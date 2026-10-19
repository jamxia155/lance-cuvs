# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Write the raw f32 inputs that `cagra_assign_bench` reads.

Reads a vector column from a Lance dataset and writes two row-major,
little-endian f32 files: `centroids.f32` and `queries.f32`.

By default the "centroids" are a random sample of `--num-centroids` dataset
vectors, which is only a proxy for trained k-means centroids (they are less
smooth, so graph recall can differ). To benchmark with real centroids, pass
`--centroids-npy` with an array of shape (k, dim) dumped from a trained index.

`--normalize` L2-normalizes both files; use it for cosine datasets, since
Lance assigns cosine as L2 on unit vectors.
"""
import argparse

import lance
import numpy as np


def column_to_matrix(table, column):
    arr = table[column].combine_chunks()
    dim = arr.type.list_size
    return arr.values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--column", default="vector")
    p.add_argument("--num-centroids", type=int, default=32768)
    p.add_argument("--num-queries", type=int, default=1 << 20)
    p.add_argument("--centroids-npy")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = lance.dataset(args.dataset)
    rng = np.random.default_rng(args.seed)
    total = ds.count_rows()

    # Queries: a contiguous prefix, so no random-access reads over a huge table.
    queries = column_to_matrix(
        ds.to_table(columns=[args.column], limit=args.num_queries), args.column
    )
    if args.centroids_npy:
        centroids = np.load(args.centroids_npy).astype(np.float32)
    else:
        ids = np.sort(rng.choice(total, size=args.num_centroids, replace=False))
        centroids = column_to_matrix(ds.take(ids, columns=[args.column]), args.column)

    if args.normalize:
        for m in (queries, centroids):
            m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    queries.tofile(f"{args.out_dir}/queries.f32")
    centroids.tofile(f"{args.out_dir}/centroids.f32")
    print(f"dim={queries.shape[1]} queries={queries.shape[0]} centroids={centroids.shape[0]}")


if __name__ == "__main__":
    main()
