# pylance-cuvs

`pylance-cuvs` provides cuVS-backed IVF_PQ training and artifact building for
Lance datasets.

It covers one narrow part of the indexing pipeline:

1. Train an IVF_PQ model with cuVS.
2. Build a partition-local artifact from a Lance dataset.
3. Pass the training output and artifact back to Lance for finalization.

It does **not** create or register a Lance index on your behalf.

## Requirements

- Linux
- Python 3.12+
- CUDA 12 runtime available on the machine
- cuVS runtime `libcuvs-cu12==26.2.0`
- A matching backend package: `pylance-cuvs-cu12`
- A Lance build at commit `6112a34bfe38618f07c099217dc3d89fd39ca6bb`
  or a descendant that preserves the same vector-build APIs

## Installation

Install the loader, the matching backend package, and the cuVS runtime into the
same Python environment:

- `pylance-cuvs`
- `pylance-cuvs-cu12`
- `libcuvs-cu12==26.2.0`

Published beta releases are wheel-only. Source distributions are not supported.

Release versions are injected from the git tag during the publish workflow.

If you are working from this repository, the shortest local setup is:

```bash
just sync-dev
just backend-develop
```

For Lance-side integration, pin Lance to commit
`6112a34bfe38618f07c099217dc3d89fd39ca6bb` first, then validate against this
beta.

The loader chooses the backend from the installed cuVS runtime package. You can
override detection when needed:

```bash
export LANCE_CUVS_BACKEND=cu12
```

Legacy overrides such as `cuvs-26-02` are still accepted for compatibility.

A second backend, `cu12-local` (`backends/cuvs_local`), builds against a local
cuVS checkout instead of a published cuVS release, for benchmarking against
`cu12` head-to-head. It is never chosen by runtime auto-detection -- select it
explicitly:

```bash
export LANCE_CUVS_BACKEND=cu12-local
```

`cu12-local` expects the local `cuvs` checkout (a sibling directory, e.g.
`../cuvs` relative to `lance-cuvs`) to already be built and installed via its
own `build.sh libcuvs`, with `INSTALL_PREFIX` set to a plain (non-conda)
directory, e.g. (run from inside the `cuvs` checkout):

```bash
export INSTALL_PREFIX="$(pwd)/install"
```

`just backend-local-develop`/`backend-local-wheel`/`rust-local-check` default
to that same path (`../cuvs/install`, relative to this repo) for
`CMAKE_PREFIX_PATH` so `cuvs-sys`'s CMake `find_package(cuvs)` can locate it;
override with `CUVS_LOCAL_INSTALL_PREFIX` if you installed elsewhere. In
practice the built extension has loaded `libcuvs_c.so`/`libcuvs.so` from
that prefix without any extra `LD_LIBRARY_PATH` -- the linker appears to
bake in an RPATH back to the CMake install's `lib/` dir. If a future build
*does* fail to find those shared libraries at import time (the loader's own
shared-library preload only searches installed Python packages'
site-packages, not an arbitrary CMake install prefix, so it won't help
here), point `LD_LIBRARY_PATH` at `$INSTALL_PREFIX/lib` (`lib`, not `lib64`
-- this build's install layout uses `lib`).

## Quick Start

```python
from pathlib import Path

import lance
import lance_cuvs
import pyarrow as pa


def vector_array(rows: int, dim: int) -> pa.FixedSizeListArray:
    values = pa.array(
        [row + axis / 100.0 for row in range(rows) for axis in range(dim)],
        type=pa.float32(),
    )
    return pa.FixedSizeListArray.from_arrays(values, dim)


dataset_uri = Path("/tmp/example.lance")
artifact_uri = Path("/tmp/example-artifact")

table = pa.table({"vector": vector_array(rows=4096, dim=16)})
lance.write_dataset(table, dataset_uri)

training = lance_cuvs.train_ivf_pq(
    dataset_uri,
    "vector",
    metric_type="L2",
    num_partitions=8,
    num_sub_vectors=4,
    sample_rate=4,
    max_iters=20,
)

artifact = lance_cuvs.build_ivf_pq_artifact(
    dataset_uri,
    "vector",
    training=training,
    artifact_uri=artifact_uri,
)

print(type(training.ivf_centroids()))
print(type(training.pq_codebook()))
print(artifact.artifact_uri)
print(artifact.files)
```

## Python API

### `lance_cuvs.train_ivf_pq(...)`

Trains an IVF_PQ model with cuVS and returns `IvfPqTrainingOutput`.

Key fields and methods:

- `num_partitions`
- `num_sub_vectors`
- `num_bits`
- `metric_type`
- `ivf_centroids()` -> `pyarrow.FixedSizeListArray`
- `pq_codebook()` -> `pyarrow.FixedSizeListArray`

### `lance_cuvs.build_ivf_pq_artifact(...)`

Builds a partition-local artifact from a Lance dataset and a previously trained
model, then returns `IvfPqArtifactOutput`.

Key fields:

- `artifact_uri`
- `files`

`artifact_uri` must resolve to the local filesystem.

## Scope

Use `pylance-cuvs` when you want cuVS to do the expensive GPU-side training and
encoding work, but you still want Lance to own index finalization.

Do not expect this package to:

- finalize an index
- register an index in a dataset
- provide a generic Lance wrapper beyond IVF_PQ training and artifact build

## Development

List available tasks with:

```bash
just
```

Run commands inside the shared development container with:

```bash
just container-shell
```

Use `--platform linux/amd64` only when you explicitly want to match the
GitHub-hosted runner architecture:

```bash
tools/run_in_container.sh --platform linux/amd64 -- bash
```

Run the CI-equivalent CPU build locally with:

```bash
just container-python-build
```

Run the Rust quality gates locally with:

```bash
just container-rust-build
```

Build all release wheels locally with:

```bash
just container-python-release
```

Create the root development environment with:

```bash
just sync-dev
```

Run loader-only tests with:

```bash
just loader-test
```

Build the root and backend wheels with:

```bash
just build-wheels
```

Test the published root wheel in a clean environment with:

```bash
just test-loader-wheel
```

Test the published wheels on a GPU-capable machine with:

```bash
just test-gpu-wheel
```

Build only the backend wheel with:

```bash
just backend-wheel
```

Build the `cu12` backend in-place with:

```bash
just backend-develop
```

Run the Rust formatting check with:

```bash
just rust-fmt-check
```

Run Clippy with the Python bindings enabled:

```bash
just rust-clippy
```

Run cargo check with the Python bindings enabled:

```bash
just rust-check
```

Run the release-style GPU smoke end to end with:

```bash
just gpu-smoke
```

## Repository Layout

- `python/lance_cuvs`
  - root loader package
- `backends/cuvs_26_02`
  - cuVS `26.02` source tree for the CUDA 12 backend package
- `backends/cuvs_26_02/src/backend.rs`
  - Lance-facing orchestration
- `backends/cuvs_26_02/src/cuda.rs`
  - CUDA / cuVS wrappers and tensor helpers
- `backends/cuvs_26_02/src/python.rs`
  - PyO3 bindings
- `backends/cuvs_local`
  - Same shape as `cuvs_26_02`, but its `cuvs`/`cuvs-sys` dependencies point
    at a local cuVS checkout (path dependency) instead of a published crate
    version. Used to benchmark cuVS API/behavior changes (e.g. GDS/kvikio
    integration) that don't exist in any released cuVS version yet, against
    the `cuvs_26_02` baseline.

## License

Apache License 2.0. See [LICENSE](LICENSE).
