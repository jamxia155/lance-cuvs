# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lance
import lance_cuvs
import pyarrow as pa
import pytest

DIM = 16
ROWS = 4096
NUM_PARTITIONS = 8
NUM_SUB_VECTORS = 4
NUM_BITS = 8


def _has_visible_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False

    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _require_gpu() -> None:
    if _has_visible_gpu():
        return

    message = "pylance-cuvs smoke test requires a CUDA-capable GPU"
    if os.environ.get("LANCE_CUVS_REQUIRE_GPU") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _vector_array(rows: int, dim: int) -> pa.FixedSizeListArray:
    values = pa.array(
        [
            row + axis / 100.0
            for row in range(rows)
            for axis in range(dim)
        ],
        type=pa.float32(),
    )
    return pa.FixedSizeListArray.from_arrays(values, dim)


@pytest.mark.gpu
def test_train_and_build_ivf_pq_artifact(tmp_path: Path) -> None:
    _require_gpu()

    dataset_uri = tmp_path / "dataset.lance"
    artifact_uri = tmp_path / "artifact"

    table = pa.table({"vector": _vector_array(ROWS, DIM)})
    lance.write_dataset(table, dataset_uri)

    training = lance_cuvs.train_ivf_pq(
        dataset_uri,
        "vector",
        metric_type="L2",
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        sample_rate=4,
        max_iters=20,
        num_bits=NUM_BITS,
        filter_nan=False,
    )

    assert training.num_partitions == NUM_PARTITIONS
    assert training.num_sub_vectors == NUM_SUB_VECTORS
    assert training.num_bits == NUM_BITS
    assert training.metric_type == "L2"

    ivf_centroids = training.ivf_centroids()
    pq_codebook = training.pq_codebook()

    assert isinstance(ivf_centroids, pa.FixedSizeListArray)
    assert len(ivf_centroids) == NUM_PARTITIONS
    assert ivf_centroids.type.list_size == DIM

    assert isinstance(pq_codebook, pa.FixedSizeListArray)
    assert len(pq_codebook) == NUM_SUB_VECTORS * (1 << NUM_BITS)
    assert pq_codebook.type.list_size == DIM // NUM_SUB_VECTORS

    artifact = lance_cuvs.build_ivf_pq_artifact(
        dataset_uri,
        "vector",
        training=training,
        artifact_uri=artifact_uri,
        batch_size=1024,
        filter_nan=False,
    )

    assert artifact.artifact_uri == str(artifact_uri)
    assert artifact.files
    assert artifact_uri.is_dir()

    for relative_path in artifact.files:
        assert (artifact_uri / relative_path).exists(), relative_path


# cuvs-cu12-local's shared libraries come from a local cuVS checkout, not
# from libcuvs-cu12's own -- running each backend in its own subprocess
# avoids ever loading two different cuVS builds into the same process
# (unsupported), and sidesteps LANCE_CUVS_BACKEND being cached per-process
# once python/lance_cuvs's loader has picked a backend once.
_BACKEND_SMOKE_SCRIPT = """
import json
import os
import sys

os.environ["LANCE_CUVS_BACKEND"] = sys.argv[1]

import lance
import lance_cuvs
import pyarrow as pa

dataset_uri = sys.argv[2]
artifact_uri = sys.argv[3]
result_path = sys.argv[4]

dim = {dim}
rows = {rows}
values = pa.array(
    [row + axis / 100.0 for row in range(rows) for axis in range(dim)],
    type=pa.float32(),
)
vectors = pa.FixedSizeListArray.from_arrays(values, dim)
table = pa.table({{"vector": vectors}})
lance.write_dataset(table, dataset_uri)

training = lance_cuvs.train_ivf_pq(
    dataset_uri,
    "vector",
    metric_type="L2",
    num_partitions={num_partitions},
    num_sub_vectors={num_sub_vectors},
    sample_rate=4,
    max_iters=20,
    num_bits={num_bits},
    filter_nan=False,
)
artifact = lance_cuvs.build_ivf_pq_artifact(
    dataset_uri,
    "vector",
    training=training,
    artifact_uri=artifact_uri,
    batch_size=1024,
    filter_nan=False,
)

ivf_centroids = training.ivf_centroids()
pq_codebook = training.pq_codebook()

with open(result_path, "w") as f:
    json.dump(
        {{
            "num_partitions": training.num_partitions,
            "num_sub_vectors": training.num_sub_vectors,
            "num_bits": training.num_bits,
            "metric_type": training.metric_type,
            "ivf_centroids_len": len(ivf_centroids),
            "ivf_centroids_list_size": ivf_centroids.type.list_size,
            "pq_codebook_len": len(pq_codebook),
            "pq_codebook_list_size": pq_codebook.type.list_size,
            "artifact_uri": artifact.artifact_uri,
            "files": artifact.files,
        }},
        f,
    )
""".format(
    dim=DIM,
    rows=ROWS,
    num_partitions=NUM_PARTITIONS,
    num_sub_vectors=NUM_SUB_VECTORS,
    num_bits=NUM_BITS,
)

_BACKEND_NOT_INSTALLED_MARKERS = (
    "is not installed",
    "no pylance-cuvs backend",
    "Unable to detect an installed cuVS runtime",
    "Failed to preload required cuVS shared libraries",
)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cu12", "cu12-local"])
def test_train_and_build_ivf_pq_artifact_backend(backend: str, tmp_path: Path) -> None:
    """Same smoke test as above, run explicitly against each backend.

    Skips a backend that isn't installed in the current environment (e.g.
    CI images that only ship the `cu12` backend wheel), rather than failing.
    """
    _require_gpu()

    dataset_uri = tmp_path / "dataset.lance"
    artifact_uri = tmp_path / "artifact"
    result_path = tmp_path / "result.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _BACKEND_SMOKE_SCRIPT,
            backend,
            str(dataset_uri),
            str(artifact_uri),
            str(result_path),
        ],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        if any(marker in proc.stderr for marker in _BACKEND_NOT_INSTALLED_MARKERS):
            pytest.skip(f"backend {backend!r} is not installed: {proc.stderr.strip()}")
        pytest.fail(
            f"backend {backend!r} smoke test failed (exit {proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    result = json.loads(result_path.read_text())

    assert result["num_partitions"] == NUM_PARTITIONS
    assert result["num_sub_vectors"] == NUM_SUB_VECTORS
    assert result["num_bits"] == NUM_BITS
    assert result["metric_type"] == "L2"

    assert result["ivf_centroids_len"] == NUM_PARTITIONS
    assert result["ivf_centroids_list_size"] == DIM

    assert result["pq_codebook_len"] == NUM_SUB_VECTORS * (1 << NUM_BITS)
    assert result["pq_codebook_list_size"] == DIM // NUM_SUB_VECTORS

    assert result["artifact_uri"] == str(artifact_uri)
    assert result["files"]
    assert artifact_uri.is_dir()

    for relative_path in result["files"]:
        assert (artifact_uri / relative_path).exists(), relative_path
