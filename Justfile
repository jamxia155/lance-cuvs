set shell := ["bash", "-euo", "pipefail", "-c"]

root := justfile_directory()
uv_project := "uv run --project '" + root + "' --no-sync"
cmake_cmd := uv_project + " python -c 'import shutil; print(shutil.which(\"cmake\") or \"\")'"
rapids_env := "export CMAKE=\"$(" + cmake_cmd + ")\"; eval \"$(" + uv_project + " python tools/rapids_env.py --format shell)\""
dist_dir := root + "/dist"
backend_wheel_compatibility := "manylinux_2_28"

# Install prefix used by the local `cuvs` checkout's `build.sh` (see
# ../cuvs/.gitignore's `/install/`). Override with
# `just --set cuvs_local_install <path>` if you built to somewhere else.
cuvs_local_install := env_var_or_default("CUVS_LOCAL_INSTALL_PREFIX", root + "/../cuvs/install")
# `rapids_env` (shared with the `cu12` recipes) exports `cuvs_DIR` pointing at
# the pip-installed `libcuvs-cu12` package -- CMake's `find_package(cuvs)`
# treats an already-set `cuvs_DIR` as authoritative and uses it directly,
# bypassing `CMAKE_PREFIX_PATH` search. Re-export it (and the other RAPIDS
# `*_DIR` hints, which would otherwise point at that same pip package's
# possibly-mismatched raft/rmm/nvtx3/rapids_logger) to the local build after
# `rapids_env` runs, so ours wins instead of being silently shadowed.
#
# `cuvs-sys`'s build script doesn't read the `CMAKE` env var (only
# `CMAKE_PREFIX_PATH`/`CONDA_PREFIX`/`LIBCUVS_USE_PYTHON`/`VIRTUAL_ENV`) --
# it just spawns bare `cmake` off `PATH`. The local cuVS checkout's installed
# CMake package config requires CMake >=4.0 (`pyproject.toml`'s `dev` group
# pins a matching `cmake` pip package), so prepend the project venv's `bin`
# to `PATH` to make sure that newer `cmake` is actually the one found,
# ahead of whatever older `cmake` the system/container image ships.
local_rapids_env := rapids_env + "; export PATH=\"" + root + "/.venv/bin:$PATH\" CMAKE_PREFIX_PATH=\"" + cuvs_local_install + ":${CMAKE_PREFIX_PATH:-}\" cuvs_DIR=\"" + cuvs_local_install + "/lib/cmake/cuvs\"; unset nvtx3_DIR raft_DIR rmm_DIR rapids_logger_DIR"

default:
  @just --list

apply-release-version tag:
  {{uv_project}} python tools/apply_release_version.py --tag {{tag}}

show-release-version tag:
  {{uv_project}} python tools/apply_release_version.py --tag {{tag}} --dry-run

sync-dev:
  uv sync --group dev

sync-dev-no-project:
  uv sync --group dev --no-install-project

loader-test: sync-dev
  {{uv_project}} pytest -q tests/test_loader.py

# `env -u CONDA_PREFIX`: some container images export `CONDA_PREFIX` as an
# empty (but present) variable even with no conda env active (`CONDA_SHLVL=0`)
# -- `maturin` only checks for presence, not a non-empty value, and refuses
# to run when both `VIRTUAL_ENV` and `CONDA_PREFIX` are set. Unsetting just
# `CONDA_PREFIX` for the `maturin` invocation avoids that false positive
# without touching `VIRTUAL_ENV` (which must stay pointed at this project's
# venv).
backend-wheel: sync-dev
  rm -rf dist
  mkdir -p dist
  {{rapids_env}} && cd backends/cuvs_26_02 && env -u CONDA_PREFIX {{uv_project}} maturin build --release --locked --compatibility {{backend_wheel_compatibility}} --auditwheel skip --out ../../dist

backend-develop: sync-dev
  {{rapids_env}} && cd backends/cuvs_26_02 && env -u CONDA_PREFIX {{uv_project}} maturin develop --release --locked

# `cuvs_local`: built against a local cuVS checkout instead of a published
# `cuvs` release, for head-to-head benchmarking against `cu12`. Requires a
# matching cuVS C++ build (component `c_api`) already built/installed and
# discoverable by `cuvs-sys`'s CMake `find_package` (e.g. via
# `CMAKE_PREFIX_PATH`) -- see backends/cuvs_local/Cargo.toml.
backend-local-wheel: sync-dev
  rm -rf dist
  mkdir -p dist
  {{local_rapids_env}} && cd backends/cuvs_local && env -u CONDA_PREFIX {{uv_project}} maturin build --release --locked --compatibility {{backend_wheel_compatibility}} --auditwheel skip --out ../../dist

backend-local-develop: sync-dev
  {{local_rapids_env}} && cd backends/cuvs_local && env -u CONDA_PREFIX {{uv_project}} maturin develop --release --locked

build-wheels: sync-dev
  rm -rf dist
  mkdir -p dist
  {{uv_project}} python -m py_compile \
    python/lance_cuvs/__init__.py \
    python/lance_cuvs/_loader.py \
    backends/cuvs_26_02/python/lance_cuvs_backend_cu12/__init__.py
  uv build --wheel --out-dir dist
  {{rapids_env}} && cd backends/cuvs_26_02 && env -u CONDA_PREFIX {{uv_project}} maturin build --release --locked --compatibility {{backend_wheel_compatibility}} --auditwheel skip --out ../../dist

python-build: build-wheels test-loader-wheel
  @:

python-release: build-wheels
  @:

rust-fmt-check:
  cargo fmt --manifest-path backends/cuvs_26_02/Cargo.toml --all --check

rust-clippy: sync-dev-no-project
  {{rapids_env}} && cargo clippy --manifest-path backends/cuvs_26_02/Cargo.toml --locked --all-targets --features python -- -D warnings

rust-check: sync-dev-no-project
  {{rapids_env}} && cargo check --manifest-path backends/cuvs_26_02/Cargo.toml --locked --all-targets --features python

rust-build: rust-fmt-check rust-clippy rust-check
  @:

rust-local-fmt-check:
  cargo fmt --manifest-path backends/cuvs_local/Cargo.toml --all --check

rust-local-clippy: sync-dev-no-project
  {{local_rapids_env}} && cargo clippy --manifest-path backends/cuvs_local/Cargo.toml --locked --all-targets --features python -- -D warnings

rust-local-check: sync-dev-no-project
  {{local_rapids_env}} && cargo check --manifest-path backends/cuvs_local/Cargo.toml --locked --all-targets --features python

rust-local-build: rust-local-fmt-check rust-local-clippy rust-local-check
  @:

test-loader-wheel:
  @root_wheel="$(find '{{dist_dir}}' -maxdepth 1 -type f -name 'pylance_cuvs-*.whl' | head -n 1)"; \
  test -n "$root_wheel"; \
  tmpdir="$(mktemp -d)"; \
  trap 'rm -rf "$tmpdir"' EXIT; \
  uv venv --python 3.12 "$tmpdir/venv"; \
  uv pip install --python "$tmpdir/venv/bin/python" pytest "$root_wheel"; \
  "$tmpdir/venv/bin/python" -m pytest -q tests/test_loader.py

test-gpu-wheel:
  @root_wheel="$(find '{{dist_dir}}' -maxdepth 1 -type f -name 'pylance_cuvs-*.whl' | head -n 1)"; \
  backend_wheel="$(find '{{dist_dir}}' -maxdepth 1 -type f -name 'pylance_cuvs_cu12-*.whl' | head -n 1)"; \
  test -n "$root_wheel"; \
  test -n "$backend_wheel"; \
  tmpdir="$(mktemp -d)"; \
  trap 'rm -rf "$tmpdir"' EXIT; \
  uv venv --python 3.12 "$tmpdir/venv"; \
  uv pip install --python "$tmpdir/venv/bin/python" \
    pytest \
    pylance \
    libcuvs-cu12==26.2.0 \
    "$root_wheel" \
    "$backend_wheel"; \
  LANCE_CUVS_REQUIRE_GPU="${LANCE_CUVS_REQUIRE_GPU:-1}" "$tmpdir/venv/bin/python" -m pytest -q tests/test_smoke.py

gpu-smoke: build-wheels test-gpu-wheel
  @:

container-shell:
  tools/run_in_container.sh -- bash

container-python-build:
  tools/run_in_container.sh -- just python-build

container-rust-build:
  tools/run_in_container.sh -- just rust-build

container-python-release:
  tools/run_in_container.sh -- just python-release

container-gpu-smoke:
  tools/run_in_container.sh --gpu -- just gpu-smoke

container-backend-wheel:
  tools/run_in_container.sh -- just backend-wheel

container-backend-local-wheel:
  tools/run_in_container.sh -- just backend-local-wheel
