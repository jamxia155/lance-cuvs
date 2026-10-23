// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Grouped-GEMM coarse assignment for the IVF_PQ transform.
//!
//! The finalized IVF centroids are grouped into `n_lists` lists (`~sqrt(K)` by
//! default). A batch is assigned by scoring the list centers, bucketing each
//! vector's `n_probes` best (list, vector) pairs by list, and running one
//! dense TF32 GEMM per list against that list's centroids (see
//! `cuvsGroupedAssign` in cuVS). The labels are then PQ-encoded with
//! `cuvsIvfPqTransformWithLabels`, like [`crate::cagra_assign`].
//!
//! Opt in with `LANCE_CUVS_ASSIGN=grouped`. Tunables (all optional):
//! - `LANCE_CUVS_GROUPED_LISTS` (default `round(sqrt(K))`)
//! - `LANCE_CUVS_GROUPED_PROBES` (default 8; more is slower and closer to exact)
//! - `LANCE_CUVS_GROUPED_KMEANS_ITERS` (default 10)
//! - `LANCE_CUVS_GROUPED_TILE_ALIGN` (default 256)

use crate::cuda::{DeviceTensor, Resources, check_cuvs};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use std::ptr;
use std::time::Instant;

const DEFAULT_PROBES: usize = 8;
const DEFAULT_KMEANS_ITERS: usize = 10;
const DEFAULT_TILE_ALIGN: usize = 256;

#[derive(Clone, Debug)]
pub(crate) struct GroupedAssignConfig {
    /// 0 means `round(sqrt(K))`.
    pub(crate) n_lists: usize,
    pub(crate) n_probes: usize,
    pub(crate) kmeans_n_iters: usize,
    pub(crate) tile_align: usize,
}

fn usize_from_env(name: &str) -> Result<Option<usize>> {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .map(Some)
            .map_err(|error| Error::invalid_input(format!("invalid {name} '{value}': {error}"))),
        Err(_) => Ok(None),
    }
}

impl GroupedAssignConfig {
    pub(crate) fn from_env() -> Result<Self> {
        Ok(Self {
            n_lists: usize_from_env("LANCE_CUVS_GROUPED_LISTS")?.unwrap_or(0),
            n_probes: usize_from_env("LANCE_CUVS_GROUPED_PROBES")?.unwrap_or(DEFAULT_PROBES),
            kmeans_n_iters: usize_from_env("LANCE_CUVS_GROUPED_KMEANS_ITERS")?
                .unwrap_or(DEFAULT_KMEANS_ITERS),
            tile_align: usize_from_env("LANCE_CUVS_GROUPED_TILE_ALIGN")?
                .unwrap_or(DEFAULT_TILE_ALIGN),
        })
    }
}

/// A grouped-assign index over the IVF centroids.
pub(crate) struct GroupedAssigner {
    index: cuvs_sys::cuvsGroupedAssignIndex_t,
}

// SAFETY: the index is immutable after `build`; `cuvsGroupedAssign` only reads
// it and works on the calling `resources`' stream.
unsafe impl Send for GroupedAssigner {}
unsafe impl Sync for GroupedAssigner {}

impl GroupedAssigner {
    /// Builds the index over `centroids` (`[n_centroids, dim]`, f32).
    ///
    /// The metric mirrors cuVS's coarse clustering metric: inner product
    /// clusters with L2, everything else keeps its own.
    pub(crate) fn build(
        resources: &Resources,
        centroids: &FixedSizeListArray,
        metric_type: DistanceType,
        config: &GroupedAssignConfig,
    ) -> Result<Self> {
        let start = Instant::now();
        let metric = match metric_type {
            DistanceType::L2 | DistanceType::Dot => cuvs_sys::cuvsDistanceType::L2Expanded,
            DistanceType::Cosine => cuvs_sys::cuvsDistanceType::CosineExpanded,
            other => {
                return Err(Error::not_supported(format!(
                    "grouped assignment does not support metric {other:?}"
                )));
            }
        };
        let dim = centroids.value_length() as usize;
        let n_centroids = centroids.len();
        if dim % 4 != 0 {
            return Err(Error::not_supported(format!(
                "grouped assignment needs a dimension divisible by 4, got {dim}"
            )));
        }
        let values = centroids
            .values()
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| Error::invalid_input("grouped assignment needs f32 IVF centroids"))?
            .values();

        let mut device_centroids = DeviceTensor::<f32>::try_new(resources, &[n_centroids, dim])?;
        let stream = resources.get_cuda_stream()?;
        device_centroids.copy_from_host_async_on(stream, values)?;
        resources.sync_stream()?;

        let mut index = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsGroupedAssignIndexCreate(&mut index) },
            "create grouped-assign index",
        )?;
        // From here `Self` owns the handle, so any error below drops it.
        let assigner = Self { index };
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsGroupedAssignBuild(
                    resources.0,
                    device_centroids.as_mut_ptr(),
                    metric,
                    config.n_lists as u32,
                    config.n_probes as u32,
                    config.kmeans_n_iters as u32,
                    config.tile_align as u32,
                    assigner.index,
                )
            },
            "build grouped-assign index over IVF centroids",
        )?;
        resources.sync_stream()?;
        eprintln!(
            "cuVS grouped assignment: {n_centroids} x {dim} centroids, n_lists={} n_probes={} \
             kmeans_n_iters={} tile_align={} build_s={:.3}",
            if config.n_lists == 0 {
                format!("sqrt(K)={}", (n_centroids as f64).sqrt().round() as usize)
            } else {
                config.n_lists.to_string()
            },
            config.n_probes,
            config.kmeans_n_iters,
            config.tile_align,
            start.elapsed().as_secs_f64()
        );
        Ok(assigner)
    }

    /// Writes each query row's nearest-centroid id into `labels` (`[rows]`,
    /// u32). Enqueued on `resources`' stream, which cuVS synchronizes once.
    pub(crate) fn assign(
        &self,
        resources: &Resources,
        queries: *mut cuvs_sys::DLManagedTensor,
        labels: &mut DeviceTensor<u32>,
    ) -> Result<()> {
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsGroupedAssign(resources.0, self.index, queries, labels.as_mut_ptr())
            },
            "assign vectors to IVF centroids with grouped GEMMs",
        )
    }
}

impl Drop for GroupedAssigner {
    fn drop(&mut self) {
        if !self.index.is_null() {
            let _ = unsafe { cuvs_sys::cuvsGroupedAssignIndexDestroy(self.index) };
        }
    }
}
