// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF-flat coarse assignment for the IVF_PQ transform.
//!
//! A second quantizer over the finalized IVF centroids: they are grouped into
//! `n_lists` lists (`~sqrt(K)` by default) with k-means, and a vector is
//! assigned by probing its `n_probes` nearest lists and scanning those lists'
//! centroids exactly. The labels are then PQ-encoded with
//! `cuvsIvfPqTransformWithLabels`, like [`crate::cagra_assign`].
//!
//! Opt in with `LANCE_CUVS_ASSIGN=ivf_flat`. Tunables (all optional):
//! - `LANCE_CUVS_IVF_ASSIGN_LISTS` (default `round(sqrt(K))`)
//! - `LANCE_CUVS_IVF_ASSIGN_PROBES` (default 16; more is slower and closer to exact)
//! - `LANCE_CUVS_IVF_ASSIGN_KMEANS_ITERS` (default 20)

use crate::cuda::{DeviceTensor, Resources, check_cuvs, narrow_i64_to_u32_async};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use std::ptr;
use std::time::Instant;

const DEFAULT_PROBES: usize = 16;
const DEFAULT_KMEANS_ITERS: usize = 20;

#[derive(Clone, Debug)]
pub(crate) struct IvfFlatAssignConfig {
    /// 0 means `round(sqrt(K))`.
    pub(crate) n_lists: usize,
    pub(crate) n_probes: usize,
    pub(crate) kmeans_n_iters: usize,
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

impl IvfFlatAssignConfig {
    pub(crate) fn from_env() -> Result<Self> {
        Ok(Self {
            n_lists: usize_from_env("LANCE_CUVS_IVF_ASSIGN_LISTS")?.unwrap_or(0),
            n_probes: usize_from_env("LANCE_CUVS_IVF_ASSIGN_PROBES")?.unwrap_or(DEFAULT_PROBES),
            kmeans_n_iters: usize_from_env("LANCE_CUVS_IVF_ASSIGN_KMEANS_ITERS")?
                .unwrap_or(DEFAULT_KMEANS_ITERS),
        })
    }
}

/// An IVF-flat index over the IVF centroids plus its search parameters.
pub(crate) struct IvfFlatAssigner {
    index: cuvs_sys::cuvsIvfFlatIndex_t,
    search_params: cuvs_sys::cuvsIvfFlatSearchParams_t,
    // The index keeps its own copy of the vectors; this only backs the build.
    _centers: DeviceTensor<f32>,
}

// SAFETY: the index and search parameters are immutable after `build`;
// `cuvsIvfFlatSearch` only reads them, and takes its own resources handle.
unsafe impl Send for IvfFlatAssigner {}
unsafe impl Sync for IvfFlatAssigner {}

impl IvfFlatAssigner {
    /// Builds the index over `centroids` (`[n_centroids, dim]`, f32).
    ///
    /// The metric mirrors cuVS's coarse clustering metric: inner product
    /// clusters with L2, everything else keeps its own.
    pub(crate) fn build(
        resources: &Resources,
        centroids: &FixedSizeListArray,
        metric_type: DistanceType,
        config: &IvfFlatAssignConfig,
    ) -> Result<Self> {
        let start = Instant::now();
        let metric = match metric_type {
            DistanceType::L2 | DistanceType::Dot => cuvs_sys::cuvsDistanceType::L2Expanded,
            DistanceType::Cosine => cuvs_sys::cuvsDistanceType::CosineExpanded,
            other => {
                return Err(Error::not_supported(format!(
                    "IVF-flat assignment does not support metric {other:?}"
                )));
            }
        };
        let dim = centroids.value_length() as usize;
        let n_centroids = centroids.len();
        let n_lists = if config.n_lists == 0 {
            ((n_centroids as f64).sqrt().round() as usize).max(1)
        } else {
            config.n_lists
        }
        .min(n_centroids);
        let n_probes = config.n_probes.clamp(1, n_lists);
        let values = centroids
            .values()
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| Error::invalid_input("IVF-flat assignment needs f32 IVF centroids"))?
            .values();

        let mut centers = DeviceTensor::<f32>::try_new(resources, &[n_centroids, dim])?;
        let stream = resources.get_cuda_stream()?;
        centers.copy_from_host_async_on(stream, values)?;
        resources.sync_stream()?;

        let mut params = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsIvfFlatIndexParamsCreate(&mut params) },
            "allocate IVF-flat index params",
        )?;
        unsafe {
            (*params).metric = metric;
            (*params).n_lists = n_lists as u32;
            (*params).kmeans_n_iters = config.kmeans_n_iters as u32;
            (*params).kmeans_trainset_fraction = 1.0;
            // Ids are the row numbers of `centroids`, which is the label.
            (*params).add_data_on_build = true;
            (*params).adaptive_centers = false;
        }
        let mut index = ptr::null_mut();
        let created = check_cuvs(
            unsafe { cuvs_sys::cuvsIvfFlatIndexCreate(&mut index) },
            "create IVF-flat index",
        );
        let built = created.and_then(|()| {
            check_cuvs(
                unsafe {
                    cuvs_sys::cuvsIvfFlatBuild(resources.0, params, centers.as_mut_ptr(), index)
                },
                "build IVF-flat index over IVF centroids",
            )
        });
        let _ = unsafe { cuvs_sys::cuvsIvfFlatIndexParamsDestroy(params) };
        let built = built.and_then(|()| resources.sync_stream());

        let mut search_params = ptr::null_mut();
        let searchable = built.and_then(|()| {
            check_cuvs(
                unsafe { cuvs_sys::cuvsIvfFlatSearchParamsCreate(&mut search_params) },
                "allocate IVF-flat search params",
            )
        });
        // From here `Self` owns the handles, so any error below drops them.
        let assigner = Self { index, search_params, _centers: centers };
        searchable?;
        unsafe {
            (*search_params).n_probes = n_probes as u32;
        }
        eprintln!(
            "cuVS IVF-flat assignment: index over {n_centroids} x {dim} centroids, \
             n_lists={n_lists} n_probes={n_probes} kmeans_n_iters={} build_s={:.3}",
            config.kmeans_n_iters,
            start.elapsed().as_secs_f64()
        );
        Ok(assigner)
    }

    /// Top-1 search: finds each query row's nearest centroid and writes its id
    /// into `labels` (`[rows]`, u32). `neighbors` (`[rows, 1]`, i64) and
    /// `distances` (`[rows, 1]`, f32) are scratch. Enqueued on `resources`'
    /// stream.
    pub(crate) fn assign(
        &self,
        resources: &Resources,
        queries: *mut cuvs_sys::DLManagedTensor,
        neighbors: &mut DeviceTensor<i64>,
        distances: &mut DeviceTensor<f32>,
        labels: &mut DeviceTensor<u32>,
        rows: usize,
    ) -> Result<()> {
        let no_filter = cuvs_sys::cuvsFilter {
            addr: 0,
            type_: cuvs_sys::cuvsFilterType::NO_FILTER,
        };
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsIvfFlatSearch(
                    resources.0,
                    self.search_params,
                    self.index,
                    queries,
                    neighbors.as_mut_ptr(),
                    distances.as_mut_ptr(),
                    no_filter,
                )
            },
            "assign vectors to IVF centroids with IVF-flat",
        )?;
        narrow_i64_to_u32_async(resources.get_cuda_stream()?, neighbors, labels, rows)
    }
}

impl Drop for IvfFlatAssigner {
    fn drop(&mut self) {
        unsafe {
            if !self.search_params.is_null() {
                let _ = cuvs_sys::cuvsIvfFlatSearchParamsDestroy(self.search_params);
            }
            if !self.index.is_null() {
                let _ = cuvs_sys::cuvsIvfFlatIndexDestroy(self.index);
            }
        }
    }
}
