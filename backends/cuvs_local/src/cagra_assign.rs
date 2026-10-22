// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph-based coarse assignment for the IVF_PQ transform.
//!
//! cuVS's `ivf_pq::transform` assigns each vector to its nearest IVF centroid
//! with an exhaustive top-1 search (`top_1_nn`). At large partition counts that
//! dominates the transform. A [`CagraAssigner`] builds a CAGRA graph over the
//! finalized centroids once and answers the same top-1 query approximately,
//! like the HNSW index Lance builds over centroids on the CPU path. The labels
//! it produces are then PQ-encoded with `cuvsIvfPqTransformWithLabels`.
//!
//! Opt in with `LANCE_CUVS_ASSIGN=cagra`. Tunables (all optional):
//! - `LANCE_CUVS_CAGRA_GRAPH_DEGREE` (default 32)
//! - `LANCE_CUVS_CAGRA_INTERMEDIATE_DEGREE` (default 2 x graph degree)
//! - `LANCE_CUVS_CAGRA_ITOPK` (default 32; larger is slower and closer to exact)
//! - `LANCE_CUVS_CAGRA_MAX_QUERIES` (default 0: cuVS decides)
//! - `LANCE_CUVS_CAGRA_MAX_ITERATIONS` (default 0: cuVS decides; lower is faster, less exact)

use crate::cuda::{DeviceTensor, DeviceTensorView, Resources, check_cuvs};
use arrow_array::{Array, FixedSizeListArray};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use std::ptr;
use std::time::Instant;

const DEFAULT_GRAPH_DEGREE: usize = 32;
const DEFAULT_ITOPK: usize = 32;

#[derive(Clone, Debug)]
pub(crate) struct CagraAssignConfig {
    pub(crate) graph_degree: usize,
    pub(crate) intermediate_graph_degree: usize,
    pub(crate) itopk_size: usize,
    pub(crate) max_queries: usize,
    /// 0 lets cuVS resolve it: `itopk / search_width` plus one per graph hop level
    /// (36 for 32768 centroids at degree 32, itopk 32). Lower is faster and less exact.
    pub(crate) max_iterations: usize,
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

impl CagraAssignConfig {
    /// Reads the `LANCE_CUVS_CAGRA_*` settings; the caller has already chosen CAGRA.
    pub(crate) fn from_env() -> Result<Self> {
        let graph_degree =
            usize_from_env("LANCE_CUVS_CAGRA_GRAPH_DEGREE")?.unwrap_or(DEFAULT_GRAPH_DEGREE);
        Ok(Self {
            graph_degree,
            intermediate_graph_degree: usize_from_env("LANCE_CUVS_CAGRA_INTERMEDIATE_DEGREE")?
                .unwrap_or(graph_degree * 2),
            itopk_size: usize_from_env("LANCE_CUVS_CAGRA_ITOPK")?.unwrap_or(DEFAULT_ITOPK),
            max_queries: usize_from_env("LANCE_CUVS_CAGRA_MAX_QUERIES")?.unwrap_or(0),
            max_iterations: usize_from_env("LANCE_CUVS_CAGRA_MAX_ITERATIONS")?.unwrap_or(0),
        })
    }
}

/// A CAGRA graph over the IVF centroids plus its search parameters.
pub(crate) struct CagraAssigner {
    index: cuvs_sys::cuvsCagraIndex_t,
    // The index keeps a non-owning view of this dataset: it must outlive it.
    dataset: cuvs_sys::cuvsDataset_t,
    search_params: cuvs_sys::cuvsCagraSearchParams_t,
    // Backs `dataset` when it is a view; declared last so it drops last.
    _centers: DeviceTensor<f32>,
}

// SAFETY: the graph and search parameters are immutable after `build`;
// `cuvsCagraSearch` only reads them, and takes its own resources handle.
unsafe impl Send for CagraAssigner {}
unsafe impl Sync for CagraAssigner {}

impl CagraAssigner {
    /// Builds the graph over `centroids` (`[n_lists, dim]`, f32).
    ///
    /// The metric mirrors cuVS's coarse clustering metric: inner product
    /// clusters with L2, everything else keeps its own.
    pub(crate) fn build(
        resources: &Resources,
        centroids: &FixedSizeListArray,
        metric_type: DistanceType,
        config: &CagraAssignConfig,
    ) -> Result<Self> {
        let start = Instant::now();
        let metric = match metric_type {
            DistanceType::L2 | DistanceType::Dot => cuvs_sys::cuvsDistanceType::L2Expanded,
            DistanceType::Cosine => cuvs_sys::cuvsDistanceType::CosineExpanded,
            other => {
                return Err(Error::not_supported(format!(
                    "CAGRA assignment does not support metric {other:?}"
                )));
            }
        };
        let dim = centroids.value_length() as usize;
        let n_lists = centroids.len();
        let values = centroids
            .values()
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| Error::invalid_input("CAGRA assignment needs f32 IVF centroids"))?
            .values();

        let mut centers = DeviceTensor::<f32>::try_new(resources, &[n_lists, dim])?;
        let stream = resources.get_cuda_stream()?;
        centers.copy_from_host_async_on(stream, values)?;
        resources.sync_stream()?;

        // cuVS wants a padded dataset. A device tensor whose rows already have the
        // padded stride can only be viewed (`centers` then stays alive with the
        // index); any other layout is copied into padded storage.
        let mut dataset = ptr::null_mut();
        let viewed = check_cuvs(
            unsafe {
                cuvs_sys::cuvsDatasetMakePaddedView(resources.0, centers.as_mut_ptr(), &mut dataset)
            },
            "view centroids as a CAGRA dataset",
        );
        if viewed.is_err() {
            dataset = ptr::null_mut();
            check_cuvs(
                unsafe {
                    cuvs_sys::cuvsDatasetMakePadded(
                        resources.0,
                        centers.as_mut_ptr(),
                        cuvs_sys::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                        &mut dataset,
                    )
                },
                "create CAGRA centroid dataset",
            )?;
        }

        let mut params = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsCagraIndexParamsCreate(&mut params) },
            "allocate CAGRA index params",
        )?;
        unsafe {
            (*params).metric = metric;
            (*params).graph_degree = config.graph_degree;
            (*params).intermediate_graph_degree = config.intermediate_graph_degree;
        }
        let mut index = ptr::null_mut();
        let created = check_cuvs(
            unsafe { cuvs_sys::cuvsCagraIndexCreate(&mut index) },
            "create CAGRA index",
        );
        let built = created.and_then(|()| {
            check_cuvs(
                unsafe { cuvs_sys::cuvsCagraBuild(resources.0, params, dataset, index) },
                "build CAGRA index over IVF centroids",
            )
        });
        let _ = unsafe { cuvs_sys::cuvsCagraIndexParamsDestroy(params) };
        let built = built.and_then(|()| resources.sync_stream());

        let mut search_params = ptr::null_mut();
        let searchable = built.and_then(|()| {
            check_cuvs(
                unsafe { cuvs_sys::cuvsCagraSearchParamsCreate(&mut search_params) },
                "allocate CAGRA search params",
            )
        });
        // From here `Self` owns the handles, so any error below drops them.
        let assigner = Self { index, dataset, search_params, _centers: centers };
        searchable?;
        unsafe {
            (*search_params).itopk_size = config.itopk_size;
            (*search_params).max_queries = config.max_queries;
            (*search_params).max_iterations = config.max_iterations;
        }
        eprintln!(
            "cuVS CAGRA assignment: graph over {n_lists} x {dim} centroids, graph_degree={} \
             intermediate_graph_degree={} itopk={} max_iterations={} max_queries={} build_s={:.3}",
            config.graph_degree,
            config.intermediate_graph_degree,
            config.itopk_size,
            config.max_iterations,
            config.max_queries,
            start.elapsed().as_secs_f64()
        );
        Ok(assigner)
    }

    /// Top-1 search: writes the nearest centroid id of each query row into
    /// `neighbors` (`[rows, 1]`, u32) and its distance into `distances`
    /// (`[rows, 1]`, f32). Enqueued on `resources`' stream.
    pub(crate) fn assign(
        &self,
        resources: &Resources,
        queries: *mut cuvs_sys::DLManagedTensor,
        neighbors: &mut DeviceTensorView,
        distances: *mut cuvs_sys::DLManagedTensor,
    ) -> Result<()> {
        let no_filter = cuvs_sys::cuvsFilter {
            addr: 0,
            type_: cuvs_sys::cuvsFilterType::NO_FILTER,
        };
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsCagraSearch(
                    resources.0,
                    self.search_params,
                    self.index,
                    queries,
                    neighbors.as_mut_ptr(),
                    distances,
                    no_filter,
                )
            },
            "assign vectors to IVF centroids with CAGRA",
        )
    }
}

impl Drop for CagraAssigner {
    fn drop(&mut self) {
        unsafe {
            if !self.search_params.is_null() {
                let _ = cuvs_sys::cuvsCagraSearchParamsDestroy(self.search_params);
            }
            if !self.index.is_null() {
                let _ = cuvs_sys::cuvsCagraIndexDestroy(self.index);
            }
            if !self.dataset.is_null() {
                let _ = cuvs_sys::cuvsDatasetDestroy(self.dataset);
            }
        }
    }
}
