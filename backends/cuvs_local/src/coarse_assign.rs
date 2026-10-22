// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Approximate coarse assignment for the IVF_PQ transform, as an alternative
//! to cuVS's exhaustive `top_1_nn`. Selected with `LANCE_CUVS_ASSIGN`:
//! `cagra` ([`crate::cagra_assign`]), `ivf_flat` ([`crate::ivf_assign`]), or
//! unset / `exact` for the default. The assigner runs once per batch slot,
//! ahead of `cuvsIvfPqTransformWithLabels`.

use crate::cagra_assign::{CagraAssignConfig, CagraAssigner};
use crate::cuda::{DeviceTensor, Resources};
use crate::ivf_assign::{IvfFlatAssignConfig, IvfFlatAssigner};
use arrow_array::FixedSizeListArray;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

pub(crate) enum CoarseAssigner {
    Cagra(CagraAssigner),
    IvfFlat(IvfFlatAssigner),
}

/// Device scratch an assigner needs next to the slot's labels.
pub(crate) struct AssignScratch {
    /// Top-1 distances, `[rows, 1]`.
    pub(crate) distances: DeviceTensor<f32>,
    /// Top-1 ids as i64, `[rows, 1]`; only IVF-flat produces these.
    pub(crate) neighbors_i64: Option<DeviceTensor<i64>>,
}

impl AssignScratch {
    pub(crate) fn try_new(
        resources: &Resources,
        assigner: &CoarseAssigner,
        max_rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            distances: DeviceTensor::try_new(resources, &[max_rows, 1])?,
            neighbors_i64: match assigner {
                CoarseAssigner::IvfFlat(_) => Some(DeviceTensor::try_new(resources, &[max_rows, 1])?),
                CoarseAssigner::Cagra(_) => None,
            },
        })
    }

    pub(crate) fn set_rows(&mut self, rows: usize) -> Result<()> {
        self.distances.set_shape(&[rows, 1])?;
        if let Some(neighbors) = self.neighbors_i64.as_mut() {
            neighbors.set_shape(&[rows, 1])?;
        }
        Ok(())
    }
}

impl CoarseAssigner {
    /// `Ok(None)` unless `LANCE_CUVS_ASSIGN` selects an approximate assigner.
    pub(crate) fn from_env(
        resources: &Resources,
        centroids: &FixedSizeListArray,
        metric_type: DistanceType,
    ) -> Result<Option<Self>> {
        match std::env::var("LANCE_CUVS_ASSIGN").ok().as_deref() {
            None | Some("") | Some("default") | Some("exact") => Ok(None),
            Some("cagra") => {
                let config = CagraAssignConfig::from_env()?;
                Ok(Some(Self::Cagra(CagraAssigner::build(
                    resources,
                    centroids,
                    metric_type,
                    &config,
                )?)))
            }
            Some("ivf_flat") => {
                let config = IvfFlatAssignConfig::from_env()?;
                Ok(Some(Self::IvfFlat(IvfFlatAssigner::build(
                    resources,
                    centroids,
                    metric_type,
                    &config,
                )?)))
            }
            Some(other) => Err(Error::invalid_input(format!(
                "invalid LANCE_CUVS_ASSIGN '{other}': expected 'cagra', 'ivf_flat' or 'exact'"
            ))),
        }
    }

    /// Writes each query row's nearest-centroid id into `labels` (`[rows]`).
    pub(crate) fn assign(
        &self,
        resources: &Resources,
        queries: *mut cuvs_sys::DLManagedTensor,
        labels: &mut DeviceTensor<u32>,
        scratch: &mut AssignScratch,
        rows: usize,
    ) -> Result<()> {
        match (self, scratch.neighbors_i64.as_mut()) {
            (Self::Cagra(cagra), _) => {
                let mut labels_2d = labels.view_with_shape(&[rows, 1])?;
                cagra.assign(resources, queries, &mut labels_2d, scratch.distances.as_mut_ptr())
            }
            (Self::IvfFlat(ivf), Some(neighbors)) => ivf.assign(
                resources,
                queries,
                neighbors,
                &mut scratch.distances,
                labels,
                rows,
            ),
            (Self::IvfFlat(_), None) => Err(Error::io("IVF-flat assignment scratch is missing")),
        }
    }
}
