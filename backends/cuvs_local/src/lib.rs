// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! CUDA 12 backend implementation for `pylance-cuvs`.
//!
//! This backend crate is intentionally narrow:
//! - it trains IVF_PQ models with cuVS,
//! - it encodes a Lance dataset into a partition-local artifact,
//! - and it stops before Lance's canonical finalize step.
//!
//! Callers are expected to pass the returned artifact and training outputs back
//! to Lance's own index creation APIs.

mod backend;
mod cuda;
// `pub`, not `mod` -- the standalone `gds_fragment_verify` bin target (a separate crate within
// this package) needs to call into this module's resolver directly, matching the archived 2.0
// prototype's `gds_fragment_verify.rs` precedent.
pub mod gds_layout;
#[cfg(feature = "python")]
mod python;

pub use backend::{
    CuvsVectorBuildBackend, IvfPqBuildParams, PartitionArtifactBuildOutput, TrainedIvfPqIndex,
    VectorBuildBackend, VectorIndexBuildOutput, VectorIndexBuildParams, VectorIndexKind,
    assign_ivf_pq_to_artifact, build_vector_index, train_ivf_pq,
};
