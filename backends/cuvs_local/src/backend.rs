// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cuda::{
    CudaEvent, CuvsIvfPqIndex, DeviceTensor, HostTensorView, MatrixBuffer, PinnedHostBuffer,
    RegisteredHostBuffer, check_cuvs, copy_tensor_to_host_f32_2d, copy_tensor_to_host_f32_3d,
    create_index_params, destroy_index_params, enable_rmm_pool_from_env, ivf_centroids_from_host,
    make_tensor_view, matrix_from_vectors, pq_codebook_from_host,
};
use arrow::compute::{concat_batches, filter};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt8Array, UInt32Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use crate::cuda::Resources;
use futures::lock::Mutex;
use futures::{
    FutureExt, SinkExt, StreamExt, TryStreamExt, channel::mpsc, future::LocalBoxFuture, stream,
};
use crate::gds_layout::resolve_fragment_column;
use lance::dataset::Dataset;
use lance::dataset::fragment::FileFragment;
use lance::index::vector::PartitionArtifactBuilder;
use lance::index::vector::utils::infer_vector_dim;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, ROW_ID, Result};
use lance_index::vector::utils::is_finite;
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_linalg::distance::DistanceType;
use log::warn;
use ndarray::Array2;
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PARTITION_ARTIFACT_METADATA_FILE_NAME: &str = "metadata.lance";
const PIPELINE_SLOTS: usize = 2;
// Deliberately separate from `PIPELINE_SLOTS`: that constant also sizes the non-GDS scan pipeline
// (`scan_transform_batches`) and the artifact-append channel shared by both paths, so widening it
// would double buffers/queue capacity for the CPU path too. This one only sizes the GDS prefetch
// pipeline's device-buffer slots (see `run_gds_prefetch_pipeline`) -- see
// `profiling/GDS_PORTING_PLAN.md` for why 2 wasn't enough to consistently hide the read behind the
// transform.
const GDS_PIPELINE_SLOTS: usize = 4;
const DEFAULT_SCAN_FRAGMENT_READAHEAD: usize = 0;
const DEFAULT_SCAN_IO_BUFFER_SIZE: u64 = 16 * 1024 * 1024 * 1024;
const DEFAULT_SCAN_BATCH_READAHEAD: usize = 32;
const DEFAULT_PREPARE_WORKERS: usize = 1;
const TRAINING_SAMPLE_CHUNK_ROWS: usize = 8 * 1024;
const TRAINING_SAMPLE_BATCH_READAHEAD: usize = 64;

/// A trained cuVS IVF_PQ model that can be reused for artifact builds.
///
/// The training outputs are exposed as Arrow arrays so callers can feed them
/// directly back into Lance's index finalization APIs.
pub struct TrainedIvfPqIndex {
    pub(crate) resources: Resources,
    pub(crate) index: CuvsIvfPqIndex,
    pub(crate) num_partitions: usize,
    pub(crate) dimension: usize,
    pub(crate) num_sub_vectors: usize,
    pub(crate) num_bits: usize,
    pub(crate) metric_type: DistanceType,
    pub(crate) ivf_centroids: FixedSizeListArray,
    pub(crate) pq_codebook: FixedSizeListArray,
}

impl TrainedIvfPqIndex {
    /// Return IVF centroids as a fixed-size list Arrow array.
    pub fn ivf_centroids(&self) -> &FixedSizeListArray {
        &self.ivf_centroids
    }

    /// Return the PQ codebook as a fixed-size list Arrow array.
    pub fn pq_codebook(&self) -> &FixedSizeListArray {
        &self.pq_codebook
    }

    /// Return the number of trained IVF partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Return the encoded PQ byte width, which equals the number of subvectors.
    pub fn pq_code_width(&self) -> usize {
        self.num_sub_vectors
    }

    /// Return the distance metric used during training.
    pub fn metric_type(&self) -> DistanceType {
        self.metric_type
    }

    /// Return the number of bits used per PQ code.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }
}

/// Parameters for a vector index build request handled by this crate.
///
/// The request describes only the backend-owned steps: training and artifact
/// generation. Lance finalization happens outside this crate.
#[derive(Clone)]
pub struct VectorIndexBuildParams {
    pub column: String,
    pub kind: VectorIndexKind,
    pub artifact_uri: String,
    pub batch_size: usize,
    pub filter_nan: bool,
}

/// Supported vector index kinds for the current backend surface.
#[derive(Clone)]
pub enum VectorIndexKind {
    /// Build an IVF_PQ artifact with cuVS.
    IvfPq(IvfPqBuildParams),
}

/// Build parameters for a cuVS IVF_PQ job.
#[derive(Clone)]
pub struct IvfPqBuildParams {
    pub num_partitions: usize,
    pub metric_type: DistanceType,
    pub num_sub_vectors: usize,
    pub sample_rate: usize,
    pub max_iters: usize,
    pub num_bits: usize,
}

/// Backend output that callers can pass to Lance finalization.
pub enum VectorIndexBuildOutput {
    /// A partition-local artifact plus the Arrow-native training outputs used
    /// to build it.
    PartitionArtifact(PartitionArtifactBuildOutput),
}

impl VectorIndexBuildOutput {
    /// Return the output artifact URI.
    pub fn artifact_uri(&self) -> &str {
        match self {
            Self::PartitionArtifact(output) => &output.artifact_uri,
        }
    }

    /// Return the artifact file list relative to the artifact root.
    pub fn files(&self) -> &[String] {
        match self {
            Self::PartitionArtifact(output) => &output.files,
        }
    }

    /// Return trained IVF centroids.
    pub fn ivf_centroids(&self) -> &FixedSizeListArray {
        match self {
            Self::PartitionArtifact(output) => &output.ivf_centroids,
        }
    }

    /// Return the trained PQ codebook.
    pub fn pq_codebook(&self) -> &FixedSizeListArray {
        match self {
            Self::PartitionArtifact(output) => &output.pq_codebook,
        }
    }
}

/// Result of building a partition-local artifact.
pub struct PartitionArtifactBuildOutput {
    pub(crate) artifact_uri: String,
    pub(crate) files: Vec<String>,
    pub(crate) ivf_centroids: FixedSizeListArray,
    pub(crate) pq_codebook: FixedSizeListArray,
}

/// Minimal backend interface for vector build providers.
pub trait VectorBuildBackend {
    /// Execute a backend build request and return a Lance-consumable output.
    fn build<'a>(
        &'a self,
        dataset: &'a Dataset,
        params: VectorIndexBuildParams,
    ) -> LocalBoxFuture<'a, Result<VectorIndexBuildOutput>>;
}

/// cuVS implementation of [`VectorBuildBackend`].
pub struct CuvsVectorBuildBackend;

impl VectorBuildBackend for CuvsVectorBuildBackend {
    fn build<'a>(
        &'a self,
        dataset: &'a Dataset,
        params: VectorIndexBuildParams,
    ) -> LocalBoxFuture<'a, Result<VectorIndexBuildOutput>> {
        async move {
            match params.kind {
                VectorIndexKind::IvfPq(build_params) => {
                    let train_start = Instant::now();
                    let trained = train_ivf_pq(
                        dataset,
                        &params.column,
                        build_params.num_partitions,
                        build_params.metric_type,
                        build_params.num_sub_vectors,
                        build_params.sample_rate,
                        build_params.max_iters,
                        build_params.num_bits,
                        params.filter_nan,
                    )
                    .await?;
                    eprintln!(
                        "cuVS train_ivf_pq time: {:.3}s",
                        train_start.elapsed().as_secs_f64()
                    );
                    let artifact_start = Instant::now();
                    let files = assign_ivf_pq_to_artifact(
                        dataset,
                        &params.column,
                        &trained,
                        &params.artifact_uri,
                        params.batch_size,
                        params.filter_nan,
                        None,
                    )
                    .await?;
                    eprintln!(
                        "cuVS assign_ivf_pq_to_artifact time: {:.3}s files={}",
                        artifact_start.elapsed().as_secs_f64(),
                        files.len()
                    );
                    Ok(VectorIndexBuildOutput::PartitionArtifact(
                        PartitionArtifactBuildOutput {
                            artifact_uri: params.artifact_uri,
                            files,
                            ivf_centroids: trained.ivf_centroids.clone(),
                            pq_codebook: trained.pq_codebook.clone(),
                        },
                    ))
                }
            }
        }
        .boxed_local()
    }
}

fn infer_dimension(dataset: &Dataset, column: &str) -> Result<usize> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "column '{column}' does not exist in dataset schema"
        ))
    })?;
    infer_vector_dim(&field.data_type())
}

fn get_column_from_batch(batch: &RecordBatch, column: &str) -> Result<ArrayRef> {
    if let Some(col) = batch.column_by_name(column) {
        return Ok(col.clone());
    }

    let parts = lance_core::datatypes::parse_field_path(column)
        .map_err(|error| Error::index(format!("failed to parse field path '{column}': {error}")))?;
    if parts.is_empty() {
        return Err(Error::index(format!("invalid empty field path: {column}")));
    }

    let mut current_array = batch
        .column_by_name(&parts[0])
        .ok_or_else(|| {
            Error::index(format!(
                "column '{column}' does not exist in batch (missing root field '{}')",
                parts[0]
            ))
        })?
        .clone();

    for part in &parts[1..] {
        let struct_array = current_array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| {
                Error::index(format!(
                    "cannot access nested field '{part}' in column '{column}': parent is not a struct"
                ))
            })?;
        current_array = struct_array
            .column_by_name(part)
            .ok_or_else(|| {
                Error::index(format!(
                    "nested field '{part}' does not exist in column '{column}'"
                ))
            })?
            .clone();
    }

    Ok(current_array)
}

fn vector_column_to_fsl(batch: &RecordBatch, column: &str) -> Result<FixedSizeListArray> {
    let array = get_column_from_batch(batch, column)?;
    match array.data_type() {
        DataType::FixedSizeList(_, _) => Ok(array.as_fixed_size_list().clone()),
        DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            Ok(list_array.values().as_fixed_size_list().clone())
        }
        _ => Err(Error::index(format!(
            "column '{column}' is not a vector column"
        ))),
    }
}

fn build_partition_batch(
    row_ids: Arc<dyn Array>,
    partitions: &[u32],
    pq_codes: &[u8],
    code_width: usize,
) -> Result<RecordBatch> {
    if pq_codes.len() != partitions.len() * code_width {
        return Err(Error::io(format!(
            "partition artifact batch expects {} PQ codes for {} rows and code width {}, got {}",
            partitions.len() * code_width,
            partitions.len(),
            code_width,
            pq_codes.len()
        )));
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(PART_ID_COLUMN, DataType::UInt32, false),
        Field::new(
            PQ_CODE_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt8, true)),
                code_width as i32,
            ),
            true,
        ),
    ]));
    let pq_codes = FixedSizeListArray::try_new_from_values(
        UInt8Array::from_iter_values(pq_codes.iter().copied()),
        code_width as i32,
    )?;
    Ok(RecordBatch::try_new(
        schema,
        vec![
            row_ids,
            Arc::new(UInt32Array::from_iter_values(partitions.iter().copied())),
            Arc::new(pq_codes),
        ],
    )?)
}

struct TransformSlot {
    input_device: DeviceTensor<f32>,
    labels_host: PinnedHostBuffer<u32>,
    labels_device: DeviceTensor<u32>,
    codes_host: PinnedHostBuffer<u8>,
    codes_device: DeviceTensor<u8>,
    h2d_start: CudaEvent,
    h2d_done: CudaEvent,
    transform_done: CudaEvent,
    output_ready: CudaEvent,
    input_vectors: Option<FixedSizeListArray>,
    input_matrix: Option<Array2<f32>>,
    input_registration: Option<RegisteredHostBuffer>,
    row_ids: Option<Arc<dyn Array>>,
    rows: usize,
}

enum PreparedMatrix {
    F32Arrow {
        vectors: FixedSizeListArray,
        rows: usize,
        dimension: usize,
    },
    Owned(Array2<f32>),
}

impl PreparedMatrix {
    fn rows(&self) -> usize {
        match self {
            Self::F32Arrow { rows, .. } => *rows,
            Self::Owned(array) => array.nrows(),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::F32Arrow { dimension, .. } => *dimension,
            Self::Owned(array) => array.ncols(),
        }
    }

    fn input_slice(&self) -> Result<&[f32]> {
        match self {
            Self::F32Arrow { vectors, .. } => {
                let values = vectors.values().as_primitive::<Float32Type>();
                Ok(values.values().as_ref())
            }
            Self::Owned(array) => array
                .as_slice_memory_order()
                .ok_or_else(|| Error::io("transform matrix is not contiguous")),
        }
    }
}

struct PreparedTransformBatch {
    row_ids: Arc<dyn Array>,
    matrix: PreparedMatrix,
    input_registration: Option<RegisteredHostBuffer>,
}

/// One resolved, ready-to-issue GDS read: `len_bytes` starting at `file_offset` in `path`, landing
/// at `dst_offset_bytes` within a `TransformSlot`'s `input_device` buffer.
///
/// A single batch typically needs more than one of these -- a fragment's row range can span
/// several physical pages, each independently offset-resolved (see the byte-range-resolution
/// design notes in `profiling/GDS_PORTING_PLAN.md`, "Fragment byte-range resolution research").
/// Resolving `path`/`file_offset`/`dst_offset_bytes` from a fragment + row range is Lance-format
/// knowledge that does not yet exist on this branch -- not yet implemented, tracked as an open
/// item in that doc.
struct GdsRead {
    path: String,
    file_offset: u64,
    dst_offset_bytes: usize,
    len_bytes: usize,
}

struct DrainedTransformBatch {
    batch: RecordBatch,
    h2d: Duration,
    transform: Duration,
    d2h: Duration,
    sync: Duration,
    build_batch: Duration,
}

#[derive(Default)]
struct LaunchTimings {
    h2d_enqueue: Duration,
    transform_call: Duration,
    d2h_enqueue: Duration,
}

#[derive(Default)]
struct ArtifactScannerStats {
    input_batches: usize,
    input_rows: usize,
    scan_wait: Duration,
    send: Duration,
}

#[derive(Default)]
struct ArtifactPrepareStats {
    workers: usize,
    input_batches: usize,
    input_rows: usize,
    raw_wait: Duration,
    send: Duration,
    vector: Duration,
    filter: Duration,
    matrix: Duration,
    register: Duration,
    registered_bytes: usize,
}

impl TransformSlot {
    fn try_new(
        resources: &Resources,
        max_rows: usize,
        dimension: usize,
        code_width: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_device: DeviceTensor::try_new(resources, &[max_rows, dimension])?,
            labels_host: PinnedHostBuffer::try_new(max_rows)?,
            labels_device: DeviceTensor::try_new(resources, &[max_rows])?,
            codes_host: PinnedHostBuffer::try_new(max_rows * code_width)?,
            codes_device: DeviceTensor::try_new(resources, &[max_rows, code_width])?,
            h2d_start: CudaEvent::try_new()?,
            h2d_done: CudaEvent::try_new()?,
            transform_done: CudaEvent::try_new()?,
            output_ready: CudaEvent::try_new()?,
            input_vectors: None,
            input_matrix: None,
            input_registration: None,
            row_ids: None,
            rows: 0,
        })
    }

    fn has_pending_output(&self) -> bool {
        self.row_ids.is_some()
    }

    fn launch(
        &mut self,
        trained: &TrainedIvfPqIndex,
        stream: cuvs_sys::cudaStream_t,
        prepared: PreparedTransformBatch,
    ) -> Result<LaunchTimings> {
        let mut timings = LaunchTimings::default();
        let code_width = trained.pq_code_width();
        let row_ids = prepared.row_ids;
        let matrix = prepared.matrix;
        let rows = matrix.rows();
        let dimension = matrix.dimension();
        let input_slice = matrix.input_slice()?;

        self.input_device.set_shape(&[rows, dimension])?;
        self.labels_device.set_shape(&[rows])?;
        self.codes_device.set_shape(&[rows, code_width])?;
        self.rows = rows;
        self.row_ids = Some(row_ids);
        self.input_registration = prepared.input_registration;

        self.h2d_start.record(stream)?;
        let h2d_enqueue_start = Instant::now();
        self.input_device
            .copy_from_host_async(&trained.resources, input_slice)?;
        timings.h2d_enqueue += h2d_enqueue_start.elapsed();
        self.h2d_done.record(stream)?;
        match matrix {
            PreparedMatrix::F32Arrow { vectors, .. } => {
                self.input_vectors = Some(vectors);
                self.input_matrix = None;
            }
            PreparedMatrix::Owned(array) => {
                self.input_vectors = None;
                self.input_matrix = Some(array);
            }
        }
        let transform_call_start = Instant::now();
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsIvfPqTransform(
                    trained.resources.0,
                    trained.index.raw,
                    self.input_device.as_mut_ptr(),
                    self.labels_device.as_mut_ptr(),
                    self.codes_device.as_mut_ptr(),
                )
            },
            "transform vectors with IVF_PQ",
        )?;
        timings.transform_call += transform_call_start.elapsed();
        self.transform_done.record(stream)?;
        let d2h_enqueue_start = Instant::now();
        self.labels_device
            .copy_to_host_async(&trained.resources, self.labels_host.prefix_mut(rows)?)?;
        self.codes_device.copy_to_host_async(
            &trained.resources,
            self.codes_host.prefix_mut(rows * code_width)?,
        )?;
        timings.d2h_enqueue += d2h_enqueue_start.elapsed();
        self.output_ready.record(stream)?;
        Ok(timings)
    }

    /// GDS counterpart to `launch()`, for when `input_device` has already been filled by the
    /// caller -- only sets up shape/bookkeeping and issues the transform + D2H copy. The read
    /// itself happens on a separate OS thread via `tokio::task::spawn_blocking`
    /// (`append_transformed_batches_via_gds`'s `spawn_gds_reads`), run concurrently with the
    /// *previous* fragment's `cuvsIvfPqTransform` call -- which blocks the calling host thread via
    /// its own internal `raft::resource::sync_stream` regardless of what's enqueued on the CUDA
    /// stream, so stream-ordering the read ahead of the transform call (tried first, reverted --
    /// see `profiling/GDS_PORTING_PLAN.md`) can't achieve overlap on its own; OS-thread-level
    /// concurrency can, since the sync only blocks the thread that calls `cuvsIvfPqTransform`, not
    /// a GDS read issued concurrently from a different thread. Since the read isn't stream work
    /// anymore, `h2d_start`/`h2d_done` bracket essentially nothing here (`h2d_s` reports ~0) --
    /// real read cost now shows up as wall-clock time in the caller, not a GPU event.
    fn launch_gds_prefetched(
        &mut self,
        trained: &TrainedIvfPqIndex,
        stream: cuvs_sys::cudaStream_t,
        row_ids: Arc<dyn Array>,
        rows: usize,
        dimension: usize,
    ) -> Result<LaunchTimings> {
        let mut timings = LaunchTimings::default();
        let code_width = trained.pq_code_width();

        self.input_device.set_shape(&[rows, dimension])?;
        self.labels_device.set_shape(&[rows])?;
        self.codes_device.set_shape(&[rows, code_width])?;
        self.rows = rows;
        self.row_ids = Some(row_ids);
        self.input_registration = None;
        self.input_vectors = None;
        self.input_matrix = None;

        self.h2d_start.record(stream)?;
        self.h2d_done.record(stream)?;

        let transform_call_start = Instant::now();
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsIvfPqTransform(
                    trained.resources.0,
                    trained.index.raw,
                    self.input_device.as_mut_ptr(),
                    self.labels_device.as_mut_ptr(),
                    self.codes_device.as_mut_ptr(),
                )
            },
            "transform vectors with IVF_PQ",
        )?;
        timings.transform_call += transform_call_start.elapsed();
        self.transform_done.record(stream)?;
        let d2h_enqueue_start = Instant::now();
        self.labels_device
            .copy_to_host_async(&trained.resources, self.labels_host.prefix_mut(rows)?)?;
        self.codes_device.copy_to_host_async(
            &trained.resources,
            self.codes_host.prefix_mut(rows * code_width)?,
        )?;
        timings.d2h_enqueue += d2h_enqueue_start.elapsed();
        self.output_ready.record(stream)?;
        Ok(timings)
    }

    fn drain_to_batch(&mut self, code_width: usize) -> Result<Option<DrainedTransformBatch>> {
        if !self.has_pending_output() {
            return Ok(None);
        }

        let sync_start = Instant::now();
        self.output_ready.synchronize()?;
        let sync = sync_start.elapsed();
        let h2d = self.h2d_done.elapsed_since(&self.h2d_start)?;
        let transform = self.transform_done.elapsed_since(&self.h2d_done)?;
        let d2h = self.output_ready.elapsed_since(&self.transform_done)?;
        self.input_registration = None;
        self.input_vectors = None;
        self.input_matrix = None;
        let row_ids = self
            .row_ids
            .take()
            .ok_or_else(|| Error::io("transform slot is missing row ids"))?;
        let build_batch_start = Instant::now();
        let batch = build_partition_batch(
            row_ids,
            self.labels_host.prefix(self.rows)?,
            self.codes_host.prefix(self.rows * code_width)?,
            code_width,
        )?;
        let build_batch = build_batch_start.elapsed();
        self.rows = 0;
        Ok(Some(DrainedTransformBatch {
            batch,
            h2d,
            transform,
            d2h,
            sync,
            build_batch,
        }))
    }
}

// Tracks the first-observed and maximum value of a per-batch timing alongside
// its running sum, to distinguish a one-time warm-up cost (first ~= max ~=
// most of the sum, rest near-zero) from a cost that recurs on every batch
// (first ~= max ~= sum / batch_count).
#[derive(Default)]
struct FirstMaxTracker {
    first: Option<Duration>,
    max: Duration,
}

impl FirstMaxTracker {
    fn record(&mut self, value: Duration) {
        if self.first.is_none() {
            self.first = Some(value);
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn first_secs(&self) -> f64 {
        self.first.map(secs).unwrap_or(0.0)
    }

    fn max_secs(&self) -> f64 {
        secs(self.max)
    }
}

#[derive(Default)]
struct ArtifactBuildStats {
    scanner_tasks: usize,
    prepare_workers: usize,
    input_batches: usize,
    input_rows: usize,
    prepared_batches: usize,
    prepared_rows: usize,
    output_batches: usize,
    output_rows: usize,
    scan_wait: Duration,
    raw_send: Duration,
    raw_wait: Duration,
    drain: Duration,
    send: Duration,
    prepare_send: Duration,
    vector: Duration,
    filter: Duration,
    matrix: Duration,
    launch: Duration,
    gpu_h2d: Duration,
    gpu_transform: Duration,
    gpu_d2h: Duration,
    gpu_h2d_first_max: FirstMaxTracker,
    gpu_transform_first_max: FirstMaxTracker,
    gpu_d2h_first_max: FirstMaxTracker,
    launch_h2d_enqueue: Duration,
    launch_transform_call: Duration,
    launch_d2h_enqueue: Duration,
    launch_h2d_enqueue_first_max: FirstMaxTracker,
    launch_transform_call_first_max: FirstMaxTracker,
    launch_d2h_enqueue_first_max: FirstMaxTracker,
    drain_sync: Duration,
    drain_build_batch: Duration,
    register: Duration,
    registered_bytes: usize,
}

impl ArtifactBuildStats {
    fn merge_scanner(&mut self, scanner: ArtifactScannerStats) {
        self.scanner_tasks += 1;
        self.input_batches += scanner.input_batches;
        self.input_rows += scanner.input_rows;
        self.scan_wait += scanner.scan_wait;
        self.raw_send += scanner.send;
    }

    fn merge_prepare(&mut self, prepare: ArtifactPrepareStats) {
        self.prepare_workers += prepare.workers;
        self.prepared_batches += prepare.input_batches;
        self.prepared_rows += prepare.input_rows;
        self.raw_wait += prepare.raw_wait;
        self.prepare_send += prepare.send;
        self.vector += prepare.vector;
        self.filter += prepare.filter;
        self.matrix += prepare.matrix;
        self.register += prepare.register;
        self.registered_bytes += prepare.registered_bytes;
    }

    fn record_output(&mut self, batch: &RecordBatch) {
        self.output_batches += 1;
        self.output_rows += batch.num_rows();
    }

    fn record_drained(&mut self, drained: &DrainedTransformBatch) {
        self.record_output(&drained.batch);
        self.gpu_h2d += drained.h2d;
        self.gpu_transform += drained.transform;
        self.gpu_d2h += drained.d2h;
        self.gpu_h2d_first_max.record(drained.h2d);
        self.gpu_transform_first_max.record(drained.transform);
        self.gpu_d2h_first_max.record(drained.d2h);
        self.drain_sync += drained.sync;
        self.drain_build_batch += drained.build_batch;
    }

    fn record_launch_timings(&mut self, timings: LaunchTimings) {
        self.launch_h2d_enqueue += timings.h2d_enqueue;
        self.launch_transform_call += timings.transform_call;
        self.launch_d2h_enqueue += timings.d2h_enqueue;
        self.launch_h2d_enqueue_first_max.record(timings.h2d_enqueue);
        self.launch_transform_call_first_max.record(timings.transform_call);
        self.launch_d2h_enqueue_first_max.record(timings.d2h_enqueue);
    }

    fn log(&self) {
        eprintln!(
            "cuVS artifact stages: scanner_tasks={} prepare_workers={} input_batches={} input_rows={} prepared_batches={} prepared_rows={} output_batches={} output_rows={} scan_wait_s={:.3} raw_send_s={:.3} raw_wait_s={:.3} drain_s={:.3} send_s={:.3} prepare_send_s={:.3} vector_s={:.3} filter_s={:.3} matrix_s={:.3} launch_s={:.3}",
            self.scanner_tasks,
            self.prepare_workers,
            self.input_batches,
            self.input_rows,
            self.prepared_batches,
            self.prepared_rows,
            self.output_batches,
            self.output_rows,
            secs(self.scan_wait),
            secs(self.raw_send),
            secs(self.raw_wait),
            secs(self.drain),
            secs(self.send),
            secs(self.prepare_send),
            secs(self.vector),
            secs(self.filter),
            secs(self.matrix),
            secs(self.launch),
        );
        eprintln!(
            "cuVS artifact h2d registration: register_s={:.3} registered_gib={:.3}",
            secs(self.register),
            self.registered_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        eprintln!(
            "cuVS artifact gpu events: h2d_s={:.3} transform_s={:.3} d2h_s={:.3}",
            secs(self.gpu_h2d),
            secs(self.gpu_transform),
            secs(self.gpu_d2h),
        );
        eprintln!(
            "cuVS artifact gpu events (first batch / max batch, to isolate one-time warm-up costs): \
             h2d first_s={:.3} max_s={:.3} | transform first_s={:.3} max_s={:.3} | d2h first_s={:.3} max_s={:.3}",
            self.gpu_h2d_first_max.first_secs(),
            self.gpu_h2d_first_max.max_secs(),
            self.gpu_transform_first_max.first_secs(),
            self.gpu_transform_first_max.max_secs(),
            self.gpu_d2h_first_max.first_secs(),
            self.gpu_d2h_first_max.max_secs(),
        );
        eprintln!(
            "cuVS artifact launch cpu: h2d_enqueue_s={:.3} transform_call_s={:.3} d2h_enqueue_s={:.3}",
            secs(self.launch_h2d_enqueue),
            secs(self.launch_transform_call),
            secs(self.launch_d2h_enqueue),
        );
        eprintln!(
            "cuVS artifact launch cpu (first batch / max batch, to isolate one-time warm-up costs): \
             h2d_enqueue first_s={:.3} max_s={:.3} | transform_call first_s={:.3} max_s={:.3} | d2h_enqueue first_s={:.3} max_s={:.3}",
            self.launch_h2d_enqueue_first_max.first_secs(),
            self.launch_h2d_enqueue_first_max.max_secs(),
            self.launch_transform_call_first_max.first_secs(),
            self.launch_transform_call_first_max.max_secs(),
            self.launch_d2h_enqueue_first_max.first_secs(),
            self.launch_d2h_enqueue_first_max.max_secs(),
        );
        eprintln!(
            "cuVS artifact drain cpu: sync_s={:.3} build_batch_s={:.3}",
            secs(self.drain_sync),
            secs(self.drain_build_batch),
        );
        eprintln!("cuVS artifact max rss: {} KiB", max_rss_kib());
    }
}

fn secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn max_rss_kib() -> i64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status == 0 {
        unsafe { usage.assume_init().ru_maxrss }
    } else {
        -1
    }
}

fn prepare_workers_from_env() -> usize {
    std::env::var("LANCE_CUVS_PREPARE_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|workers| *workers > 0)
        .unwrap_or(DEFAULT_PREPARE_WORKERS)
}

/// Opt-in switch for the GDS read path (`append_transformed_batches_via_gds`), off by default --
/// `gds_layout.rs`'s narrow scope (non-nullable `FixedSizeList<f32>`, 2.1 `FullZipLayout` only,
/// single data file per fragment, no deletion vector) means it can reject a dataset the normal
/// CPU path handles fine, so this must stay opt-in until that scope is broadened.
fn gds_read_enabled_from_env() -> bool {
    std::env::var("LANCE_CUVS_GDS_READ").ok().as_deref() == Some("1")
}

/// The dataset's local filesystem root, needed because the GDS read path (`cuvsReadLargeFile`)
/// does a raw file open, unlike `Dataset`'s own `ObjectStore` abstraction which also supports
/// non-local backends a raw open can't reach. This GDS path only ever makes sense for local block
/// storage regardless (that's the entire premise of GDS), so requiring this is a scope match, not
/// a new limitation. `Dataset::uri()` echoes back whatever was passed to `Dataset::open`, which
/// for a local dataset is a plain filesystem path (this doesn't handle a `file://`-prefixed URI
/// differently -- not needed unless/until a caller is found that opens datasets that way).
fn local_dataset_root(dataset: &Dataset) -> PathBuf {
    PathBuf::from(dataset.uri())
}

/// Row IDs for one fragment's entire row range, in physical row order -- metadata-scale (an
/// empty-projection scan, ~8 bytes/row for the `_rowid` column alone), not the bulk vector data
/// the GDS path exists to avoid re-fetching through the normal decode path. Must be in the same
/// physical order `gds_layout::resolve_fragment_column`'s pages are resolved in (page-encounter
/// order, i.e. physical row order within the fragment) for the two to line up correctly --
/// deliberately not passing `scan_in_order(false)` (which the normal CPU path uses for
/// cross-fragment concurrency) for exactly this reason.
async fn fragment_row_ids(dataset: &Dataset, fragment: &FileFragment) -> Result<ArrayRef> {
    let mut scanner = dataset.scan();
    scanner.with_fragments(vec![fragment.metadata().clone()]);
    scanner.empty_project()?;
    scanner.with_row_id();
    let mut stream = scanner.try_into_stream().await?;

    let mut parts: Vec<RecordBatch> = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        parts.push(batch);
    }
    let Some(schema) = parts.first().map(RecordBatch::schema) else {
        return Ok(Arc::new(arrow_array::UInt64Array::from(Vec::<u64>::new())));
    };
    let batch = concat_batches(&schema, &parts)?;
    batch.column_by_name(ROW_ID).cloned().ok_or_else(|| {
        Error::io(format!(
            "fragment {} row-id scan is missing the {ROW_ID} column",
            fragment.metadata().id
        ))
    })
}

fn scan_fragment_readahead_from_env() -> usize {
    std::env::var("LANCE_CUVS_SCAN_FRAGMENT_READAHEAD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SCAN_FRAGMENT_READAHEAD)
}

fn scan_batch_readahead_from_env() -> usize {
    std::env::var("LANCE_CUVS_SCAN_BATCH_READAHEAD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_SCAN_BATCH_READAHEAD)
}

fn training_sample_ranges(num_rows: usize, sample_rows: usize) -> Vec<Range<u64>> {
    let sample_rows = sample_rows.min(num_rows);
    if sample_rows == 0 {
        return Vec::new();
    }
    if sample_rows == num_rows {
        return vec![0..num_rows as u64];
    }

    let chunk_rows = TRAINING_SAMPLE_CHUNK_ROWS.min(sample_rows);
    let num_chunks = sample_rows.div_ceil(chunk_rows);
    let mut remaining = sample_rows;
    let mut ranges = Vec::with_capacity(num_chunks);
    for chunk_idx in 0..num_chunks {
        let rows = chunk_rows.min(remaining);
        remaining -= rows;
        let max_start = num_rows - rows;
        let start = if num_chunks == 1 {
            max_start / 2
        } else {
            ((chunk_idx as u128 * max_start as u128) / (num_chunks - 1) as u128) as usize
        };
        ranges.push(start as u64..(start + rows) as u64);
    }
    ranges
}

async fn sample_training_vectors(
    dataset: &Dataset,
    column: &str,
    sample_rows: usize,
) -> Result<FixedSizeListArray> {
    let num_rows = dataset.count_rows(None).await?;
    if num_rows == 0 {
        return Err(Error::invalid_input(
            "cuVS training requires at least one training vector",
        ));
    }

    let ranges = training_sample_ranges(num_rows, sample_rows);
    let projection = Arc::new(dataset.schema().project(&[column])?);
    let stream = dataset.take_scan(
        Box::pin(stream::iter(ranges.into_iter().map(Ok))),
        projection,
        TRAINING_SAMPLE_BATCH_READAHEAD,
    );
    let batches = stream.try_collect::<Vec<_>>().await?;
    let Some(schema) = batches.first().map(RecordBatch::schema) else {
        return Err(Error::invalid_input(
            "cuVS training sample did not return any vectors",
        ));
    };
    let batch = concat_batches(&schema, &batches)?;
    Ok(vector_column_to_fsl(&batch, column)?)
}

async fn scan_transform_batches(
    dataset: Dataset,
    column: String,
    batch_size: usize,
    filter_nan: bool,
    mut raw_tx: mpsc::Sender<RecordBatch>,
) -> Result<ArtifactScannerStats> {
    let mut scanner = dataset.scan();
    scanner.project(&[&column])?;
    if dataset
        .schema()
        .field(&column)
        .is_some_and(|field| field.nullable && filter_nan)
    {
        scanner.filter(&format!("{column} is not null"))?;
    }
    scanner.with_row_id();
    scanner.batch_size(batch_size);
    scanner.scan_in_order(false);
    scanner.fragment_readahead(scan_fragment_readahead_from_env());
    scanner.batch_readahead(scan_batch_readahead_from_env());
    scanner.io_buffer_size(DEFAULT_SCAN_IO_BUFFER_SIZE);
    let mut stream = scanner.try_into_stream().await?;
    let mut stats = ArtifactScannerStats::default();

    loop {
        let scan_start = Instant::now();
        let Some(batch) = stream.try_next().await? else {
            stats.scan_wait += scan_start.elapsed();
            break;
        };
        stats.scan_wait += scan_start.elapsed();
        stats.input_batches += 1;
        stats.input_rows += batch.num_rows();

        let send_start = Instant::now();
        raw_tx
            .send(batch)
            .await
            .map_err(|error| Error::io(format!("failed to forward raw batch: {error}")))?;
        stats.send += send_start.elapsed();
    }

    Ok(stats)
}

fn prepare_transform_batch(
    batch: RecordBatch,
    column: &str,
    filter_nan: bool,
    stats: &mut ArtifactPrepareStats,
) -> Result<Option<PreparedTransformBatch>> {
    stats.input_batches += 1;
    stats.input_rows += batch.num_rows();

    let vector_start = Instant::now();
    let vectors = vector_column_to_fsl(&batch, column)?;
    let row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| Error::invalid_input(format!("transform batch is missing {ROW_ID}")))?;
    stats.vector += vector_start.elapsed();

    let filter_start = Instant::now();
    let (filtered_row_ids, filtered_vectors) = if filter_nan {
        let finite_mask = is_finite(&vectors);
        let valid_rows = finite_mask.true_count();
        if valid_rows == 0 {
            stats.filter += filter_start.elapsed();
            return Ok(None);
        }
        if valid_rows != vectors.len() {
            warn!(
                "{} vectors are ignored during partition assignment because they are null or non-finite",
                vectors.len() - valid_rows
            );
        }

        let filtered_row_ids = if valid_rows == row_ids.len() {
            row_ids.clone()
        } else {
            filter(row_ids.as_ref(), &finite_mask)?
        };
        let filtered_vectors = if valid_rows == vectors.len() {
            vectors
        } else {
            let vector_column = batch.column_by_name(column).ok_or_else(|| {
                Error::invalid_input(format!(
                    "transform batch is missing vector column '{column}'"
                ))
            })?;
            let field = batch
                .schema()
                .field_with_name(column)
                .map_err(|_| {
                    Error::invalid_input(format!(
                        "transform batch schema is missing field '{column}'"
                    ))
                })?
                .clone();
            let filtered_vectors = filter(vector_column.as_ref(), &finite_mask)?;
            vector_column_to_fsl(
                &RecordBatch::try_new(
                    Arc::new(ArrowSchema::new(vec![field])),
                    vec![filtered_vectors],
                )?,
                column,
            )?
        };
        (filtered_row_ids, filtered_vectors)
    } else {
        (row_ids.clone(), vectors)
    };
    stats.filter += filter_start.elapsed();

    let matrix_start = Instant::now();
    let matrix = matrix_from_vectors(&filtered_vectors)?;
    stats.matrix += matrix_start.elapsed();

    let (prepared_matrix, input_registration) = match matrix {
        MatrixBuffer::Borrowed { values, rows, cols } => {
            let register_start = Instant::now();
            let registration = match RegisteredHostBuffer::try_new(values) {
                Ok(registration) => Some(registration),
                Err(error) => {
                    warn!(
                        "failed to register host vector buffer for CUDA H2D; falling back to pageable memory: {error}"
                    );
                    None
                }
            };
            stats.register += register_start.elapsed();
            stats.registered_bytes += registration
                .as_ref()
                .map(RegisteredHostBuffer::original_bytes)
                .unwrap_or_default();
            (
                PreparedMatrix::F32Arrow {
                    vectors: filtered_vectors,
                    rows,
                    dimension: cols,
                },
                registration,
            )
        }
        MatrixBuffer::Owned(array) => (PreparedMatrix::Owned(array), None),
    };

    Ok(Some(PreparedTransformBatch {
        row_ids: filtered_row_ids,
        matrix: prepared_matrix,
        input_registration,
    }))
}

async fn prepare_transform_batches(
    column: String,
    filter_nan: bool,
    raw_rx: Arc<Mutex<mpsc::Receiver<RecordBatch>>>,
    mut prepared_tx: mpsc::Sender<PreparedTransformBatch>,
) -> Result<ArtifactPrepareStats> {
    let mut stats = ArtifactPrepareStats {
        workers: 1,
        ..Default::default()
    };

    loop {
        let raw_wait_start = Instant::now();
        let batch = {
            let mut raw_rx = raw_rx.lock().await;
            raw_rx.next().await
        };
        stats.raw_wait += raw_wait_start.elapsed();

        let Some(batch) = batch else {
            break;
        };
        let column = column.clone();
        let (prepared, batch_stats) = tokio::task::spawn_blocking(move || {
            let mut batch_stats = ArtifactPrepareStats::default();
            let prepared = prepare_transform_batch(batch, &column, filter_nan, &mut batch_stats)?;
            Ok::<_, Error>((prepared, batch_stats))
        })
        .await
        .map_err(|error| Error::io(format!("prepare transform blocking task failed: {error}")))??;
        stats.input_batches += batch_stats.input_batches;
        stats.input_rows += batch_stats.input_rows;
        stats.vector += batch_stats.vector;
        stats.filter += batch_stats.filter;
        stats.matrix += batch_stats.matrix;
        stats.register += batch_stats.register;
        stats.registered_bytes += batch_stats.registered_bytes;

        let Some(prepared) = prepared else {
            continue;
        };
        let send_start = Instant::now();
        prepared_tx
            .send(prepared)
            .await
            .map_err(|error| Error::io(format!("failed to forward prepared batch: {error}")))?;
        stats.send += send_start.elapsed();
    }

    Ok(stats)
}

async fn append_transformed_batches_to_artifact(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    batch_size: usize,
    filter_nan: bool,
    append_tx: &mut mpsc::Sender<Result<RecordBatch>>,
) -> Result<()> {
    let code_width = trained.pq_code_width();
    let cuda_stream = trained
        .resources
        .get_cuda_stream()
        .map_err(|error| Error::io(error.to_string()))?;
    let mut slots = (0..PIPELINE_SLOTS)
        .map(|_| {
            TransformSlot::try_new(
                &trained.resources,
                batch_size,
                trained.dimension,
                code_width,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut next_slot = 0usize;
    let mut stats = ArtifactBuildStats::default();
    let prepare_workers = prepare_workers_from_env();
    if prepare_workers > 1 {
        eprintln!(
            "cuVS artifact prepare: using {} workers behind a single scanner",
            prepare_workers
        );
    }
    let (raw_tx, raw_rx) = mpsc::channel::<RecordBatch>(prepare_workers);
    let raw_rx = Arc::new(Mutex::new(raw_rx));
    let scanner_task = tokio::spawn(scan_transform_batches(
        dataset.clone(),
        column.to_string(),
        batch_size,
        filter_nan,
        raw_tx,
    ));
    let (prepared_tx, mut prepared_rx) = mpsc::channel::<PreparedTransformBatch>(PIPELINE_SLOTS);
    let prepare_tasks = (0..prepare_workers)
        .map(|_| {
            tokio::spawn(prepare_transform_batches(
                column.to_string(),
                filter_nan,
                raw_rx.clone(),
                prepared_tx.clone(),
            ))
        })
        .collect::<Vec<_>>();
    drop(prepared_tx);

    while let Some(prepared) = prepared_rx.next().await {
        let slot = &mut slots[next_slot];
        let drain_start = Instant::now();
        let transformed = if let Some(transformed) = slot.drain_to_batch(code_width)? {
            stats.drain += drain_start.elapsed();
            stats.record_drained(&transformed);
            Some(transformed)
        } else {
            stats.drain += drain_start.elapsed();
            None
        };

        let launch_start = Instant::now();
        let launch_timings = slot.launch(trained, cuda_stream, prepared)?;
        stats.launch += launch_start.elapsed();
        stats.record_launch_timings(launch_timings);

        if let Some(transformed) = transformed {
            let send_start = Instant::now();
            append_tx
                .send(Ok(transformed.batch))
                .await
                .map_err(|error| {
                    Error::io(format!("failed to forward transformed batch: {error}"))
                })?;
            stats.send += send_start.elapsed();
        }
        next_slot = (next_slot + 1) % PIPELINE_SLOTS;
    }
    for prepare_task in prepare_tasks {
        let prepare_stats = prepare_task
            .await
            .map_err(|error| Error::io(format!("prepare transform task failed: {error}")))??;
        stats.merge_prepare(prepare_stats);
    }
    let scanner_stats = scanner_task
        .await
        .map_err(|error| Error::io(format!("scanner transform task failed: {error}")))??;
    stats.merge_scanner(scanner_stats);

    for slot in &mut slots {
        let drain_start = Instant::now();
        if let Some(transformed) = slot.drain_to_batch(code_width)? {
            stats.drain += drain_start.elapsed();
            stats.record_drained(&transformed);
            let send_start = Instant::now();
            append_tx
                .send(Ok(transformed.batch))
                .await
                .map_err(|error| {
                    Error::io(format!("failed to forward transformed batch: {error}"))
                })?;
            stats.send += send_start.elapsed();
        } else {
            stats.drain += drain_start.elapsed();
        }
    }
    stats.log();
    Ok(())
}

/// Spawns a background OS thread (`tokio::task::spawn_blocking`) that issues `reads` sequentially
/// against `dst_ptr` via the plain synchronous `cuvsReadLargeFile`. Meant to run concurrently with
/// a *different* fragment's blocking `cuvsIvfPqTransform` call -- `cuvsIvfPqTransform` internally
/// calls `raft::resource::sync_stream` unconditionally before returning (confirmed by reading
/// `cpp/src/neighbors/ivf_pq/ivf_pq_transform.cuh:171` in the `cuvs` checkout), so it blocks the
/// calling host thread regardless of CUDA stream ordering -- a stream-ordered async read (tried
/// first, reverted, see `profiling/GDS_PORTING_PLAN.md`) can't overlap with it, but a read issued
/// concurrently from a *different* OS thread can, since the sync only blocks the thread that calls
/// `cuvsIvfPqTransform`.
///
/// `dst_ptr` is cast to a `usize` to cross `spawn_blocking`'s `Send` bound -- sound because it's
/// just an address (CUDA device pointers have no thread affinity) and the caller guarantees no
/// other read/write touches the same bytes until this task's `JoinHandle` is awaited (see
/// `append_transformed_batches_via_gds`'s slot-buffer-reuse-safety reasoning).
fn spawn_gds_reads(
    dst_ptr: *mut std::ffi::c_void,
    reads: Vec<GdsRead>,
    pipeline_start: Instant,
    timeline_tag: String,
) -> tokio::task::JoinHandle<Result<()>> {
    let dst_addr = dst_ptr as usize;
    tokio::task::spawn_blocking(move || {
        eprintln!(
            "[gds-timeline] {timeline_tag} spawn_blocking closure started t={:.3}s",
            pipeline_start.elapsed().as_secs_f64()
        );
        let dst_ptr = dst_addr as *mut std::ffi::c_void;
        for read in &reads {
            let path_c = std::ffi::CString::new(read.path.as_str()).map_err(|error| {
                Error::io(format!("GDS read path contains NUL byte: {error}"))
            })?;
            let dest =
                unsafe { (dst_ptr as *mut u8).add(read.dst_offset_bytes) as *mut std::ffi::c_void };
            check_cuvs(
                unsafe {
                    cuvs_sys::cuvsReadLargeFile(path_c.as_ptr(), dest, read.len_bytes, read.file_offset)
                },
                "read via GDS (background prefetch)",
            )?;
        }
        eprintln!(
            "[gds-timeline] {timeline_tag} spawn_blocking closure done t={:.3}s",
            pipeline_start.elapsed().as_secs_f64()
        );
        Ok(())
    })
}

/// A fragment's GDS reads, already spawned on a background thread (see `spawn_gds_reads`), plus
/// the row-ids/row-count needed to finish launching the transform once the reads land.
struct FragmentPrefetch {
    handle: tokio::task::JoinHandle<Result<()>>,
    row_ids: Arc<dyn Array>,
    rows: usize,
}

/// Resolves `fragment`'s byte ranges (metadata-scale) and row ids, then spawns its GDS reads (see
/// `spawn_gds_reads`) into `dst_addr`/`dst_capacity_bytes`. Returns `None` for an empty (zero-row)
/// fragment -- nothing to read or transform.
///
/// Takes `dst_addr` as a `usize` rather than `*mut c_void`: this function itself awaits
/// (`resolve_fragment_column`/`fragment_row_ids`) before ever touching the destination pointer, so
/// if it took a raw pointer directly, that pointer would be captured in this function's own future
/// state across those awaits, making the future `!Send` -- and it needs to be `Send` to run inside
/// `spawn_fragment_prefetch`'s outer `tokio::spawn`. Reconstituted into a pointer only at the very
/// end, after the last await, right where it's passed to `spawn_gds_reads`.
async fn prefetch_fragment_gds_reads(
    dataset: &Dataset,
    dataset_root: &std::path::Path,
    fragment: &FileFragment,
    column: &str,
    dimension: u64,
    dst_addr: usize,
    dst_capacity_bytes: usize,
    pipeline_start: Instant,
    timeline_tag: String,
) -> Result<Option<FragmentPrefetch>> {
    let (local_path, plans) =
        resolve_fragment_column(dataset, dataset_root, fragment, column, dimension, 4).await?;
    eprintln!(
        "[gds-timeline] {timeline_tag} resolve_fragment_column done t={:.3}s",
        pipeline_start.elapsed().as_secs_f64()
    );
    let rows: u64 = plans.iter().map(|plan| plan.num_rows).sum();
    if rows == 0 {
        return Ok(None);
    }

    let row_ids = fragment_row_ids(dataset, fragment).await?;
    if row_ids.len() != rows as usize {
        return Err(Error::io(format!(
            "fragment {}: row-id count {} does not match GDS-resolved row count {rows}",
            fragment.metadata().id,
            row_ids.len()
        )));
    }

    let path_str = local_path.to_str().ok_or_else(|| {
        Error::io(format!(
            "GDS data file path is not valid UTF-8: {}",
            local_path.display()
        ))
    })?;
    let reads: Vec<GdsRead> = plans
        .iter()
        .map(|plan| {
            let dst_offset_bytes = (plan.row_start * dimension * 4) as usize;
            let len_bytes = plan.byte_len as usize;
            let end = dst_offset_bytes
                .checked_add(len_bytes)
                .ok_or_else(|| Error::io("GDS read destination range overflow"))?;
            if end > dst_capacity_bytes {
                return Err(Error::io(format!(
                    "GDS read destination range {dst_offset_bytes}..{end} exceeds device tensor \
                     capacity {dst_capacity_bytes}"
                )));
            }
            Ok(GdsRead {
                path: path_str.to_string(),
                file_offset: plan.file_offset,
                dst_offset_bytes,
                len_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let dst_ptr = dst_addr as *mut std::ffi::c_void;
    Ok(Some(FragmentPrefetch {
        handle: spawn_gds_reads(dst_ptr, reads, pipeline_start, timeline_tag),
        row_ids,
        rows: rows as usize,
    }))
}

/// Spawns `prefetch_fragment_gds_reads` itself as an independent `tokio::spawn` task, rather than
/// `.await`ing it inline on the caller's task. This matters: `prefetch_fragment_gds_reads` does
/// real async I/O (`resolve_fragment_column`/`fragment_row_ids`, both going through Lance's
/// metadata/scanner machinery) *before* it ever calls `spawn_gds_reads` -- if that resolution were
/// awaited inline (as an earlier version of this code did), it would consume part of the very
/// overlap window the caller is trying to use for something else (the *previous* fragment's
/// blocking `cuvsIvfPqTransform` call), rather than running concurrently with it. Wrapping the
/// whole thing in `tokio::spawn` lets resolution start running as soon as this returns, not only
/// once the caller gets around to awaiting the result.
///
/// Needs owned/cloned inputs (`Dataset`/`FileFragment` are cheap, `Arc`-backed clones -- matches
/// the pattern `scan_transform_batches` already uses elsewhere in this file), since `tokio::spawn`
/// requires `'static`, ruling out borrowing from the caller's stack frame. `dst_ptr` is passed
/// through as the `usize` address `prefetch_fragment_gds_reads` itself takes -- see that
/// function's doc comment for why the raw pointer can't cross an await point.
fn spawn_fragment_prefetch(
    dataset: Dataset,
    dataset_root: PathBuf,
    fragment: FileFragment,
    column: String,
    dimension: u64,
    dst_ptr: *mut std::ffi::c_void,
    dst_capacity_bytes: usize,
    pipeline_start: Instant,
    timeline_tag: String,
) -> tokio::task::JoinHandle<Result<Option<FragmentPrefetch>>> {
    let dst_addr = dst_ptr as usize;
    let spawn_at = pipeline_start.elapsed().as_secs_f64();
    tokio::spawn(async move {
        eprintln!(
            "[gds-timeline] {timeline_tag} prefetch task polled t={:.3}s (spawned at t={:.3}s)",
            pipeline_start.elapsed().as_secs_f64(),
            spawn_at
        );
        prefetch_fragment_gds_reads(
            &dataset,
            &dataset_root,
            &fragment,
            &column,
            dimension,
            dst_addr,
            dst_capacity_bytes,
            pipeline_start,
            timeline_tag,
        )
        .await
    })
}

/// GDS counterpart to `append_transformed_batches_to_artifact`: reads straight from disk into
/// device memory via `cuvsReadLargeFile`, skipping the host scan/decode/matrix-packing pipeline
/// entirely. Overlaps a fragment's read with the *previous* fragment's transform by running it on
/// a background thread (`spawn_gds_reads`) rather than inline -- see that function's doc comment
/// for why stream-ordering alone (tried first, reverted) couldn't achieve this. Processes one
/// whole fragment per slot-launch (not a `batch_size` chunk) -- `gds_layout::
/// resolve_fragment_column` only resolves a fragment's entire row range, not arbitrary
/// sub-ranges, so device buffers are sized to the largest fragment up front rather than a
/// configurable batch size.
///
/// Narrower than the CPU path by design (see `gds_layout.rs`'s module docs): rejects
/// `filter_nan=true` on a nullable column outright (real unsupported case, not silently ignored),
/// and any fragment `resolve_fragment_column` itself rejects (deletion vector present, more than
/// one data file, non-2.1/non-`FullZipLayout` encoding, nullable column) surfaces as a hard error
/// for the whole build rather than a silent fallback -- this path is opt-in
/// (`LANCE_CUVS_GDS_READ=1`) precisely because it doesn't yet cover every dataset shape the CPU
/// path does.
async fn append_transformed_batches_via_gds(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    filter_nan: bool,
    append_tx: &mut mpsc::Sender<Result<RecordBatch>>,
) -> Result<()> {
    if filter_nan
        && dataset
            .schema()
            .field(column)
            .is_some_and(|field| field.nullable)
    {
        return Err(Error::not_supported(
            "GDS read path (LANCE_CUVS_GDS_READ=1) does not support filter_nan=true on a \
             nullable column -- disable filter_nan or use the normal CPU scan path instead",
        ));
    }

    let dataset_root = local_dataset_root(dataset);
    let fragments = dataset.get_fragments();
    if fragments.is_empty() {
        return Ok(());
    }

    let dimension = trained.dimension;
    let code_width = trained.pq_code_width();
    let cuda_stream = trained
        .resources
        .get_cuda_stream()
        .map_err(|error| Error::io(error.to_string()))?;

    let max_fragment_rows = fragments
        .iter()
        .map(|fragment| {
            fragment.metadata().physical_rows.ok_or_else(|| {
                Error::io(format!(
                    "fragment {} is missing physical_rows metadata; GDS read path requires it",
                    fragment.metadata().id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);

    let mut slots = (0..GDS_PIPELINE_SLOTS)
        .map(|_| TransformSlot::try_new(&trained.resources, max_fragment_rows, dimension, code_width))
        .collect::<Result<Vec<_>>>()?;
    // Outer prefetch tasks (see `spawn_fragment_prefetch`) spawned ahead of time for the next
    // fragment that will reuse a given slot, indexed by slot. Only ever populated *after* that
    // slot's current occupant's launch_gds_prefetched call has returned (confirming, via
    // cuvsIvfPqTransform's own internal sync, that input_device is free) -- see the safety comment
    // below, right before it's populated.
    let mut pending: Vec<Option<tokio::task::JoinHandle<Result<Option<FragmentPrefetch>>>>> =
        (0..GDS_PIPELINE_SLOTS).map(|_| None).collect();
    let mut stats = ArtifactBuildStats::default();
    // TEMPORARY diagnostic timeline (see profiling/GDS_PORTING_PLAN.md) -- remove once the
    // overlap question is settled either way. Each line is tagged `frag=N slot=S` so concurrent
    // prefetches can be told apart; note `eprintln!` itself serializes on a global stderr lock, so
    // these calls slightly perturb the very scheduling being measured (a few microseconds of
    // contention per call, negligible next to the ms-scale reads/transforms here, but worth
    // remembering if this is ever used to explain sub-millisecond timing).
    let pipeline_start = Instant::now();

    let result = run_gds_prefetch_pipeline(
        dataset,
        &dataset_root,
        &fragments,
        column,
        trained,
        dimension,
        code_width,
        cuda_stream,
        &mut slots,
        &mut pending,
        &mut stats,
        pipeline_start,
        append_tx,
    )
    .await;

    // `pending`'s JoinHandles are for `spawn_fragment_prefetch` tasks that ultimately call
    // `cuvsReadLargeFile` against a slot's raw device pointer (see `spawn_gds_reads`). Dropping a
    // JoinHandle only *detaches* the task -- it does not stop or wait for it -- so on any exit
    // from `run_gds_prefetch_pipeline` above (success, an error via `?`, or a panic unwinding
    // through it), a task spawned for a slot that hasn't been consumed yet may still be running
    // when `slots` (and the CUDA allocations its `DeviceTensor`s own) would otherwise be dropped
    // below. Awaiting every outstanding handle here, before `slots` goes out of scope, closes that
    // use-after-free window. `abort()` alone is not sufficient: it cannot preempt a
    // `spawn_blocking` closure once its OS thread has actually entered the `cuvsReadLargeFile` FFI
    // call, so the task must still be awaited to completion, not just requested to stop -- `abort`
    // here only short-circuits tasks that haven't started running yet.
    for maybe_handle in pending.drain(..) {
        if let Some(handle) = maybe_handle {
            handle.abort();
            let _ = handle.await;
        }
    }

    result
}

/// The core per-fragment prefetch/transform/drain loop for [`append_transformed_batches_via_gds`],
/// split out so the caller can guarantee every outstanding prefetch task (tracked in `pending`) is
/// joined before its target `slots` are dropped, regardless of how this function returns -- see
/// the safety comment at its call site.
#[allow(clippy::too_many_arguments)]
async fn run_gds_prefetch_pipeline(
    dataset: &Dataset,
    dataset_root: &std::path::Path,
    fragments: &[FileFragment],
    column: &str,
    trained: &TrainedIvfPqIndex,
    dimension: usize,
    code_width: usize,
    cuda_stream: cuvs_sys::cudaStream_t,
    slots: &mut [TransformSlot],
    pending: &mut [Option<tokio::task::JoinHandle<Result<Option<FragmentPrefetch>>>>],
    stats: &mut ArtifactBuildStats,
    pipeline_start: Instant,
    append_tx: &mut mpsc::Sender<Result<RecordBatch>>,
) -> Result<()> {
    for i in 0..fragments.len() {
        let slot_idx = i % GDS_PIPELINE_SLOTS;
        let timeline_tag = || format!("frag={i} slot={slot_idx}");

        // Fragment i's reads: either already running in the background (spawned as an outer
        // tokio::spawn task after fragment i-GDS_PIPELINE_SLOTS's launch, below) or, for this
        // slot's first use, spawn-and-await inline now (no overlap for the first
        // GDS_PIPELINE_SLOTS fragments -- unavoidable startup cost).
        let prefetch = match pending[slot_idx].take() {
            Some(handle) => {
                handle
                    .await
                    .map_err(|error| Error::io(format!("GDS prefetch task panicked: {error}")))??
            }
            None => {
                let dst_addr = slots[slot_idx].input_device.device_ptr() as usize;
                let dst_capacity = slots[slot_idx].input_device.capacity_bytes();
                prefetch_fragment_gds_reads(
                    dataset,
                    &dataset_root,
                    &fragments[i],
                    column,
                    dimension as u64,
                    dst_addr,
                    dst_capacity,
                    pipeline_start,
                    timeline_tag(),
                )
                .await?
            }
        };
        let Some(prefetch) = prefetch else {
            // Zero-row fragment -- nothing was ever read or launched into this slot for this
            // iteration, so there's nothing to drain either; this slot's occupant is unchanged.
            continue;
        };

        eprintln!(
            "[gds-timeline] {} begin wait for prefetch t={:.3}s",
            timeline_tag(),
            pipeline_start.elapsed().as_secs_f64()
        );
        let wait_start = Instant::now();
        prefetch
            .handle
            .await
            .map_err(|error| Error::io(format!("GDS prefetch task panicked: {error}")))??;
        let wait_elapsed = wait_start.elapsed(); // near-zero if overlap is actually working
        eprintln!(
            "[gds-timeline] {} prefetch wait done t={:.3}s",
            timeline_tag(),
            pipeline_start.elapsed().as_secs_f64()
        );

        let slot = &mut slots[slot_idx];
        let drain_start = Instant::now();
        let transformed = if let Some(transformed) = slot.drain_to_batch(code_width)? {
            stats.drain += drain_start.elapsed();
            stats.record_drained(&transformed);
            Some(transformed)
        } else {
            stats.drain += drain_start.elapsed();
            None
        };

        eprintln!(
            "[gds-timeline] {} begin transform t={:.3}s",
            timeline_tag(),
            pipeline_start.elapsed().as_secs_f64()
        );
        let launch_start = Instant::now();
        let mut launch_timings = slot.launch_gds_prefetched(
            trained,
            cuda_stream,
            prefetch.row_ids,
            prefetch.rows,
            dimension,
        )?;
        eprintln!(
            "[gds-timeline] {} transform done t={:.3}s",
            timeline_tag(),
            pipeline_start.elapsed().as_secs_f64()
        );
        launch_timings.h2d_enqueue += wait_elapsed;
        stats.launch += launch_start.elapsed();
        stats.record_launch_timings(launch_timings);

        // Safe only now: launch_gds_prefetched's cuvsIvfPqTransform call just returned, and its
        // own internal raft::resource::sync_stream guarantees the transform kernel -- the last
        // reader of input_device -- has genuinely finished, regardless of CUDA stream state.
        // Spawning any earlier (e.g. before this call) would race the read against fragment i's
        // own not-yet-issued transform.
        if i + GDS_PIPELINE_SLOTS < fragments.len() {
            let dst_ptr = slot.input_device.device_ptr();
            let dst_capacity = slot.input_device.capacity_bytes();
            let next_tag = format!("frag={} slot={slot_idx}", i + GDS_PIPELINE_SLOTS);
            eprintln!(
                "[gds-timeline] {next_tag} spawning prefetch t={:.3}s",
                pipeline_start.elapsed().as_secs_f64()
            );
            pending[slot_idx] = Some(spawn_fragment_prefetch(
                dataset.clone(),
                dataset_root.to_path_buf(),
                fragments[i + GDS_PIPELINE_SLOTS].clone(),
                column.to_string(),
                dimension as u64,
                dst_ptr,
                dst_capacity,
                pipeline_start,
                next_tag,
            ));
        }

        if let Some(transformed) = transformed {
            let send_start = Instant::now();
            append_tx
                .send(Ok(transformed.batch))
                .await
                .map_err(|error| {
                    Error::io(format!("failed to forward transformed batch: {error}"))
                })?;
            stats.send += send_start.elapsed();
        }
    }

    for slot in slots.iter_mut() {
        let drain_start = Instant::now();
        if let Some(transformed) = slot.drain_to_batch(code_width)? {
            stats.drain += drain_start.elapsed();
            stats.record_drained(&transformed);
            let send_start = Instant::now();
            append_tx
                .send(Ok(transformed.batch))
                .await
                .map_err(|error| {
                    Error::io(format!("failed to forward transformed batch: {error}"))
                })?;
            stats.send += send_start.elapsed();
        } else {
            stats.drain += drain_start.elapsed();
        }
    }
    stats.log();
    Ok(())
}

async fn append_artifact_batches(
    mut artifact: PartitionArtifactBuilder,
    mut rx: mpsc::Receiver<Result<RecordBatch>>,
) -> Result<Vec<String>> {
    let mut batches = 0usize;
    let mut rows = 0usize;
    let mut append_time = Duration::default();
    while let Some(batch) = rx.next().await {
        let batch = batch?;
        batches += 1;
        rows += batch.num_rows();
        let append_start = Instant::now();
        artifact.append_batch(&batch).await?;
        append_time += append_start.elapsed();
    }

    let finish_start = Instant::now();
    let files = artifact
        .finish(PARTITION_ARTIFACT_METADATA_FILE_NAME, None)
        .await?;
    let finish_time = finish_start.elapsed();
    eprintln!(
        "cuVS artifact append task: batches={} rows={} append_s={:.3} finish_s={:.3} files={}",
        batches,
        rows,
        secs(append_time),
        secs(finish_time),
        files.len()
    );
    Ok(files)
}

/// Train an IVF_PQ model with cuVS and return Arrow-native training outputs.
///
/// This function performs only the backend-owned training step. The returned
/// value can be reused across multiple artifact builds.
///
/// # Errors
///
/// Returns an error when the input column is missing, empty, incompatible with
/// cuVS, or when CUDA/cuVS reports a build failure.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::Dataset;
/// # use lance_cuvs::train_ivf_pq;
/// # use lance_linalg::distance::DistanceType;
/// # async fn demo(dataset: &Dataset) -> lance_core::Result<()> {
/// let training = train_ivf_pq(
///     dataset,
///     "vector",
///     256,
///     DistanceType::L2,
///     16,
///     256,
///     50,
///     8,
///     true,
/// )
/// .await?;
/// assert_eq!(training.num_partitions(), 256);
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub async fn train_ivf_pq(
    dataset: &Dataset,
    column: &str,
    num_partitions: usize,
    metric_type: DistanceType,
    num_sub_vectors: usize,
    sample_rate: usize,
    max_iters: usize,
    num_bits: usize,
    filter_nan: bool,
) -> Result<TrainedIvfPqIndex> {
    if num_bits != 8 {
        return Err(Error::not_supported(
            "cuVS IVF_PQ currently supports only num_bits=8",
        ));
    }

    let dimension = infer_dimension(dataset, column)?;
    if dimension % num_sub_vectors != 0 {
        return Err(Error::invalid_input(format!(
            "cuVS IVF_PQ requires vector dimension {} to be divisible by num_sub_vectors {}",
            dimension, num_sub_vectors
        )));
    }

    let train_rows = (num_partitions * sample_rate).max(256 * 256).max(1);
    let sample_start = Instant::now();
    let train_vectors = sample_training_vectors(dataset, column, train_rows).await?;
    eprintln!(
        "cuVS train sample time: {:.3}s rows={}",
        sample_start.elapsed().as_secs_f64(),
        train_vectors.len()
    );
    let train_vectors = if filter_nan {
        let mask = is_finite(&train_vectors);
        let filtered = filter(&train_vectors, &mask)?.as_fixed_size_list().clone();
        filtered.slice(0, train_rows.min(filtered.len()))
    } else {
        train_vectors
    };
    if train_vectors.is_empty() {
        return Err(Error::invalid_input(
            "cuVS training requires at least one non-null training vector",
        ));
    }

    let matrix = matrix_from_vectors(&train_vectors)?;
    enable_rmm_pool_from_env()?;
    let resources = Resources::new().map_err(|error| Error::io(error.to_string()))?;
    let index = CuvsIvfPqIndex::try_new()?;
    let params = create_index_params(
        metric_type,
        num_partitions,
        num_sub_vectors,
        sample_rate,
        max_iters,
        num_bits,
    )?;
    let matrix_view = matrix.view()?;
    let mut dataset_tensor = HostTensorView::try_new::<f32>(
        &[matrix_view.nrows(), matrix_view.ncols()],
        matrix_view.as_ptr() as *mut std::ffi::c_void,
    );

    let build_result = check_cuvs(
        unsafe {
            cuvs_sys::cuvsIvfPqBuild(resources.0, params, dataset_tensor.as_mut_ptr(), index.raw)
        },
        "build IVF_PQ index",
    );
    destroy_index_params(params);
    build_result?;

    let mut centers = make_tensor_view();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexGetCenters(index.raw, centers.as_mut_ptr()) },
        "get IVF centroids",
    )?;
    let ivf_centroids =
        ivf_centroids_from_host(copy_tensor_to_host_f32_2d(&resources, centers.tensor())?)?;

    let mut pq_centers = make_tensor_view();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexGetPqCenters(index.raw, pq_centers.as_mut_ptr()) },
        "get PQ codebook",
    )?;
    let (pq_codebook_values, pq_codebook_shape) =
        copy_tensor_to_host_f32_3d(&resources, pq_centers.tensor())?;
    let pq_codebook = pq_codebook_from_host(
        pq_codebook_values,
        pq_codebook_shape,
        num_sub_vectors,
        dimension,
        num_bits,
    )?;

    Ok(TrainedIvfPqIndex {
        resources,
        index,
        num_partitions,
        dimension,
        num_sub_vectors,
        num_bits,
        metric_type,
        ivf_centroids,
        pq_codebook,
    })
}

/// Build a partition-local IVF_PQ artifact from a trained model.
///
/// The output artifact is intended to be consumed by Lance's
/// `precomputed_partition_artifact_uri` finalization path.
///
/// # Errors
///
/// Returns an error when scanning, encoding, or writing the artifact fails.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::Dataset;
/// # use lance_cuvs::{assign_ivf_pq_to_artifact, train_ivf_pq};
/// # use lance_linalg::distance::DistanceType;
/// # async fn demo(dataset: &Dataset) -> lance_core::Result<()> {
/// let training = train_ivf_pq(
///     dataset,
///     "vector",
///     256,
///     DistanceType::L2,
///     16,
///     256,
///     50,
///     8,
///     true,
/// )
/// .await?;
/// let files = assign_ivf_pq_to_artifact(
///     dataset,
///     "vector",
///     &training,
///     "/tmp/lance-cuvs-artifact",
///     1024 * 128,
///     true,
/// )
/// .await?;
/// assert!(!files.is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn assign_ivf_pq_to_artifact(
    dataset: &Dataset,
    column: &str,
    trained: &TrainedIvfPqIndex,
    artifact_uri: &str,
    batch_size: usize,
    filter_nan: bool,
    storage_options: Option<&HashMap<String, String>>,
) -> Result<Vec<String>> {
    let artifact = PartitionArtifactBuilder::try_new(
        artifact_uri,
        trained.num_partitions,
        trained.pq_code_width(),
        storage_options,
    )
    .await?;

    let (mut append_tx, append_rx) = mpsc::channel::<Result<RecordBatch>>(PIPELINE_SLOTS);
    let append_task = tokio::spawn(append_artifact_batches(artifact, append_rx));

    let use_gds = gds_read_enabled_from_env();
    let append_start = Instant::now();
    let append_result = if use_gds {
        eprintln!("cuVS artifact build: LANCE_CUVS_GDS_READ=1, using the GDS read path");
        append_transformed_batches_via_gds(dataset, column, trained, filter_nan, &mut append_tx)
            .await
    } else {
        append_transformed_batches_to_artifact(
            dataset,
            column,
            trained,
            batch_size,
            filter_nan,
            &mut append_tx,
        )
        .await
    };
    drop(append_tx);
    if let Err(error) = append_result {
        append_task.abort();
        return Err(error);
    }
    eprintln!(
        "cuVS artifact append_transformed_batches time: {:.3}s",
        append_start.elapsed().as_secs_f64()
    );
    let files = append_task
        .await
        .map_err(|error| Error::io(format!("partition artifact append task failed: {error}")))??;
    Ok(files)
}

/// Execute a full backend build request.
///
/// This convenience entrypoint wraps training and artifact construction behind
/// [`VectorBuildBackend`]. It still stops before Lance finalization.
///
/// # Example
///
/// ```no_run
/// # use lance::dataset::Dataset;
/// # use lance_cuvs::{
/// #     build_vector_index, IvfPqBuildParams, VectorIndexBuildParams, VectorIndexKind,
/// # };
/// # use lance_linalg::distance::DistanceType;
/// # async fn demo(dataset: &Dataset) -> lance_core::Result<()> {
/// let output = build_vector_index(
///     dataset,
///     VectorIndexBuildParams {
///         column: "vector".to_string(),
///         kind: VectorIndexKind::IvfPq(IvfPqBuildParams {
///             num_partitions: 256,
///             metric_type: DistanceType::L2,
///             num_sub_vectors: 16,
///             sample_rate: 256,
///             max_iters: 50,
///             num_bits: 8,
///         }),
///         artifact_uri: "/tmp/lance-cuvs-artifact".to_string(),
///         batch_size: 1024 * 128,
///         filter_nan: true,
///     },
/// )
/// .await?;
/// assert!(!output.files().is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn build_vector_index(
    dataset: &Dataset,
    params: VectorIndexBuildParams,
) -> Result<VectorIndexBuildOutput> {
    CuvsVectorBuildBackend.build(dataset, params).await
}
