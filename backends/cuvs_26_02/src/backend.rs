// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::cuda::{
    CudaEvent, CuvsIvfPqIndex, DeviceTensor, HostTensorView, MatrixBuffer, PinnedHostBuffer,
    PinnedStagingPool, PinnedStagingSlot, RegisteredHostBuffer, check_cuvs,
    copy_tensor_to_host_f32_2d, copy_tensor_to_host_f32_3d, create_index_params,
    cuda_profiler_start, cuda_profiler_stop, destroy_index_params, enable_rmm_pool_from_env,
    ivf_centroids_from_host, make_tensor_view, matrix_from_vectors, pq_codebook_from_host,
};
use arrow::compute::filter;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use cuvs::Resources;
use futures::lock::Mutex;
use futures::{
    FutureExt, SinkExt, StreamExt, TryStreamExt, channel::mpsc, future::LocalBoxFuture, stream,
};
use lance::dataset::Dataset;
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
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// RAII wrapper around `nvtx::range_push!`/`range_pop!`.
///
/// `nvtx`'s push/pop are nested and thread-local (they wrap legacy
/// `nvtxRangePushA`/`nvtxRangePop`), so this guard is only safe for spans
/// that run start-to-finish on one thread with no `.await` in between --
/// pushing on one thread and popping after resuming on a different Tokio
/// worker thread would corrupt that thread's NVTX stack. Use this only for
/// synchronous code; it still correctly closes the range on early return via
/// `?`, unlike calling the macros directly. Spans that cross an `.await` are
/// left uninstrumented by NVTX and rely on the existing `Duration` timers
/// instead.
struct NvtxSpan;

impl NvtxSpan {
    fn new(message: &str) -> Self {
        nvtx::range_push!("{}", message);
        Self
    }
}

impl Drop for NvtxSpan {
    fn drop(&mut self) {
        nvtx::range_pop!();
    }
}

/// Reads this thread's cumulative CPU time (`CLOCK_THREAD_CPUTIME_ID`), which
/// only advances while this specific OS thread is actually executing --
/// unlike wall-clock time, it does not advance while the thread is blocked
/// (e.g. on I/O) or descheduled.
fn thread_cpu_time() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, appropriately-sized out-parameter for
    // `clock_gettime`; `CLOCK_THREAD_CPUTIME_ID` is always available on Linux.
    unsafe {
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts);
    }
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// Wraps a future to measure the CPU time actually spent advancing it,
/// immune to Tokio work-stealing moving the task across OS threads between
/// polls: each individual `poll()` call runs start-to-finish on a single
/// thread (migration only happens *between* polls, while the task is
/// suspended), so summing per-poll `CLOCK_THREAD_CPUTIME_ID` deltas gives a
/// migration-safe total regardless of which thread(s) executed it.
///
/// This exists because OS-level CPU sampling (nsys `--sample`) on this
/// environment lacks real kernel scheduling info (nsys itself reports
/// "Scheduling information is absent... deduced... This is inaccurate"),
/// making its `Running`-vs-blocked thread-state classification untrustworthy
/// for distinguishing genuine CPU-bound work from idle/blocked time.
/// Measuring CPU time in-process via `clock_gettime` sidesteps that entirely.
struct CpuTimedFuture<F> {
    inner: F,
    cpu_time: Duration,
}

impl<F> CpuTimedFuture<F> {
    fn new(inner: F) -> Self {
        Self {
            inner,
            cpu_time: Duration::ZERO,
        }
    }
}

impl<F: Future + Unpin> Future for CpuTimedFuture<F> {
    type Output = (F::Output, Duration);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let before = thread_cpu_time();
        let poll_result = Pin::new(&mut this.inner).poll(cx);
        this.cpu_time += thread_cpu_time().saturating_sub(before);
        match poll_result {
            Poll::Ready(output) => Poll::Ready((output, this.cpu_time)),
            Poll::Pending => Poll::Pending,
        }
    }
}

const PARTITION_ARTIFACT_METADATA_FILE_NAME: &str = "metadata.lance";
const PIPELINE_SLOTS: usize = 2;
const DEFAULT_SCAN_FRAGMENT_READAHEAD: usize = 0;
const DEFAULT_SCAN_IO_BUFFER_SIZE: u64 = 16 * 1024 * 1024 * 1024;
const DEFAULT_SCAN_BATCH_READAHEAD: usize = 32;
const DEFAULT_PREPARE_WORKERS: usize = 1;
const TRAINING_SAMPLE_CHUNK_ROWS: usize = 8 * 1024;
const TRAINING_SAMPLE_BATCH_READAHEAD: usize = 64;
// Conservative default thread count for prefaulting the training-sample
// buffer -- deliberately not scaled to all available cores, so it doesn't
// contend with the scan/decode pipeline's own concurrency (already
// observed using ~35 threads for concurrent reads) while it runs alongside
// it in the background. Overridable via LANCE_CUVS_SAMPLE_PREFAULT_THREADS.
const DEFAULT_SAMPLE_PREFAULT_THREADS: usize = 8;
// Threads per batch for copying a decoded batch into its pinned staging slot
// (LANCE_CUVS_PINNED_STAGING_COPY_THREADS). Each prepare worker copies its own
// batch, so total copy concurrency is up to prepare_workers x this.
const DEFAULT_PINNED_STAGING_COPY_THREADS: usize = 4;

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
    // Pinned staging slot the H2D copy reads from, when staging is enabled.
    // Held until `drain_to_batch` has synchronized on `output_ready` -- see
    // `PinnedStagingSlot`.
    input_staging: Option<PinnedStagingSlot>,
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
    // Already copied into the batch's pinned staging slot; the decoded Arrow
    // buffer was freed in the prepare worker right after the copy.
    Staged {
        rows: usize,
        dimension: usize,
    },
}

impl PreparedMatrix {
    fn rows(&self) -> usize {
        match self {
            Self::F32Arrow { rows, .. } | Self::Staged { rows, .. } => *rows,
            Self::Owned(array) => array.nrows(),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::F32Arrow { dimension, .. } | Self::Staged { dimension, .. } => *dimension,
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
            Self::Staged { .. } => Err(Error::io(
                "staged transform batch has no host matrix; read its pinned staging slot",
            )),
        }
    }
}

struct PreparedTransformBatch {
    row_ids: Arc<dyn Array>,
    matrix: PreparedMatrix,
    input_registration: Option<RegisteredHostBuffer>,
    input_staging: Option<PinnedStagingSlot>,
}

struct DrainedTransformBatch {
    batch: RecordBatch,
    h2d: Duration,
    transform: Duration,
    d2h: Duration,
    sync: Duration,
    event_query: Duration,
    release: Duration,
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
    scan_cpu: Duration,
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
    staging_wait: Duration,
    staging_copy: Duration,
    staged_bytes: usize,
    // Batches that did not fit a staging slot and were copied to the GPU
    // from pageable memory instead.
    staging_fallbacks: usize,
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
            input_staging: None,
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
        let _span = NvtxSpan::new("cuvs/gpu_launch");
        self.launch_inner(trained, stream, prepared)
    }

    fn launch_inner(
        &mut self,
        trained: &TrainedIvfPqIndex,
        stream: cuvs_sys::cudaStream_t,
        prepared: PreparedTransformBatch,
    ) -> Result<LaunchTimings> {
        let mut timings = LaunchTimings::default();
        let code_width = trained.pq_code_width();
        let row_ids = prepared.row_ids;
        let matrix = prepared.matrix;
        let input_staging = prepared.input_staging;
        let rows = matrix.rows();
        let dimension = matrix.dimension();
        let input_slice = match &input_staging {
            Some(staging) => staging.as_slice()?,
            None => matrix.input_slice()?,
        };

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
        // Keep the staging slot until the drain has synchronized: the copy
        // just enqueued is still reading from it.
        self.input_staging = input_staging;
        match matrix {
            PreparedMatrix::F32Arrow { vectors, .. } => {
                self.input_vectors = Some(vectors);
                self.input_matrix = None;
            }
            PreparedMatrix::Owned(array) => {
                self.input_vectors = None;
                self.input_matrix = Some(array);
            }
            PreparedMatrix::Staged { .. } => {
                self.input_vectors = None;
                self.input_matrix = None;
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

    fn drain_to_batch(&mut self, code_width: usize) -> Result<Option<DrainedTransformBatch>> {
        if !self.has_pending_output() {
            return Ok(None);
        }

        let sync_span = NvtxSpan::new("cuvs/gpu_sync");
        let sync_start = Instant::now();
        self.output_ready.synchronize()?;
        let sync = sync_start.elapsed();
        drop(sync_span);
        // `drain_s` previously didn't add up to `sync_s + build_batch_s` (off
        // by ~2.7-3.1s per run, consistently) -- this is the suspected culprit:
        // three separate `cudaEventElapsedTime` driver calls, each with its
        // own real (if individually small) call overhead.
        let event_query_span = NvtxSpan::new("cuvs/gpu_event_query");
        let event_query_start = Instant::now();
        let h2d = self.h2d_done.elapsed_since(&self.h2d_start)?;
        let transform = self.transform_done.elapsed_since(&self.h2d_done)?;
        let d2h = self.output_ready.elapsed_since(&self.transform_done)?;
        let event_query = event_query_start.elapsed();
        drop(event_query_span);
        // event_query_s turned out negligible (~0.001s/run), not the culprit
        // for drain_s's gap. This line drops `input_registration`
        // (`RegisteredHostBuffer`, whose `Drop` calls `cudaHostUnregister` --
        // confirmed non-trivial in an earlier CUDA-API trace, ~620ms/64
        // calls there) plus the input vectors/matrix buffers themselves.
        // With pinned staging enabled, this only returns the staging slot to
        // its pool: there is no registration, and the decoded buffer was
        // already freed in the prepare worker.
        let release_span = NvtxSpan::new("cuvs/gpu_release_input");
        let release_start = Instant::now();
        self.input_registration = None;
        self.input_staging = None;
        self.input_vectors = None;
        self.input_matrix = None;
        let release = release_start.elapsed();
        drop(release_span);
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
            event_query,
            release,
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
    scan_cpu: Duration,
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
    drain_event_query: Duration,
    drain_release: Duration,
    drain_build_batch: Duration,
    register: Duration,
    registered_bytes: usize,
    staging_slots: usize,
    staging_slot_bytes: usize,
    // Background allocation time for all slots (off the critical path);
    // None if it had not finished when the stage ended.
    staging_alloc: Option<Duration>,
    staging_wait: Duration,
    staging_copy: Duration,
    staged_bytes: usize,
    staging_fallbacks: usize,
}

impl ArtifactBuildStats {
    fn merge_scanner(&mut self, scanner: ArtifactScannerStats) {
        self.scanner_tasks += 1;
        self.input_batches += scanner.input_batches;
        self.input_rows += scanner.input_rows;
        self.scan_wait += scanner.scan_wait;
        self.scan_cpu += scanner.scan_cpu;
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
        self.staging_wait += prepare.staging_wait;
        self.staging_copy += prepare.staging_copy;
        self.staged_bytes += prepare.staged_bytes;
        self.staging_fallbacks += prepare.staging_fallbacks;
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
        self.drain_event_query += drained.event_query;
        self.drain_release += drained.release;
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
            "cuVS artifact stages: scanner_tasks={} prepare_workers={} input_batches={} input_rows={} prepared_batches={} prepared_rows={} output_batches={} output_rows={} scan_wait_s={:.3} scan_cpu_s={:.3} raw_send_s={:.3} raw_wait_s={:.3} drain_s={:.3} send_s={:.3} prepare_send_s={:.3} vector_s={:.3} filter_s={:.3} matrix_s={:.3} launch_s={:.3}",
            self.scanner_tasks,
            self.prepare_workers,
            self.input_batches,
            self.input_rows,
            self.prepared_batches,
            self.prepared_rows,
            self.output_batches,
            self.output_rows,
            secs(self.scan_wait),
            secs(self.scan_cpu),
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
        if self.staging_slots > 0 {
            // alloc_s: background allocation of all slots, off the critical
            // path. wait_s: time prepare workers spent blocked waiting for a
            // slot -- including, at startup, for the first slots to be
            // allocated -- so any allocation cost still on the critical path
            // shows up there, not in alloc_s.
            eprintln!(
                "cuVS artifact h2d staging: slots={} slot_mib={:.1} pinned_gib={:.3} alloc_s={} wait_s={:.3} copy_s={:.3} staged_gib={:.3} fallbacks={}",
                self.staging_slots,
                self.staging_slot_bytes as f64 / (1024.0 * 1024.0),
                (self.staging_slots * self.staging_slot_bytes) as f64 / (1024.0 * 1024.0 * 1024.0),
                self.staging_alloc
                    .map_or_else(|| "unfinished".to_string(), |alloc| format!("{:.3}", secs(alloc))),
                secs(self.staging_wait),
                secs(self.staging_copy),
                self.staged_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                self.staging_fallbacks,
            );
        }
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
            "cuVS artifact drain cpu: sync_s={:.3} event_query_s={:.3} release_s={:.3} build_batch_s={:.3}",
            secs(self.drain_sync),
            secs(self.drain_event_query),
            secs(self.drain_release),
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

fn sample_prefault_threads_from_env() -> usize {
    std::env::var("LANCE_CUVS_SAMPLE_PREFAULT_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or(DEFAULT_SAMPLE_PREFAULT_THREADS)
}

fn scan_fragment_readahead_from_env() -> usize {
    std::env::var("LANCE_CUVS_SCAN_FRAGMENT_READAHEAD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SCAN_FRAGMENT_READAHEAD)
}

/// `LANCE_CUVS_PINNED_STAGING=1` (or `true`): copy each decoded batch into a
/// pre-pinned staging slot for the H2D copy, instead of registering the
/// decoded buffer itself with `cudaHostRegister`. Off by default so the two
/// paths can be A/B'd.
fn pinned_staging_enabled_from_env() -> bool {
    matches!(
        std::env::var("LANCE_CUVS_PINNED_STAGING").ok().as_deref(),
        Some("1") | Some("true")
    )
}

fn pinned_staging_slots_from_env(default: usize) -> usize {
    std::env::var("LANCE_CUVS_PINNED_STAGING_SLOTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|slots| *slots > 0)
        .unwrap_or(default)
}

fn pinned_staging_copy_threads_from_env() -> usize {
    std::env::var("LANCE_CUVS_PINNED_STAGING_COPY_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or(DEFAULT_PINNED_STAGING_COPY_THREADS)
}

/// Rows per staging slot: the largest fragment's physical row count, capped
/// at `batch_size`. The scan yields batches within a single fragment, so this
/// is the largest batch it can produce, and sizing to `batch_size` alone
/// would waste pinned memory whenever fragments are smaller (e.g. 1 GiB slots
/// for 610 MiB batches at the default 128Ki-row `batch_size`). Falls back to
/// `batch_size` when any fragment lacks a physical row count. A batch that
/// still does not fit is copied from pageable memory rather than staged.
fn staging_slot_rows(dataset: &Dataset, batch_size: usize) -> usize {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.metadata().physical_rows)
        .try_fold(0usize, |max_rows, rows| rows.map(|rows| max_rows.max(rows)))
        .filter(|rows| *rows > 0)
        .map_or(batch_size, |rows| rows.min(batch_size))
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

    let dimension = infer_dimension(dataset, column)?;
    let ranges = training_sample_ranges(num_rows, sample_rows);
    let expected_rows: usize = ranges.iter().map(|r| (r.end - r.start) as usize).sum();
    let projection = Arc::new(dataset.schema().project(&[column])?);
    let mut stream = dataset.take_scan(
        Box::pin(stream::iter(ranges.into_iter().map(Ok))),
        projection,
        TRAINING_SAMPLE_BATCH_READAHEAD,
    );

    // Prefaulting the destination buffer (forcing its pages to be resident
    // before writing into them) turned out to cost as much as the
    // concat_batches pass it replaced (~4.4s, confirmed single-threaded via
    // profiling) -- but unlike the scan below, that cost is pure CPU/memory
    // work with no I/O wait, so it can run concurrently with the scan
    // instead of serially before or interleaved into it. Farm it out to a
    // background blocking task, parallelized across a few threads via
    // `std::thread::scope` (safe to borrow the task-local buffer across
    // threads because the scope blocks until they all finish), and collect
    // decoded batches into `pending` in the meantime instead of copying
    // them immediately -- `FixedSizeListArray` values are just Arc handles
    // into Arrow's own already-allocated decode buffers, so holding onto a
    // batch's worth of them briefly costs no extra data movement.
    let prefault_handle = tokio::task::spawn_blocking(move || {
        nvtx::mark!("cuvs/sample_prefault_start");
        let mut buf = vec![0f32; expected_rows * dimension];
        if !buf.is_empty() {
            let num_threads = sample_prefault_threads_from_env().min(buf.len()).max(1);
            let chunk_len = buf.len().div_ceil(num_threads);
            std::thread::scope(|scope| {
                for chunk in buf.chunks_mut(chunk_len) {
                    scope.spawn(move || {
                        chunk.iter_mut().for_each(|v| *v = 0.0);
                    });
                }
            });
        }
        nvtx::mark!("cuvs/sample_prefault_end");
        buf
    });

    // Instant markers (not a push/pop range): this loop crosses an `.await`,
    // and a multi-threaded Tokio runtime may resume the task on a different
    // worker thread, which would corrupt a thread-local push/pop stack.
    nvtx::mark!("cuvs/sample_collect_start");
    let collect_start = Instant::now();
    let mut pending: Vec<(usize, FixedSizeListArray)> = Vec::new();
    let mut offset_rows = 0usize;
    while let Some(batch) = stream.try_next().await? {
        let vectors = vector_column_to_fsl(&batch, column)?;
        let rows = vectors.len();
        if rows == 0 {
            continue;
        }
        pending.push((offset_rows, vectors));
        offset_rows += rows;
    }
    let collect = collect_start.elapsed();
    nvtx::mark!("cuvs/sample_collect_end");

    if offset_rows == 0 {
        return Err(Error::invalid_input(
            "cuVS training sample did not return any vectors",
        ));
    }

    nvtx::mark!("cuvs/sample_prefault_join_start");
    let prefault_wait_start = Instant::now();
    let mut sample_values = prefault_handle
        .await
        .map_err(|error| Error::io(format!("sample prefault task failed: {error}")))?;
    let prefault_wait = prefault_wait_start.elapsed();
    nvtx::mark!("cuvs/sample_prefault_join_end");

    // Fully synchronous (no `.await` inside), so a push/pop range is safe
    // here. Should be fast now that the destination pages are resident.
    let copy_span = NvtxSpan::new("cuvs/sample_copy");
    let copy_start = Instant::now();
    for (dst_offset_rows, vectors) in pending {
        let rows = vectors.len();
        let matrix = matrix_from_vectors(&vectors)?;
        let src: &[f32] = match &matrix {
            MatrixBuffer::Borrowed { values, .. } => values,
            MatrixBuffer::Owned(array) => array
                .as_slice_memory_order()
                .ok_or_else(|| Error::io("training sample matrix is not contiguous"))?,
        };
        if src.len() != rows * dimension {
            return Err(Error::io(format!(
                "training sample batch vector width mismatch: expected {} values ({rows} rows x {dimension} dim), got {}",
                rows * dimension,
                src.len()
            )));
        }
        let dst_start = dst_offset_rows * dimension;
        let dst_end = dst_start + src.len();
        if dst_end > sample_values.len() {
            sample_values.resize(dst_end, 0.0);
        }
        sample_values[dst_start..dst_end].copy_from_slice(src);
    }
    let copy = copy_start.elapsed();
    drop(copy_span);

    sample_values.truncate(offset_rows * dimension);
    eprintln!(
        "cuVS train sample collect: total_s={:.3} collect_s={:.3} prefault_wait_s={:.3} copy_s={:.3} rows={}",
        (collect + prefault_wait + copy).as_secs_f64(),
        collect.as_secs_f64(),
        prefault_wait.as_secs_f64(),
        copy.as_secs_f64(),
        offset_rows,
    );

    Ok(FixedSizeListArray::try_new_from_values(
        Float32Array::from(sample_values),
        dimension as i32,
    )?)
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

    // Instant markers, not push/pop: these spans cross an `.await`, and a
    // multi-threaded Tokio runtime may resume the task on a different
    // worker thread, which would corrupt a thread-local push/pop stack.
    // Bounds just the `try_next()` call (matching `scan_wait`'s own scope,
    // not `raw_send`'s downstream-backpressure wait). `scan_cpu` (via
    // `CpuTimedFuture`) measures genuine in-process CPU time spent advancing
    // this same call, migration-safe and independent of nsys's thread-state
    // sampling -- `scan_wait - scan_cpu` is time this call spent blocked
    // (I/O, scheduling) rather than actually computing.
    loop {
        nvtx::mark!("cuvs/scan_batch_start");
        let scan_start = Instant::now();
        let (next, cpu_time) = CpuTimedFuture::new(stream.try_next()).await;
        stats.scan_wait += scan_start.elapsed();
        stats.scan_cpu += cpu_time;
        let Some(batch) = next? else {
            nvtx::mark!("cuvs/scan_batch_end");
            break;
        };
        nvtx::mark!("cuvs/scan_batch_end");
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
    staging: Option<&Arc<PinnedStagingPool>>,
    copy_threads: usize,
    stats: &mut ArtifactPrepareStats,
) -> Result<Option<PreparedTransformBatch>> {
    stats.input_batches += 1;
    stats.input_rows += batch.num_rows();

    // `NvtxSpan` rather than raw `range_push!`/`range_pop!`: this function
    // returns early via `?` on malformed input, and it reruns on a reused
    // `spawn_blocking` thread across batches -- an unbalanced push/pop pair
    // would silently corrupt that thread's NVTX stack for every later batch.
    // `NvtxSpan`'s `Drop` closes the range unconditionally, so it's safe
    // across any exit path (it never crosses an `.await`, so the thread
    // stays fixed for the whole span).
    let vector_span = NvtxSpan::new("cuvs/prepare_vector");
    let vector_start = Instant::now();
    let vectors = vector_column_to_fsl(&batch, column)?;
    let row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| Error::invalid_input(format!("transform batch is missing {ROW_ID}")))?;
    stats.vector += vector_start.elapsed();
    drop(vector_span);

    let filter_span = NvtxSpan::new("cuvs/prepare_filter");
    let filter_start = Instant::now();
    let (filtered_row_ids, filtered_vectors) = if filter_nan {
        let finite_mask = is_finite(&vectors);
        let valid_rows = finite_mask.true_count();
        if valid_rows == 0 {
            stats.filter += filter_start.elapsed();
            drop(filter_span);
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
    drop(filter_span);

    let matrix_span = NvtxSpan::new("cuvs/prepare_matrix");
    let matrix_start = Instant::now();
    let matrix = matrix_from_vectors(&filtered_vectors)?;
    stats.matrix += matrix_start.elapsed();
    drop(matrix_span);

    let (prepared_matrix, input_registration, input_staging) = match matrix {
        MatrixBuffer::Borrowed { values, rows, cols } => match staging {
            Some(pool) if values.len() <= pool.slot_len() => {
                let stage_span = NvtxSpan::new("cuvs/prepare_stage");
                let wait_start = Instant::now();
                let mut slot = pool.acquire()?;
                stats.staging_wait += wait_start.elapsed();
                let copy_start = Instant::now();
                slot.fill_from(values, copy_threads)?;
                stats.staging_copy += copy_start.elapsed();
                stats.staged_bytes += std::mem::size_of_val(values);
                drop(stage_span);
                // `filtered_vectors` -- the decoded buffer -- is not carried
                // forward: it is freed when this function returns, here on the
                // prepare worker, instead of in the pipeline driver's drain.
                (
                    PreparedMatrix::Staged {
                        rows,
                        dimension: cols,
                    },
                    None,
                    Some(slot),
                )
            }
            Some(_) => {
                // Too large for a slot: send it from pageable memory.
                // Deliberately not registered. With staging on, only the
                // dedicated staging slots ever act as pinned H2D sources;
                // `cudaHostRegister` rounds out to whole pages, which is only
                // safe for buffers that own their pages, so keeping arbitrary
                // heap buffers out of it keeps allocator changes safe.
                stats.staging_fallbacks += 1;
                (
                    PreparedMatrix::F32Arrow {
                        vectors: filtered_vectors,
                        rows,
                        dimension: cols,
                    },
                    None,
                    None,
                )
            }
            None => {
                let register_span = NvtxSpan::new("cuvs/prepare_register");
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
                drop(register_span);
                (
                    PreparedMatrix::F32Arrow {
                        vectors: filtered_vectors,
                        rows,
                        dimension: cols,
                    },
                    registration,
                    None,
                )
            }
        },
        MatrixBuffer::Owned(array) => (PreparedMatrix::Owned(array), None, None),
    };

    Ok(Some(PreparedTransformBatch {
        row_ids: filtered_row_ids,
        matrix: prepared_matrix,
        input_registration,
        input_staging,
    }))
}

async fn prepare_transform_batches(
    column: String,
    filter_nan: bool,
    raw_rx: Arc<Mutex<mpsc::Receiver<RecordBatch>>>,
    mut prepared_tx: mpsc::Sender<PreparedTransformBatch>,
    staging: Option<Arc<PinnedStagingPool>>,
    copy_threads: usize,
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
        let staging = staging.clone();
        // `spawn_blocking` also matters for staging: acquiring a slot blocks
        // the thread until one is free.
        let (prepared, batch_stats) = tokio::task::spawn_blocking(move || {
            let mut batch_stats = ArtifactPrepareStats::default();
            let prepared = prepare_transform_batch(
                batch,
                &column,
                filter_nan,
                staging.as_ref(),
                copy_threads,
                &mut batch_stats,
            )?;
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
        stats.staging_wait += batch_stats.staging_wait;
        stats.staging_copy += batch_stats.staging_copy;
        stats.staged_bytes += batch_stats.staged_bytes;
        stats.staging_fallbacks += batch_stats.staging_fallbacks;

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
    let staging = if pinned_staging_enabled_from_env() {
        // Deadlock floor: the driver loop below only drains a transform slot
        // (releasing its staging slot) when the next prepared batch arrives.
        // While it waits for one, the prepared channel is empty and at most
        // PIPELINE_SLOTS staging slots are held by in-flight transforms, so
        // one more guarantees some prepare worker can always acquire a slot.
        let min_slots = PIPELINE_SLOTS + 1;
        let requested = pinned_staging_slots_from_env(prepare_workers + PIPELINE_SLOTS);
        let staging_slots = if requested < min_slots {
            warn!(
                "LANCE_CUVS_PINNED_STAGING_SLOTS={requested} is below the deadlock-free minimum; using {min_slots}"
            );
            min_slots
        } else {
            requested
        };
        let slot_len = staging_slot_rows(dataset, batch_size) * trained.dimension;
        // Returns immediately; slots are allocated on a background thread
        // and handed out as they become ready, so the scanner's first reads
        // overlap the allocation instead of waiting for all of it.
        let pool = PinnedStagingPool::spawn(staging_slots, slot_len)?;
        stats.staging_slots = staging_slots;
        stats.staging_slot_bytes = slot_len * std::mem::size_of::<f32>();
        eprintln!(
            "cuVS artifact prepare: pinned staging enabled: {} slots x {:.1} MiB ({:.2} GiB pinned), allocating in the background",
            staging_slots,
            stats.staging_slot_bytes as f64 / (1024.0 * 1024.0),
            (staging_slots * stats.staging_slot_bytes) as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        Some(pool)
    } else {
        None
    };
    let copy_threads = pinned_staging_copy_threads_from_env();
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
                staging.clone(),
                copy_threads,
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
    if let Some(pool) = &staging {
        stats.staging_alloc = pool.allocation_time();
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
    // Idle time waiting for the GPU pipeline to hand off a finished batch --
    // the mirror of `raw_wait`/`scan_wait` upstream. Previously unmeasured:
    // only time spent inside `append_batch` itself was logged, so there was
    // no way to tell "append is the bottleneck" from "append is idle,
    // starved by GPU/scan/prepare upstream" just from the existing timers.
    let mut recv_wait = Duration::default();
    let mut append_time = Duration::default();
    loop {
        let recv_start = Instant::now();
        let batch = rx.next().await;
        recv_wait += recv_start.elapsed();

        let Some(batch) = batch else {
            break;
        };
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
        "cuVS artifact append task: batches={} rows={} recv_wait_s={:.3} append_s={:.3} finish_s={:.3} files={}",
        batches,
        rows,
        secs(recv_wait),
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
    // Instant markers, not a push/pop range: this call crosses an `.await`,
    // and a multi-threaded Tokio runtime may resume the task on a different
    // worker thread, which would corrupt a thread-local push/pop stack.
    nvtx::mark!("cuvs/sample_training_vectors_start");
    let sample_start = Instant::now();
    let train_vectors = sample_training_vectors(dataset, column, train_rows).await?;
    nvtx::mark!("cuvs/sample_training_vectors_end");
    eprintln!(
        "cuVS train sample time: {:.3}s rows={}",
        sample_start.elapsed().as_secs_f64(),
        train_vectors.len()
    );
    let filter_start = Instant::now();
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
    eprintln!(
        "cuVS train sample filter time: {:.3}s",
        filter_start.elapsed().as_secs_f64()
    );

    let matrix_start = Instant::now();
    let matrix = matrix_from_vectors(&train_vectors)?;
    eprintln!(
        "cuVS train sample matrix time: {:.3}s",
        matrix_start.elapsed().as_secs_f64()
    );
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

    // `cuvsIvfPqBuild` runs k-means clustering plus PQ codebook training on
    // the GPU and, unlike the transform path, is not broken up into
    // enqueue-vs-sync phases here -- this wall-clock duration is the
    // straightforward ground truth for "how long did training actually
    // take on the GPU," not just an enqueue cost.
    let build_start = Instant::now();
    let build_result = check_cuvs(
        unsafe {
            cuvs_sys::cuvsIvfPqBuild(resources.0, params, dataset_tensor.as_mut_ptr(), index.raw)
        },
        "build IVF_PQ index",
    );
    destroy_index_params(params);
    build_result?;
    eprintln!(
        "cuVS cuvsIvfPqBuild time: {:.3}s",
        build_start.elapsed().as_secs_f64()
    );

    // `copy_tensor_to_host_f32_2d`/`_3d` each end with a blocking
    // `resources.sync_stream()`, so these durations also absorb any
    // outstanding GPU work queued by `cuvsIvfPqBuild` that hadn't finished
    // by the time the build call returned.
    let centroid_readback_start = Instant::now();
    let mut centers = make_tensor_view();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexGetCenters(index.raw, centers.as_mut_ptr()) },
        "get IVF centroids",
    )?;
    let ivf_centroids =
        ivf_centroids_from_host(copy_tensor_to_host_f32_2d(&resources, centers.tensor())?)?;
    eprintln!(
        "cuVS IVF centroid readback time: {:.3}s",
        centroid_readback_start.elapsed().as_secs_f64()
    );

    let codebook_readback_start = Instant::now();
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
    eprintln!(
        "cuVS PQ codebook readback time: {:.3}s",
        codebook_readback_start.elapsed().as_secs_f64()
    );

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
    let builder_start = Instant::now();
    let artifact = PartitionArtifactBuilder::try_new(
        artifact_uri,
        trained.num_partitions,
        trained.pq_code_width(),
        storage_options,
    )
    .await?;
    eprintln!(
        "cuVS artifact builder setup time: {:.3}s",
        builder_start.elapsed().as_secs_f64()
    );

    let (mut append_tx, append_rx) = mpsc::channel::<Result<RecordBatch>>(PIPELINE_SLOTS);
    let append_task = tokio::spawn(append_artifact_batches(artifact, append_rx));

    // Scope nsys/nvprof capture (when run with `--capture-range=cudaProfilerApi`)
    // to the transform/append pipeline, excluding the separately-timed k-means
    // training phase that precedes this function.
    if let Err(error) = cuda_profiler_start() {
        drop(append_tx);
        append_task.abort();
        return Err(error);
    }
    let append_start = Instant::now();
    let append_result = append_transformed_batches_to_artifact(
        dataset,
        column,
        trained,
        batch_size,
        filter_nan,
        &mut append_tx,
    )
    .await;
    // Capture elapsed time before `cuda_profiler_stop()`, not after: under
    // nsys with `--capture-range-end=stop`, stopping capture triggers a
    // blocking trace export inside that call, and its duration scales with
    // trace size (observed: ~3s for a small trace, ~17s for one bloated by
    // millions of CUPTI_ACTIVITY_KIND_SYNCHRONIZATION rows from RMM pool
    // instrumentation). Measuring after `cuda_profiler_stop()` would
    // silently fold that nsys-internal export time into a number that's
    // supposed to be pure pipeline wall clock.
    let append_elapsed = append_start.elapsed();
    drop(append_tx);
    let stop_start = Instant::now();
    if let Err(stop_error) = cuda_profiler_stop() {
        warn!("failed to stop CUDA profiler capture: {stop_error}");
    }
    let stop_elapsed = stop_start.elapsed();
    if stop_elapsed > Duration::from_millis(100) {
        eprintln!(
            "cuVS cuda_profiler_stop time: {:.3}s (likely nsys trace export, not pipeline work)",
            stop_elapsed.as_secs_f64()
        );
    }
    if let Err(error) = append_result {
        append_task.abort();
        return Err(error);
    }
    eprintln!(
        "cuVS artifact append_transformed_batches time: {:.3}s",
        append_elapsed.as_secs_f64()
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
