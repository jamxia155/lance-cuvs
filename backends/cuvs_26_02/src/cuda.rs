// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray};
use arrow_schema::DataType;
use cuvs::Resources;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use ndarray::{Array2, ArrayView2};
use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

pub(crate) type CudaEventHandle = *mut c_void;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cuvs_sys::cudaError_t;
    fn cudaFreeHost(ptr: *mut c_void) -> cuvs_sys::cudaError_t;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> cuvs_sys::cudaError_t;
    fn cudaHostUnregister(ptr: *mut c_void) -> cuvs_sys::cudaError_t;
    fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cuvs_sys::cudaMemcpyKind,
        stream: cuvs_sys::cudaStream_t,
    ) -> cuvs_sys::cudaError_t;
    fn cudaEventCreate(event: *mut CudaEventHandle) -> cuvs_sys::cudaError_t;
    fn cudaEventDestroy(event: CudaEventHandle) -> cuvs_sys::cudaError_t;
    fn cudaEventRecord(
        event: CudaEventHandle,
        stream: cuvs_sys::cudaStream_t,
    ) -> cuvs_sys::cudaError_t;
    fn cudaEventSynchronize(event: CudaEventHandle) -> cuvs_sys::cudaError_t;
    fn cudaEventElapsedTime(
        ms: *mut f32,
        start: CudaEventHandle,
        end: CudaEventHandle,
    ) -> cuvs_sys::cudaError_t;
    fn cudaProfilerStart() -> cuvs_sys::cudaError_t;
    fn cudaProfilerStop() -> cuvs_sys::cudaError_t;
}

/// Start `nsys`/`nvprof` capture when run under `--capture-range=cudaProfilerApi`.
/// A no-op outside that mode.
pub(crate) fn cuda_profiler_start() -> Result<()> {
    check_cuda(unsafe { cudaProfilerStart() }, "start CUDA profiler capture")
}

/// Stop `nsys`/`nvprof` capture started by [`cuda_profiler_start`].
pub(crate) fn cuda_profiler_stop() -> Result<()> {
    check_cuda(unsafe { cudaProfilerStop() }, "stop CUDA profiler capture")
}

pub(crate) struct CuvsIvfPqIndex {
    pub(crate) raw: cuvs_sys::cuvsIvfPqIndex_t,
}

impl CuvsIvfPqIndex {
    pub(crate) fn try_new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsIvfPqIndexCreate(&mut raw) },
            "create IVF_PQ index",
        )?;
        Ok(Self { raw })
    }
}

impl Drop for CuvsIvfPqIndex {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cuvs_sys::cuvsIvfPqIndexDestroy(self.raw) };
        }
    }
}

pub(crate) enum MatrixBuffer<'a> {
    Borrowed {
        values: &'a [f32],
        rows: usize,
        cols: usize,
    },
    Owned(Array2<f32>),
}

impl MatrixBuffer<'_> {
    pub(crate) fn view(&self) -> Result<ArrayView2<'_, f32>> {
        match self {
            Self::Borrowed { values, rows, cols } => ArrayView2::from_shape((*rows, *cols), values)
                .map_err(|error| {
                    Error::io(format!("failed to create borrowed matrix view: {error}"))
                }),
            Self::Owned(array) => Ok(array.view()),
        }
    }
}

pub(crate) struct HostTensorView {
    shape: Vec<i64>,
    tensor: cuvs_sys::DLManagedTensor,
}

impl HostTensorView {
    pub(crate) fn try_new<T: DlElement>(shape: &[usize], data: *mut c_void) -> Self {
        let shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
        let tensor = cuvs_sys::DLManagedTensor {
            dl_tensor: cuvs_sys::DLTensor {
                data,
                device: cuvs_sys::DLDevice {
                    device_type: cuvs_sys::DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: shape.len() as i32,
                dtype: T::dl_dtype(),
                shape: shape.as_ptr() as *mut i64,
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };
        Self { shape, tensor }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        debug_assert_eq!(self.shape.len(), self.tensor.dl_tensor.ndim as usize);
        &mut self.tensor
    }

    pub(crate) fn tensor(&self) -> &cuvs_sys::DLManagedTensor {
        &self.tensor
    }
}

pub(crate) trait DlElement: Copy + Default {
    fn dl_dtype() -> cuvs_sys::DLDataType;
}

impl DlElement for f32 {
    fn dl_dtype() -> cuvs_sys::DLDataType {
        cuvs_sys::DLDataType {
            code: cuvs_sys::DLDataTypeCode::kDLFloat as u8,
            bits: 32,
            lanes: 1,
        }
    }
}

impl DlElement for u8 {
    fn dl_dtype() -> cuvs_sys::DLDataType {
        cuvs_sys::DLDataType {
            code: cuvs_sys::DLDataTypeCode::kDLUInt as u8,
            bits: 8,
            lanes: 1,
        }
    }
}

impl DlElement for u32 {
    fn dl_dtype() -> cuvs_sys::DLDataType {
        cuvs_sys::DLDataType {
            code: cuvs_sys::DLDataTypeCode::kDLUInt as u8,
            bits: 32,
            lanes: 1,
        }
    }
}

pub(crate) struct DeviceTensor<T: DlElement> {
    shape: Vec<i64>,
    tensor: cuvs_sys::DLManagedTensor,
    capacity_bytes: usize,
    resources: cuvs_sys::cuvsResources_t,
    _marker: PhantomData<T>,
}

impl<T: DlElement> DeviceTensor<T> {
    pub(crate) fn try_new(resources: &Resources, shape: &[usize]) -> Result<Self> {
        let capacity_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let mut data = ptr::null_mut();
        check_cuvs(
            unsafe { cuvs_sys::cuvsRMMAlloc(resources.0, &mut data, capacity_bytes) },
            "allocate device tensor",
        )?;
        let shape = shape.iter().map(|dim| *dim as i64).collect::<Vec<_>>();
        let tensor = cuvs_sys::DLManagedTensor {
            dl_tensor: cuvs_sys::DLTensor {
                data,
                device: cuvs_sys::DLDevice {
                    device_type: cuvs_sys::DLDeviceType::kDLCUDA,
                    device_id: 0,
                },
                ndim: shape.len() as i32,
                dtype: T::dl_dtype(),
                shape: shape.as_ptr() as *mut i64,
                strides: ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: ptr::null_mut(),
            deleter: None,
        };
        Ok(Self {
            shape,
            tensor,
            capacity_bytes,
            resources: resources.0,
            _marker: PhantomData,
        })
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut cuvs_sys::DLManagedTensor {
        debug_assert_eq!(self.shape.len(), self.tensor.dl_tensor.ndim as usize);
        &mut self.tensor
    }

    pub(crate) fn set_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.len() != self.shape.len() {
            return Err(Error::io(format!(
                "device tensor rank mismatch: expected {}, got {}",
                self.shape.len(),
                shape.len()
            )));
        }
        let required_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        if required_bytes > self.capacity_bytes {
            return Err(Error::io(format!(
                "device tensor capacity {} bytes is smaller than requested shape {:?} ({} bytes)",
                self.capacity_bytes, shape, required_bytes
            )));
        }
        for (dst, src) in self.shape.iter_mut().zip(shape) {
            *dst = *src as i64;
        }
        Ok(())
    }

    fn current_len(&self) -> usize {
        self.shape.iter().map(|dim| *dim as usize).product()
    }

    fn current_bytes(&self) -> usize {
        self.current_len() * std::mem::size_of::<T>()
    }

    pub(crate) fn copy_from_host_async(&mut self, resources: &Resources, src: &[T]) -> Result<()> {
        let expected_len = self.current_len();
        if src.len() != expected_len {
            return Err(Error::io(format!(
                "device tensor copy expects {expected_len} elements, got {}",
                src.len()
            )));
        }
        check_cuda(
            unsafe {
                cuvs_sys::cudaMemcpyAsync(
                    self.tensor.dl_tensor.data,
                    src.as_ptr() as *const _,
                    self.current_bytes(),
                    cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                    resources
                        .get_cuda_stream()
                        .map_err(|e| Error::io(e.to_string()))?,
                )
            },
            "copy host tensor to device",
        )
    }

    pub(crate) fn copy_to_host_async(&self, resources: &Resources, dst: &mut [T]) -> Result<()> {
        let expected_len = self.current_len();
        if dst.len() != expected_len {
            return Err(Error::io(format!(
                "device tensor copy expects destination length {expected_len}, got {}",
                dst.len()
            )));
        }
        check_cuda(
            unsafe {
                cuvs_sys::cudaMemcpyAsync(
                    dst.as_mut_ptr() as *mut _,
                    self.tensor.dl_tensor.data,
                    self.current_bytes(),
                    cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                    resources
                        .get_cuda_stream()
                        .map_err(|e| Error::io(e.to_string()))?,
                )
            },
            "copy device tensor to host",
        )
    }
}

impl<T: DlElement> Drop for DeviceTensor<T> {
    fn drop(&mut self) {
        if !self.tensor.dl_tensor.data.is_null() {
            let _ = unsafe {
                cuvs_sys::cuvsRMMFree(
                    self.resources,
                    self.tensor.dl_tensor.data,
                    self.capacity_bytes,
                )
            };
        }
    }
}

pub(crate) struct RegisteredHostBuffer {
    ptr: *mut c_void,
    original_bytes: usize,
}

// CUDA host registration owns a process-local address range; unregistering it
// from the consumer task is safe because CUDA runtime calls are thread-safe.
unsafe impl Send for RegisteredHostBuffer {}

impl RegisteredHostBuffer {
    pub(crate) fn try_new<T>(slice: &[T]) -> Result<Self> {
        let original_bytes = std::mem::size_of_val(slice);
        if original_bytes == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                original_bytes: 0,
            });
        }

        let page_size = page_size()?;
        let start = slice.as_ptr() as usize;
        let end = start
            .checked_add(original_bytes)
            .ok_or_else(|| Error::io("registered host buffer size overflow"))?;
        let aligned_start = start & !(page_size - 1);
        let aligned_end = end
            .checked_add(page_size - 1)
            .ok_or_else(|| Error::io("registered host buffer alignment overflow"))?
            & !(page_size - 1);
        let bytes = aligned_end - aligned_start;
        let ptr = aligned_start as *mut c_void;

        check_cuda(
            unsafe { cudaHostRegister(ptr, bytes, 0) },
            "register host buffer",
        )?;
        Ok(Self {
            ptr,
            original_bytes,
        })
    }

    pub(crate) fn original_bytes(&self) -> usize {
        self.original_bytes
    }
}

impl Drop for RegisteredHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudaHostUnregister(self.ptr) };
        }
    }
}

fn page_size() -> Result<usize> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(Error::io("failed to resolve system page size"));
    }
    let page_size = page_size as usize;
    if !page_size.is_power_of_two() {
        return Err(Error::io(format!(
            "system page size {page_size} is not a power of two"
        )));
    }
    Ok(page_size)
}

pub(crate) struct PinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> PinnedHostBuffer<T> {
    pub(crate) fn try_new(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::io("pinned host allocation size overflow"))?;
        let mut raw = ptr::null_mut();
        check_cuda(
            unsafe { cudaMallocHost(&mut raw, bytes) },
            "allocate pinned host buffer",
        )?;
        Ok(Self {
            ptr: raw.cast::<T>(),
            len,
            _marker: PhantomData,
        })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub(crate) fn prefix(&self, len: usize) -> Result<&[T]> {
        if len > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than requested prefix {}",
                self.len, len
            )));
        }
        Ok(&self.as_slice()[..len])
    }

    pub(crate) fn prefix_mut(&mut self, len: usize) -> Result<&mut [T]> {
        if len > self.len {
            return Err(Error::io(format!(
                "pinned host buffer length {} is smaller than requested prefix {}",
                self.len, len
            )));
        }
        Ok(&mut self.as_mut_slice()[..len])
    }
}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudaFreeHost(self.ptr.cast::<c_void>()) };
        }
    }
}

// A `PinnedHostBuffer` exclusively owns its `cudaMallocHost` allocation -- no
// other handle aliases it -- so moving it to another thread is sound, and
// `cudaFreeHost` is thread-safe. Needed so buffers can move through the
// staging pool between prepare workers and the pipeline driver.
unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}

/// A fixed set of page-locked host buffers, allocated once per artifact build
/// and recycled across batches, used as the H2D copy source in place of
/// registering each batch's own decoded buffer with `cudaHostRegister`.
///
/// Per-batch registration measured ~3 s per scan stage for 64 x 610 MiB
/// batches, and registrations queue behind each other (~31 ms median alone,
/// 56-96 ms with one or more already in flight) while holding `mmap_lock` for
/// read during pinning, which also stalls concurrent `munmap`s. Copying into
/// an already-pinned slot replaces that with a plain memcpy. See
/// `profiling/PINNED_BUFFER_POOL_DESIGN.md`.
///
/// Slots are only ever allocated here, up front -- never per batch: a
/// per-batch `cudaMallocHost`/`cudaFreeHost` would put synchronizing CUDA
/// allocation calls back into the multi-threaded pipeline.
pub(crate) struct PinnedStagingPool {
    free: Mutex<Vec<PinnedHostBuffer<f32>>>,
    returned: Condvar,
    slot_len: usize,
}

impl PinnedStagingPool {
    pub(crate) fn try_new(slots: usize, slot_len: usize) -> Result<Arc<Self>> {
        let free = (0..slots)
            .map(|_| PinnedHostBuffer::try_new(slot_len))
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(Self {
            free: Mutex::new(free),
            returned: Condvar::new(),
            slot_len,
        }))
    }

    /// Capacity of each slot, in `f32` elements.
    pub(crate) fn slot_len(&self) -> usize {
        self.slot_len
    }

    /// Takes a free slot, blocking until one is returned if none is free.
    ///
    /// Blocks the calling OS thread: call it only from a blocking-capable
    /// thread (e.g. inside `spawn_blocking`), never directly on an async task.
    pub(crate) fn acquire(self: &Arc<Self>) -> PinnedStagingSlot {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(buffer) = free.pop() {
                return PinnedStagingSlot {
                    buffer: Some(buffer),
                    len: 0,
                    pool: Arc::clone(self),
                };
            }
            free = self
                .returned
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// One slot checked out of a [`PinnedStagingPool`]; returns itself to the pool
/// on drop.
///
/// The holder must keep the slot alive until any async copy reading from it
/// has completed -- dropping it earlier would let another batch overwrite a
/// buffer the GPU is still reading. The transform pipeline holds it in its
/// `TransformSlot` until the drain has synchronized on `output_ready`, which
/// is recorded after the H2D copy on the same stream.
pub(crate) struct PinnedStagingSlot {
    buffer: Option<PinnedHostBuffer<f32>>,
    len: usize,
    pool: Arc<PinnedStagingPool>,
}

impl PinnedStagingSlot {
    /// Copies `src` into the start of the slot, split across up to
    /// `copy_threads` threads.
    pub(crate) fn fill_from(&mut self, src: &[f32], copy_threads: usize) -> Result<()> {
        let buffer = self
            .buffer
            .as_mut()
            .ok_or_else(|| Error::io("pinned staging slot has already been released"))?;
        let dst = buffer.prefix_mut(src.len())?;
        parallel_copy(dst, src, copy_threads);
        self.len = src.len();
        Ok(())
    }

    /// The filled prefix of the slot.
    pub(crate) fn as_slice(&self) -> Result<&[f32]> {
        self.buffer
            .as_ref()
            .ok_or_else(|| Error::io("pinned staging slot has already been released"))?
            .prefix(self.len)
    }
}

impl Drop for PinnedStagingSlot {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut free = self
                .pool
                .free
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            free.push(buffer);
            drop(free);
            self.pool.returned.notify_one();
        }
    }
}

/// Below this many bytes per thread, spawning copy threads costs more than it
/// saves.
const PARALLEL_COPY_MIN_BYTES_PER_THREAD: usize = 8 * 1024 * 1024;

fn parallel_copy<T: Copy + Send + Sync>(dst: &mut [T], src: &[T], threads: usize) {
    debug_assert_eq!(dst.len(), src.len());
    let bytes = std::mem::size_of_val(src);
    let threads = threads
        .min(bytes / PARALLEL_COPY_MIN_BYTES_PER_THREAD)
        .max(1);
    if threads == 1 {
        dst.copy_from_slice(src);
        return;
    }
    let chunk = src.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (dst, src) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || dst.copy_from_slice(src));
        }
    });
}

pub(crate) struct CudaEvent {
    raw: CudaEventHandle,
}

impl CudaEvent {
    pub(crate) fn try_new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check_cuda(unsafe { cudaEventCreate(&mut raw) }, "create CUDA event")?;
        Ok(Self { raw })
    }

    pub(crate) fn record(&self, stream: cuvs_sys::cudaStream_t) -> Result<()> {
        check_cuda(
            unsafe { cudaEventRecord(self.raw, stream) },
            "record CUDA event",
        )
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        check_cuda(
            unsafe { cudaEventSynchronize(self.raw) },
            "synchronize CUDA event",
        )
    }

    pub(crate) fn elapsed_since(&self, start: &Self) -> Result<std::time::Duration> {
        let mut ms = 0.0f32;
        check_cuda(
            unsafe { cudaEventElapsedTime(&mut ms, start.raw, self.raw) },
            "measure CUDA event elapsed time",
        )?;
        Ok(std::time::Duration::from_secs_f64(ms as f64 / 1000.0))
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudaEventDestroy(self.raw) };
        }
    }
}

pub(crate) fn check_cuvs(status: cuvs_sys::cuvsError_t, context: &str) -> Result<()> {
    if status == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        return Ok(());
    }

    let message = unsafe {
        let text = cuvs_sys::cuvsGetLastErrorText();
        if text.is_null() {
            format!("{status:?}")
        } else {
            format!(
                "{status:?}: {}",
                CStr::from_ptr(text).to_string_lossy().into_owned()
            )
        }
    };
    Err(Error::io(format!("cuVS failed to {context}: {message}")))
}

pub(crate) fn check_cuda(status: cuvs_sys::cudaError_t, context: &str) -> Result<()> {
    if status == cuvs_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error::io(format!("CUDA failed to {context}: {status:?}")))
    }
}

pub(crate) fn enable_rmm_pool_from_env() -> Result<()> {
    let Some(config) = std::env::var("LANCE_CUVS_RMM_POOL").ok() else {
        return Ok(());
    };
    let (initial, max) = match config.split_once(',') {
        Some((initial, max)) => (
            initial.parse::<i32>().map_err(|error| {
                Error::invalid_input(format!(
                    "invalid LANCE_CUVS_RMM_POOL initial percent '{initial}': {error}"
                ))
            })?,
            max.parse::<i32>().map_err(|error| {
                Error::invalid_input(format!(
                    "invalid LANCE_CUVS_RMM_POOL max percent '{max}': {error}"
                ))
            })?,
        ),
        None => {
            let percent = config.parse::<i32>().map_err(|error| {
                Error::invalid_input(format!(
                    "invalid LANCE_CUVS_RMM_POOL percent '{config}': {error}"
                ))
            })?;
            (percent, percent)
        }
    };
    check_cuvs(
        unsafe { cuvs_sys::cuvsRMMPoolMemoryResourceEnable(initial, max, false) },
        "enable RMM pool memory resource",
    )
}

pub(crate) fn cuvs_distance_type(metric_type: DistanceType) -> Result<cuvs_sys::cuvsDistanceType> {
    match metric_type {
        DistanceType::L2 => Ok(cuvs_sys::cuvsDistanceType::L2Expanded),
        DistanceType::Cosine => Ok(cuvs_sys::cuvsDistanceType::CosineExpanded),
        DistanceType::Dot => Ok(cuvs_sys::cuvsDistanceType::InnerProduct),
        other => Err(Error::not_supported(format!(
            "cuVS IVF_PQ does not support metric {other:?}"
        ))),
    }
}

pub(crate) fn create_index_params(
    metric_type: DistanceType,
    num_partitions: usize,
    num_sub_vectors: usize,
    sample_rate: usize,
    max_iters: usize,
    num_bits: usize,
) -> Result<cuvs_sys::cuvsIvfPqIndexParams_t> {
    let mut params = ptr::null_mut();
    check_cuvs(
        unsafe { cuvs_sys::cuvsIvfPqIndexParamsCreate(&mut params) },
        "allocate IVF_PQ index params",
    )?;
    let metric = cuvs_distance_type(metric_type)?;
    unsafe {
        (*params).metric = metric;
        (*params).metric_arg = 0.0;
        (*params).add_data_on_build = false;
        (*params).n_lists = num_partitions as u32;
        (*params).kmeans_n_iters = max_iters as u32;
        (*params).kmeans_trainset_fraction = 1.0;
        (*params).pq_bits = num_bits as u32;
        (*params).pq_dim = num_sub_vectors as u32;
        (*params).codebook_kind =
            cuvs_sys::cuvsIvfPqCodebookGen::CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE;
        (*params).force_random_rotation = false;
        (*params).conservative_memory_allocation = false;
        (*params).max_train_points_per_pq_code = sample_rate as u32;
        (*params).codes_layout = cuvs_sys::cuvsIvfPqListLayout::CUVS_IVF_PQ_LIST_LAYOUT_FLAT;
    }
    Ok(params)
}

pub(crate) fn destroy_index_params(params: cuvs_sys::cuvsIvfPqIndexParams_t) {
    if !params.is_null() {
        let _ = unsafe { cuvs_sys::cuvsIvfPqIndexParamsDestroy(params) };
    }
}

pub(crate) fn make_tensor_view() -> HostTensorView {
    let shape = Vec::new();
    let tensor = cuvs_sys::DLManagedTensor {
        dl_tensor: cuvs_sys::DLTensor {
            data: ptr::null_mut(),
            device: cuvs_sys::DLDevice {
                device_type: cuvs_sys::DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 0,
            dtype: <f32 as DlElement>::dl_dtype(),
            shape: shape.as_ptr() as *mut i64,
            strides: ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ptr::null_mut(),
        deleter: None,
    };
    HostTensorView { shape, tensor }
}

pub(crate) fn tensor_shape(tensor: &cuvs_sys::DLManagedTensor) -> Vec<usize> {
    let dl_tensor = &tensor.dl_tensor;
    (0..dl_tensor.ndim)
        .map(|idx| unsafe { *dl_tensor.shape.add(idx as usize) as usize })
        .collect()
}

pub(crate) fn tensor_num_bytes(tensor: &cuvs_sys::DLManagedTensor) -> usize {
    let shape = tensor_shape(tensor);
    let numel = shape.into_iter().product::<usize>();
    numel * ((tensor.dl_tensor.dtype.bits as usize) / 8)
}

pub(crate) fn copy_tensor_to_host_f32_2d(
    resources: &Resources,
    tensor: &cuvs_sys::DLManagedTensor,
) -> Result<Array2<f32>> {
    let shape = tensor_shape(tensor);
    if shape.len() != 2 {
        return Err(Error::io(format!(
            "expected 2D tensor, got shape {shape:?}"
        )));
    }
    let mut array = Array2::<f32>::zeros((shape[0], shape[1]));
    let stream = resources
        .get_cuda_stream()
        .map_err(|e| Error::io(e.to_string()))?;
    if tensor.dl_tensor.strides.is_null() {
        check_cuda(
            unsafe {
                cuvs_sys::cudaMemcpyAsync(
                    array.as_mut_ptr() as *mut _,
                    tensor.dl_tensor.data,
                    tensor_num_bytes(tensor),
                    cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                    stream,
                )
            },
            "copy tensor to host",
        )?;
    } else {
        let row_stride = unsafe { *tensor.dl_tensor.strides } as usize;
        let col_stride = unsafe { *tensor.dl_tensor.strides.add(1) } as usize;
        if col_stride != 1 {
            return Err(Error::not_supported(format!(
                "copying 2D tensors with non-unit column stride is not supported: strides=({}, {})",
                row_stride, col_stride
            )));
        }
        let row_bytes = shape[1] * std::mem::size_of::<f32>();
        check_cuda(
            unsafe {
                cudaMemcpy2DAsync(
                    array.as_mut_ptr().cast::<c_void>(),
                    row_bytes,
                    tensor.dl_tensor.data,
                    row_stride * std::mem::size_of::<f32>(),
                    row_bytes,
                    shape[0],
                    cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                    stream,
                )
            },
            "copy strided tensor to host",
        )?;
    }
    resources
        .sync_stream()
        .map_err(|e| Error::io(e.to_string()))?;
    Ok(array)
}

pub(crate) fn copy_tensor_to_host_f32_3d(
    resources: &Resources,
    tensor: &cuvs_sys::DLManagedTensor,
) -> Result<(Vec<f32>, [usize; 3])> {
    let shape = tensor_shape(tensor);
    if shape.len() != 3 {
        return Err(Error::io(format!(
            "expected 3D tensor, got shape {shape:?}"
        )));
    }
    let mut values = vec![0.0f32; shape.iter().product()];
    check_cuda(
        unsafe {
            cuvs_sys::cudaMemcpyAsync(
                values.as_mut_ptr() as *mut _,
                tensor.dl_tensor.data,
                tensor_num_bytes(tensor),
                cuvs_sys::cudaMemcpyKind_cudaMemcpyDefault,
                resources
                    .get_cuda_stream()
                    .map_err(|e| Error::io(e.to_string()))?,
            )
        },
        "copy tensor to host",
    )?;
    resources
        .sync_stream()
        .map_err(|e| Error::io(e.to_string()))?;
    Ok((values, [shape[0], shape[1], shape[2]]))
}

pub(crate) fn matrix_from_vectors<'a>(vectors: &'a FixedSizeListArray) -> Result<MatrixBuffer<'a>> {
    let dim = vectors.value_length() as usize;
    match vectors.value_type() {
        DataType::Float32 => {
            let values = vectors.values().as_primitive::<Float32Type>();
            let values: &[f32] = values.values().as_ref();
            Ok(MatrixBuffer::Borrowed {
                values,
                rows: vectors.len(),
                cols: dim,
            })
        }
        DataType::Float16 => {
            let values = vectors.values().as_primitive::<Float16Type>();
            let data = values
                .values()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            Ok(MatrixBuffer::Owned(
                Array2::from_shape_vec((vectors.len(), dim), data).map_err(|error| {
                    Error::io(format!("failed to create float16 matrix copy: {error}"))
                })?,
            ))
        }
        DataType::Float64 => {
            let values = vectors.values().as_primitive::<Float64Type>();
            let data = values
                .values()
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>();
            Ok(MatrixBuffer::Owned(
                Array2::from_shape_vec((vectors.len(), dim), data).map_err(|error| {
                    Error::io(format!("failed to create float64 matrix copy: {error}"))
                })?,
            ))
        }
        other => Err(Error::not_supported(format!(
            "cuVS IVF_PQ currently supports float16/float32/float64 vectors, got {other}"
        ))),
    }
}

pub(crate) fn ivf_centroids_from_host(array: Array2<f32>) -> Result<FixedSizeListArray> {
    let dim = array.ncols() as i32;
    let values = arrow_array::Float32Array::from_iter_values(array);
    Ok(FixedSizeListArray::try_new_from_values(values, dim)?)
}

pub(crate) fn pq_codebook_from_host(
    values: Vec<f32>,
    shape: [usize; 3],
    num_sub_vectors: usize,
    dimension: usize,
    num_bits: usize,
) -> Result<FixedSizeListArray> {
    let pq_book_size = 1usize << num_bits;
    let subvector_dim = dimension / num_sub_vectors;
    let expected = [num_sub_vectors, subvector_dim, pq_book_size];
    if shape != expected {
        return Err(Error::io(format!(
            "cuVS returned incompatible PQ codebook shape: expected {expected:?}, got {shape:?}"
        )));
    }

    let mut flattened = Vec::with_capacity(values.len());
    for subspace in 0..num_sub_vectors {
        for centroid in 0..pq_book_size {
            for component in 0..subvector_dim {
                let source_idx = ((subspace * subvector_dim + component) * pq_book_size) + centroid;
                flattened.push(values[source_idx]);
            }
        }
    }

    Ok(FixedSizeListArray::try_new_from_values(
        arrow_array::Float32Array::from(flattened),
        subvector_dim as i32,
    )?)
}
