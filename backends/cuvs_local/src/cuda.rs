// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use ndarray::{Array2, ArrayView2};
use std::ffi::{CStr, c_uint, c_void};
use std::marker::PhantomData;
use std::ptr;

pub(crate) type CudaEventHandle = *mut c_void;

// cuvs-sys's bindgen build blocklists everything matching `cuda.*` (only
// `cudaDataType_t`/`cudaStream_t` are hand-declared there, for cuVS's own
// API surface) -- see cuvs-sys/build.rs and cuvs-sys/src/lib.rs in the local
// cuVS checkout. This crate needs a broader slice of the cudart C ABI for
// its own H2D/D2H pipeline, so it's hand-declared here the same way, rather
// than relying on cuvs_sys to re-export it.
pub(crate) type CudaError = c_uint;
pub(crate) const CUDA_SUCCESS: CudaError = 0;
pub(crate) type CudaMemcpyKind = c_uint;
// Deliberately not using `cudaMemcpyDefault` (kind=4, UVA-based direction
// inference): the cuVS Rust/Java bindings' own usage against `cuvsRMMAlloc`-
// backed buffers (e.g. java/.../CagraIndexImpl.java,
// MultiPartitionCagraSearchImpl.java) consistently uses an explicit
// direction instead, since `cuvsRMMAlloc`'s workspace-resource-backed
// pointers (see cuVS PR #2035) aren't reliably recognized by
// `cudaPointerGetAttributes`-based direction inference -- using
// `cudaMemcpyDefault` against them fails with `cudaErrorInvalidValue`.
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: CudaMemcpyKind = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: CudaMemcpyKind = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaFreeHost(ptr: *mut c_void) -> CudaError;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> CudaError;
    fn cudaHostUnregister(ptr: *mut c_void) -> CudaError;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
        stream: cuvs_sys::cudaStream_t,
    ) -> CudaError;
    fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: CudaMemcpyKind,
        stream: cuvs_sys::cudaStream_t,
    ) -> CudaError;
    fn cudaEventCreate(event: *mut CudaEventHandle) -> CudaError;
    fn cudaEventDestroy(event: CudaEventHandle) -> CudaError;
    fn cudaEventRecord(event: CudaEventHandle, stream: cuvs_sys::cudaStream_t) -> CudaError;
    fn cudaEventSynchronize(event: CudaEventHandle) -> CudaError;
    fn cudaEventElapsedTime(
        ms: *mut f32,
        start: CudaEventHandle,
        end: CudaEventHandle,
    ) -> CudaError;
    fn cudaGetErrorString(error: CudaError) -> *const std::ffi::c_char;
    fn cudaGetErrorName(error: CudaError) -> *const std::ffi::c_char;
    fn cudaMallocAsync(
        ptr: *mut *mut c_void,
        size: usize,
        stream: cuvs_sys::cudaStream_t,
    ) -> CudaError;
    fn cudaFreeAsync(ptr: *mut c_void, stream: cuvs_sys::cudaStream_t) -> CudaError;
}

/// `cuvsResources_t` handle owner.
///
/// `cuvs::Resources` (the safe Rust wrapper crate) privatized its raw handle
/// field in cuVS PR #2284 ("Rust: builder-based params, per-module errors"),
/// exposing only a `pub(crate)` accessor scoped to the `cuvs` crate itself.
/// This backend calls several raw `cuvs_sys` C API functions directly
/// (`cuvsIvfPqTransform`, `cuvsRMMAlloc`, ...) that have no safe wrapper in
/// the `cuvs` crate, so it needs the raw handle -- this type owns one
/// directly via `cuvs_sys`, mirroring what `cuvs::Resources` itself does
/// internally (and what it publicly exposed prior to PR #2284).
pub(crate) struct Resources(pub(crate) cuvs_sys::cuvsResources_t);

impl Resources {
    pub(crate) fn new() -> Result<Self> {
        let mut handle: cuvs_sys::cuvsResources_t = 0;
        check_cuvs(
            unsafe { cuvs_sys::cuvsResourcesCreate(&mut handle) },
            "create cuVS resources",
        )?;
        Ok(Self(handle))
    }

    pub(crate) fn get_cuda_stream(&self) -> Result<cuvs_sys::cudaStream_t> {
        let mut stream = std::mem::MaybeUninit::<cuvs_sys::cudaStream_t>::uninit();
        check_cuvs(
            unsafe { cuvs_sys::cuvsStreamGet(self.0, stream.as_mut_ptr()) },
            "get cuVS resources CUDA stream",
        )?;
        Ok(unsafe { stream.assume_init() })
    }

    pub(crate) fn sync_stream(&self) -> Result<()> {
        check_cuvs(
            unsafe { cuvs_sys::cuvsStreamSync(self.0) },
            "sync cuVS resources CUDA stream",
        )
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        let _ = unsafe { cuvs_sys::cuvsResourcesDestroy(self.0) };
    }
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
    stream: cuvs_sys::cudaStream_t,
    _marker: PhantomData<T>,
}

impl<T: DlElement> DeviceTensor<T> {
    pub(crate) fn try_new(resources: &Resources, shape: &[usize]) -> Result<Self> {
        let capacity_bytes = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let stream = resources
            .get_cuda_stream()
            .map_err(|e| Error::io(e.to_string()))?;
        let mut data = ptr::null_mut();
        // `cudaMallocAsync`/`cudaFreeAsync` (CUDA's own stream-ordered
        // allocator), not `cuvsRMMAlloc` (allocates from cuVS's RAFT
        // "workspace" resource -- documented as scoped to transient,
        // per-call scratch use, not the long-lived, reused-across-many-calls
        // buffers `TransformSlot` holds) and not plain synchronous
        // `cudaMalloc`/`cudaFree` (device-wide-serializing, unacceptable in
        // this multi-threaded pipeline).
        check_cuda(
            unsafe { cudaMallocAsync(&mut data, capacity_bytes, stream) },
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
            stream,
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
                cudaMemcpyAsync(
                    self.tensor.dl_tensor.data,
                    src.as_ptr() as *const _,
                    self.current_bytes(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    resources
                        .get_cuda_stream()
                        .map_err(|e| Error::io(e.to_string()))?,
                )
            },
            "copy host tensor to device",
        )
    }

    /// Reads `len_bytes` directly into the device tensor's buffer at `dst_offset_bytes`, via
    /// `cuvsReadLargeFile`, starting at `file_offset` in the file at `path`.
    ///
    /// `dst_offset_bytes`/`len_bytes` (rather than always filling the whole tensor) let a caller
    /// issue several reads into one buffer -- needed since a single batch's rows can span multiple
    /// physical pages within a fragment, each landing at a different device-buffer offset.
    ///
    /// Unlike `copy_from_host_async`, this is **synchronous** -- `cuvsReadLargeFile` has no CUDA
    /// stream parameter (kvikio manages its own read concurrency internally, not via CUDA
    /// streams), so the call blocks the calling thread until the read completes. Confirmed against
    /// real GDS hardware (not just compat-mode fallback), see `gds_read_smoke_test.rs` and
    /// `profiling/GDS_PORTING_PLAN.md`.
    pub(crate) fn read_from_gds(
        &mut self,
        path: &str,
        file_offset: u64,
        dst_offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<()> {
        let end = dst_offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(|| Error::io("GDS read destination range overflow"))?;
        if end > self.capacity_bytes {
            return Err(Error::io(format!(
                "GDS read destination range {dst_offset_bytes}..{end} exceeds device tensor capacity {}",
                self.capacity_bytes
            )));
        }
        let path_c = std::ffi::CString::new(path)
            .map_err(|error| Error::io(format!("GDS read path contains NUL byte: {error}")))?;
        let dst_ptr = unsafe { (self.tensor.dl_tensor.data as *mut u8).add(dst_offset_bytes) as *mut c_void };
        check_cuvs(
            unsafe { cuvs_sys::cuvsReadLargeFile(path_c.as_ptr(), dst_ptr, len_bytes, file_offset) },
            "read device tensor via GDS",
        )
    }

    /// Stream-ordered counterpart to `read_from_gds`: enqueues the read onto `stream` via
    /// `cuvsReadLargeFileAsync` and returns immediately (like `copy_from_host_async`), instead of
    /// blocking the calling thread for the read's duration. The returned `GdsReadFuture` must be
    /// finished (via `finish_gds_read_async`) only after `stream` has been synchronized past this
    /// read -- see `GdsReadFuture`'s own doc comment.
    pub(crate) fn read_from_gds_async(
        &mut self,
        path: &str,
        file_offset: u64,
        dst_offset_bytes: usize,
        len_bytes: usize,
        stream: cuvs_sys::cudaStream_t,
    ) -> Result<GdsReadFuture> {
        let end = dst_offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(|| Error::io("GDS read destination range overflow"))?;
        if end > self.capacity_bytes {
            return Err(Error::io(format!(
                "GDS read destination range {dst_offset_bytes}..{end} exceeds device tensor capacity {}",
                self.capacity_bytes
            )));
        }
        let path_c = std::ffi::CString::new(path)
            .map_err(|error| Error::io(format!("GDS read path contains NUL byte: {error}")))?;
        let dst_ptr = unsafe { (self.tensor.dl_tensor.data as *mut u8).add(dst_offset_bytes) as *mut c_void };
        let mut future_out: cuvs_sys::cuvsGdsReadFuture_t = ptr::null_mut();
        check_cuvs(
            unsafe {
                cuvs_sys::cuvsReadLargeFileAsync(
                    path_c.as_ptr(),
                    dst_ptr,
                    len_bytes,
                    file_offset,
                    stream,
                    &mut future_out,
                )
            },
            "begin async GDS read",
        )?;
        Ok(GdsReadFuture(future_out))
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
                cudaMemcpyAsync(
                    dst.as_mut_ptr() as *mut _,
                    self.tensor.dl_tensor.data,
                    self.current_bytes(),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
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
            let _ = unsafe { cudaFreeAsync(self.tensor.dl_tensor.data, self.stream) };
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
        // Round INWARD to page boundaries (ceil the start, floor the end),
        // not outward: consecutive batches are contiguous, non-page-aligned
        // slices of one shared underlying Arrow buffer, and the pipeline
        // keeps several of their RegisteredHostBuffers alive at once
        // (look-ahead prefetch). Outward rounding made adjacent batches'
        // registered ranges overlap by up to a page, which cudaHostRegister
        // either rejects outright (cudaErrorHostMemoryAlreadyRegistered) or
        // -- as observed here -- silently accepts but leaves in a state
        // that fails a later cudaMemcpyAsync with cudaErrorInvalidValue.
        // Inward rounding can never produce overlapping ranges between
        // touching slices, at the cost of leaving up to `page_size - 1`
        // bytes unregistered (pageable) at each end of the buffer.
        let inward_start = start
            .checked_add(page_size - 1)
            .ok_or_else(|| Error::io("registered host buffer alignment overflow"))?
            & !(page_size - 1);
        let inward_end = end & !(page_size - 1);
        if inward_end <= inward_start {
            // Buffer doesn't contain a full page -- nothing safe to
            // register; fall back to plain pageable memory for this batch.
            return Ok(Self {
                ptr: ptr::null_mut(),
                original_bytes: 0,
            });
        }
        let bytes = inward_end - inward_start;
        let ptr = inward_start as *mut c_void;

        check_cuda(
            // Flags=0 (not Portable): tried Portable while debugging a
            // cross-thread-registration theory for a cudaErrorInvalidValue
            // that turned out to be caused by overlapping registered ranges
            // instead (fixed above via inward page rounding), unrelated to
            // Portable. Benchmarking then showed Portable adds a real,
            // consistent per-batch cost (~70-100ms/batch on H2D, matching
            // cuVS PR investigation notes: Portable memory is documented to
            // participate across all CUDA contexts, a heavier guarantee
            // this single-context pipeline doesn't need) with no
            // correctness benefit -- matches cuvs_26_02's cuda.rs, which
            // has always used flags=0 here without issue.
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

/// A pending, stream-ordered GDS read started by `DeviceTensor::read_from_gds_async`.
///
/// Must be kept alive and passed to `finish_gds_read_async` only after the CUDA stream it was
/// enqueued on has been synchronized past it (e.g. via `CudaEvent::synchronize` on an event
/// recorded after the read on that stream) -- finishing it earlier is undefined behavior, per
/// `cuvsFinishReadLargeFileAsync`'s own contract (which this wraps 1:1). Dropping it without
/// calling `finish_gds_read_async` leaks the underlying C++ object -- deliberately not
/// implemented as a `Drop` safety net, since a `Drop` impl running before the stream has been
/// synchronized would itself be the same undefined-behavior violation this type exists to avoid;
/// leaking on an already-erroneous path is the lesser failure mode.
pub(crate) struct GdsReadFuture(cuvs_sys::cuvsGdsReadFuture_t);

// Safety: this wraps an opaque C++ object (a kvikio StreamFuture + FileHandle pair) with no
// thread-affinity of its own -- CUDA stream-ordered operations are inherently safe to enqueue
// from one thread and wait on/finish from another (this is exactly what `cudaStreamSynchronize`
// from a different thread than the enqueuing one already implies). This type exclusively owns the
// underlying handle (never aliased, no shared mutable state), so moving it across a task/thread
// boundary -- the entire reason this type exists -- is sound.
unsafe impl Send for GdsReadFuture {}

/// Completes a pending read started by `DeviceTensor::read_from_gds_async`. Must only be called
/// after the CUDA stream the read was enqueued on has been synchronized past it -- see
/// `GdsReadFuture`'s doc comment. `expected_bytes` is verified against the number of bytes
/// actually read (`cuvsFinishReadLargeFileAsync` itself fails loudly on a short read).
pub(crate) fn finish_gds_read_async(future: GdsReadFuture, expected_bytes: usize) -> Result<()> {
    check_cuvs(
        unsafe { cuvs_sys::cuvsFinishReadLargeFileAsync(future.0, expected_bytes) },
        "finish async GDS read",
    )
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

pub(crate) fn check_cuda(status: CudaError, context: &str) -> Result<()> {
    if status == CUDA_SUCCESS {
        return Ok(());
    }

    let name = unsafe {
        let ptr = cudaGetErrorName(status);
        if ptr.is_null() {
            format!("code {status}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    let message = unsafe {
        let ptr = cudaGetErrorString(status);
        if ptr.is_null() {
            "<no message>".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(Error::io(format!(
        "CUDA failed to {context}: {name} ({status}): {message}"
    )))
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
                cudaMemcpyAsync(
                    array.as_mut_ptr() as *mut _,
                    tensor.dl_tensor.data,
                    tensor_num_bytes(tensor),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
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
                    CUDA_MEMCPY_DEVICE_TO_HOST,
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
            cudaMemcpyAsync(
                values.as_mut_ptr() as *mut _,
                tensor.dl_tensor.data,
                tensor_num_bytes(tensor),
                CUDA_MEMCPY_DEVICE_TO_HOST,
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
