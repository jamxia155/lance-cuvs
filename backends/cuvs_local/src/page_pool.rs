//! Fixed pools of pinned host memory handed to Lance so that its large host buffers do not go
//! through the global allocator. Each pool is one `cudaMallocHost` region cut into fixed-size
//! slots; a request takes one whole slot, and falls back to the heap when the pool is empty or
//! the request does not fit a slot.
//!
//! * page pool: the ~32 MiB `O_DIRECT` page-read buffers of lance-io.
//!   `LANCE_CUVS_PAGE_POOL_GIB` (unset/0 = off), `LANCE_CUVS_PAGE_SLOT_MIB` (default 40).
//! * batch pool: the contiguous buffer a decoded batch is assembled into from its pages
//!   (lance-encoding), the buffer that is later copied to the device. Slots must hold a whole
//!   batch (`batch_size` rows x dim x 4 bytes, 1 GiB for 128K rows of 2048 floats), and the pool
//!   must cover every batch alive at once. `LANCE_CUVS_BATCH_POOL_GIB` (unset/0 = off),
//!   `LANCE_CUVS_BATCH_SLOT_MIB` (default 1024).
//!
//! Slices inside either pool are already page-locked, see [`contains`].

use crate::cuda::PinnedHostBuffer;
use lance_core::Result;
use lance_encoding::output_pool::{
    DecodeOutputAllocator, OutputBuf, set_decode_output_allocator,
};
use lance_io::buffer_pool::{DirectBufAllocator, PoolBuf, set_direct_buf_allocator};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const ALIGN: usize = 4096;

/// Address ranges (`[start, end)`) of the active pools, one entry per pool kind; empty when off.
static POOL_RANGES: [(AtomicUsize, AtomicUsize); 2] = [
    (AtomicUsize::new(0), AtomicUsize::new(0)),
    (AtomicUsize::new(0), AtomicUsize::new(0)),
];

/// True if `[ptr, ptr + len)` lies inside a pool's pinned region. Such memory is already
/// page-locked, so it must not be passed to `cudaHostRegister` (which fails with
/// `cudaErrorHostMemoryAlreadyRegistered`) and can be copied to the device directly.
pub(crate) fn contains(ptr: usize, len: usize) -> bool {
    POOL_RANGES.iter().any(|(start, end)| {
        let end = end.load(Ordering::Acquire);
        let start = start.load(Ordering::Acquire);
        ptr >= start && ptr.saturating_add(len) <= end
    })
}

pub(crate) struct SlotPool {
    label: &'static str,
    range_index: usize,
    // Owns the pinned region; only touched on drop.
    _region: Mutex<PinnedHostBuffer<u8>>,
    base: usize,
    slot_bytes: usize,
    free: Mutex<Vec<usize>>,
    slots: usize,
    hits: AtomicU64,
    misses_empty: AtomicU64,
    misses_too_big: AtomicU64,
    max_len: AtomicUsize,
    in_use: AtomicUsize,
    max_in_use: AtomicUsize,
}

impl SlotPool {
    /// Takes one slot for a request of `len` bytes, with the closure that returns it.
    fn take(self: &Arc<Self>, len: usize) -> Option<(NonNull<u8>, Box<dyn FnOnce() + Send + Sync>)> {
        self.max_len.fetch_max(len, Ordering::Relaxed);
        if len > self.slot_bytes {
            self.misses_too_big.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let idx = self.free.lock().expect("slot pool poisoned").pop();
        let Some(idx) = idx else {
            self.misses_empty.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.hits.fetch_add(1, Ordering::Relaxed);
        let now = self.in_use.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_in_use.fetch_max(now, Ordering::Relaxed);
        let ptr = NonNull::new((self.base + idx * self.slot_bytes) as *mut u8)?;
        let pool = self.clone();
        Some((
            ptr,
            Box::new(move || {
                pool.in_use.fetch_sub(1, Ordering::Relaxed);
                pool.free.lock().expect("slot pool poisoned").push(idx);
            }),
        ))
    }

    fn report(&self) {
        eprintln!(
            "cuVS {} pool: slots={} slot_mib={} hits={} misses_empty={} misses_too_big={} max_in_use={} max_request_mib={:.2}",
            self.label,
            self.slots,
            self.slot_bytes >> 20,
            self.hits.load(Ordering::Relaxed),
            self.misses_empty.load(Ordering::Relaxed),
            self.misses_too_big.load(Ordering::Relaxed),
            self.max_in_use.load(Ordering::Relaxed),
            self.max_len.load(Ordering::Relaxed) as f64 / (1u64 << 20) as f64,
        );
    }

    fn clear_range(&self) {
        let (start, end) = &POOL_RANGES[self.range_index];
        end.store(0, Ordering::Release);
        start.store(0, Ordering::Release);
    }
}

struct PageHandle(Arc<SlotPool>);

impl DirectBufAllocator for PageHandle {
    fn alloc(&self, len: usize) -> Option<PoolBuf> {
        let (ptr, release) = self.0.take(len)?;
        Some(PoolBuf { ptr, release })
    }
}

struct BatchHandle(Arc<SlotPool>);

impl DecodeOutputAllocator for BatchHandle {
    fn alloc(&self, len: usize) -> Option<OutputBuf> {
        let (ptr, release) = self.0.take(len)?;
        Some(OutputBuf { ptr, release })
    }
}

/// Registers the pools with lance-io / lance-encoding for their lifetime; unregisters on drop.
pub(crate) struct PagePoolGuard {
    page: Option<Arc<SlotPool>>,
    batch: Option<Arc<SlotPool>>,
}

impl Drop for PagePoolGuard {
    fn drop(&mut self) {
        if let Some(pool) = &self.page {
            set_direct_buf_allocator(None);
            pool.clear_range();
            pool.report();
        }
        if let Some(pool) = &self.batch {
            set_decode_output_allocator(None);
            pool.clear_range();
            pool.report();
        }
    }
}

fn env_f64(name: &str) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn create_pool(
    label: &'static str,
    range_index: usize,
    gib: f64,
    slot_mib: usize,
) -> Result<Arc<SlotPool>> {
    let slot_bytes = (slot_mib << 20).div_ceil(ALIGN) * ALIGN;
    let slots = ((gib * (1u64 << 30) as f64) as usize / slot_bytes).max(1);
    let start = std::time::Instant::now();
    // One extra page of slack so the first slot can be aligned to 4096.
    let mut region = PinnedHostBuffer::<u8>::try_new(slots * slot_bytes + ALIGN)?;
    let raw = region.as_mut_slice().as_mut_ptr() as usize;
    let base = raw.div_ceil(ALIGN) * ALIGN;
    let pool = Arc::new(SlotPool {
        label,
        range_index,
        _region: Mutex::new(region),
        base,
        slot_bytes,
        free: Mutex::new((0..slots).rev().collect()),
        slots,
        hits: AtomicU64::new(0),
        misses_empty: AtomicU64::new(0),
        misses_too_big: AtomicU64::new(0),
        max_len: AtomicUsize::new(0),
        in_use: AtomicUsize::new(0),
        max_in_use: AtomicUsize::new(0),
    });
    let (range_start, range_end) = &POOL_RANGES[range_index];
    range_start.store(base, Ordering::Release);
    range_end.store(base + slots * slot_bytes, Ordering::Release);
    eprintln!(
        "cuVS {label} pool: enabled, {} x {} MiB pinned = {:.1} GiB (alloc {:.1}s)",
        slots,
        slot_bytes >> 20,
        (slots * slot_bytes) as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(pool)
}

pub(crate) fn page_pool_from_env() -> Result<Option<PagePoolGuard>> {
    let page_gib = env_f64("LANCE_CUVS_PAGE_POOL_GIB");
    let batch_gib = env_f64("LANCE_CUVS_BATCH_POOL_GIB");
    let mut guard = PagePoolGuard {
        page: None,
        batch: None,
    };
    if page_gib > 0.0 {
        let pool = create_pool(
            "page",
            0,
            page_gib,
            env_usize("LANCE_CUVS_PAGE_SLOT_MIB", 40),
        )?;
        set_direct_buf_allocator(Some(Arc::new(PageHandle(pool.clone()))));
        guard.page = Some(pool);
    }
    if batch_gib > 0.0 {
        let pool = create_pool(
            "batch",
            1,
            batch_gib,
            env_usize("LANCE_CUVS_BATCH_SLOT_MIB", 1024),
        )?;
        set_decode_output_allocator(Some(Arc::new(BatchHandle(pool.clone()))));
        guard.batch = Some(pool);
    }
    Ok((guard.page.is_some() || guard.batch.is_some()).then_some(guard))
}
