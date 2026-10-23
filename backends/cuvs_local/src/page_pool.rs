//! A fixed pool of pinned host memory handed to
//! lance-io for its large `O_DIRECT` page-read buffers, so these ~32 MiB
//! allocations do not go through the global allocator. Falls back to the heap
//! when the pool is empty or a request does not fit a slot.
//!
//! `LANCE_CUVS_PAGE_POOL_GIB` (unset/0 = off), `LANCE_CUVS_PAGE_SLOT_MIB` (default 40).

use crate::cuda::PinnedHostBuffer;
use lance_core::Result;
use lance_io::buffer_pool::{DirectBufAllocator, PoolBuf, set_direct_buf_allocator};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const ALIGN: usize = 4096;

/// Address range of the active pool's slots (`[start, end)`), or empty when there is none.
static POOL_START: AtomicUsize = AtomicUsize::new(0);
static POOL_END: AtomicUsize = AtomicUsize::new(0);

/// True if `[ptr, ptr + len)` lies inside the pool's pinned region. Such memory is already
/// page-locked, so it must not be passed to `cudaHostRegister` (which fails with
/// `cudaErrorHostMemoryAlreadyRegistered`) and can be copied to the device directly.
pub(crate) fn contains(ptr: usize, len: usize) -> bool {
    let end = POOL_END.load(Ordering::Acquire);
    let start = POOL_START.load(Ordering::Acquire);
    ptr >= start && ptr.saturating_add(len) <= end
}

pub(crate) struct PagePool {
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

struct PoolHandle(Arc<PagePool>);

impl DirectBufAllocator for PoolHandle {
    fn alloc(&self, len: usize) -> Option<PoolBuf> {
        let pool = &self.0;
        pool.max_len.fetch_max(len, Ordering::Relaxed);
        if len > pool.slot_bytes {
            pool.misses_too_big.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let idx = pool.free.lock().expect("page pool poisoned").pop();
        let Some(idx) = idx else {
            pool.misses_empty.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        pool.hits.fetch_add(1, Ordering::Relaxed);
        let now = pool.in_use.fetch_add(1, Ordering::Relaxed) + 1;
        pool.max_in_use.fetch_max(now, Ordering::Relaxed);
        let ptr = NonNull::new((pool.base + idx * pool.slot_bytes) as *mut u8)?;
        let pool = pool.clone();
        Some(PoolBuf {
            ptr,
            release: Box::new(move || {
                pool.in_use.fetch_sub(1, Ordering::Relaxed);
                pool.free.lock().expect("page pool poisoned").push(idx);
            }),
        })
    }
}

/// Registers the pool with lance-io for its lifetime; unregisters on drop.
pub(crate) struct PagePoolGuard {
    pool: Arc<PagePool>,
}

impl Drop for PagePoolGuard {
    fn drop(&mut self) {
        set_direct_buf_allocator(None);
        POOL_END.store(0, Ordering::Release);
        POOL_START.store(0, Ordering::Release);
        let p = &self.pool;
        eprintln!(
            "cuVS page pool: slots={} slot_mib={} hits={} misses_empty={} misses_too_big={} max_in_use={} max_request_mib={:.2}",
            p.slots,
            p.slot_bytes >> 20,
            p.hits.load(Ordering::Relaxed),
            p.misses_empty.load(Ordering::Relaxed),
            p.misses_too_big.load(Ordering::Relaxed),
            p.max_in_use.load(Ordering::Relaxed),
            p.max_len.load(Ordering::Relaxed) as f64 / (1u64 << 20) as f64,
        );
    }
}

pub(crate) fn page_pool_from_env() -> Result<Option<PagePoolGuard>> {
    let gib: f64 = std::env::var("LANCE_CUVS_PAGE_POOL_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    if !(gib > 0.0) {
        return Ok(None);
    }
    let slot_mib: usize = std::env::var("LANCE_CUVS_PAGE_SLOT_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let slot_bytes = (slot_mib << 20).div_ceil(ALIGN) * ALIGN;
    let slots = ((gib * (1u64 << 30) as f64) as usize / slot_bytes).max(1);
    let start = std::time::Instant::now();
    // One extra page of slack so the first slot can be aligned to 4096.
    let mut region = PinnedHostBuffer::<u8>::try_new(slots * slot_bytes + ALIGN)?;
    let raw = region.as_mut_slice().as_mut_ptr() as usize;
    let base = raw.div_ceil(ALIGN) * ALIGN;
    let pool = Arc::new(PagePool {
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
    POOL_START.store(base, Ordering::Release);
    POOL_END.store(base + slots * slot_bytes, Ordering::Release);
    set_direct_buf_allocator(Some(Arc::new(PoolHandle(pool.clone()))));
    eprintln!(
        "cuVS page pool: enabled, {} x {} MiB pinned = {:.1} GiB (alloc {:.1}s)",
        slots,
        slot_bytes >> 20,
        (slots * slot_bytes) as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(Some(PagePoolGuard { pool }))
}
