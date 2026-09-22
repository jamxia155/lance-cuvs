// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Debug-only instrumentation: logs a symbolicated backtrace for every
//! Rust-side allocation at or above `LANCE_CUVS_ALLOC_TRACE_MIN_BYTES`
//! bytes (default 2MiB -- comfortably below the smallest step in the
//! 2/4/8/16/32/64/128 MiB `mmap`/`munmap` doubling pattern this is meant to
//! diagnose). Disabled unless `LANCE_CUVS_ALLOC_TRACE=1` is set; when
//! disabled this is a transparent passthrough to the system allocator with
//! one cached `OnceLock` read of overhead per call.
//!
//! This intercepts allocations made by any Rust code statically linked into
//! this `.so` (lance-file, lance-encoding, lance-io, lance-table, this
//! crate, etc.) at the point the `Layout` is requested -- upstream of
//! whatever the system allocator (glibc `malloc`, eventually `mmap` for
//! large sizes) does with it. It does not see non-Rust (CUDA driver, glibc
//! internals) allocations.
//!
//! Temporary debugging aid for the pinned-buffer-pool investigation
//! (see `profiling/PINNED_BUFFER_POOL_DESIGN.md`) -- not meant to ship
//! enabled, and not meant to stay in the tree long-term.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LANCE_CUVS_ALLOC_TRACE").is_ok())
}

fn min_bytes() -> usize {
    static MIN_BYTES: OnceLock<usize> = OnceLock::new();
    *MIN_BYTES.get_or_init(|| {
        std::env::var("LANCE_CUVS_ALLOC_TRACE_MIN_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2 * 1024 * 1024)
    })
}

thread_local! {
    // Backtrace capture/formatting itself allocates. Guard against
    // re-entering the logger from within its own capture -- skip logging
    // for that inner allocation rather than recursing.
    static IN_LOGGER: Cell<bool> = const { Cell::new(false) };
}

static ALLOC_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);

pub struct AllocTracer;

unsafe impl GlobalAlloc for AllocTracer {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        maybe_log("alloc", layout.size(), layout.align());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        maybe_log("alloc_zeroed", layout.size(), layout.align());
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        maybe_log("realloc", new_size, layout.align());
        out
    }
}

#[inline]
fn maybe_log(kind: &str, size: usize, align: usize) {
    if !enabled() || size < min_bytes() {
        return;
    }
    let should_log = IN_LOGGER.with(|f| {
        if f.get() {
            false
        } else {
            f.set(true);
            true
        }
    });
    if !should_log {
        return;
    }
    let seq = ALLOC_TRACE_SEQ.fetch_add(1, Ordering::Relaxed);
    // force_capture() always captures, unlike capture() which respects
    // RUST_BACKTRACE/RUST_LIB_BACKTRACE and may no-op if unset.
    let bt = std::backtrace::Backtrace::force_capture();
    eprintln!(
        "[alloc-trace #{seq}] {kind} size={size} align={align} tid={:?}\n{bt}\n---END #{seq}---",
        std::thread::current().id()
    );
    IN_LOGGER.with(|f| f.set(false));
}

#[global_allocator]
static GLOBAL_ALLOC_TRACER: AllocTracer = AllocTracer;
