// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Debug-only instrumentation: logs a *raw, unresolved* stack trace for
//! every Rust-side allocation at or above `LANCE_CUVS_ALLOC_TRACE_MIN_BYTES`
//! bytes (default 2MiB -- comfortably below the smallest step in the
//! 2/4/8/16/32/64/128 MiB `mmap`/`munmap` doubling pattern this is meant to
//! diagnose). Disabled unless `LANCE_CUVS_ALLOC_TRACE=1` is set; when
//! disabled this is a transparent passthrough to the system allocator with
//! one cached `OnceLock` read of overhead per call.
//!
//! Deliberately uses the `backtrace` crate's unresolved capture
//! (`Backtrace::new_unresolved`) rather than `std::backtrace::Backtrace`:
//! symbolicating against this crate's `release-with-debug` binary (embedded
//! DWARF, multiple GB) live, inside a hot allocation path, is slow enough
//! to look like a hang. This logs raw instruction pointers only; resolve
//! them offline afterward (same file-offset + `gdb -batch -ex "info line
//! *<offset>"` approach used elsewhere in this investigation -- see
//! `profiling/PINNED_BUFFER_POOL_DESIGN.md`). The `_native.abi3.so`
//! mapping line from `/proc/self/maps` is logged once so the offset math
//! can be done directly from this file, without a separate `perf` capture.
//!
//! This intercepts allocations made by any Rust code statically linked into
//! this `.so` (lance-file, lance-encoding, lance-io, lance-table, this
//! crate, etc.) at the point the `Layout` is requested -- upstream of
//! whatever the system allocator (glibc `malloc`, eventually `mmap` for
//! large sizes) does with it. It does not see non-Rust (CUDA driver, glibc
//! internals) allocations.
//!
//! Temporary debugging aid for the pinned-buffer-pool investigation --
//! not meant to ship enabled, and not meant to stay in the tree long-term.
//!
//! Also feeds `large_alloc_gauge` (always on; only allocations of 256 MiB
//! and up touch it).

use crate::large_alloc_gauge;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::CStr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Reads an environment variable via raw `libc::getenv` rather than
/// `std::env::var`/`var_os`. This is load-bearing, not a style choice:
/// `std::env::var` allocates a `String` when the variable is set, and since
/// this crate is `#[global_allocator]`, that allocation re-enters
/// `AllocTracer::alloc` -> `maybe_log` -> `enabled()`/`min_bytes()` ->
/// (tries to) `OnceLock::get_or_init` again, on the same thread, from
/// within that same `OnceLock`'s own initializer. `OnceLock` is not
/// reentrant-safe for that and deadlocks waiting on itself. `getenv`
/// returns a pointer straight into the existing environment block with no
/// allocation at all, so it can't trigger this.
fn getenv_str(name: &CStr) -> Option<&'static str> {
    unsafe {
        let ptr = libc::getenv(name.as_ptr());
        if ptr.is_null() {
            None
        } else {
            CStr::from_ptr(ptr).to_str().ok()
        }
    }
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            getenv_str(c"LANCE_CUVS_ALLOC_TRACE"),
            Some("1") | Some("true")
        )
    })
}

fn min_bytes() -> usize {
    static MIN_BYTES: OnceLock<usize> = OnceLock::new();
    *MIN_BYTES.get_or_init(|| {
        getenv_str(c"LANCE_CUVS_ALLOC_TRACE_MIN_BYTES")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2 * 1024 * 1024)
    })
}

/// The `_native.abi3.so` line(s) from `/proc/self/maps`, read once and
/// cached, so the log is self-contained for offline offset computation.
fn maps_line_once() -> &'static str {
    static MAPS: OnceLock<String> = OnceLock::new();
    MAPS.get_or_init(|| {
        std::fs::read_to_string("/proc/self/maps")
            .ok()
            .map(|contents| {
                contents
                    .lines()
                    .filter(|l| l.contains("_native.abi3.so"))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| "<failed to read /proc/self/maps>".to_string())
    })
}

thread_local! {
    // Guard against the (unlikely, but possible) case of this logger's own
    // bookkeeping allocating above the threshold and re-entering itself.
    static IN_LOGGER: Cell<bool> = const { Cell::new(false) };
}

static ALLOC_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);

pub struct AllocTracer;

unsafe impl GlobalAlloc for AllocTracer {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            large_alloc_gauge::on_alloc(layout.size());
        }
        maybe_log("alloc", layout.size(), layout.align());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            large_alloc_gauge::on_alloc(layout.size());
        }
        maybe_log("alloc_zeroed", layout.size(), layout.align());
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
        large_alloc_gauge::on_dealloc(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, new_size) };
        // On failure the original allocation is untouched.
        if !out.is_null() {
            large_alloc_gauge::on_dealloc(layout.size());
            large_alloc_gauge::on_alloc(new_size);
        }
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

    // Unresolved capture: just walks the stack (frame-pointer or CFI based),
    // no symbol/DWARF lookup at all. The earlier hangs we saw here turned
    // out to be the reentrant OnceLock deadlock in enabled()/min_bytes()
    // (see getenv_str's doc comment), not this capture -- confirmed by
    // bisecting it out and reproducing the hang without it.
    let bt = backtrace::Backtrace::new_unresolved();
    let ips: Vec<String> = bt.frames().iter().map(|f| format!("{:?}", f.ip())).collect();

    eprintln!(
        "[alloc-trace #{seq}] {kind} size={size} align={align} tid={:?}\nmaps: {}\nips: {}\n---END #{seq}---",
        std::thread::current().id(),
        maps_line_once(),
        ips.join(" "),
    );

    IN_LOGGER.with(|f| f.set(false));
}

#[global_allocator]
static GLOBAL_ALLOC_TRACER: AllocTracer = AllocTracer;
