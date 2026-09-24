// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Gauge of live large Rust allocations (>= [`MIN_BYTES`]), fed by the
//! global allocator in `alloc_trace.rs`.
//!
//! During the transform/append stage the only allocations this large are the
//! per-fragment decoded vector buffers (arrow's concat inside Lance's
//! decoder, 610 MiB each on the benchmark dataset), which live until the
//! drain frees them after the H2D copy. So the live count over the stage is
//! how many decoded batches the scanner and the backend's queues hold at
//! once -- the number of slots a pinned pool would need to serve them all.
//!
//! Counting is always on (it has to be, for the count to be right when a
//! window starts); time weighting only happens between [`begin_window`] and
//! [`end_window`]. Large allocations are rare (hundreds per run), so a mutex
//! is fine. Nothing here allocates: this runs inside the global allocator.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Well above the scan stage's 128 MiB decode intermediates, below the
/// 610 MiB decoded batch buffers.
pub(crate) const MIN_BYTES: usize = 256 * 1024 * 1024;

/// Live counts at or above this share the last histogram bucket.
const MAX_TRACKED: usize = 128;

struct State {
    live: usize,
    live_bytes: usize,
    window: Option<Window>,
}

struct Window {
    start: Instant,
    last_change: Instant,
    live_at_start: usize,
    peak: usize,
    peak_bytes: usize,
    time_to_peak: Duration,
    // time_at[k] = time spent with k large allocations live.
    time_at: [Duration; MAX_TRACKED + 1],
}

static STATE: Mutex<State> = Mutex::new(State {
    live: 0,
    live_bytes: 0,
    window: None,
});

fn lock() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

impl State {
    fn change(&mut self, allocated: bool, size: usize) {
        let now = self.window.as_ref().map(|_| Instant::now());
        if let (Some(window), Some(now)) = (self.window.as_mut(), now) {
            window.time_at[self.live.min(MAX_TRACKED)] += now - window.last_change;
            window.last_change = now;
        }
        if allocated {
            self.live += 1;
            self.live_bytes += size;
        } else {
            // Saturating: an allocation made before this `.so` was loaded
            // cannot be freed through it, but stay safe regardless.
            self.live = self.live.saturating_sub(1);
            self.live_bytes = self.live_bytes.saturating_sub(size);
        }
        if let (Some(window), Some(now)) = (self.window.as_mut(), now) {
            if self.live > window.peak {
                window.peak = self.live;
                window.peak_bytes = self.live_bytes;
                window.time_to_peak = now - window.start;
            }
        }
    }
}

#[inline]
pub(crate) fn on_alloc(size: usize) {
    if size >= MIN_BYTES {
        lock().change(true, size);
    }
}

#[inline]
pub(crate) fn on_dealloc(size: usize) {
    if size >= MIN_BYTES {
        lock().change(false, size);
    }
}

/// Starts time-weighting the live count; replaces any open window.
pub(crate) fn begin_window() {
    let now = Instant::now();
    let mut state = lock();
    let live = state.live;
    let live_bytes = state.live_bytes;
    state.window = Some(Window {
        start: now,
        last_change: now,
        live_at_start: live,
        peak: live,
        peak_bytes: live_bytes,
        time_to_peak: Duration::ZERO,
        time_at: [Duration::ZERO; MAX_TRACKED + 1],
    });
}

pub(crate) struct GaugeReport {
    pub(crate) live_at_start: usize,
    pub(crate) live_at_end: usize,
    pub(crate) peak: usize,
    pub(crate) peak_bytes: usize,
    pub(crate) time_to_peak: Duration,
    pub(crate) window: Duration,
    /// Time-weighted mean live count.
    pub(crate) mean: f64,
    /// Smallest k such that the live count was <= k for 50/90/99% of the
    /// window.
    pub(crate) p50: usize,
    pub(crate) p90: usize,
    pub(crate) p99: usize,
}

/// Closes the window opened by [`begin_window`]; `None` if none is open.
pub(crate) fn end_window() -> Option<GaugeReport> {
    let now = Instant::now();
    let mut state = lock();
    let live = state.live;
    let mut window = state.window.take()?;
    drop(state);
    window.time_at[live.min(MAX_TRACKED)] += now - window.last_change;

    let total = now - window.start;
    let total_s = total.as_secs_f64();
    let mean = if total_s > 0.0 {
        window
            .time_at
            .iter()
            .enumerate()
            .map(|(k, t)| k as f64 * t.as_secs_f64())
            .sum::<f64>()
            / total_s
    } else {
        live as f64
    };
    let percentile = |fraction: f64| {
        let mut cumulative = 0.0;
        for (k, t) in window.time_at.iter().enumerate() {
            cumulative += t.as_secs_f64();
            if cumulative >= fraction * total_s {
                return k;
            }
        }
        MAX_TRACKED
    };
    Some(GaugeReport {
        live_at_start: window.live_at_start,
        live_at_end: live,
        peak: window.peak,
        peak_bytes: window.peak_bytes,
        time_to_peak: window.time_to_peak,
        window: total,
        mean,
        p50: percentile(0.50),
        p90: percentile(0.90),
        p99: percentile(0.99),
    })
}
