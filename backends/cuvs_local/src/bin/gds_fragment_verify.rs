// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cross-check verification for `gds_layout::resolve_fragment_column` (2.1 "structural" format,
//! non-nullable `FixedSizeList<f32>` only): resolves a fragment's vector column and reads it via
//! `cuvsReadLargeFile`, then compares against the SAME rows fetched through Lance's own normal
//! decode path (`dataset.scan()`), element-for-element (exact bit comparison, not approximate).
//! Given how much of `gds_layout.rs` is "derive the formula from static reading" rather than
//! runtime-verified (see `profiling/GDS_PORTING_PLAN.md`), this is the essential empirical check
//! before trusting it in the real `TransformSlot` pipeline.
//!
//! Requires a dataset written with Lance's current default format (2.1 "structural") and a
//! non-nullable vector column -- `gds_layout.rs`'s documented narrow scope; this tool itself
//! fails loudly (not silently) if the scanned ground truth turns out to have any null rows.
//!
//! Usage:
//! ```text
//! cargo run --release --bin gds_fragment_verify -- /tmp/some_dataset.lance vector 2048
//! ```

use arrow_array::{Array, FixedSizeListArray, Float32Array};
use futures::TryStreamExt;
use lance::dataset::Dataset;
use lance_cuvs_backend_cu12_local::gds_layout::resolve_fragment_column;
use std::ffi::{CStr, CString, c_void};
use std::os::raw::c_uint;
use std::path::Path;

type CudaError = c_uint;
const CUDA_SUCCESS: CudaError = 0;
type CudaMemcpyKind = c_uint;
const CUDA_MEMCPY_DEVICE_TO_HOST: CudaMemcpyKind = 2;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaFree(ptr: *mut c_void) -> CudaError;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind) -> CudaError;
}

fn check_cuda(status: CudaError, context: &str) {
    if status != CUDA_SUCCESS {
        eprintln!("CUDA failed to {context}: cuda error {status}");
        std::process::exit(1);
    }
}

fn check_cuvs(status: cuvs_sys::cuvsError_t, context: &str) {
    if status != cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        let message = unsafe {
            let ptr = cuvs_sys::cuvsGetLastErrorText();
            if ptr.is_null() {
                "<no error text>".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        eprintln!("cuVS failed to {context}: {message}");
        std::process::exit(1);
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: gds_fragment_verify <dataset_path> <column> <dim>");
        std::process::exit(2);
    }
    let dataset_path = &args[1];
    let column = &args[2];
    let dim: u64 = args[3].parse().expect("dim must be a positive integer");

    let dataset = Dataset::open(dataset_path).await.unwrap_or_else(|error| {
        eprintln!("failed to open dataset {dataset_path}: {error}");
        std::process::exit(1);
    });

    let fragments = dataset.get_fragments();
    let fragment = fragments.first().unwrap_or_else(|| {
        eprintln!("dataset has no fragments");
        std::process::exit(1);
    });

    let dataset_root = Path::new(dataset_path);
    let (local_path, plans) = resolve_fragment_column(&dataset, dataset_root, fragment, column, dim, 4)
        .await
        .unwrap_or_else(|error| {
            eprintln!("resolve_fragment_column failed: {error}");
            std::process::exit(1);
        });
    println!("resolved {} page(s) in {}", plans.len(), local_path.display());
    for plan in &plans {
        println!(
            "  page: rows [{}, {}), values @ file_offset={} byte_len={}",
            plan.row_start,
            plan.row_start + plan.num_rows,
            plan.file_offset,
            plan.byte_len
        );
    }

    let total_rows: u64 = plans.iter().map(|p| p.num_rows).sum();
    let total_value_bytes = (total_rows * dim * 4) as usize;

    let mut dev_ptr: *mut c_void = std::ptr::null_mut();
    check_cuda(
        unsafe { cudaMalloc(&mut dev_ptr, total_value_bytes) },
        "allocate device buffer",
    );

    let path_c =
        CString::new(local_path.to_str().expect("non-utf8 path")).expect("path contains NUL");
    for plan in &plans {
        let dst_offset_bytes = (plan.row_start * dim * 4) as usize;
        let status = unsafe {
            cuvs_sys::cuvsReadLargeFile(
                path_c.as_ptr(),
                (dev_ptr as *mut u8).add(dst_offset_bytes) as *mut c_void,
                plan.byte_len as usize,
                plan.file_offset,
            )
        };
        check_cuvs(status, "cuvsReadLargeFile");
    }

    let mut gds_values = vec![0f32; (total_rows * dim) as usize];
    check_cuda(
        unsafe {
            cudaMemcpy(
                gds_values.as_mut_ptr() as *mut c_void,
                dev_ptr,
                total_value_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        },
        "copy device buffer back to host for verification",
    );
    unsafe { cudaFree(dev_ptr) };

    // Ground truth: Lance's own normal decode path. Default scan order deliberately used (not
    // scan_in_order(false), which the real production pipeline uses for throughput) -- physical
    // row order must match the GDS-resolved page order exactly for this comparison to be
    // meaningful.
    let mut scanner = dataset.scan();
    scanner.project(&[column.as_str()]).unwrap_or_else(|error| {
        eprintln!("scanner.project failed: {error}");
        std::process::exit(1);
    });
    let mut stream = scanner.try_into_stream().await.unwrap_or_else(|error| {
        eprintln!("scanner.try_into_stream failed: {error}");
        std::process::exit(1);
    });

    let mut expected_values: Vec<f32> = Vec::with_capacity((total_rows * dim) as usize);
    loop {
        let batch = stream.try_next().await.unwrap_or_else(|error| {
            eprintln!("scan failed: {error}");
            std::process::exit(1);
        });
        let Some(batch) = batch else { break };
        let column_array = batch.column_by_name(column).unwrap_or_else(|| {
            eprintln!("scanned batch missing column '{column}'");
            std::process::exit(1);
        });
        let fsl = column_array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap_or_else(|| {
                eprintln!("column '{column}' is not a FixedSizeListArray");
                std::process::exit(1);
            });
        if fsl.null_count() != 0 {
            eprintln!(
                "FAIL: column '{column}' has {} null rows -- gds_layout.rs only supports \
                 non-nullable columns, this dataset is out of scope for this verify tool",
                fsl.null_count()
            );
            std::process::exit(1);
        }
        let values = fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap_or_else(|| {
                eprintln!("column '{column}' values are not a Float32Array -- unexpected dtype");
                std::process::exit(1);
            });
        expected_values.extend_from_slice(values.values());
    }

    if expected_values.len() != gds_values.len() {
        eprintln!(
            "FAIL: element count mismatch -- GDS-resolved {} f32 elements, scanned {} f32 elements",
            gds_values.len(),
            expected_values.len()
        );
        std::process::exit(1);
    }

    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for (i, (gds_v, scanned_v)) in gds_values.iter().zip(expected_values.iter()).enumerate() {
        // Exact bitwise comparison, not approximate: these are supposed to be the literal same
        // on-disk bytes reinterpreted as f32, not a numerically-close computation, so any
        // difference at all -- including a NaN-bit-pattern difference `==` would hide -- is a
        // real bug in the byte-range resolution above.
        if gds_v.to_bits() != scanned_v.to_bits() {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, *gds_v, *scanned_v));
            }
        }
    }

    if mismatches == 0 {
        println!(
            "PASS: {total_rows} rows x {dim} dims ({} f32 elements) match Lance's own decoded \
             read exactly",
            gds_values.len()
        );
    } else {
        let (i, gds_v, scanned_v) = first_mismatch.unwrap();
        eprintln!(
            "FAIL: {mismatches} value mismatches (first at flat index {i}: GDS={gds_v}, \
             scanned={scanned_v})"
        );
        std::process::exit(1);
    }
}
