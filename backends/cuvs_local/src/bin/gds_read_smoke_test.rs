// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Standalone smoke test for `cuvs_sys::cuvsReadLargeFile`.
//!
//! Independent of `TransformSlot`/RMM -- only exercises the raw C ABI against a host buffer and a
//! plain `cudaMalloc`'d device buffer, to validate the FFI signature/semantics before anything is
//! wired into the real fragment-scanning pipeline (see backend.rs for that).

use std::ffi::{CStr, CString, c_void};
use std::fs::File;
use std::io::Write;
use std::os::raw::c_uint;
use std::path::PathBuf;

type CudaError = c_uint;
const CUDA_SUCCESS: CudaError = 0;
type CudaMemcpyKind = c_uint;
const CUDA_MEMCPY_DEVICE_TO_HOST: CudaMemcpyKind = 2;

// cuvs-sys's bindgen build blocklists everything matching `cuda.*`, so this is hand-declared here
// the same way cuda.rs does for its own cudart usage, rather than relying on cuvs_sys to re-export
// it. Plain synchronous variants are enough for a standalone smoke test -- no stream needed.
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    fn cudaFree(ptr: *mut c_void) -> CudaError;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: CudaMemcpyKind) -> CudaError;
}

fn pattern_byte(i: usize) -> u8 {
    (i % 251) as u8
}

fn last_cuvs_error() -> String {
    unsafe {
        let ptr = cuvs_sys::cuvsGetLastErrorText();
        if ptr.is_null() {
            "<no error text>".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn check_cuvs(status: cuvs_sys::cuvsError_t, context: &str) {
    if status != cuvs_sys::cuvsError_t::CUVS_SUCCESS {
        panic!("{context} failed: {}", last_cuvs_error());
    }
}

fn check_cuda(status: CudaError, context: &str) {
    if status != CUDA_SUCCESS {
        panic!("{context} failed: cuda error {status}");
    }
}

fn main() {
    const TOTAL_BYTES: usize = 4 * 1024 * 1024;

    // Defaults to the OS temp dir (often not block-storage-backed, e.g. overlayfs/tmpfs in a
    // container -- GDS legitimately can't engage there regardless of driver support). Pass a
    // directory on real local disk (e.g. an NVMe mount) as the first arg to test GDS for real.
    let dir: PathBuf = std::env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let path: PathBuf = dir.join(format!("cuvs_read_large_file_smoke_{}.bin", std::process::id()));
    {
        let mut file = File::create(&path).expect("failed to create smoke-test file");
        let data: Vec<u8> = (0..TOTAL_BYTES).map(pattern_byte).collect();
        file.write_all(&data).expect("failed to write smoke-test file");
    }
    let path_c = CString::new(path.to_str().expect("non-utf8 path")).expect("path contains NUL");

    // 1. Full-file read into a host buffer.
    {
        let mut host_buf = vec![0u8; TOTAL_BYTES];
        let status = unsafe {
            cuvs_sys::cuvsReadLargeFile(path_c.as_ptr(), host_buf.as_mut_ptr() as *mut c_void, TOTAL_BYTES, 0)
        };
        check_cuvs(status, "cuvsReadLargeFile (host, full file)");
        for (i, &byte) in host_buf.iter().enumerate() {
            assert_eq!(byte, pattern_byte(i), "host mismatch at byte {i}");
        }
        println!("PASS: host full-file read ({TOTAL_BYTES} bytes) matches byte-for-byte");
    }

    // 2. Random-access, offset read into a device buffer -- the actually load-bearing case for
    // Lance's resolved byte ranges (non-zero offset, not a full-file read).
    {
        let read_offset: usize = TOTAL_BYTES / 4;
        let read_len: usize = TOTAL_BYTES / 2;

        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        check_cuda(unsafe { cudaMalloc(&mut device_ptr, read_len) }, "cudaMalloc");

        let status = unsafe {
            cuvs_sys::cuvsReadLargeFile(path_c.as_ptr(), device_ptr, read_len, read_offset as u64)
        };
        check_cuvs(status, "cuvsReadLargeFile (device, offset read)");

        let mut host_buf = vec![0u8; read_len];
        check_cuda(
            unsafe {
                cudaMemcpy(
                    host_buf.as_mut_ptr() as *mut c_void,
                    device_ptr,
                    read_len,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy (D2H verification copy)",
        );
        check_cuda(unsafe { cudaFree(device_ptr) }, "cudaFree");

        for (i, &byte) in host_buf.iter().enumerate() {
            assert_eq!(byte, pattern_byte(read_offset + i), "device mismatch at byte {i}");
        }
        println!("PASS: device offset read ({read_len} bytes at offset {read_offset}) matches byte-for-byte");
    }

    // 3. Error path: a nonexistent file should fail cleanly, not crash.
    {
        let bogus = CString::new("/nonexistent/cuvs_read_large_file_smoke.bin").unwrap();
        let mut host_buf = vec![0u8; 16];
        let status =
            unsafe { cuvs_sys::cuvsReadLargeFile(bogus.as_ptr(), host_buf.as_mut_ptr() as *mut c_void, 16, 0) };
        assert_eq!(status, cuvs_sys::cuvsError_t::CUVS_ERROR, "expected CUVS_ERROR for a missing file");
        println!("PASS: missing-file read reports CUVS_ERROR ({})", last_cuvs_error());
    }

    std::fs::remove_file(&path).ok();
    println!("ALL PASS");
}
