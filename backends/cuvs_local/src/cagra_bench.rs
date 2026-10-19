// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Standalone microbenchmark for graph-based coarse assignment: build a CAGRA
//! graph over finalized IVF centroids, then run a k=1 search for a stream of
//! query vectors, and compare it with exact brute-force top-1 for label
//! agreement and throughput.
//!
//! Not part of the build pipeline. Run through the `cagra_assign_bench` binary:
//!
//! ```text
//! cagra_assign_bench --centroids c.f32 --queries q.f32 --dim 2048 \
//!     --batch 131072 --itopk 32,64,128 --graph-degree 32,64
//! ```
//!
//! Inputs are raw little-endian row-major f32 files (see
//! `tools/extract_cagra_bench_inputs.py`). Cosine data must already be
//! normalized: Lance assigns cosine as L2 on unit vectors, so this bench uses
//! L2 only.

use crate::cuda::{
    DeviceTensor, Resources, check_cuvs, enable_rmm_pool_from_env,
};
use std::io::Read;
use std::ptr;
use std::time::Instant;

type BenchResult<T> = std::result::Result<T, String>;

struct Args {
    centroids: String,
    queries: String,
    dim: usize,
    num_queries: Option<usize>,
    batches: Vec<usize>,
    itopk: Vec<usize>,
    graph_degrees: Vec<usize>,
    /// 0 lets cuVS choose.
    max_queries: usize,
    /// "auto", "single_cta", "multi_cta" or "multi_kernel".
    algo: String,
    warmup_batches: usize,
}

fn parse_list(value: &str) -> BenchResult<Vec<usize>> {
    value
        .split(',')
        .map(|v| v.trim().parse::<usize>().map_err(|e| format!("bad list entry '{v}': {e}")))
        .collect()
}

fn parse_args() -> BenchResult<Args> {
    let mut centroids = None;
    let mut queries = None;
    let mut dim = None;
    let mut args = Args {
        centroids: String::new(),
        queries: String::new(),
        dim: 0,
        num_queries: None,
        batches: vec![128 * 1024],
        itopk: vec![32, 64, 128],
        graph_degrees: vec![32],
        max_queries: 0,
        algo: "auto".to_string(),
        warmup_batches: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--centroids" => centroids = Some(value()?),
            "--queries" => queries = Some(value()?),
            "--dim" => dim = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?),
            "--num-queries" => {
                args.num_queries = Some(value()?.parse::<usize>().map_err(|e| e.to_string())?)
            }
            "--batch" => args.batches = parse_list(&value()?)?,
            "--itopk" => args.itopk = parse_list(&value()?)?,
            "--graph-degree" => args.graph_degrees = parse_list(&value()?)?,
            "--max-queries" => {
                args.max_queries = value()?.parse::<usize>().map_err(|e| e.to_string())?
            }
            "--algo" => args.algo = value()?,
            "--warmup-batches" => {
                args.warmup_batches = value()?.parse::<usize>().map_err(|e| e.to_string())?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    args.centroids = centroids.ok_or("--centroids is required")?;
    args.queries = queries.ok_or("--queries is required")?;
    args.dim = dim.ok_or("--dim is required")?;
    Ok(args)
}

/// Reads up to `max_values` little-endian f32 values (all of them when `None`).
fn read_f32(path: &str, max_values: Option<usize>) -> BenchResult<Vec<f32>> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let file_values = file.metadata().map_err(|e| e.to_string())?.len() as usize / 4;
    let values = max_values.map_or(file_values, |m| m.min(file_values));
    let mut bytes = vec![0u8; values * 4];
    (&file)
        .take(bytes.len() as u64)
        .read_exact(&mut bytes)
        .map_err(|e| format!("read {path}: {e}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn err<T, E: std::fmt::Display>(result: Result<T, E>) -> BenchResult<T> {
    result.map_err(|e| e.to_string())
}

struct Batches<'a> {
    queries: &'a [f32],
    dim: usize,
    batch: usize,
}

impl<'a> Batches<'a> {
    fn rows(&self) -> usize {
        self.queries.len() / self.dim
    }

    fn iter(&self) -> impl Iterator<Item = (usize, &'a [f32])> + '_ {
        let (dim, batch) = (self.dim, self.batch);
        self.queries
            .chunks(batch * dim)
            .enumerate()
            .map(move |(i, chunk)| (i * batch, chunk))
    }
}

/// Device buffers for one batch of queries and their top-1 results.
struct SearchBuffers<N: crate::cuda::DlElement> {
    queries: DeviceTensor<f32>,
    neighbors: DeviceTensor<N>,
    distances: DeviceTensor<f32>,
    capacity: usize,
}

impl<N: crate::cuda::DlElement> SearchBuffers<N> {
    fn new(res: &Resources, capacity: usize, dim: usize) -> BenchResult<Self> {
        Ok(Self {
            queries: err(DeviceTensor::try_new(res, &[capacity, dim]))?,
            neighbors: err(DeviceTensor::try_new(res, &[capacity, 1]))?,
            distances: err(DeviceTensor::try_new(res, &[capacity, 1]))?,
            capacity,
        })
    }

    /// Uploads `chunk` and resizes the output tensors to match; the copy is
    /// synchronized so callers can time the search alone.
    fn load(&mut self, res: &Resources, chunk: &[f32], dim: usize) -> BenchResult<usize> {
        let rows = chunk.len() / dim;
        debug_assert!(rows <= self.capacity);
        err(self.queries.set_shape(&[rows, dim]))?;
        err(self.neighbors.set_shape(&[rows, 1]))?;
        err(self.distances.set_shape(&[rows, 1]))?;
        let stream = err(res.get_cuda_stream())?;
        err(self.queries.copy_from_host_async_on(stream, chunk))?;
        err(res.sync_stream())?;
        Ok(rows)
    }

}

fn read_back<T: crate::cuda::DlElement>(
    tensor: &DeviceTensor<T>,
    res: &Resources,
    rows: usize,
) -> BenchResult<Vec<T>> {
    let mut out = vec![T::default(); rows];
    err(tensor.copy_to_host_async(res, &mut out))?;
    err(res.sync_stream())?;
    Ok(out)
}

pub fn run() -> BenchResult<()> {
    let args = parse_args()?;
    let dim = args.dim;

    let centroids = read_f32(&args.centroids, None)?;
    if centroids.len() % dim != 0 {
        return Err(format!("{} values is not a multiple of dim {dim}", centroids.len()));
    }
    let k = centroids.len() / dim;
    let queries = read_f32(&args.queries, args.num_queries.map(|n| n * dim))?;
    let n = queries.len() / dim;
    eprintln!("centroids: {k} x {dim}; queries: {n} x {dim}");
    if k == 0 || n == 0 {
        return Err("empty centroid or query file".to_string());
    }

    err(enable_rmm_pool_from_env())?;
    let res = err(Resources::new())?;

    let mut centroid_dev = err(DeviceTensor::<f32>::try_new(&res, &[k, dim]))?;
    let stream = err(res.get_cuda_stream())?;
    err(centroid_dev.copy_from_host_async_on(stream, &centroids))?;
    err(res.sync_stream())?;

    // Exact reference: brute-force L2 top-1.
    let max_batch = *args.batches.iter().max().unwrap();
    let ref_batch = max_batch.min(n);
    let mut bf_index = ptr::null_mut();
    err(check_cuvs(
        unsafe { cuvs_sys::cuvsBruteForceIndexCreate(&mut bf_index) },
        "create brute-force index",
    ))?;
    err(check_cuvs(
        unsafe {
            cuvs_sys::cuvsBruteForceBuild(
                res.0,
                centroid_dev.as_mut_ptr(),
                cuvs_sys::cuvsDistanceType::L2Expanded,
                0.0,
                bf_index,
            )
        },
        "build brute-force index",
    ))?;
    let no_filter = cuvs_sys::cuvsFilter { addr: 0, type_: cuvs_sys::cuvsFilterType::NO_FILTER };

    let mut ref_labels = Vec::<u32>::with_capacity(n);
    let mut ref_dists = Vec::<f32>::with_capacity(n);
    let mut bf = SearchBuffers::<i64>::new(&res, ref_batch, dim)?;
    let bf_batches = Batches { queries: &queries, dim, batch: ref_batch };
    let mut bf_search_s = 0.0;
    for (i, (_, chunk)) in bf_batches.iter().enumerate() {
        let rows = bf.load(&res, chunk, dim)?;
        let start = Instant::now();
        err(check_cuvs(
            unsafe {
                cuvs_sys::cuvsBruteForceSearch(
                    res.0,
                    bf_index,
                    bf.queries.as_mut_ptr(),
                    bf.neighbors.as_mut_ptr(),
                    bf.distances.as_mut_ptr(),
                    no_filter,
                )
            },
            "brute-force search",
        ))?;
        err(res.sync_stream())?;
        // The first batch pays one-time kernel/workspace setup; leave it out.
        if i > 0 || bf_batches.rows() <= ref_batch {
            bf_search_s += start.elapsed().as_secs_f64();
        }
        ref_labels.extend(
            read_back(&bf.neighbors, &res, rows)?
                .into_iter()
                .map(|v| v as u32),
        );
        ref_dists.extend(read_back(&bf.distances, &res, rows)?);
    }
    println!(
        "brute-force exact (fp32, batch {ref_batch}): search_s={bf_search_s:.3} \
         (first batch excluded when there are several)"
    );

    for &graph_degree in &args.graph_degrees {
        let mut params = ptr::null_mut();
        err(check_cuvs(
            unsafe { cuvs_sys::cuvsCagraIndexParamsCreate(&mut params) },
            "create CAGRA index params",
        ))?;
        unsafe {
            (*params).metric = cuvs_sys::cuvsDistanceType::L2Expanded;
            (*params).graph_degree = graph_degree;
            (*params).intermediate_graph_degree = graph_degree * 2;
        }
        let mut index = ptr::null_mut();
        err(check_cuvs(
            unsafe { cuvs_sys::cuvsCagraIndexCreate(&mut index) },
            "create CAGRA index",
        ))?;
        // Non-owning padded view (search rejects a standard one): `centroid_dev` must outlive the index.
        let mut dataset = ptr::null_mut();
        err(check_cuvs(
            unsafe {
                cuvs_sys::cuvsDatasetMakePaddedView(res.0, centroid_dev.as_mut_ptr(), &mut dataset)
            },
            "wrap centroids as a CAGRA dataset view",
        ))?;
        let build_start = Instant::now();
        let build = check_cuvs(
            unsafe { cuvs_sys::cuvsCagraBuild(res.0, params, dataset, index) },
            "build CAGRA index",
        );
        let _ = unsafe { cuvs_sys::cuvsCagraIndexParamsDestroy(params) };
        err(build)?;
        err(res.sync_stream())?;
        println!(
            "cagra build: graph_degree={graph_degree} intermediate={} build_s={:.3}",
            graph_degree * 2,
            build_start.elapsed().as_secs_f64()
        );

        for &batch in &args.batches {
            let batch = batch.min(n);
            let mut buf = SearchBuffers::<u32>::new(&res, batch, dim)?;
            let batches = Batches { queries: &queries, dim, batch };
            for &itopk in &args.itopk {
                let mut sp = ptr::null_mut();
                err(check_cuvs(
                    unsafe { cuvs_sys::cuvsCagraSearchParamsCreate(&mut sp) },
                    "create CAGRA search params",
                ))?;
                unsafe {
                    (*sp).itopk_size = itopk;
                    (*sp).max_queries = args.max_queries;
                    (*sp).algo = match args.algo.as_str() {
                        "auto" => cuvs_sys::cuvsCagraSearchAlgo::AUTO,
                        "single_cta" => cuvs_sys::cuvsCagraSearchAlgo::SINGLE_CTA,
                        "multi_cta" => cuvs_sys::cuvsCagraSearchAlgo::MULTI_CTA,
                        "multi_kernel" => cuvs_sys::cuvsCagraSearchAlgo::MULTI_KERNEL,
                        other => return Err(format!("unknown --algo {other}")),
                    };
                }

                let mut search_s = 0.0;
                let mut timed_rows = 0usize;
                let mut agree = 0usize;
                let mut tie_ok = 0usize;
                for (i, (offset, chunk)) in batches.iter().enumerate() {
                    let rows = buf.load(&res, chunk, dim)?;
                    let start = Instant::now();
                    err(check_cuvs(
                        unsafe {
                            cuvs_sys::cuvsCagraSearch(
                                res.0,
                                sp,
                                index,
                                buf.queries.as_mut_ptr(),
                                buf.neighbors.as_mut_ptr(),
                                buf.distances.as_mut_ptr(),
                                no_filter,
                            )
                        },
                        "CAGRA search",
                    ))?;
                    err(res.sync_stream())?;
                    let elapsed = start.elapsed().as_secs_f64();
                    if i >= args.warmup_batches.min(batches.rows().div_ceil(batch) - 1) {
                        search_s += elapsed;
                        timed_rows += rows;
                    }
                    let labels = read_back(&buf.neighbors, &res, rows)?;
                    let dists = read_back(&buf.distances, &res, rows)?;
                    for r in 0..rows {
                        let (label, dist) = (labels[r], dists[r]);
                        let (rl, rd) = (ref_labels[offset + r], ref_dists[offset + r]);
                        if label == rl {
                            agree += 1;
                            tie_ok += 1;
                        } else if dist <= rd * (1.0 + 1e-5) + 1e-6 {
                            // A different centroid at an equal distance.
                            tie_ok += 1;
                        }
                    }
                }
                let _ = unsafe { cuvs_sys::cuvsCagraSearchParamsDestroy(sp) };
                println!(
                    "cagra search: graph_degree={graph_degree} batch={batch} itopk={itopk} \
                     algo={} max_queries={} timed_rows={timed_rows} search_s={search_s:.3} \
                     rows_per_s={:.0} label_agree={:.5} agree_or_tie={:.5}",
                    args.algo,
                    args.max_queries,
                    timed_rows as f64 / search_s.max(1e-9),
                    agree as f64 / n as f64,
                    tie_ok as f64 / n as f64
                );
            }
        }
        let _ = unsafe { cuvs_sys::cuvsCagraIndexDestroy(index) };
        let _ = unsafe { cuvs_sys::cuvsDatasetDestroy(dataset) };
    }
    let _ = unsafe { cuvs_sys::cuvsBruteForceIndexDestroy(bf_index) };
    Ok(())
}
