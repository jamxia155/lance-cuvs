// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//
// Standalone prototype: grouped-GEMM IVF assignment of vectors to a set of centroids.
//
// The K centroids are grouped into M lists (spherical k-means). A batch of queries is assigned by
// (1) scoring the M list centers and taking each query's top-p lists, (2) bucketing the
// (list, query) pairs by list, and (3) for each list, gathering its queries and running one dense
// TF32 GEMM against that list's centroids plus a row argmax, then (4) merging each query's p
// partial results. Compared with an exhaustive TF32 GEMM + argmax over all K centroids (the same
// shape as cuVS's "unfused" backend), which is timed in the same program.
//
// Scores are inner products (the inputs are unit-norm: nearest by L2 == max dot).
//
//   nvcc -O3 -std=c++17 -arch=sm_90 grouped_assign_proto.cu -lcublas -o grouped_assign_proto
//   ./grouped_assign_proto centroids.f32 queries.f32 2048 [lists=181] [probes=8] [batch=131072]
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);   \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)
#define CB(x)                                                                              \
  do {                                                                                     \
    cublasStatus_t s_ = (x);                                                               \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                     \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__);          \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

using KV = cub::KeyValuePair<int, float>;
struct ArgMaxOp {
  __device__ KV operator()(const KV& a, const KV& b) const
  {
    return (b.value > a.value || (b.value == a.value && b.key < a.key)) ? b : a;
  }
};

static std::vector<float> read_f32(const char* path, size_t max_values)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  size_t n = std::min<size_t>(f.tellg() / 4, max_values);
  std::vector<float> v(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), n * 4);
  return v;
}

// One block per row: best (score, column) of `cols` scores.
__global__ void argmax_rows(const float* scores, int cols, float* best, int* arg)
{
  const float* row = scores + (size_t)blockIdx.x * cols;
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = row[c];
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, 256>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) { best[blockIdx.x] = r.value; arg[blockIdx.x] = r.key; }
}

// Like argmax_rows, but writes (score, id_map[col]) to the slot given by pairs[row].
__global__ void argmax_scatter(const float* scores,
                               int cols,
                               int stride,
                               const int* pairs,
                               const int* id_map,
                               float* res_score,
                               int* res_id)
{
  const float* row = scores + (size_t)blockIdx.x * stride;
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = row[c];
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, 256>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) {
    int slot       = pairs[blockIdx.x];
    res_score[slot] = r.value;
    res_id[slot]    = id_map[r.key];
  }
}

// Top-p columns of each row of `scores` [rows, M] by repeated block argmax (p small).
// SWAP mode: scores are column-major [g queries x n list centroids] with leading dimension `ld`
// (query fastest), so one thread per query loops over the list's columns with coalesced reads.
__global__ void argmax_scatter_T(const float* scores, int g, int n, int ld, const int* pairs, const int* id_map,
                                 float* res_score, int* res_id)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= g) return;
  float best = -INFINITY;
  int arg    = 0;
  for (int c = 0; c < n; c++) {
    float v = scores[(size_t)c * ld + q];
    if (v > best) { best = v; arg = c; }
  }
  int slot        = pairs[q];
  res_score[slot] = best;
  res_id[slot]    = id_map[arg];
}

// GROUPED mode: gather every (list, query) pair's query row in one launch.
__global__ void gather_all(const float* q, int D, const int* sorted_vals, int p, float* out)
{
  int query         = sorted_vals[blockIdx.x] / p;
  const float4* src = reinterpret_cast<const float4*>(q + (size_t)query * D);
  float4* dst       = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * D);
  for (int c = threadIdx.x; c < D / 4; c += blockDim.x) dst[c] = src[c];
}

// GROUPED mode: argmax of pair i's score row (rows of one list are contiguous, n floats each).
__global__ void argmax_all(const float* sc, const int* list_of_pair, const int* goff, const int* lcount, const int* ldl,
                           const long long* scoff, const int* moff, const int* sorted_vals, const int* perm_id,
                           float* res_score, int* res_id)
{
  int i = blockIdx.x, l = list_of_pair[i];
  int n = lcount[l];
  const float* row = sc + scoff[l] + (size_t)(i - goff[l]) * ldl[l];
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < n; c += blockDim.x) {
    float v = row[c];
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, 256>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) {
    int slot        = sorted_vals[i];
    res_score[slot] = r.value;
    res_id[slot]    = perm_id[moff[l] + r.key];
  }
}

__global__ void top_p_rows(const float* scores, int M, int stride, int p, int* out)
{
  extern __shared__ float row[];
  for (int c = threadIdx.x; c < M; c += blockDim.x) row[c] = scores[(size_t)blockIdx.x * stride + c];
  __syncthreads();
  using BR = cub::BlockReduce<KV, 256>;
  __shared__ typename BR::TempStorage t;
  __shared__ int chosen;
  for (int j = 0; j < p; j++) {
    KV loc(0, -INFINITY);
    for (int c = threadIdx.x; c < M; c += blockDim.x)
      if (row[c] > loc.value) loc = KV(c, row[c]);
    KV r = BR(t).Reduce(loc, ArgMaxOp());
    if (threadIdx.x == 0) { chosen = r.key; out[(size_t)blockIdx.x * p + j] = r.key; row[r.key] = -INFINITY; }
    __syncthreads();
  }
}

__global__ void init_pairs(const int* lists, int n, int* keys, int* vals)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { keys[i] = lists[i]; vals[i] = i; }
}

__global__ void count_keys(const int* keys, int n, int* counts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(counts + keys[i], 1);
}

// Gathers the query rows of pairs sorted_vals[off .. off+g) (pair / p = query) into `out`.
__global__ void gather_queries(const float* q, int D, const int* sorted_vals, int off, int p, float* out)
{
  int r        = blockIdx.x;
  int query    = sorted_vals[off + r] / p;
  const float4* src = reinterpret_cast<const float4*>(q + (size_t)query * D);
  float4* dst       = reinterpret_cast<float4*>(out + (size_t)r * D);
  for (int c = threadIdx.x; c < D / 4; c += blockDim.x) dst[c] = src[c];
}

__global__ void merge_pairs(const float* res_score, const int* res_id, int p, int nq, int* labels)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nq) return;
  float best = -INFINITY;
  int arg    = 0;
  for (int j = 0; j < p; j++) {
    float s = res_score[(size_t)q * p + j];
    if (s > best) { best = s; arg = res_id[(size_t)q * p + j]; }
  }
  labels[q] = arg;
}

// Spherical-k-means accumulate: sums[assign[i]] += x[i].
__global__ void accumulate(const float* x, const int* assign, int D, float* sums, int* counts)
{
  int i = blockIdx.x;
  int a = assign[i];
  for (int c = threadIdx.x; c < D; c += blockDim.x) atomicAdd(sums + (size_t)a * D + c, x[(size_t)i * D + c]);
  if (threadIdx.x == 0) atomicAdd(counts + a, 1);
}

__global__ void normalize_rows(float* x, int D)
{
  float* row = x + (size_t)blockIdx.x * D;
  float s    = 0;
  for (int c = threadIdx.x; c < D; c += blockDim.x) s += row[c] * row[c];
  using BR = cub::BlockReduce<float, 256>;
  __shared__ typename BR::TempStorage t;
  __shared__ float inv;
  float tot = BR(t).Sum(s);
  if (threadIdx.x == 0) inv = tot > 0 ? rsqrtf(tot) : 1.0f;
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += blockDim.x) row[c] *= inv;
}

static void gemm_tf32(cublasHandle_t h, int n, int m, int D, const float* A, const float* B, float* C, int ldc = -1)
{
  // C (row-major [m, n], row stride ldc >= n) = B (row-major [m, D]) * A (row-major [n, D])^T, TF32.
  // Callers pad only the output's leading dimension; no extra columns are computed.
  const float one = 1.f, zero = 0.f;
  if (ldc < 0) ldc = n;
  CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, D, &one, A, CUDA_R_32F, D, B, CUDA_R_32F, D,
                  &zero, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(int argc, char** argv)
{
  if (argc < 4) { fprintf(stderr, "usage: %s centroids.f32 queries.f32 dim [lists] [probes] [batch]\n", argv[0]); return 1; }
  const int D = atoi(argv[3]);
  auto hc     = read_f32(argv[1], (size_t)-1);
  const int K = hc.size() / D;
  const int M = argc > 4 ? atoi(argv[4]) : std::max(1, (int)std::lround(std::sqrt((double)K)));
  const int P = argc > 5 ? atoi(argv[5]) : 8;
  const int B = argc > 6 ? atoi(argv[6]) : 131072;
  auto hq     = read_f32(argv[2], (size_t)-1);
  const int NQ = (hq.size() / D) / B * B;  // whole batches only
  printf("K=%d D=%d M=%d probes=%d batch=%d queries=%d\n", K, D, M, P, B, NQ);

  cublasHandle_t cb;
  CB(cublasCreate(&cb));
  float *dC, *dQ;
  CK(cudaMalloc(&dC, (size_t)K * D * 4));
  CK(cudaMalloc(&dQ, (size_t)NQ * D * 4));
  CK(cudaMemcpy(dC, hc.data(), (size_t)K * D * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dQ, hq.data(), (size_t)NQ * D * 4, cudaMemcpyHostToDevice));

  // ---- Group the centroids into M lists (spherical k-means, 10 iterations). ----
  float *dS, *dSums, *dTmp;
  int *dAssign, *dCnt;
  float* dBestK;
  const int MpadS = (M + 7) / 8 * 8;
  CK(cudaMalloc(&dS, (size_t)MpadS * D * 4));
  CK(cudaMemset(dS, 0, (size_t)MpadS * D * 4));
  CK(cudaMalloc(&dSums, (size_t)M * D * 4));
  CK(cudaMalloc(&dTmp, (size_t)K * M * 4));
  CK(cudaMalloc(&dAssign, (size_t)K * 4));
  CK(cudaMalloc(&dBestK, (size_t)K * 4));
  CK(cudaMalloc(&dCnt, (size_t)M * 4));
  {
    std::mt19937 rng(7);
    std::vector<int> idx(K);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int m = 0; m < M; m++)
      CK(cudaMemcpy(dS + (size_t)m * D, dC + (size_t)idx[m] * D, (size_t)D * 4, cudaMemcpyDeviceToDevice));
    for (int it = 0; it < 10; it++) {
      gemm_tf32(cb, M, K, D, dS, dC, dTmp);
      argmax_rows<<<K, 256>>>(dTmp, M, dBestK, dAssign);
      CK(cudaMemset(dSums, 0, (size_t)M * D * 4));
      CK(cudaMemset(dCnt, 0, (size_t)M * 4));
      accumulate<<<K, 256>>>(dC, dAssign, D, dSums, dCnt);
      normalize_rows<<<M, 256>>>(dSums, D);
      std::vector<int> cnt(M);
      CK(cudaMemcpy(cnt.data(), dCnt, (size_t)M * 4, cudaMemcpyDeviceToHost));
      for (int m = 0; m < M; m++)  // keep the old center for an empty list
        if (cnt[m] > 0) CK(cudaMemcpy(dS + (size_t)m * D, dSums + (size_t)m * D, (size_t)D * 4, cudaMemcpyDeviceToDevice));
    }
    gemm_tf32(cb, M, K, D, dS, dC, dTmp);
    argmax_rows<<<K, 256>>>(dTmp, M, dBestK, dAssign);
    if (std::getenv("BALANCE") && K % M == 0) {
      // Exactly balanced lists (K/M centroids each): greedy capacity-constrained assignment over all
      // (centroid, list) scores in descending order, then re-center; repeated a few times.
      const int cap   = K / M;
      const int biter = std::getenv("BAL_ITERS") ? atoi(std::getenv("BAL_ITERS")) : 5;
      std::vector<float> sc((size_t)K * M);
      std::vector<int> ha(K);
      for (int it = 0; it <= biter; it++) {
        gemm_tf32(cb, M, K, D, dS, dC, dTmp);
        CK(cudaMemcpy(sc.data(), dTmp, sc.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<uint64_t> keys(sc.size());
        for (size_t i = 0; i < sc.size(); i++) {
          uint32_t u; std::memcpy(&u, &sc[i], 4);
          u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // order-preserving float -> uint
          keys[i] = ((uint64_t)u << 32) | (uint32_t)i;
        }
        std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());
        std::fill(ha.begin(), ha.end(), -1);
        std::vector<int> used(M, 0);
        for (uint64_t k : keys) {
          uint32_t i = (uint32_t)k;
          int c = i / M, l = i % M;
          if (ha[c] < 0 && used[l] < cap) { ha[c] = l; used[l]++; }
        }
        CK(cudaMemcpy(dAssign, ha.data(), (size_t)K * 4, cudaMemcpyHostToDevice));
        if (it == biter) break;
        CK(cudaMemset(dSums, 0, (size_t)M * D * 4));
        CK(cudaMemset(dCnt, 0, (size_t)M * 4));
        accumulate<<<K, 256>>>(dC, dAssign, D, dSums, dCnt);
        normalize_rows<<<M, 256>>>(dSums, D);
        CK(cudaMemcpy(dS, dSums, (size_t)M * D * 4, cudaMemcpyDeviceToDevice));
      }
      printf("balanced lists: %d centroids each\n", cap);
    }
  }
  // Lists in CSR form; centroids permuted so each list is contiguous.
  std::vector<int> hAssign(K);
  CK(cudaMemcpy(hAssign.data(), dAssign, (size_t)K * 4, cudaMemcpyDeviceToHost));
  std::vector<int> mcount(M, 0), moff(M + 1, 0);
  for (int a : hAssign) mcount[a]++;
  for (int m = 0; m < M; m++) moff[m + 1] = moff[m] + mcount[m];
  std::vector<int> fill(moff.begin(), moff.end() - 1), cperm_id(K);
  std::vector<float> hperm((size_t)K * D);
  for (int c = 0; c < K; c++) {
    int pos         = fill[hAssign[c]]++;
    cperm_id[pos]   = c;
    std::copy(hc.begin() + (size_t)c * D, hc.begin() + (size_t)(c + 1) * D, hperm.begin() + (size_t)pos * D);
  }
  int nmax = *std::max_element(mcount.begin(), mcount.end());
  // Round GEMM output leading dimensions up to a multiple of 8 floats (cuBLAS tensor-core kernels
  // want 16-byte aligned leading dimensions); NOPAD=1 turns it off for A/B comparisons.
  const int ALIGN = std::getenv("PADTO") ? atoi(std::getenv("PADTO")) : std::getenv("NOPAD") ? 1 : 8;  // PADTO=128/256 + PADM=1: tile-size padding proxy
  auto pad = [&](int x) { return (x + ALIGN - 1) / ALIGN * ALIGN; };
  const int Mpad = pad(M), nmax_pad = pad(nmax);
  printf("lists: avg %.1f, max %d, min %d centroids\n", (double)K / M, nmax, *std::min_element(mcount.begin(), mcount.end()));
  float* dPerm;
  int *dPermId;
  CK(cudaMalloc(&dPerm, ((size_t)K + ALIGN + 16) * D * 4));
  CK(cudaMemset(dPerm, 0, ((size_t)K + ALIGN + 16) * D * 4));
  CK(cudaMalloc(&dPermId, (size_t)K * 4));
  CK(cudaMemcpy(dPerm, hperm.data(), (size_t)K * D * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dPermId, cperm_id.data(), (size_t)K * 4, cudaMemcpyHostToDevice));

  // ---- Buffers for the grouped assignment. ----
  const int NP = B * P;
  float *dS1, *dQg, *dSc, *dResScore;
  int *dLists, *dKeys, *dKeysSorted, *dVals, *dValsSorted, *dResId, *dCounts, *dLabels;
  CK(cudaMalloc(&dS1, (size_t)B * Mpad * 4));
  // Per-stream scratch (gathered queries, list scores) and cuBLAS handles; STREAMS=1 reproduces the serial loop.
  const int NS = std::getenv("STREAMS") ? std::max(1, atoi(std::getenv("STREAMS"))) : 8;
  std::vector<cudaStream_t> streams(NS);
  std::vector<cublasHandle_t> handles(NS);
  std::vector<float*> qgs(NS), scs(NS);
  std::vector<cudaEvent_t> joinEv(NS);
  cudaEvent_t forkEv;
  CK(cudaEventCreateWithFlags(&forkEv, cudaEventDisableTiming));
  for (int i = 0; i < NS; i++) {
    CK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    CB(cublasCreate(&handles[i]));
    CB(cublasSetStream(handles[i], streams[i]));
    CK(cudaMalloc(&qgs[i], (size_t)B * D * 4));
    CK(cudaMalloc(&scs[i], (size_t)B * nmax_pad * 4));
    CK(cudaEventCreateWithFlags(&joinEv[i], cudaEventDisableTiming));
  }
  (void)dQg; (void)dSc;
  CK(cudaMalloc(&dResScore, (size_t)NP * 4));
  CK(cudaMalloc(&dResId, (size_t)NP * 4));
  CK(cudaMalloc(&dLists, (size_t)NP * 4));
  CK(cudaMalloc(&dKeys, (size_t)NP * 4));
  CK(cudaMalloc(&dKeysSorted, (size_t)NP * 4));
  CK(cudaMalloc(&dVals, (size_t)NP * 4));
  CK(cudaMalloc(&dValsSorted, (size_t)NP * 4));
  CK(cudaMalloc(&dCounts, (size_t)M * 4));
  CK(cudaMalloc(&dLabels, (size_t)NQ * 4));
  void* dTmpSort = nullptr;
  size_t sortBytes = 0;
  const int endBit = std::max(1, (int)std::ceil(std::log2((double)M)));
  cub::DeviceRadixSort::SortPairs(dTmpSort, sortBytes, dKeys, dKeysSorted, dVals, dValsSorted, NP, 0, endBit);
  CK(cudaMalloc(&dTmpSort, sortBytes));

  cudaEvent_t e0, e1, e2, e3, e4;
  for (auto* e : {&e0, &e1, &e2, &e3, &e4}) CK(cudaEventCreate(e));
  double t_stage1 = 0, t_sort = 0, t_lists = 0, t_total = 0;
  const bool profile = std::getenv("PROFILE") != nullptr;
  const bool padm    = std::getenv("PADM") != nullptr;
  const char* modeEnv = std::getenv("MODE");  // loop (default) | swap | grouped
  const bool swap     = modeEnv && std::string(modeEnv) == "swap";
  const bool grouped  = modeEnv && std::string(modeEnv) == "grouped";
  float* dQgAll = nullptr;
  float* dScAll = nullptr;
  size_t scAllCap = 0;
  const void **dPA = nullptr, **dPB = nullptr, **dPC = nullptr;
  int *dGoff = nullptr, *dLd = nullptr, *dLcnt = nullptr, *dMoff = nullptr;
  long long* dScoff = nullptr;
  if (grouped) {
    scAllCap = (size_t)NP * (nmax + 8) + 64;
    CK(cudaMalloc(&dQgAll, (size_t)NP * D * 4));
    CK(cudaMalloc(&dScAll, scAllCap * 4));
    CK(cudaMalloc(&dPA, (size_t)M * 8)); CK(cudaMalloc(&dPB, (size_t)M * 8)); CK(cudaMalloc(&dPC, (size_t)M * 8));
    CK(cudaMalloc(&dGoff, (size_t)M * 4)); CK(cudaMalloc(&dLcnt, (size_t)M * 4)); CK(cudaMalloc(&dLd, (size_t)M * 4)); CK(cudaMalloc(&dMoff, (size_t)M * 4));
    CK(cudaMalloc(&dScoff, (size_t)M * 8));
  }
  std::vector<cudaEvent_t> pe(4 * (size_t)M);
  if (profile) for (auto& e : pe) CK(cudaEventCreate(&e));
  double t_gather = 0, t_gemm = 0, t_argmax = 0, gtot = 0;
  const int nb = NQ / B;
  for (int rep = 0; rep < nb + 1; rep++) {  // rep 0 is a warm-up on batch 0
    int b            = rep == 0 ? 0 : rep - 1;
    const float* q   = dQ + (size_t)b * B * D;
    CK(cudaEventRecord(e0));
    // (1) list scores and each query's top-p lists
    gemm_tf32(cb, M, B, D, dS, q, dS1, Mpad);
    top_p_rows<<<B, 256, (size_t)M * 4>>>(dS1, M, Mpad, P, dLists);
    CK(cudaEventRecord(e1));
    // (2) bucket (list, query) pairs by list
    init_pairs<<<(NP + 255) / 256, 256>>>(dLists, NP, dKeys, dVals);
    CK(cudaMemset(dCounts, 0, (size_t)M * 4));
    count_keys<<<(NP + 255) / 256, 256>>>(dKeys, NP, dCounts);
    cub::DeviceRadixSort::SortPairs(dTmpSort, sortBytes, dKeys, dKeysSorted, dVals, dValsSorted, NP, 0, endBit);
    std::vector<int> counts(M);
    CK(cudaMemcpy(counts.data(), dCounts, (size_t)M * 4, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e2));
    CK(cudaEventRecord(forkEv));
    for (int i = 0; i < NS; i++) CK(cudaStreamWaitEvent(streams[i], forkEv, 0));
    // (3) per-list gather + GEMM + argmax
    int off = 0;
    for (int m = 0; m < M && !grouped; m++) {
      int g = counts[m], n = mcount[m];
      if (g > 0 && n > 0) {
        const int si = m % NS;
        cudaStream_t st = streams[si];
        float *qg = qgs[si], *sc = scs[si];
        if (profile) CK(cudaEventRecord(pe[4 * m + 0], st));
        gather_queries<<<g, 256, 0, st>>>(q, D, dValsSorted, off, P, qg);
        if (profile) CK(cudaEventRecord(pe[4 * m + 1], st));
        int npad = pad(n);
        // PADM=1 (diagnostic): also compute the padding columns, so the GEMM's m is aligned.
        if (swap) {
          // queries as GEMM-M (col-major C is [g x n], ld g); no padding of the list side
          gemm_tf32(handles[si], g, n, D, qg, dPerm + (size_t)moff[m] * D, sc, g);
          if (profile) CK(cudaEventRecord(pe[4 * m + 2], st));
          argmax_scatter_T<<<(g + 127) / 128, 128, 0, st>>>(sc, g, n, g, dValsSorted + off, dPermId + moff[m], dResScore, dResId);
        } else {
          gemm_tf32(handles[si], padm ? npad : n, g, D, dPerm + (size_t)moff[m] * D, qg, sc, npad);
          if (profile) CK(cudaEventRecord(pe[4 * m + 2], st));
          argmax_scatter<<<g, 256, 0, st>>>(sc, n, npad, dValsSorted + off, dPermId + moff[m], dResScore, dResId);
        }
        if (profile) CK(cudaEventRecord(pe[4 * m + 3], st));
        gtot += (double)g;
      }
      off += g;
    }
    if (profile && NS == 1 && rep > 0) {
      CK(cudaEventSynchronize(pe[4 * M - 1]));
      for (int m = 0; m < M; m++) {
        if (!(counts[m] > 0 && mcount[m] > 0)) continue;
        float x;
        CK(cudaEventElapsedTime(&x, pe[4 * m + 0], pe[4 * m + 1])); t_gather += x;
        CK(cudaEventElapsedTime(&x, pe[4 * m + 1], pe[4 * m + 2])); t_gemm += x;
        CK(cudaEventElapsedTime(&x, pe[4 * m + 2], pe[4 * m + 3])); t_argmax += x;
      }
    }
    if (grouped) {
      // one gather, one grouped-batched GEMM call (one group per list), one argmax
      std::vector<int> hm, hn, hk, hlda, hldb, hldc, hgs, hgoff(M, 0), hl(M, 0), hld(M, 0);
      std::vector<long long> hscoff(M, 0);
      std::vector<const float*> hA, hB;
      std::vector<float*> hC;
      long long so = 0;
      int o2       = 0;
      for (int m = 0; m < M; m++) {
        int g = counts[m], n = mcount[m];
        hgoff[m] = o2; hl[m] = n; hld[m] = (n + 7) / 8 * 8; hscoff[m] = so;
        if (g > 0 && n > 0) {
          hm.push_back(n); hn.push_back(g); hk.push_back(D); hlda.push_back(D); hldb.push_back(D); hldc.push_back(hld[m]);
          hgs.push_back(1);
          hA.push_back(dPerm + (size_t)moff[m] * D);
          hB.push_back(dQgAll + (size_t)o2 * D);
          hC.push_back(dScAll + so);
          so += (long long)g * hld[m];
        }
        o2 += g;
      }
      int G = (int)hm.size();
      if ((size_t)so > scAllCap) { fprintf(stderr, "score buffer too small\n"); return 1; }
      std::vector<cublasOperation_t> ta(G, CUBLAS_OP_T), tb(G, CUBLAS_OP_N);
      std::vector<float> al(G, 1.f), be(G, 0.f);
      CK(cudaMemcpy(dPA, hA.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dPB, hB.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dPC, hC.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dGoff, hgoff.data(), M * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dLcnt, hl.data(), M * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dLd, hld.data(), M * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dScoff, hscoff.data(), M * 8, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dMoff, moff.data(), M * 4, cudaMemcpyHostToDevice));
      cudaEvent_t g0, g1, g2;
      CK(cudaEventCreate(&g0)); CK(cudaEventCreate(&g1)); CK(cudaEventCreate(&g2));
      CK(cudaEventRecord(g0));
      gather_all<<<NP, 256>>>(q, D, dValsSorted, P, dQgAll);
      CK(cudaEventRecord(g1));
      CB(cublasGemmGroupedBatchedEx(cb, ta.data(), tb.data(), hm.data(), hn.data(), hk.data(), al.data(),
                                    (const void* const*)dPA, CUDA_R_32F, hlda.data(), (const void* const*)dPB, CUDA_R_32F,
                                    hldb.data(), be.data(), (void* const*)dPC, CUDA_R_32F, hldc.data(), G, hgs.data(),
                                    CUBLAS_COMPUTE_32F_FAST_TF32));
      CK(cudaEventRecord(g2));
      if (std::getenv("CHECK") && rep == 1) {
        // compare each group's grouped-GEMM scores with a plain per-list GEMM
        int gi = 0, bad = 0;
        for (int m = 0; m < M; m++) {
          int g = counts[m], n = mcount[m];
          if (!(g > 0 && n > 0)) continue;
          gemm_tf32(cb, n, g, D, dPerm + (size_t)moff[m] * D, dQgAll + (size_t)hgoff[m] * D, scs[0], n);
          std::vector<float> a((size_t)g * n), b((size_t)g * n);
          CK(cudaMemcpy(a.data(), scs[0], a.size() * 4, cudaMemcpyDeviceToHost));
          { std::vector<float> raw((size_t)g * hld[m]); CK(cudaMemcpy(raw.data(), dScAll + hscoff[m], raw.size() * 4, cudaMemcpyDeviceToHost));
            for (int r = 0; r < g; r++) for (int c = 0; c < n; c++) b[(size_t)r * n + c] = raw[(size_t)r * hld[m] + c]; }
          double mx = 0;
          for (size_t i = 0; i < a.size(); i++) mx = std::max(mx, (double)std::fabs(a[i] - b[i]));
          if (mx > 1e-2) { if (bad++ < 5) printf("group %d (list %d, g=%d n=%d) max diff %.4f\n", gi, m, g, n, mx); }
          gi++;
        }
        printf("CHECK: %d / %d groups differ\n", bad, gi);
      }
      argmax_all<<<NP, 256>>>(dScAll, dKeysSorted, dGoff, dLcnt, dLd, dScoff, dMoff, dValsSorted, dPermId, dResScore, dResId);
      if (profile) {
        CK(cudaEventSynchronize(g2)); float a, b;
        CK(cudaEventElapsedTime(&a, g0, g1)); CK(cudaEventElapsedTime(&b, g1, g2));
        if (rep > 0) { t_gather += a; t_gemm += b; }
      }
      CK(cudaEventDestroy(g0)); CK(cudaEventDestroy(g1)); CK(cudaEventDestroy(g2));
    }
    for (int i = 0; i < NS; i++) {
      CK(cudaEventRecord(joinEv[i], streams[i]));
      CK(cudaStreamWaitEvent(0, joinEv[i], 0));
    }
    CK(cudaEventRecord(e3));
    // (4) merge each query's p partial results
    merge_pairs<<<(B + 255) / 256, 256>>>(dResScore, dResId, P, B, dLabels + (size_t)b * B);
    CK(cudaEventRecord(e4));
    CK(cudaEventSynchronize(e4));
    if (rep > 0) {
      float a, bb, c, d, tot;
      CK(cudaEventElapsedTime(&a, e0, e1)); CK(cudaEventElapsedTime(&bb, e1, e2));
      CK(cudaEventElapsedTime(&c, e2, e3)); CK(cudaEventElapsedTime(&d, e3, e4));
      CK(cudaEventElapsedTime(&tot, e0, e4));
      t_stage1 += a; t_sort += bb; t_lists += c; t_total += tot;
      (void)d;
    }
  }
  printf("grouped:     %.2f ms/batch  = %.3f us/row   (stage1 %.2f, sort+count %.2f, per-list %.2f ms)\n",
         t_total / nb, t_total / nb * 1e3 / B, t_stage1 / nb, t_sort / nb, t_lists / nb);

  if (profile)
    printf("per-kernel (summed over lists, ms/batch): gather %.2f  gemm %.2f  argmax %.2f   (avg group %.0f queries/list)\n",
           t_gather / nb, t_gemm / nb, t_argmax / nb, gtot / (nb + 1) / M);
  // ---- Exhaustive TF32 GEMM + argmax baseline, in tiles. ----
  const int T = 4096;
  float *dScBase, *dBest;
  int *dArg, *dExact;
  CK(cudaMalloc(&dScBase, (size_t)T * K * 4));
  CK(cudaMalloc(&dBest, (size_t)T * 4));
  CK(cudaMalloc(&dArg, (size_t)T * 4));
  CK(cudaMalloc(&dExact, (size_t)NQ * 4));
  double t_exact = 0;
  for (int rep = 0; rep < nb + 1; rep++) {
    int b = rep == 0 ? 0 : rep - 1;
    CK(cudaEventRecord(e0));
    for (int t0 = 0; t0 < B; t0 += T) {
      const float* q = dQ + ((size_t)b * B + t0) * D;
      gemm_tf32(cb, K, T, D, dC, q, dScBase);
      argmax_rows<<<T, 256>>>(dScBase, K, dBest, dArg);
      CK(cudaMemcpy(dExact + (size_t)b * B + t0, dArg, (size_t)T * 4, cudaMemcpyDeviceToDevice));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (rep > 0) t_exact += ms;
  }
  printf("exhaustive:  %.2f ms/batch  = %.3f us/row   (TF32 GEMM over all %d centroids + argmax)\n",
         t_exact / nb, t_exact / nb * 1e3 / B, K);

  std::vector<int> g(NQ), e(NQ);
  CK(cudaMemcpy(g.data(), dLabels, (size_t)NQ * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(e.data(), dExact, (size_t)NQ * 4, cudaMemcpyDeviceToHost));
  size_t agree = 0;
  for (int i = 0; i < NQ; i++) agree += g[i] == e[i];
  printf("label agreement with the exhaustive TF32 result: %.5f  speedup %.2fx\n", (double)agree / NQ, t_exact / t_total);
  return 0;
}
