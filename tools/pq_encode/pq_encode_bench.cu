// Standalone check and benchmark for the PQ-encode paths of cuvsIvfPqTransformWithLabels,
// selected with LANCE_CUVS_PQ_ENCODE_BACKEND (baseline | simt | tensor | tensor3).
//
// Builds a small IVF-PQ model (training only) on synthetic clustered vectors through the cuVS C
// API, assigns coarse labels once, then runs TransformWithLabels with each backend on the same
// rows and reports
//   - time per row (residuals + encode; the labels are given, so no coarse assignment),
//   - agreement of the codes with `baseline` (bytes and rows),
//   - on the first --check-rows rows, each path's extra distortion against the exact argmin in
//     FP64: sum(d_chosen - d_best) / sum(d_best), and how many (row, subspace) codes are not the
//     exact argmin (counting exact distance ties as optimal).
//
// Build (from the lanceDB root):
//   nvcc -O2 -std=c++17 -I cuvs/install/include -I miniconda3/envs/cuvs-cu134/include \
//     lance-cuvs/tools/pq_encode/pq_encode_bench.cu -o pq_encode_bench \
//     -L cuvs/install/lib -lcuvs_c -Xlinker -rpath=$PWD/cuvs/install/lib \
//     -Xlinker -rpath=$PWD/miniconda3/envs/cuvs-cu134/lib
// Run it with the conda libssl/libcrypto preloaded, as `run_local` does.

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(x)                                                                    \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)
#define CHECK_CUVS(x)                                                                    \
  do {                                                                                   \
    cuvsError_t e_ = (x);                                                                \
    if (e_ != CUVS_SUCCESS) {                                                            \
      std::fprintf(stderr, "%s:%d cuVS error: %s\n", __FILE__, __LINE__, cuvsGetLastErrorText()); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

static __device__ __forceinline__ uint64_t mix(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

static __device__ __forceinline__ float approx_normal(uint64_t seed)
{
  const uint64_t h = mix(seed);
  float s          = 0.0f;
  for (int i = 0; i < 4; i++) {
    s += static_cast<float>((h >> (16 * i)) & 0xffff) * (1.0f / 65536.0f);
  }
  return (s - 2.0f) * 1.7320508f;  // mean 0, variance 1
}

// row i = base[i % n_modes] + noise * N(0, 1)
__global__ void gen_kernel(float* out, int64_t n_rows, int dim, int n_modes, float noise, uint64_t seed)
{
  const int64_t total = n_rows * dim;
  for (int64_t t = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; t < total;
       t += int64_t{gridDim.x} * blockDim.x) {
    const int64_t i = t / dim;
    const int c     = static_cast<int>(t % dim);
    const int mode  = static_cast<int>(mix(seed ^ (static_cast<uint64_t>(i) * 31)) % n_modes);
    out[t] = approx_normal(seed + 1 + static_cast<uint64_t>(mode) * dim + c) +
             noise * approx_normal(seed + 7919 + static_cast<uint64_t>(t));
  }
}

static DLManagedTensor tensor(void* ptr, uint8_t code, uint8_t bits, int ndim, int64_t* shape, int dev)
{
  DLManagedTensor t{};
  t.dl_tensor.data        = ptr;
  t.dl_tensor.device      = {dev >= 0 ? kDLCUDA : kDLCPU, dev >= 0 ? dev : 0};
  t.dl_tensor.ndim        = ndim;
  t.dl_tensor.dtype       = {code, bits, 1};
  t.dl_tensor.shape       = shape;
  t.dl_tensor.strides     = nullptr;
  t.dl_tensor.byte_offset = 0;
  return t;
}

static std::vector<std::string> split(const std::string& s)
{
  std::vector<std::string> out;
  size_t p = 0;
  while (p <= s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    out.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  return out;
}

int main(int argc, char** argv)
{
  int64_t n_train = 200000, n_rows = 262144, check_rows = 2048;
  int dim = 2048, pq_dim = 256, n_lists = 256, reps = 5;
  std::string backends = "baseline,simt,tensor,tensor3";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const char* v       = argv[i + 1];
    if (a == "--n-train") n_train = std::atoll(v);
    else if (a == "--n-rows") n_rows = std::atoll(v);
    else if (a == "--check-rows") check_rows = std::atoll(v);
    else if (a == "--dim") dim = std::atoi(v);
    else if (a == "--pq-dim") pq_dim = std::atoi(v);
    else if (a == "--n-lists") n_lists = std::atoi(v);
    else if (a == "--reps") reps = std::atoi(v);
    else if (a == "--backends") backends = v;
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  const int pq_len = dim / pq_dim;
  check_rows       = std::min(check_rows, n_rows);
  std::printf("n_train=%lld n_rows=%lld dim=%d pq_dim=%d pq_len=%d n_lists=%d\n",
              (long long)n_train, (long long)n_rows, dim, pq_dim, pq_len, n_lists);

  cuvsResources_t res;
  CHECK_CUVS(cuvsResourcesCreate(&res));

  float *d_train, *d_rows;
  CHECK_CUDA(cudaMalloc(&d_train, size_t(n_train) * dim * 4));
  CHECK_CUDA(cudaMalloc(&d_rows, size_t(n_rows) * dim * 4));
  gen_kernel<<<4096, 256>>>(d_train, n_train, dim, n_lists * 4, 0.6f, 1234);
  gen_kernel<<<4096, 256>>>(d_rows, n_rows, dim, n_lists * 4, 0.6f, 1234);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Train (no data added): a metric-agnostic residual codebook, so use plain L2 and the host
  // reconstruction below is exactly x - center[label].
  cuvsIvfPqIndexParams_t params;
  CHECK_CUVS(cuvsIvfPqIndexParamsCreate(&params));
  params->metric                   = L2Expanded;
  params->add_data_on_build        = false;
  params->n_lists                  = n_lists;
  params->pq_bits                  = 8;
  params->pq_dim                   = pq_dim;
  params->codebook_kind            = CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE;
  params->kmeans_n_iters           = 10;
  params->kmeans_trainset_fraction = 1.0;
  cuvsIvfPqIndex_t index;
  CHECK_CUVS(cuvsIvfPqIndexCreate(&index));
  int64_t train_shape[2] = {n_train, dim};
  DLManagedTensor train_t = tensor(d_train, kDLFloat, 32, 2, train_shape, 0);
  std::printf("building the model ...\n");
  CHECK_CUVS(cuvsIvfPqBuild(res, params, &train_t, index));
  CHECK_CUVS(cuvsStreamSync(res));

  // Labels once, with the default path (they only choose the residual's centre).
  uint32_t* d_labels;
  uint8_t* d_codes;
  CHECK_CUDA(cudaMalloc(&d_labels, size_t(n_rows) * 4));
  CHECK_CUDA(cudaMalloc(&d_codes, size_t(n_rows) * pq_dim));
  int64_t rows_shape[2]   = {n_rows, dim};
  int64_t labels_shape[1] = {n_rows};
  int64_t codes_shape[2]  = {n_rows, pq_dim};
  DLManagedTensor rows_t   = tensor(d_rows, kDLFloat, 32, 2, rows_shape, 0);
  DLManagedTensor labels_t = tensor(d_labels, kDLUInt, 32, 1, labels_shape, 0);
  DLManagedTensor codes_t  = tensor(d_codes, kDLUInt, 8, 2, codes_shape, 0);
  setenv("LANCE_CUVS_PQ_ENCODE_BACKEND", "baseline", 1);
  CHECK_CUVS(cuvsIvfPqTransform(res, index, &rows_t, &labels_t, &codes_t));
  CHECK_CUVS(cuvsStreamSync(res));

  // Host copies for the exact check: the first rows, the centres, the codebook.
  std::vector<float> h_rows(size_t(check_rows) * dim);
  std::vector<uint32_t> h_labels(check_rows);
  CHECK_CUDA(cudaMemcpy(h_rows.data(), d_rows, h_rows.size() * 4, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_labels.data(), d_labels, check_rows * 4, cudaMemcpyDeviceToHost));
  DLManagedTensor centers_t{}, pqc_t{};
  CHECK_CUVS(cuvsIvfPqIndexGetCenters(index, &centers_t));
  CHECK_CUVS(cuvsIvfPqIndexGetPqCenters(index, &pqc_t));
  // The centres may be a strided view of the padded [n_lists, dim_ext] matrix.
  const int64_t center_cols   = centers_t.dl_tensor.shape[1];
  const int64_t center_stride = centers_t.dl_tensor.strides ? centers_t.dl_tensor.strides[0] : center_cols;
  std::printf("centers tensor: shape [%lld, %lld] row stride %lld\n",
              (long long)centers_t.dl_tensor.shape[0], (long long)center_cols, (long long)center_stride);
  std::vector<float> h_centers(size_t(n_lists) * center_cols), h_pqc(size_t(pq_dim) * pq_len * 256);
  CHECK_CUDA(cudaMemcpy2D(h_centers.data(), center_cols * 4, centers_t.dl_tensor.data, center_stride * 4,
                          center_cols * 4, n_lists, cudaMemcpyDeviceToHost));
  {
    // The residual is rotation * (x - centre); the host check below assumes the identity.
    DLManagedTensor rot_t{};
    CHECK_CUVS(cuvsIvfPqIndexGetRotationMatrix(index, &rot_t));
    const int64_t n = rot_t.dl_tensor.shape[0];
    const int64_t m = rot_t.dl_tensor.shape[1];
    const int64_t rs = rot_t.dl_tensor.strides ? rot_t.dl_tensor.strides[0] : m;
    std::vector<float> rot(size_t(n) * m);
    CHECK_CUDA(cudaMemcpy2D(rot.data(), m * 4, rot_t.dl_tensor.data, rs * 4, m * 4, n, cudaMemcpyDeviceToHost));
    bool ident = n == m;
    for (int64_t i = 0; ident && i < n; i++)
      for (int64_t k = 0; k < m; k++)
        if (rot[size_t(i) * m + k] != (i == k ? 1.0f : 0.0f)) { ident = false; break; }
    std::printf("rotation matrix [%lld, %lld]: %s\n", (long long)n, (long long)m,
                ident ? "identity" : "NOT the identity (host check is invalid)");
  }
  CHECK_CUDA(cudaMemcpy(h_pqc.data(), pqc_t.dl_tensor.data, h_pqc.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<float> h_res(size_t(check_rows) * dim);
  for (int64_t r = 0; r < check_rows; r++) {
    for (int c = 0; c < dim; c++) {
      h_res[size_t(r) * dim + c] = h_rows[size_t(r) * dim + c] - h_centers[size_t(h_labels[r]) * center_cols + c];
    }
  }
  // exact best distance per (row, subspace)
  std::vector<double> best(size_t(check_rows) * pq_dim);
  std::vector<double> dists(256);
  for (int64_t r = 0; r < check_rows; r++) {
    for (int j = 0; j < pq_dim; j++) {
      double b = 1e300;
      for (int l = 0; l < 256; l++) {
        double d = 0;
        for (int k = 0; k < pq_len; k++) {
          const double t = double(h_res[size_t(r) * dim + j * pq_len + k]) - double(h_pqc[(size_t(j) * pq_len + k) * 256 + l]);
          d += t * t;
        }
        b = std::min(b, d);
      }
      best[size_t(r) * pq_dim + j] = b;
    }
  }

  std::vector<uint8_t> base_codes(size_t(n_rows) * pq_dim), codes(size_t(n_rows) * pq_dim);
  std::printf("\n%-9s %10s %9s %12s %12s %14s %14s\n", "backend", "ms/call", "us/row", "bytes!=base",
              "rows!=base", "not-exact-code", "extra distort");
  for (const auto& be : split(backends)) {
    setenv("LANCE_CUVS_PQ_ENCODE_BACKEND", be.c_str(), 1);
    // one warm-up call, then timed calls
    CHECK_CUVS(cuvsIvfPqTransformWithLabels(res, index, &rows_t, &labels_t, &codes_t));
    CHECK_CUVS(cuvsStreamSync(res));
    double total_ms = 0;
    for (int it = 0; it < reps; it++) {
      const auto t0 = std::chrono::steady_clock::now();
      CHECK_CUVS(cuvsIvfPqTransformWithLabels(res, index, &rows_t, &labels_t, &codes_t));
      CHECK_CUVS(cuvsStreamSync(res));
      total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    CHECK_CUDA(cudaMemcpy(codes.data(), d_codes, codes.size(), cudaMemcpyDeviceToHost));
    if (be == "baseline") base_codes = codes;
    int64_t bad_bytes = 0, bad_rows = 0;
    for (int64_t r = 0; r < n_rows; r++) {
      bool bad = false;
      for (int j = 0; j < pq_dim; j++) {
        if (codes[size_t(r) * pq_dim + j] != base_codes[size_t(r) * pq_dim + j]) { bad_bytes++; bad = true; }
      }
      bad_rows += bad;
    }
    int64_t not_exact = 0;
    double extra = 0, sum_best = 0;
    for (int64_t r = 0; r < check_rows; r++) {
      for (int j = 0; j < pq_dim; j++) {
        const int l = codes[size_t(r) * pq_dim + j];
        double d = 0;
        for (int k = 0; k < pq_len; k++) {
          const double t = double(h_res[size_t(r) * dim + j * pq_len + k]) - double(h_pqc[(size_t(j) * pq_len + k) * 256 + l]);
          d += t * t;
        }
        const double b = best[size_t(r) * pq_dim + j];
        if (d > b) not_exact++;
        extra += d - b;
        sum_best += b;
      }
    }
    const double ms = total_ms / reps;
    std::printf("%-9s %10.3f %9.4f %12lld %12lld %14lld %14.3e\n", be.c_str(), ms,
                ms * 1e3 / double(n_rows), (long long)bad_bytes, (long long)bad_rows,
                (long long)not_exact, extra / sum_best);
  }
  std::printf("(%lld of %lld codes checked for the exact argmin: %lld rows x %d subspaces)\n",
              (long long)(check_rows * pq_dim), (long long)(n_rows * pq_dim), (long long)check_rows, pq_dim);
  return 0;
}
