/***************************************************************************************************
 * NVFP4 reduction-order probe.
 *
 * Empirically compares native cuBLASLt NVFP4 GEMM against software reduction-order hypotheses.
 * This is intended to reverse-engineer the observable Blackwell TCGEN05/cuBLASLt accumulation
 * behavior as far as the public API allows.
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../../common.cuh"

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                     \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "       \
                << cudaGetErrorString(err) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

#define CHECK_CUBLAS(call)                                                        \
  do {                                                                            \
    cublasStatus_t status = call;                                                 \
    if (status != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLASLt error in " << __FILE__ << ":" << __LINE__ << ": "   \
                << status << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

static bool try_cublas(cublasStatus_t status, const char *expr, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLASLt setup failed at line " << line << " for " << expr
              << ": " << status << std::endl;
    return false;
  }
  return true;
}

#define TRY_CUBLAS(call)                                                          \
  do {                                                                            \
    if (!try_cublas(call, #call, __LINE__)) {                                     \
      destroy();                                                                  \
      return false;                                                               \
    }                                                                             \
  } while (0)

template <typename T>
struct CudaType;

template <>
struct CudaType<__nv_bfloat16> {
  static constexpr cudaDataType_t value = CUDA_R_16BF;
  static constexpr const char *name = "bf16";
};

template <>
struct CudaType<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
  static constexpr const char *name = "fp32";
};

struct CublasLtNvfp4Gemm {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic = {};
  void *workspace = nullptr;
  size_t workspace_size = 128 * 1024 * 1024;

  template <typename T>
  T get_config_attr(cublasLtMatmulAlgoConfigAttributes_t attr, const char *name) const {
    T value = 0;
    size_t size_written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
        &heuristic.algo, attr, &value, sizeof(value), &size_written);
    if (status == CUBLAS_STATUS_SUCCESS) {
      std::cout << "  " << name << "=" << static_cast<uint64_t>(value) << std::endl;
    }
    return value;
  }

  template <typename T>
  T get_cap_attr(cublasLtMatmulAlgoCapAttributes_t attr, const char *name) const {
    T value = 0;
    size_t size_written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoCapGetAttribute(
        &heuristic.algo, attr, &value, sizeof(value), &size_written);
    if (status == CUBLAS_STATUS_SUCCESS) {
      std::cout << "  " << name << "=0x" << std::hex << static_cast<uint64_t>(value)
                << std::dec << std::endl;
    }
    return value;
  }

  template <typename T>
  T get_config_attr_quiet(cublasLtMatmulAlgoConfigAttributes_t attr) const {
    T value = 0;
    size_t size_written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(
        &heuristic.algo, attr, &value, sizeof(value), &size_written);
    return value;
  }

  template <typename T>
  T get_cap_attr_quiet(cublasLtMatmulAlgoCapAttributes_t attr) const {
    T value = 0;
    size_t size_written = 0;
    cublasLtMatmulAlgoCapGetAttribute(
        &heuristic.algo, attr, &value, sizeof(value), &size_written);
    return value;
  }

  void print_algo_metadata() const {
    std::cout << "cuBLASLt heuristic metadata:" << std::endl;
    get_config_attr<int32_t>(CUBLASLT_ALGO_CONFIG_ID, "algo_id");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_TILE_ID, "tile_id");
    get_config_attr<int32_t>(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, "splitk_num");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, "reduction_scheme");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, "custom_option");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_STAGES_ID, "stages_id");
    get_config_attr<uint16_t>(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, "cluster_shape_id");
    get_cap_attr<uint64_t>(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, "cap_numerical_impl_flags");
  }

  bool init(int m, int n, int k, cudaDataType_t output_type) {
    TRY_CUBLAS(cublasLtCreate(&handle));
    TRY_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_4F_E2M1, k, n, k));
    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_4F_E2M1, k, m, k));
    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layout_c, output_type, n, m, n));
    TRY_CUBLAS(cublasLtMatrixLayoutCreate(&layout_d, output_type, n, m, n));

    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    TRY_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    TRY_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    void *dummy_scale_ptr = workspace;
    TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &dummy_scale_ptr, sizeof(dummy_scale_ptr)));
    TRY_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &dummy_scale_ptr, sizeof(dummy_scale_ptr)));

    int returned_results = 0;
    TRY_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, layout_a, layout_b, layout_c, layout_d,
        preference, 1, &heuristic, &returned_results));
    if (returned_results == 0) {
      std::cerr << "No cuBLASLt NVFP4 algorithm found." << std::endl;
      destroy();
      return false;
    }
    print_algo_metadata();
    return true;
  }

  template <typename OutputT>
  void run(
      const __nv_fp4x2_e2m1 *a,
      const __nv_fp4x2_e2m1 *b,
      const __nv_fp8_e4m3 *a_scale,
      const __nv_fp8_e4m3 *b_scale,
      float alpha,
      OutputT *d,
      cudaStream_t stream) {
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &b_scale, sizeof(b_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &a_scale, sizeof(a_scale)));

    float beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(
        handle, matmul_desc, &alpha,
        b, layout_a,
        a, layout_b,
        &beta,
        d, layout_c,
        d, layout_d,
        &heuristic.algo, workspace, workspace_size, stream));
  }

  void destroy() {
    if (workspace != nullptr) CHECK_CUDA(cudaFree(workspace));
    if (preference != nullptr) CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    if (layout_a != nullptr) CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layout_a));
    if (layout_b != nullptr) CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layout_b));
    if (layout_c != nullptr) CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layout_c));
    if (layout_d != nullptr) CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layout_d));
    if (matmul_desc != nullptr) CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmul_desc));
    if (handle != nullptr) CHECK_CUBLAS(cublasLtDestroy(handle));
    workspace = nullptr;
    preference = nullptr;
    layout_a = nullptr;
    layout_b = nullptr;
    layout_c = nullptr;
    layout_d = nullptr;
    matmul_desc = nullptr;
    handle = nullptr;
  }

  void write_algo_metadata_json(std::ostream &out) const {
    out << "\"algo_id\":" << get_config_attr_quiet<int32_t>(CUBLASLT_ALGO_CONFIG_ID)
        << ",\"tile_id\":" << get_config_attr_quiet<uint32_t>(CUBLASLT_ALGO_CONFIG_TILE_ID)
        << ",\"splitk_num\":" << get_config_attr_quiet<int32_t>(CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
        << ",\"reduction_scheme\":"
        << get_config_attr_quiet<uint32_t>(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
        << ",\"custom_option\":"
        << get_config_attr_quiet<uint32_t>(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)
        << ",\"stages_id\":" << get_config_attr_quiet<uint32_t>(CUBLASLT_ALGO_CONFIG_STAGES_ID)
        << ",\"cluster_shape_id\":"
        << get_config_attr_quiet<uint16_t>(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)
        << ",\"cap_numerical_impl_flags\":"
        << get_cap_attr_quiet<uint64_t>(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS);
  }
};

template <typename OutputT>
__device__ inline OutputT from_float(float x) {
  if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
    return __float2bfloat16_rn(x);
  } else {
    return x;
  }
}

__device__ inline float fp4_term(
    const __nv_fp4x2_e2m1 *a_packed,
    const __nv_fp4x2_e2m1 *b_packed,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    int row,
    int col,
    int k_idx,
    int k) {
  constexpr int block_size = 16;
  int pair_idx = k_idx / 2;
  float2 a_vals = static_cast<float2>(a_packed[row * (k / 2) + pair_idx]);
  float2 b_vals = static_cast<float2>(b_packed[col * (k / 2) + pair_idx]);
  float a = (k_idx & 1) == 0 ? a_vals.x : a_vals.y;
  float b = (k_idx & 1) == 0 ? b_vals.x : b_vals.y;
  int k_block = k_idx / block_size;
  int k_blocks = k / block_size;
  float a_s = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
      a_scale[scale_swizzle_idx(row, k_block, k_blocks)]);
  float b_s = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
      b_scale[scale_swizzle_idx(col, k_block, k_blocks)]);
  return __fmul_rn(__fmul_rn(a, a_s), __fmul_rn(b, b_s));
}

template <typename OutputT, bool ReverseK>
__global__ void reference_seq_kernel(
    OutputT *d,
    const __nv_fp4x2_e2m1 *a_packed,
    const __nv_fp4x2_e2m1 *b_packed,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int step = 0; step < k; ++step) {
    int k_idx = ReverseK ? (k - 1 - step) : step;
    acc = __fadd_rn(acc, fp4_term(a_packed, b_packed, a_scale, b_scale, row, col, k_idx, k));
  }
  d[row * n + col] = from_float<OutputT>(__fmul_rn(acc, alpha));
}

template <typename OutputT, int GroupSize, bool InnerReverse, bool OuterReverse>
__global__ void reference_grouped_kernel(
    OutputT *d,
    const __nv_fp4x2_e2m1 *a_packed,
    const __nv_fp4x2_e2m1 *b_packed,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  int groups = k / GroupSize;
  for (int outer_step = 0; outer_step < groups; ++outer_step) {
    int group = OuterReverse ? (groups - 1 - outer_step) : outer_step;
    int base = group * GroupSize;
    float partial = 0.0f;
    for (int inner_step = 0; inner_step < GroupSize; ++inner_step) {
      int inner = InnerReverse ? (GroupSize - 1 - inner_step) : inner_step;
      partial = __fadd_rn(
          partial, fp4_term(a_packed, b_packed, a_scale, b_scale, row, col, base + inner, k));
    }
    acc = __fadd_rn(acc, partial);
  }
  d[row * n + col] = from_float<OutputT>(__fmul_rn(acc, alpha));
}

template <int Count>
__device__ inline float pairwise_sum(float (&values)[Count]) {
  int active = Count;
  while (active > 1) {
    int out = 0;
    for (int i = 0; i + 1 < active; i += 2) {
      values[out++] = __fadd_rn(values[i], values[i + 1]);
    }
    if ((active & 1) != 0) {
      values[out++] = values[active - 1];
    }
    active = out;
  }
  return values[0];
}

template <typename OutputT, int GroupSize, bool OuterReverse>
__global__ void reference_grouped_pairwise_inner_kernel(
    OutputT *d,
    const __nv_fp4x2_e2m1 *a_packed,
    const __nv_fp4x2_e2m1 *b_packed,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  int groups = k / GroupSize;
  for (int outer_step = 0; outer_step < groups; ++outer_step) {
    int group = OuterReverse ? (groups - 1 - outer_step) : outer_step;
    int base = group * GroupSize;
    float terms[GroupSize];
    #pragma unroll
    for (int inner = 0; inner < GroupSize; ++inner) {
      terms[inner] = fp4_term(a_packed, b_packed, a_scale, b_scale, row, col, base + inner, k);
    }
    acc = __fadd_rn(acc, pairwise_sum(terms));
  }
  d[row * n + col] = from_float<OutputT>(__fmul_rn(acc, alpha));
}

template <typename T>
static float host_to_float(T value) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return kittens::base_types::convertor<float, __nv_bfloat16>::convert(value);
  } else {
    return value;
  }
}

struct DiffStats {
  double mean = 0.0;
  double max = 0.0;
  size_t nonzero = 0;
  size_t first = 0;
  float first_native = 0.0f;
  float first_ref = 0.0f;
};

template <typename OutputT>
static DiffStats compare_outputs(const OutputT *native, const OutputT *reference, int m, int n) {
  size_t count = static_cast<size_t>(m) * n;
  std::vector<OutputT> h_native(count);
  std::vector<OutputT> h_reference(count);
  CHECK_CUDA(cudaMemcpy(h_native.data(), native, count * sizeof(OutputT), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_reference.data(), reference, count * sizeof(OutputT), cudaMemcpyDeviceToHost));

  DiffStats stats;
  bool saw_first = false;
  for (size_t i = 0; i < count; ++i) {
    float a = host_to_float(h_native[i]);
    float b = host_to_float(h_reference[i]);
    double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
    stats.mean += diff;
    stats.max = std::max(stats.max, diff);
    if (a != b) {
      ++stats.nonzero;
      if (!saw_first) {
        saw_first = true;
        stats.first = i;
        stats.first_native = a;
        stats.first_ref = b;
      }
    }
  }
  stats.mean /= static_cast<double>(count);
  return stats;
}

template <typename OutputT>
static void print_stats(const std::string &name, const OutputT *native, const OutputT *reference, int m, int n) {
  DiffStats stats = compare_outputs(native, reference, m, n);
  int row = static_cast<int>(stats.first / n);
  int col = static_cast<int>(stats.first % n);
  std::cout << std::left << std::setw(24) << name
            << " mean=" << std::scientific << std::setprecision(8) << stats.mean
            << " max=" << stats.max
            << " nz=" << std::dec << stats.nonzero;
  if (stats.nonzero != 0) {
    std::cout << " first=(" << row << "," << col << ")"
              << " native=" << std::setprecision(10) << stats.first_native
              << " ref=" << stats.first_ref;
  }
  std::cout << std::endl;
}

static int host_scale_swizzle_idx(int row, int k_block, int K_blocks) {
  int M_block = row / 128;
  int K_block_groups = K_blocks / 4;
  int K_block_group = k_block / 4;
  int row_in_32 = row % 32;
  int tile_in_block = (row / 32) % 4;
  int kb_in_block = k_block % 4;

  int block_base = (M_block * K_block_groups + K_block_group) * 512;
  int local_idx = row_in_32 * 16 + tile_in_block * 4 + kb_in_block;
  return block_base + local_idx;
}

static float host_fp4_e2m1(uint8_t nibble) {
  static constexpr float lut[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return lut[nibble & 0x0f];
}

static float host_fp4_value(const std::vector<uint8_t> &packed, int row, int k_idx, int k) {
  uint8_t byte = packed[static_cast<size_t>(row) * (k / 2) + (k_idx / 2)];
  uint8_t nibble = (k_idx & 1) == 0 ? (byte & 0x0f) : (byte >> 4);
  return host_fp4_e2m1(nibble);
}

static void write_float_array(std::ostream &out, const std::vector<float> &values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << std::setprecision(9) << values[i];
  }
  out << "]";
}

static std::vector<float> host_fp4_row(const std::vector<uint8_t> &packed, int row, int k) {
  std::vector<float> values(k);
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    values[k_idx] = host_fp4_value(packed, row, k_idx, k);
  }
  return values;
}

static std::vector<float> host_scale_row(
    const std::vector<__nv_fp8_e4m3> &scales, int row, int k) {
  int k_blocks = k / 16;
  std::vector<float> values(k_blocks);
  for (int k_block = 0; k_block < k_blocks; ++k_block) {
    values[k_block] = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
        scales[host_scale_swizzle_idx(row, k_block, k_blocks)]);
  }
  return values;
}

struct MismatchRecord {
  size_t flat_idx = 0;
  int row = 0;
  int col = 0;
  float native = 0.0f;
  float seq_fwd = 0.0f;
  double diff = 0.0;
};

template <typename OutputT>
static bool emit_corpus_for_output(
    std::ostream &out,
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    float scale_min,
    float scale_max,
    int m,
    int n,
    int k,
    int corpus_limit) {
  std::cout << "\nCorpus output type: " << CudaType<OutputT>::name << std::endl;
  CublasLtNvfp4Gemm gemm;
  if (!gemm.init(m, n, k, CudaType<OutputT>::value)) {
    std::cout << "Skipping corpus for " << CudaType<OutputT>::name << ": unsupported." << std::endl;
    return false;
  }

  size_t d_size = static_cast<size_t>(m) * n;
  OutputT *native = nullptr;
  OutputT *reference = nullptr;
  CHECK_CUDA(cudaMalloc(&native, d_size * sizeof(OutputT)));
  CHECK_CUDA(cudaMalloc(&reference, d_size * sizeof(OutputT)));

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  gemm.run(a, b, a_scale, b_scale, alpha, native, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  reference_seq_kernel<OutputT, false>
      <<<grid, block>>>(reference, a, b, a_scale, b_scale, alpha, m, n, k);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<OutputT> h_native(d_size);
  std::vector<OutputT> h_reference(d_size);
  CHECK_CUDA(cudaMemcpy(h_native.data(), native, d_size * sizeof(OutputT), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_reference.data(), reference, d_size * sizeof(OutputT), cudaMemcpyDeviceToHost));

  std::vector<MismatchRecord> mismatches;
  mismatches.reserve(std::min<size_t>(d_size, static_cast<size_t>(corpus_limit * 8)));
  for (size_t i = 0; i < d_size; ++i) {
    float native_value = host_to_float(h_native[i]);
    float seq_value = host_to_float(h_reference[i]);
    double diff = std::abs(static_cast<double>(native_value) - static_cast<double>(seq_value));
    if (native_value != seq_value) {
      MismatchRecord record;
      record.flat_idx = i;
      record.row = static_cast<int>(i / n);
      record.col = static_cast<int>(i % n);
      record.native = native_value;
      record.seq_fwd = seq_value;
      record.diff = diff;
      mismatches.push_back(record);
    }
  }

  std::sort(mismatches.begin(), mismatches.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.diff == rhs.diff) {
      return lhs.flat_idx < rhs.flat_idx;
    }
    return lhs.diff > rhs.diff;
  });
  if (static_cast<int>(mismatches.size()) > corpus_limit) {
    mismatches.resize(corpus_limit);
  }

  out << "{\"kind\":\"meta\",\"output_type\":\"" << CudaType<OutputT>::name << "\""
      << ",\"m\":" << m
      << ",\"n\":" << n
      << ",\"k\":" << k
      << ",\"alpha\":" << std::setprecision(9) << alpha
      << ",\"scale_min\":" << scale_min
      << ",\"scale_max\":" << scale_max
      << ",\"mismatch_baseline\":\"seq_fwd\""
      << ",\"mismatch_count\":" << mismatches.size()
      << ",\"cublaslt\":{";
  gemm.write_algo_metadata_json(out);
  out << "}}\n";

  size_t fp4_a_size = static_cast<size_t>(m) * k / 2;
  size_t fp4_b_size = static_cast<size_t>(n) * k / 2;
  size_t scale_a_size = static_cast<size_t>(m) * (k / 16);
  size_t scale_b_size = static_cast<size_t>(n) * (k / 16);

  std::vector<uint8_t> h_a(fp4_a_size * sizeof(__nv_fp4x2_e2m1));
  std::vector<uint8_t> h_b(fp4_b_size * sizeof(__nv_fp4x2_e2m1));
  std::vector<__nv_fp8_e4m3> h_a_scale(scale_a_size);
  std::vector<__nv_fp8_e4m3> h_b_scale(scale_b_size);
  CHECK_CUDA(cudaMemcpy(h_a.data(), a, h_a.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_b.data(), b, h_b.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_a_scale.data(), a_scale, scale_a_size * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_b_scale.data(), b_scale, scale_b_size * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));

  for (const auto &record : mismatches) {
    std::vector<float> a_values = host_fp4_row(h_a, record.row, k);
    std::vector<float> b_values = host_fp4_row(h_b, record.col, k);
    std::vector<float> a_scales = host_scale_row(h_a_scale, record.row, k);
    std::vector<float> b_scales = host_scale_row(h_b_scale, record.col, k);
    out << "{\"kind\":\"record\",\"output_type\":\"" << CudaType<OutputT>::name << "\""
        << ",\"m\":" << m
        << ",\"n\":" << n
        << ",\"k\":" << k
        << ",\"row\":" << record.row
        << ",\"col\":" << record.col
        << ",\"flat_idx\":" << record.flat_idx
        << ",\"native\":" << std::setprecision(10) << record.native
        << ",\"seq_fwd\":" << std::setprecision(10) << record.seq_fwd
        << ",\"abs_diff\":" << std::setprecision(10) << record.diff
        << ",\"alpha\":" << std::setprecision(9) << alpha
        << ",\"scale_block_size\":16"
        << ",\"a_fp4\":";
    write_float_array(out, a_values);
    out << ",\"b_fp4\":";
    write_float_array(out, b_values);
    out << ",\"a_scales\":";
    write_float_array(out, a_scales);
    out << ",\"b_scales\":";
    write_float_array(out, b_scales);
    out << "}\n";
  }

  std::cout << "corpus records emitted for " << CudaType<OutputT>::name
            << ": " << mismatches.size() << std::endl;

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(native));
  CHECK_CUDA(cudaFree(reference));
  gemm.destroy();
  return true;
}

template <typename OutputT, bool ReverseK>
static void run_seq_candidate(
    const std::string &name,
    OutputT *native,
    OutputT *reference,
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  reference_seq_kernel<OutputT, ReverseK>
      <<<grid, block>>>(reference, a, b, a_scale, b_scale, alpha, m, n, k);
  CHECK_CUDA(cudaDeviceSynchronize());
  print_stats(name, native, reference, m, n);
}

template <typename OutputT, int GroupSize, bool InnerReverse, bool OuterReverse>
static void run_group_candidate(
    const std::string &name,
    OutputT *native,
    OutputT *reference,
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  reference_grouped_kernel<OutputT, GroupSize, InnerReverse, OuterReverse>
      <<<grid, block>>>(reference, a, b, a_scale, b_scale, alpha, m, n, k);
  CHECK_CUDA(cudaDeviceSynchronize());
  print_stats(name, native, reference, m, n);
}

template <typename OutputT, int GroupSize, bool OuterReverse>
static void run_pairwise_inner_candidate(
    const std::string &name,
    OutputT *native,
    OutputT *reference,
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  reference_grouped_pairwise_inner_kernel<OutputT, GroupSize, OuterReverse>
      <<<grid, block>>>(reference, a, b, a_scale, b_scale, alpha, m, n, k);
  CHECK_CUDA(cudaDeviceSynchronize());
  print_stats(name, native, reference, m, n);
}

template <typename OutputT>
static void run_candidates(
    OutputT *native,
    OutputT *reference,
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  run_seq_candidate<OutputT, false>("seq_fwd", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_seq_candidate<OutputT, true>("seq_rev", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 16, false, false>("g16_if_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 16, false, true>("g16_if_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 32, false, false>("g32_if_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 32, false, true>("g32_if_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 64, false, false>("g64_if_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 64, false, true>("g64_if_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 64, true, false>("g64_ir_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 64, true, true>("g64_ir_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 16, false>("g16_pair_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 16, true>("g16_pair_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 32, false>("g32_pair_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 32, true>("g32_pair_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 64, false>("g64_pair_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 64, true>("g64_pair_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 128, false>("g128_pair_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_pairwise_inner_candidate<OutputT, 128, true>("g128_pair_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 128, false, false>("g128_if_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 128, false, true>("g128_if_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 256, false, false>("g256_if_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 256, false, true>("g256_if_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 512, false, false>("g512_if_of", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
  run_group_candidate<OutputT, 512, false, true>("g512_if_or", native, reference, a, b, a_scale, b_scale, alpha, m, n, k);
}

template <typename OutputT>
static bool run_probe(
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m,
    int n,
    int k) {
  std::cout << "\nOutput type: " << CudaType<OutputT>::name << std::endl;
  CublasLtNvfp4Gemm gemm;
  if (!gemm.init(m, n, k, CudaType<OutputT>::value)) {
    std::cout << "Skipping " << CudaType<OutputT>::name << " output: unsupported by selected cuBLASLt path." << std::endl;
    return false;
  }

  size_t d_size = static_cast<size_t>(m) * n;
  OutputT *native = nullptr;
  OutputT *reference = nullptr;
  CHECK_CUDA(cudaMalloc(&native, d_size * sizeof(OutputT)));
  CHECK_CUDA(cudaMalloc(&reference, d_size * sizeof(OutputT)));

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  gemm.run(a, b, a_scale, b_scale, alpha, native, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  run_candidates(native, reference, a, b, a_scale, b_scale, alpha, m, n, k);

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(native));
  CHECK_CUDA(cudaFree(reference));
  gemm.destroy();
  return true;
}

template <typename OutputT>
static bool run_metadata_probe(int m, int n, int k) {
  std::cout << "\nOutput type: " << CudaType<OutputT>::name << std::endl;
  CublasLtNvfp4Gemm gemm;
  if (!gemm.init(m, n, k, CudaType<OutputT>::value)) {
    std::cout << "Skipping " << CudaType<OutputT>::name << " output: unsupported by selected cuBLASLt path." << std::endl;
    return false;
  }
  gemm.destroy();
  return true;
}

int main(int argc, char **argv) {
  int m = argc > 1 ? std::atoi(argv[1]) : 128;
  int n = argc > 2 ? std::atoi(argv[2]) : 256;
  int k = argc > 3 ? std::atoi(argv[3]) : 6144;
  float alpha = argc > 4 ? std::atof(argv[4]) : 1.0f;
  float scale_min = argc > 5 ? std::atof(argv[5]) : 0.25f;
  float scale_max = argc > 6 ? std::atof(argv[6]) : 2.0f;
  bool metadata_only = false;
  std::string corpus_path;
  std::string corpus_output = "both";
  int corpus_limit = 16;

  for (int i = 7; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--metadata-only") {
      metadata_only = true;
    } else if (arg == "--emit-corpus") {
      if (i + 1 >= argc) {
        std::cerr << "--emit-corpus requires a path" << std::endl;
        return EXIT_FAILURE;
      }
      corpus_path = argv[++i];
    } else if (arg == "--corpus-output") {
      if (i + 1 >= argc) {
        std::cerr << "--corpus-output requires fp32, bf16, or both" << std::endl;
        return EXIT_FAILURE;
      }
      corpus_output = argv[++i];
    } else if (arg == "--corpus-limit") {
      if (i + 1 >= argc) {
        std::cerr << "--corpus-limit requires an integer" << std::endl;
        return EXIT_FAILURE;
      }
      corpus_limit = std::atoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (corpus_limit <= 0) {
    std::cerr << "--corpus-limit must be positive" << std::endl;
    return EXIT_FAILURE;
  }
  if (corpus_output != "fp32" && corpus_output != "bf16" && corpus_output != "both") {
    std::cerr << "--corpus-output must be fp32, bf16, or both" << std::endl;
    return EXIT_FAILURE;
  }

  if (m % 128 != 0 || n % 128 != 0 || k % 512 != 0) {
    std::cerr << "Expected M,N multiples of 128 and K multiple of 512." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "NVFP4 reduction-order probe" << std::endl;
  std::cout << "M=" << m << " N=" << n << " K=" << k << " alpha=" << alpha << std::endl;
  std::cout << "scale_range=[" << scale_min << ", " << scale_max << "]" << std::endl;

  if (metadata_only) {
    bool ran_fp32 = run_metadata_probe<float>(m, n, k);
    bool ran_bf16 = run_metadata_probe<__nv_bfloat16>(m, n, k);
    return (ran_fp32 && ran_bf16) ? 0 : 2;
  }

  size_t fp4_a_size = static_cast<size_t>(m) * k / 2;
  size_t fp4_b_size = static_cast<size_t>(n) * k / 2;
  size_t scale_a_size = static_cast<size_t>(m) * (k / 16);
  size_t scale_b_size = static_cast<size_t>(n) * (k / 16);

  __nv_fp4x2_e2m1 *a = nullptr;
  __nv_fp4x2_e2m1 *b = nullptr;
  __nv_fp8_e4m3 *a_scale = nullptr;
  __nv_fp8_e4m3 *b_scale = nullptr;
  CHECK_CUDA(cudaMalloc(&a, fp4_a_size * sizeof(__nv_fp4x2_e2m1)));
  CHECK_CUDA(cudaMalloc(&b, fp4_b_size * sizeof(__nv_fp4x2_e2m1)));
  CHECK_CUDA(cudaMalloc(&a_scale, scale_a_size * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&b_scale, scale_b_size * sizeof(__nv_fp8_e4m3)));

  fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t *>(a), fp4_a_size, 2026, 0.0f, 255.0f);
  fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t *>(b), fp4_b_size, 2027, 0.0f, 255.0f);
  fill<__nv_fp8_e4m3, FillMode::RANDOM>(a_scale, scale_a_size, 2028, scale_min, scale_max);
  fill<__nv_fp8_e4m3, FillMode::RANDOM>(b_scale, scale_b_size, 2029, scale_min, scale_max);
  CHECK_CUDA(cudaDeviceSynchronize());

  if (!corpus_path.empty()) {
    std::ofstream out(corpus_path);
    if (!out) {
      std::cerr << "Failed to open corpus path: " << corpus_path << std::endl;
      return EXIT_FAILURE;
    }
    bool ran = true;
    if (corpus_output == "fp32" || corpus_output == "both") {
      ran = emit_corpus_for_output<float>(
          out, a, b, a_scale, b_scale, alpha, scale_min, scale_max, m, n, k, corpus_limit) && ran;
    }
    if (corpus_output == "bf16" || corpus_output == "both") {
      ran = emit_corpus_for_output<__nv_bfloat16>(
          out, a, b, a_scale, b_scale, alpha, scale_min, scale_max, m, n, k, corpus_limit) && ran;
    }
    std::cout << "wrote corpus: " << corpus_path << std::endl;
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(a_scale));
    CHECK_CUDA(cudaFree(b_scale));
    return ran ? 0 : 2;
  }

  bool ran_fp32 = run_probe<float>(a, b, a_scale, b_scale, alpha, m, n, k);
  run_probe<__nv_bfloat16>(a, b, a_scale, b_scale, alpha, m, n, k);

  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(a_scale));
  CHECK_CUDA(cudaFree(b_scale));
  return ran_fp32 ? 0 : 2;
}
