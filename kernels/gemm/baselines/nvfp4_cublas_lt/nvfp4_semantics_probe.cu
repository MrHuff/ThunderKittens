/***************************************************************************************************
 * NVFP4 TCGEN05 semantics probe.
 *
 * Compares native cuBLASLt NVFP4 GEMM against several software references using
 * the same packed FP4 E2M1 payloads and UE4M3 block-scale buffers.
 *
 * This is a correctness diagnostic, not a benchmark.
 **************************************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
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
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "  " << name << "=<unavailable:" << status << ">" << std::endl;
      return 0;
    }
    std::cout << "  " << name << "=" << static_cast<uint64_t>(value) << std::endl;
    return value;
  }

  template <typename T>
  T get_cap_attr(cublasLtMatmulAlgoCapAttributes_t attr, const char *name) const {
    T value = 0;
    size_t size_written = 0;
    cublasStatus_t status = cublasLtMatmulAlgoCapGetAttribute(
        &heuristic.algo, attr, &value, sizeof(value), &size_written);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "  " << name << "=<unavailable:" << status << ">" << std::endl;
      return 0;
    }
    std::cout << "  " << name << "=0x" << std::hex << static_cast<uint64_t>(value)
              << std::dec << std::endl;
    return value;
  }

  void print_algo_metadata() const {
    std::cout << "cuBLASLt heuristic metadata:" << std::endl;
    get_config_attr<int32_t>(CUBLASLT_ALGO_CONFIG_ID, "algo_id");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_TILE_ID, "tile_id");
    get_config_attr<int32_t>(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, "splitk_num");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, "reduction_scheme");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, "cta_swizzling");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, "custom_option");
    get_config_attr<uint32_t>(CUBLASLT_ALGO_CONFIG_STAGES_ID, "stages_id");
    get_config_attr<uint16_t>(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, "inner_shape_id");
    get_config_attr<uint16_t>(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, "cluster_shape_id");
    get_cap_attr<uint32_t>(CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, "cap_splitk_support");
    get_cap_attr<uint32_t>(CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, "cap_reduction_scheme_mask");
    get_cap_attr<uint64_t>(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, "cap_numerical_impl_flags");
  }

  void init(int m, int n, int k) {
    CHECK_CUBLAS(cublasLtCreate(&handle));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));

    // cuBLASLt sees its A as our column-major B and its B as our row-major A.
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_4F_E2M1, k, n, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_4F_E2M1, k, m, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, n, m, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, n, m, n));

    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    void *dummy_scale_ptr = workspace;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &dummy_scale_ptr, sizeof(dummy_scale_ptr)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &dummy_scale_ptr, sizeof(dummy_scale_ptr)));

    int returned_results = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, layout_a, layout_b, layout_c, layout_d,
        preference, 1, &heuristic, &returned_results));
    if (returned_results == 0) {
      std::cerr << "No cuBLASLt NVFP4 algorithm found." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    print_algo_metadata();
  }

  void run(
      const __nv_fp4x2_e2m1 *a,
      const __nv_fp4x2_e2m1 *b,
      const __nv_fp8_e4m3 *a_scale,
      const __nv_fp8_e4m3 *b_scale,
      float alpha,
      __nv_bfloat16 *d,
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
  }
};

__device__ inline float bf16_round_trip(float x) {
  __nv_bfloat16 y = __float2bfloat16_rn(x);
  return __bfloat162float(y);
}

template <int Variant>
__global__ void reference_nvfp4_variant_kernel(
    __nv_bfloat16 *d,
    const __nv_fp4x2_e2m1 *a_packed,
    const __nv_fp4x2_e2m1 *b_packed,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m, int n, int k) {
  constexpr int block_size = 16;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  int k_blocks = k / block_size;
  for (int kk = 0; kk < k; kk += 2) {
    int a_idx = row * (k / 2) + kk / 2;
    int b_idx = col * (k / 2) + kk / 2;

    float2 a_vals = static_cast<float2>(a_packed[a_idx]);
    float2 b_vals = static_cast<float2>(b_packed[b_idx]);

    int k_block0 = kk / block_size;
    int k_block1 = (kk + 1) / block_size;

    float a_s0 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
        a_scale[scale_swizzle_idx(row, k_block0, k_blocks)]);
    float a_s1 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
        a_scale[scale_swizzle_idx(row, k_block1, k_blocks)]);
    float b_s0 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
        b_scale[scale_swizzle_idx(col, k_block0, k_blocks)]);
    float b_s1 = kittens::base_types::convertor<float, __nv_fp8_e4m3>::convert(
        b_scale[scale_swizzle_idx(col, k_block1, k_blocks)]);

    if constexpr (Variant == 0) {
      // Mathematical PTX/MLIR form: (A * scale_a) * (B * scale_b).
      acc += (a_vals.x * a_s0) * (b_vals.x * b_s0);
      acc += (a_vals.y * a_s1) * (b_vals.y * b_s1);
    } else if constexpr (Variant == 1) {
      // Product first, then combined block scale.
      acc += (a_vals.x * b_vals.x) * (a_s0 * b_s0);
      acc += (a_vals.y * b_vals.y) * (a_s1 * b_s1);
    } else if constexpr (Variant == 2) {
      // Dequantized operands rounded to BF16 before multiply.
      acc += bf16_round_trip(a_vals.x * a_s0) * bf16_round_trip(b_vals.x * b_s0);
      acc += bf16_round_trip(a_vals.y * a_s1) * bf16_round_trip(b_vals.y * b_s1);
    } else if constexpr (Variant == 3) {
      // Each scaled product rounded to BF16 before FP32 accumulation.
      acc += bf16_round_trip((a_vals.x * a_s0) * (b_vals.x * b_s0));
      acc += bf16_round_trip((a_vals.y * a_s1) * (b_vals.y * b_s1));
    } else if constexpr (Variant == 4) {
      // BF16 running accumulator. This should be worse; it is a sanity bound.
      acc = bf16_round_trip(acc + (a_vals.x * a_s0) * (b_vals.x * b_s0));
      acc = bf16_round_trip(acc + (a_vals.y * a_s1) * (b_vals.y * b_s1));
    }
  }

  d[row * n + col] = __float2bfloat16_rn(acc * alpha);
}

template <int Variant>
static void reference_nvfp4_variant(
    __nv_bfloat16 *d,
    const __nv_fp4x2_e2m1 *a,
    const __nv_fp4x2_e2m1 *b,
    const __nv_fp8_e4m3 *a_scale,
    const __nv_fp8_e4m3 *b_scale,
    float alpha,
    int m, int n, int k) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  reference_nvfp4_variant_kernel<Variant>
      <<<grid, block>>>(d, a, b, a_scale, b_scale, alpha, m, n, k);
}

struct DiffStats {
  double mean = 0.0;
  double max = 0.0;
  size_t nonzero = 0;
  size_t first = 0;
  float first_native = 0.0f;
  float first_ref = 0.0f;
};

static DiffStats compare_outputs(
    const __nv_bfloat16 *native,
    const __nv_bfloat16 *reference,
    size_t count) {
  std::vector<__nv_bfloat16> h_native(count);
  std::vector<__nv_bfloat16> h_reference(count);
  CHECK_CUDA(cudaMemcpy(h_native.data(), native, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_reference.data(), reference, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  DiffStats stats;
  bool saw_first = false;
  for (size_t i = 0; i < count; ++i) {
    float a = kittens::base_types::convertor<float, __nv_bfloat16>::convert(h_native[i]);
    float b = kittens::base_types::convertor<float, __nv_bfloat16>::convert(h_reference[i]);
    double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
    stats.mean += diff;
    stats.max = std::max(stats.max, diff);
    if (diff != 0.0) {
      stats.nonzero++;
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

static void print_stats(
    const std::string &name,
    const __nv_bfloat16 *native,
    const __nv_bfloat16 *reference,
    int m, int n) {
  DiffStats stats = compare_outputs(native, reference, static_cast<size_t>(m) * n);
  int row = static_cast<int>(stats.first / n);
  int col = static_cast<int>(stats.first % n);
  std::cout << std::left << std::setw(28) << name
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

int main(int argc, char **argv) {
  int m = argc > 1 ? std::atoi(argv[1]) : 512;
  int n = argc > 2 ? std::atoi(argv[2]) : 512;
  int k = argc > 3 ? std::atoi(argv[3]) : 64;
  float alpha = argc > 4 ? std::atof(argv[4]) : 1.0f;
  bool metadata_only = argc > 5 && std::string(argv[5]) == "--metadata-only";

  if (m % 128 != 0 || n % 128 != 0 || k % 64 != 0) {
    std::cerr << "Expected M,N multiples of 128 and K multiple of 64." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "NVFP4 TCGEN05 semantics probe" << std::endl;
  std::cout << "M=" << m << " N=" << n << " K=" << k << " alpha=" << alpha << std::endl;
  std::cout << "Native: cuBLASLt CUDA_R_4F_E2M1 + VEC16_UE4M3 scales, FP32 compute, BF16 output" << std::endl;

  if (metadata_only) {
    CublasLtNvfp4Gemm gemm;
    gemm.init(m, n, k);
    gemm.destroy();
    return 0;
  }

  size_t fp4_a_size = static_cast<size_t>(m) * k / 2;
  size_t fp4_b_size = static_cast<size_t>(n) * k / 2;
  size_t scale_a_size = static_cast<size_t>(m) * (k / 16);
  size_t scale_b_size = static_cast<size_t>(n) * (k / 16);
  size_t d_size = static_cast<size_t>(m) * n;

  __nv_fp4x2_e2m1 *a = nullptr;
  __nv_fp4x2_e2m1 *b = nullptr;
  __nv_fp8_e4m3 *a_scale = nullptr;
  __nv_fp8_e4m3 *b_scale = nullptr;
  __nv_bfloat16 *d_native = nullptr;
  __nv_bfloat16 *d_ref0 = nullptr;
  __nv_bfloat16 *d_ref1 = nullptr;
  __nv_bfloat16 *d_ref2 = nullptr;
  __nv_bfloat16 *d_ref3 = nullptr;
  __nv_bfloat16 *d_ref4 = nullptr;

  CHECK_CUDA(cudaMalloc(&a, fp4_a_size * sizeof(__nv_fp4x2_e2m1)));
  CHECK_CUDA(cudaMalloc(&b, fp4_b_size * sizeof(__nv_fp4x2_e2m1)));
  CHECK_CUDA(cudaMalloc(&a_scale, scale_a_size * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&b_scale, scale_b_size * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA(cudaMalloc(&d_native, d_size * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_ref0, d_size * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_ref1, d_size * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_ref2, d_size * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_ref3, d_size * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_ref4, d_size * sizeof(__nv_bfloat16)));

  fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t *>(a), fp4_a_size, 2026, 0.0f, 255.0f);
  fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t *>(b), fp4_b_size, 2027, 0.0f, 255.0f);
  fill<__nv_fp8_e4m3, FillMode::RANDOM>(a_scale, scale_a_size, 2028, 0.25f, 2.0f);
  fill<__nv_fp8_e4m3, FillMode::RANDOM>(b_scale, scale_b_size, 2029, 0.25f, 2.0f);
  fill<__nv_bfloat16, FillMode::CONSTANT>(d_native, d_size, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  CublasLtNvfp4Gemm gemm;
  gemm.init(m, n, k);
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  gemm.run(a, b, a_scale, b_scale, alpha, d_native, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  reference_nvfp4_variant<0>(d_ref0, a, b, a_scale, b_scale, alpha, m, n, k);
  reference_nvfp4_variant<1>(d_ref1, a, b, a_scale, b_scale, alpha, m, n, k);
  reference_nvfp4_variant<2>(d_ref2, a, b, a_scale, b_scale, alpha, m, n, k);
  reference_nvfp4_variant<3>(d_ref3, a, b, a_scale, b_scale, alpha, m, n, k);
  reference_nvfp4_variant<4>(d_ref4, a, b, a_scale, b_scale, alpha, m, n, k);
  CHECK_CUDA(cudaDeviceSynchronize());

  print_stats("(A*sA)*(B*sB)", d_native, d_ref0, m, n);
  print_stats("(A*B)*(sA*sB)", d_native, d_ref1, m, n);
  print_stats("bf16(A*sA),bf16(B*sB)", d_native, d_ref2, m, n);
  print_stats("bf16(each product)", d_native, d_ref3, m, n);
  print_stats("bf16 running acc", d_native, d_ref4, m, n);

  CHECK_CUDA(cudaStreamDestroy(stream));
  gemm.destroy();
  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(a_scale));
  CHECK_CUDA(cudaFree(b_scale));
  CHECK_CUDA(cudaFree(d_native));
  CHECK_CUDA(cudaFree(d_ref0));
  CHECK_CUDA(cudaFree(d_ref1));
  CHECK_CUDA(cudaFree(d_ref2));
  CHECK_CUDA(cudaFree(d_ref3));
  CHECK_CUDA(cudaFree(d_ref4));
  return 0;
}
