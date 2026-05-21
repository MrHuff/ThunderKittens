// ================================================================
// MXFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "mxfp4_gemm.cuh"
// mxfp4_quantize.cuh removed — use standalone mxfp4_v2 quantizer
#include "mxfp4_batched_gemm.cuh"
#include "mxfp4_split2_accum_gemm.cuh"
#include "mxfp4_split3_accum_gemm.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <optional>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_entrypoint(const __grid_constant__ mxfp4_gemm::globals<C> g) {
    mxfp4_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = mxfp4_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " NUM_D_TILES=" << C::NUM_D_TILES
              << " OVERLAP_EPI=" << C::OVERLAP_EPI << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K / 2 + size_t(N) * K / 2 + size_t(M) * N * 2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp4x2_e2m1*> d_A(arg_group_count);
    std::vector<__nv_fp4x2_e2m1*> d_B(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_B_sc(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_D[i], M*N*sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M*N*sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i*100, 0.0f, 255.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i*100 + 1, 0.0f, 255.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_A_sc[i], M*K/32, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_B_sc[i], N*K/32, seed + i*100 + 3, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device (MXFP4 with E8M0 scales, block size 32)
    reference_blockscaled_gemm<__nv_fp4x2_e2m1, __nv_fp8_e8m0, __nv_bfloat16, 32>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{d_A_sc[i], M/128, K/128, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{d_B_sc[i], N/128, K/128, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Bg, Bsg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    // Prepare kernel launch configuration
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_correctness(d_D[0], d_D_ref, M * N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_A_sc[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    int N;
    bool ncu = false;

    // Template parameters: Nb, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP_SIZE, NUM_D_TILES, OVERLAP_EPI
    N = 1024;
    run_benchmark<mxfp4_gemm::config<128, 5, 4, 12, 2, true>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<mxfp4_gemm::config<256, 5, 8, 12, 2, true>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<mxfp4_gemm::config<256, 5, 8, 8, 2, false>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<mxfp4_gemm::config<256, 4, 16, 16, 4, false>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<mxfp4_gemm::config<256, 4, 8, 8, 2, false>>(N, N, N, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"

namespace {

using mxfp4_onepass_cfg1 = mxfp4_split2_accum_gemm::config<128, 5, 4, 12, 2, true, 2, false>;
using mxfp4_onepass_cfg3 = mxfp4_split2_accum_gemm::config<256, 5, 8, 4, 2, false, 2, false>;
using mxfp4_onepass_cfg5 = mxfp4_split2_accum_gemm::config<256, 5, 8, 12, 2, false, 2, false>;
using mxfp4_split3_onepass_cfg1 = mxfp4_split3_accum_gemm::config<128, 5, 4, 12, 2, true, 2, false>;
using mxfp4_split3_onepass_cfg3 = mxfp4_split3_accum_gemm::config<256, 5, 8, 4, 2, false, 2, false>;
using mxfp4_split3_onepass_cfg5 = mxfp4_split3_accum_gemm::config<256, 5, 8, 12, 2, false, 2, false>;

void check_fp4_matrix(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
}

void check_output_matrix(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
}

template <typename GL>
GL tensor_to_gl_tma_view(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2 || t.dim() == 4, name, " must be 2D or 4D");

    if constexpr (std::is_same_v<typename GL::dtype, kittens::fp4e2m1_2>) {
        TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
    } else if constexpr (std::is_same_v<typename GL::dtype, kittens::fp8e8m0>) {
        TORCH_CHECK(t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte,
                    name, " must be fp8e8m0/uint8");
    } else if constexpr (std::is_same_v<typename GL::dtype, kittens::bf16>) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    }

    int b = 1;
    int d = 1;
    int r = 1;
    int c = 1;

    if (t.dim() == 2) {
        TORCH_CHECK(t.stride(1) == 1, name, " 2D TMA view must have unit inner stride");
        TORCH_CHECK(t.stride(0) >= t.size(1), name, " 2D TMA leading stride is smaller than logical width");
        r = static_cast<int>(t.size(0));
        c = static_cast<int>(t.stride(0));
    } else {
        TORCH_CHECK(t.stride(3) == 1, name, " 4D TMA view must have unit innermost stride");
        TORCH_CHECK(t.stride(2) == t.size(3), name, " 4D TMA inner tile stride mismatch");
        TORCH_CHECK(t.stride(1) == t.size(2) * t.size(3), name, " 4D TMA depth stride mismatch");
        TORCH_CHECK(t.stride(0) % t.stride(1) == 0, name, " 4D TMA batch stride mismatch");
        b = static_cast<int>(t.size(0));
        d = static_cast<int>(t.stride(0) / t.stride(1));
        r = static_cast<int>(t.size(2));
        c = static_cast<int>(t.size(3));
        TORCH_CHECK(d >= t.size(1), name, " 4D TMA leading depth is smaller than logical depth");
    }

    return kittens::make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()), b, d, r, c);
}

template <typename GL>
GL tensor_to_gl_tma_2d_slice(
    const at::Tensor& t,
    const char* name,
    int64_t row_start,
    int64_t col_start,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.stride(1) == 1, name, " slice must have unit inner stride");
    TORCH_CHECK(row_start >= 0 && col_start >= 0 && rows > 0 && cols > 0, name, " invalid slice");
    TORCH_CHECK(row_start + rows <= t.size(0), name, " row slice exceeds tensor");
    TORCH_CHECK(col_start + cols <= t.stride(0), name, " col slice exceeds leading stride");

    if constexpr (std::is_same_v<typename GL::dtype, kittens::fp4e2m1_2>) {
        TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
    } else if constexpr (std::is_same_v<typename GL::dtype, kittens::bf16>) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    } else {
        TORCH_CHECK(false, name, " unsupported 2D sliced TMA dtype");
    }

    const int64_t element_offset = row_start * t.stride(0) + col_start;
    const auto* ptr = static_cast<const char*>(t.data_ptr()) + element_offset * t.element_size();
    return kittens::make_gl<GL>(
        reinterpret_cast<uint64_t>(ptr),
        1,
        1,
        static_cast<int>(rows),
        static_cast<int>(t.stride(0)));
}

template <typename GL>
GL tensor_to_gl_tma_scale_slice(
    const at::Tensor& t,
    const char* name,
    int64_t row_start,
    int64_t k_start,
    int64_t rows,
    int64_t k_size
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 4, name, " must be 4D");
    TORCH_CHECK(t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte,
                name, " must be fp8e8m0/uint8");
    TORCH_CHECK(t.stride(3) == 1, name, " last stride must be contiguous");
    TORCH_CHECK(t.stride(2) == t.size(3), name, " inner tile stride mismatch");
    TORCH_CHECK(t.stride(1) == t.size(2) * t.size(3), name, " depth stride mismatch");
    TORCH_CHECK(t.stride(0) % t.stride(1) == 0, name, " batch stride mismatch");
    TORCH_CHECK(row_start >= 0 && k_start >= 0 && rows > 0 && k_size > 0, name, " invalid scale slice");
    TORCH_CHECK(row_start % 128 == 0 && k_start % 128 == 0 && rows % 128 == 0 && k_size % 128 == 0,
                name, " scale slices must be 128-aligned");
    const int64_t row_block_start = row_start / 128;
    const int64_t k_block_start = k_start / 128;
    const int64_t row_blocks = rows / 128;
    const int64_t k_blocks = k_size / 128;
    TORCH_CHECK(row_block_start + row_blocks <= t.size(0), name, " row scale slice exceeds tensor");
    TORCH_CHECK(k_block_start + k_blocks <= t.size(1), name, " K scale slice exceeds tensor");

    const int64_t element_offset = row_block_start * t.stride(0) + k_block_start * t.stride(1);
    const auto* ptr = static_cast<const char*>(t.data_ptr()) + element_offset * t.element_size();
    return kittens::make_gl<GL>(
        reinterpret_cast<uint64_t>(ptr),
        static_cast<int>(row_blocks),
        static_cast<int>(t.stride(0) / t.stride(1)),
        static_cast<int>(t.size(2)),
        static_cast<int>(t.size(3)));
}

void check_tilemask(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kByte, name, " must be uint8");
    TORCH_CHECK(t.size(0) == rows, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols, name, " second dim mismatch");
}

bool rope_tensor_disabled(const at::Tensor& t) {
    return t.numel() == 0;
}

bool is_power_of_two(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void check_rope_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t seq_len,
    int64_t rotary_dim
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    TORCH_CHECK(t.size(1) == rotary_dim / 2, name, " rotary_dim/2 mismatch");
}

void check_rope_epilogue_args(
    const at::Tensor& D,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    TORCH_CHECK(rope_seq_len > 0, "rope_seq_len must be positive");
    TORCH_CHECK(rope_head_dim > 0, "rope_head_dim must be positive");
    TORCH_CHECK(rope_rotary_dim > 0, "rope_rotary_dim must be positive");
    TORCH_CHECK((rope_head_dim % 2) == 0, "rope_head_dim must be even");
    TORCH_CHECK((rope_rotary_dim % 2) == 0, "rope_rotary_dim must be even");
    TORCH_CHECK(rope_rotary_dim <= rope_head_dim, "rope_rotary_dim must be <= rope_head_dim");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D.size(1) % rope_head_dim == 0, "output cols must be divisible by rope_head_dim");
    check_rope_tensor(rope_cos, "rope_cos", rope_seq_len, rope_rotary_dim);
    check_rope_tensor(rope_sin, "rope_sin", rope_seq_len, rope_rotary_dim);
    kittens::py::device_check(D, rope_cos, rope_sin);
}

void check_rope_live64_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t seq_len
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == seq_len, name, " seq_len mismatch");
    TORCH_CHECK(t.size(1) == 32, name, " second dim must equal 32");
    TORCH_CHECK(t.size(2) == 2, name, " third dim must equal 2");
}

void check_rope_live64_args(
    const at::Tensor& D,
    const at::Tensor& rope_cs,
    int64_t rope_seq_len
) {
    TORCH_CHECK(rope_seq_len > 0, "rope_seq_len must be positive");
    TORCH_CHECK(is_power_of_two(rope_seq_len), "rope_seq_len must be a power of two");
    TORCH_CHECK(D.size(0) % rope_seq_len == 0, "output rows must be divisible by rope_seq_len");
    TORCH_CHECK(D.size(1) % 64 == 0, "output cols must be divisible by 64");
    check_rope_live64_tensor(rope_cs, "rope_cs", rope_seq_len);
    kittens::py::device_check(D, rope_cs);
}

__global__ void deepseek_mla_inverse_rope_pack_grad_kernel(
    const __nv_bfloat16* __restrict__ grad_q,
    const __nv_bfloat16* __restrict__ grad_kv,
    const __nv_bfloat16* __restrict__ grad_kpe,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    __nv_bfloat16* __restrict__ out,
    int64_t total,
    int q_dim,
    int qk_head_dim,
    int rope_dim,
    int kv_lora_rank,
    int kv_pad_dim,
    int kpe_pad_dim,
    int padded_n,
    int seq_len
) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (
        int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        linear < total;
        linear += stride
    ) {
        const int row = static_cast<int>(linear / padded_n);
        const int col = static_cast<int>(linear - static_cast<int64_t>(row) * padded_n);
        const int kpe_base = q_dim + kv_pad_dim;
        __nv_bfloat16 value = __float2bfloat16_rn(0.0f);

        if (col < q_dim) {
            const int head_col = col % qk_head_dim;
            if (head_col < rope_dim) {
                const int pair_col = col - head_col + ((head_col / 2) * 2);
                const float x = __bfloat162float(grad_q[static_cast<int64_t>(row) * q_dim + pair_col]);
                const float y = __bfloat162float(grad_q[static_cast<int64_t>(row) * q_dim + pair_col + 1]);
                const int rope_offset = (row % seq_len) * (rope_dim / 2) + (head_col / 2);
                const float c = rope_cos[rope_offset];
                const float s = rope_sin[rope_offset];
                const float rotated = (head_col & 1) == 0 ? (x * c + y * s) : (y * c - x * s);
                value = __float2bfloat16_rn(rotated);
            } else {
                value = grad_q[static_cast<int64_t>(row) * q_dim + col];
            }
        } else if (col < kpe_base) {
            const int kv_col = col - q_dim;
            if (kv_col < kv_lora_rank) {
                value = grad_kv[static_cast<int64_t>(row) * kv_lora_rank + kv_col];
            }
        } else {
            const int kpe_col = col - kpe_base;
            if (kpe_col < rope_dim) {
                const int pair_col = (kpe_col / 2) * 2;
                const float x = __bfloat162float(grad_kpe[static_cast<int64_t>(row) * rope_dim + pair_col]);
                const float y = __bfloat162float(grad_kpe[static_cast<int64_t>(row) * rope_dim + pair_col + 1]);
                const int rope_offset = (row % seq_len) * (rope_dim / 2) + (kpe_col / 2);
                const float c = rope_cos[rope_offset];
                const float s = rope_sin[rope_offset];
                const float rotated = (kpe_col & 1) == 0 ? (x * c + y * s) : (y * c - x * s);
                value = __float2bfloat16_rn(rotated);
            } else if (kpe_col < kpe_pad_dim) {
                value = __float2bfloat16_rn(0.0f);
            }
        }

        out[linear] = value;
    }
}

void mxfp4_deepseek_mla_inverse_rope_pack_grad_entrypoint(
    const at::Tensor& grad_q,
    const at::Tensor& grad_kv,
    const at::Tensor& grad_kpe,
    const at::Tensor& rope_cos,
    const at::Tensor& rope_sin,
    at::Tensor& out,
    int64_t seq_len,
    int64_t n_heads,
    int64_t qk_head_dim,
    int64_t rope_dim,
    int64_t kv_lora_rank,
    int64_t kv_pad_dim,
    int64_t kpe_pad_dim
) {
    TORCH_CHECK(grad_q.is_cuda() && grad_kv.is_cuda() && grad_kpe.is_cuda() && out.is_cuda(),
                "DeepSeek MLA grad pack expects CUDA tensors");
    TORCH_CHECK(grad_q.is_contiguous() && grad_kv.is_contiguous() &&
                grad_kpe.is_contiguous() && out.is_contiguous(),
                "DeepSeek MLA grad pack expects contiguous tensors");
    TORCH_CHECK(grad_q.dim() == 2 && grad_kv.dim() == 2 && grad_kpe.dim() == 2 && out.dim() == 2,
                "DeepSeek MLA grad pack expects 2D tensors");
    TORCH_CHECK(grad_q.scalar_type() == at::kBFloat16 &&
                grad_kv.scalar_type() == at::kBFloat16 &&
                grad_kpe.scalar_type() == at::kBFloat16 &&
                out.scalar_type() == at::kBFloat16,
                "DeepSeek MLA grad pack tensors must be bf16");
    TORCH_CHECK(seq_len > 0 && n_heads > 0 && qk_head_dim > 0 && rope_dim > 0,
                "DeepSeek MLA grad pack received invalid dimensions");
    TORCH_CHECK((rope_dim % 2) == 0 && rope_dim <= qk_head_dim,
                "DeepSeek MLA grad pack requires an even rope_dim <= qk_head_dim");
    const int64_t M = grad_q.size(0);
    const int64_t q_dim = n_heads * qk_head_dim;
    const int64_t padded_n = q_dim + kv_pad_dim + kpe_pad_dim;
    TORCH_CHECK(M % seq_len == 0, "DeepSeek MLA grad pack rows must be divisible by seq_len");
    TORCH_CHECK(grad_q.size(1) == q_dim, "DeepSeek MLA grad_q width mismatch");
    TORCH_CHECK(grad_kv.size(0) == M && grad_kv.size(1) == kv_lora_rank,
                "DeepSeek MLA grad_kv shape mismatch");
    TORCH_CHECK(grad_kpe.size(0) == M && grad_kpe.size(1) == rope_dim,
                "DeepSeek MLA grad_kpe shape mismatch");
    TORCH_CHECK(out.size(0) == M && out.size(1) == padded_n,
                "DeepSeek MLA packed grad output shape mismatch");
    check_rope_tensor(rope_cos, "rope_cos", seq_len, rope_dim);
    check_rope_tensor(rope_sin, "rope_sin", seq_len, rope_dim);
    kittens::py::device_check(grad_q, grad_kv, grad_kpe, rope_cos, rope_sin, out);

    const int threads = 256;
    const int64_t total = M * padded_n;
    const int64_t needed_blocks = (total + threads - 1) / threads;
    int max_blocks = 32768;
    if (const char* env = std::getenv("MXFP4_DEEPSEEK_MLA_PACK_GRAD_BLOCKS")) {
        max_blocks = std::max(1, std::atoi(env));
    }
    const int blocks = static_cast<int>(needed_blocks < max_blocks ? needed_blocks : max_blocks);
    deepseek_mla_inverse_rope_pack_grad_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_kpe.data_ptr<at::BFloat16>()),
        rope_cos.data_ptr<float>(),
        rope_sin.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        total,
        static_cast<int>(q_dim),
        static_cast<int>(qk_head_dim),
        static_cast<int>(rope_dim),
        static_cast<int>(kv_lora_rank),
        static_cast<int>(kv_pad_dim),
        static_cast<int>(kpe_pad_dim),
        static_cast<int>(padded_n),
        static_cast<int>(seq_len));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static bool use_rope_live64_rht32() {
    return std::getenv("MXFP4_GEMM_ROPE_LIVE64_RHT32") != nullptr;
}

template <typename C>
static void check_rope_live64_rht32_config(bool enabled) {
    if (enabled) {
        TORCH_CHECK((C::Nb / C::EPI_PIPE_DEPTH) % 32 == 0,
                    "MXFP4_GEMM_ROPE_LIVE64_RHT32 requires epilogue fragments divisible by 32 columns");
    }
}

template <typename C>
using rope_live64_rht32_config = mxfp4_gemm::config<
    C::Nb,
    C::LOAD_PIPE_DEPTH,
    C::EPI_PIPE_DEPTH,
    C::SUPERGROUP_SIZE,
    C::NUM_D_TILES,
    C::OVERLAP_EPI,
    C::Kb,
    true>;

void check_mxfp4_scale_tensor(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols,
    bool allow_views
) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    if (!allow_views) {
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    }
    TORCH_CHECK(t.dim() == 4, name, " must be 4D");
    TORCH_CHECK(
        t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte,
        name, " must be fp8 e8m0 or byte view"
    );
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 128, name, " second dim mismatch");
    TORCH_CHECK(t.size(2) == 32, name, " third dim must equal 32");
    TORCH_CHECK(t.size(3) == 16, name, " fourth dim must equal 16");

    if (allow_views) {
        TORCH_CHECK(t.stride(3) == 1, name, " last stride must be contiguous");
        TORCH_CHECK(t.stride(2) == 16, name, " stride(2) must equal 16");
        TORCH_CHECK(t.stride(1) == 512, name, " stride(1) must equal 512");
        TORCH_CHECK(t.stride(0) >= t.size(1) * 512, name, " leading stride too small");
        const auto data_ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((data_ptr & 0xF) == 0, name, " data pointer must be 16-byte aligned");
        TORCH_CHECK((t.stride(1) % 16) == 0, name, " stride(1) must be 16-byte aligned");
        TORCH_CHECK((t.stride(0) % 16) == 0, name, " stride(0) must be 16-byte aligned");
    }
}

template <typename ST>
void encode_mxfp4_scale_tensor_map(CUtensorMap* desc, const at::Tensor& t, const char* name) {
    static_assert(std::is_same_v<typename ST::dtype, kittens::fp8e8m0>,
                  "MXFP4 scale TMA helper assumes fp8e8m0 logical elements");
    static_assert(!ST::swizzle, "MXFP4 scale TMA helper only supports non-swizzled tiles");

    check_mxfp4_scale_tensor(t, name, t.size(0) * 128, t.size(1) * 128, true);

    uint64_t gmem_shape[4] = {
        static_cast<uint64_t>(t.size(3)),
        static_cast<uint64_t>(t.size(2)),
        static_cast<uint64_t>(t.size(1)),
        static_cast<uint64_t>(t.size(0)),
    };
    uint64_t gmem_stride[3] = {
        static_cast<uint64_t>(t.stride(2)),
        static_cast<uint64_t>(t.stride(1)),
        static_cast<uint64_t>(t.stride(0)),
    };
    uint32_t smem_shape[4] = {
        static_cast<uint32_t>(ST::cols),
        static_cast<uint32_t>(ST::rows),
        1, 1,
    };
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        4,
        t.data_ptr(),
        gmem_shape,
        gmem_stride,
        smem_shape,
        smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, name, " TMA creation failed");
}

void check_mxfp4_split2_dgrad_inputs(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list
) {
    check_fp4_matrix(A_full, "A_full");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full M must be a multiple of 128");
    TORCH_CHECK((A_full.size(1) * 2) % 128 == 0, "A_full K must be a multiple of 128");

    const int n = static_cast<int>(A_sc_list.size());
    TORCH_CHECK(n == 2, "split2 one-pass dgrad expects exactly 2 A scale tensors");
    TORCH_CHECK(
        n == static_cast<int>(A_col_offsets.size()) &&
        n == static_cast<int>(A_col_widths.size()) &&
        n == static_cast<int>(B_list.size()) &&
        n == static_cast<int>(B_sc_list.size()),
        "all split2 one-pass inputs must have length 2"
    );

    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets must be non-negative");
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths must be positive");
        TORCH_CHECK(
            A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
            "A_full slice exceeds packed width"
        );
        TORCH_CHECK((A_col_widths[i] * 2) % 128 == 0, "split widths must be multiples of 128");
        check_mxfp4_scale_tensor(A_sc_list[i], "A_sc_list[i]", A_full.size(0), A_col_widths[i] * 2, true);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i], "B_list packed K must match A_col_widths");
        TORCH_CHECK(B_list[i].size(0) % 128 == 0, "B_list rows must be multiples of 128");
        check_mxfp4_scale_tensor(B_sc_list[i], "B_sc_list[i]", B_list[i].size(0), B_list[i].size(1) * 2, false);
        kittens::py::device_check(A_full, A_sc_list[i], B_list[i], B_sc_list[i]);
    }
}

void check_mxfp4_split3_dgrad_inputs(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list
) {
    check_fp4_matrix(A_full, "A_full");
    TORCH_CHECK(A_full.size(0) % 128 == 0, "A_full M must be a multiple of 128");
    TORCH_CHECK((A_full.size(1) * 2) % 128 == 0, "A_full K must be a multiple of 128");

    const int n = static_cast<int>(A_sc_list.size());
    TORCH_CHECK(n == 3, "split3 one-pass dgrad expects exactly 3 A scale tensors");
    TORCH_CHECK(
        n == static_cast<int>(A_col_offsets.size()) &&
        n == static_cast<int>(A_col_widths.size()) &&
        n == static_cast<int>(B_list.size()) &&
        n == static_cast<int>(B_sc_list.size()),
        "all split3 one-pass inputs must have length 3"
    );

    for (int i = 0; i < n; ++i) {
        TORCH_CHECK(A_col_offsets[i] >= 0, "A_col_offsets must be non-negative");
        TORCH_CHECK(A_col_widths[i] > 0, "A_col_widths must be positive");
        TORCH_CHECK(
            A_col_offsets[i] + A_col_widths[i] <= A_full.size(1),
            "A_full slice exceeds packed width"
        );
        TORCH_CHECK((A_col_widths[i] * 2) % 128 == 0, "split widths must be multiples of 128");
        check_mxfp4_scale_tensor(A_sc_list[i], "A_sc_list[i]", A_full.size(0), A_col_widths[i] * 2, true);
        check_fp4_matrix(B_list[i], "B_list[i]");
        TORCH_CHECK(B_list[i].size(1) == A_col_widths[i], "B_list packed K must match A_col_widths");
        TORCH_CHECK(B_list[i].size(0) % 128 == 0, "B_list rows must be multiples of 128");
        check_mxfp4_scale_tensor(B_sc_list[i], "B_sc_list[i]", B_list[i].size(0), B_list[i].size(1) * 2, false);
        kittens::py::device_check(A_full, A_sc_list[i], B_list[i], B_sc_list[i]);
    }
}


template <typename C>
void launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out
) {
    using G = mxfp4_split2_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 2; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        TORCH_CHECK(fp4_cols > 0, "A_col_widths must be positive");
        TORCH_CHECK((2 * fp4_cols) % C::Kb == 0,
                    "one-pass split2 dgrad expects reduction widths aligned to Kb=", C::Kb);
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);

        uint64_t gmem_shape[5] = {
            static_cast<uint64_t>(swizzle_elements),
            static_cast<uint64_t>(M),
            static_cast<uint64_t>((fp4_cols + swizzle_elements - 1) / swizzle_elements),
            1, 1
        };
        uint64_t gmem_stride[4] = {
            static_cast<uint64_t>(a_full_row_stride),
            128,
            static_cast<uint64_t>(M * a_full_row_stride),
            static_cast<uint64_t>(M * a_full_row_stride)
        };
        uint32_t smem_shape[5] = {
            static_cast<uint32_t>(swizzle_elements),
            static_cast<uint32_t>(C::Mb / 2),
            1, 1, 1
        };
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        CUresult result = cuTensorMapEncodeTiled(
            &g_host.A_tma[i],
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            5,
            const_cast<void*>(data_ptr),
            gmem_shape,
            gmem_stride,
            smem_shape,
            smem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(result == CUDA_SUCCESS, "One-pass split2 MXFP4 A TMA creation failed for batch ", i);

        if (A_sc_list[i].is_contiguous()) {
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        } else {
            encode_mxfp4_scale_tensor_map<typename G::A_sc_tile>(&g_host.A_sc_tma[i], A_sc_list[i], "A_sc_list[i]");
        }

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, mxfp4_split2_accum_gemm::kernel<C>>(g_host);
}

template <typename C>
void launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out
) {
    using G = mxfp4_split3_accum_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const uint8_t* a_base = reinterpret_cast<const uint8_t*>(A_full.data_ptr());
    const int64_t a_full_row_stride = K_total_fp4;

    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);

    for (int i = 0; i < 3; ++i) {
        constexpr int64_t swizzle_elements = 128;
        const int64_t fp4_cols = A_col_widths[i];
        const int64_t fp4_offset = A_col_offsets[i];
        const void* data_ptr = a_base + fp4_offset;

        TORCH_CHECK(fp4_cols > 0, "A_col_widths must be positive");
        TORCH_CHECK((2 * fp4_cols) % C::Kb == 0,
                    "one-pass split3 dgrad expects reduction widths aligned to Kb=", C::Kb);
        g_host.num_red_blocks[i] = static_cast<int>((2 * fp4_cols) / C::Kb);

        uint64_t gmem_shape[5] = {
            static_cast<uint64_t>(swizzle_elements),
            static_cast<uint64_t>(M),
            static_cast<uint64_t>((fp4_cols + swizzle_elements - 1) / swizzle_elements),
            1, 1
        };
        uint64_t gmem_stride[4] = {
            static_cast<uint64_t>(a_full_row_stride),
            128,
            static_cast<uint64_t>(M * a_full_row_stride),
            static_cast<uint64_t>(M * a_full_row_stride)
        };
        uint32_t smem_shape[5] = {
            static_cast<uint32_t>(swizzle_elements),
            static_cast<uint32_t>(C::Mb / 2),
            1, 1, 1
        };
        uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

        CUresult result = cuTensorMapEncodeTiled(
            &g_host.A_tma[i],
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            5,
            const_cast<void*>(data_ptr),
            gmem_shape,
            gmem_stride,
            smem_shape,
            smem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(result == CUDA_SUCCESS, "One-pass split3 MXFP4 A TMA creation failed for batch ", i);

        if (A_sc_list[i].is_contiguous()) {
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        } else {
            encode_mxfp4_scale_tensor_map<typename G::A_sc_tile>(&g_host.A_sc_tma[i], A_sc_list[i], "A_sc_list[i]");
        }

        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    }

    auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
    memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

    kittens::py::launch_kernel<C, G, mxfp4_split3_accum_gemm::kernel<C>>(g_host);
}


void launch_mxfp4_split2_dgrad_gemm_strided_onepass(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 1:
            launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<mxfp4_onepass_cfg1>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 3:
            launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<mxfp4_onepass_cfg3>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 5:
            launch_mxfp4_split2_dgrad_gemm_strided_onepass_with_config<mxfp4_onepass_cfg5>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown MXFP4 split2 one-pass config_idx=", resolved_idx);
    }
}

void launch_mxfp4_split3_dgrad_gemm_strided_onepass(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int config_idx
) {
    int resolved_idx = config_idx;
    if (resolved_idx < 0) {
        resolved_idx = 5;
    }
    switch (resolved_idx) {
        case 1:
            launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config<mxfp4_split3_onepass_cfg1>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 3:
            launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config<mxfp4_split3_onepass_cfg3>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        case 5:
            launch_mxfp4_split3_dgrad_gemm_strided_onepass_with_config<mxfp4_split3_onepass_cfg5>(
                A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list, D_out);
            break;
        default:
            TORCH_CHECK(false, "Unknown MXFP4 split3 one-pass config_idx=", resolved_idx);
    }
}


} // namespace

template <typename C>
static void launch_mxfp4_gemm_dense(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor* output_scale = nullptr
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .output_scale = output_scale == nullptr ? nullptr : output_scale->data_ptr<float>(),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_mxfp4_gemm_dense_residual(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    auto r_gl = kittens::py::tensor_to_gl<typename G::D_gl>(R);
    memcpy(&g.R_tma, &r_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void launch_mxfp4_gemm_masked(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .tilemask_ptr = tilemask.data_ptr<uint8_t>(),
        .tilemask_rows = static_cast<int>(tilemask.size(0)),
        .tilemask_cols = static_cast<int>(tilemask.size(1)),
        .tilemask_transposed = tilemask_transposed
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

void mxfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    // Single config that works for all shapes with Kb=256.
    // config<256,5,8,4,2,false> = Nb=256, LOAD_PIPE=5, EPI=8, SG=4, DT=2, no overlap
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>(A, A_sc, B, B_sc, D);
}

void mxfp4_gemm_scaled_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &output_scale
) {
    TORCH_CHECK(output_scale.is_cuda(), "output_scale must be a CUDA scalar tensor");
    TORCH_CHECK(output_scale.scalar_type() == at::kFloat, "output_scale must be float32");
    TORCH_CHECK(output_scale.numel() == 1, "output_scale must contain one element");
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, false, true>>(
        A, A_sc, B, B_sc, D, &output_scale);
}

void mxfp4_gemm_residual_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D
) {
    check_output_matrix(R, "R", D.size(0), D.size(1));
    kittens::py::device_check(A, A_sc, B, B_sc, R, D);
    launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, true>>(
        A, A_sc, B, B_sc, R, D);
}

void mxfp4_gemm_residual_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &R,
    at::Tensor &D,
    int config_id
) {
    check_output_matrix(R, "R", D.size(0), D.size(1));
    kittens::py::device_check(A, A_sc, B, B_sc, R, D);

    //                     Nb   LOAD EPI  SG  DT  OVERLAP Kb   RHT   RESIDUAL
    switch (config_id) {
    case 0:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  4, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 1:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 4, 16,  4, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 2:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  8, 2, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 3:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8, 12, 4, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 4:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 5:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5, 16,  4, 2, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 6:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 4,  8, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 7:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  4, 4, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 8:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 4, 16, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 9:  launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  8,  4, 2, true,  256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    case 10: launch_mxfp4_gemm_dense_residual<mxfp4_gemm::config<256, 5,  4, 12, 2, false, 256, false, true>>(A, A_sc, B, B_sc, R, D); break;
    default: TORCH_CHECK(false, "Invalid residual config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_gemm_k128_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    // Same launch shape as the default consumer, but with Kb=128 so the
    // reduction granularity matches a single 128-column tile.
    launch_mxfp4_gemm_dense<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>(A, A_sc, B, B_sc, D);
}

void mxfp4_gemm_masked_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D
) {
    int64_t mask_rows = tilemask_transposed ? A_sc.size(1) : A_sc.size(0);
    int64_t mask_cols = tilemask_transposed ? A_sc.size(0) : A_sc.size(1);
    check_tilemask(tilemask, "tilemask", mask_rows, mask_cols);

    launch_mxfp4_gemm_masked<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>(
        A, A_sc, B, B_sc, tilemask, tilemask_transposed, D);
}

void mxfp4_gemm_masked_k128_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &tilemask,
    bool tilemask_transposed,
    at::Tensor &D
) {
    int64_t mask_rows = tilemask_transposed ? A_sc.size(1) : A_sc.size(0);
    int64_t mask_cols = tilemask_transposed ? A_sc.size(0) : A_sc.size(1);
    check_tilemask(tilemask, "tilemask", mask_rows, mask_cols);

    launch_mxfp4_gemm_masked<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>(
        A, A_sc, B, B_sc, tilemask, tilemask_transposed, D);
}


// ================================================================
// Config-selectable GEMM for tile tuning sweeps.
// ================================================================
template <typename C>
static void run_gemm_with_config(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .tilemask_ptr = nullptr,
        .tilemask_rows = 0,
        .tilemask_cols = 0,
        .tilemask_transposed = false
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_with_config_rope(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cos,
    const at::Tensor &rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .rope = {
            .cos = rope_cos.data_ptr<float>(),
            .sin = rope_sin.data_ptr<float>(),
            .seq_len = static_cast<int>(rope_seq_len),
            .head_dim = static_cast<int>(rope_head_dim),
            .rotary_dim = static_cast<int>(rope_rotary_dim),
        },
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_with_config_rope_live64_impl(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    using G = mxfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .rope_live64 = {
            .cs = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>()),
            .seq_len = static_cast<int>(rope_seq_len),
            .seq_mask = static_cast<int>(rope_seq_len - 1),
        },
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

template <typename C>
static void run_gemm_with_config_rope_live64(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    const bool apply_rht32 = use_rope_live64_rht32();
    check_rope_live64_rht32_config<C>(apply_rht32);
    if (apply_rht32) {
        run_gemm_with_config_rope_live64_impl<rope_live64_rht32_config<C>>(
            A, A_sc, B, B_sc, D, rope_cs, rope_seq_len);
    } else {
        run_gemm_with_config_rope_live64_impl<C>(
            A, A_sc, B, B_sc, D, rope_cs, rope_seq_len);
    }
}

void mxfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D, int config_id
) {
    //                     Nb   LOAD EPI  SG  DT  OVERLAP
    switch (config_id) {
    // Defaults (used by mxfp4_gemm_entrypoint)
    case 0:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 1:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    // Best from Kb=128 sweep
    case 2:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 3:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D); break;
    // Additional candidates
    case 4:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 5:  run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 6:  run_gemm_with_config<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 7:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 8:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 9:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 10: run_gemm_with_config<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(A, A_sc, B, B_sc, D); break;

    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_gemm_rope_live64_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len
) {
    check_rope_live64_args(D, rope_cs, rope_seq_len);
    run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>(
        A, A_sc, B, B_sc, D, rope_cs, rope_seq_len);
}

void mxfp4_gemm_rope_live64_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cs,
    int64_t rope_seq_len,
    int config_id
) {
    check_rope_live64_args(D, rope_cs, rope_seq_len);
    switch (config_id) {
    case 0:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 1:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 2:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 3:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 4:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 5:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 6:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 7:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 8:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 9:  run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    case 10: run_gemm_with_config_rope_live64<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cs, rope_seq_len); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_gemm_rope_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cos,
    const at::Tensor &rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim
) {
    check_rope_epilogue_args(D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim);
    run_gemm_with_config_rope<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>(
        A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim);
}

void mxfp4_gemm_rope_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &rope_cos,
    const at::Tensor &rope_sin,
    int64_t rope_seq_len,
    int64_t rope_head_dim,
    int64_t rope_rotary_dim,
    int config_id
) {
    check_rope_epilogue_args(D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim);
    switch (config_id) {
    case 0:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 1:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 2:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 3:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 4:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 5:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 6:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 7:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 8:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 9:  run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    case 10: run_gemm_with_config_rope<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(A, A_sc, B, B_sc, D, rope_cos, rope_sin, rope_seq_len, rope_head_dim, rope_rotary_dim); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

// ================================================================
// True Batched GEMM entrypoint
// D_out_list[i] = A_list[i] × B_list[i]^T, independently per batch.
// ================================================================
void mxfp4_batched_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;

        for (int i = 0; i < n; ++i) {
            int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = tensor_to_gl_tma_view<typename G::A_fp4x2_gl>(A_list[i], "A_list");
            auto a_sc_gl = tensor_to_gl_tma_view<typename G::A_sc_gl>(A_sc_list[i], "A_sc_list");
            auto b_gl = tensor_to_gl_tma_view<typename G::B_fp4x2_gl>(B_list[i], "B_list");
            auto b_sc_gl = tensor_to_gl_tma_view<typename G::B_sc_gl>(B_sc_list[i], "B_sc_list");
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    // For batched GEMM, use MMA_PER_TILE-friendly configs to avoid resource overflow.
    // DeepSeek expert hidden dims such as 1408 require Kb=128; Kb=256 would skip
    // the final 128-wide reduction tile.
    const int64_t K0 = 2 * A_list[0].size(1);
    if (N_out % 256 != 0 && N_out % 128 == 0) {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 128>>();
        }
    } else if (N_out <= 4096) {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>();
        }
    } else {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 128>>();
        }
    }
}

void mxfp4_grouped_gemm_strided_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int64_t num_batches,
    int64_t m_per_batch,
    int64_t n_per_batch,
    int64_t k_per_batch,
    int64_t a_row_stride,
    int64_t a_k_stride,
    int64_t b_row_stride,
    int64_t b_k_stride,
    int64_t d_row_stride,
    int config_id = -1
) {
    TORCH_CHECK(num_batches > 0, "num_batches must be positive");
    TORCH_CHECK(A.is_cuda() && A_sc.is_cuda() && B.is_cuda() && B_sc.is_cuda() && D.is_cuda(),
                "mxfp4_grouped_gemm_strided expects CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && A_sc.is_contiguous() && B.is_contiguous() && B_sc.is_contiguous() && D.is_contiguous(),
                "mxfp4_grouped_gemm_strided expects contiguous tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "mxfp4_grouped_gemm_strided expects flat 2D A/B/D tensors");
    TORCH_CHECK(m_per_batch > 0 && n_per_batch > 0 && k_per_batch > 0,
                "m/n/k per batch must be positive");
    TORCH_CHECK(D.size(1) == n_per_batch, "D second dim must equal n_per_batch");

    const int64_t M = m_per_batch;
    const int64_t N_out = n_per_batch;
    const int64_t K0 = k_per_batch;

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.uniform_strided = true;
        g_host.num_batches = (int)num_batches;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(K0 / C::Kb);
        g_host.total_spatial_tiles = 0;
        TORCH_CHECK(g_host.num_row_blocks > 0 && g_host.num_col_blocks > 0 && g_host.num_red_blocks > 0,
                    "mxfp4_grouped_gemm_strided expects positive tile counts");
        TORCH_CHECK(M % C::Mb == 0, "mxfp4_grouped_gemm_strided M must be a multiple of ", C::Mb);
        TORCH_CHECK(N_out % C::Nb == 0, "mxfp4_grouped_gemm_strided N must be a multiple of ", C::Nb);
        TORCH_CHECK(K0 % C::Kb == 0, "mxfp4_grouped_gemm_strided K must be a multiple of ", C::Kb);
        TORCH_CHECK(a_row_stride % 128 == 0 && b_row_stride % 128 == 0 && d_row_stride % 128 == 0,
                    "row strides must be multiples of 128");
        TORCH_CHECK(a_k_stride % C::Kb == 0 && b_k_stride % C::Kb == 0,
                    "K strides must be multiples of the selected K tile");
        g_host.a_row_block_stride = (int)(a_row_stride / 128);
        g_host.a_k_block_stride = (int)(a_k_stride / C::Kb);
        g_host.b_row_block_stride = (int)(b_row_stride / 128);
        g_host.b_k_block_stride = (int)(b_k_stride / C::Kb);
        g_host.d_row_block_stride = (int)(d_row_stride / 128);

        auto a_gl = tensor_to_gl_tma_view<typename G::A_fp4x2_gl>(A, "A");
        auto a_sc_gl = tensor_to_gl_tma_view<typename G::A_sc_gl>(A_sc, "A_sc");
        auto b_gl = tensor_to_gl_tma_view<typename G::B_fp4x2_gl>(B, "B");
        auto b_sc_gl = tensor_to_gl_tma_view<typename G::B_sc_gl>(B_sc, "B_sc");
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D);
        memcpy(&g_host.A_tma[0], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[0], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[0], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[0], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.D_tma[0], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    auto build_nb128_kb128 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, false, 128>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, true,  128>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  8, 2, true,  128>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 4, true,  128>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 2, false, 128>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided Nb=128 config_id: ", cfg, " (valid: 0-4)");
        }
    };
    auto build_nb128_kb256 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, false, 256>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  4, 2, true,  256>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8,  8, 2, true,  256>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 4, true,  256>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<128, 5,  8, 12, 2, false, 256>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided Nb=128 config_id: ", cfg, " (valid: 0-4)");
        }
    };
    auto build_nb256_kb128 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false, 128>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true,  128>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true,  128>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true,  128>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false, 128>>(); break;
        case 5: build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false, 128>>(); break;
        case 6: build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false, 128>>(); break;
        case 7: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false, 128>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided config_id: ", cfg, " (valid: 0-7)");
        }
    };
    auto build_nb256_kb256 = [&](int cfg) {
        switch (cfg) {
        case 0: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false, 256>>(); break;
        case 1: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true,  256>>(); break;
        case 2: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true,  256>>(); break;
        case 3: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true,  256>>(); break;
        case 4: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false, 256>>(); break;
        case 5: build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false, 256>>(); break;
        case 6: build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false, 256>>(); break;
        case 7: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false, 256>>(); break;
        default: TORCH_CHECK(false, "Invalid grouped strided config_id: ", cfg, " (valid: 0-7)");
        }
    };

    if (N_out % 256 != 0 && N_out % 128 == 0) {
        if (K0 % 256 == 0) {
            if (config_id >= 0) build_nb128_kb256((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 256>>();
        } else {
            if (config_id >= 0) build_nb128_kb128((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 128>>();
        }
    } else if (N_out <= 4096) {
        if (K0 % 256 == 0) {
            if (config_id >= 0) build_nb256_kb256((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>();
        } else {
            if (config_id >= 0) build_nb256_kb128((int)config_id);
            else build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>();
        }
    } else {
        if (K0 % 256 == 0) {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 256>>();
        } else {
            build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 128>>();
        }
    }
}

void mxfp4_batched_gemm_slices_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const std::vector<int64_t> &A_row_starts,
    const std::vector<int64_t> &A_k_starts,
    const std::vector<int64_t> &B_row_starts,
    const std::vector<int64_t> &B_k_starts,
    const std::vector<int64_t> &D_row_starts,
    const std::vector<int64_t> &D_col_starts,
    const std::vector<int64_t> &M_list,
    const std::vector<int64_t> &N_list,
    const std::vector<int64_t> &K_list,
    int config_id = -1
) {
    const int n = static_cast<int>(M_list.size());
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(
        A_row_starts.size() == M_list.size() &&
        A_k_starts.size() == M_list.size() &&
        B_row_starts.size() == M_list.size() &&
        B_k_starts.size() == M_list.size() &&
        D_row_starts.size() == M_list.size() &&
        D_col_starts.size() == M_list.size() &&
        N_list.size() == M_list.size() &&
        K_list.size() == M_list.size(),
        "all sliced batched GEMM metadata lists must have equal length");
    TORCH_CHECK(A.is_cuda() && A_sc.is_cuda() && B.is_cuda() && B_sc.is_cuda() && D.is_cuda(),
                "mxfp4_batched_gemm_slices expects CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "A, B, and D must be 2D");
    TORCH_CHECK(A_sc.dim() == 4 && B_sc.dim() == 4, "scale tensors must be 4D");
    TORCH_CHECK(D.scalar_type() == at::kBFloat16, "D must be bf16");
    TORCH_CHECK(D.stride(1) == 1, "D must have unit inner stride");

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M_list[0] / C::Mb);
        g_host.num_col_blocks = (int)(N_list[0] / C::Nb);
        g_host.num_red_blocks = (int)(K_list[0] / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;
        g_host.uniform_strided = false;

        for (int i = 0; i < n; ++i) {
            TORCH_CHECK(M_list[i] > 0 && N_list[i] > 0 && K_list[i] > 0,
                        "sliced batched GEMM dimensions must be positive");
            TORCH_CHECK(M_list[i] % C::Mb == 0, "D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(N_list[i] % C::Nb == 0, "D cols must be a multiple of ", C::Nb);
            TORCH_CHECK(K_list[i] % C::Kb == 0, "K must be a multiple of ", C::Kb);
            TORCH_CHECK(A_k_starts[i] % C::Kb == 0 && B_k_starts[i] % C::Kb == 0,
                        "K starts must align to selected K tile");
            TORCH_CHECK(D_row_starts[i] >= 0 && D_col_starts[i] >= 0,
                        "D starts must be non-negative");
            TORCH_CHECK(D_row_starts[i] + M_list[i] <= D.size(0), "D row slice exceeds tensor");
            TORCH_CHECK(D_col_starts[i] + N_list[i] <= D.stride(0), "D col slice exceeds leading stride");

            const int row_blocks = (int)(M_list[i] / C::Mb);
            const int col_blocks = (int)(N_list[i] / C::Nb);
            const int red_blocks = (int)(K_list[i] / C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = tensor_to_gl_tma_2d_slice<typename G::A_fp4x2_gl>(
                A, "A", A_row_starts[i], A_k_starts[i] / 2, M_list[i], K_list[i] / 2);
            auto a_sc_gl = tensor_to_gl_tma_scale_slice<typename G::A_sc_gl>(
                A_sc, "A_sc", A_row_starts[i], A_k_starts[i], M_list[i], K_list[i]);
            auto b_gl = tensor_to_gl_tma_2d_slice<typename G::B_fp4x2_gl>(
                B, "B", B_row_starts[i], B_k_starts[i] / 2, N_list[i], K_list[i] / 2);
            auto b_sc_gl = tensor_to_gl_tma_scale_slice<typename G::B_sc_gl>(
                B_sc, "B_sc", B_row_starts[i], B_k_starts[i], N_list[i], K_list[i]);
            auto d_gl = tensor_to_gl_tma_2d_slice<typename G::D_gl>(
                D, "D", D_row_starts[i], D_col_starts[i], M_list[i], N_list[i]);

            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    const int64_t N_out = N_list[0];
    const int64_t K0 = K_list[0];
    auto run_auto = [&]() {
        if (N_out % 256 != 0 && N_out % 128 == 0) {
            if (K0 % 256 == 0) {
                build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 256>>();
            } else {
                build_and_launch.template operator()<mxfp4_gemm::config<128, 5, 8, 4, 2, false, 128>>();
            }
        } else if (N_out <= 4096) {
            if (K0 % 256 == 0) {
                build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 256>>();
            } else {
                build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false, 128>>();
            }
        } else {
            if (K0 % 256 == 0) {
                build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 256>>();
            } else {
                build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false, 128>>();
            }
        }
    };

    if (config_id < 0) {
        run_auto();
        return;
    }
    switch (config_id) {
    case 0:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(); break;
    case 1:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(); break;
    case 2:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(); break;
    case 3:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(); break;
    case 4:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(); break;
    case 5:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(); break;
    case 6:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(); break;
    case 7:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(); break;
    case 8:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(); break;
    case 9:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-9)");
    }
}

void mxfp4_batched_gemm_config_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    int config_id
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;

        for (int i = 0; i < n; ++i) {
            int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm_config expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm_config D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm_config D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm_config K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = tensor_to_gl_tma_view<typename G::A_fp4x2_gl>(A_list[i], "A_list");
            auto a_sc_gl = tensor_to_gl_tma_view<typename G::A_sc_gl>(A_sc_list[i], "A_sc_list");
            auto b_gl = tensor_to_gl_tma_view<typename G::B_fp4x2_gl>(B_list[i], "B_list");
            auto b_sc_gl = tensor_to_gl_tma_view<typename G::B_sc_gl>(B_sc_list[i], "B_sc_list");
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    switch (config_id) {
    case 0:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(); break;
    case 1:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(); break;
    case 2:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(); break;
    case 3:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(); break;
    case 4:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(); break;
    case 5:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(); break;
    case 6:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(); break;
    case 7:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(); break;
    case 8:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(); break;
    case 9:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-9)");
    }
}

void mxfp4_batched_gemm_rope_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    const std::vector<at::Tensor> &rope_cos_list,
    const std::vector<at::Tensor> &rope_sin_list,
    const std::vector<int64_t> &rope_seq_len_list,
    const std::vector<int64_t> &rope_head_dim_list,
    const std::vector<int64_t> &rope_rotary_dim_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());
    TORCH_CHECK(n == (int)rope_cos_list.size());
    TORCH_CHECK(n == (int)rope_sin_list.size());
    TORCH_CHECK(n == (int)rope_seq_len_list.size());
    TORCH_CHECK(n == (int)rope_head_dim_list.size());
    TORCH_CHECK(n == (int)rope_rotary_dim_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;
        g_host.uniform_strided = false;

        for (int i = 0; i < n; ++i) {
            const int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            const int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            const int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm_rope expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm_rope D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm_rope D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm_rope K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            if (!rope_tensor_disabled(rope_cos_list[i]) && !rope_tensor_disabled(rope_sin_list[i])) {
                check_rope_epilogue_args(
                    D_out_list[i],
                    rope_cos_list[i],
                    rope_sin_list[i],
                    rope_seq_len_list[i],
                    rope_head_dim_list[i],
                    rope_rotary_dim_list[i]);
                g_host.rope[i].cos = rope_cos_list[i].data_ptr<float>();
                g_host.rope[i].sin = rope_sin_list[i].data_ptr<float>();
                g_host.rope[i].seq_len = static_cast<int>(rope_seq_len_list[i]);
                g_host.rope[i].head_dim = static_cast<int>(rope_head_dim_list[i]);
                g_host.rope[i].rotary_dim = static_cast<int>(rope_rotary_dim_list[i]);
            }
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    int forced_config = -1;
    if (const char* env = std::getenv("MXFP4_BATCHED_GEMM_ROPE_CONFIG_ID")) {
        forced_config = std::atoi(env);
    }
    if (forced_config >= 0) {
        switch (forced_config) {
        case 0:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(); break;
        case 1:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(); break;
        case 2:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(); break;
        case 3:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(); break;
        case 4:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(); break;
        case 5:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(); break;
        case 6:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(); break;
        case 7:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(); break;
        case 8:  build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(); break;
        case 9:  build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(); break;
        case 10: build_and_launch.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(); break;
        default: TORCH_CHECK(false, "Invalid MXFP4_BATCHED_GEMM_ROPE_CONFIG_ID: ", forced_config);
        }
        return;
    }
    const int64_t K = 2 * A_list[0].size(1);
    if (
        n == 3
        && M >= 32768
        && K == 2048
        && D_out_list[0].size(1) == 3072
        && D_out_list[1].size(1) == 512
        && D_out_list[2].size(1) == 256
        && rope_head_dim_list[0] == 192
        && rope_rotary_dim_list[0] == 64
        && rope_head_dim_list[2] == 64
        && rope_rotary_dim_list[2] == 64
    ) {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 12, 2, false>>();
    } else if (N_out <= 4096) {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false>>();
    }
}

void mxfp4_batched_gemm_rope_live64_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    const std::vector<at::Tensor> &rope_cs_list,
    const std::vector<int64_t> &rope_seq_len_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());
    TORCH_CHECK(n == (int)rope_cs_list.size());
    TORCH_CHECK(n == (int)rope_seq_len_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;
        g_host.uniform_strided = false;

        for (int i = 0; i < n; ++i) {
            const int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            const int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            const int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm_rope_live64 expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm_rope_live64 D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm_rope_live64 D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm_rope_live64 K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            if (!rope_tensor_disabled(rope_cs_list[i])) {
                check_rope_live64_args(D_out_list[i], rope_cs_list[i], rope_seq_len_list[i]);
                g_host.rope_live64[i].cs = reinterpret_cast<const float2*>(rope_cs_list[i].data_ptr<float>());
                g_host.rope_live64[i].seq_len = static_cast<int>(rope_seq_len_list[i]);
                g_host.rope_live64[i].seq_mask = static_cast<int>(rope_seq_len_list[i] - 1);
            }
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    const bool apply_rht32 = use_rope_live64_rht32();
    auto build_selected = [&]<typename Base>() {
        check_rope_live64_rht32_config<Base>(apply_rht32);
        if (apply_rht32) {
            build_and_launch.template operator()<rope_live64_rht32_config<Base>>();
        } else {
            build_and_launch.template operator()<Base>();
        }
    };

    if (N_out <= 4096) {
        build_selected.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_selected.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false>>();
    }
}

void mxfp4_batched_gemm_rope_live64_config_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    std::vector<at::Tensor> &D_out_list,
    const std::vector<at::Tensor> &rope_cs_list,
    const std::vector<int64_t> &rope_seq_len_list,
    int config_id
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= mxfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", mxfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)D_out_list.size());
    TORCH_CHECK(n == (int)rope_cs_list.size());
    TORCH_CHECK(n == (int)rope_seq_len_list.size());

    const int64_t M = D_out_list[0].size(0);
    const int64_t N_out = D_out_list[0].size(1);

    auto build_and_launch = [&]<typename C>() {
        using G = mxfp4_batched_gemm::globals<C>;
        G g_host {};
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);
        g_host.tile_offsets[0] = 0;
        g_host.total_spatial_tiles = 0;
        g_host.uniform_strided = false;

        for (int i = 0; i < n; ++i) {
            const int row_blocks = (int)(D_out_list[i].size(0) / C::Mb);
            const int col_blocks = (int)(D_out_list[i].size(1) / C::Nb);
            const int red_blocks = (int)(2 * A_list[i].size(1) / C::Kb);
            TORCH_CHECK(row_blocks > 0 && col_blocks > 0 && red_blocks > 0,
                        "mxfp4_batched_gemm_rope_live64_config expects positive tile counts");
            TORCH_CHECK(D_out_list[i].size(0) % C::Mb == 0,
                        "mxfp4_batched_gemm_rope_live64_config D rows must be a multiple of ", C::Mb);
            TORCH_CHECK(D_out_list[i].size(1) % C::Nb == 0,
                        "mxfp4_batched_gemm_rope_live64_config D cols must be a multiple of ", C::Nb);
            TORCH_CHECK((2 * A_list[i].size(1)) % C::Kb == 0,
                        "mxfp4_batched_gemm_rope_live64_config K must be a multiple of ", C::Kb);
            g_host.num_row_blocks_by_batch[i] = row_blocks;
            g_host.num_col_blocks_by_batch[i] = col_blocks;
            g_host.num_red_blocks_by_batch[i] = red_blocks;
            g_host.total_spatial_tiles += row_blocks * col_blocks;
            g_host.tile_offsets[i + 1] = g_host.total_spatial_tiles;

            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc_list[i]);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc_list[i]);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            if (!rope_tensor_disabled(rope_cs_list[i])) {
                check_rope_live64_args(D_out_list[i], rope_cs_list[i], rope_seq_len_list[i]);
                g_host.rope_live64[i].cs = reinterpret_cast<const float2*>(rope_cs_list[i].data_ptr<float>());
                g_host.rope_live64[i].seq_len = static_cast<int>(rope_seq_len_list[i]);
                g_host.rope_live64[i].seq_mask = static_cast<int>(rope_seq_len_list[i] - 1);
            }
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    const bool apply_rht32 = use_rope_live64_rht32();
    auto build_selected = [&]<typename Base>() {
        check_rope_live64_rht32_config<Base>(apply_rht32);
        if (apply_rht32) {
            build_and_launch.template operator()<rope_live64_rht32_config<Base>>();
        } else {
            build_and_launch.template operator()<Base>();
        }
    };

    switch (config_id) {
    case 0:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(); break;
    case 1:  build_selected.template operator()<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(); break;
    case 2:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(); break;
    case 3:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(); break;
    case 4:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(); break;
    case 5:  build_selected.template operator()<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(); break;
    case 6:  build_selected.template operator()<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(); break;
    case 7:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(); break;
    case 8:  build_selected.template operator()<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(); break;
    case 9:  build_selected.template operator()<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(); break;
    case 10: build_selected.template operator()<mxfp4_gemm::config<256, 5,  4, 12, 2, false>>(); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-10)");
    }
}

void mxfp4_split2_dgrad_strided_onepass_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_mxfp4_split2_dgrad_inputs(
        A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list);
    TORCH_CHECK(B_list.size() == 2, "split2 one-pass dgrad expects exactly 2 B tensors");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_mxfp4_split2_dgrad_gemm_strided_onepass(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        static_cast<int>(config_idx));
}

void mxfp4_split3_dgrad_strided_onepass_gemm_entrypoint(
    const at::Tensor& A_full,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<int64_t>& A_col_offsets,
    const std::vector<int64_t>& A_col_widths,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    at::Tensor& D_out,
    int64_t config_idx
) {
    check_mxfp4_split3_dgrad_inputs(
        A_full, A_sc_list, A_col_offsets, A_col_widths, B_list, B_sc_list);
    TORCH_CHECK(B_list.size() == 3, "split3 one-pass dgrad expects exactly 3 B tensors");
    check_output_matrix(D_out, "D_out", A_full.size(0), B_list[0].size(0));
    launch_mxfp4_split3_dgrad_gemm_strided_onepass(
        A_full,
        A_sc_list,
        A_col_offsets,
        A_col_widths,
        B_list,
        B_sc_list,
        D_out,
        static_cast<int>(config_idx));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_gemm", &mxfp4_gemm_entrypoint);
    m.def("mxfp4_gemm_scaled", &mxfp4_gemm_scaled_entrypoint,
          "MXFP4 GEMM with an extra CUDA scalar epilogue multiplier",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("output_scale"));
    m.def("mxfp4_gemm_residual", &mxfp4_gemm_residual_entrypoint,
          "Dense GEMM with fused bf16 residual add in the epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("R"), pybind11::arg("D"));
    m.def("mxfp4_gemm_residual_config", &mxfp4_gemm_residual_config_entrypoint,
          "Dense GEMM with fused bf16 residual add and explicit kernel config",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("R"), pybind11::arg("D"),
          pybind11::arg("config_id"));
    m.def("mxfp4_gemm_k128", &mxfp4_gemm_k128_entrypoint);
    m.def("mxfp4_gemm_masked", &mxfp4_gemm_masked_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"), pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_masked_k128", &mxfp4_gemm_masked_k128_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"), pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_config", &mxfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("mxfp4_gemm_rope", &mxfp4_gemm_rope_entrypoint,
          "GEMM with a RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cos"), pybind11::arg("rope_sin"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("rope_head_dim"),
          pybind11::arg("rope_rotary_dim"));
    m.def("mxfp4_gemm_rope_live64", &mxfp4_gemm_rope_live64_entrypoint,
          "GEMM with an exact-shape live64 RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"));
    m.def("mxfp4_gemm_rope_live64_config", &mxfp4_gemm_rope_live64_config_entrypoint,
          "GEMM with selectable tile config and an exact-shape live64 RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cs"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("config_id"));
    m.def("mxfp4_gemm_rope_config", &mxfp4_gemm_rope_config_entrypoint,
          "GEMM with selectable tile config and a RoPE epilogue",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("rope_cos"), pybind11::arg("rope_sin"),
          pybind11::arg("rope_seq_len"),
          pybind11::arg("rope_head_dim"),
          pybind11::arg("rope_rotary_dim"),
          pybind11::arg("config_id"));
    m.def("mxfp4_deepseek_mla_inverse_rope_pack_grad",
          &mxfp4_deepseek_mla_inverse_rope_pack_grad_entrypoint,
          "DeepSeek MLA backward helper: inverse RoPE q/k_pe grads and pack padded dY",
          pybind11::arg("grad_q"),
          pybind11::arg("grad_kv"),
          pybind11::arg("grad_kpe"),
          pybind11::arg("rope_cos"),
          pybind11::arg("rope_sin"),
          pybind11::arg("out"),
          pybind11::arg("seq_len"),
          pybind11::arg("n_heads"),
          pybind11::arg("qk_head_dim"),
          pybind11::arg("rope_dim"),
          pybind11::arg("kv_lora_rank"),
          pybind11::arg("kv_pad_dim"),
          pybind11::arg("kpe_pad_dim"));

    m.def("mxfp4_batched_gemm", &mxfp4_batched_gemm_entrypoint,
          "True Batched GEMM: D_i = A_i × B_i^T, independently per batch",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"));
    m.def("mxfp4_batched_gemm_slices", &mxfp4_batched_gemm_slices_entrypoint,
          "True Batched GEMM from bulk tensors plus per-batch row/K/output slices",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("A_row_starts"), pybind11::arg("A_k_starts"),
          pybind11::arg("B_row_starts"), pybind11::arg("B_k_starts"),
          pybind11::arg("D_row_starts"), pybind11::arg("D_col_starts"),
          pybind11::arg("M_list"), pybind11::arg("N_list"), pybind11::arg("K_list"),
          pybind11::arg("config_id") = -1);
    m.def("mxfp4_grouped_gemm_strided", &mxfp4_grouped_gemm_strided_entrypoint,
          "Uniform grouped GEMM over flat packed tensors using one TMA descriptor per operand",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"),
          pybind11::arg("num_batches"),
          pybind11::arg("m_per_batch"),
          pybind11::arg("n_per_batch"),
          pybind11::arg("k_per_batch"),
          pybind11::arg("a_row_stride"),
          pybind11::arg("a_k_stride"),
          pybind11::arg("b_row_stride"),
          pybind11::arg("b_k_stride"),
          pybind11::arg("d_row_stride"),
          pybind11::arg("config_id") = -1);
    m.def("mxfp4_batched_gemm_config", &mxfp4_batched_gemm_config_entrypoint,
          "True Batched GEMM with selectable tile config",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"), pybind11::arg("config_id"));
    m.def("mxfp4_batched_gemm_rope", &mxfp4_batched_gemm_rope_entrypoint,
          "True Batched GEMM with an optional per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cos_list"), pybind11::arg("rope_sin_list"),
          pybind11::arg("rope_seq_len_list"),
          pybind11::arg("rope_head_dim_list"),
          pybind11::arg("rope_rotary_dim_list"));
    m.def("mxfp4_batched_gemm_rope_live64", &mxfp4_batched_gemm_rope_live64_entrypoint,
          "True Batched GEMM with an exact-shape live64 per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cs_list"),
          pybind11::arg("rope_seq_len_list"));
    m.def("mxfp4_batched_gemm_rope_live64_config", &mxfp4_batched_gemm_rope_live64_config_entrypoint,
          "True Batched GEMM with selectable tile config and an exact-shape live64 per-batch RoPE epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"),
          pybind11::arg("rope_cs_list"),
          pybind11::arg("rope_seq_len_list"),
          pybind11::arg("config_id"));
    m.def("mxfp4_split2_dgrad_strided_onepass_gemm",
          &mxfp4_split2_dgrad_strided_onepass_gemm_entrypoint,
          "MXFP4 split2 one-pass dgrad GEMM with strided row slices",
          pybind11::arg("A_full"),
          pybind11::arg("A_sc_list"),
          pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"),
          pybind11::arg("D_out"),
          pybind11::arg("config_idx") = -1);
    m.def("mxfp4_split3_dgrad_strided_onepass_gemm",
          &mxfp4_split3_dgrad_strided_onepass_gemm_entrypoint,
          "MXFP4 split3 one-pass dgrad GEMM with strided row slices",
          pybind11::arg("A_full"),
          pybind11::arg("A_sc_list"),
          pybind11::arg("A_col_offsets"),
          pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"),
          pybind11::arg("D_out"),
          pybind11::arg("config_idx") = -1);
}

#endif
