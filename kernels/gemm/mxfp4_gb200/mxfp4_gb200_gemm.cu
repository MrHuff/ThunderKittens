// ================================================================
// MXFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "mxfp4_gemm.cuh"
// mxfp4_quantize.cuh removed — use standalone mxfp4_v2 quantizer
#include "mxfp4_batched_gemm.cuh"
#include "mxfp4_split2_accum_gemm.cuh"
#include <cstdint>
#include <cstring>
#include <optional>

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

} // namespace

void mxfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    int K = A.size(1) * 2;
    int N_out = D.size(1);
    // Single config that works for all shapes with Kb=256.
    // config<256,5,8,4,2,false> = Nb=256, LOAD_PIPE=5, EPI=8, SG=4, DT=2, no overlap
    using C = mxfp4_gemm::config<256, 5, 8, 4, 2, false>;
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

    using C = mxfp4_gemm::config<256, 5, 8, 4, 2, false>;
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

void mxfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D, int config_id
) {
    //                     Nb   LOAD EPI  SG  DT  OVERLAP
    // Trimmed to 10 configs for faster compile with Kb=256
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

    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-9)");
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

        for (int i = 0; i < n; ++i) {
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
        }
        kittens::py::launch_kernel<C, G, mxfp4_batched_gemm::kernel<C>>(g_host);
    };

    // For batched GEMM, use MMA_PER_TILE-friendly configs to avoid resource overflow
    if (N_out <= 4096) {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.template operator()<mxfp4_gemm::config<256, 4, 16, 4, 2, false>>();
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_gemm", &mxfp4_gemm_entrypoint);
    m.def("mxfp4_gemm_masked", &mxfp4_gemm_masked_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("tilemask"), pybind11::arg("tilemask_transposed"),
          pybind11::arg("D"));
    m.def("mxfp4_gemm_config", &mxfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("config_id"));

    m.def("mxfp4_batched_gemm", &mxfp4_batched_gemm_entrypoint,
          "True Batched GEMM: D_i = A_i × B_i^T, independently per batch",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"));
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
}

#endif
