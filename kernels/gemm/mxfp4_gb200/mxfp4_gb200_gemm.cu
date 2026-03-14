// ================================================================
// MXFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "mxfp4_gemm.cuh"
#include "mxfp4_quantize.cuh"
#include "mxfp4_batched_gemm.cuh"
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

void mxfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    int K = A.size(1) * 2;
    int N_out = D.size(1);
    if (K <= 2048 && N_out <= 4096) {
        using C = mxfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = mxfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
        };
        kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
    } else if (K <= 2048) {
        using C = mxfp4_gemm::config<256, 4, 16, 4, 2, false>;
        using G = mxfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
        };
        kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
    } else {
        using C = mxfp4_gemm::config<256, 4, 16, 12, 4, false>;
        using G = mxfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
        };
        kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
    }
}

void mxfp4_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp4x2,
    at::Tensor &A_sc
) {
    using C = mxfp4_quantize::config;
    using G = mxfp4_quantize::globals;

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(A_bf16),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl>(A_sc)
    };
    kittens::py::launch_kernel<C, G, mxfp4_quantize::kernel>(g);
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
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
    };
    kittens::py::launch_kernel<C, G, mxfp4_gemm::kernel<C>>(g);
}

void mxfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &D, int config_id
) {
    //                     Nb   LOAD EPI  SG  DT  OVERLAP
    switch (config_id) {
    // ── Original 10 ──
    case 0:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  1, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 1:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 2:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 3:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 4:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 5:  run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 6:  run_gemm_with_config<mxfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 7:  run_gemm_with_config<mxfp4_gemm::config<128, 5,  4, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 8:  run_gemm_with_config<mxfp4_gemm::config<128, 4,  4, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 9:  run_gemm_with_config<mxfp4_gemm::config<256, 4, 16, 12, 4, false>>(A, A_sc, B, B_sc, D); break;

    // ── Nb=256 LP=5 EP=8 SG sweep (best family) ──
    case 10: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  1, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 11: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  2, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 12: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  8, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 13: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 14: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  1, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 15: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  2, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 16: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, B, B_sc, D); break;

    // ── Nb=256 LP=5 EP=16 SG sweep ──
    case 17: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 18: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 19: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 20: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 21: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  1, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 22: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  8, 2, true >>(A, A_sc, B, B_sc, D); break;

    // ── Nb=256 LP=4 EP=8 SG sweep ──
    case 23: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8,  1, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 24: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 25: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8,  2, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 26: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8,  8, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 27: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 28: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8, 12, 2, true >>(A, A_sc, B, B_sc, D); break;

    // ── Nb=256 LP=3 (shallower pipeline) ──
    case 29: run_gemm_with_config<mxfp4_gemm::config<256, 3,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 30: run_gemm_with_config<mxfp4_gemm::config<256, 3, 16,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 31: run_gemm_with_config<mxfp4_gemm::config<256, 3,  8, 12, 2, true >>(A, A_sc, B, B_sc, D); break;

    // ── DT=4 variants (more output tiles) ──
    case 32: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8,  4, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 33: run_gemm_with_config<mxfp4_gemm::config<256, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D); break;
    case 34: run_gemm_with_config<mxfp4_gemm::config<256, 4,  8, 12, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 35: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16,  4, 4, true >>(A, A_sc, B, B_sc, D); break;
    case 36: run_gemm_with_config<mxfp4_gemm::config<256, 5, 16, 12, 4, true >>(A, A_sc, B, B_sc, D); break;

    // ── Nb=128 EP=8 ──
    case 37: run_gemm_with_config<mxfp4_gemm::config<128, 5,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 38: run_gemm_with_config<mxfp4_gemm::config<128, 5,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 39: run_gemm_with_config<mxfp4_gemm::config<128, 5,  8,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 40: run_gemm_with_config<mxfp4_gemm::config<128, 5,  8, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 41: run_gemm_with_config<mxfp4_gemm::config<128, 4,  8,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 42: run_gemm_with_config<mxfp4_gemm::config<128, 4,  8, 12, 2, false>>(A, A_sc, B, B_sc, D); break;

    // ── Nb=128 EP=4 more SG ──
    case 43: run_gemm_with_config<mxfp4_gemm::config<128, 5,  4,  1, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 44: run_gemm_with_config<mxfp4_gemm::config<128, 5,  4,  2, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 45: run_gemm_with_config<mxfp4_gemm::config<128, 5,  4,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 46: run_gemm_with_config<mxfp4_gemm::config<128, 5,  4,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 47: run_gemm_with_config<mxfp4_gemm::config<128, 4,  4,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 48: run_gemm_with_config<mxfp4_gemm::config<128, 4,  4,  1, 2, false>>(A, A_sc, B, B_sc, D); break;

    // ── Nb=256 LP=4 EP=16 OVL=T ──
    case 49: run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  1, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 50: run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 51: run_gemm_with_config<mxfp4_gemm::config<256, 4, 16, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 52: run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  2, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 53: run_gemm_with_config<mxfp4_gemm::config<256, 4, 16,  8, 2, false>>(A, A_sc, B, B_sc, D); break;

    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-53)");
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_gemm", &mxfp4_gemm_entrypoint);
    m.def("mxfp4_gemm_config", &mxfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("mxfp4_quantize", &mxfp4_quantize_entrypoint);
    m.def("mxfp4_batched_gemm", &mxfp4_batched_gemm_entrypoint,
          "True Batched GEMM: D_i = A_i × B_i^T, independently per batch",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"),
          pybind11::arg("D_out_list"));
}

#endif
