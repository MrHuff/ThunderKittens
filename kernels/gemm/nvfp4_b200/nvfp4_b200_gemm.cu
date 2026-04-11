// ================================================================
// NVFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "nvfp4_gemm.cuh"
#include "nvfp4_quantize.cuh"
#include "nvfp4_batched_accum_gemm.cuh"
#include "nvfp4_accum_gemm.cuh"
#include "nvfp4_fused_gemm.cuh"
#include "nvfp4_persistent_gemm.cuh"
#include <optional>

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = nvfp4_gemm::globals<C>;

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
    std::vector<__nv_fp8_e4m3*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B_sc(arg_group_count);
    std::vector<float*> d_A_sc_global(arg_group_count);
    std::vector<float*> d_B_sc_global(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp4x2_e2m1)/2);
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e4m3)/16);
        cudaMalloc(&d_A_sc_global[i], sizeof(float));
        cudaMalloc(&d_B_sc_global[i], sizeof(float));
        cudaMalloc(&d_D[i], M * N * sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M * N * sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, seed + i * 100, 0.0f, 255.0f);
        fill<uint8_t, FillMode::RANDOM>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, seed + i * 100 + 1, 0.0f, 255.0f);    
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A_sc[i], M*K/16, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B_sc[i], N*K/16, seed + i*100 + 3, 0.1f, 10.0f);
        fill<float, FillMode::RANDOM>(d_A_sc_global[i], 1, seed + i * 100 + 4, 0.1f, 10.0f);
        fill<float, FillMode::RANDOM>(d_B_sc_global[i], 1, seed + i * 100 + 5, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_nvfp4_gemm<__nv_bfloat16>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], d_A_sc_global[0], d_B_sc_global[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    // Note: The kernel expects scales as half, but we store fp8e4m3. Reinterpret the pointers.
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_fp4x2_gl Ag{d_A[i], nullptr, nullptr, M, K/2};
        typename G::A_sc_gl Asg{reinterpret_cast<half*>(d_A_sc[i]), nullptr, M/128, K/64, nullptr};
        typename G::A_sc_global_gl Asgg{d_A_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::B_fp4x2_gl Bg{d_B[i], nullptr, nullptr, N, K/2};
        typename G::B_sc_gl Bsg{reinterpret_cast<half*>(d_B_sc[i]), nullptr, N/128, K/64, nullptr};
        typename G::B_sc_global_gl Bsgg{d_B_sc_global[i], nullptr, nullptr, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Asgg, Bg, Bsg, Bsgg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));
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
        cudaFree(d_A_sc[i]);
        cudaFree(d_A_sc_global[i]);
        cudaFree(d_B[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_B_sc_global[i]);
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
    run_benchmark<nvfp4_gemm::config<128, 5, 4, 12, 2, true>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<nvfp4_gemm::config<256, 5, 8, 4, 2, true>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<nvfp4_gemm::config<256, 4, 16, 1, 2, false>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<nvfp4_gemm::config<256, 4, 16, 12, 2, false>>(N, N, N, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"
#include "ATen/Functions.h"

void nvfp4_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    int K = B.size(1) * 2;
    int N_out = D.size(1);
    if (K <= 2048 && N_out <= 4096) {
        // Dgrad + small-N shapes: sweep-optimized config
        // Nb=256, LOAD_PIPE=5, EPI_PIPE=8, SG=4, OVL=false
        // 1.33x faster than Nb=128 on Wo dgrad, 1.49x on small-M dgrad
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = 0,
            .k_dim = 0,
            .use_split_D = false,
            .b_sg_per_tile = nullptr,
            .silu_dim = 0
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else if (K <= 2048) {
        // Sweep-optimized: Nb=512 beats Nb=1024 by 22% for large-N (e.g. N=6144)
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = 0,
            .k_dim = 0,
            .use_split_D = false,
            .b_sg_per_tile = nullptr
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else {
        using C = nvfp4_gemm::config<256, 4, 8, 12, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = 0,
            .k_dim = 0,
            .use_split_D = false,
            .b_sg_per_tile = nullptr
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    }
}

// ================================================================
// Non-PDL standard GEMM: USE_PDL=false, CLUSTER_SIZE=1.
// Safe for CUDA graph capture and replay.
// Regular nvfp4_gemm uses PDL + CLUSTER_SIZE=2 which do not replay
// correctly inside CUDA graphs.
// ================================================================
void nvfp4_gemm_nopdl_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    // USE_PDL=false (8th arg), CLUSTER_SIZE=2 (9th arg, default — kernel requires cluster pairs)
    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .use_split_D = false,
        .b_sg_per_tile = nullptr,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

// Grouped GEMM: concatenated weights with per-tile B_sc_global
// B_sg_per_tile: [num_col_tiles] float, pre-computed on GPU by Python.
//   Each entry has the B_sg value for that column tile's group.
void nvfp4_grouped_gemm_entrypoint(
    const at::Tensor &A,              // [M, K/2] fp4
    const at::Tensor &A_sc,           // [M/16, K/16] fp8
    const at::Tensor &A_sc_global,    // [1] float
    const at::Tensor &B,              // [N_total, K/2] fp4 (concatenated weights)
    const at::Tensor &B_sc,           // [N_total/16, K/16] fp8
    const at::Tensor &B_sg_per_tile,  // [num_col_tiles] float — pre-computed per-tile B_sg (on GPU)
    at::Tensor &D,                    // [M, N_total] or [M, Nq] bf16
    std::optional<at::Tensor> D_K_opt = std::nullopt, // Optional K output
    std::optional<at::Tensor> D_V_opt = std::nullopt, // Optional V output
    int silu_dim = 0                  // Apply SiLU to output columns [0, silu_dim). 0 = disabled.
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }
    bool use_split_D = (D_K_opt.has_value() && D_V_opt.has_value());

    int K = B.size(1) * 2;
    if (K <= 2048) {
        // Sweep-optimized: Nb=512 beats Nb=1024 by 22% for QKV fwd (K=2048, N=6144)
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
            .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
            .use_split_D = use_split_D,
            .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
            .silu_dim = silu_dim
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    } else {
        // Sweep-optimized: SG=4 OVL=false beats SG=12 OVL=true by 5-6% for wgrad at M=32K-65K
        using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false>;
        using G = nvfp4_gemm::globals<C>;
        G g {
            .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
            .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
            .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
            .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
            .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
            .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
            .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .D_V = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
            .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
            .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
            .use_split_D = use_split_D,
            .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
            .silu_dim = silu_dim
        };
        kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
    }
}


// ================================================================
// Non-PDL Grouped GEMM: same as above but with USE_PDL=false.
// Safe for multi-stream (side stream) and CUDA graph capture.
// PDL (Programmatic Dependent Launch) requires per-stream arrive/wait
// which deadlocks on side streams without prior PDL kernel launches.
// ================================================================
void nvfp4_grouped_gemm_nopdl_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sc_global,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_per_tile,
    at::Tensor &D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    static thread_local at::Tensor dummy_bsg;
    if (!dummy_bsg.defined()) {
        dummy_bsg = at::zeros({1}, at::dtype(at::kFloat).device(at::kCUDA));
    }
    bool use_split_D = (D_K_opt.has_value() && D_V_opt.has_value());

    // USE_PDL=false, CLUSTER_SIZE=2 (default) — safe for multi-stream
    // USE_PDL=false (8th arg), CLUSTER_SIZE=2 (9th arg — kernel requires cluster pairs)
    using C = nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>;
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(dummy_bsg),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .use_split_D = use_split_D,
        .b_sg_per_tile = B_sg_per_tile.data_ptr<float>(),
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}


void nvfp4_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp4x2,
    at::Tensor &A_sc,
    at::Tensor &A_sc_global,
    bool scale_2d
) {
    using C = nvfp4_quantize::quantize_config;
    using G = nvfp4_quantize::globals;

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<G::A_bf16_gl>(A_bf16),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
        .A_sc = kittens::py::tensor_to_gl<G::A_sc_gl, false>(A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<G::A_sc_global_gl>(A_sc_global)
    };

    // MUST use PyTorch's current stream — bare <<<>>> uses default stream 0
    // which races with PyTorch ops on the current stream, causing NaN Heisenbug.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nvfp4_quantize::zero_kernel<<<1, 1, 0, stream>>>(g);
    nvfp4_quantize::absmax_kernel<<<nvfp4_quantize::absmax_config::NUM_BLOCKS, nvfp4_quantize::absmax_config::NUM_THREADS, 0, stream>>>(g);
    nvfp4_quantize::divide_kernel<<<1, 1, 0, stream>>>(g);
    if (scale_2d) kittens::py::launch_kernel<C, G, nvfp4_quantize::quantize_kernel<true>>(g);
    else          kittens::py::launch_kernel<C, G, nvfp4_quantize::quantize_kernel<false>>(g);

    // Fixup FP8 E4M3 NaN in scale tensor: overflow during quantization can produce
    // NaN bit patterns (0x7F) which poison downstream MMA operations.
    {
        int64_t sc_numel = A_sc.numel();
        int threads = 256;
        int blocks = ((sc_numel / 4) + threads - 1) / threads;
        fp8_nan_fixup_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<uint8_t*>(A_sc.data_ptr()), sc_numel);
    }
}

at::Tensor fp32_to_fp4x2_entrypoint(at::Tensor A_fp32) {
    using C = nvfp4_utils::config;
    using G = nvfp4_utils::globals;

    auto options = A_fp32.options().dtype(at::kFloat4_e2m1fn_x2).requires_grad(false);
    at::Tensor A_fp4x2 = at::empty({A_fp32.size(0), A_fp32.size(1) / 2}, options);

    G g {
        .A_fp32 = kittens::py::tensor_to_gl<G::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<C, G, nvfp4_utils::fp32_to_fp4x2_kernel>(g);

    return A_fp4x2;
}

at::Tensor fp4x2_to_fp32_entrypoint(at::Tensor A_fp4x2) {
    using C = nvfp4_utils::config;
    using G = nvfp4_utils::globals;

    auto options = A_fp4x2.options().dtype(at::kFloat).requires_grad(false);
    at::Tensor A_fp32 = at::empty({A_fp4x2.size(0), A_fp4x2.size(1) * 2}, options);

    G g {
        .A_fp32 = kittens::py::tensor_to_gl<G::A_fp32_gl>(A_fp32),
        .A_fp4x2 = kittens::py::tensor_to_gl<G::A_fp4x2_gl>(A_fp4x2),
    };
    kittens::py::launch_kernel<C, G, nvfp4_utils::fp4x2_to_fp32_kernel>(g);

    return A_fp32;
}

// ================================================================
// Config-selectable GEMM for tile tuning sweeps.
// config_id selects from pre-compiled configs below.
// ================================================================
template <typename C>
static void run_gemm_with_config(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    using G = nvfp4_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0, .k_dim = 0, .use_split_D = false, .b_sg_per_tile = nullptr, .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

#define NVFP4_GEMM_CONFIG_CASES(X) \
    X(0,  256, 4, 16,  1, 2, false) \
    X(1,  256, 4, 16,  4, 2, false) \
    X(2,  256, 4, 16, 12, 2, false) \
    X(3,  256, 5,  8,  4, 2, true ) \
    X(4,  256, 5,  8, 12, 2, true ) \
    X(5,  256, 5,  8,  4, 2, false) \
    X(6,  256, 4,  8, 12, 2, false) \
    X(7,  128, 5,  4, 12, 2, true ) \
    X(8,  128, 4,  4, 12, 2, false) \
    X(9,  128, 5,  4,  4, 2, true ) \
    X(10, 256, 5, 16,  4, 2, true ) \
    X(11, 256, 5, 16, 12, 2, true ) \
    X(12, 256, 5,  8,  1, 2, false) \
    X(13, 256, 5,  8,  2, 2, false) \
    X(14, 256, 5,  8,  8, 2, false) \
    X(15, 256, 5,  8, 12, 2, false) \
    X(16, 256, 5,  8,  1, 2, true ) \
    X(17, 256, 5,  8,  2, 2, true ) \
    X(18, 256, 5,  8,  8, 2, true ) \
    X(19, 256, 5, 16,  1, 2, false) \
    X(20, 256, 5, 16,  2, 2, false) \
    X(21, 256, 5, 16,  8, 2, false) \
    X(22, 256, 5, 16,  1, 2, true ) \
    X(23, 256, 5, 16,  2, 2, true ) \
    X(24, 256, 5, 16,  8, 2, true ) \
    X(25, 256, 4,  8,  1, 2, false) \
    X(26, 256, 4,  8,  4, 2, false) \
    X(27, 256, 4,  8,  2, 2, false) \
    X(28, 256, 4,  8,  8, 2, false) \
    X(29, 256, 3,  8,  4, 2, false) \
    X(30, 256, 3, 16,  4, 2, false) \
    X(31, 256, 4, 16,  2, 2, false) \
    X(32, 256, 4, 16,  8, 2, false) \
    X(33, 256, 4, 16,  1, 2, true ) \
    X(34, 256, 4, 16,  4, 2, true ) \
    X(35, 128, 5,  8,  4, 2, false) \
    X(36, 128, 5,  8, 12, 2, false) \
    X(37, 128, 5,  8,  1, 2, false) \
    X(38, 128, 5,  8,  4, 2, true ) \
    X(39, 128, 5,  8, 12, 2, true ) \
    X(40, 128, 4,  8,  4, 2, false) \
    X(41, 128, 4,  8, 12, 2, false) \
    X(42, 128, 5,  4,  1, 2, false) \
    X(43, 128, 5,  4,  2, 2, false) \
    X(44, 128, 5,  4,  1, 2, true ) \
    X(45, 128, 4,  4,  4, 2, false) \
    X(46, 128, 4,  4,  1, 2, false)

#define NVFP4_GEMM_DISPATCH_CASE(ID, NB, LP, EP, SG, DT, OVERLAP) \
    case ID: run_gemm_with_config<nvfp4_gemm::config<NB, LP, EP, SG, DT, OVERLAP>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

#define NVFP4_GEMM_DISPATCH_NOPDL_CASE(ID, NB, LP, EP, SG, DT, OVERLAP) \
    case ID: run_gemm_with_config<nvfp4_gemm::config<NB, LP, EP, SG, DT, OVERLAP, 256, false, 2>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

void nvfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D, int config_id
) {
    switch (config_id) {
        NVFP4_GEMM_CONFIG_CASES(NVFP4_GEMM_DISPATCH_CASE)
        // NOTE: Nb=128 + EP=16 is INVALID (D_tile cols = 128/16 = 8 < 16 minimum)
        default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-46)");
    }
}

void nvfp4_gemm_config_nopdl_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D, int config_id
) {
    switch (config_id) {
        NVFP4_GEMM_CONFIG_CASES(NVFP4_GEMM_DISPATCH_NOPDL_CASE)
        default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-46)");
    }
}

#undef NVFP4_GEMM_DISPATCH_NOPDL_CASE
#undef NVFP4_GEMM_DISPATCH_CASE
#undef NVFP4_GEMM_CONFIG_CASES

// ================================================================
// Batched GEMM entrypoint (z-dim parallel): D_i = A_i × B_i^T
// Each batch writes to a separate output buffer.
// ================================================================
void nvfp4_batched_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)D_list.size());

    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int K_first = (int)(A_list[0].size(1) * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        for (int i = 0; i < n; ++i) {
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
    };

    if (K_first <= 2048 && N_out <= 4096) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else if (K_first <= 2048) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    }
}

// ================================================================
// Accumulating Batched GEMM: all batches accumulate into a single D_out.
// Batch 0 stores via TMA, batches 1+ load-add-store with per-tile
// semaphores for ordering. Eliminates the separate sum3 kernel.
// ================================================================
void nvfp4_accum_gemm_v2_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    at::Tensor &D_out,
    at::Tensor &tile_done_buf
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_accum_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_accum_gemm::MAX_BATCHES);

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int K_first = (int)(A_list[0].size(1) * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_accum_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        // Per-tile completion counters (zeroed by caller)
        int num_tiles = g_host.num_row_blocks * 2 * g_host.num_col_blocks;
        TORCH_CHECK(tile_done_buf.numel() >= num_tiles,
                    "tile_done_buf too small: need ", num_tiles, " got ", tile_done_buf.numel());
        g_host.tile_done = tile_done_buf.data_ptr<int>();

        for (int i = 0; i < n; ++i) {
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }

        // Single shared D output TMA descriptor
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
        memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        kittens::py::launch_kernel<C, G, nvfp4_accum_gemm::kernel<C>>(g_host);
    };

    build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
}

// ================================================================
// Strided batched GEMM: reads A from a full (M, K_total/2) buffer
// with per-batch column offsets, avoiding .contiguous() copies.
// ================================================================

// Create a TMA descriptor for FP4 data with custom row stride.
// This allows reading from a sub-region of a larger contiguous buffer
// without requiring the sub-region itself to be contiguous.
static void create_strided_fp4_tma(
    CUtensorMap *tma_map,
    const void  *global_addr,    // pointer to start of sub-region
    int64_t      rows,           // M (number of rows)
    int64_t      sub_cols,       // N_g/2 (columns in this batch's sub-region)
    int64_t      full_row_stride // N_total/2 * sizeof(fp4x2) = total bytes between rows
) {
    // FP4 TMA: 5D with 128B swizzle (matching st_fp4e2m1_2 tile layout)
    constexpr uint32_t  tma_dim = 5;
    constexpr int64_t   swizzle_elements = 128;  // 128B / sizeof(fp4x2=1) = 128 elements

    uint64_t gmem_shape [5];
    uint64_t gmem_stride[4];
    uint32_t smem_shape [5];
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    gmem_shape[0] = (uint64_t)swizzle_elements;
    gmem_shape[1] = (uint64_t)rows;
    gmem_shape[2] = (uint64_t)(sub_cols + swizzle_elements - 1) / swizzle_elements;
    gmem_shape[3] = 1;
    gmem_shape[4] = 1;

    // KEY: gmem_stride[0] uses full_row_stride, NOT sub_cols
    gmem_stride[0] = (uint64_t)full_row_stride;
    gmem_stride[1] = 128;  // swizzle_bytes
    gmem_stride[2] = (uint64_t)rows * full_row_stride;
    gmem_stride[3] = (uint64_t)rows * full_row_stride;

    // Shared memory tile shape — must match the FP4 tile used by the kernel
    // st_fp4e2m1_2<Mb/2, Kb/2>: e.g. <64, 128> → smem rows=64, cols=128
    // The TMA loads swizzle_elements per inner dim and tile_height per outer dim
    smem_shape[0] = swizzle_elements;
    smem_shape[1] = 64;  // will be overridden below based on actual tile
    smem_shape[2] = 1;   // sub_cols / swizzle_elements;
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,  // fp4x2 → uint8
        tma_dim,
        const_cast<void*>(global_addr),
        gmem_shape, gmem_stride,
        smem_shape, smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "Failed to create strided FP4 TMA descriptor");
}

// Create a TMA descriptor for FP8 scale data with custom layout.
// Scales: [ntm, ntk, 512] where ntm = M/128, ntk = N_g/64.
// Scale tiles are stored separately per-group already in our case,
// but for strided A scales, we use the same approach.
static void create_strided_sc_tma(
    CUtensorMap *tma_map,
    const void  *global_addr,
    int64_t      depth,        // ntm = M/128
    int64_t      sub_rows,     // ntk = N_g/64
    int64_t      full_depth_stride  // bytes between depth slices in full buffer
) {
    // Scale TMA: 5D with 128B swizzle (matching st_hf<4, 256> tile layout)
    // The scale layout is (1, depth=ntm, rows=ntk, cols=256) with half type
    constexpr uint32_t  tma_dim = 5;
    constexpr int64_t   swizzle_elements = 64;  // 128B / sizeof(half=2) = 64 elements

    uint64_t gmem_shape [5];
    uint64_t gmem_stride[4];
    uint32_t smem_shape [5];
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    gmem_shape[0] = (uint64_t)swizzle_elements;
    gmem_shape[1] = (uint64_t)depth;  // ntm
    gmem_shape[2] = (uint64_t)(256 + swizzle_elements - 1) / swizzle_elements;  // 256/64 = 4
    gmem_shape[3] = (uint64_t)sub_rows;  // ntk
    gmem_shape[4] = 1;

    gmem_stride[0] = (uint64_t)256 * sizeof(uint16_t);  // stride between ntm slices = 256 * 2 = 512B
    gmem_stride[1] = 128;  // swizzle_bytes
    gmem_stride[2] = (uint64_t)depth * 256 * sizeof(uint16_t);  // stride between ntk slices
    gmem_stride[3] = full_depth_stride;  // stride between batches (if applicable)

    smem_shape[0] = swizzle_elements;
    smem_shape[1] = 4;   // tile height
    smem_shape[2] = 256 / swizzle_elements;  // 4
    smem_shape[3] = 1;
    smem_shape[4] = 1;

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        tma_dim,
        const_cast<void*>(global_addr),
        gmem_shape, gmem_stride,
        smem_shape, smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "Failed to create strided scale TMA descriptor");
}

// Strided batched GEMM: A FP4 data is read from a single full buffer with per-batch offsets.
// Scales are still passed per-batch (contiguous) since their copies are negligible.
// This avoids the expensive .contiguous() copies for FP4 narrow(1,...) views (0.4ms at M=64K).
void nvfp4_batched_gemm_strided_entrypoint(
    // A FP4: single full row-quantized buffer
    const at::Tensor &A_full,           // [M, K_total/2] fp4x2 (full concatenated row FP4)
    // A scales: per-batch contiguous (unchanged from regular batched GEMM)
    const std::vector<at::Tensor> &A_sc_list,  // per-batch scale tensors
    const std::vector<at::Tensor> &A_sg_list,  // per-batch [1] float32
    const std::vector<int64_t> &A_col_offsets,  // per-batch FP4 column offsets (in fp4x2 elements)
    const std::vector<int64_t> &A_col_widths,   // per-batch FP4 column widths (= N_g/2)
    // B: per-batch (unchanged)
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_sg_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)D_list.size());
    TORCH_CHECK(n == (int)A_col_offsets.size());
    TORCH_CHECK(n == (int)A_col_widths.size());
    TORCH_CHECK(n == (int)A_sc_list.size());

    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int64_t K_total_fp4 = A_full.size(1);  // total fp4x2 columns
    const int K_first = (int)(A_col_widths[0] * 2);  // first batch's K in elements

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_col_widths[0] / C::Kb);

        const uint8_t *a_base = (const uint8_t*)A_full.data_ptr();
        const int64_t a_full_row_stride = K_total_fp4;  // bytes per row (sizeof(fp4x2) = 1)

        for (int i = 0; i < n; ++i) {
            const int64_t fp4_cols = A_col_widths[i];     // N_g/2 in fp4x2 elements
            const int64_t fp4_offset = A_col_offsets[i];  // column offset in fp4x2

            // --- A FP4 TMA: strided ---
            // Create TMA for (M, fp4_cols) sub-region from (M, K_total_fp4) buffer
            // by overriding gmem_stride[0] to use full row stride.
            {
                // First create reference TMA via tensor_to_gl on a dummy contiguous tensor
                // to get correct smem_shape and other params, then override.
                // Actually, build from scratch matching create_tensor_map<st_fp4e2m1_2<128,128>, 2>:
                constexpr int64_t swizzle_elements = 128;  // 128B / sizeof(fp4x2=1) = 128
                const void *data_ptr = a_base + fp4_offset;
                
                uint64_t gmem_shape [5] = {
                    (uint64_t)swizzle_elements,                             // dim0: swizzle block
                    (uint64_t)M,                                            // dim1: rows
                    (uint64_t)(fp4_cols + swizzle_elements - 1) / swizzle_elements, // dim2: col-blocks
                    1, 1                                                    // dim3,4: depth, batch
                };
                uint64_t gmem_stride[4] = {
                    (uint64_t)a_full_row_stride,       // stride between rows (KEY: full buffer stride)
                    128,                               // swizzle_bytes
                    (uint64_t)M * a_full_row_stride,   // stride between depth slices
                    (uint64_t)M * a_full_row_stride    // stride between batches
                };
                // smem tile: st_fp4e2m1_2<Mb/2=128, Kb/2=128> → rows=128, cols=128
                uint32_t smem_shape [5] = {(uint32_t)swizzle_elements, (uint32_t)(C::Mb/2), 1, 1, 1};
                uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

                CUresult result = cuTensorMapEncodeTiled(
                    &g_host.A_tma[i],
                    CU_TENSOR_MAP_DATA_TYPE_UINT8,
                    5,
                    const_cast<void*>(data_ptr),
                    gmem_shape, gmem_stride,
                    smem_shape, smem_stride,
                    CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                );
                TORCH_CHECK(result == CUDA_SUCCESS,
                    "Strided FP4 TMA creation failed for batch ", i);
            }

            // --- A Scale TMA: standard (using tensor_to_gl) ---
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            // --- B and D: standard (unchanged) ---
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
    };

    if (K_first <= 2048 && N_out <= 4096) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else if (K_first <= 2048) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    }
}

// ================================================================
// Non-PDL batched GEMM strided: USE_PDL=false for CUDA graph safety.
// ================================================================
void nvfp4_batched_gemm_strided_nopdl_entrypoint(
    const at::Tensor &A_full,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<int64_t> &A_col_offsets,
    const std::vector<int64_t> &A_col_widths,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_sg_list.size();
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)D_list.size());
    TORCH_CHECK(n == (int)A_col_offsets.size());
    TORCH_CHECK(n == (int)A_col_widths.size());
    TORCH_CHECK(n == (int)A_sc_list.size());

    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);
    const int64_t K_total_fp4 = A_full.size(1);
    const int K_first = (int)(A_col_widths[0] * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_col_widths[0] / C::Kb);

        const uint8_t *a_base = (const uint8_t*)A_full.data_ptr();
        const int64_t a_full_row_stride = K_total_fp4;

        for (int i = 0; i < n; ++i) {
            const int64_t fp4_cols = A_col_widths[i];
            const int64_t fp4_offset = A_col_offsets[i];

            {
                constexpr int64_t swizzle_elements = 128;
                const void *data_ptr = a_base + fp4_offset;
                uint64_t gmem_shape [5] = {
                    (uint64_t)swizzle_elements, (uint64_t)M,
                    (uint64_t)(fp4_cols + swizzle_elements - 1) / swizzle_elements, 1, 1
                };
                uint64_t gmem_stride[4] = {
                    (uint64_t)a_full_row_stride, 128,
                    (uint64_t)M * a_full_row_stride, (uint64_t)M * a_full_row_stride
                };
                uint32_t smem_shape [5] = {(uint32_t)swizzle_elements, (uint32_t)(C::Mb/2), 1, 1, 1};
                uint32_t smem_stride[5] = {1, 1, 1, 1, 1};
                CUresult result = cuTensorMapEncodeTiled(
                    &g_host.A_tma[i], CU_TENSOR_MAP_DATA_TYPE_UINT8, 5,
                    const_cast<void*>(data_ptr), gmem_shape, gmem_stride,
                    smem_shape, smem_stride,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                );
                TORCH_CHECK(result == CUDA_SUCCESS, "Strided FP4 TMA (nopdl) failed for batch ", i);
            }

            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);
            memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_gemm::kernel<C>>(g_host);
    };

    // USE_PDL=false (8th arg), CLUSTER_SIZE=2 (9th arg)
    build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false, 256, false, 2>>();
}

// ================================================================
// Fused split dgrad+sum: slices concatenated dY, runs batched GEMM
// (z-dim parallel), then sums the per-split outputs.
// ================================================================

// Forward declaration — defined below
void nvfp4_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_list,
    at::Tensor &D_out
);
void nvfp4_split_dgrad_sum(
    // Concatenated row-quantized gradient: dy_cat_q
    const at::Tensor &A_fp4_cat,     // [M, N_total/2] fp4x2 (row-quantized dY)
    const at::Tensor &A_sc_cat,      // [ntm, ntk_total, 512] fp8 (row scales)
    const std::vector<at::Tensor> &A_sg_list,  // [n_splits] each [1] float32 (per-split global scale)
    // Per-split column-quantized weights
    const std::vector<at::Tensor> &B_fp4_list,  // [n_splits] each [K, N_i/2] fp4x2
    const std::vector<at::Tensor> &B_sc_list,   // [n_splits] each [ntm_c, ntk_c_i, 512] fp8
    const at::Tensor &B_sg_cat,                  // [n_splits] float32
    // Split dimensions
    const std::vector<int64_t> &split_dims,      // [q_dim, k_dim, v_dim]
    // Output
    at::Tensor &D_out                            // [M, K] bf16 — accumulated dgrad
) {
    const int n_splits = (int)split_dims.size();
    TORCH_CHECK(n_splits == (int)B_fp4_list.size());
    TORCH_CHECK(n_splits == (int)B_sc_list.size());
    TORCH_CHECK(n_splits == (int)A_sg_list.size());

    const int64_t M = D_out.size(0);
    const int64_t K = D_out.size(1);

    // Slice concatenated A into per-split tensors for batched GEMM
    auto a_fp4_bytes = A_fp4_cat.view(c10::ScalarType::Byte);
    auto a_sc_bytes = A_sc_cat.view(c10::ScalarType::Byte);

    std::vector<at::Tensor> A_list, A_sc_list_v, B_sg_list;
    std::vector<at::Tensor> D_list;

    int64_t fp4_col_offset = 0;
    int64_t sc_col_offset = 0;
    for (int i = 0; i < n_splits; ++i) {
        const int64_t N_i = split_dims[i];
        const int64_t fp4_cols_i = N_i / 2;
        const int64_t sc_tiles_i = N_i / 64;

        A_list.push_back(
            a_fp4_bytes.narrow(1, fp4_col_offset, fp4_cols_i)
                .contiguous().view(at::kFloat4_e2m1fn_x2)
        );
        A_sc_list_v.push_back(
            a_sc_bytes.narrow(1, sc_col_offset, sc_tiles_i)
                .contiguous().view(at::kFloat8_e4m3fn)
        );
        B_sg_list.push_back(B_sg_cat.narrow(0, i, 1));
        D_list.push_back(at::empty({M, K}, D_out.options()));

        fp4_col_offset += fp4_cols_i;
        sc_col_offset += sc_tiles_i;
    }

    // Z-dim parallel batched GEMM: one kernel launch, per-batch outputs
    nvfp4_batched_gemm_entrypoint(
        A_list, A_sc_list_v, A_sg_list,
        B_fp4_list, B_sc_list, B_sg_list,
        D_list
    );

    // Sum per-split outputs into D_out
    D_out.copy_(D_list[0]);
    for (int i = 1; i < n_splits; ++i) {
        D_out.add_(D_list[i]);
    }
}

// ================================================================
// ================================================================
// Simple CUDA kernel to sum N tensors element-wise into output
// ================================================================
__global__ void sum_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(A[idx], B[idx]);
    }
}

__global__ void sum3_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    // Vectorized path: 8 bf16 = 128 bits = int4 per iteration
    const int64_t vec_numel = numel / 8;
    const int4* A4 = reinterpret_cast<const int4*>(A);
    const int4* B4 = reinterpret_cast<const int4*>(B);
    const int4* C4 = reinterpret_cast<const int4*>(C);
    int4* out4 = reinterpret_cast<int4*>(out);

    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_numel; i += stride) {
        int4 a = A4[i];
        int4 b = B4[i];
        int4 c = C4[i];

        // Reinterpret as bf162 pairs and add
        __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&a);
        __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(&b);
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(&c);

        int4 r;
        __nv_bfloat162* r2 = reinterpret_cast<__nv_bfloat162*>(&r);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            r2[j] = __hadd2(__hadd2(a2[j], b2[j]), c2[j]);
        }
        out4[i] = r;
    }

    // Scalar tail for non-8-aligned remainder
    int64_t tail_start = vec_numel * 8;
    for (int64_t i = tail_start + threadIdx.x; i < numel; i += blockDim.x) {
        if (blockIdx.x == 0)
            out[i] = __hadd(__hadd(A[i], B[i]), C[i]);
    }
}

__global__ void sum4_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ D,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), __hadd(C[idx], D[idx]));
    }
}

// ================================================================
// Batched GEMM with Fused Accumulation — TRUE in-kernel accumulation.
// D_out = sum_i(A_i × B_i^T), accumulated in TMEM registers.
// No intermediate D buffers, no sum3 kernel.
// ================================================================
void nvfp4_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,       // per-batch [M, K/2] fp4x2
    const std::vector<at::Tensor> &A_sc_list,    // per-batch [ntm, ntk, 512] fp8
    const std::vector<at::Tensor> &A_sg_list,    // per-batch [1] float32
    const std::vector<at::Tensor> &B_list,       // per-batch [N, K/2] fp4x2
    const std::vector<at::Tensor> &B_sc_list,    // per-batch [ntm_b, ntk, 512] fp8
    const std::vector<at::Tensor> &B_sg_list,    // per-batch [1] float32
    at::Tensor &D_out                            // accumulated [M, N] bf16
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0 && n <= 4, "num_batches must be 1..4");
    TORCH_CHECK(D_out.dim() == 2);

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);

    if (n == 1) {
        // Single batch: just run regular GEMM directly into D_out
        std::vector<at::Tensor> D_list = {D_out};
        nvfp4_batched_gemm_entrypoint(A_list, A_sc_list, A_sg_list,
                                       B_list, B_sc_list, B_sg_list, D_list);
        return;
    }

    // Use the real in-kernel accumulation kernel
    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_accum_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        for (int i = 0; i < n; ++i) {
            auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
            auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
                A_sc_list[i], 1,
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(0)/128 : A_sc_list[i].size(0),
                A_sc_list[i].dim() == 2 ? A_sc_list[i].size(1)/4 : A_sc_list[i].size(1), 256);
            auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
            auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
                B_sc_list[i], 1,
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(0)/128 : B_sc_list[i].size(0),
                B_sc_list[i].dim() == 2 ? B_sc_list[i].size(1)/4 : B_sc_list[i].size(1), 256);
            memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }

        // Single output D TMA descriptor (accumulated result)
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
        memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        kittens::py::launch_kernel<C, G, nvfp4_batched_accum_gemm::kernel<C>>(g_host);
    };

    build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
}

// ================================================================
// Standalone fused 3-way bf16 sum: out = A + B + C
// Reads 3 inputs, writes 1 output in a single kernel launch.
// ================================================================
void sum3_bf16_entrypoint(
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    at::Tensor &out
) {
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous() && out.is_contiguous());
    const int64_t numel = A.numel();
    TORCH_CHECK(numel == B.numel() && numel == C.numel() && numel == out.numel());
    const int threads = 256;
    // vec_numel = numel/8 (each thread handles 8 bf16 via int4 loads)
    const int64_t vec_numel = numel / 8;
    int blocks = (int)((vec_numel + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;  // grid-striding handles the rest
    auto stream = at::cuda::getCurrentCUDAStream();
    sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(C.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        numel);
}
// ================================================================
// Fused Quantize+GEMM: A is bf16, B is pre-quantized NVFP4
// Mode 1: constant SCALE_MAX (USE_CTA_AMAX=false) — fastest
// Mode 2: CTA-level amax (USE_CTA_AMAX=true) — better accuracy
// ================================================================
template <typename C>
static inline void launch_fused_gemm_with_config(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    using G = nvfp4_fused_gemm::globals<C>;

    const int M = A_bf16.size(0);
    const int K = A_bf16.size(1);

    // Allocate scratch buffers for quantized A reuse across col tiles
    auto A_scratch_byte = at::zeros({M, K/2}, at::TensorOptions().dtype(at::kByte).device(A_bf16.device()));
    auto A_scratch = A_scratch_byte.view(at::kFloat4_e2m1fn_x2);
    auto A_sc_scratch = at::zeros({M/128, K/64, 512}, at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_bf16.device()));
    // Per-(cta_half, K_stage) completion flags, zeroed before each launch
    // num_cta_halves = M / 128 (each 256-row block has 2 CTA halves)
    // num_red_blocks = K / 256
    const int num_flags = (M / 128) * (K / 256);
    auto A_scratch_ready = at::zeros({num_flags}, at::TensorOptions().dtype(at::kInt).device(A_bf16.device()));

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1),
            256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .use_split_D = false,
        .b_sg_per_tile = nullptr,
        .silu_dim = 0,
        .A_scratch = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_scratch),
        .A_sc_scratch = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_scratch, 1, M/128, K/64, 256),
        .A_scratch_ready = (uint32_t*)A_scratch_ready.data_ptr<int>(),
    };

    kittens::py::launch_kernel<C, G, nvfp4_fused_gemm::kernel<C>>(g);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    const int K = A_bf16.size(1);
    const int N = D.size(1);

    if (N % 256 == 0 && N <= 2048) {
        if (K <= 256 && N <= 256) {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 4, 4, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>>(
                A_bf16, B, B_sc, B_sc_global, D);
        } else {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 4, 8, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>>(
                A_bf16, B, B_sc, B_sc_global, D);
        }
    } else {
        if (N % 256 != 0) {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 5, 4, 4, 2, false, USE_CTA_AMAX>>(
                A_bf16, B, B_sc, B_sc_global, D);
        } else {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<256, 4, 8, 4, 2, false, USE_CTA_AMAX>>(
                A_bf16, B, B_sc, B_sc_global, D);
        }
    }
}

void nvfp4_fused_gemm_entrypoint(
    const at::Tensor &A_bf16,       // [M, K] bf16 activations
    const at::Tensor &B,            // [N, K/2] fp4x2
    const at::Tensor &B_sc,         // [N/128, K/64, 512] fp8
    const at::Tensor &B_sc_global,  // [1] float32
    at::Tensor &D                   // [M, N] bf16 output
) {
    int M = A_bf16.size(0);
    int K = A_bf16.size(1);
    int N = D.size(1);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::kFloat4_e2m1fn_x2, "B must be fp4x2");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128");
    dispatch_fused_gemm<false>(A_bf16, B, B_sc, B_sc_global, D);
}

// CTA-level amax version — pre-scans A for per-CTA max|A|
void nvfp4_fused_gemm_cta_amax_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    int M = A_bf16.size(0);
    int K = A_bf16.size(1);
    int N = D.size(1);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::kFloat4_e2m1fn_x2, "B must be fp4x2");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 128 == 0, "N must be multiple of 128");
    dispatch_fused_gemm<true>(A_bf16, B, B_sc, B_sc_global, D);
}

// ================================================================
// Persistent Quantize→GEMM: single kernel launch.
// Quantizes A (and optionally B) to FP4 in HBM, then runs GEMM.
// Uses constant SCALE_MAX (no amax scan).
// ================================================================
template <typename KC>
static __cluster_dims__(KC::CLUSTER_SIZE) __launch_bounds__(KC::NUM_THREADS)
__global__ void persistent_gemm_entry(const __grid_constant__ nvfp4_persistent_gemm::globals<KC> g) {
    nvfp4_persistent_gemm::kernel<KC>(g);
}

void nvfp4_persistent_gemm_entrypoint(
    const at::Tensor &A_bf16,       // [M, K] bf16
    const at::Tensor &B_bf16,       // [N, K] bf16
    at::Tensor &D                   // [M, N] bf16 output
) {
    int M = A_bf16.size(0);
    int K = A_bf16.size(1);
    int N = B_bf16.size(0);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 256 == 0, "N must be multiple of 256");

    constexpr float SCALE_MAX_DEC = 65504.0f / (6.0f * 448.0f);

    // Allocate FP4 scratch buffers in HBM
    auto A_fp4 = at::empty({M, K/2}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2).device(A_bf16.device()));
    auto A_sc  = at::empty({M/128, K/64, 512}, at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_bf16.device()));
    auto A_sg  = at::full({1}, SCALE_MAX_DEC, at::TensorOptions().dtype(at::kFloat).device(A_bf16.device()));
    auto B_fp4 = at::empty({N, K/2}, at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2).device(A_bf16.device()));
    auto B_sc  = at::empty({N/128, K/64, 512}, at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(A_bf16.device()));
    auto B_sg  = at::full({1}, SCALE_MAX_DEC, at::TensorOptions().dtype(at::kFloat).device(A_bf16.device()));

    // Grid barrier counter
    auto barrier = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(A_bf16.device()));

    using C = nvfp4_persistent_gemm::config<256, 5, 8, 4, 2, false>;
    using G = nvfp4_persistent_gemm::globals<C>;

    G g {
        // Phase 1: quantize
        .A_bf16  = kittens::py::tensor_to_gl<typename G::Q_bf16_gl>(A_bf16),
        .A_q_fp4 = kittens::py::tensor_to_gl<typename G::Q_fp4_gl>(A_fp4),
        .A_q_sc  = kittens::py::tensor_to_gl<typename G::Q_sc_gl, false>(A_sc, 1, M/128, K/64, 256),
        .B_bf16  = kittens::py::tensor_to_gl<typename G::Q_bf16_gl>(B_bf16),
        .B_q_fp4 = kittens::py::tensor_to_gl<typename G::Q_fp4_gl>(B_fp4),
        .B_q_sc  = kittens::py::tensor_to_gl<typename G::Q_sc_gl, false>(B_sc, 1, N/128, K/64, 256),
        .quantize_b = true,
        // Phase 2: GEMM (same HBM data)
        .A       = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_fp4),
        .A_sc    = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, M/128, K/64, 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sg),
        .B       = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_fp4),
        .B_sc    = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, N/128, K/64, 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sg),
        .D       = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .barrier = barrier.data_ptr<int>(),
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    auto smem = g.dynamic_shared_memory();
    cudaFuncSetAttribute(persistent_gemm_entry<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    LaunchConfig<true, false> lc(g.grid(), g.block(), smem, (cudaStream_t)stream, C::CLUSTER_SIZE);
    cudaLaunchKernelEx(lc, persistent_gemm_entry<C>, g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm_entrypoint);
    m.def("nvfp4_gemm_nopdl", &nvfp4_gemm_nopdl_entrypoint,
          "Non-PDL GEMM for CUDA graph capture (CLUSTER_SIZE=1, USE_PDL=false)");
    m.def("nvfp4_gemm_config", &nvfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("nvfp4_gemm_config_nopdl", &nvfp4_gemm_config_nopdl_entrypoint,
          "Non-PDL GEMM with selectable tile config (for sweeping side-stream tails)",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("nvfp4_grouped_gemm", &nvfp4_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K_opt") = std::nullopt, pybind11::arg("D_V_opt") = std::nullopt,
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_grouped_gemm_nopdl", &nvfp4_grouped_gemm_nopdl_entrypoint,
          "Non-PDL grouped GEMM for multi-stream and CUDA graph usage",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K_opt") = std::nullopt, pybind11::arg("D_V_opt") = std::nullopt,
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_split_dgrad_sum", &nvfp4_split_dgrad_sum,
          "Fused split dgrad: slice concatenated row-quantized gradient → batched GEMM + accumulation",
          pybind11::arg("A_fp4_cat"), pybind11::arg("A_sc_cat"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_fp4_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_cat"),
          pybind11::arg("split_dims"), pybind11::arg("D_out"));
    m.def("nvfp4_accum_gemm_v2", &nvfp4_accum_gemm_v2_entrypoint,
          "Accumulating Batched GEMM: D_out = sum_i(A_i × B_i^T), fused in epilogue",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_out"), pybind11::arg("tile_done_buf"));
    m.def("nvfp4_batched_accum_gemm", &nvfp4_batched_accum_gemm_entrypoint,
          "True Batched GEMM with Fused Accumulation: D_out = sum_i(A_i × B_i^T)",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_out"));
    m.def("nvfp4_batched_gemm", &nvfp4_batched_gemm_entrypoint,
          "True Batched GEMM (z-dim parallel): D_i = A_i × B_i^T",
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_batched_gemm_strided", &nvfp4_batched_gemm_strided_entrypoint,
          "Strided Batched GEMM: reads A FP4 from full buffer with column offsets, avoiding .contiguous()",
          pybind11::arg("A_full"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_batched_gemm_strided_nopdl", &nvfp4_batched_gemm_strided_nopdl_entrypoint,
          "Non-PDL batched GEMM strided (USE_PDL=false) for CUDA graph safety",
          pybind11::arg("A_full"), pybind11::arg("A_sc_list"), pybind11::arg("A_sg_list"),
          pybind11::arg("A_col_offsets"), pybind11::arg("A_col_widths"),
          pybind11::arg("B_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_quantize", &nvfp4_quantize_entrypoint);
    m.def("sum3_bf16", &sum3_bf16_entrypoint,
          "Fused 3-way bf16 sum: out = A + B + C (single kernel)",
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("out"));
    m.def("fp32_to_fp4x2", &fp32_to_fp4x2_entrypoint);
    m.def("fp4x2_to_fp32", &fp4x2_to_fp32_entrypoint);
    m.def("nvfp4_fused_gemm", &nvfp4_fused_gemm_entrypoint,
          "Fused Quantize+GEMM: constant SCALE_MAX (fastest, no pre-scan)",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_cta_amax", &nvfp4_fused_gemm_cta_amax_entrypoint,
          "Fused Quantize+GEMM: CTA-level amax pre-scan (better accuracy)",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
    m.def("nvfp4_persistent_gemm", &nvfp4_persistent_gemm_entrypoint,
          "Persistent Quantize->GEMM: quantize A+B to HBM then GEMM, single kernel",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));

    // Fused TE→TK GEMM: takes raw TE NVFP4 tensors + dimensions.
    // ALL tensor manipulation (view, reshape, amax*recip) happens in C++.
    // Python side passes only raw data pointers + integer dimensions.
    //
    // Args:
    //   a_fp4_data:   raw fp4 packed data (any shape, viewed as fp4x2)
    //   a_scale_inv:  flat swizzled scales (any shape, reshaped to tiles)
    //   a_amax:       [1] float32
    //   a_M, a_K:     dimensions of A matrix
    //   b_fp4_data, b_scale_inv, b_amax, b_M, b_K: same for B
    //   out:          [a_M, b_M] bf16 pre-allocated output
    m.def("nvfp4_gemm_from_te", [](
        const at::Tensor &a_fp4_data,
        const at::Tensor &a_scale_inv,
        const at::Tensor &a_amax,
        int64_t a_M, int64_t a_K,
        const at::Tensor &b_fp4_data,
        const at::Tensor &b_scale_inv,
        const at::Tensor &b_amax,
        int64_t b_M, int64_t b_K,
        at::Tensor &out
    ) {
        const float NVFP4_SCALE_RECIP = 1.0f / (6.0f * 448.0f);

        // View fp4_data as fp4x2
        auto A = a_fp4_data.view(at::kFloat4_e2m1fn_x2);
        auto B = b_fp4_data.view(at::kFloat4_e2m1fn_x2);

        // Reshape flat scales to tile layout and view as fp8
        int64_t a_ntm = a_M / 128, a_ntk = a_K / 64;
        int64_t b_ntm = b_M / 128, b_ntk = b_K / 64;
        auto A_sc = a_scale_inv.reshape({a_ntm, a_ntk, 512}).view(at::kFloat8_e4m3fn);
        auto B_sc = b_scale_inv.reshape({b_ntm, b_ntk, 512}).view(at::kFloat8_e4m3fn);

        // Compute sc_global
        auto A_sg = a_amax.mul(NVFP4_SCALE_RECIP);
        auto B_sg = b_amax.mul(NVFP4_SCALE_RECIP);

        nvfp4_gemm_entrypoint(A, A_sc, A_sg, B, B_sc, B_sg, out);
    });
}

#endif
