// ================================================================
// NVFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "nvfp4_gemm.cuh"
#include "nvfp4_quantize.cuh"
#include "nvfp4_batched_accum_gemm.cuh"
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
        using C = nvfp4_gemm::config<256, 4, 16, 4, 2, false>;
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
        using C = nvfp4_gemm::config<256, 4, 16, 4, 2, false>;
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
        using C = nvfp4_gemm::config<256, 4, 8, 12, 2, false>;
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

void nvfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D, int config_id
) {
    //                     Nb   LOAD EPI  SG  DT  OVERLAP
    switch (config_id) {
    case 0: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 1: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 2: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 3: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 4: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 5: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 6: run_gemm_with_config<nvfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 7: run_gemm_with_config<nvfp4_gemm::config<128, 5,  4, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 8: run_gemm_with_config<nvfp4_gemm::config<128, 4,  4, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 9: run_gemm_with_config<nvfp4_gemm::config<128, 5,  4,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 10: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 11: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-11)");
    }
}

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
        build_and_launch.operator()<nvfp4_gemm::config<256, 4, 16, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 4, 8, 12, 2, false>>();
    }
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
// True Batched GEMM entrypoint with Fused Accumulation
// D_out = sum_i(A_i × B_i^T), independently per batch but output is accumulated.
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
    TORCH_CHECK(n > 0 && n <= nvfp4_batched_accum_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_batched_accum_gemm::MAX_BATCHES);
    TORCH_CHECK(n == (int)A_sc_list.size());
    TORCH_CHECK(n == (int)A_sg_list.size());
    TORCH_CHECK(n == (int)B_list.size());
    TORCH_CHECK(n == (int)B_sc_list.size());
    TORCH_CHECK(n == (int)B_sg_list.size());
    TORCH_CHECK(D_out.dim() == 2);

    const int64_t M = D_out.size(0);
    const int64_t N_out = D_out.size(1);
    const int K_first = (int)(A_list[0].size(1) * 2);

    auto build_and_launch = [&]<typename C>() {
        using G = nvfp4_batched_accum_gemm::globals<C>;
        G g_host;
        memset(&g_host, 0, sizeof(G));
        g_host.num_batches = n;
        g_host.num_row_blocks = (int)(M / C::Mb);
        g_host.num_col_blocks = (int)(N_out / C::Nb);
        g_host.num_red_blocks = (int)(2 * A_list[0].size(1) / C::Kb);

        for (int i = 0; i < n; ++i) {
            // Input TMA descriptors
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

            // Output TMA descriptor is fixed for all batches
            if (i == 0) {
                auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_out);
                memcpy(&g_host.D_tma, &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
            }

            // Scale pointers
            g_host.A_sg[i] = A_sg_list[i].data_ptr<float>();
            g_host.B_sg[i] = B_sg_list[i].data_ptr<float>();
        }
        kittens::py::launch_kernel<C, G, nvfp4_batched_accum_gemm::kernel<C>>(g_host);
    };

    if (K_first <= 2048 && N_out <= 4096) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else if (K_first <= 2048) {
        build_and_launch.operator()<nvfp4_gemm::config<256, 4, 16, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 4, 8, 12, 2, false>>();
    }
}

PYBIND11_MODULE(_C, m) {
    m.def("nvfp4_gemm", &nvfp4_gemm_entrypoint);
    m.def("nvfp4_gemm_config", &nvfp4_gemm_config_entrypoint,
          "GEMM with selectable tile config (for sweeping)",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"), pybind11::arg("config_id"));
    m.def("nvfp4_grouped_gemm", &nvfp4_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_per_tile"),
          pybind11::arg("D"), pybind11::arg("D_K_opt") = std::nullopt, pybind11::arg("D_V_opt") = std::nullopt,
          pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_split_dgrad_sum", &nvfp4_split_dgrad_sum,
          "Fused split dgrad: slice concatenated row-quantized gradient → batched GEMM + accumulation",
          pybind11::arg("A_fp4_cat"), pybind11::arg("A_sc_cat"), pybind11::arg("A_sg_list"),
          pybind11::arg("B_fp4_list"), pybind11::arg("B_sc_list"), pybind11::arg("B_sg_cat"),
          pybind11::arg("split_dims"), pybind11::arg("D_out"));
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
    m.def("nvfp4_quantize", &nvfp4_quantize_entrypoint);
    m.def("fp32_to_fp4x2", &fp32_to_fp4x2_entrypoint);
    m.def("fp4x2_to_fp32", &fp4x2_to_fp32_entrypoint);

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