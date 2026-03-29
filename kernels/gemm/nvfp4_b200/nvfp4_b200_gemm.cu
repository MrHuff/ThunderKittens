// ================================================================
// NVFP4 GEMM Module — Main compilation unit.
// Includes kernel headers and provides entrypoints + pybind11.
// ================================================================
#include "nvfp4_gemm.cuh"
#include "nvfp4_quantize.cuh"
#include "nvfp4_batched_accum_gemm.cuh"
#include "nvfp4_fused_gemm.cuh"
#include "nvfp4_fused_gemm_both_bf16.cuh"
#include <cstring>
#include <string>
#include <vector>
#include <optional>

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_entrypoint(const __grid_constant__ nvfp4_gemm::globals<C> g) {
    nvfp4_gemm::kernel<C>(g);
}

template <typename C>
__cluster_dims__(C::CLUSTER_SIZE) __launch_bounds__(C::NUM_THREADS)
__global__ void kernel_both_bf16_entrypoint(
    const __grid_constant__ nvfp4_fused_gemm_both_bf16::globals<C> g
) {
    nvfp4_fused_gemm_both_bf16::kernel<C>(g);
}

template <typename C>
static inline void launch_both_bf16_standalone(
    const nvfp4_fused_gemm_both_bf16::globals<C> &g
) {
    CUDACHECK(cudaFuncSetAttribute(
        kernel_both_bf16_entrypoint<C>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
#if defined(KITTENS_BLACKWELL)
    if constexpr (C::CLUSTER_SIZE > 8) {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_both_bf16_entrypoint<C>,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1));
    }
#endif
    if constexpr (C::CLUSTER_SIZE <= 1) {
        LaunchConfig<false, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(), 0);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel_both_bf16_entrypoint<C>, g));
    } else {
        LaunchConfig<true, false> launch_config(
            g.grid(), g.block(), g.dynamic_shared_memory(), 0, C::CLUSTER_SIZE);
        CUDACHECK(cudaLaunchKernelEx(
            launch_config, kernel_both_bf16_entrypoint<C>, g));
    }
}

#ifdef SHARED_B_DEBUG_STANDALONE

struct SharedBDebugStandaloneOptions {
    std::string mode = "shared_b_cta";
    int m = 1024;
    int n = 1024;
    int k = 1024;
};

static inline void print_shared_b_debug_standalone_usage(const char *argv0) {
    std::cout
        << "Usage:\n"
        << "  " << argv0 << " --mode {shared_b_const,shared_b_cta,shared_b_transport_const,shared_b_transport_cta}"
        << " [--m M] [--n N] [--k K]\n"
        << "  " << argv0 << " shared_b_const 256 256 256\n";
}

static inline bool parse_int_arg(const char *arg, int &out) {
    try {
        out = std::stoi(arg);
        return true;
    } catch (...) {
        return false;
    }
}

static inline bool parse_shared_b_debug_standalone_args(
    int argc,
    char **argv,
    SharedBDebugStandaloneOptions &opts
) {
    int positional_ints = 0;
    bool positional_mode_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--mode") {
            if (i + 1 >= argc) return false;
            opts.mode = argv[++i];
        } else if (arg == "--m") {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], opts.m)) return false;
        } else if (arg == "--n") {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], opts.n)) return false;
        } else if (arg == "--k") {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], opts.k)) return false;
        } else if (!arg.empty() && arg[0] != '-' && !positional_mode_set) {
            opts.mode = arg;
            positional_mode_set = true;
        } else if (!arg.empty() && arg[0] != '-') {
            int value = 0;
            if (!parse_int_arg(arg.c_str(), value)) return false;
            if (positional_ints == 0) opts.m = value;
            else if (positional_ints == 1) opts.n = value;
            else if (positional_ints == 2) opts.k = value;
            else return false;
            ++positional_ints;
        } else {
            return false;
        }
    }
    return true;
}

static inline void print_bf16_buffer_summary(
    const std::vector<__nv_bfloat16> &h_D
) {
    double checksum = 0.0;
    double abs_max = 0.0;
    size_t finite_count = 0;
    for (const __nv_bfloat16 &x : h_D) {
        const float val = kittens::base_types::convertor<float, __nv_bfloat16>::convert(x);
        if (std::isfinite(val)) {
            ++finite_count;
            checksum += static_cast<double>(val);
            abs_max = std::max(abs_max, static_cast<double>(std::abs(val)));
        }
    }
    std::cout << "finite elements: " << finite_count << " / " << h_D.size() << "\n";
    std::cout << "checksum:        " << checksum << "\n";
    std::cout << "abs max:         " << abs_max << "\n";
}

template <bool USE_CTA_AMAX>
static inline int run_shared_b_debug_standalone_once(
    const SharedBDebugStandaloneOptions &opts,
    bool transport_only
) {
    using C = nvfp4_fused_gemm_both_bf16::config<
        USE_CTA_AMAX, 1, 2, 4, 2, 2, false, false, false, true, false, true, 128>;
    using G = nvfp4_fused_gemm_both_bf16::globals<C>;

    if ((opts.m % 256) != 0 || (opts.n % 256) != 0 || (opts.k % 128) != 0) {
        std::cerr << "Shared-B standalone runner expects M/N multiples of 256 and K a multiple of 128.\n";
        return 2;
    }

    __nv_bfloat16 *d_A = nullptr;
    __nv_bfloat16 *d_B = nullptr;
    __nv_bfloat16 *d_D = nullptr;

    const size_t a_count = static_cast<size_t>(opts.m) * opts.k;
    const size_t b_count = static_cast<size_t>(opts.n) * opts.k;
    const size_t d_count = static_cast<size_t>(opts.m) * opts.n;
    const float init_bound = 1.0f / std::sqrt(std::sqrt(static_cast<float>(opts.k)));

    CUDACHECK(cudaMalloc(&d_A, a_count * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_B, b_count * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_D, d_count * sizeof(__nv_bfloat16)));

    fill<__nv_bfloat16, FillMode::RANDOM>(d_A, a_count, 2024, -init_bound, init_bound);
    fill<__nv_bfloat16, FillMode::RANDOM>(d_B, b_count, 2025, -init_bound, init_bound);
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D, d_count, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    G g{
        .A_bf16 = kittens::make_gl<typename G::A_bf16_gl>(
            reinterpret_cast<uint64_t>(d_A), 1, 1, opts.m, opts.k),
        .B_bf16 = kittens::make_gl<typename G::B_bf16_gl>(
            reinterpret_cast<uint64_t>(d_B), 1, 1, opts.n, opts.k),
        .D = kittens::make_gl<typename G::D_gl>(
            reinterpret_cast<uint64_t>(d_D), 1, 1, opts.m, opts.n),
        .debug_cta0_a_ptr = nullptr,
        .debug_cta1_a_ptr = nullptr,
        .debug_cta0_sc_ptr = nullptr,
        .debug_cta1_sc_ptr = nullptr,
        .debug_a_stride = 0,
        .debug_transport_only = transport_only,
        .debug_main_dump_only = false,
        .debug_front_half_mode = 0,
    };

    std::cout << "Mode: " << opts.mode
              << "  M=" << opts.m << " N=" << opts.n << " K=" << opts.k
              << "  smem=" << g.dynamic_shared_memory() << " bytes"
              << "  cluster=" << C::CLUSTER_SIZE
              << "  threads=" << C::NUM_THREADS << "\n";

    cudaGetLastError();
    launch_both_bf16_standalone<C>(g);
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(sync_err) << "\n";
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_D);
        return 1;
    }

    std::cout << "Kernel completed successfully.\n";
    if (!transport_only) {
        std::vector<__nv_bfloat16> h_D(d_count);
        CUDACHECK(cudaMemcpy(
            h_D.data(), d_D, d_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        print_bf16_buffer_summary(h_D);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    return 0;
}

#endif

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

int main(int argc, char **argv) {
#ifdef SHARED_B_DEBUG_STANDALONE
    SharedBDebugStandaloneOptions opts;
    if (!parse_shared_b_debug_standalone_args(argc, argv, opts)) {
        print_shared_b_debug_standalone_usage(argv[0]);
        return 1;
    }

    if (opts.mode == "shared_b_const") {
        return run_shared_b_debug_standalone_once<false>(opts, false);
    } else if (opts.mode == "shared_b_cta") {
        return run_shared_b_debug_standalone_once<true>(opts, false);
    } else if (opts.mode == "shared_b_transport_const") {
        return run_shared_b_debug_standalone_once<false>(opts, true);
    } else if (opts.mode == "shared_b_transport_cta") {
        return run_shared_b_debug_standalone_once<true>(opts, true);
    }

    print_shared_b_debug_standalone_usage(argv[0]);
    return 1;
#else
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
#endif
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
            .v_dim = 0,
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
            .v_dim = 0,
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
            .v_dim = 0,
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
        .v_dim = 0,
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
    bool use_split_D = D_K_opt.has_value();
    int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;

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
            .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value()) : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
            .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
            .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
            .v_dim = use_split_D ? v_dim : 0,
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
            .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value()) : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
            .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
            .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
            .v_dim = use_split_D ? v_dim : 0,
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
    bool use_split_D = D_K_opt.has_value();
    int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;

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
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value()) : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value()) : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = use_split_D ? v_dim : 0,
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
        .q_dim = 0, .k_dim = 0, .v_dim = 0, .use_split_D = false, .b_sg_per_tile = nullptr, .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_gemm::kernel<C>>(g);
}

void nvfp4_gemm_config_entrypoint(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &D, int config_id
) {
    //                     Nb   LOAD EPI  SG  DT  OVERLAP  Mb   PDL
    switch (config_id) {
    // ── Original 12 configs (0-11) ──────────────────────────────
    case 0:  run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 1:  run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 2:  run_gemm_with_config<nvfp4_gemm::config<256, 4, 16, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 3:  run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 4:  run_gemm_with_config<nvfp4_gemm::config<256, 5,  8, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 5:  run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 6:  run_gemm_with_config<nvfp4_gemm::config<256, 4,  8, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 7:  run_gemm_with_config<nvfp4_gemm::config<128, 5,  4, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 8:  run_gemm_with_config<nvfp4_gemm::config<128, 4,  4, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 9:  run_gemm_with_config<nvfp4_gemm::config<128, 5,  4,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 10: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 11: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=256: SG sweep with EP=8, LP=5 (best base) ───────────
    case 12: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 13: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  2, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 14: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  8, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 15: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 16: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  1, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 17: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  2, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 18: run_gemm_with_config<nvfp4_gemm::config<256, 5,  8,  8, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=256: SG sweep with EP=16, LP=5 ──────────────────────
    case 19: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 20: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  2, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 21: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  8, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 22: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  1, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 23: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  2, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 24: run_gemm_with_config<nvfp4_gemm::config<256, 5, 16,  8, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=256: LP=4 with EP=8, SG sweep ────────────────────────
    case 25: run_gemm_with_config<nvfp4_gemm::config<256, 4,  8,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 26: run_gemm_with_config<nvfp4_gemm::config<256, 4,  8,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 27: run_gemm_with_config<nvfp4_gemm::config<256, 4,  8,  2, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 28: run_gemm_with_config<nvfp4_gemm::config<256, 4,  8,  8, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=256: LP=3 exploration ────────────────────────────────
    case 29: run_gemm_with_config<nvfp4_gemm::config<256, 3,  8,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 30: run_gemm_with_config<nvfp4_gemm::config<256, 3, 16,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=256: SG sweep with EP=16, LP=4 ──────────────────────
    case 31: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  2, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 32: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  8, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 33: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  1, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 34: run_gemm_with_config<nvfp4_gemm::config<256, 4, 16,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=128: EP=8 sweep (gap in original) ────────────────────
    case 35: run_gemm_with_config<nvfp4_gemm::config<128, 5,  8,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 36: run_gemm_with_config<nvfp4_gemm::config<128, 5,  8, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 37: run_gemm_with_config<nvfp4_gemm::config<128, 5,  8,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 38: run_gemm_with_config<nvfp4_gemm::config<128, 5,  8,  4, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 39: run_gemm_with_config<nvfp4_gemm::config<128, 5,  8, 12, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 40: run_gemm_with_config<nvfp4_gemm::config<128, 4,  8,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 41: run_gemm_with_config<nvfp4_gemm::config<128, 4,  8, 12, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // ── Nb=128: EP=4, SG=1 and SG=2 (gap in original) ──────────
    case 42: run_gemm_with_config<nvfp4_gemm::config<128, 5,  4,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 43: run_gemm_with_config<nvfp4_gemm::config<128, 5,  4,  2, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 44: run_gemm_with_config<nvfp4_gemm::config<128, 5,  4,  1, 2, true >>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 45: run_gemm_with_config<nvfp4_gemm::config<128, 4,  4,  4, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;
    case 46: run_gemm_with_config<nvfp4_gemm::config<128, 4,  4,  1, 2, false>>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, D); break;

    // NOTE: Nb=128 + EP=16 is INVALID (D_tile cols = 128/16 = 8 < 16 minimum)

    default: TORCH_CHECK(false, "Invalid config_id: ", config_id, " (valid: 0-46)");
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
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    } else {
        build_and_launch.operator()<nvfp4_gemm::config<256, 5, 8, 4, 2, false>>();
    }
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
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), C[idx]);
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
// Batched GEMM with Accumulation — implemented as batched_gemm + sum
// D_out = sum_i(A_i × B_i^T), all batches produce separate outputs then sum.
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

    // Allocate per-batch output buffers
    std::vector<at::Tensor> D_list;
    D_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        D_list.push_back(at::empty({M, N_out}, D_out.options()));
    }

    // Run batched GEMM with separate outputs
    nvfp4_batched_gemm_entrypoint(A_list, A_sc_list, A_sg_list,
                                   B_list, B_sc_list, B_sg_list, D_list);

    // Sum outputs with a simple CUDA kernel
    const int64_t numel = M * N_out;
    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (n == 2) {
        sum_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else if (n == 3) {
        sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else { // n == 4
        sum4_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[3].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    }
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
    const int blocks = (int)((numel + threads - 1) / threads);
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
    at::Tensor &D,
    at::Tensor *debug_cta0_a = nullptr,
    at::Tensor *debug_cta1_a = nullptr,
    at::Tensor *debug_cta0_sc = nullptr,
    at::Tensor *debug_cta1_sc = nullptr
) {
    using G = nvfp4_fused_gemm::globals<C>;
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
        .debug_cta0_a_ptr = debug_cta0_a ? reinterpret_cast<uint8_t*>(debug_cta0_a->data_ptr()) : nullptr,
        .debug_cta1_a_ptr = debug_cta1_a ? reinterpret_cast<uint8_t*>(debug_cta1_a->data_ptr()) : nullptr,
        .debug_cta0_sc_ptr = debug_cta0_sc ? reinterpret_cast<uint8_t*>(debug_cta0_sc->data_ptr()) : nullptr,
        .debug_cta1_sc_ptr = debug_cta1_sc ? reinterpret_cast<uint8_t*>(debug_cta1_sc->data_ptr()) : nullptr,
        .debug_a_stride = debug_cta0_a ? static_cast<int>(debug_cta0_a->size(1))
                       : (debug_cta1_a ? static_cast<int>(debug_cta1_a->size(1)) : 0)
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

    if (N % 256 == 0) {
        if (K <= 256 && N <= 256) {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 4, 4, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>>(
                A_bf16, B, B_sc, B_sc_global, D);
        } else {
            launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 4, 8, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>>(
                A_bf16, B, B_sc, B_sc_global, D);
        }
    } else {
        launch_fused_gemm_with_config<nvfp4_fused_gemm::config<128, 5, 4, 4, 2, false, USE_CTA_AMAX>>(
            A_bf16, B, B_sc, B_sc_global, D);
    }
}

template <typename C>
static inline void launch_fused_gemm_both_bf16_with_config(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor *debug_cta0_a = nullptr,
    at::Tensor *debug_cta1_a = nullptr,
    at::Tensor *debug_cta0_sc = nullptr,
    at::Tensor *debug_cta1_sc = nullptr,
    bool debug_transport_only = false,
    bool debug_main_dump_only = false,
    int debug_front_half_mode = 0
) {
    using G = nvfp4_fused_gemm_both_bf16::globals<C>;
    G g{
        .A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16),
        .B_bf16 = kittens::py::tensor_to_gl<typename G::B_bf16_gl>(B_bf16),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .debug_cta0_a_ptr = debug_cta0_a ? reinterpret_cast<uint8_t*>(debug_cta0_a->data_ptr()) : nullptr,
        .debug_cta1_a_ptr = debug_cta1_a ? reinterpret_cast<uint8_t*>(debug_cta1_a->data_ptr()) : nullptr,
        .debug_cta0_sc_ptr = debug_cta0_sc ? reinterpret_cast<uint8_t*>(debug_cta0_sc->data_ptr()) : nullptr,
        .debug_cta1_sc_ptr = debug_cta1_sc ? reinterpret_cast<uint8_t*>(debug_cta1_sc->data_ptr()) : nullptr,
        .debug_a_stride = debug_cta0_a ? static_cast<int>(debug_cta0_a->size(1))
                       : (debug_cta1_a ? static_cast<int>(debug_cta1_a->size(1)) : 0),
        .debug_transport_only = debug_transport_only,
        .debug_main_dump_only = debug_main_dump_only,
        .debug_front_half_mode = debug_front_half_mode,
    };
    kittens::py::launch_kernel<C, G, nvfp4_fused_gemm_both_bf16::kernel<C>>(g);
}

template <typename C>
static inline void launch_fused_gemm_both_bf16_transport_debug_with_config(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using G = nvfp4_fused_gemm_both_bf16::transport_debug_globals<C>;
    G g{
        .A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16),
        .debug_cta0_a_ptr = reinterpret_cast<uint8_t *>(debug_cta0_a.data_ptr()),
        .debug_cta1_a_ptr = reinterpret_cast<uint8_t *>(debug_cta1_a.data_ptr()),
        .debug_cta0_sc_ptr = reinterpret_cast<uint8_t *>(debug_cta0_sc.data_ptr()),
        .debug_cta1_sc_ptr = reinterpret_cast<uint8_t *>(debug_cta1_sc.data_ptr()),
        .debug_a_stride = static_cast<int>(debug_cta0_a.size(1)),
    };
    kittens::py::launch_kernel<C, G, nvfp4_fused_gemm_both_bf16::transport_debug_kernel<C>>(g);
}

template <typename C>
static inline void launch_fused_gemm_both_bf16_clustered_transport_debug_with_config(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &debug_a_owner,
    at::Tensor &debug_a_recv,
    at::Tensor &debug_a_owner_sc,
    at::Tensor &debug_a_recv_sc,
    at::Tensor &debug_b_owner,
    at::Tensor &debug_b_recv,
    at::Tensor &debug_b_owner_sc,
    at::Tensor &debug_b_recv_sc
) {
    using G = nvfp4_fused_gemm_both_bf16::clustered_transport_debug_globals<C>;
    G g{
        .A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16),
        .B_bf16 = kittens::py::tensor_to_gl<typename G::B_bf16_gl>(B_bf16),
        .debug_a_owner_ptr = reinterpret_cast<uint8_t *>(debug_a_owner.data_ptr()),
        .debug_a_recv_ptr = reinterpret_cast<uint8_t *>(debug_a_recv.data_ptr()),
        .debug_a_owner_sc_ptr = reinterpret_cast<uint8_t *>(debug_a_owner_sc.data_ptr()),
        .debug_a_recv_sc_ptr = reinterpret_cast<uint8_t *>(debug_a_recv_sc.data_ptr()),
        .debug_b_owner_ptr = reinterpret_cast<uint8_t *>(debug_b_owner.data_ptr()),
        .debug_b_recv_ptr = reinterpret_cast<uint8_t *>(debug_b_recv.data_ptr()),
        .debug_b_owner_sc_ptr = reinterpret_cast<uint8_t *>(debug_b_owner_sc.data_ptr()),
        .debug_b_recv_sc_ptr = reinterpret_cast<uint8_t *>(debug_b_recv_sc.data_ptr()),
        .debug_stride = static_cast<int>(debug_a_owner.size(1)),
    };
    kittens::py::launch_kernel<C, G, nvfp4_fused_gemm_both_bf16::clustered_transport_debug_kernel<C>>(g);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 2, 4, 4>>(
            A_bf16, B_bf16, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_b_2cta_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<
            USE_CTA_AMAX, 1, 2, 4, 2, 2, false, false, false, true, false, true, 128>>(
            A_bf16, B_bf16, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    int debug_front_half_mode = 0
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<
            USE_CTA_AMAX, 1, 2, 4, 2, 2, false, false, false, true, false, true, 128>>(
            A_bf16, B_bf16, D,
            nullptr, nullptr, nullptr, nullptr,
            true, false, debug_front_half_mode);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_cluster_2x2_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<
            USE_CTA_AMAX, 1, 1, 1, 1, 4, false, false, false, false, true>>(
            A_bf16, B_bf16, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_cluster_2x2_transport_only_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<
            USE_CTA_AMAX, 1, 1, 1, 1, 4, false, false, false, false, true>>(
            A_bf16, B_bf16, D,
            nullptr, nullptr, nullptr, nullptr,
            true, false);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_quadcol_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 4, 1, 1, 1>>(
            A_bf16, B_bf16, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true>>(
            A_bf16, B_bf16, D);
}

template <typename C>
static inline void check_both_bf16_shared_a_transport_dump_tensor_shapes(
    const at::Tensor &debug_cta0_a,
    const at::Tensor &debug_cta1_a,
    const at::Tensor &debug_cta0_sc,
    const at::Tensor &debug_cta1_sc
) {
    const int expected_a_rows = C::Mb;
    const int expected_a_cols = C::Kb / 2;
    const int expected_sc_bytes = sizeof(typename nvfp4_fused_gemm_both_bf16::transport_debug_globals<C>::A_sc_tile);
    TORCH_CHECK(debug_cta0_a.is_cuda() && debug_cta1_a.is_cuda(), "debug A dump tensors must be CUDA tensors");
    TORCH_CHECK(debug_cta0_sc.is_cuda() && debug_cta1_sc.is_cuda(), "debug scale dump tensors must be CUDA tensors");
    TORCH_CHECK(debug_cta0_a.is_contiguous() && debug_cta1_a.is_contiguous(),
                "debug A dump tensors must be contiguous");
    TORCH_CHECK(debug_cta0_sc.is_contiguous() && debug_cta1_sc.is_contiguous(),
                "debug scale dump tensors must be contiguous");
    TORCH_CHECK(debug_cta0_a.dtype() == at::kByte && debug_cta1_a.dtype() == at::kByte,
                "debug A dump tensors must be uint8");
    TORCH_CHECK(debug_cta0_sc.dtype() == at::kByte && debug_cta1_sc.dtype() == at::kByte,
                "debug scale dump tensors must be uint8");
    TORCH_CHECK(debug_cta0_a.dim() == 2 && debug_cta1_a.dim() == 2,
                "debug A dump tensors must be 2D");
    TORCH_CHECK(debug_cta0_a.size(0) == expected_a_rows && debug_cta0_a.size(1) == expected_a_cols,
                "debug_cta0_a must have shape [", expected_a_rows, ", ", expected_a_cols, "]");
    TORCH_CHECK(debug_cta1_a.size(0) == expected_a_rows && debug_cta1_a.size(1) == expected_a_cols,
                "debug_cta1_a must have shape [", expected_a_rows, ", ", expected_a_cols, "]");
    TORCH_CHECK(debug_cta0_sc.numel() == expected_sc_bytes && debug_cta1_sc.numel() == expected_sc_bytes,
                "debug scale dump tensors must each have ", expected_sc_bytes, " bytes");
}

template <typename C>
static inline void check_both_bf16_clustered_transport_dump_tensor_shapes(
    const at::Tensor &debug_a_owner,
    const at::Tensor &debug_a_recv,
    const at::Tensor &debug_a_owner_sc,
    const at::Tensor &debug_a_recv_sc,
    const at::Tensor &debug_b_owner,
    const at::Tensor &debug_b_recv,
    const at::Tensor &debug_b_owner_sc,
    const at::Tensor &debug_b_recv_sc
) {
    const int expected_rows = C::Mb;
    const int expected_cols = C::Kb / 2;
    const int expected_sc_bytes =
        sizeof(typename nvfp4_fused_gemm_both_bf16::clustered_transport_debug_globals<C>::A_sc_tile);
    auto check_fp4_tensor = [&](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.dtype() == at::kByte, name, " must be uint8");
        TORCH_CHECK(tensor.dim() == 2, name, " must be 2D");
        TORCH_CHECK(tensor.size(0) == expected_rows && tensor.size(1) == expected_cols,
                    name, " must have shape [", expected_rows, ", ", expected_cols, "]");
    };
    auto check_sc_tensor = [&](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.dtype() == at::kByte, name, " must be uint8");
        TORCH_CHECK(tensor.numel() == expected_sc_bytes,
                    name, " must have ", expected_sc_bytes, " bytes");
    };
    check_fp4_tensor(debug_a_owner, "debug_a_owner");
    check_fp4_tensor(debug_a_recv, "debug_a_recv");
    check_fp4_tensor(debug_b_owner, "debug_b_owner");
    check_fp4_tensor(debug_b_recv, "debug_b_recv");
    check_sc_tensor(debug_a_owner_sc, "debug_a_owner_sc");
    check_sc_tensor(debug_a_recv_sc, "debug_a_recv_sc");
    check_sc_tensor(debug_b_owner_sc, "debug_b_owner_sc");
    check_sc_tensor(debug_b_recv_sc, "debug_b_recv_sc");
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_cluster_2x2_debug_dump(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &debug_a_owner,
    at::Tensor &debug_a_recv,
    at::Tensor &debug_a_owner_sc,
    at::Tensor &debug_a_recv_sc,
    at::Tensor &debug_b_owner,
    at::Tensor &debug_b_recv,
    at::Tensor &debug_b_owner_sc,
    at::Tensor &debug_b_recv_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::clustered_transport_debug_config<USE_CTA_AMAX>;
    check_both_bf16_clustered_transport_dump_tensor_shapes<DebugConfig>(
        debug_a_owner, debug_a_recv, debug_a_owner_sc, debug_a_recv_sc,
        debug_b_owner, debug_b_recv, debug_b_owner_sc, debug_b_recv_sc);
    launch_fused_gemm_both_bf16_clustered_transport_debug_with_config<DebugConfig>(
        A_bf16, B_bf16,
        debug_a_owner, debug_a_recv, debug_a_owner_sc, debug_a_recv_sc,
        debug_b_owner, debug_b_recv, debug_b_owner_sc, debug_b_recv_sc);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::transport_debug_config<USE_CTA_AMAX>;
    check_both_bf16_shared_a_transport_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_transport_debug_with_config<DebugConfig>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::transport_debug_config<USE_CTA_AMAX, true>;
    check_both_bf16_shared_a_transport_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_transport_debug_with_config<DebugConfig>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::transport_debug_config<USE_CTA_AMAX, false, true>;
    check_both_bf16_shared_a_transport_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_transport_debug_with_config<DebugConfig>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::transport_debug_config<USE_CTA_AMAX, true, true>;
    check_both_bf16_shared_a_transport_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_transport_debug_with_config<DebugConfig>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

template <typename C>
static inline void check_both_bf16_shared_a_dump_tensor_shapes(
    const at::Tensor &debug_cta0_a,
    const at::Tensor &debug_cta1_a,
    const at::Tensor &debug_cta0_sc,
    const at::Tensor &debug_cta1_sc
) {
    const int expected_a_rows = C::Mb;
    const int expected_a_cols = C::Kb / 2;
    const int expected_sc_bytes = sizeof(typename nvfp4_fused_gemm_both_bf16::globals<C>::A_sc_tile);
    TORCH_CHECK(debug_cta0_a.is_cuda() && debug_cta1_a.is_cuda(), "debug A dump tensors must be CUDA tensors");
    TORCH_CHECK(debug_cta0_sc.is_cuda() && debug_cta1_sc.is_cuda(), "debug scale dump tensors must be CUDA tensors");
    TORCH_CHECK(debug_cta0_a.is_contiguous() && debug_cta1_a.is_contiguous(),
                "debug A dump tensors must be contiguous");
    TORCH_CHECK(debug_cta0_sc.is_contiguous() && debug_cta1_sc.is_contiguous(),
                "debug scale dump tensors must be contiguous");
    TORCH_CHECK(debug_cta0_a.dtype() == at::kByte && debug_cta1_a.dtype() == at::kByte,
                "debug A dump tensors must be uint8");
    TORCH_CHECK(debug_cta0_sc.dtype() == at::kByte && debug_cta1_sc.dtype() == at::kByte,
                "debug scale dump tensors must be uint8");
    TORCH_CHECK(debug_cta0_a.dim() == 2 && debug_cta1_a.dim() == 2,
                "debug A dump tensors must be 2D");
    TORCH_CHECK(debug_cta0_a.size(0) == expected_a_rows && debug_cta0_a.size(1) == expected_a_cols,
                "debug_cta0_a must have shape [", expected_a_rows, ", ", expected_a_cols, "]");
    TORCH_CHECK(debug_cta1_a.size(0) == expected_a_rows && debug_cta1_a.size(1) == expected_a_cols,
                "debug_cta1_a must have shape [", expected_a_rows, ", ", expected_a_cols, "]");
    TORCH_CHECK(debug_cta0_sc.numel() == expected_sc_bytes && debug_cta1_sc.numel() == expected_sc_bytes,
                "debug scale dump tensors must each have ", expected_sc_bytes, " bytes");
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_local_a_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true, true>>(
            A_bf16, B_bf16, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    launch_fused_gemm_both_bf16_with_config<
        nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true, true, true>>(
            A_bf16, B_bf16, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_debug_dump(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true>;
    check_both_bf16_shared_a_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_with_config<DebugConfig>(
        A_bf16, B_bf16, D,
        &debug_cta0_a, &debug_cta1_a, &debug_cta0_sc, &debug_cta1_sc,
        false, true);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true>;
    check_both_bf16_shared_a_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_with_config<DebugConfig>(
        A_bf16, B_bf16, D,
        &debug_cta0_a, &debug_cta1_a, &debug_cta0_sc, &debug_cta1_sc,
        true);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true, true>;
    check_both_bf16_shared_a_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_with_config<DebugConfig>(
        A_bf16, B_bf16, D,
        &debug_cta0_a, &debug_cta1_a, &debug_cta0_sc, &debug_cta1_sc,
        true);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig = nvfp4_fused_gemm_both_bf16::config<USE_CTA_AMAX, 1, 1, 8, 1, 2, true, true>;
    check_both_bf16_shared_a_dump_tensor_shapes<DebugConfig>(
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    launch_fused_gemm_both_bf16_with_config<DebugConfig>(
        A_bf16, B_bf16, D,
        &debug_cta0_a, &debug_cta1_a, &debug_cta0_sc, &debug_cta1_sc,
        false, true);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_shared_a_debug(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D,
    at::Tensor *debug_cta0_a = nullptr,
    at::Tensor *debug_cta1_a = nullptr,
    at::Tensor *debug_cta0_sc = nullptr,
    at::Tensor *debug_cta1_sc = nullptr
) {
    launch_fused_gemm_with_config<
        nvfp4_fused_gemm::config<256, 2, 8, 4, 2, false, USE_CTA_AMAX, 128, true, 2, 1, 1, true>>(
            A_bf16, B, B_sc, B_sc_global, D,
            debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

template <typename C>
static inline void check_shared_a_dump_tensor_shapes(
    const at::Tensor &debug_cta0_a,
    const at::Tensor &debug_cta1_a,
    const at::Tensor &debug_cta0_sc,
    const at::Tensor &debug_cta1_sc
) {
    const int expected_a_rows = C::ROW_SLICE;
    const int expected_a_cols = C::Kb / 2;
    const int expected_sc_bytes = sizeof(typename nvfp4_fused_gemm::globals<C>::A_sc_tile);
    TORCH_CHECK(debug_cta0_a.is_cuda() && debug_cta1_a.is_cuda(), "debug A dump tensors must be CUDA tensors");
    TORCH_CHECK(debug_cta0_sc.is_cuda() && debug_cta1_sc.is_cuda(), "debug scale dump tensors must be CUDA tensors");
    TORCH_CHECK(debug_cta0_a.is_contiguous() && debug_cta1_a.is_contiguous(),
                "debug A dump tensors must be contiguous");
    TORCH_CHECK(debug_cta0_sc.is_contiguous() && debug_cta1_sc.is_contiguous(),
                "debug scale dump tensors must be contiguous");
    TORCH_CHECK(debug_cta0_a.dtype() == at::kByte && debug_cta1_a.dtype() == at::kByte,
                "debug A dump tensors must be uint8");
    TORCH_CHECK(debug_cta0_sc.dtype() == at::kByte && debug_cta1_sc.dtype() == at::kByte,
                "debug scale dump tensors must be uint8");
    TORCH_CHECK(debug_cta0_a.dim() == 2 && debug_cta1_a.dim() == 2,
                "debug A dump tensors must be 2D");
    TORCH_CHECK(debug_cta0_a.size(0) == expected_a_rows && debug_cta0_a.size(1) == expected_a_cols,
                "debug_cta0_a must have shape [", expected_a_rows, ", ", expected_a_cols, "]");
    TORCH_CHECK(debug_cta1_a.size(0) == expected_a_rows && debug_cta1_a.size(1) == expected_a_cols,
                "debug_cta1_a must have shape [", expected_a_rows, ", ", expected_a_cols, "]");
    TORCH_CHECK(debug_cta0_sc.numel() == expected_sc_bytes && debug_cta1_sc.numel() == expected_sc_bytes,
                "debug scale dump tensors must each have ", expected_sc_bytes, " bytes");
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

void nvfp4_fused_gemm_both_bf16_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    const int M = A_bf16.size(0);
    const int K = A_bf16.size(1);
    const int N = B_bf16.size(0);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bf16");
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(A_bf16.is_contiguous() && B_bf16.is_contiguous() && D.is_contiguous(),
                "A, B, and D must be contiguous");
    TORCH_CHECK(B_bf16.size(1) == K, "B second dimension must match A second dimension");
    TORCH_CHECK(D.size(0) == M && D.size(1) == N, "D must have shape [M, N]");
    TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");

    dispatch_fused_gemm_both_bf16<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    const int M = A_bf16.size(0);
    const int K = A_bf16.size(1);
    const int N = B_bf16.size(0);

    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B must be bf16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bf16");
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA tensors");
    TORCH_CHECK(A_bf16.is_contiguous() && B_bf16.is_contiguous() && D.is_contiguous(),
                "A, B, and D must be contiguous");
    TORCH_CHECK(B_bf16.size(1) == K, "B second dimension must match A second dimension");
    TORCH_CHECK(D.size(0) == M && D.size(1) == N, "D must have shape [M, N]");
    TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");

    dispatch_fused_gemm_both_bf16<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA transport-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_producer_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA producer-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 1);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 2);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA B-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 3);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 4);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA B-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 5);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 6);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA B-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 7);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_stage1_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-stage1-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 8);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_stage1_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA B-stage1-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 9);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_per_stage_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-quant-per-stage-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 10);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_per_stage_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA B-quant-per-stage-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 11);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-quant-then-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 12);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_then_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA B-quant-then-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 13);
}

void nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_skip_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA A-quant-then-skip-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<false>(A_bf16, B_bf16, D, 14);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_transport_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA transport-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_producer_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax producer-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 1);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 2);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax B-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 3);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 4);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax B-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 5);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 6);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax B-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 7);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_stage1_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-stage1-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 8);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_stage1_quant_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax B-stage1-quant-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 9);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_per_stage_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-quant-per-stage-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 10);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_per_stage_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax B-quant-per-stage-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 11);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-quant-then-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 12);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_then_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax B-quant-then-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 13);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_skip_wait_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && B_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 128 == 0,
                "shared-B 2CTA CTA-amax A-quant-then-skip-wait-only debug path requires M/N multiples of 256 and K multiple of 128");
    dispatch_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug<true>(A_bf16, B_bf16, D, 14);
}

void nvfp4_fused_gemm_both_bf16_quadcol_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 512 == 0,
                "quad-column debug path requires M multiple of 128, K multiple of 256, N multiple of 512");
    dispatch_fused_gemm_both_bf16_quadcol_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 512 == 0,
                "quad-column debug path requires M multiple of 128, K multiple of 256, N multiple of 512");
    dispatch_fused_gemm_both_bf16_quadcol_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "clustered 2x2 debug path requires M, N, and K to be multiples of 256");
    dispatch_fused_gemm_both_bf16_cluster_2x2_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "clustered 2x2 debug path requires M, N, and K to be multiples of 256");
    dispatch_fused_gemm_both_bf16_cluster_2x2_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cluster_2x2_transport_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "clustered 2x2 transport-only debug path requires M, N, and K to be multiples of 256");
    dispatch_fused_gemm_both_bf16_cluster_2x2_transport_only_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_transport_only_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 256 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "clustered 2x2 transport-only debug path requires M, N, and K to be multiples of 256");
    dispatch_fused_gemm_both_bf16_cluster_2x2_transport_only_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &debug_a_owner,
    at::Tensor &debug_a_recv,
    at::Tensor &debug_a_owner_sc,
    at::Tensor &debug_a_recv_sc,
    at::Tensor &debug_b_owner,
    at::Tensor &debug_b_recv,
    at::Tensor &debug_b_owner_sc,
    at::Tensor &debug_b_recv_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16 && B_bf16.dtype() == at::kBFloat16,
                "A and B must be bfloat16");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256 &&
                B_bf16.size(0) == 256 && B_bf16.size(1) == 256,
                "clustered 2x2 dump helper is fixed to 256 x 256 x 256");
    dispatch_fused_gemm_both_bf16_cluster_2x2_debug_dump<false>(
        A_bf16, B_bf16,
        debug_a_owner, debug_a_recv, debug_a_owner_sc, debug_a_recv_sc,
        debug_b_owner, debug_b_recv, debug_b_owner_sc, debug_b_recv_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &debug_a_owner,
    at::Tensor &debug_a_recv,
    at::Tensor &debug_a_owner_sc,
    at::Tensor &debug_a_recv_sc,
    at::Tensor &debug_b_owner,
    at::Tensor &debug_b_recv,
    at::Tensor &debug_b_owner_sc,
    at::Tensor &debug_b_recv_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda(), "A and B must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16 && B_bf16.dtype() == at::kBFloat16,
                "A and B must be bfloat16");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256 &&
                B_bf16.size(0) == 256 && B_bf16.size(1) == 256,
                "clustered 2x2 dump helper is fixed to 256 x 256 x 256");
    dispatch_fused_gemm_both_bf16_cluster_2x2_debug_dump<true>(
        A_bf16, B_bf16,
        debug_a_owner, debug_a_recv, debug_a_owner_sc, debug_a_recv_sc,
        debug_b_owner, debug_b_recv, debug_b_owner_sc, debug_b_recv_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "shared-A 2CTA debug path requires M multiple of 128, K multiple of 256, N multiple of 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "shared-A 2CTA debug path requires M multiple of 128, K multiple of 256, N multiple of 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "shared-A 2CTA local-A debug path requires M multiple of 128, K multiple of 256, N multiple of 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_local_a_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "shared-A 2CTA local-A debug path requires M multiple of 128, K multiple of 256, N multiple of 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_local_a_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "shared-A 2CTA publish-only local-A debug path requires M multiple of 128, K multiple of 256, N multiple of 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug<false>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_publish_only_local_a_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) % 128 == 0 && A_bf16.size(1) % 256 == 0 && B_bf16.size(0) % 256 == 0,
                "shared-A 2CTA publish-only local-A debug path requires M multiple of 128, K multiple of 256, N multiple of 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug<true>(A_bf16, B_bf16, D);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_debug_dump<false>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_debug_dump<true>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A full-transport dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump<false>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A full-transport dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump<true>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A full-transport local-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump<false>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A full-transport local-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump<true>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A local-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump<false>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B_bf16,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda() && B_bf16.is_cuda() && D.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(B_bf16.dtype() == at::kBFloat16, "B_bf16 must be bfloat16");
    TORCH_CHECK(D.dtype() == at::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2 && B_bf16.dim() == 2 && D.dim() == 2, "all tensors must be 2D");
    TORCH_CHECK(A_bf16.size(1) == B_bf16.size(1), "A and B inner dimensions must match");
    TORCH_CHECK(A_bf16.size(0) == D.size(0), "D rows must match A rows");
    TORCH_CHECK(B_bf16.size(0) == D.size(1), "D cols must match B rows");
    TORCH_CHECK(A_bf16.size(0) == 256 && B_bf16.size(0) == 512 && A_bf16.size(1) == 256,
                "both-bf16 shared-A local-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump<true>(
        A_bf16, B_bf16, D, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump<false>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump<true>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport local-A dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump<false>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport local-A dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump<true>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport import dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump<false>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport import dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump<true>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport import local-A dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump<false>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_local_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    TORCH_CHECK(A_bf16.is_cuda(), "A_bf16 must be CUDA");
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A_bf16 must be bfloat16");
    TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
    TORCH_CHECK(A_bf16.size(0) == 256 && A_bf16.size(1) == 256,
                "both-bf16 shared-A transport import local-A dump helper is fixed to A shape 256 x 256");
    dispatch_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump<true>(
        A_bf16, debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

template <bool USE_CTA_AMAX>
static inline void check_shared_a_debug_inputs(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    const at::Tensor &D
) {
    const int M = A_bf16.size(0);
    const int K = A_bf16.size(1);
    const int N = D.size(1);
    TORCH_CHECK(A_bf16.dtype() == at::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == at::kFloat4_e2m1fn_x2, "B must be fp4x2");
    TORCH_CHECK(B_sc_global.dtype() == at::kFloat, "B_sc_global must be float32");
    TORCH_CHECK(M % 256 == 0, "M must be multiple of 256");
    TORCH_CHECK(K % 256 == 0, "K must be multiple of 256");
    TORCH_CHECK(N % 512 == 0, "Shared-A debug backend requires N to be a multiple of 512");
}

void nvfp4_fused_gemm_shared_a_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    check_shared_a_debug_inputs<false>(A_bf16, B, B_sc, B_sc_global, D);
    dispatch_fused_gemm_shared_a_debug<false>(A_bf16, B, B_sc, B_sc_global, D);
}

void nvfp4_fused_gemm_cta_amax_shared_a_debug_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D
) {
    check_shared_a_debug_inputs<true>(A_bf16, B, B_sc, B_sc_global, D);
    dispatch_fused_gemm_shared_a_debug<true>(A_bf16, B, B_sc, B_sc_global, D);
}

template <bool USE_CTA_AMAX>
static inline void dispatch_fused_gemm_shared_a_debug_dump(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    using DebugConfig =
        nvfp4_fused_gemm::config<256, 2, 8, 4, 2, false, USE_CTA_AMAX, 128, true, 2, 1, 1, true>;
    check_shared_a_dump_tensor_shapes<DebugConfig>(debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
    dispatch_fused_gemm_shared_a_debug<USE_CTA_AMAX>(
        A_bf16, B, B_sc, B_sc_global, D,
        &debug_cta0_a, &debug_cta1_a, &debug_cta0_sc, &debug_cta1_sc);
}

void nvfp4_fused_gemm_shared_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    check_shared_a_debug_inputs<false>(A_bf16, B, B_sc, B_sc_global, D);
    TORCH_CHECK(A_bf16.size(0) == 256 && D.size(1) == 512 && A_bf16.size(1) == 256,
                "Shared-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_shared_a_debug_dump<false>(
        A_bf16, B, B_sc, B_sc_global, D,
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
}

void nvfp4_fused_gemm_cta_amax_shared_a_debug_dump_entrypoint(
    const at::Tensor &A_bf16,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sc_global,
    at::Tensor &D,
    at::Tensor &debug_cta0_a,
    at::Tensor &debug_cta1_a,
    at::Tensor &debug_cta0_sc,
    at::Tensor &debug_cta1_sc
) {
    check_shared_a_debug_inputs<true>(A_bf16, B, B_sc, B_sc_global, D);
    TORCH_CHECK(A_bf16.size(0) == 256 && D.size(1) == 512 && A_bf16.size(1) == 256,
                "Shared-A dump helper is fixed to shape 256 x 512 x 256");
    dispatch_fused_gemm_shared_a_debug_dump<true>(
        A_bf16, B, B_sc, B_sc_global, D,
        debug_cta0_a, debug_cta1_a, debug_cta0_sc, debug_cta1_sc);
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
    m.def("nvfp4_fused_gemm_both_bf16", &nvfp4_fused_gemm_both_bf16_entrypoint,
          "Experimental fused GEMM: quantize both bf16 operands on the fly (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax", &nvfp4_fused_gemm_both_bf16_cta_amax_entrypoint,
          "Experimental fused GEMM: quantize both bf16 operands on the fly (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B experiment (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B experiment (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B front-half bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_producer_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_producer_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B producer-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_wait_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-wait-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_wait_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-wait-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-quant-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-quant-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_stage1_quant_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_stage1_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-stage1-only quant bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_stage1_quant_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_stage1_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-stage1-only quant bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_per_stage_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_per_stage_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-quant with per-stage destination slots (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_per_stage_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_per_stage_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-quant with per-stage destination slots (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_wait_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B quant once then only wait stage 1 (A path, constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_then_wait_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_then_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B quant once then only wait stage 1 (B path, constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_skip_wait_only_debug", &nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_skip_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B quant once then skip stage-1 raw wait (A path, constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_transport_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_transport_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B front-half bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_producer_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_producer_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B producer-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_wait_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-wait-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_wait_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-wait-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-quant-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-quant-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_stage1_quant_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_stage1_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-stage1-only quant bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_stage1_quant_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_stage1_quant_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-stage1-only quant bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_per_stage_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_per_stage_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B A-quant with per-stage destination slots (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_per_stage_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_per_stage_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B B-quant with per-stage destination slots (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_wait_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B quant once then only wait stage 1 (A path, CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_then_wait_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_then_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B quant once then only wait stage 1 (B path, CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_skip_wait_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_skip_wait_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2CTA shared-B quant once then skip stage-1 raw wait (A path, CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_quadcol_debug", &nvfp4_fused_gemm_both_bf16_quadcol_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: same-CTA quad-column A-reuse experiment (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: same-CTA quad-column A-reuse experiment (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cluster_2x2_debug", &nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2x2 clustered A/B reuse experiment (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2x2 clustered A/B reuse experiment (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cluster_2x2_transport_only_debug", &nvfp4_fused_gemm_both_bf16_cluster_2x2_transport_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2x2 clustered A/B reuse transport-only bring-up (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_transport_only_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_transport_only_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: 2x2 clustered A/B reuse transport-only bring-up (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_dump", &nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_dump_entrypoint,
          "Dump helper for 2x2 clustered both-bf16 A/B transport (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("debug_a_owner"), pybind11::arg("debug_a_recv"),
          pybind11::arg("debug_a_owner_sc"), pybind11::arg("debug_a_recv_sc"),
          pybind11::arg("debug_b_owner"), pybind11::arg("debug_b_recv"),
          pybind11::arg("debug_b_owner_sc"), pybind11::arg("debug_b_recv_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_dump_entrypoint,
          "Dump helper for 2x2 clustered both-bf16 A/B transport (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("debug_a_owner"), pybind11::arg("debug_a_recv"),
          pybind11::arg("debug_a_owner_sc"), pybind11::arg("debug_a_recv_sc"),
          pybind11::arg("debug_b_owner"), pybind11::arg("debug_b_recv"),
          pybind11::arg("debug_b_owner_sc"), pybind11::arg("debug_b_recv_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA shared-A experiment (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA shared-A experiment (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA debug backend with CTA1 local A quantization (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA debug backend with CTA1 local A quantization (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA debug backend with remote publish enabled and CTA1 local A consumption (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_publish_only_local_a_debug", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_publish_only_local_a_debug_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA debug backend with remote publish enabled and CTA1 local A consumption (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA shared-A dump helper (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA shared-A dump helper (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: full-kernel transport-only shared-A dump helper (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: full-kernel transport-only shared-A dump helper (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: full-kernel transport-only local-A dump helper (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: full-kernel transport-only local-A dump helper (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA local-A dump helper (constant scale)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 fused GEMM: cross-CTA local-A dump helper (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B_bf16"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: cross-CTA shared-A payload dump helper (constant scale)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: cross-CTA shared-A payload dump helper (CTA amax)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: local-A control dump helper (constant scale)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: local-A control dump helper (CTA amax)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: shared-A receive+import dump helper (constant scale)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: shared-A receive+import dump helper (CTA amax)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: local-A receive+import control dump helper (constant scale)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_local_a_debug_dump", &nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_local_a_debug_dump_entrypoint,
          "Developer-only both-bf16 transport microkernel: local-A receive+import control dump helper (CTA amax)",
          pybind11::arg("A_bf16"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_shared_a_debug", &nvfp4_fused_gemm_shared_a_debug_entrypoint,
          "Developer-only isolated cross-CTA shared-A fused GEMM",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_cta_amax_shared_a_debug", &nvfp4_fused_gemm_cta_amax_shared_a_debug_entrypoint,
          "Developer-only isolated cross-CTA shared-A fused GEMM (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"));
    m.def("nvfp4_fused_gemm_shared_a_debug_dump", &nvfp4_fused_gemm_shared_a_debug_dump_entrypoint,
          "Developer-only isolated cross-CTA shared-A dump helper for 256x512x256",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));
    m.def("nvfp4_fused_gemm_cta_amax_shared_a_debug_dump", &nvfp4_fused_gemm_cta_amax_shared_a_debug_dump_entrypoint,
          "Developer-only isolated cross-CTA shared-A dump helper for 256x512x256 (CTA amax)",
          pybind11::arg("A_bf16"), pybind11::arg("B"),
          pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("D"),
          pybind11::arg("debug_cta0_a"), pybind11::arg("debug_cta1_a"),
          pybind11::arg("debug_cta0_sc"), pybind11::arg("debug_cta1_sc"));

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
