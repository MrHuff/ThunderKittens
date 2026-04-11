#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// YOUR TURN: Build a B200 GEMM From Scratch!
//
// This is a skeleton file for you to implement a high-performance BF16 GEMM
// on Blackwell (GB200) using ThunderKittens. Fill in each section marked with
// TODO. The comments explain what needs to go there and why.
//
// Goal: C[M×N] = A[M×K] × B[K×N],  all bf16, M=N=K=4096
//
// The launch harness (launch.cu) will:
//   1. Allocate A, B, C on the GPU
//   2. Fill A, B with random data
//   3. Call your matmul() function
//   4. Compare against a reference GEMM
//   5. Report TFLOPS and errors
//
// Suggested progression:
//   Stage 1: Get a simple warp-level MMA working (aim for ~50 TFLOPS)
//   Stage 2: Add TMA loads + pipeline (aim for ~300 TFLOPS)
//   Stage 3: Add warp specialization (aim for ~700 TFLOPS)
//   Stage 4: Add epilogue pipelining (aim for ~1000 TFLOPS)
//   Stage 5: Add 2-CTA cluster (aim for ~1200 TFLOPS)
//   Stage 6: Add CLC persistence (aim for ~1350 TFLOPS)
//
// Refer to WALKTHROUGH.md for detailed explanations of each concept.
// ============================================================================


// ===== STEP 1: Define your tile dimensions =====
// These control the output tile each CTA computes.
// Larger tiles → better compute/memory ratio, but need more SMEM/regs.
//
// Typical good values for B200:
//   TILE_M=128, TILE_N=256, TILE_K=64
//
// TODO: Choose your tile sizes
static constexpr int TILE_M = 128;   // output rows per consumer
static constexpr int TILE_N = 128;   // output cols (start smaller, then try 256)
static constexpr int TILE_K = 64;    // K dimension per iteration

// TODO: Choose pipeline depth (how many SMEM buffers for double/multi-buffering)
static constexpr int PIPE_STAGES = 3;

// TODO: How many threads? Start with 1 warpgroup (128 threads), then add more.
// Each warpgroup = 4 warps × 32 threads = 128 threads
static constexpr int NUM_THREADS = 128;


// ===== STEP 2: Define ThunderKittens tile types =====
// TK uses typed tiles instead of raw arrays.
//
// Shared tiles: st_bf<rows, cols> — lives in shared memory (SMEM)
// Register tiles: rt_bf<rows, cols> — lives in warp registers
// Tensor tiles: tt<float, rows, cols> — lives in TMEM (Blackwell only)
// Global layouts: gl<bf16, batch, depth, rows, cols, TileType...>
//
// The TileType parameter tells TK which TMA descriptors to create.

// TODO: Define your shared-memory tile types
using a_tile = st_bf<TILE_M, TILE_K>;    // A tile: TILE_M × TILE_K
using b_tile = st_bf<TILE_K, TILE_N>;    // B tile: TILE_K × TILE_N

// TODO: Define the output epilogue tile (for TMEM → SMEM → GMEM)
// For simple start: same as full output tile
using d_tile = st_bf<TILE_M, TILE_N>;

// TODO: Define TMEM accumulator type (Blackwell tensor cores accumulate in FP32)
using d_tt_t = tt<float, TILE_M, TILE_N>;

// TODO: Define global layout descriptors
// gl<elem_type, batch_dim, depth_dim, row_dim, col_dim, TilesToMakeTMADescFor...>
// Use -1 for runtime-determined dimensions
using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;


// ===== STEP 3: Write your GPU kernel =====
//
// Key TK operations you'll need (see WALKTHROUGH.md for details):
//
// Memory:
//   tma::load_async(smem_tile, global_layout, {row, col}, semaphore)
//   tma::expect_bytes(semaphore, nbytes)
//   tma::store_async(global_layout, smem_tile, {row, col})
//
// Synchronization:
//   init_semaphore(sem, thread_count, tx_count)
//   wait(sem, phase)
//   arrive(sem)
//
// Compute (warp-level WMMA — simpler start):
//   warp::load(reg, smem)
//   warp::mma_AB(dst_reg, a_reg, b_reg, acc_reg)
//   warp::store(global, reg, {coords})
//
// Compute (warpgroup-level tcgen05 — Blackwell, much faster):
//   mm_AB(tmem_acc, a_smem, b_smem, finished_sem)     // first iter: zero + mul
//   mma_AB(tmem_acc, a_smem, b_smem, finished_sem)    // accumulate
//   detail::tcgen05::commit<1>(done_sem)               // make TMEM results visible
//   warpgroup::load_async(reg, tmem_subtile)           // TMEM → registers
//   tensor_load_wait()                                 // wait for TMEM→reg
//   warpgroup::store(smem, reg)                        // registers → SMEM
//
// Cluster (advanced):
//   tma::cluster::load_async(smem, layout, coords, sem, mask, cta)
//   everyone::tma::cluster::sync()
//   cluster_ctarank()
//
// CLC (persistent, most advanced):
//   clc::schedule(handle, sem)
//   clc::query(handle) → {.success, .x, .y, .z}
//

__global__
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int N
) {
    // ---- STEP 3a: Prefetch TMA descriptors ----
    // Only thread 0 needs to do this. It loads the TMA descriptor into L2 cache.
    if (threadIdx.x == 0) {
        A_layout.template prefetch_tma<a_tile>();
        B_layout.template prefetch_tma<b_tile>();
        D_layout.template prefetch_tma<d_tile>();
    }

    // ---- STEP 3b: Figure out which tile this CTA computes ----
    // For the simplest version: blockIdx.x = row tile, blockIdx.y = col tile
    // TODO: Map blockIdx to (row_tile, col_tile) coordinates
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;
    int num_k_iters = N / TILE_K;

    // ---- STEP 3c: Allocate shared memory ----
    // TK uses a shared_allocator to manage dynamic SMEM:
    //   extern __shared__ int __shm[];
    //   tma_swizzle_allocator al((int*)&__shm[0]);
    //   a_tile &my_a = al.allocate<a_tile>();      // single tile
    //   a_tile (&arr)[3] = al.allocate<a_tile, 3>(); // array of 3
    //
    // TODO: Allocate your SMEM tiles
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    // For a simple single-buffered version:
    a_tile &a_smem = al.allocate<a_tile>();
    b_tile &b_smem = al.allocate<b_tile>();

    // TODO: Allocate semaphores
    __shared__ semaphore smem_arrived, smem_finished;

    // ---- STEP 3d: Initialize semaphores (ONE thread only) ----
    // init_semaphore(sem, thread_arrive_count, transaction_byte_count)
    // TODO: init your semaphores
    if (threadIdx.x == 0) {
        init_semaphore(smem_arrived, 0, 1);
        init_semaphore(smem_finished, 0, 0);
    }
    __syncthreads();

    // ---- STEP 3e: Allocate TMEM accumulator (Blackwell) ----
    // tensor_allocator<rows, cols> tm_alloc{};
    // d_tt_t accum = tm_alloc.allocate<d_tt_t>(offset);
    //
    // For WMMA (simpler): use rt_fl<M, N> accum in registers instead.
    //
    // TODO: Allocate your accumulator

    // ---- STEP 3f: Main compute loop over K ----
    // For each K iteration:
    //   1. Load A[tile_row, k] and B[k, tile_col] into SMEM
    //   2. Compute: accum += A_smem × B_smem
    //
    // TODO: Implement the main loop
    for (int k = 0; k < num_k_iters; k++) {

        // TODO: Issue TMA loads (thread 0 only)
        // tma::expect_bytes(smem_arrived, sizeof(a_tile) + sizeof(b_tile));
        // tma::load_async(a_smem, A_layout, {tile_row, k}, smem_arrived);
        // tma::load_async(b_smem, B_layout, {k, tile_col}, smem_arrived);

        // TODO: Wait for data to arrive
        // wait(smem_arrived, phase);

        // TODO: Compute MMA
        // For WMMA:
        //   warp::load(a_reg, a_smem);
        //   warp::mma_AB(c_reg, a_reg, b_reg, c_reg);
        // For tcgen05:
        //   if (k == 0) mm_AB(accum, a_smem, b_smem, smem_finished);
        //   else        mma_AB(accum, a_smem, b_smem, smem_finished);
    }

    // ---- STEP 3g: Store result back to GMEM ----
    // Path: TMEM → registers → SMEM → TMA store → GMEM
    // Or for WMMA: registers → GMEM directly
    //
    // TODO: Store your results
}


// ===== STEP 4: Host-side launch function =====
// The harness calls: void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N)
//
void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    // TODO: Create global layouts
    a_gl A_layout{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl B_layout{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    d_gl D_layout{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};

    // TODO: Calculate grid dimensions
    // Simple: one CTA per output tile
    dim3 grid(N / TILE_N, N / TILE_M);
    dim3 block(NUM_THREADS);

    // TODO: Calculate SMEM size
    constexpr size_t smem_size = sizeof(a_tile) + sizeof(b_tile) + 1024;

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // TODO: Launch kernel
    // Simple version:
    matmul_kernel<<<grid, block, smem_size>>>(A_layout, B_layout, D_layout, N);

    // Advanced (with cluster):
    // LaunchConfig<true, false> launch_config(grid, block, smem_size, 0, CLUSTER_SIZE);
    // cudaLaunchKernelEx(launch_config, matmul_kernel, A_layout, B_layout, D_layout, N);
}

#include "launch.cu"
