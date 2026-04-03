#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// Level 11: Kernel Parameter Tuning
//
// This level is identical to Level 10's architecture (persistent CLC kernel)
// but exposes all tunable parameters as compile-time constants. The goal is
// to build with different configurations and observe their effect on
// throughput, register usage, and SMEM consumption.
//
// Build with:
//   make clean && make run
//
// To sweep configs, modify the constants in the "TUNING KNOBS" section below.
//
// ============================================================================
//
// === TUNING KNOBS ===
// Try different values and rebuild to see the effect!
//
// Knob               | What it controls                           | Recommended range
// ------------------- ------------------------------------------- -----------------
// TILE_M             | Output tile rows (per consumer)            | 64, 128
// TILE_N             | Output tile cols (total)                   | 128, 256
// TILE_K             | K-dimension per iteration                  | 32, 64, 128
// PIPE_STAGES        | TMA load pipeline depth                   | 2, 3, 4, 5, 6, 7
// CLUSTER_SIZE       | CTAs per cluster                           | 1, 2
// NUM_CONSUMERS      | Consumer warpgroups per CTA                | 1, 2
// EPI_PIPE_DEPTH     | Epilogue slices (TMEM→SMEM→GMEM chunks)   | 1, 2, 4, 8
// NUM_D_TILES        | SMEM double-buffering for stores           | 1, 2
// SUPERGROUP_SIZE    | L2 swizzle group width                     | 1, 2, 4, 8, 16
// ============================================================================

// ===== TUNING KNOBS — MODIFY THESE =====
// Config A: "Wide & Deep" — our Level 10 default
//   128×256 tiles, 4-stage pipe, 2 consumers, 8 epilogue slices
#ifndef CONFIG
#define CONFIG 0
#endif

#if CONFIG == 0
// Level 10 default (known good: ~1346 TFLOPS)
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 1
// "Deeper pipeline with narrow tile" — narrower N + 5-stage pipe
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;             // ← halved to free SMEM
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;          // ← 4 stages fits
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 1;        // ← 1 consumer = less SMEM for A tiles
static constexpr int EPI_PIPE_DEPTH = 4;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 2
// "Wider K" — fewer iters, more work per iter
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 128;             // ← doubled from 64
static constexpr int PIPE_STAGES = 2;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 3
// "Single consumer" — fewer warpgroups, simpler sync
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 1;        // ← only 1 consumer
static constexpr int EPI_PIPE_DEPTH = 4;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 4
// "No cluster" — single CTA, no cluster overhead
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 1;         // ← no cluster
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 5
// "Narrow tile" — smaller N output, more grid parallelism
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;             // ← halved
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 4;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 6
// "Production N=4096" — matches bf16_b200 production config exactly
//   config<256, 256, 64, 4, false, 4, 8>
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 7
// "Shallow pipe" — minimal pipeline stages
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 2;          // ← minimal
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

#elif CONFIG == 8
// "Wide supergroup" — larger L2 tile swizzle
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 2;
static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 8;      // ← wider swizzle

#endif

static constexpr int MMA_PIPE_DEPTH = 1;
static constexpr int CLC_PIPE_DEPTH = 1;
static constexpr int NUM_WARPS = (NUM_CONSUMERS + 1) * 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

// ===== Derived tile types =====
using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N / CLUSTER_SIZE>;
using d_tile = st_bf<TILE_M, TILE_N / EPI_PIPE_DEPTH>;
using d_tt_t = tt<float, TILE_M, TILE_N>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

struct matmul_globals {
    a_gl a;
    b_gl b;
    d_gl d;
};

// ===== Kernel (same architecture as Level 10, parameterized) =====
__global__
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ matmul_globals g) {
    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<a_tile>();
        g.b.template prefetch_tma<b_tile>();
        g.d.template prefetch_tma<d_tile>();
    }

    const int cta_rank = CLUSTER_SIZE > 1 ? cluster_ctarank() : 0;
    const int iters_per_task = g.a.cols() / TILE_K;
    const int rblks = g.d.rows() / (CLUSTER_SIZE * NUM_CONSUMERS * TILE_M);
    const int cblks = g.d.cols() / TILE_N;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile (&a_smem)[PIPE_STAGES][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_STAGES, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_STAGES]                = al.allocate<b_tile, PIPE_STAGES>();
    d_tile (&d_smem)[NUM_CONSUMERS][NUM_D_TILES]  = al.allocate<d_tile, NUM_CONSUMERS, NUM_D_TILES>();

    tensor_allocator<1, CLUSTER_SIZE, false> tm_alloc{};

    __shared__ uint32_t tmem_addr;
    __shared__ clc::handle clc_handle[CLC_PIPE_DEPTH];
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore schedule_arrived[CLC_PIPE_DEPTH], schedule_finished[CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[PIPE_STAGES], inputs_finished[PIPE_STAGES];
    __shared__ semaphore outputs_arrived[NUM_CONSUMERS], outputs_finished[MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (2+NUM_CONSUMERS)*CLUSTER_SIZE+NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE*NUM_CONSUMERS);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    // ---- PRODUCER WARPGROUP ----
    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            int2 tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/CLUSTER_SIZE);
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < NUM_CONSUMERS; i++)
                        tma::cluster::load_async(a_smem[input_ring][i], g.a,
                            {(tile_coord.x*CLUSTER_SIZE+cta_rank)*NUM_CONSUMERS+i, idx},
                            inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b,
                        {idx, tile_coord.y*CLUSTER_SIZE+cta_rank},
                        inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring = ring_advance<PIPE_STAGES>(input_ring);
                }
                wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, schedule.x/CLUSTER_SIZE);
                else break;
            }
        }
        else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0) {
                    wait(schedule_finished[task_iter%CLC_PIPE_DEPTH], ((task_iter+CLC_PIPE_DEPTH)/CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%CLC_PIPE_DEPTH], schedule_arrived[task_iter%CLC_PIPE_DEPTH]);
                }
                tma::expect_bytes(schedule_arrived[task_iter%CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        }
        else if (cta_rank == 0 && warpgroup::warpid() < NUM_CONSUMERS && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < MMA_PIPE_DEPTH; i++)
                d_tt[i] = tm_alloc.allocate<d_tt_t>((i+warpgroup::warpid())*TILE_N);
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
                wait(outputs_finished[task_iter%MMA_PIPE_DEPTH], ((task_iter+MMA_PIPE_DEPTH)/MMA_PIPE_DEPTH)%2);
                for (int idx = 0; idx < iters_per_task; idx++) {
                    tma::expect_bytes(inputs_arrived[input_ring],
                        (CLUSTER_SIZE*NUM_CONSUMERS*sizeof(a_tile) + 2*sizeof(b_tile))/NUM_CONSUMERS);
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    if (idx == 0) mm2_AB (d_tt[task_iter%MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    else          mma2_AB(d_tt[task_iter%MMA_PIPE_DEPTH], a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    update_phasebit<0>(bitfield, input_ring);
                    input_ring = ring_advance<PIPE_STAGES>(input_ring);
                }
                detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
                if (!schedule.success) break;
            }
        }
    }
    // ---- CONSUMER/EPILOGUE WARPGROUPS ----
    else {
        using epilogue_group = group<WARPGROUP_WARPS*NUM_CONSUMERS>;
        warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();
        if (epilogue_group::warpid() == 0) {
            tm_alloc.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_alloc.set_addr(tmem_addr);
        d_tt_t d_tt[MMA_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < MMA_PIPE_DEPTH; i++)
            d_tt[i] = tm_alloc.allocate<d_tt_t>((i+warpgroup::groupid())*TILE_N);
        int2 tile_coord, next_tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/CLUSTER_SIZE);

        for (int task_iter = 0; true; task_iter++) {
            tile_coord = next_tile_coord;
            wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, schedule.x/CLUSTER_SIZE);
            wait(outputs_arrived[warpgroup::groupid()], task_iter%2);
            rt_bf<TILE_M/4, TILE_N/EPI_PIPE_DEPTH> d_reg[EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++)
                warpgroup::load_async(d_reg[i], d_tt[task_iter%MMA_PIPE_DEPTH].template subtile<tt<float, TILE_M, TILE_N/EPI_PIPE_DEPTH>>(0, TILE_N/EPI_PIPE_DEPTH*i));
            tensor_load_wait();
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(outputs_finished[task_iter%MMA_PIPE_DEPTH], 0);
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                warpgroup::tma::store_async_read_wait<NUM_D_TILES-1>();
                warpgroup::sync(warpgroup::groupid()+1);
                warpgroup::store(d_smem[warpgroup::groupid()][i%NUM_D_TILES], d_reg[i]);
                warpgroup::sync(warpgroup::groupid()+1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%NUM_D_TILES],
                    {(CLUSTER_SIZE*tile_coord.x+cta_rank)*NUM_CONSUMERS+warpgroup::groupid(), EPI_PIPE_DEPTH*tile_coord.y+i});
            }
            if (!schedule.success) break;
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

// ===== Host-side launch =====
void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    a_gl Ag{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl Bg{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    d_gl Dg{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    matmul_globals g{Ag, Bg, Dg};

    int grid = (N / (CLUSTER_SIZE * NUM_CONSUMERS * TILE_M)) * (N / TILE_N) * CLUSTER_SIZE;

    constexpr size_t smem_size = sizeof(a_tile) * PIPE_STAGES * NUM_CONSUMERS +
                                  sizeof(b_tile) * PIPE_STAGES +
                                  sizeof(d_tile) * NUM_D_TILES * NUM_CONSUMERS + 1024;
    static_assert(smem_size <= MAX_SHARED_MEMORY - 1024);

    // Print config info
    static bool printed = false;
    if (!printed) {
        printf("Config %d: TILE_M=%d, TILE_N=%d, TILE_K=%d, PIPE=%d, CLUSTER=%d, CONSUMERS=%d, EPI=%d, SMEM=%zuB\n",
            CONFIG, TILE_M, TILE_N, TILE_K, PIPE_STAGES, CLUSTER_SIZE, NUM_CONSUMERS, EPI_PIPE_DEPTH, smem_size);
        printed = true;
    }

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    LaunchConfig<true, false> launch_config(
        dim3(grid), dim3(NUM_THREADS), smem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, matmul_kernel, g);
}

#include "launch.cu"
