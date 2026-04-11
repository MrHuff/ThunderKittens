#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// Level 10: Persistent Kernel with CLC (Cluster Launch Control)
//
// This builds on Level 09's 2-CTA clustered design and makes it PERSISTENT.
// Instead of launching one CTA per tile and exiting, CTAs stay resident and
// loop over multiple tiles via Blackwell's CLC hardware scheduler.
//
// Key additions over Level 09:
//   1. CLC (clc::schedule / clc::query) — hardware work distributor
//   2. Persistent main loop — CTAs loop until CLC says "no more work"
//   3. outputs_finished semaphore — MMA waits for epilogue to drain TMEM
//   4. CLC scheduler warp (warp 2 of producer WG)
//   5. PDL (Programmatic Dependent Launch) — overlap next kernel launch
//   6. Supergroup-swizzled tile ordering for L2 cache locality
//   7. TMEM provision/deprovision lifecycle management
//   8. LaunchConfig with cluster + PDL attributes
// ============================================================================

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 256;
static constexpr int TILE_K = 64;
static constexpr int PIPE_STAGES = 4;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int CLC_PIPE_DEPTH = 1;
static constexpr int MMA_PIPE_DEPTH = 1;

static constexpr int EPI_PIPE_DEPTH = 8;
static constexpr int NUM_D_TILES = 2;
static constexpr int SUPERGROUP_SIZE = 4;

static constexpr int NUM_CONSUMERS = 2;
static constexpr int NUM_WARPS = (NUM_CONSUMERS + 1) * 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N / 2>;
using d_tile = st_bf<TILE_M, TILE_N / EPI_PIPE_DEPTH>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;

struct matmul_globals {
    a_gl a;
    b_gl b;
    d_gl d;
};

__global__
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ matmul_globals g) {
    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<a_tile>();
        g.b.template prefetch_tma<b_tile>();
        g.d.template prefetch_tma<d_tile>();
    }

    const int cta_rank = cluster_ctarank();
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

    // ================================================================
    //  PRODUCER WARPGROUP (warpgroup 2)
    //  4 warps with distinct roles:
    //    warp 3: TMA data loader
    //    warp 2: CLC scheduler (asks HW for next tile)
    //    warps 0-1: MMA issuers (CTA 0 only — one per consumer)
    // ================================================================
    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            // ---- TMA LOADER (persistent loop) ----
            int input_ring = 0;
            int2 tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, blockIdx.x/CLUSTER_SIZE);
            everyone::tma::cluster::wait();
            for (int task_iter = 0; true; task_iter++) {
                // Load ALL K iterations for current tile
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                    #pragma unroll
                    for (int i = 0; i < NUM_CONSUMERS; i++)
                        tma::cluster::load_async(a_smem[input_ring][i], g.a,
                            {(tile_coord.x*2+cta_rank)*NUM_CONSUMERS+i, idx},
                            inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    tma::cluster::load_async(b_smem[input_ring], g.b,
                        {idx, tile_coord.y*2+cta_rank},
                        inputs_arrived[input_ring], (uint16_t)(1<<cta_rank), 0);
                    update_phasebit<1>(bitfield, input_ring);
                    input_ring = ring_advance<PIPE_STAGES>(input_ring);
                }
                // After finishing current tile's loads, check CLC for next task
                wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, schedule.x/CLUSTER_SIZE);
                else break;  // No more tiles — exit
            }
        }
        else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            // ---- CLC SCHEDULER ----
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
            // ---- MMA ISSUERS (persistent loop) ----
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_alloc.set_addr(tmem_addr);
            d_tt_t d_tt[MMA_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < MMA_PIPE_DEPTH; i++)
                d_tt[i] = tm_alloc.allocate<d_tt_t>((i+warpgroup::warpid())*TILE_N);
            int input_ring = 0;
            for (int task_iter = 0; true; task_iter++) {
                // Check CLC: is there a next task after this one?
                wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
                // Wait for epilogue to finish draining TMEM from previous iter
                wait(outputs_finished[task_iter%MMA_PIPE_DEPTH], ((task_iter+MMA_PIPE_DEPTH)/MMA_PIPE_DEPTH)%2);
                // Run MMA across all K iterations
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
    // ================================================================
    //  CONSUMER/EPILOGUE WARPGROUPS 0 and 1
    //  Wait for MMA results, drain TMEM → registers → SMEM → GMEM
    // ================================================================
    else {
        using epilogue_group = group<WARPGROUP_WARPS*NUM_CONSUMERS>;
        warpgroup::increase_registers<224>();
        everyone::tma::cluster::wait_aligned();
        // TMEM provisioning: one warp provisions, all wait
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
            // Wait for CLC schedule
            wait(schedule_arrived[task_iter%CLC_PIPE_DEPTH], (task_iter/CLC_PIPE_DEPTH)%2);
            auto schedule = clc::query(clc_handle[task_iter%CLC_PIPE_DEPTH]);
            warpgroup::sync(warpgroup::groupid()+1);
            warpgroup::tma::cluster::arrive(schedule_finished[task_iter%CLC_PIPE_DEPTH], 0);
            if (schedule.success) next_tile_coord = get_swizzled_2d_idx<SUPERGROUP_SIZE>(rblks, cblks, schedule.x/CLUSTER_SIZE);
            // Wait for MMA to produce TMEM results
            wait(outputs_arrived[warpgroup::groupid()], task_iter%2);
            // Epilogue: TMEM → regs → SMEM → GMEM
            rt_bf<TILE_M/4, TILE_N/EPI_PIPE_DEPTH> d_reg[EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++)
                warpgroup::load_async(d_reg[i], d_tt[task_iter%MMA_PIPE_DEPTH].template subtile<tt<float, TILE_M, TILE_N/EPI_PIPE_DEPTH>>(0, TILE_N/EPI_PIPE_DEPTH*i));
            tensor_load_wait();
            warpgroup::sync(warpgroup::groupid()+1);
            // Note: PDL (pdl::arrive/wait) is disabled here because the educational
            // launch.cu harness doesn't use PDL-aware launch attributes. In production,
            // PDL allows overlapping this kernel's tail with the next kernel's startup.
            warpgroup::tma::cluster::arrive(outputs_finished[task_iter%MMA_PIPE_DEPTH], 0);
            #pragma unroll
            for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
                warpgroup::tma::store_async_read_wait<NUM_D_TILES-1>();
                warpgroup::sync(warpgroup::groupid()+1);
                warpgroup::store(d_smem[warpgroup::groupid()][i%NUM_D_TILES], d_reg[i]);
                warpgroup::sync(warpgroup::groupid()+1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.d, d_smem[warpgroup::groupid()][i%NUM_D_TILES],
                    {(2*tile_coord.x+cta_rank)*NUM_CONSUMERS+warpgroup::groupid(), EPI_PIPE_DEPTH*tile_coord.y+i});
            }
            if (!schedule.success) break;
        }
        // TMEM deprovisioning
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_rank);
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    a_gl Ag{reinterpret_cast<bf16*>(A), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    b_gl Bg{reinterpret_cast<bf16*>(B), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    d_gl Dg{reinterpret_cast<bf16*>(C), nullptr, nullptr, (unsigned long)N, (unsigned long)N};
    matmul_globals g{Ag, Bg, Dg};

    // Grid: one cluster per output tile
    int grid = (N / (CLUSTER_SIZE * NUM_CONSUMERS * TILE_M)) * (N / TILE_N) * CLUSTER_SIZE;

    constexpr size_t smem_size = sizeof(a_tile) * PIPE_STAGES * NUM_CONSUMERS +
                                  sizeof(b_tile) * PIPE_STAGES +
                                  sizeof(d_tile) * NUM_D_TILES * NUM_CONSUMERS + 1024;
    static_assert(smem_size <= MAX_SHARED_MEMORY - 1024);

    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    LaunchConfig<true, false> launch_config(
        dim3(grid), dim3(NUM_THREADS), smem_size, 0, CLUSTER_SIZE);
    cudaLaunchKernelEx(launch_config, matmul_kernel, g);
}

#include "launch.cu"
