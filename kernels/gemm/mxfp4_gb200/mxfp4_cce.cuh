#pragma once
// ================================================================
// MXFP4 Fused Cross-Entropy Kernel — CCE variant of mxfp4_gemm
//
// Instead of materializing [M, V] logits, computes:
//   lse[M]               = logsumexp over all vocab columns
//   neg_correct_logit[M]  = -logit at target column
//
// Loss = mean(neg_correct_logit + lse)
//
// Architecture: same as mxfp4_gemm but the consumer epilogue
// computes online logsumexp via TK's row_max/row_sum + exp/sub_col,
// then atomically merges per-tile LSE into global lse[] via logaddexp.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace mxfp4_cce {

// Atomic logaddexp: atomically merge val into *addr
__device__ inline void atomic_logaddexp(float* addr, float val) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    while (true) {
        float old_f = __uint_as_float(old);
        float mx = fmaxf(old_f, val);
        float new_f;
        if (__isinff(mx) && mx < 0.f) {
            new_f = -INFINITY;
        } else {
            new_f = mx + __logf(__expf(old_f - mx) + __expf(val - mx));
        }
        unsigned int assumed = old;
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(new_f));
        if (old == assumed) break;
    }
}

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_EPI_PIPE_DEPTH > 0);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_NUM_D_TILES > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/128;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, B_sc_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;

    // CCE outputs (raw pointers, not TMA)
    float* lse;                // [M] — initialized to -inf
    float* neg_correct_logit;  // [M] — initialized to 0
    const int64_t* targets;    // [M] — target indices
    int M;                     // actual M (unpadded)
    int N;                     // actual N (unpadded)

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A[C::MMA_PER_TILE];
        B_sc_tile B[C::B_SC_SIZE * C::MMA_PER_TILE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        int padded_M = A.rows();
        int padded_N = B.rows();
        // Match GEMM grid: each CTA handles Mb/2 rows (CLUSTER_SIZE=2 CTAs per Mb tile)
        int num_row_blocks = padded_M / (C::Mb / 2);
        int num_col_blocks = padded_N / C::Nb;
        int total = num_row_blocks * num_col_blocks;
        int max_blocks = num_sms();
        // Round down to multiple of CLUSTER_SIZE
        int grid_size = min(total, max_blocks);
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(grid_size);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;

    const int padded_M = g.A.rows();
    const int padded_N = g.B.rows();
    const int num_row_blocks = padded_M / C::Mb;
    const int num_col_blocks = padded_N / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_iters_per_block = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    // ======================== PRODUCER (identical to GEMM) ========================
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        tma::cluster::load_async(input_scales[stage].A[k], g.A_sc,
                            {row_block_idx*2 + cta_id, i*C::MMA_PER_TILE + k, 0, 0},
                            scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    }
                    if constexpr (C::B_SC_SIZE == 2) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            tma::cluster::load_async(input_scales[stage].B[cta_id * C::MMA_PER_TILE + k], g.B_sc,
                                {col_block_idx*2 + cta_id, i*C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        }
                    } else if (cta_id == 0) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            tma::cluster::load_async(input_scales[stage].B[k], g.B_sc,
                                {col_block_idx, i*C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        }
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // ======================== MMA WARP (identical to GEMM) ========================
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*16 + k*16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*C::B_SC_SIZE*16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                        if constexpr (C::B_SC_SIZE == 2) {
                            auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*C::B_SC_SIZE*16 + 16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[C::MMA_PER_TILE + k]);
                        }
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                                        inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
            }
        }
    // ======================== CONSUMER — CCE EPILOGUE ========================
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;

        // Register tile and col vector types for this warp
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using col_vec_t = typename subtile_rt::col_vec;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Online logsumexp running state per row in this warp
            col_vec_t running_max;
            col_vec_t running_sumexp;
            // Initialize to -inf / 0
            #pragma unroll
            for (int i = 0; i < running_max.outer_dim; i++) {
                #pragma unroll
                for (int j = 0; j < running_max.inner_dim; j++) {
                    running_max[i][j].x = -INFINITY;
                    running_max[i][j].y = -INFINITY;
                    running_sumexp[i][j].x = 0.f;
                    running_sumexp[i][j].y = 0.f;
                }
            }

            // Process subtiles one at a time, matching GEMM OVERLAP_EPI pattern.
            // tensor_load_wait only on last tile — async loads are pipelined.
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                subtile_rt D_reg_fl;
                warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));

                if (epi == C::EPI_PIPE_DEPTH - 1) {
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                }

                // Scale by MXFP4 alpha
                warp::mul(D_reg_fl, D_reg_fl, MXFP4_ALPHA);

                // Extract correct-class logits from registers
                {
                    int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                    int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
                    
                    warp::apply(D_reg_fl, D_reg_fl, [&](int local_row, int local_col, float val) -> float {
                        int global_row = warp_row_base + local_row;
                        int global_col = col_start + local_col;
                        if (global_row < g.M && global_col < g.N) {
                            int64_t target = g.targets[global_row];
                            if ((int)target == global_col) {
                                atomicAdd(&g.neg_correct_logit[global_row], -val);
                            }
                        }
                        return val;
                    });
                }

                // Online logsumexp
                col_vec_t tile_max;
                warp::row_max(tile_max, D_reg_fl);

                subtile_rt shifted;
                warp::sub_row(shifted, D_reg_fl, tile_max);

                subtile_rt exp_tile;
                warp::exp(exp_tile, shifted);

                col_vec_t tile_sumexp;
                warp::row_sum(tile_sumexp, exp_tile);

                // Merge into running state
                #pragma unroll
                for (int i = 0; i < col_vec_t::outer_dim; i++) {
                    #pragma unroll
                    for (int j = 0; j < col_vec_t::inner_dim; j++) {
                        float old_max_x = running_max[i][j].x;
                        float new_max_x = fmaxf(old_max_x, tile_max[i][j].x);
                        float old_max_y = running_max[i][j].y;
                        float new_max_y = fmaxf(old_max_y, tile_max[i][j].y);

                        if (new_max_x > -INFINITY) {
                            running_sumexp[i][j].x = running_sumexp[i][j].x * __expf(old_max_x - new_max_x) +
                                                     tile_sumexp[i][j].x * __expf(tile_max[i][j].x - new_max_x);
                        }
                        if (new_max_y > -INFINITY) {
                            running_sumexp[i][j].y = running_sumexp[i][j].y * __expf(old_max_y - new_max_y) +
                                                     tile_sumexp[i][j].y * __expf(tile_max[i][j].y - new_max_y);
                        }
                        running_max[i][j].x = new_max_x;
                        running_max[i][j].y = new_max_y;
                    }
                }
            }

            // After all EPI subtiles: compute per-row LSE and atomically merge
            // LSE = running_max + log(running_sumexp)
            // Then atomic_logaddexp into global lse[row]
            //
            // MMA register tile row-major layout:
            // For each 16x16 subtile within the col_vec:
            //   accum.x = reduced value for row (laneid/4)       within the subtile
            //   accum.y = reduced value for row (laneid/4) + 8   within the subtile
            // After shuffle, within each group of 4 threads (laneid & 0x1C), all 4
            // hold the same values. So only one per group (laneid % 4 == 0) writes.
            //
            // col_vec[i][0].x = row i*16 + laneid/4     (within warp's portion)
            // col_vec[i][0].y = row i*16 + laneid/4 + 8 (within warp's portion)
            
            const int lane_id = threadIdx.x % 32;
            const int warpid_in_wg = warpgroup::warpid();  // 0..3
            
            // Each warp in warpgroup handles Mb/8 rows = 32 rows  
            int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            int warp_row_offset = warpid_in_wg * (C::Mb / 8);  // 32 rows per warp
            
            if (lane_id % 4 == 0) {  // One writer per group of 4
                #pragma unroll
                for (int i = 0; i < col_vec_t::outer_dim; i++) {
                    // Within tile i (16 rows): .x = row laneid/4, .y = row laneid/4+8
                    int global_row_x = tile_row_base + warp_row_offset + i * 16 + lane_id / 4;
                    int global_row_y = global_row_x + 8;
                    
                    float max_x = running_max[i][0].x;
                    float max_y = running_max[i][0].y;
                    float sumexp_x = running_sumexp[i][0].x;
                    float sumexp_y = running_sumexp[i][0].y;

                    if (global_row_x < g.M && max_x > -INFINITY) {
                        float local_lse = max_x + __logf(sumexp_x);
                        atomic_logaddexp(&g.lse[global_row_x], local_lse);
                    }
                    if (global_row_y < g.M && max_y > -INFINITY) {
                        float local_lse = max_y + __logf(sumexp_y);
                        atomic_logaddexp(&g.lse[global_row_y], local_lse);
                    }
                }
            }

            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();  // drain async state (matches GEMM)
        warpgroup::pdl::arrive();                     // must be unconditional
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace mxfp4_cce
