#pragma once
// ================================================================
// NVFP4 CCE Backward v5 dC — owner-flipped fused pass
//
// Grid ownership: (vocab_superblock, k_block)
// One vocab superblock = 2 adjacent 128-row vocab blocks.
//
// For each row-block we:
//  1. recompute/quantize the low vocab block into CTA0-local staged G^T
//  2. recompute/quantize the high vocab block into CTA1-local staged G^T
//  3. launch one generic-style 2-CTA full-output GEMM:
//       dC_super += Gt_super @ E_col^T
//
// This deliberately avoids the toxic M=128 half_tt phase-3 contract.
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v5_dC {

__device__ __forceinline__ uint8_t quantize_fp4_pair_v5(float v0, float v1, float rcp_scale) {
    const float2 scaled = {v0 * rcp_scale, v1 * rcp_scale};
    return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
}

__device__ __forceinline__ void store_b8_cluster_v5(uint32_t local_addr, int target_cta, uint8_t value) {
    uint32_t addr = local_addr;
    if (target_cta != cluster_ctarank()) {
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                     : "=r"(addr)
                     : "r"(local_addr), "r"(target_cta));
    }
    asm volatile("{st.shared::cluster.b8 [%0], %1;}" :: "r"(addr), "r"((uint32_t)value));
}

__device__ __forceinline__ void store_b8_local_v5(uint32_t local_addr, uint8_t value) {
    asm volatile("{st.shared.b8 [%0], %1;}" :: "r"(local_addr), "r"((uint32_t)value));
}

__device__ __forceinline__ void arrive_remote_cluster_v5(semaphore &sem, int target_cta, uint32_t count = 1) {
    uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(remote_addr)
                 : "r"(local_addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
                 :: "r"(remote_addr), "r"(count)
                 : "memory");
}

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = false;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = 4;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int Nb_out = 128;

    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int P3_MMA_PER_TILE = Nb / 64;
    static constexpr int P3_SCALE_CHUNKS = Mb / 64;
    static constexpr int P3_A_SCALE_CHUNKS = Mb / 64;
    static constexpr int NUM_D_TILES = 2;
};

template <typename C>
struct debug_globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    using P3_B_fp4x2_tile = st_fp4e2m1_2<C::Nb_out/2, C::Mb/2>;
    using P3_B_sc_tile    = st_hf<4, 256, false>;

    // Full 128-row local G^T view per CTA for the 256-row superblock phase-3 path.
    using Gt_fp4_row_tile = st_fp4e2m1_2<C::Nb, C::Mb/2>;
    // Match the generic NVFP4 GEMM scale staging contract exactly.
    // The producer later reinterprets this as 4 contiguous 32x16 fp8 chunks
    // for load_mxnv_scale_async2, just like the working generic path.
    using Gt_sc_row_tile  = st_hf<4, 256, false>;

    using Out_tile      = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH>;
    using Out_sm_tile   = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;
    using Debug_raw_tile = st_fl<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;

    using P3_B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, P3_B_fp4x2_tile>;
    using P3_B_sc_gl        = gl<half,       1, -1, -1, 256, P3_B_sc_tile>;
    using P3_B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using Out_gl            = gl<bf16,       1,  1, -1, -1, Out_tile>;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;

    P3_B_fp4x2_gl     E_col;
    P3_B_sc_gl        E_col_sc;
    P3_B_sc_global_gl E_col_sc_global;
    Out_gl            dC_out;

    // Developer-only buffers. The production path keeps them null.
    uint8_t*          debug_gt_fp4_ptr;
    uint8_t*          debug_gt_sc_ptr;
    bf16*             debug_p3_out_ptr;
    float*            debug_p3_out_raw_ptr;
    int               debug_gt_fp4_stride;
    int               debug_p3_out_stride;
    int               debug_p3_out_raw_stride;
    int               debug_trace_mode;

    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;
    int K;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct output_tiles_t {
        Out_sm_tile D[1];
    };
    struct fp4_staging_t {
        Gt_fp4_row_tile Gt_row;
        Gt_sc_row_tile  Gt_row_sc;
    };
    struct p3_tiles_t {
        P3_B_fp4x2_tile B;
    };
    struct p3_scales_t {
        P3_B_sc_tile B_sc;
    };

    __host__ inline dim3 grid() const {
        const int num_vocab_blocks = B.rows() / C::Nb;
        const int num_vocab_superblocks = (num_vocab_blocks + 1) / 2;
        const int num_k_blocks = dC_out.cols() / C::Nb_out;
        int total = debug_trace_mode ? C::CLUSTER_SIZE : num_vocab_superblocks * num_k_blocks;
        int grid_size = debug_trace_mode ? C::CLUSTER_SIZE : min(total, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int phase1_smem = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(output_tiles_t);
        constexpr int fp4_smem = sizeof(fp4_staging_t) + 1024;
        constexpr int p3_smem = sizeof(p3_tiles_t) + 1024 +
                                sizeof(p3_scales_t) + 1024;
        constexpr int base_total = phase1_smem + fp4_smem + p3_smem;
        static_assert(base_total + (int)sizeof(Debug_raw_tile) <= MAX_SHARED_MEMORY - 1024);
        const int debug_smem = (debug_p3_out_raw_ptr != nullptr) ? (int)sizeof(Debug_raw_tile) : 0;
        return base_total + debug_smem;
    }
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    using P3_B_fp4x2_tile = st_fp4e2m1_2<C::Nb_out/2, C::Mb/2>;
    using P3_B_sc_tile    = st_hf<4, 256, false>;

    using Gt_fp4_row_tile = st_fp4e2m1_2<C::Nb, C::Mb/2>;
    using Gt_sc_row_tile  = st_hf<4, 256, false>;

    using Out_tile       = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH>;
    using Out_sm_tile    = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;
    using Debug_raw_tile = st_fl<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;

    using P3_B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, P3_B_fp4x2_tile>;
    using P3_B_sc_gl        = gl<half,       1, -1, -1, 256, P3_B_sc_tile>;
    using P3_B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using Out_gl            = gl<bf16,       1,  1, -1, -1, Out_tile>;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;

    P3_B_fp4x2_gl     E_col;
    P3_B_sc_gl        E_col_sc;
    P3_B_sc_global_gl E_col_sc_global;
    Out_gl            dC_out;

    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;
    int K;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct output_tiles_t {
        Out_sm_tile D[1];
    };
    struct fp4_staging_t {
        Gt_fp4_row_tile Gt_row;
        Gt_sc_row_tile  Gt_row_sc;
    };
    struct p3_tiles_t {
        P3_B_fp4x2_tile B;
    };
    struct p3_scales_t {
        P3_B_sc_tile B_sc;
    };

    __host__ inline dim3 grid() const {
        const int num_vocab_blocks = B.rows() / C::Nb;
        const int num_vocab_superblocks = (num_vocab_blocks + 1) / 2;
        const int num_k_blocks = dC_out.cols() / C::Nb_out;
        int total = num_vocab_superblocks * num_k_blocks;
        int grid_size = min(total, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int phase1_smem = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(output_tiles_t);
        constexpr int fp4_smem = sizeof(fp4_staging_t) + 1024;
        constexpr int p3_smem = sizeof(p3_tiles_t) + 1024 +
                                sizeof(p3_scales_t) + 1024;
        constexpr int base_total = phase1_smem + fp4_smem + p3_smem;
        static_assert(base_total <= MAX_SHARED_MEMORY - 1024);
        return base_total;
    }
};

template <typename C, typename G, bool Debug>
__device__ inline void kernel_impl(const G& g) {

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.E_col.template prefetch_tma<typename G::P3_B_fp4x2_tile>();
        g.E_col_sc.template prefetch_tma<typename G::P3_B_sc_tile>();
        g.dC_out.template prefetch_tma<typename G::Out_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;

    const int num_vocab_blocks = g.B.rows() / C::Nb;
    const int num_vocab_superblocks = (num_vocab_blocks + 1) / 2;
    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_k_blocks = g.dC_out.cols() / C::Nb_out;
    const int num_blocks = num_vocab_superblocks * num_k_blocks;
    const int num_iters_per_row = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_k_blocks;

    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::output_tiles_t &output_tiles = sm_allocator.allocate<G::output_tiles_t>();
    typename G::fp4_staging_t  &fp4_staging = sm_allocator.allocate<G::fp4_staging_t>();
    typename G::p3_tiles_t     &p3_tiles = sm_allocator.allocate<G::p3_tiles_t>();
    typename G::p3_scales_t    &p3_scales = sm_allocator.allocate<G::p3_scales_t>();
    typename G::Debug_raw_tile *debug_raw = nullptr;
    if constexpr (Debug) {
        if (g.debug_p3_out_raw_ptr) {
            debug_raw = &sm_allocator.allocate<typename G::Debug_raw_tile>();
        }
    }
    auto should_stop_block = [&](int block_idx) {
        if constexpr (Debug) return g.debug_trace_mode > 0 && block_idx >= g.debug_trace_mode;
        else                  return false;
    };
    auto should_stop_row_block = [&](int row_block_idx) {
        if constexpr (Debug) return g.debug_trace_mode > 0 && row_block_idx > 0;
        else                  return false;
    };

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
                    __shared__ semaphore a_super_ready;
    __shared__ semaphore epi_finished;
    __shared__ semaphore p3_tiles_arrived;
    __shared__ semaphore p3_scales_arrived;
    __shared__ semaphore p3_inputs_finished;
    __shared__ semaphore p3_outputs_arrived;
    __shared__ semaphore p3_outputs_finished;
    __shared__ float gt_row_sg;

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(a_super_ready, C::CLUSTER_SIZE, 0);
        init_semaphore(epi_finished, 1, 0);
        init_semaphore(p3_tiles_arrived, 0, 1);
        init_semaphore(p3_scales_arrived, 0, 1);
        init_semaphore(p3_inputs_finished, 0, 1);
        init_semaphore(p3_outputs_arrived, 0, 1);
        init_semaphore(p3_outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS * C::PRODUCER_WARPGROUPS>::warpid();
        const int lane_id = threadIdx.x % 32;
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                if (should_stop_block(block_idx)) break;
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
                const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
                const int k_block_idx = idx_within_supergroup / vocab_superblocks_in_group;

                for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                    if (should_stop_row_block(row_block_idx)) break;
                    for (int subpass = 0; subpass < 2; ++subpass) {
                        const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                        if (vocab_block_idx >= num_vocab_blocks) continue;
                        for (int i = 0; i < num_iters_per_row; ++i) {
                            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                            tma::cluster::load_async(
                                input_tiles[stage].A, g.A,
                                {row_block_idx * 2 + cta_id, i},
                                tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                            tma::cluster::load_async(
                                input_tiles[stage].B, g.B,
                                {vocab_block_idx * 2 + cta_id, i},
                                tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                            update_phasebit<1>(phasebits, stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                    }

                    wait(p3_inputs_finished, get_phasebit<1>(phasebits, 6));
                    tma::cluster::load_async(
                        p3_tiles.B, g.E_col,
                        {k_block_idx * 2 + cta_id, row_block_idx},
                        p3_tiles_arrived, (uint16_t)(1 << cta_id), 0);
                    update_phasebit<1>(phasebits, 6);
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                if (should_stop_block(block_idx)) break;
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
                const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
                const int k_block_idx = idx_within_supergroup / vocab_superblocks_in_group;

                for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                    if (should_stop_row_block(row_block_idx)) break;
                    for (int subpass = 0; subpass < 2; ++subpass) {
                        const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                        if (vocab_block_idx >= num_vocab_blocks) continue;
                        for (int i = 0; i < num_iters_per_row; ++i) {
                            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                            tma::cluster::load_async(
                                input_scales[stage].A, g.A_sc,
                                {row_block_idx * 2 + cta_id, i, 0},
                                scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                            if (cta_id == 0) {
                                tma::cluster::load_async(
                                    input_scales[stage].B[0], g.B_sc,
                                    {vocab_block_idx, i, 0},
                                    scales_arrived[stage], (uint16_t)(0b11), 0);
                            }
                            update_phasebit<1>(phasebits, stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                    }

                    wait(p3_inputs_finished, get_phasebit<1>(phasebits, 6));
                    tma::cluster::load_async(
                        p3_scales.B_sc, g.E_col_sc,
                        {k_block_idx, row_block_idx, 0},
                        p3_scales_arrived, (uint16_t)(1 << cta_id), 0);
                    update_phasebit<1>(phasebits, 6);
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            int produced_blocks = 0;
            int epi_phase = 1;

            auto phase1_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto p3_tm     = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm   = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm   = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

            constexpr int p3_sc_offset = 256;
            auto p3_A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::P3_A_SCALE_CHUNKS>>(p3_sc_offset);
            auto p3_B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::P3_SCALE_CHUNKS>>(
                p3_sc_offset + 4 * C::P3_A_SCALE_CHUNKS);

            auto do_phase1 = [&](auto &accum) {
                for (int i = 0; i < num_iters_per_row; ++i) {
                    tma::expect_bytes(scales_arrived[stage], 2 * sizeof(typename G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_sub = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                            stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(A_sc_tm_sub, A_sc_sm_sub);

                        auto B_sc_tm_sub = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                            stage * C::MMA_PER_TILE * 32 + ii * 16);
                        auto &B_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(B_sc_tm_sub, B_sc_sm_sub);
                    }

                    tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(typename G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) {
                        mm2_ABt(
                            accum, input_tiles[stage].A, input_tiles[stage].B,
                            A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]);
                    } else {
                        mma2_ABt(
                            accum, input_tiles[stage].A, input_tiles[stage].B,
                            A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]);
                    }
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            };

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                if (should_stop_block(block_idx)) break;
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
                const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
                const int k_block_idx = idx_within_supergroup / vocab_superblocks_in_group;

                if (produced_blocks > 0) {
                    wait(epi_finished, epi_phase);
                    update_phasebit<1>(phasebits, 9);
                    epi_phase ^= 1;
                }

                int phase0 = 0;
                for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                    if (should_stop_row_block(row_block_idx)) break;

                    for (int subpass = 0; subpass < 2; ++subpass) {
                        const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                        const int iter_phase = phase0;

                        tensor_after_thread_sync();

                        if (vocab_block_idx < num_vocab_blocks) {
                            do_phase1(phase1_tm);
                            tensor_commit<2>(outputs_arrived);
                            tensor_after_thread_sync();
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                            wait(a_super_ready, iter_phase);
                        } else {
                            // Tail: leave the peer-local A tile zero-filled.
                            wait(a_super_ready, iter_phase);
                        }
                        phase0 ^= 1;
                    }

                    wait(p3_outputs_finished, get_phasebit<1>(phasebits, 8));
                    update_phasebit<1>(phasebits, 8);

                    #pragma unroll
                    for (int ii = 0; ii < C::P3_A_SCALE_CHUNKS; ++ii) {
                        auto p3_A_sc_sub = p3_A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        auto &A_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&fp4_staging.Gt_row_sc.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(p3_A_sc_sub, A_sc_sm_sub);
                    }

                    tma::expect_bytes(p3_scales_arrived, 2 * sizeof(typename G::p3_scales_t));
                    wait(p3_scales_arrived, get_phasebit<0>(phasebits, 6));
                    #pragma unroll
                    for (int ii = 0; ii < C::P3_SCALE_CHUNKS; ++ii) {
                        auto p3_B_sc_sub = p3_B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        auto &B_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&p3_scales.B_sc.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(p3_B_sc_sub, B_sc_sm_sub);
                    }

                    tma::expect_bytes(p3_tiles_arrived, 2 * sizeof(typename G::p3_tiles_t));
                    wait(p3_tiles_arrived, get_phasebit<0>(phasebits, 6));
                    mm2_ABt(
                        p3_tm, fp4_staging.Gt_row, p3_tiles.B,
                        p3_A_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_A_SCALE_CHUNKS * 16>>(0),
                        p3_B_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_SCALE_CHUNKS * 32>>(0),
                        p3_inputs_finished);
                    tensor_commit<2>(p3_outputs_arrived);
                    tensor_after_thread_sync();
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    update_phasebit<0>(phasebits, 6);
                }
                ++produced_blocks;
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto phase1_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto p3_tm     = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}];
        const int lane_id = threadIdx.x % 32;
        using logits_rt = rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        using logits_rt_bf = rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        using out_rt    = rt_fl<C::Nb / 4, C::Nb_out / C::EPI_PIPE_DEPTH>;
        using out_rt_bf = rt_bf<C::Nb / 4, C::Nb_out / C::EPI_PIPE_DEPTH>;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            if (should_stop_block(block_idx)) break;
            const int supergroup_idx = block_idx / num_blocks_per_supergroup;
            const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
            const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
            const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
            const int k_block_idx = idx_within_supergroup / vocab_superblocks_in_group;

            out_rt D_acc[C::EPI_PIPE_DEPTH];
            // Clear our local staged A tile once per row-block sweep iteration.
            auto clear_a_super = [&]() {
                constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                uint8_t *fp4_bytes = reinterpret_cast<uint8_t*>(&fp4_staging.Gt_row.data[0]);
                for (int idx = threadIdx.x; idx < (int)sizeof(typename G::Gt_fp4_row_tile); idx += CONSUMER_THREADS) {
                    fp4_bytes[idx] = 0;
                }
                uint8_t *sc_bytes = reinterpret_cast<uint8_t*>(&fp4_staging.Gt_row_sc.data[0]);
                for (int idx = threadIdx.x; idx < (int)sizeof(typename G::Gt_sc_row_tile); idx += CONSUMER_THREADS) {
                    sc_bytes[idx] = 0;
                }
            };

            int phase0 = 0;
            for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                if (should_stop_row_block(row_block_idx)) break;
                clear_a_super();
                warpgroup::sync(1);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                for (int subpass = 0; subpass < 2; ++subpass) {
                    const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                    const int iter_phase = phase0;
                    const bool valid_subpass = vocab_block_idx < num_vocab_blocks;

                    if (valid_subpass) {
                        wait(outputs_arrived, iter_phase);

                        const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                        const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

                        logits_rt D_regs_fl[C::EPI_PIPE_DEPTH];
                        logits_rt_bf D_regs_bf[C::EPI_PIPE_DEPTH];
                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                            warpgroup::load_async(
                                D_regs_fl[epi],
                                phase1_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                        }
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);

                        int my_targets_x[logits_rt::height];
                        int my_targets_y[logits_rt::height];
                        float my_lse_x[logits_rt::height];
                        float my_lse_y[logits_rt::height];
                        #pragma unroll
                        for (int i = 0; i < logits_rt::height; ++i) {
                            const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                            const int global_row_y = global_row_x + 8;
                            my_targets_x[i] = (global_row_x < g.M) ? static_cast<int>(g.targets[global_row_x]) : -1;
                            my_targets_y[i] = (global_row_y < g.M) ? static_cast<int>(g.targets[global_row_y]) : -1;
                            my_lse_x[i] = (global_row_x < g.M) ? g.lse[global_row_x] : INFINITY;
                            my_lse_y[i] = (global_row_y < g.M) ? g.lse[global_row_y] : INFINITY;
                        }

                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                            auto &D_fl = D_regs_fl[epi];
                            warp::mul(D_fl, D_fl, global_scale);
                            const int col_start = vocab_block_idx * C::Nb + epi * (C::Nb / C::EPI_PIPE_DEPTH);

                            #pragma unroll
                            for (int i = 0; i < logits_rt::height; ++i) {
                                const float lse_x = my_lse_x[i];
                                const float lse_y = my_lse_y[i];
                                #pragma unroll
                                for (int j = 0; j < logits_rt::width; ++j) {
                                    #pragma unroll
                                    for (int kk = 0; kk < 4; ++kk) {
                                        const float lse_val = (kk % 2 == 0) ? lse_x : lse_y;
                                        D_fl.tiles[i][j].data[kk].x = __expf(D_fl.tiles[i][j].data[kk].x - lse_val);
                                        D_fl.tiles[i][j].data[kk].y = __expf(D_fl.tiles[i][j].data[kk].y - lse_val);
                                    }
                                }
                            }

                            #pragma unroll
                            for (int i = 0; i < logits_rt::height; ++i) {
                                const int tgt_x = my_targets_x[i];
                                if (tgt_x >= col_start && tgt_x < col_start + (C::Nb / C::EPI_PIPE_DEPTH)) {
                                    const int local_col = tgt_x - col_start;
                                    const int j_idx = local_col / 16;
                                    const int within_tile = local_col % 16;
                                    const int k_half = within_tile / 8;
                                    const int pair_pos = (within_tile % 8) / 2;
                                    if ((lane_id % 4) == pair_pos) {
                                        const int k_idx = k_half * 2;
                                        if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                                        else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                                    }
                                }
                                const int tgt_y = my_targets_y[i];
                                if (tgt_y >= col_start && tgt_y < col_start + (C::Nb / C::EPI_PIPE_DEPTH)) {
                                    const int local_col = tgt_y - col_start;
                                    const int j_idx = local_col / 16;
                                    const int within_tile = local_col % 16;
                                    const int k_half = within_tile / 8;
                                    const int pair_pos = (within_tile % 8) / 2;
                                    if ((lane_id % 4) == pair_pos) {
                                        const int k_idx = k_half * 2 + 1;
                                        if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                                        else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                                    }
                                }
                            }

                            #pragma unroll
                            for (int i = 0; i < logits_rt::height; ++i) {
                                const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                                const int global_row_y = global_row_x + 8;
                                #pragma unroll
                                for (int j = 0; j < logits_rt::width; ++j) {
                                    #pragma unroll
                                    for (int kk = 0; kk < 4; ++kk) {
                                        if (kk % 2 == 0 && global_row_x >= g.M) {
                                            D_fl.tiles[i][j].data[kk].x = 0.0f;
                                            D_fl.tiles[i][j].data[kk].y = 0.0f;
                                        }
                                        if (kk % 2 == 1 && global_row_y >= g.M) {
                                            D_fl.tiles[i][j].data[kk].x = 0.0f;
                                            D_fl.tiles[i][j].data[kk].y = 0.0f;
                                        }
                                    }
                                }
                            }

                            warp::mul(D_fl, D_fl, g.grad_scale);
                            warp::copy(D_regs_bf[epi], D_fl);
                        }

                        if (warpgroup::warpid() == 0 && lane_id == 0) {
                            constexpr float kFp4Max = 6.0f;
                            constexpr float kE4M3Max = 448.0f;
                            gt_row_sg = fmaxf(g.grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
                        }
                        warpgroup::sync(1);

                        const float g_sg = gt_row_sg;
                        const float g_sg_rcp = 1.0f / g_sg;
                        const uint32_t gt_fp4_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row.data[0]));
                        const uint32_t gt_sc_base  = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row_sc.data[0]));
                        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
                        constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                        constexpr int LOCAL_ROW16_BLOCKS = (C::Mb / 2) / 16;
                        constexpr int ROW16_BLOCKS_PER_PASS = CONSUMER_THREADS / SUBTILE_COLS;
                        static_assert(LOCAL_ROW16_BLOCKS % ROW16_BLOCKS_PER_PASS == 0);

                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                            warpgroup::sync(1);
                            warpgroup::store(output_tiles.D[0], D_regs_bf[epi]);
                            warpgroup::sync(1);

                            const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
                            const int col_in_epi = threadIdx.x % SUBTILE_COLS;
                            const int row16_block_base = threadIdx.x / SUBTILE_COLS;
                            const int local_gc = epi * SUBTILE_COLS + col_in_epi;

                            #pragma unroll
                            for (int row16_pass = 0; row16_pass < LOCAL_ROW16_BLOCKS / ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                                const int row16_block = row16_block_base + row16_pass * ROW16_BLOCKS_PER_PASS;
                                if (row16_block >= LOCAL_ROW16_BLOCKS) continue;

                                const int local_row_base = row16_block * 16;
                                const int global_row_base = tile_row_base + local_row_base;
                                float col_amax = 0.0f;
                                #pragma unroll
                                for (int r = 0; r < 16; ++r) {
                                    bf16 value_bf;
                                    move<bf16>::lds(value_bf, G::Out_sm_tile::idx(d_base, {local_row_base + r, col_in_epi}));
                                    if (global_row_base + r < g.M) {
                                        col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value_bf)));
                                    }
                                }

                                const float col_scale = col_amax * (1.0f / 6.0f);
                                const float col_rcp = (col_amax > 0.0f) ? (6.0f / col_amax) : 0.0f;
                                const int local_row_pair_base = cta_id * (C::Mb / 4) + local_row_base / 2;

                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const int global_row = global_row_base + pair * 2;
                                    if (global_row >= g.M) continue;

                                    bf16 value0_bf;
                                    move<bf16>::lds(value0_bf, G::Out_sm_tile::idx(d_base, {local_row_base + pair * 2, col_in_epi}));
                                    const float v0 = __bfloat162float(value0_bf);
                                    float v1 = 0.0f;
                                    if (global_row + 1 < g.M) {
                                        bf16 value1_bf;
                                        move<bf16>::lds(value1_bf, G::Out_sm_tile::idx(d_base, {local_row_base + pair * 2 + 1, col_in_epi}));
                                        v1 = __bfloat162float(value1_bf);
                                    }

                                    const uint8_t fp4_pair = quantize_fp4_pair_v5(v0, v1, col_rcp);
                                    const uint32_t fp4_addr = G::Gt_fp4_row_tile::idx(
                                        gt_fp4_base, {local_gc, local_row_pair_base + pair});
                                    if (subpass == cta_id) store_b8_local_v5(fp4_addr, fp4_pair);
                                    else                  store_b8_cluster_v5(fp4_addr, subpass, fp4_pair);
                                }

                                const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(col_scale * g_sg_rcp);
                                const int local_row_m = cta_id * (C::Mb / 2) + local_row_base;
                                const int m_kgroup = local_row_m / 64;
                                const int m_16_in_64 = (local_row_m / 16) % 4;
                                const int sr = local_gc % 32;
                                const int rr = (local_gc / 32) % 4;
                                const int byte_idx = m_kgroup * 512 + sr * 16 + rr * 4 + m_16_in_64;
                                const uint8_t sc_byte = *reinterpret_cast<const uint8_t*>(&csc);
                                if (subpass == cta_id) store_b8_local_v5(gt_sc_base + byte_idx, sc_byte);
                                else                  store_b8_cluster_v5(gt_sc_base + byte_idx, subpass, sc_byte);
                            }
                        }
                    }

                    warpgroup::sync(1);
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                    if (warpgroup::warpid() == 0 && lane_id == 0) {
                        arrive(a_super_ready, 1);
                        arrive_remote_cluster_v5(a_super_ready, 1 - cta_id, 1);
                    }
                    phase0 ^= 1;
                }

                wait(p3_outputs_arrived, get_phasebit<0>(phasebits, 8));
                out_rt D_reg_fl[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::load_async(
                        D_reg_fl[epi],
                        p3_tm.template subtile<full_tt_fl<C::Nb_out / C::EPI_PIPE_DEPTH>>(
                            0,
                            epi * (C::Nb_out / C::EPI_PIPE_DEPTH)));
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(p3_outputs_finished, 0, 1);
                update_phasebit<0>(phasebits, 8);

                const float p3_scale = gt_row_sg * g.E_col_sc_global[{0}];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    if constexpr (Debug) {
                        if (g.debug_p3_out_raw_ptr &&
                            cluster_id == 0 &&
                            block_idx == 0 &&
                            row_block_idx == 0) {
                            warpgroup::store(*debug_raw, D_reg_fl[epi]);
                            warpgroup::sync(1);
                            const uint32_t raw_base = static_cast<uint32_t>(__cvta_generic_to_shared(&debug_raw->data[0]));
                            constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                            for (int idx = threadIdx.x; idx < C::Nb * (C::Nb_out / C::EPI_PIPE_DEPTH); idx += CONSUMER_THREADS) {
                                const int row = idx / (C::Nb_out / C::EPI_PIPE_DEPTH);
                                const int col = idx % (C::Nb_out / C::EPI_PIPE_DEPTH);
                                float value;
                                move<float>::lds(value, G::Debug_raw_tile::idx(raw_base, {row, col}));
                                g.debug_p3_out_raw_ptr[(cta_id * C::Nb + row) * g.debug_p3_out_raw_stride +
                                                       epi * (C::Nb_out / C::EPI_PIPE_DEPTH) + col] = value;
                            }
                            warpgroup::sync(1);
                        }
                    }
                    warp::mul(D_reg_fl[epi], D_reg_fl[epi], p3_scale);
                    if (row_block_idx == 0) warp::copy(D_acc[epi], D_reg_fl[epi]);
                    else                    warp::add(D_acc[epi], D_acc[epi], D_reg_fl[epi]);
                }
            }

            const int vocab_superblock_base = vocab_superblock_idx * 2;
            out_rt_bf D_reg_bf[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                warp::copy(D_reg_bf[epi], D_acc[epi]);
            }
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                warpgroup::tma::store_async_read_wait<0>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[0], D_reg_bf[epi]);
                warpgroup::sync(1);

                if constexpr (Debug) {
                    if (g.debug_p3_out_ptr &&
                        cluster_id == 0 &&
                        block_idx == 0) {
                        const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
                        constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                        for (int idx = threadIdx.x; idx < C::Nb * (C::Nb_out / C::EPI_PIPE_DEPTH); idx += CONSUMER_THREADS) {
                            const int row = idx / (C::Nb_out / C::EPI_PIPE_DEPTH);
                            const int col = idx % (C::Nb_out / C::EPI_PIPE_DEPTH);
                            bf16 value;
                            move<bf16>::lds(value, G::Out_sm_tile::idx(d_base, {row, col}));
                            g.debug_p3_out_ptr[(cta_id * C::Nb + row) * g.debug_p3_out_stride +
                                               epi * (C::Nb_out / C::EPI_PIPE_DEPTH) + col] = value;
                        }
                        warpgroup::sync(1);
                    }
                }

                warpgroup::sync(1);

                if (vocab_superblock_base + cta_id < num_vocab_blocks) {
                    const int global_vocab_base = (vocab_superblock_base + cta_id) * C::Nb;
                    const int global_k_base = (C::EPI_PIPE_DEPTH * k_block_idx + epi) * (C::Nb_out / C::EPI_PIPE_DEPTH);
                    constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                    const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
                    for (int idx = threadIdx.x; idx < C::Nb * (C::Nb_out / C::EPI_PIPE_DEPTH); idx += CONSUMER_THREADS) {
                        const int row = idx / (C::Nb_out / C::EPI_PIPE_DEPTH);
                        const int col = idx % (C::Nb_out / C::EPI_PIPE_DEPTH);
                        bf16 value_f;
                        move<bf16>::lds(value_f, G::Out_sm_tile::idx(d_base, {row, col}));
                        g.dC_out.raw_ptr[(global_vocab_base + row) * g.dC_out.cols() + (global_k_base + col)] =
                            value_f;
                    }
                    warpgroup::sync(1);
                }
            }

            warpgroup::tma::store_async_read_wait<0>();
            if (cta_id == 0 && warpgroup::warpid() == 0 && lane_id == 0) {
                __threadfence_block();
                arrive(epi_finished, 1);
            }

            if constexpr (Debug) {
                if (g.debug_gt_fp4_ptr && cluster_id == 0 && block_idx == 0) {
                    constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                    const uint32_t gt_fp4_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row.data[0]));
                    const uint32_t gt_sc_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row_sc.data[0]));
                    for (int idx = threadIdx.x; idx < C::Nb * (C::Mb / 2); idx += CONSUMER_THREADS) {
                        const int row = idx / (C::Mb / 2);
                        const int col = idx % (C::Mb / 2);
                        uint32_t value_u32;
                        asm volatile("ld.shared.b8 %0, [%1];\n" : "=r"(value_u32) : "r"(G::Gt_fp4_row_tile::idx(gt_fp4_base, {row, col})));
                        g.debug_gt_fp4_ptr[(cta_id * C::Nb + row) * g.debug_gt_fp4_stride + col] = static_cast<uint8_t>(value_u32);
                    }
                    for (int idx = threadIdx.x; idx < (int)sizeof(typename G::Gt_sc_row_tile); idx += CONSUMER_THREADS) {
                        g.debug_gt_sc_ptr[cta_id * (int)sizeof(typename G::Gt_sc_row_tile) + idx] =
                            reinterpret_cast<const uint8_t*>(&fp4_staging.Gt_row_sc.data[0])[idx];
                    }
                }
            }
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

template <typename C>
__device__ inline void kernel(const globals<C>& g) {
    kernel_impl<C, globals<C>, false>(g);
}

template <typename C>
__device__ inline void debug_kernel(const debug_globals<C>& g) {
    kernel_impl<C, debug_globals<C>, true>(g);
}

// Developer-only probe placeholder. The benchmark path does not rely on it.
template <typename C>
struct p3_probe_globals {
    const uint8_t *gt_fp4_ptr;
    const uint8_t *gt_sc_ptr;
    const uint8_t *p3_b_fp4_ptr;
    const uint8_t *p3_b_sc_ptr;
    float gt_sg;
    float p3_b_sg;
    bf16 *out_bf_ptr;
    float *out_raw_ptr;
    int gt_fp4_stride;
    int p3_b_fp4_stride;
    int out_stride;
    int out_raw_stride;
    int probe_mode;

    __host__ inline dim3 grid() const { return dim3(1); }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const { return 0; }
};

template <typename C>
__device__ inline void p3_probe_kernel(const p3_probe_globals<C>&) {}

} // namespace nvfp4_cce_backward_v5_dC
