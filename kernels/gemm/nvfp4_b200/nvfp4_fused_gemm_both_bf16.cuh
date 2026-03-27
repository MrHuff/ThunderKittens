#pragma once
// ================================================================
// Experimental two-sided fused NVFP4 GEMM.
//
// Accepts bf16 A and bf16 B, quantizes both operands on the fly inside
// a single CTA, then performs NVFP4 MMA:
//   D = quantize(A_bf16) x quantize(B_bf16)^T
//
// This is intentionally separate from the production fused kernel so we
// can benchmark a real "both operands in-kernel" prototype without
// destabilizing the current A-only fused path.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_fused_gemm_both_bf16 {

static constexpr float SCALE_MAX_DEC = 65504.0f / (6.0f * 448.0f);
static constexpr float SCALE_MAX_ENC = 1.0f / SCALE_MAX_DEC;

static constexpr int QUANT_TILE_M = 128;
static constexpr int QUANT_TILE_N = 128;
static constexpr int K_BLOCK_SIZE = 16;

template <
    bool _USE_CTA_AMAX,
    int _COL_TILES_PER_BLOCK = 1,
    int _LOAD_PIPE_DEPTH = 2,
    int _EPI_PIPE_DEPTH = 4,
    int _NUM_D_TILES = (_EPI_PIPE_DEPTH > 1 ? 4 : 1),
    int _CLUSTER_SIZE = 1,
    bool _SHARE_A_ACROSS_CTAS = false>
struct config {
    static_assert(_COL_TILES_PER_BLOCK == 1 || _COL_TILES_PER_BLOCK == 2 || _COL_TILES_PER_BLOCK == 4,
                  "Both-bf16 fused kernel supports 1, 2, or 4 column tiles per block");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 4,
                  "Both-bf16 fused kernel supports load pipe depth 1..4");
    static_assert(_EPI_PIPE_DEPTH > 0 && 128 % _EPI_PIPE_DEPTH == 0,
                  "Both-bf16 fused kernel requires EPI_PIPE_DEPTH to divide Nb=128");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be positive");
    static_assert(!_SHARE_A_ACROSS_CTAS || (_CLUSTER_SIZE == 2 && _COL_TILES_PER_BLOCK == 1),
                  "Shared-A both-bf16 path expects 2-CTA clusters and one local column tile per CTA");
    static constexpr bool USE_CTA_AMAX = _USE_CTA_AMAX;
    static constexpr int COL_TILES_PER_BLOCK = _COL_TILES_PER_BLOCK;
    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    static constexpr bool SHARE_A_ACROSS_CTAS = _SHARE_A_ACROSS_CTAS;
    static constexpr int TMEM_NCTA = SHARE_A_ACROSS_CTAS ? 1 : CLUSTER_SIZE;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int A_QUANT_WARPGROUPS = 1;
    static constexpr int B_QUANT_WARPGROUPS = 1;
    static constexpr int MMA_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS =
        CONSUMER_WARPGROUPS + A_QUANT_WARPGROUPS + B_QUANT_WARPGROUPS + MMA_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int QUANT_SUB_TILES = Kb / QUANT_TILE_N;
    static constexpr int MMA_PER_TILE = Kb / 64;

    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int NUM_D_TILES = _NUM_D_TILES;
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;
};

template <typename C>
struct globals {
    using A_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using B_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb, C::Kb / 2>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb, C::Kb / 2>;
    using A_sc_tile = st_hf<4, 256, false>;
    using B_sc_tile = st_hf<4, 256, false>;
    using D_tile = st_bf<C::Mb, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using B_bf16_gl = gl<bf16, 1, 1, -1, -1, B_bf16_tile>;
    using D_gl = gl<bf16, 1, 1, -1, -1, D_tile>;

    A_bf16_gl A_bf16;
    B_bf16_gl B_bf16;
    D_gl D;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B[C::COL_TILES_PER_BLOCK];
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::COL_TILES_PER_BLOCK];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };
    struct quant_buf_t {
        A_bf16_tile bf16_tile;
    };
    struct a_export_t {
        alignas(16) uint8_t data[C::Mb * (C::Kb / 2)];
    };
    struct a_scale_export_t {
        alignas(16) uint8_t data[sizeof(A_sc_tile)];
    };

    __host__ inline dim3 grid() const {
        if constexpr (C::SHARE_A_ACROSS_CTAS) {
            return dim3(D.cols() / C::Nb, D.rows() / C::Mb);
        }
        return dim3(D.cols() / (C::Nb * C::COL_TILES_PER_BLOCK), D.rows() / C::Mb);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int smem =
            sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
            sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
            sizeof(outputs_t) +
            2 * sizeof(quant_buf_t) + 1024 +
            (C::SHARE_A_ACROSS_CTAS
                 ? C::LOAD_PIPE_DEPTH * (sizeof(a_export_t) + sizeof(a_scale_export_t)) + 1024
                 : 0);
        static_assert(smem <= MAX_SHARED_MEMORY - 1024);
        return smem;
    }
};

template <typename QuantBuf, typename Fp4Tile, typename ScTile>
__device__ inline void quantize_operand_subtile_constant(
    QuantBuf &quant_buf,
    Fp4Tile &out_tile,
    ScTile &out_scales,
    int sub_tile_idx,
    uint32_t export_canonical_smem_base = 0
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    bf16_2 bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF + col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col = k_block_idx * K_BLOCK_SIZE + ((local_tid + j) * 2) % K_BLOCK_SIZE;
                const int offset = (tile_row * QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_smem)) + offset);
            }
        }
    }

    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            bf16_2 amax_pair = __habs2(bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                amax_pair = __hmax2(amax_pair, __habs2(bf16_reg[col_half][i][j]));
            }
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(amax_pair.x, amax_pair.y));
        }
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * SCALE_MAX_ENC);
        }

        const uint32_t tile_base = static_cast<uint32_t>(__cvta_generic_to_shared(&out_tile.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec * SCALE_MAX_DEC, 1e-12f);
            const int fp4_col_base =
                sub_tile_idx * (QUANT_TILE_N / 2) +
                (k_block_idx + col_half * NUM_K_BLOCKS_HALF) * K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int fp4_col = fp4_col_base + ((local_tid + j) & 7);
                const uint32_t swizzled_addr = Fp4Tile::idx(tile_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(bf16_reg[col_half][i][j].x) * s_enc,
                    __bfloat162float(bf16_reg[col_half][i][j].y) * s_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
                if (export_canonical_smem_base != 0) {
                    asm volatile("{st.shared.b8 [%0], %1;}"
                        :: "r"(export_canonical_smem_base + tile_row * QUANT_TILE_N + fp4_col),
                           "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
                }
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t *>(&out_scales.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[1][0])));
    }
}

template <typename QuantBuf, typename Fp4Tile, typename ScTile>
__device__ inline void quantize_operand_subtile_cta_amax(
    QuantBuf &quant_buf,
    Fp4Tile &out_tile,
    ScTile &out_scales,
    int sub_tile_idx,
    int quant_sync_bar_id,
    float *running_amax,
    float *warp_max_buf,
    uint32_t export_canonical_smem_base = 0
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    bf16_2 bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF + col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col = k_block_idx * K_BLOCK_SIZE + ((local_tid + j) * 2) % K_BLOCK_SIZE;
                const int offset = (tile_row * QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_smem)) + offset);
            }
        }
    }

    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            bf16_2 amax_pair = __habs2(bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                amax_pair = __hmax2(amax_pair, __habs2(bf16_reg[col_half][i][j]));
            }
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(amax_pair.x, amax_pair.y));
        }
    }

    float thread_max = 0.0f;
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            thread_max = fmaxf(thread_max, amax_all[col_half][i]);
        }
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    const int warp_in_wg = local_tid / 32;
    const int lane = local_tid % 32;
    if (lane == 0) {
        warp_max_buf[warp_in_wg] = thread_max;
    }
    warpgroup::sync(quant_sync_bar_id);
    const float subtile_amax = fmaxf(
        fmaxf(warp_max_buf[0], warp_max_buf[1]),
        fmaxf(warp_max_buf[2], warp_max_buf[3]));
    warpgroup::sync(quant_sync_bar_id);

    if (local_tid == 0) {
        *running_amax = fmaxf(*running_amax, fmaxf(subtile_amax, 1e-12f));
    }

    const float s_global_dec = fmaxf(subtile_amax, 1e-12f) / (6.0f * 448.0f);
    const float s_global_enc = (6.0f * 448.0f) / fmaxf(subtile_amax, 1e-12f);

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * s_global_enc);
        }

        const uint32_t tile_base = static_cast<uint32_t>(__cvta_generic_to_shared(&out_tile.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec * s_global_dec, 1e-12f);
            const int fp4_col_base =
                sub_tile_idx * (QUANT_TILE_N / 2) +
                (k_block_idx + col_half * NUM_K_BLOCKS_HALF) * K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int fp4_col = fp4_col_base + ((local_tid + j) & 7);
                const uint32_t swizzled_addr = Fp4Tile::idx(tile_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(bf16_reg[col_half][i][j].x) * s_enc,
                    __bfloat162float(bf16_reg[col_half][i][j].y) * s_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
                if (export_canonical_smem_base != 0) {
                    asm volatile("{st.shared.b8 [%0], %1;}"
                        :: "r"(export_canonical_smem_base + tile_row * QUANT_TILE_N + fp4_col),
                           "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
                }
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t *>(&out_scales.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[1][0])));
    }
}

template <typename C>
__device__ inline void kernel_shared_a_cross_cta(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::SHARE_A_ACROSS_CTAS, "shared-A helper only supports shared-A configs");
    static_assert(C::CLUSTER_SIZE == 2 && C::COL_TILES_PER_BLOCK == 1,
                  "shared-A helper expects 2-CTA clusters and one local output tile per CTA");

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int row_block_idx = blockIdx.y;
    const int col_block_idx = clusterIdx().x;
    const int cta_col_idx = col_block_idx * C::CLUSTER_SIZE + cta_id;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;
    if (threadIdx.x == 0 && g.D.rows() == 256 && g.D.cols() == 512 && g.A_bf16.cols() == 256 && blockIdx.y == 0) {
        printf("shared_a_2cta dbg: blockIdx.x=%d clusterIdx.x=%d cta_id=%d cta_col_idx=%d\n",
               (int)blockIdx.x, col_block_idx, cta_id, cta_col_idx);
    }

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
    typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::a_export_t (&a_remote_fp4)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::a_export_t, C::LOAD_PIPE_DEPTH>();
    typename G::a_scale_export_t (&a_remote_sc)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::a_scale_export_t, C::LOAD_PIPE_DEPTH>();

    tensor_allocator<1, C::TMEM_NCTA, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore b_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore a_stage_released[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_remote_ready[C::LOAD_PIPE_DEPTH];
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            init_semaphore(a_quant_done[stage], 0, 1);
            init_semaphore(b_quant_done[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
            init_semaphore(a_stage_released[stage], 0, C::CLUSTER_SIZE);
            init_semaphore(a_remote_ready[stage], 0, 1);
        }
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
            init_semaphore(b_bf16_sub_arrived[sub], 0, 1);
        }
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
        tm_allocator.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }
    wait(tmem_provisioned, 0);
    if (warpgroup_id == 0 || warpgroup_id == 3) {
        tm_allocator.set_addr(tmem_addr);
    }
    everyone::tma::cluster::wait();
    __syncthreads();

    if (warpgroup_id == 1) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        uint32_t release_phasebits = 0xFFFF0000;
        uint32_t remote_ready_phasebits = 0xFFFF0000;
        int sub_phase = 0;
        int iter_idx = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (cta_id == 0 && is_leader) {
                running_amax_a = 0.0f;
            }
            warpgroup::sync(a_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx, ++iter_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));

            if (cta_id == 0) {
                if (iter_idx >= C::LOAD_PIPE_DEPTH) {
                    if (is_leader) {
                        arrive(a_stage_released[stage], 1);
                    }
                    wait(a_stage_released[stage], get_phasebit<1>(release_phasebits, stage));
                }

                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                        tma::load_async(
                            quant_buf_a.bf16_tile, g.A_bf16,
                            {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            a_bf16_sub_arrived[sub]);
                    }
                    wait(a_bf16_sub_arrived[sub], sub_phase);
                    const uint32_t export_base = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&a_remote_fp4[stage].data[0]));
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            a_quant_bar_id, &running_amax_a, warp_max_buf_a, export_base);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub, export_base);
                    }
                    warpgroup::sync(a_quant_bar_id);
                }
                sub_phase ^= 1;

                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_remote_sc[stage].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                    sizeof(input_scales[stage].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                if (local_tid == 0 && g.D.rows() == 256 && g.D.cols() == 512 && g.A_bf16.cols() == 256 &&
                    row_block_idx == 0 && red_block_idx == 0) {
                    const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(&input_tiles[stage].A.data[0]));
                    const uint32_t a_byte0 = nvfp4_fused_gemm::ld_shared_u8(decltype(input_tiles[stage].A)::idx(a_base, {0, 0}));
                    const uint32_t sc_byte0 = nvfp4_fused_gemm::ld_shared_u8(
                        static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])));
                    printf("shared_a_2cta A src cta=%d a0=%u sc0=%u\n", cta_id, a_byte0, sc_byte0);
                }

                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (is_leader) {
                    arrive(a_quant_done[stage], 1);
                    nvfp4_fused_gemm::arrive_remote_cluster(a_remote_ready[stage], 1, 1);
                }
                if (iter_idx >= C::LOAD_PIPE_DEPTH) {
                    update_phasebit<1>(release_phasebits, stage);
                }
            } else {
                if (iter_idx >= C::LOAD_PIPE_DEPTH && is_leader) {
                    nvfp4_fused_gemm::arrive_remote_cluster(a_stage_released[stage], 0, 1);
                }
                wait(a_remote_ready[stage], get_phasebit<1>(remote_ready_phasebits, stage));
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                nvfp4_fused_gemm::copy_shared_cluster_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_remote_sc[stage].data[0])),
                    0,
                    sizeof(input_scales[stage].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                nvfp4_fused_gemm::import_shared_cluster_canonical_fp4_tile(
                    input_tiles[stage].A,
                    C::Mb,
                    C::Kb / 2,
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_remote_fp4[stage].data[0])),
                    0,
                    local_tid,
                    128);

                warpgroup::sync(a_quant_bar_id);
                if (local_tid == 0 && g.D.rows() == 256 && g.D.cols() == 512 && g.A_bf16.cols() == 256 &&
                    row_block_idx == 0 && red_block_idx == 0) {
                    const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(&input_tiles[stage].A.data[0]));
                    const uint32_t a_byte0 = nvfp4_fused_gemm::ld_shared_u8(decltype(input_tiles[stage].A)::idx(a_base, {0, 0}));
                    const uint32_t sc_byte0 = nvfp4_fused_gemm::ld_shared_u8(
                        static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])));
                    printf("shared_a_2cta A dst cta=%d a0=%u sc0=%u\n", cta_id, a_byte0, sc_byte0);
                }
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    arrive(a_quant_done[stage], 1);
                }
                update_phasebit<1>(remote_ready_phasebits, stage);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if (warpgroup_id == 2) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int b_quant_bar_id = 3;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_b = 0.0f;
            }
            warpgroup::sync(b_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            #pragma unroll
            for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                if (is_leader) {
                    tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                    tma::load_async(
                        quant_buf_b.bf16_tile, g.B_bf16,
                        {cta_col_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                        b_bf16_sub_arrived[sub]);
                }
                wait(b_bf16_sub_arrived[sub], sub_phase);
                if constexpr (C::USE_CTA_AMAX) {
                    quantize_operand_subtile_cta_amax(
                        quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub,
                        b_quant_bar_id, &running_amax_b, warp_max_buf_b);
                } else {
                    quantize_operand_subtile_constant(
                        quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub);
                }
                warpgroup::sync(b_quant_bar_id);
            }
            sub_phase ^= 1;

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (is_leader) {
                arrive(b_quant_done[stage], 1);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if (warpgroup_id == 3 && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 0) {
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            uint32_t stage = 0;
            uint32_t mma_phasebits = 0;

            tensor_after_thread_sync();
            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                #pragma unroll
                for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                    auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &A_sc_sm_subtile =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                    auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &B_sc_sm_subtile =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                }

                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                if (red_block_idx == 0) {
                    mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                } else {
                    mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                }
                kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                mma_phasebits ^= (1u << stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
            tensor_commit<1>(outputs_arrived);
        }
    }

    if (warpgroup_id == 0) {
        wait(outputs_arrived, 0);
        const float a_sg_dec = C::USE_CTA_AMAX
            ? ((cta_id == 0 ? running_amax_a : nvfp4_fused_gemm::load_running_amax_remote<C>(&running_amax_a, 0)) /
               (6.0f * 448.0f))
            : SCALE_MAX_DEC;
        const float b_sg_dec = C::USE_CTA_AMAX ? running_amax_b / (6.0f * 448.0f) : SCALE_MAX_DEC;
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
            warpgroup::load_async(
                D_reg_fl,
                out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * epi));
            warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
            warp::copy(D_reg[epi], D_reg_fl);
        }

        tensor_load_wait();
        tensor_before_thread_sync();
        warpgroup::sync(1);

        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                g.D, output_tiles.D[epi % C::NUM_D_TILES],
                {row_block_idx, C::EPI_PIPE_DEPTH * cta_col_idx + epi});
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if constexpr (C::SHARE_A_ACROSS_CTAS) {
        kernel_shared_a_cross_cta<C>(g);
        return;
    }

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int row_block_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
    typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore b_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            init_semaphore(a_quant_done[stage], 0, 1);
            init_semaphore(b_quant_done[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
        }
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
            init_semaphore(b_bf16_sub_arrived[sub], 0, 1);
        }
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
        tm_allocator.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }
    wait(tmem_provisioned, 0);
    if (warpgroup_id == 0 || warpgroup_id == 3) {
        tm_allocator.set_addr(tmem_addr);
    }
    everyone::tma::cluster::wait();
    __syncthreads();

    if (warpgroup_id == 1) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_a = 0.0f;
            }
            warpgroup::sync(a_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            #pragma unroll
            for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                if (is_leader) {
                    tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                    tma::load_async(
                        quant_buf_a.bf16_tile, g.A_bf16,
                        {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                        a_bf16_sub_arrived[sub]);
                }
                wait(a_bf16_sub_arrived[sub], sub_phase);
                if constexpr (C::USE_CTA_AMAX) {
                    quantize_operand_subtile_cta_amax(
                        quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                        a_quant_bar_id, &running_amax_a, warp_max_buf_a);
                } else {
                    quantize_operand_subtile_constant(
                        quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub);
                }
                warpgroup::sync(a_quant_bar_id);
            }
            sub_phase ^= 1;

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (is_leader) {
                arrive(a_quant_done[stage], 1);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if (warpgroup_id == 2) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int b_quant_bar_id = 3;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_b = 0.0f;
            }
            warpgroup::sync(b_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            #pragma unroll
            for (int col_tile = 0; col_tile < C::COL_TILES_PER_BLOCK; ++col_tile) {
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                        tma::load_async(
                            quant_buf_b.bf16_tile, g.B_bf16,
                            {col_block_idx * C::COL_TILES_PER_BLOCK + col_tile,
                             red_block_idx * C::QUANT_SUB_TILES + sub},
                            b_bf16_sub_arrived[sub]);
                    }
                    wait(b_bf16_sub_arrived[sub], sub_phase);
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_b, input_tiles[stage].B[col_tile], input_scales[stage].B[col_tile], sub,
                            b_quant_bar_id, &running_amax_b, warp_max_buf_b);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_b, input_tiles[stage].B[col_tile], input_scales[stage].B[col_tile], sub);
                    }
                    warpgroup::sync(b_quant_bar_id);
                }
                sub_phase ^= 1;
            }

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (is_leader) {
                arrive(b_quant_done[stage], 1);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if (warpgroup_id == 3 && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 0) {
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            uint32_t stage = 0;
            uint32_t mma_phasebits = 0;

            if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    if (red_block_idx == 0) {
                        mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                    } else {
                        mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                    mma_phasebits ^= (1u << stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
            } else if constexpr (C::COL_TILES_PER_BLOCK == 2) {
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 8 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_0 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                        auto B_sc_tm_subtile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_1 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    if (red_block_idx == 0) {
                        mm_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mm_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                    } else {
                        mma_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mma_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                    mma_phasebits ^= (1u << stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
            } else {
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                auto out_tm_2 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(256);
                auto out_tm_3 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(384);
                auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 8 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_2 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 12 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_3 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_0 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                        auto B_sc_tm_subtile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_1 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_1, B_sc_sm_subtile_1);

                        auto B_sc_tm_subtile_2 = B_sc_tm_2.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_2 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[2].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_2, B_sc_sm_subtile_2);

                        auto B_sc_tm_subtile_3 = B_sc_tm_3.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_3 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[3].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_3, B_sc_sm_subtile_3);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_2 = B_sc_tm_2.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_3 = B_sc_tm_3.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    if (red_block_idx == 0) {
                        mm_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mm_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                        mm_ABt(out_tm_2, input_tiles[stage].A, input_tiles[stage].B[2], A_sc_tm_tile, B_sc_tm_tile_2);
                        mm_ABt(out_tm_3, input_tiles[stage].A, input_tiles[stage].B[3], A_sc_tm_tile, B_sc_tm_tile_3);
                    } else {
                        mma_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mma_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                        mma_ABt(out_tm_2, input_tiles[stage].A, input_tiles[stage].B[2], A_sc_tm_tile, B_sc_tm_tile_2);
                        mma_ABt(out_tm_3, input_tiles[stage].A, input_tiles[stage].B[3], A_sc_tm_tile, B_sc_tm_tile_3);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                    mma_phasebits ^= (1u << stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
            }
        }
    }

    if (warpgroup_id == 0) {
        wait(outputs_arrived, 0);
        const float a_sg_dec = C::USE_CTA_AMAX ? running_amax_a / (6.0f * 448.0f) : SCALE_MAX_DEC;
        const float b_sg_dec = C::USE_CTA_AMAX ? running_amax_b / (6.0f * 448.0f) : SCALE_MAX_DEC;
        if constexpr (C::COL_TILES_PER_BLOCK == 1) {
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

            rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                warpgroup::load_async(
                    D_reg_fl,
                    out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                        0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                warp::copy(D_reg[epi], D_reg_fl);
            }

            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);

            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                    g.D, output_tiles.D[epi % C::NUM_D_TILES],
                    {row_block_idx, C::EPI_PIPE_DEPTH * col_block_idx + epi});
            }
        } else if constexpr (C::COL_TILES_PER_BLOCK == 2) {
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

            auto store_dual_tile = [&](auto &out_tm_sel, int col_tile_offset) {
                rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(
                        D_reg_fl,
                        out_tm_sel.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                            0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                    warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                    warp::copy(D_reg[epi], D_reg_fl);
                }

                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                        g.D, output_tiles.D[epi % C::NUM_D_TILES],
                        {row_block_idx,
                         C::EPI_PIPE_DEPTH * (col_block_idx * C::COL_TILES_PER_BLOCK + col_tile_offset) + epi});
                }
            };

            store_dual_tile(out_tm_0, 0);
            store_dual_tile(out_tm_1, 1);
        } else {
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto out_tm_2 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(256);
            auto out_tm_3 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(384);

            auto store_quad_tile = [&](auto &out_tm_sel, int col_tile_offset) {
                rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(
                        D_reg_fl,
                        out_tm_sel.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                            0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                    warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                    warp::copy(D_reg[epi], D_reg_fl);
                }

                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                        g.D, output_tiles.D[epi % C::NUM_D_TILES],
                        {row_block_idx,
                         C::EPI_PIPE_DEPTH * (col_block_idx * C::COL_TILES_PER_BLOCK + col_tile_offset) + epi});
                }
            };

            store_quad_tile(out_tm_0, 0);
            store_quad_tile(out_tm_1, 1);
            store_quad_tile(out_tm_2, 2);
            store_quad_tile(out_tm_3, 3);
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
}

} // namespace nvfp4_fused_gemm_both_bf16
