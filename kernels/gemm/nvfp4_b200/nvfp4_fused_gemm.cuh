#pragma once
// ================================================================
// NVFP4 Fused Quantize + GEMM Kernel
// Accepts bf16 A input, quantizes on-the-fly to NVFP4, then GEMMs.
// B is pre-quantized NVFP4.
// D = quantize(A_bf16) × B_fp4^T
//
// Template flag USE_CTA_AMAX controls the global scale strategy:
//   false: constant SCALE_MAX = bf16_max/(6*448) — fast, no sync
//   true:  per-sub-tile amax via warpgroup reduction of the per-block
//          amaxes already computed during quantization. No pre-scan.
//          Running max tracked in SMEM for the epilogue.
//
// Architecture: 3 warpgroups (384 threads)
//   WG0 (consumer):  epilogue — TMEM → registers → TMA store to HBM
//   WG1 (quantizer): TMA-loads bf16 A sub-tiles → quantizes to FP4 + scales
//   WG2 (producer):  TMA-loads B FP4/scales → MMA orchestration
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_fused_gemm {

// Constant global scale (used when USE_CTA_AMAX=false)
static constexpr float SCALE_MAX_DEC = 65504.0f / (6.0f * 448.0f);
static constexpr float SCALE_MAX_ENC = 1.0f / SCALE_MAX_DEC;

// Quantization tile constants
static constexpr int QUANT_TILE_M = 128;
static constexpr int QUANT_TILE_N = 128;
static constexpr int K_BLOCK_SIZE = 16;

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI, bool _USE_CTA_AMAX = false, int _Mb = 256, bool _USE_PDL = true, int _CLUSTER_SIZE = 2, int _COL_TILES_PER_BLOCK = 1, int _TMEM_NCTA = _CLUSTER_SIZE, bool _SHARE_A_ACROSS_CTAS = false>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_Mb == 128 || _Mb == 256, "Fused kernel supports Mb=128 or Mb=256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_EPI_PIPE_DEPTH > 0);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_NUM_D_TILES > 0);
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2);
    static_assert(_TMEM_NCTA == 1 || _TMEM_NCTA == 2, "TMEM_NCTA must be 1 or 2");
    static_assert(_COL_TILES_PER_BLOCK == 1 || (_COL_TILES_PER_BLOCK == 2 && _Nb == 128),
                  "Dual-column fused backend currently supports Nb=128 only");
    static_assert(!_SHARE_A_ACROSS_CTAS || (_CLUSTER_SIZE == 2 && _TMEM_NCTA == 1 && _COL_TILES_PER_BLOCK == 1 && _Nb == 256 && _Mb == 128),
                  "Cross-CTA shared-A backend currently supports CLUSTER_SIZE=2, TMEM_NCTA=1, Mb=128, Nb=256");

    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    static constexpr bool USE_PDL = _USE_PDL;
    static constexpr bool USE_CTA_AMAX = _USE_CTA_AMAX;
    static constexpr int COL_TILES_PER_BLOCK = _COL_TILES_PER_BLOCK;
    static constexpr int TMEM_NCTA = _TMEM_NCTA;
    static constexpr bool SHARE_A_ACROSS_CTAS = _SHARE_A_ACROSS_CTAS;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;  // 384

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int BLOCK_M = Mb;
    static constexpr int ROW_SPLIT_CTAS = SHARE_A_ACROSS_CTAS ? 1 : CLUSTER_SIZE;
    static constexpr int ROW_SLICE = BLOCK_M / ROW_SPLIT_CTAS;
    static_assert(BLOCK_M % ROW_SPLIT_CTAS == 0, "BLOCK_M must divide evenly across row-split CTAs");
    static constexpr int CTA_N = Nb * COL_TILES_PER_BLOCK;
    static constexpr int BLOCK_N = CTA_N * (SHARE_A_ACROSS_CTAS ? CLUSTER_SIZE : 1);
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;

    static constexpr int QUANT_SUB_TILES = Kb / QUANT_TILE_N; // 2
    static constexpr int A_ROW_TILES_PER_BLOCK = BLOCK_M / QUANT_TILE_M;
    static_assert(ROW_SLICE == QUANT_TILE_M, "Current fused quantizer assumes 128-row per-CTA A slices");
};

template <typename C>
struct globals {
    using A_bf16_tile  = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<C::ROW_SLICE, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/C::TMEM_NCTA, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::ROW_SLICE, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_bf16_gl      = gl<bf16,      1,  1, -1, -1, A_bf16_tile>;
    using B_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,      1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,     1,  1,  1,  1>;
    using D_gl           = gl<bf16,      1,  1, -1, -1, D_tile>;

    A_bf16_gl      A_bf16;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;
    D_gl           D;

    D_gl           D_K;
    D_gl           D_V;
    int            q_dim;
    int            k_dim;
    bool           use_split_D;
    const float*   b_sg_per_tile;
    int            silu_dim;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B[C::COL_TILES_PER_BLOCK];
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::COL_TILES_PER_BLOCK][C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };
    struct quant_buf_t {
        A_bf16_tile bf16_tile;
    };
    struct a_export_t {
        alignas(16) uint8_t data[C::ROW_SLICE * (C::Kb/2)];
    };
    struct a_scale_export_t {
        alignas(16) uint8_t data[sizeof(A_sc_tile)];
    };

    __host__ inline dim3 grid() const {
        int d_cols = use_split_D ? (q_dim + k_dim + D_V.cols()) : D.cols();
        return dim3(min((D.rows()/C::ROW_SLICE)*(d_cols/C::CTA_N), num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int shared_a_transport_smem =
            C::SHARE_A_ACROSS_CTAS
                ? (sizeof(a_export_t) +
                   sizeof(a_scale_export_t) +
                   sizeof(a_export_t) * C::LOAD_PIPE_DEPTH +
                   sizeof(a_scale_export_t) * C::LOAD_PIPE_DEPTH)
                : 0;
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t) +
                                               sizeof(quant_buf_t) + 1024 +
                                               shared_a_transport_smem +
                                               64; // running_amax SMEM + warp_max_buf
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }

    uint8_t*       debug_cta0_a_ptr;
    uint8_t*       debug_cta1_a_ptr;
    uint8_t*       debug_cta0_sc_ptr;
    uint8_t*       debug_cta1_sc_ptr;
    int            debug_a_stride;
};

__device__ inline void st_shared_cluster_b8(uint32_t dst_addr, uint32_t value);
__device__ inline void st_shared_cluster_b32(uint32_t dst_addr, uint32_t value);
__device__ inline uint32_t map_cluster_addr(uint32_t local_addr, int target_cta);
__device__ inline void arrive_remote_cluster(semaphore &sem, int target_cta, uint32_t count = 1);


// ================================================================
// Constant-scale quantization for one 128x128 bf16 sub-tile.
// Processes one 64-column half at a time so the fast path carries a
// smaller live register set than the CTA-amax variant.
// Called by all 128 threads of the quantizer warpgroup.
// ================================================================
template <typename G>
__device__ inline void quantize_subtile_constant(
    typename G::quant_buf_t &quant_buf,
    typename G::input_tiles_t &out_tile,
    typename G::input_scales_t &out_scales,
    int sub_tile_idx,
    uint32_t export_a_canonical_smem_base = 0,
    uint32_t remote_a_scale_local_smem_base = 0,
    int remote_target_cta = -1,
    bool mirror_remote_scale = false
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;  // 4
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;                      // 8

    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &A_bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8)%NUM_K_BLOCKS_HALF + col_half*NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*K_BLOCK_SIZE + ((local_tid+j)*2)%K_BLOCK_SIZE;
                const int offset = (tile_row*QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }

    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            bf16_2 _amax = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                _amax = __hmax2(_amax, __habs2(A_bf16_reg[col_half][i][j]));
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(_amax.x, _amax.y));
        }
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            A_sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * SCALE_MAX_ENC);

        const uint32_t a_tile_smem_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(&out_tile.A.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(A_sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec*SCALE_MAX_DEC, 1e-12f);
            const int fp4_col_base = sub_tile_idx * (QUANT_TILE_N/2) +
                                     (k_block_idx + col_half*NUM_K_BLOCKS_HALF)*K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int fp4_col = fp4_col_base + ((local_tid+j)&7);
                const uint32_t swizzled_addr = decltype(out_tile.A)::idx(
                    a_tile_smem_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x)*s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y)*s_enc
                };
                const uint32_t packed_fp4 =
                    static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(packed_fp4));
                if (export_a_canonical_smem_base != 0) {
                    asm volatile("{st.shared.b8 [%0], %1;}"
                        :: "r"(export_a_canonical_smem_base + tile_row * (QUANT_TILE_N) + fp4_col),
                           "r"(packed_fp4));
                }
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t*>(&out_scales.A.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[1][0])));
        if (mirror_remote_scale) {
            st_shared_cluster_b32(map_cluster_addr(remote_a_scale_local_smem_base + mma_base_0 + scale_offset, remote_target_cta),
                                  *reinterpret_cast<uint32_t*>(&A_sc_reg[0][0]));
            st_shared_cluster_b32(map_cluster_addr(remote_a_scale_local_smem_base + mma_base_1 + scale_offset, remote_target_cta),
                                  *reinterpret_cast<uint32_t*>(&A_sc_reg[1][0]));
        }
    }
}

// ================================================================
// CTA-amax quantization for one 128x128 bf16 sub-tile.
// Keeps the two 64-column halves live together so it can reduce a
// single sub-tile max before deriving per-block scales.
// Called by all 128 threads of the quantizer warpgroup.
// ================================================================
template <typename G>
__device__ inline void quantize_subtile_cta_amax(
    typename G::quant_buf_t &quant_buf,
    typename G::input_tiles_t &out_tile,
    typename G::input_scales_t &out_scales,
    int sub_tile_idx,
    int quant_sync_bar_id,
    float *running_amax,
    float *warp_max_buf,
    uint32_t export_a_canonical_smem_base = 0,
    uint32_t remote_a_scale_local_smem_base = 0,
    int remote_target_cta = -1,
    bool mirror_remote_scale = false
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;  // 4
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;                      // 8

    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &A_bf16_smem = quant_buf.bf16_tile;

    // Load from SMEM
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8)%NUM_K_BLOCKS_HALF + col_half*NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*K_BLOCK_SIZE + ((local_tid+j)*2)%K_BLOCK_SIZE;
                const int offset = (tile_row*QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }

    // Compute per-16-element block amaxes (both col halves)
    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            bf16_2 _amax = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                _amax = __hmax2(_amax, __habs2(A_bf16_reg[col_half][i][j]));
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(_amax.x, _amax.y));
        }
    }

    // Warpgroup-reduce all per-block amaxes to get sub-tile amax.
    float thread_max = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < 2; ch++)
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            thread_max = fmaxf(thread_max, amax_all[ch][i]);

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));

    const int warp_in_wg = local_tid / 32;
    const int lane = local_tid % 32;
    if (lane == 0) warp_max_buf[warp_in_wg] = thread_max;
    warpgroup::sync(quant_sync_bar_id);
    float subtile_amax = fmaxf(fmaxf(warp_max_buf[0], warp_max_buf[1]),
                               fmaxf(warp_max_buf[2], warp_max_buf[3]));
    warpgroup::sync(quant_sync_bar_id);

    subtile_amax = fmaxf(subtile_amax, 1e-12f);
    if (local_tid == 0) {
        float old = *running_amax;
        *running_amax = fmaxf(old, subtile_amax);
    }

    const float s_global_dec = subtile_amax / (6.0f * 448.0f);
    const float s_global_enc = (6.0f * 448.0f) / subtile_amax;

    // Compute FP8 block scales and quantize to FP4
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            A_sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * s_global_enc);

        const uint32_t a_tile_smem_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(&out_tile.A.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(A_sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec*s_global_dec, 1e-12f);
            const int fp4_col_base = sub_tile_idx * (QUANT_TILE_N/2) +
                                     (k_block_idx + col_half*NUM_K_BLOCKS_HALF)*K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int fp4_col = fp4_col_base + ((local_tid+j)&7);
                const uint32_t swizzled_addr = decltype(out_tile.A)::idx(
                    a_tile_smem_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x)*s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y)*s_enc
                };
                const uint32_t packed_fp4 =
                    static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(packed_fp4));
                if (export_a_canonical_smem_base != 0) {
                    asm volatile("{st.shared.b8 [%0], %1;}"
                        :: "r"(export_a_canonical_smem_base + tile_row * (QUANT_TILE_N) + fp4_col),
                           "r"(packed_fp4));
                }
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t*>(&out_scales.A.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[1][0])));
        if (mirror_remote_scale) {
            st_shared_cluster_b32(map_cluster_addr(remote_a_scale_local_smem_base + mma_base_0 + scale_offset, remote_target_cta),
                                  *reinterpret_cast<uint32_t*>(&A_sc_reg[0][0]));
            st_shared_cluster_b32(map_cluster_addr(remote_a_scale_local_smem_base + mma_base_1 + scale_offset, remote_target_cta),
                                  *reinterpret_cast<uint32_t*>(&A_sc_reg[1][0]));
        }
    }
}


// SiLU activation
template <typename RT>
__device__ inline void apply_silu_inplace(RT &D_reg) {
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                auto &v = D_reg.tiles[i][j].data[k];
                v.x = v.x / (1.0f + __expf(-v.x));
                v.y = v.y / (1.0f + __expf(-v.y));
            }
        }
    }
}

__device__ inline uint32_t map_cluster_addr(uint32_t local_addr, int target_cta) {
    if (target_cta == cluster_ctarank()) {
        return local_addr;
    }
    uint32_t mapped_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(mapped_addr)
                 : "r"(local_addr), "r"(target_cta));
    return mapped_addr;
}

__device__ inline uint32_t ld_shared_cluster_u32(uint32_t src_addr) {
    uint32_t value;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n"
                 : "=r"(value)
                 : "r"(src_addr));
    return value;
}

__device__ inline uint32_t ld_shared_cluster_u8(uint32_t src_addr) {
    uint32_t value;
    asm volatile("ld.shared::cluster.b8 %0, [%1];\n"
                 : "=r"(value)
                 : "r"(src_addr));
    return value;
}

__device__ inline uint32_t ld_shared_u8(uint32_t src_addr) {
    uint32_t value;
    asm volatile("ld.shared.b8 %0, [%1];\n"
                 : "=r"(value)
                 : "r"(src_addr));
    return value;
}

__device__ inline uint32_t ld_shared_u32(uint32_t src_addr) {
    uint32_t value;
    asm volatile("ld.shared.b32 %0, [%1];\n"
                 : "=r"(value)
                 : "r"(src_addr));
    return value;
}

__device__ inline void st_shared_cluster_b8(uint32_t dst_addr, uint32_t value) {
    asm volatile("st.shared::cluster.b8 [%0], %1;\n" :: "r"(dst_addr), "r"(value));
}

__device__ inline void st_shared_cluster_b32(uint32_t dst_addr, uint32_t value) {
    asm volatile("st.shared::cluster.b32 [%0], %1;\n" :: "r"(dst_addr), "r"(value));
}

__device__ inline void arrive_remote_cluster(semaphore &sem, int target_cta, uint32_t count) {
    const uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    const uint32_t remote_addr = map_cluster_addr(local_addr, target_cta);
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
                 :: "r"(remote_addr), "r"(count)
                 : "memory");
}

template <typename T>
__device__ inline T ld_shared_cluster(uint32_t src_addr) {
    static_assert(sizeof(T) == sizeof(uint32_t), "ld_shared_cluster only supports 32-bit types");
    const uint32_t bits = ld_shared_cluster_u32(src_addr);
    return *reinterpret_cast<const T*>(&bits);
}

__device__ inline void copy_shared_cluster_bytes(
    uint32_t dst_local_addr,
    uint32_t src_local_addr,
    int src_cta,
    int bytes,
    int thread_linear_idx,
    int thread_count
) {
    for (int byte_off = thread_linear_idx * 4; byte_off < bytes; byte_off += thread_count * 4) {
        const uint32_t value = ld_shared_cluster_u32(
            map_cluster_addr(src_local_addr + byte_off, src_cta));
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(dst_local_addr + byte_off), "r"(value));
    }
}

__device__ inline void copy_shared_cluster_bytes_b8(
    uint32_t dst_local_addr,
    uint32_t src_local_addr,
    int src_cta,
    int bytes,
    int thread_linear_idx,
    int thread_count
) {
    for (int byte_off = thread_linear_idx; byte_off < bytes; byte_off += thread_count) {
        asm volatile("st.shared.b8 [%0], %1;\n"
                     :: "r"(dst_local_addr + byte_off),
                        "r"(ld_shared_cluster_u8(map_cluster_addr(src_local_addr + byte_off, src_cta))));
    }
}

__device__ inline void copy_shared_local_bytes(
    uint32_t dst_local_addr,
    uint32_t src_local_addr,
    int bytes,
    int thread_linear_idx,
    int thread_count
) {
    for (int byte_off = thread_linear_idx * 4; byte_off < bytes; byte_off += thread_count * 4) {
        asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(dst_local_addr + byte_off), "r"(ld_shared_u32(src_local_addr + byte_off)));
    }
}

__device__ inline void mirror_shared_bytes_to_cluster(
    uint32_t src_local_addr,
    uint32_t dst_local_addr,
    int dst_cta,
    int bytes,
    int thread_linear_idx,
    int thread_count
) {
    for (int byte_off = thread_linear_idx * 4; byte_off < bytes; byte_off += thread_count * 4) {
        st_shared_cluster_b32(
            map_cluster_addr(dst_local_addr + byte_off, dst_cta),
            ld_shared_u32(src_local_addr + byte_off));
    }
}

template <typename Tile>
__device__ inline void import_shared_cluster_canonical_fp4_tile(
    Tile &dst_tile,
    int rows,
    int cols,
    uint32_t src_local_addr,
    int src_cta,
    int thread_linear_idx,
    int thread_count
) {
    const uint32_t dst_local_base = static_cast<uint32_t>(__cvta_generic_to_shared(&dst_tile.data[0]));
    const int total_bytes = rows * cols;
    for (int byte_off = thread_linear_idx * 4; byte_off < total_bytes; byte_off += thread_count * 4) {
        const uint32_t packed = ld_shared_cluster_u32(
            map_cluster_addr(src_local_addr + byte_off, src_cta));
        #pragma unroll
        for (int lane_byte = 0; lane_byte < 4; ++lane_byte) {
            const int linear_idx = byte_off + lane_byte;
            if (linear_idx < total_bytes) {
                const int row = linear_idx / cols;
                const int col = linear_idx % cols;
                const uint32_t dst_addr = Tile::idx(dst_local_base, {row, col});
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(dst_addr),
                       "r"((packed >> (8 * lane_byte)) & 0xFFu));
            }
        }
    }
}

template <typename Tile>
__device__ inline void import_local_canonical_fp4_tile(
    Tile &dst_tile,
    int rows,
    int cols,
    uint32_t src_local_addr,
    int thread_linear_idx,
    int thread_count
) {
    const uint32_t dst_local_base = static_cast<uint32_t>(__cvta_generic_to_shared(&dst_tile.data[0]));
    const int total_bytes = rows * cols;
    for (int linear_idx = thread_linear_idx; linear_idx < total_bytes; linear_idx += thread_count) {
        const int row = linear_idx / cols;
        const int col = linear_idx % cols;
        const uint32_t dst_addr = Tile::idx(dst_local_base, {row, col});
        asm volatile("{st.shared.b8 [%0], %1;}"
            :: "r"(dst_addr),
               "r"(ld_shared_u8(src_local_addr + linear_idx)));
    }
}

template <typename Tile>
__device__ inline void dump_swizzled_fp4_tile_canonical(
    const Tile &tile,
    int rows,
    int cols,
    uint8_t *dst_ptr,
    int dst_stride,
    int thread_linear_idx,
    int thread_count
) {
    const uint32_t tile_base = static_cast<uint32_t>(__cvta_generic_to_shared(&tile.data[0]));
    for (int linear_idx = thread_linear_idx; linear_idx < rows * cols; linear_idx += thread_count) {
        const int row = linear_idx / cols;
        const int col = linear_idx % cols;
        dst_ptr[row * dst_stride + col] =
            static_cast<uint8_t>(ld_shared_u8(Tile::idx(tile_base, {row, col})));
    }
}

__device__ inline void dump_shared_raw_bytes(
    const void *src_ptr,
    uint8_t *dst_ptr,
    int bytes,
    int thread_linear_idx,
    int thread_count
) {
    const uint32_t src_base = static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr));
    for (int idx = thread_linear_idx; idx < bytes; idx += thread_count) {
        dst_ptr[idx] = static_cast<uint8_t>(ld_shared_u8(src_base + idx));
    }
}

template <typename C>
__device__ inline int output_row_tile_idx(int row_block_idx, int cta_id) {
    return row_block_idx * C::A_ROW_TILES_PER_BLOCK + (C::SHARE_A_ACROSS_CTAS ? 0 : cta_id);
}

template <typename C>
__device__ inline int output_col_tile_idx(int col_block_idx, int cta_id, int col_tile = 0) {
    if constexpr (C::SHARE_A_ACROSS_CTAS) {
        return col_block_idx * C::CLUSTER_SIZE + cta_id;
    } else {
        return col_block_idx * C::COL_TILES_PER_BLOCK + col_tile;
    }
}

template <typename C>
__device__ inline float load_running_amax_remote(float *running_amax_ptr, int src_cta) {
    const uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(running_amax_ptr));
    return ld_shared_cluster<float>(map_cluster_addr(local_addr, src_cta));
}

template <typename C>
__device__ inline void kernel_shared_a_cross_cta(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::SHARE_A_ACROSS_CTAS, "Cross-CTA helper only supports shared-A configs");
    static_assert(C::CLUSTER_SIZE == 2, "Cross-CTA helper assumes two-CTA clusters");
    static_assert(C::TMEM_NCTA == 1, "Cross-CTA helper uses per-CTA local TMEM");
    static_assert(C::Nb == 256 && C::COL_TILES_PER_BLOCK == 1, "Cross-CTA helper expects one 256-column local tile");
    static_assert(!C::OVERLAP_EPI, "Cross-CTA helper currently supports non-overlapped epilogues only");

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::BLOCK_M;
    const int N_total = g.D.cols();
    const int num_col_blocks = N_total / C::BLOCK_N;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();
    typename G::quant_buf_t     &quant_buf                         = sm_allocator.allocate<G::quant_buf_t>();
    typename G::a_export_t      &a_export_fp4                      = sm_allocator.allocate<typename G::a_export_t>();
    typename G::a_scale_export_t &a_export_sc                     = sm_allocator.allocate<typename G::a_scale_export_t>();
    typename G::a_export_t      (&a_recv_fp4)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::a_export_t, C::LOAD_PIPE_DEPTH>();
    typename G::a_scale_export_t (&a_recv_sc)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::a_scale_export_t, C::LOAD_PIPE_DEPTH>();

    __shared__ float running_amax;
    __shared__ float warp_max_buf[4];

    tensor_allocator<1, C::TMEM_NCTA, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_stage_released[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_payload_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_scale_arrived[C::LOAD_PIPE_DEPTH];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
            init_semaphore(a_stage_released[i], 0, C::CLUSTER_SIZE);
            init_semaphore(a_quant_done[i], 0, 1);
            init_semaphore(a_payload_arrived[i], 0, 1);
            init_semaphore(a_scale_arrived[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, 1);
        #pragma unroll
        for (int s = 0; s < C::QUANT_SUB_TILES; ++s) {
            init_semaphore(bf16_sub_arrived[s], 0, 1);
        }
        if constexpr (C::USE_CTA_AMAX) {
            running_amax = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == C::CONSUMER_WARPGROUPS) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int quant_sync_bar_id = 2;
        uint32_t release_phasebits = 0xFFFF0000;
        uint32_t payload_arrived_phasebits = 0;
        int iter_idx = 0;

        if constexpr (C::USE_PDL) {
            if (warp::laneid() == 0) pdl::wait();
        }
        everyone::tma::cluster::wait();

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;

            if constexpr (C::USE_CTA_AMAX) {
                if (cta_id == 0) {
                    if (is_leader) running_amax = 0.0f;
                    warpgroup::sync(quant_sync_bar_id);
                }
            }

            int bf16_sub_phase = 0;
            for (int i = 0; i < num_red_blocks; ++i, ++iter_idx) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));

                if (cta_id == 0) {
                    if (iter_idx >= C::LOAD_PIPE_DEPTH) {
                        if (is_leader) arrive(a_stage_released[stage], 1);
                        wait(a_stage_released[stage], get_phasebit<1>(release_phasebits, stage));
                    }

                    #pragma unroll
                    for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                        if (is_leader) {
                            tma::expect(bf16_sub_arrived[sub], quant_buf.bf16_tile);
                            tma::load_async(
                                quant_buf.bf16_tile,
                                g.A_bf16,
                                {row_block_idx * C::A_ROW_TILES_PER_BLOCK, i * C::QUANT_SUB_TILES + sub},
                                bf16_sub_arrived[sub]
                            );
                        }
                        wait(bf16_sub_arrived[sub], bf16_sub_phase);
                        const uint32_t export_a_canonical_smem_base = static_cast<uint32_t>(
                            __cvta_generic_to_shared(&a_export_fp4.data[0]));
                        if constexpr (C::USE_CTA_AMAX) {
                            quantize_subtile_cta_amax<G>(
                                quant_buf, input_tiles[stage], input_scales[stage],
                                sub, quant_sync_bar_id, &running_amax, warp_max_buf,
                                export_a_canonical_smem_base);
                        } else {
                            quantize_subtile_constant<G>(
                                quant_buf, input_tiles[stage], input_scales[stage], sub,
                                export_a_canonical_smem_base);
                        }
                        warpgroup::sync(quant_sync_bar_id);
                    }
                    bf16_sub_phase ^= 1;
                    copy_shared_local_bytes(
                        static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc.data[0])),
                        static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                        sizeof(input_scales[stage].A),
                        local_tid,
                        128);
                    warpgroup::sync(quant_sync_bar_id);

                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                    if (is_leader) {
                        const int target_cta = 1;
                        tma::cluster::expect_bytes(
                            a_payload_arrived[stage],
                            sizeof(typename G::a_export_t),
                            target_cta);
                        tma::cluster::expect_bytes(
                            a_scale_arrived[stage],
                            sizeof(typename G::a_scale_export_t),
                            target_cta);
                        tma::cluster::store_async(
                            reinterpret_cast<void*>(&a_recv_fp4[stage]),
                            reinterpret_cast<void*>(&a_export_fp4),
                            sizeof(typename G::a_export_t),
                            target_cta,
                            a_payload_arrived[stage]);
                        tma::cluster::store_async(
                            reinterpret_cast<void*>(&a_recv_sc[stage]),
                            reinterpret_cast<void*>(&a_export_sc),
                            sizeof(typename G::a_scale_export_t),
                            target_cta,
                            a_scale_arrived[stage]);
                        tma::store_async_read_wait<0>();
                        arrive(a_quant_done[stage], 1);
                    }
                    if (cluster_id == 0 && block_idx == 0 && i == 0) {
                        if (g.debug_cta0_a_ptr != nullptr) {
                            dump_shared_raw_bytes(
                                &a_export_fp4.data[0], g.debug_cta0_a_ptr,
                                C::ROW_SLICE * (C::Kb/2), local_tid, 128);
                        }
                        if (g.debug_cta0_sc_ptr != nullptr) {
                            dump_shared_raw_bytes(
                                &a_export_sc.data[0], g.debug_cta0_sc_ptr,
                                sizeof(input_scales[stage].A), local_tid, 128);
                        }
                    }
                    if (iter_idx >= C::LOAD_PIPE_DEPTH) {
                        update_phasebit<1>(release_phasebits, stage);
                    }
                } else {
                    if (iter_idx >= C::LOAD_PIPE_DEPTH && is_leader) {
                        arrive_remote_cluster(a_stage_released[stage], 0, 1);
                    }
                    wait(a_payload_arrived[stage], get_phasebit<0>(payload_arrived_phasebits, stage));
                    wait(a_scale_arrived[stage], get_phasebit<0>(payload_arrived_phasebits, stage));
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    copy_shared_local_bytes(
                        static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                        static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc[stage].data[0])),
                        sizeof(input_scales[stage].A),
                        local_tid,
                        128);
                    warpgroup::sync(quant_sync_bar_id);
                    import_local_canonical_fp4_tile(
                        input_tiles[stage].A,
                        C::ROW_SLICE,
                        C::Kb / 2,
                        static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4[stage].data[0])),
                        local_tid,
                        128);

                    warpgroup::sync(quant_sync_bar_id);
                    if (cluster_id == 0 && block_idx == 0 && i == 0) {
                        if (g.debug_cta1_a_ptr != nullptr) {
                            dump_shared_raw_bytes(
                                &a_recv_fp4[stage].data[0],
                                g.debug_cta1_a_ptr,
                                C::ROW_SLICE * (C::Kb/2), local_tid, 128);
                        }
                        if (g.debug_cta1_sc_ptr != nullptr) {
                            dump_shared_raw_bytes(
                                &a_recv_sc[stage].data[0], g.debug_cta1_sc_ptr,
                                sizeof(input_scales[stage].A), local_tid, 128);
                        }
                    }
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                    if (is_leader) {
                        arrive(a_quant_done[stage], 1);
                    }
                    update_phasebit<0>(payload_arrived_phasebits, stage);
                }

                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        }
    }
    else if (warpgroup_id == C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS) {
        if (warp::elect_leader()) {
            int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
            if (warp_id == 3) {
                if constexpr (C::USE_PDL) pdl::wait();
                everyone::tma::cluster::wait();
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                    int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                    int cta_col_idx = output_col_tile_idx<C>(col_block_idx, cta_id);

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        tma::load_async(input_tiles[stage].B[0], g.B, {cta_col_idx, i}, tiles_arrived[stage]);
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
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                    int cta_col_idx = output_col_tile_idx<C>(col_block_idx, cta_id);

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        #pragma unroll
                        for (int sc = 0; sc < C::B_SC_SIZE; ++sc) {
                            tma::load_async(input_scales[stage].B[0][sc], g.B_sc, {cta_col_idx * C::B_SC_SIZE + sc, i, 0}, scales_arrived[stage]);
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            } else if (warp_id == 0) {
                uint32_t mma_a_quant_phasebits = 0;
                everyone::tma::cluster::wait();
                wait(tmem_provisioned, 0);
                tm_allocator.set_addr(tmem_addr);

                auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::B_SC_SIZE*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256 + 4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                    tensor_after_thread_sync();
                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(a_quant_done[stage], (mma_a_quant_phasebits >> stage) & 0b1);
                        tma::expect_bytes(scales_arrived[stage], C::B_SC_SIZE * sizeof(typename G::B_sc_tile));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));

                        #pragma unroll
                        for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16 + ii*16);
                            auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16*32*ii);
                            load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                            #pragma unroll
                            for (int sc = 0; sc < C::B_SC_SIZE; ++sc) {
                                auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*(16*C::B_SC_SIZE) + ii*C::B_SC_SIZE*16 + sc*16);
                                auto &B_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][sc].data[0]) + 16*32*ii);
                                load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                            }
                        }

                        tma::expect_bytes(tiles_arrived[stage], sizeof(typename G::B_fp4x2_tile));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                        auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16);
                        auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16*C::B_SC_SIZE>>(stage*C::MMA_PER_TILE*(16*C::B_SC_SIZE));
                        if (i == 0) {
                            mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                        } else {
                            mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                        }
                        kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                        update_phasebit<0>(phasebits, stage);
                        mma_a_quant_phasebits ^= (1u << stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                    tensor_commit<1>(outputs_arrived);
                    update_phasebit<1>(phasebits, 0);
                }
            }
        }
    }
    else {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        const float default_b_sg = g.B_sc_global[{0}];

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            int row_tile_idx = output_row_tile_idx<C>(row_block_idx, cta_id);
            int col_tile_idx = output_col_tile_idx<C>(col_block_idx, cta_id);

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            float a_sg_dec = SCALE_MAX_DEC;
            if constexpr (C::USE_CTA_AMAX) {
                a_sg_dec = ((cta_id == 0) ? running_amax : load_running_amax_remote<C>(&running_amax, 0)) / (6.0f * 448.0f);
            }

            rt_bf<C::ROW_SLICE / 4, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                rt_fl<C::ROW_SLICE / 4, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                float gs = a_sg_dec * default_b_sg;
                if (g.b_sg_per_tile != nullptr) {
                    gs = a_sg_dec * g.b_sg_per_tile[col_tile_idx];
                }
                warp::mul(D_reg_fl, D_reg_fl, gs);
                warp::copy(D_reg[i], D_reg_fl);
            }

            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                arrive(outputs_finished);
            }

            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                    g.D,
                    output_tiles.D[i%C::NUM_D_TILES],
                    {row_tile_idx, C::EPI_PIPE_DEPTH*col_tile_idx + i}
                );
            }
            update_phasebit<0>(phasebits, 0);
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

// ================================================================
// Main kernel
// ================================================================
template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if constexpr (C::SHARE_A_ACROSS_CTAS) {
        kernel_shared_a_cross_cta<C>(g);
    } else {

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
        if (g.use_split_D) {
            g.D_K.template prefetch_tma<typename G::D_tile>();
            g.D_V.template prefetch_tma<typename G::D_tile>();
        }
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int N_total = g.use_split_D ? (g.q_dim + g.k_dim + g.D_V.cols()) : g.D.cols();
    const int num_col_blocks = N_total / C::BLOCK_N;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();
    typename G::quant_buf_t     &quant_buf                         = sm_allocator.allocate<G::quant_buf_t>();

    // SMEM for CTA-amax mode: running max amax + warp reduction buffer
    __shared__ float running_amax;
    __shared__ float warp_max_buf[4];

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Barriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore bf16_sub_arrived[C::QUANT_SUB_TILES];
    // Cluster-aggregated A-quantization completion for the single MMA warp on CTA 0.
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
            init_semaphore(a_quant_done[i], 0, C::CLUSTER_SIZE);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int s = 0; s < C::QUANT_SUB_TILES; s++)
            init_semaphore(bf16_sub_arrived[s], 0, 1);
        if constexpr (C::USE_CTA_AMAX)
            running_amax = 0.0f;
    }
    everyone::tma::cluster::arrive_aligned();

    // ================================================================
    // WG1: Quantizer
    // ================================================================
    if (warpgroup_id == C::CONSUMER_WARPGROUPS) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int quant_sync_bar_id = 2;

        if constexpr (C::USE_PDL) {
            if (warp::laneid() == 0) pdl::wait();
        }
        everyone::tma::cluster::wait();

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;

            // Reset running amax for this output tile
            if constexpr (C::USE_CTA_AMAX) {
                if (is_leader) running_amax = 0.0f;
                warpgroup::sync(quant_sync_bar_id);
            }

            int bf16_sub_phase = 0; // Alternates each K iteration

            for (int i = 0; i < num_red_blocks; ++i) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));

                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; sub++) {
                    if (is_leader) {
                        tma::expect(bf16_sub_arrived[sub], quant_buf.bf16_tile);
                        tma::load_async(quant_buf.bf16_tile, g.A_bf16,
                            {row_block_idx*2 + cta_id, i * C::QUANT_SUB_TILES + sub},
                            bf16_sub_arrived[sub]);
                    }
                    wait(bf16_sub_arrived[sub], bf16_sub_phase);

                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_subtile_cta_amax<G>(
                            quant_buf, input_tiles[stage], input_scales[stage],
                            sub, quant_sync_bar_id, &running_amax, warp_max_buf);
                    } else {
                        quantize_subtile_constant<G>(
                            quant_buf, input_tiles[stage], input_scales[stage], sub);
                    }

                    // All four warps must finish consuming quant_buf before the next sub-tile TMA overwrites it.
                    warpgroup::sync(quant_sync_bar_id);
                }
                bf16_sub_phase ^= 1; // Toggle phase for next K iteration

                // Publish direct DSMEM writes before CTA 0's MMA warp reads remote A/scales.
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                if (is_leader) {
                    if (cta_id == 0) {
                        arrive(a_quant_done[stage], 1);
                    } else {
                        tma::cluster::arrive(a_quant_done[stage], 0, 1);
                    }
                }

                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        }
    }
    // ================================================================
    // WG2: Producer — TMA loads B + MMA
    // ================================================================
    else if (warpgroup_id == C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS) {
        if (warp::elect_leader()) {
            int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
            if (warp_id == 3) {
                if constexpr (C::USE_PDL) pdl::wait();
                everyone::tma::cluster::wait();
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                    int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                            tma::cluster::load_async(input_tiles[stage].B[0], g.B,
                                {col_block_idx*2 + cta_id, i},
                                tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                        } else {
                            #pragma unroll
                            for (int col_tile = 0; col_tile < C::COL_TILES_PER_BLOCK; ++col_tile) {
                                tma::cluster::load_async(input_tiles[stage].B[col_tile], g.B,
                                    {(col_block_idx*C::COL_TILES_PER_BLOCK + col_tile)*2 + cta_id, i},
                                    tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                            }
                        }
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
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        if constexpr (C::COL_TILES_PER_BLOCK == 1 && C::B_SC_SIZE == 2) {
                            tma::cluster::load_async(input_scales[stage].B[0][cta_id], g.B_sc,
                                {col_block_idx*2 + cta_id, i, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        } else if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                            if (cta_id == 0) {
                                tma::cluster::load_async(input_scales[stage].B[0][0], g.B_sc,
                                    {col_block_idx, i, 0},
                                    scales_arrived[stage], (uint16_t)(0b11), 0);
                            }
                        } else if (cta_id == 0) {
                            #pragma unroll
                            for (int col_tile = 0; col_tile < C::COL_TILES_PER_BLOCK; ++col_tile) {
                                tma::cluster::load_async(input_scales[stage].B[col_tile][0], g.B_sc,
                                    {col_block_idx*C::COL_TILES_PER_BLOCK + col_tile, i, 0},
                                    scales_arrived[stage], (uint16_t)(0b11), 0);
                            }
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            } else if (cta_id == 0 && warp_id == 0) {
                // MMA orchestration
                uint32_t mma_a_quant_phasebits = 0;
                everyone::tma::cluster::wait();
                wait(tmem_provisioned, 0);
                tm_allocator.set_addr(tmem_addr);
                if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                    auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
                    auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

                    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        tensor_after_thread_sync();
                        for (int i = 0; i < num_red_blocks; i++) {
                            tma::cluster::wait(a_quant_done[stage], (mma_a_quant_phasebits >> stage) & 0b1);
                            tma::expect_bytes(scales_arrived[stage],
                                C::CLUSTER_SIZE * C::B_SC_SIZE * sizeof(typename G::B_sc_tile));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));

                            #pragma unroll
                            for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                                auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                                auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                                load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                                auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16);
                                auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][0].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                                if constexpr (C::B_SC_SIZE == 2) {
                                    auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16+16);
                                    auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][1].data[0])+16*32*ii);
                                    load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                                }
                            }

                            tma::expect_bytes(tiles_arrived[stage],
                                C::CLUSTER_SIZE * sizeof(typename G::B_fp4x2_tile));
                            wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));

                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                            if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0],
                                                A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                                B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                                inputs_finished[stage]);
                            else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0],
                                                A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                                B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                                inputs_finished[stage]);
                            update_phasebit<0>(phasebits, stage);
                            mma_a_quant_phasebits ^= (1u << stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                        tensor_commit<2>(outputs_arrived);
                        update_phasebit<1>(phasebits, 0);
                    }
                } else {
                    auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                    auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE>>(256);
                    auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE>>(256 + 4*C::MMA_PER_TILE);
                    auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE>>(256 + 8*C::MMA_PER_TILE);

                    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        tensor_after_thread_sync();
                        for (int i = 0; i < num_red_blocks; i++) {
                            tma::cluster::wait(a_quant_done[stage], (mma_a_quant_phasebits >> stage) & 0b1);
                            tma::expect_bytes(scales_arrived[stage],
                                C::CLUSTER_SIZE * C::COL_TILES_PER_BLOCK * sizeof(typename G::B_sc_tile));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));

                            #pragma unroll
                            for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                                auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                                auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                                load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);

                                auto B_sc_tm_subtile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                                auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][0].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                                auto B_sc_tm_subtile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                                auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1][0].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                            }

                            tma::expect_bytes(tiles_arrived[stage],
                                C::CLUSTER_SIZE * C::COL_TILES_PER_BLOCK * sizeof(typename G::B_fp4x2_tile));
                            wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));

                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                            if (i == 0) {
                                mm2_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0],
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                        B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0));
                                mm2_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1],
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                        B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                        inputs_finished[stage]);
                            } else {
                                mma2_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0],
                                         A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                         B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0));
                                mma2_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1],
                                         A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                         B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                         inputs_finished[stage]);
                            }
                            update_phasebit<0>(phasebits, stage);
                            mma_a_quant_phasebits ^= (1u << stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                        tensor_commit<2>(outputs_arrived);
                        update_phasebit<1>(phasebits, 0);
                    }
                }
            }
        }
    }
    // ================================================================
    // WG0: Consumer — epilogue
    // ================================================================
    else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        const float default_b_sg = g.B_sc_global[{0}];

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            float a_sg_dec;
            if constexpr (C::USE_CTA_AMAX) {
                a_sg_dec = running_amax / (6.0f * 448.0f);
            } else {
                a_sg_dec = SCALE_MAX_DEC;
            }

            if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

                if constexpr (C::OVERLAP_EPI) {
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg;
                        warpgroup::load_async(D_reg, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                        if (i == C::EPI_PIPE_DEPTH - 1) {
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            warpgroup::sync(1);
                            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                        }
                        {
                            float gs = a_sg_dec * default_b_sg;
                            if (g.b_sg_per_tile != nullptr) {
                                gs = a_sg_dec * g.b_sg_per_tile[col_block_idx];
                            }
                            warp::mul(D_reg, D_reg, gs);
                        }
                        if (g.silu_dim > 0) {
                            int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.silu_dim) apply_silu_inplace(D_reg);
                        }
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg);
                        warpgroup::sync(1);
                        if (g.use_split_D) {
                            int col_offset = C::EPI_PIPE_DEPTH*col_block_idx + i;
                            int col_offset_elems = col_offset * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.q_dim) {
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, col_offset});
                            } else if (col_offset_elems < g.q_dim + g.k_dim) {
                                int k_col_offset = col_offset - (g.q_dim / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_K, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, k_col_offset});
                            } else {
                                int v_col_offset = col_offset - ((g.q_dim + g.k_dim) / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_V, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, v_col_offset});
                            }
                        } else {
                            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
                        }
                    }
                } else {
                    rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                        warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                        {
                            float gs = a_sg_dec * default_b_sg;
                            if (g.b_sg_per_tile != nullptr) {
                                gs = a_sg_dec * g.b_sg_per_tile[col_block_idx];
                            }
                            warp::mul(D_reg_fl, D_reg_fl, gs);
                        }
                        if (g.silu_dim > 0) {
                            int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.silu_dim) apply_silu_inplace(D_reg_fl);
                        }
                        warp::copy(D_reg[i], D_reg_fl);
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                        warpgroup::sync(1);
                        if (g.use_split_D) {
                            int col_offset = C::EPI_PIPE_DEPTH*col_block_idx + i;
                            int col_offset_elems = col_offset * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.q_dim) {
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, col_offset});
                            } else if (col_offset_elems < g.q_dim + g.k_dim) {
                                int k_col_offset = col_offset - (g.q_dim / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_K, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, k_col_offset});
                            } else {
                                int v_col_offset = col_offset - ((g.q_dim + g.k_dim) / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_V, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, v_col_offset});
                            }
                        } else {
                            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
                        }
                    }
                }
            } else {
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

                auto store_dual_tile = [&](auto &out_tm_sel, int col_tile_offset, bool release_outputs) {
                    rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                        warpgroup::load_async(D_reg_fl, out_tm_sel.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                        warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * default_b_sg);
                        warp::copy(D_reg[i], D_reg_fl);
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    if (release_outputs) {
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    }
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                            g.D, output_tiles.D[i%C::NUM_D_TILES],
                            {row_block_idx*2 + cta_id,
                             C::EPI_PIPE_DEPTH*(col_block_idx*C::COL_TILES_PER_BLOCK + col_tile_offset) + i});
                    }
                };

                store_dual_tile(out_tm_0, 0, false);
                store_dual_tile(out_tm_1, 1, true);
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
    }
}

} // namespace nvfp4_fused_gemm
