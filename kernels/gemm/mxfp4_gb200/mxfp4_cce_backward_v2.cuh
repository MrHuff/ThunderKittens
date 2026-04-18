#pragma once
// ================================================================
// MXFP4 CCE Backward v2 — Fused Softmax Gradient + FP4 Quantization
//
// Same architecture as NVFP4 backward v2 but using MXFP4 format:
//   - E8M0 block scales (vs FP8 E4M3 in NVFP4)
//   - Per-32-element scale blocks (vs per-16 in NVFP4)
//   - Fixed MXFP4_ALPHA = 1/36 (vs dynamic global_scale)
//   - MMA_PER_TILE = 2 (128-element K-reduction)
//
// Template flag USE_BF16_ACCUM:
//   true  → store G as BF16 via TMA (for BF16 dE/dC GEMMs)
//   false → quantize G to MXFP4 on-the-fly (for FP4 dE/dC GEMMs)
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace mxfp4_cce_backward_v2 {

// =========================================================================
// Config
// =========================================================================
// QUANT_MODE: 0=RTE (default), 1=ENCODE-centric (ceil), 2=DECODE-centric (floor)
template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _USE_BF16_ACCUM = true, bool _PINGPONG = true, int _QUANT_MODE = 0>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

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
    static constexpr int B_SC_SIZE = 1;        // Nb/128
    static constexpr int MMA_PER_TILE = 2;     // Kb/128 (MXFP4)

    static constexpr int NUM_D_TILES = 2;
    static constexpr bool USE_BF16_ACCUM = _USE_BF16_ACCUM;
    static constexpr int QUANT_MODE = _QUANT_MODE;  // 0=RTE, 1=ENCODE, 2=DECODE
};

// =========================================================================
// FP4 E2M1 quantization helpers
// =========================================================================
__device__ __forceinline__ uint8_t float_to_fp4(float val) {
    float aval = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t enc;
    if      (aval < 0.25f)  enc = 0;       // 0
    else if (aval < 0.75f)  enc = 1;       // 0.5
    else if (aval < 1.25f)  enc = 2;       // 1.0
    else if (aval < 1.75f)  enc = 3;       // 1.5
    else if (aval < 2.5f)   enc = 4;       // 2.0
    else if (aval < 3.5f)   enc = 5;       // 3.0
    else if (aval < 5.0f)   enc = 6;       // 4.0
    else                    enc = 7;       // 6.0
    return sign | enc;
}

__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float rcp_scale) {
    uint8_t q0 = float_to_fp4(v0 * rcp_scale);
    uint8_t q1 = float_to_fp4(v1 * rcp_scale);
    return q0 | (q1 << 4);
}

// E8M0 conversion: float → pure exponent with round-ties-to-even
__device__ __forceinline__ uint8_t float_to_e8m0_rn(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t val_u32 = __float_as_uint(val);
    uint8_t exponent = (val_u32 >> 23) & 0xFF;
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    constexpr uint32_t half = 1u << 22;  // 2^(23-1)
    bool round_up = (mantissa > half) || (mantissa == half && (exponent & 1));
    if (round_up && exponent < 0xFE) {
        ++exponent;
    }
    return exponent;
}

// Encode-centric: ceil exponent → 2^exp >= val always (scale >= amax → no clipping)
__device__ __forceinline__ uint8_t float_to_e8m0_ceil(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    if (mant > 0 && exp < 0xFE) ++exp;
    return exp;
}

// Decode-centric: floor exponent → 2^exp <= val always (scale < amax → fills range)
__device__ __forceinline__ uint8_t float_to_e8m0_floor(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    return exp;
}

// Dispatch based on mode: 0=RTE, 1=ENCODE, 2=DECODE
template<int MODE>  // 0=RTE, 1=ENCODE(ceil), 2=DECODE(floor)
__device__ __forceinline__ uint8_t float_to_e8m0_dispatch(float val) {
    if constexpr (MODE == 1) return float_to_e8m0_ceil(val);
    else if constexpr (MODE == 2) return float_to_e8m0_floor(val);
    else return float_to_e8m0_rn(val);
}

// Reciprocal of 2^(e8m0 - 127)
__device__ __forceinline__ float exp2f_rcp_e8m0(uint8_t e8m0) {
    if (e8m0 == 0) return 0.0f;
    // 2^(e8m0-127) has bit pattern (e8m0 << 23)
    // reciprocal = 2^(127-e8m0)
    uint32_t bits = (uint32_t)(254 - e8m0) << 23;
    return __uint_as_float(bits);
}

// =========================================================================
// Globals
// =========================================================================
template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;       // MXFP4: E8M0 scales
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    // FP4 output tile for quantized G
    using G_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;

    using A_fp4x2_gl   = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl      = gl<fp8e8m0,  -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl   = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl      = gl<fp8e8m0,  -1, -1, 32, 16, B_sc_tile>;
    using D_gl         = gl<bf16,      1,  1, -1, -1, D_tile>;
    using G_fp4x2_gl   = gl<fp4e2m1_2, 1,  1, -1, -1, G_fp4x2_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;
    D_gl       D_out;        // BF16 output (BF16 mode)

    // FP4 output (FP4 mode)
    G_fp4x2_gl G_fp4_row;    // Quantized G in fp4x2 format
    uint8_t*   G_sc_row;     // E8M0 scales in swizzled layout
    uint8_t*   G_fp4_col_ptr;   // fp4x2 at [col, row/2], size (V, M/2)
    uint8_t*   G_sc_col_ptr;    // E8M0 scales in TK 3D format [V/128, M/128, 512]
    int        G_sc_col_kgroups; // = M/128
    uint8_t*   G_tilemask;      // [M/128, N/128] tile activity mask
    int        G_tilemask_cols; // = N/128

    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A[C::MMA_PER_TILE];
        B_sc_tile B[C::MMA_PER_TILE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        int padded_M = A.rows();
        int padded_N = B.rows();
        int num_row_blocks = padded_M / C::Mb;
        int num_col_blocks = padded_N / C::Nb;
        int total = num_row_blocks * num_col_blocks;
        int max_blocks = num_sms();
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

// =========================================================================
// Main kernel
// =========================================================================
template <typename C>
__device__ inline void backward_kernel_v2(const globals<C>& g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        if constexpr (C::USE_BF16_ACCUM) {
            g.D_out.template prefetch_tma<typename G::D_tile>();
        }
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

    // ======================== PRODUCER ========================
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
                    if (cta_id == 0) {
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
            // ======================== MMA WARP ========================
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

            int phase = 0;

            auto do_mma_block = [&](auto& accum) {
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*16 + k*16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            };

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                if (phase == 0) do_mma_block(out_tm_0);
                else            do_mma_block(out_tm_1);
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
                phase ^= 1;
            }
        }
    // ======================== CONSUMER ========================
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

        constexpr float MXFP4_ALPHA = 1.0f / 36.0f;

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;  // 128/4 = 32
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;       // 32-wide FP32 register tile
        using subtile_rt_bf = rt_bf<C::Mb / 8, SUBTILE_COLS>;

        const int lane_id = threadIdx.x % 32;
        int phase = 0;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

            subtile_rt D_regs_fl[C::EPI_PIPE_DEPTH];
            subtile_rt_bf D_regs_bf[C::EPI_PIPE_DEPTH];

            auto load_from_accum = [&](auto& accum) {
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    warpgroup::load_async(D_regs_fl[epi], accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                }
            };
            if (phase == 0) load_from_accum(out_tm_0);
            else            load_from_accum(out_tm_1);

            int my_targets_x[subtile_rt::height];
            int my_targets_y[subtile_rt::height];
            float my_lse_x[subtile_rt::height];
            float my_lse_y[subtile_rt::height];
            #pragma unroll
            for (int i = 0; i < subtile_rt::height; i++) {
                int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                my_targets_x[i] = (global_row_x < g.M) ? (int)g.targets[global_row_x] : -1;
                my_targets_y[i] = (global_row_y < g.M) ? (int)g.targets[global_row_y] : -1;
                my_lse_x[i] = (global_row_x < g.M) ? g.lse[global_row_x] : INFINITY;
                my_lse_y[i] = (global_row_y < g.M) ? g.lse[global_row_y] : INFINITY;
            }

            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);

            // Compute softmax gradient
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                subtile_rt& D_fl = D_regs_fl[epi];
                warp::mul(D_fl, D_fl, MXFP4_ALPHA);
                int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;

                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    float lse_x = my_lse_x[i];
                    float lse_y = my_lse_y[i];
                    #pragma unroll
                    for (int j = 0; j < subtile_rt::width; j++) {
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            float lse_val = (k % 2 == 0) ? lse_x : lse_y;
                            D_fl.tiles[i][j].data[k].x = __expf(D_fl.tiles[i][j].data[k].x - lse_val);
                            D_fl.tiles[i][j].data[k].y = __expf(D_fl.tiles[i][j].data[k].y - lse_val);
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    int tgt_x = my_targets_x[i];
                    if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
                        int local_col = tgt_x - col_start;
                        int j_idx = local_col / 16;
                        int within_tile = local_col % 16;
                        int k_half = within_tile / 8;
                        int pair_pos = (within_tile % 8) / 2;
                        if ((lane_id % 4) == pair_pos) {
                            int k_idx = k_half * 2;
                            if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                            else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                        }
                    }
                    int tgt_y = my_targets_y[i];
                    if (tgt_y >= col_start && tgt_y < col_start + SUBTILE_COLS) {
                        int local_col = tgt_y - col_start;
                        int j_idx = local_col / 16;
                        int within_tile = local_col % 16;
                        int k_half = within_tile / 8;
                        int pair_pos = (within_tile % 8) / 2;
                        if ((lane_id % 4) == pair_pos) {
                            int k_idx = k_half * 2 + 1;
                            if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                            else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                    int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                    #pragma unroll
                    for (int j = 0; j < subtile_rt::width; j++) {
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            if (k % 2 == 0 && global_row_x >= g.M) {
                                D_fl.tiles[i][j].data[k].x = 0.0f;
                                D_fl.tiles[i][j].data[k].y = 0.0f;
                            }
                            if (k % 2 == 1 && global_row_y >= g.M) {
                                D_fl.tiles[i][j].data[k].x = 0.0f;
                                D_fl.tiles[i][j].data[k].y = 0.0f;
                            }
                        }
                    }
                }
                warp::mul(D_fl, D_fl, g.grad_scale);
                warp::copy(D_regs_bf[epi], D_fl);
            }

            // Free TMEM
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            // CUT filtering
            bool tile_is_filtered = false;
            if (g.filter_eps > 0.0f) {
                float local_max = 0.0f;
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    #pragma unroll
                    for (int i = 0; i < subtile_rt::height; i++) {
                        #pragma unroll
                        for (int j = 0; j < subtile_rt::width; j++) {
                            #pragma unroll
                            for (int k = 0; k < 4; k++) {
                                local_max = fmaxf(local_max, fabsf(D_regs_fl[epi].tiles[i][j].data[k].x));
                                local_max = fmaxf(local_max, fabsf(D_regs_fl[epi].tiles[i][j].data[k].y));
                            }
                        }
                    }
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
                __shared__ float filter_max_smem[WARPGROUP_WARPS];
                if (lane_id == 0) filter_max_smem[warpgroup::warpid()] = local_max;
                warpgroup::sync(1);
                float global_max = 0.0f;
                for (int w = 0; w < WARPGROUP_WARPS; w++)
                    global_max = fmaxf(global_max, filter_max_smem[w]);
                tile_is_filtered = (global_max < g.filter_eps);
            }

            if (g.G_tilemask != nullptr && threadIdx.x == 0) {
                const int row_tile_idx = row_block_idx * 2 + cta_id;
                g.G_tilemask[row_tile_idx * g.G_tilemask_cols + col_block_idx] =
                    static_cast<uint8_t>(!tile_is_filtered);
            }

            // Store output
            if (!tile_is_filtered) {
                if constexpr (C::USE_BF16_ACCUM) {
                    // BF16 path: TMA store G_bf16 (same as v1)
                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_regs_bf[epi]);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            g.D_out, output_tiles.D[epi % C::NUM_D_TILES],
                            {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + epi});
                    }
                } else {
                    // ═══════ FP4 PATH: quantize G to MXFP4 on-the-fly ═══════
                    // MXFP4: per-32-element blocks with E8M0 scales
                    // Each subtile (epi) is 32 cols wide = exactly one MX scale block!
                    //
                    // Register layout per thread per (i,j) tile:
                    //   data[0].x, data[0].y → row_x, col pair 0
                    //   data[1].x, data[1].y → row_y, col pair 0
                    //   data[2].x, data[2].y → row_x, col pair 1 (col+8)
                    //   data[3].x, data[3].y → row_y, col pair 1 (col+8)
                    //
                    // subtile_rt::width = 2 (32 cols / 16 cols per j-tile)
                    // So j=0 covers cols 0..15, j=1 covers cols 16..31
                    // Combined j=0,1 → one 32-element MX block per row
                    //
                    // Step 1: Compute per-row, per-32-col-block amax
                    // Step 2: Convert amax to E8M0 scale
                    // Step 3: Quantize with coeff = 6.0 / scale
                    // Step 4: Pack fp4x2 bytes, write to global

                    const int ntk = g.N / 128;  // number of col-tiles in K dimension
                    const bool do_col = g.G_fp4_col_ptr != nullptr && g.G_sc_col_ptr != nullptr;
                    uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.G_fp4_row.raw_ptr);
                    const int row_fp4_stride = g.G_fp4_row.cols();
                    const int col_fp4_stride = g.A.rows() / 2;

                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                        subtile_rt& D_fl = D_regs_fl[epi];
                        int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;

                        #pragma unroll
                        for (int i = 0; i < subtile_rt::height; i++) {
                            int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                            int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;

                            // Collect all 8 values per row across j=0,1 (32 cols total)
                            float vals_x[8], vals_y[8];
                            #pragma unroll
                            for (int j = 0; j < subtile_rt::width; j++) {
                                vals_x[j*4 + 0] = D_fl.tiles[i][j].data[0].x;
                                vals_x[j*4 + 1] = D_fl.tiles[i][j].data[0].y;
                                vals_x[j*4 + 2] = D_fl.tiles[i][j].data[2].x;
                                vals_x[j*4 + 3] = D_fl.tiles[i][j].data[2].y;
                                vals_y[j*4 + 0] = D_fl.tiles[i][j].data[1].x;
                                vals_y[j*4 + 1] = D_fl.tiles[i][j].data[1].y;
                                vals_y[j*4 + 2] = D_fl.tiles[i][j].data[3].x;
                                vals_y[j*4 + 3] = D_fl.tiles[i][j].data[3].y;
                            }

                            // Per-row, per-32-col-block amax
                            // Each thread has 8 values, reduce across 4 lanes
                            float amax_x = 0.0f, amax_y = 0.0f;
                            #pragma unroll
                            for (int v = 0; v < 8; v++) {
                                amax_x = fmaxf(amax_x, fabsf(vals_x[v]));
                                amax_y = fmaxf(amax_y, fabsf(vals_y[v]));
                            }
                            // Warp shuffle across 4 threads sharing a row
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset <<= 1) {
                                amax_x = fmaxf(amax_x, __shfl_xor_sync(0xFFFFFFFF, amax_x, offset));
                                amax_y = fmaxf(amax_y, __shfl_xor_sync(0xFFFFFFFF, amax_y, offset));
                            }

                            // Convert amax to E8M0 scale (mode-dependent rounding)
                            uint8_t e8m0_x = (amax_x <= 1e-9f) ? 0 : float_to_e8m0_dispatch<C::QUANT_MODE>(amax_x);
                            uint8_t e8m0_y = (amax_y <= 1e-9f) ? 0 : float_to_e8m0_dispatch<C::QUANT_MODE>(amax_y);
                            float rcp_x = exp2f_rcp_e8m0(e8m0_x);
                            float rcp_y = exp2f_rcp_e8m0(e8m0_y);
                            float coeff_x = 6.0f * rcp_x;
                            float coeff_y = 6.0f * rcp_y;

                            // Quantize and pack FP4 pairs
                            #pragma unroll
                            for (int j = 0; j < subtile_rt::width; j++) {
                                // Low cols: data[0].x/y (cols base_col, base_col+1)
                                uint8_t fp4_x_lo = quantize_fp4_pair(
                                    D_fl.tiles[i][j].data[0].x, D_fl.tiles[i][j].data[0].y, coeff_x);
                                uint8_t fp4_x_hi = quantize_fp4_pair(
                                    D_fl.tiles[i][j].data[2].x, D_fl.tiles[i][j].data[2].y, coeff_x);
                                uint8_t fp4_y_lo = quantize_fp4_pair(
                                    D_fl.tiles[i][j].data[1].x, D_fl.tiles[i][j].data[1].y, coeff_y);
                                uint8_t fp4_y_hi = quantize_fp4_pair(
                                    D_fl.tiles[i][j].data[3].x, D_fl.tiles[i][j].data[3].y, coeff_y);

                                // Write FP4 data to global
                                int global_byte_lo = col_start / 2 + (lane_id % 4) + j * 8;
                                int global_byte_hi = global_byte_lo + 4;
                                if (global_row_x < g.M) {
                                    row_fp4_ptr[global_row_x * row_fp4_stride + global_byte_lo] = fp4_x_lo;
                                    row_fp4_ptr[global_row_x * row_fp4_stride + global_byte_hi] = fp4_x_hi;
                                }
                                if (global_row_y < g.M) {
                                    row_fp4_ptr[global_row_y * row_fp4_stride + global_byte_lo] = fp4_y_lo;
                                    row_fp4_ptr[global_row_y * row_fp4_stride + global_byte_hi] = fp4_y_hi;
                                }
                            }

                            // Write E8M0 scale (one per row per 32-col block)
                            // Scale layout: (row_blk * ntk + col_blk) * 512 + j_in_tile * 16 + grp * 4
                            // For this tile: row_blk = global_row/128, col_blk = col_start/128
                            // j_in_tile = row % 32, grp = (row % 128) / 32
                            // Each MX block = 32 cols → one scale per epi subtile
                            if ((lane_id % 4) == 0) {
                                int sc_row_blk_x = global_row_x / 128;
                                int sc_col_blk   = col_start / 128;
                                int epi_block    = epi;  // which 32-col block within the 128-col tile

                                if (global_row_x < g.M) {
                                    int j_in_tile = global_row_x % 32;
                                    int grp       = (global_row_x % 128) / 32;
                                    int base = (sc_row_blk_x * ntk + sc_col_blk) * 512 + j_in_tile * 16 + grp * 4;
                                    g.G_sc_row[base + epi_block] = e8m0_x;
                                }
                                if (global_row_y < g.M) {
                                    int sc_row_blk_y = global_row_y / 128;
                                    int j_in_tile = global_row_y % 32;
                                    int grp       = (global_row_y % 128) / 32;
                                    int base = (sc_row_blk_y * ntk + sc_col_blk) * 512 + j_in_tile * 16 + grp * 4;
                                    g.G_sc_row[base + epi_block] = e8m0_y;
                                }
                            }
                        }
                    }

                    if (do_col) {
                        const int lane_col_pair = (lane_id % 4) * 2;
                        const int row_pair_idx = lane_id / 4;
                        const bool is_scale_lane = row_pair_idx == 0;
                        const int m_kgroup = warp_row_base / 128;
                        const int m_32_in_128 = (warp_row_base / 32) % 4;
                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                            subtile_rt& D_fl = D_regs_fl[epi];
                            int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;

                            #pragma unroll
                            for (int j = 0; j < subtile_rt::width; j++) {
                                int gc0 = col_start + j * 16 + lane_col_pair;
                                int gc1 = gc0 + 1;
                                int gc8 = gc0 + 8;
                                int gc9 = gc8 + 1;

                                float col_amax[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                #pragma unroll
                                for (int i = 0; i < subtile_rt::height; i++) {
                                    col_amax[0] = fmaxf(col_amax[0], fabsf(D_fl.tiles[i][j].data[0].x));
                                    col_amax[1] = fmaxf(col_amax[1], fabsf(D_fl.tiles[i][j].data[0].y));
                                    col_amax[2] = fmaxf(col_amax[2], fabsf(D_fl.tiles[i][j].data[2].x));
                                    col_amax[3] = fmaxf(col_amax[3], fabsf(D_fl.tiles[i][j].data[2].y));
                                    col_amax[0] = fmaxf(col_amax[0], fabsf(D_fl.tiles[i][j].data[1].x));
                                    col_amax[1] = fmaxf(col_amax[1], fabsf(D_fl.tiles[i][j].data[1].y));
                                    col_amax[2] = fmaxf(col_amax[2], fabsf(D_fl.tiles[i][j].data[3].x));
                                    col_amax[3] = fmaxf(col_amax[3], fabsf(D_fl.tiles[i][j].data[3].y));
                                }
                                #pragma unroll
                                for (int v = 0; v < 4; v++) {
                                    col_amax[v] = fmaxf(col_amax[v], __shfl_xor_sync(0xFFFFFFFF, col_amax[v], 4));
                                    col_amax[v] = fmaxf(col_amax[v], __shfl_xor_sync(0xFFFFFFFF, col_amax[v], 8));
                                    col_amax[v] = fmaxf(col_amax[v], __shfl_xor_sync(0xFFFFFFFF, col_amax[v], 16));
                                }

                                uint8_t col_e8m0[4];
                                float col_coeff[4];
                                #pragma unroll
                                for (int v = 0; v < 4; v++) {
                                    col_e8m0[v] = (col_amax[v] <= 1e-9f) ? 0 : float_to_e8m0_dispatch<C::QUANT_MODE>(col_amax[v]);
                                    col_coeff[v] = 6.0f * exp2f_rcp_e8m0(col_e8m0[v]);
                                }

                                #pragma unroll
                                for (int i = 0; i < subtile_rt::height; i++) {
                                    int global_row_pair_lo = warp_row_base + i * 16 + row_pair_idx;
                                    int global_row_pair_hi = global_row_pair_lo + 8;

                                    float vals_x[4] = {
                                        D_fl.tiles[i][j].data[0].x,
                                        D_fl.tiles[i][j].data[0].y,
                                        D_fl.tiles[i][j].data[2].x,
                                        D_fl.tiles[i][j].data[2].y,
                                    };
                                    float vals_y[4] = {
                                        D_fl.tiles[i][j].data[1].x,
                                        D_fl.tiles[i][j].data[1].y,
                                        D_fl.tiles[i][j].data[3].x,
                                        D_fl.tiles[i][j].data[3].y,
                                    };

                                    float row_x_pair_hi[4];
                                    float row_y_pair_hi[4];
                                    #pragma unroll
                                    for (int v = 0; v < 4; v++) {
                                        row_x_pair_hi[v] = __shfl_xor_sync(0xFFFFFFFF, vals_x[v], 4);
                                        row_y_pair_hi[v] = __shfl_xor_sync(0xFFFFFFFF, vals_y[v], 4);
                                    }

                                    if ((row_pair_idx % 2) == 0) {
                                        if (gc0 < g.N && global_row_pair_lo < g.M) {
                                            float pair_hi = (global_row_pair_lo + 1 < g.M) ? row_x_pair_hi[0] : 0.0f;
                                            g.G_fp4_col_ptr[gc0 * col_fp4_stride + global_row_pair_lo / 2] =
                                                quantize_fp4_pair(vals_x[0], pair_hi, col_coeff[0]);
                                        }
                                        if (gc1 < g.N && global_row_pair_lo < g.M) {
                                            float pair_hi = (global_row_pair_lo + 1 < g.M) ? row_x_pair_hi[1] : 0.0f;
                                            g.G_fp4_col_ptr[gc1 * col_fp4_stride + global_row_pair_lo / 2] =
                                                quantize_fp4_pair(vals_x[1], pair_hi, col_coeff[1]);
                                        }
                                        if (gc8 < g.N && global_row_pair_lo < g.M) {
                                            float pair_hi = (global_row_pair_lo + 1 < g.M) ? row_x_pair_hi[2] : 0.0f;
                                            g.G_fp4_col_ptr[gc8 * col_fp4_stride + global_row_pair_lo / 2] =
                                                quantize_fp4_pair(vals_x[2], pair_hi, col_coeff[2]);
                                        }
                                        if (gc9 < g.N && global_row_pair_lo < g.M) {
                                            float pair_hi = (global_row_pair_lo + 1 < g.M) ? row_x_pair_hi[3] : 0.0f;
                                            g.G_fp4_col_ptr[gc9 * col_fp4_stride + global_row_pair_lo / 2] =
                                                quantize_fp4_pair(vals_x[3], pair_hi, col_coeff[3]);
                                        }

                                        if (gc0 < g.N && global_row_pair_hi < g.M) {
                                            float pair_hi = (global_row_pair_hi + 1 < g.M) ? row_y_pair_hi[0] : 0.0f;
                                            g.G_fp4_col_ptr[gc0 * col_fp4_stride + global_row_pair_hi / 2] =
                                                quantize_fp4_pair(vals_y[0], pair_hi, col_coeff[0]);
                                        }
                                        if (gc1 < g.N && global_row_pair_hi < g.M) {
                                            float pair_hi = (global_row_pair_hi + 1 < g.M) ? row_y_pair_hi[1] : 0.0f;
                                            g.G_fp4_col_ptr[gc1 * col_fp4_stride + global_row_pair_hi / 2] =
                                                quantize_fp4_pair(vals_y[1], pair_hi, col_coeff[1]);
                                        }
                                        if (gc8 < g.N && global_row_pair_hi < g.M) {
                                            float pair_hi = (global_row_pair_hi + 1 < g.M) ? row_y_pair_hi[2] : 0.0f;
                                            g.G_fp4_col_ptr[gc8 * col_fp4_stride + global_row_pair_hi / 2] =
                                                quantize_fp4_pair(vals_y[2], pair_hi, col_coeff[2]);
                                        }
                                        if (gc9 < g.N && global_row_pair_hi < g.M) {
                                            float pair_hi = (global_row_pair_hi + 1 < g.M) ? row_y_pair_hi[3] : 0.0f;
                                            g.G_fp4_col_ptr[gc9 * col_fp4_stride + global_row_pair_hi / 2] =
                                                quantize_fp4_pair(vals_y[3], pair_hi, col_coeff[3]);
                                        }
                                    }
                                }

                                if (is_scale_lane) {
                                    if (gc0 < g.N) {
                                        int depth = gc0 / 128;
                                        int sr = gc0 % 32;
                                        int rr = (gc0 / 32) % 4;
                                        int chunk = depth * g.G_sc_col_kgroups + m_kgroup;
                                        int byte_idx = sr * 16 + rr * 4 + m_32_in_128;
                                        g.G_sc_col_ptr[chunk * 512 + byte_idx] = col_e8m0[0];
                                    }
                                    if (gc1 < g.N) {
                                        int depth = gc1 / 128;
                                        int sr = gc1 % 32;
                                        int rr = (gc1 / 32) % 4;
                                        int chunk = depth * g.G_sc_col_kgroups + m_kgroup;
                                        int byte_idx = sr * 16 + rr * 4 + m_32_in_128;
                                        g.G_sc_col_ptr[chunk * 512 + byte_idx] = col_e8m0[1];
                                    }
                                    if (gc8 < g.N) {
                                        int depth = gc8 / 128;
                                        int sr = gc8 % 32;
                                        int rr = (gc8 / 32) % 4;
                                        int chunk = depth * g.G_sc_col_kgroups + m_kgroup;
                                        int byte_idx = sr * 16 + rr * 4 + m_32_in_128;
                                        g.G_sc_col_ptr[chunk * 512 + byte_idx] = col_e8m0[2];
                                    }
                                    if (gc9 < g.N) {
                                        int depth = gc9 / 128;
                                        int sr = gc9 % 32;
                                        int rr = (gc9 / 32) % 4;
                                        int chunk = depth * g.G_sc_col_kgroups + m_kgroup;
                                        int byte_idx = sr * 16 + rr * 4 + m_32_in_128;
                                        g.G_sc_col_ptr[chunk * 512 + byte_idx] = col_e8m0[3];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            update_phasebit<0>(phasebits, 0);
            phase ^= 1;
        }
        warpgroup::sync(1);
        if constexpr (C::USE_BF16_ACCUM) {
            warpgroup::tma::store_async_read_wait<0>();
        }
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace mxfp4_cce_backward_v2
