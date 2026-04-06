#pragma once
// ================================================================
// NVFP4 CCE Backward v3 — Fused Softmax Gradient + FP4 Quantization
//
// Phase 1: FP4 GEMM recomputes logits = E_fp4 @ C_fp4^T
// Phase 2: Consumer computes G = grad_scale * (softmax(logits) - 1[target])
// Phase 3: Consumer stores G to global memory:
//   BF16 mode (USE_BF16_ACCUM=true):  Store G as BF16 via TMA
//   FP4  mode (USE_BF16_ACCUM=false): Quantize G to NVFP4 on-the-fly
//          (row-wise for dE = G @ C, col-wise for dC = G.T @ E)
//          Per-16-element FP8 micro scales plus an analytic tensor scale
//          derived from grad_scale (no tensor-global G amax pass required)
//
// After this kernel, separate GEMM calls compute dE/dC:
//   dE = G_output @ C_quantized   (FP4 or BF16 GEMM)
//   dC = G_output^T @ E_quantized (FP4 or BF16 GEMM)
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v3 {

// =========================================================================
// Config
// =========================================================================
template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _USE_BF16_ACCUM = true, bool _PINGPONG = true>
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
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = 2;
    static constexpr bool USE_BF16_ACCUM = _USE_BF16_ACCUM;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_3wg {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_rowregs {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_aligned {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = true;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_rowregs_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_rowregs_s4 {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 4;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_rowregs_s3 {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 3;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_plainstage {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = true;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_bf16cache {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = true;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_paircache {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = true;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowregs {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowregs_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <typename C>
struct config_traits_3wg {
    static constexpr bool USE_COL_PAIR_STAGE = false;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool COL_QUANT_FROM_DBF_REGS = false;
    static constexpr bool DIRECT_QUANT_WARPBALANCED = false;
    static constexpr bool DIRECT_ROW_WARPBALANCED = false;
    static constexpr bool DIRECT_COL_WARPBALANCED = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_ROWRECORD_REGS = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = false;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_QUANT_NOINLINE_HELPER = false;
    static constexpr bool ROW_QUANT_USE_HW_FP4X2 = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
    static constexpr bool ROW_PAIR_STAGE_PADDED_LAYOUT = false;
    static constexpr bool ROW_WAITS_FOR_COL_RECYCLE = false;
    static constexpr bool ROW_WAITS_FOR_COL_DONE_ROW16 = false;
    static constexpr bool ROW_QUANT_IMMEDIATE_PER_ROW16 = false;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = false;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = false;
    static constexpr bool DEBUG_SKIP_ROW_READY_WAIT = false;
    static constexpr bool DEBUG_SKIP_COL_READY_WAIT = false;
    static constexpr bool DEBUG_SKIP_ROW_RECYCLE_WAIT = false;
    static constexpr bool DEBUG_SKIP_COL_RECYCLE_WAIT = false;
    static constexpr bool DEBUG_ROW_WAIT_ON_COL_READY = false;
    static constexpr bool DEBUG_DISABLE_CONSUMER_ROW_STAGE_PRODUCTION = false;
    static constexpr bool DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION = false;
    static constexpr bool ENABLE_STOREADD_COMBO = false;
};

template <typename C>
struct config_traits_col_ready_per_row16 {
    static constexpr bool value = false;
};

template <typename C, typename = void>
struct config_traits_row_pair_stage_slotmajor {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_pair_stage_slotmajor<C, std::void_t<decltype(config_traits_3wg<C>::ROW_PAIR_STAGE_SLOTMAJOR)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_PAIR_STAGE_SLOTMAJOR;
};

template <typename C, typename = void>
struct config_traits_row_pair_stage_floatcache {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_pair_stage_floatcache<C, std::void_t<decltype(config_traits_3wg<C>::ROW_PAIR_STAGE_FLOATCACHE)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_PAIR_STAGE_FLOATCACHE;
};

template <typename C, typename = void>
struct config_traits_row_pair_stage_padded {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_pair_stage_padded<C, std::void_t<decltype(config_traits_3wg<C>::ROW_PAIR_STAGE_PADDED_LAYOUT)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_PAIR_STAGE_PADDED_LAYOUT;
};

template <typename C, typename = void>
struct config_traits_row_pair_stage_lanepair {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_pair_stage_lanepair<C, std::void_t<decltype(config_traits_3wg<C>::ROW_PAIR_STAGE_LANEPAIR_LAYOUT)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_PAIR_STAGE_LANEPAIR_LAYOUT;
};

template <typename C, typename = void>
struct config_traits_col_pair_stage_padded {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_col_pair_stage_padded<C, std::void_t<decltype(config_traits_3wg<C>::COL_PAIR_STAGE_PADDED_LAYOUT)>> {
    static constexpr bool value = config_traits_3wg<C>::COL_PAIR_STAGE_PADDED_LAYOUT;
};

template <typename C, typename = void>
struct config_traits_row_quant_from_rowrecord_regs {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_quant_from_rowrecord_regs<C, std::void_t<decltype(config_traits_3wg<C>::ROW_QUANT_FROM_ROWRECORD_REGS)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_QUANT_FROM_ROWRECORD_REGS;
};

template <typename C, typename = void>
struct config_traits_row_quant_noinline_helper {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_quant_noinline_helper<C, std::void_t<decltype(config_traits_3wg<C>::ROW_QUANT_NOINLINE_HELPER)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_QUANT_NOINLINE_HELPER;
};

template <typename C, typename = void>
struct config_traits_row_quant_use_hw_fp4x2 {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_quant_use_hw_fp4x2<C, std::void_t<decltype(config_traits_3wg<C>::ROW_QUANT_USE_HW_FP4X2)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_QUANT_USE_HW_FP4X2;
};

template <typename C, typename = void>
struct config_traits_storeadd_combo {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_storeadd_combo<C, std::void_t<decltype(config_traits_3wg<C>::ENABLE_STOREADD_COMBO)>> {
    static constexpr bool value = config_traits_3wg<C>::ENABLE_STOREADD_COMBO;
};

template <typename C, typename = void>
struct config_traits_separate_backhalf_consumers {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_separate_backhalf_consumers<C, std::void_t<decltype(config_traits_3wg<C>::SEPARATE_BACKHALF_CONSUMERS)>> {
    static constexpr bool value = config_traits_3wg<C>::SEPARATE_BACKHALF_CONSUMERS;
};

template <typename C, typename = void>
struct config_traits_combo_de_warpgroups {
    static constexpr int value = 0;
};

template <typename C>
struct config_traits_combo_de_warpgroups<C, std::void_t<decltype(config_traits_3wg<C>::COMBO_DE_WARPGROUPS)>> {
    static constexpr int value = config_traits_3wg<C>::COMBO_DE_WARPGROUPS;
};

template <typename C, typename = void>
struct config_traits_combo_dc_warpgroups {
    static constexpr int value = 0;
};

template <typename C>
struct config_traits_combo_dc_warpgroups<C, std::void_t<decltype(config_traits_3wg<C>::COMBO_DC_WARPGROUPS)>> {
    static constexpr int value = config_traits_3wg<C>::COMBO_DC_WARPGROUPS;
};

template <typename C, typename = void>
struct config_traits_row_waits_for_col_recycle {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_waits_for_col_recycle<C, std::void_t<decltype(config_traits_3wg<C>::ROW_WAITS_FOR_COL_RECYCLE)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_WAITS_FOR_COL_RECYCLE;
};

template <typename C, typename = void>
struct config_traits_debug_disable_row_quant_work {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_disable_row_quant_work<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_DISABLE_ROW_QUANT_WORK)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_DISABLE_ROW_QUANT_WORK;
};

template <typename C, typename = void>
struct config_traits_debug_disable_col_quant_work {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_disable_col_quant_work<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_DISABLE_COL_QUANT_WORK)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_DISABLE_COL_QUANT_WORK;
};

template <typename C, typename = void>
struct config_traits_debug_skip_row_ready_wait {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_skip_row_ready_wait<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_SKIP_ROW_READY_WAIT)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_SKIP_ROW_READY_WAIT;
};

template <typename C, typename = void>
struct config_traits_debug_skip_col_ready_wait {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_skip_col_ready_wait<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_SKIP_COL_READY_WAIT)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_SKIP_COL_READY_WAIT;
};

template <typename C, typename = void>
struct config_traits_debug_skip_row_recycle_wait {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_skip_row_recycle_wait<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_SKIP_ROW_RECYCLE_WAIT)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_SKIP_ROW_RECYCLE_WAIT;
};

template <typename C, typename = void>
struct config_traits_debug_skip_col_recycle_wait {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_skip_col_recycle_wait<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_SKIP_COL_RECYCLE_WAIT)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_SKIP_COL_RECYCLE_WAIT;
};

template <typename C, typename = void>
struct config_traits_debug_row_wait_on_col_ready {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_row_wait_on_col_ready<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_ROW_WAIT_ON_COL_READY)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_ROW_WAIT_ON_COL_READY;
};

template <typename C, typename = void>
struct config_traits_debug_disable_consumer_row_stage_production {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_disable_consumer_row_stage_production<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_DISABLE_CONSUMER_ROW_STAGE_PRODUCTION)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_DISABLE_CONSUMER_ROW_STAGE_PRODUCTION;
};

template <typename C, typename = void>
struct config_traits_debug_disable_consumer_col_stage_production {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_debug_disable_consumer_col_stage_production<C, std::void_t<decltype(config_traits_3wg<C>::DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION)>> {
    static constexpr bool value = config_traits_3wg<C>::DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION;
};

template <typename C, typename = void>
struct config_traits_row_waits_for_col_done_row16 {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_waits_for_col_done_row16<C, std::void_t<decltype(config_traits_3wg<C>::ROW_WAITS_FOR_COL_DONE_ROW16)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_WAITS_FOR_COL_DONE_ROW16;
};

template <typename C, typename = void>
struct config_traits_row_quant_immediate_per_row16 {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_row_quant_immediate_per_row16<C, std::void_t<decltype(config_traits_3wg<C>::ROW_QUANT_IMMEDIATE_PER_ROW16)>> {
    static constexpr bool value = config_traits_3wg<C>::ROW_QUANT_IMMEDIATE_PER_ROW16;
};

template <typename C, typename = void>
struct config_traits_col_quant_from_dbf_regs {
    static constexpr bool value = false;
};

template <typename C>
struct config_traits_col_quant_from_dbf_regs<C, std::void_t<decltype(config_traits_3wg<C>::COL_QUANT_FROM_DBF_REGS)>> {
    static constexpr bool value = config_traits_3wg<C>::COL_QUANT_FROM_DBF_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowslotrecord_rowsync_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = true;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_rcp_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = true;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_floatcache_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_ROWRECORD_REGS = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowrecordregs_dualfloatcache_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowrecordregs_dualfloatcache_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_ROWRECORD_REGS = true;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_consumerotf {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 0;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_consumerotf<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = false;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool COL_QUANT_FROM_DBF_REGS = true;
    static constexpr bool DIRECT_QUANT_WARPBALANCED = false;
    static constexpr bool DIRECT_ROW_WARPBALANCED = false;
    static constexpr bool DIRECT_COL_WARPBALANCED = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_ROWRECORD_REGS = true;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_consumerotf_warpbalanced {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 0;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_consumerotf_warpbalanced<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = false;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool COL_QUANT_FROM_DBF_REGS = true;
    static constexpr bool DIRECT_QUANT_WARPBALANCED = true;
    static constexpr bool DIRECT_ROW_WARPBALANCED = true;
    static constexpr bool DIRECT_COL_WARPBALANCED = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_ROWRECORD_REGS = true;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
    static constexpr bool ROW_PAIR_STAGE_PADDED_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_QUANT_NOINLINE_HELPER = true;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_rowhelper<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool ROW_QUANT_USE_HW_FP4X2 = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>>
    : config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool ENABLE_STOREADD_COMBO = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_5wg {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COMBO_DE_WARPGROUPS = 1;
    static constexpr int COMBO_DC_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS =
        COL_QUANTIZER_WARPGROUPS + COMBO_DE_WARPGROUPS + COMBO_DC_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS =
        CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_5wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>>
    : config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool ENABLE_STOREADD_COMBO = true;
    static constexpr bool SEPARATE_BACKHALF_CONSUMERS = true;
    static constexpr int COMBO_DE_WARPGROUPS = 1;
    static constexpr int COMBO_DC_WARPGROUPS = 1;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_combo_storeadd_5wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = true;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = true;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_epipipe<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitcolready<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_ROW_WAIT_ON_COL_READY = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowstub<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colstub<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowwaitskip<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = true;
    static constexpr bool DEBUG_SKIP_ROW_READY_WAIT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colwaitskip<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = true;
    static constexpr bool DEBUG_SKIP_COL_READY_WAIT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_rowrecycleskip<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_SKIP_ROW_RECYCLE_WAIT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_colrecycleskip<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_SKIP_COL_RECYCLE_WAIT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_noearly<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = true;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = true;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = true;
    static constexpr bool DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_norowprod<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = true;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = true;
    static constexpr bool DEBUG_DISABLE_CONSUMER_ROW_STAGE_PRODUCTION = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready {
    using base = experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = base::ROW_QUANTIZER_WARPGROUPS;
    static constexpr int COL_QUANTIZER_WARPGROUPS = base::COL_QUANTIZER_WARPGROUPS;
    static constexpr int QUANTIZER_WARPGROUPS = base::QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = base::OVERLAP_EPI;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int BF16_STAGE_COUNT = base::BF16_STAGE_COUNT;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
    static constexpr bool CONSUMER_DO_ROW = base::CONSUMER_DO_ROW;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = base::COL_HELPERS_USE_ALL_QUANTIZER_WGS;
    static constexpr bool USE_COL_PLAIN_STAGE = base::USE_COL_PLAIN_STAGE;
    static constexpr bool EARLY_COL_READY = base::EARLY_COL_READY;
    static constexpr bool CACHE_COL_VALUES = base::CACHE_COL_VALUES;
    static constexpr bool CACHE_COL_VALUES_BF16 = base::CACHE_COL_VALUES_BF16;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = base::CACHE_COL_VALUES_BF16_PAIRS;
    static constexpr bool FAST_ALIGNED_QUANT = base::FAST_ALIGNED_QUANT;
    static constexpr bool ROW_QUANT_FROM_REGS = base::ROW_QUANT_FROM_REGS;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_rowwaitcolready<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = true;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = true;
    static constexpr bool DEBUG_ROW_WAIT_ON_COL_READY = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = false;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_dtile<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = false;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowwgdiag_rowpaironly<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool ROW_QUANT_IMMEDIATE_PER_ROW16 = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_rowimmediate_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool ROW_WAITS_FOR_COL_RECYCLE = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_colthenrow_seq<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
    static constexpr bool ROW_WAITS_FOR_COL_DONE_ROW16 = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_coldone_row16_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = true;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpair_rowpair_packed_rowsync_dualfloatcache_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_col_ready_per_row16<experimental_config_colwg_colpair_rowpair_rowrecord_rowsync_row16ready_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool value = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_lanepairrecord_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_lanepairrecord_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = true;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowpair_rowrecord_rcp_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowpair_rowrecord_rcp_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = true;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = true;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = true;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowfromcol_overlap {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = true;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowfromcol_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = true;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowregs<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowregs_overlap<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowleader {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowleader<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = true;
    static constexpr bool ROW_QUANT_ROWDUAL = false;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_colwg_colpair_rowdual {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 0;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG, int _EPI_PIPE_DEPTH>
struct config_traits_3wg<experimental_config_colwg_colpair_rowdual<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _PINGPONG, _EPI_PIPE_DEPTH>> {
    static constexpr bool USE_COL_PAIR_STAGE = true;
    static constexpr bool USE_ROW_PAIR_STAGE = false;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = false;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = false;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = false;
    static constexpr bool USE_ROW_RCP_STAGE = false;
    static constexpr bool PACK_COL_FP4_U64 = true;
    static constexpr bool ROW_QUANT_ROWLEADER = false;
    static constexpr bool ROW_QUANT_ROWDUAL = true;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct experimental_config_col2wg {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0 && (128 % _EPI_PIPE_DEPTH) == 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 0;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 2;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = true;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = true;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true>
struct experimental_config_4wg {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int ROW_QUANTIZER_WARPGROUPS = 1;
    static constexpr int COL_QUANTIZER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = ROW_QUANTIZER_WARPGROUPS + COL_QUANTIZER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
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
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int BF16_STAGE_COUNT = 2;
    static constexpr int NUM_D_TILES = BF16_STAGE_COUNT;
    static constexpr bool USE_BF16_ACCUM = false;
    static constexpr bool CONSUMER_DO_ROW = false;
    static constexpr bool COL_HELPERS_USE_ALL_QUANTIZER_WGS = false;
    static constexpr bool USE_COL_PLAIN_STAGE = false;
    static constexpr bool EARLY_COL_READY = false;
    static constexpr bool CACHE_COL_VALUES = false;
    static constexpr bool CACHE_COL_VALUES_BF16 = false;
    static constexpr bool CACHE_COL_VALUES_BF16_PAIRS = false;
    static constexpr bool FAST_ALIGNED_QUANT = false;
    static constexpr bool ROW_QUANT_FROM_REGS = false;
};

// =========================================================================
// FP4 quantization helpers
// =========================================================================

// FP4 E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
// Quantize float to nearest FP4 E2M1 (4-bit nibble: 1 sign + 2 exp + 1 mantissa)
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

// Pack pair of values into fp4x2 byte: lo nibble = first, hi nibble = second
__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float rcp_scale) {
    uint8_t q0 = float_to_fp4(v0 * rcp_scale);
    uint8_t q1 = float_to_fp4(v1 * rcp_scale);
    return q0 | (q1 << 4);
}

__device__ __forceinline__ bf16 bf16_from_bits(uint16_t bits) {
    bf16 value;
    *reinterpret_cast<uint16_t*>(&value) = bits;
    return value;
}

__device__ __forceinline__ uint16_t bf16_bits(bf16 value) {
    return *reinterpret_cast<uint16_t*>(&value);
}

__device__ __forceinline__ bf16_2 bf16x2_from_bits(uint32_t bits) {
    bf16_2 value;
    *reinterpret_cast<uint32_t*>(&value) = bits;
    return value;
}

__device__ __forceinline__ uint32_t bf16x2_bits(bf16_2 value) {
    return *reinterpret_cast<uint32_t*>(&value);
}

__device__ __forceinline__ void store_bf16x2_bits(bf16_2 &dst, uint32_t bits) {
    *reinterpret_cast<uint32_t*>(&dst) = bits;
}

__device__ __forceinline__ void store_bf16x2_pair_bits(bf16_2 *dst, uint32_t bits0, uint32_t bits1) {
    *reinterpret_cast<uint64_t*>(dst) = static_cast<uint64_t>(bits0) | (static_cast<uint64_t>(bits1) << 32);
}

template <typename G>
__device__ __forceinline__ void store_col_pair_stage_pair(
    typename G::col_pair_stage_t *stages,
    int bf_stage,
    int row16_block,
    int col_pair_idx,
    int pair_slot,
    uint32_t bits0,
    uint32_t bits1)
{
    if constexpr (G::DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION) {
        return;
    }
    if constexpr (G::COL_PAIR_STAGE_PADDED_LAYOUT) {
        store_bf16x2_pair_bits(
            &stages[bf_stage].pairs_padded[row16_block][col_pair_idx][pair_slot][0],
            bits0,
            bits1);
    } else {
        store_bf16x2_pair_bits(
            &stages[bf_stage].pairs[row16_block][col_pair_idx][pair_slot][0],
            bits0,
            bits1);
    }
}

__device__ __forceinline__ void store_u64x2(uint64_t *dst, uint64_t bits0, uint64_t bits1);

template <typename G>
__device__ __forceinline__ void store_row_pair_stage_record(
    typename G::row_pair_stage_t *stages,
    int bf_stage,
    int row16_block,
    int group16,
    int pair_slot,
    int lane_pair,
    uint64_t row_bits_lo,
    uint64_t row_bits_hi)
{
    if constexpr (G::DEBUG_DISABLE_CONSUMER_ROW_STAGE_PRODUCTION) {
        return;
    }
    if constexpr (G::ROW_PAIR_STAGE_ROWRECORD) {
        if constexpr (G::ROW_PAIR_STAGE_SLOTMAJOR) {
            stages[bf_stage].row_records_slotmajor[row16_block][pair_slot][group16][0][lane_pair] = row_bits_lo;
            stages[bf_stage].row_records_slotmajor[row16_block][pair_slot][group16][1][lane_pair] = row_bits_hi;
        } else if constexpr (G::ROW_PAIR_STAGE_PADDED_LAYOUT) {
            stages[bf_stage].row_records_padded[row16_block][group16][pair_slot][0][lane_pair] = row_bits_lo;
            stages[bf_stage].row_records_padded[row16_block][group16][pair_slot][1][lane_pair] = row_bits_hi;
        } else if constexpr (G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT) {
            store_u64x2(
                &stages[bf_stage].lanepair_records[row16_block][group16][pair_slot][lane_pair][0],
                row_bits_lo, row_bits_hi);
        } else {
            stages[bf_stage].row_records[row16_block][group16][pair_slot][0][lane_pair] = row_bits_lo;
            stages[bf_stage].row_records[row16_block][group16][pair_slot][1][lane_pair] = row_bits_hi;
        }
    } else {
        store_u64x2(
            &stages[bf_stage].packed[row16_block][group16][pair_slot][lane_pair][0],
            row_bits_lo, row_bits_hi);
    }
}

template <typename G>
__device__ __forceinline__ bf16_2 load_col_pair_stage_pair(
    const typename G::col_pair_stage_t *stages,
    int bf_stage,
    int row16_block,
    int col_pair_idx,
    int pair_slot,
    int lane)
{
    if constexpr (G::COL_PAIR_STAGE_PADDED_LAYOUT) {
        return stages[bf_stage].pairs_padded[row16_block][col_pair_idx][pair_slot][lane];
    } else {
        return stages[bf_stage].pairs[row16_block][col_pair_idx][pair_slot][lane];
    }
}

__device__ __forceinline__ void store_u64x2(uint64_t *dst, uint64_t bits0, uint64_t bits1) {
    *reinterpret_cast<ulonglong2*>(dst) =
        make_ulonglong2(static_cast<unsigned long long>(bits0),
                        static_cast<unsigned long long>(bits1));
}

__device__ __forceinline__ void store_global_u64(uint8_t* dst, uint64_t value) {
    *reinterpret_cast<uint64_t*>(dst) = value;
}

template <typename G>
__device__ __forceinline__ void store_combo_row_stage_u64(
    typename G::combo_row_stage_t *combo_row_stage,
    int local_row,
    int local_fp4x2_col_base,
    uint64_t packed_fp4)
{
    if constexpr (!G::ENABLE_STOREADD_COMBO) {
        (void)combo_row_stage;
        (void)local_row;
        (void)local_fp4x2_col_base;
        (void)packed_fp4;
        return;
    } else {
        uint8_t *fp4_bytes = reinterpret_cast<uint8_t*>(&combo_row_stage->G_row.data[0]);
        store_global_u64(&fp4_bytes[local_row * G::G_fp4_row_tile::cols + local_fp4x2_col_base], packed_fp4);
    }
}

template <typename G>
__device__ __forceinline__ void store_combo_row_stage_scale(
    typename G::combo_row_stage_t *combo_row_stage,
    int local_row,
    int local_col_16,
    uint8_t scale_byte)
{
    if constexpr (!G::ENABLE_STOREADD_COMBO) {
        (void)combo_row_stage;
        (void)local_row;
        (void)local_col_16;
        (void)scale_byte;
        return;
    } else {
        uint8_t *sc_bytes = reinterpret_cast<uint8_t*>(&combo_row_stage->G_row_sc.data[0]);
        const int sr = local_row % 32;
        const int rr = (local_row / 32) % 4;
        const int kgroup = local_col_16 / 64;
        const int col_16_in_64 = (local_col_16 / 16) % 4;
        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
        sc_bytes[kgroup * 512 + byte_idx] = scale_byte;
    }
}

template <typename G>
__device__ __forceinline__ void store_combo_col_stage_u64(
    typename G::combo_col_stage_t *combo_col_stage,
    int local_col,
    int local_row_pair_base,
    uint64_t packed_fp4)
{
    if constexpr (!G::ENABLE_STOREADD_COMBO) {
        (void)combo_col_stage;
        (void)local_col;
        (void)local_row_pair_base;
        (void)packed_fp4;
        return;
    } else {
        uint8_t *fp4_bytes = reinterpret_cast<uint8_t*>(&combo_col_stage->Gt_row.data[0]);
        store_global_u64(&fp4_bytes[local_col * G::combo_col_fp4_tile::cols + local_row_pair_base], packed_fp4);
    }
}

template <typename G>
__device__ __forceinline__ void store_combo_col_stage_scale(
    typename G::combo_col_stage_t *combo_col_stage,
    int local_col,
    int local_row_base,
    uint8_t scale_byte)
{
    if constexpr (!G::ENABLE_STOREADD_COMBO) {
        (void)combo_col_stage;
        (void)local_col;
        (void)local_row_base;
        (void)scale_byte;
        return;
    } else {
        uint8_t *sc_bytes = reinterpret_cast<uint8_t*>(&combo_col_stage->Gt_row_sc.data[0]);
        const int sr = local_col % 32;
        const int rr = (local_col / 32) % 4;
        const int m_kgroup = local_row_base / 64;
        const int m_16_in_64 = (local_row_base / 16) % 4;
        const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
        sc_bytes[m_kgroup * 512 + byte_idx] = scale_byte;
    }
}

__device__ __forceinline__ uint64_t shfl_u64_sync(unsigned mask, uint64_t value, int src_lane, int width = warpSize) {
    const uint32_t lo = __shfl_sync(mask, static_cast<uint32_t>(value), src_lane, width);
    const uint32_t hi = __shfl_sync(mask, static_cast<uint32_t>(value >> 32), src_lane, width);
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

template <bool USE_HW_FP4X2 = false>
__device__ __forceinline__ uint8_t quantize_rowrecord_cached_pair(float2 pair, float rcp_scale) {
    if constexpr (USE_HW_FP4X2) {
        const float2 scaled = make_float2(pair.x * rcp_scale, pair.y * rcp_scale);
        return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
    } else {
        return quantize_fp4_pair(pair.x, pair.y, rcp_scale);
    }
}

template <bool USE_HW_FP4X2 = false>
__device__ __forceinline__ void quantize_rowrecord_pairbits_to_global(
    uint64_t pair_bits_0,
    uint64_t pair_bits_1,
    uint64_t pair_bits_2,
    uint64_t pair_bits_3,
    uint8_t* row_fp4_ptr,
    int row_fp4_stride,
    uint8_t* row_sc_ptr,
    int row_sc_kgroups,
    int global_row,
    int col_start,
    int group16,
    float g_sg_rcp,
    float g_sg,
    bool encode_centric,
    uint8_t* combo_row_fp4_ptr = nullptr,
    int combo_row_fp4_stride = 0,
    uint8_t* combo_row_sc_ptr = nullptr,
    int combo_local_row = -1,
    int combo_local_col_start = 0) {
    constexpr float k_fp4_max = 6.0f;
    constexpr float k_e4m3_max = 448.0f;
    float2 cached_pairs[8];
    const uint64_t pair_bits_arr[4] = {pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3};
    float amax = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const uint64_t pair_bits = pair_bits_arr[pair];
        const bf16_2 pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
        const bf16_2 pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
        const float2 pair_lo = __bfloat1622float2(pair_lo_bf);
        const float2 pair_hi = __bfloat1622float2(pair_hi_bf);
        cached_pairs[pair] = pair_lo;
        cached_pairs[pair + 4] = pair_hi;
        amax = fmaxf(amax, fabsf(pair_lo.x));
        amax = fmaxf(amax, fabsf(pair_lo.y));
        amax = fmaxf(amax, fabsf(pair_hi.x));
        amax = fmaxf(amax, fabsf(pair_hi.y));
    }

    const float rcp_scale = (amax > 0.0f) ? (k_fp4_max / amax) : 0.0f;
    const int global_col_16 = col_start + group16 * 16;
    const int fp4x2_col_base = global_col_16 / 2;
    uint64_t packed_fp4 = 0;
    #pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        packed_fp4 |= static_cast<uint64_t>(
            quantize_rowrecord_cached_pair<USE_HW_FP4X2>(cached_pairs[pair], rcp_scale)) << (pair * 8);
    }
    store_global_u64(
        &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
        packed_fp4);
    if (combo_row_fp4_ptr != nullptr && combo_local_row >= 0) {
        const int local_col_16 = combo_local_col_start + group16 * 16;
        const int local_fp4x2_col_base = local_col_16 / 2;
        store_global_u64(
            &combo_row_fp4_ptr[combo_local_row * combo_row_fp4_stride + local_fp4x2_col_base],
            packed_fp4);
    }

    const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
    float stored_scale = scale * g_sg_rcp;
    if (encode_centric) {
        stored_scale = fminf(rcp_scale * g_sg, k_e4m3_max);
    }
    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
    const int kgroup = global_col_16 / 64;
    const int col_16_in_64 = (global_col_16 / 16) % 4;
    const int depth = global_row / 128;
    const int sr = global_row % 32;
    const int rr = (global_row / 32) % 4;
    const int chunk = depth * row_sc_kgroups + kgroup;
    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
    const uint8_t scale_byte = *reinterpret_cast<const uint8_t*>(&sc);
    row_sc_ptr[chunk * 512 + byte_idx] = scale_byte;
    if (combo_row_sc_ptr != nullptr && combo_local_row >= 0) {
        const int local_col_16 = combo_local_col_start + group16 * 16;
        const int local_kgroup = local_col_16 / 64;
        const int local_col_16_in_64 = (local_col_16 / 16) % 4;
        const int local_sr = combo_local_row % 32;
        const int local_rr = (combo_local_row / 32) % 4;
        const int local_byte_idx = local_sr * 16 + local_rr * 4 + local_col_16_in_64;
        combo_row_sc_ptr[local_kgroup * 512 + local_byte_idx] = scale_byte;
    }
}

template <bool USE_HW_FP4X2 = false>
__device__ __noinline__ void quantize_rowrecord_pairbits_to_global_noinline(
    uint64_t pair_bits_0,
    uint64_t pair_bits_1,
    uint64_t pair_bits_2,
    uint64_t pair_bits_3,
    uint8_t* row_fp4_ptr,
    int row_fp4_stride,
    uint8_t* row_sc_ptr,
    int row_sc_kgroups,
    int global_row,
    int col_start,
    int group16,
    float g_sg_rcp,
    float g_sg,
    bool encode_centric,
    uint8_t* combo_row_fp4_ptr = nullptr,
    int combo_row_fp4_stride = 0,
    uint8_t* combo_row_sc_ptr = nullptr,
    int combo_local_row = -1,
    int combo_local_col_start = 0) {
    quantize_rowrecord_pairbits_to_global<USE_HW_FP4X2>(
        pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3,
        row_fp4_ptr, row_fp4_stride,
        row_sc_ptr, row_sc_kgroups,
        global_row, col_start, group16,
        g_sg_rcp, g_sg, encode_centric,
        combo_row_fp4_ptr, combo_row_fp4_stride,
        combo_row_sc_ptr, combo_local_row, combo_local_col_start);
}

// =========================================================================
// Globals
// =========================================================================
template <typename C>
struct globals {
    // FP4 tiles for logit recomputation (A=E_fp4, B=C_fp4)
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    // BF16 output tile (for BF16 mode — same as v1)
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;
    using D_helper_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH, false>;

    // FP4 output tiles (for FP4 mode — row-quantized G)
    // Row-quantized G: each tile is (Mb/2, Nb/2) in fp4x2 layout
    using G_fp4_row_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;
    // Scale tile for row-quantized: per-16-element blocks
    // For row-quant of (128, 128): 128 rows, 128/16 = 8 scale blocks per row
    // = 128 * 8 = 1024 FP8 scales = st_fp8e4m3<128, 8>
    // But TK NVFP4 scale layout is [rows/128, cols/64, 512] fp8
    using G_sc_row_tile  = st_hf<4, 256, false>; // Same layout as input scales

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    // FP4 G output globals
    using G_fp4_row_gl   = gl<fp4e2m1_2,  1,  1, -1, -1, G_fp4_row_tile>;
    using G_sc_row_gl    = gl<half,       1, -1, -1, 256, G_sc_row_tile>;
    using G_sg_row_gl    = gl<float,      1,  1,  1,  1>;

    // FP4 inputs
    A_fp4x2_gl     A;         // E_fp4 (M_padded, K)
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;         // C_fp4 (V_padded, K)
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;

    // BF16 mode output
    D_gl           D_out;     // BF16 grad_logits (M, V)

    // FP4 mode outputs (row-quantized G for dE = G @ C)
    G_fp4_row_gl   G_fp4_row;
    // Scale in TK 3D format: [M/128, V/64, 512] fp8
    // byte = (row%32)*16 + (row/32%4)*4 + (col/16%4)
    // depth=row/128, k_group=col/64
    uint8_t*       G_sc_row_ptr;    // TK 3D scale tensor, cast to uint8_t
    int            G_sc_row_kgroups; // = V/64
    // Tensor-wide decode scale for G. We derive this analytically from
    // grad_scale so the FP8 micro-scales stay representable without a
    // separate global-amax reduction over G.
    G_sg_row_gl    G_sg_row;

    // FP4 mode outputs (col-quantized G^T for dC = E^T @ G = (G^T)^T @ E)
    // Stored transposed as (V, M) so it's row-quantized G^T — usable as B operand
    uint8_t*       G_fp4_col_ptr;   // fp4x2 at [col, row/2], size (V, M/2)
    // Col-quant scale in TK 3D format: [V/128, M/64, 512] fp8
    uint8_t*       G_sc_col_ptr;
    int            G_sc_col_kgroups; // = M/64

    // Backward inputs
    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;    // V (vocab)
    bool encode_centric;  // false=decode: store E4M3(scale), true=encode: store E4M3(1/scale)

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
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

template <typename C>
struct globals_2ctaSdupB {
    using base = globals<C>;

    using A_fp4x2_tile = typename base::A_fp4x2_tile;
    using A_sc_tile    = typename base::A_sc_tile;
    using B_fp4x2_tile = typename base::B_fp4x2_tile;
    using B_sc_tile    = typename base::B_sc_tile;
    using D_tile       = st_bf<C::Mb/2, C::Nb/(2 * C::EPI_PIPE_DEPTH)>;
    using G_fp4_row_tile = typename base::G_fp4_row_tile;
    using G_sc_row_tile  = typename base::G_sc_row_tile;

    using A_fp4x2_gl     = typename base::A_fp4x2_gl;
    using A_sc_gl        = typename base::A_sc_gl;
    using A_sc_global_gl = typename base::A_sc_global_gl;
    using B_fp4x2_gl     = typename base::B_fp4x2_gl;
    using B_sc_gl        = typename base::B_sc_gl;
    using B_sc_global_gl = typename base::B_sc_global_gl;
    using D_gl           = typename base::D_gl;
    using G_fp4_row_gl   = typename base::G_fp4_row_gl;
    using G_sc_row_gl    = typename base::G_sc_row_gl;
    using G_sg_row_gl    = typename base::G_sg_row_gl;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;
    D_gl           D_out;

    G_fp4_row_gl   G_fp4_row;
    uint8_t*       G_sc_row_ptr;
    int            G_sc_row_kgroups;
    G_sg_row_gl    G_sg_row;

    uint8_t*       G_fp4_col_ptr;
    uint8_t*       G_sc_col_ptr;
    int            G_sc_col_kgroups;

    const float*   lse;
    const int64_t* targets;
    float          grad_scale;
    float          filter_eps;
    int            M;
    int            N;
    bool           encode_centric;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B[2];
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[2][C::B_SC_SIZE];
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

template <typename C>
struct globals_3wg {
    static constexpr int COL_HELPER_SLOTS = C::COL_HELPERS_USE_ALL_QUANTIZER_WGS ?
        C::QUANTIZER_WARPGROUPS :
        ((C::COL_QUANTIZER_WARPGROUPS > 0) ? C::COL_QUANTIZER_WARPGROUPS : 1);
    static constexpr bool USE_COL_HELPER_MAILBOX = (C::COL_QUANTIZER_WARPGROUPS > 1);
    static constexpr bool USE_COL_PLAIN_STAGE = C::USE_COL_PLAIN_STAGE;
    static constexpr bool USE_COL_PAIR_STAGE = config_traits_3wg<C>::USE_COL_PAIR_STAGE;
    static constexpr bool COL_PAIR_STAGE_PADDED_LAYOUT = config_traits_col_pair_stage_padded<C>::value;
    static constexpr bool COL_QUANT_FROM_DBF_REGS = config_traits_col_quant_from_dbf_regs<C>::value;
    static constexpr bool DIRECT_QUANT_WARPBALANCED = config_traits_3wg<C>::DIRECT_QUANT_WARPBALANCED;
    static constexpr bool DIRECT_ROW_WARPBALANCED = config_traits_3wg<C>::DIRECT_ROW_WARPBALANCED;
    static constexpr bool DIRECT_COL_WARPBALANCED = config_traits_3wg<C>::DIRECT_COL_WARPBALANCED;
    static constexpr bool USE_ROW_PAIR_STAGE = config_traits_3wg<C>::USE_ROW_PAIR_STAGE;
    static constexpr bool ROW_QUANT_FROM_COL_PAIR_STAGE = config_traits_3wg<C>::ROW_QUANT_FROM_COL_PAIR_STAGE;
    static constexpr bool ROW_QUANT_FROM_ROWRECORD_REGS = config_traits_row_quant_from_rowrecord_regs<C>::value;
    static constexpr bool ROW_PAIR_STAGE_ROWRECORD = config_traits_3wg<C>::ROW_PAIR_STAGE_ROWRECORD;
    static constexpr bool ROW_PAIR_STAGE_WARP_SYNC_ONLY = config_traits_3wg<C>::ROW_PAIR_STAGE_WARP_SYNC_ONLY;
    static constexpr bool ROW_PAIR_STAGE_SLOTMAJOR = config_traits_row_pair_stage_slotmajor<C>::value;
    static constexpr bool ROW_PAIR_STAGE_LANEPAIR_LAYOUT = config_traits_row_pair_stage_lanepair<C>::value;
    static constexpr bool ROW_PAIR_STAGE_PADDED_LAYOUT = config_traits_row_pair_stage_padded<C>::value;
    static constexpr bool COL_READY_PER_ROW16 = config_traits_col_ready_per_row16<C>::value;
    static constexpr bool USE_ROW_RCP_STAGE = config_traits_3wg<C>::USE_ROW_RCP_STAGE;
    static constexpr bool ROW_PAIR_STAGE_FLOATCACHE = config_traits_row_pair_stage_floatcache<C>::value;
    static constexpr bool PACK_COL_FP4_U64 = config_traits_3wg<C>::PACK_COL_FP4_U64;
    static constexpr bool ROW_QUANT_ROWLEADER = config_traits_3wg<C>::ROW_QUANT_ROWLEADER;
    static constexpr bool ROW_QUANT_ROWDUAL = config_traits_3wg<C>::ROW_QUANT_ROWDUAL;
    static constexpr bool ROW_QUANT_NOINLINE_HELPER = config_traits_row_quant_noinline_helper<C>::value;
    static constexpr bool ROW_QUANT_USE_HW_FP4X2 = config_traits_row_quant_use_hw_fp4x2<C>::value;
    static constexpr bool ROW_WAITS_FOR_COL_RECYCLE = config_traits_row_waits_for_col_recycle<C>::value;
    static constexpr bool ROW_WAITS_FOR_COL_DONE_ROW16 = config_traits_row_waits_for_col_done_row16<C>::value;
    static constexpr bool ROW_QUANT_IMMEDIATE_PER_ROW16 = config_traits_row_quant_immediate_per_row16<C>::value;
    static constexpr bool DEBUG_DISABLE_ROW_QUANT_WORK = config_traits_debug_disable_row_quant_work<C>::value;
    static constexpr bool DEBUG_DISABLE_COL_QUANT_WORK = config_traits_debug_disable_col_quant_work<C>::value;
    static constexpr bool DEBUG_SKIP_ROW_READY_WAIT = config_traits_debug_skip_row_ready_wait<C>::value;
    static constexpr bool DEBUG_SKIP_COL_READY_WAIT = config_traits_debug_skip_col_ready_wait<C>::value;
    static constexpr bool DEBUG_SKIP_ROW_RECYCLE_WAIT = config_traits_debug_skip_row_recycle_wait<C>::value;
    static constexpr bool DEBUG_SKIP_COL_RECYCLE_WAIT = config_traits_debug_skip_col_recycle_wait<C>::value;
    static constexpr bool DEBUG_ROW_WAIT_ON_COL_READY = config_traits_debug_row_wait_on_col_ready<C>::value;
    static constexpr bool DEBUG_DISABLE_CONSUMER_ROW_STAGE_PRODUCTION = config_traits_debug_disable_consumer_row_stage_production<C>::value;
    static constexpr bool DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION = config_traits_debug_disable_consumer_col_stage_production<C>::value;
    static constexpr bool ENABLE_STOREADD_COMBO = config_traits_storeadd_combo<C>::value;
    static constexpr bool SEPARATE_BACKHALF_CONSUMERS = config_traits_separate_backhalf_consumers<C>::value;
    static constexpr int COMBO_DE_WARPGROUPS = config_traits_combo_de_warpgroups<C>::value;
    static constexpr int COMBO_DC_WARPGROUPS = config_traits_combo_dc_warpgroups<C>::value;
    static constexpr int FRONTEND_QUANTIZER_WARPGROUPS =
        SEPARATE_BACKHALF_CONSUMERS
            ? (C::ROW_QUANTIZER_WARPGROUPS + C::COL_QUANTIZER_WARPGROUPS)
            : C::QUANTIZER_WARPGROUPS;
    static constexpr bool FRONTEND_SINGLE_QUANTIZER_WG = (FRONTEND_QUANTIZER_WARPGROUPS == 1);
    static constexpr bool FRONTEND_MULTI_QUANTIZER_WG = (FRONTEND_QUANTIZER_WARPGROUPS > 1);
    static constexpr int COMBO_MODE_FULL = 0;
    static constexpr int COMBO_MODE_GONLY = 1;
    static constexpr int COMBO_MODE_DEONLY = 2;
    static constexpr int COMBO_MODE_DCONLY = 3;
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;
    using D_helper_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH, false>;
    using G_fp4_row_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;
    using G_sc_row_tile  = st_hf<4, 256, false>;
    using combo_row_sc_tile = st_hf<C::Nb / 64, 256, false>;
    using combo_col_fp4_tile = st_fp4e2m1_2<C::Nb, C::Mb/2>;
    using combo_col_sc_tile = st_hf<4, 256, false>;
    using combo_p3_C_tile = st_fp4e2m1_2<C::Nb/2, C::Nb/2>;
    using combo_p3_C_sc_tile = st_hf<C::Nb / 128, 256, false>;
    using combo_p3_E_tile = st_fp4e2m1_2<C::Nb/2, C::Mb/2>;
    using combo_p3_E_sc_tile = st_hf<4, 256, false>;
    using combo_dE_tile = st_bf<C::Mb/2, C::Nb / C::EPI_PIPE_DEPTH>;
    using combo_dC_tile = st_bf<C::Nb, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using G_fp4_row_gl   = gl<fp4e2m1_2,  1,  1, -1, -1, G_fp4_row_tile>;
    using G_sc_row_gl    = gl<half,       1, -1, -1, 256, G_sc_row_tile>;
    using G_sg_row_gl    = gl<float,      1,  1,  1,  1>;
    using combo_p3_C_gl = gl<fp4e2m1_2, 1, 1, -1, -1, combo_p3_C_tile>;
    using combo_p3_C_sc_gl = gl<half, 1, -1, -1, 256, combo_p3_C_sc_tile>;
    using combo_p3_C_sc_global_gl = gl<float, 1, 1, 1, 1>;
    using combo_p3_E_gl = gl<fp4e2m1_2, 1, 1, -1, -1, combo_p3_E_tile>;
    using combo_p3_E_sc_gl = gl<half, 1, -1, -1, 256, combo_p3_E_sc_tile>;
    using combo_p3_E_sc_global_gl = gl<float, 1, 1, 1, 1>;
    using combo_dE_gl = gl<bf16, 1, 1, -1, -1, combo_dE_tile>;
    using combo_dC_gl = gl<bf16, 1, 1, -1, -1, combo_dC_tile>;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;
    G_fp4_row_gl   G_fp4_row;
    uint8_t*       G_sc_row_ptr;
    int            G_sc_row_kgroups;
    G_sg_row_gl    G_sg_row;
    uint8_t*       G_fp4_col_ptr;
    uint8_t*       G_sc_col_ptr;
    int            G_sc_col_kgroups;
    const float*   lse;
    const int64_t* targets;
    float          grad_scale;
    float          filter_eps;
    int            M;
    int            N;
    bool           encode_centric;
    combo_p3_C_gl  C_col;
    combo_p3_C_sc_gl C_col_sc;
    combo_p3_C_sc_global_gl C_col_sc_global;
    combo_p3_E_gl  E_col;
    combo_p3_E_sc_gl E_col_sc;
    combo_p3_E_sc_global_gl E_col_sc_global;
    combo_dE_gl    dE_out;
    combo_dC_gl    dC_out;
    int            combo_mode;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct bf16_stage_t {
        D_tile D;
    };
    struct col_plain_stage_t {
        D_helper_tile D;
    };
    struct alignas(16) col_pair_stage_t {
        union {
            bf16_2 pairs[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/2][8][2];
            bf16_2 pairs_padded[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/2][8][4];
        };
    };
    struct alignas(16) row_pair_stage_t {
        union {
            uint64_t packed[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/16][8][4][2];
            uint64_t row_records[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/16][8][2][4];
            uint64_t row_records_padded[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/16][8][2][5];
            uint64_t row_records_slotmajor[C::Mb/32][8][C::Nb/C::EPI_PIPE_DEPTH/16][2][4];
            uint64_t lanepair_records[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/16][8][4][2];
        };
    };
    struct row_rcp_stage_t {
        float rcp[C::Mb/32][C::Nb/C::EPI_PIPE_DEPTH/16][8][2];
    };
    struct col_mailbox_stage_t {
        D_helper_tile D[COL_HELPER_SLOTS];
    };
    struct combo_row_stage_t {
        G_fp4_row_tile G_row;
        combo_row_sc_tile G_row_sc;
    };
    struct combo_col_stage_t {
        combo_col_fp4_tile Gt_row;
        combo_col_sc_tile Gt_row_sc;
    };
    struct combo_p3_tiles_t {
        combo_p3_C_tile C_operand;
        combo_p3_E_tile E_operand;
    };
    struct combo_p3_scales_t {
        combo_p3_C_sc_tile C_sc;
        combo_p3_E_sc_tile E_sc;
    };
    struct combo_p3_c_tile_stage_t {
        combo_p3_C_tile C_operand;
    };
    struct combo_de_scales_stage_t {
        combo_p3_C_sc_tile C_sc;
    };
    struct combo_de_output_stage_t {
        combo_dE_tile dE;
    };
    struct combo_dc_tile_stage_t {
        combo_p3_E_tile E_operand;
    };
    struct combo_dc_scales_stage_t {
        combo_p3_E_sc_tile E_sc;
    };
    struct combo_dc_output_stage_t {
        combo_dC_tile dC;
    };
    struct combo_output_stage_t {
        union {
            combo_dE_tile dE;
            combo_dC_tile dC;
            combo_p3_E_tile E_operand;
        };
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
        static_assert(!ENABLE_STOREADD_COMBO || (sizeof(combo_row_stage_t) % 1024 == 0));
        static_assert(!ENABLE_STOREADD_COMBO || (sizeof(combo_col_stage_t) % 1024 == 0));
        static_assert(!ENABLE_STOREADD_COMBO || (sizeof(combo_p3_c_tile_stage_t) % 1024 == 0));
        static_assert(!ENABLE_STOREADD_COMBO || (sizeof(combo_output_stage_t) % 1024 == 0));
        constexpr int combo_dynamic_shared_memory =
            !ENABLE_STOREADD_COMBO ? 0 :
            (SEPARATE_BACKHALF_CONSUMERS
                 ? (sizeof(combo_row_stage_t) +
                    sizeof(combo_col_stage_t) +
                    sizeof(combo_p3_c_tile_stage_t) +
                    sizeof(combo_de_scales_stage_t) +
                    sizeof(combo_output_stage_t) + 1024)
                 : (sizeof(combo_row_stage_t) +
                    sizeof(combo_col_stage_t) +
                    sizeof(combo_p3_c_tile_stage_t) +
                    sizeof(combo_output_stage_t) + 1024));
        constexpr int _dynamic_shared_memory =
            sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
            sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
            sizeof(bf16_stage_t) * C::BF16_STAGE_COUNT + 1024 +
            (USE_COL_PLAIN_STAGE ? (sizeof(col_plain_stage_t) * C::BF16_STAGE_COUNT + 1024) : 0) +
            (USE_COL_PAIR_STAGE ? (sizeof(col_pair_stage_t) * C::BF16_STAGE_COUNT + 1024) : 0) +
            (USE_ROW_PAIR_STAGE ? (sizeof(row_pair_stage_t) * C::BF16_STAGE_COUNT + 1024) : 0) +
            (USE_ROW_RCP_STAGE ? (sizeof(row_rcp_stage_t) * C::BF16_STAGE_COUNT + 1024) : 0) +
            (USE_COL_HELPER_MAILBOX ? (sizeof(col_mailbox_stage_t) * C::BF16_STAGE_COUNT + 1024) : 0) +
            combo_dynamic_shared_memory;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
using globals_4wg = globals_3wg<C>;

template <typename C>
using globals_5wg = globals_3wg<C>;

// =========================================================================
// Main kernel
// =========================================================================
template <typename C, bool USE_COL_SCRATCH = false>
__device__ inline void backward_kernel_v3_impl(const globals<C>& g) {
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
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if (cta_id == 0) tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
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
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

            int phase = 0;

            auto do_mma_block = [&](auto& accum) {
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16);
                        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0])+16*32*ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
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

        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;
        const float g_sg = g.G_sg_row[{0}];
        const float g_sg_rcp = 1.0f / g_sg;

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        constexpr int ROW16_BLOCKS = C::Mb / 32;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using subtile_rt_bf = rt_bf<C::Mb / 8, SUBTILE_COLS>;
        __shared__ bf16 col_quant_scratch[WARPGROUP_WARPS][SUBTILE_COLS][16];

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

            // ---- Step 1: Load logits from TMEM ----
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

            // Precompute per-lane target and LSE
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

            // ---- Step 2: Compute softmax gradient ----
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                subtile_rt& D_fl = D_regs_fl[epi];
                warp::mul(D_fl, D_fl, global_scale);
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

            // Free TMEM — MMA can start next block
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            // ---- Step 3: CUT filtering ----
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

            // ---- Step 4: Store output ----
            if (!tile_is_filtered) {
                if constexpr (C::USE_BF16_ACCUM) {
                    // ═══════════ BF16 PATH: store G as BF16 (same as v1) ═══════════
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
                    // ═══════ FP4 PATH: quantize G to NVFP4 from canonical BF16 shared slices ════════
                    // Materialize each epilogue slice once to shared memory, then
                    // derive both row- and col-quantized outputs from the same
                    // canonical BF16 layout. This keeps the existing CTA-local
                    // micro-scale contract while avoiding the old register-layout-
                    // dependent scatter path.
                    uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.G_fp4_row.raw_ptr);
                    uint8_t* row_sc_ptr = g.G_sc_row_ptr;
                    uint8_t* col_fp4_ptr = g.G_fp4_col_ptr;
                    uint8_t* col_sc_ptr = g.G_sc_col_ptr;
                    const int row_fp4_stride = g.G_fp4_row.cols();
                    const int row_sc_kgroups = g.G_sc_row_kgroups;
                    const int col_fp4_stride = g.A.rows() / 2;
                    const int col_sc_kgroups = g.G_sc_col_kgroups;
                    const bool encode_centric = g.encode_centric;
                    constexpr float FP4_MAX = 6.0f;
                    constexpr float E4M3_MAX = 448.0f;

                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                        const int smem_slot = epi % C::NUM_D_TILES;
                        int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[smem_slot], D_regs_bf[epi]);
                        warpgroup::sync(1);

                        const uint32_t d_base = static_cast<uint32_t>(
                            __cvta_generic_to_shared(&output_tiles.D[smem_slot].data[0]));

                        // ═══ ROW QUANTIZATION: per-row, per-16-col micro-scales ═══
                        const int quant_row = threadIdx.x;
                        if (quant_row < C::Mb / 2) {
                            const int global_row = tile_row_base + quant_row;
                            const bool full_row = global_row < g.M;
                            #pragma unroll
                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                bf16_2 vals[8];
                                float amax = 0.0f;
                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const int col = group16 * 16 + pair * 2;
                                    move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                                    amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                    amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                                }

                                const float scale = amax * (1.0f / FP4_MAX);
                                const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;

                                if (global_row < g.M) {
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        const uint8_t fp4_pair = quantize_fp4_pair(
                                            __bfloat162float(vals[pair].x),
                                            __bfloat162float(vals[pair].y),
                                            rcp_scale);
                                        const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                    }

                                    float stored_scale = scale * g_sg_rcp;
                                    if (encode_centric) {
                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                    }
                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                    const int global_col_16 = col_start + group16 * 16;
                                    const int kgroup = global_col_16 / 64;
                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                    const int depth = global_row / 128;
                                    const int sr = global_row % 32;
                                    const int rr = (global_row / 32) % 4;
                                    const int chunk = depth * row_sc_kgroups + kgroup;
                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                        *reinterpret_cast<const uint8_t*>(&sc);
                                }
                            }
                        }

                        // ═══ COLUMN QUANTIZATION: per-column, per-16-row micro-scales ═══
                        if constexpr (USE_COL_SCRATCH) {
                            const int warp_row16_base = warpgroup::warpid();
                            if (lane_id < SUBTILE_COLS) {
                                #pragma unroll
                                for (int row16_pass = 0; row16_pass < ROW16_BLOCKS / WARPGROUP_WARPS; ++row16_pass) {
                                    const int row16_block = warp_row16_base + row16_pass * WARPGROUP_WARPS;
                                    const int local_row_base = row16_block * 16;
                                    const int global_row_base = tile_row_base + local_row_base;
                                    const int global_col = col_start + lane_id;

                                    #pragma unroll
                                    for (int r = 0; r < 16; ++r) {
                                        bf16 value;
                                        move<bf16>::lds(value, G::D_tile::idx(d_base, {local_row_base + r, lane_id}));
                                        col_quant_scratch[warp_row16_base][lane_id][r] = value;
                                    }
                                    __syncwarp();

                                    if (global_col < g.N) {
                                        float col_amax = 0.0f;
                                        #pragma unroll
                                        for (int r = 0; r < 16; ++r) {
                                            const bf16 value = col_quant_scratch[warp_row16_base][lane_id][r];
                                            if (global_row_base + r < g.M) {
                                                col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value)));
                                            }
                                        }

                                        const float col_scale = col_amax * (1.0f / FP4_MAX);
                                        const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                        const int global_row_pair_base = global_row_base / 2;

                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const int global_row = global_row_base + pair * 2;
                                            if (global_row < g.M) {
                                                const float v0 = __bfloat162float(col_quant_scratch[warp_row16_base][lane_id][pair * 2]);
                                                float v1 = 0.0f;
                                                if (global_row + 1 < g.M) {
                                                    v1 = __bfloat162float(col_quant_scratch[warp_row16_base][lane_id][pair * 2 + 1]);
                                                }
                                                col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                    quantize_fp4_pair(v0, v1, col_rcp);
                                            }
                                        }

                                        float stored_scale = col_scale * g_sg_rcp;
                                        if (encode_centric) {
                                            stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                        }
                                        const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                        const int depth = global_col / 128;
                                        const int sr = global_col % 32;
                                        const int rr = (global_col / 32) % 4;
                                        const int m_kgroup = global_row_base / 64;
                                        const int m_16_in_64 = (global_row_base / 16) % 4;
                                        const int chunk = depth * col_sc_kgroups + m_kgroup;
                                        const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                        col_sc_ptr[chunk * 512 + byte_idx] =
                                            *reinterpret_cast<const uint8_t*>(&csc);
                                    }
                                    __syncwarp();
                                }
                            }
                        } else {
                            constexpr int ROW16_BLOCKS_PER_PASS = C::NUM_THREADS / 2 / SUBTILE_COLS;
                            const int col_in_epi = threadIdx.x % SUBTILE_COLS;
                            const int row16_block_base = threadIdx.x / SUBTILE_COLS;
                            #pragma unroll
                            for (int row16_pass = 0; row16_pass < ROW16_BLOCKS / ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                                const int row16_block = row16_block_base + row16_pass * ROW16_BLOCKS_PER_PASS;
                                const int global_col = col_start + col_in_epi;
                                if (row16_block < ROW16_BLOCKS && global_col < g.N) {
                                    const int local_row_base = row16_block * 16;
                                    const int global_row_base = tile_row_base + local_row_base;
                                    float col_amax = 0.0f;
                                    #pragma unroll
                                    for (int r = 0; r < 16; ++r) {
                                        bf16 value;
                                        move<bf16>::lds(value, G::D_tile::idx(d_base, {local_row_base + r, col_in_epi}));
                                        if (global_row_base + r < g.M) {
                                            col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value)));
                                        }
                                    }

                                    const float col_scale = col_amax * (1.0f / FP4_MAX);
                                    const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;

                                    const int global_row_pair_base = global_row_base / 2;
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        const int global_row = global_row_base + pair * 2;
                                        if (global_row < g.M) {
                                            bf16 value0_bf;
                                            move<bf16>::lds(value0_bf, G::D_tile::idx(d_base, {local_row_base + pair * 2, col_in_epi}));
                                            const float v0 = __bfloat162float(value0_bf);
                                            float v1 = 0.0f;
                                            if (global_row + 1 < g.M) {
                                                bf16 value1_bf;
                                                move<bf16>::lds(value1_bf, G::D_tile::idx(d_base, {local_row_base + pair * 2 + 1, col_in_epi}));
                                                v1 = __bfloat162float(value1_bf);
                                            }
                                            col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                quantize_fp4_pair(v0, v1, col_rcp);
                                        }
                                    }

                                    float stored_scale = col_scale * g_sg_rcp;
                                    if (encode_centric) {
                                        stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                    }
                                    const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                    const int depth = global_col / 128;
                                    const int sr = global_col % 32;
                                    const int rr = (global_col / 32) % 4;
                                    const int m_kgroup = global_row_base / 64;
                                    const int m_16_in_64 = (global_row_base / 16) % 4;
                                    const int chunk = depth * col_sc_kgroups + m_kgroup;
                                    const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                    col_sc_ptr[chunk * 512 + byte_idx] =
                                        *reinterpret_cast<const uint8_t*>(&csc);
                                }
                            }
                        }

                        warpgroup::sync(1);
                    }  // for epi
                }  // if !C::USE_BF16_ACCUM
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

template <typename C, bool DO_ROW, bool DO_COL, bool LOCAL_S_PER_CTA = false, bool STORE_BF16_OUT = false>
__device__ inline void backward_kernel_v3_streaming_impl(const globals<C>& g) {
    if (g.filter_eps > 0.0f) {
        backward_kernel_v3_impl<C, false>(g);
        return;
    }

    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        if constexpr (G::ENABLE_STOREADD_COMBO) {
            g.C_col.template prefetch_tma<typename G::combo_p3_C_tile>();
            g.C_col_sc.template prefetch_tma<typename G::combo_p3_C_sc_tile>();
            g.E_col.template prefetch_tma<typename G::combo_p3_E_tile>();
            g.E_col_sc.template prefetch_tma<typename G::combo_p3_E_sc_tile>();
            g.dE_out.template prefetch_tma<typename G::combo_dE_tile>();
            g.dC_out.template prefetch_tma<typename G::combo_dC_tile>();
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
        if constexpr (LOCAL_S_PER_CTA) init_semaphore(outputs_finished, 0, 1);
        else                           init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

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
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if (cta_id == 0) tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if ((LOCAL_S_PER_CTA && warp_id == 0) ||
                   (!LOCAL_S_PER_CTA && cta_id == 0 && warp_id == 0)) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

            int phase = 0;

            auto do_mma_block = [&](auto& accum) {
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16);
                        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0])+16*32*ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
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
                if constexpr (LOCAL_S_PER_CTA) tensor_commit<2>(outputs_arrived, static_cast<uint16_t>(1u << cta_id));
                else                           tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
                phase ^= 1;
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

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;
        const float g_sg = g.G_sg_row[{0}];
        const float g_sg_rcp = 1.0f / g_sg;

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        constexpr int ROW16_BLOCKS = C::Mb / 32;
        constexpr int ROW16_BLOCKS_PER_PASS = C::NUM_THREADS / 2 / SUBTILE_COLS;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
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

            const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
            const bool full_tile_rows = tile_row_base + (C::Mb / 2) <= g.M;

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

            uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.G_fp4_row.raw_ptr);
            uint8_t* row_sc_ptr = g.G_sc_row_ptr;
            uint8_t* col_fp4_ptr = g.G_fp4_col_ptr;
            uint8_t* col_sc_ptr = g.G_sc_col_ptr;
            const int row_fp4_stride = g.G_fp4_row.cols();
            const int row_sc_kgroups = g.G_sc_row_kgroups;
            const int col_fp4_stride = g.A.rows() / 2;
            const int col_sc_kgroups = g.G_sc_col_kgroups;
            const bool encode_centric = g.encode_centric;
            constexpr float FP4_MAX = 6.0f;
            constexpr float E4M3_MAX = 448.0f;
            auto& accum = (phase == 0) ? out_tm_0 : out_tm_1;

            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                subtile_rt D_fl;
                subtile_rt_bf D_bf;
                warpgroup::load_async(D_fl, accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                warp::mul(D_fl, D_fl, global_scale);
                const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;

                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    const float lse_x = my_lse_x[i];
                    const float lse_y = my_lse_y[i];
                    #pragma unroll
                    for (int j = 0; j < subtile_rt::width; j++) {
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            const float lse_val = (k % 2 == 0) ? lse_x : lse_y;
                            D_fl.tiles[i][j].data[k].x = __expf(D_fl.tiles[i][j].data[k].x - lse_val);
                            D_fl.tiles[i][j].data[k].y = __expf(D_fl.tiles[i][j].data[k].y - lse_val);
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    const int tgt_x = my_targets_x[i];
                    if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
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
                    if (tgt_y >= col_start && tgt_y < col_start + SUBTILE_COLS) {
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
                for (int i = 0; i < subtile_rt::height; i++) {
                    const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                    const int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
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
                warp::copy(D_bf, D_fl);

                if (epi == C::EPI_PIPE_DEPTH - 1) {
                    if constexpr (LOCAL_S_PER_CTA) {
                        if (warpgroup::warpid() == 0 && warp::laneid() == 0) arrive(outputs_finished);
                    }
                    else                           warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                }

                const int smem_slot = epi % C::NUM_D_TILES;
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[smem_slot], D_bf);
                warpgroup::sync(1);

                const uint32_t d_base = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&output_tiles.D[smem_slot].data[0]));

                if constexpr (STORE_BF16_OUT) {
                    bf16* d_out_ptr = reinterpret_cast<bf16*>(g.D_out.raw_ptr);
                    const int d_out_stride = g.D_out.cols();
                    for (int linear = threadIdx.x; linear < (C::Mb / 2) * SUBTILE_COLS; linear += C::NUM_THREADS) {
                        const int local_row = linear / SUBTILE_COLS;
                        const int local_col = linear % SUBTILE_COLS;
                        const int global_row = tile_row_base + local_row;
                        const int global_col = col_start + local_col;
                        if (global_row < g.M && global_col < g.N) {
                            bf16 value;
                            move<bf16>::lds(value, G::D_tile::idx(d_base, {local_row, local_col}));
                            d_out_ptr[global_row * d_out_stride + global_col] = value;
                        }
                    }
                    warpgroup::sync(1);
                }

                if constexpr (DO_ROW) {
                    const int quant_row = threadIdx.x;
                    if (quant_row < C::Mb / 2) {
                        const int global_row = tile_row_base + quant_row;
                        #pragma unroll
                        for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                            bf16_2 vals[8];
                            float amax = 0.0f;
                            #pragma unroll
                            for (int pair = 0; pair < 8; ++pair) {
                                const int col = group16 * 16 + pair * 2;
                                move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                                amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                            }

                            const float scale = amax * (1.0f / FP4_MAX);
                            const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;

                            if (global_row < g.M) {
                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const uint8_t fp4_pair = quantize_fp4_pair(
                                        __bfloat162float(vals[pair].x),
                                        __bfloat162float(vals[pair].y),
                                        rcp_scale);
                                    const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                    row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                }

                                float stored_scale = scale * g_sg_rcp;
                                if (encode_centric) {
                                    stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                }
                                const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                const int global_col_16 = col_start + group16 * 16;
                                const int kgroup = global_col_16 / 64;
                                const int col_16_in_64 = (global_col_16 / 16) % 4;
                                const int depth = global_row / 128;
                                const int sr = global_row % 32;
                                const int rr = (global_row / 32) % 4;
                                const int chunk = depth * row_sc_kgroups + kgroup;
                                const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                row_sc_ptr[chunk * 512 + byte_idx] =
                                    *reinterpret_cast<const uint8_t*>(&sc);
                            }
                        }
                    }
                }

                if constexpr (DO_COL) {
                    const int col_in_epi = threadIdx.x % SUBTILE_COLS;
                    const int row16_block_base = threadIdx.x / SUBTILE_COLS;
                    #pragma unroll
                    for (int row16_pass = 0; row16_pass < ROW16_BLOCKS / ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                        const int row16_block = row16_block_base + row16_pass * ROW16_BLOCKS_PER_PASS;
                        const int global_col = col_start + col_in_epi;
                        if (row16_block < ROW16_BLOCKS && global_col < g.N) {
                            const int local_row_base = row16_block * 16;
                            const int global_row_base = tile_row_base + local_row_base;
                            float col_amax = 0.0f;
                            #pragma unroll
                            for (int r = 0; r < 16; ++r) {
                                bf16 value;
                                move<bf16>::lds(value, G::D_tile::idx(d_base, {local_row_base + r, col_in_epi}));
                                if (global_row_base + r < g.M) {
                                    col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value)));
                                }
                            }

                            const float col_scale = col_amax * (1.0f / FP4_MAX);
                            const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                            const int global_row_pair_base = global_row_base / 2;

                            #pragma unroll
                            for (int pair = 0; pair < 8; ++pair) {
                                const int global_row = global_row_base + pair * 2;
                                if (global_row < g.M) {
                                    bf16 value0_bf;
                                    move<bf16>::lds(value0_bf, G::D_tile::idx(d_base, {local_row_base + pair * 2, col_in_epi}));
                                    const float v0 = __bfloat162float(value0_bf);
                                    float v1 = 0.0f;
                                    if (global_row + 1 < g.M) {
                                        bf16 value1_bf;
                                        move<bf16>::lds(value1_bf, G::D_tile::idx(d_base, {local_row_base + pair * 2 + 1, col_in_epi}));
                                        v1 = __bfloat162float(value1_bf);
                                    }
                                    col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                        quantize_fp4_pair(v0, v1, col_rcp);
                                }
                            }

                            float stored_scale = col_scale * g_sg_rcp;
                            if (encode_centric) {
                                stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                            }
                            const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                            const int depth = global_col / 128;
                            const int sr = global_col % 32;
                            const int rr = (global_col / 32) % 4;
                            const int m_kgroup = global_row_base / 64;
                            const int m_16_in_64 = (global_row_base / 16) % 4;
                            const int chunk = depth * col_sc_kgroups + m_kgroup;
                            const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                            col_sc_ptr[chunk * 512 + byte_idx] =
                                *reinterpret_cast<const uint8_t*>(&csc);
                        }
                    }
                }

                warpgroup::sync(1);
            }

            update_phasebit<0>(phasebits, 0);
            phase ^= 1;
        }
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

template <typename C, bool DO_ROW, bool DO_COL>
__device__ inline void backward_kernel_v3_streaming_2ctaSdupB_impl(const globals_2ctaSdupB<C>& g) {
    using G = globals_2ctaSdupB<C>;

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
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<typename G::outputs_t>();

    tensor_allocator<1, 1, false> tm_allocator;

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
        init_semaphore(outputs_finished, 0, 1);
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS * C::PRODUCER_WARPGROUPS>::warpid();
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
                    tma::load_async(input_tiles[stage].A, g.A, {row_block_idx * 2 + cta_id, i}, tiles_arrived[stage]);
                    tma::load_async(input_tiles[stage].B[0], g.B, {col_block_idx * 2 + 0, i}, tiles_arrived[stage]);
                    tma::load_async(input_tiles[stage].B[1], g.B, {col_block_idx * 2 + 1, i}, tiles_arrived[stage]);
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
                    tma::load_async(input_scales[stage].A, g.A_sc, {row_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage]);
                    #pragma unroll
                    for (int sc = 0; sc < C::B_SC_SIZE; ++sc) {
                        tma::load_async(input_scales[stage].B[0][sc], g.B_sc, {col_block_idx * 2 + 0, i, sc}, scales_arrived[stage]);
                        tma::load_async(input_scales[stage].B[1][sc], g.B_sc, {col_block_idx * 2 + 1, i, sc}, scales_arrived[stage]);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 0) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb / 2>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb / 2>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE>>(256 + 4 * C::MMA_PER_TILE * (C::LOAD_PIPE_DEPTH + 1));

            constexpr uint32_t SCALE_BYTES = sizeof(typename G::A_sc_tile) + 2 * C::B_SC_SIZE * sizeof(typename G::B_sc_tile);
            constexpr uint32_t TILE_BYTES = sizeof(typename G::A_fp4x2_tile) + 2 * sizeof(typename G::B_fp4x2_tile);

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::expect_bytes(scales_arrived[stage], SCALE_BYTES);
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_sub = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_sub, A_sc_sm_sub);

                        auto B_sc_tm_sub_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        auto &B_sc_sm_sub_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_sub_0, B_sc_sm_sub_0);

                        auto B_sc_tm_sub_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        auto &B_sc_sm_sub_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1][0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_sub_1, B_sc_sm_sub_1);
                    }

                    tma::expect_bytes(tiles_arrived[stage], TILE_BYTES);
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(0);
                    auto B_sc_tm_tile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(0);

                    if (i == 0) {
                        mm_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mm_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                    } else {
                        mma_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mma_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
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

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb / 2>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb / 2>>(128);

        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;
        const float g_sg = g.G_sg_row[{0}];
        const float g_sg_rcp = 1.0f / g_sg;

        constexpr int LOCAL_N = C::Nb / 2;
        constexpr int SUBTILE_COLS = LOCAL_N / C::EPI_PIPE_DEPTH;
        constexpr int ROW16_BLOCKS = C::Mb / 32;
        constexpr int ROW16_BLOCKS_PER_PASS = C::NUM_THREADS / 2 / SUBTILE_COLS;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using subtile_rt_bf = rt_bf<C::Mb / 8, SUBTILE_COLS>;
        const int lane_id = threadIdx.x % 32;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
            const bool full_tile_rows = tile_row_base + (C::Mb / 2) <= g.M;

            int my_targets_x[subtile_rt::height];
            int my_targets_y[subtile_rt::height];
            float my_lse_x[subtile_rt::height];
            float my_lse_y[subtile_rt::height];
            #pragma unroll
            for (int i = 0; i < subtile_rt::height; ++i) {
                int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                my_targets_x[i] = (global_row_x < g.M) ? (int)g.targets[global_row_x] : -1;
                my_targets_y[i] = (global_row_y < g.M) ? (int)g.targets[global_row_y] : -1;
                my_lse_x[i] = (global_row_x < g.M) ? g.lse[global_row_x] : INFINITY;
                my_lse_y[i] = (global_row_y < g.M) ? g.lse[global_row_y] : INFINITY;
            }

            uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.G_fp4_row.raw_ptr);
            uint8_t* row_sc_ptr = g.G_sc_row_ptr;
            uint8_t* col_fp4_ptr = g.G_fp4_col_ptr;
            uint8_t* col_sc_ptr = g.G_sc_col_ptr;
            const int row_fp4_stride = g.G_fp4_row.cols();
            const int row_sc_kgroups = g.G_sc_row_kgroups;
            const int col_fp4_stride = g.A.rows() / 2;
            const int col_sc_kgroups = g.G_sc_col_kgroups;
            const bool encode_centric = g.encode_centric;
            constexpr float FP4_MAX = 6.0f;
            constexpr float E4M3_MAX = 448.0f;

            auto consume_local_tile = [&](auto& accum, int col_tile_offset, bool release_outputs) {
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    subtile_rt D_fl;
                    subtile_rt_bf D_bf;
                    warpgroup::load_async(D_fl, accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);

                    warp::mul(D_fl, D_fl, global_scale);
                    const int col_start = col_block_idx * C::Nb + col_tile_offset * LOCAL_N + epi * SUBTILE_COLS;

                    #pragma unroll
                    for (int i = 0; i < subtile_rt::height; ++i) {
                        const float lse_x = my_lse_x[i];
                        const float lse_y = my_lse_y[i];
                        #pragma unroll
                        for (int j = 0; j < subtile_rt::width; ++j) {
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const float lse_val = (k % 2 == 0) ? lse_x : lse_y;
                                D_fl.tiles[i][j].data[k].x = __expf(D_fl.tiles[i][j].data[k].x - lse_val);
                                D_fl.tiles[i][j].data[k].y = __expf(D_fl.tiles[i][j].data[k].y - lse_val);
                            }
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < subtile_rt::height; ++i) {
                        const int tgt_x = my_targets_x[i];
                        if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
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
                        if (tgt_y >= col_start && tgt_y < col_start + SUBTILE_COLS) {
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
                    for (int i = 0; i < subtile_rt::height; ++i) {
                        const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                        const int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                        #pragma unroll
                        for (int j = 0; j < subtile_rt::width; ++j) {
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
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
                    warp::copy(D_bf, D_fl);

                    if (release_outputs && epi == C::EPI_PIPE_DEPTH - 1) {
                        if (warpgroup::warpid() == 0 && warp::laneid() == 0) arrive(outputs_finished);
                    }

                    const int smem_slot = epi % C::NUM_D_TILES;
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[smem_slot], D_bf);
                    warpgroup::sync(1);

                    const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[smem_slot].data[0]));

                    if constexpr (DO_ROW) {
                        const int quant_row = threadIdx.x;
                        if (quant_row < C::Mb / 2) {
                            const int global_row = tile_row_base + quant_row;
                            const bool full_row = global_row < g.M;
                            #pragma unroll
                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                bf16_2 vals[8];
                                float amax = 0.0f;
                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const int col = group16 * 16 + pair * 2;
                                    move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                                    amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                    amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                                }
                                const float scale = amax * (1.0f / FP4_MAX);
                                const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                if constexpr (C::FAST_ALIGNED_QUANT) {
                                    if (full_row) {
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const uint8_t fp4_pair = quantize_fp4_pair(
                                                __bfloat162float(vals[pair].x),
                                                __bfloat162float(vals[pair].y),
                                                rcp_scale);
                                            const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                            row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                        }
                                        float stored_scale = scale * g_sg_rcp;
                                        if (encode_centric) {
                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                        }
                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                        const int global_col_16 = col_start + group16 * 16;
                                        const int kgroup = global_col_16 / 64;
                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                        const int depth = global_row / 128;
                                        const int sr = global_row % 32;
                                        const int rr = (global_row / 32) % 4;
                                        const int chunk = depth * row_sc_kgroups + kgroup;
                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                        row_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&sc);
                                        continue;
                                    }
                                }
                                if (global_row < g.M) {
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        const uint8_t fp4_pair = quantize_fp4_pair(
                                            __bfloat162float(vals[pair].x),
                                            __bfloat162float(vals[pair].y),
                                            rcp_scale);
                                        const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                    }

                                    float stored_scale = scale * g_sg_rcp;
                                    if (encode_centric) stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                    const int global_col_16 = col_start + group16 * 16;
                                    const int kgroup = global_col_16 / 64;
                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                    const int depth = global_row / 128;
                                    const int sr = global_row % 32;
                                    const int rr = (global_row / 32) % 4;
                                    const int chunk = depth * row_sc_kgroups + kgroup;
                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                    row_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&sc);
                                }
                            }
                        }
                    }

                    if constexpr (DO_COL) {
                        const int col_in_epi = threadIdx.x % SUBTILE_COLS;
                        const int row16_block_base = threadIdx.x / SUBTILE_COLS;
                        #pragma unroll
                        for (int row16_pass = 0; row16_pass < ROW16_BLOCKS / ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                            const int row16_block = row16_block_base + row16_pass * ROW16_BLOCKS_PER_PASS;
                            const int global_col = col_start + col_in_epi;
                            if (row16_block < ROW16_BLOCKS && global_col < g.N) {
                                const int local_row_base = row16_block * 16;
                                const int global_row_base = tile_row_base + local_row_base;
                                float col_amax = 0.0f;
                                #pragma unroll
                                for (int r = 0; r < 16; ++r) {
                                    bf16 value;
                                    move<bf16>::lds(value, G::D_tile::idx(d_base, {local_row_base + r, col_in_epi}));
                                    if (global_row_base + r < g.M) col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value)));
                                }

                                const float col_scale = col_amax * (1.0f / FP4_MAX);
                                const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                const int global_row_pair_base = global_row_base / 2;

                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const int global_row = global_row_base + pair * 2;
                                    if (global_row < g.M) {
                                        bf16 value0_bf;
                                        move<bf16>::lds(value0_bf, G::D_tile::idx(d_base, {local_row_base + pair * 2, col_in_epi}));
                                        const float v0 = __bfloat162float(value0_bf);
                                        float v1 = 0.0f;
                                        if (global_row + 1 < g.M) {
                                            bf16 value1_bf;
                                            move<bf16>::lds(value1_bf, G::D_tile::idx(d_base, {local_row_base + pair * 2 + 1, col_in_epi}));
                                            v1 = __bfloat162float(value1_bf);
                                        }
                                        col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                            quantize_fp4_pair(v0, v1, col_rcp);
                                    }
                                }

                                float stored_scale = col_scale * g_sg_rcp;
                                if (encode_centric) stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                const int depth = global_col / 128;
                                const int sr = global_col % 32;
                                const int rr = (global_col / 32) % 4;
                                const int m_kgroup = global_row_base / 64;
                                const int m_16_in_64 = (global_row_base / 16) % 4;
                                const int chunk = depth * col_sc_kgroups + m_kgroup;
                                const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                col_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&csc);
                            }
                        }
                    }

                    warpgroup::sync(1);
                }
            };

            consume_local_tile(out_tm_0, 0, false);
            consume_local_tile(out_tm_1, 1, true);

            update_phasebit<0>(phasebits, 0);
        }

        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

template <typename C, bool DO_ROW, bool DO_COL, int STATIC_COMBO_MODE = -1, int EXTRA_INACTIVE_WG_BEGIN = -1>
__device__ inline void backward_kernel_v3_streaming_3wg_impl(const globals_3wg<C>& g) {
    using G = globals_3wg<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    constexpr bool STATIC_FRONTEND_ONLY_GONLY =
        G::SEPARATE_BACKHALF_CONSUMERS &&
        (STATIC_COMBO_MODE == G::COMBO_MODE_GONLY);
    constexpr bool STATIC_FRONTEND_ONLY_DEONLY =
        G::SEPARATE_BACKHALF_CONSUMERS &&
        (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY);
    constexpr bool EFFECTIVE_ENABLE_STOREADD_COMBO =
        G::ENABLE_STOREADD_COMBO && !STATIC_FRONTEND_ONLY_GONLY;
    constexpr bool EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS =
        G::SEPARATE_BACKHALF_CONSUMERS && EFFECTIVE_ENABLE_STOREADD_COMBO;
    constexpr int EFFECTIVE_QUANTIZER_WARPGROUPS =
        STATIC_FRONTEND_ONLY_GONLY ? G::FRONTEND_QUANTIZER_WARPGROUPS
                                   : (STATIC_FRONTEND_ONLY_DEONLY ? 0 : C::QUANTIZER_WARPGROUPS);
    constexpr int EFFECTIVE_PRODUCER_WARPGROUP_ID =
        C::CONSUMER_WARPGROUPS + EFFECTIVE_QUANTIZER_WARPGROUPS;
    const int effective_warpgroup_id = warpgroup_id;
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
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::bf16_stage_t (&bf16_epi_stage)[C::BF16_STAGE_COUNT] =
        sm_allocator.allocate<typename G::bf16_stage_t, C::BF16_STAGE_COUNT>();
    typename G::col_plain_stage_t *col_plain_stage = nullptr;
    if constexpr (G::USE_COL_PLAIN_STAGE) {
        auto &col_plain_stage_ref =
            sm_allocator.template allocate<typename G::col_plain_stage_t, C::BF16_STAGE_COUNT>();
        col_plain_stage = &col_plain_stage_ref[0];
    }
    typename G::col_pair_stage_t *col_pair_stage = nullptr;
    if constexpr (G::USE_COL_PAIR_STAGE) {
        auto &col_pair_stage_ref =
            sm_allocator.template allocate<typename G::col_pair_stage_t, C::BF16_STAGE_COUNT>();
        col_pair_stage = &col_pair_stage_ref[0];
    }
    typename G::row_pair_stage_t *row_pair_stage = nullptr;
    if constexpr (G::USE_ROW_PAIR_STAGE) {
        auto &row_pair_stage_ref =
            sm_allocator.template allocate<typename G::row_pair_stage_t, C::BF16_STAGE_COUNT>();
        row_pair_stage = &row_pair_stage_ref[0];
    }
    typename G::row_rcp_stage_t *row_rcp_stage = nullptr;
    if constexpr (G::USE_ROW_RCP_STAGE) {
        auto &row_rcp_stage_ref =
            sm_allocator.template allocate<typename G::row_rcp_stage_t, C::BF16_STAGE_COUNT>();
        row_rcp_stage = &row_rcp_stage_ref[0];
    }
    typename G::combo_row_stage_t *combo_row_stage = nullptr;
    typename G::combo_col_stage_t *combo_col_stage = nullptr;
    typename G::combo_p3_c_tile_stage_t *combo_p3_c_tile_stage = nullptr;
    typename G::combo_de_scales_stage_t *combo_de_scales_stage = nullptr;
    typename G::combo_de_output_stage_t *combo_de_output_stage = nullptr;
    typename G::combo_dc_tile_stage_t *combo_dc_tile_stage = nullptr;
    typename G::combo_dc_scales_stage_t *combo_dc_scales_stage = nullptr;
    typename G::combo_dc_output_stage_t *combo_dc_output_stage = nullptr;
    typename G::combo_output_stage_t *combo_output_stage = nullptr;
    if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO) {
        if constexpr (EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
            combo_row_stage = &sm_allocator.template allocate<typename G::combo_row_stage_t>();
            combo_col_stage = &sm_allocator.template allocate<typename G::combo_col_stage_t>();
            combo_p3_c_tile_stage = &sm_allocator.template allocate<typename G::combo_p3_c_tile_stage_t>();
            combo_de_scales_stage = &sm_allocator.template allocate<typename G::combo_de_scales_stage_t>();
            combo_output_stage = &sm_allocator.template allocate<typename G::combo_output_stage_t>();
            combo_de_output_stage =
                reinterpret_cast<typename G::combo_de_output_stage_t *>(combo_output_stage);
            combo_dc_tile_stage =
                reinterpret_cast<typename G::combo_dc_tile_stage_t *>(combo_output_stage);
            combo_dc_scales_stage =
                reinterpret_cast<typename G::combo_dc_scales_stage_t *>(&input_scales[0]);
            combo_dc_output_stage =
                reinterpret_cast<typename G::combo_dc_output_stage_t *>(combo_output_stage);
        } else {
            combo_row_stage = &sm_allocator.template allocate<typename G::combo_row_stage_t>();
            combo_col_stage = &sm_allocator.template allocate<typename G::combo_col_stage_t>();
            combo_p3_c_tile_stage = &sm_allocator.template allocate<typename G::combo_p3_c_tile_stage_t>();
            combo_output_stage = &sm_allocator.template allocate<typename G::combo_output_stage_t>();
        }
    }

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore slice_row_ready[C::BF16_STAGE_COUNT];
    __shared__ semaphore slice_col_ready[C::BF16_STAGE_COUNT];
    __shared__ semaphore slice_col_ready_row16[C::BF16_STAGE_COUNT][C::Mb / 32];
    __shared__ semaphore slice_row_recycled[C::BF16_STAGE_COUNT];
    __shared__ semaphore slice_col_recycled[C::BF16_STAGE_COUNT];
    __shared__ volatile int slice_col_done_row16[C::BF16_STAGE_COUNT][C::Mb / 32];
    __shared__ semaphore combo_row_stage_ready;
    __shared__ semaphore combo_g_row_ready[C::BF16_STAGE_COUNT];
    __shared__ semaphore combo_g_row_recycled[C::BF16_STAGE_COUNT];
    __shared__ semaphore combo_g_col_ready[C::BF16_STAGE_COUNT];
    __shared__ semaphore combo_g_col_recycled[C::BF16_STAGE_COUNT];
    __shared__ semaphore combo_p3_c_tiles_arrived;
    __shared__ semaphore combo_p3_c_scales_arrived;
    __shared__ semaphore combo_p3_e_tiles_arrived;
    __shared__ semaphore combo_p3_e_scales_arrived;
    __shared__ semaphore combo_p3_inputs_finished;
    __shared__ semaphore combo_p3_outputs_arrived;
    __shared__ semaphore combo_de_p3_c_tiles_arrived;
    __shared__ semaphore combo_de_p3_c_scales_arrived;
    __shared__ semaphore combo_de_p3_inputs_finished;
    __shared__ semaphore combo_de_p3_outputs_arrived;
    __shared__ semaphore combo_dc_p3_e_tiles_arrived;
    __shared__ semaphore combo_dc_p3_e_scales_arrived;
    __shared__ semaphore combo_dc_p3_inputs_finished;
    __shared__ semaphore combo_dc_p3_outputs_arrived;
    __shared__ semaphore combo_backhalf_de_done;
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
        constexpr int row_recycle_arrivals = G::FRONTEND_SINGLE_QUANTIZER_WG ? 1 : (C::ROW_QUANTIZER_WARPGROUPS > 0 ? C::ROW_QUANTIZER_WARPGROUPS : 1);
        constexpr int col_recycle_arrivals = G::FRONTEND_SINGLE_QUANTIZER_WG ? 1 : (C::COL_QUANTIZER_WARPGROUPS > 0 ? C::COL_QUANTIZER_WARPGROUPS : 1);
        #pragma unroll
        for (int i = 0; i < C::BF16_STAGE_COUNT; ++i) {
            init_semaphore(slice_row_ready[i], 0, 1);
            init_semaphore(slice_col_ready[i], 0, 1);
            #pragma unroll
            for (int row16 = 0; row16 < C::Mb / 32; ++row16) {
                init_semaphore(slice_col_ready_row16[i][row16], 0, 1);
                slice_col_done_row16[i][row16] = 0;
            }
            init_semaphore(slice_row_recycled[i], 0, row_recycle_arrivals);
            init_semaphore(slice_col_recycled[i], 0, col_recycle_arrivals);
        }
        if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO) {
            if constexpr (EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                #pragma unroll
                for (int i = 0; i < C::BF16_STAGE_COUNT; ++i) {
                    init_semaphore(combo_g_row_ready[i], 0, 1);
                    init_semaphore(combo_g_row_recycled[i], 0, 1);
                    init_semaphore(combo_g_col_ready[i], 0, 1);
                    init_semaphore(combo_g_col_recycled[i], 0, 1);
                }
                init_semaphore(combo_de_p3_c_tiles_arrived, 0, 1);
                init_semaphore(combo_de_p3_c_scales_arrived, 0, 1);
                init_semaphore(combo_de_p3_inputs_finished, 0, 1);
                init_semaphore(combo_de_p3_outputs_arrived, 0, 1);
                init_semaphore(combo_dc_p3_e_tiles_arrived, 0, 1);
                init_semaphore(combo_dc_p3_e_scales_arrived, 0, 1);
                init_semaphore(combo_dc_p3_inputs_finished, 0, 1);
                init_semaphore(combo_dc_p3_outputs_arrived, 0, 1);
                init_semaphore(combo_backhalf_de_done, 0, 1);
            } else {
                init_semaphore(combo_row_stage_ready, 0, 1);
                init_semaphore(combo_p3_c_tiles_arrived, 0, 1);
                init_semaphore(combo_p3_c_scales_arrived, 0, 1);
                init_semaphore(combo_p3_e_tiles_arrived, 0, 1);
                init_semaphore(combo_p3_e_scales_arrived, 0, 1);
                init_semaphore(combo_p3_inputs_finished, 0, 1);
                init_semaphore(combo_p3_outputs_arrived, 0, 1);
            }
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if constexpr (EXTRA_INACTIVE_WG_BEGIN >= 0) {
        if (warpgroup_id >= EXTRA_INACTIVE_WG_BEGIN) {
            everyone::tma::cluster::wait_aligned();
            return;
        }
    }

    if constexpr (STATIC_FRONTEND_ONLY_GONLY) {
        constexpr int first_inactive_backhalf_wg =
            EFFECTIVE_PRODUCER_WARPGROUP_ID + C::PRODUCER_WARPGROUPS;
        constexpr int end_inactive_backhalf_wg =
            C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS;
        if (warpgroup_id >= first_inactive_backhalf_wg &&
            warpgroup_id < end_inactive_backhalf_wg) {
            everyone::tma::cluster::wait_aligned();
            return;
        }
    }
    if constexpr (STATIC_FRONTEND_ONLY_DEONLY) {
        constexpr int first_inactive_backhalf_wg =
            EFFECTIVE_PRODUCER_WARPGROUP_ID + C::PRODUCER_WARPGROUPS + G::COMBO_DE_WARPGROUPS;
        constexpr int end_inactive_backhalf_wg =
            C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS;
        if (warpgroup_id >= first_inactive_backhalf_wg &&
            warpgroup_id < end_inactive_backhalf_wg) {
            everyone::tma::cluster::wait_aligned();
            return;
        }
    }

    if (effective_warpgroup_id == EFFECTIVE_PRODUCER_WARPGROUP_ID && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS * C::PRODUCER_WARPGROUPS>::warpid();
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
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
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
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    if (cta_id == 0) {
                        tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

            int phase = 0;

            auto do_mma_block = [&](auto& accum) {
                for (int i = 0; i < num_iters_per_block; ++i) {
                    tma::expect_bytes(scales_arrived[stage], 2 * sizeof(typename G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16);
                        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(typename G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                        inputs_finished[stage]);
                    else mma2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                  A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                  B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
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
    } else if (effective_warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;
        const float g_sg = g.G_sg_row[{0}];
        const float g_sg_rcp = 1.0f / g_sg;
        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using subtile_rt_bf = rt_bf<C::Mb / 8, SUBTILE_COLS>;
        const int lane_id = threadIdx.x % 32;
        uint32_t slice_phasebits = 0xFFFF0000;
        uint32_t slice_row_recycle_phasebits = 0xFFFF0000;
        uint32_t slice_col_recycle_phasebits = 0xFFFF0000;
        int phase = 0;
        uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.G_fp4_row.raw_ptr);
        uint8_t* row_sc_ptr = g.G_sc_row_ptr;
        uint8_t* col_fp4_ptr = g.G_fp4_col_ptr;
        uint8_t* col_sc_ptr = g.G_sc_col_ptr;
        const int row_fp4_stride = g.G_fp4_row.cols();
        const int row_sc_kgroups = g.G_sc_row_kgroups;
        const int col_fp4_stride = g.A.rows() / 2;
        const int col_sc_kgroups = g.G_sc_col_kgroups;
        const bool encode_centric = g.encode_centric;
        constexpr bool COMBO_MODE_IS_STATIC = (STATIC_COMBO_MODE >= 0);
        constexpr bool STATIC_COMBO_DO_DE =
            COMBO_MODE_IS_STATIC &&
            (STATIC_COMBO_MODE == G::COMBO_MODE_FULL || STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY);
        constexpr bool STATIC_COMBO_DO_DC =
            COMBO_MODE_IS_STATIC &&
            (STATIC_COMBO_MODE == G::COMBO_MODE_FULL || STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY);
        constexpr bool STATIC_COMBO_GONLY =
            COMBO_MODE_IS_STATIC && (STATIC_COMBO_MODE == G::COMBO_MODE_GONLY);
        const bool combo_use_de =
            G::ENABLE_STOREADD_COMBO &&
            G::SEPARATE_BACKHALF_CONSUMERS &&
            (COMBO_MODE_IS_STATIC
                ? STATIC_COMBO_DO_DE
                : (g.combo_mode == G::COMBO_MODE_FULL || g.combo_mode == G::COMBO_MODE_DEONLY));
        constexpr bool combo_debug_skip_consumer_row_recycle_wait = false;
        constexpr bool combo_debug_force_row_ready_publish = false;
        uint32_t combo_row_recycle_phasebits = 0xFFFF0000;
        constexpr int CONSUMER_WG_SYNC_ID =
            STATIC_FRONTEND_ONLY_GONLY ? 3 : (G::ROW_QUANT_IMMEDIATE_PER_ROW16 ? 3 : 1);
        constexpr float FP4_MAX = 6.0f;
        constexpr float E4M3_MAX = 448.0f;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
            const bool full_tile_rows = tile_row_base + (C::Mb / 2) <= g.M;

            int my_targets_x[subtile_rt::height];
            int my_targets_y[subtile_rt::height];
            float my_lse_x[subtile_rt::height];
            float my_lse_y[subtile_rt::height];
            #pragma unroll
            for (int i = 0; i < subtile_rt::height; ++i) {
                int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                my_targets_x[i] = (global_row_x < g.M) ? (int)g.targets[global_row_x] : -1;
                my_targets_y[i] = (global_row_y < g.M) ? (int)g.targets[global_row_y] : -1;
                my_lse_x[i] = (global_row_x < g.M) ? g.lse[global_row_x] : INFINITY;
                my_lse_y[i] = (global_row_y < g.M) ? g.lse[global_row_y] : INFINITY;
            }

            auto& accum = (phase == 0) ? out_tm_0 : out_tm_1;
            auto wait_for_quant_stage = [&](int bf_stage) {
                if constexpr (DO_ROW && !C::CONSUMER_DO_ROW) {
                    if constexpr (!G::DEBUG_SKIP_ROW_RECYCLE_WAIT) {
                        wait(slice_row_recycled[bf_stage], get_phasebit<1>(slice_row_recycle_phasebits, bf_stage));
                    }
                }
                if constexpr (DO_COL) {
                    if constexpr (G::COL_QUANT_FROM_DBF_REGS) {
                    } else if constexpr (G::USE_COL_PAIR_STAGE || (DO_ROW && C::ROW_QUANT_FROM_REGS && C::CONSUMER_DO_ROW && !C::EARLY_COL_READY)) {
                        if constexpr (!G::DEBUG_SKIP_COL_RECYCLE_WAIT) {
                            wait(slice_col_recycled[bf_stage], get_phasebit<1>(slice_col_recycle_phasebits, bf_stage));
                        }
                    } else {
                        wait(slice_col_recycled[bf_stage], get_phasebit<1>(slice_phasebits, bf_stage));
                    }
                }
                if constexpr (DO_ROW && DO_COL && G::ROW_WAITS_FOR_COL_DONE_ROW16) {
                    if (warpgroup::warpid() == 0 && warp::laneid() < (C::Mb / 32)) {
                        slice_col_done_row16[bf_stage][warp::laneid()] = 0;
                    }
                    warpgroup::sync(CONSUMER_WG_SYNC_ID);
                }
            };
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                const int bf_stage = epi % C::BF16_STAGE_COUNT;
                if constexpr (!C::OVERLAP_EPI) {
                    wait_for_quant_stage(bf_stage);
                }

                subtile_rt_bf D_bf;
                const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                const bool combo_stage_first_epi = (epi == 0);
                const bool combo_stage_publish_epi = (epi == C::EPI_PIPE_DEPTH - 1);
                typename G::combo_row_stage_t *combo_row_stage_slot = nullptr;
                uint8_t* combo_row_fp4_local_ptr = nullptr;
                uint8_t* combo_row_sc_local_ptr = nullptr;
                if constexpr (G::ENABLE_STOREADD_COMBO) {
                    if (combo_use_de) {
                        combo_row_stage_slot = combo_row_stage;
                        combo_row_fp4_local_ptr =
                            reinterpret_cast<uint8_t*>(&combo_row_stage_slot->G_row.data[0]);
                        combo_row_sc_local_ptr =
                            reinterpret_cast<uint8_t*>(&combo_row_stage_slot->G_row_sc.data[0]);
                    }
                }
                if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                    if constexpr (DO_ROW && C::CONSUMER_DO_ROW) {
                        if (combo_use_de && combo_stage_first_epi) {
                            if constexpr (!combo_debug_skip_consumer_row_recycle_wait) {
                                wait(combo_g_row_recycled[0], get_phasebit<1>(combo_row_recycle_phasebits, 0));
                            }
                            uint8_t *combo_row_bytes = reinterpret_cast<uint8_t*>(&combo_row_stage_slot->G_row.data[0]);
                            for (int idx = warpgroup::warpid() * WARP_THREADS + lane_id;
                                 idx < static_cast<int>(sizeof(typename G::combo_row_stage_t));
                                 idx += WARPGROUP_WARPS * WARP_THREADS) {
                                combo_row_bytes[idx] = 0;
                            }
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                            if constexpr (combo_debug_force_row_ready_publish) {
                                if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                    __threadfence_block();
                                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                    arrive(combo_g_row_ready[0]);
                                }
                            }
                        }
                    }
                }
                {
                    subtile_rt D_fl;
                    warpgroup::load_async(D_fl, accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(CONSUMER_WG_SYNC_ID);

                    warp::mul(D_fl, D_fl, global_scale);

                    #pragma unroll
                    for (int i = 0; i < subtile_rt::height; ++i) {
                        const float lse_x = my_lse_x[i];
                        const float lse_y = my_lse_y[i];
                        #pragma unroll
                        for (int j = 0; j < subtile_rt::width; ++j) {
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const float lse_val = (k % 2 == 0) ? lse_x : lse_y;
                                D_fl.tiles[i][j].data[k].x = __expf(D_fl.tiles[i][j].data[k].x - lse_val);
                                D_fl.tiles[i][j].data[k].y = __expf(D_fl.tiles[i][j].data[k].y - lse_val);
                            }
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < subtile_rt::height; ++i) {
                        const int tgt_x = my_targets_x[i];
                        if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
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
                        if (tgt_y >= col_start && tgt_y < col_start + SUBTILE_COLS) {
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
                    for (int i = 0; i < subtile_rt::height; ++i) {
                        const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                        const int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                        #pragma unroll
                        for (int j = 0; j < subtile_rt::width; ++j) {
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
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
                    warp::copy(D_bf, D_fl);
                }

                if (epi == C::EPI_PIPE_DEPTH - 1 && !EFFECTIVE_ENABLE_STOREADD_COMBO) {
                    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                }

                if constexpr (C::OVERLAP_EPI) {
                    wait_for_quant_stage(bf_stage);
                }

                if constexpr (DO_ROW || DO_COL) {
                    if constexpr (G::COL_QUANT_FROM_DBF_REGS) {
                        static_assert(C::QUANTIZER_WARPGROUPS == 0,
                                      "consumer-owned direct OTF epilogue requires zero quantizer warpgroups");
                        static_assert(C::CONSUMER_DO_ROW,
                                      "consumer-owned direct OTF epilogue requires consumer-owned row quantization");
                        const int lane_row = lane_id / 4;
                        const int lane_pair = lane_id % 4;
                        const int local_warp_row_base = warpgroup::warpid() * (C::Mb / 8);

                        if constexpr (DO_ROW) {
                            #pragma unroll
                            for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                #pragma unroll
                                for (int row_half = 0; row_half < 2; ++row_half) {
                                    const bool writer_lane = (lane_row & 1) == 0;
                                    const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                    #pragma unroll
                                    for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                        const bf16_2 vals0 = D_bf.tiles[i][group16].data[row_half];
                                        const bf16_2 vals1 = D_bf.tiles[i][group16].data[row_half + 2];
                                        const uint32_t vals0_bits = bf16x2_bits(vals0);
                                        const uint32_t vals1_bits = bf16x2_bits(vals1);
                                        const uint32_t vals0_peer_bits =
                                            __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                        const uint32_t vals1_peer_bits =
                                            __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                        if constexpr (G::DIRECT_ROW_WARPBALANCED) {
                                            const uint64_t row_bits_lo =
                                                static_cast<uint64_t>(vals0_bits) |
                                                (static_cast<uint64_t>(vals1_bits) << 32);
                                            const uint64_t row_bits_hi =
                                                static_cast<uint64_t>(vals0_peer_bits) |
                                                (static_cast<uint64_t>(vals1_peer_bits) << 32);
                                            const bf16_2 row0_pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_lo));
                                            const bf16_2 row0_pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_lo >> 32));
                                            const bf16_2 row1_pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_hi));
                                            const bf16_2 row1_pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_hi >> 32));

                                            const float2 row0_pair_lo = make_float2(
                                                __bfloat162float(row0_pair_lo_bf.x),
                                                __bfloat162float(row0_pair_lo_bf.y));
                                            const float2 row0_pair_hi = make_float2(
                                                __bfloat162float(row0_pair_hi_bf.x),
                                                __bfloat162float(row0_pair_hi_bf.y));
                                            const float2 row1_pair_lo = make_float2(
                                                __bfloat162float(row1_pair_lo_bf.x),
                                                __bfloat162float(row1_pair_lo_bf.y));
                                            const float2 row1_pair_hi = make_float2(
                                                __bfloat162float(row1_pair_hi_bf.x),
                                                __bfloat162float(row1_pair_hi_bf.y));

                                            float row0_amax = 0.0f;
                                            row0_amax = fmaxf(row0_amax, fabsf(row0_pair_lo.x));
                                            row0_amax = fmaxf(row0_amax, fabsf(row0_pair_lo.y));
                                            row0_amax = fmaxf(row0_amax, fabsf(row0_pair_hi.x));
                                            row0_amax = fmaxf(row0_amax, fabsf(row0_pair_hi.y));
                                            row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 1, 4));
                                            row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 2, 4));

                                            float row1_amax = 0.0f;
                                            row1_amax = fmaxf(row1_amax, fabsf(row1_pair_lo.x));
                                            row1_amax = fmaxf(row1_amax, fabsf(row1_pair_lo.y));
                                            row1_amax = fmaxf(row1_amax, fabsf(row1_pair_hi.x));
                                            row1_amax = fmaxf(row1_amax, fabsf(row1_pair_hi.y));
                                            row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 1, 4));
                                            row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 2, 4));

                                            const float row0_rcp = (row0_amax > 0.0f) ? (FP4_MAX / row0_amax) : 0.0f;
                                            const float row1_rcp = (row1_amax > 0.0f) ? (FP4_MAX / row1_amax) : 0.0f;
                                            const unsigned row0_byte_lo = quantize_fp4_pair(
                                                row0_pair_lo.x, row0_pair_lo.y, row0_rcp);
                                            const unsigned row0_byte_hi = quantize_fp4_pair(
                                                row0_pair_hi.x, row0_pair_hi.y, row0_rcp);
                                            const unsigned row1_byte_lo = quantize_fp4_pair(
                                                row1_pair_lo.x, row1_pair_lo.y, row1_rcp);
                                            const unsigned row1_byte_hi = quantize_fp4_pair(
                                                row1_pair_hi.x, row1_pair_hi.y, row1_rcp);

                                            uint64_t packed_row0 = 0;
                                            uint64_t packed_row1 = 0;
                                            #pragma unroll
                                            for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                packed_row0 |= static_cast<uint64_t>(
                                                    __shfl_sync(0xffffffff, static_cast<int>(row0_byte_lo), src_pair, 4) & 0xff)
                                                    << (src_pair * 8);
                                                packed_row0 |= static_cast<uint64_t>(
                                                    __shfl_sync(0xffffffff, static_cast<int>(row0_byte_hi), src_pair, 4) & 0xff)
                                                    << ((src_pair + 4) * 8);
                                                packed_row1 |= static_cast<uint64_t>(
                                                    __shfl_sync(0xffffffff, static_cast<int>(row1_byte_lo), src_pair, 4) & 0xff)
                                                    << (src_pair * 8);
                                                packed_row1 |= static_cast<uint64_t>(
                                                    __shfl_sync(0xffffffff, static_cast<int>(row1_byte_hi), src_pair, 4) & 0xff)
                                                    << ((src_pair + 4) * 8);
                                            }

                                            if (writer_lane && lane_pair == 0) {
                                                const int local_row0 =
                                                    local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                                const int global_row0 = tile_row_base + local_row0;
                                                const int global_row1 = global_row0 + 1;
                                                const bool row0_in_bounds = full_tile_rows || global_row0 < g.M;
                                                const bool row1_in_bounds = full_tile_rows || global_row1 < g.M;
                                                const int global_col_16 = col_start + group16 * 16;
                                                const int fp4x2_col_base = global_col_16 / 2;
                                                const int kgroup = global_col_16 / 64;
                                                const int col_16_in_64 = (global_col_16 / 16) % 4;

                                                if (row0_in_bounds) {
                                                    store_global_u64(
                                                        &row_fp4_ptr[global_row0 * row_fp4_stride + fp4x2_col_base],
                                                        packed_row0);
                                                    const float row0_scale = row0_rcp > 0.0f ? (1.0f / row0_rcp) : 0.0f;
                                                    float stored_scale = row0_scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(row0_rcp * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int depth = global_row0 / 128;
                                                    const int sr = global_row0 % 32;
                                                    const int rr = (global_row0 / 32) % 4;
                                                    const int chunk = depth * row_sc_kgroups + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                                if (row1_in_bounds) {
                                                    store_global_u64(
                                                        &row_fp4_ptr[global_row1 * row_fp4_stride + fp4x2_col_base],
                                                        packed_row1);
                                                    const float row1_scale = row1_rcp > 0.0f ? (1.0f / row1_rcp) : 0.0f;
                                                    float stored_scale = row1_scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(row1_rcp * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int depth = global_row1 / 128;
                                                    const int sr = global_row1 % 32;
                                                    const int rr = (global_row1 / 32) % 4;
                                                    const int chunk = depth * row_sc_kgroups + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                            }
                                        } else if (writer_lane) {
                                                const uint64_t row_bits_lo =
                                                    static_cast<uint64_t>(vals0_bits) |
                                                    (static_cast<uint64_t>(vals1_bits) << 32);
                                                const uint64_t row_bits_hi =
                                                    static_cast<uint64_t>(vals0_peer_bits) |
                                                    (static_cast<uint64_t>(vals1_peer_bits) << 32);
                                                const unsigned writer_mask = __activemask();
                                                const int row_lane_base = (lane_id / 4) * 4;
                                                const uint64_t pair_bits_lo_0 = shfl_u64_sync(writer_mask, row_bits_lo, row_lane_base + 0);
                                                const uint64_t pair_bits_lo_1 = shfl_u64_sync(writer_mask, row_bits_lo, row_lane_base + 1);
                                                const uint64_t pair_bits_lo_2 = shfl_u64_sync(writer_mask, row_bits_lo, row_lane_base + 2);
                                                const uint64_t pair_bits_lo_3 = shfl_u64_sync(writer_mask, row_bits_lo, row_lane_base + 3);
                                                const uint64_t pair_bits_hi_0 = shfl_u64_sync(writer_mask, row_bits_hi, row_lane_base + 0);
                                                const uint64_t pair_bits_hi_1 = shfl_u64_sync(writer_mask, row_bits_hi, row_lane_base + 1);
                                                const uint64_t pair_bits_hi_2 = shfl_u64_sync(writer_mask, row_bits_hi, row_lane_base + 2);
                                                const uint64_t pair_bits_hi_3 = shfl_u64_sync(writer_mask, row_bits_hi, row_lane_base + 3);

                                                if (lane_pair == 0) {
                                                    const int local_row0 =
                                                        local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                                    const int global_row0 = tile_row_base + local_row0;
                                                    const int global_row1 = global_row0 + 1;
                                                    const bool row0_in_bounds = full_tile_rows || global_row0 < g.M;
                                                    const bool row1_in_bounds = full_tile_rows || global_row1 < g.M;

                                                    if (row0_in_bounds) {
                                                        quantize_rowrecord_pairbits_to_global(
                                                            pair_bits_lo_0, pair_bits_lo_1, pair_bits_lo_2, pair_bits_lo_3,
                                                            row_fp4_ptr, row_fp4_stride,
                                                            row_sc_ptr, row_sc_kgroups,
                                                            global_row0, col_start, group16,
                                                            g_sg_rcp, g_sg, encode_centric);
                                                    }
                                                    if (row1_in_bounds) {
                                                        quantize_rowrecord_pairbits_to_global(
                                                            pair_bits_hi_0, pair_bits_hi_1, pair_bits_hi_2, pair_bits_hi_3,
                                                            row_fp4_ptr, row_fp4_stride,
                                                            row_sc_ptr, row_sc_kgroups,
                                                            global_row1, col_start, group16,
                                                            g_sg_rcp, g_sg, encode_centric);
                                                    }
                                                }
                                        }
                                    }
                                }
                            }
                        }

                        if constexpr (DO_ROW && DO_COL) {
                            __syncwarp();
                        }

                        if constexpr (DO_COL) {
                            #pragma unroll
                            for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                const int local_row_base = local_warp_row_base + i * 16;
                                const int global_row_base = tile_row_base + local_row_base;
                                const int global_row_pair_base = global_row_base / 2;
                                #pragma unroll
                                for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                    const int col16 = lane_id & 15;
                                    const int col_in_epi = group16 * 16 + col16;
                                    const int global_col = col_start + col_in_epi;
                                    const int lane_pair_src = (col16 & 7) >> 1;
                                    const bool use_hi = col16 >= 8;
                                    const bool use_y = (col16 & 1) != 0;
                                    const uint32_t local_lo_bits_0 =
                                        bf16x2_bits(D_bf.tiles[i][group16].data[0]);
                                    const uint32_t local_hi_bits_0 =
                                        bf16x2_bits(D_bf.tiles[i][group16].data[2]);
                                    const uint32_t local_lo_bits_1 =
                                        bf16x2_bits(D_bf.tiles[i][group16].data[1]);
                                    const uint32_t local_hi_bits_1 =
                                        bf16x2_bits(D_bf.tiles[i][group16].data[3]);
                                    if constexpr (G::DIRECT_COL_WARPBALANCED) {
                                        const bool col_team_active = global_col < g.N;
                                        const bool lower_owner_lane = lane_id < 16;
                                        float partial_amax = 0.0f;
                                        {
                                            const uint32_t pass_lo_bits = local_lo_bits_0;
                                            const uint32_t pass_hi_bits = local_hi_bits_0;
                                            float lower_partial_amax = 0.0f;
                                            #pragma unroll
                                            for (int local_pair = 0; local_pair < 4; ++local_pair) {
                                                const int src_even = local_pair * 8 + lane_pair_src;
                                                const int src_odd = src_even + 4;
                                                const uint32_t even_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_even);
                                                const uint32_t odd_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_odd);
                                                const uint32_t even_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_even);
                                                const uint32_t odd_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_odd);
                                                const uint32_t even_bits = use_hi ? even_hi_bits : even_lo_bits;
                                                const uint32_t odd_bits = use_hi ? odd_hi_bits : odd_lo_bits;
                                                const uint32_t pair_bits = use_y
                                                    ? ((even_bits >> 16) | (odd_bits & 0xffff0000u))
                                                    : ((even_bits & 0x0000ffffu) | (odd_bits << 16));
                                                const bf16_2 bf_pair = bf16x2_from_bits(pair_bits);
                                                const int global_row = global_row_base + local_pair * 2;
                                                const float v0 =
                                                    (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 =
                                                    (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                if (col_team_active && lower_owner_lane) {
                                                    lower_partial_amax = fmaxf(lower_partial_amax, fabsf(v0));
                                                    lower_partial_amax = fmaxf(lower_partial_amax, fabsf(v1));
                                                }
                                            }
                                            if (lower_owner_lane) {
                                                partial_amax = lower_partial_amax;
                                            }
                                        }
                                        {
                                            const uint32_t pass_lo_bits = local_lo_bits_1;
                                            const uint32_t pass_hi_bits = local_hi_bits_1;
                                            float upper_partial_amax = 0.0f;
                                            #pragma unroll
                                            for (int local_pair = 0; local_pair < 4; ++local_pair) {
                                                const int src_even = local_pair * 8 + lane_pair_src;
                                                const int src_odd = src_even + 4;
                                                const uint32_t even_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_even);
                                                const uint32_t odd_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_odd);
                                                const uint32_t even_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_even);
                                                const uint32_t odd_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_odd);
                                                const uint32_t even_bits = use_hi ? even_hi_bits : even_lo_bits;
                                                const uint32_t odd_bits = use_hi ? odd_hi_bits : odd_lo_bits;
                                                const uint32_t pair_bits = use_y
                                                    ? ((even_bits >> 16) | (odd_bits & 0xffff0000u))
                                                    : ((even_bits & 0x0000ffffu) | (odd_bits << 16));
                                                const bf16_2 bf_pair = bf16x2_from_bits(pair_bits);
                                                const int global_row = global_row_base + (4 + local_pair) * 2;
                                                const float v0 =
                                                    (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 =
                                                    (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                if (col_team_active && !lower_owner_lane) {
                                                    upper_partial_amax = fmaxf(upper_partial_amax, fabsf(v0));
                                                    upper_partial_amax = fmaxf(upper_partial_amax, fabsf(v1));
                                                }
                                            }
                                            if (!lower_owner_lane) {
                                                partial_amax = upper_partial_amax;
                                            }
                                        }

                                        float col_amax =
                                            fmaxf(partial_amax, __shfl_xor_sync(0xffffffff, partial_amax, 16));
                                        const float col_scale = col_amax * (1.0f / FP4_MAX);
                                        const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                        uint32_t local_packed_fp4 = 0;
                                        {
                                            const uint32_t pass_lo_bits = local_lo_bits_0;
                                            const uint32_t pass_hi_bits = local_hi_bits_0;
                                            uint32_t lower_packed_fp4 = 0;
                                            #pragma unroll
                                            for (int local_pair = 0; local_pair < 4; ++local_pair) {
                                                const int src_even = local_pair * 8 + lane_pair_src;
                                                const int src_odd = src_even + 4;
                                                const uint32_t even_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_even);
                                                const uint32_t odd_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_odd);
                                                const uint32_t even_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_even);
                                                const uint32_t odd_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_odd);
                                                const uint32_t even_bits = use_hi ? even_hi_bits : even_lo_bits;
                                                const uint32_t odd_bits = use_hi ? odd_hi_bits : odd_lo_bits;
                                                const uint32_t pair_bits = use_y
                                                    ? ((even_bits >> 16) | (odd_bits & 0xffff0000u))
                                                    : ((even_bits & 0x0000ffffu) | (odd_bits << 16));
                                                const bf16_2 bf_pair = bf16x2_from_bits(pair_bits);
                                                const int global_row = global_row_base + local_pair * 2;
                                                const float v0 =
                                                    (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 =
                                                    (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                if (col_team_active && lower_owner_lane) {
                                                    lower_packed_fp4 |= static_cast<uint32_t>(
                                                                            quantize_fp4_pair(v0, v1, col_rcp))
                                                                        << (local_pair * 8);
                                                }
                                            }
                                            if (lower_owner_lane) {
                                                local_packed_fp4 = lower_packed_fp4;
                                            }
                                        }
                                        {
                                            const uint32_t pass_lo_bits = local_lo_bits_1;
                                            const uint32_t pass_hi_bits = local_hi_bits_1;
                                            uint32_t upper_packed_fp4_local = 0;
                                            #pragma unroll
                                            for (int local_pair = 0; local_pair < 4; ++local_pair) {
                                                const int src_even = local_pair * 8 + lane_pair_src;
                                                const int src_odd = src_even + 4;
                                                const uint32_t even_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_even);
                                                const uint32_t odd_lo_bits =
                                                    __shfl_sync(0xffffffff, pass_lo_bits, src_odd);
                                                const uint32_t even_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_even);
                                                const uint32_t odd_hi_bits =
                                                    __shfl_sync(0xffffffff, pass_hi_bits, src_odd);
                                                const uint32_t even_bits = use_hi ? even_hi_bits : even_lo_bits;
                                                const uint32_t odd_bits = use_hi ? odd_hi_bits : odd_lo_bits;
                                                const uint32_t pair_bits = use_y
                                                    ? ((even_bits >> 16) | (odd_bits & 0xffff0000u))
                                                    : ((even_bits & 0x0000ffffu) | (odd_bits << 16));
                                                const bf16_2 bf_pair = bf16x2_from_bits(pair_bits);
                                                const int global_row = global_row_base + (4 + local_pair) * 2;
                                                const float v0 =
                                                    (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 =
                                                    (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                if (col_team_active && !lower_owner_lane) {
                                                    upper_packed_fp4_local |= static_cast<uint32_t>(
                                                                                    quantize_fp4_pair(v0, v1, col_rcp))
                                                                                << (local_pair * 8);
                                                }
                                            }
                                            if (!lower_owner_lane) {
                                                local_packed_fp4 = upper_packed_fp4_local;
                                            }
                                        }
                                        const uint32_t upper_packed_fp4 =
                                            __shfl_xor_sync(0xffffffff, static_cast<int>(local_packed_fp4), 16);
                                        if (col_team_active && lower_owner_lane) {
                                            const uint64_t packed_fp4 =
                                                static_cast<uint64_t>(local_packed_fp4) |
                                                (static_cast<uint64_t>(upper_packed_fp4) << 32);
                                            store_global_u64(
                                                col_fp4_ptr + global_col * col_fp4_stride + global_row_pair_base,
                                                packed_fp4);

                                            float stored_scale = col_scale * g_sg_rcp;
                                            if (encode_centric) {
                                                stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                            }
                                            const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                            const int depth = global_col / 128;
                                            const int sr = global_col % 32;
                                            const int rr = (global_col / 32) % 4;
                                            const int m_kgroup = global_row_base / 64;
                                            const int m_16_in_64 = (global_row_base / 16) % 4;
                                            const int chunk = depth * col_sc_kgroups + m_kgroup;
                                            const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                            col_sc_ptr[chunk * 512 + byte_idx] =
                                                *reinterpret_cast<const uint8_t*>(&csc);
                                        }
                                    } else {
                                        const bool col_owner = (lane_id < 16) && (global_col < g.N);
                                        float col_amax = 0.0f;
                                        uint64_t packed_fp4 = 0;
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const int row_half = pair >> 2;
                                            const int row_pair = pair & 3;
                                            const uint32_t local_lo_bits = row_half == 0 ? local_lo_bits_0 : local_lo_bits_1;
                                            const uint32_t local_hi_bits = row_half == 0 ? local_hi_bits_0 : local_hi_bits_1;
                                            const int src_even = row_pair * 8 + lane_pair_src;
                                            const int src_odd = src_even + 4;
                                            const uint32_t even_lo_bits = __shfl_sync(0xffffffff, local_lo_bits, src_even);
                                            const uint32_t odd_lo_bits = __shfl_sync(0xffffffff, local_lo_bits, src_odd);
                                            const uint32_t even_hi_bits = __shfl_sync(0xffffffff, local_hi_bits, src_even);
                                            const uint32_t odd_hi_bits = __shfl_sync(0xffffffff, local_hi_bits, src_odd);
                                            if (col_owner) {
                                                const uint32_t even_bits = use_hi ? even_hi_bits : even_lo_bits;
                                                const uint32_t odd_bits = use_hi ? odd_hi_bits : odd_lo_bits;
                                                const uint32_t pair_bits = use_y
                                                    ? ((even_bits >> 16) | (odd_bits & 0xffff0000u))
                                                    : ((even_bits & 0x0000ffffu) | (odd_bits << 16));
                                                const bf16_2 bf_pair = bf16x2_from_bits(pair_bits);
                                                const int global_row = global_row_base + pair * 2;
                                                const float v0 = (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 = (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                col_amax = fmaxf(col_amax, fabsf(v0));
                                                col_amax = fmaxf(col_amax, fabsf(v1));
                                            }
                                        }

                                        const float col_scale = col_amax * (1.0f / FP4_MAX);
                                        const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const int row_half = pair >> 2;
                                            const int row_pair = pair & 3;
                                            const uint32_t local_lo_bits = row_half == 0 ? local_lo_bits_0 : local_lo_bits_1;
                                            const uint32_t local_hi_bits = row_half == 0 ? local_hi_bits_0 : local_hi_bits_1;
                                            const int src_even = row_pair * 8 + lane_pair_src;
                                            const int src_odd = src_even + 4;
                                            const uint32_t even_lo_bits = __shfl_sync(0xffffffff, local_lo_bits, src_even);
                                            const uint32_t odd_lo_bits = __shfl_sync(0xffffffff, local_lo_bits, src_odd);
                                            const uint32_t even_hi_bits = __shfl_sync(0xffffffff, local_hi_bits, src_even);
                                            const uint32_t odd_hi_bits = __shfl_sync(0xffffffff, local_hi_bits, src_odd);
                                            if (col_owner) {
                                                const uint32_t even_bits = use_hi ? even_hi_bits : even_lo_bits;
                                                const uint32_t odd_bits = use_hi ? odd_hi_bits : odd_lo_bits;
                                                const uint32_t pair_bits = use_y
                                                    ? ((even_bits >> 16) | (odd_bits & 0xffff0000u))
                                                    : ((even_bits & 0x0000ffffu) | (odd_bits << 16));
                                                const bf16_2 bf_pair = bf16x2_from_bits(pair_bits);
                                                const int global_row = global_row_base + pair * 2;
                                                const float v0 = (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 = (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(v0, v1, col_rcp))
                                                              << (pair * 8);
                                            }
                                        }
                                        if (col_owner) {
                                            store_global_u64(col_fp4_ptr + global_col * col_fp4_stride + global_row_pair_base, packed_fp4);

                                            float stored_scale = col_scale * g_sg_rcp;
                                            if (encode_centric) {
                                                stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                            }
                                            const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                            const int depth = global_col / 128;
                                            const int sr = global_col % 32;
                                            const int rr = (global_col / 32) % 4;
                                            const int m_kgroup = global_row_base / 64;
                                            const int m_16_in_64 = (global_row_base / 16) % 4;
                                            const int chunk = depth * col_sc_kgroups + m_kgroup;
                                            const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                            col_sc_ptr[chunk * 512 + byte_idx] =
                                                *reinterpret_cast<const uint8_t*>(&csc);
                                        }
                                    }
                                }
                            }
                        }

                        warpgroup::sync(1);
                    } else if constexpr (G::USE_COL_PAIR_STAGE &&
                                         (DO_COL || C::CONSUMER_DO_ROW || G::ROW_QUANT_FROM_COL_PAIR_STAGE || C::ROW_QUANT_FROM_REGS)) {
                        static_assert(
                            C::CONSUMER_DO_ROW ||
                                (G::USE_ROW_PAIR_STAGE &&
                                 G::ROW_PAIR_STAGE_ROWRECORD &&
                                 G::ROW_PAIR_STAGE_FLOATCACHE &&
                                 G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT),
                            "col-pair mailbox path requires consumer-owned row quantization or lane-pair rowrecord row-quantizer support");
                        const int lane_row = lane_id / 4;
                        const int lane_pair = lane_id % 4;
                        const int local_warp_row_base = warpgroup::warpid() * (C::Mb / 8);

                        if constexpr (C::ROW_QUANT_FROM_REGS && C::EARLY_COL_READY) {
                            if constexpr (DO_COL) {
                                #pragma unroll
                                for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                    const int row16_block = (local_warp_row_base + i * 16) / 16;
                                    #pragma unroll
                                    for (int row_half = 0; row_half < 2; ++row_half) {
                                        const int pair_slot = row_half * 4 + lane_row / 2;
                                        const bool writer_lane = (lane_row & 1) == 0;
                                        const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                        #pragma unroll
                                        for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                            const bf16_2 vals0 = D_bf.tiles[i][group16].data[row_half];
                                            const bf16_2 vals1 = D_bf.tiles[i][group16].data[row_half + 2];
                                            const int col_base = group16 * 16 + lane_pair * 2;
                                            const int col_pair_base = col_base / 2;
                                            const uint32_t vals0_bits = bf16x2_bits(vals0);
                                            const uint32_t vals0_peer_bits =
                                                __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 0,
                                                    pair_slot,
                                                    (vals0_bits & 0x0000ffffu) | (vals0_peer_bits << 16),
                                                    (vals0_bits >> 16) | (vals0_peer_bits & 0xffff0000u));
                                            }
                                            const uint32_t vals1_bits = bf16x2_bits(vals1);
                                            const uint32_t vals1_peer_bits =
                                                __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                            if constexpr (G::USE_ROW_RCP_STAGE && DO_ROW) {
                                                const bf16_2 vals0_peer = bf16x2_from_bits(vals0_peer_bits);
                                                const bf16_2 vals1_peer = bf16x2_from_bits(vals1_peer_bits);
                                                float row0_amax = fmaxf(
                                                    fmaxf(fabsf(__bfloat162float(vals0.x)), fabsf(__bfloat162float(vals0.y))),
                                                    fmaxf(fabsf(__bfloat162float(vals1.x)), fabsf(__bfloat162float(vals1.y))));
                                                float row1_amax = fmaxf(
                                                    fmaxf(fabsf(__bfloat162float(vals0_peer.x)), fabsf(__bfloat162float(vals0_peer.y))),
                                                    fmaxf(fabsf(__bfloat162float(vals1_peer.x)), fabsf(__bfloat162float(vals1_peer.y))));
                                                row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 1, 4));
                                                row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 2, 4));
                                                row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 1, 4));
                                                row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 2, 4));
                                                if (writer_lane && lane_pair == 0) {
                                                    const int local_row0 = local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                                    const int global_row0 = tile_row_base + local_row0;
                                                    const int global_row1 = global_row0 + 1;
                                                    const bool row0_in_bounds = full_tile_rows || global_row0 < g.M;
                                                    const bool row1_in_bounds = full_tile_rows || global_row1 < g.M;
                                                    const float row0_rcp = row0_amax > 0.0f ? (FP4_MAX / row0_amax) : 0.0f;
                                                    const float row1_rcp = row1_amax > 0.0f ? (FP4_MAX / row1_amax) : 0.0f;
                                                    row_rcp_stage[bf_stage].rcp[row16_block][group16][pair_slot][0] = row0_in_bounds ? row0_rcp : 0.0f;
                                                    row_rcp_stage[bf_stage].rcp[row16_block][group16][pair_slot][1] = row1_in_bounds ? row1_rcp : 0.0f;
                                                }
                                            }
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 4,
                                                    pair_slot,
                                                    (vals1_bits & 0x0000ffffu) | (vals1_peer_bits << 16),
                                                    (vals1_bits >> 16) | (vals1_peer_bits & 0xffff0000u));
                                            }
                                        }
                                    }
                                }
                                warpgroup::sync(1);
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                warpgroup::sync(1);
                                if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                    arrive(slice_col_ready[bf_stage]);
                                }
                                update_phasebit<1>(slice_col_recycle_phasebits, bf_stage);
                            }

                            if constexpr (DO_ROW && C::CONSUMER_DO_ROW) {
                                #pragma unroll
                                for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                    #pragma unroll
                                    for (int row_half = 0; row_half < 2; ++row_half) {
                                        const int local_row = local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                        const int global_row = tile_row_base + local_row;
                                        #pragma unroll
                                        for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                            const bf16_2 vals0 = D_bf.tiles[i][group16].data[row_half];
                                            const bf16_2 vals1 = D_bf.tiles[i][group16].data[row_half + 2];
                                            if constexpr (G::ROW_QUANT_ROWLEADER) {
                                                const uint32_t vals0_bits = bf16x2_bits(vals0);
                                                const uint32_t vals1_bits = bf16x2_bits(vals1);
                                                const int row_lane_base = lane_row << 2;
                                                uint32_t gathered_vals0[4];
                                                uint32_t gathered_vals1[4];
                                                #pragma unroll
                                                for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                    gathered_vals0[src_pair] =
                                                        __shfl_sync(0xffffffff, vals0_bits, row_lane_base + src_pair);
                                                    gathered_vals1[src_pair] =
                                                        __shfl_sync(0xffffffff, vals1_bits, row_lane_base + src_pair);
                                                }
                                                if (lane_pair == 0 && global_row < g.M) {
                                                    float amax = 0.0f;
                                                    #pragma unroll
                                                    for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                        const bf16_2 gathered0 = bf16x2_from_bits(gathered_vals0[src_pair]);
                                                        const bf16_2 gathered1 = bf16x2_from_bits(gathered_vals1[src_pair]);
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered0.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered0.y)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered1.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered1.y)));
                                                    }

                                                    const float scale = amax * (1.0f / FP4_MAX);
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;
                                                    #pragma unroll
                                                    for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                        const bf16_2 gathered0 = bf16x2_from_bits(gathered_vals0[src_pair]);
                                                        const bf16_2 gathered1 = bf16x2_from_bits(gathered_vals1[src_pair]);
                                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + src_pair] =
                                                            quantize_fp4_pair(
                                                                __bfloat162float(gathered0.x),
                                                                __bfloat162float(gathered0.y),
                                                                rcp_scale);
                                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + 4 + src_pair] =
                                                            quantize_fp4_pair(
                                                                __bfloat162float(gathered1.x),
                                                                __bfloat162float(gathered1.y),
                                                                rcp_scale);
                                                    }

                                                    float stored_scale = scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int kgroup = global_col_16 / 64;
                                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                    const int depth = global_row / 128;
                                                    const int sr = global_row % 32;
                                                    const int rr = (global_row / 32) % 4;
                                                    const int chunk = depth * row_sc_kgroups + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                            } else if constexpr (G::ROW_QUANT_ROWDUAL) {
                                                const uint32_t vals0_bits = bf16x2_bits(vals0);
                                                const uint32_t vals1_bits = bf16x2_bits(vals1);
                                                const int row_lane_base = lane_row << 2;
                                                const uint32_t owner_bits = ((lane_pair & 1) == 0) ? vals0_bits : vals1_bits;
                                                uint32_t gathered_bits[4];
                                                float half_amax = 0.0f;
                                                #pragma unroll
                                                for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                    const uint32_t bits =
                                                        __shfl_sync(0xffffffff, owner_bits, row_lane_base + src_pair);
                                                    gathered_bits[src_pair] = bits;
                                                    const bf16_2 gathered = bf16x2_from_bits(bits);
                                                    half_amax = fmaxf(half_amax, fabsf(__bfloat162float(gathered.x)));
                                                    half_amax = fmaxf(half_amax, fabsf(__bfloat162float(gathered.y)));
                                                }

                                                if (global_row < g.M && lane_pair < 2) {
                                                    const float other_half_amax =
                                                        __shfl_sync(0xffffffff, half_amax, row_lane_base + (lane_pair ^ 1));
                                                    const float amax = fmaxf(half_amax, other_half_amax);
                                                    const float scale = amax * (1.0f / FP4_MAX);
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2 + lane_pair * 4;
                                                    #pragma unroll
                                                    for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                        const bf16_2 gathered = bf16x2_from_bits(gathered_bits[src_pair]);
                                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + src_pair] =
                                                            quantize_fp4_pair(
                                                                __bfloat162float(gathered.x),
                                                                __bfloat162float(gathered.y),
                                                                rcp_scale);
                                                    }

                                                    if (lane_pair == 0) {
                                                        float stored_scale = scale * g_sg_rcp;
                                                        if (encode_centric) {
                                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                        }
                                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                        const int kgroup = global_col_16 / 64;
                                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                        const int depth = global_row / 128;
                                                        const int sr = global_row % 32;
                                                        const int rr = (global_row / 32) % 4;
                                                        const int chunk = depth * row_sc_kgroups + kgroup;
                                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                        row_sc_ptr[chunk * 512 + byte_idx] =
                                                            *reinterpret_cast<const uint8_t*>(&sc);
                                                    }
                                                }
                                            } else {
                                                const float v00 = __bfloat162float(vals0.x);
                                                const float v01 = __bfloat162float(vals0.y);
                                                const float v10 = __bfloat162float(vals1.x);
                                                const float v11 = __bfloat162float(vals1.y);
                                                const int depth = global_row / 128;
                                                const int sr = global_row % 32;
                                                const int rr = (global_row / 32) % 4;
                                                const int row_chunk_base = depth * row_sc_kgroups;

                                                if (full_tile_rows || global_row < g.M) {
                                                    float amax = fmaxf(fmaxf(fabsf(v00), fabsf(v01)),
                                                                       fmaxf(fabsf(v10), fabsf(v11)));
                                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
                                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));

                                                    const float scale = amax * (1.0f / FP4_MAX);
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;
                                                    row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + lane_pair] =
                                                        quantize_fp4_pair(v00, v01, rcp_scale);
                                                    row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + 4 + lane_pair] =
                                                        quantize_fp4_pair(v10, v11, rcp_scale);

                                                    if (lane_pair == 0) {
                                                        float stored_scale = scale * g_sg_rcp;
                                                        if (encode_centric) {
                                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                        }
                                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                        const int kgroup = global_col_16 / 64;
                                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                        const int chunk = row_chunk_base + kgroup;
                                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                        row_sc_ptr[chunk * 512 + byte_idx] =
                                                            *reinterpret_cast<const uint8_t*>(&sc);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else if constexpr (C::ROW_QUANT_FROM_REGS) {
                            #pragma unroll
                            for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                const int row16_block = (local_warp_row_base + i * 16) / 16;
                                #pragma unroll
                                for (int row_half = 0; row_half < 2; ++row_half) {
                                    const int local_row = local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                    const int global_row = tile_row_base + local_row;
                                    const int pair_slot = row_half * 4 + lane_row / 2;
                                    const bool writer_lane = (lane_row & 1) == 0;
                                    const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                    #pragma unroll
                                    for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                        const bf16_2 vals0 = D_bf.tiles[i][group16].data[row_half];
                                        const bf16_2 vals1 = D_bf.tiles[i][group16].data[row_half + 2];
                                        const int col_base = group16 * 16 + lane_pair * 2;
                                        const int col_pair_base = col_base / 2;

                                        if constexpr (DO_COL) {
                                            const uint32_t vals0_bits = bf16x2_bits(vals0);
                                            const uint32_t vals0_peer_bits =
                                                __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 0,
                                                    pair_slot,
                                                    (vals0_bits & 0x0000ffffu) | (vals0_peer_bits << 16),
                                                    (vals0_bits >> 16) | (vals0_peer_bits & 0xffff0000u));
                                            }
                                            const uint32_t vals1_bits = bf16x2_bits(vals1);
                                            const uint32_t vals1_peer_bits =
                                                __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 4,
                                                    pair_slot,
                                                    (vals1_bits & 0x0000ffffu) | (vals1_peer_bits << 16),
                                                    (vals1_bits >> 16) | (vals1_peer_bits & 0xffff0000u));
                                            }
                                        }

                                        if constexpr (DO_ROW) {
                                            if constexpr (G::ROW_QUANT_ROWLEADER) {
                                                const uint32_t vals0_bits = bf16x2_bits(vals0);
                                                const uint32_t vals1_bits = bf16x2_bits(vals1);
                                                const int row_lane_base = lane_row << 2;
                                                uint32_t gathered_vals0[4];
                                                uint32_t gathered_vals1[4];
                                                #pragma unroll
                                                for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                    gathered_vals0[src_pair] =
                                                        __shfl_sync(0xffffffff, vals0_bits, row_lane_base + src_pair);
                                                    gathered_vals1[src_pair] =
                                                        __shfl_sync(0xffffffff, vals1_bits, row_lane_base + src_pair);
                                                }
                                                if (lane_pair == 0 && global_row < g.M) {
                                                    float amax = 0.0f;
                                                    #pragma unroll
                                                    for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                        const bf16_2 gathered0 = bf16x2_from_bits(gathered_vals0[src_pair]);
                                                        const bf16_2 gathered1 = bf16x2_from_bits(gathered_vals1[src_pair]);
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered0.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered0.y)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered1.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(gathered1.y)));
                                                    }

                                                    const float scale = amax * (1.0f / FP4_MAX);
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;
                                                    #pragma unroll
                                                    for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                        const bf16_2 gathered0 = bf16x2_from_bits(gathered_vals0[src_pair]);
                                                        const bf16_2 gathered1 = bf16x2_from_bits(gathered_vals1[src_pair]);
                                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + src_pair] =
                                                            quantize_fp4_pair(
                                                                __bfloat162float(gathered0.x),
                                                                __bfloat162float(gathered0.y),
                                                                rcp_scale);
                                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + 4 + src_pair] =
                                                            quantize_fp4_pair(
                                                                __bfloat162float(gathered1.x),
                                                                __bfloat162float(gathered1.y),
                                                                rcp_scale);
                                                    }

                                                    float stored_scale = scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int kgroup = global_col_16 / 64;
                                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                    const int depth = global_row / 128;
                                                    const int sr = global_row % 32;
                                                    const int rr = (global_row / 32) % 4;
                                                    const int chunk = depth * row_sc_kgroups + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                            } else if constexpr (G::ROW_QUANT_ROWDUAL) {
                                                const uint32_t vals0_bits = bf16x2_bits(vals0);
                                                const uint32_t vals1_bits = bf16x2_bits(vals1);
                                                const int row_lane_base = lane_row << 2;
                                                const uint32_t owner_bits = ((lane_pair & 1) == 0) ? vals0_bits : vals1_bits;
                                                uint32_t gathered_bits[4];
                                                float half_amax = 0.0f;
                                                #pragma unroll
                                                for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                    const uint32_t bits =
                                                        __shfl_sync(0xffffffff, owner_bits, row_lane_base + src_pair);
                                                    gathered_bits[src_pair] = bits;
                                                    const bf16_2 gathered = bf16x2_from_bits(bits);
                                                    half_amax = fmaxf(half_amax, fabsf(__bfloat162float(gathered.x)));
                                                    half_amax = fmaxf(half_amax, fabsf(__bfloat162float(gathered.y)));
                                                }

                                                if (global_row < g.M && lane_pair < 2) {
                                                    const float other_half_amax =
                                                        __shfl_sync(0xffffffff, half_amax, row_lane_base + (lane_pair ^ 1));
                                                    const float amax = fmaxf(half_amax, other_half_amax);
                                                    const float scale = amax * (1.0f / FP4_MAX);
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2 + lane_pair * 4;
                                                    #pragma unroll
                                                    for (int src_pair = 0; src_pair < 4; ++src_pair) {
                                                        const bf16_2 gathered = bf16x2_from_bits(gathered_bits[src_pair]);
                                                        row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + src_pair] =
                                                            quantize_fp4_pair(
                                                                __bfloat162float(gathered.x),
                                                                __bfloat162float(gathered.y),
                                                                rcp_scale);
                                                    }

                                                    if (lane_pair == 0) {
                                                        float stored_scale = scale * g_sg_rcp;
                                                        if (encode_centric) {
                                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                        }
                                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                        const int kgroup = global_col_16 / 64;
                                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                        const int depth = global_row / 128;
                                                        const int sr = global_row % 32;
                                                        const int rr = (global_row / 32) % 4;
                                                        const int chunk = depth * row_sc_kgroups + kgroup;
                                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                        row_sc_ptr[chunk * 512 + byte_idx] =
                                                            *reinterpret_cast<const uint8_t*>(&sc);
                                                    }
                                                }
                                            } else {
                                                const float v00 = __bfloat162float(vals0.x);
                                                const float v01 = __bfloat162float(vals0.y);
                                                const float v10 = __bfloat162float(vals1.x);
                                                const float v11 = __bfloat162float(vals1.y);
                                                const int depth = global_row / 128;
                                                const int sr = global_row % 32;
                                                const int rr = (global_row / 32) % 4;
                                                const int row_chunk_base = depth * row_sc_kgroups;

                                                if (full_tile_rows || global_row < g.M) {
                                                    float amax = fmaxf(fmaxf(fabsf(v00), fabsf(v01)),
                                                                       fmaxf(fabsf(v10), fabsf(v11)));
                                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
                                                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));

                                                    const float scale = amax * (1.0f / FP4_MAX);
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;
                                                    row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + lane_pair] =
                                                        quantize_fp4_pair(v00, v01, rcp_scale);
                                                    row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + 4 + lane_pair] =
                                                        quantize_fp4_pair(v10, v11, rcp_scale);

                                                    if (lane_pair == 0) {
                                                        float stored_scale = scale * g_sg_rcp;
                                                        if (encode_centric) {
                                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                        }
                                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                        const int kgroup = global_col_16 / 64;
                                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                        const int chunk = row_chunk_base + kgroup;
                                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                        row_sc_ptr[chunk * 512 + byte_idx] =
                                                            *reinterpret_cast<const uint8_t*>(&sc);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if constexpr (DO_COL) {
                                #pragma unroll
                                for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                    const int row16_block = (local_warp_row_base + i * 16) / 16;
                                    #pragma unroll
                                    for (int row_half = 0; row_half < 2; ++row_half) {
                                        const int pair_slot = row_half * 4 + lane_row / 2;
                                        #pragma unroll
                                        for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                            const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                            const bool writer_lane = (lane_row & 1) == 0;
                                            const int col_base = group16 * 16 + lane_pair * 2;
                                            const int col_pair_base = col_base / 2;
                                            const bf16_2 vals0 =
                                                D_bf.tiles[i][group16].data[row_half];
                                            const bf16_2 vals1 =
                                                D_bf.tiles[i][group16].data[row_half + 2];
                                            const uint32_t vals0_bits = bf16x2_bits(vals0);
                                            const uint32_t vals0_peer_bits =
                                                __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 0,
                                                    pair_slot,
                                                    (vals0_bits & 0x0000ffffu) | (vals0_peer_bits << 16),
                                                    (vals0_bits >> 16) | (vals0_peer_bits & 0xffff0000u));
                                            }
                                            const uint32_t vals1_bits = bf16x2_bits(vals1);
                                            const uint32_t vals1_peer_bits =
                                                __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 4,
                                                    pair_slot,
                                                    (vals1_bits & 0x0000ffffu) | (vals1_peer_bits << 16),
                                                    (vals1_bits >> 16) | (vals1_peer_bits & 0xffff0000u));
                                            }
                                            if constexpr (G::USE_ROW_PAIR_STAGE && DO_ROW) {
                                                if constexpr (G::USE_ROW_RCP_STAGE) {
                                                    const bf16_2 vals0_peer = bf16x2_from_bits(vals0_peer_bits);
                                                    const bf16_2 vals1_peer = bf16x2_from_bits(vals1_peer_bits);
                                                    float row0_amax = fmaxf(
                                                        fmaxf(fabsf(__bfloat162float(vals0.x)), fabsf(__bfloat162float(vals0.y))),
                                                        fmaxf(fabsf(__bfloat162float(vals1.x)), fabsf(__bfloat162float(vals1.y))));
                                                    float row1_amax = fmaxf(
                                                        fmaxf(fabsf(__bfloat162float(vals0_peer.x)), fabsf(__bfloat162float(vals0_peer.y))),
                                                        fmaxf(fabsf(__bfloat162float(vals1_peer.x)), fabsf(__bfloat162float(vals1_peer.y))));
                                                    row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 1, 4));
                                                    row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 2, 4));
                                                    row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 1, 4));
                                                    row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 2, 4));
                                                    if (writer_lane && lane_pair == 0) {
                                                        const int local_row0 = local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                                        const int global_row0 = tile_row_base + local_row0;
                                                        const int global_row1 = global_row0 + 1;
                                                        const bool row0_in_bounds = full_tile_rows || global_row0 < g.M;
                                                        const bool row1_in_bounds = full_tile_rows || global_row1 < g.M;
                                                        const float row0_rcp = row0_amax > 0.0f ? (FP4_MAX / row0_amax) : 0.0f;
                                                        const float row1_rcp = row1_amax > 0.0f ? (FP4_MAX / row1_amax) : 0.0f;
                                                        row_rcp_stage[bf_stage].rcp[row16_block][group16][pair_slot][0] = row0_in_bounds ? row0_rcp : 0.0f;
                                                        row_rcp_stage[bf_stage].rcp[row16_block][group16][pair_slot][1] = row1_in_bounds ? row1_rcp : 0.0f;
                                                    }
                                                }
                                                if (writer_lane) {
                                                    const uint64_t row_bits_lo =
                                                        static_cast<uint64_t>(vals0_bits) |
                                                        (static_cast<uint64_t>(vals1_bits) << 32);
                                                    const uint64_t row_bits_hi =
                                                        static_cast<uint64_t>(vals0_peer_bits) |
                                                        (static_cast<uint64_t>(vals1_peer_bits) << 32);
                                                    store_row_pair_stage_record<G>(
                                                        row_pair_stage,
                                                        bf_stage,
                                                        row16_block,
                                                        group16,
                                                        pair_slot,
                                                        lane_pair,
                                                        row_bits_lo,
                                                        row_bits_hi);
                                                }
                                            }
                                        }
                                    }
                                    if constexpr (C::EARLY_COL_READY &&
                                                  G::COL_READY_PER_ROW16 &&
                                                  G::ROW_QUANT_IMMEDIATE_PER_ROW16 &&
                                                  DO_ROW &&
                                                  G::USE_ROW_PAIR_STAGE &&
                                                  G::ROW_PAIR_STAGE_ROWRECORD &&
                                                  G::ROW_PAIR_STAGE_FLOATCACHE &&
                                                  G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT) {
                                        __syncwarp();
                                        __threadfence_block();
                                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                        if (warp::laneid() == 0) {
                                            arrive(slice_col_ready_row16[bf_stage][row16_block]);
                                        }
                                    }
                                }
                                if constexpr (C::EARLY_COL_READY &&
                                              G::COL_READY_PER_ROW16 &&
                                              G::ROW_QUANT_IMMEDIATE_PER_ROW16 &&
                                              DO_ROW &&
                                              G::USE_ROW_PAIR_STAGE &&
                                              G::ROW_PAIR_STAGE_ROWRECORD &&
                                              G::ROW_PAIR_STAGE_FLOATCACHE &&
                                              G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT) {
                                    __syncwarp();
                                    #pragma unroll
                                    for (int local_i = 0; local_i < subtile_rt_bf::height; ++local_i) {
                                        const int row16_block = (local_warp_row_base + local_i * 16) / 16;
                                        const bool row16_lane_active = ((lane_id >> 4) == local_i);
                                        if (row16_lane_active) {
                                            const int quant_row = local_warp_row_base + local_i * 16 + (lane_id & 15);
                                            const int global_row = tile_row_base + quant_row;
                                            const int depth = global_row / 128;
                                            const int sr = global_row % 32;
                                            const int rr = (global_row / 32) % 4;
                                            const int row_chunk_base = depth * row_sc_kgroups;
                                            const bool row_in_bounds = full_tile_rows || global_row < g.M;
                                            const int row_pair_slot = (lane_id & 15) / 2;
                                            const int row_pair_lane = lane_id & 1;
                                            #pragma unroll
                                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                                const uint64_t pair_bits_0 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][0][row_pair_lane];
                                                const uint64_t pair_bits_1 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][1][row_pair_lane];
                                                const uint64_t pair_bits_2 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][2][row_pair_lane];
                                                const uint64_t pair_bits_3 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][3][row_pair_lane];
                                                if (row_in_bounds) {
                                                    float2 cached_pairs[8];
                                                    const uint64_t pair_bits_arr[4] = {
                                                        pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3
                                                    };
                                                    float amax = 0.0f;
                                                    #pragma unroll
                                                    for (int pair = 0; pair < 4; ++pair) {
                                                        const uint64_t pair_bits = pair_bits_arr[pair];
                                                        const bf16_2 pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                                        const bf16_2 pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                                        const float2 pair_lo = make_float2(
                                                            __bfloat162float(pair_lo_bf.x),
                                                            __bfloat162float(pair_lo_bf.y));
                                                        const float2 pair_hi = make_float2(
                                                            __bfloat162float(pair_hi_bf.x),
                                                            __bfloat162float(pair_hi_bf.y));
                                                        cached_pairs[pair] = pair_lo;
                                                        cached_pairs[pair + 4] = pair_hi;
                                                        amax = fmaxf(amax, fabsf(pair_lo.x));
                                                        amax = fmaxf(amax, fabsf(pair_lo.y));
                                                        amax = fmaxf(amax, fabsf(pair_hi.x));
                                                        amax = fmaxf(amax, fabsf(pair_hi.y));
                                                    }
                                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;
                                                    uint64_t packed_fp4 = 0;
                                                    #pragma unroll
                                                    for (int pair = 0; pair < 8; ++pair) {
                                                        packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                            cached_pairs[pair].x,
                                                            cached_pairs[pair].y,
                                                            rcp_scale)) << (pair * 8);
                                                    }
                                                    store_global_u64(
                                                        &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                        packed_fp4);
                                                    const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                                    float stored_scale = scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int kgroup = global_col_16 / 64;
                                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                    const int chunk = row_chunk_base + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                            }
                                        }
                                    }
                                } else if constexpr (C::EARLY_COL_READY &&
                                              G::COL_READY_PER_ROW16 &&
                                              (!G::ROW_QUANT_IMMEDIATE_PER_ROW16 || !DO_ROW)) {
                                    __syncwarp();
                                    __threadfence_block();
                                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                    if (warp::laneid() == 0) {
                                        #pragma unroll
                                        for (int local_i = 0; local_i < subtile_rt_bf::height; ++local_i) {
                                            const int published_row16_block = (local_warp_row_base + local_i * 16) / 16;
                                            arrive(slice_col_ready_row16[bf_stage][published_row16_block]);
                                        }
                                    }
                                }
                            }

                            if constexpr (G::ROW_QUANT_FROM_COL_PAIR_STAGE && DO_ROW && !DO_COL) {
                                #pragma unroll
                                for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                    const int row16_block = (local_warp_row_base + i * 16) / 16;
                                    #pragma unroll
                                    for (int row_half = 0; row_half < 2; ++row_half) {
                                        const int pair_slot = row_half * 4 + lane_row / 2;
                                        #pragma unroll
                                        for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                            const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                            const bool writer_lane = (lane_row & 1) == 0;
                                            const int col_base = group16 * 16 + lane_pair * 2;
                                            const int col_pair_base = col_base / 2;
                                            const bf16_2 vals0 =
                                                D_bf.tiles[i][group16].data[row_half];
                                            const bf16_2 vals1 =
                                                D_bf.tiles[i][group16].data[row_half + 2];
                                            const uint32_t vals0_bits = bf16x2_bits(vals0);
                                            const uint32_t vals0_peer_bits =
                                                __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 0,
                                                    pair_slot,
                                                    (vals0_bits & 0x0000ffffu) | (vals0_peer_bits << 16),
                                                    (vals0_bits >> 16) | (vals0_peer_bits & 0xffff0000u));
                                            }
                                            const uint32_t vals1_bits = bf16x2_bits(vals1);
                                            const uint32_t vals1_peer_bits =
                                                __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                            if (writer_lane) {
                                                store_col_pair_stage_pair<G>(
                                                    col_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    col_pair_base + 4,
                                                    pair_slot,
                                                    (vals1_bits & 0x0000ffffu) | (vals1_peer_bits << 16),
                                                    (vals1_bits >> 16) | (vals1_peer_bits & 0xffff0000u));
                                            }
                                        }
                                    }
                                }
                            }

                            if constexpr (G::USE_ROW_PAIR_STAGE && DO_ROW && !DO_COL) {
                                #pragma unroll
                                for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                    const int row16_block = (local_warp_row_base + i * 16) / 16;
                                    #pragma unroll
                                    for (int row_half = 0; row_half < 2; ++row_half) {
                                        const int pair_slot = row_half * 4 + lane_row / 2;
                                        const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                        const bool writer_lane = (lane_row & 1) == 0;
                                        #pragma unroll
                                        for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                            const uint32_t vals0_bits =
                                                bf16x2_bits(D_bf.tiles[i][group16].data[row_half]);
                                            const uint32_t vals1_bits =
                                                bf16x2_bits(D_bf.tiles[i][group16].data[row_half + 2]);
                                            const uint32_t vals0_peer_bits =
                                                __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                            const uint32_t vals1_peer_bits =
                                                __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                            if constexpr (G::USE_ROW_RCP_STAGE) {
                                                const bf16_2 vals0 = bf16x2_from_bits(vals0_bits);
                                                const bf16_2 vals1 = bf16x2_from_bits(vals1_bits);
                                                const bf16_2 vals0_peer = bf16x2_from_bits(vals0_peer_bits);
                                                const bf16_2 vals1_peer = bf16x2_from_bits(vals1_peer_bits);
                                                float row0_amax = fmaxf(
                                                    fmaxf(fabsf(__bfloat162float(vals0.x)), fabsf(__bfloat162float(vals0.y))),
                                                    fmaxf(fabsf(__bfloat162float(vals1.x)), fabsf(__bfloat162float(vals1.y))));
                                                float row1_amax = fmaxf(
                                                    fmaxf(fabsf(__bfloat162float(vals0_peer.x)), fabsf(__bfloat162float(vals0_peer.y))),
                                                    fmaxf(fabsf(__bfloat162float(vals1_peer.x)), fabsf(__bfloat162float(vals1_peer.y))));
                                                row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 1, 4));
                                                row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 2, 4));
                                                row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 1, 4));
                                                row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 2, 4));
                                                if (writer_lane && lane_pair == 0) {
                                                    const int local_row0 = local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                                    const int global_row0 = tile_row_base + local_row0;
                                                    const int global_row1 = global_row0 + 1;
                                                    const bool row0_in_bounds = full_tile_rows || global_row0 < g.M;
                                                    const bool row1_in_bounds = full_tile_rows || global_row1 < g.M;
                                                    const float row0_rcp = row0_amax > 0.0f ? (FP4_MAX / row0_amax) : 0.0f;
                                                    const float row1_rcp = row1_amax > 0.0f ? (FP4_MAX / row1_amax) : 0.0f;
                                                    row_rcp_stage[bf_stage].rcp[row16_block][group16][pair_slot][0] = row0_in_bounds ? row0_rcp : 0.0f;
                                                    row_rcp_stage[bf_stage].rcp[row16_block][group16][pair_slot][1] = row1_in_bounds ? row1_rcp : 0.0f;
                                                }
                                            }
                                            if (writer_lane) {
                                                const uint64_t row_bits_lo =
                                                    static_cast<uint64_t>(vals0_bits) |
                                                    (static_cast<uint64_t>(vals1_bits) << 32);
                                                const uint64_t row_bits_hi =
                                                    static_cast<uint64_t>(vals0_peer_bits) |
                                                    (static_cast<uint64_t>(vals1_peer_bits) << 32);
                                                store_row_pair_stage_record<G>(
                                                    row_pair_stage,
                                                    bf_stage,
                                                    row16_block,
                                                    group16,
                                                    pair_slot,
                                                    lane_pair,
                                                    row_bits_lo,
                                                    row_bits_hi);
                                            }
                                        }
                                    }
                                }
                            }

                            if constexpr (DO_COL && C::EARLY_COL_READY) {
                                if constexpr (G::COL_READY_PER_ROW16) {
                                    update_phasebit<1>(slice_col_recycle_phasebits, bf_stage);
                                } else {
                                    warpgroup::sync(CONSUMER_WG_SYNC_ID);
                                    __threadfence_block();
                                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                    warpgroup::sync(CONSUMER_WG_SYNC_ID);
                                    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                        arrive(slice_col_ready[bf_stage]);
                                    }
                                    update_phasebit<1>(slice_col_recycle_phasebits, bf_stage);
                                }
                            }

                            if constexpr (DO_ROW && C::CONSUMER_DO_ROW) {
                                if constexpr (DO_COL && G::ROW_WAITS_FOR_COL_RECYCLE) {
                                    wait(slice_col_recycled[bf_stage], get_phasebit<1>(slice_col_recycle_phasebits, bf_stage));
                                }
                                if constexpr (G::ROW_QUANT_FROM_ROWRECORD_REGS) {
                                    #pragma unroll
                                    for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                        #pragma unroll
                                        for (int row_half = 0; row_half < 2; ++row_half) {
                                            const bool writer_lane = (lane_row & 1) == 0;
                                            const int peer_lane = ((lane_row ^ 1) << 2) | lane_pair;
                                            #pragma unroll
                                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                                const bf16_2 vals0 = D_bf.tiles[i][group16].data[row_half];
                                                const bf16_2 vals1 = D_bf.tiles[i][group16].data[row_half + 2];
                                                const uint32_t vals0_bits = bf16x2_bits(vals0);
                                                const uint32_t vals1_bits = bf16x2_bits(vals1);
                                                const uint32_t vals0_peer_bits =
                                                    __shfl_sync(0xffffffff, vals0_bits, peer_lane);
                                                const uint32_t vals1_peer_bits =
                                                    __shfl_sync(0xffffffff, vals1_bits, peer_lane);
                                                uint64_t row_bits_lo = 0;
                                                uint64_t row_bits_hi = 0;
                                                if (writer_lane) {
                                                    row_bits_lo =
                                                        static_cast<uint64_t>(vals0_bits) |
                                                        (static_cast<uint64_t>(vals1_bits) << 32);
                                                    row_bits_hi =
                                                        static_cast<uint64_t>(vals0_peer_bits) |
                                                        (static_cast<uint64_t>(vals1_peer_bits) << 32);
                                                }
                                                const bf16_2 row0_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_lo));
                                                const bf16_2 row0_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_lo >> 32));
                                                const bf16_2 row1_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_hi));
                                                const bf16_2 row1_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(row_bits_hi >> 32));
                                                float row0_amax = 0.0f;
                                                float row1_amax = 0.0f;
                                                uint8_t row0_lo_fp4 = 0;
                                                uint8_t row0_hi_fp4 = 0;
                                                uint8_t row1_lo_fp4 = 0;
                                                uint8_t row1_hi_fp4 = 0;
                                                if (writer_lane) {
                                                    const float row0_lo_x = __bfloat162float(row0_lo_bf.x);
                                                    const float row0_lo_y = __bfloat162float(row0_lo_bf.y);
                                                    const float row0_hi_x = __bfloat162float(row0_hi_bf.x);
                                                    const float row0_hi_y = __bfloat162float(row0_hi_bf.y);
                                                    const float row1_lo_x = __bfloat162float(row1_lo_bf.x);
                                                    const float row1_lo_y = __bfloat162float(row1_lo_bf.y);
                                                    const float row1_hi_x = __bfloat162float(row1_hi_bf.x);
                                                    const float row1_hi_y = __bfloat162float(row1_hi_bf.y);
                                                    row0_amax = fmaxf(fmaxf(fabsf(row0_lo_x), fabsf(row0_lo_y)),
                                                                      fmaxf(fabsf(row0_hi_x), fabsf(row0_hi_y)));
                                                    row1_amax = fmaxf(fmaxf(fabsf(row1_lo_x), fabsf(row1_lo_y)),
                                                                      fmaxf(fabsf(row1_hi_x), fabsf(row1_hi_y)));
                                                    row0_lo_fp4 = quantize_fp4_pair(row0_lo_x, row0_lo_y, 1.0f);
                                                    row0_hi_fp4 = quantize_fp4_pair(row0_hi_x, row0_hi_y, 1.0f);
                                                    row1_lo_fp4 = quantize_fp4_pair(row1_lo_x, row1_lo_y, 1.0f);
                                                    row1_hi_fp4 = quantize_fp4_pair(row1_hi_x, row1_hi_y, 1.0f);
                                                }
                                                row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 1, 4));
                                                row0_amax = fmaxf(row0_amax, __shfl_xor_sync(0xffffffff, row0_amax, 2, 4));
                                                row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 1, 4));
                                                row1_amax = fmaxf(row1_amax, __shfl_xor_sync(0xffffffff, row1_amax, 2, 4));

                                                if (writer_lane) {
                                                    const int local_row0 =
                                                        local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                                    const int global_row0 = tile_row_base + local_row0;
                                                    const int global_row1 = global_row0 + 1;
                                                    const bool row0_in_bounds = full_tile_rows || global_row0 < g.M;
                                                    const bool row1_in_bounds = full_tile_rows || global_row1 < g.M;
                                                    const float row0_rcp = row0_amax > 0.0f ? (FP4_MAX / row0_amax) : 0.0f;
                                                    const float row1_rcp = row1_amax > 0.0f ? (FP4_MAX / row1_amax) : 0.0f;
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;

                                                    const float row0_lo_x = __bfloat162float(row0_lo_bf.x);
                                                    const float row0_lo_y = __bfloat162float(row0_lo_bf.y);
                                                    const float row0_hi_x = __bfloat162float(row0_hi_bf.x);
                                                    const float row0_hi_y = __bfloat162float(row0_hi_bf.y);
                                                    const float row1_lo_x = __bfloat162float(row1_lo_bf.x);
                                                    const float row1_lo_y = __bfloat162float(row1_lo_bf.y);
                                                    const float row1_hi_x = __bfloat162float(row1_hi_bf.x);
                                                    const float row1_hi_y = __bfloat162float(row1_hi_bf.y);
                                                    row0_lo_fp4 = quantize_fp4_pair(row0_lo_x, row0_lo_y, row0_rcp);
                                                    row0_hi_fp4 = quantize_fp4_pair(row0_hi_x, row0_hi_y, row0_rcp);
                                                    row1_lo_fp4 = quantize_fp4_pair(row1_lo_x, row1_lo_y, row1_rcp);
                                                    row1_hi_fp4 = quantize_fp4_pair(row1_hi_x, row1_hi_y, row1_rcp);

                                                    if (row0_in_bounds) {
                                                        row_fp4_ptr[global_row0 * row_fp4_stride + fp4x2_col_base + lane_pair] = row0_lo_fp4;
                                                        row_fp4_ptr[global_row0 * row_fp4_stride + fp4x2_col_base + lane_pair + 4] = row0_hi_fp4;
                                                        if (lane_pair == 0) {
                                                            const float scale = row0_rcp > 0.0f ? (1.0f / row0_rcp) : 0.0f;
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(row0_rcp * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int depth = global_row0 / 128;
                                                            const int sr = global_row0 % 32;
                                                            const int rr = (global_row0 / 32) % 4;
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = depth * row_sc_kgroups + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                        }
                                                    }
                                                    if (row1_in_bounds) {
                                                        row_fp4_ptr[global_row1 * row_fp4_stride + fp4x2_col_base + lane_pair] = row1_lo_fp4;
                                                        row_fp4_ptr[global_row1 * row_fp4_stride + fp4x2_col_base + lane_pair + 4] = row1_hi_fp4;
                                                        if (lane_pair == 0) {
                                                            const float scale = row1_rcp > 0.0f ? (1.0f / row1_rcp) : 0.0f;
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(row1_rcp * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int depth = global_row1 / 128;
                                                            const int sr = global_row1 % 32;
                                                            const int rr = (global_row1 / 32) % 4;
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = depth * row_sc_kgroups + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if constexpr (G::USE_ROW_PAIR_STAGE || G::ROW_QUANT_FROM_COL_PAIR_STAGE) {
                                    if constexpr (G::USE_ROW_PAIR_STAGE && G::ROW_PAIR_STAGE_WARP_SYNC_ONLY) {
                                        __syncwarp();
                                    } else {
                                        warpgroup::sync(CONSUMER_WG_SYNC_ID);
                                    }
                                    } else {
                                        warpgroup::store(bf16_epi_stage[bf_stage].D, D_bf);
                                        warpgroup::sync(CONSUMER_WG_SYNC_ID);
                                    }

                                    const uint32_t d_base = (G::USE_ROW_PAIR_STAGE || G::ROW_QUANT_FROM_COL_PAIR_STAGE) ? 0u : static_cast<uint32_t>(
                                        __cvta_generic_to_shared(&bf16_epi_stage[bf_stage].D.data[0]));
                                    if constexpr (DO_COL && G::ROW_QUANT_IMMEDIATE_PER_ROW16) {
                                        // Row quant for this fused path already ran during the
                                        // per-row16 producer/publish loop above.
                                    } else if constexpr (G::USE_ROW_PAIR_STAGE) {
                                        if constexpr (!(DO_COL && G::ROW_QUANT_IMMEDIATE_PER_ROW16)) {
                                        const int quant_row = threadIdx.x;
                                        if (quant_row < C::Mb / 2) {
                                            const int global_row = tile_row_base + quant_row;
                                            const int depth = global_row / 128;
                                            const int sr = global_row % 32;
                                            const int rr = (global_row / 32) % 4;
                                            const int row_chunk_base = depth * row_sc_kgroups;
                                            const bool row_in_bounds = full_tile_rows || global_row < g.M;
                                            const int row16_block = quant_row / 16;
                                            const int row_pair_slot = (quant_row % 16) / 2;
                                            const int row_pair_lane = quant_row & 1;
                                            if constexpr (DO_COL && G::ROW_WAITS_FOR_COL_DONE_ROW16) {
                                                while (slice_col_done_row16[bf_stage][row16_block] == 0) {
                                                }
                                                __threadfence_block();
                                            }
                                            #pragma unroll
                                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                                if constexpr (G::ROW_PAIR_STAGE_ROWRECORD) {
                                                    uint64_t pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3;
                                                    if constexpr (G::ROW_PAIR_STAGE_SLOTMAJOR) {
                                                        const ulonglong2 rec01 =
                                                            *reinterpret_cast<const ulonglong2*>(
                                                                &row_pair_stage[bf_stage].row_records_slotmajor[row16_block][row_pair_slot][group16][row_pair_lane][0]);
                                                        const ulonglong2 rec23 =
                                                            *reinterpret_cast<const ulonglong2*>(
                                                                &row_pair_stage[bf_stage].row_records_slotmajor[row16_block][row_pair_slot][group16][row_pair_lane][2]);
                                                        pair_bits_0 = static_cast<uint64_t>(rec01.x);
                                                        pair_bits_1 = static_cast<uint64_t>(rec01.y);
                                                        pair_bits_2 = static_cast<uint64_t>(rec23.x);
                                                        pair_bits_3 = static_cast<uint64_t>(rec23.y);
                                                    } else if constexpr (G::ROW_PAIR_STAGE_PADDED_LAYOUT) {
                                                        pair_bits_0 = row_pair_stage[bf_stage].row_records_padded[row16_block][group16][row_pair_slot][row_pair_lane][0];
                                                        pair_bits_1 = row_pair_stage[bf_stage].row_records_padded[row16_block][group16][row_pair_slot][row_pair_lane][1];
                                                        pair_bits_2 = row_pair_stage[bf_stage].row_records_padded[row16_block][group16][row_pair_slot][row_pair_lane][2];
                                                        pair_bits_3 = row_pair_stage[bf_stage].row_records_padded[row16_block][group16][row_pair_slot][row_pair_lane][3];
                                                    } else if constexpr (G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT) {
                                                        pair_bits_0 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][0][row_pair_lane];
                                                        pair_bits_1 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][1][row_pair_lane];
                                                        pair_bits_2 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][2][row_pair_lane];
                                                        pair_bits_3 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][3][row_pair_lane];
                                                    } else {
                                                        const ulonglong2 rec01 =
                                                            *reinterpret_cast<const ulonglong2*>(
                                                                &row_pair_stage[bf_stage].row_records[row16_block][group16][row_pair_slot][row_pair_lane][0]);
                                                        const ulonglong2 rec23 =
                                                            *reinterpret_cast<const ulonglong2*>(
                                                                &row_pair_stage[bf_stage].row_records[row16_block][group16][row_pair_slot][row_pair_lane][2]);
                                                        pair_bits_0 = static_cast<uint64_t>(rec01.x);
                                                        pair_bits_1 = static_cast<uint64_t>(rec01.y);
                                                        pair_bits_2 = static_cast<uint64_t>(rec23.x);
                                                        pair_bits_3 = static_cast<uint64_t>(rec23.y);
                                                    }
                                                    if (row_in_bounds) {
                                                        float rcp_scale;
                                                        if constexpr (G::USE_ROW_RCP_STAGE) {
                                                            rcp_scale =
                                                                row_rcp_stage[bf_stage].rcp[row16_block][group16][row_pair_slot][row_pair_lane];
                                                        } else if constexpr (G::ROW_PAIR_STAGE_FLOATCACHE) {
                                                            if constexpr (G::ROW_QUANT_NOINLINE_HELPER) {
                                                                quantize_rowrecord_pairbits_to_global_noinline<G::ROW_QUANT_USE_HW_FP4X2>(
                                                                    pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3,
                                                                    row_fp4_ptr, row_fp4_stride,
                                                                    row_sc_ptr, row_sc_kgroups,
                                                                    global_row, col_start, group16,
                                                                    g_sg_rcp, g_sg, encode_centric,
                                                                    combo_row_fp4_local_ptr, G::G_fp4_row_tile::cols,
                                                                    combo_row_sc_local_ptr, quant_row,
                                                                    epi * SUBTILE_COLS);
                                                            } else if constexpr (G::ROW_QUANT_USE_HW_FP4X2) {
                                                                quantize_rowrecord_pairbits_to_global<true>(
                                                                    pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3,
                                                                    row_fp4_ptr, row_fp4_stride,
                                                                    row_sc_ptr, row_sc_kgroups,
                                                                    global_row, col_start, group16,
                                                                    g_sg_rcp, g_sg, encode_centric,
                                                                    combo_row_fp4_local_ptr, G::G_fp4_row_tile::cols,
                                                                    combo_row_sc_local_ptr, quant_row,
                                                                    epi * SUBTILE_COLS);
                                                            } else {
                                                            float2 cached_pairs[8];
                                                            const uint64_t pair_bits_arr[4] = {
                                                                pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3
                                                            };
                                                            float amax = 0.0f;
                                                            #pragma unroll
                                                            for (int pair = 0; pair < 4; ++pair) {
                                                                const uint64_t pair_bits = pair_bits_arr[pair];
                                                                const bf16_2 pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                                                const bf16_2 pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                                                const float2 pair_lo = make_float2(
                                                                    __bfloat162float(pair_lo_bf.x),
                                                                    __bfloat162float(pair_lo_bf.y));
                                                                const float2 pair_hi = make_float2(
                                                                    __bfloat162float(pair_hi_bf.x),
                                                                    __bfloat162float(pair_hi_bf.y));
                                                                cached_pairs[pair] = pair_lo;
                                                                cached_pairs[pair + 4] = pair_hi;
                                                                amax = fmaxf(amax, fabsf(pair_lo.x));
                                                                amax = fmaxf(amax, fabsf(pair_lo.y));
                                                                amax = fmaxf(amax, fabsf(pair_hi.x));
                                                                amax = fmaxf(amax, fabsf(pair_hi.y));
                                                            }
                                                            rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                            const int global_col_16 = col_start + group16 * 16;
                                                            const int fp4x2_col_base = global_col_16 / 2;
                                                            uint64_t packed_fp4 = 0;
                                                            #pragma unroll
                                                            for (int pair = 0; pair < 8; ++pair) {
                                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                    cached_pairs[pair].x,
                                                                    cached_pairs[pair].y,
                                                                    rcp_scale)) << (pair * 8);
                                                            }
                                                            store_global_u64(
                                                                &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                                packed_fp4);
                                                            const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = row_chunk_base + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                            }
                                                        } else {
                                                            bf16_2 cached_lo[4];
                                                            bf16_2 cached_hi[4];
                                                            cached_lo[0] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_0));
                                                            cached_hi[0] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_0 >> 32));
                                                            cached_lo[1] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_1));
                                                            cached_hi[1] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_1 >> 32));
                                                            cached_lo[2] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_2));
                                                            cached_hi[2] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_2 >> 32));
                                                            cached_lo[3] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_3));
                                                            cached_hi[3] = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_3 >> 32));
                                                            float amax = 0.0f;
                                                            #pragma unroll
                                                            for (int pair = 0; pair < 4; ++pair) {
                                                                amax = fmaxf(amax, fabsf(__bfloat162float(cached_lo[pair].x)));
                                                                amax = fmaxf(amax, fabsf(__bfloat162float(cached_lo[pair].y)));
                                                                amax = fmaxf(amax, fabsf(__bfloat162float(cached_hi[pair].x)));
                                                                amax = fmaxf(amax, fabsf(__bfloat162float(cached_hi[pair].y)));
                                                            }
                                                            rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                            const int global_col_16 = col_start + group16 * 16;
                                                            const int fp4x2_col_base = global_col_16 / 2;
                                                            uint64_t packed_fp4 = 0;
                                                            #pragma unroll
                                                            for (int pair = 0; pair < 4; ++pair) {
                                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                    __bfloat162float(cached_lo[pair].x),
                                                                    __bfloat162float(cached_lo[pair].y),
                                                                    rcp_scale)) << (pair * 8);
                                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                    __bfloat162float(cached_hi[pair].x),
                                                                    __bfloat162float(cached_hi[pair].y),
                                                                    rcp_scale)) << ((pair + 4) * 8);
                                                            }
                                                            store_global_u64(
                                                                &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                                packed_fp4);
                                                            const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = row_chunk_base + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                        }
                                                        if constexpr (G::USE_ROW_RCP_STAGE) {
                                                            const int global_col_16 = col_start + group16 * 16;
                                                            const int fp4x2_col_base = global_col_16 / 2;
                                                            uint64_t packed_fp4 = 0;
                                                            const bf16_2 pair0_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_0));
                                                            const bf16_2 pair0_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_0 >> 32));
                                                            const bf16_2 pair1_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_1));
                                                            const bf16_2 pair1_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_1 >> 32));
                                                            const bf16_2 pair2_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_2));
                                                            const bf16_2 pair2_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_2 >> 32));
                                                            const bf16_2 pair3_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_3));
                                                            const bf16_2 pair3_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits_3 >> 32));
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair0_lo.x),
                                                                __bfloat162float(pair0_lo.y),
                                                                rcp_scale)) << 0;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair1_lo.x),
                                                                __bfloat162float(pair1_lo.y),
                                                                rcp_scale)) << 8;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair2_lo.x),
                                                                __bfloat162float(pair2_lo.y),
                                                                rcp_scale)) << 16;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair3_lo.x),
                                                                __bfloat162float(pair3_lo.y),
                                                                rcp_scale)) << 24;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair0_hi.x),
                                                                __bfloat162float(pair0_hi.y),
                                                                rcp_scale)) << 32;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair1_hi.x),
                                                                __bfloat162float(pair1_hi.y),
                                                                rcp_scale)) << 40;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair2_hi.x),
                                                                __bfloat162float(pair2_hi.y),
                                                                rcp_scale)) << 48;
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair3_hi.x),
                                                                __bfloat162float(pair3_hi.y),
                                                                rcp_scale)) << 56;
                                                            store_global_u64(
                                                                &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                                packed_fp4);
                                                            const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = row_chunk_base + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                        }
                                                    }
                                                } else {
                                                    if constexpr (G::ROW_PAIR_STAGE_FLOATCACHE) {
                                                        float2 cached_pairs[8];
                                                        float amax = 0.0f;
                                                        #pragma unroll
                                                        for (int pair = 0; pair < 4; ++pair) {
                                                            const uint64_t pair_bits =
                                                                row_pair_stage[bf_stage].packed[row16_block][group16][row_pair_slot][pair][row_pair_lane];
                                                            const bf16_2 pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                                            const bf16_2 pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                                            const float2 pair_lo = make_float2(
                                                                __bfloat162float(pair_lo_bf.x),
                                                                __bfloat162float(pair_lo_bf.y));
                                                            const float2 pair_hi = make_float2(
                                                                __bfloat162float(pair_hi_bf.x),
                                                                __bfloat162float(pair_hi_bf.y));
                                                            cached_pairs[pair] = pair_lo;
                                                            cached_pairs[pair + 4] = pair_hi;
                                                            amax = fmaxf(amax, fabsf(pair_lo.x));
                                                            amax = fmaxf(amax, fabsf(pair_lo.y));
                                                            amax = fmaxf(amax, fabsf(pair_hi.x));
                                                            amax = fmaxf(amax, fabsf(pair_hi.y));
                                                        }

                                                        const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                        if (row_in_bounds) {
                                                            const int global_col_16 = col_start + group16 * 16;
                                                            const int fp4x2_col_base = global_col_16 / 2;
                                                            uint64_t packed_fp4 = 0;
                                                            #pragma unroll
                                                            for (int pair = 0; pair < 8; ++pair) {
                                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                    cached_pairs[pair].x,
                                                                    cached_pairs[pair].y,
                                                                    rcp_scale)) << (pair * 8);
                                                            }
                                                            store_global_u64(
                                                                &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                                packed_fp4);
                                                            const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = row_chunk_base + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                        }
                                                    } else {
                                                        float amax = 0.0f;
                                                        uint64_t packed_row_pairs[4];
                                                        #pragma unroll
                                                        for (int pair = 0; pair < 4; ++pair) {
                                                            const uint64_t pair_bits =
                                                                row_pair_stage[bf_stage].packed[row16_block][group16][row_pair_slot][pair][row_pair_lane];
                                                            packed_row_pairs[pair] = pair_bits;
                                                            const bf16_2 pair_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                                            const bf16_2 pair_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                                            amax = fmaxf(amax, fabsf(__bfloat162float(pair_lo.x)));
                                                            amax = fmaxf(amax, fabsf(__bfloat162float(pair_lo.y)));
                                                            amax = fmaxf(amax, fabsf(__bfloat162float(pair_hi.x)));
                                                            amax = fmaxf(amax, fabsf(__bfloat162float(pair_hi.y)));
                                                        }

                                                        const float scale = amax * (1.0f / FP4_MAX);
                                                        const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;

                                                        if (row_in_bounds) {
                                                            const int global_col_16 = col_start + group16 * 16;
                                                            const int fp4x2_col_base = global_col_16 / 2;
                                                            uint64_t packed_fp4 = 0;
                                                            #pragma unroll
                                                            for (int pair = 0; pair < 4; ++pair) {
                                                                const uint64_t pair_bits = packed_row_pairs[pair];
                                                                const bf16_2 pair_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                                                const bf16_2 pair_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                    __bfloat162float(pair_lo.x),
                                                                    __bfloat162float(pair_lo.y),
                                                                    rcp_scale)) << (pair * 8);
                                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                    __bfloat162float(pair_hi.x),
                                                                    __bfloat162float(pair_hi.y),
                                                                    rcp_scale)) << ((pair + 4) * 8);
                                                            }
                                                            store_global_u64(
                                                                &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                                packed_fp4);
                                                            float stored_scale = scale * g_sg_rcp;
                                                            if (encode_centric) {
                                                                stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                            }
                                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                            const int kgroup = global_col_16 / 64;
                                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                            const int chunk = row_chunk_base + kgroup;
                                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                                *reinterpret_cast<const uint8_t*>(&sc);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        const int quant_row = threadIdx.x;
                                        if (quant_row < C::Mb / 2) {
                                            const int global_row = tile_row_base + quant_row;
                                            const int depth = global_row / 128;
                                            const int sr = global_row % 32;
                                            const int rr = (global_row / 32) % 4;
                                            const int row_chunk_base = depth * row_sc_kgroups;
                                            const bool row_in_bounds = full_tile_rows || global_row < g.M;
                                            const int row16_block = quant_row / 16;
                                            const int row_pair_slot = (quant_row % 16) / 2;
                                            const int row_pair_lane = quant_row & 1;
                                            #pragma unroll
                                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                                float amax = 0.0f;
                                                uint64_t packed_row_pairs[4];
                                                #pragma unroll
                                                for (int pair = 0; pair < 4; ++pair) {
                                                    if constexpr (G::ROW_QUANT_FROM_COL_PAIR_STAGE) {
                                                        const uint64_t pair0_bits = [&]() {
                                                            if constexpr (G::COL_PAIR_STAGE_PADDED_LAYOUT) {
                                                                return *reinterpret_cast<const uint64_t*>(
                                                                    &col_pair_stage[bf_stage].pairs_padded[row16_block][group16 * 8 + pair][row_pair_slot][0]);
                                                            } else {
                                                                return *reinterpret_cast<const uint64_t*>(
                                                                    &col_pair_stage[bf_stage].pairs[row16_block][group16 * 8 + pair][row_pair_slot][0]);
                                                            }
                                                        }();
                                                        const uint64_t pair1_bits = [&]() {
                                                            if constexpr (G::COL_PAIR_STAGE_PADDED_LAYOUT) {
                                                                return *reinterpret_cast<const uint64_t*>(
                                                                    &col_pair_stage[bf_stage].pairs_padded[row16_block][group16 * 8 + pair + 4][row_pair_slot][0]);
                                                            } else {
                                                                return *reinterpret_cast<const uint64_t*>(
                                                                    &col_pair_stage[bf_stage].pairs[row16_block][group16 * 8 + pair + 4][row_pair_slot][0]);
                                                            }
                                                        }();
                                                        const uint32_t col0_bits = static_cast<uint32_t>(pair0_bits);
                                                        const uint32_t col1_bits = static_cast<uint32_t>(pair0_bits >> 32);
                                                        const uint32_t col4_bits = static_cast<uint32_t>(pair1_bits);
                                                        const uint32_t col5_bits = static_cast<uint32_t>(pair1_bits >> 32);
                                                        const uint16_t row0_bits = row_pair_lane == 0 ?
                                                            static_cast<uint16_t>(col0_bits) :
                                                            static_cast<uint16_t>(col0_bits >> 16);
                                                        const uint16_t row1_bits = row_pair_lane == 0 ?
                                                            static_cast<uint16_t>(col1_bits) :
                                                            static_cast<uint16_t>(col1_bits >> 16);
                                                        const uint16_t row4_bits = row_pair_lane == 0 ?
                                                            static_cast<uint16_t>(col4_bits) :
                                                            static_cast<uint16_t>(col4_bits >> 16);
                                                        const uint16_t row5_bits = row_pair_lane == 0 ?
                                                            static_cast<uint16_t>(col5_bits) :
                                                            static_cast<uint16_t>(col5_bits >> 16);
                                                        const bf16_2 pair_lo = bf16x2_from_bits(
                                                            static_cast<uint32_t>(row0_bits) |
                                                            (static_cast<uint32_t>(row1_bits) << 16));
                                                        const bf16_2 pair_hi = bf16x2_from_bits(
                                                            static_cast<uint32_t>(row4_bits) |
                                                            (static_cast<uint32_t>(row5_bits) << 16));
                                                        packed_row_pairs[pair] =
                                                            static_cast<uint64_t>(bf16x2_bits(pair_lo)) |
                                                            (static_cast<uint64_t>(bf16x2_bits(pair_hi)) << 32);
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(pair_lo.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(pair_lo.y)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(pair_hi.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(pair_hi.y)));
                                                    } else {
                                                        bf16_2 vals_pair;
                                                        const int col = group16 * 16 + pair * 2;
                                                        move<bf16_2>::lds(vals_pair, G::D_tile::idx(d_base, {quant_row, col}));
                                                        packed_row_pairs[pair] = bf16x2_bits(vals_pair);
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals_pair.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals_pair.y)));
                                                    }
                                                }
                                                if constexpr (!G::ROW_QUANT_FROM_COL_PAIR_STAGE) {
                                                    #pragma unroll
                                                    for (int pair = 4; pair < 8; ++pair) {
                                                        bf16_2 vals_pair;
                                                        const int col = group16 * 16 + pair * 2;
                                                        move<bf16_2>::lds(vals_pair, G::D_tile::idx(d_base, {quant_row, col}));
                                                        packed_row_pairs[pair] = bf16x2_bits(vals_pair);
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals_pair.x)));
                                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals_pair.y)));
                                                    }
                                                }

                                                const float scale = amax * (1.0f / FP4_MAX);
                                                const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;

                                                if (row_in_bounds) {
                                                    const int global_col_16 = col_start + group16 * 16;
                                                    const int fp4x2_col_base = global_col_16 / 2;
                                                    if constexpr (G::ROW_QUANT_FROM_COL_PAIR_STAGE) {
                                                        uint64_t packed_fp4 = 0;
                                                        #pragma unroll
                                                        for (int pair = 0; pair < 4; ++pair) {
                                                            const uint64_t pair_bits = packed_row_pairs[pair];
                                                            const bf16_2 pair_lo = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                                            const bf16_2 pair_hi = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair_lo.x),
                                                                __bfloat162float(pair_lo.y),
                                                                rcp_scale)) << (pair * 8);
                                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                                __bfloat162float(pair_hi.x),
                                                                __bfloat162float(pair_hi.y),
                                                                rcp_scale)) << ((pair + 4) * 8);
                                                        }
                                                        store_global_u64(
                                                            &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                                            packed_fp4);
                                                    } else {
                                                        #pragma unroll
                                                        for (int pair = 0; pair < 8; ++pair) {
                                                            const bf16_2 vals_pair = bf16x2_from_bits(static_cast<uint32_t>(packed_row_pairs[pair]));
                                                            row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + pair] =
                                                                quantize_fp4_pair(
                                                                    __bfloat162float(vals_pair.x),
                                                                    __bfloat162float(vals_pair.y),
                                                                    rcp_scale);
                                                        }
                                                    }

                                                    float stored_scale = scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int kgroup = global_col_16 / 64;
                                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                    const int chunk = row_chunk_base + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                            }
                                        }
                                        }
                                    }
                                }
                            }
                        }
                        if constexpr (DO_COL && !C::EARLY_COL_READY) {
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                            if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                arrive(slice_col_ready[bf_stage]);
                            }
                            update_phasebit<1>(slice_col_recycle_phasebits, bf_stage);
                        }
                    } else {
                        warpgroup::sync(CONSUMER_WG_SYNC_ID);
                        warpgroup::store(bf16_epi_stage[bf_stage].D, D_bf);
                        warpgroup::sync(CONSUMER_WG_SYNC_ID);

                        if constexpr (DO_ROW && DO_COL && !C::EARLY_COL_READY && C::ROW_QUANT_FROM_REGS) {
                            if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            }
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                        }

                        if constexpr (DO_COL && C::EARLY_COL_READY) {
                            if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                arrive(slice_col_ready[bf_stage]);
                            }
                        }

                        if constexpr (DO_ROW && C::CONSUMER_DO_ROW) {
                            if constexpr (C::ROW_QUANT_FROM_REGS) {
                                const int lane_row = lane_id / 4;
                                const int lane_pair = lane_id % 4;
                                const int local_warp_row_base = warpgroup::warpid() * (C::Mb / 8);
                                #pragma unroll
                                for (int i = 0; i < subtile_rt_bf::height; ++i) {
                                    #pragma unroll
                                    for (int row_half = 0; row_half < 2; ++row_half) {
                                        const int local_row = local_warp_row_base + i * 16 + row_half * 8 + lane_row;
                                        const int global_row = tile_row_base + local_row;
                                        if (full_tile_rows || global_row < g.M) {
                                            #pragma unroll
                                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                                const bf16_2 vals0 = D_bf.tiles[i][group16].data[row_half];
                                                const bf16_2 vals1 = D_bf.tiles[i][group16].data[row_half + 2];
                                                const float v00 = __bfloat162float(vals0.x);
                                                const float v01 = __bfloat162float(vals0.y);
                                                const float v10 = __bfloat162float(vals1.x);
                                                const float v11 = __bfloat162float(vals1.y);
                                                const int depth = global_row / 128;
                                                const int sr = global_row % 32;
                                                const int rr = (global_row / 32) % 4;
                                                const int row_chunk_base = depth * row_sc_kgroups;
                                                float amax = fmaxf(fmaxf(fabsf(v00), fabsf(v01)),
                                                                   fmaxf(fabsf(v10), fabsf(v11)));
                                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
                                                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));

                                                const float scale = amax * (1.0f / FP4_MAX);
                                                const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                                const int global_col_16 = col_start + group16 * 16;
                                                const int fp4x2_col_base = global_col_16 / 2;
                                                row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + lane_pair] =
                                                    quantize_fp4_pair(v00, v01, rcp_scale);
                                                row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base + 4 + lane_pair] =
                                                    quantize_fp4_pair(v10, v11, rcp_scale);

                                                if (lane_pair == 0) {
                                                    float stored_scale = scale * g_sg_rcp;
                                                    if (encode_centric) {
                                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                                    }
                                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                                    const int kgroup = global_col_16 / 64;
                                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                                    const int chunk = row_chunk_base + kgroup;
                                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                                        *reinterpret_cast<const uint8_t*>(&sc);
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                const uint32_t d_base = static_cast<uint32_t>(
                                    __cvta_generic_to_shared(&bf16_epi_stage[bf_stage].D.data[0]));
                                const uint32_t d_col_stage_base = [&]() {
                                    if constexpr (G::USE_COL_PLAIN_STAGE) {
                                        return static_cast<uint32_t>(__cvta_generic_to_shared(&col_plain_stage[bf_stage].D.data[0]));
                                    } else {
                                        return 0u;
                                    }
                                }();
                                const int quant_row = threadIdx.x;
                                if (quant_row < C::Mb / 2) {
                                    const int global_row = tile_row_base + quant_row;
                                    #pragma unroll
                                    for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                        bf16_2 vals[8];
                                        float amax = 0.0f;
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const int col = group16 * 16 + pair * 2;
                                            move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                                            if constexpr (DO_COL && G::USE_COL_PLAIN_STAGE) {
                                                move<bf16_2>::sts(G::D_helper_tile::idx(d_col_stage_base, {quant_row, col}), vals[pair]);
                                            }
                                            amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                            amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                                        }

                                        const float scale = amax * (1.0f / FP4_MAX);
                                        const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;

                                        if (global_row < g.M) {
                                            #pragma unroll
                                            for (int pair = 0; pair < 8; ++pair) {
                                                const uint8_t fp4_pair = quantize_fp4_pair(
                                                    __bfloat162float(vals[pair].x),
                                                    __bfloat162float(vals[pair].y),
                                                    rcp_scale);
                                                const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                                row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                            }

                                            float stored_scale = scale * g_sg_rcp;
                                            if (encode_centric) {
                                                stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                            }
                                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                            const int global_col_16 = col_start + group16 * 16;
                                            const int kgroup = global_col_16 / 64;
                                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                                            const int depth = global_row / 128;
                                            const int sr = global_row % 32;
                                            const int rr = (global_row / 32) % 4;
                                            const int chunk = depth * row_sc_kgroups + kgroup;
                                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                            row_sc_ptr[chunk * 512 + byte_idx] =
                                                *reinterpret_cast<const uint8_t*>(&sc);
                                        }
                                    }
                                }
                            }
                        }

                        if constexpr (DO_ROW && C::CONSUMER_DO_ROW && C::ROW_QUANT_FROM_REGS) {
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                        }

                        if constexpr (DO_COL && G::USE_COL_PLAIN_STAGE) {
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                        }

                        if constexpr (DO_ROW && !C::CONSUMER_DO_ROW && G::USE_ROW_PAIR_STAGE && !G::ROW_PAIR_STAGE_WARP_SYNC_ONLY) {
                            warpgroup::sync(CONSUMER_WG_SYNC_ID);
                        }

                        if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                            if constexpr (DO_ROW && !C::CONSUMER_DO_ROW) {
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                arrive(slice_row_ready[bf_stage]);
                            }
                            if constexpr (DO_COL && !C::EARLY_COL_READY) {
                                if constexpr (G::USE_COL_PLAIN_STAGE || C::ROW_QUANT_FROM_REGS) {
                                    __threadfence_block();
                                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                }
                                arrive(slice_col_ready[bf_stage]);
                            }
                        }
                        if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                            if constexpr (DO_ROW && C::CONSUMER_DO_ROW) {
                                if (combo_use_de && combo_stage_publish_epi) {
                                    warpgroup::sync(CONSUMER_WG_SYNC_ID);
                                    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                                        __threadfence_block();
                                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                        arrive(combo_g_row_ready[0]);
                                    }
                                }
                            }
                        }
                        if constexpr (DO_ROW && !C::CONSUMER_DO_ROW) {
                            update_phasebit<1>(slice_row_recycle_phasebits, bf_stage);
                        }
                        if constexpr (DO_COL) {
                            if constexpr (G::FRONTEND_MULTI_QUANTIZER_WG &&
                                          C::EARLY_COL_READY &&
                                          G::USE_COL_PAIR_STAGE) {
                            } else if constexpr (G::USE_COL_PAIR_STAGE ||
                                          (DO_ROW && C::ROW_QUANT_FROM_REGS && C::CONSUMER_DO_ROW && !C::EARLY_COL_READY)) {
                                update_phasebit<1>(slice_col_recycle_phasebits, bf_stage);
                            } else {
                                update_phasebit<1>(slice_phasebits, bf_stage);
                            }
                        }
                        if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                            if constexpr (DO_ROW && C::CONSUMER_DO_ROW) {
                                if (combo_use_de && combo_stage_publish_epi) {
                                    update_phasebit<1>(combo_row_recycle_phasebits, 0);
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO) {
                if constexpr (!EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                    warpgroup::sync(CONSUMER_WG_SYNC_ID);
                    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        arrive(combo_row_stage_ready);
                    }
                }
            }

            update_phasebit<0>(phasebits, 0);
            phase ^= 1;
        }
        warpgroup::sync(CONSUMER_WG_SYNC_ID);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if constexpr (!EFFECTIVE_ENABLE_STOREADD_COMBO) {
            if (warpgroup::warpid() == 0) tm_allocator.deprovision();
        }
    } else if (effective_warpgroup_id < C::CONSUMER_WARPGROUPS + EFFECTIVE_QUANTIZER_WARPGROUPS) {
        constexpr bool ROW_IN_QUANTIZER = DO_ROW && !C::CONSUMER_DO_ROW && (C::QUANTIZER_WARPGROUPS > 0);
        constexpr bool COL_IN_QUANTIZER = DO_COL && (C::QUANTIZER_WARPGROUPS > 0);
        if constexpr (!(ROW_IN_QUANTIZER || COL_IN_QUANTIZER) && !EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
            return;
        }
        everyone::tma::cluster::wait_aligned();
        if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO) {
            wait(tmem_provisioned, 0);
        }

        const float g_sg = g.G_sg_row[{0}];
        const float g_sg_rcp = 1.0f / g_sg;
        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        constexpr int ROW16_BLOCKS = C::Mb / 32;
        constexpr int QUANTIZER_WG_THREADS = WARP_THREADS * WARPGROUP_WARPS;
        constexpr float FP4_MAX = 6.0f;
        constexpr float E4M3_MAX = 448.0f;

        const int quant_thread = threadIdx.x - warpgroup_id * QUANTIZER_WG_THREADS;
        const int quant_lane = quant_thread % WARP_THREADS;
        uint32_t slice_phasebits = 0xFFFF0000;
        uint32_t slice_row_ready_phasebits = 0xFFFF0000;
        uint32_t slice_col_ready_phasebits = 0xFFFF0000;
        uint32_t combo_row_ready_phasebits = 0xFFFF0000;
        uint32_t combo_col_ready_phasebits = 0xFFFF0000;
        uint32_t combo_col_recycle_phasebits = 0xFFFF0000;

        uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.G_fp4_row.raw_ptr);
        uint8_t* row_sc_ptr = g.G_sc_row_ptr;
        uint8_t* col_fp4_ptr = g.G_fp4_col_ptr;
        uint8_t* col_sc_ptr = g.G_sc_col_ptr;
        const int row_fp4_stride = g.G_fp4_row.cols();
        const int row_sc_kgroups = g.G_sc_row_kgroups;
        const int col_fp4_stride = g.A.rows() / 2;
        const int col_sc_kgroups = g.G_sc_col_kgroups;
        const bool encode_centric = g.encode_centric;
        int combo_row_stage_phase = 0;
        int combo_c_tiles_phase = 0;
        int combo_c_scales_phase = 0;
        int combo_e_tiles_phase = 0;
        int combo_e_scales_phase = 0;
        int combo_p3_outputs_phase = 0;
        int combo_backhalf_de_done_phase = 0;
        constexpr bool COMBO_MODE_IS_STATIC = (STATIC_COMBO_MODE >= 0);
        constexpr bool STATIC_COMBO_DO_DE =
            COMBO_MODE_IS_STATIC &&
            (STATIC_COMBO_MODE == G::COMBO_MODE_FULL || STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY);
        constexpr bool STATIC_COMBO_DO_DC =
            COMBO_MODE_IS_STATIC &&
            (STATIC_COMBO_MODE == G::COMBO_MODE_FULL || STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY);
        constexpr bool STATIC_COMBO_GONLY =
            COMBO_MODE_IS_STATIC && (STATIC_COMBO_MODE == G::COMBO_MODE_GONLY);
        const bool combo_do_de = G::ENABLE_STOREADD_COMBO &&
                                 (COMBO_MODE_IS_STATIC
                                      ? STATIC_COMBO_DO_DE
                                      : (g.combo_mode == G::COMBO_MODE_FULL || g.combo_mode == G::COMBO_MODE_DEONLY));
        const bool combo_do_dc = G::ENABLE_STOREADD_COMBO &&
                                 (COMBO_MODE_IS_STATIC
                                      ? STATIC_COMBO_DO_DC
                                      : (g.combo_mode == G::COMBO_MODE_FULL || g.combo_mode == G::COMBO_MODE_DCONLY));
        constexpr bool combo_debug_skip_sep_de_row_ready_wait = false;
        constexpr bool combo_debug_skip_sep_de_output_wait = false;
        constexpr bool combo_debug_skip_sep_de_input_wait = false;
        constexpr bool combo_debug_skip_sep_de_scale_wait = false;
        constexpr bool combo_debug_skip_sep_de_tile_wait = false;
        constexpr bool combo_debug_skip_sep_de_body = true;
        constexpr bool combo_debug_skip_sep_de_drain = false;
        constexpr bool combo_debug_skip_sep_de_row_sc_preload = true;
        if constexpr (G::ENABLE_STOREADD_COMBO) {
            tm_allocator.set_addr(tmem_addr);
        }

        constexpr int first_quantizer_wg = C::CONSUMER_WARPGROUPS;
        constexpr int first_col_quantizer_wg =
            EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS
                ? (STATIC_FRONTEND_ONLY_DEONLY
                       ? (EFFECTIVE_PRODUCER_WARPGROUP_ID + C::PRODUCER_WARPGROUPS)
                       : first_quantizer_wg)
                : first_quantizer_wg + (G::FRONTEND_SINGLE_QUANTIZER_WG ? G::FRONTEND_QUANTIZER_WARPGROUPS : C::ROW_QUANTIZER_WARPGROUPS);
        constexpr int first_de_consumer_wg =
            EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS
                ? (STATIC_FRONTEND_ONLY_DEONLY
                       ? (EFFECTIVE_PRODUCER_WARPGROUP_ID + C::PRODUCER_WARPGROUPS)
                       : (first_col_quantizer_wg + C::COL_QUANTIZER_WARPGROUPS))
                : first_col_quantizer_wg;
        constexpr int first_dc_consumer_wg = first_de_consumer_wg + (EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS ? G::COMBO_DE_WARPGROUPS : 0);
        const bool is_row_quantizer_wg =
            !EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS &&
            (warpgroup_id >= first_quantizer_wg) && (warpgroup_id < first_col_quantizer_wg);
        const bool is_col_quantizer_wg =
            EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS
                ? (warpgroup_id >= first_col_quantizer_wg &&
                   warpgroup_id < first_de_consumer_wg)
                : (G::FRONTEND_MULTI_QUANTIZER_WG &&
                   (warpgroup_id >= first_col_quantizer_wg) &&
                   (warpgroup_id < C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS));
        const bool is_de_consumer_wg =
            EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS &&
            (warpgroup_id >= first_de_consumer_wg) &&
            (warpgroup_id < first_dc_consumer_wg);
        const bool is_dc_consumer_wg =
            EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS &&
            (warpgroup_id >= first_dc_consumer_wg) &&
            (warpgroup_id < C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS);
        const int quantizer_sync_id = EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS
            ? (STATIC_FRONTEND_ONLY_DEONLY
                ? (is_de_consumer_wg ? 4 : 2)
                : (is_col_quantizer_wg ? 2 : (is_de_consumer_wg ? 3 : 4)))
            : (G::FRONTEND_MULTI_QUANTIZER_WG
                ? (is_row_quantizer_wg ? 2 : 3)
                : (!C::CONSUMER_DO_ROW ? 2 : 1));
        if constexpr (G::FRONTEND_MULTI_QUANTIZER_WG || EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
            if (!is_row_quantizer_wg && !is_col_quantizer_wg && !is_de_consumer_wg && !is_dc_consumer_wg) {
                return;
            }
        }
        if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
            if (((COMBO_MODE_IS_STATIC && STATIC_COMBO_GONLY) ||
                 (!COMBO_MODE_IS_STATIC && g.combo_mode == G::COMBO_MODE_GONLY)) &&
                (is_de_consumer_wg || is_dc_consumer_wg)) {
                return;
            }
            if (((COMBO_MODE_IS_STATIC && STATIC_COMBO_DO_DE && !STATIC_COMBO_DO_DC) ||
                 (!COMBO_MODE_IS_STATIC && g.combo_mode == G::COMBO_MODE_DEONLY)) &&
                is_col_quantizer_wg) {
                return;
            }
            if (((COMBO_MODE_IS_STATIC && STATIC_COMBO_DO_DE && !STATIC_COMBO_DO_DC) ||
                 (!COMBO_MODE_IS_STATIC && g.combo_mode == G::COMBO_MODE_DEONLY)) &&
                is_dc_consumer_wg) {
                return;
            }
            if (((COMBO_MODE_IS_STATIC && STATIC_COMBO_DO_DC && !STATIC_COMBO_DO_DE) ||
                 (!COMBO_MODE_IS_STATIC && g.combo_mode == G::COMBO_MODE_DCONLY)) &&
                is_de_consumer_wg) {
                return;
            }
        }

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            const bool full_tile_rows = tile_row_base + (C::Mb / 2) <= g.M;
            if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                if ((is_de_consumer_wg && !combo_do_de) ||
                    (is_dc_consumer_wg && !combo_do_dc)) {
                    continue;
                }
            }

            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                const int bf_stage = epi % C::BF16_STAGE_COUNT;
                const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                const bool combo_stage_first_epi = (epi == 0);
                const bool combo_stage_publish_epi = (epi == C::EPI_PIPE_DEPTH - 1);
                typename G::combo_col_stage_t *combo_col_stage_slot = nullptr;
                uint8_t* combo_col_fp4_local_ptr = nullptr;
                uint8_t* combo_col_sc_local_ptr = nullptr;
                if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO) {
                    if (combo_do_dc) {
                        combo_col_stage_slot = combo_col_stage;
                        combo_col_fp4_local_ptr =
                            reinterpret_cast<uint8_t*>(&combo_col_stage_slot->Gt_row.data[0]);
                        combo_col_sc_local_ptr =
                            reinterpret_cast<uint8_t*>(&combo_col_stage_slot->Gt_row_sc.data[0]);
                    }
                }

                if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                    if (is_col_quantizer_wg && combo_do_dc && combo_stage_first_epi) {
                        wait(combo_g_col_recycled[0], get_phasebit<1>(combo_col_recycle_phasebits, 0));
                        uint8_t *combo_col_bytes = reinterpret_cast<uint8_t*>(&combo_col_stage_slot->Gt_row.data[0]);
                        for (int idx = warpgroup::warpid() * WARP_THREADS + quant_lane;
                             idx < static_cast<int>(sizeof(typename G::combo_col_stage_t));
                             idx += WARPGROUP_WARPS * WARP_THREADS) {
                            combo_col_bytes[idx] = 0;
                        }
                        warpgroup::sync(quantizer_sync_id);
                    }
                    if (is_de_consumer_wg && combo_do_de && combo_stage_publish_epi) {
                        if constexpr (!combo_debug_skip_sep_de_row_ready_wait) {
                            wait(combo_g_row_ready[0], get_phasebit<0>(combo_row_ready_phasebits, 0));
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        }
                    }
                    if (is_dc_consumer_wg && combo_do_dc && combo_stage_publish_epi) {
                        wait(combo_g_col_ready[0], get_phasebit<0>(combo_col_ready_phasebits, 0));
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    }
                }

                    if constexpr (G::FRONTEND_SINGLE_QUANTIZER_WG) {
                        if constexpr (ROW_IN_QUANTIZER) {
                            if constexpr (!G::DEBUG_SKIP_ROW_READY_WAIT) {
                                wait(slice_row_ready[bf_stage], get_phasebit<0>(slice_row_ready_phasebits, bf_stage));
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            }
                        }
                        if constexpr (COL_IN_QUANTIZER) {
                            if constexpr (!G::COL_READY_PER_ROW16) {
                                if constexpr (!G::DEBUG_SKIP_COL_READY_WAIT) {
                                    if constexpr (G::USE_COL_PAIR_STAGE || (DO_ROW && C::ROW_QUANT_FROM_REGS && C::CONSUMER_DO_ROW && !C::EARLY_COL_READY)) {
                                        wait(slice_col_ready[bf_stage], get_phasebit<0>(slice_col_ready_phasebits, bf_stage));
                                    } else {
                                        wait(slice_col_ready[bf_stage], get_phasebit<0>(slice_phasebits, bf_stage));
                                    }
                                }
                            }
                        }
                    } else {
                    if constexpr (ROW_IN_QUANTIZER) {
                        if (is_row_quantizer_wg) {
                            if constexpr (!G::DEBUG_SKIP_ROW_READY_WAIT) {
                                if constexpr (G::DEBUG_ROW_WAIT_ON_COL_READY && DO_COL) {
                                    wait(slice_col_ready[bf_stage], get_phasebit<0>(slice_col_ready_phasebits, bf_stage));
                                } else {
                                    wait(slice_row_ready[bf_stage], get_phasebit<0>(slice_row_ready_phasebits, bf_stage));
                                }
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            }
                        }
                    }
                    if constexpr (COL_IN_QUANTIZER) {
                        if (is_col_quantizer_wg) {
                            if constexpr (!G::COL_READY_PER_ROW16) {
                                if constexpr (!G::DEBUG_SKIP_COL_READY_WAIT) {
                                    if constexpr (G::USE_COL_PAIR_STAGE || (DO_ROW && C::ROW_QUANT_FROM_REGS && C::CONSUMER_DO_ROW && !C::EARLY_COL_READY)) {
                                        wait(slice_col_ready[bf_stage], get_phasebit<0>(slice_col_ready_phasebits, bf_stage));
                                    } else {
                                        wait(slice_col_ready[bf_stage], get_phasebit<0>(slice_phasebits, bf_stage));
                                    }
                                }
                            }
                        }
                    }
                }
                if constexpr (COL_IN_QUANTIZER && (G::USE_COL_PLAIN_STAGE || G::USE_COL_PAIR_STAGE || C::EARLY_COL_READY || C::ROW_QUANT_FROM_REGS) && !G::COL_READY_PER_ROW16) {
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                }

                const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_epi_stage[bf_stage].D.data[0]));
                const uint32_t d_col_base = [&]() {
                    if constexpr (G::USE_COL_PLAIN_STAGE) {
                        return static_cast<uint32_t>(__cvta_generic_to_shared(&col_plain_stage[bf_stage].D.data[0]));
                    } else {
                        return d_base;
                    }
                }();
                auto load_col_value = [&](bf16 &value, int local_row, int local_col) {
                    if constexpr (G::USE_COL_PLAIN_STAGE) {
                        move<bf16>::lds(value, G::D_helper_tile::idx(d_col_base, {local_row, local_col}));
                    } else {
                        move<bf16>::lds(value, G::D_tile::idx(d_col_base, {local_row, local_col}));
                    }
                };

                if constexpr (ROW_IN_QUANTIZER) {
                    if constexpr (!G::DEBUG_DISABLE_ROW_QUANT_WORK) {
                    constexpr bool SINGLE_ROW_ONLY_LANEPAIR_FLOATCACHE =
                        G::FRONTEND_SINGLE_QUANTIZER_WG &&
                        !COL_IN_QUANTIZER &&
                        G::USE_ROW_PAIR_STAGE &&
                        G::ROW_PAIR_STAGE_ROWRECORD &&
                        G::ROW_PAIR_STAGE_FLOATCACHE &&
                        G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT;
                    if constexpr (SINGLE_ROW_ONLY_LANEPAIR_FLOATCACHE) {
                        const int quant_row = quant_thread;
                        if (quant_row < C::Mb / 2) {
                            const int global_row = tile_row_base + quant_row;
                            const int row16_block = quant_row / 16;
                            const int row_pair_slot = (quant_row % 16) / 2;
                            const int row_pair_lane = quant_row & 1;
                            const int depth = global_row / 128;
                            const int sr = global_row % 32;
                            const int rr = (global_row / 32) % 4;
                            const int row_chunk_base = depth * row_sc_kgroups;
                            const bool row_in_bounds = full_tile_rows || global_row < g.M;
                            #pragma unroll
                            for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                const uint64_t pair_bits_0 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][0][row_pair_lane];
                                const uint64_t pair_bits_1 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][1][row_pair_lane];
                                const uint64_t pair_bits_2 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][2][row_pair_lane];
                                const uint64_t pair_bits_3 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][3][row_pair_lane];
                                if (row_in_bounds) {
                                    float2 cached_pairs[8];
                                    const uint64_t pair_bits_arr[4] = {
                                        pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3
                                    };
                                    float amax = 0.0f;
                                    #pragma unroll
                                    for (int pair = 0; pair < 4; ++pair) {
                                        const uint64_t pair_bits = pair_bits_arr[pair];
                                        const bf16_2 pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                        const bf16_2 pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                        const float2 pair_lo = make_float2(
                                            __bfloat162float(pair_lo_bf.x),
                                            __bfloat162float(pair_lo_bf.y));
                                        const float2 pair_hi = make_float2(
                                            __bfloat162float(pair_hi_bf.x),
                                            __bfloat162float(pair_hi_bf.y));
                                        cached_pairs[pair] = pair_lo;
                                        cached_pairs[pair + 4] = pair_hi;
                                        amax = fmaxf(amax, fabsf(pair_lo.x));
                                        amax = fmaxf(amax, fabsf(pair_lo.y));
                                        amax = fmaxf(amax, fabsf(pair_hi.x));
                                        amax = fmaxf(amax, fabsf(pair_hi.y));
                                    }
                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                    const int global_col_16 = col_start + group16 * 16;
                                    const int fp4x2_col_base = global_col_16 / 2;
                                    uint64_t packed_fp4 = 0;
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                            cached_pairs[pair].x,
                                            cached_pairs[pair].y,
                                            rcp_scale)) << (pair * 8);
                                    }
                                    store_global_u64(
                                        &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                        packed_fp4);
                                    const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                    float stored_scale = scale * g_sg_rcp;
                                    if (encode_centric) {
                                        stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                    }
                                    const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                    const int kgroup = global_col_16 / 64;
                                    const int col_16_in_64 = (global_col_16 / 16) % 4;
                                    const int chunk = row_chunk_base + kgroup;
                                    const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                    row_sc_ptr[chunk * 512 + byte_idx] =
                                        *reinterpret_cast<const uint8_t*>(&sc);
                                }
                            }
                        }
                    } else if constexpr (G::FRONTEND_SINGLE_QUANTIZER_WG) {
                        constexpr int ROW_THREADS = QUANTIZER_WG_THREADS / 2;
                        if (quant_thread < ROW_THREADS) {
                            const int quant_row = quant_thread;
                            if (quant_row < C::Mb / 2) {
                                const int global_row = tile_row_base + quant_row;
                                #pragma unroll
                                for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                    bf16_2 vals[8];
                                    float amax = 0.0f;
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        const int col = group16 * 16 + pair * 2;
                                        move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                                    }
                                    const float scale = amax * (1.0f / FP4_MAX);
                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                    if (global_row < g.M) {
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const uint8_t fp4_pair = quantize_fp4_pair(
                                                __bfloat162float(vals[pair].x),
                                                __bfloat162float(vals[pair].y),
                                                rcp_scale);
                                            const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                            row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                        }
                                        float stored_scale = scale * g_sg_rcp;
                                        if (encode_centric) {
                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                        }
                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                        const int global_col_16 = col_start + group16 * 16;
                                        const int kgroup = global_col_16 / 64;
                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                        const int depth = global_row / 128;
                                        const int sr = global_row % 32;
                                        const int rr = (global_row / 32) % 4;
                                        const int chunk = depth * row_sc_kgroups + kgroup;
                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                        row_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&sc);
                                    }
                                }
                            }
                        }
                    } else if (is_row_quantizer_wg) {
                        if constexpr (G::USE_ROW_PAIR_STAGE &&
                                      G::ROW_PAIR_STAGE_ROWRECORD &&
                                      G::ROW_PAIR_STAGE_FLOATCACHE &&
                                      G::ROW_PAIR_STAGE_LANEPAIR_LAYOUT) {
                            const int quant_row = quant_thread;
                            if (quant_row < C::Mb / 2) {
                                const int global_row = tile_row_base + quant_row;
                                const int row16_block = quant_row / 16;
                                const int row_pair_slot = (quant_row % 16) / 2;
                                const int row_pair_lane = quant_row & 1;
                                const int depth = global_row / 128;
                                const int sr = global_row % 32;
                                const int rr = (global_row / 32) % 4;
                                const int row_chunk_base = depth * row_sc_kgroups;
                                const bool row_in_bounds = full_tile_rows || global_row < g.M;
                                #pragma unroll
                                for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                    const uint64_t pair_bits_0 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][0][row_pair_lane];
                                    const uint64_t pair_bits_1 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][1][row_pair_lane];
                                    const uint64_t pair_bits_2 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][2][row_pair_lane];
                                    const uint64_t pair_bits_3 = row_pair_stage[bf_stage].lanepair_records[row16_block][group16][row_pair_slot][3][row_pair_lane];
                                    if (row_in_bounds) {
                                        float2 cached_pairs[8];
                                        const uint64_t pair_bits_arr[4] = {
                                            pair_bits_0, pair_bits_1, pair_bits_2, pair_bits_3
                                        };
                                        float amax = 0.0f;
                                        #pragma unroll
                                        for (int pair = 0; pair < 4; ++pair) {
                                            const uint64_t pair_bits = pair_bits_arr[pair];
                                            const bf16_2 pair_lo_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits));
                                            const bf16_2 pair_hi_bf = bf16x2_from_bits(static_cast<uint32_t>(pair_bits >> 32));
                                            const float2 pair_lo = make_float2(
                                                __bfloat162float(pair_lo_bf.x),
                                                __bfloat162float(pair_lo_bf.y));
                                            const float2 pair_hi = make_float2(
                                                __bfloat162float(pair_hi_bf.x),
                                                __bfloat162float(pair_hi_bf.y));
                                            cached_pairs[pair] = pair_lo;
                                            cached_pairs[pair + 4] = pair_hi;
                                            amax = fmaxf(amax, fabsf(pair_lo.x));
                                            amax = fmaxf(amax, fabsf(pair_lo.y));
                                            amax = fmaxf(amax, fabsf(pair_hi.x));
                                            amax = fmaxf(amax, fabsf(pair_hi.y));
                                        }
                                        const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                        const int global_col_16 = col_start + group16 * 16;
                                        const int fp4x2_col_base = global_col_16 / 2;
                                        uint64_t packed_fp4 = 0;
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                cached_pairs[pair].x,
                                                cached_pairs[pair].y,
                                                rcp_scale)) << (pair * 8);
                                        }
                                        store_global_u64(
                                            &row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col_base],
                                            packed_fp4);
                                        const float scale = rcp_scale > 0.0f ? (1.0f / rcp_scale) : 0.0f;
                                        float stored_scale = scale * g_sg_rcp;
                                        if (encode_centric) {
                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                        }
                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                        const int kgroup = global_col_16 / 64;
                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                        const int chunk = row_chunk_base + kgroup;
                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                        row_sc_ptr[chunk * 512 + byte_idx] =
                                            *reinterpret_cast<const uint8_t*>(&sc);
                                    }
                                }
                            }
                        } else {
                            const int quant_row = quant_thread;
                            if (quant_row < C::Mb / 2) {
                                const int global_row = tile_row_base + quant_row;
                                #pragma unroll
                                for (int group16 = 0; group16 < SUBTILE_COLS / 16; ++group16) {
                                    bf16_2 vals[8];
                                    float amax = 0.0f;
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        const int col = group16 * 16 + pair * 2;
                                        move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                        amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                                    }
                                    const float scale = amax * (1.0f / FP4_MAX);
                                    const float rcp_scale = (amax > 0.0f) ? (FP4_MAX / amax) : 0.0f;
                                    if (global_row < g.M) {
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const uint8_t fp4_pair = quantize_fp4_pair(
                                                __bfloat162float(vals[pair].x),
                                                __bfloat162float(vals[pair].y),
                                                rcp_scale);
                                            const int fp4x2_col = (col_start + group16 * 16 + pair * 2) / 2;
                                            row_fp4_ptr[global_row * row_fp4_stride + fp4x2_col] = fp4_pair;
                                        }
                                        float stored_scale = scale * g_sg_rcp;
                                        if (encode_centric) {
                                            stored_scale = fminf(rcp_scale * g_sg, E4M3_MAX);
                                        }
                                        const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(stored_scale);
                                        const int global_col_16 = col_start + group16 * 16;
                                        const int kgroup = global_col_16 / 64;
                                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                                        const int depth = global_row / 128;
                                        const int sr = global_row % 32;
                                        const int rr = (global_row / 32) % 4;
                                        const int chunk = depth * row_sc_kgroups + kgroup;
                                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                                        row_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&sc);
                                    }
                                }
                            }
                        }
                    }
                    }
                }

                if constexpr (COL_IN_QUANTIZER) {
                    if constexpr (!G::DEBUG_DISABLE_COL_QUANT_WORK) {
                    if constexpr (G::FRONTEND_SINGLE_QUANTIZER_WG) {
                        constexpr int ROW_THREADS = ROW_IN_QUANTIZER ? (QUANTIZER_WG_THREADS / 2) : 0;
                        constexpr int COL_THREADS = QUANTIZER_WG_THREADS - ROW_THREADS;
                        constexpr int COL_ROW16_BLOCKS_PER_PASS = COL_THREADS / SUBTILE_COLS;
                        static_assert(COL_ROW16_BLOCKS_PER_PASS > 0);
                        const int col_thread = quant_thread - ROW_THREADS;
                        if (col_thread >= 0) {
                            const int col_in_epi = col_thread % SUBTILE_COLS;
                            const int row16_block_base = col_thread / SUBTILE_COLS;
                            #pragma unroll
                            for (int row16_pass = 0; row16_pass < (ROW16_BLOCKS + COL_ROW16_BLOCKS_PER_PASS - 1) / COL_ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                                const int row16_block = row16_block_base + row16_pass * COL_ROW16_BLOCKS_PER_PASS;
                                const int global_col = col_start + col_in_epi;
                                    if (row16_block < ROW16_BLOCKS && global_col < g.N) {
                                        if constexpr (G::COL_READY_PER_ROW16) {
                                            wait(slice_col_ready_row16[bf_stage][row16_block], get_phasebit<0>(slice_col_ready_phasebits, bf_stage));
                                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                        }
                                    const int local_row_base = row16_block * 16;
                                    const int global_row_base = tile_row_base + local_row_base;
                                    const int global_row_pair_base = global_row_base / 2;
                                    const int depth = global_col / 128;
                                    const int sr = global_col % 32;
                                    const int rr = (global_col / 32) % 4;
                                    const int m_kgroup = global_row_base / 64;
                                    const int m_16_in_64 = (global_row_base / 16) % 4;
                                    const int chunk = depth * col_sc_kgroups + m_kgroup;
                                    const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                    if constexpr (G::USE_COL_PAIR_STAGE) {
                                        const int col_pair_idx = col_in_epi / 2;
                                        const int col_pair_lane = col_in_epi % 2;
                                        float col_amax = 0.0f;
                                        uint64_t packed_fp4 = 0;
                                        if constexpr (C::CACHE_COL_VALUES) {
                                            float2 cached_pairs[8];
                                            #pragma unroll
                                            for (int pair = 0; pair < 8; ++pair) {
                                                const int global_row = global_row_base + pair * 2;
                                                const bf16_2 bf_pair =
                                                    load_col_pair_stage_pair<G>(col_pair_stage, bf_stage, row16_block, col_pair_idx, pair, col_pair_lane);
                                                const float v0 = (global_row < g.M) ? __bfloat162float(bf_pair.x) : 0.0f;
                                                const float v1 = (global_row + 1 < g.M) ? __bfloat162float(bf_pair.y) : 0.0f;
                                                cached_pairs[pair] = make_float2(v0, v1);
                                                col_amax = fmaxf(col_amax, fabsf(v0));
                                                col_amax = fmaxf(col_amax, fabsf(v1));
                                            }
                                            const float col_scale = col_amax * (1.0f / FP4_MAX);
                                            const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                            #pragma unroll
                                            for (int pair = 0; pair < 8; ++pair) {
                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(
                                                    cached_pairs[pair].x,
                                                    cached_pairs[pair].y,
                                                    col_rcp)) << (pair * 8);
                                            }
                                            if constexpr (G::PACK_COL_FP4_U64) {
                                                store_global_u64(col_fp4_ptr + global_col * col_fp4_stride + global_row_pair_base, packed_fp4);
                                            } else {
                                                #pragma unroll
                                                for (int pair = 0; pair < 8; ++pair) {
                                                    col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                        static_cast<uint8_t>(packed_fp4 >> (pair * 8));
                                                }
                                            }
                                            if (combo_do_dc) {
                                                const int local_col = epi * SUBTILE_COLS + col_in_epi;
                                                const int local_row_pair_base = local_row_base / 2;
                                                store_combo_col_stage_u64<G>(
                                                    combo_col_stage_slot,
                                                    local_col,
                                                    local_row_pair_base,
                                                    packed_fp4);
                                            }
                                            float stored_scale = col_scale * g_sg_rcp;
                                            if (encode_centric) {
                                                stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                            }
                                            const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                                const uint8_t csc_byte = *reinterpret_cast<const uint8_t*>(&csc);
                                                col_sc_ptr[chunk * 512 + byte_idx] = csc_byte;
                                                if (combo_do_dc) {
                                                    const int local_col = epi * SUBTILE_COLS + col_in_epi;
                                                    store_combo_col_stage_scale<G>(
                                                        combo_col_stage_slot,
                                                        local_col,
                                                        local_row_base,
                                                        csc_byte);
                                                }
                                        } else {
                                            bf16_2 cached_pairs[8];
                                            #pragma unroll
                                            for (int pair = 0; pair < 8; ++pair) {
                                                cached_pairs[pair] =
                                                    load_col_pair_stage_pair<G>(col_pair_stage, bf_stage, row16_block, col_pair_idx, pair, col_pair_lane);
                                                const float v0 = __bfloat162float(cached_pairs[pair].x);
                                                const float v1 = (global_row_base + pair * 2 + 1 < g.M) ? __bfloat162float(cached_pairs[pair].y) : 0.0f;
                                                col_amax = fmaxf(col_amax, fabsf(v0));
                                                col_amax = fmaxf(col_amax, fabsf(v1));
                                            }
                                            const float col_scale = col_amax * (1.0f / FP4_MAX);
                                            const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                            #pragma unroll
                                            for (int pair = 0; pair < 8; ++pair) {
                                                const int global_row = global_row_base + pair * 2;
                                                const float v0 = (global_row < g.M) ? __bfloat162float(cached_pairs[pair].x) : 0.0f;
                                                const float v1 = (global_row + 1 < g.M) ? __bfloat162float(cached_pairs[pair].y) : 0.0f;
                                                packed_fp4 |= static_cast<uint64_t>(quantize_fp4_pair(v0, v1, col_rcp)) << (pair * 8);
                                            }
                                            if constexpr (G::PACK_COL_FP4_U64) {
                                                store_global_u64(col_fp4_ptr + global_col * col_fp4_stride + global_row_pair_base, packed_fp4);
                                            } else {
                                                #pragma unroll
                                                for (int pair = 0; pair < 8; ++pair) {
                                                    col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                        static_cast<uint8_t>(packed_fp4 >> (pair * 8));
                                                }
                                            }
                                            if (combo_do_dc) {
                                                const int local_col = epi * SUBTILE_COLS + col_in_epi;
                                                const int local_row_pair_base = local_row_base / 2;
                                                store_combo_col_stage_u64<G>(
                                                    combo_col_stage_slot,
                                                    local_col,
                                                    local_row_pair_base,
                                                    packed_fp4);
                                            }
                                            float stored_scale = col_scale * g_sg_rcp;
                                            if (encode_centric) {
                                                stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                            }
                                            const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                                const uint8_t csc_byte = *reinterpret_cast<const uint8_t*>(&csc);
                                                col_sc_ptr[chunk * 512 + byte_idx] = csc_byte;
                                                if (combo_do_dc) {
                                                    const int local_col = epi * SUBTILE_COLS + col_in_epi;
                                                    store_combo_col_stage_scale<G>(
                                                        combo_col_stage_slot,
                                                        local_col,
                                                        local_row_base,
                                                        csc_byte);
                                                }
                                        }
                                        if constexpr (DO_ROW && G::ROW_WAITS_FOR_COL_DONE_ROW16) {
                                            __syncwarp();
                                            if (col_in_epi == 0) {
                                                __threadfence_block();
                                                slice_col_done_row16[bf_stage][row16_block] = 1;
                                            }
                                        }
                                    } else {
                                        const bool full_row16 = (global_row_base + 15) < g.M;
                                        float cached_vals[16];
                                        bf16_2 cached_pairs[8];
                                        float col_amax = 0.0f;
                                        if constexpr (C::CACHE_COL_VALUES_BF16_PAIRS) {
                                            if (full_row16) {
                                                #pragma unroll
                                                for (int pair = 0; pair < 8; ++pair) {
                                                    load_col_value(cached_pairs[pair].x, local_row_base + pair * 2, col_in_epi);
                                                    load_col_value(cached_pairs[pair].y, local_row_base + pair * 2 + 1, col_in_epi);
                                                    col_amax = fmaxf(col_amax, fabsf(__bfloat162float(cached_pairs[pair].x)));
                                                    col_amax = fmaxf(col_amax, fabsf(__bfloat162float(cached_pairs[pair].y)));
                                                }
                                            } else {
                                                #pragma unroll
                                                for (int pair = 0; pair < 8; ++pair) {
                                                    const int global_row = global_row_base + pair * 2;
                                                    bf16 value0_bf;
                                                    load_col_value(value0_bf, local_row_base + pair * 2, col_in_epi);
                                                    cached_pairs[pair].x = value0_bf;
                                                    cached_pairs[pair].y = __float2bfloat16(0.0f);
                                                    const float v0 = (global_row < g.M) ? __bfloat162float(value0_bf) : 0.0f;
                                                    float v1 = 0.0f;
                                                    if (global_row + 1 < g.M) {
                                                        bf16 value1_bf;
                                                        load_col_value(value1_bf, local_row_base + pair * 2 + 1, col_in_epi);
                                                        cached_pairs[pair].y = value1_bf;
                                                        v1 = __bfloat162float(value1_bf);
                                                    }
                                                    col_amax = fmaxf(col_amax, fabsf(v0));
                                                    col_amax = fmaxf(col_amax, fabsf(v1));
                                                }
                                            }
                                        } else {
                                            #pragma unroll
                                            for (int r = 0; r < 16; ++r) {
                                                bf16 value;
                                                load_col_value(value, local_row_base + r, col_in_epi);
                                                const float fv = (global_row_base + r < g.M) ? __bfloat162float(value) : 0.0f;
                                                if constexpr (C::CACHE_COL_VALUES) {
                                                    cached_vals[r] = fv;
                                                }
                                                col_amax = fmaxf(col_amax, fabsf(fv));
                                            }
                                        }
                                        const float col_scale = col_amax * (1.0f / FP4_MAX);
                                        const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                        if constexpr (C::FAST_ALIGNED_QUANT) {
                                            if (full_row16) {
                                                #pragma unroll
                                                for (int pair = 0; pair < 8; ++pair) {
                                                    bf16 value0_bf;
                                                    bf16 value1_bf;
                                                    load_col_value(value0_bf, local_row_base + pair * 2, col_in_epi);
                                                    load_col_value(value1_bf, local_row_base + pair * 2 + 1, col_in_epi);
                                                    col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                        quantize_fp4_pair(__bfloat162float(value0_bf),
                                                                          __bfloat162float(value1_bf),
                                                                          col_rcp);
                                                }
                                                float stored_scale = col_scale * g_sg_rcp;
                                                if (encode_centric) {
                                                    stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                                }
                                                const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                                col_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&csc);
                                                continue;
                                            }
                                        }
                                        #pragma unroll
                                        for (int pair = 0; pair < 8; ++pair) {
                                            const int global_row = global_row_base + pair * 2;
                                            if constexpr (C::CACHE_COL_VALUES_BF16_PAIRS) {
                                                if (full_row16) {
                                                    col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                        quantize_fp4_pair(__bfloat162float(cached_pairs[pair].x),
                                                                          __bfloat162float(cached_pairs[pair].y),
                                                                          col_rcp);
                                                    continue;
                                                }
                                            }
                                            if (global_row < g.M) {
                                                float v0, v1;
                                                if constexpr (C::CACHE_COL_VALUES_BF16_PAIRS) {
                                                    v0 = __bfloat162float(cached_pairs[pair].x);
                                                    v1 = (global_row + 1 < g.M) ? __bfloat162float(cached_pairs[pair].y) : 0.0f;
                                                } else if constexpr (C::CACHE_COL_VALUES) {
                                                    v0 = cached_vals[pair * 2];
                                                    v1 = cached_vals[pair * 2 + 1];
                                                } else {
                                                    bf16 value0_bf;
                                                    load_col_value(value0_bf, local_row_base + pair * 2, col_in_epi);
                                                    v0 = __bfloat162float(value0_bf);
                                                    v1 = 0.0f;
                                                    if (global_row + 1 < g.M) {
                                                        bf16 value1_bf;
                                                        load_col_value(value1_bf, local_row_base + pair * 2 + 1, col_in_epi);
                                                        v1 = __bfloat162float(value1_bf);
                                                    }
                                                }
                                                col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                                    quantize_fp4_pair(v0, v1, col_rcp);
                                            }
                                        }
                                        float stored_scale = col_scale * g_sg_rcp;
                                        if (encode_centric) {
                                            stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                        }
                                        const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                        col_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&csc);
                                    }
                                    if constexpr (DO_ROW && G::ROW_WAITS_FOR_COL_DONE_ROW16) {
                                        __syncwarp();
                                        if (col_in_epi == 0) {
                                            __threadfence_block();
                                            slice_col_done_row16[bf_stage][row16_block] = 1;
                                        }
                                    }
                                }
                            }
                        }
                    } else if (is_col_quantizer_wg) {
                        constexpr int COL_THREADS = QUANTIZER_WG_THREADS * C::COL_QUANTIZER_WARPGROUPS;
                        constexpr int COL_ROW16_BLOCKS_PER_PASS = COL_THREADS / SUBTILE_COLS;
                        static_assert(COL_ROW16_BLOCKS_PER_PASS > 0);
                        const int col_quantizer_rank = warpgroup_id - first_col_quantizer_wg;
                        const int col_thread = col_quantizer_rank * QUANTIZER_WG_THREADS + quant_thread;
                        const int col_in_epi = col_thread % SUBTILE_COLS;
                        const int row16_block_base = col_thread / SUBTILE_COLS;
                        #pragma unroll
                        for (int row16_pass = 0; row16_pass < (ROW16_BLOCKS + COL_ROW16_BLOCKS_PER_PASS - 1) / COL_ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                            const int row16_block = row16_block_base + row16_pass * COL_ROW16_BLOCKS_PER_PASS;
                            const int global_col = col_start + col_in_epi;
                            if (row16_block < ROW16_BLOCKS && global_col < g.N) {
                                if constexpr (G::COL_READY_PER_ROW16) {
                                    wait(slice_col_ready_row16[bf_stage][row16_block], get_phasebit<0>(slice_col_ready_phasebits, bf_stage));
                                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                }
                                const int local_row_base = row16_block * 16;
                                const int global_row_base = tile_row_base + local_row_base;
                                const int global_row_pair_base = global_row_base / 2;
                                const int depth = global_col / 128;
                                const int sr = global_col % 32;
                                const int rr = (global_col / 32) % 4;
                                const int m_kgroup = global_row_base / 64;
                                const int m_16_in_64 = (global_row_base / 16) % 4;
                                const int chunk = depth * col_sc_kgroups + m_kgroup;
                                const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
                                float cached_vals[16];
                                bf16_2 cached_pairs[8];
                                float col_amax = 0.0f;
                                if constexpr (C::CACHE_COL_VALUES_BF16_PAIRS) {
                                    #pragma unroll
                                    for (int pair = 0; pair < 8; ++pair) {
                                        const int global_row = global_row_base + pair * 2;
                                        bf16 value0_bf;
                                        load_col_value(value0_bf, local_row_base + pair * 2, col_in_epi);
                                        cached_pairs[pair].x = value0_bf;
                                        cached_pairs[pair].y = __float2bfloat16(0.0f);
                                        const float v0 = (global_row < g.M) ? __bfloat162float(value0_bf) : 0.0f;
                                        float v1 = 0.0f;
                                        if (global_row + 1 < g.M) {
                                            bf16 value1_bf;
                                            load_col_value(value1_bf, local_row_base + pair * 2 + 1, col_in_epi);
                                            cached_pairs[pair].y = value1_bf;
                                            v1 = __bfloat162float(value1_bf);
                                        }
                                        col_amax = fmaxf(col_amax, fabsf(v0));
                                        col_amax = fmaxf(col_amax, fabsf(v1));
                                    }
                                } else {
                                    #pragma unroll
                                    for (int r = 0; r < 16; ++r) {
                                        bf16 value;
                                        load_col_value(value, local_row_base + r, col_in_epi);
                                        const float fv = (global_row_base + r < g.M) ? __bfloat162float(value) : 0.0f;
                                        if constexpr (C::CACHE_COL_VALUES) {
                                            cached_vals[r] = fv;
                                        }
                                        col_amax = fmaxf(col_amax, fabsf(fv));
                                    }
                                }
                                const float col_scale = col_amax * (1.0f / FP4_MAX);
                                const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const int global_row = global_row_base + pair * 2;
                                    if (global_row < g.M) {
                                        float v0, v1;
                                        if constexpr (C::CACHE_COL_VALUES_BF16_PAIRS) {
                                            v0 = __bfloat162float(cached_pairs[pair].x);
                                            v1 = (global_row + 1 < g.M) ? __bfloat162float(cached_pairs[pair].y) : 0.0f;
                                        } else if constexpr (C::CACHE_COL_VALUES) {
                                            v0 = cached_vals[pair * 2];
                                            v1 = cached_vals[pair * 2 + 1];
                                        } else {
                                            bf16 value0_bf;
                                            load_col_value(value0_bf, local_row_base + pair * 2, col_in_epi);
                                            v0 = __bfloat162float(value0_bf);
                                            v1 = 0.0f;
                                            if (global_row + 1 < g.M) {
                                                bf16 value1_bf;
                                                load_col_value(value1_bf, local_row_base + pair * 2 + 1, col_in_epi);
                                                v1 = __bfloat162float(value1_bf);
                                            }
                                        }
                                        col_fp4_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                                            quantize_fp4_pair(v0, v1, col_rcp);
                                    }
                                }
                                float stored_scale = col_scale * g_sg_rcp;
                                if (encode_centric) {
                                    stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
                                }
                                const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
                                col_sc_ptr[chunk * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&csc);
                            }
                            if constexpr (DO_ROW && G::ROW_WAITS_FOR_COL_DONE_ROW16) {
                                __syncwarp();
                                if (col_in_epi == 0) {
                                    __threadfence_block();
                                    slice_col_done_row16[bf_stage][row16_block] = 1;
                                }
                            }
                        }
                    }
                    }
                }

                if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                    constexpr int combo_de_a_scale_chunks = C::Nb / 64;
                    constexpr int combo_de_b_scale_chunks = C::Nb / 128;
                    constexpr int combo_dc_a_scale_chunks = C::Nb / 64;
                    constexpr int combo_dc_b_scale_chunks = C::Nb / 64;
                    using combo_de_rt = rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
                    using combo_de_rt_bf = rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
                    using combo_dc_rt = rt_fl<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;
                    using combo_dc_rt_bf = rt_bf<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;
                    const bool combo_issue_leader = (warpgroup::warpid() == 0) && (quant_lane == 0);
                    auto combo_de_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(256);
                    auto combo_dc_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(384);
                    auto combo_de_a_sc_tm =
                        tm_allocator.template allocate<full_tt_fp8e4m3<16 * combo_de_a_scale_chunks>>(512);
                    auto combo_de_b_sc_tm =
                        tm_allocator.template allocate<full_tt_fp8e4m3<32 * combo_de_b_scale_chunks>>(
                            512 + 4 * combo_de_a_scale_chunks);
                    auto combo_dc_a_sc_tm =
                        tm_allocator.template allocate<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(640);
                    auto combo_dc_b_sc_tm =
                        tm_allocator.template allocate<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(
                            640 + 4 * combo_dc_a_scale_chunks);

                    if (combo_stage_publish_epi && is_de_consumer_wg && combo_do_de) {
                        const float combo_de_scale = g_sg * g.C_col_sc_global[{0}];
                        if constexpr (!combo_debug_skip_sep_de_row_sc_preload) {
                            #pragma unroll
                            for (int ii = 0; ii < combo_de_a_scale_chunks; ++ii) {
                                auto combo_a_sc_sub =
                                    combo_de_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                auto &combo_g_sc_sm_sub =
                                    *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                        reinterpret_cast<uint64_t>(&combo_row_stage->G_row_sc.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(combo_a_sc_sub, combo_g_sc_sm_sub);
                            }
                        }

                        int combo_de_c_tiles_phase = 0;
                        int combo_de_c_scales_phase = 0;
                        int combo_de_inputs_phase = 1;
                        int combo_de_outputs_phase = 0;
                        int prev_k_block_idx = -1;
                        const int combo_num_k_blocks =
                            combo_debug_skip_sep_de_body ? 0 : (g.dE_out.cols() / C::Nb);

                        #pragma unroll 1
                        for (int k_block_idx = 0; k_block_idx < combo_num_k_blocks; ++k_block_idx) {
                            if (prev_k_block_idx >= 0) {
                                if constexpr (!combo_debug_skip_sep_de_drain) {
                                    if constexpr (!combo_debug_skip_sep_de_output_wait) {
                                        wait(combo_de_p3_outputs_arrived, combo_de_outputs_phase);
                                        combo_de_outputs_phase ^= 1;
                                    }
                                    #pragma unroll
                                    for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                                        combo_de_rt D_reg_fl;
                                        combo_de_rt_bf D_reg_bf;
                                        warpgroup::tma::store_async_read_wait<0>();
                                        warpgroup::sync(quantizer_sync_id);
                                        warpgroup::load_async(
                                            D_reg_fl,
                                            combo_de_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                                0,
                                                combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                                        tensor_load_wait();
                                        tensor_before_thread_sync();
                                        warpgroup::sync(quantizer_sync_id);
                                        warp::mul(D_reg_fl, D_reg_fl, combo_de_scale);
                                        warp::copy(D_reg_bf, D_reg_fl);
                                        warpgroup::store(combo_de_output_stage->dE, D_reg_bf);
                                        warpgroup::sync(quantizer_sync_id);
                                        warpgroup::tma::store_add_async(
                                            g.dE_out, combo_de_output_stage->dE,
                                            {row_block_idx * 2 + cta_id, prev_k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                                        tensor_after_thread_sync();
                                    }
                                    warpgroup::tma::store_async_read_wait<0>();
                                }
                            }

                            if (combo_issue_leader) {
                                if constexpr (!combo_debug_skip_sep_de_input_wait) {
                                    wait(combo_de_p3_inputs_finished, combo_de_inputs_phase);
                                    combo_de_inputs_phase ^= 1;
                                }
                                tma::expect_bytes(combo_de_p3_c_tiles_arrived, sizeof(typename G::combo_p3_C_tile));
                                tma::load_async(
                                    combo_p3_c_tile_stage->C_operand, g.C_col,
                                    {k_block_idx * 2 + cta_id, col_block_idx},
                                    combo_de_p3_c_tiles_arrived);
                                tma::load_async(
                                    combo_de_scales_stage->C_sc, g.C_col_sc,
                                    {k_block_idx, col_block_idx, 0},
                                    combo_de_p3_c_scales_arrived);
                            }
                            warpgroup::sync(quantizer_sync_id);

                            if (combo_issue_leader) {
                                tma::expect_bytes(combo_de_p3_c_scales_arrived, sizeof(typename G::combo_p3_C_sc_tile));
                                if constexpr (!combo_debug_skip_sep_de_scale_wait) {
                                    wait(combo_de_p3_c_scales_arrived, combo_de_c_scales_phase);
                                    combo_de_c_scales_phase ^= 1;
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);

                            #pragma unroll
                            for (int ii = 0; ii < combo_de_b_scale_chunks; ++ii) {
                                auto combo_b_sc_sub =
                                    combo_de_b_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                auto &combo_c_sc_sm_sub =
                                    *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                        reinterpret_cast<uint64_t>(&combo_de_scales_stage->C_sc.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(combo_b_sc_sub, combo_c_sc_sm_sub);
                            }

                            if (combo_issue_leader) {
                                if constexpr (!combo_debug_skip_sep_de_tile_wait) {
                                    wait(combo_de_p3_c_tiles_arrived, combo_de_c_tiles_phase);
                                    combo_de_c_tiles_phase ^= 1;
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);

                            mm2_ABt(
                                combo_de_tm, combo_row_stage->G_row, combo_p3_c_tile_stage->C_operand,
                                combo_de_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_de_a_scale_chunks>>(0),
                                combo_de_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_de_b_scale_chunks>>(0),
                                combo_de_p3_inputs_finished);
                            tensor_commit<2>(combo_de_p3_outputs_arrived);
                            tensor_after_thread_sync();
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                            warpgroup::sync(quantizer_sync_id);
                            prev_k_block_idx = k_block_idx;
                        }

                        if (prev_k_block_idx >= 0) {
                            if constexpr (!combo_debug_skip_sep_de_drain) {
                                if constexpr (!combo_debug_skip_sep_de_output_wait) {
                                    wait(combo_de_p3_outputs_arrived, combo_de_outputs_phase);
                                    combo_de_outputs_phase ^= 1;
                                }
                                #pragma unroll
                                for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                                    combo_de_rt D_reg_fl;
                                    combo_de_rt_bf D_reg_bf;
                                    warpgroup::tma::store_async_read_wait<0>();
                                    warpgroup::sync(quantizer_sync_id);
                                    warpgroup::load_async(
                                        D_reg_fl,
                                        combo_de_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                            0,
                                            combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                                    tensor_load_wait();
                                    tensor_before_thread_sync();
                                    warpgroup::sync(quantizer_sync_id);
                                    warp::mul(D_reg_fl, D_reg_fl, combo_de_scale);
                                    warp::copy(D_reg_bf, D_reg_fl);
                                    warpgroup::store(combo_de_output_stage->dE, D_reg_bf);
                                    warpgroup::sync(quantizer_sync_id);
                                    warpgroup::tma::store_add_async(
                                        g.dE_out, combo_de_output_stage->dE,
                                        {row_block_idx * 2 + cta_id, prev_k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                                    tensor_after_thread_sync();
                                }
                                warpgroup::tma::store_async_read_wait<0>();
                            }
                        }
                        if (combo_do_dc) {
                            warpgroup::sync(quantizer_sync_id);
                            if (warpgroup::warpid() == 0 && quant_lane == 0) {
                                arrive(combo_backhalf_de_done);
                            }
                        }
                    }

                    if (combo_stage_publish_epi && is_dc_consumer_wg && combo_do_dc) {
                        if (combo_do_de) {
                            wait(combo_backhalf_de_done, combo_backhalf_de_done_phase);
                            combo_backhalf_de_done_phase ^= 1;
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        }
                        const float combo_dc_scale = g_sg * g.E_col_sc_global[{0}];
                        #pragma unroll
                        for (int ii = 0; ii < combo_dc_a_scale_chunks; ++ii) {
                            auto combo_a_sc_sub =
                                combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                            auto &combo_gt_sc_sm_sub =
                                *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                    reinterpret_cast<uint64_t>(&combo_col_stage->Gt_row_sc.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(combo_a_sc_sub, combo_gt_sc_sm_sub);
                        }

                        int combo_dc_e_tiles_phase = 0;
                        int combo_dc_e_scales_phase = 0;
                        int combo_dc_inputs_phase = 1;
                        int combo_dc_outputs_phase = 0;
                        int prev_k_block_idx = -1;
                        const int combo_num_k_blocks = g.dC_out.cols() / C::Nb;

                        #pragma unroll 1
                        for (int k_block_idx = 0; k_block_idx < combo_num_k_blocks; ++k_block_idx) {
                            if (prev_k_block_idx >= 0) {
                                wait(combo_dc_p3_outputs_arrived, combo_dc_outputs_phase);
                                combo_dc_outputs_phase ^= 1;
                                #pragma unroll
                                for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                                    combo_dc_rt D_reg_fl;
                                    combo_dc_rt_bf D_reg_bf;
                                    warpgroup::tma::store_async_read_wait<0>();
                                    warpgroup::sync(quantizer_sync_id);
                                    warpgroup::load_async(
                                        D_reg_fl,
                                        combo_dc_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                            0,
                                            combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                                    tensor_load_wait();
                                    tensor_before_thread_sync();
                                    warpgroup::sync(quantizer_sync_id);
                                    warp::mul(D_reg_fl, D_reg_fl, combo_dc_scale);
                                    warp::copy(D_reg_bf, D_reg_fl);
                                    warpgroup::store(combo_dc_output_stage->dC, D_reg_bf);
                                    warpgroup::sync(quantizer_sync_id);
                                    warpgroup::tma::store_add_async(
                                        g.dC_out, combo_dc_output_stage->dC,
                                        {col_block_idx, prev_k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                                    tensor_after_thread_sync();
                                }
                                warpgroup::tma::store_async_read_wait<0>();
                            }

                            if (combo_issue_leader) {
                                wait(combo_dc_p3_inputs_finished, combo_dc_inputs_phase);
                                combo_dc_inputs_phase ^= 1;
                                tma::expect_bytes(combo_dc_p3_e_tiles_arrived, sizeof(typename G::combo_p3_E_tile));
                                tma::load_async(
                                    combo_dc_tile_stage->E_operand, g.E_col,
                                    {k_block_idx * 2 + cta_id, row_block_idx},
                                    combo_dc_p3_e_tiles_arrived);
                                tma::load_async(
                                    combo_dc_scales_stage->E_sc, g.E_col_sc,
                                    {k_block_idx, row_block_idx, 0},
                                    combo_dc_p3_e_scales_arrived);
                            }
                            warpgroup::sync(quantizer_sync_id);

                            if (combo_issue_leader) {
                                tma::expect_bytes(combo_dc_p3_e_scales_arrived, sizeof(typename G::combo_p3_E_sc_tile));
                                wait(combo_dc_p3_e_scales_arrived, combo_dc_e_scales_phase);
                                combo_dc_e_scales_phase ^= 1;
                            }
                            warpgroup::sync(quantizer_sync_id);

                            #pragma unroll
                            for (int ii = 0; ii < combo_dc_b_scale_chunks; ++ii) {
                                auto combo_b_sc_sub =
                                    combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                auto &combo_e_sc_sm_sub =
                                    *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                        reinterpret_cast<uint64_t>(&combo_dc_scales_stage->E_sc.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
                            }

                            if (combo_issue_leader) {
                                wait(combo_dc_p3_e_tiles_arrived, combo_dc_e_tiles_phase);
                                combo_dc_e_tiles_phase ^= 1;
                            }
                            warpgroup::sync(quantizer_sync_id);

                            mm2_ABt(
                                combo_dc_tm, combo_col_stage->Gt_row, combo_dc_tile_stage->E_operand,
                                combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(0),
                                combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(0),
                                combo_dc_p3_inputs_finished);
                            tensor_commit<2>(combo_dc_p3_outputs_arrived);
                            tensor_after_thread_sync();
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                            warpgroup::sync(quantizer_sync_id);
                            prev_k_block_idx = k_block_idx;
                        }

                        if (prev_k_block_idx >= 0) {
                            wait(combo_dc_p3_outputs_arrived, combo_dc_outputs_phase);
                            combo_dc_outputs_phase ^= 1;
                            #pragma unroll
                            for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                                combo_dc_rt D_reg_fl;
                                combo_dc_rt_bf D_reg_bf;
                                warpgroup::tma::store_async_read_wait<0>();
                                warpgroup::sync(quantizer_sync_id);
                                warpgroup::load_async(
                                    D_reg_fl,
                                    combo_dc_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                        0,
                                        combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                                tensor_load_wait();
                                tensor_before_thread_sync();
                                warpgroup::sync(quantizer_sync_id);
                                warp::mul(D_reg_fl, D_reg_fl, combo_dc_scale);
                                warp::copy(D_reg_bf, D_reg_fl);
                                warpgroup::store(combo_dc_output_stage->dC, D_reg_bf);
                                warpgroup::sync(quantizer_sync_id);
                                warpgroup::tma::store_add_async(
                                    g.dC_out, combo_dc_output_stage->dC,
                                    {col_block_idx, prev_k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                                tensor_after_thread_sync();
                            }
                            warpgroup::tma::store_async_read_wait<0>();
                        }
                    }
                }

                warpgroup::sync(quantizer_sync_id);
                if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                    if (warpgroup::warpid() == 0 && quant_lane == 0) {
                        if (is_col_quantizer_wg && combo_do_dc && combo_stage_publish_epi) {
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            arrive(combo_g_col_ready[0]);
                        }
                        if (is_de_consumer_wg && combo_do_de && combo_stage_publish_epi) {
                            arrive(combo_g_row_recycled[0]);
                        }
                        if (is_dc_consumer_wg && combo_do_dc && combo_stage_publish_epi) {
                            arrive(combo_g_col_recycled[0]);
                        }
                    }
                }
                if (warpgroup::warpid() == 0 && quant_lane == 0) {
                    if constexpr (G::FRONTEND_SINGLE_QUANTIZER_WG) {
                        if constexpr (ROW_IN_QUANTIZER) {
                            arrive(slice_row_recycled[bf_stage]);
                        }
                        if constexpr (COL_IN_QUANTIZER) {
                            arrive(slice_col_recycled[bf_stage]);
                        }
                    } else {
                        if constexpr (ROW_IN_QUANTIZER) {
                            if (is_row_quantizer_wg) {
                                arrive(slice_row_recycled[bf_stage]);
                            }
                        }
                        if constexpr (COL_IN_QUANTIZER) {
                            if (is_col_quantizer_wg) {
                                arrive(slice_col_recycled[bf_stage]);
                            }
                        }
                    }
                }
                if constexpr (ROW_IN_QUANTIZER) {
                    if constexpr (G::FRONTEND_SINGLE_QUANTIZER_WG) {
                        if constexpr (G::DEBUG_ROW_WAIT_ON_COL_READY && DO_COL) {
                            update_phasebit<0>(slice_col_ready_phasebits, bf_stage);
                        } else {
                            update_phasebit<0>(slice_row_ready_phasebits, bf_stage);
                        }
                    } else if (is_row_quantizer_wg) {
                        if constexpr (G::DEBUG_ROW_WAIT_ON_COL_READY && DO_COL) {
                            update_phasebit<0>(slice_col_ready_phasebits, bf_stage);
                        } else {
                            update_phasebit<0>(slice_row_ready_phasebits, bf_stage);
                        }
                    }
                }
                if constexpr (COL_IN_QUANTIZER) {
                    if constexpr (G::FRONTEND_SINGLE_QUANTIZER_WG) {
                        if constexpr (G::USE_COL_PAIR_STAGE || (DO_ROW && C::ROW_QUANT_FROM_REGS && C::CONSUMER_DO_ROW && !C::EARLY_COL_READY)) {
                            update_phasebit<0>(slice_col_ready_phasebits, bf_stage);
                        } else {
                            update_phasebit<0>(slice_phasebits, bf_stage);
                        }
                    } else if (is_col_quantizer_wg) {
                        if constexpr (G::USE_COL_PAIR_STAGE || (DO_ROW && C::ROW_QUANT_FROM_REGS && C::CONSUMER_DO_ROW && !C::EARLY_COL_READY)) {
                            update_phasebit<0>(slice_col_ready_phasebits, bf_stage);
                        } else {
                            update_phasebit<0>(slice_phasebits, bf_stage);
                        }
                    }
                }
                if constexpr (G::ENABLE_STOREADD_COMBO && G::SEPARATE_BACKHALF_CONSUMERS) {
                    if (is_col_quantizer_wg && combo_do_dc && combo_stage_publish_epi) {
                        update_phasebit<1>(combo_col_recycle_phasebits, 0);
                    }
                    if (is_de_consumer_wg && combo_do_de && combo_stage_publish_epi) {
                        update_phasebit<0>(combo_row_ready_phasebits, 0);
                    }
                    if (is_dc_consumer_wg && combo_do_dc && combo_stage_publish_epi) {
                        update_phasebit<0>(combo_col_ready_phasebits, 0);
                    }
                }
            }

            if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO && !EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                static_assert(C::QUANTIZER_WARPGROUPS == 1,
                              "store-add combo currently requires the stable single-quantizer public v3 seam");
                static_assert(C::CONSUMER_DO_ROW,
                              "store-add combo currently requires consumer-owned row staging");
                auto *combo_p3_c_tile = &combo_p3_c_tile_stage->C_operand;
                auto *combo_p3_e_tile = &combo_output_stage->E_operand;
                auto *combo_p3_scales =
                    reinterpret_cast<typename G::combo_p3_scales_t *>(&input_scales[0]);
                auto *combo_p3_c_scales = &combo_p3_scales->C_sc;
                auto *combo_p3_e_scales = &combo_p3_scales->E_sc;

                warpgroup::sync(quantizer_sync_id);
                wait(combo_row_stage_ready, combo_row_stage_phase);
                combo_row_stage_phase ^= 1;
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                auto combo_p3_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                constexpr int combo_sc_offset = 256;
                constexpr int combo_de_a_scale_chunks = C::Nb / 64;
                constexpr int combo_de_b_scale_chunks = C::Nb / 128;
                constexpr int combo_dc_a_scale_chunks = C::Nb / 64;
                constexpr int combo_dc_b_scale_chunks = C::Nb / 64;
                auto combo_de_a_sc_tm =
                    tm_allocator.template allocate<full_tt_fp8e4m3<16 * combo_de_a_scale_chunks>>(combo_sc_offset);
                auto combo_de_b_sc_tm =
                    tm_allocator.template allocate<full_tt_fp8e4m3<32 * combo_de_b_scale_chunks>>(
                        combo_sc_offset + 4 * combo_de_a_scale_chunks);
                auto combo_dc_a_sc_tm =
                    tm_allocator.template allocate<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(combo_sc_offset);
                auto combo_dc_b_sc_tm =
                    tm_allocator.template allocate<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(
                        combo_sc_offset + 4 * combo_dc_a_scale_chunks);

                using combo_de_rt = rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
                using combo_de_rt_bf = rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
                using combo_dc_rt = rt_fl<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;
                using combo_dc_rt_bf = rt_bf<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;
                const bool combo_issue_leader = (warpgroup::warpid() == 0) && (quant_lane == 0);
                constexpr bool combo_debug_single_k = false;
                constexpr bool combo_debug_skip_store_add = false;
                constexpr bool combo_debug_skip_p3_mma = false;
                constexpr bool combo_debug_skip_p3_output_wait = false;
                constexpr bool combo_debug_skip_p3_output_drain = false;
                constexpr bool combo_debug_skip_de_scales = false;
                constexpr bool combo_debug_skip_de_tiles = false;
                constexpr bool combo_debug_skip_de_tile_wait = false;
                constexpr bool combo_debug_skip_dc_scales = false;
                constexpr bool combo_debug_skip_dc_tiles = false;
                constexpr bool combo_debug_skip_dc_tile_wait = false;
                const int combo_num_k_blocks_full =
                    combo_do_de ? (g.dE_out.cols() / C::Nb) : (g.dC_out.cols() / C::Nb);
                const int combo_num_k_blocks =
                    combo_debug_single_k ? min(combo_num_k_blocks_full, 1) : combo_num_k_blocks_full;

                auto combo_wait_p3_output = [&]() {
                    wait(combo_p3_outputs_arrived, combo_p3_outputs_phase);
                    combo_p3_outputs_phase ^= 1;
                };

                const bool combo_do_de_epi = combo_do_de;
                const bool combo_do_dc_epi = combo_do_dc;
                if (combo_do_de_epi || combo_do_dc_epi) {
                    constexpr bool combo_debug_skip_local_a_scales = false;
                    if (combo_do_de_epi) {
                        const float combo_de_scale = g_sg * g.C_col_sc_global[{0}];
                        if constexpr (!combo_debug_skip_local_a_scales) {
                            #pragma unroll
                            for (int ii = 0; ii < combo_de_a_scale_chunks; ++ii) {
                                auto combo_a_sc_sub =
                                    combo_de_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                auto &combo_g_sc_sm_sub =
                                    *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                        reinterpret_cast<uint64_t>(&combo_row_stage->G_row_sc.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(combo_a_sc_sub, combo_g_sc_sm_sub);
                            }
                        }
                        #pragma unroll 1
                        for (int k_block_idx = 0; k_block_idx < combo_num_k_blocks; ++k_block_idx) {
                            if (combo_issue_leader) {
                                if constexpr (!combo_debug_skip_de_tiles) {
                                    if constexpr (!combo_debug_skip_de_tile_wait) {
                                        tma::expect_bytes(combo_p3_c_tiles_arrived, sizeof(typename G::combo_p3_C_tile));
                                    }
                                    tma::load_async(
                                        *combo_p3_c_tile, g.C_col,
                                        {k_block_idx * 2 + cta_id, col_block_idx},
                                        combo_p3_c_tiles_arrived);
                                }
                                if constexpr (!combo_debug_skip_de_scales) {
                                    tma::load_async(
                                        *combo_p3_c_scales, g.C_col_sc,
                                        {k_block_idx, col_block_idx, 0},
                                        combo_p3_c_scales_arrived);
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);

                            if constexpr (!combo_debug_skip_de_scales) {
                                if (combo_issue_leader) {
                                    tma::expect_bytes(combo_p3_c_scales_arrived, sizeof(typename G::combo_p3_C_sc_tile));
                                    wait(combo_p3_c_scales_arrived, combo_c_scales_phase);
                                    combo_c_scales_phase ^= 1;
                                }
                                warpgroup::sync(quantizer_sync_id);
                            }

                            if constexpr (!combo_debug_skip_de_scales) {
                                #pragma unroll
                                for (int ii = 0; ii < combo_de_b_scale_chunks; ++ii) {
                                    auto combo_b_sc_sub =
                                        combo_de_b_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                    auto &combo_c_sc_sm_sub =
                                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                            reinterpret_cast<uint64_t>(&combo_p3_c_scales->data[0]) + 16 * 32 * ii);
                                    load_mxnv_scale_async2(combo_b_sc_sub, combo_c_sc_sm_sub);
                                }
                            }

                            if constexpr (!combo_debug_skip_de_tiles) {
                                if (combo_issue_leader) {
                                    if constexpr (!combo_debug_skip_de_tile_wait) {
                                        wait(combo_p3_c_tiles_arrived, combo_c_tiles_phase);
                                        combo_c_tiles_phase ^= 1;
                                    }
                                }
                                warpgroup::sync(quantizer_sync_id);
                            }

                            if constexpr (!combo_debug_skip_p3_mma) {
                                mm2_ABt(
                                    combo_p3_tm, combo_row_stage->G_row, *combo_p3_c_tile,
                                    combo_de_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_de_a_scale_chunks>>(0),
                                    combo_de_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_de_b_scale_chunks>>(0),
                                    combo_p3_inputs_finished);
                                tensor_commit<2>(combo_p3_outputs_arrived);
                                tensor_after_thread_sync();
                                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                                warpgroup::sync(quantizer_sync_id);
                                if constexpr (!combo_debug_skip_p3_output_wait) {
                                    combo_wait_p3_output();
                                    if constexpr (!combo_debug_skip_p3_output_drain) {
                                        #pragma unroll
                                        for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                                            combo_de_rt D_reg_fl;
                                            combo_de_rt_bf D_reg_bf;
                                            warpgroup::tma::store_async_read_wait<0>();
                                            warpgroup::sync(quantizer_sync_id);
                                            warpgroup::load_async(
                                                D_reg_fl,
                                                combo_p3_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                                    0,
                                                    combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                                            tensor_load_wait();
                                            tensor_before_thread_sync();
                                            warpgroup::sync(quantizer_sync_id);
                                            warp::mul(D_reg_fl, D_reg_fl, combo_de_scale);
                                            warp::copy(D_reg_bf, D_reg_fl);
                                            warpgroup::store(combo_output_stage->dE, D_reg_bf);
                                            warpgroup::sync(quantizer_sync_id);
                                            if constexpr (!combo_debug_skip_store_add) {
                                                warpgroup::tma::store_add_async(
                                                    g.dE_out, combo_output_stage->dE,
                                                    {row_block_idx * 2 + cta_id, k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                                            }
                                            tensor_after_thread_sync();
                                        }
                                        if constexpr (!combo_debug_skip_store_add) {
                                            warpgroup::tma::store_async_read_wait<0>();
                                        }
                                    }
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);
                            tensor_after_thread_sync();
                        }
                    }

                    if (combo_do_dc_epi) {
                        const float combo_dc_scale = g_sg * g.E_col_sc_global[{0}];
                        if constexpr (!combo_debug_skip_local_a_scales) {
                            #pragma unroll
                            for (int ii = 0; ii < combo_dc_a_scale_chunks; ++ii) {
                                auto combo_a_sc_sub =
                                    combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                auto &combo_gt_sc_sm_sub =
                                    *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                        reinterpret_cast<uint64_t>(&combo_col_stage->Gt_row_sc.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(combo_a_sc_sub, combo_gt_sc_sm_sub);
                            }
                        }
                        #pragma unroll 1
                        for (int k_block_idx = 0; k_block_idx < combo_num_k_blocks; ++k_block_idx) {
                            if (combo_issue_leader) {
                                if constexpr (!combo_debug_skip_dc_tiles) {
                                    if constexpr (!combo_debug_skip_dc_tile_wait) {
                                        tma::expect_bytes(combo_p3_e_tiles_arrived, sizeof(typename G::combo_p3_E_tile));
                                    }
                                    tma::load_async(
                                        *combo_p3_e_tile, g.E_col,
                                        {k_block_idx * 2 + cta_id, row_block_idx},
                                        combo_p3_e_tiles_arrived);
                                }
                                if constexpr (!combo_debug_skip_dc_scales) {
                                    tma::load_async(
                                        *combo_p3_e_scales, g.E_col_sc,
                                        {k_block_idx, row_block_idx, 0},
                                        combo_p3_e_scales_arrived);
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);

                            if constexpr (!combo_debug_skip_dc_scales) {
                                if (combo_issue_leader) {
                                    tma::expect_bytes(combo_p3_e_scales_arrived, sizeof(typename G::combo_p3_E_sc_tile));
                                    wait(combo_p3_e_scales_arrived, combo_e_scales_phase);
                                    combo_e_scales_phase ^= 1;
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);

                            if constexpr (!combo_debug_skip_dc_scales) {
                                #pragma unroll
                                for (int ii = 0; ii < combo_dc_b_scale_chunks; ++ii) {
                                    auto combo_b_sc_sub =
                                        combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                                    auto &combo_e_sc_sm_sub =
                                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                            reinterpret_cast<uint64_t>(&combo_p3_e_scales->data[0]) + 16 * 32 * ii);
                                        load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
                                }
                            }

                            if constexpr (!combo_debug_skip_dc_tiles) {
                                if (combo_issue_leader) {
                                    if constexpr (!combo_debug_skip_dc_tile_wait) {
                                        wait(combo_p3_e_tiles_arrived, combo_e_tiles_phase);
                                        combo_e_tiles_phase ^= 1;
                                    }
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);

                            if constexpr (!combo_debug_skip_p3_mma) {
                                mm2_ABt(
                                    combo_p3_tm, combo_col_stage->Gt_row, *combo_p3_e_tile,
                                    combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(0),
                                    combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(0),
                                    combo_p3_inputs_finished);
                                tensor_commit<2>(combo_p3_outputs_arrived);
                                tensor_after_thread_sync();
                                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                                warpgroup::sync(quantizer_sync_id);
                                if constexpr (!combo_debug_skip_p3_output_wait) {
                                    combo_wait_p3_output();
                                    if constexpr (!combo_debug_skip_p3_output_drain) {
                                        #pragma unroll
                                        for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                                            combo_dc_rt D_reg_fl;
                                            combo_dc_rt_bf D_reg_bf;
                                            warpgroup::tma::store_async_read_wait<0>();
                                            warpgroup::sync(quantizer_sync_id);
                                            warpgroup::load_async(
                                                D_reg_fl,
                                                combo_p3_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                                    0,
                                                    combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                                            tensor_load_wait();
                                            tensor_before_thread_sync();
                                            warpgroup::sync(quantizer_sync_id);
                                            warp::mul(D_reg_fl, D_reg_fl, combo_dc_scale);
                                            warp::copy(D_reg_bf, D_reg_fl);
                                            warpgroup::store(combo_output_stage->dC, D_reg_bf);
                                            warpgroup::sync(quantizer_sync_id);
                                            if constexpr (!combo_debug_skip_store_add) {
                                                warpgroup::tma::store_add_async(
                                                    g.dC_out, combo_output_stage->dC,
                                                    {col_block_idx, k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                                            }
                                            tensor_after_thread_sync();
                                        }
                                        if constexpr (!combo_debug_skip_store_add) {
                                            warpgroup::tma::store_async_read_wait<0>();
                                        }
                                    }
                                }
                            }
                            warpgroup::sync(quantizer_sync_id);
                            tensor_after_thread_sync();
                        }
                    }
                }

                warpgroup::sync(quantizer_sync_id);
                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
            }
        }
        warpgroup::sync(quantizer_sync_id);
        if constexpr (EFFECTIVE_ENABLE_STOREADD_COMBO) {
            if constexpr (EFFECTIVE_SEPARATE_BACKHALF_CONSUMERS) {
                const bool combo_deprovision_owner =
                    combo_do_dc ? is_dc_consumer_wg :
                    (combo_do_de ? is_de_consumer_wg : is_col_quantizer_wg);
                if (combo_deprovision_owner && warpgroup::warpid() == 0) tm_allocator.deprovision();
            } else {
                if (warpgroup::warpid() == 0) tm_allocator.deprovision();
            }
        }
    }
}

template <typename C>
__device__ inline void backward_kernel_v3(const globals<C>& g) {
    backward_kernel_v3_impl<C, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_colscratch(const globals<C>& g) {
    backward_kernel_v3_impl<C, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, true, true, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_replayonly(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, false, false, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_rowonly(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, true, false, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_rowonly_storebf16(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, true, false, false, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_colonly(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, false, true, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_2ctaS(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, true, true, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_2ctaS_replayonly(const globals<C>& g) {
    backward_kernel_v3_streaming_impl<C, false, false, true>(g);
}

template <typename C>
__global__ void backward_kernel_v3_col2pass_stage2(
    const bf16* __restrict__ D_scratch,
    int scratch_stride,
    const float* __restrict__ G_sg_row,
    uint8_t* __restrict__ G_fp4_col_ptr,
    uint8_t* __restrict__ G_sc_col_ptr,
    int col_fp4_stride,
    int col_sc_kgroups,
    int M,
    int N,
    bool encode_centric)
{
    constexpr int TILE_ROWS = C::Mb / 2;
    constexpr int TILE_COLS = C::Nb;
    constexpr float FP4_MAX = 6.0f;
    constexpr float E4M3_MAX = 448.0f;

    const int col_in_tile = threadIdx.x;
    if (col_in_tile >= TILE_COLS) return;

    const int half_row_block_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;
    const int global_col = col_block_idx * TILE_COLS + col_in_tile;
    if (global_col >= N) return;

    const float g_sg = G_sg_row[0];
    const float g_sg_rcp = 1.0f / g_sg;
    const int tile_row_base = half_row_block_idx * TILE_ROWS;

    #pragma unroll
    for (int row16_block = 0; row16_block < TILE_ROWS / 16; ++row16_block) {
        const int global_row_base = tile_row_base + row16_block * 16;
        float col_amax = 0.0f;

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int global_row = global_row_base + r;
            if (global_row < M) {
                const bf16 value = D_scratch[global_row * scratch_stride + global_col];
                col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value)));
            }
        }

        const float col_scale = col_amax * (1.0f / FP4_MAX);
        const float col_rcp = (col_amax > 0.0f) ? (FP4_MAX / col_amax) : 0.0f;
        const int global_row_pair_base = global_row_base / 2;

        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const int global_row = global_row_base + pair * 2;
            if (global_row < M) {
                const float v0 = __bfloat162float(D_scratch[global_row * scratch_stride + global_col]);
                float v1 = 0.0f;
                if (global_row + 1 < M) {
                    v1 = __bfloat162float(D_scratch[(global_row + 1) * scratch_stride + global_col]);
                }
                G_fp4_col_ptr[global_col * col_fp4_stride + global_row_pair_base + pair] =
                    quantize_fp4_pair(v0, v1, col_rcp);
            }
        }

        float stored_scale = col_scale * g_sg_rcp;
        if (encode_centric) {
            stored_scale = fminf(col_rcp * g_sg, E4M3_MAX);
        }
        const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(stored_scale);
        const int depth = global_col / 128;
        const int sr = global_col % 32;
        const int rr = (global_col / 32) % 4;
        const int m_kgroup = global_row_base / 64;
        const int m_16_in_64 = (global_row_base / 16) % 4;
        const int chunk = depth * col_sc_kgroups + m_kgroup;
        const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
        G_sc_col_ptr[chunk * 512 + byte_idx] =
            *reinterpret_cast<const uint8_t*>(&csc);
    }
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_2ctaSdupB(const globals_2ctaSdupB<C>& g) {
    backward_kernel_v3_streaming_2ctaSdupB_impl<C, true, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_2ctaSdupB_replayonly(const globals_2ctaSdupB<C>& g) {
    backward_kernel_v3_streaming_2ctaSdupB_impl<C, false, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_3wg(const globals_3wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, true, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_3wg_replayonly(const globals_3wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, false, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_3wg_rowonly(const globals_3wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, true, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_3wg_colonly(const globals_3wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, false, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_4wg(const globals_4wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, true, true>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_4wg_replayonly(const globals_4wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, false, false>(g);
}

template <typename C>
__device__ inline void backward_kernel_v3_streaming_5wg(const globals_5wg<C>& g) {
    backward_kernel_v3_streaming_3wg_impl<C, true, true>(g);
}

template <typename C, int STATIC_COMBO_MODE>
__device__ inline void backward_kernel_v3_streaming_5wg_mode(const globals_5wg<C>& g) {
    if constexpr (STATIC_COMBO_MODE == globals_5wg<C>::COMBO_MODE_DEONLY) {
        backward_kernel_v3_streaming_3wg_impl<C, true, false, STATIC_COMBO_MODE>(g);
    } else if constexpr (STATIC_COMBO_MODE == globals_5wg<C>::COMBO_MODE_DCONLY) {
        backward_kernel_v3_streaming_3wg_impl<C, false, true, STATIC_COMBO_MODE>(g);
    } else {
        backward_kernel_v3_streaming_3wg_impl<C, true, true, STATIC_COMBO_MODE>(g);
    }
}

} // namespace nvfp4_cce_backward_v3
