#pragma once

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_localcta_swiglu_quant_gemm {

static constexpr float LOCALCTA_PREPARED_MIN_NONZERO_SCALE = 0.001953125f;

template <
    int _LOAD_PIPE_DEPTH,
    int _SUPERGROUP_SIZE,
    bool _USE_PDL = true,
    int _Nb = 128,
    int _Kb = 128,
    bool _USE_SQRELU = false,
    bool _ENCODE_CENTRIC = true,
    bool _V5_SCALAR_SG = false,
    bool _AMAX_ONLY = false>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "W13 SwiGLU producer supports 128- or 256-column tiles");
    static_assert(_Kb == 128 || _Kb == 256, "W13 SwiGLU producer supports 128- or 256-wide reduction tiles");
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = _USE_PDL;
    static constexpr bool USE_SQRELU = _USE_SQRELU;
    static constexpr bool ENCODE_CENTRIC = _ENCODE_CENTRIC;
    static constexpr bool V5_SCALAR_SG = _V5_SCALAR_SG;
    static constexpr bool AMAX_ONLY = _AMAX_ONLY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _Nb / 32;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int NUM_D_TILES = 2;
    static constexpr int NUM_HALVES = Nb / 128;
    static constexpr int EPI_PER_HALF = EPI_PIPE_DEPTH / NUM_HALVES;
    static constexpr int A_SC_TMEM_OFFSET = 256;
    static constexpr int A_SC_TMEM_COLS = (16 * MMA_PER_TILE * LOAD_PIPE_DEPTH) / 4;
    static constexpr int B_SC_TMEM_COLS = (32 * B_SC_SIZE * MMA_PER_TILE * LOAD_PIPE_DEPTH) / 4;
    static constexpr int B1_SC_TMEM_OFFSET = A_SC_TMEM_OFFSET + A_SC_TMEM_COLS;
    static constexpr int B3_SC_TMEM_OFFSET = B1_SC_TMEM_OFFSET + B_SC_TMEM_COLS;
};

template <typename _GL>
struct tma_dev_proxy {
    using identifier = ducks::gl::identifier;
    using T = typename _GL::T;
    using T2 = typename _GL::T2;
    using dtype = typename _GL::dtype;
    static constexpr int __b__ = _GL::__b__;
    static constexpr int __d__ = _GL::__d__;
    static constexpr int __r__ = _GL::__r__;
    static constexpr int __c__ = _GL::__c__;

    const CUtensorMap* dev_tma;

    __device__ explicit tma_dev_proxy(const CUtensorMap* _dev_tma) : dev_tma(_dev_tma) {}

    template<int axis> __device__ inline size_t shape() const { return 0; }
    template<int axis> __device__ inline size_t stride() const { return 0; }

    template<typename U, int axis> __device__ inline const CUtensorMap* get_tma() const {
        return dev_tma;
    }
    __device__ inline void prefetch() const {
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(dev_tma)) : "memory");
    }
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<half,       1, -1, -1, 256, B_sc_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B1;
    B_sc_gl    B1_sc;
    B_fp4x2_gl B3;
    B_sc_gl    B3_sc;
    const float* A_sg;
    int A_sg_stride;
    const float* A_sg_chunk_grid;
    int A_sg_chunk_stride;
    const float* B1_sg;
    int B1_sg_stride;
    const float* B1_sg_chunk_grid;
    int B1_sg_chunk_stride;
    const float* B3_sg;
    int B3_sg_stride;
    const float* B3_sg_chunk_grid;
    int B3_sg_chunk_stride;

    uint8_t* row_fp4;
    uint8_t* row_sc;
    float* row_sg;
    uint8_t* col_fp4;
    uint8_t* col_sc;
    float* col_sg;
    float* global_amax;
    int M;
    int H;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B1;
        B_fp4x2_tile B3;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B1[C::B_SC_SIZE];
        B_sc_tile B3[C::B_SC_SIZE];
    };

    __host__ inline dim3 grid() const {
        const int num_row_blocks = M / C::Mb;
        const int num_col_blocks = H / C::Nb;
        int grid_size = min(num_row_blocks * num_col_blocks * C::CLUSTER_SIZE, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__device__ inline void apply_w13_chunk_scales_to_stage(
    typename globals<C>::input_scales_t &scales,
    const float *A_sg_chunks,
    int A_sg_stride,
    const float *A_sg_final,
    int A_sg_final_stride,
    const float *B1_sg_chunks,
    int B1_sg_stride,
    const float *B1_sg_final,
    int B1_sg_final_stride,
    const float *B3_sg_chunks,
    int B3_sg_stride,
    const float *B3_sg_final,
    int B3_sg_final_stride,
    int a_chunk_row,
    int b_chunk_row_0,
    int row_block_idx,
    int col_block_idx,
    int chunk_base)
{
    constexpr int B_BLOCKS_PER_OUTER_SG = 256 / C::Nb;
    const int b_outer_idx = col_block_idx / B_BLOCKS_PER_OUTER_SG;

    #pragma unroll
    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
        const int chunk_k = chunk_base + ii / 2;
        if (A_sg_chunks != nullptr) {
            const float a_chunk_sg = A_sg_chunks[a_chunk_row * A_sg_stride + chunk_k];
            const float a_final_sg = (A_sg_final != nullptr)
                ? fmaxf(A_sg_final[row_block_idx * A_sg_final_stride], 1.0e-12f)
                : 1.0f;
            const float a_scale = a_chunk_sg / a_final_sg;
            auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&scales.A.data[0]) + 16 * 32 * ii);
            nvfp4_gemm::scale_shared_fp8_tile(A_sc_sm_subtile, a_scale);
        }

        if (B1_sg_chunks != nullptr) {
            const float b1_final_sg = (B1_sg_final != nullptr)
                ? fmaxf(B1_sg_final[b_outer_idx * B1_sg_final_stride], 1.0e-12f)
                : 1.0f;
            const float b1_chunk_sg_0 = B1_sg_chunks[b_chunk_row_0 * B1_sg_stride + chunk_k];
            const float b1_scale_0 = b1_chunk_sg_0 / b1_final_sg;
            auto &B1_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&scales.B1[0].data[0]) + 16 * 32 * ii);
            nvfp4_gemm::scale_shared_fp8_tile(B1_sc_sm_subtile_0, b1_scale_0);
            if constexpr (C::B_SC_SIZE == 2) {
                const float b1_chunk_sg_1 = B1_sg_chunks[(b_chunk_row_0 + 1) * B1_sg_stride + chunk_k];
                const float b1_scale_1 = b1_chunk_sg_1 / b1_final_sg;
                auto &B1_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                    reinterpret_cast<uint64_t>(&scales.B1[1].data[0]) + 16 * 32 * ii);
                nvfp4_gemm::scale_shared_fp8_tile(B1_sc_sm_subtile_1, b1_scale_1);
            }
        }

        if (B3_sg_chunks != nullptr) {
            const float b3_final_sg = (B3_sg_final != nullptr)
                ? fmaxf(B3_sg_final[b_outer_idx * B3_sg_final_stride], 1.0e-12f)
                : 1.0f;
            const float b3_chunk_sg_0 = B3_sg_chunks[b_chunk_row_0 * B3_sg_stride + chunk_k];
            const float b3_scale_0 = b3_chunk_sg_0 / b3_final_sg;
            auto &B3_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&scales.B3[0].data[0]) + 16 * 32 * ii);
            nvfp4_gemm::scale_shared_fp8_tile(B3_sc_sm_subtile_0, b3_scale_0);
            if constexpr (C::B_SC_SIZE == 2) {
                const float b3_chunk_sg_1 = B3_sg_chunks[(b_chunk_row_0 + 1) * B3_sg_stride + chunk_k];
                const float b3_scale_1 = b3_chunk_sg_1 / b3_final_sg;
                auto &B3_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                    reinterpret_cast<uint64_t>(&scales.B3[1].data[0]) + 16 * 32 * ii);
                nvfp4_gemm::scale_shared_fp8_tile(B3_sc_sm_subtile_1, b3_scale_1);
            }
        }
    }
}

static constexpr float LOCALCTA_GLOBAL_SCALE_NUM = 1493.0f;
static constexpr float V5_GLOBAL_SCALE_NUM = 2688.0f;
static constexpr float LOCALCTA_MIN_NONZERO_SCALE = 0.001953125f;
static constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__ void atomic_max_float_bits(float* addr, float val) {
    if (val <= 0.0f) {
        return;
    }
    unsigned int* p = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *p;
    const unsigned int want = __float_as_uint(val);
    while (want > old) {
        old = atomicCAS(p, old, want);
    }
}

__device__ __forceinline__ uint8_t float_to_fp4(float val) {
    float aval = fabsf(val);
    uint8_t sign = ((__float_as_uint(val) >> 31) << 3);
    uint8_t enc =
        static_cast<uint8_t>(aval >= 0.25f) +
        static_cast<uint8_t>(aval >= 0.75f) +
        static_cast<uint8_t>(aval >= 1.25f) +
        static_cast<uint8_t>(aval >= 1.75f) +
        static_cast<uint8_t>(aval >= 2.5f) +
        static_cast<uint8_t>(aval >= 3.5f) +
        static_cast<uint8_t>(aval >= 5.0f);
    return sign | enc;
}

__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float coeff) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg.b8 f0, f1, f2, f3; \n\t"
        ".reg.f32 x, y; \n\t"
        "mul.rn.f32 x, %1, %3; \n\t"
        "mul.rn.f32 y, %2, %3; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, x, y; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, x, y; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, x, y; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, x, y; \n\t"
        "mov.b32 %0, {f0, f1, f2, f3}; \n"
        "}"
        : "=r"(out)
        : "f"(v0), "f"(v1), "f"(coeff));
    return static_cast<uint8_t>(out & 0xffu);
}

__device__ __forceinline__ uint8_t fp8e4m3_byte(float value) {
    fp8e4m3 out = static_cast<fp8e4m3>(value);
    return reinterpret_cast<const uint8_t&>(out);
}

__device__ __forceinline__ float fp8e4m3_round_to_float(float value) {
    fp8e4m3 out = static_cast<fp8e4m3>(value);
    return static_cast<float>(out);
}

__device__ __forceinline__ float localcta_encode_scale(float amax) {
    if (amax == 0.0f) {
        return 1.0f;
    }
    const float scale = LOCALCTA_GLOBAL_SCALE_NUM / amax;
    if (scale == 0.0f) {
        return 1.0f;
    }
    return fminf(scale, 3.4028235e+38f);
}

__device__ __forceinline__ float v5_encode_scale(float amax) {
    if (amax == 0.0f) {
        return 1.0f;
    }
    const float scale = V5_GLOBAL_SCALE_NUM / amax;
    if (scale == 0.0f) {
        return 1.0f;
    }
    return fminf(scale, 3.4028235e+38f);
}

__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_rn(
    const uint64_t in03,
    const uint64_t in47,
    const float coeff)
{
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg.b64 scaling_coeff_2x; \n\t"
        "mov.b64 scaling_coeff_2x, {%3, %3}; \n\t"
        ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
        "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"
        ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "cvt.f32.bf16 v4, v4_bf16; \n\t"
        "cvt.f32.bf16 v5, v5_bf16; \n\t"
        "cvt.f32.bf16 v6, v6_bf16; \n\t"
        "cvt.f32.bf16 v7, v7_bf16; \n\t"
        ".reg.b64 v01, v23, v45, v67; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mov.b64 v45, {v4, v5}; \n\t"
        "mov.b64 v67, {v6, v7}; \n\t"
        "mul.f32x2 v01, v01, scaling_coeff_2x; \n\t"
        "mul.f32x2 v23, v23, scaling_coeff_2x; \n\t"
        "mul.f32x2 v45, v45, scaling_coeff_2x; \n\t"
        "mul.f32x2 v67, v67, scaling_coeff_2x; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "mov.b64 {v5, v4}, v45; \n\t"
        "mov.b64 {v7, v6}, v67; \n\t"
        ".reg.b8 f0, f1, f2, f3; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
        "mov.b32 %0, {f0, f1, f2, f3};\n"
        "}"
        : "=r"(out)
        : "l"(in03), "l"(in47), "f"(coeff));
    return out;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}

template <typename T>
__device__ __forceinline__ uint32_t map_shared_cluster_addr(const T* ptr, int dst_cta) {
    uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(remote_addr)
                 : "r"(local_addr), "r"(dst_cta));
    return remote_addr;
}

__device__ __forceinline__ float cluster_load_shared_f32(const float* ptr, int src_cta) {
    const uint32_t remote_addr = map_shared_cluster_addr(ptr, src_cta);
    uint32_t bits;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(bits) : "r"(remote_addr));
    return *reinterpret_cast<const float*>(&bits);
}

__device__ __forceinline__ uint32_t cluster_load_shared_u32(const uint32_t* ptr, int src_cta) {
    const uint32_t remote_addr = map_shared_cluster_addr(ptr, src_cta);
    uint32_t value;
    asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(value) : "r"(remote_addr));
    return value;
}

template <typename C>
__device__ __forceinline__ void localcta_block_quant_params(
    float block_amax,
    float chunk_s_enc,
    float chunk_sg,
    float& coeff,
    uint8_t& stored_scale_byte)
{
    (void)chunk_sg;
    if constexpr (C::ENCODE_CENTRIC) {
        __nv_fp8_e4m3 mult_fp8 = static_cast<__nv_fp8_e4m3>(FP8_E4M3_MAX);
        float mult = FP8_E4M3_MAX;
        if (block_amax > 0.0f && chunk_s_enc > 0.0f) {
            mult = fminf(6.0f / (block_amax * chunk_s_enc), 3.4028235e+38f);
            mult_fp8 = static_cast<__nv_fp8_e4m3>(mult);
        }
        const float mult_val = static_cast<float>(mult_fp8);
        coeff = mult_val * chunk_s_enc;
        const __nv_fp8_e4m3 stored = static_cast<__nv_fp8_e4m3>(1.0f / mult_val);
        stored_scale_byte = *reinterpret_cast<const uint8_t*>(&stored);
    } else {
        __nv_fp8_e4m3 stored = static_cast<__nv_fp8_e4m3>(0.0f);
        if (chunk_s_enc > 0.0f) {
            stored = static_cast<__nv_fp8_e4m3>(
                fminf((block_amax / 6.0f) * chunk_s_enc, 3.4028235e+38f));
        }
        const float stored_val = static_cast<float>(stored);
        coeff = (stored_val > 0.0f && chunk_s_enc > 0.0f)
            ? fminf(chunk_s_enc / stored_val, 3.4028235e+38f)
            : 0.0f;
        stored_scale_byte = *reinterpret_cast<const uint8_t*>(&stored);
    }
}

template <typename C, typename subtile_rt>
__device__ __noinline__ float stage_swiglu_pairs(
    subtile_rt& D1_fl,
    subtile_rt& D3_fl,
    bf16_2 (*pairs)[16][33],
    int epi_slot,
    int lane_id,
    int warp_row_base,
    int logical_col_start,
    int M,
    int H)
{
    const int lane_byte = lane_id % 4;
    const int row_pair_idx = lane_id / 4;
    float local_amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < subtile_rt::height; i++) {
        (void)row_pair_idx;
        #pragma unroll
        for (int j = 0; j < subtile_rt::width; j++) {
            const int pair_base = j * 8 + lane_byte;

            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float h1_x = __bfloat162float(
                    __float2bfloat16_rn(D1_fl.tiles[i][j].data[d].x));
                const float h1_y = __bfloat162float(
                    __float2bfloat16_rn(D1_fl.tiles[i][j].data[d].y));
                const float h3_x = __bfloat162float(
                    __float2bfloat16_rn(D3_fl.tiles[i][j].data[d].x));
                const float h3_y = __bfloat162float(
                    __float2bfloat16_rn(D3_fl.tiles[i][j].data[d].y));
                const float sig_x = 1.0f / (1.0f + __expf(-h1_x));
                const float sig_y = 1.0f / (1.0f + __expf(-h1_y));
                const float out_x = h1_x * sig_x * h3_x;
                const float out_y = h1_y * sig_y * h3_y;

                const int col_slot = pair_base + ((d >= 2) ? 4 : 0);
                const int row_slot = i * 16 + row_pair_idx + ((d & 1) ? 8 : 0);
                const bf16_2 out_pair = bf16_2{
                    __float2bfloat16_rn(out_x),
                    __float2bfloat16_rn(out_y)};
                pairs[epi_slot][col_slot][row_slot] = out_pair;
                local_amax = fmaxf(local_amax, fabsf(__bfloat162float(out_pair.x)));
                local_amax = fmaxf(local_amax, fabsf(__bfloat162float(out_pair.y)));
            }
        }
    }
    return local_amax;
}

template <typename C, typename subtile_rt>
__device__ __noinline__ float stage_sqrelu_pairs(
    subtile_rt& D1_fl,
    bf16_2 (*pairs)[16][33],
    int epi_slot,
    int lane_id)
{
    const int lane_byte = lane_id % 4;
    const int row_pair_idx = lane_id / 4;
    float local_amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < subtile_rt::height; i++) {
        (void)row_pair_idx;
        #pragma unroll
        for (int j = 0; j < subtile_rt::width; j++) {
            const int pair_base = j * 8 + lane_byte;

            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float h1_x = __bfloat162float(
                    __float2bfloat16_rn(D1_fl.tiles[i][j].data[d].x));
                const float h1_y = __bfloat162float(
                    __float2bfloat16_rn(D1_fl.tiles[i][j].data[d].y));
                const float relu_x = fmaxf(h1_x, 0.0f);
                const float relu_y = fmaxf(h1_y, 0.0f);
                const float out_x = relu_x * relu_x;
                const float out_y = relu_y * relu_y;

                const int col_slot = pair_base + ((d >= 2) ? 4 : 0);
                const int row_slot = i * 16 + row_pair_idx + ((d & 1) ? 8 : 0);
                const bf16_2 out_pair = bf16_2{
                    __float2bfloat16_rn(out_x),
                    __float2bfloat16_rn(out_y)};
                pairs[epi_slot][col_slot][row_slot] = out_pair;
                local_amax = fmaxf(local_amax, fabsf(__bfloat162float(out_pair.x)));
                local_amax = fmaxf(local_amax, fabsf(__bfloat162float(out_pair.y)));
            }
        }
    }
    return local_amax;
}

template <typename C>
__device__ __noinline__ float amax_from_stage(
    bf16_2 (*pairs)[16][33],
    int lane_id)
{
    float local_amax = 0.0f;
    #pragma unroll
    for (int epi_slot = 0; epi_slot < C::EPI_PER_HALF; ++epi_slot) {
        #pragma unroll
        for (int col_slot = 0; col_slot < 16; ++col_slot) {
            const bf16_2 v = pairs[epi_slot][col_slot][lane_id];
            local_amax = fmaxf(local_amax, fabsf(__bfloat162float(v.x)));
            local_amax = fmaxf(local_amax, fabsf(__bfloat162float(v.y)));
        }
    }
    return warp_reduce_max(local_amax);
}

template <typename C>
__device__ __noinline__ void quantize_rows_from_stage(
    const globals<C>& g,
    bf16_2 (*pairs)[16][33],
    int epi_slot,
    int lane_id,
    int warp_row_base,
    int logical_col_start,
    float chunk_s_enc,
    float chunk_sg)
{
    const int local_row = lane_id;
    const int global_row = warp_row_base + local_row;
    const int row_fp4_stride = g.H / 2;
    const int row_ntk_total = g.H / 64;
    const int sc_row_blk = global_row / 128;
    const int j_in_tile = global_row % 32;
    const int grp = (global_row % 128) / 32;

    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        float block_amax = 0.0f;
        __align__(8) bf16_2 cached[8];
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const bf16_2 v = pairs[epi_slot][group * 8 + pair][local_row];
            cached[pair] = v;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(v.x)));
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(v.y)));
        }

        float coeff;
        uint8_t stored_scale;
        localcta_block_quant_params<C>(block_amax, chunk_s_enc, chunk_sg, coeff, stored_scale);
        if constexpr (C::USE_SQRELU && !C::V5_SCALAR_SG) {
            __nv_fp8_e4m3 raw_scale;
            *reinterpret_cast<uint8_t*>(&raw_scale) = stored_scale;
            __nv_fp8_e4m3 prepared_scale =
                static_cast<__nv_fp8_e4m3>(static_cast<float>(raw_scale) * chunk_sg);
            if (
                static_cast<float>(raw_scale) > 0.0f
                && chunk_sg > 0.0f
                && static_cast<float>(prepared_scale) == 0.0f
            ) {
                prepared_scale = static_cast<__nv_fp8_e4m3>(
                    LOCALCTA_PREPARED_MIN_NONZERO_SCALE);
            }
            stored_scale = *reinterpret_cast<const uint8_t*>(&prepared_scale);
        }

        const uint32_t packed_lo = mul_cvt_bf16_to_fp4_8x_rn(
            *reinterpret_cast<const uint64_t*>(&cached[0]),
            *reinterpret_cast<const uint64_t*>(&cached[2]),
            coeff);
        const uint32_t packed_hi = mul_cvt_bf16_to_fp4_8x_rn(
            *reinterpret_cast<const uint64_t*>(&cached[4]),
            *reinterpret_cast<const uint64_t*>(&cached[6]),
            coeff);
        const uint64_t packed = static_cast<uint64_t>(packed_lo) |
                                (static_cast<uint64_t>(packed_hi) << 32);

        const int logical_col = logical_col_start + group * 16;
        *reinterpret_cast<uint64_t*>(
            &g.row_fp4[global_row * row_fp4_stride + logical_col / 2]) = packed;

        const int col64 = logical_col / 64;
        const int scale_in64 = (logical_col % 64) / 16;
        const int scale_base = (sc_row_blk * row_ntk_total + col64) * 512 +
                               j_in_tile * 16 + grp * 4 + scale_in64;
        g.row_sc[scale_base] = stored_scale;
    }
}

template <typename C>
__device__ __noinline__ void quantize_cols_from_stage(
    const globals<C>& g,
    bf16_2 (*pairs)[16][33],
    int epi_slot,
    int lane_id,
    int warp_row_base,
    int logical_col_start,
    float chunk_s_enc,
    float chunk_sg)
{
    const int local_col = lane_id;
    const int local_col_pair = local_col >> 1;
    const bool use_y = (local_col & 1) != 0;
    const int logical_col = logical_col_start + local_col;
    const int col_fp4_stride = g.M / 2;
    const int col_ntk_total = g.M / 64;
    const int col_chunk = logical_col / 128;
    const int j_in_tile = logical_col % 32;
    const int grp = (logical_col % 128) / 32;

    #pragma unroll
    for (int row_group = 0; row_group < 2; ++row_group) {
        float block_amax = 0.0f;
        __align__(8) bf16_2 cached[8];
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const int row0 = row_group * 16 + pair * 2;
            const bf16_2 v0 = pairs[epi_slot][local_col_pair][row0 + 0];
            const bf16_2 v1 = pairs[epi_slot][local_col_pair][row0 + 1];
            cached[pair] = use_y ? bf16_2{v0.y, v1.y} : bf16_2{v0.x, v1.x};
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(cached[pair].x)));
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(cached[pair].y)));
        }

        float coeff;
        uint8_t stored_scale;
        localcta_block_quant_params<C>(block_amax, chunk_s_enc, chunk_sg, coeff, stored_scale);

        const uint32_t packed_lo = mul_cvt_bf16_to_fp4_8x_rn(
            *reinterpret_cast<const uint64_t*>(&cached[0]),
            *reinterpret_cast<const uint64_t*>(&cached[2]),
            coeff);
        const uint32_t packed_hi = mul_cvt_bf16_to_fp4_8x_rn(
            *reinterpret_cast<const uint64_t*>(&cached[4]),
            *reinterpret_cast<const uint64_t*>(&cached[6]),
            coeff);
        const uint64_t packed = static_cast<uint64_t>(packed_lo) |
                                (static_cast<uint64_t>(packed_hi) << 32);

        const int global_row_pair = warp_row_base / 2 + row_group * 8;
        *reinterpret_cast<uint64_t*>(
            &g.col_fp4[logical_col * col_fp4_stride + global_row_pair]) = packed;

        const int global_row = warp_row_base + row_group * 16;
        const int row64 = global_row / 64;
        const int scale_in64 = (global_row % 64) / 16;
        const int scale_base = (col_chunk * col_ntk_total + row64) * 512 +
                               j_in_tile * 16 + grp * 4 + scale_in64;
        g.col_sc[scale_base] = stored_scale;
    }
}

template <typename C>
__device__ inline void kernel(const globals<C>& g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B1.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B1_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.B3.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B3_sc.template prefetch_tma<typename G::B_sc_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.M / C::Mb;
    const int num_col_blocks = g.H / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    const bool has_any_chunk_grid =
        g.A_sg_chunk_grid != nullptr ||
        g.B1_sg_chunk_grid != nullptr ||
        g.B3_sg_chunk_grid != nullptr;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scale_tiles_ready[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_prepared[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(scale_tiles_ready[i], 0, 1);
            init_semaphore(scales_prepared[i], 0, C::CLUSTER_SIZE);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3 && warp::elect_leader()) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B1, g.B1, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B3, g.B3, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2 && warp::elect_leader()) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if constexpr (C::B_SC_SIZE == 2) {
                        tma::cluster::load_async(input_scales[stage].B1[cta_id], g.B1_sc, {col_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                        tma::cluster::load_async(input_scales[stage].B3[cta_id], g.B3_sc, {col_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    } else if (cta_id == 0) {
                        tma::cluster::load_async(input_scales[stage].B1[0], g.B1_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                        tma::cluster::load_async(input_scales[stage].B3[0], g.B3_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 1) {
            if (!has_any_chunk_grid) {
                return;
            }
            everyone::tma::cluster::wait();
            const int lane = warp::laneid();
            uint32_t ready_phasebits = 0;
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_red_blocks; ++i) {
                    if (cta_id == 0) {
                        if (lane == 0) {
                            tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                            arrive(scale_tiles_ready[stage], 1);
                            tma::cluster::arrive(scale_tiles_ready[stage], 1, 1);
                            update_phasebit<0>(phasebits, stage);
                        }
                    } else if (lane == 0) {
                        wait(scale_tiles_ready[stage], get_phasebit<0>(ready_phasebits, stage));
                        update_phasebit<0>(ready_phasebits, stage);
                    }
                    __syncwarp();

                    apply_w13_chunk_scales_to_stage<C>(
                        input_scales[stage],
                        g.A_sg_chunk_grid,
                        g.A_sg_chunk_stride < 0 ? -g.A_sg_chunk_stride : g.A_sg_chunk_stride,
                        g.A_sg,
                        g.A_sg_stride,
                        g.B1_sg_chunk_grid,
                        g.B1_sg_chunk_stride < 0 ? -g.B1_sg_chunk_stride : g.B1_sg_chunk_stride,
                        g.B1_sg,
                        g.B1_sg_stride,
                        C::USE_SQRELU ? nullptr : g.B3_sg_chunk_grid,
                        g.B3_sg_chunk_stride < 0 ? -g.B3_sg_chunk_stride : g.B3_sg_chunk_stride,
                        C::USE_SQRELU ? nullptr : g.B3_sg,
                        g.B3_sg_stride,
                        row_block_idx * 2 + cta_id,
                        col_block_idx * C::B_SC_SIZE,
                        row_block_idx,
                        col_block_idx,
                        i * (C::Kb / 128));

                    __syncwarp();
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    if (lane == 0) {
                        if (cta_id == 0) {
                            arrive(scales_prepared[stage], 1);
                        } else {
                            tma::cluster::arrive(scales_prepared[stage], 0, 1);
                        }
                    }
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0 && warp::elect_leader()) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out1_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out3_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(
                C::A_SC_TMEM_OFFSET);
            auto B1_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::B_SC_SIZE*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(
                C::B1_SC_TMEM_OFFSET);
            auto B3_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::B_SC_SIZE*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(
                C::B3_SC_TMEM_OFFSET);
            uint32_t output_phasebits = 0xFFFF0000;
            uint32_t prepared_phasebits = 0;
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                (void)row_block_idx;
                wait(outputs_finished, get_phasebit<1>(output_phasebits, 0));
                tensor_after_thread_sync();

                for (int i = 0; i < num_red_blocks; i++) {
                    if (has_any_chunk_grid) {
                        wait(scales_prepared[stage], get_phasebit<0>(prepared_phasebits, stage));
                    } else {
                        tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    }
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                        auto B1_sc_tm_subtile_0 = B1_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16+ii*C::B_SC_SIZE*16);
                        auto &B1_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B1[0].data[0])+16*32*ii);
                        load_mxnv_scale_async2(B1_sc_tm_subtile_0, B1_sc_sm_subtile_0);
                        if constexpr (!C::USE_SQRELU) {
                            auto B3_sc_tm_subtile_0 = B3_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16+ii*C::B_SC_SIZE*16);
                            auto &B3_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B3[0].data[0])+16*32*ii);
                            load_mxnv_scale_async2(B3_sc_tm_subtile_0, B3_sc_sm_subtile_0);
                        }
                        if constexpr (C::B_SC_SIZE == 2) {
                            auto B1_sc_tm_subtile_1 = B1_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16+ii*C::B_SC_SIZE*16+16);
                            auto &B1_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B1[1].data[0])+16*32*ii);
                            load_mxnv_scale_async2(B1_sc_tm_subtile_1, B1_sc_sm_subtile_1);
                            if constexpr (!C::USE_SQRELU) {
                                auto B3_sc_tm_subtile_1 = B3_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16+ii*C::B_SC_SIZE*16+16);
                                auto &B3_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B3[1].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B3_sc_tm_subtile_1, B3_sc_sm_subtile_1);
                            }
                        }
                    }
                    if (has_any_chunk_grid) {
                        update_phasebit<0>(prepared_phasebits, stage);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    auto A_sc_stage = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16);
                    auto B1_sc_stage = B1_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*C::B_SC_SIZE*16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16);
                    if (i == 0) {
                        if constexpr (C::USE_SQRELU) {
                            mm2_ABt(out1_tm, input_tiles[stage].A, input_tiles[stage].B1,
                                    A_sc_stage, B1_sc_stage, inputs_finished[stage]);
                        } else {
                            auto B3_sc_stage = B3_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*C::B_SC_SIZE*16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16);
                            mm2_ABt(out1_tm, input_tiles[stage].A, input_tiles[stage].B1,
                                    A_sc_stage, B1_sc_stage);
                            mm2_ABt(out3_tm, input_tiles[stage].A, input_tiles[stage].B3,
                                    A_sc_stage, B3_sc_stage, inputs_finished[stage]);
                        }
                    } else {
                        if constexpr (C::USE_SQRELU) {
                            mma2_ABt(out1_tm, input_tiles[stage].A, input_tiles[stage].B1,
                                     A_sc_stage, B1_sc_stage, inputs_finished[stage]);
                        } else {
                            auto B3_sc_stage = B3_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*C::B_SC_SIZE*16>>(stage*C::MMA_PER_TILE*C::B_SC_SIZE*16);
                            mma2_ABt(out1_tm, input_tiles[stage].A, input_tiles[stage].B1,
                                     A_sc_stage, B1_sc_stage);
                            mma2_ABt(out3_tm, input_tiles[stage].A, input_tiles[stage].B3,
                                     A_sc_stage, B3_sc_stage, inputs_finished[stage]);
                        }
                    }
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(output_phasebits, 0);
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
        auto out1_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out3_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        static_assert(SUBTILE_COLS == 32, "localCTA SwiGLU producer assumes 32-column epilogue slices");
        static_assert(C::EPI_PER_HALF == 4, "localCTA SwiGLU producer assumes four 32-column slices per 128-column chunk");
        __shared__ bf16_2 staged[C::NUM_HALVES][WARPGROUP_WARPS][C::EPI_PER_HALF][16][33];
        __shared__ float warp_amax[C::NUM_HALVES][WARPGROUP_WARPS];
        __shared__ float chunk_s_enc[C::NUM_HALVES];
        __shared__ float chunk_sg[C::NUM_HALVES];
        uint32_t output_phasebits = 0xFFFF0000;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            const int wg_warp = warpgroup::warpid();
            const int lane_id = warp::laneid();
            const int warp_row_base = (row_block_idx * 2 + cta_id) * (C::Mb / 2) + wg_warp * 32;

            wait(outputs_arrived, get_phasebit<0>(output_phasebits, 0));
            constexpr int B_BLOCKS_PER_OUTER_SG = 256 / C::Nb;
            const int b_outer_idx = col_block_idx / B_BLOCKS_PER_OUTER_SG;
            const float a_sg = g.A_sg[row_block_idx * g.A_sg_stride];
            const float gs1 = a_sg * g.B1_sg[b_outer_idx * g.B1_sg_stride];
            float gs3 = 1.0f;
            if constexpr (!C::USE_SQRELU) {
                gs3 = a_sg * g.B3_sg[b_outer_idx * g.B3_sg_stride];
            }

            #pragma unroll
            for (int half = 0; half < C::NUM_HALVES; ++half) {
                bf16_2 (*pairs)[16][33] = staged[half][wg_warp];
                float local_amax = 0.0f;
                #pragma unroll
                for (int e = 0; e < C::EPI_PER_HALF; ++e) {
                    const int epi = half * C::EPI_PER_HALF + e;
                    subtile_rt D1_acc;
                    warpgroup::load_async(
                        D1_acc,
                        out1_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                    subtile_rt D3_acc;
                    if constexpr (!C::USE_SQRELU) {
                        warpgroup::load_async(
                            D3_acc,
                            out3_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);

                    warp::mul(D1_acc, D1_acc, gs1);
                    if constexpr (!C::USE_SQRELU) {
                        warp::mul(D3_acc, D3_acc, gs3);
                    }
                    warpgroup::sync(1);
                    const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    if constexpr (C::USE_SQRELU) {
                        local_amax = fmaxf(
                        local_amax,
                        stage_sqrelu_pairs<C>(
                            D1_acc, pairs, e, lane_id));
                    } else {
                        local_amax = fmaxf(
                            local_amax,
                            stage_swiglu_pairs<C>(
                                D1_acc, D3_acc, pairs, e, lane_id,
                                warp_row_base, col_start, g.M, g.H));
                    }
                    warpgroup::sync(1);
                }
                local_amax = warp_reduce_max(local_amax);
                if (lane_id == 0) {
                    warp_amax[half][wg_warp] = local_amax;
                }
                warpgroup::sync(1);
            }
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
            tensor_after_thread_sync();
            update_phasebit<0>(output_phasebits, 0);

            #pragma unroll
            for (int half = 0; half < C::NUM_HALVES; ++half) {
                if (wg_warp == 0 && lane_id == 0) {
                    float amax = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < WARPGROUP_WARPS; ++w) {
                        amax = fmaxf(amax, warp_amax[half][w]);
                    }
                    if constexpr (C::AMAX_ONLY) {
                        atomic_max_float_bits(g.global_amax, amax);
                    } else if constexpr (C::V5_SCALAR_SG) {
                        const float global_amax = fmaxf(*g.global_amax, 0.0f);
                        chunk_s_enc[half] = v5_encode_scale(global_amax);
                        chunk_sg[half] = global_amax / V5_GLOBAL_SCALE_NUM;
                        if (row_block_idx == 0 && col_block_idx == 0 && cta_id == 0 && half == 0) {
                            g.row_sg[0] = chunk_sg[half];
                            g.col_sg[0] = chunk_sg[half];
                        }
                    } else {
                        chunk_s_enc[half] = localcta_encode_scale(amax);
                        chunk_sg[half] = amax / LOCALCTA_GLOBAL_SCALE_NUM;

                        const int row_chunk = row_block_idx * 2 + cta_id;
                        const int col_chunk = col_block_idx * C::NUM_HALVES + half;
                        const int row_sg_cols = g.H / 128;
                        const int col_sg_cols = g.M / 128;
                        g.row_sg[row_chunk * row_sg_cols + col_chunk] = chunk_sg[half];
                        g.col_sg[col_chunk * col_sg_cols + row_chunk] = chunk_sg[half];
                    }
                }
                warpgroup::sync(1);
            }

            if constexpr (C::AMAX_ONLY) {
                continue;
            }

            #pragma unroll
            for (int half = 0; half < C::NUM_HALVES; ++half) {
                bf16_2 (*pairs)[16][33] = staged[half][wg_warp];
                #pragma unroll
                for (int e = 0; e < C::EPI_PER_HALF; ++e) {
                    const int epi = half * C::EPI_PER_HALF + e;
                    const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    quantize_rows_from_stage<C>(
                        g, pairs, e, lane_id, warp_row_base, col_start,
                        chunk_s_enc[half], chunk_sg[half]);
                    quantize_cols_from_stage<C>(
                        g, pairs, e, lane_id, warp_row_base, col_start,
                        chunk_s_enc[half], chunk_sg[half]);
                }
                warpgroup::sync(1);
            }
        }
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace nvfp4_localcta_swiglu_quant_gemm
