#pragma once

#include "kittens.cuh"

namespace h_mxfp4_tile_carrier {

constexpr int TILE = 128;
constexpr float TILE_INV_ELEMENTS = 0x1p-14f;

__device__ __forceinline__ uint8_t fp4(float x) {
    const float a = fabsf(x);
    const uint8_t sign = static_cast<uint8_t>(__float_as_uint(x) >> 28) & 8u;
    const uint8_t e = static_cast<uint8_t>(a >= .25f) +
        static_cast<uint8_t>(a >= .75f) + static_cast<uint8_t>(a >= 1.25f) +
        static_cast<uint8_t>(a >= 1.75f) + static_cast<uint8_t>(a >= 2.5f) +
        static_cast<uint8_t>(a >= 3.5f) + static_cast<uint8_t>(a >= 5.0f);
    return sign | e;
}

__device__ __forceinline__ uint8_t fp4_pair(float x, float y, float rcp) {
    return fp4(x * rcp) | static_cast<uint8_t>(fp4(y * rcp) << 4);
}

// MX mode=1: encode-centric E8M0.  The scale always encloses amax.
__device__ __forceinline__ uint8_t e8m0_mode1(float amax) {
    if (!(amax > 0.0f)) return 0;
    uint32_t u = __float_as_uint(amax);
    uint8_t e = static_cast<uint8_t>((u >> 23) & 0xffu);
    if ((u & 0x7fffffu) != 0 && e < 0xfeu) ++e;
    return e;
}

__device__ __forceinline__ float fp4_rcp(uint8_t e) {
    if (e == 0) return 0.0f;
    return 6.0f * __uint_as_float(static_cast<uint32_t>(254 - e) << 23);
}

template <typename RT>
__device__ __forceinline__ float sumsq(const RT& x) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float a = __bfloat162float(x.tiles[i][j].data[k].x);
                const float b = __bfloat162float(x.tiles[i][j].data[k].y);
                s = fmaf(a, a, s);
                s = fmaf(b, b, s);
            }
        }
    }
    return s;
}

__device__ __forceinline__ float tile_rsqrt(float s, float* scratch, float eps) {
    const int lane = kittens::warp::laneid();
    const int warp = kittens::warpgroup::warpid();
    #pragma unroll
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffffu, s, d);
    if (lane == 0) scratch[warp] = s;
    kittens::warpgroup::sync(1);
    if (warp == 0) {
        float t = lane < kittens::WARPGROUP_WARPS ? scratch[lane] : 0.0f;
        #pragma unroll
        for (int d = 16; d; d >>= 1) t += __shfl_down_sync(0xffffffffu, t, d);
        if (lane == 0) scratch[kittens::WARPGROUP_WARPS] = rsqrtf(fmaf(t, TILE_INV_ELEMENTS, eps));
    }
    kittens::warpgroup::sync(1);
    return scratch[kittens::WARPGROUP_WARPS];
}

template <typename RT>
__device__ __forceinline__ void normalize_gamma_to_stage(
    const RT& z, kittens::bf16_2 (*pairs)[33], float r, const kittens::bf16* gamma,
    int col_base) {
    const int lane = kittens::warp::laneid();
    const int lane_byte = lane & 3;
    const int row_pair = lane >> 2;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int p = j * 8 + lane_byte;
            const int c0 = col_base + p * 2;
            const int c1 = c0 + 8;
            const auto& q = z.tiles[i][j].data;
            pairs[p][i * 16 + row_pair] = kittens::bf16_2{
                __float2bfloat16_rn((__bfloat162float(q[0].x) * r) * __bfloat162float(gamma[c0])),
                __float2bfloat16_rn((__bfloat162float(q[0].y) * r) * __bfloat162float(gamma[c0 + 1]))};
            pairs[p][i * 16 + row_pair + 8] = kittens::bf16_2{
                __float2bfloat16_rn((__bfloat162float(q[1].x) * r) * __bfloat162float(gamma[c0])),
                __float2bfloat16_rn((__bfloat162float(q[1].y) * r) * __bfloat162float(gamma[c0 + 1]))};
            pairs[p + 4][i * 16 + row_pair] = kittens::bf16_2{
                __float2bfloat16_rn((__bfloat162float(q[2].x) * r) * __bfloat162float(gamma[c1])),
                __float2bfloat16_rn((__bfloat162float(q[2].y) * r) * __bfloat162float(gamma[c1 + 1]))};
            pairs[p + 4][i * 16 + row_pair + 8] = kittens::bf16_2{
                __float2bfloat16_rn((__bfloat162float(q[3].x) * r) * __bfloat162float(gamma[c1])),
                __float2bfloat16_rn((__bfloat162float(q[3].y) * r) * __bfloat162float(gamma[c1 + 1]))};
        }
    }
}

template <typename G>
__device__ __forceinline__ void emit_32x32(
    const G& g, kittens::bf16_2 (*pairs)[33], int warp_row, int col_start) {
    const int lane = kittens::warp::laneid();
    const int row = warp_row + lane;
    kittens::bf16_2 cached[16];
    float amax = 0.0f;
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        cached[p] = pairs[p][lane];
        amax = fmaxf(amax, fabsf(__bfloat162float(cached[p].x)));
        amax = fmaxf(amax, fabsf(__bfloat162float(cached[p].y)));
    }
    const uint8_t e = e8m0_mode1(amax);
    const float rcp = fp4_rcp(e);
    uint8_t* row_ptr = reinterpret_cast<uint8_t*>(g.h_row_fp4);
    #pragma unroll
    for (int p = 0; p < 16; ++p)
        row_ptr[row * (g.h_cols / 2) + col_start / 2 + p] =
            fp4_pair(__bfloat162float(cached[p].x), __bfloat162float(cached[p].y), rcp);
    const int rsb = row / 128, jig = row % 32, grp = (row % 128) / 32;
    g.h_row_sc[((rsb * (g.h_cols / 128) + col_start / 128) * 512) + jig * 16 + grp * 4 + (col_start % 128) / 32] = e;

    const int local_col = lane;
    const int pair_col = local_col >> 1;
    const bool y = local_col & 1;
    float camax = 0.0f;
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        const auto a = pairs[pair_col][p * 2];
        const auto b = pairs[pair_col][p * 2 + 1];
        cached[p] = y ? kittens::bf16_2{a.y, b.y} : kittens::bf16_2{a.x, b.x};
        camax = fmaxf(camax, fabsf(__bfloat162float(cached[p].x)));
        camax = fmaxf(camax, fabsf(__bfloat162float(cached[p].y)));
    }
    const uint8_t ce = e8m0_mode1(camax);
    const float crcp = fp4_rcp(ce);
    const int gc = col_start + local_col;
    uint8_t* col_ptr = reinterpret_cast<uint8_t*>(g.h_col_fp4);
    #pragma unroll
    for (int p = 0; p < 16; ++p)
        col_ptr[gc * (g.h_rows / 2) + warp_row / 2 + p] =
            fp4_pair(__bfloat162float(cached[p].x), __bfloat162float(cached[p].y), crcp);
    const int chunk = (gc / 128) * (g.h_rows / 128) + warp_row / 128;
    const int idx = (gc % 32) * 16 + ((gc / 32) % 4) * 4 + ((warp_row / 32) % 4);
    g.h_col_sc[chunk * 512 + idx] = ce;
}

} // namespace h_mxfp4_tile_carrier
