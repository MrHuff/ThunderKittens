#pragma once

#include <type_traits>

namespace nvfp4_rope_epilogue {

struct rope_live64_desc {
    const float2* cs = nullptr;
    int seq_len = 0;
    int seq_mask = 0;

    __host__ __device__ inline bool enabled() const {
        return cs != nullptr && seq_len > 0;
    }
};

template <typename Packed>
__device__ inline Packed rotate_pair(const Packed& packed, float cos_v, float sin_v);

template <>
__device__ inline float2 rotate_pair<float2>(const float2& packed, float cos_v, float sin_v) {
    return float2{
        packed.x * cos_v - packed.y * sin_v,
        packed.y * cos_v + packed.x * sin_v,
    };
}

template <>
__device__ inline kittens::bf16_2 rotate_pair<kittens::bf16_2>(
    const kittens::bf16_2& packed,
    float cos_v,
    float sin_v
) {
    const float even = __bfloat162float(packed.x);
    const float odd = __bfloat162float(packed.y);
    return kittens::bf16_2{
        __float2bfloat16_rn(even * cos_v - odd * sin_v),
        __float2bfloat16_rn(odd * cos_v + even * sin_v),
    };
}

template <int HEAD_DIM, int ROTARY_OFFSET, int ROTARY_DIM, kittens::ducks::rt::row_layout RT>
__device__ inline void apply_inplace_live64(
    RT& tile,
    const rope_live64_desc& rope,
    int global_row_base,
    int global_col_base
) {
    static_assert(HEAD_DIM > 0, "RoPE head dimension must be positive");
    static_assert(ROTARY_OFFSET >= 0 && ROTARY_OFFSET % 2 == 0,
                  "RoPE offset must be non-negative and even");
    static_assert(ROTARY_DIM > 0 && ROTARY_DIM <= 64 && ROTARY_DIM % 2 == 0,
                  "RoPE dimension must be positive, even, and at most 64");
    static_assert(ROTARY_OFFSET + ROTARY_DIM <= HEAD_DIM,
                  "RoPE range must fit within each head");
    if (!rope.enabled()) {
        return;
    }

    // Live64 configurations emit 32-column epilogue tiles. DeepSeek's
    // [nope=128, rope=64] head layout is therefore tile-aligned, so reject
    // non-RoPE tiles once instead of doing a runtime modulo for every pair.
    const int head_col_base = global_col_base % HEAD_DIM;
    if (head_col_base < ROTARY_OFFSET ||
        head_col_base >= ROTARY_OFFSET + ROTARY_DIM) {
        return;
    }

    static_assert(
        std::is_same_v<typename RT::dtype, float2> || std::is_same_v<typename RT::dtype, kittens::bf16_2>,
        "RoPE epilogue supports float2 and bf16_2 register tiles"
    );

    constexpr int tile_row_dim = RT::tile_size_row;
    constexpr int tile_col_dim = RT::tile_size_col;
    const int warp_row_base = kittens::warpgroup::warpid() * RT::rows;
    const int warp_lane = kittens::warp::laneid();

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; ++k) {
                const int row =
                    global_row_base +
                    warp_row_base +
                    i * tile_row_dim +
                    (k % 2) * (tile_row_dim / 2) +
                    warp_lane / 4;
                const int local_col_even =
                    j * tile_col_dim +
                    (k / 2) * (tile_col_dim / 2) +
                    (warp_lane % 4) * 2;
                const int rotary_pair =
                    (head_col_base - ROTARY_OFFSET + local_col_even) >> 1;
                const float2 cs = rope.cs[(row & rope.seq_mask) * 32 + rotary_pair];
                tile.tiles[i][j].data[k] = rotate_pair(tile.tiles[i][j].data[k], cs.x, cs.y);
            }
        }
    }
}

} // namespace nvfp4_rope_epilogue
