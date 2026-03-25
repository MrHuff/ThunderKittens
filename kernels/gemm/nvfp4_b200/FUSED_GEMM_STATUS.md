# Fused NVFP4 Quantization+GEMM — Status & Walkthrough

## Architecture

3-warpgroup fused kernel: **WG0** (consumer/epilogue) · **WG1** (quantizer) · **WG2** (producer+MMA)

The quantizer loads BF16 activations via CTA-scope TMA, quantizes inline to FP4+scales,
and writes directly into the GEMM's pipeline SMEM slots — eliminating the separate
quantize→GMEM→TMA-load round-trip.

## Bugs Found & Fixed

### 1. `tma::expect_bytes` — B-only byte counts
The MMA warp's `expect_bytes` originally matched the full A+B original. Since A is now written
directly (not via TMA), the byte count must reflect **B-only TMA arrivals**.
After empirical testing, **1×sizeof(B_fp4x2_tile)** is correct for tiles, and
**B_SC_SIZE×sizeof(B_sc_tile)** for scales. The 2× factor in the original is due to
cluster-scope TMA arrivals from both CTAs.

### 2. `a_quant_done` wait missing in MMA warp
Added `wait(a_quant_done[stage])` before reading A data from SMEM — ensures the quantizer
is done writing before MMA reads.

### 3. Phase bit tracking for `a_quant_done` in warps 2 & 3
Producer warps 2 (scales loader) and 3 (tiles loader) waited on `a_quant_done` using
`get_phasebit<0>` but never called `update_phasebit<0>`. On the 2nd K-tile with the same
pipeline stage, the wait passed through immediately (stale phase). Fixed by adding
`update_phasebit<0>` after each wait.

### 4. `bf16_sub_arrived` phase hardcoded to 0
The quantizer's `wait(bf16_sub_arrived[sub], 0)` used a hardcoded phase. Since these barriers
are reused across K iterations, the phase must alternate. Fixed with a `bf16_sub_phase` counter
that XORs after each K iteration.

### 5. **SMEM layout mismatch** (current blocker)
The quantizer writes FP4 bytes via raw `st.shared.b8` with **linear offsets**, but
`st_fp4e2m1_2<128,128>` uses TK's **128-byte swizzle** layout. When TMA loads FP4 from GMEM,
it applies the swizzle on the way into SMEM. Our direct writes skip TMA → data lands in wrong
locations → `unspecified launch failure`.

## Swizzle Analysis

### TK's `st_fp4e2m1_2<128,128>` Swizzle
From [st.cuh](file:///workspace/codebases/fp4_matmul/ThunderKittens/include/types/shared/st.cuh#L107-L119):

```
swizzle_bytes  = 128
swizzle_repeat = 1024  (= swizzle_bytes * 8)
subtile_cols   = 128   (= swizzle_bytes / sizeof(fp4e2m1_2))

linear_addr = base + row * 128 + col
swizzle_bits = ((linear_addr % 1024) >> 7) << 4
swizzled_addr = linear_addr ^ swizzle_bits
```

This XORs bits [6:4] of the address with bits [9:7], which is the standard CUDA 128B swizzle.

### Scale Tile: No Swizzle
`st_hf<4, 256, false>` — the `false` template argument disables swizzling.
Scale writes can use linear offsets.

### Standalone Quantize Pipeline
From [persistent_quantize.cuh](file:///workspace/codebases/fp4_matmul/TK_quantisation/nvfp4_v5/persistent_quantize.cuh):
1. `v3_rowwise_scaling()` writes FP4 to a **linear** SMEM output buffer
2. TMA stores this linear buffer to GMEM
3. GEMM's TMA load applies swizzle on the way into the `st_fp4e2m1_2` tile

For scales, `swizzle_scales_row_inplace()` applies a separate swizzle (`koffset*512 + j*16 + grp*4 + k_byte`)
before TMA-storing. This swizzle matches the GEMM's scale tile layout.

## Fix Plan: Swizzled Direct Writes

Replace the raw `st.shared.b8` in `quantize_subtile` with a swizzle-aware write:

```cpp
// Instead of:
// offset = tile_row * 128 + fp4_col
// st.shared.b8 [base + offset], value

// Use TK's idx function:
uint32_t smem_addr = out_tile.A.idx(
    static_cast<uint32_t>(__cvta_generic_to_shared(&out_tile.A.data[0])),
    {tile_row, fp4_col});
asm volatile("{st.shared.b8 [%0], %1;}" :: "r"(smem_addr), "r"(value));
```

This uses the tile's own `idx()` method which applies the swizzle formula correctly.
The same approach works for the 4-byte scale writes via `st.shared.b32`.

## Remaining Issues

1. **`__syncthreads()` in CTA_AMAX mode**: The warpgroup-local amax reduction uses
   `__syncthreads()` which syncs ALL 384 threads across 3 warpgroups → deadlock.
   Must replace with warpgroup-scoped sync (named barrier or `__syncwarp` chain).

2. **Performance**: Register spilling (152-204 bytes) may need tuning after correctness.
