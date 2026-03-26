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

### 5. SMEM layout mismatch
The quantizer writes FP4 bytes via raw `st.shared.b8` with **linear offsets**, but
`st_fp4e2m1_2<128,128>` uses TK's **128-byte swizzle** layout. When TMA loads FP4 from GMEM,
it applies the swizzle on the way into SMEM. Our direct writes skip TMA → data lands in wrong
locations → `unspecified launch failure`.

Fixed by switching direct FP4 writes to `out_tile.A.idx(...)`, which applies the same
128-byte swizzle the GEMM path expects.

### 6. CTA_AMAX warpgroup sync bugs
The CTA-level amax path had two synchronization problems:
1. `__syncthreads()` after `running_amax = 0.0f`, which deadlocked because only the
   quantizer warpgroup reaches that point.
2. `__syncwarp()` around the shared `warp_max_buf[4]` reduction scratch, which only
   synchronized each 32-thread warp rather than the full 128-thread quantizer warpgroup.

Fixed by using a dedicated `warpgroup::sync(bar_id)` sequence for:
- `running_amax` reset visibility
- `warp_max_buf` write completion before the inter-warp read
- `warp_max_buf` reuse safety
- `quant_buf` reuse safety before the next BF16 sub-tile TMA overwrites it

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

## Build / Run Notes

- `common.mk` now resolves `nvcc` from `CUDA_HOME` or `/usr/local/cuda/bin/nvcc`, so
  `make` does not depend on the caller's `PATH` being pre-populated.
- PyTorch extension builds embed Torch's library path as an rpath, so `_C` imports
  without a manual `LD_LIBRARY_PATH` export.
- `test_fused_gemm.py` now fails fast with a clear error when CUDA device access is unavailable.
- `test_fused_gemm.py` now benchmarks five paths:
  - separate `quantize + gemm`
  - fused A-only constant-scale
  - fused A-only CTA-amax
  - experimental both-bf16 fused constant-scale
  - experimental both-bf16 fused CTA-amax
- New public APIs for the experimental two-sided prototype:
  - `nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D)`
  - `nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D)`

## Experimental Both-BF16 Prototype

There is now a separate experimental kernel in
`nvfp4_fused_gemm_both_bf16.cuh`. It is a real one-kernel prototype that accepts
both operands as bf16 and quantizes both on the fly before MMA.

Architecture:

- 4 warpgroups per CTA
  - WG0: consumer / epilogue
  - WG1: A quantizer
  - WG2: B quantizer
  - WG3: MMA orchestrator
- Tile shape is currently fixed at `128 x 128 x 256`.
- CTA-amax mode tracks separate running amax values for A and B, then applies
  `a_sg_dec * b_sg_dec` in the epilogue.

Important limitations of this first version:

- It is a single-CTA prototype with no cross-tile reuse for either operand.
- It compiles and runs, but ptxas reports very large spills:
  - constant-scale: `868` bytes spill stores / `868` bytes spill loads
  - CTA-amax: `876` bytes spill stores / `876` bytes spill loads
- Because it does not reuse quantized A or B across neighboring output tiles, it is
  not performance-competitive with the separate baseline yet.

## Current Validation

- `compute-sanitizer --tool memcheck python3 -u test_fused_gemm.py 256 256 256`
  completed with `0` errors.
- `python3 -u test_fused_gemm.py 256 256 256`
- `python3 -u test_fused_gemm.py 1024 1024 1024`
- `python3 -u test_fused_gemm.py 2048 2048 2048`
- `python3 -u test_fused_gemm.py 4096 4096 4096`
- `python3 -u test_fused_gemm.py 256 384 256`
  all complete successfully.
- The new experimental both-bf16 public APIs also complete successfully on:
  - `256 x 256 x 256`
  - `1024 x 1024 x 1024`
  - `2048 x 2048 x 2048`
  - `4096 x 4096 x 4096`
- The current fused dispatch is:
  - `N % 256 == 0 && N <= 2048`: dual-column prototype `config<128, *, *, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>`
  - `N % 256 == 0 && N > 2048`: single-column wide-tile `config<256, 4, 8, 4, 2, false, USE_CTA_AMAX>`
  - `N % 256 != 0`: single-column fallback `config<128, *, *, 4, 2, false, USE_CTA_AMAX>`
- The dual-column constant-scale kernels currently compile with small but nonzero spills:
  `8` bytes spill stores / `12` bytes spill loads.
- A cross-CTA shared-A prototype exists in-tree, but it is **not** currently dispatched. It still
  miscomputes CTA1's imported A tile on `256 x 512 x 256` (left-half cosine `~0.988`, right-half
  cosine `~0.56` against the separate baseline), so it is gated off until the DSMEM handoff is correct.

## Current Measurements

### Sweep Summary

- `256`: fused constant is faster than separate.
- `1024`: fused constant is effectively at parity with separate.
- `2048`: fused constant remains slower than separate.
- `4096`: the restored wide-tile path is still much slower than separate; CTA-amax accuracy also collapses.

### Key Benchmark Breakdowns

`256 x 256 x 256`

- Separate quantize only: `0.018 ms`
- Separate GEMM only: `0.006 ms`
- Separate quantize + GEMM: `0.023 ms`
- Fused constant: `0.010 ms`
- Fused CTA-amax: `0.011 ms`
- Both-bf16 fused constant: `0.021 ms`
- Both-bf16 fused CTA-amax: `0.021 ms`

`1024 x 1024 x 1024`

- Separate quantize only: `0.019 ms`
- Separate GEMM only: `0.007 ms`
- Separate quantize + GEMM: `0.025 ms`
- Fused constant: `0.026 ms`
- Fused CTA-amax: `0.029 ms`
- Both-bf16 fused constant: `0.043 ms`
- Both-bf16 fused CTA-amax: `0.045 ms`

`2048 x 2048 x 2048`

- Separate quantize only: `0.021 ms`
- Separate GEMM only: `0.009 ms`
- Separate quantize + GEMM: `0.029 ms`
- Fused constant: `0.049 ms`
- Fused CTA-amax: `0.054 ms`
- Both-bf16 fused constant: `0.151 ms`
- Both-bf16 fused CTA-amax: `0.155 ms`

`4096 x 4096 x 4096`

- Separate quantize only: `0.026 ms`
- Separate GEMM only: `0.031 ms`
- Separate quantize + GEMM: `0.060 ms`
- Fused constant: `0.345 ms`
- Fused CTA-amax: `0.382 ms`
- Both-bf16 fused constant: `1.002 ms`
- Both-bf16 fused CTA-amax: `1.039 ms`

## Root Cause of Large-Shape Slowdown

The fused kernel quantizes A inside the per-output-tile loop:

- the quantizer loop iterates over `block_idx`, which encodes both `row_block_idx` and `col_block_idx`
- the BF16 A tile load depends only on `row_block_idx`
- therefore the same A row block is re-quantized once per output-column tile

This is why the current fused path behaves well at `256` but loses as `N` grows:

- `256` with `Nb=128`: only `2` column tiles, so duplicated A work is still small
- `1024` with `Nb=256`: `4` column tiles
- `2048` with `Nb=256`: `8` column tiles
- `4096` with `Nb=256`: `16` column tiles

The standalone pipeline quantizes A once and reuses it across all column tiles, so the current
fused design has an algorithmic disadvantage at larger `N`, independent of register spilling.

## Dual-Column Prototype Result

The literal `2 x 128` dual-column prototype was implemented and validated for both constant-scale
and CTA-amax paths. It is sanitizer-clean, and it helps at small/mid sizes:

- `256`: clear win over separate
- `1024`: near parity

But it does **not** fix the `2048+` wall, because its effective block width is still `256` columns.
That means it rearranges the wide-tile work into two `128`-wide accumulators without increasing how
many output columns a single A quantization feeds. At `4096`, the prototype was actually worse than
the existing wide-tile backend, so dispatch is currently gated to use it only up to `2048`.

## Updated Next Step

Further config tuning alone is unlikely to beat separate `quantize + gemm` on large shapes.
The next meaningful optimization is structural:

1. Increase the *effective* fused output width beyond the current `256` columns, so one A quantization
   genuinely feeds more output-column work.
2. The most plausible routes are:
   - a wider multi-output backend that reuses A across more than `256` columns
   - or a cluster/dataflow change where multiple output-column workers consume the same quantized A stage
3. Keep CTA-amax as a secondary mode; its `4096` accuracy collapse should not block constant-scale tuning.
4. If the cross-CTA route is revisited, fix the DSMEM A-tile handoff before re-enabling dispatch.
   The current prototype compiles and runs, but the copied/imported A payload on CTA1 is not bitwise
   equivalent to a local quantize.

## Both-BF16 Takeaway

The new two-sided prototype answers the immediate question: simply quantizing both bf16
operands inside one CTA is not enough to beat separate `quantize + gemm`.

- It is slower than the current A-only fused kernel at every tested size.
- It is much slower than the separate baseline at `2048` and `4096`.
- Its CTA-amax mode is numerically better than the current A-only fused CTA-amax at `4096`,
  but that comes with very large spill cost and no meaningful throughput win.

So the next meaningful speed path is still structural reuse, not “just fuse more work into
one CTA.” A winning both-bf16 design needs a 2D reuse schedule where quantized A and
quantized B each feed multiple output tiles before being discarded.
