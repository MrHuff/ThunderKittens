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
- The current prototype uses a `LOAD_PIPE_DEPTH=2` ping-pong pipeline for
  quantized FP4 tiles and FP8 scales.
- CTA-amax mode tracks separate running amax values for A and B, then applies
  `a_sg_dec * b_sg_dec` in the epilogue.

Important limitations of the current version:

- It is a single-CTA prototype with no cross-tile reuse for either operand.
- The first pipelined attempt broke correctness until two handoff bugs were fixed:
  - the BF16-subtile TMA semaphore phase must toggle per reduction block, not per stage slot
  - the TMEM scale scratch needs true per-stage storage rather than aliasing stage `0`
- The repaired ping-pong version compiles with much smaller spills:
  - constant-scale: `108` bytes spill stores / `120` bytes spill loads
  - CTA-amax: `0` bytes spill stores / `0` bytes spill loads
- Because it does not reuse quantized A or B across neighboring output tiles, it is
  not performance-competitive with the separate baseline yet.
- Follow-up tuning on 2026-03-26 added compile-time support for deeper load-pipe variants,
  but that alone did not move the large shapes:
  - a 3-stage single-column variant kept `2048` roughly flat (`0.122 ms`) and made `4096`
    much worse (`1.951 ms` constant-scale)
  - a same-CTA dual-column `4096+` experiment reduced the constant-scale regression versus
    the 3-stage try, but was still slower than the known-good 2-stage single-column path and
    the CTA-amax sweep stopped making forward progress at `4096`
  - a 5-warpgroup "parallel B quantizer" follow-up for the dual-column constant path also lost:
    it compiled only after shrinking epilogue staging, but ptxas reported `356/404` bytes of
    spills and `16` barriers, and the `4096` both-bf16 constant path regressed to `2.050 ms`
  - the public both-bf16 dispatcher is therefore back on the original 2-stage single-column
    ping-pong path while the generalized config plumbing stays available for future experiments
- A new developer-only same-CTA quad-column experiment is now wired in-tree:
  - kernel support exists for `COL_TILES_PER_BLOCK=4`, so one quantized A stage can feed
    four adjacent `128`-column bf16-B tiles inside a single CTA before the stage is recycled
  - to fit shared memory and cut bring-up overhead, the current debug dispatch uses
    `LOAD_PIPE_DEPTH=1`, `EPI_PIPE_DEPTH=1`, and `NUM_D_TILES=1`; this keeps the experiment
    focused on column reuse rather than overlapping loads or a deep epilogue
  - new debug entrypoints:
    - `nvfp4_fused_gemm_both_bf16_quadcol_debug`
    - `nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug`
  - test harness entry:
    - `python3 -u test_fused_gemm.py --both-bf16-quadcol 1024 4096 4096`
  - this experiment is intentionally **not** the default public both-bf16 path yet; it was
    added during a no-GPU session and still needs B200 correctness/perf validation
  - compile-only signal is poor even after trimming the config:
    - constant-scale: `1984` bytes stack, `2300` bytes spill stores / `2300` bytes spill loads
    - CTA-amax: `2000` bytes stack, `2312` bytes spill stores / `2348` bytes spill loads
    - so wider same-CTA reuse by itself does not currently look promising without a deeper
      register-pressure rewrite or a different work partition
- A new developer-only both-bf16 cross-CTA shared-A experiment is also wired in-tree:
  - one CTA quantizes A once and exports canonical FP4 bytes plus A scales
  - the sibling CTA imports that A stage through cluster shared memory, then quantizes its own B locally
  - each CTA still owns one local `128`-column output tile, so the cluster gets A reuse without the
    quad-column same-CTA register blow-up
  - new debug entrypoints:
    - `nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug`
    - `nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug`
  - test harness entry:
    - `python3 -u test_fused_gemm.py --both-bf16-shared-a-2cta 1024 2048 2048`
  - compile-only signal is encouraging relative to the quad-column path:
    - constant-scale: `96` bytes stack, `44` bytes spill stores / `56` bytes spill loads
    - CTA-amax: `48` bytes stack, `0` bytes spill stores / `0` bytes spill loads
  - the manual `ld.shared::cluster` receive path is now replaced by a DSMEM-style async transport:
    - sender-side remote `expect_bytes(...)`
    - async shared-cluster remote stores into CTA1-owned receive scratch
    - CTA1 waits on a local completion semaphore before importing the canonical FP4 tile
  - runtime bring-up on `256 x 512 x 256` now confirms that the backend launches and both CTAs produce
    strongly correlated output tiles, but the transport is still not byte-exact:
    - constant-scale dump mismatches remain (`A tile: 16354`, `A scales: 2039`)
    - CTA-amax dump mismatches remain (`A tile: 8532`, `A scales: 2048`)
    - constant-scale tile cosine matrix versus separate now has all four `128`-column tiles on-diagonal
      at about `0.976-0.984`, so the path is no longer "dead right half", but a small number of
      catastrophic outliers still make the full-output cosine only `0.694`
    - CTA-amax also improved from the original dead-half failure mode, but still only reaches
      about `0.705` cosine against the separate baseline
  - public both-bf16 dispatch remains unchanged; this backend is still debug-only

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
- The quad-column same-CTA both-bf16 experiment currently has compile coverage only from this
  session; runtime validation is still pending on a GPU-enabled B200 host.
- The cross-CTA shared-A both-bf16 experiment has now been rerun on:
  - `python3 -u test_fused_gemm.py --both-bf16-shared-a-2cta 256 512 256`
  It launches and produces sane tile structure, but it still fails the dump-level transport gate and
  is not yet safe to promote.
- The current fused dispatch is:
  - `N % 256 == 0 && N <= 2048`: dual-column prototype `config<128, *, *, 4, 2, false, USE_CTA_AMAX, 256, true, 2, 2>`
  - `N % 256 == 0 && N > 2048`: single-column wide-tile `config<256, 4, 8, 4, 2, false, USE_CTA_AMAX>`
  - `N % 256 != 0`: single-column fallback `config<128, *, *, 4, 2, false, USE_CTA_AMAX>`
- The dual-column constant-scale kernels currently compile with small but nonzero spills:
  `8` bytes spill stores / `12` bytes spill loads.
- A cross-CTA shared-A prototype exists in-tree, but it is **not** currently dispatched. It still
  miscomputes the DSMEM-imported A/scales on `256 x 512 x 256`, even after replacing the original
  manual cluster loads. The current debug run is no longer a "dead half" failure; instead, all four
  `128`-column tiles are structurally correlated with the separate baseline, but dump mismatches and
  a handful of extreme outliers still keep the full-output cosine low (`~0.694` constant,
  `~0.705` CTA-amax). It stays gated off until that transport/scale mismatch is fully resolved.

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
- Both-bf16 fused constant: `0.011 ms`
- Both-bf16 fused CTA-amax: `0.012 ms`

`1024 x 1024 x 1024`

- Separate quantize only: `0.019 ms`
- Separate GEMM only: `0.007 ms`
- Separate quantize + GEMM: `0.025 ms`
- Fused constant: `0.026 ms`
- Fused CTA-amax: `0.029 ms`
- Both-bf16 fused constant: `0.029 ms`
- Both-bf16 fused CTA-amax: `0.030 ms`

`2048 x 2048 x 2048`

- Separate quantize only: `0.021 ms`
- Separate GEMM only: `0.009 ms`
- Separate quantize + GEMM: `0.029 ms`
- Fused constant: `0.049 ms`
- Fused CTA-amax: `0.054 ms`
- Both-bf16 fused constant: `0.122 ms`
- Both-bf16 fused CTA-amax: `0.127 ms`

`4096 x 4096 x 4096`

- Separate quantize only: `0.026 ms`
- Separate GEMM only: `0.031 ms`
- Separate quantize + GEMM: `0.060 ms`
- Fused constant: `0.345 ms`
- Fused CTA-amax: `0.382 ms`
- Both-bf16 fused constant: `0.866 ms`
- Both-bf16 fused CTA-amax: `0.901 ms`

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

The two-sided prototype now answers a more specific question: a real ping-pong fused kernel
that quantizes both bf16 operands in one CTA can be made correct and materially faster than
the first cut, but it still does not beat separate `quantize + gemm`.

- The repaired ping-pong path is much faster than the original both-bf16 prototype:
  - `256`: `0.021 ms` -> `0.011 ms`
  - `1024`: `0.043 ms` -> `0.029 ms`
  - `2048`: `0.151 ms` -> `0.122 ms`
  - `4096`: `1.002 ms` -> `0.866 ms`
- A same-CTA dual-column follow-up (`COL_TILES_PER_BLOCK=2`) was also tried and kept
  correct, but it was slower than the single-column ping-pong baseline:
  - `256`: `0.023 ms`
  - `1024`: `0.064 ms`
  - `2048`: `0.137 ms`
  so it is currently gated off.
- It is still slower than the current A-only fused kernel at every tested size.
- It is still much slower than the separate baseline at `2048` and `4096`.

So the next meaningful speed path is still structural reuse, not “just fuse more work into
one CTA.” A winning both-bf16 design needs a 2D reuse schedule where quantized A and
quantized B each feed multiple output tiles before being discarded.

## Shared-A 2CTA Debug Update

Latest isolated repro:

```bash
python3 -u test_fused_gemm.py --both-bf16-shared-a-2cta 256 512 256
```

Current state:

- The catastrophic errors are still confined to the odd `128`-column tiles, which means the
  remaining failure is still on the `cta_id == 1` path.
- The diagonal tile cosines are high (`~0.98-0.99`), but a small number of CTA1 outputs are
  catastrophically wrong:
  - constant-scale outliers jump to values like `+-262144`
  - CTA-amax outliers collapse toward `0`
- The strongest recent experiments did **not** fix it:
  - leader-only bulk DSMEM copies for A/scales instead of striped `st.async`
  - adding `fence.proxy.async.shared::cluster`
  - splitting the FP4-payload and scale-payload completion onto separate semaphores

Useful interpretation:

- The transport is still the leading suspect, but the newer outlier print shows the failure mode
  more precisely than before:
  - the bad coordinates live in tiles `1` and `3`, i.e. the CTA1-owned output tiles
  - CTA0-owned tiles remain in the expected accuracy band
- So the next highest-value debug move is to bypass shared-A on CTA1 temporarily
  and let CTA1 quantize A locally. If that fixes the odd tiles immediately, the rest of the
  CTA1 B/MMA/epilogue path is fine and the problem is purely the shared-A import path.

### CTA1 Local-A Diagnostic Result

That diagnostic is now implemented as a separate developer-only backend. On the same
`256 x 512 x 256` repro:

- CTA1-local-A makes the dump checks go fully bitwise-equal for both constant-scale and CTA-amax:
  - A tile mismatches: `0`
  - A scale mismatches: `0`
- End-to-end numerics also snap back to the healthy both-bf16 baseline:
  - constant-scale cosine vs separate: `0.9758429527`
  - CTA-amax cosine vs separate: `0.9935339093`

So the remaining bug is confirmed to be transport-only. CTA1's local B quantization, MMA,
TMEM use, and epilogue are all fine when A import is removed from the equation.

### Current Best Shared-A Transport Checkpoint

The best stable shared-A receive experiment so far is:

- CTA1 pulls the remote payload with `copy_shared_cluster_bytes_b8(...)`
- then does a `warpgroup::sync(...)`
- then copies scales into `input_scales` and imports the canonical FP4 bytes into the local swizzled A tile

Observed result on the same repro:

- Constant-scale dump mismatches improve a lot, but are not fixed:
  - A tile mismatches: `2099`
  - A scales mismatches: `2048`
- Constant-scale no longer throws huge `+-262144` outliers or launch-fails.
  The CTA1-owned output tiles now collapse cleanly to zero instead:
  - tiles `1` and `3` have cosine `0.0000` vs separate
  - full-output cosine is `0.6915594935`
- CTA-amax remains partially live but still wrong:
  - full-output cosine is `0.7048626542`

One follow-up hybrid attempt (`b32` for scales, `b8` for FP4) regressed immediately back to
an unspecified launch failure, so the all-`b8` receive with the added post-copy sync is the
best current checkpoint.

Practical conclusion:

- The transport bug is now narrowed further:
  - CTA1-local quantization proves the consumer path is good
  - all-`b8` receive plus sync proves the raw CTA1 payload can be made much less corrupted
  - but the received scale scratch is still effectively dead, which is why constant-scale
    odd tiles zero out instead of matching the baseline
- The next fix should focus specifically on why CTA1's copied A-scale scratch is not surviving
  the receive path, rather than continuing to question the rest of the GEMM pipeline.

## Dedicated-Producer 4WG Refactor

Implemented a fresh single-CTA both-bf16 production backend with a real 4WG split:

- `WG0`: dedicated producer for bf16 A/B TMA fetches
- `WG1`: A quantizer only
- `WG2`: B quantizer only
- `WG3`: MMA plus epilogue/store

This backend was implemented and benchmarked behind the existing public API shape band, but it
is currently left **gated off** because it did not beat the older single-CTA ping-pong kernel.
The public both-bf16 dispatcher is therefore still on the older 2-stage path.

### What Changed

- The new backend keeps the quantized stage ring (`LOAD_PIPE_DEPTH=3`) but moves raw A/B
  fetch issuance out of the quantizer warpgroups.
- Because a full raw bf16 stage ring does not fit in SMEM for this kernel shape, the producer
  uses single raw A and B staging buffers and drives them independently with separate producer
  warps inside `WG0`.
- Epilogue/store moved from the old consumer warpgroup into `WG3` so `WG0` stays load-only.

### Validation

Build:

```bash
make -B -j1
```

Public runs:

```bash
python3 -u test_fused_gemm.py 256 256 256
python3 -u test_fused_gemm.py 1024 1024 1024
python3 -u test_fused_gemm.py 2048 2048 2048
python3 -u test_fused_gemm.py 4096 4096 4096
```

Sanitizer smoke:

```bash
compute-sanitizer --tool memcheck python3 -u test_fused_gemm.py 256 256 256
```

Current result: all public paths still run, and the sanitizer smoke passed with `0` errors.
The dedicated-producer measurements below were taken before the dispatcher was gated back to the
older public both-bf16 path.

### Benchmark Outcome

The new 4WG backend is correct, but it does **not** beat separate `quantize + gemm`.
It also does not beat the older both-bf16 ping-pong kernel on the large-shape band.

Measured dedicated-producer both-bf16 timings:

- `2048`: dedicated-producer const `0.128 ms`, CTA-amax `0.130 ms`
- `4096`: dedicated-producer const `0.894 ms`, CTA-amax `0.929 ms`

Reference baselines from the same runs:

- `2048`: separate `0.029 ms`, A-only fused `0.049 ms`
- `4096`: separate `0.060 ms`, A-only fused `0.345 ms`

Compile-time signal for the new dedicated-producer both-bf16 config:

- constant: `104` bytes spill stores / `208` bytes spill loads
- CTA-amax: `272` bytes spill stores / `512` bytes spill loads

So decoupling the fetch issue path from the quantizers helped a little structurally, but it
did not change the main economics: the kernel is still dominated by on-the-fly quantization
work with too little reuse per quantized operand tile.

### Current Takeaway

This refactor answers the utilization question more clearly:

- the old both-bf16 kernel was indeed making quantizer warpgroups do double duty
- separating producer from quantization is feasible and the implementation now exists in-tree
- but overlap alone is not enough to win at `2048+`

The next meaningful speed path is still structural reuse, not more local pipeline cleanup.
To beat separate `quantize + gemm`, the both-bf16 design likely needs one or both of:

- wider effective reuse of quantized A across more output columns
- wider effective reuse of quantized B across more output rows

The A-only fused path was left on the existing production backend in this round. The public
both-bf16 dispatcher was also left on the older production backend after the dedicated-producer
measurements came back slower.

## Shared-B 2CTA Debug Split Results

Current work is on the new internal shared-B `2CTA` backend only. Public dispatch is still
unchanged.

The most useful split results so far are:

- `producer-only` passes for both constant and CTA-amax at `256 x 256 x 256`
- `A-wait-only` passes for both constant and CTA-amax at `256 x 256 x 256`
- `B-wait-only` passes for both constant and CTA-amax at `256 x 256 x 256`
- `A-quant-only` passes at `256 x 256 x 128`, but hangs on the constant kernel at `256 x 256 x 256`
- `B-quant-only` passes the constant kernel at `256 x 256 x 128`, then hangs on CTA-amax
- `B-quant-only` hangs on the constant kernel at `256 x 256 x 256`
- `A-stage1-quant-only` passes for both constant and CTA-amax at `256 x 256 x 256`
- `B-stage1-quant-only` reaches CTA-amax launch at `256 x 256 x 256`, so the constant stage-1
  pass completes there too
- `A-quant-per-stage-only` still hangs on the constant kernel at `256 x 256 x 256`
- `A-quant-then-wait-only` also hangs on the constant kernel at `256 x 256 x 256`

Current interpretation:

- the shared-B producer and raw-arrival semaphore contract is working
- the second raw BF16 stage itself is readable; stage-1-only bring-up is clean
- the constant-path `K=256` hang needs the **first quantize pass** to happen before the second-stage
  progress breaks
- giving transport-only quantization distinct destination slots per stage still does **not** fix
  the hang, so the failure is not just reuse of `input_tiles[0]` / `input_scales[0]`
- `A-quant-then-wait-only` hanging means the first A quantization pass is already poisoning the
  second-stage wait path before the second quantization body even starts
- the strongest current hypothesis is shared-memory corruption from the first quantization pass
  clobbering stage-1 raw data or semaphore state
- the B CTA-amax path still has a separate single-stage bug in its peer-amax / global-scale path

The current highest-value next step is to verify what the first quantization pass corrupts in the
shared-B transport-only backend, for example by dumping or checksumming stage-1 raw buffers and
nearby semaphore storage before and after the first quantize iteration. Once the constant path is
clean, come back to the separate B CTA-amax peer-amax exchange bug.

## Shared-B 2CTA Current Checkpoint

The shared-B `2CTA` debug backend advanced substantially after the earlier split work:

- restoring the older cluster-MMA contract in the real kernel helped immediately:
  - `a_quant_done` and `b_quant_done` are cluster-scoped again
  - only `cta_id == 0 && warp_id == 0` issues `mm2_ABt` / `mma2_ABt`
  - non-leader CTA uses `tma::cluster::arrive(..., 0, 1)` instead of running cluster MMA itself
- with that fix, `256 x 256 x 256` full shared-B debug became numerically sane for both modes

Measured `256 x 256 x 256` shared-B debug quality versus separate quantize+GEMM:

- constant:
  - full cosine `0.975947`
  - quadrants all land around `0.975-0.976`
- CTA-amax:
  - full cosine `0.994232`
  - quadrants land around `0.993-0.996`

So the shared-B dataflow is now correct enough at the bring-up shape to be worth pushing on.

### Shared-B Standalone Checkpoint

The shared-B `2CTA` debug backend now has a standalone runner in the non-`TORCH_COMPILE`
path of `nvfp4_b200_gemm.cu`, with a dedicated local build target:

```bash
make -B -j1 shared_b_debug_runner
./nvfp4_b200_shared_b_debug.out --mode shared_b_const --m 1024 --n 1024 --k 1024
./nvfp4_b200_shared_b_debug.out --mode shared_b_cta --m 1024 --n 1024 --k 1024
```

The crucial peer-amax fix was replacing the earlier async scalar hop with a direct remote
shared-memory `b32` store plus explicit remote mbarrier arrival. The old failure signature
was:

- repeated WG2 `pre-peer`
- no `post-peer`
- `1024` CTA-amax hang in both transport-only and full shared-B debug modes

Current standalone status:

- `shared_b_const 256 256 256`: passes
- `shared_b_cta 256 256 256`: passes
- `shared_b_const 1024 1024 1024`: passes
- `shared_b_cta 1024 1024 1024`: passes
- `compute-sanitizer --tool memcheck ./nvfp4_b200_shared_b_debug.out --mode shared_b_cta --m 1024 --n 1024 --k 1024`: `0` errors
- `shared_b_const 2048 2048 2048`: passes
- `shared_b_cta 2048 2048 2048`: passes

Representative standalone summaries:

- `1024` constant:
  - finite `1048576 / 1048576`
  - checksum `58.5263`
  - abs max `1.64844`
- `1024` CTA-amax:
  - finite `1048576 / 1048576`
  - checksum `-100.787`
  - abs max `1.64844`
- `2048` constant:
  - finite `4194304 / 4194304`
  - checksum `-486.919`
  - abs max `0.980469`
- `2048` CTA-amax:
  - finite `4194304 / 4194304`
  - checksum `346.124`
  - abs max `1.79688`

One debug-only branch is still broken:

- `shared_b_transport_cta 1024 1024 1024` still fails with `unspecified launch failure`

That no longer blocks the main shared-B backend. The full shared-B CTA-amax kernel now runs
through WG2 peer exchange, WG3 scale wait, and cluster MMA at `1024` and `2048`.

### Python Validation Checkpoint

After rebuilding the extension with the same peer-amax fix, the Python shared-B debug harness
also completes cleanly again:

```bash
CUDA_VISIBLE_DEVICES=1 TK_SKIP_CUDA_PREFLIGHT=1 python3 -u test_fused_gemm.py --both-bf16-shared-b-2cta 1024 1024 1024
CUDA_VISIBLE_DEVICES=1 TK_SKIP_CUDA_PREFLIGHT=1 python3 -u test_fused_gemm.py --both-bf16-shared-b-2cta 2048 2048 2048
```

`1024 x 1024 x 1024` versus separate quantize+GEMM:

- shared-B constant:
  - max diff `1.187500`
  - mean diff `0.194954`
  - cosine `0.9711900949`
- shared-B CTA-amax:
  - max diff `2.078125`
  - mean diff `0.232026`
  - cosine `0.9867668152`

`2048 x 2048 x 2048` versus separate quantize+GEMM:

- shared-B constant:
  - max diff `1.203125`
  - mean diff `0.183925`
  - cosine `0.9736415148`
- shared-B CTA-amax:
  - max diff `2.578125`
  - mean diff `0.305810`
  - cosine `0.9868634939`

At both shapes, the shared-B debug outputs match the current public both-bf16 path very closely.
Public dispatch is still unchanged: the shared-B backend is debug-only until it is benchmarked
against separate `quantize + gemm`.

### Host Runtime Noise

The Torch/CUDA host stack is still somewhat noisy on this machine:

- fresh Python processes sometimes hit `cudaGetDeviceCount() -> Error 304`
- `TK_SKIP_CUDA_PREFLIGHT=1` still helps avoid false negatives
- stale GPU jobs can still poison new Torch sessions even when `nvidia-smi` looks fine

Those are harness issues, not current kernel blockers.

### Immediate Next Step

The next work item is no longer deadlock debugging. The shared-B backend is now ready for
performance comparison:

1. add or reuse a timing path for the shared-B debug backend
2. compare shared-B constant and CTA-amax against:
   - separate quantize+GEMM
   - current public both-bf16 backend
3. only then decide whether to promote shared-B for `2048+`

## Shared-B Public Promotion Checkpoint

The shared-B `2CTA` backend is now promoted into the public both-bf16 dispatcher for the
large-shape band only:

- `M >= 2048`
- `N >= 2048`
- `M % 256 == 0`
- `N % 256 == 0`
- `K % 256 == 0`

Smaller shapes still use the older single-CTA ping-pong backend.

Public benchmark results after promotion:

`2048 x 2048 x 2048`

- separate `quantize + gemm`: `0.029 ms`
- public both-bf16 const: `0.070 ms`
- public both-bf16 CTA-amax: `0.076 ms`

This replaces the old public both-bf16 numbers at roughly:

- const: `0.126 ms`
- CTA-amax: `0.129 ms`

So the promoted shared-B path improves public both-bf16 throughput by about `1.8x` at `2048`.

`4096 x 4096 x 4096`

- separate `quantize + gemm`: `0.063 ms`
- public both-bf16 const: `1.073 ms`
- public both-bf16 CTA-amax: `1.409 ms`

On the same shape, the old public both-bf16 path was around:

- const: `1.959 ms`
- CTA-amax: `2.739 ms`

So the promoted shared-B path also improves the public both-bf16 path materially at `4096`,
but it still does **not** beat separate `quantize + gemm`.

Numerically, the promoted public both-bf16 path remains sane after promotion:

- `2048` const cosine vs separate: `0.9736415148`
- `2048` CTA-amax cosine vs separate: `0.9866641760`
- `4096` const cosine vs separate: `0.9742028713`
- `4096` CTA-amax cosine vs separate: `0.9861819148`

Important conclusion:

- shared-B is the best current public both-bf16 fused path for large shapes
- but the deeper structural conclusion did not change: this direct streaming design is still
  slower than separate `quantize + gemm` at large square shapes because quantization reuse is
  still tile-local rather than global

### Why More Pipelining Probably Will Not Fix It

Nsight Compute on the promoted shared-B constant backend at `2048 x 2048 x 2048`:

```bash
ncu --target-processes all \
    --section SpeedOfLight \
    --section SchedulerStats \
    --section WarpStateStats \
    --section Occupancy \
    ./nvfp4_b200_shared_b_debug.out --mode shared_b_const --m 2048 --n 2048 --k 2048
```

Key counters from that run:

- compute throughput: about `28.1%`
- memory throughput: about `20.5%`
- DRAM throughput: about `1.66%`
- active warps per scheduler: about `3.36`
- eligible warps per scheduler: about `0.42`
- cycles with no eligible warp: about `63.8%`
- achieved occupancy: about `20.8%`
- theoretical occupancy: `25%`
- top Nsight guidance: about `4.1` cycles of L1TEX scoreboard dependency, about `44.3%` of
  the issue gap

Interpretation:

- the promoted shared-B kernel is not bandwidth-bound
- it is also not tensor-core-saturated
- it is mostly latency/eligibility-limited: too few eligible warps, too much dependency waiting,
  and only one cluster resident because the kernel is already large in threads, registers, and SMEM

That means more overlap can still improve constants a bit, but it does **not** change the main
economic problem: the kernel still re-quantizes A and B for each `256 x 256` output block instead
of reusing those quantized strips globally the way separate `quantize + gemm` does.

We also tried one last low-risk tuning sweep on the shared-B backend:

- `LOAD_PIPE_DEPTH=1` instead of `2`
- `NUM_D_TILES=1` instead of `2`

Observed outcome:

- `LOAD_PIPE_DEPTH=1` improved the shared-B constant path at `2048` by only about `1%`
- the same `LOAD_PIPE_DEPTH=1` variant became unstable or incorrect at `4096`
- `NUM_D_TILES=1` was slower at `2048`

So the straightforward pipe-depth / output-buffer knobs do not rescue the large-shape case.
The most likely remaining path to actually beat separate `quantize + gemm` would need much
larger-lifetime reuse of quantized strips, not just deeper or cleaner single-block streaming.

### Mirror Hybrid Control: A Pre-Quantized, B Streamed

The existing production A-only fused kernel already serves as the first hybrid control:

- **A streamed / B pre-quantized**: public `nvfp4_fused_gemm`

To complete the feasibility picture, we added a debug-only mirror hybrid on top of the shared-B
`2CTA` backend:

- **A pre-quantized / B streamed**: debug-only `prequant_a_shared_b_*`

That mirror path now:

- builds in both the extension and the standalone runner
- runs cleanly at `256` and `1024` in standalone constant and CTA-amax modes
- passes standalone memcheck at `1024` CTA-amax
- runs through the Python harness at `256`, `1024`, `2048`, and `4096`

What this control was meant to tell us:

- if **A pre-quantized / B streamed** got close to separate, then B streaming would look like the
  better direction
- if it still lost badly, then the conclusion would be stronger: current hardware is mainly losing
  on block-local streaming itself, not just on which operand gets streamed

Observed Python comparison against separate `quantize + gemm`:

`2048 x 2048 x 2048`

- separate `quantize + gemm`: `0.028 ms`, cosine `0.9909662008` vs bf16 ref
- A-only fused const (`A streamed / B pre-quantized`): `0.048 ms`, cosine `0.9870330691` vs separate
- mirror hybrid const (`A pre-quantized / B streamed`): `0.051 ms`, cosine `0.6620744467` vs separate
- shared-B both-bf16 const (`A streamed / B streamed`): `0.069 ms`, cosine `0.9736415148` vs separate

`4096 x 4096 x 4096`

- separate `quantize + gemm`: `0.060 ms`, cosine `0.9909696579` vs bf16 ref
- A-only fused const (`A streamed / B pre-quantized`): `0.609 ms`, cosine `0.9869642258` vs separate
- mirror hybrid const (`A pre-quantized / B streamed`): `0.609 ms`, cosine `0.6624910831` vs separate
- shared-B both-bf16 const (`A streamed / B streamed`): `1.064 ms`, cosine `0.9742028713` vs separate

Interpretation:

- performance-wise, the mirror hybrid lands in essentially the same band as the existing A-only
  fused path, especially at `4096`
- that is the useful new lesson: **streaming exactly one operand is materially cheaper than
  streaming both operands**, but still far slower than separate `quantize + gemm` at large square
  shapes
- the mirror hybrid's numerics are currently **not** trustworthy; cosine stays near `0.66` across
  shapes, which strongly suggests a remaining pre-quantized-A staging/layout bug in this debug path
  rather than a meaningful algorithmic accuracy limit
- because of that, the mirror path should be treated as a feasibility/control experiment only, not
  as a candidate backend

Even with that caveat, the broad conclusion is now clearer:

- on current hardware, **one-sided streaming is feasible** and lands in the same rough performance
  band regardless of which side pays the streaming tax
- but **tile-local direct streaming still does not match separate `quantize + gemm`** on large
  square GEMMs
- the missing ingredient is still wider-lifetime reuse of quantized strips, not more pipeline depth

## 2026-03-29: A-only shared-A transport fixed, but backend still loses

We revisited the older A-only cross-CTA shared-A debug backend in `nvfp4_fused_gemm.cuh` to see
whether a structurally better one-sided reuse path could beat the current public A-only fused
kernel.

The important transport fix was:

- the shared-A receiver semaphores (`a_payload_arrived` / `a_scale_arrived`) were waiting on the
  wrong phase-bit half
- the A-scale export path was also not matching the working shared-B contract
- after switching those waits to the lower phase-bit half and making A-scale export follow the
  same `local input_scales -> export scratch -> async cluster store` pattern as shared-B, direct
  dump probes became exact for both constant-scale and CTA-amax

Clean direct dump result at `256 x 512 x 256`:

- constant dump: imported A payload exact, imported A scales exact
- CTA-amax dump: imported A payload exact, imported A scales exact

Direct end-to-end kernel result at `256 x 512 x 256`:

- shared-A constant runs cleanly and matches separate with cosine about `0.9876`
- shared-A CTA-amax runs cleanly and matches separate with cosine about `0.9966`

Larger-shape follow-up:

- `2048` shared-A constant now launches and stays finite
- `2048` shared-A CTA-amax still hits `cudaErrorLaunchFailure`

Constant-only timing comparison:

- `2048`: separate `0.0299 ms`, public A-only fused `0.0500 ms`, shared-A debug constant `0.0656 ms`
- `4096`: separate `0.0634 ms`, public A-only fused `0.3488 ms`, shared-A debug constant `0.4611 ms`

So the shared-A transport bug is finally fixed, but the backend still does **not** improve the
current public A-only fused path:

- it is slower than the existing public A-only fused kernel at both `2048` and `4096`
- CTA-amax is still not stable at `2048`
- that makes the result more decisive, not less: even after transport is correct, this particular
  cross-CTA shared-A backend does not overturn the broader conclusion that current tile-local
  streaming designs are dominated by repeated quantization tax

One last scheduling-only sweep was still worth trying after the transport fix:

- disabling `USE_PDL` did not help; shared-A constant stayed at about `0.0657 ms` at `2048` and
  about `0.4613 ms` at `4096`
- increasing `SUPERGROUP_SIZE` from `4` to `8` and then `16` on the original PDL-enabled path
  shaved a little constant-path time off the debug backend:
  - `2048`: shared-A constant improved from about `0.0656 ms` to about `0.0642 ms`
  - `4096`: shared-A constant improved from about `0.4611 ms` to about `0.4598 ms`
- the public A-only fused kernel still remains clearly faster:
  - `2048`: public A-only fused about `0.0485 ms`
  - `4096`: public A-only fused about `0.3466 ms`
- shared-A CTA-amax still fails at `2048` with `cudaErrorLaunchFailure`

Interpretation:

- there is a little row-group scheduling overhead to trim in the shared-A path
- but the size of the win is only a few percent and does not materially change the ranking
- that is strong evidence that the remaining gap is not a “hidden pipeline knob” problem
- even after fixing transport and nudging scheduling, this shared-A backend still loses to the
  simpler public A-only fused kernel, which is itself still behind separate `quantize + gemm`

## 2026-03-29: Bottom Line on Streaming FP4 Into GEMM

At this point the engineering conclusion is fairly clear for **pure dense square GEMMs** on the
current B200 path:

- streaming FP4 quantization directly into the GEMM hot loop is **feasible**
- it is sometimes **usefully faster than a worse fused design**
- but it is **not** competitive with separate `quantize + gemm` once shapes are large and reuse is
  high

Fresh end-to-end confirmation on the current tree:

`2048 x 2048 x 2048`

- separate `quantize + gemm`: `0.030 ms`
- A-only fused (`A streamed / B pre-quantized`): `0.049 ms`
- mirror hybrid (`A pre-quantized / B streamed`): `0.052 ms`
- both-bf16 shared-B (`A streamed / B streamed`): `0.070 ms`

`4096 x 4096 x 4096`

- separate `quantize + gemm`: `0.061 ms`
- A-only fused (`A streamed / B pre-quantized`): `0.348 ms`
- mirror hybrid (`A pre-quantized / B streamed`): `0.353 ms`
- both-bf16 shared-B (`A streamed / B streamed`): `0.547 ms`

That gives a pretty consistent ranking:

1. separate `quantize + gemm`
2. one-sided streaming
3. two-sided streaming

The profile evidence explains why. The promoted shared-B path was:

- not bandwidth-bound
- not tensor-core-saturated
- mostly **latency / eligibility limited**
- running with very low eligible-warp count and high dependency waiting

So deeper or cleaner pipelining can shave constants, but it cannot remove the core tax:

- separate `quantize + gemm` quantizes each operand once and reuses the result globally
- streaming fused kernels quantize in the hot loop and only reuse the result over a small local
  output block
- as `M` and `N` grow, that repeated-quantization tax dominates

The one-sided controls are especially informative:

- A-only fused and the mirror hybrid land in nearly the same performance band
- that strongly suggests the problem is **not mainly which side is streamed**
- the problem is the **block-local lifetime** of streamed quantized tiles on current hardware

What this does **not** mean:

- it does **not** mean “fusion is always bad”
- it does **not** mean “streaming quantization can never win”

What it *does* mean for this repo and hardware target is:

- for standalone GEMM, current tile-local FP4 streaming is a poor economic trade versus
  pre-quantization
- fusion may still make sense when it removes additional memory traffic or kernels beyond the
  quantize step itself
- examples would be cases where the quantized tile is immediately consumed by more work than just
  GEMM, or where fusion also absorbs a downstream reduction / normalization / pointwise stage

That last point is still an inference, not something we proved here. We did **not** benchmark a
full “quantize + GEMM + softmax” style fused pipeline in this work. So the validated statement is
more specific:

- on this hardware, for the GEMM-only workloads tested here, we could not pipeline our way around
  the repeated-quantization tax
- wider-lifetime reuse of quantized strips would be needed to change that conclusion materially
