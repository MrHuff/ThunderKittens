# NVFP4 `v5` Status Notes

## What `v5` Does

`v5` tries to avoid materializing `G` to HBM.

Instead of:

1. recompute logits
2. form `G`
3. write quantized `G`
4. run later GEMMs for `dE` and `dC`

it does two fused owner-flipped passes:

- `dE` pass:
  - phase 1 recomputes logits for one `(row_block, vocab_block)`
  - consumer replays softmax/LSE, subtracts the target, applies `grad_scale`, and quantizes `G`
  - phase 3 immediately multiplies the staged `G_row` against `C_col`
  - partial `dE` stays on-chip until the vocab sweep finishes

- `dC` pass:
  - phase 1 recomputes logits for one `(row_block, vocab_block)` pair inside a vocab superblock
  - consumer forms `G`, writes col-quantized `G^T` staging, and publishes it cluster-wide
  - phase 3 multiplies the staged `G^T` against `E_col`
  - partial `dC` stays on-chip until the row sweep finishes

Important NVFP4 contract:

- quantization is CTA-local
- FP8 micro-scales are CTA-local/per-16 reductions
- there is no tensor-global `G` amax pass
- the analytic scalar (`fp4_row_sg` / `gt_row_sg`) is ABI compatibility, not the real decode information

## Why `v5` Still Sucks

The problem is mostly architectural, not semantic.

`v5` already uses:

- tensor cores for phase 1 and phase 3
- ping-pong staging
- CTA-local FP4 quantization
- CTA-local `amax` reductions

But it still loses badly because it pays the replay/handoff cost twice:

- once in fused `dE`
- again in fused `dC`

The expensive part is not raw DRAM traffic. The expensive part is keeping too much state alive on-chip while bouncing through barriers and scoreboards.

Measured behavior on current Blackwell/TK runs:

- DRAM throughput is tiny, roughly `~1%`
- achieved occupancy is roughly `~10%`
- eligible warps per scheduler are extremely low, around `~0.13`
- the dominant issue gap is L1TEX scoreboard stall, not memory bandwidth

So `v5` is mostly limited by:

- high register pressure
- large live SMEM/TMEM footprint
- low warp availability to hide on-chip latency
- repeated handoff between TMA, shared staging, TMEM, and consumer accumulation

## What Moves Around A Lot

The hot path moves the same logical information through several on-chip representations:

1. TMA loads FP4 payloads and FP8 scales for phase 1 into shared memory
2. phase 1 writes logits into TMEM accumulators
3. consumer reads logits from TMEM into registers
4. consumer computes `softmax(logits) - one_hot(target)` and scales it
5. consumer materializes BF16 slices into shared memory
6. consumer requantizes those BF16 slices into FP4 payload + FP8 micro-scales
7. producer loads the phase 3 B operand into shared memory
8. producer loads phase 3 scales into TMEM scale slots
9. phase 3 writes output into TMEM
10. consumer reads phase 3 output from TMEM and accumulates/stores BF16

That is the core `v5` tax: even without writing `G` to HBM, we still move `G` through enough on-chip layouts that the handoff machinery becomes the bottleneck.

## What Helped

- lowering `LOAD_PIPE_DEPTH` from `4` to `2`
- fixing `dC` reuse so one staged `G^T` is reused across two adjacent `k` blocks
- reducing some live epilogue state on `dE`

These helped because they reduced live state and stage pressure.

## What Did Not Translate Into A Real Win

- widening reuse too aggressively
  - mathematically valid
  - but it widens live output state and pushes register pressure back up

- splitting `dC` into a 3-warpgroup "producer / math / epilogue" pipeline
  - it was possible to make the sandbox correct
  - but not to turn it into a compelling standalone kernel
  - the extra handoff / scoreboard / codegen cost ate the intended overlap win

- "hybrid" `v5`: keep phase 1 in FP4, but stage BF16 `G` / `G^T` on chip and run phase 3 in BF16
  - this made the kernels exact across the main validation shapes:
    - `dC`: `256x256x512`, `4096x256x512`, `256x256x8192`
    - `dE`: `256x256x512`, `4096x256x512`
    - combo: `4096x4096x32000`
  - but it did not improve the economics
  - on `4Kx4K->32K`, measured direct timings were:
    - hybrid `dC`: `53.423 ms` vs public `dC`: `38.530 ms`
    - hybrid `dE`: `68.419 ms` vs public `dE`: `60.604 ms`
    - hybrid combo: `217.274 ms` vs public combo: `41.038 ms`
    - shipped experimental combo (`exp dE + public dC`): `35.683 ms`
  - so removing on-the-fly FP4 requantization was not enough
  - the BF16 intermediate removed control-path ugliness, but the larger live BF16 scratch/state made the fused path even less attractive

- trying to rescue the old toxic `half_tt` `dC` output contract

- treating this as a DRAM bandwidth problem
  - the profiler data does not support that

## What Finally Worked

- a Triton-style one-pass exact combo kernel
  - keep only the logits GEMM in FP4
  - take dense BF16 `E` and `C` in natural row-major layout
  - compute the softmax-gradient tile once
  - immediately update both `dE` and `dC` from that same tile
  - do not stage `G` / `G^T`
  - do not split into separate replay-style `dE` / `dC` kernels
  - do not use the owner-flipped phase-3 pipeline

- this path is exact on the main validation set:
  - `256x256x512`
  - `4096x256x512`
  - `256x256x8192`
  - `4096x4096x32000`
  - tiny `compute-sanitizer --tool memcheck` is clean

- on `4Kx4K->32K`, current direct timings are:
  - Triton-style exact combo: `20.48 ms`
  - Triton `cce_exact` backward: `23.07 ms`
  - Triton `cce` backward: `8.74 ms`
  - public `nv-v5-combo`: `41.09 ms`
  - shipped experimental combo (`exp dE + public dC`): `35.70 ms`

- so the important new fact is:
  - `v5` does not have to lose
  - but it only starts to look viable when it copies the Triton backward schedule instead of the replay / owner-flipped schedule
  - the win comes from computing the gradient tile once and consuming it twice, not from making the old split pipeline ever more elaborate

## Current Practical Conclusion

- the replay-style `v5` family is still a dead end
  - 3-warpgroup `dC` did not become a good kernel
  - the BF16-intermediate hybrid did not become a good kernel
  - owner-flipped split passes still pay too much replay / handoff / live-state tax

- the meaningful `v5` direction is now different:
  - one-pass combo only
  - Triton-style schedule
  - FP4 only where it buys something directly, i.e. the logits GEMM
  - immediate double-consumption of the BF16 gradient tile

- that means the practical comparison is no longer just `v5` vs `v3`
  - `v3` is still the right answer if we want a materialized-intermediate design
  - but a Triton-style one-pass `v5` combo is now a legitimate non-materialized alternative

- the current result is good but not complete
  - it beats `cce_exact`
  - it is still well behind filtered Triton `cce`
  - so the remaining work, if any, should focus on this one-pass combo kernel only

## Current Best Bounded Conclusion

On current Blackwell/TK resource economics, replay-style 1-pass FP4 without materialized `G` loses for one of two reasons:

- either reuse is too small, so `G` is recomputed/requantized too often
- or reuse is widened, so enough state stays live that occupancy collapses and scoreboard stalls dominate

That is why the old split `v5` designs still lose.

But the Triton-style one-pass result shows a narrower claim is more accurate:

- materializing `G` is not strictly required to beat `cce_exact`
- what matters is computing the softmax-gradient tile once and consuming it twice with very little extra on-chip choreography

So the two serious paths are now:

- `v3` style:
  - compute `G` once
  - quantize it once
  - store quantized `G`
  - let later GEMMs run with much lighter on-chip state

- Triton-style `v5` combo:
  - recompute logits once
  - form the BF16 grad tile once
  - consume it immediately for both `dE` and `dC`
  - avoid staged `G` / `G^T` and avoid separate replay passes

So the next serious work should either:

- prioritize `v3`, especially the fused FP4 epilogue/materialization kernel
- or stay entirely on the Triton-style one-pass combo path

and in either case preserve the NVFP4 rule:

- CTA-local quantization
- CTA-local `amax`
- no tensor-global `G` amax
