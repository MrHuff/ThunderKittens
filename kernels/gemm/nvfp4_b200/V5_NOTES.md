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

- trying to rescue the old toxic `half_tt` `dC` output contract

- treating this as a DRAM bandwidth problem
  - the profiler data does not support that

## Current Best Bounded Conclusion

On current Blackwell/TK resource economics, 1-pass FP4 without materialized `G` loses for one of two reasons:

- either reuse is too small, so `G` is recomputed/requantized too often
- or reuse is widened, so enough state stays live that occupancy collapses and scoreboard stalls dominate

That is why `v3` is still better:

- compute `G` once
- quantize it once
- store quantized `G`
- let later GEMMs run with much lighter on-chip state

So the next serious work should prioritize `v3`, especially the fused FP4 epilogue/materialization kernel, while preserving the NVFP4 rule:

- CTA-local quantization
- CTA-local `amax`
- no tensor-global `G` amax
