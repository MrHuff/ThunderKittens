# localCTA v3 GEMM Contract

`localCTA_epilogue_v3` is the consumer-side companion to `nvfp4_CTA_local_v3`.

## Goal

Keep the fast NVFP4 producer/consumer structure and remove the `v1` requirement that prepared scales already contain the outer correction.

`v3` fast kernels consume:

- FP4 payloads
- FP8 microscales
- FP32 row-tile outer scales
- FP32 col-tile outer scales

with no `Q8(sc * sg)` folding.

## Core invariant

Unlike the old raw localCTA contract, `v3` outer scales are K-invariant for the GEMM tile.

First landing geometry is fixed:

- `Mb = 256`
- `Nb = 256`
- `Kb = 256`

So for one output tile:

- `row_sg_v3[row_block_idx]` is constant across the full K reduction
- `col_sg_v3[col_block_idx]` is constant across the full K reduction

That makes the consumer structure identical to regular fast NVFP4 GEMM:

1. load FP4 payloads and FP8 microscales
2. run the full K reduction in TMEM
3. load the final accumulator tile
4. multiply once by `row_sg_v3[row_block_idx] * col_sg_v3[col_block_idx]`
5. run the output epilogue / store path

## Rejected designs

- Folding `sc * sg` into FP8 prepared scales
- Rewriting staged FP8 scales in shared memory
- Carrying K-varying SG through the consumer
- Per-K consumer accumulation in registers

All of those either recreate the `v1` underflow failure mode or destroy kernel performance.

## Required end state

- no SG folding into prepared FP8 scales
- no chunk-grid SG ABI inside `v3`
- fast regular/grouped/batched/batched-accum localCTA kernels accept tile-global outer scales
- correction stays in `float`
- `v1` remains unchanged as the baseline until `v3` passes parity, training, and MFU gates
