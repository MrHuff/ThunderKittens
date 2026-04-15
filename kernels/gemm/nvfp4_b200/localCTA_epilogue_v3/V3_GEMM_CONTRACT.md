# localCTA v3 GEMM Contract

`localCTA_epilogue_v3` is the consumer-side companion to `nvfp4_CTA_local_v3`.

## Goal

Keep the fast localCTA GEMM family, but remove the `v1` requirement that prepared scales already contain the SG correction.

`v3` fast kernels should consume:

- FP4 payloads
- FP8 microscales
- FP32 correction tensors

with no `Q8(sc * sg)` folding.

## Design constraint

`v5` can multiply one global correction after accumulation because its outer scale is K-invariant for the tile.

Current localCTA is different: correction varies across the reduction loop. `v3` therefore needs a correction grid aligned to the reduction slices the fast GEMM already stages. That preserves a clean GEMM ABI without reintroducing the slow raw/direct localCTA path.

## Required end state

- no SG folding into prepared FP8 scales
- fast regular/grouped/batched localCTA kernels accept explicit correction tensors
- correction stays in `float`
- `v1` remains unchanged as the baseline until `v3` passes parity, training, and MFU gates
