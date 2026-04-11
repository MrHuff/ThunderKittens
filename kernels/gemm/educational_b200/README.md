# ThunderKittens Educational GEMM Kernels (Blackwell)

This folder builds up the B200 GEMM piece-by-piece. It is only for educational purposes.

Change the `LEVEL` field in the `Makefile` to `01` - `10`, then `make clean && make run`.

- Level 01 (6 TFLOPs): Simple for loop (float) -- this is faster than bf16 because bf16 gets implicitly converted to floats first on cuda cores
- Level 02 (6 TFLOPs): Simple for loop (bf16)
- Level 03 (11 TFLOPs): Use shared memory
- Level 04 (26 TFLOPs): Use tensor cores (WMMA)
- Level 05 (55 TFLOPs): Use TMA for global<->shared memory transfers (+ WMMA)
- Level 06 (293 TFLOPs): Use tensor cores (tcgen05 MMA) with TMA
- Level 07 (731 TFLOPs): Use pipelined warp specialization (TMA loader + MMA issuer)
- Level 08 (1050 TFLOPs): Use epilogue pipelining
- Level 09 (1285 TFLOPs): Use 2-CTA cluster and warpgroup-level parallelism
- Level 10 (1346 TFLOPs): Use persistent kernel with CLC (Cluster Launch Control) + supergroup swizzling
- Level 11: Parameter tuning — same architecture as Level 10 with tunable knobs (tile size, pipeline depth, consumers, etc.)

## Level 11: Parameter Tuning

Build with different configs using `-DCONFIG=N`:
```bash
nvcc level_11.cu [flags] -DCONFIG=0 -o level_11_c0.out  # see Makefile for full flags
```

| Config | TILE_N | PIPE | CONSUMERS | EPI | SMEM   | TFLOPS | Key Change                   |
|--------|--------|------|-----------|-----|--------|--------|------------------------------|
| 0      | 256    | 4    | 2         | 8   | 230KB  | 1298   | Baseline (Level 10 default)  |
| 1      | 128    | 4    | 1         | 4   | 116KB  | 1127   | Narrow tile + 1 consumer     |
| 5      | 128    | 4    | 2         | 4   | 198KB  | 1187   | Narrow tile + 2 consumers    |
| 7      | 256    | 2    | 2         | 8   | 132KB  | 1132   | Shallow 2-stage pipeline     |
| 8      | 256    | 4    | 2         | 8   | 230KB  | 1278   | Wider supergroup (SG=8)      |

## From Scratch

`from_scratch.cu` — empty skeleton with TODO markers for implementing a GEMM from zero. See `WALKTHROUGH.md` for reference.

Note: Our full kernel in ``../bf16_b200`` gives 1540 TFLOPs for the default GEMM size in these examples ($M=N=K=4096$).
