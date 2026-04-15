## Backward Sweep

Command:

```bash
CUDA_VISIBLE_DEVICES=2 python3 /workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py \
  --backward-sweep --shape-set canonical6 --warmup 1 --iters 3
```

Scope:

- NVFP4:
  - `v2-native`, `v2-enc`, `v2-dec`
  - `v3-native`, `v3-enc`, `v3-dec`
  - `v6-3wg`, `v6-5wg`
- MXFP4:
  - `v2-rte`, `v2-enc`, `v2-dec`
  - `v3-rte`, `v3-enc`, `v3-dec`

Important caveat:

- `v6` rows are timing-only in this sweep.
- The current public `v6` ABI does not expose the localCTA chunk-grid (`sg_chunks`) metadata needed for trustworthy full-backward cosine checks with localCTA tails.
- The current public `v6` API only exposes the legacy scalar `G_sg_row`, which is enough for the existing bridged NV GEMM ABI, but not enough to drive numerically correct localCTA `dC` in the same way as the standalone localCTA quantizer.

## Update 2026-04-14: MX family flag + memory reporting

The benchmark harness now supports:

```bash
CUDA_VISIBLE_DEVICES=2 python3 /workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py \
  --backward-sweep --shape-set small2 --fp4-family mx --report-memory --warmup 0 --iters 1
```

and the same `--fp4-family` / `--report-memory` controls for NV.

Important contract correction:

- MX `v3` is benchmarked through its public FP4 row+col backward API.
- MX `v2` is **not** benchmarked through an equivalent public FP4 row+col API because that API does not exist today.
- The public MX `v2` FP4 wrappers only expose row outputs, so full backward remains a hybrid benchmark surface:
  - BF16 backward via `backward_v2_bf16_*`
  - mode-matched `mxfp4_quantize_for_gemm(...)` on `G` and `G^T`
  - MX GEMM tails for `dE` and `dC`

Memory methodology:

- `PeakAlloc(MB)` and `PeakResv(MB)` are per-variant CUDA deltas over the shared cached-input baseline.
- `Contract(MB)` is the unique-storage size of variant-owned tensors only; shared cached inputs and view aliases are excluded.

Validated memory reruns:

- MX small2:
  - `256x256->512`
  - `512x256->512`
- MX large spot:
  - `4Kx4K->32K`
- NV small2 sanity check:
  - `256x256->512`
  - `512x256->512`

Representative MX results with memory columns:

| Shape | Variant | Tail | Time (ms) | cos(dE) | cos(dC) | PeakAlloc (MB) | Contract (MB) |
|---|---|---|---:|---:|---:|---:|---:|
| `256x256->512` | `v2-enc` | `mx_hybrid` | 0.520 | 0.9933 | 0.9932 | 1.01 | 0.76 |
| `256x256->512` | `v3-enc` | `mx_gemm` | 0.505 | 0.9933 | 0.9932 | 0.76 | 0.51 |
| `512x256->512` | `v2-enc` | `mx_hybrid` | 0.403 | 0.9934 | 0.9932 | 1.77 | 1.27 |
| `512x256->512` | `v3-enc` | `mx_gemm` | 0.225 | 0.9934 | 0.9932 | 1.27 | 0.77 |
| `4Kx4K->32K` | `v2-enc` | `mx_hybrid` | 1.533 | 0.9934 | 0.9934 | 914.81 | 664.81 |
| `4Kx4K->32K` | `v3-enc` | `mx_gemm` | 3.610 | 0.9934 | 0.9934 | 665.38 | 414.81 |

Current interpretation:

- the family selector and memory reporting are now valid on both NV and MX surfaces
- MX `v2` and MX `v3` are no longer being compared under a false “same public FP4 ABI” assumption
- MX `v2` currently trades more memory and extra quantization work for its hybrid full-backward path

## Update 2026-04-14: Larger shapes + Triton BF16 baseline

The harness now supports:

```bash
CUDA_VISIBLE_DEVICES=2 python3 /workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py \
  --backward-sweep --shape-set xlarge4 --fp4-family nv --include-triton-bf16
```

with `xlarge4`:

- `4Kx7K->256K`
- `16Kx4K->32K`
- `8Kx8K->128K`
- `16Kx8K->128K`

and an optional Triton BF16 row driven by `cut_cross_entropy.linear_cross_entropy(..., impl="cce")`.

Key results gathered on `CUDA_VISIBLE_DEVICES=2`:

| Shape | `v2` | `v3` | `Triton BF16 CCE` | Notes |
|---|---:|---:|---:|---|
| `4Kx7K->256K` | `12.298 ms` (`dec 11.123`) | public row+col path does not return promptly | `1134.887 ms` | the earlier `+122.07 GiB` OOM was a wrapper placeholder bug and has since been fixed |
| `16Kx4K->32K` | `3.838 ms` (`dec 3.823`) | `v3-enc 5.992 ms`, `v3-dec 6.023 ms` | `988.593 ms` | `v3-native 236.276 ms` is an outlier and should not be treated as the stable `v3` surface |
| `8Kx8K->128K` | `v2-enc 13.211 ms` | `v3-enc` one-shot did not return promptly (>100s) | `1842.606 ms` | direct spot check, no memory pass |
| `16Kx8K->128K` | `v2-enc 60.695 ms` | not promoted; `8Kx8K->128K` already showed `v3` not returning promptly | `1319.789 ms` | direct spot check, no memory pass |

Memory observations from the completed `--report-memory` xlarge runs:

- `4Kx7K->256K`
  - `v2-native PeakAlloc=6681.00 MB`, `Contract=5556.00 MB`
  - `v6-5wg PeakAlloc=4681.92 MB`, `Contract=4681.92 MB`
  - `triton-cce PeakAlloc=7112.92 MB`, `Contract=7112.00 MB`
- `16Kx4K->32K`
  - `v2-native PeakAlloc=1940.50 MB`, `Contract=1378.00 MB`
  - `v3-enc PeakAlloc=2893.63 MB`, `Contract=940.50 MB`
  - `triton-cce PeakAlloc=756.28 MB`, `Contract=756.00 MB`

Interpretation:

- `v2` continues to scale cleanly into the larger-shape regime.
- Triton BF16 CCE remains orders of magnitude slower on backward wall time than `v2` on these shapes, even before considering FP4 tail advantages.
- `v3` is no longer just “a bit behind” at large shapes:
  - on `16Kx4K->32K`, stable `enc/dec` are still materially slower than `v2`
- on vocab-heavy or square frontier shapes, the public row+col path still fails to return promptly even after the dead-placeholder OOM fix
- the large-shape problem is now clearly a scalability issue in `v3` materialization, not a small constant-factor gap

## Update 2026-04-14: Clean isolated compare mode

The harness now supports a fresh-subprocess compare mode:

```bash
CUDA_VISIBLE_DEVICES=2 python3 /workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py \
  --isolated-compare --shape-set xlarge4 --config-label '16Kx4K->32K' \
  --warmup 0 --iters 1 --isolated-timeout-s 90
```

This mode runs each row from the same raw BF16 inputs in a fresh process and reports:

- `Time (ms)`
- `Peak>Raw(MB)` = peak CUDA alloc above the raw BF16 input baseline
- `Peak>Quant(MB)` = peak CUDA alloc above the pre-quantized-input baseline

Why this matters:

- the older in-process sweep can be distorted by shared cached tensors and prior allocator state
- `v3` in particular was giving conflicting large-shape signals between the in-process sweep and manual spot checks

Clean isolated results on `CUDA_VISIBLE_DEVICES=2`:

| Shape | Variant | Time (ms) | Peak>Raw (MB) | Peak>Quant (MB) | Status |
|---|---|---:|---:|---:|---|
| `16Kx4K->32K` | `v2-enc` | 13.036 | 2153.19 | 1940.50 | `OK` |
| `16Kx4K->32K` | `v3-enc` | — | — | — | `TIMEOUT` |
| `16Kx4K->32K` | `triton-cce` | 196.182 | 756.28 | 378.09 | `OK` |
| `4Kx7K->256K` | `v2-enc` | 39.444 | 8684.52 | 6681.00 | `OK` |
| `4Kx7K->256K` | `v3-enc` | — | — | — | `TIMEOUT` |
| `4Kx7K->256K` | `triton-cce` | 659.113 | 7112.92 | 3556.87 | `OK` |

Current interpretation:

- the clean isolated surface is harsher on `v3` than the older in-process sweep
- `v2` still returns quickly under isolation
- `v3` currently does not have a reliable clean large-shape comparison surface:
  - timeout on `16Kx4K->32K`
  - timeout on `4Kx7K->256K`
- for large-shape decision-making, the isolated numbers should be treated as the more trustworthy surface

## Update 2026-04-14: `v3` xlarge split after the OOM fix

The `+122.07 GiB` request on `4Kx7K->256K` was caused by an unused combo placeholder in `nvfp4_cce_backward_v3.cu` that still allocated `max(M, N) x max(M, N)` BF16 for public `3wg` launches. That dead placeholder is now reduced to a legal `128x32` BF16 tile, matching the minimal col-only path contract.

After rebuilding `v3`:

- `4Kx7K->256K`
  - `rowonly`: returns cleanly
  - `colonly`: returns cleanly
  - public full row+col: still hangs / times out
- `16Kx4K->32K`
  - `rowonly`: returns cleanly
  - `colonly`: fails with `CUDA error: unspecified launch failure`
  - public full row+col: still hangs / times out

So the large-shape `v3` picture is now:

- the absurd OOM was real but fixed
- the remaining failures are runtime path bugs
- they are not the same bug on every shape:
  - large-`M` col-only failure on `16Kx4K->32K`
  - large-`V` row+col interaction failure on `4Kx7K->256K`

## Update 2026-04-14: `v3-enc` fix

The original 2026-04-13 sweep in this file predates the public `v3-enc` contract fix.
Those historical `v3-enc` rows should no longer be treated as current.

Root cause:

- public NVFP4 `enc` and `dec` do not expose different raw GEMM-facing scale-byte tensors in the standalone quantizer
- `v3-enc` had drifted into a different raw-output contract
- the fix was to route the public `v3-enc` wrapper back onto the same raw-output contract used by the standalone NV quantizer and to centralize the internal scale-contract helper usage in `nvfp4_cce_backward_v3.cuh`

Validation reruns after the fix:

```bash
CUDA_VISIBLE_DEVICES=2 python3 /workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py \
  --backward-sweep --shape-set small2 --warmup 0 --iters 1

CUDA_VISIBLE_DEVICES=2 python3 /workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py \
  --backward-sweep --shape-set large4 --warmup 0 --iters 1
```

Post-fix NVFP4 `v3-enc` spot check:

| Shape | Time (ms) | cos(dE) | cos(dC) |
|---|---:|---:|---:|
| `256x256->512` | 0.195 | 0.9930 | 0.9933 |
| `512x256->512` | 0.291 | 0.9932 | 0.9935 |
| `4Kx4K->32K` | 1.588 | 0.9940 | 0.9939 |
| `4Kx8K->32K` | 1.990 | 0.9940 | 0.9940 |
| `8Kx4K->32K` | 3.012 | 0.9941 | 0.9939 |
| `4Kx4K->128K` | 6.048 | 0.9941 | 0.9940 |

Current status:

- `v3-enc` is now numerically healthy on the NVFP4 sweep shapes that were rerun
- `v3-enc`, `v3-native`, and `v3-dec` are now functionally in the same regime on public NVFP4 backward
- the raw output block below remains as the original 2026-04-13 snapshot and is retained only as historical context

## Key Findings

- `v2-native` and `v2-enc` are identical in the current shipped full-backward contract. Both are BF16 backward + standalone NV quantizer + NV GEMM tails.
- `v3-native` tracks `v2-native` numerically on all sweep shapes.
- `v3-dec` is healthy and slightly stronger numerically than `v3-native`.
- the original 2026-04-13 `v3-enc` rows in this file are stale; see the 2026-04-14 update above
- after the contract fix, `v3-enc` is numerically healthy on the rerun sweep shapes
- MXFP4 `enc` / `dec` sweeps are healthy.
- `v6` timing with localCTA tails is in the right rough range, but the missing chunk-grid ABI means these rows should not be treated as validated numerical results yet.

## Raw Output

```text
========================================================================================================================
  BACKWARD SWEEP MODE
  NV: v2 current/native + explicit encode/decode, v3 current/native + explicit encode/decode, v6 localCTA full
  MX: v2/v3 across rte/encode/decode
  Note: v6 rows are timing-only today; the public v6 ABI does not expose localCTA chunk-grid metadata for trustworthy full-backward cosine checks.
  Shapes: canonical6
========================================================================================================================

========================================================================================================================
256x256->512  [M=256, K=256, V=512]
========================================================================================================================
NVFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-native          | nv_gemm    |      0.143 |     0.9930 |     0.9933 | OK
v2-enc             | nv_gemm    |      0.143 |     0.9930 |     0.9933 | OK
v2-dec             | nv_gemm    |      0.109 |     0.9955 |     0.9955 | OK
v3-native          | nv_gemm    |      0.117 |     0.9930 |     0.9933 | OK
v3-enc             | nv_gemm    |      0.096 |     0.0379 |     0.0435 | OK
v3-dec             | nv_gemm    |      0.098 |     0.9955 |     0.9955 | OK
v6-3wg             | localcta   |      0.087 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)
v6-5wg             | localcta   |      0.087 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)

MXFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-rte             | mx_gemm    |      0.143 |     0.9912 |     0.9912 | OK
v2-enc             | mx_gemm    |      0.140 |     0.9933 |     0.9932 | OK
v2-dec             | mx_gemm    |      0.130 |     0.9323 |     0.9336 | OK
v3-rte             | mx_gemm    |      0.140 |     0.9912 |     0.9912 | OK
v3-enc             | mx_gemm    |      0.134 |     0.9933 |     0.9932 | OK
v3-dec             | mx_gemm    |      0.132 |     0.9820 |     0.9826 | OK

========================================================================================================================
512x256->512  [M=512, K=256, V=512]
========================================================================================================================
NVFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-native          | nv_gemm    |      0.216 |     0.9932 |     0.9935 | OK
v2-enc             | nv_gemm    |      0.216 |     0.9932 |     0.9935 | OK
v2-dec             | nv_gemm    |      0.196 |     0.9954 |     0.9954 | OK
v3-native          | nv_gemm    |      0.174 |     0.9932 |     0.9935 | OK
v3-enc             | nv_gemm    |      0.177 |     0.0476 |     0.0439 | OK
v3-dec             | nv_gemm    |      0.162 |     0.9954 |     0.9954 | OK
v6-3wg             | localcta   |      0.154 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)
v6-5wg             | localcta   |      0.139 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)

MXFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-rte             | mx_gemm    |      0.210 |     0.9911 |     0.9911 | OK
v2-enc             | mx_gemm    |      0.213 |     0.9934 |     0.9932 | OK
v2-dec             | mx_gemm    |      0.199 |     0.9343 |     0.9357 | OK
v3-rte             | mx_gemm    |      0.156 |     0.9911 |     0.9911 | OK
v3-enc             | mx_gemm    |      0.151 |     0.9934 |     0.9932 | OK
v3-dec             | mx_gemm    |      0.151 |     0.9823 |     0.9823 | OK

========================================================================================================================
4Kx4K->32K  [M=4096, K=4096, V=32000]
========================================================================================================================
NVFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-native          | nv_gemm    |      1.082 |     0.9940 |     0.9939 | OK
v2-enc             | nv_gemm    |      1.082 |     0.9940 |     0.9939 | OK
v2-dec             | nv_gemm    |      1.067 |     0.9955 |     0.9955 | OK
v3-native          | nv_gemm    |      1.598 |     0.9940 |     0.9939 | OK
v3-enc             | nv_gemm    |      1.614 |     0.0055 |     0.0054 | OK
v3-dec             | nv_gemm    |      1.616 |     0.9955 |     0.9955 | OK
v6-3wg             | localcta   |      2.919 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)
v6-5wg             | localcta   |      2.880 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)

MXFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-rte             | mx_gemm    |      1.533 |     0.9912 |     0.9912 | OK
v2-enc             | mx_gemm    |      1.518 |     0.9934 |     0.9934 | OK
v2-dec             | mx_gemm    |      1.521 |     0.9833 |     0.9833 | OK
v3-rte             | mx_gemm    |      3.314 |     0.9912 |     0.9912 | OK
v3-enc             | mx_gemm    |      3.304 |     0.9934 |     0.9934 | OK
v3-dec             | mx_gemm    |      3.212 |     0.9833 |     0.9833 | OK

========================================================================================================================
4Kx8K->32K  [M=4096, K=8192, V=32000]
========================================================================================================================
NVFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-native          | nv_gemm    |      1.594 |     0.9940 |     0.9940 | OK
v2-enc             | nv_gemm    |      1.594 |     0.9940 |     0.9940 | OK
v2-dec             | nv_gemm    |      1.587 |     0.9955 |     0.9955 | OK
v3-native          | nv_gemm    |      2.004 |     0.9940 |     0.9940 | OK
v3-enc             | nv_gemm    |      2.000 |     0.0055 |     0.0054 | OK
v3-dec             | nv_gemm    |      2.002 |     0.9955 |     0.9955 | OK
v6-3wg             | localcta   |      4.939 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)
v6-5wg             | localcta   |      4.940 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)

MXFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-rte             | mx_gemm    |      2.046 |     0.9912 |     0.9912 | OK
v2-enc             | mx_gemm    |      2.041 |     0.9934 |     0.9934 | OK
v2-dec             | mx_gemm    |      2.043 |     0.9833 |     0.9833 | OK
v3-rte             | mx_gemm    |      3.679 |     0.9912 |     0.9912 | OK
v3-enc             | mx_gemm    |      3.656 |     0.9934 |     0.9934 | OK
v3-dec             | mx_gemm    |      3.589 |     0.9833 |     0.9833 | OK

========================================================================================================================
8Kx4K->32K  [M=8192, K=4096, V=32000]
========================================================================================================================
NVFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-native          | nv_gemm    |      1.961 |     0.9941 |     0.9939 | OK
v2-enc             | nv_gemm    |      1.961 |     0.9941 |     0.9939 | OK
v2-dec             | nv_gemm    |      1.946 |     0.9955 |     0.9955 | OK
v3-native          | nv_gemm    |      3.028 |     0.9941 |     0.9939 | OK
v3-enc             | nv_gemm    |      3.016 |     0.0055 |     0.0054 | OK
v3-dec             | nv_gemm    |      3.026 |     0.9955 |     0.9955 | OK
v6-3wg             | localcta   |      5.289 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)
v6-5wg             | localcta   |      5.285 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)

MXFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-rte             | mx_gemm    |      2.902 |     0.9912 |     0.9912 | OK
v2-enc             | mx_gemm    |      2.903 |     0.9934 |     0.9934 | OK
v2-dec             | mx_gemm    |      2.912 |     0.9833 |     0.9833 | OK
v3-rte             | mx_gemm    |      6.452 |     0.9912 |     0.9912 | OK
v3-enc             | mx_gemm    |      6.412 |     0.9934 |     0.9934 | OK
v3-dec             | mx_gemm    |      6.233 |     0.9833 |     0.9833 | OK

========================================================================================================================
4Kx4K->128K  [M=4096, K=4096, V=128000]
========================================================================================================================
NVFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-native          | nv_gemm    |      3.922 |     0.9941 |     0.9941 | OK
v2-enc             | nv_gemm    |      3.922 |     0.9941 |     0.9941 | OK
v2-dec             | nv_gemm    |      3.915 |     0.9953 |     0.9955 | OK
v3-native          | nv_gemm    |      6.050 |     0.9941 |     0.9941 | OK
v3-enc             | nv_gemm    |      6.061 |     0.0029 |     0.0027 | OK
v3-dec             | nv_gemm    |      6.053 |     0.9953 |     0.9955 | OK
v6-3wg             | localcta   |     11.005 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)
v6-5wg             | localcta   |     11.006 |          - |          - | TIMING_ONLY(localCTA sg_chunks missing from public v6 ABI)

MXFP4
Variant            | Tail       |  Time (ms) |    cos(dE) |    cos(dC) | Status
─────────────────────────────────────────────────────────────────────────────────────────────
v2-rte             | mx_gemm    |      6.232 |     0.9912 |     0.9912 | OK
v2-enc             | mx_gemm    |      6.200 |     0.9934 |     0.9934 | OK
v2-dec             | mx_gemm    |      6.212 |     0.9833 |     0.9833 | OK
v3-rte             | mx_gemm    |     12.819 |     0.9912 |     0.9912 | OK
v3-enc             | mx_gemm    |     12.755 |     0.9934 |     0.9934 | OK
v3-dec             | mx_gemm    |     12.400 |     0.9833 |     0.9833 | OK
```

## Large NVFP4 Re-Run After `v3` Wrapper Memory Cleanup

Command:

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u fp4_cce_TK/bench_v2_vs_v3.py
```

Method:
- large-shape NVFP4-only rerun
- `warmup=2`, `iters=5`
- `v2` rows use the shipped BF16 backward + quantize + NV GEMM tail contract
- `v3` rows use the rebuilt public `rowhwfp4` path after shrinking the unused BF16 `D_out` placeholder to a minimal TMA-valid tile

### `4Kx4K->32K`

| Variant | Time (ms) | cos(dE) | cos(dC) |
|---|---:|---:|---:|
| `v2-native` | 1.035 | 0.9940 | 0.9939 |
| `v2-enc` | 1.032 | 0.9940 | 0.9939 |
| `v2-dec` | 1.039 | 0.9955 | 0.9955 |
| `v3-native` | 1.596 | 0.9940 | 0.9939 |
| `v3-enc` | 1.593 | 0.0055 | 0.0054 |
| `v3-dec` | 1.597 | 0.9955 | 0.9955 |

### `4Kx8K->32K`

| Variant | Time (ms) | cos(dE) | cos(dC) |
|---|---:|---:|---:|
| `v2-native` | 1.592 | 0.9940 | 0.9940 |
| `v2-enc` | 1.581 | 0.9940 | 0.9940 |
| `v2-dec` | 1.579 | 0.9955 | 0.9955 |
| `v3-native` | 1.987 | 0.9940 | 0.9940 |
| `v3-enc` | 1.986 | 0.0053 | 0.0054 |
| `v3-dec` | 1.987 | 0.9955 | 0.9955 |

### `8Kx4K->32K`

| Variant | Time (ms) | cos(dE) | cos(dC) |
|---|---:|---:|---:|
| `v2-native` | 1.951 | 0.9941 | 0.9939 |
| `v2-enc` | 1.945 | 0.9941 | 0.9939 |
| `v2-dec` | 1.935 | 0.9955 | 0.9955 |
| `v3-native` | 2.998 | 0.9941 | 0.9939 |
| `v3-enc` | 2.993 | 0.0055 | 0.0055 |
| `v3-dec` | 3.002 | 0.9955 | 0.9955 |

### `4Kx4K->128K`

| Variant | Time (ms) | cos(dE) | cos(dC) |
|---|---:|---:|---:|
| `v2-native` | 3.904 | 0.9941 | 0.9940 |
| `v2-enc` | 3.904 | 0.9941 | 0.9940 |
| `v2-dec` | 3.887 | 0.9955 | 0.9955 |
| `v3-native` | 6.029 | 0.9941 | 0.9940 |
| `v3-enc` | 6.031 | 0.0026 | 0.0027 |
| `v3-dec` | 6.034 | 0.9955 | 0.9955 |

Takeaways:
- `v2` remains the fastest NVFP4 end-to-end backward contract across the full large set.
- `v3-native` is still consistently slower by roughly `1.25x` to `1.54x`.
- `v3-dec` remains numerically healthy and essentially tied with `v3-native`.
- the large-shape `v3-enc` rows in the original 2026-04-13 table are superseded by the 2026-04-14 rerun above.

## 2026-04-15 update: public `v3` large-shape path fixed

The previous large-shape `TIMEOUT` and `OOM` conclusions for public `v3` are now superseded.

After the dead-placeholder fix, the remaining large-shape failures were traced to the public
`rowhwfp4_row16ready_overlap` synchronization IDs. Promoting the experimental
`consumersync3_quantizersync2` variant into the public dispatch in
`nvfp4_cce_backward_v3.cu` makes the public `v3` path return cleanly on the full isolated
`xlarge4` set.

Reference normal-shape isolated compare on `CUDA_VISIBLE_DEVICES=2`, `warmup=2`, `iters=5`:

| Config | Variant | Time (ms) | Peak>Raw(MB) | Peak>Quant(MB) | Status |
|---|---|---:|---:|---:|---|
| `4Kx4K->32K` | `v2-enc` | 1.043 | 831.27 | 672.63 | `OK` |
| `4Kx4K->32K` | `v3-enc` | 1.547 | 581.27 | 422.63 | `OK` |
| `4Kx4K->32K` | `triton-cce` | 47.184 | 564.12 | 282.07 | `OK` |

Clean isolated `xlarge4` compare on `CUDA_VISIBLE_DEVICES=2`, `warmup=0`, `iters=1`:

| Config | Variant | Time (ms) | Peak>Raw(MB) | Peak>Quant(MB) | Status |
|---|---|---:|---:|---:|---|
| `4Kx7K->256K` | `v2-enc` | 39.494 | 8684.52 | 6681.00 | `OK` |
| `4Kx7K->256K` | `v3-enc` | 285.110 | 6684.52 | 4681.01 | `OK` |
| `4Kx7K->256K` | `triton-cce` | 654.781 | 7112.92 | 3556.87 | `OK` |
| `16Kx4K->32K` | `v2-enc` | 13.030 | 2153.19 | 1940.50 | `OK` |
| `16Kx4K->32K` | `v3-enc` | 248.314 | 1153.20 | 940.51 | `OK` |
| `16Kx4K->32K` | `triton-cce` | 191.815 | 756.28 | 378.09 | `OK` |
| `8Kx8K->128K` | `v2-enc` | 45.332 | 6450.03 | 5253.00 | `OK` |
| `8Kx8K->128K` | `v3-enc` | 286.359 | 4450.04 | 3253.01 | `OK` |
| `8Kx8K->128K` | `triton-cce` | 748.733 | 4256.61 | 2128.52 | `OK` |
| `16Kx8K->128K` | `v2-enc` | 85.747 | 9775.06 | 8506.00 | `OK` |
| `16Kx8K->128K` | `v3-enc` | 335.787 | 5775.07 | 4506.01 | `OK` |
| `16Kx8K->128K` | `triton-cce` | 1485.488 | 4512.74 | 2256.55 | `OK` |

Current interpretation:

- public `v3` is now operational on the large isolated surface
- `v3` really does use less memory than `v2` on these shapes
- `v3` is still much slower than `v2`, and on `16Kx4K->32K` it is slower than Triton BF16 CCE
- the remaining `v3` problem is throughput, not the earlier catastrophic placeholder allocation
