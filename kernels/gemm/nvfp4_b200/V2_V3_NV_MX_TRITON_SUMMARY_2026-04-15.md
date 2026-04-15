# `v2` / `v3` backward comparison against Triton BF16

## Setup

- Date: `2026-04-15`
- Device: `CUDA_VISIBLE_DEVICES=3`
- Command shape set: `canonical6`
- Benchmark command:

```bash
python3 -u fp4_cce_TK/bench_v2_vs_v3.py \
  --backward-sweep \
  --shape-set canonical6 \
  --report-memory \
  --include-triton-bf16 \
  --warmup 2 \
  --iters 5
```

- Error metric: cosine similarity against the BF16 reference backward.
- Memory columns:
  - `PeakAlloc`: peak CUDA allocation delta over the shared input baseline
  - `Contract`: explicit unique-storage bytes owned by the variant

## Caveats

- NVFP4 `v2` and `v3` are both full public FP4-output backward paths.
- MXFP4 `v2` is still the `mx_hybrid` path:
  - BF16 backward
  - mode-matched MX quantization of `G` and `G^T`
  - MX GEMM tails
- MXFP4 `v3` is the public FP4-output `mx_gemm` path.
- Triton BF16 uses `cut_cross_entropy.linear_cross_entropy(..., impl="cce")`.

## Executive summary

- NVFP4:
  - `v2` is the throughput winner on every shape and both `enc` and `dec`.
  - `v3` is consistently lower-footprint than `v2`.
  - `dec` is slightly more accurate than `enc`.
- MXFP4:
  - `v2` is again faster than `v3`.
  - `v3` is again lower-footprint than `v2`.
  - `enc` is materially better than `dec` on accuracy.
- Triton BF16:
  - exact by construction
  - much slower than both FP4 families on all benchmarked shapes

## NVFP4 encode

Format per cell: `time ms | cos(dE) / cos(dC) | PeakAlloc MB / Contract MB`

| Shape | `v2-enc` | `v3-enc` | `triton-cce` |
|---|---|---|---|
| `256x256->512` | `0.097 | 0.9930 / 0.9933 | 0.77 / 0.62` | `0.143 | 0.9930 / 0.9933 | 0.52 / 0.52` | `0.466 | 1.0000 / 1.0000 | 0.75 / 0.75` |
| `512x256->512` | `0.194 | 0.9932 / 0.9935 | 1.28 / 1.00` | `0.166 | 0.9932 / 0.9935 | 0.79 / 0.78` | `0.525 | 1.0000 / 1.0000 | 1.01 / 1.00` |
| `4Kx4K->32K` | `1.066 | 0.9940 / 0.9939 | 673.19 / 532.00` | `1.593 | 0.9940 / 0.9939 | 422.63 / 422.63` | `14.062 | 1.0000 / 1.0000 | 564.12 / 564.00` |
| `4Kx8K->32K` | `1.567 | 0.9940 / 0.9940 | 954.69 / 814.00` | `1.975 | 0.9940 / 0.9940 | 704.63 / 704.63` | `28.276 | 1.0000 / 1.0000 | 1128.19 / 1128.00` |
| `8Kx4K->32K` | `1.926 | 0.9941 / 0.9939 | 1097.25 / 814.00` | `2.997 | 0.9941 / 0.9939 | 597.26 / 595.25` | `27.496 | 1.0000 / 1.0000 | 628.17 / 628.00` |
| `4Kx4K->128K` | `3.858 | 0.9941 / 0.9941 | 2594.50 / 2032.00` | `5.958 | 0.9941 / 0.9941 | 1594.51 / 1594.50` | `54.258 | 1.0000 / 1.0000 | 2064.30 / 2064.00` |

## NVFP4 decode

| Shape | `v2-dec` | `v3-dec` | `triton-cce` |
|---|---|---|---|
| `256x256->512` | `0.146 | 0.9955 / 0.9955 | 0.77 / 0.62` | `0.132 | 0.9955 / 0.9955 | 0.52 / 0.52` | `0.466 | 1.0000 / 1.0000 | 0.75 / 0.75` |
| `512x256->512` | `0.177 | 0.9954 / 0.9954 | 1.28 / 1.00` | `0.161 | 0.9954 / 0.9954 | 0.79 / 0.78` | `0.525 | 1.0000 / 1.0000 | 1.01 / 1.00` |
| `4Kx4K->32K` | `1.056 | 0.9955 / 0.9955 | 673.19 / 532.00` | `1.592 | 0.9955 / 0.9955 | 422.63 / 422.63` | `14.062 | 1.0000 / 1.0000 | 564.12 / 564.00` |
| `4Kx8K->32K` | `1.580 | 0.9955 / 0.9955 | 954.63 / 814.00` | `1.974 | 0.9955 / 0.9955 | 704.63 / 704.63` | `28.276 | 1.0000 / 1.0000 | 1128.19 / 1128.00` |
| `8Kx4K->32K` | `1.935 | 0.9955 / 0.9955 | 1097.25 / 814.00` | `2.998 | 0.9955 / 0.9955 | 597.26 / 595.25` | `27.496 | 1.0000 / 1.0000 | 628.17 / 628.00` |
| `4Kx4K->128K` | `3.844 | 0.9955 / 0.9955 | 2594.50 / 2032.00` | `5.952 | 0.9955 / 0.9955 | 1594.51 / 1594.50` | `54.258 | 1.0000 / 1.0000 | 2064.30 / 2064.00` |

## MXFP4 encode

| Shape | `v2-enc` | `v3-enc` | `triton-cce` |
|---|---|---|---|
| `256x256->512` | `0.210 | 0.9933 / 0.9932 | 1.01 / 0.76` | `0.140 | 0.9933 / 0.9932 | 0.76 / 0.51` | `0.473 | 1.0000 / 1.0000 | 0.75 / 0.75` |
| `512x256->512` | `0.213 | 0.9934 / 0.9932 | 1.77 / 1.27` | `0.141 | 0.9934 / 0.9932 | 1.27 / 0.77` | `0.455 | 1.0000 / 1.0000 | 1.01 / 1.00` |
| `4Kx4K->32K` | `1.511 | 0.9934 / 0.9934 | 914.81 / 664.81` | `3.273 | 0.9934 / 0.9934 | 664.81 / 414.81` | `14.076 | 1.0000 / 1.0000 | 564.12 / 564.00` |
| `4Kx8K->32K` | `2.025 | 0.9932 / 0.9934 | 1196.81 / 946.81` | `3.624 | 0.9932 / 0.9934 | 946.81 / 696.81` | `28.263 | 1.0000 / 1.0000 | 1128.19 / 1128.00` |
| `8Kx4K->32K` | `2.876 | 0.9929 / 0.9934 | 1580.62 / 1079.62` | `6.353 | 0.9929 / 0.9934 | 1079.62 / 579.62` | `27.361 | 1.0000 / 1.0000 | 628.17 / 628.00` |
| `4Kx4K->128K` | `6.107 | 0.9932 / 0.9934 | 3564.00 / 2563.25` | `12.634 | 0.9932 / 0.9934 | 2564.00 / 1563.25` | `54.095 | 1.0000 / 1.0000 | 2064.30 / 2064.00` |

## MXFP4 decode

| Shape | `v2-dec` | `v3-dec` | `triton-cce` |
|---|---|---|---|
| `256x256->512` | `0.210 | 0.9323 / 0.9336 | 1.01 / 0.76` | `0.138 | 0.9820 / 0.9826 | 0.76 / 0.51` | `0.473 | 1.0000 / 1.0000 | 0.75 / 0.75` |
| `512x256->512` | `0.209 | 0.9343 / 0.9357 | 1.77 / 1.27` | `0.143 | 0.9823 / 0.9823 | 1.27 / 0.77` | `0.455 | 1.0000 / 1.0000 | 1.01 / 1.00` |
| `4Kx4K->32K` | `1.518 | 0.9833 / 0.9833 | 914.81 / 664.81` | `3.185 | 0.9833 / 0.9833 | 664.81 / 414.81` | `14.076 | 1.0000 / 1.0000 | 564.12 / 564.00` |
| `4Kx8K->32K` | `2.033 | 0.9829 / 0.9833 | 1196.81 / 946.81` | `3.548 | 0.9828 / 0.9833 | 946.81 / 696.81` | `28.263 | 1.0000 / 1.0000 | 1128.19 / 1128.00` |
| `8Kx4K->32K` | `2.883 | 0.9831 / 0.9833 | 1580.62 / 1079.62` | `6.177 | 0.9831 / 0.9833 | 1079.62 / 579.62` | `27.361 | 1.0000 / 1.0000 | 628.17 / 628.00` |
| `4Kx4K->128K` | `6.141 | 0.9832 / 0.9833 | 3563.25 / 2563.25` | `12.296 | 0.9832 / 0.9833 | 2564.00 / 1563.25` | `54.095 | 1.0000 / 1.0000 | 2064.30 / 2064.00` |

## Conclusions

- If the objective is raw throughput, `v2` is the clear frontrunner in both NVFP4 and MXFP4.
- If the objective is reduced memory footprint, `v3` is the better result in both NVFP4 and MXFP4.
- On NVFP4, `enc` and `dec` have essentially identical speed, with `dec` slightly better cosine.
- On MXFP4, `enc` is clearly better than `dec` on cosine.
- Triton BF16 remains the exact baseline, but it is not competitive on speed in this regime.
