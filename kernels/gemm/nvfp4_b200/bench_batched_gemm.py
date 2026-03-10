#!/usr/bin/env python3
"""Benchmark: correctness + speed of standard GEMM vs batched GEMM."""
import sys, torch
sys.path.insert(0, '.')
from _C import nvfp4_quantize, nvfp4_gemm, nvfp4_batched_gemm

torch.random.manual_seed(42)

def quantize(A):
    M, Kh = A.shape; K = Kh * 2
    fp4 = torch.empty(M, Kh, dtype=torch.float4_e2m1fn_x2, device='cuda')
    sc  = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device='cuda')
    sg  = torch.empty(1, dtype=torch.float32, device='cuda')
    nvfp4_quantize(A, fp4, sc, sg, False)
    return fp4, sc, sg

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

# ─── Shapes to test ───
SHAPES = [
    (16384, 2048, 2048),
    (16384, 4096, 4096),
    (16384, 8192, 8192),
]

NUM_BATCHES = [1, 2, 3, 5]

print("=" * 110)
print(f"{'Shape':>22s} | {'Method':>14s} | {'NaN':>6s} | {'Time(ms)':>10s} | {'TFLOPs':>8s} | {'Correctness':>20s}")
print("=" * 110)

for M, K, N in SHAPES:
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / (K ** 0.5)
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / (K ** 0.5)
    Aq = quantize(A); Bq = quantize(B)
    flops_per_gemm = 2.0 * M * N * K

    # ── Standard GEMM (reference) ──
    D_ref = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    nvfp4_gemm(Aq[0], Aq[1], Aq[2], Bq[0], Bq[1], Bq[2], D_ref)
    torch.cuda.synchronize()
    ref_nan = D_ref.isnan().sum().item()

    def run_std():
        nvfp4_gemm(Aq[0], Aq[1], Aq[2], Bq[0], Bq[1], Bq[2], D_ref)
    std_ms = bench(run_std)
    std_tflops = flops_per_gemm / (std_ms * 1e-3) / 1e12

    print(f"{M}x{K}x{N:>5d} | {'std_gemm':>14s} | {ref_nan:>6d} | {std_ms:>10.3f} | {std_tflops:>8.1f} | {'(reference)':>20s}")

    # ── Batched GEMM with N copies ──
    for nb in NUM_BATCHES:
        try:
            D_b = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_batched_gemm(
                [Aq[0]] * nb, [Aq[1]] * nb, [Aq[2]] * nb,
                [Bq[0]] * nb, [Bq[1]] * nb, [Bq[2]] * nb, D_b)
            torch.cuda.synchronize()
            b_nan = D_b.isnan().sum().item()

            # Correctness: should equal nb * ref
            mask = ~D_b.isnan() & ~D_ref.isnan()
            if mask.sum() > 0:
                maxdiff = abs(D_b.float()[mask] - nb * D_ref.float()[mask]).max().item()
            else:
                maxdiff = float('nan')

            def run_batched(nb_=nb):
                nvfp4_batched_gemm(
                    [Aq[0]] * nb_, [Aq[1]] * nb_, [Aq[2]] * nb_,
                    [Bq[0]] * nb_, [Bq[1]] * nb_, [Bq[2]] * nb_, D_b)
            b_ms = bench(run_batched)
            total_flops = flops_per_gemm * nb
            b_tflops = total_flops / (b_ms * 1e-3) / 1e12

            # Compare: N individual std GEMMs
            def run_n_std(nb_=nb):
                for _ in range(nb_):
                    nvfp4_gemm(Aq[0], Aq[1], Aq[2], Bq[0], Bq[1], Bq[2], D_ref)
            n_std_ms = bench(run_n_std)
            speedup = n_std_ms / b_ms

            correctness = f"maxd={maxdiff:.4f}" if not (maxdiff != maxdiff) else "NaN"
            print(f"{' ':>22s} | {'batched_x' + str(nb):>14s} | {b_nan:>6d} | {b_ms:>10.3f} | {b_tflops:>8.1f} | {correctness:>20s}")
            print(f"{' ':>22s} | {str(nb) + 'x_std_loop':>14s} | {'':>6s} | {n_std_ms:>10.3f} | {'':>8s} | {'speedup=' + f'{speedup:.2f}x':>20s}")
        except Exception as e:
            print(f"{' ':>22s} | {'batched_x' + str(nb):>14s} | {'ERROR':>6s} | {'':>10s} | {'':>8s} | {str(e)[:40]:>20s}")
            torch.cuda.synchronize()  # reset CUDA state

    print("-" * 110)

print("DONE")
