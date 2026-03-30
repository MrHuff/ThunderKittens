"""
Apples-to-apples benchmark: persistent kernel vs localCTA quantize + fast GEMM.
Both use per-block amax (no global amax scan), both use the same GEMM kernel.
"""
import sys
from pathlib import Path
import torch
torch.random.manual_seed(42)

# Add paths for localCTA modules
ROOT = Path(__file__).resolve().parent
LOCALCTA_GEMM = ROOT / "localCTA_epilogue"
LOCALCTA_QUANT = ROOT.parents[3] / "TK_quantisation" / "nvfp4_CTA_local_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCALCTA_GEMM))
sys.path.insert(0, str(LOCALCTA_QUANT))

from _C import nvfp4_persistent_gemm  # type: ignore

# Try loading localCTA modules
try:
    import _C_nv_localcta_gemm as local_gemm  # type: ignore
    import _tk_quant_localcta as local_q  # type: ignore
    HAS_LOCALCTA = True
except ImportError as e:
    print(f"WARNING: localCTA modules not available: {e}")
    print("  Build: cd TK_quantisation/nvfp4_CTA_local_v1 && make -B")
    print("  Build: cd localCTA_epilogue && make -B")
    HAS_LOCALCTA = False

# Also load regular TK for 3-way comparison
from _C import nvfp4_gemm, nvfp4_quantize  # type: ignore


def bench(fn, warmup=10, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run_case(M, N, K):
    print(f"\n{'='*72}")
    print(f"  M={M}, N={N}, K={K}")
    print(f"{'='*72}")

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    D_pers = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # === Persistent kernel ===
    ms_pers = bench(lambda: nvfp4_persistent_gemm(A, B, D_pers))

    # === LocalCTA: quantize_prepared + fast_gemm ===
    if HAS_LOCALCTA:
        D_lcta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

        # Quantize (prepared = chunk scales baked into FP8 block scales)
        def lcta_e2e():
            Aq = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
            Bq = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
            local_gemm.nvfp4_localcta_fast_gemm(
                Aq[0], Aq[1], Bq[0], Bq[1], D_lcta)

        ms_lcta = bench(lcta_e2e)

        # Also measure components
        ms_lcta_q = bench(lambda: [
            local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True),
            local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)])
        Aq = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
        Bq = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
        ms_lcta_g = bench(lambda: local_gemm.nvfp4_localcta_fast_gemm(
            Aq[0], Aq[1], Bq[0], Bq[1], D_lcta))

        # Correctness
        lcta_e2e()
        nvfp4_persistent_gemm(A, B, D_pers)
        torch.cuda.synchronize()
        cos_lcta = torch.nn.functional.cosine_similarity(
            D_pers.flatten().float().unsqueeze(0),
            D_lcta.flatten().float().unsqueeze(0)).item()
    else:
        ms_lcta = ms_lcta_q = ms_lcta_g = float('nan')
        cos_lcta = float('nan')

    # === Regular TK: quantize + gemm ===
    A_fp4 = torch.empty(M, K//2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sg = torch.empty(1, dtype=torch.float32, device="cuda")
    B_fp4 = torch.empty(N, K//2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N//128, K//64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sg = torch.empty(1, dtype=torch.float32, device="cuda")
    D_tk = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    ms_tk = bench(lambda: [nvfp4_quantize(A, A_fp4, A_sc, A_sg, False),
                            nvfp4_quantize(B, B_fp4, B_sc, B_sg, False),
                            nvfp4_gemm(A_fp4, A_sc, A_sg, B_fp4, B_sc, B_sg, D_tk)])

    flops = 2.0 * M * N * K
    pers_tflops = flops / (ms_pers * 1e-3) / 1e12
    lcta_tflops = flops / (ms_lcta * 1e-3) / 1e12 if HAS_LOCALCTA else 0
    tk_tflops = flops / (ms_tk * 1e-3) / 1e12

    print(f"    Regular TK:       {ms_tk:.4f} ms  ({tk_tflops:.0f} TFLOPs)")
    if HAS_LOCALCTA:
        print(f"    LocalCTA sep:     {ms_lcta:.4f} ms  ({lcta_tflops:.0f} TFLOPs)  [Q={ms_lcta_q:.4f} G={ms_lcta_g:.4f}]")
    print(f"    Persistent:       {ms_pers:.4f} ms  ({pers_tflops:.0f} TFLOPs)")

    if HAS_LOCALCTA:
        delta = ms_lcta - ms_pers
        if delta > 0:
            print(f"    → Persistent wins vs localCTA by {delta:.4f} ms ({delta/ms_lcta*100:.1f}%)")
        else:
            print(f"    → LocalCTA wins vs persistent by {-delta:.4f} ms ({-delta/ms_pers*100:.1f}%)")
        print(f"    Cos(pers vs lcta): {cos_lcta:.6f}")

    delta_tk = ms_tk - ms_pers
    if delta_tk > 0:
        print(f"    → Persistent wins vs TK by {delta_tk:.4f} ms ({delta_tk/ms_tk*100:.1f}%)")
    else:
        print(f"    → TK wins vs persistent by {-delta_tk:.4f} ms ({-delta_tk/ms_pers*100:.1f}%)")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print("\n*** QKV Forward shapes (M × 6144 × 2048) ***")
    for M in [2048, 4096, 8192, 16384, 32768, 65536]:
        try:
            run_case(M, 6144, 2048)
        except Exception as e:
            print(f"  M={M}: FAILED: {e}")

    print("\n\n*** QKV Dgrad shapes (M × 2048 × 6144) ***")
    for M in [2048, 4096, 8192, 16384, 32768]:
        try:
            run_case(M, 2048, 6144)
        except Exception as e:
            print(f"  M={M}: FAILED: {e}")

    print("\n\n*** Square shapes ***")
    for s in [4096, 8192]:
        try:
            run_case(s, s, s)
        except Exception as e:
            print(f"  {s}³: FAILED: {e}")
