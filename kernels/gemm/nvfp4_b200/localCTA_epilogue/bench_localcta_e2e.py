import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
QUANT_ROOT = ROOT.parents[4] / "TK_quantisation" / "nvfp4_CTA_local_v1"
V5_ROOT = ROOT.parents[4] / "TK_quantisation" / "nvfp4_v5"
LEGACY_GEMM_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_ROOT))
sys.path.insert(0, str(V5_ROOT))
sys.path.insert(0, str(LEGACY_GEMM_ROOT))

import _C_nv_localcta_gemm as local_gemm  # type: ignore
import _C as legacy_gemm  # type: ignore
import _tk_quant_localcta as local_q  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore


def bench(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def constant_chunk_grid(rows: int, cols: int, sg: torch.Tensor) -> torch.Tensor:
    return torch.full((rows // 128, cols // 128), sg.item(), device=sg.device, dtype=torch.float32)


def main() -> None:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K ** 0.25)

    D_local = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    D_v5 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    t_local_q = bench(lambda: local_q.tk_localcta_quantize_for_gemm(A, False, True))
    t_v5_q = bench(lambda: q_v5.tk_quantize_for_gemm(A, False, True))

    A_local = local_q.tk_localcta_quantize_for_gemm(A, False, True)
    B_local = local_q.tk_localcta_quantize_for_gemm(B, False, True)
    A_v5 = q_v5.tk_quantize_for_gemm(A, False, True)
    B_v5 = q_v5.tk_quantize_for_gemm(B, False, True)

    t_local_deq = bench(lambda: local_q.tk_localcta_reconstruct_row(A_local[0], A_local[1], A_local[4]))
    sg_grid_v5 = constant_chunk_grid(M, K, A_v5[4])
    t_v5_deq = bench(lambda: local_q.tk_localcta_reconstruct_row(A_v5[0], A_v5[1], sg_grid_v5))

    t_local_gemm = bench(lambda: local_gemm.nvfp4_localcta_gemm(
        A_local[0], A_local[1], A_local[4],
        B_local[0], B_local[1], B_local[4],
        D_local,
    ))
    t_v5_gemm = bench(lambda: legacy_gemm.nvfp4_gemm(
        A_v5[0], A_v5[1], A_v5[4],
        B_v5[0], B_v5[1], B_v5[4],
        D_v5,
    ))

    def local_e2e():
        Aq = local_q.tk_localcta_quantize_for_gemm(A, False, True)
        Bq = local_q.tk_localcta_quantize_for_gemm(B, False, True)
        local_gemm.nvfp4_localcta_gemm(
            Aq[0], Aq[1], Aq[4],
            Bq[0], Bq[1], Bq[4],
            D_local,
        )

    t_local_e2e = bench(local_e2e, warmup=2, iters=5)

    def v5_e2e():
        Aq = q_v5.tk_quantize_for_gemm(A, False, True)
        Bq = q_v5.tk_quantize_for_gemm(B, False, True)
        legacy_gemm.nvfp4_gemm(Aq[0], Aq[1], Aq[4], Bq[0], Bq[1], Bq[4], D_v5)

    t_v5_e2e = bench(v5_e2e, warmup=2, iters=5)

    print(f"quant_only_ms localCTA={t_local_q:.3f} v5={t_v5_q:.3f}")
    print(f"dequant_only_ms localCTA={t_local_deq:.3f} v5={t_v5_deq:.3f}")
    print(f"gemm_only_ms localCTA={t_local_gemm:.3f} v5={t_v5_gemm:.3f}")
    print(f"e2e_ms localCTA={t_local_e2e:.3f} v5={t_v5_e2e:.3f}")


if __name__ == "__main__":
    main()
