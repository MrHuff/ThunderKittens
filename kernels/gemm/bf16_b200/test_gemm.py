import sys
import time

import torch

from _C import bf16_gemm, bf16_gemm_out  # type: ignore


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "nonzero": float((diff != 0).sum().item()),
    }


def main() -> None:
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    torch.manual_seed(1234)
    a = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / (k**0.25)).contiguous()
    b = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / (k**0.25)).contiguous()

    out = bf16_gemm(a, b)
    ref = torch.mm(a, b.t())
    stats = _diff_stats(out, ref)
    print(f"shape: M={m} N={n} K={k}")
    print(
        "TK BF16 vs torch.mm: "
        f"max={stats['max']:.8f} mean={stats['mean']:.8f} nonzero={int(stats['nonzero'])}"
    )

    out2 = torch.empty_like(out)
    bf16_gemm_out(a, b, out2)
    if not torch.equal(out, out2):
        raise AssertionError("bf16_gemm and bf16_gemm_out produced different outputs")

    warmup = 5
    iters = 20
    for _ in range(warmup):
        bf16_gemm_out(a, b, out2)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        bf16_gemm_out(a, b, out2)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    tflops = (2.0 * m * n * k) / elapsed / 1e12
    print(f"avg_time_us={elapsed * 1e6:.2f} tflops={tflops:.2f}")


if __name__ == "__main__":
    main()
