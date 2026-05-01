import sys
import time

import torch

from _C import (  # type: ignore
    bf16_gemm,
    bf16_gemm_alpha,
    bf16_gemm_alpha_k16,
    bf16_gemm_alpha_k32,
    bf16_gemm_alpha_k128,
    bf16_gemm_alpha_cublaslt,
    bf16_gemm_alpha_reverse,
    bf16_gemm_alpha_reverse_k16,
    bf16_gemm_out,
)


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

    alpha = 3.94728715e-7
    out_scaled = bf16_gemm(a, b, alpha)
    ref_scaled = (torch.mm(a.float(), b.t().float()) * alpha).to(torch.bfloat16)
    scaled_stats = _diff_stats(out_scaled, ref_scaled)
    print(
        "TK scaled BF16 vs fp32-mm scaled: "
        f"max={scaled_stats['max']:.8f} "
        f"mean={scaled_stats['mean']:.8f} "
        f"nonzero={int(scaled_stats['nonzero'])}"
    )

    alpha_tensor = torch.tensor(alpha, device="cuda", dtype=torch.float32)
    out_scaled_tensor = bf16_gemm_alpha(a, b, alpha_tensor)
    if not torch.equal(out_scaled, out_scaled_tensor):
        raise AssertionError("host-alpha and device-alpha GEMM outputs differ")

    for name, fn in (
        ("k16", bf16_gemm_alpha_k16),
        ("k32", bf16_gemm_alpha_k32),
        ("k128", bf16_gemm_alpha_k128),
    ):
        if k % int(name[1:]) != 0:
            continue
        out_variant = fn(a, b, alpha_tensor)
        if out_variant.shape != out_scaled.shape or not torch.isfinite(out_variant).all():
            raise AssertionError(f"{name} GEMM output is invalid")
        variant_stats = _diff_stats(out_variant, ref_scaled)
        print(
            f"TK scaled BF16 {name} vs fp32-mm scaled: "
            f"max={variant_stats['max']:.8f} "
            f"mean={variant_stats['mean']:.8f} "
            f"nonzero={int(variant_stats['nonzero'])}"
        )

    out_reverse = bf16_gemm_alpha_reverse(a, b, alpha_tensor)
    if k == 64 and not torch.equal(out_scaled_tensor, out_reverse):
        raise AssertionError("reverse-K64 should match forward order for a single K tile")

    out_reverse_k16 = bf16_gemm_alpha_reverse_k16(a, b, alpha_tensor)
    if out_reverse_k16.shape != out_scaled.shape or not torch.isfinite(out_reverse_k16).all():
        raise AssertionError("reverse-K16 GEMM output is invalid")

    out_cublaslt = bf16_gemm_alpha_cublaslt(a, b, alpha_tensor)
    if out_cublaslt.shape != out_scaled.shape or not torch.isfinite(out_cublaslt).all():
        raise AssertionError("cuBLASLt BF16 GEMM output is invalid")
    cublaslt_stats = _diff_stats(out_cublaslt, ref_scaled)
    print(
        "cuBLASLt scaled BF16 vs fp32-mm scaled: "
        f"max={cublaslt_stats['max']:.8f} "
        f"mean={cublaslt_stats['mean']:.8f} "
        f"nonzero={int(cublaslt_stats['nonzero'])}"
    )

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
