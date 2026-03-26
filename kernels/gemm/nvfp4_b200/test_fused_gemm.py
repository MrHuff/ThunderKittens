"""
Test script for NVFP4 Fused Quantize+GEMM kernel.

Compares 5 modes:
1. Separate quantize→GEMM pipeline (global amax, 2-pass)
2. Fused GEMM with constant SCALE_MAX (no pre-scan)
3. Fused GEMM with CTA-level amax (pre-scan)
4. Experimental both-bf16 fused GEMM with constant SCALE_MAX
5. Experimental both-bf16 fused GEMM with CTA-level amax
All compared against PyTorch bf16 matmul reference.

Usage:
  python test_fused_gemm.py [M] [N] [K]
  python test_fused_gemm.py --sweep
"""
import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from _C import (nvfp4_fused_gemm, nvfp4_fused_gemm_cta_amax,
                nvfp4_fused_gemm_both_bf16, nvfp4_fused_gemm_both_bf16_cta_amax,
                nvfp4_fused_gemm_shared_a_debug,
                nvfp4_fused_gemm_cta_amax_shared_a_debug,
                nvfp4_fused_gemm_shared_a_debug_dump,
                nvfp4_fused_gemm_cta_amax_shared_a_debug_dump,
                nvfp4_gemm, nvfp4_quantize)  # type: ignore


DEBUG_A_ROWS = 128
DEBUG_A_COLS = 128
DEBUG_A_SC_BYTES = 2048


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> None:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    max_diff = (A - A_ref).abs().max().item()
    mean_diff = (A - A_ref).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        A.flatten().unsqueeze(0), A_ref.flatten().unsqueeze(0)
    ).item()
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Cos sim:   {cos_sim:.10f}")


def fused_backend_label(N: int) -> str:
    if N % 256 == 0:
        return "dual-column reuse backend"
    return "single-column fallback"


def compare_byte_dump(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    equal = torch.equal(lhs, rhs)
    diff = (lhs.to(torch.int16) - rhs.to(torch.int16)).abs()
    max_diff = diff.max().item()
    mismatch_count = (lhs != rhs).sum().item()
    if mismatch_count:
        lhs_flat = lhs.flatten()
        rhs_flat = rhs.flatten()
        first_mismatch = (lhs_flat != rhs_flat).nonzero()[0].item()
        window_end = min(first_mismatch + 8, lhs_flat.numel())
        lhs_window = lhs_flat[first_mismatch:window_end].tolist()
        rhs_window = rhs_flat[first_mismatch:window_end].tolist()
        print(
            f"  {name}: equal={equal}, mismatches={mismatch_count}, first mismatch={first_mismatch}, "
            f"max byte diff={max_diff}, lhs[{first_mismatch}:{window_end}]={lhs_window}, "
            f"rhs[{first_mismatch}:{window_end}]={rhs_window}"
        )
    else:
        print(f"  {name}: equal={equal}, mismatches=0, max byte diff={max_diff}")


def print_split_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> None:
    half = A.size(1) // 2
    print(f"  {name} left half:")
    check_diff(f"{name}_left", A[:, :half], A_ref[:, :half])
    print(f"  {name} right half:")
    check_diff(f"{name}_right", A[:, half:], A_ref[:, half:])


def run_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  M={M}, N={N}, K={K}")
    print(f"  Fused backend: {fused_backend_label(N)}")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    D_ref = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)

    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    D_separate = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()
    print(f"\n[0] Separate quantize→GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)

    D_fused_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_const)
    torch.cuda.synchronize()
    print(f"\n[1] Fused (constant SCALE_MAX) vs bf16 ref:")
    check_diff("fused_const vs ref", D_fused_const, D_ref)
    print(f"\n[1] Fused (constant SCALE_MAX) vs separate:")
    check_diff("fused_const vs sep", D_fused_const, D_separate)

    D_fused_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm_cta_amax(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_cta)
    torch.cuda.synchronize()
    print(f"\n[2] Fused (CTA amax) vs bf16 ref:")
    check_diff("fused_cta vs ref", D_fused_cta, D_ref)
    print(f"\n[2] Fused (CTA amax) vs separate:")
    check_diff("fused_cta vs sep", D_fused_cta, D_separate)

    D_both_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_both_const)
    torch.cuda.synchronize()
    print(f"\n[3] Fused both-bf16 (constant SCALE_MAX) vs bf16 ref:")
    check_diff("fused_both_const vs ref", D_both_const, D_ref)
    print(f"\n[3] Fused both-bf16 (constant SCALE_MAX) vs separate:")
    check_diff("fused_both_const vs sep", D_both_const, D_separate)

    D_both_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_both_cta)
    torch.cuda.synchronize()
    print(f"\n[4] Fused both-bf16 (CTA amax) vs bf16 ref:")
    check_diff("fused_both_cta vs ref", D_both_cta, D_ref)
    print(f"\n[4] Fused both-bf16 (CTA amax) vs separate:")
    check_diff("fused_both_cta vs sep", D_both_cta, D_separate)

    NUM_WARMUPS = 5
    NUM_ITERS = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def bench(fn, label):
        for _ in range(NUM_WARMUPS):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(NUM_ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / NUM_ITERS
        flops = 2.0 * M * N * K
        tflops = flops * 1e-12
        print(f"  {label}: {ms:.3f} ms  ({tflops/ms*1e3:.2f} TFLOPs)")

    print(f"\n{'='*72}")
    print("Benchmarks:")
    bench(lambda: nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False),
          "Separate quantize only  ")
    bench(lambda: nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate),
          "Separate GEMM only      ")
    bench(lambda: [nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False),
                   nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)],
          "Separate (quantize+GEMM)  ")
    bench(lambda: nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_const),
          "Fused (constant SCALE_MAX)")
    bench(lambda: nvfp4_fused_gemm_cta_amax(A_bf16, B_fp4x2, B_sc, B_sc_global, D_fused_cta),
          "Fused (CTA amax)          ")
    bench(lambda: nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_both_const),
          "Both-bf16 fused (const)   ")
    bench(lambda: nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_both_cta),
          "Both-bf16 fused (CTA)     ")
    print(f"{'='*72}")


def run_shared_a_debug_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  Shared-A Debug  M={M}, N={N}, K={K}")
    print("  Public fused backend remains unchanged during this run")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    D_ref = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)

    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    D_separate = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()

    print("\n[0] Separate quantize->GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)

    D_public_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_public_const)
    D_public_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm_cta_amax(A_bf16, B_fp4x2, B_sc, B_sc_global, D_public_cta)

    D_shared_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_shared_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    do_dump = (M, N, K) == (256, 512, 256)
    if do_dump:
        dump_cta0_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        dump_cta1_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        dump_cta0_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
        dump_cta1_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

        nvfp4_fused_gemm_shared_a_debug_dump(
            A_bf16, B_fp4x2, B_sc, B_sc_global, D_shared_const,
            dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[1] Shared-A debug dump checks (constant SCALE_MAX):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        nvfp4_fused_gemm_cta_amax_shared_a_debug_dump(
            A_bf16, B_fp4x2, B_sc, B_sc_global, D_shared_cta,
            dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[2] Shared-A debug dump checks (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)
    else:
        nvfp4_fused_gemm_shared_a_debug(A_bf16, B_fp4x2, B_sc, B_sc_global, D_shared_const)
        nvfp4_fused_gemm_cta_amax_shared_a_debug(A_bf16, B_fp4x2, B_sc, B_sc_global, D_shared_cta)
        torch.cuda.synchronize()

    print("\n[1] Shared-A debug (constant SCALE_MAX) vs bf16 ref:")
    check_diff("shared_a_const vs ref", D_shared_const, D_ref)
    print("\n[1] Shared-A debug (constant SCALE_MAX) vs separate:")
    check_diff("shared_a_const vs sep", D_shared_const, D_separate)
    print_split_diff("shared_a_const vs sep", D_shared_const, D_separate)

    print("\n[2] Shared-A debug (CTA amax) vs bf16 ref:")
    check_diff("shared_a_cta vs ref", D_shared_cta, D_ref)
    print("\n[2] Shared-A debug (CTA amax) vs separate:")
    check_diff("shared_a_cta vs sep", D_shared_cta, D_separate)
    print_split_diff("shared_a_cta vs sep", D_shared_cta, D_separate)

    if not torch.isfinite(D_shared_const).all() or not torch.isfinite(D_shared_cta).all():
        print("\nSkipping Shared-A debug benchmarks because the isolated backend produced non-finite outputs.")
        return

    NUM_WARMUPS = 5
    NUM_ITERS = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def bench(fn, label):
        for _ in range(NUM_WARMUPS):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(NUM_ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / NUM_ITERS
        flops = 2.0 * M * N * K
        tflops = flops * 1e-12
        print(f"  {label}: {ms:.3f} ms  ({tflops/ms*1e3:.2f} TFLOPs)")

    print(f"\n{'='*72}")
    print("Shared-A Debug Benchmarks:")
    bench(lambda: [nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False),
                   nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)],
          "Separate (quantize+GEMM)    ")
    bench(lambda: nvfp4_fused_gemm(A_bf16, B_fp4x2, B_sc, B_sc_global, D_public_const),
          "Public fused (constant)     ")
    bench(lambda: nvfp4_fused_gemm_cta_amax(A_bf16, B_fp4x2, B_sc, B_sc_global, D_public_cta),
          "Public fused (CTA amax)     ")
    bench(lambda: nvfp4_fused_gemm_shared_a_debug(A_bf16, B_fp4x2, B_sc, B_sc_global, D_shared_const),
          "Shared-A debug (constant)   ")
    bench(lambda: nvfp4_fused_gemm_cta_amax_shared_a_debug(A_bf16, B_fp4x2, B_sc, B_sc_global, D_shared_cta),
          "Shared-A debug (CTA amax)   ")
    print(f"{'='*72}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device unavailable. Run test_fused_gemm.py on a GPU-enabled B200 host."
        )

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        for size in (256, 1024, 2048, 4096):
            run_case(size, size, size)
            print()
    elif len(sys.argv) > 1 and sys.argv[1] == "--shared-a-debug":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_shared_a_debug_case(M, N, K)
    else:
        M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
        run_case(M, N, K)
