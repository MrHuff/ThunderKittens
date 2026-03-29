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
  python test_fused_gemm.py --both-bf16-quadcol [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-transport-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-producer-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-b-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-wait-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-b-wait-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-quant-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-b-quant-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-stage1-quant-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-b-stage1-quant-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-quant-per-stage-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-b-quant-per-stage-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-quant-then-wait-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-b-quant-then-wait-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-b-2cta-a-quant-then-skip-wait-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-cluster-2x2 [M] [N] [K]
  python test_fused_gemm.py --both-bf16-cluster-2x2-dump-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-cluster-2x2-transport-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-a-2cta [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-a-2cta-publish-only-local-a [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-a-2cta-dump-only [M] [N] [K]
  python test_fused_gemm.py --both-bf16-shared-a-2cta-transport-only [M] [K]
  python test_fused_gemm.py --both-bf16-shared-a-2cta-transport-import-only [M] [K]
"""
import os
import sys
import time
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

def warmup_cuda(max_attempts: int = 5, sleep_s: float = 1.0) -> bool:
    for attempt in range(max_attempts):
        try:
            if torch.cuda.is_available():
                torch.empty(1, device="cuda")
                return True
        except Exception:
            pass
        if attempt + 1 < max_attempts:
            time.sleep(sleep_s)
    return False


def cuda_retry(fn, max_attempts: int = 5, sleep_s: float = 1.0):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RuntimeError as exc:
            last_exc = exc
            msg = str(exc)
            if "Error 304" not in msg and "cudaGetDeviceCount" not in msg:
                raise
            if attempt + 1 < max_attempts:
                time.sleep(sleep_s)
    raise last_exc


# Warm up CUDA through plain Torch before importing the extension.
# This avoids a flaky import-time cudaGetDeviceCount() failure after prior bad launches.
CUDA_READY = warmup_cuda()


def load_ext() -> None:
    global nvfp4_fused_gemm, nvfp4_fused_gemm_cta_amax
    global nvfp4_fused_gemm_both_bf16, nvfp4_fused_gemm_both_bf16_cta_amax
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_producer_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_stage1_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_stage1_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_per_stage_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_per_stage_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_then_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_skip_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_transport_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_producer_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_stage1_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_stage1_quant_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_per_stage_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_per_stage_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_then_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_skip_wait_only_debug
    global nvfp4_fused_gemm_both_bf16_quadcol_debug, nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug
    global nvfp4_fused_gemm_both_bf16_cluster_2x2_debug, nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug
    global nvfp4_fused_gemm_both_bf16_cluster_2x2_transport_only_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_transport_only_debug
    global nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_dump, nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug, nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_dump, nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug, nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_publish_only_local_a_debug
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_debug_dump
    global nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump
    global nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_local_a_debug_dump
    global nvfp4_fused_gemm_shared_a_debug, nvfp4_fused_gemm_cta_amax_shared_a_debug
    global nvfp4_fused_gemm_shared_a_debug_dump, nvfp4_fused_gemm_cta_amax_shared_a_debug_dump
    global nvfp4_gemm, nvfp4_quantize

    if "nvfp4_fused_gemm" in globals():
        return

    from _C import (nvfp4_fused_gemm, nvfp4_fused_gemm_cta_amax,
                    nvfp4_fused_gemm_both_bf16, nvfp4_fused_gemm_both_bf16_cta_amax,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_producer_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_stage1_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_stage1_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_per_stage_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_per_stage_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_then_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_skip_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_transport_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_producer_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_stage1_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_stage1_quant_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_per_stage_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_per_stage_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_then_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_skip_wait_only_debug,
                    nvfp4_fused_gemm_both_bf16_quadcol_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug,
                    nvfp4_fused_gemm_both_bf16_cluster_2x2_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug,
                    nvfp4_fused_gemm_both_bf16_cluster_2x2_transport_only_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_transport_only_debug,
                    nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_publish_only_local_a_debug,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_debug_dump,
                    nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump,
                    nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_local_a_debug_dump,
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


def check_diff_quadrants(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> None:
    M, N = A.shape
    m_mid = M // 2
    n_mid = N // 2
    quadrants = [
        ("top-left", A[:m_mid, :n_mid], A_ref[:m_mid, :n_mid]),
        ("top-right", A[:m_mid, n_mid:], A_ref[:m_mid, n_mid:]),
        ("bot-left", A[m_mid:, :n_mid], A_ref[m_mid:, :n_mid]),
        ("bot-right", A[m_mid:, n_mid:], A_ref[m_mid:, n_mid:]),
    ]
    print(f"  Quadrants for {name}:")
    for label, tile, tile_ref in quadrants:
        tile = tile.to(torch.float32)
        tile_ref = tile_ref.to(torch.float32)
        max_diff = (tile - tile_ref).abs().max().item()
        mean_diff = (tile - tile_ref).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            tile.flatten().unsqueeze(0), tile_ref.flatten().unsqueeze(0)
        ).item()
        print(
            f"    {label:<9} max={max_diff:.6f} mean={mean_diff:.6f} cos={cos_sim:.10f}"
        )


def fused_backend_label(N: int) -> str:
    if N % 256 == 0:
        return "dual-column reuse backend"
    return "single-column fallback"


def both_bf16_backend_label(M: int, N: int, K: int) -> str:
    if M >= 2048 and N >= 2048 and M % 256 == 0 and N % 256 == 0 and K % 256 == 0:
        return "shared-B 2CTA backend (Kb=128, cluster MMA)"
    return "single-CTA ping-pong (2-stage)"


def both_bf16_shared_b_2cta_backend_label(M: int, N: int, K: int) -> str:
    if M % 256 == 0 and N % 256 == 0 and K % 128 == 0:
        return "2CTA shared-B debug backend (Kb=128, cluster MMA)"
    return "unsupported"


def both_bf16_quadcol_backend_label(N: int, K: int) -> str:
    if N % 512 == 0:
        return "same-CTA quad-column A-reuse debug backend (1-stage, 1-epi)"
    return "unsupported"


def both_bf16_cluster_2x2_backend_label(M: int, N: int, K: int) -> str:
    if M % 256 == 0 and N % 256 == 0 and K % 256 == 0:
        return "2x2 clustered A/B reuse debug backend"
    return "unsupported"


def both_bf16_shared_a_2cta_backend_label(N: int, K: int) -> str:
    if N % 256 == 0:
        return "cross-CTA shared-A debug backend"
    return "unsupported"


def both_bf16_shared_a_2cta_local_a_backend_label(N: int, K: int) -> str:
    if N % 256 == 0:
        return "cross-CTA debug backend with CTA1 local A quantization"
    return "unsupported"


def both_bf16_shared_a_2cta_publish_only_local_a_backend_label(N: int, K: int) -> str:
    if N % 256 == 0:
        return "cross-CTA debug backend with remote publish active and CTA1 local A consumption"
    return "unsupported"


def both_bf16_shared_a_2cta_transport_backend_label() -> str:
    return "cross-CTA shared-A transport microkernel"


def both_bf16_shared_a_2cta_transport_import_backend_label() -> str:
    return "cross-CTA shared-A transport+import microkernel"


def both_bf16_shared_a_2cta_full_transport_backend_label() -> str:
    return "cross-CTA shared-A full-loop transport debug"


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


def print_tile_cos_matrix(name: str, A: torch.Tensor, A_ref: torch.Tensor, tile_cols: int = 128) -> None:
    if A.size(1) % tile_cols != 0 or A_ref.size(1) % tile_cols != 0:
        return
    tiles = A.size(1) // tile_cols
    if tiles > 8:
        return
    A32 = A.to(torch.float32)
    R32 = A_ref.to(torch.float32)
    print(f"  {name} tile cosine matrix ({tile_cols}-col tiles):")
    for i in range(tiles):
        row = []
        Ai = A32[:, i * tile_cols:(i + 1) * tile_cols].flatten()
        for j in range(tiles):
            Rj = R32[:, j * tile_cols:(j + 1) * tile_cols].flatten()
            cos = torch.nn.functional.cosine_similarity(
                Ai.unsqueeze(0), Rj.unsqueeze(0)
            ).item()
            row.append(f"{cos: .4f}")
        print("   ", i, " ".join(row))


def print_top_outliers(name: str, A: torch.Tensor, A_ref: torch.Tensor, limit: int = 8) -> None:
    diff = (A.to(torch.float32) - A_ref.to(torch.float32)).abs()
    flat = diff.flatten()
    k = min(limit, flat.numel())
    if k == 0:
        return
    vals, idxs = torch.topk(flat, k)
    cols = A.size(1)
    print(f"  {name} top abs-diff coordinates:")
    for rank in range(k):
        idx = idxs[rank].item()
        row = idx // cols
        col = idx % cols
        lhs = A[row, col].item()
        rhs = A_ref[row, col].item()
        print(
            f"    {rank}: ({row}, {col}) diff={vals[rank].item():.6f} "
            f"lhs={lhs} rhs={rhs}"
        )


def run_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  M={M}, N={N}, K={K}")
    print(f"  Fused backend: {fused_backend_label(N)}")
    print(f"  Both-bf16 backend: {both_bf16_backend_label(M, N, K)}")
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


def run_both_bf16_quadcol_debug_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Quad-Column Debug  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_quadcol_backend_label(N, K)}")
    print("  Public both-bf16 backend remains unchanged during this run")
    print(f"{'='*72}")

    A_bf16 = cuda_retry(lambda: torch.randn(M, K, dtype=torch.bfloat16, device="cuda")) / K**0.25
    B_bf16 = cuda_retry(lambda: torch.randn(N, K, dtype=torch.bfloat16, device="cuda")) / K**0.25
    D_ref = cuda_retry(lambda: torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16))

    A_fp4x2 = cuda_retry(lambda: torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda"))
    A_sc = cuda_retry(lambda: torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda"))
    A_sc_global = cuda_retry(lambda: torch.empty(1, dtype=torch.float32, device="cuda"))
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)

    B_fp4x2 = cuda_retry(lambda: torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda"))
    B_sc = cuda_retry(lambda: torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda"))
    B_sc_global = cuda_retry(lambda: torch.empty(1, dtype=torch.float32, device="cuda"))
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    D_separate = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()

    D_public_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_public_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_quad_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_quad_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_public_const)
    nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_public_cta)
    nvfp4_fused_gemm_both_bf16_quadcol_debug(A_bf16, B_bf16, D_quad_const)
    nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug(A_bf16, B_bf16, D_quad_cta)
    torch.cuda.synchronize()

    print("\n[0] Separate quantize->GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)

    print("\n[1] Public both-bf16 (constant) vs separate:")
    check_diff("public_both_const vs sep", D_public_const, D_separate)
    print("\n[2] Quad-column debug (constant) vs separate:")
    check_diff("quadcol_const vs sep", D_quad_const, D_separate)
    print("\n[3] Public both-bf16 (CTA amax) vs separate:")
    check_diff("public_both_cta vs sep", D_public_cta, D_separate)
    print("\n[4] Quad-column debug (CTA amax) vs separate:")
    check_diff("quadcol_cta vs sep", D_quad_cta, D_separate)

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
    print("Both-BF16 Quad-Column Debug Benchmarks:")
    bench(lambda: nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_public_const),
          "Public both-bf16 (const)      ")
    bench(lambda: nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_public_cta),
          "Public both-bf16 (CTA)        ")
    bench(lambda: nvfp4_fused_gemm_both_bf16_quadcol_debug(A_bf16, B_bf16, D_quad_const),
          "Quad-column debug (const)     ")
    bench(lambda: nvfp4_fused_gemm_both_bf16_cta_amax_quadcol_debug(A_bf16, B_bf16, D_quad_cta),
          "Quad-column debug (CTA)       ")
    print(f"{'='*72}")


def run_both_bf16_shared_b_2cta_debug_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-B 2CTA Debug  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_b_2cta_backend_label(M, N, K)}")
    print("  Public both-bf16 backend remains unchanged during this run")
    print(f"{'='*72}")

    A_bf16 = cuda_retry(lambda: torch.randn(M, K, dtype=torch.bfloat16, device="cuda")) / K**0.25
    B_bf16 = cuda_retry(lambda: torch.randn(N, K, dtype=torch.bfloat16, device="cuda")) / K**0.25
    D_ref = cuda_retry(lambda: torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16))

    A_fp4x2 = cuda_retry(lambda: torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda"))
    A_sc = cuda_retry(lambda: torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda"))
    A_sc_global = cuda_retry(lambda: torch.empty(1, dtype=torch.float32, device="cuda"))
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)

    B_fp4x2 = cuda_retry(lambda: torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda"))
    B_sc = cuda_retry(lambda: torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda"))
    B_sc_global = cuda_retry(lambda: torch.empty(1, dtype=torch.float32, device="cuda"))
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    D_separate = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()

    D_public_const = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))
    D_public_cta = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))
    D_shared_b_const = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))
    D_shared_b_cta = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))

    nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_public_const)
    torch.cuda.synchronize()
    nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_public_cta)
    torch.cuda.synchronize()
    print("\n[-] Launching shared-B 2CTA debug constant kernel...")
    nvfp4_fused_gemm_both_bf16_shared_b_2cta_debug(A_bf16, B_bf16, D_shared_b_const)
    torch.cuda.synchronize()
    print("[-] Launching shared-B 2CTA debug CTA-amax kernel...")
    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_debug(A_bf16, B_bf16, D_shared_b_cta)
    torch.cuda.synchronize()

    print("\n[0] Separate quantize->GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)
    print("\n[1] Public both-bf16 (constant) vs separate:")
    check_diff("public_both_const vs sep", D_public_const, D_separate)
    print("\n[2] Shared-B 2CTA debug (constant) vs separate:")
    check_diff("shared_b_2cta_const vs sep", D_shared_b_const, D_separate)
    check_diff_quadrants("shared_b_2cta_const vs sep", D_shared_b_const, D_separate)
    print("\n[3] Public both-bf16 (CTA amax) vs separate:")
    check_diff("public_both_cta vs sep", D_public_cta, D_separate)
    print("\n[4] Shared-B 2CTA debug (CTA amax) vs separate:")
    check_diff("shared_b_2cta_cta vs sep", D_shared_b_cta, D_separate)
    check_diff_quadrants("shared_b_2cta_cta vs sep", D_shared_b_cta, D_separate)


def run_both_bf16_shared_b_2cta_transport_only_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-B 2CTA Transport-Only  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_b_2cta_backend_label(M, N, K)}")
    print("  This run stops before WG3 MMA/epilogue to isolate the front-half pipeline")
    print(f"{'='*72}")

    A_bf16 = cuda_retry(lambda: torch.randn(M, K, dtype=torch.bfloat16, device="cuda")) / K**0.25
    B_bf16 = cuda_retry(lambda: torch.randn(N, K, dtype=torch.bfloat16, device="cuda")) / K**0.25
    D = cuda_retry(lambda: torch.zeros(M, N, dtype=torch.bfloat16, device="cuda"))

    print("\n[-] Launching shared-B 2CTA transport-only constant kernel...")
    nvfp4_fused_gemm_both_bf16_shared_b_2cta_transport_only_debug(A_bf16, B_bf16, D)
    torch.cuda.synchronize()
    print("[-] Launching shared-B 2CTA transport-only CTA-amax kernel...")
    nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_transport_only_debug(A_bf16, B_bf16, D)
    torch.cuda.synchronize()
    print("\n[+] Shared-B 2CTA transport-only front half completed.")


def run_both_bf16_shared_b_2cta_split_case(
    mode_label: str,
    const_kernel,
    cta_kernel,
    M: int,
    N: int,
    K: int,
) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-B 2CTA {mode_label}  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_b_2cta_backend_label(M, N, K)}")
    print("  This run isolates one front-half path inside the shared-B debug kernel")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    print(f"\n[-] Launching shared-B 2CTA {mode_label.lower()} constant kernel...")
    const_kernel(A_bf16, B_bf16, D)
    torch.cuda.synchronize()
    print(f"[-] Launching shared-B 2CTA {mode_label.lower()} CTA-amax kernel...")
    cta_kernel(A_bf16, B_bf16, D)
    torch.cuda.synchronize()
    print(f"\n[+] Shared-B 2CTA {mode_label.lower()} path completed.")


def run_both_bf16_cluster_2x2_debug_case(M: int, N: int, K: int, dump_only: bool = False) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Clustered 2x2 Debug  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_cluster_2x2_backend_label(M, N, K)}")
    print("  Public both-bf16 backend remains unchanged during this run")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25

    if (M, N, K) == (256, 256, 256):
        debug_a_owner = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        debug_a_recv = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        debug_a_owner_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
        debug_a_recv_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
        debug_b_owner = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        debug_b_recv = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        debug_b_owner_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
        debug_b_recv_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

        nvfp4_fused_gemm_both_bf16_cluster_2x2_debug_dump(
            A_bf16, B_bf16,
            debug_a_owner, debug_a_recv, debug_a_owner_sc, debug_a_recv_sc,
            debug_b_owner, debug_b_recv, debug_b_owner_sc, debug_b_recv_sc)
        torch.cuda.synchronize()
        print("\n[-] Clustered 2x2 transport dump checks (constant SCALE_MAX):")
        compare_byte_dump("A tile", debug_a_owner, debug_a_recv)
        compare_byte_dump("A scales", debug_a_owner_sc, debug_a_recv_sc)
        compare_byte_dump("B tile", debug_b_owner, debug_b_recv)
        compare_byte_dump("B scales", debug_b_owner_sc, debug_b_recv_sc)

        nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug_dump(
            A_bf16, B_bf16,
            debug_a_owner, debug_a_recv, debug_a_owner_sc, debug_a_recv_sc,
            debug_b_owner, debug_b_recv, debug_b_owner_sc, debug_b_recv_sc)
        torch.cuda.synchronize()
        print("\n[-] Clustered 2x2 transport dump checks (CTA amax):")
        compare_byte_dump("A tile", debug_a_owner, debug_a_recv)
        compare_byte_dump("A scales", debug_a_owner_sc, debug_a_recv_sc)
        compare_byte_dump("B tile", debug_b_owner, debug_b_recv)
        compare_byte_dump("B scales", debug_b_owner_sc, debug_b_recv_sc)

        if dump_only:
            return

    D_ref = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)
    D_separate = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()

    D_public_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_public_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_cluster_const = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_cluster_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_public_const)
    torch.cuda.synchronize()
    nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_public_cta)
    torch.cuda.synchronize()
    print("\n[-] Launching clustered 2x2 debug constant kernel...")
    nvfp4_fused_gemm_both_bf16_cluster_2x2_debug(A_bf16, B_bf16, D_cluster_const)
    torch.cuda.synchronize()
    print("[-] Launching clustered 2x2 debug CTA-amax kernel...")
    nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_debug(A_bf16, B_bf16, D_cluster_cta)
    torch.cuda.synchronize()

    print("\n[0] Separate quantize->GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)
    print("\n[1] Public both-bf16 (constant) vs separate:")
    check_diff("public_both_const vs sep", D_public_const, D_separate)
    print("\n[2] Clustered 2x2 debug (constant) vs separate:")
    check_diff("cluster_2x2_const vs sep", D_cluster_const, D_separate)
    print("\n[3] Public both-bf16 (CTA amax) vs separate:")
    check_diff("public_both_cta vs sep", D_public_cta, D_separate)
    print("\n[4] Clustered 2x2 debug (CTA amax) vs separate:")
    check_diff("cluster_2x2_cta vs sep", D_cluster_cta, D_separate)


def run_both_bf16_shared_a_2cta_debug_case(
    M: int, N: int, K: int, dump_only: bool = False, local_a_only: bool = False
) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-A 2CTA Debug  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_a_2cta_backend_label(N, K)}")
    print("  Public both-bf16 backend remains unchanged during this run")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    D_ref = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)

    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)

    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    D_separate = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()

    D_public = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_public_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_shared = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_shared_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_shared_local = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_shared_local_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    if (M, N, K) == (256, 512, 256):
        dump_cta0_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        dump_cta1_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
        dump_cta0_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
        dump_cta1_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

        if not local_a_only:
            nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump(
                A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
            torch.cuda.synchronize()
            print("\n[-] Both-bf16 Shared-A 2CTA transport precheck (constant SCALE_MAX):")
            compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
            compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Both-bf16 Shared-A 2CTA transport local-A control (constant SCALE_MAX):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        if not dump_only and not local_a_only:
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump(
                A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
            torch.cuda.synchronize()
            print("\n[-] Both-bf16 Shared-A 2CTA transport precheck (CTA amax):")
            compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
            compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

            nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump(
                A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
            torch.cuda.synchronize()
            print("\n[-] Both-bf16 Shared-A 2CTA transport local-A control (CTA amax):")
            compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
            compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        if dump_only or local_a_only:
            return

    def run_step(label, fn):
        print(f"\n[-] Launching {label}...")
        fn()
        torch.cuda.synchronize()

    run_step(
        "public both-bf16 constant",
        lambda: nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_public),
    )
    run_step(
        "public both-bf16 CTA amax",
        lambda: nvfp4_fused_gemm_both_bf16_cta_amax(A_bf16, B_bf16, D_public_cta),
    )
    run_step(
        "shared-A 2CTA debug constant",
        lambda: nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug(A_bf16, B_bf16, D_shared),
    )
    run_step(
        "shared-A 2CTA debug CTA amax",
        lambda: nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug(A_bf16, B_bf16, D_shared_cta),
    )
    run_step(
        "shared-A 2CTA local-A debug constant",
        lambda: nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug(A_bf16, B_bf16, D_shared_local),
    )
    run_step(
        "shared-A 2CTA local-A debug CTA amax",
        lambda: nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug(A_bf16, B_bf16, D_shared_local_cta),
    )

    print("\n[0] Separate quantize->GEMM vs bf16 ref:")
    check_diff("sep vs ref", D_separate, D_ref)
    print("\n[1] Public both-bf16 (constant) vs separate:")
    check_diff("public_both_const vs sep", D_public, D_separate)
    print("\n[2] Shared-A 2CTA debug (constant) vs separate:")
    check_diff("shared_a_2cta_const vs sep", D_shared, D_separate)
    print_split_diff("shared_a_2cta_const vs sep", D_shared, D_separate)
    print_tile_cos_matrix("shared_a_2cta_const vs sep", D_shared, D_separate)
    print_top_outliers("shared_a_2cta_const vs sep", D_shared, D_separate)
    print("\n[3] Public both-bf16 (CTA amax) vs separate:")
    check_diff("public_both_cta vs sep", D_public_cta, D_separate)
    print("\n[4] Shared-A 2CTA debug (CTA amax) vs separate:")
    check_diff("shared_a_2cta_cta vs sep", D_shared_cta, D_separate)
    print_split_diff("shared_a_2cta_cta vs sep", D_shared_cta, D_separate)
    print_tile_cos_matrix("shared_a_2cta_cta vs sep", D_shared_cta, D_separate)
    print_top_outliers("shared_a_2cta_cta vs sep", D_shared_cta, D_separate)
    print("\n[5] Shared-A 2CTA local-A debug (constant) vs separate:")
    check_diff("shared_a_2cta_local_a_const vs sep", D_shared_local, D_separate)
    print_split_diff("shared_a_2cta_local_a_const vs sep", D_shared_local, D_separate)
    print_tile_cos_matrix("shared_a_2cta_local_a_const vs sep", D_shared_local, D_separate)
    print_top_outliers("shared_a_2cta_local_a_const vs sep", D_shared_local, D_separate)
    print("\n[6] Shared-A 2CTA local-A debug (CTA amax) vs separate:")
    check_diff("shared_a_2cta_local_a_cta vs sep", D_shared_local_cta, D_separate)
    print_split_diff("shared_a_2cta_local_a_cta vs sep", D_shared_local_cta, D_separate)
    print_tile_cos_matrix("shared_a_2cta_local_a_cta vs sep", D_shared_local_cta, D_separate)
    print_top_outliers("shared_a_2cta_local_a_cta vs sep", D_shared_local_cta, D_separate)


def run_both_bf16_shared_a_2cta_transport_debug_case(
    M: int = 256, K: int = 256, local_a_only: bool = False
) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-A 2CTA Transport Debug  M={M}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_a_2cta_transport_backend_label()}")
    print("  This is the A-only transport microkernel, not the full fused GEMM path")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    dump_cta0_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta1_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta0_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
    dump_cta1_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

    if not local_a_only:
        nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Transport microkernel dump checks (constant SCALE_MAX):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Transport microkernel dump checks (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_local_a_debug_dump(
        A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
    torch.cuda.synchronize()
    print("\n[-] Transport microkernel local-A control (constant SCALE_MAX):")
    compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
    compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    if not local_a_only:
        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_local_a_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Transport microkernel local-A control (CTA amax):")
    compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
    compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)


def run_both_bf16_shared_a_2cta_publish_only_local_a_case(M: int, N: int, K: int) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-A 2CTA Publish-Only Local-A Debug  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_a_2cta_publish_only_local_a_backend_label(N, K)}")
    print("  CTA0 still performs remote shared-A publish; CTA1 consumes locally quantized A")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    D_ref = torch.matmul(A_bf16, B_bf16.T).to(torch.bfloat16)

    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    A_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)

    B_fp4x2 = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    B_sc = torch.empty(N // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    B_sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)

    D_separate = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, D_separate)
    torch.cuda.synchronize()

    D_public = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_publish = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    D_publish_cta = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    def run_step(label, fn):
        print(f"\n[-] Launching {label}...")
        fn()
        torch.cuda.synchronize()

    run_step(
        "public both-bf16 constant",
        lambda: nvfp4_fused_gemm_both_bf16(A_bf16, B_bf16, D_public),
    )
    run_step(
        "publish-only local-A debug constant",
        lambda: nvfp4_fused_gemm_both_bf16_shared_a_2cta_publish_only_local_a_debug(A_bf16, B_bf16, D_publish),
    )
    run_step(
        "publish-only local-A debug CTA amax",
        lambda: nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_publish_only_local_a_debug(A_bf16, B_bf16, D_publish_cta),
    )

    print("\n[0] Public both-bf16 (constant) vs separate:")
    check_diff("public_both_const vs sep", D_public, D_separate)
    print("\n[1] Publish-only local-A debug (constant) vs separate:")
    check_diff("publish_only_local_a_const vs sep", D_publish, D_separate)
    print("\n[2] Publish-only local-A debug (CTA amax) vs separate:")
    check_diff("publish_only_local_a_cta vs sep", D_publish_cta, D_separate)
    print("\n[3] Publish-only local-A debug (constant) vs bf16 ref:")
    check_diff("publish_only_local_a_const vs ref", D_publish, D_ref)


def run_both_bf16_shared_a_2cta_transport_import_debug_case(
    M: int = 256, K: int = 256, local_a_only: bool = False
) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-A 2CTA Transport+Import Debug  M={M}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_a_2cta_transport_import_backend_label()}")
    print("  This checks receive-plus-import into the swizzled local A tile in isolation")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    dump_cta0_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta1_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta0_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
    dump_cta1_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

    if not local_a_only:
        nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Transport+import microkernel dump checks (constant SCALE_MAX):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Transport+import microkernel dump checks (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    nvfp4_fused_gemm_both_bf16_shared_a_2cta_transport_import_local_a_debug_dump(
        A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
    torch.cuda.synchronize()
    print("\n[-] Transport+import microkernel local-A control (constant SCALE_MAX):")
    compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
    compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    if not local_a_only:
        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_transport_import_local_a_debug_dump(
            A_bf16, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Transport+import microkernel local-A control (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)


def run_both_bf16_shared_a_2cta_full_transport_debug_case(
    M: int = 256, N: int = 512, K: int = 256, local_a_only: bool = False
) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-A 2CTA Full-Loop Transport Debug  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_a_2cta_full_transport_backend_label()}")
    print("  This checks the same shared-A receive/import path inside the real looped kernel")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dump_cta0_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta1_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta0_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
    dump_cta1_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

    if not local_a_only:
        nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_debug_dump(
            A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Full-loop transport dump checks (constant SCALE_MAX):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        D.zero_()
        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_debug_dump(
            A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Full-loop transport dump checks (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    D.zero_()
    nvfp4_fused_gemm_both_bf16_shared_a_2cta_full_transport_local_a_debug_dump(
        A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
    torch.cuda.synchronize()
    print("\n[-] Full-loop transport local-A control (constant SCALE_MAX):")
    compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
    compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    if not local_a_only:
        D.zero_()
        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_full_transport_local_a_debug_dump(
            A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Full-loop transport local-A control (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)


def run_both_bf16_shared_a_2cta_main_dump_case(
    M: int = 256, N: int = 512, K: int = 256, local_a_only: bool = False
) -> None:
    print(f"{'='*72}")
    print(f"  Both-BF16 Shared-A 2CTA Main-Kernel Dump  M={M}, N={N}, K={K}")
    print(f"  Experimental backend: {both_bf16_shared_a_2cta_backend_label(N, K)}")
    print("  This dumps the actual end-to-end shared-A kernel after its first-stage import")
    print(f"{'='*72}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dump_cta0_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta1_a = torch.empty((DEBUG_A_ROWS, DEBUG_A_COLS), dtype=torch.uint8, device="cuda")
    dump_cta0_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")
    dump_cta1_sc = torch.empty(DEBUG_A_SC_BYTES, dtype=torch.uint8, device="cuda")

    if not local_a_only:
        nvfp4_fused_gemm_both_bf16_shared_a_2cta_debug_dump(
            A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Main-kernel dump checks (constant SCALE_MAX):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

        D.zero_()
        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_debug_dump(
            A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Main-kernel dump checks (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    D.zero_()
    nvfp4_fused_gemm_both_bf16_shared_a_2cta_local_a_debug_dump(
        A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
    torch.cuda.synchronize()
    print("\n[-] Main-kernel local-A control (constant SCALE_MAX):")
    compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
    compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)

    if not local_a_only:
        D.zero_()
        nvfp4_fused_gemm_both_bf16_cta_amax_shared_a_2cta_local_a_debug_dump(
            A_bf16, B_bf16, D, dump_cta0_a, dump_cta1_a, dump_cta0_sc, dump_cta1_sc)
        torch.cuda.synchronize()
        print("\n[-] Main-kernel local-A control (CTA amax):")
        compare_byte_dump("A tile", dump_cta0_a, dump_cta1_a)
        compare_byte_dump("A scales", dump_cta0_sc, dump_cta1_sc)


if __name__ == '__main__':
    skip_cuda_preflight = os.environ.get("TK_SKIP_CUDA_PREFLIGHT") == "1"
    if not skip_cuda_preflight and not CUDA_READY:
        raise RuntimeError(
            "CUDA device unavailable. Run test_fused_gemm.py on a GPU-enabled B200 host."
        )
    load_ext()

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        for size in (256, 1024, 2048, 4096):
            run_case(size, size, size)
            print()
    elif len(sys.argv) > 1 and sys.argv[1] == "--shared-a-debug":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_shared_a_debug_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-quadcol":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
        run_both_bf16_quadcol_debug_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_debug_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-transport-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_transport_only_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-producer-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "Producer-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_producer_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_producer_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-b-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "B-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-wait-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Wait-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_wait_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_wait_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-b-wait-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "B-Wait-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_wait_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_wait_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-quant-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Quant-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-b-quant-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "B-Quant-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-stage1-quant-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Stage1-Quant-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_stage1_quant_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_stage1_quant_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-b-stage1-quant-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "B-Stage1-Quant-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_stage1_quant_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_stage1_quant_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-quant-per-stage-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Quant-Per-Stage-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_per_stage_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_per_stage_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-b-quant-per-stage-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "B-Quant-Per-Stage-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_per_stage_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_per_stage_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-quant-then-wait-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Quant-Then-Wait-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_wait_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_wait_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-b-quant-then-wait-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "B-Quant-Then-Wait-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_b_quant_then_wait_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_b_quant_then_wait_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-b-2cta-a-quant-then-skip-wait-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_b_2cta_split_case(
            "A-Quant-Then-Skip-Wait-Only",
            nvfp4_fused_gemm_both_bf16_shared_b_2cta_a_quant_then_skip_wait_only_debug,
            nvfp4_fused_gemm_both_bf16_cta_amax_shared_b_2cta_a_quant_then_skip_wait_only_debug,
            M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-cluster-2x2":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_cluster_2x2_debug_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-cluster-2x2-dump-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_cluster_2x2_debug_case(M, N, K, dump_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-cluster-2x2-transport-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        print(f"{'='*72}")
        print(f"  Both-BF16 Clustered 2x2 Transport-Only  M={M}, N={N}, K={K}")
        print(f"{'='*72}")
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K**0.25
        B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.25
        D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        print("[-] Launching clustered 2x2 transport-only constant kernel...")
        nvfp4_fused_gemm_both_bf16_cluster_2x2_transport_only_debug(A_bf16, B_bf16, D)
        torch.cuda.synchronize()
        print("[-] Launching clustered 2x2 transport-only CTA-amax kernel...")
        nvfp4_fused_gemm_both_bf16_cta_amax_cluster_2x2_transport_only_debug(A_bf16, B_bf16, D)
        torch.cuda.synchronize()
        print("Transport-only clustered 2x2 kernels completed.")
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
        run_both_bf16_shared_a_2cta_debug_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-dump-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_debug_case(M, N, K, dump_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-local-a-dump-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_debug_case(M, N, K, dump_only=True, local_a_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-publish-only-local-a":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_publish_only_local_a_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-transport-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        run_both_bf16_shared_a_2cta_transport_debug_case(M, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-transport-local-a-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        run_both_bf16_shared_a_2cta_transport_debug_case(M, K, local_a_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-transport-import-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        run_both_bf16_shared_a_2cta_transport_import_debug_case(M, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-transport-import-local-a-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
        run_both_bf16_shared_a_2cta_transport_import_debug_case(M, K, local_a_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-full-transport-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_full_transport_debug_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-full-transport-local-a-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_full_transport_debug_case(M, N, K, local_a_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-main-dump-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_main_dump_case(M, N, K)
    elif len(sys.argv) > 1 and sys.argv[1] == "--both-bf16-shared-a-2cta-main-local-a-dump-only":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 256
        N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        run_both_bf16_shared_a_2cta_main_dump_case(M, N, K, local_a_only=True)
    else:
        M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        K = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
        run_case(M, N, K)
