from pathlib import Path


TK_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KERNELS = 18
CLUSTER_LIFETIME_HEADERS = {
    "kernels/gemm/mxfp4_gb200/mxfp4_atb_gemm.cuh": 1,
    "kernels/gemm/mxfp4_gb200/mxfp4_batched_gemm.cuh": 1,
    "kernels/gemm/mxfp4_gb200/mxfp4_gemm.cuh": 1,
    "kernels/gemm/mxfp4_gb200/mxfp4_silu_dgrad_quant_gemm.cuh": 1,
    "kernels/gemm/mxfp4_gb200/mxfp4_split2_accum_gemm.cuh": 1,
    "kernels/gemm/mxfp4_gb200/mxfp4_split3_accum_gemm.cuh": 1,
    "kernels/gemm/nvfp4_b200/nvfp4_accum_gemm.cuh": 1,
    "kernels/gemm/nvfp4_b200/nvfp4_batched_accum_gemm.cuh": 1,
    "kernels/gemm/nvfp4_b200/nvfp4_batched_gemm.cuh": 1,
    "kernels/gemm/nvfp4_b200/nvfp4_gemm.cuh": 2,
    "kernels/gemm/nvfp4_b200/nvfp4_split2_accum_gemm.cuh": 2,
    "kernels/gemm/nvfp4_b200/nvfp4_split3_accum_gemm.cuh": 1,
    (
        "kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
        "nvfp4_localcta_batched_kernel.cuh"
    ): 1,
    (
        "kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
        "nvfp4_localcta_kernel.cuh"
    ): 2,
    (
        "kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
        "nvfp4_localcta_silu_dgrad_quant_gemm.cuh"
    ): 1,
}


def test_clustered_gemms_have_complete_entry_and_exit_phases() -> None:
    """Keep every warp and clustered CTA live across the full kernel."""

    total = 0
    for relative_path, expected in CLUSTER_LIFETIME_HEADERS.items():
        path = TK_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        counts = {
            "entry_arrive": text.count(
                "everyone::tma::cluster::arrive_aligned();"
            ),
            "entry_wait": text.count(
                "everyone::tma::cluster::wait_aligned();"
            ),
            "role_wait": text.count("everyone::tma::cluster::wait();"),
            "exit_arrive": text.count(
                "barrier.cluster.arrive.relaxed.aligned;"
            ),
            "exit_wait": text.count("barrier.cluster.wait.aligned;"),
        }
        assert counts == {
            "entry_arrive": expected,
            "entry_wait": expected,
            "role_wait": 0,
            "exit_arrive": expected,
            "exit_wait": expected,
        }, path
        total += expected

    assert total == EXPECTED_KERNELS
