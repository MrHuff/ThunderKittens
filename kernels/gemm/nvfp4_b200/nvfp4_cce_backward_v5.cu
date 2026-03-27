// NVFP4 CCE Backward v5 — owner-flipped fused passes
#include "nvfp4_cce_backward_v5_dE.cuh"
#include "nvfp4_cce_backward_v5_dC.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

template <typename C>
static void launch_backward_v5_dE(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    at::Tensor &dE_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dE::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .C_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(C_col),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            C_col_sc, 1, C_col_sc.size(0), C_col_sc.size(1), 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(C_col_sc_global),
        .dE_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dE_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dE::kernel<C>>(g);
}

using bwd_v5_dE_fp4_L4_SG8 = nvfp4_cce_backward_v5_dE::config<2, 8, true>;
using bwd_v5_dC_fp4_L4_SG8 = nvfp4_cce_backward_v5_dC::config<2, 8, true>;

template <typename C>
static void launch_backward_v5_dC(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::kernel<C>>(g);
}

template <typename C>
static void launch_debug_backward_v5_dC_stage(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out,
    at::Tensor &debug_gt_fp4, at::Tensor &debug_gt_sc,
    at::Tensor &debug_p3_b_fp4, at::Tensor &debug_p3_b_sc, at::Tensor &debug_p3_out,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC::debug_globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .debug_p3_b_fp4_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_fp4.data_ptr()),
        .debug_p3_b_sc_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_sc.data_ptr()),
        .debug_gt_fp4_ptr = reinterpret_cast<uint8_t*>(debug_gt_fp4.data_ptr()),
        .debug_gt_sc_ptr = reinterpret_cast<uint8_t*>(debug_gt_sc.data_ptr()),
        .debug_p3_out_ptr = reinterpret_cast<bf16*>(debug_p3_out.data_ptr()),
        .debug_p3_out_raw_ptr = nullptr,
        .debug_p3_b_fp4_stride = static_cast<int>(debug_p3_b_fp4.size(1)),
        .debug_gt_fp4_stride = static_cast<int>(debug_gt_fp4.size(1)),
        .debug_p3_out_stride = static_cast<int>(debug_p3_out.size(1)),
        .debug_p3_out_raw_stride = 0,
        .debug_trace_mode = 0,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::debug_kernel<C>>(g);
}

template <typename C>
static void launch_debug_backward_v5_dC_p3_probe(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    const at::Tensor &E_col, const at::Tensor &E_col_sc, const at::Tensor &E_col_sc_global,
    at::Tensor &dC_out,
    at::Tensor &debug_gt_fp4, at::Tensor &debug_gt_sc,
    at::Tensor &debug_gt_sc_cluster, at::Tensor &debug_p3_a_sc_chunks,
    at::Tensor &debug_p3_b_fp4, at::Tensor &debug_p3_b_sc,
    at::Tensor &debug_p3_out_raw, at::Tensor &debug_p3_out,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, int K, int debug_trace_mode, int debug_p3_contract, int debug_b_payload_source, int debug_b_sc_source,
    float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v5_dC::debug_globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim() == 2 ? A_sc.size(0) / 128 : A_sc.size(0),
            A_sc.dim() == 2 ? A_sc.size(1) / 4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim() == 2 ? B_sc.size(0) / 128 : B_sc.size(0),
            B_sc.dim() == 2 ? B_sc.size(1) / 4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .E_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(E_col),
        .E_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            E_col_sc, 1, E_col_sc.size(0), E_col_sc.size(1), 256),
        .E_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(E_col_sc_global),
        .dC_out = kittens::py::tensor_to_gl<typename G::Out_gl>(dC_out),
        .debug_p3_b_fp4_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_fp4.data_ptr()),
        .debug_p3_b_sc_ptr = reinterpret_cast<uint8_t*>(debug_p3_b_sc.data_ptr()),
        .debug_gt_fp4_ptr = reinterpret_cast<uint8_t*>(debug_gt_fp4.data_ptr()),
        .debug_gt_sc_ptr = reinterpret_cast<uint8_t*>(debug_gt_sc.data_ptr()),
        .debug_p3_out_ptr = reinterpret_cast<bf16*>(debug_p3_out.data_ptr()),
        .debug_p3_out_raw_ptr = reinterpret_cast<float*>(debug_p3_out_raw.data_ptr()),
        .debug_p3_b_fp4_stride = static_cast<int>(debug_p3_b_fp4.size(1)),
        .debug_gt_fp4_stride = static_cast<int>(debug_gt_fp4.size(1)),
        .debug_p3_out_stride = static_cast<int>(debug_p3_out.size(1)),
        .debug_p3_out_raw_stride = static_cast<int>(debug_p3_out_raw.size(1)),
        .debug_trace_mode = debug_trace_mode,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M,
        .N = N,
        .K = K,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::debug_kernel<C>>(g);
}

template <typename C>
static void launch_debug_backward_v5_dC_mm2_probe(
    const at::Tensor &gt_fp4,
    const at::Tensor &gt_sc,
    const at::Tensor &p3_b_fp4,
    const at::Tensor &p3_b_sc,
    at::Tensor &debug_p3_out_raw,
    at::Tensor &debug_p3_out,
    float gt_sg,
    float p3_b_sg,
    int probe_mode)
{
    using G = nvfp4_cce_backward_v5_dC::p3_probe_globals<C>;
    G g {
        .gt_fp4_ptr = reinterpret_cast<const uint8_t*>(gt_fp4.data_ptr()),
        .gt_sc_ptr = reinterpret_cast<const uint8_t*>(gt_sc.data_ptr()),
        .p3_b_fp4_ptr = reinterpret_cast<const uint8_t*>(p3_b_fp4.data_ptr()),
        .p3_b_sc_ptr = reinterpret_cast<const uint8_t*>(p3_b_sc.data_ptr()),
        .gt_sg = gt_sg,
        .p3_b_sg = p3_b_sg,
        .out_bf_ptr = reinterpret_cast<bf16*>(debug_p3_out.data_ptr()),
        .out_raw_ptr = reinterpret_cast<float*>(debug_p3_out_raw.data_ptr()),
        .gt_fp4_stride = static_cast<int>(gt_fp4.size(1)),
        .p3_b_fp4_stride = static_cast<int>(p3_b_fp4.size(1)),
        .out_stride = static_cast<int>(debug_p3_out.size(1)),
        .out_raw_stride = static_cast<int>(debug_p3_out_raw.size(1)),
        .probe_mode = probe_mode,
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v5_dC::p3_probe_kernel<C>>(g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v5_dE_fp4_L4_SG8", &launch_backward_v5_dE<bwd_v5_dE_fp4_L4_SG8>,
          "NVFP4 CCE backward v5 fused dE pass L4 SG8");
    m.def("backward_v5_dC_fp4_L4_SG8", &launch_backward_v5_dC<bwd_v5_dC_fp4_L4_SG8>,
          "NVFP4 CCE backward v5 fused dC pass L4 SG8");
    m.def("debug_v5_dC_stage_fp4_L4_SG8", &launch_debug_backward_v5_dC_stage<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC stage dump L4 SG8");
    m.def("debug_v5_dC_p3_probe_fp4_L4_SG8", &launch_debug_backward_v5_dC_p3_probe<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC phase-3 contract probe L4 SG8");
    m.def("debug_v5_dC_trace_fp4_L4_SG8", &launch_debug_backward_v5_dC_p3_probe<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only NVFP4 CCE v5 dC tiny trace L4 SG8");
    m.def("debug_v5_dC_mm2_probe_fp4_L4_SG8", &launch_debug_backward_v5_dC_mm2_probe<bwd_v5_dC_fp4_L4_SG8>,
          "Developer-only isolated NVFP4 dC phase-3 mm2 probe L4 SG8");
}
#endif
