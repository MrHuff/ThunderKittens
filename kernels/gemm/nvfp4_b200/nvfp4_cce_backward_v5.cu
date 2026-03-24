// NVFP4 CCE Backward v5 — owner-flipped fused dE pass
#include "nvfp4_cce_backward_v5_dE.cuh"

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

using bwd_v5_dE_fp4_L4_SG8 = nvfp4_cce_backward_v5_dE::config<4, 8, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v5_dE_fp4_L4_SG8", &launch_backward_v5_dE<bwd_v5_dE_fp4_L4_SG8>,
          "NVFP4 CCE backward v5 fused dE pass L4 SG8");
}
#endif
