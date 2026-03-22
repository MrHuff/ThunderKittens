// NVFP4 CCE Backward v4 — Fully Fused One-Pass (Algorithm 3) MVP
#include "nvfp4_cce_backward_v4.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

// BF16 mode: outputs BF16 grad_logits AND accumulates dE via fused Phase 3
// For MVP: just BF16 output (same as v1), Phase 3 dE GEMM is TODO
template <typename C>
static void launch_backward_v4_bf16(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = nvfp4_cce_backward_v4::globals<C>;

    // Dummies must be >= SMEM tile size for TMA descriptor creation
    // P3_B_fp4x2_tile = st_fp4e2m1_2<Nb_out/2, Nb/2> = <64, 64>
    auto dummy_fp4_ccol = A.new_empty({C::Nb_out/2, C::Nb/2}, A.options().dtype(at::kFloat4_e2m1fn_x2));
    // G_fp4_row_tile = st_fp4e2m1_2<Mb/2, Nb/2> = <128, 64>
    auto dummy_fp4_grow = A.new_empty({C::Mb/2, C::Nb/2}, A.options().dtype(at::kFloat4_e2m1fn_x2));
    auto dummy_sc  = A.new_empty({4, 256}, A.options().dtype(c10::kHalf));
    auto dummy_sg  = A.new_empty({1}, A.options().dtype(c10::kFloat));
    // dE_tile = st_bf<Mb/2, Nb_out> = <128, 128>
    auto dummy_dE  = A.new_empty({C::Mb/2, C::Nb_out}, A.options().dtype(c10::kBFloat16));

    int K = 0;  // TEMP: disable Phase 3 K-loop for testing Phase 2b

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        // Phase 3 C_col: use dummy for BF16 mode
        .C_col = kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(dummy_fp4_ccol),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(dummy_sc, 1, 1, 1, 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(dummy_sg),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .dE_out = kittens::py::tensor_to_gl<typename G::dE_gl>(dummy_dE),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(dummy_fp4_grow),
        .G_sc_row_ptr = nullptr,
        .G_sc_row_kgroups = 1,
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(dummy_sg),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 1,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N, .K = K,
        .encode_centric = false
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v4::backward_kernel_v4<C>>(g);
}

// Config instantiations — BF16 only for MVP (outputs BF16 G for separate dC GEMM)
using bwd_v4_bf16_L4_SG8 = nvfp4_cce_backward_v4::config<4, 8, true, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v4_bf16_L4_SG8", &launch_backward_v4_bf16<bwd_v4_bf16_L4_SG8>,
          "NVFP4 CCE backward v4 (BF16+dE fused) L4 SG8");
}
#endif
