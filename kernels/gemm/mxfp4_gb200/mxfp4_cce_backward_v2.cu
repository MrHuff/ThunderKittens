// MXFP4 CCE Backward v2 — Fused softmax gradient + optional FP4 quantization
#include "mxfp4_cce_backward_v2.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

// BF16 mode: outputs BF16 grad_logits
template <typename C>
static void launch_backward_v2_bf16(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    const auto padded_M = A.size(0);
    const auto padded_N = B.size(0);
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(
            A.new_empty({padded_M, padded_N / 2}, A.options().dtype(c10::kFloat4_e2m1fn_x2))),
        .G_sc_row = nullptr,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

// FP4 mode: outputs quantized G in MXFP4 format
template <typename C>
static void launch_backward_v2_fp4(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            A.new_empty({1, 1}, A.options().dtype(c10::kBFloat16))),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(G_fp4_row),
        .G_sc_row = G_sc_row.data_ptr<uint8_t>(),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

// Config instantiations
using bwd_v2_bf16_L5_SG8 = mxfp4_cce_backward_v2::config<5, 8, true, true>;
using bwd_v2_bf16_L4_SG8 = mxfp4_cce_backward_v2::config<4, 8, true, true>;
using bwd_v2_fp4_L5_SG8  = mxfp4_cce_backward_v2::config<5, 8, false, true>;
using bwd_v2_fp4_L4_SG8  = mxfp4_cce_backward_v2::config<4, 8, false, true>;
// Encode-centric: ceil E8M0 exponent (QUANT_MODE=1)
using bwd_v2_fp4_enc_L4_SG8 = mxfp4_cce_backward_v2::config<4, 8, false, true, 1>;
// Decode-centric: floor E8M0 exponent (QUANT_MODE=2)
using bwd_v2_fp4_dec_L4_SG8 = mxfp4_cce_backward_v2::config<4, 8, false, true, 2>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v2_bf16_L5_SG8", &launch_backward_v2_bf16<bwd_v2_bf16_L5_SG8>,
          "MXFP4 CCE backward v2 (BF16 output) L5 SG8");
    m.def("backward_v2_bf16_L4_SG8", &launch_backward_v2_bf16<bwd_v2_bf16_L4_SG8>,
          "MXFP4 CCE backward v2 (BF16 output) L4 SG8");
    m.def("backward_v2_fp4_L5_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_L5_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE output) L5 SG8");
    m.def("backward_v2_fp4_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE output) L4 SG8");
    m.def("backward_v2_fp4_enc_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_enc_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric output) L4 SG8");
    m.def("backward_v2_fp4_dec_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_dec_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric output) L4 SG8");
}
#endif
