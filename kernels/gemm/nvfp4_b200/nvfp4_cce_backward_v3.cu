// NVFP4 CCE Backward v2 — Fused softmax gradient + optional FP4 quantization
#include "nvfp4_cce_backward_v3.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

// BF16 mode: outputs BF16 grad_logits (identical interface to v1)
template <typename C>
static void launch_backward_v3_bf16(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    static_assert(C::USE_BF16_ACCUM, "Must use BF16 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;

    // Create dummy tensors for unused FP4 output fields
    auto dummy_fp4 = A.new_empty({1, 1}, A.options().dtype(c10::kByte));
    auto dummy_sc  = A.new_empty({1, 1}, A.options().dtype(c10::kHalf));
    auto dummy_sg  = A.new_empty({1}, A.options().dtype(c10::kFloat));

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
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(dummy_fp4),
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
        .M = M, .N = N,
        .encode_centric = false
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3<C>>(g);
}

// FP4 mode: outputs quantized G in NVFP4 format for separate GEMM calls
template <typename C>
static void launch_backward_v3_fp4(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row, at::Tensor &G_sg_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f,
    bool encode_centric = false)
{
    static_assert(!C::USE_BF16_ACCUM, "Must use FP4 config for this function");
    using G = nvfp4_cce_backward_v3::globals<C>;

    // Dummy BF16 output — D_out is unused in FP4 mode but TMA needs valid tensor
    auto dummy_bf16 = A.new_empty({M, N}, A.options().dtype(c10::kBFloat16));

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
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(dummy_bf16),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(G_fp4_row),
        .G_sc_row_ptr = reinterpret_cast<uint8_t*>(G_sc_row.data_ptr()),
        .G_sc_row_kgroups = N / 64,  // V/64 k-groups for row-quant scale
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(G_sg_row),
        .G_fp4_col_ptr = reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        .G_sc_col_ptr = reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        .G_sc_col_kgroups = M / 64,  // M/64 k-groups for col-quant scale
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N,
        .encode_centric = encode_centric
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v3::backward_kernel_v3<C>>(g);
}

// Config instantiations
using bwd_v3_bf16_L4_SG8 = nvfp4_cce_backward_v3::config<4, 8, true, true>;
using bwd_v3_fp4_L4_SG8  = nvfp4_cce_backward_v3::config<4, 8, false, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v3_bf16_L4_SG8", &launch_backward_v3_bf16<bwd_v3_bf16_L4_SG8>,
          "NVFP4 CCE backward v2 (BF16 output) L4 SG8");
    m.def("backward_v3_fp4_L4_SG8", &launch_backward_v3_fp4<bwd_v3_fp4_L4_SG8>,
          "NVFP4 CCE backward v2 (FP4 output) L4 SG8");
}
#endif
