#include <torch/extension.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <optional>
#include <tuple>
#include <vector>

#include "../../../../../TK_quantisation/nvfp4_CTA_local_v1/localcta_reconstruct.cuh"

namespace py = pybind11;

namespace {

torch::Tensor dequantize_rowwise(
    const at::Tensor &fp4,
    const at::Tensor &sc,
    const at::Tensor &sg_chunks
) {
    TORCH_CHECK(fp4.is_cuda() && sc.is_cuda() && sg_chunks.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(fp4.scalar_type() == torch::kFloat4_e2m1fn_x2, "fp4 dtype mismatch");
    TORCH_CHECK(sc.scalar_type() == torch::kFloat8_e4m3fn, "scale dtype mismatch");
    TORCH_CHECK(sg_chunks.scalar_type() == torch::kFloat32, "sg dtype mismatch");

    const int rows = (int)fp4.size(0);
    const int cols = (int)fp4.size(1) * 2;
    auto out = torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(fp4.device()));
    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t numel = (int64_t)rows * cols;
    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);
    tk_localcta_reconstruct::reconstruct_rowwise_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp4x2_e2m1*>(fp4.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(sc.data_ptr()),
        sg_chunks.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        rows, cols, (int)sg_chunks.size(0), (int)sg_chunks.size(1));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dequantize_rowwise failed: ", cudaGetErrorString(err));
    return out;
}

void apply_silu_prefix(at::Tensor &x, int silu_dim) {
    if (silu_dim <= 0) {
        return;
    }
    TORCH_CHECK(silu_dim <= x.size(1), "silu_dim exceeds output width");
    auto slice = x.narrow(1, 0, silu_dim);
    slice.copy_(slice * at::sigmoid(slice));
}

at::Tensor matmul_localcta(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sg_chunks,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_chunks
) {
    auto A_bf16 = dequantize_rowwise(A, A_sc, A_sg_chunks);
    auto B_bf16 = dequantize_rowwise(B, B_sc, B_sg_chunks);
    return at::matmul(A_bf16, B_bf16.transpose(0, 1));
}

void check_gemm_shapes(const at::Tensor &A,
                       const at::Tensor &A_sc,
                       const at::Tensor &A_sg_chunks,
                       const at::Tensor &B,
                       const at::Tensor &B_sc,
                       const at::Tensor &B_sg_chunks) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A_sc.dim() == 3 && B_sc.dim() == 3, "A_sc and B_sc must be 3D");
    TORCH_CHECK(A_sg_chunks.dim() == 2 && B_sg_chunks.dim() == 2,
                "chunk scale grids must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share K/2 width");
    TORCH_CHECK(A.size(0) % 128 == 0 && B.size(0) % 128 == 0, "M and N must be multiples of 128");
    TORCH_CHECK(A.size(1) % 64 == 0, "K/2 must be multiple of 64");
    TORCH_CHECK(A_sg_chunks.size(0) == A.size(0) / 128 &&
                A_sg_chunks.size(1) == (A.size(1) * 2) / 128,
                "A chunk grid shape mismatch");
    TORCH_CHECK(B_sg_chunks.size(0) == B.size(0) / 128 &&
                B_sg_chunks.size(1) == (B.size(1) * 2) / 128,
                "B chunk grid shape mismatch");
}

}  // namespace

void nvfp4_localcta_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sg_chunks,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_chunks,
    at::Tensor &D
) {
    check_gemm_shapes(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    auto result = matmul_localcta(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    D.copy_(result);
}

void nvfp4_localcta_grouped_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &A_sg_chunks,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &B_sg_chunks,
    at::Tensor &D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    check_gemm_shapes(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    auto result = matmul_localcta(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    apply_silu_prefix(result, silu_dim);

    if (!D_K_opt.has_value()) {
        D.copy_(result);
        return;
    }

    const int q_dim = (int)D.size(1);
    const int k_dim = (int)D_K_opt.value().size(1);
    const int v_dim = D_V_opt.has_value() ? (int)D_V_opt.value().size(1) : 0;
    TORCH_CHECK(q_dim + k_dim + v_dim == result.size(1),
                "split outputs do not cover grouped output width");

    D.copy_(result.narrow(1, 0, q_dim));
    D_K_opt.value().copy_(result.narrow(1, q_dim, k_dim));
    if (D_V_opt.has_value()) {
        D_V_opt.value().copy_(result.narrow(1, q_dim + k_dim, v_dim));
    }
}

void nvfp4_localcta_batched_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_chunks_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_chunks_list,
    std::vector<at::Tensor> &D_list
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n == (int)A_sc_list.size() &&
                n == (int)A_sg_chunks_list.size() &&
                n == (int)B_list.size() &&
                n == (int)B_sc_list.size() &&
                n == (int)B_sg_chunks_list.size() &&
                n == (int)D_list.size(),
                "batched input list sizes must match");

    for (int i = 0; i < n; ++i) {
        nvfp4_localcta_gemm_entrypoint(
            A_list[i], A_sc_list[i], A_sg_chunks_list[i],
            B_list[i], B_sc_list[i], B_sg_chunks_list[i],
            D_list[i]);
    }
}

void nvfp4_localcta_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor> &A_list,
    const std::vector<at::Tensor> &A_sc_list,
    const std::vector<at::Tensor> &A_sg_chunks_list,
    const std::vector<at::Tensor> &B_list,
    const std::vector<at::Tensor> &B_sc_list,
    const std::vector<at::Tensor> &B_sg_chunks_list,
    at::Tensor &D_out
) {
    const int n = (int)A_list.size();
    TORCH_CHECK(n > 0, "need at least one batch");

    auto accum = torch::zeros_like(D_out);
    for (int i = 0; i < n; ++i) {
        auto result = matmul_localcta(
            A_list[i], A_sc_list[i], A_sg_chunks_list[i],
            B_list[i], B_sc_list[i], B_sg_chunks_list[i]);
        accum.add_(result);
    }
    D_out.copy_(accum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_localcta_gemm", &nvfp4_localcta_gemm_entrypoint,
          py::arg("A"), py::arg("A_sc"), py::arg("A_sg_chunks"),
          py::arg("B"), py::arg("B_sc"), py::arg("B_sg_chunks"),
          py::arg("D"));
    m.def("nvfp4_localcta_grouped_gemm", &nvfp4_localcta_grouped_gemm_entrypoint,
          py::arg("A"), py::arg("A_sc"), py::arg("A_sg_chunks"),
          py::arg("B"), py::arg("B_sc"), py::arg("B_sg_chunks"),
          py::arg("D"), py::arg("D_K") = std::nullopt, py::arg("D_V") = std::nullopt,
          py::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_batched_gemm", &nvfp4_localcta_batched_gemm_entrypoint,
          py::arg("A_list"), py::arg("A_sc_list"), py::arg("A_sg_chunks_list"),
          py::arg("B_list"), py::arg("B_sc_list"), py::arg("B_sg_chunks_list"),
          py::arg("D_list"));
    m.def("nvfp4_localcta_batched_accum_gemm", &nvfp4_localcta_batched_accum_gemm_entrypoint,
          py::arg("A_list"), py::arg("A_sc_list"), py::arg("A_sg_chunks_list"),
          py::arg("B_list"), py::arg("B_sc_list"), py::arg("B_sg_chunks_list"),
          py::arg("D_out"));
}
