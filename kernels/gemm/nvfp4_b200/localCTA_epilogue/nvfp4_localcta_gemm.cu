#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

#include "pyutils/torchutils.cuh"
#include "nvfp4_localcta_kernel.cuh"
#include "nvfp4_localcta_batched_kernel.cuh"

namespace {

constexpr int kScaleBytesPerTile = 512;
using localcta_regular_smalln_config = nvfp4_localcta_gemm::config<128, 5, 4, 12, 2, true, 256, true, 2, 128>;
using localcta_regular_smallk_config = nvfp4_localcta_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;
using localcta_regular_largek_config = nvfp4_localcta_gemm::config<256, 5, 8, 12, 2, false, 256, true, 2, 128>;
using localcta_parity_config = nvfp4_localcta_gemm::config<256, 5, 8, 4, 2, false, 256, true, 2, 128>;

void check_fp4_matrix(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat4_e2m1fn_x2, name, " must be fp4x2");
}

void check_scale_tensor(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 3, name, " must be 3D");
    TORCH_CHECK(t.scalar_type() == at::kFloat8_e4m3fn, name, " must be fp8 e4m3");
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 64, name, " second dim mismatch");
    TORCH_CHECK(t.size(2) == kScaleBytesPerTile, name, " third dim must be 512");
}

void check_chunk_grid(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.size(0) == rows / 128, name, " first dim mismatch");
    TORCH_CHECK(t.size(1) == cols / 128, name, " second dim mismatch");
}

void check_output_matrix(const at::Tensor& t, const char* name, int64_t rows, int64_t cols) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
}

void check_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks
) {
    check_fp4_matrix(A, "A");
    check_fp4_matrix(B, "B");
    TORCH_CHECK(A.size(0) % 128 == 0 && B.size(0) % 128 == 0, "M and N must be multiples of 128");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share packed K");
    TORCH_CHECK((A.size(1) * 2) % 128 == 0, "K must be a multiple of 128");
    check_scale_tensor(A_sc, "A_sc", A.size(0), A.size(1) * 2);
    check_scale_tensor(B_sc, "B_sc", B.size(0), B.size(1) * 2);
    check_chunk_grid(A_sg_chunks, "A_sg_chunks", A.size(0), A.size(1) * 2);
    check_chunk_grid(B_sg_chunks, "B_sg_chunks", B.size(0), B.size(1) * 2);
    kittens::py::device_check(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
}

template <typename C>
void launch_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    using G = nvfp4_localcta_gemm::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sg_chunks = A_sg_chunks.data_ptr<float>(),
        .A_sg_stride = static_cast<int>(A_sg_chunks.size(1)),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sg_chunks = B_sg_chunks.data_ptr<float>(),
        .B_sg_stride = static_cast<int>(B_sg_chunks.size(1)),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .q_dim = 0,
        .k_dim = 0,
        .v_dim = 0,
        .use_split_D = false,
        .silu_dim = 0
    };
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel<C>>(g);
}

template <typename C>
void launch_grouped_gemm_with_config(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    using G = nvfp4_localcta_gemm::globals<C>;
    const bool use_split_D = D_K_opt.has_value();
    const int v_dim = D_V_opt.has_value() ? static_cast<int>(D_V_opt.value().size(1)) : 0;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1, A_sc.size(0), A_sc.size(1), 256),
        .A_sg_chunks = A_sg_chunks.data_ptr<float>(),
        .A_sg_stride = static_cast<int>(A_sg_chunks.size(1)),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1, B_sc.size(0), B_sc.size(1), 256),
        .B_sg_chunks = B_sg_chunks.data_ptr<float>(),
        .B_sg_stride = static_cast<int>(B_sg_chunks.size(1)),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_K = use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                           : kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_V = D_V_opt.has_value() ? kittens::py::tensor_to_gl<typename G::D_gl>(D_V_opt.value())
                                   : (use_split_D ? kittens::py::tensor_to_gl<typename G::D_gl>(D_K_opt.value())
                                                  : kittens::py::tensor_to_gl<typename G::D_gl>(D)),
        .q_dim = use_split_D ? static_cast<int>(D.size(1)) : 0,
        .k_dim = use_split_D ? static_cast<int>(D_K_opt.value().size(1)) : 0,
        .v_dim = use_split_D ? v_dim : 0,
        .use_split_D = use_split_D,
        .silu_dim = silu_dim
    };
    kittens::py::launch_kernel<C, G, nvfp4_localcta_gemm::kernel<C>>(g);
}

void launch_regular_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    const int64_t M = D.size(0);
    const int64_t K = A.size(1) * 2;
    const int64_t N = D.size(1);
    if (K <= 2048 && M <= 1024 && N <= 2048) {
        launch_gemm_with_config<localcta_regular_smalln_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    } else if (K <= 2048) {
        launch_gemm_with_config<localcta_regular_smallk_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    } else {
        launch_gemm_with_config<localcta_regular_largek_config>(
            A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
    }
}

void launch_grouped_gemm(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt,
    std::optional<at::Tensor> D_V_opt,
    int silu_dim
) {
    launch_grouped_gemm_with_config<localcta_parity_config>(
        A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks,
        D, D_K_opt, D_V_opt, silu_dim);
}

void check_batched_inputs(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list
) {
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n > 0, "batched GEMM requires at least one batch");
    TORCH_CHECK(n == static_cast<int>(A_sc_list.size()) &&
                n == static_cast<int>(A_sg_chunks_list.size()) &&
                n == static_cast<int>(B_list.size()) &&
                n == static_cast<int>(B_sc_list.size()) &&
                n == static_cast<int>(B_sg_chunks_list.size()),
                "all batched input lists must have the same length");
    for (int i = 0; i < n; ++i) {
        check_gemm_inputs(
            A_list[i], A_sc_list[i], A_sg_chunks_list[i],
            B_list[i], B_sc_list[i], B_sg_chunks_list[i]);
    }
}

template <typename C>
void launch_batched_gemm_with_config(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    using G = nvfp4_localcta_batched_gemm::globals<C>;
    G g_host;
    memset(&g_host, 0, sizeof(G));

    const int n = static_cast<int>(A_list.size());
    const int64_t M = D_list[0].size(0);
    const int64_t N_out = D_list[0].size(1);

    g_host.num_batches = n;
    g_host.num_row_blocks = static_cast<int>(M / C::Mb);
    g_host.num_col_blocks = static_cast<int>(N_out / C::Nb);
    g_host.num_red_blocks = static_cast<int>((2 * A_list[0].size(1)) / C::Kb);

    for (int i = 0; i < n; ++i) {
        auto a_gl = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A_list[i]);
        auto a_sc_gl = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc_list[i], 1, A_sc_list[i].size(0), A_sc_list[i].size(1), 256);
        auto b_gl = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B_list[i]);
        auto b_sc_gl = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc_list[i], 1, B_sc_list[i].size(0), B_sc_list[i].size(1), 256);
        auto d_gl = kittens::py::tensor_to_gl<typename G::D_gl>(D_list[i]);

        memcpy(&g_host.A_tma[i], &a_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.A_sc_tma[i], &a_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_tma[i], &b_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.B_sc_tma[i], &b_sc_gl.tma_descs.tma_desc, sizeof(CUtensorMap));
        memcpy(&g_host.D_tma[i], &d_gl.tma_descs.tma_desc, sizeof(CUtensorMap));

        g_host.A_sg_chunks[i] = A_sg_chunks_list[i].data_ptr<float>();
        g_host.A_sg_stride[i] = static_cast<int>(A_sg_chunks_list[i].size(1));
        g_host.B_sg_chunks[i] = B_sg_chunks_list[i].data_ptr<float>();
        g_host.B_sg_stride[i] = static_cast<int>(B_sg_chunks_list[i].size(1));
    }

    kittens::py::launch_kernel<C, G, nvfp4_localcta_batched_gemm::kernel<C>>(g_host);
}

void launch_batched_gemm(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    launch_batched_gemm_with_config<localcta_parity_config>(
        A_list, A_sc_list, A_sg_chunks_list, B_list, B_sc_list, B_sg_chunks_list, D_list);
}

__global__ void sum_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(A[idx], B[idx]);
    }
}

__global__ void sum3_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), C[idx]);
    }
}

__global__ void sum4_tensors_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ D,
    __nv_bfloat16* __restrict__ out,
    int64_t numel
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __hadd(__hadd(A[idx], B[idx]), __hadd(C[idx], D[idx]));
    }
}

}  // namespace

void nvfp4_localcta_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D
) {
    check_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    check_output_matrix(D, "D", A.size(0), B.size(0));
    launch_regular_gemm(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks, D);
}

void nvfp4_localcta_grouped_gemm_entrypoint(
    const at::Tensor& A,
    const at::Tensor& A_sc,
    const at::Tensor& A_sg_chunks,
    const at::Tensor& B,
    const at::Tensor& B_sc,
    const at::Tensor& B_sg_chunks,
    at::Tensor& D,
    std::optional<at::Tensor> D_K_opt = std::nullopt,
    std::optional<at::Tensor> D_V_opt = std::nullopt,
    int silu_dim = 0
) {
    check_gemm_inputs(A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks);
    TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kBFloat16,
                "D must be a contiguous CUDA bf16 tensor");
    if (D_K_opt.has_value()) {
        TORCH_CHECK(D_K_opt.value().is_cuda() && D_K_opt.value().is_contiguous() &&
                    D_K_opt.value().scalar_type() == at::kBFloat16,
                    "D_K must be a contiguous CUDA bf16 tensor");
    }
    if (D_V_opt.has_value()) {
        TORCH_CHECK(D_V_opt.value().is_cuda() && D_V_opt.value().is_contiguous() &&
                    D_V_opt.value().scalar_type() == at::kBFloat16,
                    "D_V must be a contiguous CUDA bf16 tensor");
    }

    launch_grouped_gemm(
        A, A_sc, A_sg_chunks, B, B_sc, B_sg_chunks,
        D, D_K_opt, D_V_opt, silu_dim);
}

void nvfp4_localcta_batched_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    std::vector<at::Tensor>& D_list
) {
    check_batched_inputs(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list);

    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= nvfp4_localcta_batched_gemm::MAX_BATCHES,
                "num_batches must be 1..", nvfp4_localcta_batched_gemm::MAX_BATCHES);
    TORCH_CHECK(n == static_cast<int>(D_list.size()),
                "D_list length must match number of batches");

    for (int i = 0; i < n; ++i) {
        check_output_matrix(D_list[i], "D_list[i]", A_list[i].size(0), B_list[i].size(0));
    }

    launch_batched_gemm(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list,
        D_list);
}

void nvfp4_localcta_batched_accum_gemm_entrypoint(
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& A_sc_list,
    const std::vector<at::Tensor>& A_sg_chunks_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& B_sc_list,
    const std::vector<at::Tensor>& B_sg_chunks_list,
    at::Tensor& D_out
) {
    check_batched_inputs(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list);
    const int n = static_cast<int>(A_list.size());
    TORCH_CHECK(n <= 4, "num_batches must be 1..4");
    check_output_matrix(D_out, "D_out", A_list[0].size(0), B_list[0].size(0));

    if (n == 1) {
        launch_regular_gemm(
            A_list[0], A_sc_list[0], A_sg_chunks_list[0],
            B_list[0], B_sc_list[0], B_sg_chunks_list[0], D_out);
        return;
    }

    std::vector<at::Tensor> D_list;
    D_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        D_list.push_back(at::empty_like(D_out));
    }

    launch_batched_gemm(
        A_list, A_sc_list, A_sg_chunks_list,
        B_list, B_sc_list, B_sg_chunks_list,
        D_list);

    const int64_t numel = D_out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (n == 2) {
        sum_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else if (n == 3) {
        sum3_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    } else {
        sum4_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(D_list[0].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[1].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[2].data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(D_list[3].data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(D_out.data_ptr()),
            numel);
    }

    CUDACHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_localcta_gemm", &nvfp4_localcta_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"));
    m.def("nvfp4_localcta_grouped_gemm", &nvfp4_localcta_grouped_gemm_entrypoint,
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sg_chunks"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sg_chunks"),
          pybind11::arg("D"), pybind11::arg("D_K") = std::nullopt,
          pybind11::arg("D_V") = std::nullopt, pybind11::arg("silu_dim") = 0);
    m.def("nvfp4_localcta_batched_gemm", &nvfp4_localcta_batched_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_list"));
    m.def("nvfp4_localcta_batched_accum_gemm", &nvfp4_localcta_batched_accum_gemm_entrypoint,
          pybind11::arg("A_list"), pybind11::arg("A_sc_list"),
          pybind11::arg("A_sg_chunks_list"), pybind11::arg("B_list"),
          pybind11::arg("B_sc_list"), pybind11::arg("B_sg_chunks_list"),
          pybind11::arg("D_out"));
}
