#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <optional>

namespace c5_rms_bwd {

__global__ void row_partial_dot_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ gamma,
    float* __restrict__ partial_dot,
    int64_t rows,
    int64_t hidden_size
) {
    __shared__ float scratch[32];
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t slice = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;
    if (row >= rows || slice >= hidden_size / 32 || tid >= 32) {
        return;
    }
    const int64_t col = slice * 32 + tid;
    const int64_t idx = row * hidden_size + col;
    const float g = gamma == nullptr ? 1.0f : __bfloat162float(gamma[col]);
    scratch[tid] = __bfloat162float(x[idx]) * __bfloat162float(dy[idx]) * g;
    __syncthreads();
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_dot[row * (hidden_size / 32) + slice] = scratch[0];
    }
}

__global__ void row_dot_reduce_kernel(
    const float* __restrict__ partial_dot,
    float* __restrict__ dot,
    int64_t rows,
    int64_t partial_cols
) {
    __shared__ float scratch[256];
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    float sum = 0.0f;
    const int64_t base = row * partial_cols;
    for (int64_t col = tid; col < partial_cols; col += blockDim.x) {
        sum += partial_dot[base + col];
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dot[row] = scratch[0];
    }
}

__global__ void rmsnorm_dx_apply_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ coeff,
    const float* __restrict__ dot,
    __nv_bfloat16* __restrict__ dx,
    int64_t numel,
    int64_t hidden_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const int64_t row = idx / hidden_size;
    const int64_t col = idx - row * hidden_size;
    const float r = coeff[row];
    const float g = gamma == nullptr ? 1.0f : __bfloat162float(gamma[col]);
    const float x_val = __bfloat162float(x[idx]);
    const float dy_gamma = __bfloat162float(dy[idx]) * g;
    const float dx_val = r * dy_gamma - x_val * (r * r * r / static_cast<float>(hidden_size)) * dot[row];
    dx[idx] = __float2bfloat16(dx_val);
}

inline void check_matrix_bf16(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.dim() == 2, name, " must be rank-2 [M,N]");
}

inline void check_gamma(const std::optional<at::Tensor>& gamma_opt, int64_t hidden_size) {
    if (!gamma_opt.has_value()) {
        return;
    }
    const at::Tensor& gamma = gamma_opt.value();
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16, "gamma must be bf16");
    TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == hidden_size,
                "gamma must have shape [hidden_size]");
}

inline void check_partial_dot(const at::Tensor& partial_dot, int64_t rows, int64_t hidden_size) {
    TORCH_CHECK(partial_dot.is_cuda(), "partial_dot must be CUDA");
    TORCH_CHECK(partial_dot.is_contiguous(), "partial_dot must be contiguous");
    TORCH_CHECK(partial_dot.scalar_type() == at::kFloat, "partial_dot must be float32");
    TORCH_CHECK(partial_dot.dim() == 2 &&
                    partial_dot.size(0) == rows &&
                    partial_dot.size(1) == hidden_size / 32,
                "partial_dot must have shape [M, hidden_size / 32]");
}

inline void check_vector_fp32(const at::Tensor& t, const char* name, int64_t rows) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 1 && t.numel() == rows, name, " must have shape [M]");
}

inline void partial_dot_entrypoint(
    const at::Tensor& x,
    const at::Tensor& dy,
    at::Tensor& partial_dot,
    std::optional<at::Tensor> gamma_opt = std::nullopt
) {
    check_matrix_bf16(x, "x");
    check_matrix_bf16(dy, "dy");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    const int64_t rows = x.size(0);
    const int64_t hidden_size = x.size(1);
    TORCH_CHECK(hidden_size > 0 && hidden_size % 32 == 0,
                "hidden_size must be positive and divisible by 32");
    check_gamma(gamma_opt, hidden_size);
    check_partial_dot(partial_dot, rows, hidden_size);
    if (rows == 0) {
        return;
    }
    const __nv_bfloat16* gamma_ptr = gamma_opt.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(gamma_opt.value().data_ptr())
        : nullptr;
    dim3 grid(rows, hidden_size / 32);
    dim3 block(32);
    row_partial_dot_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
        gamma_ptr,
        partial_dot.data_ptr<float>(),
        rows,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void reduce_dot_entrypoint(
    const at::Tensor& partial_dot,
    at::Tensor& dot,
    int64_t hidden_size
) {
    TORCH_CHECK(hidden_size > 0 && hidden_size % 32 == 0,
                "hidden_size must be positive and divisible by 32");
    TORCH_CHECK(partial_dot.is_cuda(), "partial_dot must be CUDA");
    TORCH_CHECK(partial_dot.is_contiguous(), "partial_dot must be contiguous");
    TORCH_CHECK(partial_dot.scalar_type() == at::kFloat, "partial_dot must be float32");
    TORCH_CHECK(partial_dot.dim() == 2, "partial_dot must be rank-2 [M, hidden_size / 32]");
    TORCH_CHECK(partial_dot.size(1) == hidden_size / 32,
                "partial_dot second dimension must equal hidden_size / 32");
    const int64_t rows = partial_dot.size(0);
    check_vector_fp32(dot, "dot", rows);
    if (rows == 0) {
        return;
    }
    row_dot_reduce_kernel<<<static_cast<int>(rows), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial_dot.data_ptr<float>(),
        dot.data_ptr<float>(),
        rows,
        partial_dot.size(1)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void apply_dx_entrypoint(
    const at::Tensor& x,
    const at::Tensor& dy,
    const at::Tensor& coeff,
    const at::Tensor& dot,
    at::Tensor& dx,
    int64_t hidden_size,
    std::optional<at::Tensor> gamma_opt = std::nullopt
) {
    check_matrix_bf16(x, "x");
    check_matrix_bf16(dy, "dy");
    check_matrix_bf16(dx, "dx");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    TORCH_CHECK(dx.sizes() == x.sizes(), "dx shape must match x");
    TORCH_CHECK(hidden_size == x.size(1), "hidden_size must equal x.shape[1]");
    TORCH_CHECK(hidden_size > 0 && hidden_size % 32 == 0,
                "hidden_size must be positive and divisible by 32");
    const int64_t rows = x.size(0);
    check_gamma(gamma_opt, hidden_size);
    check_vector_fp32(coeff, "coeff", rows);
    check_vector_fp32(dot, "dot", rows);
    const int64_t numel = x.numel();
    if (numel == 0) {
        return;
    }
    const __nv_bfloat16* gamma_ptr = gamma_opt.has_value()
        ? reinterpret_cast<const __nv_bfloat16*>(gamma_opt.value().data_ptr())
        : nullptr;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    rmsnorm_dx_apply_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
        gamma_ptr,
        coeff.data_ptr<float>(),
        dot.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
        numel,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace c5_rms_bwd
