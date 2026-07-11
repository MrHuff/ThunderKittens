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
    constexpr int WARPS_PER_BLOCK = 4;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t slice = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp;
    if (row >= rows || slice >= hidden_size / 32) {
        return;
    }
    const int64_t col = slice * 32 + lane;
    const int64_t idx = row * hidden_size + col;
    const float g = gamma == nullptr ? 1.0f : __bfloat162float(gamma[col]);
    float sum = __bfloat162float(x[idx]) * __bfloat162float(dy[idx]) * g;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        partial_dot[row * (hidden_size / 32) + slice] = sum;
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

__global__ void rmsnorm_dx_apply_native_order_kernel(
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
    const float inv = coeff[row];
    const float g = gamma == nullptr ? 1.0f : __bfloat162float(gamma[col]);
    const float x_val = __bfloat162float(x[idx]);
    const float dy_val = __bfloat162float(dy[idx]);
    const float dot_mean = dot[row] / static_cast<float>(hidden_size);
    const float inv3_dot = inv * inv * inv * dot_mean;
    const float dx_val = inv * (dy_val * g) - x_val * inv3_dot;
    dx[idx] = __float2bfloat16(dx_val);
}

__global__ void dgamma_native_order_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dy,
    const float* __restrict__ coeff,
    float* __restrict__ dgamma,
    int64_t rows,
    int64_t hidden_size
) {
    __shared__ float scratch[256];
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if (col >= hidden_size) {
        return;
    }
    const int tid = threadIdx.x;
    float partial = 0.0f;
    for (int64_t row = tid; row < rows; row += blockDim.x) {
        const int64_t idx = row * hidden_size + col;
        partial += (
            __bfloat162float(dy[idx])
            * __bfloat162float(x[idx])
            * coeff[row]
        );
    }
    scratch[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dgamma[col] = scratch[0];
    }
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void row_tile_partial_dgamma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dy,
    const float* __restrict__ coeff,
    float* __restrict__ partial_dgamma,
    int64_t rows,
    int64_t hidden_size
) {
    __shared__ float warp_sums[8];
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    const int64_t row_tile = static_cast<int64_t>(blockIdx.y);
    if (col >= hidden_size) {
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t row_start = row_tile * 256;
    const int64_t row_end = min(row_start + 256, rows);
    float partial = 0.0f;
    for (int64_t row = row_start + tid; row < row_end; row += blockDim.x) {
        const int64_t idx = row * hidden_size + col;
        partial += (
            __bfloat162float(x[idx])
            * __bfloat162float(dy[idx])
            * coeff[row]
        );
    }
    partial = warp_sum(partial);
    if (lane == 0) {
        warp_sums[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < 8 ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            partial_dgamma[row_tile * hidden_size + col] = block_sum;
        }
    }
}

__global__ void partial_dgamma_reduce_kernel(
    const float* __restrict__ partial_dgamma,
    float* __restrict__ dgamma,
    int64_t row_tiles,
    int64_t hidden_size
) {
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if (col >= hidden_size) {
        return;
    }
    const int lane = threadIdx.x;
    float sum = 0.0f;
    for (int64_t tile = lane; tile < row_tiles; tile += 32) {
        sum += partial_dgamma[tile * hidden_size + col];
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        dgamma[col] = sum;
    }
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

inline void check_partial_dgamma(const at::Tensor& partial_dgamma, int64_t rows, int64_t hidden_size) {
    const int64_t row_tiles = (rows + 255) / 256;
    TORCH_CHECK(partial_dgamma.is_cuda(), "partial_dgamma must be CUDA");
    TORCH_CHECK(partial_dgamma.is_contiguous(), "partial_dgamma must be contiguous");
    TORCH_CHECK(partial_dgamma.scalar_type() == at::kFloat, "partial_dgamma must be float32");
    TORCH_CHECK(partial_dgamma.dim() == 2 &&
                    partial_dgamma.size(0) == row_tiles &&
                    partial_dgamma.size(1) == hidden_size,
                "partial_dgamma must have shape [ceil(M / 256), hidden_size]");
}

inline void check_dgamma(const at::Tensor& dgamma, int64_t hidden_size) {
    TORCH_CHECK(dgamma.is_cuda(), "dgamma must be CUDA");
    TORCH_CHECK(dgamma.is_contiguous(), "dgamma must be contiguous");
    TORCH_CHECK(dgamma.scalar_type() == at::kFloat, "dgamma must be float32");
    TORCH_CHECK(dgamma.dim() == 1 && dgamma.numel() == hidden_size,
                "dgamma must have shape [hidden_size]");
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
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid(rows, (hidden_size / 32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(32 * WARPS_PER_BLOCK);
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

inline void apply_dx_native_order_entrypoint(
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
    rmsnorm_dx_apply_native_order_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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

inline void dgamma_native_order_entrypoint(
    const at::Tensor& x,
    const at::Tensor& dy,
    const at::Tensor& coeff,
    at::Tensor& dgamma
) {
    check_matrix_bf16(x, "x");
    check_matrix_bf16(dy, "dy");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    const int64_t rows = x.size(0);
    const int64_t hidden_size = x.size(1);
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    check_vector_fp32(coeff, "coeff", rows);
    check_dgamma(dgamma, hidden_size);
    if (rows == 0) {
        dgamma.zero_();
        return;
    }
    dgamma_native_order_kernel<<<static_cast<int>(hidden_size), 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
        coeff.data_ptr<float>(),
        dgamma.data_ptr<float>(),
        rows,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void partial_dgamma_entrypoint(
    const at::Tensor& x,
    const at::Tensor& dy,
    const at::Tensor& coeff,
    at::Tensor& partial_dgamma
) {
    check_matrix_bf16(x, "x");
    check_matrix_bf16(dy, "dy");
    TORCH_CHECK(dy.sizes() == x.sizes(), "dy shape must match x");
    const int64_t rows = x.size(0);
    const int64_t hidden_size = x.size(1);
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    check_vector_fp32(coeff, "coeff", rows);
    check_partial_dgamma(partial_dgamma, rows, hidden_size);
    if (rows == 0) {
        return;
    }
    constexpr int threads = 256;
    dim3 grid(static_cast<unsigned int>(hidden_size), static_cast<unsigned int>((rows + 255) / 256));
    row_tile_partial_dgamma_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr()),
        coeff.data_ptr<float>(),
        partial_dgamma.data_ptr<float>(),
        rows,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void reduce_dgamma_entrypoint(
    const at::Tensor& partial_dgamma,
    at::Tensor& dgamma,
    int64_t hidden_size
) {
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    TORCH_CHECK(partial_dgamma.is_cuda(), "partial_dgamma must be CUDA");
    TORCH_CHECK(partial_dgamma.is_contiguous(), "partial_dgamma must be contiguous");
    TORCH_CHECK(partial_dgamma.scalar_type() == at::kFloat, "partial_dgamma must be float32");
    TORCH_CHECK(partial_dgamma.dim() == 2, "partial_dgamma must be rank-2 [ceil(M / 256), hidden_size]");
    TORCH_CHECK(partial_dgamma.size(1) == hidden_size,
                "partial_dgamma second dimension must equal hidden_size");
    check_dgamma(dgamma, hidden_size);
    const int64_t row_tiles = partial_dgamma.size(0);
    if (row_tiles == 0) {
        dgamma.zero_();
        return;
    }
    constexpr int threads = 32;
    partial_dgamma_reduce_kernel<<<static_cast<int>(hidden_size), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial_dgamma.data_ptr<float>(),
        dgamma.data_ptr<float>(),
        row_tiles,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace c5_rms_bwd
