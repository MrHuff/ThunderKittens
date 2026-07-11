#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace c1_rms_reduce {

__global__ void row_rms_coeff_kernel(
    const float* __restrict__ row_rms_partial,
    float* __restrict__ coeff,
    int64_t rows,
    int64_t partial_cols,
    int64_t hidden_size,
    float eps
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
        sum += row_rms_partial[base + col];
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
        coeff[row] = rsqrtf(scratch[0] / static_cast<float>(hidden_size) + eps);
    }
}

__global__ void row_rms_coeff_warp_kernel(
    const float* __restrict__ row_rms_partial,
    float* __restrict__ coeff,
    int64_t rows,
    int64_t partial_cols,
    int64_t hidden_size,
    float eps
) {
    constexpr int WARPS_PER_BLOCK = 4;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    if (row >= rows) {
        return;
    }
    const int64_t base = row * partial_cols;
    float low = lane < partial_cols ? row_rms_partial[base + lane] : 0.0f;
    if (lane + 64 < partial_cols) {
        low += row_rms_partial[base + lane + 64];
    }
    float high = lane + 32 < partial_cols ? row_rms_partial[base + lane + 32] : 0.0f;
    if (lane + 96 < partial_cols) {
        high += row_rms_partial[base + lane + 96];
    }
    float sum = low + high;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        coeff[row] = rsqrtf(sum / static_cast<float>(hidden_size) + eps);
    }
}

inline void check_row_rms_reduce_args(
    const at::Tensor& row_rms_partial,
    const at::Tensor& coeff,
    int64_t hidden_size
) {
    TORCH_CHECK(row_rms_partial.is_cuda(), "row_rms_partial must be CUDA");
    TORCH_CHECK(coeff.is_cuda(), "coeff must be CUDA");
    TORCH_CHECK(row_rms_partial.scalar_type() == at::kFloat, "row_rms_partial must be float32");
    TORCH_CHECK(coeff.scalar_type() == at::kFloat, "coeff must be float32");
    TORCH_CHECK(row_rms_partial.is_contiguous(), "row_rms_partial must be contiguous");
    TORCH_CHECK(coeff.is_contiguous(), "coeff must be contiguous");
    TORCH_CHECK(row_rms_partial.dim() == 2, "row_rms_partial must be rank-2 [M, N/32]");
    TORCH_CHECK(coeff.dim() == 1, "coeff must be rank-1 [M]");
    TORCH_CHECK(row_rms_partial.size(0) == coeff.size(0), "coeff length must equal row_rms_partial rows");
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    TORCH_CHECK(hidden_size % 32 == 0, "hidden_size must be divisible by 32");
    TORCH_CHECK(row_rms_partial.size(1) == hidden_size / 32,
                "row_rms_partial second dimension must equal hidden_size / 32");
}

inline void row_rms_reduce_entrypoint(
    const at::Tensor& row_rms_partial,
    at::Tensor& coeff,
    int64_t hidden_size,
    double eps
) {
    check_row_rms_reduce_args(row_rms_partial, coeff, hidden_size);
    TORCH_CHECK(eps >= 0.0, "eps must be non-negative");
    const int64_t rows = row_rms_partial.size(0);
    const int64_t partial_cols = row_rms_partial.size(1);
    if (rows == 0) {
        return;
    }
    if (partial_cols <= 128) {
        constexpr int WARPS_PER_BLOCK = 4;
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        dim3 block(32 * WARPS_PER_BLOCK);
        row_rms_coeff_warp_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            row_rms_partial.data_ptr<float>(),
            coeff.data_ptr<float>(),
            rows,
            partial_cols,
            hidden_size,
            static_cast<float>(eps)
        );
    } else {
        dim3 grid(rows);
        dim3 block(256);
        row_rms_coeff_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            row_rms_partial.data_ptr<float>(),
            coeff.data_ptr<float>(),
            rows,
            partial_cols,
            hidden_size,
            static_cast<float>(eps)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace c1_rms_reduce
