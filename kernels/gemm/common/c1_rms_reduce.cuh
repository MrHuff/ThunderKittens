#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

#include "kittens.cuh"

namespace c1_rms_reduce {

inline void check_no_overlap(
    const at::Tensor& writable,
    const char* writable_name,
    const at::Tensor& input,
    const char* input_name
) {
    const auto writable_begin = reinterpret_cast<uintptr_t>(writable.data_ptr());
    const auto input_begin = reinterpret_cast<uintptr_t>(input.data_ptr());
    const auto writable_bytes = static_cast<uintptr_t>(writable.nbytes());
    const auto input_bytes = static_cast<uintptr_t>(input.nbytes());
    if (writable_bytes == 0 || input_bytes == 0) {
        return;
    }
    TORCH_CHECK(
        writable_begin + writable_bytes <= input_begin ||
            input_begin + input_bytes <= writable_begin,
        "C4 ", writable_name, " must not overlap ", input_name
    );
}

template <typename... ReadTensors>
inline void check_c2_c3_output_overlap(
    const at::Tensor& row_rms_partial,
    const at::Tensor& coeff,
    const at::Tensor& D,
    const ReadTensors&... reads
) {
    check_no_overlap(coeff, "coeff", row_rms_partial, "row_rms_partial");
    check_no_overlap(coeff, "coeff", D, "D");
    (check_no_overlap(coeff, "coeff", reads, "a GEMM read input"), ...);
    check_no_overlap(D, "D", row_rms_partial, "row_rms_partial");
    (check_no_overlap(D, "D", reads, "a GEMM read input"), ...);
}

template <bool EARLY_PDL_ARRIVE = false>
__global__ void row_rms_coeff_kernel(
    const float* __restrict__ row_rms_partial,
    float* __restrict__ coeff,
    int64_t rows,
    int64_t partial_cols,
    int64_t hidden_size,
    float eps
) {
    if constexpr (EARLY_PDL_ARRIVE) {
        if (threadIdx.x == 0) {
            kittens::pdl::arrive();
        }
    }
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

template <bool EARLY_PDL_ARRIVE = false>
__global__ void row_rms_coeff_warp_kernel(
    const float* __restrict__ row_rms_partial,
    float* __restrict__ coeff,
    int64_t rows,
    int64_t partial_cols,
    int64_t hidden_size,
    float eps
) {
    if constexpr (EARLY_PDL_ARRIVE) {
        if (threadIdx.x == 0) {
            kittens::pdl::arrive();
        }
    }
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
    TORCH_CHECK(row_rms_partial.get_device() == coeff.get_device(),
                "row_rms_partial and coeff must be on the same CUDA device");
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
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto tensor_device = row_rms_partial.get_device();
    if (C10_UNLIKELY(stream.device_index() != tensor_device)) {
        const c10::cuda::CUDAGuard device_guard(tensor_device);
        if (partial_cols <= 128) {
            constexpr int WARPS_PER_BLOCK = 4;
            dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
            dim3 block(32 * WARPS_PER_BLOCK);
            row_rms_coeff_warp_kernel<><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
            row_rms_coeff_kernel<><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                row_rms_partial.data_ptr<float>(),
                coeff.data_ptr<float>(),
                rows,
                partial_cols,
                hidden_size,
                static_cast<float>(eps)
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    if (partial_cols <= 128) {
        constexpr int WARPS_PER_BLOCK = 4;
        dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        dim3 block(32 * WARPS_PER_BLOCK);
        row_rms_coeff_warp_kernel<><<<grid, block, 0, stream>>>(
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
        row_rms_coeff_kernel<><<<grid, block, 0, stream>>>(
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

inline void check_row_rms_reduce_pdl_args(
    const at::Tensor& row_rms_partial,
    const at::Tensor& coeff,
    int64_t hidden_size,
    double eps
) {
    check_row_rms_reduce_args(row_rms_partial, coeff, hidden_size);
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0,
                "C4 eps must be finite and non-negative");
    const int64_t rows = row_rms_partial.size(0);
    const int64_t partial_cols = row_rms_partial.size(1);
    TORCH_CHECK(rows > 0, "C4 overlap requires M > 0");
    TORCH_CHECK(partial_cols <= 128,
                "C4 PDL C2 reducer supports at most 128 partial columns");
}

inline void launch_row_rms_reduce_pdl_unchecked(
    const at::Tensor& row_rms_partial,
    at::Tensor& coeff,
    int64_t hidden_size,
    double eps
) {
    const int64_t rows = row_rms_partial.size(0);
    const int64_t partial_cols = row_rms_partial.size(1);
    if (rows == 0) {
        return;
    }
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(32 * WARPS_PER_BLOCK);
    row_rms_coeff_warp_kernel<true><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        row_rms_partial.data_ptr<float>(),
        coeff.data_ptr<float>(),
        rows,
        partial_cols,
        hidden_size,
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void row_rms_reduce_pdl_entrypoint(
    const at::Tensor& row_rms_partial,
    at::Tensor& coeff,
    int64_t hidden_size,
    double eps
) {
    check_row_rms_reduce_pdl_args(row_rms_partial, coeff, hidden_size, eps);
    const c10::cuda::CUDAGuard device_guard(row_rms_partial.device());
    launch_row_rms_reduce_pdl_unchecked(row_rms_partial, coeff, hidden_size, eps);
}

} // namespace c1_rms_reduce
