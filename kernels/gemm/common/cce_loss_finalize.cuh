#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace fp4_cce {

template <int BLOCK_THREADS>
__global__ void finalize_loss_kernel(
    const float* __restrict__ lse,
    const float* __restrict__ neg_target_logit,
    const int64_t* __restrict__ targets,
    float* __restrict__ loss,
    int64_t* __restrict__ valid_count,
    int rows,
    int64_t ignore_index) {
    static_assert(BLOCK_THREADS % 32 == 0);
    constexpr int WARPS = BLOCK_THREADS / 32;
    __shared__ float warp_sums[WARPS];
    __shared__ int warp_counts[WARPS];

    float thread_sum = 0.0f;
    int thread_count = 0;
    for (int row = threadIdx.x; row < rows; row += BLOCK_THREADS) {
        if (targets[row] != ignore_index) {
            thread_sum += lse[row] + neg_target_logit[row];
            ++thread_count;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        thread_count += __shfl_down_sync(0xffffffff, thread_count, offset);
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = thread_sum;
        warp_counts[warp] = thread_count;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < WARPS ? warp_sums[lane] : 0.0f;
        int block_count = lane < WARPS ? warp_counts[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            block_count += __shfl_down_sync(0xffffffff, block_count, offset);
        }
        if (lane == 0) {
            valid_count[0] = static_cast<int64_t>(block_count);
            loss[0] = block_count > 0 ? block_sum / static_cast<float>(block_count) : 0.0f;
        }
    }
}

inline void launch_finalize_loss(
    const at::Tensor& lse,
    const at::Tensor& neg_target_logit,
    const at::Tensor& targets,
    at::Tensor& loss,
    at::Tensor& valid_count,
    int rows,
    int64_t ignore_index) {
    TORCH_CHECK(lse.is_cuda() && neg_target_logit.is_cuda() && targets.is_cuda(),
                "CCE inputs must be CUDA tensors");
    TORCH_CHECK(loss.is_cuda() && valid_count.is_cuda(),
                "CCE outputs must be CUDA tensors");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && neg_target_logit.scalar_type() == at::kFloat,
                "CCE LSE and target-logit tensors must be float32");
    TORCH_CHECK(targets.scalar_type() == at::kLong, "CCE targets must be int64");
    TORCH_CHECK(loss.scalar_type() == at::kFloat && loss.numel() == 1,
                "CCE loss must be a scalar float32 tensor");
    TORCH_CHECK(valid_count.scalar_type() == at::kLong && valid_count.numel() == 1,
                "CCE valid_count must be a scalar int64 tensor");
    TORCH_CHECK(rows >= 0 && rows <= lse.numel() && rows <= neg_target_logit.numel() &&
                    rows <= targets.numel(),
                "CCE row count exceeds an input tensor");
    TORCH_CHECK(lse.get_device() == targets.get_device() &&
                    neg_target_logit.get_device() == targets.get_device() &&
                    loss.get_device() == targets.get_device() &&
                    valid_count.get_device() == targets.get_device(),
                "CCE tensors must be on the same CUDA device");

    const c10::cuda::CUDAGuard device_guard(targets.device());
    constexpr int THREADS = 256;
    finalize_loss_kernel<THREADS><<<1, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        lse.data_ptr<float>(),
        neg_target_logit.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        valid_count.data_ptr<int64_t>(),
        rows,
        ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fp4_cce
