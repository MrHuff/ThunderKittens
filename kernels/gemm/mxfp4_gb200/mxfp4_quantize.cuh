#pragma once
// ================================================================
// MXFP4 Quantization Kernel
// Single-pass: bf16 → FP4 + E8M0 scales (no global amax needed)
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace mxfp4_quantize {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPS = 2; // 64 threads, 2 rows per thread
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_SIZE = 128;       // This should not change
    static constexpr int K_BLOCK_SIZE = 32;     // MXFP4 block size = 32

    using A_bf16_tile  = st_bf<TILE_SIZE, TILE_SIZE, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_SIZE, TILE_SIZE/2, false>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;

    using A_bf16_gl  = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;

    A_bf16_gl  A_bf16;  // M x N
    A_fp4x2_gl A_fp4x2; // M x (N/2)
    A_sc_gl    A_sc;     // (M/128) x (N/128) x 32 x 16

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_SIZE, A_bf16.rows() / TILE_SIZE);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TILE_SIZE * TILE_SIZE * sizeof(bf16) + 1024;
    }
};

__device__ inline void kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    globals::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem = *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_tile &A_sc_smem = *reinterpret_cast<globals::A_sc_tile *>(
        reinterpret_cast<uint64_t>(&A_fp4x2_smem) + sizeof(A_fp4x2_smem));

    // Calculate indices
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, A_bf16_smem);
        tma::load_async(A_bf16_smem, G.A_bf16, {row, col}, inputs_arrived);
    }

    // Wait for the TMA load to complete
    __syncthreads();
    wait(inputs_arrived, 0);

    // We have 64 threads per block. Each thread handles 2 rows of 128 elements.
    constexpr int ROWS_PER_THREAD = 2;
    constexpr int NUM_K_BLOCKS = globals::TILE_SIZE / globals::K_BLOCK_SIZE; // 4
    constexpr int N_PER_K_BLOCK = globals::TILE_SIZE / 2 / NUM_K_BLOCKS;     // 16
    bf16_2 A_bf16_reg[ROWS_PER_THREAD][NUM_K_BLOCKS][N_PER_K_BLOCK];
    fp8e8m0 A_sc_reg[ROWS_PER_THREAD][NUM_K_BLOCKS];

    // Load input matrix from shared memory (custom swizzling)
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int tile_row = tid + (r*64);
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // 8 SMEM banks per K-block
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                int offset = (tile_row*globals::TILE_SIZE + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[r][i][j], static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }
    __syncthreads();

    // Perform MXFP4 quantization with E8M0 scales
    // Scale convention matches TE MXFP4 decode-centric (no global scale):
    //   E8M0 = round(log2(amax)) + 127  (tracks raw amax, not amax/6)
    //   Quantize: x * (6.0 / 2^exponent)  (divides by scale, multiplies by FP4 max)
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int tile_row = tid + (r*64);
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            int k_block_idx = (i + tid/8) % NUM_K_BLOCKS;

            // Calculate absolute maximum over the block of 32 elements
            bf16_2 amax = __habs2(A_bf16_reg[r][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                amax = __hmax2(amax, __habs2(A_bf16_reg[r][i][j]));

            // Compute E8M0 scale: round(log2(amax)) + 127
            float block_amax = __bfloat162float(__hmax(amax.x, amax.y));
            int e8m0_val;
            if (block_amax <= 1e-9f) {
                e8m0_val = 0;  // min scale for zero blocks
            } else {
                int exp = (int)roundf(log2f(block_amax));
                e8m0_val = min(max(exp + 127, 0), 255);
            }
            A_sc_reg[r][k_block_idx].__x = (__nv_fp8_storage_t)e8m0_val;
            // Quantize factor = 6.0 / 2^exponent = FP4_MAX / scale
            float scale_inv = 6.0f / static_cast<float>(A_sc_reg[r][k_block_idx]);

            // Quantize to FP4 and store to shared memory
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                float2 scaled = {
                    __bfloat162float(A_bf16_reg[r][i][j].x) * scale_inv,
                    __bfloat162float(A_bf16_reg[r][i][j].y) * scale_inv
                };
                int offset = tile_row * globals::TILE_SIZE / 2 + tile_col / 2;
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_fp4x2_smem)) + offset)
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
            }
        }

        // Store the scales to shared memory following NVIDIA's scale swizzle layout
        int scale_offset = (tile_row % 32) * 16 + // row
                           (tile_row / 32) * 4;   // column
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem)) + scale_offset)
               "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[r][0])));
    }

    // Store to global memory
    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_fp4x2, A_fp4x2_smem, {row, col});
        tma::store_async(G.A_sc, A_sc_smem, {row, col, 0, 0});
    }
}

} // namespace mxfp4_quantize
