#pragma once

#include "nvfp4_cce_backward_v3.cuh"

namespace nvfp4_cce_backward_v6 {

template <typename C>
using globals_5wg = nvfp4_cce_backward_v3::globals_5wg<C>;

template <typename C>
using globals_3wg = nvfp4_cce_backward_v3::globals_3wg<C>;

template <typename C, int SYNC_ID>
__device__ inline void load_combo_row_stage_from_global(
    typename globals_5wg<C>::combo_row_stage_t& stage,
    const globals_5wg<C>& g,
    int row_block_idx,
    int col_block_idx,
    int cta_id)
{
    const int lane_id = threadIdx.x % WARP_THREADS;
    const int row_fp4_stride = g.G_fp4_row.cols();
    const int global_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
    const int global_fp4x2_col_base = col_block_idx * (C::Nb / 2);
    uint8_t* dst_fp4 = reinterpret_cast<uint8_t*>(&stage.G_row.data[0]);
    uint8_t* dst_sc = reinterpret_cast<uint8_t*>(&stage.G_row_sc.data[0]);
    const uint8_t* src_fp4 = reinterpret_cast<const uint8_t*>(g.G_fp4_row.raw_ptr);
    const uint8_t* src_sc = reinterpret_cast<const uint8_t*>(g.G_sc_row_ptr);

    for (int idx = warpgroup::warpid() * WARP_THREADS + lane_id;
         idx < static_cast<int>(sizeof(typename globals_5wg<C>::combo_row_stage_t));
         idx += WARPGROUP_WARPS * WARP_THREADS) {
        dst_fp4[idx] = 0;
    }
    for (int local_row = warpgroup::warpid(); local_row < C::Mb / 2; local_row += WARPGROUP_WARPS) {
        const int global_row = global_row_base + local_row;
        const int fp4_offset = global_row * row_fp4_stride + global_fp4x2_col_base;
        const int local_offset = local_row * globals_5wg<C>::G_fp4_row_tile::cols;
        for (int fp4x2_col = lane_id; fp4x2_col < globals_5wg<C>::G_fp4_row_tile::cols; fp4x2_col += WARP_THREADS) {
            dst_fp4[local_offset + fp4x2_col] = src_fp4[fp4_offset + fp4x2_col];
        }
        const int depth = global_row / 128;
        const int sr = global_row % 32;
        const int rr = (global_row / 32) % 4;
        for (int local_col16_idx = lane_id; local_col16_idx < C::Nb / 16; local_col16_idx += WARP_THREADS) {
            const int global_col16 = col_block_idx * (C::Nb / 16) * 16 + local_col16_idx * 16;
            const int global_kgroup = global_col16 / 64;
            const int col16_in_64 = (global_col16 / 16) % 4;
            const int chunk = depth * g.G_sc_row_kgroups + global_kgroup;
            const int byte_idx = sr * 16 + rr * 4 + col16_in_64;
            const int local_kgroup = local_col16_idx / 4;
            const int local_byte_idx = (local_row % 32) * 16 + ((local_row / 32) % 4) * 4 + col16_in_64;
            dst_sc[local_kgroup * 512 + local_byte_idx] = src_sc[chunk * 512 + byte_idx];
        }
    }
    return;
}

template <typename C, int SYNC_ID>
__device__ inline void load_combo_col_stage_from_global(
    typename globals_5wg<C>::combo_col_stage_t& stage,
    const globals_5wg<C>& g,
    int row_block_idx,
    int col_block_idx)
{
    const int lane_id = threadIdx.x % WARP_THREADS;
    const int col_fp4_stride = g.A.rows() / 2;
    const int global_col_base = col_block_idx * C::Nb;
    const int global_row_pair_base = row_block_idx * (C::Mb / 2);
    uint8_t* dst_fp4 = reinterpret_cast<uint8_t*>(&stage.Gt_row.data[0]);
    uint8_t* dst_sc = reinterpret_cast<uint8_t*>(&stage.Gt_row_sc.data[0]);
    const uint8_t* src_fp4 = reinterpret_cast<const uint8_t*>(g.G_fp4_col_ptr);
    const uint8_t* src_sc = reinterpret_cast<const uint8_t*>(g.G_sc_col_ptr);

    for (int idx = warpgroup::warpid() * WARP_THREADS + lane_id;
         idx < static_cast<int>(sizeof(typename globals_5wg<C>::combo_col_stage_t));
         idx += WARPGROUP_WARPS * WARP_THREADS) {
        dst_fp4[idx] = 0;
    }
    warpgroup::sync(SYNC_ID);

    for (int local_col = warpgroup::warpid(); local_col < C::Nb; local_col += WARPGROUP_WARPS) {
        const int global_col = global_col_base + local_col;
        const int src_offset = global_col * col_fp4_stride + global_row_pair_base;
        const int dst_offset = local_col * globals_5wg<C>::combo_col_fp4_tile::cols;
        for (int row_pair = lane_id; row_pair < globals_5wg<C>::combo_col_fp4_tile::cols; row_pair += WARP_THREADS) {
            dst_fp4[dst_offset + row_pair] = src_fp4[src_offset + row_pair];
        }
        const int sr = global_col % 32;
        const int rr = (global_col / 32) % 4;
        for (int local_row16 = lane_id; local_row16 < C::Mb / 16; local_row16 += WARP_THREADS) {
            const int global_row_base16 = row_block_idx * C::Mb + local_row16 * 16;
            const int m_kgroup = global_row_base16 / 64;
            const int m_16_in_64 = (global_row_base16 / 16) % 4;
            const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
            dst_sc[m_kgroup * 512 + byte_idx] = src_sc[m_kgroup * 512 + byte_idx];
        }
    }
    warpgroup::sync(SYNC_ID);
}

template <typename C>
__device__ inline void issue_combo_de_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    uint32_t &combo_de_tmem_addr,
    semaphore &combo_de_tmem_provisioned,
    semaphore &combo_de_p3_c_tiles_arrived,
    semaphore &combo_de_p3_c_scales_arrived,
    semaphore &combo_de_p3_inputs_finished,
    semaphore (&combo_de_p3_outputs_arrived)[2],
    semaphore (&combo_de_p3_outputs_finished)[2])
{
    const bool combo_issue_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);
    (void)cta_id;
    (void)cluster_id;
    (void)combo_de_tmem_addr;
    (void)combo_de_tmem_provisioned;
    (void)combo_de_p3_c_tiles_arrived;
    (void)combo_de_p3_c_scales_arrived;
    (void)combo_de_p3_outputs_arrived;
    (void)combo_de_p3_outputs_finished;
    if (combo_issue_leader) {
        g.G_sg_row[{0}] = 271.0f;
    }
    return;
}

template <typename C>
__device__ inline void drain_combo_de_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    uint32_t &combo_de_tmem_addr,
    semaphore &combo_de_tmem_provisioned,
    semaphore &combo_de_p3_inputs_finished,
    semaphore (&combo_de_p3_outputs_arrived)[2],
    semaphore (&combo_de_p3_outputs_finished)[2])
{
    (void)combo_de_tmem_addr;
    (void)combo_de_tmem_provisioned;
    (void)combo_de_p3_inputs_finished;
    (void)combo_de_p3_outputs_arrived;
    (void)combo_de_p3_outputs_finished;
    (void)cta_id;
    (void)cluster_id;
    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
        g.G_sg_row[{0}] = 273.0f;
    }
    return;
}

template <typename C>
__device__ inline void issue_combo_dc_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    uint32_t &combo_dc_tmem_addr,
    semaphore &combo_dc_tmem_provisioned,
    semaphore &combo_dc_p3_e_tiles_arrived,
    semaphore &combo_dc_p3_e_scales_arrived,
    semaphore &combo_dc_p3_inputs_finished,
    semaphore (&combo_dc_p3_outputs_arrived)[2],
    semaphore (&combo_dc_p3_outputs_finished)[2])
{
    using G = globals_5wg<C>;
    constexpr int combo_dc_a_scale_chunks = C::Nb / 64;
    constexpr int combo_dc_b_scale_chunks = C::Nb / 64;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    auto &combo_col_stage = sm_allocator.template allocate<typename G::combo_col_stage_t>();
    auto &combo_dc_tile_stage = sm_allocator.template allocate<typename G::combo_dc_tile_stage_t>();
    auto &combo_dc_scales_stage = sm_allocator.template allocate<typename G::combo_dc_scales_stage_t>();
    auto &combo_dc_output_stage = sm_allocator.template allocate<typename G::combo_dc_output_stage_t>();
    (void)combo_dc_output_stage;

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    const bool combo_issue_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);

    if (warpgroup::warpid() == 0) {
        tm_allocator.provision(combo_dc_tmem_addr);
        warp::arrive(combo_dc_tmem_provisioned);
    }
    wait(combo_dc_tmem_provisioned, 0);
    tm_allocator.set_addr(combo_dc_tmem_addr);

    auto combo_dc_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
    auto combo_dc_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(C::Nb);
    auto combo_dc_a_sc_tm =
        tm_allocator.template allocate<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(2 * C::Nb);
    auto combo_dc_b_sc_tm =
        tm_allocator.template allocate<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(
            2 * C::Nb + 4 * combo_dc_a_scale_chunks);

    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_col_blocks = g.B.rows() / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;

    int combo_dc_e_tiles_phase = 0;
    int combo_dc_e_scales_phase = 0;
    int combo_dc_inputs_phase = 1;
    int combo_dc_outputs_finished_phase[2] = {0, 0};
    bool first_input_issue = true;
    bool first_output_use[2] = {true, true};

    #pragma unroll 1
    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
        const int supergroup_idx = block_idx / num_blocks_per_supergroup;
        const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
        const int rows_in_supergroup =
            min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
        const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
        const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
        const int col_block_idx = idx_within_supergroup / rows_in_supergroup;

        load_combo_col_stage_from_global<C, 4>(combo_col_stage, g, row_block_idx, col_block_idx);
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        warpgroup::sync(4);

        #pragma unroll
        for (int ii = 0; ii < combo_dc_a_scale_chunks; ++ii) {
            auto combo_a_sc_sub =
                combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
            auto &combo_gt_sc_sm_sub =
                *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                    reinterpret_cast<uint64_t>(&combo_col_stage.Gt_row_sc.data[0]) + 16 * 32 * ii);
            load_mxnv_scale_async2(combo_a_sc_sub, combo_gt_sc_sm_sub);
        }

        const int num_k_blocks = g.dC_out.cols() / C::Nb;
        #pragma unroll 1
        for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const int slot = k_block_idx & 1;
            auto combo_dc_tm = (slot == 0) ? combo_dc_tm_0 : combo_dc_tm_1;

            if (combo_issue_leader) {
                if (!first_input_issue) {
                    wait(combo_dc_p3_inputs_finished, combo_dc_inputs_phase);
                    combo_dc_inputs_phase ^= 1;
                }
                if (!first_output_use[slot]) {
                    wait(combo_dc_p3_outputs_finished[slot], combo_dc_outputs_finished_phase[slot]);
                    tensor_after_thread_sync();
                    combo_dc_outputs_finished_phase[slot] ^= 1;
                }
                tma::expect_bytes(combo_dc_p3_e_tiles_arrived, sizeof(typename G::combo_p3_E_tile));
                tma::load_async(
                    combo_dc_tile_stage.E_operand, g.E_col,
                    {k_block_idx * 2 + cta_id, row_block_idx},
                    combo_dc_p3_e_tiles_arrived);
                tma::expect_bytes(combo_dc_p3_e_scales_arrived, sizeof(typename G::combo_p3_E_sc_tile));
                tma::load_async(
                    combo_dc_scales_stage.E_sc, g.E_col_sc,
                    {k_block_idx, row_block_idx, 0},
                    combo_dc_p3_e_scales_arrived);
            }
            warpgroup::sync(4);

            if (combo_issue_leader) {
                wait(combo_dc_p3_e_scales_arrived, combo_dc_e_scales_phase);
                combo_dc_e_scales_phase ^= 1;
            }
            warpgroup::sync(4);

            #pragma unroll
            for (int ii = 0; ii < combo_dc_b_scale_chunks; ++ii) {
                auto combo_b_sc_sub =
                    combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                auto &combo_e_sc_sm_sub =
                    *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                        reinterpret_cast<uint64_t>(&combo_dc_scales_stage.E_sc.data[0]) + 16 * 32 * ii);
                load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
            }

            if (combo_issue_leader) {
                wait(combo_dc_p3_e_tiles_arrived, combo_dc_e_tiles_phase);
                combo_dc_e_tiles_phase ^= 1;
            }
            warpgroup::sync(4);

            mm2_ABt(
                combo_dc_tm, combo_col_stage.Gt_row, combo_dc_tile_stage.E_operand,
                combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(0),
                combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(0),
                combo_dc_p3_inputs_finished);
            tensor_commit<2>(combo_dc_p3_outputs_arrived[slot]);
            tensor_after_thread_sync();
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            warpgroup::sync(4);

            first_input_issue = false;
            first_output_use[slot] = false;
        }
    }
}

template <typename C>
__device__ inline void drain_combo_dc_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    uint32_t &combo_dc_tmem_addr,
    semaphore &combo_dc_tmem_provisioned,
    semaphore (&combo_dc_p3_outputs_arrived)[2],
    semaphore (&combo_dc_p3_outputs_finished)[2])
{
    using G = globals_5wg<C>;
    using combo_dc_rt = rt_fl<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;
    using combo_dc_rt_bf = rt_bf<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    auto &combo_col_stage = sm_allocator.template allocate<typename G::combo_col_stage_t>();
    auto &combo_dc_tile_stage = sm_allocator.template allocate<typename G::combo_dc_tile_stage_t>();
    auto &combo_dc_scales_stage = sm_allocator.template allocate<typename G::combo_dc_scales_stage_t>();
    auto &combo_dc_output_stage = sm_allocator.template allocate<typename G::combo_dc_output_stage_t>();
    (void)combo_col_stage;
    (void)combo_dc_tile_stage;
    (void)combo_dc_scales_stage;

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    const bool combo_drain_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);

    wait(combo_dc_tmem_provisioned, 0);
    tm_allocator.set_addr(combo_dc_tmem_addr);

    auto combo_dc_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
    auto combo_dc_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(C::Nb);

    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_col_blocks = g.B.rows() / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    const int num_k_blocks = g.dC_out.cols() / C::Nb;
    const float combo_dc_scale = g.G_sg_row[{0}] * g.E_col_sc_global[{0}];

    int combo_dc_outputs_arrived_phase[2] = {0, 0};

    #pragma unroll 1
    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
        const int supergroup_idx = block_idx / num_blocks_per_supergroup;
        const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
        const int rows_in_supergroup =
            min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
        const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
        const int col_block_idx = idx_within_supergroup / rows_in_supergroup;

        #pragma unroll 1
        for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const int slot = k_block_idx & 1;
            auto combo_dc_tm = (slot == 0) ? combo_dc_tm_0 : combo_dc_tm_1;

            if (combo_drain_leader) {
                wait(combo_dc_p3_outputs_arrived[slot], combo_dc_outputs_arrived_phase[slot]);
                combo_dc_outputs_arrived_phase[slot] ^= 1;
            }
            warpgroup::sync(6);

            #pragma unroll
            for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
                combo_dc_rt D_reg_fl;
                combo_dc_rt_bf D_reg_bf;
                warpgroup::tma::store_async_read_wait<0>();
                warpgroup::sync(6);
                warpgroup::load_async(
                    D_reg_fl,
                    combo_dc_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                        0,
                        combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(6);
                warp::mul(D_reg_fl, D_reg_fl, combo_dc_scale);
                warp::copy(D_reg_bf, D_reg_fl);
                warpgroup::store(combo_dc_output_stage.dC, D_reg_bf);
                warpgroup::sync(6);
                warpgroup::tma::store_add_async(
                    g.dC_out, combo_dc_output_stage.dC,
                    {col_block_idx, k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
                tensor_after_thread_sync();
            }
            warpgroup::tma::store_async_read_wait<0>();
            warpgroup::sync(6);

            if (combo_drain_leader) {
                warpgroup::tma::cluster::arrive(combo_dc_p3_outputs_finished[slot], 0, 1);
            }
        }
    }

    warpgroup::tma::store_async_read_wait<0>();
    if (warpgroup::warpid() == 0) {
        tm_allocator.deprovision();
    }
}

template <typename C, int STATIC_COMBO_MODE>
__device__ inline void backward_kernel_v6_streaming_5wg_bridge(const globals_5wg<C>& g) {
    using G = globals_5wg<C>;
    using FrontendC = typename C::base;
    using FrontendG = nvfp4_cce_backward_v3::globals_3wg<FrontendC>;
    static_assert(sizeof(FrontendG) == sizeof(globals_5wg<C>),
                  "v6 bridge alias expects the 5-WG and public-v3 globals layouts to match");

    __shared__ volatile int frontend_done[3];
    __shared__ semaphore combo_bridge_de_done;
    __shared__ volatile int combo_bridge_de_done_flag;

    __shared__ uint32_t combo_de_tmem_addr;
    __shared__ semaphore combo_de_tmem_provisioned;
    __shared__ semaphore combo_de_p3_c_tiles_arrived;
    __shared__ semaphore combo_de_p3_c_scales_arrived;
    __shared__ semaphore combo_de_p3_inputs_finished;
    __shared__ semaphore combo_de_p3_outputs_arrived[2];
    __shared__ semaphore combo_de_p3_outputs_finished[2];

    __shared__ uint32_t combo_dc_tmem_addr;
    __shared__ semaphore combo_dc_tmem_provisioned;
    __shared__ semaphore combo_dc_p3_e_tiles_arrived;
    __shared__ semaphore combo_dc_p3_e_scales_arrived;
    __shared__ semaphore combo_dc_p3_inputs_finished;
    __shared__ semaphore combo_dc_p3_outputs_arrived[2];
    __shared__ semaphore combo_dc_p3_outputs_finished[2];

    if (threadIdx.x == 0) {
        frontend_done[0] = 0;
        frontend_done[1] = 0;
        frontend_done[2] = 0;
        combo_bridge_de_done_flag = 0;
        if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY ||
                      STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
            g.C_col.template prefetch_tma<typename G::combo_p3_C_tile>();
            g.C_col_sc.template prefetch_tma<typename G::combo_p3_C_sc_tile>();
            g.dE_out.template prefetch_tma<typename G::combo_dE_tile>();
            init_semaphore(combo_de_tmem_provisioned, 0, 1);
            init_semaphore(combo_de_p3_c_tiles_arrived, 0, 1);
            init_semaphore(combo_de_p3_c_scales_arrived, 0, 1);
            init_semaphore(combo_de_p3_inputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                init_semaphore(combo_de_p3_outputs_arrived[i], 0, 1);
                init_semaphore(combo_de_p3_outputs_finished[i], 0, C::CLUSTER_SIZE);
            }
        }
        if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY ||
                      STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
            g.E_col.template prefetch_tma<typename G::combo_p3_E_tile>();
            g.E_col_sc.template prefetch_tma<typename G::combo_p3_E_sc_tile>();
            g.dC_out.template prefetch_tma<typename G::combo_dC_tile>();
            init_semaphore(combo_dc_tmem_provisioned, 0, 1);
            init_semaphore(combo_dc_p3_e_tiles_arrived, 0, 1);
            init_semaphore(combo_dc_p3_e_scales_arrived, 0, 1);
            init_semaphore(combo_dc_p3_inputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                init_semaphore(combo_dc_p3_outputs_arrived[i], 0, 1);
                init_semaphore(combo_dc_p3_outputs_finished[i], 0, C::CLUSTER_SIZE);
            }
        }
        if constexpr (STATIC_COMBO_MODE != G::COMBO_MODE_GONLY) {
            init_semaphore(combo_bridge_de_done, 0, 1);
        }
    }
    __syncthreads();

    const FrontendG& g_front = reinterpret_cast<const FrontendG&>(g);
    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const bool wg_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);

    nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg_impl<
        FrontendC, true, true, -1, 3>(g_front);

    if (warpgroup_id < 3 && wg_leader) {
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        frontend_done[warpgroup_id] = 1;
    }
    if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY) {
        if (warpgroup_id == 2) {
            if (wg_leader) {
                while (frontend_done[2] != 7) {}
                g.G_sg_row[{0}] = 293.0f;
            }
            return;
        }
        if (warpgroup_id == 0 && wg_leader) {
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            frontend_done[2] = 7;
        }
        return;
    }
    if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY ||
                  STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
        everyone::tma::cluster::arrive_aligned();
        everyone::tma::cluster::wait_aligned();
    }

    if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_GONLY) {
        if (warpgroup_id < 2) {
            return;
        }
        return;
    } else if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY) {
        return;
    } else if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY) {
        if (warpgroup_id < 2) {
            return;
        }
        if (warpgroup_id == 2) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(4);
            issue_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_e_tiles_arrived, combo_dc_p3_e_scales_arrived,
                combo_dc_p3_inputs_finished,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
            wait(combo_bridge_de_done, 0);
        } else if (warpgroup_id == 4) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(6);
            drain_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
            if (wg_leader) {
                arrive(combo_bridge_de_done);
            }
        }
        return;
    } else if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
        if (warpgroup_id < 2) {
            return;
        }
        if (warpgroup_id == 2) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(4);
            issue_combo_de_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_de_tmem_addr, combo_de_tmem_provisioned,
                combo_de_p3_c_tiles_arrived, combo_de_p3_c_scales_arrived,
                combo_de_p3_inputs_finished,
                combo_de_p3_outputs_arrived, combo_de_p3_outputs_finished);
            wait(combo_bridge_de_done, 0);
            issue_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_e_tiles_arrived, combo_dc_p3_e_scales_arrived,
                combo_dc_p3_inputs_finished,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
        } else if (warpgroup_id == 3) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(5);
            drain_combo_de_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_de_tmem_addr, combo_de_tmem_provisioned,
                combo_de_p3_inputs_finished,
                combo_de_p3_outputs_arrived, combo_de_p3_outputs_finished);
            if (wg_leader) {
                arrive(combo_bridge_de_done);
            }
        } else if (warpgroup_id == 4) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(6);
            wait(combo_bridge_de_done, 0);
            drain_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
        }
        return;
    }
}

template <typename C>
__device__ inline void backward_kernel_v6_streaming_5wg_gonly(const globals_5wg<C>& g) {
    backward_kernel_v6_streaming_5wg_bridge<C, globals_5wg<C>::COMBO_MODE_GONLY>(g);
}

template <typename C, int STATIC_COMBO_MODE>
__device__ inline void backward_kernel_v6_streaming_5wg_mode(const globals_5wg<C>& g) {
    static_assert(C::NUM_WARPGROUPS == 5,
                  "v6 dedicated combo entrypoint is reserved for 5-WG configs.");
    backward_kernel_v6_streaming_5wg_bridge<C, STATIC_COMBO_MODE>(g);
}

template <typename C>
__device__ inline void backward_kernel_v6_streaming_5wg(const globals_5wg<C>& g) {
    static_assert(C::NUM_WARPGROUPS == 5,
                  "v6 dedicated combo entrypoint is reserved for 5-WG configs.");
    backward_kernel_v6_streaming_5wg_bridge<C, globals_5wg<C>::COMBO_MODE_FULL>(g);
}

}  // namespace nvfp4_cce_backward_v6
