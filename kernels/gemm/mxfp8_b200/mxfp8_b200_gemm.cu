#include "kittens.cuh"

using namespace kittens;

namespace mxfp8_gemm {

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI, bool _FP8_OUTPUT = false, typename _AB = fp8e4m3, bool _CENTERED_FP8_OUTPUT = false>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0, "LOAD_PIPE_DEPTH must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be greater than 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2, "NUM_D_TILES must be at least 2 if EPI_PIPE_DEPTH > 1");

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;
    static constexpr bool FP8_OUTPUT = _FP8_OUTPUT;
    static constexpr bool CENTERED_FP8_OUTPUT = _CENTERED_FP8_OUTPUT;
    using AB = _AB;

    static_assert(
        !(FP8_OUTPUT && CENTERED_FP8_OUTPUT),
        "plain and centered FP8 output modes are mutually exclusive");
    static_assert(
        !CENTERED_FP8_OUTPUT || _Nb / _EPI_PIPE_DEPTH == 32,
        "centered FP8 output requires one 32-value block per epilogue tile");

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 128;
    static constexpr int B_SC_SIZE = Nb/128;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename C>
struct globals {
    using A_fp8_tile = st<typename C::AB, C::Mb / 2, C::Kb>;
    using A_sc_tile  = st_fp8e8m0<32, 16, false>;
    using B_fp8_tile = st<typename C::AB, C::Nb / 2, C::Kb>;
    using B_sc_tile  = st_fp8e8m0<32, 16, false>;
    using D_tile     = std::conditional_t<
        C::FP8_OUTPUT || C::CENTERED_FP8_OUTPUT,
        st_fp8e4m3<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH, false>,
        st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>
    >;

    using A_gl    = gl<typename C::AB,  1,  1, -1, -1, A_fp8_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;
    using B_gl    = gl<typename C::AB,  1,  1, -1, -1, B_fp8_tile>;
    using B_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, B_sc_tile>;
    using D_gl    = std::conditional_t<
        C::FP8_OUTPUT || C::CENTERED_FP8_OUTPUT,
        gl<fp8e4m3, 1, 1, -1, -1, D_tile>,
        gl<bf16, 1, 1, -1, -1, D_tile>
    >;

    A_gl A;       // M x K
    A_sc_gl A_sc; // (M // 128) x (K // 128) x 32 x 16
    B_gl B;       // N x K
    B_sc_gl B_sc; // (N // 128) x (K // 128) x 32 x 16
    D_gl D;       // M x N
    bf16* D_center = nullptr; // M x (N / 32), row-major BF16 block maxima
    int D_center_stride = 0;

    struct input_tiles_t {
        A_fp8_tile A;
        B_fp8_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        return dim3(min((D.rows()/(C::Mb/2))*(D.cols()/C::Nb), num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
struct scaled_globals {
    globals<C> gemm;
    const float* output_scale;

    __host__ inline dim3 grid() const { return gemm.grid(); }
    __host__ inline dim3 block() const { return gemm.block(); }
    __host__ inline int dynamic_shared_memory() const {
        return gemm.dynamic_shared_memory();
    }
};

template <typename Tile>
__device__ inline void scale_bf16_register_tile(Tile& tile, float scale) {
    static_assert(std::is_same_v<typename Tile::dtype, bf16_2>);
    // Match BF16 TensorIterator in-place multiplication by a CUDA FP32 scalar:
    // the scalar is first cast to the destination dtype, then both BF16 inputs
    // are widened for the arithmetic.  Keeping this conversion explicit is
    // required for bitwise parity at non-power-of-two scales.
    const float bf16_scale = __bfloat162float(__float2bfloat16_rn(scale));
    #pragma unroll
    for (int i = 0; i < Tile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < Tile::width; ++j) {
            #pragma unroll
            for (int k = 0; k < Tile::packed_per_tile; ++k) {
                float2 value = __bfloat1622float2(tile.tiles[i][j].data[k]);
                value.x *= bf16_scale;
                value.y *= bf16_scale;
                tile.tiles[i][j].data[k] = __float22bfloat162_rn(value);
            }
        }
    }
}

template <typename C, bool OUTPUT_SCALE>
__device__ inline void stage_output_tile(
    const globals<C>& g,
    typename globals<C>::D_tile& output_tile,
    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>& D_reg,
    int row_block_idx,
    int col_block_idx,
    int epi,
    int cta_id,
    float output_scale) {
    static_assert(
        !OUTPUT_SCALE || !(C::FP8_OUTPUT || C::CENTERED_FP8_OUTPUT),
        "scaled MXFP8 GEMM supports BF16 output only");
    if constexpr (C::CENTERED_FP8_OUTPUT) {
        using D_bf_t = rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        D_bf_t residual;
        warp::copy(residual, D_reg);
        typename D_bf_t::col_vec center;
        warp::row_max(center, residual);
        warp::sub_row(residual, residual, center);

        rt_fp8e4m3<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_fp8;
        warp::copy(D_fp8, residual);
        warpgroup::store(output_tile, D_fp8);

        const int lane = laneid();
        if ((lane & 3) == 0) {
            const int warp_row_base =
                row_block_idx * C::Mb + cta_id * (C::Mb / 2) +
                warpgroup::warpid() * (C::Mb / 8);
            const int center_col =
                col_block_idx * C::EPI_PIPE_DEPTH + epi;
            #pragma unroll
            for (int outer = 0;
                 outer < decltype(center)::outer_dim;
                 ++outer) {
                const int local_row = outer * 16 + lane / 4;
                const bf16_2 values = center[outer][0];
                g.D_center[
                    static_cast<int64_t>(warp_row_base + local_row) *
                        g.D_center_stride + center_col] =
                    values.x;
                g.D_center[
                    static_cast<int64_t>(warp_row_base + local_row + 8) *
                        g.D_center_stride + center_col] =
                    values.y;
            }
        }
    } else if constexpr (C::FP8_OUTPUT) {
        rt_fp8e4m3<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_fp8;
        warp::copy(D_fp8, D_reg);
        warpgroup::store(output_tile, D_fp8);
    } else {
        if constexpr (OUTPUT_SCALE) {
            // Preserve the existing numerical boundary exactly: tensor-memory
            // accumulators have already been rounded into BF16 D_reg.  Widen
            // each packed BF16 pair to float2, apply the runtime FP32 scalar,
            // and round once more to BF16 before the existing shared/HBM store.
            scale_bf16_register_tile(D_reg, output_scale);
        }
        warpgroup::store(output_tile, D_reg);
    }
}

template <typename C, bool OUTPUT_SCALE>
__device__ inline void kernel_impl(
    const globals<C> &g,
    const float* output_scale_ptr) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp8_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp8_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int num_col_blocks = g.D.cols() / C::Nb;
    const int num_blocks = num_col_blocks * num_row_blocks;
    const int num_iters_per_block = g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Declare tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            // Load input scales to shared memory
            pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx * 2 + cta_id, i, 0, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    if constexpr (C::B_SC_SIZE == 2) tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {col_block_idx * 2 + cta_id, i, 0, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    else if (cta_id == 0)            tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiply
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::LOAD_PIPE_DEPTH>>(384);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                #pragma unroll 2
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*16);
                    load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A);
                    auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*32);
                    load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[0]);
                    if constexpr (C::B_SC_SIZE == 2) {
                        auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*32+16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[1]);
                    }
                    constexpr int tile_transaction_bytes =
                        (std::is_same_v<typename C::AB, fp6e3m2> ||
                         std::is_same_v<typename C::AB, fp6e2m3>)
                            ? (3 * sizeof(G::input_tiles_t)) / 2
                            : 2 * sizeof(G::input_tiles_t);
                    tma::expect_bytes(tiles_arrived[stage], tile_transaction_bytes);
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e8m0<32>>(stage * 32),
                                        inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                update_phasebit<1>(phasebits, 0);
                tensor_commit<2>(outputs_arrived);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // Consumer group
        float output_scale = 1.0f;
        if constexpr (OUTPUT_SCALE) {
            output_scale = *output_scale_ptr;
        }
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Load the output from tensor memory into registers and store to HBM
            if constexpr (C::OVERLAP_EPI) {
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg;
                    warpgroup::load_async(D_reg, out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0
                    }
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    stage_output_tile<C, OUTPUT_SCALE>(
                        g, output_tiles.D[i%C::NUM_D_TILES], D_reg,
                        row_block_idx, col_block_idx, i, cta_id,
                        output_scale);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                }
            } else {
                rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                    warpgroup::load_async(D_reg[i], out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    stage_output_tile<C, OUTPUT_SCALE>(
                        g, output_tiles.D[i%C::NUM_D_TILES], D_reg[i],
                        row_block_idx, col_block_idx, i, cta_id,
                        output_scale);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                }
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_id);
            wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    kernel_impl<C, false>(g, nullptr);
}

template <typename C>
__device__ inline void scaled_kernel(const scaled_globals<C> &g) {
    static_assert(
        !(C::FP8_OUTPUT || C::CENTERED_FP8_OUTPUT),
        "scaled MXFP8 GEMM supports BF16 output only");
    kernel_impl<C, true>(g.gemm, g.output_scale);
}

} // namespace mxfp8_gemm

namespace mxfp8_quantize {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPS = 2; // 64 threads, 2 rows per thread
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

template <typename T>
struct globals {
    static constexpr int TILE_SIZE = 128;   // This should not change
    static constexpr int K_BLOCK_SIZE = 32; // This should not change

    using A_bf16_tile = st_bf<TILE_SIZE, TILE_SIZE, false>;
    using A_data_tile = st<T, TILE_SIZE, TILE_SIZE, false>;
    using A_sc_tile   = st_fp8e8m0<32, 16, false>;

    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using A_data_gl = gl<T, 1, 1, -1, -1, A_data_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;

    A_bf16_gl A_bf16; // M x N
    A_data_gl A_data; // M x N logical elements; FP6 is compact in global memory
    A_sc_gl A_sc;     // (M // 128) x (N // 128) x 32 x 16

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_SIZE, A_bf16.rows() / TILE_SIZE);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TILE_SIZE * TILE_SIZE * sizeof(bf16) + 1024;
    }
};

template <typename T>
__device__ inline void kernel(const globals<T> &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename globals<T>::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<typename globals<T>::A_bf16_tile>();
    typename globals<T>::A_data_tile &A_data_smem = *reinterpret_cast<typename globals<T>::A_data_tile *>(&A_bf16_smem);
    typename globals<T>::A_sc_tile   &A_sc_smem = *reinterpret_cast<typename globals<T>::A_sc_tile *>(
        reinterpret_cast<uint64_t>(&A_data_smem) + sizeof(A_data_smem));

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

    // We have 64 threads per block. Each thread handles 2 rows of 128 elements
    constexpr int ROWS_PER_THREAD = 2;
    constexpr int NUM_K_BLOCKS = globals<T>::TILE_SIZE / globals<T>::K_BLOCK_SIZE; // 4
    constexpr int N_PER_K_BLOCK = globals<T>::TILE_SIZE / 2 / NUM_K_BLOCKS;        // 16
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
                int tile_col = k_block_idx*globals<T>::K_BLOCK_SIZE + ((tid+j)*2)%globals<T>::K_BLOCK_SIZE;
                int offset = (tile_row*globals<T>::TILE_SIZE + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[r][i][j], static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }
    __syncthreads();

    // Perform MXFP8 quantization
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int tile_row = tid + (r*64);
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS; i++) {
            int k_block_idx = (i + tid/8) % NUM_K_BLOCKS; // 8 SMEM banks per K-block

            // Calculate absolute maximum
            bf16_2 amax = __habs2(A_bf16_reg[r][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                amax = __hmax2(amax, __habs2(A_bf16_reg[r][i][j]));

            // Compute scales
            // Must narrow to e8m0, rounding towards positive infinity and saturating to finite, then clamp
            // https://arxiv.org/pdf/2506.08027
            constexpr float max_norm =
                std::is_same_v<T, fp8e4m3> ? 448.0f :
                std::is_same_v<T, fp6e3m2> ? 28.0f : 7.5f;
            float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) / max_norm, 0.000000000001f); // in theory lower clamp is not needed
            A_sc_reg[r][k_block_idx].__x = __nv_cvt_float_to_e8m0(scale, __NV_SATFINITE, cudaRoundPosInf); // causes stack frame, but ignorable
            float scale_inv = 1.0f / static_cast<float>(A_sc_reg[r][k_block_idx]); // utilizes the float() operator defined in __nv_fp8x2_e8m0

            // Quantize and store to shared memory
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                int tile_col = k_block_idx*globals<T>::K_BLOCK_SIZE + ((tid+j)*2)%globals<T>::K_BLOCK_SIZE;
                int offset = (tile_row*globals<T>::TILE_SIZE + tile_col) * sizeof(T);
                T A_data_reg[2] = {
                    T(__bfloat162float(A_bf16_reg[r][i][j].x) * scale_inv),
                    T(__bfloat162float(A_bf16_reg[r][i][j].y) * scale_inv)
                };
                asm volatile("{st.shared.b16 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_data_smem)) + offset)
                       "h"(*reinterpret_cast<uint16_t *>(&A_data_reg[0])));
            }
        }

        // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
        // but we do have to follow this complicated layout pattern made by NVIDIA:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        int scale_offset = (tile_row % 32) * 16 + // row
                           (tile_row / 32) * 4;   // column
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem)) + scale_offset)
               "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[r][0])));
    }

    // Store to global memory
    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_data, A_data_smem, {row, col});
        tma::store_async(G.A_sc,  A_sc_smem,  {row, col, 0, 0});
    }
}

} // namespace mxfp8_quantize

#ifndef TORCH_COMPILE

#include "../common.cuh"

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel_entrypoint(const __grid_constant__ mxfp8_gemm::globals<C> g) {
    mxfp8_gemm::kernel<C>(g);
}

template <typename C>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    using G = mxfp8_gemm::globals<C>;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb 
              << " SUPERGROUP_SIZE=" << C::SUPERGROUP_SIZE << " LOAD_PIPE_DEPTH=" << C::LOAD_PIPE_DEPTH 
              << " EPI_PIPE_DEPTH=" << C::EPI_PIPE_DEPTH << " NUM_D_TILES=" << C::NUM_D_TILES 
              << " OVERLAP_EPI=" << C::OVERLAP_EPI << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = size_t(M) * K + size_t(N) * K + size_t(M) * N * 2;
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_fp8_e4m3*> d_A(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_A_sc(arg_group_count);
    std::vector<__nv_fp8_e8m0*> d_B_sc(arg_group_count);
    std::vector<__nv_bfloat16*> d_D(arg_group_count);
    __nv_bfloat16* d_D_ref;
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_B[i], N*K*sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_A_sc[i], M*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_B_sc[i], N*K*sizeof(__nv_fp8_e8m0)/32);
        cudaMalloc(&d_D[i], M*N*sizeof(__nv_bfloat16));
    }
    cudaMalloc(&d_D_ref, M*N*sizeof(__nv_bfloat16));

    // Initialize matrices with random values on device
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -448.0f, 448.0f);
        fill<__nv_fp8_e4m3, FillMode::RANDOM>(d_B[i], N*K, seed + i*100 + 1, -448.0f, 448.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_A_sc[i], M*K/32, seed + i*100 + 2, 0.1f, 10.0f);
        fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_B_sc[i], N*K/32, seed + i*100 + 3, 0.1f, 10.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_D[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_D_ref, M*N, 0.0f);

    // Compute reference GEMM on device
    reference_blockscaled_gemm<__nv_fp8_e4m3, __nv_fp8_e8m0, __nv_bfloat16, 32>(
        d_D_ref, d_A[0], d_B[0], d_A_sc[0], d_B_sc[0], M, N, K);
    cudaDeviceSynchronize();

    // Prepare kernel inputs
    std::vector<G> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename G::A_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename G::A_sc_gl Asg{d_A_sc[i], M/128, K/128, nullptr, nullptr};
        typename G::B_gl Bg{d_B[i], nullptr, nullptr, N, K};
        typename G::B_sc_gl Bsg{d_B_sc[i], N/128, K/128, nullptr, nullptr};
        typename G::D_gl Dg{d_D[i], nullptr, nullptr, M, N};
        g.push_back(G{Ag, Asg, Bg, Bsg, Dg});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(kernel_entrypoint<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    // Prepare kernel launch configuration
    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, C::CLUSTER_SIZE);

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel_entrypoint<C>, g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Check correctness
    check_correctness(d_D[0], d_D_ref, M * N);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_A_sc[i]);
        cudaFree(d_B_sc[i]);
        cudaFree(d_D[i]);
    }
    cudaFree(d_D_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

int main() {
    int N;
    bool ncu = false;

    // Template parameters: Nb, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP_SIZE, NUM_D_TILES, OVERLAP_EPI
    N = 1024;
    run_benchmark<mxfp8_gemm::config<128, 5, 4, 12, 2, true>>(N, N, N, ncu);
    N = 2048;
    run_benchmark<mxfp8_gemm::config<256, 5, 8, 12, 2, true>>(N, N, N, ncu);
    N = 4096;
    run_benchmark<mxfp8_gemm::config<256, 5, 8, 8, 2, false>>(N, N, N, ncu);
    N = 8192;
    run_benchmark<mxfp8_gemm::config<256, 6, 16, 16, 4, false>>(N, N, N, ncu);
    N = 16384;
    run_benchmark<mxfp8_gemm::config<256, 4, 8, 8, 2, false>>(N, N, N, ncu);

    return 0;
}

#else

#include "pyutils/torchutils.cuh"

template <typename C>
void mxfp8_gemm_entrypoint_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    bf16* D_center = nullptr,
    int D_center_stride = 0
) {
    using G = mxfp8_gemm::globals<C>;

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D),
        .D_center = D_center,
        .D_center_stride = D_center_stride
    };
    kittens::py::launch_kernel<C, G, mxfp8_gemm::kernel<C>>(g);
}

void mxfp8_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    using C = mxfp8_gemm::config<256, 6, 16, 12, 4, false, false>;
    mxfp8_gemm_entrypoint_config<C>(A, A_sc, B, B_sc, D);
}

template <typename C>
void mxfp8_gemm_scaled_entrypoint_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &output_scale
) {
    using BaseG = mxfp8_gemm::globals<C>;
    using G = mxfp8_gemm::scaled_globals<C>;

    BaseG gemm {
        .A = kittens::py::tensor_to_gl<typename BaseG::A_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename BaseG::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename BaseG::B_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename BaseG::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename BaseG::D_gl>(D),
        .D_center = nullptr,
        .D_center_stride = 0
    };
    G g {
        .gemm = gemm,
        .output_scale = output_scale.data_ptr<float>()
    };
    kittens::py::launch_kernel<C, G, mxfp8_gemm::scaled_kernel<C>>(g);
}

void check_mxfp8_output_scale(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    const at::Tensor &D,
    const at::Tensor &output_scale
) {
    TORCH_CHECK(
        output_scale.is_cuda(),
        "output_scale must be a CUDA scalar tensor");
    TORCH_CHECK(
        output_scale.scalar_type() == at::kFloat,
        "output_scale must be float32");
    TORCH_CHECK(
        output_scale.numel() == 1,
        "output_scale must contain one element");
    TORCH_CHECK(
        output_scale.is_contiguous(),
        "output_scale must be contiguous");
    kittens::py::device_check(A, A_sc, B, B_sc, D, output_scale);
}

void mxfp8_gemm_scaled_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &output_scale
) {
    check_mxfp8_output_scale(A, A_sc, B, B_sc, D, output_scale);
    using C = mxfp8_gemm::config<256, 6, 16, 12, 4, false, false>;
    mxfp8_gemm_scaled_entrypoint_config<C>(
        A, A_sc, B, B_sc, D, output_scale);
}

void mxfp8_gemm_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    int64_t config_id
) {
    switch (config_id) {
    case 0:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 6, 16, 12, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 1:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  1, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 2:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  2, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 3:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  4, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 4:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  8, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 5:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4, 16, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 6:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  4, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 7:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  8, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 8:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4, 12, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 9:  mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5,  4, 16, 2, true >>(A, A_sc, B, B_sc, D); break;
    case 10: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 4,  4,  8, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 11: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 4,  4, 12, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 12: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 4,  4, 16, 2, false>>(A, A_sc, B, B_sc, D); break;
    case 13: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 6, 16,  4, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 14: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 6, 16,  8, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 15: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 6, 16, 16, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 16: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5, 16,  4, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 17: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5, 16,  8, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 18: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5, 16, 12, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 19: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 5, 16, 16, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 20: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<256, 6, 16, 12, 4, true >>(A, A_sc, B, B_sc, D); break;
    case 21: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<128, 6,  8,  8, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 22: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<128, 6,  8, 12, 4, false>>(A, A_sc, B, B_sc, D); break;
    case 23: mxfp8_gemm_entrypoint_config<mxfp8_gemm::config<128, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D); break;
    default:
        TORCH_CHECK(false, "invalid MXFP8 GEMM config_id: ", config_id, " (valid: 0-23)");
    }
}

void mxfp8_gemm_scaled_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    const at::Tensor &output_scale,
    int64_t config_id
) {
    check_mxfp8_output_scale(A, A_sc, B, B_sc, D, output_scale);
    switch (config_id) {
    case 0:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 6, 16, 12, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 1:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  1, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 2:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  2, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 3:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  4, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 4:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  8, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 5:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4, 16, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 6:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  4, 2, true >>(A, A_sc, B, B_sc, D, output_scale); break;
    case 7:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4,  8, 2, true >>(A, A_sc, B, B_sc, D, output_scale); break;
    case 8:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4, 12, 2, true >>(A, A_sc, B, B_sc, D, output_scale); break;
    case 9:  mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5,  4, 16, 2, true >>(A, A_sc, B, B_sc, D, output_scale); break;
    case 10: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 4,  4,  8, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 11: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 4,  4, 12, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 12: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 4,  4, 16, 2, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 13: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 6, 16,  4, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 14: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 6, 16,  8, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 15: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 6, 16, 16, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 16: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5, 16,  4, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 17: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5, 16,  8, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 18: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5, 16, 12, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 19: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 5, 16, 16, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 20: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<256, 6, 16, 12, 4, true >>(A, A_sc, B, B_sc, D, output_scale); break;
    case 21: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<128, 6,  8,  8, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 22: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<128, 6,  8, 12, 4, false>>(A, A_sc, B, B_sc, D, output_scale); break;
    case 23: mxfp8_gemm_scaled_entrypoint_config<mxfp8_gemm::config<128, 5,  8, 12, 4, true >>(A, A_sc, B, B_sc, D, output_scale); break;
    default:
        TORCH_CHECK(
            false,
            "invalid scaled MXFP8 GEMM config_id: ", config_id,
            " (valid: 0-23)");
    }
}

void mxfp8_gemm_fp8_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    using C = mxfp8_gemm::config<256, 6, 8, 12, 4, false, true>;
    mxfp8_gemm_entrypoint_config<C>(A, A_sc, B, B_sc, D);
}

template <typename C>
void mxfp8_gemm_centered_fp8_entrypoint_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    at::Tensor &D_center
) {
    static_assert(C::CENTERED_FP8_OUTPUT);
    constexpr int center_block = C::Nb / C::EPI_PIPE_DEPTH;
    TORCH_CHECK(
        D.is_cuda() &&
            D.scalar_type() == at::ScalarType::Float8_e4m3fn &&
            D.dim() == 2 && D.is_contiguous(),
        "centered MXFP8 GEMM data output must be contiguous CUDA E4M3 [M, N]");
    TORCH_CHECK(
        D_center.is_cuda() &&
            D_center.scalar_type() == at::ScalarType::BFloat16 &&
            D_center.dim() == 2 && D_center.is_contiguous(),
        "centered MXFP8 GEMM centers must be contiguous CUDA BF16 [M, N/32]");
    TORCH_CHECK(
        D_center.device() == D.device(),
        "centered MXFP8 GEMM outputs must be on one CUDA device");
    TORCH_CHECK(
        D_center.size(0) == D.size(0) &&
            D_center.size(1) * center_block == D.size(1),
        "centered MXFP8 GEMM center shape mismatch");
    mxfp8_gemm_entrypoint_config<C>(
        A, A_sc, B, B_sc, D,
        reinterpret_cast<bf16*>(D_center.data_ptr()),
        static_cast<int>(D_center.size(1)));
}

void mxfp8_gemm_centered_fp8_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D,
    at::Tensor &D_center
) {
    using C = mxfp8_gemm::config<
        256, 6, 8, 12, 4, false, false, fp8e4m3, true>;
    mxfp8_gemm_centered_fp8_entrypoint_config<C>(
        A, A_sc, B, B_sc, D, D_center);
}

template <typename T>
void mx_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_data,
    at::Tensor &A_sc
) {
    using C = mxfp8_quantize::config;
    using G = mxfp8_quantize::globals<T>;

    const int rows = static_cast<int>(A_bf16.size(0));
    const int cols = static_cast<int>(A_bf16.size(1));
    TORCH_CHECK(A_bf16.dim() == 2 && A_bf16.is_contiguous(),
                "MX quantizer input must be a contiguous 2D tensor");
    TORCH_CHECK(rows % 128 == 0 && cols % 128 == 0,
                "MX quantizer dimensions must be divisible by 128");
    TORCH_CHECK(A_data.dim() == 2 && A_data.is_contiguous() && A_data.size(0) == rows,
                "MX quantizer output must be a contiguous 2D tensor with matching rows");
    if constexpr (std::is_same_v<T, fp6e3m2> || std::is_same_v<T, fp6e2m3>) {
        TORCH_CHECK(A_data.dtype() == at::ScalarType::Byte,
                    "MXFP6 quantizer output must use uint8 packed-U6 storage");
        TORCH_CHECK(A_data.size(1) * 4 == cols * 3,
                    "MXFP6 packed row width must be three bytes per four values");
    } else {
        TORCH_CHECK(A_data.size(1) == cols,
                    "MXFP8 quantizer output must match the input shape");
    }

    G g {
        .A_bf16 = kittens::py::tensor_to_gl<typename G::A_bf16_gl>(A_bf16),
        .A_data = kittens::py::tensor_to_gl<typename G::A_data_gl, false>(A_data, 1, 1, rows, cols),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc)
    };
    kittens::py::launch_kernel<C, G, mxfp8_quantize::kernel<T>>(g);
}

void mxfp8_quantize_entrypoint(
    const at::Tensor &A_bf16,
    at::Tensor &A_fp8,
    at::Tensor &A_sc
) {
    mx_quantize_entrypoint<fp8e4m3>(A_bf16, A_fp8, A_sc);
}

template <typename T, typename C>
void mxfp6_gemm_entrypoint_config(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    static_assert(std::is_same_v<typename C::AB, T>);
    using G = mxfp8_gemm::globals<C>;

    TORCH_CHECK(A.dtype() == at::ScalarType::Byte && B.dtype() == at::ScalarType::Byte,
                "MXFP6 operands must use uint8 packed-U6 storage");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.is_contiguous() && B.is_contiguous(),
                "MXFP6 operands must be contiguous 2D tensors");
    const int M = static_cast<int>(A.size(0));
    const int N = static_cast<int>(B.size(0));
    TORCH_CHECK((A.size(1) * 4) % 3 == 0 && (B.size(1) * 4) % 3 == 0,
                "MXFP6 packed row width must be divisible by three bytes");
    const int K = static_cast<int>(A.size(1) * 4 / 3);
    TORCH_CHECK(static_cast<int>(B.size(1) * 4 / 3) == K, "MXFP6 GEMM K mismatch");
    TORCH_CHECK(K % 128 == 0, "MXFP6 GEMM K must be divisible by 128");
    TORCH_CHECK(D.size(0) == M && D.size(1) == N && D.dtype() == at::ScalarType::BFloat16,
                "MXFP6 GEMM output must be BF16 [M, N]");

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_gl, false>(A, 1, 1, M, K),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_gl, false>(B, 1, 1, N, K),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(D)
    };
    kittens::py::launch_kernel<C, G, mxfp8_gemm::kernel<C>>(g);
}

template <typename T>
void mxfp6_gemm_entrypoint(
    const at::Tensor &A,
    const at::Tensor &A_sc,
    const at::Tensor &B,
    const at::Tensor &B_sc,
    at::Tensor &D
) {
    using C = mxfp8_gemm::config<256, 6, 16, 12, 4, false, false, T>;
    mxfp6_gemm_entrypoint_config<T, C>(A, A_sc, B, B_sc, D);
}

void mxfp6_e3m2_quantize_entrypoint(const at::Tensor &A, at::Tensor &Q, at::Tensor &S) {
    mx_quantize_entrypoint<fp6e3m2>(A, Q, S);
}

void mxfp6_e2m3_quantize_entrypoint(const at::Tensor &A, at::Tensor &Q, at::Tensor &S) {
    mx_quantize_entrypoint<fp6e2m3>(A, Q, S);
}

void mxfp6_e3m2_gemm_entrypoint(const at::Tensor &A, const at::Tensor &AS, const at::Tensor &B, const at::Tensor &BS, at::Tensor &D) {
    mxfp6_gemm_entrypoint<fp6e3m2>(A, AS, B, BS, D);
}

void mxfp6_e2m3_gemm_entrypoint(const at::Tensor &A, const at::Tensor &AS, const at::Tensor &B, const at::Tensor &BS, at::Tensor &D) {
    mxfp6_gemm_entrypoint<fp6e2m3>(A, AS, B, BS, D);
}

void mxfp6_e2m3_gemm_config_entrypoint(
    const at::Tensor &A,
    const at::Tensor &AS,
    const at::Tensor &B,
    const at::Tensor &BS,
    at::Tensor &D,
    int64_t config_id
) {
    using T = fp6e2m3;
    switch (config_id) {
    case 0:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 6, 16, 12, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 1:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 6, 16, 16, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 2:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 6, 16,  8, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 3:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 6, 16,  4, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 4:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 5, 16, 16, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 5:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 5, 16, 12, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 6:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 5, 16,  8, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 7:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 5, 16,  4, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 8:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 6, 16, 12, 4, true,  false, T>>(A, AS, B, BS, D); break;
    case 9:  mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<128, 6,  8,  8, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 10: mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<128, 6,  8, 12, 4, false, false, T>>(A, AS, B, BS, D); break;
    case 11: mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<128, 5,  8, 12, 4, true,  false, T>>(A, AS, B, BS, D); break;
    case 12: mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 5,  4, 12, 2, false, false, T>>(A, AS, B, BS, D); break;
    case 13: mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 5,  4, 12, 2, true,  false, T>>(A, AS, B, BS, D); break;
    case 14: mxfp6_gemm_entrypoint_config<T, mxfp8_gemm::config<256, 4,  4, 12, 2, false, false, T>>(A, AS, B, BS, D); break;
    default:
        TORCH_CHECK(false, "invalid MXFP6 E2M3 GEMM config_id: ", config_id, " (valid: 0-14)");
    }
}

PYBIND11_MODULE(_C_mxfp8, m) {
    m.def("mxfp8_gemm", &mxfp8_gemm_entrypoint);
    m.def("mxfp8_gemm_config", &mxfp8_gemm_config_entrypoint);
    m.def(
        "mxfp8_gemm_scaled",
        &mxfp8_gemm_scaled_entrypoint,
        "MXFP8 GEMM with a CUDA FP32 scalar applied after BF16 rounding",
        pybind11::arg("A"), pybind11::arg("A_sc"),
        pybind11::arg("B"), pybind11::arg("B_sc"),
        pybind11::arg("D"), pybind11::arg("output_scale"));
    m.def(
        "mxfp8_gemm_scaled_config",
        &mxfp8_gemm_scaled_config_entrypoint,
        "Configured MXFP8 GEMM with post-BF16 CUDA FP32 scaling",
        pybind11::arg("A"), pybind11::arg("A_sc"),
        pybind11::arg("B"), pybind11::arg("B_sc"),
        pybind11::arg("D"), pybind11::arg("output_scale"),
        pybind11::arg("config_id"));
    m.def("mxfp8_gemm_fp8", &mxfp8_gemm_fp8_entrypoint);
    m.def(
        "mxfp8_gemm_centered_fp8",
        &mxfp8_gemm_centered_fp8_entrypoint);
    m.def("mxfp8_quantize", &mxfp8_quantize_entrypoint);
    m.def("mxfp6_e3m2_quantize", &mxfp6_e3m2_quantize_entrypoint);
    m.def("mxfp6_e2m3_quantize", &mxfp6_e2m3_quantize_entrypoint);
    m.def("mxfp6_e3m2_gemm", &mxfp6_e3m2_gemm_entrypoint);
    m.def("mxfp6_e2m3_gemm", &mxfp6_e2m3_gemm_entrypoint);
    m.def("mxfp6_e2m3_gemm_config", &mxfp6_e2m3_gemm_config_entrypoint);
}

#endif
