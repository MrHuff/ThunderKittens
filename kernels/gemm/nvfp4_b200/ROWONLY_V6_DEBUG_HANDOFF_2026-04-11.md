## Row-Only `v6` / Dedicated `dC` Handoff

Date: `2026-04-11`

Branch: `fp4_shenanigans`

Workspace state at write time:
- source tree is intentionally dirty and contains the accumulated `v2`/`v3`/`v5`/`v6` branch work
- the built `v6` artifact was restored to the safe default path before finishing this handoff

### Goal

The active line of work is a dedicated row-only `v6` frontend path:
- materialize only `G_row` to HBM
- keep the existing row-based `dE` bridge
- run `dC` on-chip through a dedicated combo back-half WG
- avoid materializing `G_col` / col scales globally

The main reason for the pivot was that the fully fused row+col path was correct but too expensive on-chip:
- col fusion drove register pressure and occupancy collapse
- the split row-only route is the best remaining path to reduce resident HBM and make fused backward viable

### Primary Files

Most active files for this debugging thread:
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v3.cuh`
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v3.cu`
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v6.cu`
- `kernels/gemm/nvfp4_b200/Makefile.cce_backward_v6`

Supporting branch files that also changed during the larger effort:
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v2.cu`
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v2.cuh`
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v5.cu`
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v5_combo_tritonstyle_experimental.cuh`
- `kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v6.cuh`
- `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu`
- `kernels/gemm/nvfp4_b200/V5_NOTES.md`
- `kernels/gemm/nvfp4_b200/Makefile.cce_backward_v3`
- `kernels/gemm/mxfp4_gb200/mxfp4_cce_backward_v2.cu`
- `kernels/gemm/mxfp4_gb200/mxfp4_cce_backward_v3.cu`
- `kernels/gemm/mxfp4_gb200/mxfp4_cce_backward_v3.cuh`
- `include/ops/group/memory/tile/tensor_to_register.cuh`
- `include/ops/thread/mma/tcgen05.cuh`

Non-repo helper used repeatedly for quick runtime smoke:
- `/tmp/rowonly_dconly_case.py`

### What Landed In Source

#### 1. Dedicated row-only wrapper and bridge plumbing

The branch now contains a true row-only experimental path behind:
- `V6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND=1`

Important properties of that route:
- dedicated wrapper instantiates as `DO_ROW=true, DO_COL=false`
- `v6` experimental bridge allocates row outputs only
- internal combo `dC` operand staging is still kept alive even though global `G_col` materialization is removed
- the safe default `v6` path is still the normal bridge when the switch is off

#### 2. Earlier deadlock / teardown fixes already in branch

Across the debugging thread, the row-only route accumulated several real fixes, including:
- true row-only wrapper instead of inheriting the generic `DO_COL=true` shape
- row-only early/late publish cleanup
- skipping deadlocking final `combo_dc_p3_outputs_arrived` behavior for the dedicated route
- `outputs_finished` lifecycle repairs for the dedicated route
- row-only-specific internal col staging kept alive for the `dC` back-half
- multiple stage-reuse / ready / recycle fixes in the shared combo machinery

These were enough to make some small dedicated row-only `dConly` cases live earlier in the thread.

#### 3. Latest direct-load simplification in the single-front-end col path

The newest changes in `nvfp4_cce_backward_v3.cuh` were:
- the dedicated row-only combo `dC` route stops taking the inherited `USE_COL_PAIR_STAGE` path inside the live single-front-end col quant loop
- instead, that route now uses the direct `load_col_value(...)` path

That simplification exposed a real bug, and the fix is also in source now:
- the direct-load branch had stopped writing the internal `combo_col_stage` FP4/scales that the dedicated `dC` back-half still consumes
- `store_combo_col_stage_u64(...)` and `store_combo_col_stage_scale(...)` were restored in both:
  - the fast aligned full-row16 subpath
  - the general direct-load subpath

Related local variables introduced for that fix:
- `local_row_pair_base = local_row_base / 2`
- `local_col = epi * SUBTILE_COLS + col_in_epi`

### Current Runtime State

#### Safe artifact state

Before ending, the default safe `v6` extension was rebuilt without the experimental flag. Direct import of the built module reports:

```python
experimental_rowonly_combo_dc_frontend_enabled == False
```

That means the workspace ends on a safe default runtime artifact even though the source tree still contains the experimental row-only work.

#### Experimental build state

The latest experimental row-only `v6` build compiles cleanly.

Recent ptxas signal:
- dedicated row-only `gonly`: about `112` registers
- dedicated row-only `dconly`: about `128` registers

#### Latest runtime confidence

The latest source fixes compiled, but post-fix runtime validation was blocked by unstable CUDA initialization on this machine:
- intermittent `cudaGetDeviceCount error 304`
- repeated `torch.cuda.is_available() == False`
- repeated `torch.cuda.device_count() == 0`

So the latest source state is compile-verified, but not fully runtime-verified after the newest direct-load + combo-stage-write repair.

### Last Grounded Runtime Observations

Before the latest CUDA flapping became the dominant problem:
- plain experimental `256x256x256` row-only repro reached:
  - `warmup ok`
  - `exp_flag True`
  - `case 256x256x256 launching`
  - `case 256x256x256 launched`
- but it did not reach `synced` in that specific run

Earlier in the thread, several smaller row-only `dConly` cases were made live with intermediate fixes, but the latest direct-load change needs revalidation under a stable CUDA runtime.

### Commands Used Repeatedly

Build the default safe `v6` artifact:

```bash
make -B -j1 -f Makefile.cce_backward_v6 \
  NVCC_GENCODE='-gencode arch=compute_100a,code=sm_100a'
```

Build the experimental dedicated row-only `v6` path:

```bash
make -B -j1 -f Makefile.cce_backward_v6 \
  NVCC_GENCODE='-gencode arch=compute_100a,code=sm_100a' \
  EXTRA_NVCCFLAGS='-DV6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND=1'
```

Build with a debug cut:

```bash
make -B -j1 -f Makefile.cce_backward_v6 \
  NVCC_GENCODE='-gencode arch=compute_100a,code=sm_100a' \
  EXTRA_NVCCFLAGS='-DV6_PUBLICV3_USE_ROWONLY_COMBO_DC_FRONTEND=1 -DROWONLY_COMBO_DC_DEBUG_CUT=200'
```

Run the small dedicated row-only `dConly` repro:

```bash
timeout 40s /workspace/codebases/fp4_matmul/.venv/bin/python -u /tmp/rowonly_dconly_case.py 256 256 256 1
```

Basic CUDA health check:

```bash
nvidia-smi
timeout 20s /workspace/codebases/fp4_matmul/.venv/bin/python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

Clean stale GPU holders when the driver/runtime wedges:

```bash
fuser -k -v /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools
```

Verify the built module flag directly:

```bash
timeout 20s /workspace/codebases/fp4_matmul/.venv/bin/python - <<'PY'
import importlib.util
so='/workspace/codebases/cce/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200/_C_nv_cce_backward_v6.cpython-312-aarch64-linux-gnu.so'
spec=importlib.util.spec_from_file_location('_C_nv_cce_backward_v6', so)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(m.experimental_rowonly_combo_dc_frontend_enabled)
PY
```

### What Still Needs Fixing

#### 1. Revalidate the latest direct-load row-only path

The very next step should be a clean runtime recheck of the current source, because the newest fix changed live dataflow:
- direct-load path replaces inherited `USE_COL_PAIR_STAGE` for the dedicated row-only route
- `combo_col_stage` FP4/scale writes were restored for the direct-load subpaths

This needs fresh runtime confirmation on:
- `256x256x256`
- `256x256x512`
- `512x512x256`

#### 2. If liveness still fails, cut inside the direct-load corridor

If the current source still hangs, the next useful seam is inside the dedicated single-front-end direct-load path rather than the old inherited col-pair-stage path.

Good follow-up cut points:
- just before first `load_col_value(...)`
- just after first `store_combo_col_stage_u64(...)`
- just after first `store_combo_col_stage_scale(...)`
- first handoff point where the dedicated `dC` back-half consumes that internal combo stage

#### 3. Stabilize the CUDA runtime enough to trust results

Right now the machine is a real blocker:
- CUDA init repeatedly fails with `304`
- Torch sometimes loses all devices mid-iteration

Until that stabilizes, it is too easy to mistake driver/runtime churn for kernel regressions.

### Suggested Immediate Validation Order

1. Kill stale GPU holders if needed.
2. Rebuild experimental row-only `v6`.
3. Re-run:
   - `256x256x256`
   - `256x256x512`
   - `512x512x256`
4. If those are live, recheck the first larger case:
   - `512x1024x256`
5. Only then retry hot cases.

### Safety Notes

- No public API changes were intentionally made in this handoff step.
- The safe default `v6` artifact was restored before finishing.
- The source tree still contains the experimental row-only route and the surrounding branch work.

### Continuation: `v2` / `v3` / `v6` Status Check (`2026-04-11 22:49:10 UTC`)

This pass was meant to answer a narrower question than the main row-only debug thread:
- what actually exists today for NV forward and backward across `v2`, `v3`, and `v6`
- what imports cleanly
- what still runtime-blocks on this machine

#### Forward status

The forward side is not symmetric with backward in the current tree.

What is present:
- `Makefile.cce` currently builds `nvfp4_cce.cu` into `_C_nv_cce*.so`
- the built artifact present in-tree is:
  - `kernels/gemm/nvfp4_b200/_C_nv_cce.cpython-312-aarch64-linux-gnu.so`
- that module imports cleanly if `libtorch_python.so` is preloaded first
- current exported symbol set is just:
  - `nvfp4_cce`
- in this continuation pass, a missing `Makefile.cce_v2` was restored and the older versioned forward artifact was rebuilt:
  - `kernels/gemm/nvfp4_b200/Makefile.cce_v2`
  - `kernels/gemm/nvfp4_b200/_C_nv_cce_v2.cpython-312-aarch64-linux-gnu.so`
- the rebuilt `_C_nv_cce_v2` artifact imports cleanly and exports:
  - `pp_L3_SG8`
  - `pp_L4_SG8`

What is missing / inconsistent:
- two forward ABIs now coexist:
  - unversioned `_C_nv_cce` exporting `nvfp4_cce`
  - versioned `_C_nv_cce_v2` exporting `pp_L3_SG8` / `pp_L4_SG8`
- several forward benchmark helpers still hard-code the versioned `_C_nv_cce_v2*.so` path, for example:
  - `fp4_cce_TK/bench_fp4_cce_tk.py`
  - `fp4_cce_TK/bench_cce_fwdbwd.py`
  - `fp4_cce_TK/bench_cce_e2e.py`
  - `fp4_cce_TK/bench_cce_all.py`
- those helpers should now load again at the file level, but the repo still has two different forward entrypoint shapes in circulation

Practical forward status:
- a versioned `v2` forward artifact now exists again and matches the older helper expectations
- no standalone NV forward `v3` or `v6` source/build path was found under `kernels/gemm/nvfp4_b200`
- so a real "forward `v2` vs `v3` vs `v6`" status table is still not meaningful here, but for a different reason now: only `v2` is actually present as a versioned forward path

#### Backward status

Backward is much cleaner structurally.

Built artifacts present:
- `_C_nv_cce_backward_v2.cpython-312-aarch64-linux-gnu.so`
- `_C_nv_cce_backward_v3.cpython-312-aarch64-linux-gnu.so`
- `_C_nv_cce_backward_v6.cpython-312-aarch64-linux-gnu.so`

Import status:
- `v2`: imports cleanly
  - key public exports:
    - `backward_v2_bf16_L4_SG8`
    - `backward_v2_fp4_L4_SG8`
    - `experimental_backward_v2_bf16_epipipe_L4_SG8`
    - `experimental_backward_v2_fp4_epipipe_L4_SG8`
- `v3`: imports cleanly
  - export surface is large (`199` visible exports in this build)
  - public entrypoints include:
    - `backward_v3_bf16_L4_SG8`
    - `backward_v3_fp4_L4_SG8`
  - many experimental col-WG / row-only / col-only variants are present
- `v6`: imports cleanly
  - visible exports are the public-v3-front-half combo family
  - default built flag reports:
    - `experimental_rowonly_combo_dc_frontend_enabled == False`
  - exported combo entrypoints include:
    - `experimental_backward_v6_combo_publicv3_fp4_L4_SG8`
    - `experimental_backward_v6_combo_publicv3_fp4_gonly_L4_SG8`
    - `experimental_backward_v6_combo_publicv3_fp4_dEonly_L4_SG8`
    - `experimental_backward_v6_combo_publicv3_fp4_dConly_L4_SG8`
    - `experimental_backward_v6_combo_publicv3_fp4_5wg_*`

Practical backward status:
- source + built artifacts for `v2` / `v3` / `v6` are all present
- all three import successfully
- `v6` remains on the safe default built artifact in this workspace

#### Runtime recheck status

This machine is not globally broken, but GPU selection matters.

Grounded runtime picture from this pass:
- `nvidia-smi` is clean
- four `GB200` devices enumerate
- no GPU processes were active during the checks
- `torch.cuda.is_available()` returns `True`
- `torch.cuda.device_count()` returns `4`

Device-level allocation probe:
- `cuda:0` fails on the first tiny allocation/sync with:
  - `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
- `cuda:1`, `cuda:2`, and `cuda:3` successfully allocate and synchronize

What that means operationally:
- the earlier "runtime is broken" conclusion was too broad
- the real problem is that default-device probing lands on a bad GPU (`0`) unless the run is pinned away from it
- for this box, meaningful status checks should explicitly set:
  - `CUDA_VISIBLE_DEVICES=1`
  - or another known-good device (`2` / `3`)

GPU-1 rerun results:
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v3-exp-check --warmup 0 --iters 1`
  - completed successfully
  - results:
    - `256x512x256`: row fp4/sc and col fp4/sc all `True`
    - `512x512x256`: row fp4/sc and col fp4/sc all `True`
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v6-publicv3-combo-check --warmup 0 --iters 1`
  - did not complete within `240s`
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-check --warmup 0 --iters 1`
  - did not complete within `180s`

GPU-1 timing datapoints (`warmup=0`, `iters=1`, so these are cold single-shot numbers, not steady-state throughput):
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v2-phase-breakdown --config-label '4Kx4K->32K' --warmup 0 --iters 1`
  - `bwd=0.532 ms`
  - `quant=0.195 ms`
  - `dE=0.264 ms`
  - `dC=0.218 ms`
  - `full=1.065 ms`
  - `tail=0.482 ms`
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --only nv-v3-fused --config-label '4Kx4K->32K' --warmup 0 --iters 1`
  - `202.089 ms`
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --only nv-v3-dE --config-label '4Kx4K->32K' --warmup 0 --iters 1`
  - `0.335 ms`
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --only nv-v3-dC --config-label '4Kx4K->32K' --warmup 0 --iters 1`
  - `0.335 ms`

All-device smoke matrix:
- `device 0`
  - tiny alloc/sync: fail
  - `--v3-exp-check`: fail in `ensure_cuda_ready()`
- `device 1`
  - tiny alloc/sync: pass
  - `--v3-exp-check`: pass
- `device 2`
  - tiny alloc/sync: pass
  - `--v3-exp-check`: pass
- `device 3`
  - tiny alloc/sync: pass
  - `--v3-exp-check`: pass

More localized `v6` signal:
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-direct-mode dConly --warmup 0 --iters 1`
  - reaches:
    - `setup_start`
    - `inputs_ready`
    - `quant_ready`
    - `lse_ready`
    - `direct nv-v6-pv35dC`
    - `launching`
  - then stalls after launch
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-direct-mode gonly --config-label '4Kx4K->32K' --warmup 0 --iters 1`
  - reaches:
    - `setup_start`
    - `inputs_ready`
    - `quant_ready`
    - `lse_ready`
    - `direct nv-v6-pv35g`
    - `launching`
    - `launched`
  - then stalls after launch return
- `CUDA_VISIBLE_DEVICES=1 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-direct-mode dEonly --config-label '4Kx4K->32K' --warmup 0 --iters 1`
  - reaches:
    - `setup_start`
    - `inputs_ready`
    - `quant_ready`
    - `lse_ready`
    - `direct nv-v6-pv35dE`
    - `launching`
  - then stalls before the host reports `launched`
- the same host-side behavior is still seen for `dConly` in the direct probe:
  - reaches `launching`
  - does not print `launched`

Important nuance:
- the bench script header text is misleading for the `v6` flags because the header-selection branch does not include them
- the execution path still reaches the `v6` logic later, so the timeout results above are still meaningful

#### Current bottom line

If the question is "what is the status right now":
- forward:
  - both `_C_nv_cce` and `_C_nv_cce_v2` are now present
  - `_C_nv_cce_v2` was rebuilt in this continuation pass and restores the old helper load target
  - no standalone forward `v3` / `v6` path was found here
- backward:
  - `v2`, `v3`, and `v6` all exist as built modules and import cleanly
  - `v6` default artifact is still the safe non-rowonly build
- runtime:
  - GPU `0` is bad for allocation/sync on this machine
  - GPUs `1`, `2`, and `3` are good enough to run checks
  - public/experimental `v3` check passes on GPUs `1`, `2`, and `3`
  - `v6` combo and `v6` 5WG paths still fail to complete on GPU `1`

#### Recommended next steps from this status pass

1. Fix or work around the CUDA allocation/sync failure first; until then, kernel status checks are noise.
2. Pin all follow-up validation away from GPU `0`:
   - use `CUDA_VISIBLE_DEVICES=1`
   - or `2` / `3`
3. Treat `v3` as currently healthy on the small exactness checks and focus the next debugging cuts on `v6` only.
4. For `v6`, the most grounded current seam is the 5WG `dConly` path after the direct launch point:
   - `setup_start`
   - `inputs_ready`
   - `quant_ready`
   - `lse_ready`
   - `direct nv-v6-pv35dC`
   - `launching`
   - then no forward progress
5. Use the direct-mode distinction between `gonly` and `dE`/`dC`:
   - `gonly` reaches `launched` and then stalls
   - `dEonly` / `dConly` do not report `launched`
   - so `dE` / `dC` are failing even earlier than `gonly`
6. If forward coverage matters in the same sweep, decide which forward ABI is canonical now:
   - keep `_C_nv_cce` as the active path and update the helper scripts
   - or keep `_C_nv_cce_v2` as the compatibility path and accept that the repo has two forward frontends

### Continuation: public `v3` / bridged `v6` rebuild sweep (`2026-04-12 01:30 UTC`)

What changed in this pass:
- rebuilt `_C_nv_cce_backward_v3.cpython-312-aarch64-linux-gnu.so` from current source after fixing the compile-only `globals_3wg` aggregate wiring
- rebuilt `_C_nv_cce_backward_v6.cpython-312-aarch64-linux-gnu.so`
- changed the default public frontend alias in `nvfp4_cce_backward_v6.cu` to `experimental_config_colwg_consumerotf_warpbalanced<4, 8, true, 4>`
- changed `launch_backward_v3_fp4_public_dispatch_L4_SG8()` to the same consumer-OTF warp-balanced frontend
- fixed the temporary 3WG combo placeholder tensors in `nvfp4_cce_backward_v3.cu` from invalid `1x1` BF16 scratch to `max(M, N) x max(M, N)` BF16 scratch so rebuilt public `v3` no longer dies during TMA descriptor creation

Concrete runtime status after those rebuilds:
- `v6` is now launch-stable on the new default frontend
  - direct non-5WG `gonly` returns cleanly
  - direct non-5WG `combo` returns cleanly
  - direct `5wg gonly` also returns cleanly
  - `--v6-publicv3-combo-check` passes on both small shapes
- rebuilt public `v3` no longer hangs or throws the earlier TMA descriptor error
  - `--v3-exp-check` now runs to completion

But the rebuilt public `v3` is still functionally wrong:
- small raw-check against the rebuilt experimental warp-balanced frontend:
  - `256x512x256`: `row fp4=True`, `row sc=True`, `col fp4=False`, `col sc=False`
  - `512x512x256`: `row fp4=True`, `row sc=True`, `col fp4=False`, `col sc=False`
- large `4Kx4K->32K` end-to-end timing on GPU `1`:
  - `v2 = 1.061 ms`
  - `v3 = 11.122 ms`
  - `v3ov = 11.107 ms`
  - `v3 bwd = 10.626 ms`
  - `v3 dE = 0.205 ms`
  - `v3 dC = 0.193 ms`
  - `cos(dE) = 0.9940`
  - `cos(dC) = 0.4990`

Interpretation:
- the new consumer-OTF warp-balanced frontend solved the launch-stability problem
- it did not solve correctness
- row materialization is fine
- col materialization is not fine
- because `v6` is currently using this rebuilt public-frontend path as its bridge source, `v6` is only validated relative to the rebuilt public chain, not relative to the original correct `v2` target

Candidate search notes from this pass:
- `experimental_backward_v3_fp4_colwg_consumerotf_warpbalanced_L4_SG8` matches the rebuilt public path exactly, so it is not an alternate fix; it is the current broken behavior
- the older/heavier col-WG variants that were probed next did not return quickly enough to promote immediately:
  - `consumerotf`
  - `colwg`
  - `plainstage`
  - `bf16cache`
  - `rowhwfp4_row16ready_overlap`

Current bottom line after this rebuild sweep:
- `v6` launch stability is improved
- rebuilt public `v3` is still not correct on the col path
- `v6` should not be considered fixed yet because its current bridge reference inherits that rebuilt public-frontend behavior

Most grounded next step:
1. Stop treating launch stability and correctness as the same problem.
2. Keep the placeholder-size fix in `nvfp4_cce_backward_v3.cu`; that one was real and necessary.
3. Revert the public frontend alias away from `consumerotf_warpbalanced` once a correct candidate is identified.
4. Use the existing `--v3-exp-...-check` modes to keep narrowing candidates, but judge them against `v2`-level `dC` behavior, not just against the rebuilt public path.

### Continuation: restored-good `v3` artifact + `v6` bridge recheck (`2026-04-12 03:10 UTC`)

What was changed in this pass:
- kept the current-source rebuilds on disk, but restored the live runtime `v3` module by copying the known-good `737fa9df` build back onto:
  - `_C_nv_cce_backward_v3.cpython-312-aarch64-linux-gnu.so`
- preserved the current-source rebuilt `v3` artifact as:
  - `_C_nv_cce_backward_v3.current_apr12_source_rebuild.so`
- patched `nvfp4_cce_backward_v6.cu` so the standard public-`v3` bridge frontend initializes the combo-side fields the same way the standalone current `launch_experimental_backward_v3_fp4_3wg()` helper does
- rebuilt `_C_nv_cce_backward_v6.cpython-312-aarch64-linux-gnu.so`

Runtime facts that are now established:
- the restored-good live `v3` artifact gives the expected end-to-end numbers again on GPU `2`
  - `4Kx4K->32K`, `warmup=2`, `iters=5`
  - `v2 = 2.998 ms`
  - `v3 = 4.358 ms`
  - `v3ov = 4.352 ms`
  - `v3 bwd = 2.969 ms`
  - `v3 dE = 0.200 ms`
  - `v3 dC = 0.189 ms`
  - `cos(dE) = 0.9940`
  - `cos(dC) = 0.9939`
- current-source `v3` is still not fixed at the source level
  - several current-source col-WG candidates byte-match the restored-good module on `256x512x256`
  - but the same candidates do not complete the large `4Kx4K->32K` fused run
  - confirmed examples:
    - `colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap`
    - `colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap`
    - `colpair_rowpair_rowrecordpad_rowsync_dualfloatcache_row16ready_overlap`
    - `colpairpad_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap`
    - `colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowrcp_row16ready_overlap`
- standard non-5WG `v6` is still not functionally correct against the restored-good `v3`
  - `--v6-publicv3-combo-check` on both small shapes still reports:
    - `row fp4 = True`
    - `row sc = True`
    - `col fp4 = False`
    - `col sc = False`
    - `dE = True`
    - `dC = False`
  - direct `gonly` isolation confirms the frontend itself is already wrong on the col side:
    - `experimental_backward_v6_combo_publicv3_fp4_gonly_L4_SG8`
    - `row fp4 = True`
    - `row sc = True`
    - `col fp4 = False`
    - `col sc = False`
- the helper-parity patch in `nvfp4_cce_backward_v6.cu` did not change that outcome
- rebuilt standard `v6` timing is still essentially unchanged:
  - `4Kx4K->32K`, `warmup=2`, `iters=5`
  - `nv-v6-pv3cmb = 26.308 ms`
  - `nv-v6-pv3dE = 26.159 ms`
  - `nv-v6-pv3dC = 26.151 ms`

Interpretation after this pass:
- there are now two separate truths:
  - operational `v3` can be made healthy immediately by using the restored-good artifact
  - current-source `v3` is still regressed at scale and needs a real source rollback or deeper kernel surgery
- standard `v6` is not just slow; it is wrong on the col path even when checked against the restored-good `v3`
- because `v6 gonly` already mismatches `G_col`, the remaining bug is in the `v6` public-front-half path itself, not only in the bridged `dC` tail

Most useful next step from here:
1. Treat the restored-good `v3` artifact as the runtime baseline for all further measurements.
2. Stop spending cycles on current-source `v3` candidate sweeps until the source-level rollback plan is clearer; small exactness alone is not predictive.
3. Debug `v6` from `gonly` first, because that is the smallest reproducer that already shows the bad col materialization.
4. Compare the `v6` `gonly` frontend launch against the exact standalone `v3` path that produced the restored-good artifact, not against the current-source helper alone.

### Continuation: public-frontend identity check + isolated row/col large-shape probes (`2026-04-12 03:03 UTC`)

What was established in this pass:
- the old known-good `737fa9df` source did **not** wire public `v3` to `consumerotf_warpbalanced`
  - old `launch_backward_v3_fp4_public_dispatch_L4_SG8()` dispatches to:
    - `experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap`
  - current source dispatches to:
    - `experimental_config_colwg_consumerotf_warpbalanced`
- this corrects the earlier assumption that the restored-good live `v3` behavior was explained by the consumer-OTF public alias alone

What was tried:
- temporarily changed the default `v6` bridge frontend in `nvfp4_cce_backward_v6.cu` from consumer-OTF to the legacy rowhwfp4 public frontend
- rebuilt `v6`
- result:
  - small `--v6-publicv3-combo-check` / `--v6-publicv3-combo-5wg-check` no longer returned promptly
  - large `4Kx4K->32K` direct `gonly` still reached `launched` and then stalled
- that experiment was reverted
- live `v3` artifact was restored again from the old good `.so`
- `v6` was rebuilt back to the prior consumer-OTF bridge default

Isolated current-source large-shape probes:
- temporarily swapped the current-source rebuilt `v3` `.so` back into the live path only for isolate testing
- `consumerotf_warpbalanced_rowonly` is healthy at large shape
  - GPU `1`
  - `4Kx4K->32K`, `warmup=2`, `iters=5`
  - `nv-v3-cotfwbr = 2.690 ms`
- large-shape col-only probes timed out at `90s`
  - `rowhwfp4_row16ready_overlap_colonly`
  - `colpair_rowpair_lanepairrecord_rowsync_dualfloatcache_row16ready_overlap_colonly`
  - `colpair_rowpair_rowrecord_rowsync_dualfloatcache_row16ready_overlap_colonly`

Interpretation after these isolates:
- the current-source large-shape regression is now much more specifically localized
  - row-side materialization can still run at scale
  - col-side materialization is the part that broadly stops making forward progress
- a hybrid `v6` bridge of:
  - consumer-OTF `rowonly`
  - plus a different current-source `colonly`
  is not a viable quick fix, because every promising current-source large col-only candidate tested here timed out
- the remaining grounded direction is to recover the old-good col path itself, not just to keep swapping current-source public aliases

Old-good large-shape isolate targets (after restoring the known-good live `v3` `.so`):
- GPU `1`, `4Kx4K->32K`, `warmup=2`, `iters=5`
  - `rowhwfp4_rowonly = 0.506 ms`
- GPU `2`, `4Kx4K->32K`, `warmup=2`, `iters=5`
  - `rowhwfp4_colonly = 1.863 ms`

Why these matter:
- they give a concrete target for the current-source recovery effort
- the issue is no longer just “current col hangs”
- the issue is “old-good col-only completes in about `1.863 ms`, while current-source col-only candidates do not complete within `90s`”

### Continuation: `4wg` replay-only vs quantizer-participation isolate (`2026-04-12 11:58 UTC`)

This pass used the preserved current-source rebuild:
- `_C_nv_cce_backward_v3.current_apr12_source_rebuild.so`

and compared it directly against the old known-good `737fa9df` artifact on GPU `2`.

The key new result is that current-source `v3` has a second grounded regression outside the rowhwfp4 col-only thread:
- a plain non-combo `4wg` path still replays correctly
- but the same config wedges as soon as the quantizer-side path participates

Concrete reproducer:
- function family:
  - `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_*_L4_SG8`
- important config properties:
  - dedicated row + col quantizer WGs (`4wg`)
  - `DEBUG_DISABLE_ROW_QUANT_WORK = true`
  - `DEBUG_DISABLE_COL_QUANT_WORK = true`
  - `DEBUG_DISABLE_CONSUMER_COL_STAGE_PRODUCTION = true`

Current-source results:
- `4Kx4K->32K`
  - `..._replayonly_L4_SG8`
  - launched and completed
  - `elapsed_s = 0.202`
  - `row_sum = 8189034805`
  - `col_sum = 0`
  - `colsc_sum = 0`
- `4Kx4K->32K`
  - `..._rowonly_L4_SG8`
  - reached `launching -> launched`
  - then timed out at `45s`
- `4Kx4K->32K`
  - `..._rowwaitskip_rowonly_L4_SG8`
  - reached `launching -> launched`
  - then timed out at `45s`
- `4Kx4K->32K`
  - `..._rowrecycleskip_rowonly_L4_SG8`
  - reached `launching -> launched`
  - then timed out at `45s`
- `256x256->512`
  - `..._rowonly_L4_SG8`
  - reached `launching -> launched`
  - then timed out at `20s`

Old-good comparison:
- `4Kx4K->32K`
  - `..._rowonly_L4_SG8`
  - launched and completed
  - `elapsed_s = 0.184`
  - `row_sum = 8189034805`
  - `col_sum = 0`
  - `colsc_sum = 0`

Interpretation:
- this is not just a large-shape block-reuse problem; the current-source `4wg` stubbed row-only path already wedges on the smallest one-block case
- this is not a producer/replay problem; replay-only is healthy on the same current-source build
- this is not explained just by the first obvious waits:
  - skipping row-ready wait does not recover it
  - skipping row-recycle wait does not recover it
- the regression starts after replay-only hands off to the generic quantizer-participation path, even when quant work itself is stubbed out

What this means for the broader thread:
- current-source `v3` has at least two grounded regressions now:
  - single-frontend rowhwfp4 family: large-shape col-side non-progress
  - multi-quantizer `4wg` family: quantizer-participation path wedges even with stubbed work
- this `4wg` result does not directly fix the `v6` row-only bridge, but it does show the current shared `backward_kernel_v3_streaming_3wg` quantizer-side control flow is no longer trustworthy as a general source baseline

Most useful next step after this isolate:
1. Treat the live old-good `v3` artifact as the only trustworthy end-to-end timing baseline.
2. Keep `v6` debugging on the smallest `gonly` repro against that restored-good baseline.
3. If source-level `v3` recovery is resumed, split the work:
   - rowhwfp4 single-frontend col-path regression
   - `4wg` multi-quantizer quantizer-participation regression

### Continuation: paired wait-skip isolates (`2026-04-12 18:33 UTC`)

This pass turned the earlier single-wait probes into paired skip probes, built as separate debug artifacts so the live good `v3` `.so` stayed untouched.

Built artifacts:
- `_C_nv_cce_backward_v3.rowskipboth_apr12.cpython-312-aarch64-linux-gnu.so`
- `_C_nv_cce_backward_v3.rowhwfp4_rowskipboth_apr12.cpython-312-aarch64-linux-gnu.so`
- `_C_nv_cce_backward_v3.rowhwfp4_colskipboth_apr12.cpython-312-aarch64-linux-gnu.so`

Loader note for any repeat:
- these custom filenames still export `PyInit__C_nv_cce_backward_v3`
- direct Python loads must therefore force the module name:
  - `spec_from_file_location("_C_nv_cce_backward_v3", so_path)`

#### `4wg_bothstub_nocolprod`: paired row-wait skip is real

Config family:
- `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_4wg_bothstub_nocolprod_*_L4_SG8`

New current-source debug config:
- `..._rowskipboth_*`
- source hook:
  - `DEBUG_SKIP_ROW_READY_WAIT = true`
  - `DEBUG_SKIP_ROW_RECYCLE_WAIT = true`

Small shape (`256x256->512`, GPU `2`):
- `rowskipboth_replayonly`
  - completed
  - `elapsed_s = 0.204`
  - `row_sum = 8191436`
  - `col_sum = 0`
  - `colsc_sum = 0`
- `rowskipboth_rowonly`
  - completed
  - `elapsed_s = 0.201`
  - `row_sum = 8191436`
  - `col_sum = 0`
  - `colsc_sum = 0`

Large shape (`4Kx4K->32K`, GPU `2`):
- same rebuilt artifact, original current-source `..._rowonly`
  - still reaches `launching -> launched`
  - still times out at `30s`
- `rowskipboth_replayonly`
  - completed
  - `elapsed_s = 0.225`
  - `row_sum = 8584027111`
  - `row_md5 = cc256fa7ed9f58db67d47976e65789da`
  - `row_sc_md5 = 7eb24d865a14fa3227633816800522c1`
- `rowskipboth_rowonly`
  - completed
  - second hot run in same process: `elapsed_s = 0.023`
  - `row_sum = 8584027111`
  - `row_md5 = cc256fa7ed9f58db67d47976e65789da`
  - `row_sc_md5 = 7eb24d865a14fa3227633816800522c1`
- old-good `737fa9df` `..._rowonly`
  - completed
  - `elapsed_s = 0.199`
  - `row_sum = 8584027111`
  - `row_md5 = cc256fa7ed9f58db67d47976e65789da`
  - `row_sc_md5 = 7eb24d865a14fa3227633816800522c1`

Interpretation:
- the `4wg` current-source row wedge is specifically recoverable by skipping both row waits together
- skipping only one row wait was not enough
- this is not just “it returns now”; the recovered row output byte-matches both:
  - current-source `rowskipboth_replayonly`
  - old-good `737fa9df` `rowonly`

So for this `4wg` family, the row-side semaphore pair is now the grounded culprit, not just a random codegen perturbation.

#### Public `rowhwfp4` large-shape col-only: deeper than obvious waits

Target family:
- `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_*_colonly_L4_SG8`

Existing public-path facts before this pass:
- base `rowhwfp4_colonly` hangs at large shape
- single `colwaitskip` did not recover it
- single `colrecycleskip` did not recover it

New paired-skip probes on `4Kx4K->32K`, GPU `2`:
- base `rowhwfp4_colonly` from the rebuilt debug artifact
  - reached `launching -> launched`
  - timed out at `30s`
- new `rowhwfp4_rowskipboth_colonly`
  - reached `launching -> launched`
  - timed out at `60s`
- new `rowhwfp4_colskipboth_colonly`
  - reached `launching -> launched`
  - timed out at `60s`

Interpretation:
- the public `rowhwfp4` large-shape col-only stall is not explained by:
  - row-ready wait alone
  - row-recycle wait alone
  - both row waits together
  - col-ready wait alone
  - col-recycle wait alone
  - both col waits together

That means the public col-path regression is deeper than the obvious semaphore waits. The next useful debug cut is no longer “toggle more wait bits”; it is instrumentation after launch inside the `rowhwfp4` col path itself:
- col-pair / lane-pair record production
- row-pair-to-col consumption ordering
- phasebit publication/consumption beyond the first ready/recycle gates

Bottom line after this pass:
- `4wg` regression: row-side paired-wait bug, now localized and reproducible with a byte-stable recovery
- public `rowhwfp4` regression: still unresolved, and now proven to be beyond the first obvious wait contracts

#### Tiny public `rowhwfp4` col-only trace: stall is before row16 publish

Timestamp:
- `2026-04-12 14:06:23 UTC`

Tiny-shape target:
- `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_*_colonly_L4_SG8`
- shape: `M=256, N=256, K=512`
- device: `GPU 2`

Sanity baseline on old-good live artifact (`737fa9df` build):
- `rowhwfp4_colonly`
  - `launching -> launched -> synced`
  - `elapsed_s = 0.169`
  - `row_sum = 0`
  - `row_sc_sum = 0`
  - `col_sum = 1542311`
  - `col_sc_sum = 516096`
  - `col_md5 = 74c8cbebbf57b8a30dab09a04e7e8ae8`
  - `col_sc_md5 = afe020786b83b793c2bbd9468097ff6e`

Current-source tiny probes before trace:
- `rowhwfp4_collegacy_colonly`
  - `launching -> launched`
  - timed out
- `rowhwfp4_colcut1_colonly`
  - `launching -> launched`
  - timed out
- `rowhwfp4_colcut2_colonly`
  - `launching -> launched`
  - timed out
- `rowhwfp4_colcut3_colonly`
  - `launching -> launched`
  - timed out

Interpretation from those pre-trace probes:
- restoring the old direct HBM col-store path was not enough to recover the tiny repro
- the existing `colcut*` early-return plumbing is not a trustworthy localization tool here because it still does not unwind the CTA cleanly

New `coltrace` probe:
- built a separate module:
  - `_C_nv_cce_backward_v3.rowhwfp4_coltrace_apr12.cpython-312-aarch64-linux-gnu.so`
- export:
  - `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_colonly_L4_SG8`

First trace cut:
- added trace prints for:
  - `step=1` after first `slice_col_ready_row16` wait
  - `step=2` after first `col_amax`
  - `step=3` after first col FP4 + scale store
  - `step=4` after `slice_col_done_row16[bf_stage][row16_block] = 1`
  - `step=5` after post-epilogue `warpgroup::sync(quantizer_sync_id)`
  - `step=6` after `arrive(slice_col_recycled[bf_stage])`
- tiny run result:
  - printed `launching`
  - printed `launched`
  - printed `step=1` on both CTAs
  - printed `step=2` on both CTAs
  - printed `step=3` on both CTAs
  - never printed `step=4`
  - timed out

Second trace cut:
- forced rebuild after adding finer checkpoints:
  - `step=31` immediately before the post-store `__syncwarp()`
  - `step=32` immediately after that `__syncwarp()`
  - `step=33` immediately after `__threadfence_block()` and before `slice_col_done_row16[...] = 1`
- tiny run result:
  - again printed only `step=1`, `step=2`, `step=3`
  - never printed `step=31`
  - timed out

Grounded interpretation:
- the stall is earlier than the row16 publish tail itself
- at least one traced lane in each CTA finishes the first col store path and reaches `step=3`
- but the warp never reaches even the first instruction immediately after that store block (`step=31`)
- so the current wedge is inside the remaining store-side / same-iteration control-flow, not in:
  - `slice_col_done_row16` publication
  - the post-store `__syncwarp()`
  - the post-publish `threadfence`

Comparison against the old-good source:
- the alias itself did not drift
- `experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap` still has the same key traits in current source and `737fa9df`:
  - `USE_COL_PAIR_STAGE = true`
  - `COL_PAIR_STAGE_PADDED_LAYOUT = true`
  - `USE_ROW_PAIR_STAGE = true`
  - `ROW_PAIR_STAGE_ROWRECORD = true`
  - `ROW_PAIR_STAGE_WARP_SYNC_ONLY = true`
  - `ROW_PAIR_STAGE_FLOATCACHE = true`
  - `PACK_COL_FP4_U64 = true`
  - `ROW_PAIR_STAGE_LANEPAIR_LAYOUT = true`
  - `ROW_QUANT_USE_HW_FP4X2 = true`
  - `COL_READY_PER_ROW16 = true`
- the active base-path branch selection is also unchanged between current source and `737fa9df`:
  - `CACHE_COL_VALUES = true`
  - `CACHE_COL_VALUES_BF16 = false`
  - `CACHE_COL_VALUES_BF16_PAIRS = false`
  - `FAST_ALIGNED_QUANT = false`
- the old-good source path around this tiny repro is still materially simpler:
  - direct `col_fp4_ptr[...] = quantize_fp4_pair(...)`
  - direct FP8 scale byte store
  - then the same `__syncwarp()` / `slice_col_done_row16` tail
- because the `collegacy` probe still hangs, the new `maybe_store_global_col_*` wrappers are not by themselves sufficient to explain the regression

Next useful cut:
- instrument or temporarily simplify the active store-side path *before* the post-store `__syncwarp()`:
  - start with the exact branch actually taken by this config:
    - `CACHE_COL_VALUES = true`
    - `CACHE_COL_VALUES_BF16_PAIRS = false`
    - `FAST_ALIGNED_QUANT = false`
    - `PACK_COL_FP4_U64 = true`
  - isolate `PACK_COL_FP4_U64` vs per-byte store behavior on this exact branch
  - confirm whether a non-traced lane is hanging in the per-pair store loop rather than after it

#### Additional tiny public `rowhwfp4` col-only branch toggles: all still hang

Follow-up probe module:
- `_C_nv_cce_backward_v3.rowhwfp4_colbyte_apr12.cpython-312-aarch64-linux-gnu.so`

New exact-config exports:
- `..._colbyte_colonly_L4_SG8`
  - forces `PACK_COL_FP4_U64 = false`
- `..._colbytelegacy_colonly_L4_SG8`
  - forces `PACK_COL_FP4_U64 = false`
  - also restores the old direct col quant/store path
- `..._colnocache_colonly_L4_SG8`
  - forces:
    - `CACHE_COL_VALUES = false`
    - `CACHE_COL_VALUES_BF16 = false`
    - `CACHE_COL_VALUES_BF16_PAIRS = false`

Tiny-shape results on `GPU 2`, `M=256, N=256, K=512`:
- `colbyte_colonly`
  - `launching -> launched`
  - timed out
- `colbytelegacy_colonly`
  - `launching -> launched`
  - timed out
- `colnocache_colonly`
  - `launching -> launched`
  - timed out

Interpretation:
- packed-u64 HBM stores are not the sole blocker
- direct-vs-helper col HBM store choice is not the sole blocker
- cached-col-values vs reload-on-demand is not the sole blocker

Current grounded exclusion list for the tiny public `rowhwfp4 colonly` wedge:
- not explained by the obvious row waits
- not explained by the obvious col waits
- not explained by:
  - helper `maybe_store_global_col_*` vs legacy direct stores
  - packed-u64 vs byte HBM col stores
  - `CACHE_COL_VALUES` enabled vs disabled

That leaves the next useful cut narrower:
- instrument the exact active row16 store loop at per-lane granularity, especially:
  - before/after `load_col_pair_stage_pair`
  - before/after the `pair` loop
  - before/after the scale-byte store
- or run the tiny repro under a CUDA sanitizer if available, to catch invalid memory / barrier divergence in non-traced lanes

#### April 12 follow-up: corrected `coltrace` interpretation, producer double-publish found, partial source fix

Important correction to the earlier trace read:
- the previous inference from missing `step=31/32/33/4` was wrong for `colonly`
- those markers sit under:
  - `if constexpr (DO_ROW && G::ROW_WAITS_FOR_COL_DONE_ROW16) { ... }`
- the `rowhwfp4 colonly` launches use `DO_ROW = false`, so absence of `step=31/4` never localized the wedge

Re-run details with widened `public_colonly_trace`:
- tiny repro:
  - `GPU 2`
  - `M=256, N=256, K=512`
  - module: `_C_nv_cce_backward_v3.rowhwfp4_coltrace_apr12.cpython-312-aarch64-linux-gnu.so`
  - export: `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_colonly_L4_SG8`
- result before any source fix:
  - pass `0` completed the active cached-values branch:
    - all traced lanes reached `step=40/41/42/43/44`
  - producer-side trace showed the same CTA publishing `slice_col_ready_row16` from both site `B` and site `D`
  - the producer published row16 blocks `0..7`
  - the consumer still only logged `step=1/2/3` for the first pass and never reached `step=5/6`

Key source diff vs old-good `737fa9df`:
- old-good source has the site `B` row16-ready publish path
- current source had an additional later block:
  - `if constexpr (C::EARLY_COL_READY && G::COL_READY_PER_ROW16 && DO_COL && !C::ROW_QUANT_FROM_REGS) { ... arrive(slice_col_ready_row16[bf_stage][published_row16_block]); }`
- that later site `D` block does not exist in `737fa9df`
- for `rowhwfp4_row16ready_overlap`, both site `B` and site `D` were active, so current source was double-publishing the same per-row16 ready semaphores

Applied source change:
- removed the extra site `D` `slice_col_ready_row16` publish block from `nvfp4_cce_backward_v3.cuh`
- rebuilt:
  - trace probe: `_C_nv_cce_backward_v3.rowhwfp4_coltrace_apr12.cpython-312-aarch64-linux-gnu.so`
  - current source artifact: `_C_nv_cce_backward_v3.current_apr12_source_rebuild.so`

Post-fix trace result:
- producer trace now shows only site `B`
- consumer trace advances beyond the first pass:
  - it now reaches `step=1/2/3` for `row16=4`
- this means the duplicate site `D` publish was a real regression and fixing it removes the original immediate pass-0 deadlock

But the source fix is not complete yet:
- uninstrumented current-source `rowhwfp4 colonly` still times out after `launching -> launched`
- traced current-source path still does not reach `step=5/6`
- so there is at least one more later wedge after the second-pass handoff

Current best localization:
- fixed:
  - duplicate per-row16 ready publish at site `D`
- still broken:
  - later `rowhwfp4 colonly` progress after the second-pass handoff
- next useful cut:
  - instrument which row16 block / participating quant thread fails after `row16=4`
  - or compare the remaining post-pass sync / recycle logic around the public col-only quantizer epilogue against `737fa9df`

#### April 12 follow-up: valid trace-cut now reaches the col quantizer recycle sync; kernel still hangs

I corrected one bad assumption in the earlier trace-cut work:
- the helper `public_colonly_release_recycled_and_cut()` was previously inert under `coltrace`
- reason:
  - `PUBLIC_COLONLY_STAGE_CUTS_ENABLED` depends on `DEBUG_PUBLIC_COLONLY_STAGE_CUT > 0`
  - the trace probe only sets `DEBUG_PUBLIC_COLONLY_TRACE = true`
- I widened the helper gate so trace mode actually executes the debug cut path

Tiny repro:
- `GPU 2`
- `M=256, N=256, K=512`
- module:
  - `_C_nv_cce_backward_v3.rowhwfp4_coltrace_apr12.cpython-312-aarch64-linux-gnu.so`
- export:
  - `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_colonly_L4_SG8`

New trace result with the now-live cut path:
- pass `1` still reaches the same store boundary:
  - all 8 traced leaders print `public col pass1 store-done`
- after that, both CTAs on `wg=1` print:
  - `public col cut presync cta=0 wg=1`
  - `public col cut presync cta=1 wg=1`
  - `public col cut postsync cta=0 wg=1`
  - `public col cut postsync cta=1 wg=1`
- the kernel still times out

What this means:
- the remaining wedge is no longer inside the col-quantizer WG's immediate `warpgroup::sync(quantizer_sync_id)` after pass `1`
- the col-quantizer WG can enter the debug recycle path and get past that sync
- the surviving hang is in a peer path that the cut does not retire
  - most likely the other WG (`wg=0`) / producer-consumer cleanup side

Useful structural clue from follow-up patch attempt:
- I tried to advance the recycle phasebit from the same quantizer-side debug cut
- that does not compile:
  - `slice_col_recycle_phasebits` is not in scope at the quantizer-side helper / pass-local trace site
- this is consistent with the remaining stuck path being owned in a different scope than the col-quantizer WG's local recycle-arrival code

Current best next cut:
- instrument the peer `wg=0` path after `site=B` publish and before its next stage-reuse wait
- specifically target waits that consume:
  - `slice_col_recycled`
  - its associated recycle-phase bookkeeping
- do not spend more time on the col-quantizer store loop itself unless a new trace disproves the current cut result

#### April 12 follow-up: `wg=0` clears the first recycle wait; remaining hang is later

I followed the previous section by instrumenting the consumer-side `wait_for_quant_stage()` path that uses:
- `wait(slice_col_recycled[bf_stage], get_phasebit<1>(slice_col_recycle_phasebits, bf_stage))`

Probe result on the same tiny repro:
- before any `site=B` publish, both CTAs on `wg=0` print:
  - `public col consumer prewait cta=0 wg=0 bf=0 epi=0`
  - `public col consumer prewait cta=1 wg=0 bf=0 epi=0`
- both CTAs also print the matching postwait:
  - `public col consumer postwait cta=0 wg=0 bf=0 epi=0`
  - `public col consumer postwait cta=1 wg=0 bf=0 epi=0`
- later, both CTAs on `wg=1` still complete the pass-1 debug cut:
  - `public col pass1 cut presync ...`
  - `public col pass1 cut done ...`
- kernel still times out

What this rules out:
- the surviving hang is not:
  - the first `wg=0` recycle wait for `bf=0, epi=0`
  - the `wg=1` pass-1 debug cut sync
  - the immediate debug recycle-arrival path that follows that sync

Tighter localization now:
- `wg=0` and `wg=1` both make visible progress past the first producer/consumer handshake
- the remaining wedge is later:
  - likely a later `epi` / `bf_stage` reuse wait
  - or another cleanup / tail path after the first public-col handoff, rather than the first pass-1 col store boundary

Next useful cut from here:
- widen the `wg=0` wait trace from only `bf=0, epi=0` to later `epi` values
- if possible, trace:
  - second-stage `wait_for_quant_stage()` calls
  - the next wait after `site=B` publish on `wg=0`
- avoid re-instrumenting the pass-1 col quantizer loop unless a later trace regresses the current result

#### April 12 follow-up: natural tiny repro stalls first at `wg=0 bf=0 epi=2`; quantizer tail still never prints

I removed the trace-only early return and reran the natural tiny repro on:
- `GPU 2`
- `M=N=256, K=512`
- module:
  - `_C_nv_cce_backward_v3.rowhwfp4_coltrace_apr12.cpython-312-aarch64-linux-gnu.so`
- export:
  - `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_colonly_L4_SG8`

New result:
- `wg=0` clears the first two consumer recycle waits on both CTAs:
  - `bf=0, epi=0`: `prewait -> postwait`
  - `bf=1, epi=1`: `prewait -> postwait`
- the first consumer wait that does not clear is:
  - `public col consumer prewait cta=0 wg=0 bf=0 epi=2`
  - `public col consumer prewait cta=1 wg=0 bf=0 epi=2`
- on the quantizer side, the same run still reaches the pass-1 store boundary:
  - all traced leaders print `public col pass1 store-done`
- but no quantizer-tail markers ever print before timeout:
  - no `public col tail presync`
  - no `public col tail postsync`
  - no `public col recycle arrive`

What this means:
- the first clean natural stall remains:
  - `wg=0` reusing `bf_stage=0` for `epi=2`
- the missing wake is still upstream of the tail publish/recycle block
- the kernel is hanging before the final quantizer-side tail sync / recycle arrival, not at the already-instrumented tail itself

#### April 12 follow-up: source-side build break from trace edit fixed; `qtid=0` finishes all `epi` row16 loops

While adding a post-loop marker, I temporarily broke the source build by dropping one closing brace in the `run_single_frontend_quantizer_wg` branch before the `is_col_quantizer_wg` `else if`.

Fix:
- restored the missing close so the active branch structure is again:
  - finish `row16_pass` loop
  - optional `cut 203`
  - exit the `run_single_frontend_quantizer_wg` branch

After rebuilding the trace probe cleanly, I widened the `row16 loop done` marker to print for all `epi` values.

Result on the same tiny repro:
- both CTAs print:
  - `public col row16 loop done cta=0 wg=1 qtid=0 bf=0 epi=0`
  - `public col row16 loop done cta=1 wg=1 qtid=0 bf=0 epi=0`
  - `public col row16 loop done cta=0 wg=1 qtid=0 bf=1 epi=1`
  - `public col row16 loop done cta=1 wg=1 qtid=0 bf=1 epi=1`
  - `public col row16 loop done cta=0 wg=1 qtid=0 bf=0 epi=2`
  - `public col row16 loop done cta=1 wg=1 qtid=0 bf=0 epi=2`
  - `public col row16 loop done cta=0 wg=1 qtid=0 bf=1 epi=3`
  - `public col row16 loop done cta=1 wg=1 qtid=0 bf=1 epi=3`
- the kernel still times out
- there are still no tail markers:
  - no `public col tail presync`
  - no `public col tail postsync`
  - no `public col recycle arrive`

Tighter localization now:
- at least the traced quantizer leader (`qtid=0`, both CTAs) finishes the active row16 quantization loop for all four `epi` stages
- so the surviving wedge is later than the main public-col row16 loop
- but earlier than the final quantizer tail sync / recycle arrival

Best next cut from here:
- instrument the post-epi quantizer cleanup path between:
  - `public col row16 loop done`
  - `public col tail presync`
- likely candidates are the cleanup / combo / fused-col-flush path(s) after the `epi` loop, rather than the row16 quantization loop itself

#### April 12 follow-up: tile-valid combo dummy fixes the tiny host-side descriptor bug, but not the col-only kernel wedge

The current-source `globals_3wg` now requires the combo fields even for `3wg colonly`, but old-good `737fa9df` did not populate those fields at all in `launch_experimental_backward_v3_fp4_3wg_colonly`.

That made the tiny `rowhwfp4 colonly` launcher sensitive to whatever dummy BF16 tensor we used for:
- `dE_out`
- `dC_out`

The temporary `1x1` dummy was invalid for the combo TMA tile:
- `combo_dC_tile = st_bf<C::Nb, C::Nb / C::EPI_PIPE_DEPTH>`
- on the active config that means `128 x 32`

Launcher fix:
- changed the `3wg colonly` dummy combo output from `1x1` BF16 to:
  - `{C::Nb, C::Nb / C::EPI_PIPE_DEPTH}`
- rationale:
  - this keeps the combo TMA descriptor legal
  - while still forcing `combo_num_k_blocks = g.dC_out.cols() / C::Nb = 32 / 128 = 0`

Validation:
- built a fresh probe module:
  - `_C_nv_cce_backward_v3.rowhwfp4_coltrace_tilevalid_apr12.cpython-312-aarch64-linux-gnu.so`
- reran the same tiny natural repro on `GPU 2`

New result:
- the host-side TMA descriptor failure is gone
- the kernel again reaches the same runtime trace frontier as before:
  - `public col row16 loop done ... epi=0..3`
  - `public col postloop flags ... combo_de=0 combo_dc=0 sep=0 fused_sm=0 fused_colq=0 col_in_q=1 dedicated=0`
- it still times out
- and it still does **not** print:
  - `public col post combo block`

What this means:
- the `1x1` dummy was a real launcher bug for tiny `3wg colonly`
- but it was not the root cause of the surviving `rowhwfp4 colonly` hang
- after removing that launcher bug, the wedge is still inside the shared post-loop block:
  - after `public col postloop flags`
  - before `public col post combo block`

`v6` design note to preserve while debugging from `v3`:
- the `v6` target remains:
  - never instantiate the softmax in BF16
  - materialize row and col directly to FP4
- so any eventual `v6` fix should preserve that direction rather than reintroduce a BF16 softmax staging path

#### April 12 follow-up: corrected late-trace gating shows the shared post-loop block completes; the surviving public col-only wedge is later

The earlier late-trace conclusion was polluted by a probe-side gating bug.

For this public `rowhwfp4 colonly` path:
- `G::FRONTEND_SINGLE_QUANTIZER_WG = true`
- the active frontend quantizer leader is `first_quantizer_wg` (`wg=1` on the tiny repro)
- so the late markers should not have been keyed off `is_col_quantizer_wg` or `first_col_quantizer_wg`

I corrected the probe-only late-trace leader gating and rebuilt:
- `_C_nv_cce_backward_v3.rowhwfp4_coltraceleader2_apr12.cpython-312-aarch64-linux-gnu.so`

Tiny natural repro (`256x256->512`, `GPU 2`) on the corrected probe:
- still reaches the full quantizer loop frontier for both CTAs:
  - `public col row16 loop done ... epi=0..3`
  - `public col postloop flags ... combo_de=0 combo_dc=0 sep=0 fused_sm=0 fused_colq=0 col_in_q=1 dedicated=0`
- and now, with the corrected leader gating, also reaches for both CTAs / all four `epi`s:
  - `public col enter shared block`
  - `public col late shared block`
  - `public col post combo block`
- it still times out
- and it still does **not** print:
  - `public col pre tail`
  - `public col tail presync`
  - `public col tail postsync`
  - `public col recycle arrive`

Important interpretation:
- the lack of `public col pre fused branch` is not informative here, because that marker sits under:
  - `if constexpr (G::FUSED_SOFTMAX_QUANT && G::FUSED_COL_IN_QUANTIZER && COL_IN_QUANTIZER)`
- and the active public path proves `fused_sm=0` / `fused_colq=0`
- so the fused branch is compile-time dead on this path

Updated localization:
- the surviving public `rowhwfp4 colonly` wedge is **not** in the row16 quantizer loop
- it is **not** in the shared combo block either
- the current runtime gap is:
  - after `public col post combo block`
  - before the next reachable marker `public col pre tail`
- source window to inspect next:
  - roughly [nvfp4_cce_backward_v3.cuh](/workspace/codebases/cce/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v3.cuh:14565)
  - through [nvfp4_cce_backward_v3.cuh](/workspace/codebases/cce/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_cce_backward_v3.cuh:14864)

Practical next cut:
- trace or simplify the non-fused, non-combo post-combo cleanup path immediately before `public col pre tail`
- do **not** spend another round on the fused-col branch for this public repro; it is currently compile-time off

#### April 13 follow-up: late-cut probes showed the real blocker is WG0, not the frontend WG

I implemented the planned late public-col stage cuts (`colcut5..8`) plus trace-wrapped `coltracecut5/6`, built a sidecar probe module:
- `_C_nv_cce_backward_v3.rowhwfp4_colcutslate_apr13.cpython-312-aarch64-linux-gnu.so`

Tiny natural repro (`256x256->512`, `GPU 2`) results:

`coltracecut5`:
- `wg=1` reaches:
  - `public col post combo block`
  - `public col cut5 wg arrive`
  - `public col cut presync/postsync`
- the kernel still times out

Interpretation:
- the cut helper is not a clean global escape at this late point
- but it does prove the frontend/quantizer WG reaches the post-combo boundary

`coltracecut6`:
- `wg=1` reaches `public col post combo block` for all `epi=0..3`
- it still never prints `public col pre tail`
- the kernel times out

I then added per-WG late prints. That changed the picture materially:
- only `wg=1` reaches:
  - `public col shared block wg arrive`
  - `public col post combo wg arrive`
  - `public col cut5 wg arrive`
- `wg=0` never reaches the shared-block boundary at all

So the surviving hang is not primarily “frontend WG post-combo cleanup” anymore. The real blocker is earlier:
- `wg=0` stalls before the shared-block handoff
- `wg=1` is just the part we happened to be tracing first

I then pushed the `wg=0` probe deeper through the active consumer mailbox path. Verified checkpoints:
- `wg=0` reaches final-epi recycle completion:
  - `public col consumer postwait ... bf=1 epi=3`
- `wg=0` reaches the final-epi row16-ready publish tail:
  - `public col consumer after siteB ... bf=1 epi=3`
- `wg=0` reaches the final-epi recycle phase update:
  - `public col consumer after recycle phasebit ... bf=1 epi=3`
- the kernel still times out
- and `wg=0` still never reaches the shared-block boundary

Updated localization:
- current source `rowhwfp4 colonly` hang is in `wg=0` inside the active `USE_COL_PAIR_STAGE` / `EARLY_COL_READY` consumer path
- the surviving gap is now:
  - after final-epi `site B`
  - after final-epi `slice_col_recycle_phasebits` update
  - before the end of that consumer col-pair branch / before `public col consumer colpair branch done`
  - and therefore also before `public col shared block wg arrive`

Final confirmation from the corrected branch-end probe:
- I moved the `public col consumer colpair branch done` checkpoint to the actual end of the active `USE_COL_PAIR_STAGE` consumer branch
- rebuilt the sidecar probe
- reran the same tiny `coltracecut5` repro
- `public col consumer after recycle phasebit` still prints
- `public col consumer colpair branch done` still does **not** print

Practical next cut:
- inspect the `wg=0` consumer tail immediately after the `DO_COL && C::EARLY_COL_READY` recycle-phase update in the active `USE_COL_PAIR_STAGE` branch
- do not spend the next round on `wg=1` `pre tail`; that was a secondary symptom

--- April 13 late follow-up

I corrected the `public col consumer colpair branch done` checkpoint placement, rebuilt the same sidecar probe, and reran the tiny natural repro.

Confirmed result:
- `wg=0` **does** reach:
  - `public col consumer colpair branch done ... bf=1 epi=3`
- `wg=0` still does **not** reach any later exit-side probes from the successfully rebuilt probe binary:
  - `public col consumer pre/post rowregs sync`
  - `public col consumer pre/post plain sync`
  - `public col consumer pre/post rowpair sync`
  - `public col consumer after generic recycle`
  - `public col after consumer section`
- `wg=1` behavior is unchanged:
  - it reaches `public col pre shared block`
  - it reaches `public col shared block wg arrive`
  - it reaches `public col post combo block`
  - the kernel still times out overall

Updated localization:
- the surviving `wg=0` wedge is now tighter:
  - after the active `USE_COL_PAIR_STAGE` consumer branch returns to the outer flow
  - before any of the later consumer-exit sync markers or the generic recycle update
- in practice this leaves the gap between:
  - `public col consumer colpair branch done`
  - and the first unconditional consumer tail block around line `11120`

Current source note:
- I started one more probe pass to bracket that `11120` arrive block directly (`pre/post arrive block` markers), but I interrupted the rebuild after a long PTXAS pass to avoid spending another full compile cycle without a new runtime result.
- So the live, last-verified probe result is the one above; the very latest source edits are present in `nvfp4_cce_backward_v3.cuh` but were not runtime-validated yet.

--- April 13 corrected colonly-path follow-up

I corrected a bad assumption in the late probe chain.

What was wrong:
- the tiny repro uses `backward_kernel_v3_streaming_3wg_colonly`
- the last two probe rounds were instrumenting an earlier consumer branch in the generic `3wg` body, not the active late `colonly` path that owns:
  - `public col post combo block`
  - `public col pre tail`
  - `public col tail presync/postsync`

What I re-verified on the active sidecar:
- tiny repro still hangs on:
  - `experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_coltrace_colonly_L4_SG8`
- filtered trace still shows only:
  - `public col post combo block ... epi=0`
  - `public col post combo block ... epi=1`
  - `public col post combo block ... epi=2`
  - `public col post combo block ... epi=3`
- and never shows:
  - `public col pre tail`
  - `public col tail presync`
  - `public col tail postsync`

Cross-check with the cut variant:
- `coltracecut6` also times out on the same tiny repro
- it prints the same final marker set as plain `coltrace`
- so the active wedge is still before the stage-6 tail gate in the real `3wg_colonly` body

Current best localization:
- active source hang is in the `3wg_colonly` path between:
  - `public col post combo block`
  - and the first `public col pre tail` print
- the earlier `11120` consumer-tail theory was from the wrong function body and should not be used as the next debug target

Current source state:
- `nvfp4_cce_backward_v3.cuh` now has new probe prints in the correct late path:
  - `public col after cut5 block`
  - `public col before pre tail gate`
- I started the rebuild for that corrected probe, but stopped it before completion after confirming the previous probe thread was chasing the wrong function body.

Next cut:
- rebuild the corrected sidecar probe and rerun the tiny trace
- if `after cut5 block` prints but `before pre tail gate` does not, the remaining wedge is inside the cleanup band immediately after the cut-5 block
- if both print, then the failure is at or immediately before the existing `pre tail` site

--- April 13 late-trace + cut51 follow-up

I added a probe-only late-trace mode to the minimal `rowhwfp4 colonly` sidecar and used it to remove the old `printf`-volume ambiguity.

Probe-only source changes:
- `nvfp4_cce_backward_v3.cuh`
  - added `DEBUG_PUBLIC_COLONLY_LATE_TRACE`
  - kept only the late markers alive in that mode
  - added `gap1..gap5` markers across the straight-line cleanup band after `public col after cut5 block`
  - exposed `public col cut presync/postsync` in late-trace builds when a public-col stage cut is actually active
- `nvfp4_cce_backward_v3_probe_rowhwfp4_colonly.cu`
  - exported:
    - `..._collatetrace_colonly_L4_SG8`
    - `..._collatetracecut5_colonly_L4_SG8`
    - `..._collatetracecut51_colonly_L4_SG8`
    - `..._collatetracecut6_colonly_L4_SG8`

Tiny repro used for all checks:
- GPU `2`
- `M=256, N=256, K=512`
- forward quant via `_C.cpython-312-aarch64-linux-gnu.so`
- backward probe via `_C_nv_cce_backward_v3_probe_rowhwfp4_colonly.cpython-312-aarch64-linux-gnu.so`

Base late-trace result:
- the active `wg=1` frontend leader now cleanly proves:
  - `public col after cut5 block`
  - `public col gap1`
- it still never reaches:
  - `public col pre fused branch`
  - `public col gap2`
  - `public col before pre tail gate`
- the config banner still says:
  - `combo_de=0`
  - `combo_dc=0`
  - `sep=0`
  - `fused_sm=0`
  - `fused_colq=0`
  - `col_in_q=1`
  - `dedicated=0`

What that means:
- the old “maybe this is just `printf` truncation” theory is no longer tenable
- the late stream really stops after `gap1`
- but `gap1` sits immediately before the `if constexpr (G::FUSED_SOFTMAX_QUANT && G::FUSED_COL_IN_QUANTIZER && COL_IN_QUANTIZER)` region, while the banner still claims that region should be compile-time dead
- so the remaining paradox is now explicit: either
  - the fused-region block is somehow still the live wedge, or
  - the kernel-wide hang is dominated by another warpgroup before the later `wg=1` prints can be observed

Cut results:
- `collatetracecut5`
  - prints through `public col post combo block`
  - never reaches `public col after cut5 block`
  - still hangs overall
- `collatetracecut51`
  - prints:
    - `public col after cut5 block`
    - `public col gap1`
    - `public col cut presync`
    - `public col cut postsync`
  - still hangs overall
- so even an immediate cut after `gap1` does not make the tiny repro return

Why cut51 matters:
- it proves the `wg=1` quantizer path is not the only live problem
- by the time `wg=1` reaches `gap1`, some other work is already unrecoverably stalled

Full-trace `cut51` cross-check:
- I reran `coltracecut51` and filtered to the old consumer markers
- `wg=0` still progresses through:
  - `public col consumer prewait/postwait` for `epi=0,1,2,3`
  - `public col consumer after siteB` at `epi=3`
  - `public col consumer after recycle phasebit`
  - `public col consumer colpair branch done`
- and then nothing later appears:
  - no `public col consumer pre rowregs sync`
  - no `post rowregs sync`
  - no `pre/plain sync`
  - no `pre rowpair sync`
  - no `pre arrive block`
  - no `after consumer section`

Updated localization:
- kernel-wide hang is now best understood as:
  - `wg=1` can reach the late cleanup band up to `gap1`
  - but `wg=0` is still wedged in the consumer tail after `public col consumer colpair branch done`
- therefore the missing later `wg=1` tail prints are not sufficient evidence that `wg=1` is the primary blocker
- the next real debug target is back on the `wg=0` consumer-exit path, specifically the first tail sync/recycle block after `public col consumer colpair branch done`

---

## April 13: consumer recycle wait pinned to `bf=0, epi=2`

I kept the tiny natural repro on GPU `2` and narrowed the live base hang further with late-only probe prints in the slim `rowhwfp4 colonly` module.

New late-trace facts on the uncut base path:
- all four warps of the active quantizer WG (`wg=1`) reach:
  - `public col gap1 warp`
  - `public col gap1b warp`
- the same uncut base path still never shows:
  - `public col gap2`
  - `public col recycle arrive`
- but `wg=1` does still reach:
  - `public col nextblock enter ... block=1`

So the important update is:
- `wg=1` is not dead in block `0`; it gets through block `0` far enough to start block `1`
- the real live wedge in the base repro is still `wg=0`

Late consumer-tail traces on the same uncut base path:
- `wg=0` prints:
  - `public col consumer prewait ... bf=0 epi=0`
  - `public col consumer postwait ... bf=0 epi=0`
  - `public col consumer prewait ... bf=1 epi=1`
  - `public col consumer postwait ... bf=1 epi=1`
  - `public col consumer prewait ... bf=0 epi=2`
- it never prints:
  - `public col consumer postwait ... bf=0 epi=2`
  - any later `wg=0` final-tail marker

That pins the live base hang to:
- `wg=0`
- block `0`
- the `wait(slice_col_recycled[bf_stage], ...)` reuse wait
- specifically at `bf_stage=0`, `epi=2`

Interpretation:
- the stalled handoff is now very specific: the third `epi` pass is waiting for the stage-`0` col recycle event or its expected phase
- this is no longer a vague “late cleanup band” problem

Old-good source comparison:
- detached old-good tree: `/workspace/codebases/tk_v3_old_737fa9df/.../nvfp4_cce_backward_v3.cuh`
- old-good has a `warpgroup::sync(quantizer_sync_id);` immediately before the quantizer leader publishes:
  - `arrive(slice_col_recycled[bf_stage])`
- current source had lost that sync

What I changed in source:
- restored that missing pre-publish quantizer sync in current `nvfp4_cce_backward_v3.cuh`

What happened after restoring the sync:
- no behavioral change on the tiny base repro
- it still stalls at:
  - `public col consumer prewait ... bf=0 epi=2`
- and still does not show a matching `public col recycle arrive` before timeout

Current best hypothesis:
- the remaining bug is a bad `slice_col_recycled[0]` handoff
- either the publish never happens for the `epi=0 -> epi=2` reuse cycle
- or the consumer-side `slice_col_recycle_phasebits` expectation is wrong for that reuse

Most useful next cut from here:
- compare the active consumer-side `slice_col_recycle_phasebits` updates against old-good in the exact `DO_COL && C::EARLY_COL_READY && G::COL_READY_PER_ROW16` path
- or add a one-shot probe that proves whether the single-quantizer `arrive(slice_col_recycled[0])` site is actually reached before the blocked `epi=2` wait

---

## April 13: siteA fix holds, remaining wedge moves into the `epi=2` consumer body

I kept the tiny natural repro on GPU `2` and stayed on the slim probe module. The source now includes the real siteA fix:

- in the narrow public single-quantizer col-only row16-ready path, the early consumer-side
  `update_phasebit<1>(slice_col_recycle_phasebits, bf_stage)` is skipped
- the restored old-good `warpgroup::sync(quantizer_sync_id)` before `arrive(slice_col_recycled[bf_stage])` is still present

What changed in behavior after the siteA fix:

- the original base wait bug is fixed:
  - `public col consumer postwait ... bf=0 epi=2` now prints
- the kernel still times out

New trace facts from the updated base late trace:

- `wg=0` still reaches:
  - `public col consumer prewait/postwait ... bf=0 epi=0`
  - `public col consumer prewait/postwait ... bf=1 epi=1`
  - `public col consumer prewait/postwait ... bf=0 epi=2`
- `wg=1` still reaches:
  - `public col gap1`
  - `public col gap1b`

Important correction from the later cut-`105` probe:

- I added targeted late markers for:
  - `public col epi body enter`
  - `public col colpair branch enter`
  - `public col producer loop enter`
  - `public col producer loop i-done`
  - `public col siteB ...`
- on the live repro, the only new late marker that appears is:
  - `public col epi body enter cta=0/1 wg=0 bf=0 epi=2`
- none of these later cut-`105` markers appear:
  - `public col colpair branch enter`
  - `public col producer loop enter`
  - any `public col siteB ...`

At the same time, other consumer warps keep advancing:

- `public col consumer after recycle phasebit ... bf=1 epi=3` prints for nonzero warps
- `public col consumer colpair branch done ... bf=1 epi=3` prints for nonzero warps
- warp `0` is still missing from those late `epi=3` prints

Updated localization:

- the surviving source hang is no longer:
  - the `bf=0, epi=2` recycle wait
  - the late `siteB` publish block
  - the immediate consumer tail after `public col consumer colpair branch done`
- the live warp-`0` wedge is now best localized to the main consumer `epi=2` body:
  - after `wait_for_quant_stage(bf_stage=0, epi=2)` returns
  - after `public col epi body enter`
  - before the later tail branch markers (`colpair branch enter` / `producer loop enter`)

Practical interpretation:

- the remaining bug moved upstream once the bad siteA phasebit update was removed
- the next useful split is inside the `epi=2` consumer compute band roughly between:
  - the `D_fl` / `D_bf` softmax-materialization body starting near line `8336`
  - the later tail branch selection near line `9300`

This is the current best handoff point. The next probe should place one or two coarse warp-`0` late markers inside that `8336 -> 9300` band rather than adding more tail-only prints.

## 2026-04-13 follow-up: consumer barrier-id probe

New probe-only change:

- added a config-level consumer sync-id override
- built `consumersync3` sidecars for the public `rowhwfp4 row16ready_overlap` `colonly` path

Result:

- plain `consumersync3 colonly` still times out on the tiny natural repro
  - command: `CUDA_VISIBLE_DEVICES=2 timeout 30s python3 /tmp/v3_probe_run.py experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_colonly_L4_SG8`
  - result: `RC:124`

But the barrier-id conclusion is still real:

- `collatetracecut105 + consumersync3` returns cleanly
  - command: `CUDA_VISIBLE_DEVICES=2 timeout 25s python3 /tmp/v3_probe_run.py experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_collatetracecut105_consumersync3_colonly_L4_SG8`
  - result: `RC:0`, `PY_DONE`
- in that returning probe, `block=1` fully executes:
  - `block=1 bf=0 epi=2`: `pre/post loadsync`, `body branch enter`, `colpair branch enter`, `siteB done`
  - `block=1 bf=1 epi=3`: `pre/post loadsync`, `body branch enter`, `colpair branch enter`, `siteB done`
  - final line: `public col consumer phase flipped ... phase=0`, then `PY_DONE`

Interpretation:

- switching the consumer load-block sync off barrier `1` is necessary
- it is not sufficient by itself for a quiet kernel
- the traced build proves the same source path can complete once the original barrier collision is removed
- the remaining bug is now timing-sensitive / ordering-sensitive, not a hard functional impossibility in the block-1 body

Additional probe-only fixes that did **not** make the quiet kernel return:

- `consumersync3 + post generic recycle sync`
- `consumersync3 + post generic recycle sync + post consumer-section sync`

Both still time out on the same tiny repro (`RC:124`).

Current best localization:

- there are two separable issues now:
  1. hard barrier-id problem at the consumer load-block sync
     - `sync(3)` resolves it
  2. a remaining timing/order bug in the quiet kernel
     - trace-heavy `cut105` runs finish
     - quiet `consumersync3` runs do not

Recommended next cut:

- compare the current `rowhwfp4` consumer loop directly against `737fa9df` at the anchored sites:
  - `CONSUMER_WG_SYNC_ID`
  - `warpgroup::load_async(D_fl)` / `tensor_load_wait()` / `tensor_before_thread_sync()`
  - both `update_phasebit<1>(slice_col_recycle_phasebits, bf_stage)` sites
- do not assume the remaining failure is fixed by more generic tail syncs; the two explicit post-recycle/post-section sync probes did not solve the quiet hang.

## 2026-04-13 follow-up: consumer tail cleared, quantizer tail still live

New probe-only changes:

- added trace-only markers around the consumer tail:
  - `consumer pre/post deprovision`
- added probe exports for a `cut106` experiment after `consumer final postsync`
- added trace-only markers around the quantizer tail:
  - `quant final presync`
  - `quant final postsync`
  - `quant branch end`

Results:

- quiet `consumersync3 + quantizersync2` still times out
  - command: `CUDA_VISIBLE_DEVICES=2 timeout 40s python3 /tmp/v3_probe_run.py experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_colonly_L4_SG8`
  - result: `RC:124`
- traced `collatetracecut105 + consumersync3 + quantizersync2` returns cleanly
  - command: `CUDA_VISIBLE_DEVICES=2 timeout 40s python3 /tmp/v3_probe_run.py experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_collatetracecut105_consumersync3_quantizersync2_colonly_L4_SG8`
  - result: `RC:0`, `PY_DONE`
  - tail is materially the same as the returning `consumersync3` tracecut105 path
- quiet `consumersync3_cut106` still times out
  - command: `CUDA_VISIBLE_DEVICES=2 timeout 40s python3 /tmp/v3_probe_run.py experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_cut106_colonly_L4_SG8`
  - result: `RC:124`
- traced `collatetracecut106_consumersync3` also times out
  - it does not reach the intended post-postsync cut site in practice
  - log still shows the block-1 consumer path and phase flips, then the quantizer side continues and hangs

Most important new result from full `coltrace_consumersync3`:

- the consumer branch is no longer the blocker in the traced hang
- the full traced repro reaches:
  - `public col consumer final presync`
  - `public col consumer final postsync`
  - `public col consumer pre deprovision`
  - `public col consumer post deprovision`
- the full traced repro still times out after that
- none of the new quantizer-tail markers print:
  - no `public col quant final presync`
  - no `public col quant final postsync`
  - no `public col quant branch end`

Interpretation:

- consumer-side final sync and TMEM deprovision are not the remaining issue
- the surviving traced hang is entirely on the quantizer side
- in the full-trace build, the quantizer does not reach its final sync or branch end
- the last visible quantizer progress is still the earlier public-col epilogue band:
  - `post combo block`
  - `after cut5 block`

Current best localization:

- remaining live issue is quantizer-side, after the public-col `post combo` / `gap1b` region and before the quantizer branch final sync
- consumer-tail cleanup can be deprioritized for now

Recommended next cut:

- keep the probe light; full trace perturbs scheduling too much
- add one or two late-trace-only markers in the non-fused quantizer epilogue after `gap1b` and before the branch-level final quantizer sync
- prefer `late trace` over `full trace` for the next pass, because the full-trace build now clearly reaches consumer shutdown but still distorts the quantizer schedule enough to wedge earlier

## 2026-04-13 follow-up: sparse late trace confirms the wedge is after `gap1b`

New probe-only changes:

- added a plain `collatetrace_consumersync3` export
- added sparse late-trace markers for:
  - `common tail presync/postsync`
  - `quant final presync/postsync`
  - `quant branch end`
- added targeted `cut109` / `cut110` attempts at the common-tail sync seam on `epi=2`
- added per-warp late markers and a cta-0 representative-lane sample (`lane=0/8/16/24`) for `epi=2`

Results:

- plain sparse `collatetrace_consumersync3` still times out
  - command: `CUDA_VISIBLE_DEVICES=2 timeout 40s python3 /tmp/v3_probe_run.py experimental_backward_v3_fp4_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_collatetrace_consumersync3_colonly_L4_SG8`
  - result: `RC:124`
- `collatetracecut109_consumersync3` also times out
  - result: `RC:124`
- neither plain sparse late trace nor `cut109` ever shows:
  - `public col common tail presync`
  - `public col common tail postsync`
  - `public col quant final presync`
  - `public col quant final postsync`
  - `public col quant branch end`

Most important new localization:

- on the sparse path, all four quantizer warps do reach `gap1b` for the problematic `epi=2`
  - `warp=0,1,2,3` all print `public col warp gap1b ... bf=0 epi=2`
- on `cta=0`, the representative lane sample also reaches `gap1b` for `epi=2`
  - `warp=0,1,2,3`
  - `lane=0,8,16,24`
- despite that, the kernel still never reaches the common-tail seam immediately after

Interpretation:

- the remaining stall is now tightly constrained to the tiny band after `gap1b` and before the common post-epi quantizer sync
- it is not just a `warp0` problem
- it is not just a single obvious warpleader problem
- at least a broad representative subset of lanes across all four quantizer warps reaches `gap1b`
- the missing participants are therefore either:
  - unsampled lanes inside those same warps, or
  - a hidden scheduling/reconvergence issue in that immediate post-`gap1b` region

Current best next step:

- do one more cta-0 lane census in the same sparse style for the remaining lane classes (`4/12/20/28`, then if needed `2/6/...`), or
- switch from probe-only prints to a direct source diff against `737fa9df` for the exact public-col block spanning:
  - `gap1b`
  - the dead fused-only branch split
  - the common post-epi quantizer sync / recycle publish seam

## 2026-04-13 follow-up: source `rowhwfp4` fused path is back, but large standalone `colonly` is still broken

Source changes that held:

- kept the public-col tail brace fix in `nvfp4_cce_backward_v3.cuh`
  - the non-fused public-col tail is no longer accidentally nested under the fused-only branch
- restored the old-good siteA `slice_col_recycle_phasebits` update in the public `rowhwfp4` path
- repointed the public `backward_v3_fp4_L4_SG8` dispatch in `nvfp4_cce_backward_v3.cu` back to `rowhwfp4`
- then bound `backward_v3_fp4_L4_SG8` directly to the same `launch_experimental_backward_v3_fp4_3wg<rowhwfp4>` entrypoint because the wrapper path was intermittently surfacing launch-failure noise during bring-up

Current small-shape status:

- `CUDA_VISIBLE_DEVICES=2 python3 bench_v2_vs_v3.py --v3-exp-check --warmup 0 --iters 1`
  - now passes again
  - `256x512x256`: `row=True`, `row_sc=True`, `col=True`, `col_sc=True`
  - `512x512x256`: `row=True`, `row_sc=True`, `col=True`, `col_sc=True`

Important large-shape source result:

- the large-shape restored source path is usable through the explicit `rowhwfp4` export even though the legacy `nv-v3-fused` surface remains flaky
- on `4Kx4K->32K`, GPU `2`, `warmup=2`, `iters=5`:
  - explicit `rowhwfp4` `rowonly`: `0.540 ms`
  - explicit `rowhwfp4` fused front-half: `1.148 ms`
  - explicit `rowhwfp4` standalone `colonly`: still fails with `cudaErrorLaunchFailure`

Direct one-config timing snapshot:

- custom direct timing on `4Kx4K->32K`, GPU `2`, `warmup=2`, `iters=5`
  - `v2 full`: `1.053 ms`
  - explicit source `v3 full` (`rowhwfp4`): `1.564 ms`
  - explicit source `v3 overlap`: `1.555 ms`
  - explicit source `v3 fused`: `1.144 ms`
  - explicit source `v3 dE`: `0.202 ms`
  - explicit source `v3 dC`: `0.188 ms`

Accuracy snapshot:

- direct large-shape compare against the same-run `v2` path:
  - `cos(dE vs v2) = 1.0000`
  - `cos(dC vs v2) = 1.0000`

Interpretation:

- the source-restored `rowhwfp4` fused path is no longer the main blocker
- it runs on the target large shape and is numerically aligned with `v2`
- performance is still worse than `v2`, and almost all of the excess is in the fused front-half (`1.144 ms`)
- the remaining source bug is narrower now:
  - large standalone `rowhwfp4 colonly` still crashes
  - the legacy public `nv-v3-fused` timing surface is still noisy/flaky enough that direct explicit timing is the more trustworthy read

Recommended next step:

- stop spending time on tiny `colonly` probe scaffolding unless the standalone `colonly` path becomes necessary for shipping
- use the working explicit `rowhwfp4` fused entry as the active large-shape timing/correctness surface
- if public `v3` must be stable as a named API, keep debugging why `backward_v3_fp4_L4_SG8` still behaves less predictably than the explicit `rowhwfp4` export despite targeting the same kernel
- for `v6`, treat the design constraint as unchanged:
  - do not instantiate BF16 softmax materialization
  - row and col should go straight to FP4

Quick `v6` recheck after the `v3` source recovery:

- `CUDA_VISIBLE_DEVICES=2 python3 bench_v2_vs_v3.py --v6-publicv3-combo-check --warmup 0 --iters 1`
  - still fails on the col side
  - `256x512x256`: `row=True`, `row_sc=True`, `col=False`, `col_sc=False`, `dE=True`, `dC=False`
  - `512x512x256`: `row=True`, `row_sc=True`, `col=False`, `col_sc=False`, `dE=True`, `dC=False`
- so `v6` did not get fixed automatically by the `v3` source recovery; it is still blocked on its own col / `dC` side behavior

## 2026-04-13 follow-up: `v6` bridge frontend was still on the wrong path; default now uses `rowhwfp4`

Root cause:

- `nvfp4_cce_backward_v6.cu` was already instantiating the combo kernels with the `rowhwfp4` configs:
  - `experimental_config_*_rowhwfp4_*_combo_storeadd`
- but the bridge frontend alias `pub_bwd_v3_fp4_frontend_L4_SG8` was still defaulting to `consumerotf_warpbalanced`
- so `v6` was front-halving through the wrong public-v3 col path before handing off to the combo tail

Source change:

- changed the default `pub_bwd_v3_fp4_frontend_L4_SG8` alias in `nvfp4_cce_backward_v6.cu` from:
  - `experimental_config_colwg_consumerotf_warpbalanced<4, 8, true, 4>`
- to:
  - `experimental_config_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap<4, 8, true, 4>`
- kept the opt-in consumer-OTF compile flags intact, so this only changes the default bridge frontend

After rebuild:

- small `3wg` combo check now passes
  - `CUDA_VISIBLE_DEVICES=2 python3 bench_v2_vs_v3.py --v6-publicv3-combo-check --warmup 0 --iters 1`
  - `256x512x256`: `row=True`, `row_sc=True`, `col=True`, `col_sc=True`, `dE=True`, `dC=True`
  - `512x512x256`: `row=True`, `row_sc=True`, `col=True`, `col_sc=True`, `dE=True`, `dC=True`
- small `5wg` combo check also now passes
  - `CUDA_VISIBLE_DEVICES=2 python3 bench_v2_vs_v3.py --v6-publicv3-combo-5wg-check --warmup 0 --iters 1`
  - `256x512x256`: `row=True`, `row_sc=True`, `col=True`, `col_sc=True`, `dE=True`, `dC=True`
  - `512x512x256`: `row=True`, `row_sc=True`, `col=True`, `col_sc=True`, `dE=True`, `dC=True`

Large-shape timing / runtime snapshot:

- direct custom timing on `4Kx4K->32K`, GPU `2`, `warmup=2`, `iters=5`
  - `v6 combo (3wg)`: `1.542 ms`
- direct single large `3wg combo` launch through the benchmark surface returns cleanly
  - `launching -> launched -> synced`
- repeated `5wg combo` timing still does not behave like a stable benchmark surface yet
  - direct `5wg combo` reaches `launching` and then stalls on the large shape
  - repeated custom `5wg combo` timing loop also timed out

Current `v6` state:

- `3wg` default combo path is functionally restored on the small gates and has a usable large-shape timing surface again
- `5wg` is improved on small exactness but is still not large-shape stable
- the `v6` design constraint still holds:
  - no BF16 softmax materialization
  - row and col should go straight to FP4

## 2026-04-13 follow-up: selective `5wg` bridge split fixes large direct liveness; repeated combo timing still needs zeroed accumulate outputs

Source change in `nvfp4_cce_backward_v6.cu`:

- kept the dedicated row-only `5wg` frontend only for:
  - `experimental_backward_v6_combo_publicv3_fp4_5wg_gonly_L4_SG8`
  - `experimental_backward_v6_combo_publicv3_fp4_5wg_dEonly_L4_SG8`
- routed `5wg` full `combo` and `dConly` back through the generic `rowhwfp4` bridge
- reverted `experimental_rowonly_combo_dc_frontend_enabled` to the compile-flag-backed value instead of hard-coding `True`

Why this split was needed:

- the broad “force all `5wg` modes through dedicated row-only” rewrite was too aggressive
- tiny direct probes showed:
  - `gonly`: returned
  - `dEonly`: returned
  - `dConly`: hung
  - full `combo`: hung
- so the dedicated row-only `5wg` frontend is viable for the pure row path and `dE` bridge, but not for the `dC` side

Large-shape direct status after the selective split (`CUDA_VISIBLE_DEVICES=2`, `4Kx4K->32K`):

- `--v6-publicv3-combo-5wg-direct-mode gonly`
  - `launching -> launched -> synced`
- `--v6-publicv3-combo-5wg-direct-mode dEonly`
  - `launching -> launched -> synced`
- `--v6-publicv3-combo-5wg-direct-mode dConly`
  - `launching -> launched -> synced`
- `--v6-publicv3-combo-5wg-direct-mode combo`
  - `launching -> launched -> synced`

That means the earlier large-shape `5wg` launch blocker is gone for one-shot runs.

Remaining timing/reuse issue:

- a strict per-iteration `launch -> torch.cuda.synchronize()` loop still wedges on the second launch for:
  - `5wg dConly`
  - `5wg combo`
- the same reuse problem also reproduces on the `3wg` side for:
  - direct `dConly`
  - full `combo`
- in contrast:
  - `5wg gonly` repeats cleanly on the same buffers
  - `5wg dEonly` repeats cleanly on the same buffers

Key isolation result:

- the repeat hang is not a general launch bug
- it goes away if the accumulate outputs are scrubbed between launches
- for `dConly`, zeroing reused `dC_out` (and resetting `G_sg_row`) before each launch is enough
- for full `combo`, zeroing reused `dE_out` + `dC_out` (and resetting `G_sg_row`) before each launch is enough
- zeroing the FP4 materialization buffers was not needed for the passing reuse tests

Interpretation:

- the current combo / `dC` timing surfaces are accumulation-style
- reusing dirty BF16 destination tensors across launches is not a valid benchmark setup for these paths
- the earlier `bench_v2_vs_v3.py --v6-publicv3-combo*-mode ...` hangs were therefore mixing a real liveness symptom with a benchmark-harness bug: it was reusing `dE_out` / `dC_out` without clearing them between timed launches

Useful manual steady-state timing observations from the zeroed-output sync loops:

- `5wg gonly`: steady-state around `0.33 ms`
- `5wg dEonly`: steady-state around `0.55 ms`
- `5wg dConly`: after zeroing `dC_out`, later iterations were in the low-single-digit-ms band (`~2.2-3.1 ms` in the direct sync loop)
- `5wg combo`: after zeroing `dE_out` + `dC_out`, later iterations were likewise in the low-single-digit-ms band (`~2.3-3.2 ms` in the direct sync loop)
- `3wg combo`: with zeroed `dE_out` + `dC_out`, observed steady iterations were about `3.2 ms`

Best next step from here:

- do not treat the remaining repeated-timing hang as a front-half failure anymore
- either:
  - patch the benchmark harness to zero `dE_out` / `dC_out` before each `v6` combo / `dConly` timed launch, or
  - make the wrapper semantics explicit that these accumulate outputs must be zeroed by the caller
- only after the timing surface is made valid is it worth comparing `3wg` vs `5wg` combo throughput in earnest

Harness follow-up:

- patched `/workspace/codebases/cce/fp4_matmul/fp4_cce_TK/bench_v2_vs_v3.py`
  - added a `setup_fn` path in the benchmark helper so `dE_out` / `dC_out` can be zeroed outside the timed region for the `v6` combo timing modes
  - wired that setup path into both:
    - `--v6-publicv3-combo-mode ...`
    - `--v6-publicv3-combo-5wg-mode ...`
- after that harness fix, the previously hanging timed large combo modes now return cleanly on `CUDA_VISIBLE_DEVICES=2`

Current large timed combo snapshot (`4Kx4K->32K`, `warmup=2`, `iters=5`):

- `CUDA_VISIBLE_DEVICES=2 ... bench_v2_vs_v3.py --v6-publicv3-combo-mode combo --config-label '4Kx4K->32K' --warmup 2 --iters 5`
  - `nv-v6-pv3cmb`: `1.534 ms`
- `CUDA_VISIBLE_DEVICES=2 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-mode combo --config-label '4Kx4K->32K' --warmup 2 --iters 5`
  - `nv-v6-pv35cmb`: `1.537 ms`

Current large timed `5wg` breakdown snapshot (`4Kx4K->32K`, `warmup=2`, `iters=5`):

- `CUDA_VISIBLE_DEVICES=2 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-mode gonly --config-label '4Kx4K->32K' --warmup 2 --iters 5`
  - `nv-v6-pv35g`: `0.535 ms`
- `CUDA_VISIBLE_DEVICES=2 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-mode dEonly --config-label '4Kx4K->32K' --warmup 2 --iters 5`
  - `nv-v6-pv35dE`: `0.772 ms`
- `CUDA_VISIBLE_DEVICES=2 ... bench_v2_vs_v3.py --v6-publicv3-combo-5wg-mode dConly --config-label '4Kx4K->32K' --warmup 2 --iters 5`
  - `nv-v6-pv35dC`: `1.375 ms`

Interpretation update:

- with a valid timing setup, `5wg combo` is now benchmarkable again
- on the current source, `5wg combo` is effectively tied with `3wg combo`, not clearly better
- the fused `5wg combo` path is still meaningfully better than the naive sum of isolated pieces:
  - `0.535 + 0.772 + 1.375 = 2.682 ms`
  - vs fused `5wg combo = 1.537 ms`
- so the next optimization pass should target real throughput deltas now that the timing surface is trustworthy, rather than more basic liveness/debugging

## 2026-04-13 localCTA-tail full-backward snapshot

Goal for this pass:

- get an apples-to-apples view of full backward (`G` materialization + `dE` + `dC`) when the tail GEMMs also use the localCTA family
- for the localCTA-tail variants, pre-quantize `C^T` and `E^T` once outside the timed region, then run:
  - front-half materialization (`v2` quantized `G`, public `v3`, or `v6 gonly`)
  - bridge scalar `G_sg` into localCTA chunk-grid form
  - `nvfp4_localcta_gemm` for `dE`
  - `nvfp4_localcta_gemm` for `dC`

Wrapper fix that unblocked the comparison:

- `3wg` `gonly` / `dEonly` were still routed through the generic public-v3 bridge, while `5wg` already used the dedicated row-only frontend
- patched `nvfp4_cce_backward_v6.cu` so `launch_experimental_backward_v6_combo_publicv3_fp4_bridge<ComboMode>` now mirrors `5wg` for `ComboMode == gonly/dEonly`
- after rebuild:
  - direct large `experimental_backward_v6_combo_publicv3_fp4_gonly_L4_SG8` now reaches `launching -> launched -> synced`
  - timed large `3wg gonly` is now `0.535 ms`

Refreshed large-shape localCTA-tail timings on `CUDA_VISIBLE_DEVICES=2`, `4Kx4K->32K`, `warmup=2`, `iters=5`:

- `v2 localCTA full`: `2.779 ms`
- `v3 localCTA full`: `7.353 ms`
- `v6 3wg localCTA full`: `2.414 ms`
- `v6 5wg localCTA full`: `5.548 ms`

Notes on the `v3 localCTA` measurement:

- a naive repeated timing loop still wedges on reused outputs
- a manual per-iteration sync loop is stable if the setup step scrubs all reused materialization/tail buffers outside the timed region:
  - `G_sg_row`
  - `dE_out`
  - `dC_out`
  - `G_sc_row`
  - `G_sc_col`
  - `G_fp4_col`
- with that scrub in place, the large standalone timed iterations were stable at:
  - `7.353 / 7.350 / 7.356 / 7.351 / 7.355 ms`
  - average `7.353 ms`

Current-path comparison points already established on the same shape / device family:

- `v2 current full`: `~1.05 ms` from the stable benchmark surface before this pass
- `v3 current full`: `~1.56 ms` from the restored `rowhwfp4` public path checkpoint
- `v6 3wg current combo`: `1.529 ms`
- `v6 5wg current combo`: `1.528 ms`

Interpretation:

- the earlier huge localCTA numbers were polluted by process pressure / stale runs and should not be used
- with the `3wg gonly` wrapper fixed, localCTA-tail full backward is now benchmarkable on both `v2` and `v6 3wg/5wg`
- `v3` is also benchmarkable now, but only with the explicit reuse scrub above
- the updated `v2 localCTA` number is much closer to expectations, which supports the suspicion that the first comparison was not trustworthy
- remaining open question: whether the direct `nvfp4_localcta_gemm` bridge is still leaving `v2` on a slower localCTA GEMM path than a prepared/fast localCTA tail would

## 2026-04-13 backward sweep note

Canonical 6-shape backward sweep results are recorded in:

- [BACKWARD_SWEEP_2026-04-13.md](/workspace/codebases/cce/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200/BACKWARD_SWEEP_2026-04-13.md)

Short version:

- `v2-native` / `v3-native` remain numerically healthy.
- `v3-dec` is healthy.
- NVFP4 `v3-enc` is still functionally wrong across the sweep.
- MXFP4 `enc` / `dec` rows are healthy.
- `v6` localCTA rows are timing-only in the current public ABI because the bridge does not expose the per-chunk localCTA `sg` grids needed for trustworthy full-backward cosine checks.

## 2026-04-14 `v3-enc` contract fix

The 2026-04-13 sweep note above is now stale with respect to NVFP4 `v3-enc`.

Root cause and fix:

- standalone public NV quantization does not expose a distinct raw GEMM-facing encode-vs-decode scale-byte ABI
- `v3-enc` had drifted into a different raw-output contract
- fixed `launch_backward_v3_fp4_public_enc_L4_SG8(...)` to use the same public raw-output contract as the standalone quantizer, and collapsed the duplicated scale-byte math in `nvfp4_cce_backward_v3.cuh` onto the shared helper path

Post-fix reruns on `CUDA_VISIBLE_DEVICES=2`:

- `--backward-sweep --shape-set small2 --warmup 0 --iters 1`
- `--backward-sweep --shape-set large4 --warmup 0 --iters 1`

Representative NVFP4 `v3-enc` results after the fix:

- `256x256->512`: `0.195 ms`, `cos(dE)=0.9930`, `cos(dC)=0.9933`
- `512x256->512`: `0.291 ms`, `cos(dE)=0.9932`, `cos(dC)=0.9935`
- `4Kx4K->32K`: `1.588 ms`, `cos(dE)=0.9940`, `cos(dC)=0.9939`
- `4Kx8K->32K`: `1.990 ms`, `cos(dE)=0.9940`, `cos(dC)=0.9940`
- `8Kx4K->32K`: `3.012 ms`, `cos(dE)=0.9941`, `cos(dC)=0.9939`
- `4Kx4K->128K`: `6.048 ms`, `cos(dE)=0.9941`, `cos(dC)=0.9940`

Current interpretation:

- `v3-enc` is no longer functionally broken on the public NVFP4 sweep surface
- `v3-native`, `v3-enc`, and `v3-dec` are now all numerically healthy
- the remaining question is performance and footprint, not `v3-enc` correctness

## 2026-04-14 MX family flag + memory footprint

Benchmark harness updates landed in `fp4_cce_TK/bench_v2_vs_v3.py`:

- `--fp4-family {nv,mx,both}`
- `--report-memory`

Validated behavior:

- NV small2 memory reporting is working.
- MX small2 and `4Kx4K->32K` memory reporting are working.
- MX `v3` uses its public FP4 row+col backward ABI.
- MX `v2` does not have an equivalent public FP4 row+col ABI, so full backward remains:
  - BF16 backward
  - mode-matched MX quantization of `G` and `G^T`
  - MX GEMM tails

Representative MX results on `CUDA_VISIBLE_DEVICES=2`, `warmup=0`, `iters=1`:

- `256x256->512`
  - `v2-enc`: `0.520 ms`, `cos(dE)=0.9933`, `cos(dC)=0.9932`, `PeakAlloc=1.01 MB`, `Contract=0.76 MB`
  - `v3-enc`: `0.505 ms`, `cos(dE)=0.9933`, `cos(dC)=0.9932`, `PeakAlloc=0.76 MB`, `Contract=0.51 MB`
- `512x256->512`
  - `v2-enc`: `0.403 ms`, `cos(dE)=0.9934`, `cos(dC)=0.9932`, `PeakAlloc=1.77 MB`, `Contract=1.27 MB`
  - `v3-enc`: `0.225 ms`, `cos(dE)=0.9934`, `cos(dC)=0.9932`, `PeakAlloc=1.27 MB`, `Contract=0.77 MB`
- `4Kx4K->32K`
  - `v2-enc`: `1.533 ms`, `cos(dE)=0.9934`, `cos(dC)=0.9934`, `PeakAlloc=914.81 MB`, `Contract=664.81 MB`
  - `v3-enc`: `3.610 ms`, `cos(dE)=0.9934`, `cos(dC)=0.9934`, `PeakAlloc=665.38 MB`, `Contract=414.81 MB`

Interpretation:

- the old assumption that MX `v2` and MX `v3` were directly comparable through the same public FP4 row+col ABI was wrong
- `v3` currently carries the better public MXFP4 materialization contract
- `v2` can still win on time in the hybrid path, but it pays for the extra BF16 `G` plus requantization in memory footprint

## 2026-04-14 xlarge NV + Triton comparison

The benchmark harness now has:

- `xlarge4` shape set
- `--include-triton-bf16`

where the Triton row uses `cut_cross_entropy.linear_cross_entropy(..., impl="cce")` as the original BF16 CCE baseline.

Completed NV/Triton large-shape comparisons on `CUDA_VISIBLE_DEVICES=2`:

- `4Kx7K->256K`
  - `v2-native`: `12.298 ms`
  - `v2-dec`: `11.123 ms`
  - `v3`: public row+col path OOM (`+122.07 GiB` attempted alloc)
  - `triton-cce`: `1134.887 ms`
- `16Kx4K->32K`
  - `v2-native`: `3.838 ms`
  - `v2-dec`: `3.823 ms`
  - `v3-enc`: `5.992 ms`
  - `v3-dec`: `6.023 ms`
  - `triton-cce`: `988.593 ms`
- direct one-shot spot checks:
  - `8Kx8K->128K`: `v2-enc 13.211 ms`, `triton-cce 1842.606 ms`, `v3-enc` did not return promptly
  - `16Kx8K->128K`: `v2-enc 60.695 ms`, `triton-cce 1319.789 ms`

What this means:

- `v2` is not just winning the older moderate shapes; it is the only path here that still looks operationally strong on frontier shapes.
- Triton BF16 CCE is useful as a correctness/backward baseline, but it is nowhere near `v2` on wall-clock backward time in this regime.
- `v3` large-shape weakness is now clearly a scalability issue in the public row+col materialization path:
  - stable but slower on `16Kx4K->32K`
  - OOM or non-returning on the heavier shapes

## 2026-04-14 clean isolated compare

Added a subprocess-based clean compare mode in `bench_v2_vs_v3.py`:

- `--isolated-compare`
- `--isolated-timeout-s`

This runs `v2-enc`, `v3-enc`, and `triton-cce` from the same raw BF16 inputs in fresh processes and reports:

- `Time (ms)`
- `Peak>Raw(MB)`
- `Peak>Quant(MB)`

Clean isolated results on `CUDA_VISIBLE_DEVICES=2`:

- `16Kx4K->32K`
  - `v2-enc`: `13.036 ms`, `Peak>Raw=2153.19 MB`, `Peak>Quant=1940.50 MB`
  - `v3-enc`: `TIMEOUT`
  - `triton-cce`: `196.182 ms`, `Peak>Raw=756.28 MB`, `Peak>Quant=378.09 MB`
- `4Kx7K->256K`
  - `v2-enc`: `39.484 ms`, `Peak>Raw=8684.52 MB`, `Peak>Quant=6681.00 MB`
  - `v3-enc`: `CUDA OOM`, attempted `+122.07 GiB`
  - `triton-cce`: `654.368 ms`, `Peak>Raw=7112.92 MB`, `Peak>Quant=3556.87 MB`

Interpretation:

- the clean-isolated surface is the current trustworthy one for large-shape decisions
- the older in-process `v3` rows should no longer be treated as definitive where they conflict with the isolated mode
- `v3` currently lacks a stable large-shape production comparison surface; `v2` does not

## 2026-04-14 v3 large-shape memory / runtime split

The `+122.07 GiB` `v3` OOM was a real wrapper bug, not the true row+col contract size.

Source fix in `nvfp4_cce_backward_v3.cu`:

- the public `3wg` wrappers were still allocating an unused combo placeholder as `max(M, N) x max(M, N)` BF16
- on `4Kx7K->256K`, `max(M, N)=256000`, so that dead placeholder tried to allocate `256000 x 256000` BF16, which is about `122.07 GiB`
- that placeholder is now replaced with the same legal minimum `128x32` BF16 tile already used by the col-only path

After rebuild, the clean isolated `4Kx7K->256K` compare changed from OOM to timeout:

- `v2-enc`: `39.444 ms`, `Peak>Raw=8684.52 MB`, `Peak>Quant=6681.00 MB`, `OK`
- `v3-enc`: `TIMEOUT`
- `triton-cce`: `659.113 ms`, `Peak>Raw=7112.92 MB`, `Peak>Quant=3556.87 MB`, `OK`

So the absurd OOM is fixed, but `v3` still has real large-shape runtime failures.

Direct large-shape front-half splits on the rebuilt `v3` binary:

- `16Kx4K->32K`
  - public full `backward_v3_fp4_enc_L4_SG8`: hangs / times out
  - `rowonly`: returns cleanly in about `0.257 s`
  - `colonly`: fails with `CUDA error: unspecified launch failure`
- `4Kx7K->256K`
  - public full `backward_v3_fp4_enc_L4_SG8`: hangs / times out
  - `rowonly`: returns cleanly
  - `colonly`: returns cleanly

Current interpretation:

- `v3` does not actually need `122 GiB`; that was a dead placeholder bug and is gone
- there are still two independent large-shape runtime problems in the public row+col path:
  - large-`M` col-only failure (`16Kx4K->32K`)
  - large-`V` row+col interaction failure (`4Kx7K->256K`), since both row-only and col-only return by themselves

## 2026-04-15: public `v3` large-shape fix promoted

The remaining public large-shape `v3` failures were fixed by promoting the sync-ID override that
had already worked as a debug sidecar.

Source change:

- in `nvfp4_cce_backward_v3.cu`, public dispatch now uses
  `bwd_v3_fp4_public_colwg_colpairpad_rowpair_lanepairrecord_rowsync_dualfloatcache_rowhwfp4_row16ready_overlap_consumersync3_quantizersync2_L4_SG8`
  instead of the prior plain `...row16ready_overlap_L4_SG8`

Validation after rebuild:

- `--v3-exp-check` passes on both small shapes
- clean isolated compare, `CUDA_VISIBLE_DEVICES=2`
  - `4Kx4K->32K`: `v3-enc 1.547 ms`, `Peak>Raw=581.27 MB`, `Peak>Quant=422.63 MB`, `OK`
  - `16Kx4K->32K`: `v3-enc 248.314 ms`, `Peak>Raw=1153.20 MB`, `Peak>Quant=940.51 MB`, `OK`
  - `4Kx7K->256K`: `v3-enc 285.110 ms`, `Peak>Raw=6684.52 MB`, `Peak>Quant=4681.01 MB`, `OK`
  - `8Kx8K->128K`: `v3-enc 286.359 ms`, `Peak>Raw=4450.04 MB`, `Peak>Quant=3253.01 MB`, `OK`
  - `16Kx8K->128K`: `v3-enc 335.787 ms`, `Peak>Raw=5775.07 MB`, `Peak>Quant=4506.01 MB`, `OK`

Matched reference rows from the same isolated compare:

- `4Kx4K->32K`
  - `v2-enc 1.043 ms`, `Peak>Raw=831.27 MB`, `Peak>Quant=672.63 MB`
  - `triton-cce 47.184 ms`, `Peak>Raw=564.12 MB`, `Peak>Quant=282.07 MB`
- `16Kx4K->32K`
  - `v2-enc 13.030 ms`, `Peak>Raw=2153.19 MB`, `Peak>Quant=1940.50 MB`
  - `triton-cce 191.815 ms`, `Peak>Raw=756.28 MB`, `Peak>Quant=378.09 MB`
- `4Kx7K->256K`
  - `v2-enc 39.494 ms`, `Peak>Raw=8684.52 MB`, `Peak>Quant=6681.00 MB`
  - `triton-cce 654.781 ms`, `Peak>Raw=7112.92 MB`, `Peak>Quant=3556.87 MB`

Net state after the fix:

- the earlier `+122.07 GiB` OOM was a separate wrapper bug and remains fixed
- public `v3` is now operational on the large isolated surface
- `v3` memory footprint is lower than `v2` on these large shapes, which matches the intended contract
- `v3` throughput is still poor; the remaining problem is performance, not the earlier memory blow-up
