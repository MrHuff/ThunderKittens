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
