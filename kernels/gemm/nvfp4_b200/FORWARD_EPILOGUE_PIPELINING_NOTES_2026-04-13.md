## Forward Pass Walkthrough

This note covers the current NVFP4 CCE forward path in:

- `nvfp4_cce.cuh`
- `nvfp4_cce_v2.cuh`
- `nvfp4_cce_v2.cu`

The forward kernel is not a conventional GEMM that materializes an `[M, V]` logits tensor in HBM and then runs a separate softmax/loss pass. The consumer epilogue works directly on TMEM-resident accumulators, computes the cross-entropy reduction state, and only writes compact outputs:

- `lse[M]`
- `neg_correct_logit[M]`
- a tiny `D_scratch` pacing store

That changes where the next throughput win is likely to come from.

### Base structure

The forward path inherits the producer/consumer split from the GEMM-style kernels:

- producer WG:
  - streams `A/B` tiles and scales from TMA into shared memory
  - launches tensorcore MMA into TMEM accumulators
- consumer WG:
  - drains TMEM accumulator subtiles into registers
  - scales by `A_sc_global * B_sc_global`
  - extracts target logits
  - updates online logsumexp state
  - emits only the reduced CE outputs

The reference implementation in `nvfp4_cce.cuh` already has an explicit deferred epilogue split:

1. phase 1a: process the first half of the output subtiles inline
2. phase 1b: batch-load the second half of the TMEM subtiles into FP32 registers
3. signal `outputs_finished`
4. phase 2: finish the deferred CE work from registers while producer MMA can move on
5. phase 3: finalize `lse`

The key comment in the source is accurate: the optimization is about decoupling CE work from the TMEM residency window, not about reducing arithmetic.

### What `v2` changes

`nvfp4_cce_v2.cuh` pushes that idea further with ping-pong accumulators:

- two TMEM accumulator slots are allocated at offsets `0` and `128`
- the producer alternates between them
- the consumer always drains the inactive slot into registers, then immediately signals `outputs_finished`

The critical sequence in the consumer is:

1. wait for `outputs_arrived`
2. batch-load all `EPI_PIPE_DEPTH` subtiles from the active accumulator into registers
3. `tensor_load_wait()`
4. `warpgroup::sync(1)`
5. `warpgroup::tma::cluster::arrive(outputs_finished, 0, 1)`
6. run the full `consumer_epilogue(...)` from registers

So `v2` already does the right first-order thing: TMEM is released before the heavy CE epilogue finishes.

### Why forward still looks pipeline-limited

The likely remaining bottleneck is not "HBM logits writeback". There is no large logits writeback.

The likely bottleneck is the paced consumer epilogue after TMEM drain:

- rowwise target extraction
- row max / subtract / exp / sum reductions
- running logsumexp merge
- atomics into `lse`
- atomics into `neg_correct_logit`
- pacing stores through `D_scratch`

Even though TMEM is released early, that consumer work still occupies the consumer WG for the block. If it runs too long relative to producer-side MMA cadence, the producer can end up waiting on the next consumer handoff despite the accumulator ping-pong.

### What to optimize next

The next useful experiment is to pipeline the epilogue more aggressively after register drain, not to pipeline a logits store path.

Concretely:

1. keep the current early TMEM release
2. split the post-drain epilogue into a fast local stage and a paced reduction/store stage
3. let the producer continue on the next accumulator while the consumer finishes:
   - target extraction
   - logsumexp merge
   - atomics
   - scratch pacing stores

Two plausible cuts:

- Cut A: separate "register drain + local row stats" from "global atomics + pacing stores"
- Cut B: keep per-subtile CE math in the hot path, but defer final row merges and atomics into a later mini-phase

### Practical implication

If forward throughput is under target, the first thing to inspect is not the MMA loop. The producer side is already reasonably decoupled. The better target is the consumer epilogue pacing after `outputs_finished`, especially any work that does not need the accumulator live once the registers are loaded.

That is the same direction the existing deferred-epilogue comments in `nvfp4_cce.cuh` are already pointing at. The next step is to extend that idea, not replace it with a conventional writeback pipeline.
