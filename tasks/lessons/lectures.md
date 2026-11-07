# Lessons from the HPML lectures — correct diagnostics & profiling for DUST

Distilled from the course lectures (`../../lectures/`), filtered to what makes our **gradient
diagnostics and performance profiling correct for DUST specifically** — a forward-only, no-backprop
trainer whose step is thousands of tiny forward passes. Each point is: concept → implication for DUST →
what we already built or still need to build.

**Read status (all read in full, 2026-11-07):** Lecture 1 (85 pp), Lecture 2 (90 pp), Lecture 3
(119 pp), Lecture 4 (100 pp), Lecture 5 CUDA Basics (107 pp) + part 2 (116 pp), Lecture 6 Advanced CUDA
(116 pp). The two guest lectures are intentionally skipped. **Lecture 5 "part 2" is a revised reprint of
part 1** (same deck, plus an animated 7-step compile-stages sequence) — no DUST-relevant content beyond
part 1, so it is treated as one lecture.

---

## Bottom line (read this first)

1. **Our workload is almost certainly launch-bound, and the course says so directly.** A step is set by
   `T_step ≈ max(Σ CPU-launch-time, Σ GPU-kernel-time)` (L3 s37). DUST issues thousands of tiny
   forwards through Python hooks; each kernel finishes before the CPU can issue the next, so the CPU
   sets the step time and **a faster GPU or fewer FLOPs does not help**. L1 names the causes verbatim
   ("launch overhead from many small operations", "low occupancy"); L3 calls the trace signature a
   "thin comb".
2. **Measure the regime before trusting any FLOP number — and FLOPs are the wrong axis anyway.** A
   roofline needs FLOPs **and bytes moved**; a launch-bound step sits *below* the roofs where neither
   more compute nor more bandwidth helps. Report **three regimes**: launch/latency-bound,
   bandwidth-bound, compute-bound.
3. **The fix ladder for a launch-bound DUST step (in order):**
   - **(a) Vectorize the population into one batched forward.** This is the single highest-value lever
     (L3 shows a Python loop vs a broadcast op at ~1900× on the same result). It removes launches *at
     the source* and also fixes the tiny-matmul under-utilization — it beats compiling the loop.
   - **(b) CUDA Graphs** (hand-built, or `torch.compile(mode="reduce-overhead")`) for the launches that
     remain: record the fixed-shape forward once, replay per draw with one launch (L6 appendix A2 is
     the DUST pattern verbatim).
   - **(c) `torch.compile` (default)** to fuse the elementwise noise/scale/activation chains.
   - **Blocker:** our forward-hooks are the hazard. Anything host-side in a hook — `.item()`, a branch
     on a tensor value, Python-sampled noise, arbitrary Python — causes a torch.compile **graph break**
     (recurring cost, lost fusion) *and* disqualifies CUDA Graphs (no sync/alloc allowed inside
     capture). The same three sins break all three fixes at once. Hooks must be pure-GPU, sync-free,
     static-shape.
4. **The honest comparison is two numbers, not one:** steps/second **and** time-to-target-quality
   (= time/step × steps needed) — our gradient-quality-vs-population curve feeds exactly this.
5. **Hardware caveat:** the lecture roofline numbers are **H100**; our target is **A100 40 GB**. Use
   A100 datasheet peaks, matched to DUST's dtype (illustrative A100 ridge: FP32 ≈ 12 FLOP/byte,
   BF16-tensor ≈ 200 FLOP/byte — confirm the exact SXM/PCIe part).

---

## What we already got right (validated by the lectures)

- **CUDA-event GPU timing + warmup discard + median/p10/p90 + stated scope** in `accounting.py` is
  exactly L2/L3's prescribed method (L3 s47, s17; L6 s96).
- **Recording both GPU-event time and host wall-clock per step, and flagging launch-bound when host ≫
  GPU**, is precisely the L6 s96 divergence test — that gap *is* the launch tax. Keep it.
- **The double-batch launch-bound test** in `profiling.py` is L3 s37 test #1 / L4 s25 / the L2
  `T ≈ L + N/B` test. Valid *only if each timed region is synced* (L5 s38).
- **Running under `no_grad` ⇒ no autograd graph, no saved activations** = inference-level memory (L3
  s25/s26/s34). Our `analytic_memory` correctly drops the backward terms.
- **`record_function('dust/<phase>')` phase labels** match L3 s52.
- **`checkpoint.py` saving optimizer momentum + perturbation RNG + step** is the full-state resume L3
  s24/s105-108 implies; the RNG/generator state is load-bearing because DUST is seed-driven.

## Corrections to our earlier (inferred) notes

- **"CUDA events, not wall-clock + synchronize" was too absolute.** The real rule (L3 s47/s62, L4 s85):
  *never time GPU work without a sync.* CUDA events are preferred for per-kernel timing;
  `perf_counter()` **with** `torch.cuda.synchronize()` is valid for whole-step timing. Our timer is
  fine; the wording was wrong.
- **Our autograd-memory list missed "workspace."** Training memory = params + grads + optimizer state +
  saved activations **+ workspace** (L3 s26). Does not change our no_grad memory model, but note it.
- **A caveat we had not stated:** `gradient_check.py`'s backprop reference needs `requires_grad` +
  saved activations, so it has a **higher, asymmetric memory profile** than the DUST estimate — they
  are not memory-comparable, and the reference path must **not** use `inference_mode` (see below).

---

## Lecture 1 — Why High-Performance ML

- **"Fewer FLOPs can still mean longer runtime"; the answer comes from a measurement, not from reading
  the algorithm.** Named causes include launch overhead from many small ops and low occupancy. →
  DUST reduces FLOPs (no backward) but hits both named causes. → Keep "launch-bound" a *hypothesis to
  confirm with a trace*, never an assumption.
- **Step-time attribution** ("where did the 40 ms go?": kernel compute / HBM / NCCL / Python-launch /
  other). → On one A100 (no NCCL) expect DUST's split dominated by **Python-launch + other**, not
  kernel compute. → *Build:* `accounting.py` measured-cost should attribute the step into those
  buckets; the launch+other fraction *is* the key metric.
- **Compute has outpaced HBM bandwidth; batch-1 decode is bandwidth/latency-bound.** → A DUST per-draw
  forward is the batch-1-decode analog (low weight reuse, streaming weights for tiny arithmetic). →
  Supports the bytes/roofline view below.
- **Named metrics: bytes-moved-per-FLOP and MFU (measured FLOP/s ÷ peak).** → We track FLOPs but not
  bytes — missing a co-equal metric. → *Build:* add bytes + MFU; expect *very low MFU* for DUST, which
  is itself the "not compute-bound" diagnostic.

## Lecture 2 — Performance Engineering + PyTorch Fundamentals

- **Async execution: a CPU timer around GPU work measures only launch/dispatch.** Sync before start and
  before reading the end, or use CUDA events. → Validates our timer; the single most common bug for
  this exact workload.
- **Warm-up is workload-dependent, not a fixed count** (context init, cuBLAS/cuDNN load, allocator pool
  growth, autotuning new shapes). → Thousands of tiny ops drive allocator growth/autotuning. → *Refine:*
  warm up until step time **stabilizes**, not a fixed discard.
- **Report a distribution + a reproducibility record.** → *Build:* have the tools emit the record **plus
  DUST knobs: population size, noise σ, #perturbed nodes/layers, seed** — attach it to the
  quality-vs-population curve especially.
- **Roofline: `I = FLOPs / bytes`, `P ≤ min(P_peak, I·BW)`, ridge `I = P_peak/BW`.** Left of ridge =
  bandwidth-bound; a higher compute peak cannot help there. → DUST's tiny forwards are low-intensity →
  far left (or below). → *Build (the roofline view):* add bytes-moved, compute `I`, place the step on an
  **A100** roofline matched to dtype. **Key byte question:** does DUST re-stream the **weights from HBM
  every draw** (→ massively bandwidth-bound, like batch-1 decode) or keep them resident? The bytes model
  must distinguish these — it names the fix (batch the population so each weight load amortizes).
- **CRITICAL: the roofline has no "launch roof."** `T ≈ L + N/B`: small N ⇒ fixed launch latency `L`
  dominates; "a launch costs a few µs whether the kernel runs 2 µs or 2 s." A launch-bound step sits
  *below* the bandwidth roof, which roofline alone cannot explain. → *Flag:* roofline is **necessary but
  not sufficient**; report the three regimes and pair it with the launch tests.
- **Amdahl: `S_max = 1/(1−p)`; which part you speed up matters more than by how much.** → If a DUST step
  is mostly Python/launch, any FLOP/kernel win is capped (e.g. 70% launch ⇒ ≤1.4×). → *Build:* attribute
  the step into fractions and apply Amdahl to **bound** the achievable gain; `p` must be *measured*.
- **Critical path ≠ bottleneck; a utilization number is not a diagnosis; summing per-op durations
  answers a different question.** → A per-op table **under-counts** a DUST step because it hides the
  inter-kernel gaps — and the gaps *are* the launch overhead. → *Build:* `profiling.py` must read the
  **timeline/trace gaps**, not just the table; report **GPU-idle gap per step** and **mean kernel
  duration vs mean inter-kernel gap** (the decisive DUST number).

## Lecture 3 — PyTorch Runtime and Profiling

- **Launch-bound mechanism (s30, s37):** every op runs a full CPU path (Python → dispatcher → autograd →
  ATen → `cudaLaunchKernel`) and returns when the kernel is *queued*. Many tiny ops ⇒ CPU can't issue
  fast enough. → This is DUST's expected regime.
- **`inference_mode` is stronger than `no_grad` (s31-34).** `no_grad` drops the autograd dispatch key;
  `inference_mode` *also* skips view/version-counter bookkeeping ⇒ less per-op CPU cost. → *Build/fix
  (GPU-validate):* run the **DUST forward path under `inference_mode`** for cheaper launches. **Caveat:**
  inference_mode tensors can't later enter autograd, so `gradient_check.py`'s **backprop reference must
  stay under normal grad** — only the DUST estimate path gets inference_mode.
- **Vectorize the Python loop (s38, s62-76) — the biggest single lesson.** A row-loop (~12k launches,
  55.87 ms) vs one broadcast op (0.029 ms, ~1900×), same result; 3072 GPU events → 2. → DUST's
  per-draw loop is this anti-pattern. → **Batch the population dimension** (fix (a) in the ladder).
- **Hidden syncs (s40-41):** `.item()`, `.cpu()`, `.tolist()`, `print(tensor)`, `bool(tensor)`,
  `.nonzero()`, boolean-mask indexing, `.all()`. "One sync per step is fine, one per inner loop is not."
  Find them with `torch.cuda.set_sync_debug_mode("warn"|"error")`; fix by accumulating on the GPU with
  `.detach()` and reading once. → Our hot draw-loops are sync-clean; keep them that way, and this is a
  hard requirement for CUDA-Graph capture.
- **torch.profiler (s49-55):** `key_averages().table(sort_by=...)` — `self_cuda_time_total` for GPU
  hotspots, `self_cpu_time_total` for launch overhead; `schedule(skip_first/wait/warmup/active)`;
  `record_function`; `export_chrome_trace` → Perfetto; self ≠ total. "Many small ops with CPU time well
  above CUDA time is the launch-bound pattern." **`with_flops`** gives estimated matmul FLOPs ⇒ achieved
  throughput. → *Build:* use `with_flops`; sort both ways; expect the thin-comb / CPU≫CUDA signature.
- **Five trace patterns (s77):** GPU gaps / **thin comb = launch-bound** / long sync / H2D-on-path /
  one big kernel. "The first four mean the GPU is waiting; only the fifth means it is slow." → DUST's
  expected pattern is the thin comb — classify it.
- **dtype precision (s15-16; L4 s18):** FP32 = 23 mantissa bits, TF32 = 10, **BF16 = 7**, FP16 = 10. →
  A bf16 DUST forward vs an fp32 true-gradient reference differs by more than rounding — `gradient_check`
  must control dtype (we support fp32 and bf16-match).
- **Views/strides/`.contiguous()` (s18-20, s27-28):** a transpose is free; the copy cost shows up in the
  *next* kernel. → Adding noise to strided activation views inside a hook can trigger a hidden
  `.contiguous()` copy. → *Build:* `profiling.py` should watch for unexpected copy kernels in the hook
  path.
- **Memory timeline (s89):** forward-only should show flat, equal humps with nothing held across steps;
  a rising sawtooth means a tensor is retained (e.g. storing loss without `.detach()`).
- **Optimizer cost (s24, s105-112):** the step updates each parameter *and* its buffers; SGD = 8 B/param
  (p+g), SGD+momentum = 12 B (+1 buffer), AdamW = 16 B (+2, m&v). The step is **memory-bound** — use
  `foreach` (CUDA default) or `fused`, not a per-tensor for-loop (same launch-bound comb). → Checkpoint
  the buffers (done); don't let the optimizer reintroduce launch overhead.
- **Steps/s vs time-to-target (s113):** judge any change that alters the math (DUST vs backprop,
  precision, batch) by **two numbers**, never step time alone.
- **Tool map (s48):** nvidia-smi/nvtop (is the GPU busy), profiler table (which ops), trace → Perfetto
  (over time), HTA (GPU idle per rank), Nsight Systems (whole system), Nsight Compute (one kernel),
  py-spy/cProfile (Python hotspots).

## Lecture 4 — torch.compile

- **Verdict for DUST: helps, but not by itself, and the hooks are the hazard.** Rank the fixes:
  vectorize the population first → CUDA Graphs / `reduce-overhead` for the rest → `default` for fusion.
- **Graph break (s56-62):** Dynamo stops the graph, runs Python, restarts — blocking fusion across it
  *every call*. Causes: branch on a tensor value, `.item()`, `print(tensor)`, untraceable Python/C — i.e.
  what a DUST hook does if it reads an activation, injects Python-sampled noise, scores by a loss value,
  or branches. The three sins (`.item()`, data-dependent branch, arbitrary Python in hooks) break fusion
  **and** CUDA Graphs at once. → *Build:* add a **graph-break / compiled-region counter** to
  `profiling.py` (a compiled DUST step should be 1-2 regions, not one-per-draw); find breaks with
  `TORCH_LOGS="graph_breaks"`, `torch._dynamo.explain`, counting "Torch-Compiled Region" spans; fix
  tensor branches with `torch.where`/`torch.cond`; keep untraceable helpers eager via
  `torch.compiler.disable`; use `fullgraph=True` to raise on the first break.
- **Guards & recompiles (s63-74):** guards are checked every call; a guard failure costs **one compile
  per new version** (one-time); a graph break costs Python **every call** (recurring) — breaks are the
  worse disease. Past `recompile_limit` (8) it silently falls back to eager. → Keep **population and
  model shapes static** ⇒ one compile and CUDA-Graph eligibility; log `TORCH_LOGS="recompiles"`.
- **Fusion saves launches *and* HBM traffic (s17):** 3 eager elementwise kernels (6 HBM transfers,
  384 MB) → 1 fused (2, 128 MB). → DUST's noise-add/scale/activation are prime fusion targets; model the
  saving in `accounting.py` and check it in the trace.
- **Compile cost / break-even (s82-89):** report **cold-compile, warm-first-call, steady-state median,
  peak memory** as *separate* numbers; break-even = compile-time ÷ per-step saving. A GEMM-bound workload
  shows no steady-state speedup → compiling never pays; a pointwise chain gained ~3.8×. → DUST does many
  forwards/step, so compile amortizes **iff** steady-state is actually faster (depends on
  pointwise-heavy vs GEMM-heavy). → Always compare **warmed, steady-state** numbers; profile *after*
  warm-up.
- **AOTAutograd's backward is irrelevant to DUST (no backward); compile still works forward-only.**
  Inductor kernel prefixes in a trace: `triton_poi_` pointwise, `triton_red_`/`triton_per_` reductions,
  `extern_kernels` a cuBLAS call — many tiny pointwise kernels remaining ⇒ fusion didn't happen.
- **Autocast keeps softmax/layernorm/loss + FP32 master weights in FP32.** → Reinforces: the
  true-gradient reference must be FP32. Use `scaled_dot_product_attention` if the net has attention.

## Lecture 5 — CUDA Basics

- **Tiny kernels are dominated by launch cost (~a few µs each); "add 64 values" should go to the CPU or
  be fused (s9).** → DUST's thousands of tiny forwards are the textbook launch-bound pathology; the
  analytic FLOP model **understates** real cost because launch overhead isn't FLOPs — so a large
  measured/analytic gap *is* the launch-bound signature. Keep analytic and measured side by side.
- **Launches are async; sync before reading results or stopping a timer (s38).** → Validates the sync
  requirement; the double-batch test is only valid with a sync in each region.
- **A kernel boundary is a grid-wide barrier; default-stream kernels run strictly in order (s83).** →
  Per-draw forwards on the default stream **serialize** — no overlap, the per-launch CPU gap repeats
  thousands of times. → Motivates CUDA Graphs / population-batching.
- **First launch can JIT from PTX and cache to `~/.nv/ComputeCache` (s58).** → Warm-up discard is
  mandatory; discard enough to clear JIT + autotune.
- **Occupancy & waves (A7 p104, s66-68):** an A100/H100 needs massive parallelism; blocks schedule in
  waves `⌈blocks/SMs⌉`; the SM hides ~500-cycle HBM latency only by having enough warps. A tiny per-draw
  matmul launches few blocks ⇒ most SMs idle ⇒ wasteful even with zero launch overhead. → *Build:* report
  **grid size / wave count** per forward; blocks ≪ #SMs ⇒ parallelism-starved. → Strengthens the case for
  batching the population.
- **Roofline (s90-92):** naive matmul ≈ 0.25 FLOP/byte ⇒ deeply memory-bound (~1% of peak); **tiling
  raises reuse toward the ridge.** → Don't hand-tune DUST's tiny GEMMs; the win is amortizing launches
  and raising intensity by batching, not a faster small kernel.
- **Coalescing (s74-76, A6):** global memory moves in 32-byte sectors; strided/transpose/AoS access
  wastes 8-16× bandwidth. → If a hook reshapes/transposes activations, a tiny matmul can be
  bandwidth-wasting *on top of* launch-bound — separate the two with Nsight Compute's memory section.

## Lecture 6 — Advanced CUDA & Triton (the appendix is the payload)

- **CUDA Graphs (appendix A2, p105) — the named launch-bound fix, and it fits DUST almost perfectly.**
  Record the fixed-shape forward once, replay with a single launch. The slide's recipe *is* the DUST
  pattern: warm up on a side stream → `with torch.cuda.graph(g): static_y = model(static_x)` →
  `static_x.copy_(new_batch); g.replay()`. → DUST replays the same fixed-shape `no_grad` forward
  thousands of times — exactly the static-shape/static-address/repeated-launch case. Each draw becomes
  "copy noise into a static buffer + `g.replay()`". **Requirements/breakers:** static shapes, **static
  tensor addresses**, warm-up before capture, **no CPU sync / `.item()` inside**. → *Concrete DUST
  caveats:* write noise into a **fixed-address buffer** (`noise_buf.normal_()`), never a fresh
  `torch.randn` (allocation moves the address); hooks add noise as pure GPU ops with no Python branch;
  the reward reduction stays on-GPU, scalars pulled only after a batch of replays. `torch.compile(
  mode="reduce-overhead")` uses CUDA graphs internally — the lower-effort first try; a hand-built
  `CUDAGraph` gives more control.
- **The deeper fix the lecture keeps implying: batch the population.** "Fuse small work into a larger
  kernel" (L5 s9), "need many warps to hide latency" (s68), "cuBLAS runs best on large GEMMs" (s59-62).
  → A population axis turns many tiny GEMMs into one large GEMM, fixing launch overhead **and** SM
  under-utilization, and moving toward the compute-bound ridge. CUDA Graphs fixes only launches;
  batching fixes launches *and* the tiny-matmul waste. → *Build:* compare **three regimes — naive loop,
  CUDA-graphed loop, population-batched** — on cost *and* gradient quality, folding the cost axis into
  the existing quality-vs-population curve.
- **Tool-to-job (A3, p106): launch-bound is a Nsight *Systems* finding, not Nsight Compute.** It shows as
  **gaps between kernels on the GPU row while the CPU row runs ahead** issuing launches. → *Build:*
  `profiling.py` should produce an **nsys timeline** and explicitly check for CPU-ahead-of-GPU gaps —
  that visual *is* the launch-bound proof.
- **NVTX (A4, p107):** `torch.cuda.nvtx.range("forward")` labels regions on the nsys timeline
  (`nsys profile -t cuda,nvtx,osrt`). → *Build (cheap, high value):* wrap each draw/population and the
  scoring step in NVTX ranges so the trace separates "forward draws" from "noise gen" from "reward
  reduction" and the per-draw gap is directly measurable.
- **Nsight Compute (A5, p108), only after confirming a kernel is GPU-bound:** `--section SpeedOfLight`
  (compute- vs memory-bound), `MemoryWorkloadAnalysis` (achieved bandwidth / bank conflicts),
  `Occupancy` (achieved), `WarpStateStats` (stall reasons). → Answers "is this tiny matmul even worth
  optimizing." Don't run ncu on a launch-bound step — the kernel isn't the problem.
- **CUDA events time GPU work inside a stream (p96).** Caveat for launch-bound steps: event time looks
  *small* while wall-clock is *large* — **report both; the divergence is the launch tax** (our
  `likely_launch_bound` flag already does this). Cross-check against `triton.testing.do_bench` (p81),
  which does warmup+median correctly.
- **Occupancy — measure before you "fix" it (p49-55):** fast kernels (cuBLAS) also run at low occupancy
  (Volkov); enough independent work per thread hides latency even at low occupancy. → Report *achieved*
  occupancy, treat low occupancy as a symptom to explain, not a target; no occupancy-tuning module.
- **Streams/overlap is NOT DUST's primary lever (p91-95).** DUST is compute-launch-bound and runs
  `no_grad` on-device, not copy-bound. Overlap needs all three of pinned memory + non-default streams +
  async copies, and the legacy default stream serializes with everything. → Build a streams module only
  if profiling shows real transfer cost (e.g. to overlap host-side noise gen) — otherwise skip it.
- **Tensor-core peaks are dtype-specific (p63-68):** PyTorch uses tensor cores for FP32 matmul only with
  `set_float32_matmul_precision("high")` or bf16 autocast. → `accounting.py`'s FLOP→time and roofline
  peak must be **dtype-aware** (don't score a bf16-tensor-core forward against the FP32 roof).

---

## Build / fix backlog (where each lesson lands)

**`accounting.py`**
- Add bytes-moved (dtype-aware) → arithmetic intensity `I = FLOP/byte` → **A100** roofline placement +
  MFU. Distinguish "weights re-streamed per draw" vs "weights resident".
- Report **three regimes** (launch/latency, bandwidth, compute) and an **Amdahl** step-fraction
  attribution (measured GPU-compute fraction `p`) to bound achievable gain.
- Attribute measured step time into {kernel compute, HBM, Python/launch, other}; report grid size /
  wave count per forward; use the profiler's `with_flops` for achieved throughput.
- Emit a reproducibility record carrying DUST knobs (population, σ, #perturbed sites, seed).

**`profiling.py`**
- Read the **trace gaps**, not just the per-op table: GPU-idle gap per step + mean-kernel-vs-gap
  (thin-comb classifier), Python/dispatch CPU-op fraction, hidden `.contiguous()`/copy detection.
- Produce an **nsys timeline + NVTX ranges**; check for CPU-ahead-of-GPU gaps (the launch-bound proof).
- For the compiled path: **graph-break / compiled-region counter** + `TORCH_LOGS` graph_breaks &
  recompiles; profile *after* warm-up.
- Per-kernel deep dive via **ncu** (SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / WarpStateStats)
  only once a kernel is confirmed GPU-bound.

**`gradient_check.py`**
- Reference gradient under **normal grad + FP32** (never `inference_mode`, never bf16); DUST estimate
  path under `inference_mode`. Note the memory asymmetry between the two.
- Fold a **cost axis** into the quality-vs-population curve; judge variants by **steps/s AND
  time-to-target**.

**`dust.py` / `estimate` (implementation candidates — need GPU validation; see Tracks 1 & perf
extensions in `../TODO.md`)**
- Prefer `inference_mode` over `no_grad` on the estimate path.
- **Vectorize the population into a batched forward** (highest-value lever); keep the inner loop
  sync-free (accumulate rewards as GPU tensors, reduce once); keep shapes static; noise into a
  fixed-address buffer.
- Then CUDA Graphs (`torch.compile(mode="reduce-overhead")` first, hand-built `CUDAGraph` if hooks/RNG
  block it); audit hooks for `.item()` / branches / arbitrary Python first.

## One-line takeaway

Measure the regime before trusting any FLOP number: DUST is almost certainly **launch-bound**, which the
roofline alone cannot show, so the profiler must read **trace gaps** (thin comb, CPU-ahead-of-GPU) — and
the cure is to **vectorize the population** and then **CUDA-Graph / compile** the fixed-shape forward,
which only works once the hooks are pure-GPU, sync-free and static.
