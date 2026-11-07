# DUST project — refined TODO

Updated 2026-11-07. Companion to `PROJECT_PLAN.md` (the full plan) and `lessons/lectures.md` (how we
measure performance). Each item says what it is and why it matters, not how it's wired.

Legend: `[x]` done · `[~]` in progress · `[ ]` todo · **(GPU)** needs a GPU to validate.

**Where the code lives:** our contribution is in `new_implementation/` (gradient_check, accounting,
profiling, checkpoint, groups — see `new_implementation/README.md`). The original DUST repo stays at
the root (`dust.py`, `model.py`, `prepare_data.py`) so the README's training commands keep working.
Only `dust.py` was modified in place (the estimate/apply split); everything else of ours is additive.

---

## Foundation — done

- [x] **`new_implementation/groups.py` — one shared vocabulary for which parts of the network get perturbed together**
      (the output projections, the hidden MLP, the attention "hubs", embeddings, the head, and the
      attention internals). It maps every weight to its group, says which later layers a given
      perturbation can reach, and counts how many noise draws each group gets. Diagnostics, the cost
      model, and the allocator all read from this one file so they can never disagree.
- [x] **`dust.py` — split one training step into "estimate" then "apply".** Estimate does the
      forward passes and produces a gradient without touching the weights; apply normalizes it, runs
      a safety check, and takes the optimizer step. Same numbers as before, but now the estimate half
      can be measured, profiled, and reused on its own.
- [x] **`dust.py` — label each phase of the estimate** (clean pass, direct draws, embedding, hubs,
      head, local, scalars) so the profiler timeline reads in those terms instead of a wall of raw ops.

## Measurement stack — done (grade the gradient, count the cost, time the run)

- [x] **`new_implementation/gradient_check.py` — score DUST's gradient against the truth.** (Renamed
      from `diagnostics.py`: it checks one thing — gradient fidelity — so the name now says so.)
      Computes the exact backprop gradient to compare against, runs the DUST estimate without changing
      weights, and reports how well they agree (direction and magnitude) per group. Repeats over seeds
      to separate bias from noise, and reproduces the paper's "estimate improves with more draws" curve.
      It does **not** track training or held-out loss over time — that is the experiment campaign
      (below), a separate and later measurement.
- [x] **`new_implementation/accounting.py` — count the cost on paper, and measure it on hardware.**
      Two halves. (1) *Analytic:* from the model's dimensions and the per-group draw counts it counts
      the arithmetic **by formula** — no GPU, nothing runs — and reports it as "forward-pass
      equivalents" (how many clean forward passes' worth of work one update costs), aware that a
      perturbation deep in the network re-runs fewer layers than one near the input. This is what makes
      variants A/B/C/D comparable before we spend any GPU time, and it sizes memory. (2) *Measured:*
      times real steps correctly (CUDA events, warmup discarded, median/p10/p90) and flags when a step
      is limited by CPU launch overhead, not GPU math. (The roofline — FLOPs vs bytes moved — lives
      in `roofline.py`; see Performance extensions.)
- [x] **`new_implementation/profiling.py` — the torch.profiler harness.** Produces the per-op and
      per-phase breakdown and a timeline trace, plus a quick "double the batch and see if the time
      doubles" test that says whether we're GPU-bound or launch-bound before trusting any savings number.
- [x] **`new_implementation/checkpoint.py` — resume a run exactly where it stopped.** Saves everything
      needed to continue bit-for-bit (weights, optimizer momentum, the noise generator's state, the
      allocator's state, the step count), writes it safely, one file per worker.
- [x] **Audited the whole stack against the profiling lecture** and fixed what was off: the timing
      method, warmup, reporting percentiles, a stability guard in the curve fit, the precision the
      reference gradient is computed at, and a caveat that FLOP savings may not show up in wall-clock.

## Tests for the measurement stack (CPU) — in progress

The modules above were written without tests. Each test checks against an independent truth, and a
deliberately broken version of the code must fail it. Run: `python -m pytest new_implementation/`.

- [x] **`test_checkpoint.py`** — a run saved after 2 steps, rebuilt from different seeds, loaded and
      continued is bit-for-bit the 4-step run that never stopped (weights, momentum, noise generator);
      resuming without the generator or the momentum must diverge. Also: bf16 embeddings and the
      per-group learning rates round-trip, no `.tmp` is left behind, per-rank files, world mismatch.
- [x] **`test_gradient_check.py`** — `compare` on known answers; the backprop reference against a
      finite difference along each parameter's gradient (must also reject a half-size gradient);
      seeded, weight-preserving estimates; `fit_cosine_law` recovers a known law; on a tiny model the
      estimate's cosine to backprop rises with population (head 0.07 → 0.18 at 8× draws), and its
      scale matches the centering factor (head projection 0.506 vs the predicted 1/2).
      Fixed two bugs it found in `gradient_check.py`: one seed gave a NaN variance (now raises), and a
      fit with no positive ceiling returned a complex `c_max` (now NaN with a note).
- [ ] `test_groups.py`, `test_accounting.py` (FLOPs vs `FlopCounterMode`, as `test_roofline.py`
      does), `test_profiling.py` (runs end to end on a tiny model).
- [ ] **CI** — run `pytest new_implementation/` on CPU on every push (GitHub Actions). Team decision.
- Note: the estimate's expected size is (1 − 1/n) of the gradient (centering includes the draw
  itself; n = nrep for direct sites, the batch for the head), and smaller still for σ-smoothed and
  attention-internal groups (measured 0.25–0.44 on the tiny model). Read `rel_norm` with that in mind.
- Note: `checkpoint.py` works but `dust.py` never calls it, so runs cannot resume yet.

## Validation — todo (deferred until we have a GPU)

- [ ] **(GPU)** Confirm the new two-half step gives identical numbers to the old fused step under a
      matched seed.
- [ ] **(GPU)** Confirm our "quality improves with more draws" curve matches the paper's within noise.
      This is the cheapest convincing proof the estimator is faithful — the gate for baseline A.
- [ ] **(GPU)** Run the launch-bound test on real hardware, to settle GPU-bound vs launch-bound
      **before** we trust any predicted FLOP saving.

## Track 1 — only re-run the part of the network a perturbation affects (baseline A) — todo

A perturbation injected at some layer can only change the layers after it. Re-running the whole network
to score it is wasted work. This track caches everything up to the injection point and re-runs just the
tail.

- [x] **`model.py`** — add a way to run the model starting from a chosen layer using cached inputs, and
      rebuild the normal forward pass on top of it. (`GPT.forward_from`; `forward` is now
      `forward_from(0, ...)`, and can record the stream entering each block.)
- [x] **`dust.py`** — during the clean pass, cache each block's input; then score the direct draws by
      re-running only the layers after the injection point instead of the full network. On by default;
      `--full-replay` (`DUST(..., suffix=False)`) keeps the public code's behaviour as the upstream
      reference, with the same noise stream. Saved in `config.json` as `execution`.
- [x] **Gate, CPU half:** `new_implementation/test_suffix.py` — on a tiny fp32 model, suffix and full
      replay give identical gradient estimates under a matched seed (max relative difference 0.0), and
      block-execution counts prove draws at block l skip blocks before l. A deliberate off-by-one start
      block fails the check. Predicted saving from `accounting.analytic_cost`: 15.5–17.9% of update
      FLOPs across the shipped recipes (16.4% at 1M/16k).
- [ ] **(GPU) Gate, GPU half:** rerun the equivalence under bf16 autocast (expect small, not zero,
      differences: the replayed prefix runs at batch n·B vs B), then measure runtime and peak memory vs
      `--full-replay` (wall-clock may not move if we turn out to be launch-bound — see lessons L3.1).
- Note: this is the **only** paper optimization still missing from the public code — the other two
  shortcuts are already in it. It's also the groundwork that multiplexing (Track 3) builds on.

## Track 2 — spend noise draws where they help most (B, our idea) — todo

The paper hand-tunes how many draws each group gets and freezes it. Our idea is to decide it live, giving
more draws to the groups whose signal is noisiest.

- [x] **Math verified.** The literature confirms the rule: draws in proportion to each group's noise
      minimizes total gradient error, and a group's importance already enters through its noise, so we
      don't weight it separately. We can measure that noise for free by splitting each step's draws in
      half and comparing. One trap to avoid (don't compare already-averaged numbers). Written up in
      `lessons/literature-findings.md`.
- [ ] **`new_implementation/allocation.py`** — a policy that picks each group's draw count from the measured noise and the
      half-to-half agreement, keeps the total budget fixed, respects a minimum per group, and only
      recomputes every so often.
- [ ] **Free measurement hook** — read each group's noise and self-agreement off the two halves it
      already computes, so adaptivity costs no extra passes.
- [ ] **Integrate** the policy between estimate and apply, and save its state in the checkpoint.
- [ ] **Compare** B against the fixed allocation at equal budget, then find how much smaller a budget B
      needs to match the fixed one's quality.

## Track 3 — perturb several groups at once and untangle them (C) — todo

Instead of testing one group per forward pass, perturb several at once using coded patterns, then
separate out each group's individual effect afterward — the same information from far fewer passes.

- [x] **Math verified.** The decode is exact to first order, but a second-order cross-term (curvature
      between groups) survives and biases the result when groups interact. The fix is a specific
      structured code, small noise, and only multiplexing groups in the same or neighboring layers. The
      catch: the groups cheapest to multiplex together are exactly the ones most prone to this bias.
      Written up in `lessons/literature-findings.md`.
- [ ] **Codebook + decoder** — build the coded patterns and the routine that recovers each group,
      and prove recovery on a controlled linear test case first.
- [ ] **Experiment** — multiplex two groups that share a re-run; compare against testing them
      separately and against plain random codes; measure the real bias and variance in the actual
      (nonlinear) model.
- [ ] **Constraint** — only multiplex same/adjacent-layer groups, or the shared-compute saving erodes.

## Track D — combine B and C — todo

- [ ] Make adaptive allocation and multiplexing agree on sizes (draw counts vs code lengths), integrate
      them, and attribute the speedup to each piece against baselines A, B, and C.

## Model sizes — todo

- [ ] Add the two smaller model configs the size study needs; check their parameter counts and where the
      value embeddings sit; train them with the paper's small-model recipe, not the main-run one.

## Measurement & performance refinements — todo (from the lectures; detail in `lessons/lectures.md`)

All lectures are now read in full. The measurement stack above is built and matches the course method;
these are the refinements the lectures call for, grouped by file.

- [x] **roofline + regimes — `new_implementation/roofline.py`.** Per-op FLOPs, dtype-aware bytes,
      kernel launches and host syncs for every phase of one update (the same phase names as the
      profiler labels), placed on an **A100** (or H100) roofline: compute / bandwidth / launch floors,
      the regime per phase, best-case MFU, plus `mfu()` and `amdahl()` for measured numbers. Models
      both SDPA kernels the explicit mask allows (`efficient`, `math`) and suffix vs full replay.
      `python new_implementation/roofline.py --population 256`. Checked on CPU
      (`test_roofline.py`): its matmul FLOPs match a real `DUST.estimate` under `FlopCounterMode`
      exactly, in both suffix and full-replay modes.
      **Prediction worth testing:** at K=256 the update is *bandwidth*-bound, not launch-bound —
      3.4 s GPU floor vs a 0.34 s launch floor, 67% of GPU time in unfused elementwise ops (rotary,
      relu², the fp32 logit cap every draw, norms, residual ops); weights are <1% of bytes, so they
      are not "re-streamed like decode". With the `math` attention kernel it is 8.2 s, dominated by
      the materialised [T, T] scores.
- [ ] **(GPU) roofline calibration.** Check which SDPA kernel runs; calibrate the per-launch latency
      from a trace; compare predicted vs measured per-phase time; split measured step time into kernel
      compute / HBM / launch / other and feed the measured fractions to `amdahl()`.
- [ ] **profiling — actually see the launch overhead.** Read the trace *gaps*, not just the per-op
      table (GPU-idle gap per step, mean-kernel-vs-gap "thin comb" test); add an nsys timeline with NVTX
      region labels; add a graph-break / compiled-region counter for the compiled path; run Nsight
      Compute on a kernel only once it's confirmed GPU-bound.
- [ ] **gradient_check — fair reference + a cost axis.** Reference under normal-grad FP32, DUST estimate
      under `inference_mode` (not the same memory profile — note it); fold a cost axis into the
      quality-vs-population curve; judge variants by steps/s **and** time-to-target, not step time alone.
- [ ] **the launch-bound fix (biggest lever), GPU-validate.** Vectorize the population into one batched
      forward; keep the inner loop sync-free and shapes static; then CUDA Graphs /
      `torch.compile(mode="reduce-overhead")` on the fixed-shape forward. First audit the hooks for
      `.item()` / branch-on-tensor / arbitrary Python — those break fusion **and** CUDA-Graph capture at
      once. This supersedes the old "scope decision": the course is built on exactly this, so treat it as
      first-class, not out of scope.

## Experiment campaign — todo

- [ ] **Runner** (plan P4): sweep method × budget × seed with equal-budget learning-rate tuning and
      machine-readable output.
- [ ] **Matrix:** freeze the comparison set; run all four variants at the small scale, then the best at
      the larger scale; 3 seeds on the key comparisons; a backprop reference per model size.
- [ ] **Plots:** gradient quality vs compute, held-out loss vs wall/GPU-hours, how the allocation shifts
      over training, how far we can multiplex, and the best model at a fixed budget.

## Literature — done (3 subagents)

Full write-up in `lessons/literature-findings.md`; sources per folder in each `bibliography.md`.

- [x] **General (18 PDFs)** — the DUST estimator is mathematically sound: it's a known
      Gaussian-smoothing gradient estimate, unbiased with a small bias that shrinks as the noise shrinks.
      Its error falls as you add draws, scaled by the **width** of the layer, not the parameter count —
      which is why it beats weight-space methods. One correction to the paper's own wording: the
      signal-to-noise grows with the square root of (draws ÷ width), so you need at least about as many
      draws as a layer is wide before interference cancels. The centering baseline is essential.
- [x] **Adaptive allocation (11 PDFs)** — confirms our idea: draws proportional to each group's noise
      minimize total gradient error, and importance already rides inside that noise rather than needing a
      separate weight.
- [x] **Multiplexing (11 PDFs)** — confirms the decode is exact to first order; the surviving
      second-order bias is real, which is exactly why the structured codes, small noise, and
      same/adjacent-layer-only rule are needed.
- [x] Verdicts folded into Track 2 and Track 3 above.
