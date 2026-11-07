# new_implementation/ — DUST gradient-check, accounting & profiling

Our team's additions to the DUST codebase for the HPML project. Everything here is **new** or
**additive**. The original training path (`../dust.py`, `../model.py`, `../prepare_data.py`) is
untouched except for one in-place refactor noted below, so the run commands in the top-level
`../README.md` still work exactly as before.

## What DUST is (30 seconds)

DUST trains a transformer **without backpropagation**. Instead of a backward pass, it jiggles the
network's activations with small random noise, runs **forward passes only**, and keeps the noise that
lowered the loss. Averaging the reward-weighted noise over many draws (a "population") gives an
estimate of the gradient, which a normal optimizer then applies. One training step is therefore
thousands of tiny forward passes rather than one forward + one backward.

Two consequences drive everything in this folder:

1. The gradient is an **estimate**, so we need a way to check how good it is.
2. The step is **many tiny kernels**, so we need to measure performance carefully — it is likely
   limited by CPU kernel-launch overhead, not GPU math, which changes what "faster" even means.

## The files

- **`gradient_check.py`** — the **ruler**. Computes the true backprop gradient and scores how close
  DUST's noisy estimate is (direction via cosine, magnitude via relative norm), per module group, plus
  variance/bias over seeds and the paper's "quality improves with more draws" curve. It checks one
  step's gradient fidelity only; comparing training/held-out loss over a run is the experiment
  campaign, not this file. (Renamed from `diagnostics.py` — the old name implied a grab-bag; this does
  one thing.)
- **`accounting.py`** — the **cost ledger**, two halves. *Analytic:* counts the arithmetic of one
  update by formula, from the model's dimensions and the per-group draw counts — nothing runs — and
  reports it in "forward-pass equivalents" (how many clean forward passes' worth of work one update
  costs), so variants A/B/C/D are comparable on paper before any GPU time, and memory can be sized.
  *Measured:* times real steps the right way (CUDA events, warmup discarded, median/p10/p90) and flags
  launch-bound steps.
- **`roofline.py`** — the **roofline**: accounting's FLOPs paired with bytes moved, kernel launches and
  host syncs, per op and per phase, placed on an A100/H100 roofline to name each phase's regime
  (compute / bandwidth / launch) and the best-case MFU; plus `mfu()` and `amdahl()` for measured
  numbers. `python roofline.py --population 256 [--attention math] [--full-replay] [--json]`.
  `test_roofline.py` pins its matmul FLOPs to a real `DUST.estimate` under `FlopCounterMode`.
- **`profiling.py`** — the **torch.profiler harness**: the per-op / per-phase table and a Perfetto
  trace, plus a double-batch test that says whether the step is GPU-bound or launch-bound before we
  trust any savings number.
- **`checkpoint.py`** — **full-state save/resume**: weights, optimizer momentum, the perturbation RNG,
  the step counter, and the adaptive-allocator state, written atomically, one file per worker. (The
  stock code saves weights only, which cannot resume a run cleanly.)
- **`groups.py`** — the **shared vocabulary**: which parameters belong to which perturbation "group",
  which later layers a perturbation reaches, and how many draws each group gets. One source of truth
  so the three tools above never disagree. No torch dependency.

## What changed in the original code

- **`../dust.py`** — the one in-place change. `DUST.step` was split into `estimate()` (the forward
  passes that produce a gradient, no weight change) + `apply()` (normalize, safety-check, optimizer
  step), with a thin `step()` wrapper that is bit-identical to the old fused step. Each phase of
  `estimate()` is tagged with `record_function('dust/<phase>')` so a profiler trace reads in DUST
  terms (clean / direct / embedding / hubs / head / local / scalars). This split is what lets
  `gradient_check.py` run the estimate without ever touching the weights.
- **Cached-suffix execution (Track 1, baseline A).** `../model.py` gains `GPT.forward_from(start, x,
  x0, idx, ...)` (the full `forward` is `forward_from(0, ...)` and can record the residual stream
  entering each block). `../dust.py` caches those streams in the clean pass, and a direct draw at block
  l reruns only blocks l onward; the token embedding still reruns everything. `--full-replay` restores
  the public code's whole-network replay for the upstream reference. `test_suffix.py` is the CPU
  equivalence gate (`python -m pytest new_implementation/test_suffix.py`).
- Nothing else in the original repo was modified.

## How it wires together

`accounting` imports `groups`; `profiling` imports `accounting`; `gradient_check` imports `groups` and
reaches back to `../dust.py` for one helper (`average_ranks`) through a small `sys.path` shim at the
top of the file, so it works both run as a script and imported as a module. **No file in the original
repo imports anything here**, which is why moving this folder out cannot break the training path.

## Where the rest lives

- `../tasks/PROJECT_PLAN.md` — the full project plan.
- `../tasks/TODO.md` — done vs. remaining, per track, in plain language.
- `../tasks/lessons/` — distilled lecture notes (how to profile and diagnose correctly) and the
  literature math verdicts for the two novel methods.
