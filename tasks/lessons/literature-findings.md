# Literature math verdicts

Math verification from the three literature subagents (PDFs in `../../literature/`, per-folder
`bibliography.md`). This file captures the **verdicts and the implications for our code**, which the
bibliographies do not. Sections filled in as each agent completes.

---

## General (transformer + backprop/non-backprop) — DONE

**Verdict: DUST's estimator is correct** (verified analytically and by Monte Carlo). 16 PDFs in
`../../literature/general/`.

### The estimator
- Gaussian-smoothing / Stein identity: for `a ~ N(0,I)`, `∇_y E[f(y+σa)] = (1/σ)E[f(y+σa)·a] → ∇f`
  as `σ→0`. DUST's `g_hat_t = −(1/(Kσ)) Σ_i r_t^{(i)} a_t^{(i)}` is exactly the K-sample,
  baseline-subtracted form of this, crediting token t's perturbation with the γ-discounted future
  loss. So `g_hat_t` is an **unbiased estimate of the smoothed activation gradient** (the "output
  error"); bias vs the true gradient is `O(σ²)` (odd terms cancel by symmetry). `G_W = Σ_t g_hat_t
  x_tᵀ` is then the same `delta·xᵀ` outer product backprop forms. Descent sign is correct.
- **The loss-centering baseline (`l̃`) is NOT optional.** Without it the variance carries a
  `d·f(y)²/σ²` term that blows up as `σ→0` (MC: 2.2e8 vs 3.8e3 at σ=1e-2). Subtracting the mean
  perturbed loss removes it. → Our centering (`direct_errors` draw-chunk mean; `groups`/coordination
  rule) must be preserved exactly by B and C, or variance explodes.

### Variance scaling (the ZO cost)
- Per draw: `Var ≈ (d+1)‖∇f‖²`. Over K draws: **`Var ≈ (d/K)‖∇f‖²`** — classic `O(d/K)`, so
  `K = O(d)` draws for fixed relative error.
- **`d` = per-layer ACTIVATION WIDTH (e.g. 512), not parameter count.** This is exactly why node
  perturbation (DUST) beats weight-space ES/EGGROLL, whose variance is `O(#params) = O(d_out·d_in)`.
  Per-token "virtual population" multiplies effective K by sequence length for free.

### Correction to DUST's SNR heuristic (sharp, actionable)
- DUST says "signal ∝ K, noise ∝ √K, so SNR ∝ √K." The **K-dependence is right**, but the absolute
  SNR is **`√(K/d)`, not `√K`** (MC: SNR/√(K/d) = 0.985, flat in K). There is a hidden `1/√d`.
- **Consequence: interference only becomes negligible once `K ≳ d` (the layer width).** Below that the
  estimate is dominated by noise. DUST's narrative omits this `1/√d`.
- Caveat: the `√K` law assumes **independent zero-mean per-draw errors**. Perturbing many
  tokens/layers in one pass adds a cross-term interference floor the clean argument misses — which is
  why DUST needs separate passes per layer type, per-head jittering, and γ-discounted attention credit.

### Implications for our two methods
- **B (adaptive allocation):** per-group variance ≈ `(d_g / K_g)·‖∇_g‖²`. Neyman says `draws_g ∝ σ_g`,
  and here `σ_g ∝ √(d_g)·‖∇_g‖`. So optimal allocation weights by **√(activation width) × gradient
  magnitude**, not variance alone. Our measured-variance/split-agreement signal captures `d_g‖∇_g‖²`
  empirically (good), but we should sanity-check it against this `√(d_g)‖∇_g‖` closed form, and the
  result confirms allocation must include a **magnitude** term, not just raw variance. [Confirm with
  the adaptive-allocation agent's Neyman verdict.]
- **C (multiplexing):** the `√K` independence assumption is exactly what multiplexing violates —
  simultaneous coded perturbation adds the cross-term interference floor. The nonlinear bias is the
  thing to quantify. [See the multiplexing agent's second-order bias verdict.]

### Files / gaps
- 16 PDFs saved (architecture: Vaswani, RoPE, RMSNorm, GPT-3, Primer/squared-ReLU, attention-sinks;
  backprop: Rumelhart 1986, Lillicrap 2020; non-backprop: EGGROLL, Salimans ES, MeZO, forward
  gradients, Ren local losses, feedback alignment ×2, predictive coding) + DUST webpage (HTML).
- Not obtained (no open PDF): Nesterov & Spokoiny 2017 (ZO theory; paywalled), Werfel 2003 /
  Widrow-Lehr 1990 node-perturbation classics (cited as references only).
- Top reads: Salimans 2017 (ES smoothing + variance), Sarkar 2025 (EGGROLL baseline), Ren 2023
  (local-loss variance control — DUST's ancestor), Malladi 2023 (MeZO), Nesterov-Spokoiny 2017 (O(d)
  theory), Rumelhart 1986 (the `delta·xᵀ` form).

---

## Multiplexing (Method C) — DONE

**Verdict: sound at first order; conditionally sound at second order.** 10 PDFs in
`../../literature/multiplexing/`.

### Decode estimator (first order — exact)
- Code matrix `C` (M×G, ±1, orthogonal `CᵀC = M·I`), per-pass loss change `ΔL_m = L_m − L_0`:
  **`r̂ = (1/M) Cᵀ ΔL`** → under linearity `r̂ = r` exactly; matched-filter / least-squares inverse, the
  `1/M` is just `‖c_g‖² = M`. Use **zero-column-sum codes** (Hadamard minus the all-ones column) so the
  baseline `L_0` cancels. Same template as Tandon gradient-coding (`AB=1`).

### Second-order bias (the crux)
- Taylor to 2nd order, `H_gh` = inter-group Hessian block. Decoding group k:
  `r̂_k = r_k + Σ_{g<h} q_{gh}·T_{k,g,h} + O(σ³)`, with `q_{gh} = σ²·a_gᵀH_gh a_h`.
- **Self-curvature cancels** for any balanced code (`Σ_m s_{m,k}=0`) — the balanced code *is* antithetic
  sampling baked in.
- **Cross-curvature `q_gh` SURVIVES:**
  - **Rademacher:** zero-mean (unbiased in expectation) but leaks `O(σ²‖H_gh‖/√M)` noise per draw.
  - **Hadamard:** Walsh identity `s_g·s_h = s_{g⊕h}` → **structured aliasing**: cross-term killed exactly
    *unless* `k = g⊕h`, where it leaks at **full strength**. Exact first-order separation, but dumps each
    cross-term onto one decoded group.
- **Condition for approximate unbiasedness: (i) small σ** (relative bias `O(σ‖H_gh‖/‖∇_g‖)→0`) **and
  (ii) weak inter-group coupling `H_gh≈0`.** Breaks for strongly-interacting groups at non-negligible σ
  — exactly what DUST's separate-pass design avoids.

### Concrete fixes
- **Hadamard:** assign groups to an **XOR-Sidon / sum-free column set** (no `g⊕h` equals any active `k`)
  → cancels *all* 2nd-order cross-terms exactly. This is the design lever for C.
- **Rademacher:** average over independent code redraws.

### Variance-reduction vs compute-saving — report SEPARATELY
- **Signal-independent noise** (minibatch/label/FP): multiplexing genuinely **reduces variance ∝ M** —
  the real Fellgett/matched-filter win; orthogonal/Hadamard provably optimal (Choromanski **Thm 3.2**:
  unbiased, strictly lower MSE than i.i.d.).
- **Signal-dependent nonlinearity** (cross-curvature): multiplexing **does not reduce it — it creates
  it** (Hadamard aliasing / Rademacher `O(σ²/√M)`). The Fellgett *disadvantage* (shot-noise regime).
- Shared pass saves compute **only when groups share their downstream suffix** (same/adjacent layers): a
  multiplexed pass costs the **shallowest** group's suffix for all groups.

### THE KEY TENSION (headline for C)
The shared suffix that makes multiplexing **cheap** is the shared nonlinearity that makes `H_gh`
**large**. So the cheapest-to-multiplex groups (same layer) are the **most bias-prone**, and the safe
groups (distant, weakly coupled) **save the least compute**. → Start C on same/adjacent-layer groups,
use XOR-Sidon Hadamard codes, keep σ small, and measure the bias empirically (our planned linear-example
check catches first order; only the nonlinear transformer reveals the surviving cross-curvature).

### Files / top reads
- 10 PDFs: Spall 1992 (SPSA), Nesterov-Spokoiny 2017, Salimans 2017 (antithetic), Choromanski 2018
  (orthogonal ES, Thm 3.2), Maheswaranathan 2019 (Guided ES), Rowland 2018 (coupled MC), Tandon 2017
  (gradient coding), Baydin 2022, MeZO, Nijima 2025 (Hadamard multiplex imaging — Fellgett advantage
  *and* disadvantage). Harwit-Sloane 1979 classical source is a book (cited only).
- Top reads: Choromanski 2018 (Thm 3.2), Spall 1992, Tandon 2017, Nijima 2025, Nesterov-Spokoiny 2017.

---

## Adaptive allocation (Method B) — DONE

**Verdict: our plan `K_i ∝ σ_i` is correct** (minimizes total gradient MSE). 10 PDFs in
`../../literature/adaptive-allocation/`.

### The formula to implement
- Neyman (1934): allocate ∝ `W_h·σ_h`; with per-draw cost, ∝ `W_h·σ_h/√c_h`. For DUST, with per-draw
  group variance `v_i` (so `E‖ĝ_i−ḡ_i‖² = v_i/K_i`) and `σ_i = √v_i`, minimizing total gradient MSE
  `Σ_i v_i/K_i` s.t. `Σ K_i = K` gives:

  ```
  K_i = K · σ_i / Σ_j σ_j        # draws ∝ per-group standard deviation  ← ship this
  ```

- Cost-weighted `K_i ∝ σ_i/√c_i`: in DUST a draw is one forward pass; if cost is ~equal across groups
  the term drops. But with Track-1 suffix execution a draw's cost IS block-depth-dependent, so once
  suffix lands, set `c_i` = that group's mean suffix cost from `accounting`. Worth revisiting.
- SNR/norm-test target `K_i ∝ v_i/‖ḡ_i‖²`: switch to this **if using Adam** (per-coordinate
  normalization makes direction matter more than scale) or if group gradient norms are very unequal
  (pure-σ over-serves large-but-noisy groups and starves small-but-important ones).

### Reconciliation with the general-agent result (they AGREE)
- General agent: per-draw group variance `v_i ≈ d_i·‖∇_i‖²` ⇒ `σ_i ≈ √(d_i)·‖∇_i‖` (`d_i` = activation
  width). Adaptive agent: allocate `K_i ∝ σ_i`.
- ⇒ `K_i ∝ √(d_i)·‖∇_i‖`: allocation **already weights by both activation width and gradient
  magnitude**, because both live inside `σ_i`. Magnitude is not a separate knob for the total-MSE
  objective — it only pulls out if you change the objective (SNR/Adam target `K_i ∝ v_i/‖∇_i‖² ≈ d_i`).
  Measuring `σ_i` empirically auto-includes `√d_i`, so our measured-variance approach needs no explicit
  `d_i` term.

### CRITICAL implementation bug to avoid
`σ_i` must be the **per-draw** std. Do NOT compare raw variances of the current `K_i`-averaged
estimates across groups — those carry a `1/K_i` factor and are not comparable. Rescale to per-draw
(multiply a current estimate's variance by its own `K_i`) before allocating.

### Split-estimate signals (verified)
Split `K_i` draws into two independent halves → estimates `a, b`:
- **Squared disagreement → absolute σ:** `E‖a−b‖² = 4v_i/K_i` ⇒ `σ_i ≈ (√K_i/2)·‖a−b‖` (K-invariant).
  **Feed the plain Neyman `K_i ∝ σ_i`.**
- **Cosine → relative SNR:** `E[cos(a,b)] ≈ ρ/(ρ+1)`, `ρ = ‖ḡ_i‖²/(2v_i/K_i)`; `(1−cos)/cos` is the
  norm-test ratio. **Use with the SNR target.**
- Both are one extra reduction over the two halves we already compute. Match signal to objective:
  `‖a−b‖`↔σ-Neyman, `cos`↔SNR.

### Pitfalls (bake into `allocation.py`)
1. **Thrash** (σ̂ rel-error ≈ `1/√(2(K_i−1))`): EMA-smooth, recompute every N steps, hysteresis, cap
   per-recompute movement.
2. **Mid-run bias:** size `K_i` from the **previous** step's statistics (one-step delay), not the draws
   you're currently using.
3. **Min-allocation + rounding:** `K_min` floor steals from groups whose optimum is below it (cap and
   re-solve on the remainder); round to the `world×chunk` multiple via **largest-remainder (Hamilton)
   apportionment** so `Σ K_i = K`. `Σ v_i/K_i` is flat near optimum ⇒ rounding is cheap; large `K_min`
   or coarse `world×chunk` vs `K` is what hurts.
4. **Exploration:** keep the per-group minimum as a **measurement floor** (cutting draws degrades the σ̂
   used to re-decide the group).

### Files / top reads
- 10 PDFs: Neyman 1934, Byrd 2012, Bollapragada 2018 (norm/inner-product tests), McCandlish 2018
  (gradient noise scale), Zhao 2015, Alain 2015 (p∝‖∇f_i‖), Katharopoulos 2018, Li 2017 (Hyperband),
  MeZO, Sparse-MeZO. Cochran *Sampling Techniques* cited only.
- Top reads: Neyman 1934, Bollapragada 2018, McCandlish 2018, Byrd 2012, Alain 2015.

---

## Cross-method summary

- **B math checks out.** Ship `K_i ∝ σ_i`, σ measured per-draw from split-half `‖a−b‖`; it already
  encodes `√(d_i)·‖∇_i‖`. Consider the SNR target under Adam. Mind the per-draw-rescale bug; use
  Hamilton apportionment.
- **C math checks out at first order only.** Exact decode `r̂ = (1/M)CᵀΔL`; second-order cross-curvature
  bias survives and breaks for coupled groups at non-negligible σ. Fix with XOR-Sidon Hadamard codes,
  small σ, same/adjacent-layer groups. The cheap-to-multiplex groups are the bias-prone ones.
- **Both rely on the centering baseline** (general-agent result): without loss-centering, variance
  blows up as `d·f²/σ²`. B and C must preserve DUST's reward centering exactly.
