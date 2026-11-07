# Adaptive Allocation — Annotated Bibliography

Literature for **Method B: Adaptive Population Allocation** — dynamically redistributing a
fixed total draw budget across DUST module groups (writers / hidden / hubs / embedding / head /
attention-internals) based on measured per-group estimator quality (per-group variance, or
split-estimate agreement as an SNR proxy), instead of the fixed hand-tuned allocation.

Core question answered by this category: *given a fixed total draw budget and per-group
estimator variance, what is the variance-minimizing way to split draws across groups?*
Answer (verified below): the **Neyman / optimal-allocation** rule, draws ∝ per-group standard
deviation (cost- and importance-weighted variants derived in `MATH-VERIFICATION` at the end).

PDFs in this folder are self-contained copies (the repo keeps per-category copies; e.g. MeZO
also appears under `../general/` and `../multiplexing/`).

---

## 1. Optimal / stratified-sampling allocation (the governing theorem)

### Neyman (1934) — On the Two Different Aspects of the Representative Method
- **File:** `neyman1934_representative-method.pdf`
- **Citation:** J. Neyman. "On the Two Different Aspects of the Representative Method: The Method
  of Stratified Sampling and the Method of Purposive Selection." *Journal of the Royal
  Statistical Society*, 97(4):558–625, 1934. DOI 10.2307/2342192. JSTOR 2342192.
- **URL:** https://www.jstor.org/stable/2342192 (scan: https://errorstatistics.com/wp-content/uploads/2019/01/neyman-1934.pdf)
- **Relevance:** The origin of optimal (Neyman) allocation. Proves that to minimize the variance
  of a stratified estimate under a fixed total sample size, the per-stratum sample count m_i must
  be **proportional to M_i·S_i** (stratum size × within-stratum standard deviation) — see the
  scanned text, "when the numbers m_i are proportional to the products M_i S_i." This is the exact
  theorem behind Method B: with equal-weight groups it collapses to **draws ∝ per-group σ**.

### Cochran — Sampling Techniques (textbook reference, no free PDF)
- **Citation:** W. G. Cochran. *Sampling Techniques*, 3rd ed., Wiley, 1977, Chapter 5
  (optimum allocation / Neyman allocation), §5.5.
- **Relevance:** The standard modern statement and proof (Cauchy–Schwarz / Lagrange) of Neyman
  allocation and its **cost-weighted** form n_h ∝ W_h σ_h / √c_h. Cite as the clean reference for
  the allocation formula we implement. (Book — not redistributable; included for citation only.)

---

## 2. Adaptive batch / sample-size selection in stochastic optimization

### Byrd, Chin, Nocedal & Wu (2012) — Sample Size Selection in Optimization Methods for ML
- **File:** `byrd2012_sample-size-selection.pdf`
- **Citation:** R. H. Byrd, G. M. Chin, J. Nocedal, Y. Wu. "Sample size selection in optimization
  methods for machine learning." *Mathematical Programming*, 134(1):127–155, 2012.
- **URL:** https://optimization-online.org/2011/11/3226/ (PDF: https://optimization-online.org/wp-content/uploads/2011/11/3226.pdf)
- **Relevance:** Foundational "grow the sample when the gradient is too noisy" method. Its **norm
  condition** ‖∇J_S − ∇J‖² ≤ θ‖∇J_S‖² is estimated from the *sample variance* of the batch
  gradient and triggers a sample-size increase — i.e. required sample size ≈ Var / (θ‖∇J_S‖²).
  This is exactly the per-group "how many draws do I need to hit a target accuracy" calculation
  Method B must do, and shows variance alone is not enough — it is variance *relative to gradient
  norm*.

### Bollapragada, Byrd & Nocedal (2018) — Adaptive Sampling Strategies for Stochastic Optimization
- **File:** `bollapragada2018_adaptive-sampling-strategies.pdf`
- **Citation:** R. Bollapragada, R. Byrd, J. Nocedal. "Adaptive Sampling Strategies for Stochastic
  Optimization." *SIAM Journal on Optimization*, 28(4):3312–3343, 2018. arXiv:1710.11258.
- **URL:** https://arxiv.org/abs/1710.11258
- **Relevance:** The key reference for *how to decide a sample size from measured noise.* Defines
  the **norm test** E‖∇F_S − ∇F‖² ≤ θ²‖∇F‖² and the stronger **inner-product test** (eqs 2.3–2.6):
  Var_i(∇F_iᵀ∇F)/|S| ≤ θ²‖∇F‖⁴, plus an orthogonality test. These give the correct
  magnitude-aware target for each group: allocate draws so per-group variance ≤ θ² × (gradient
  magnitude)². Directly tells Method B how to convert a split-agreement / variance measurement
  into a draw count.

### McCandlish, Kaplan & Amodei (2018) — An Empirical Model of Large-Batch Training
- **File:** `mccandlish2018_empirical-model-large-batch.pdf`
- **Citation:** S. McCandlish, J. Kaplan, D. Amodei. "An Empirical Model of Large-Batch Training."
  arXiv:1812.06162, 2018.
- **URL:** https://arxiv.org/abs/1812.06162
- **Relevance:** The **gradient noise scale** B_simple = tr(Σ)/‖G‖² (eq 2.9) is the single most
  useful statistic for Method B: it is the ratio of total gradient variance to squared gradient
  norm, i.e. the batch size at which noise and signal balance. Eq 2.10 shows
  E‖G_est − G‖²/‖G‖² = B_simple/B — the normalized gradient error equals noise-scale / batch.
  Per-group B_simple tells us which groups are starved; it is exactly the quantity the split-half
  cosine estimates. Also motivates the cadence: noise scale *grows* during training, so allocation
  must track it.

---

## 3. Importance / adaptive sampling for SGD (the "magnitude-weighted" precedent)

### Zhao & Zhang (2015) — Stochastic Optimization with Importance Sampling
- **File:** `zhao2015_importance-sampling-sgd.pdf`
- **Citation:** P. Zhao, T. Zhang. "Stochastic Optimization with Importance Sampling for
  Regularized Loss Minimization." *ICML 2015*, PMLR 37:1–9. arXiv:1401.2753.
- **URL:** https://arxiv.org/abs/1401.2753
- **Relevance:** Proves that non-uniform sampling proportional to each term's gradient
  Lipschitz/magnitude reduces the variance of the SGD gradient estimate and improves convergence.
  Same optimization as Neyman allocation (effort ∝ per-element "spread"), and evidence that the
  optimal weighting involves gradient *magnitude*, not variance alone — relevant to the
  magnitude-weighting question for Method B.

### Alain, Lamb, Sankar, Courville & Bengio (2015) — Variance Reduction in SGD by Distributed Importance Sampling
- **File:** `alain2015_variance-reduction-importance-sampling.pdf`
- **Citation:** G. Alain, A. Lamb, C. Sankar, A. Courville, Y. Bengio. "Variance Reduction in SGD
  by Distributed Importance Sampling." arXiv:1511.06481, 2015/2016.
- **URL:** https://arxiv.org/abs/1511.06481
- **Relevance:** Shows the **variance-minimizing sampling distribution is p_i ∝ ‖∇f_i‖** (per-sample
  gradient norm) and builds a distributed system to measure those norms cheaply and reallocate
  sampling effort — a direct structural analog of Method B redistributing draws across groups using
  cheap per-group noise measurements. Also a template for the measurement/staleness machinery.

### Katharopoulos & Fleuret (2018) — Not All Samples Are Created Equal
- **File:** `katharopoulos2018_importance-sampling.pdf`
- **Citation:** A. Katharopoulos, F. Fleuret. "Not All Samples Are Created Equal: Deep Learning
  with Importance Sampling." *ICML 2018*, PMLR 80. arXiv:1803.00942.
- **URL:** https://arxiv.org/abs/1803.00942
- **Relevance:** Practical importance sampling with a *cheap upper-bound proxy* for the per-sample
  gradient norm, plus a principled rule for **when importance sampling actually helps** (an
  effective-sample-size / variance-reduction threshold). Method B needs the same "is the signal
  worth acting on?" gate before reallocating, and the same idea of using a cheap proxy (split
  agreement) instead of the exact per-group variance.

---

## 4. Budget allocation under uncertainty (analogy)

### Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2017) — Hyperband
- **File:** `li2017_hyperband.pdf`
- **Citation:** L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar. "Hyperband: A Novel
  Bandit-Based Approach to Hyperparameter Optimization." *JMLR* 18:1–52, 2018. arXiv:1603.06560.
- **URL:** https://arxiv.org/abs/1603.06560
- **Relevance:** Successive-halving / bandit view of splitting a fixed resource budget across
  competing options with noisy quality estimates. Useful framing for the *exploration/exploitation*
  pitfall in Method B: cut a group's draws too aggressively and its variance estimate degrades
  (fewer draws → noisier signal → cannot recover), motivating a per-group **minimum allocation** as
  a measurement floor. Optional/analogy-level.

---

## 5. Zeroth-order specifics (layer-wise / adaptive perturbation budgets)

### Malladi, Gao, Nichani, Damian, Lee, Chen & Arora (2023) — MeZO
- **File:** `malladi2023_mezo.pdf`
- **Citation:** S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, S. Arora.
  "Fine-Tuning Language Models with Just Forward Passes." *NeurIPS 2023*. arXiv:2305.17333.
- **URL:** https://arxiv.org/abs/2305.17333
- **Relevance:** Canonical memory-efficient zeroth-order (SPSA-style) LLM fine-tuner — the ZO
  baseline family DUST belongs to. Establishes that ZO gradient-estimate variance scales with the
  number of effective parameters perturbed, which is the mechanism Method B exploits by moving
  draws toward the groups whose perturbations are noisiest. (Also in `../general/`, `../multiplexing/`.)

### Liu, Zhu, Gong, Cheng, Hsieh & You (2024) — Sparse MeZO
- **File:** `liu2024_sparse-mezo.pdf`
- **Citation:** Y. Liu, Z. Zhu, C. Gong, M. Cheng, C.-J. Hsieh, Y. You. "Sparse MeZO: Less
  Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning." arXiv:2402.15751, 2024.
- **URL:** https://arxiv.org/abs/2402.15751
- **Relevance:** Closest ZO precedent for *non-uniform perturbation allocation*: perturbing only a
  chosen (sparse / layer-selective) subset of parameters reduces ZO variance and improves accuracy,
  empirical evidence that where you spend perturbation budget matters and that concentrating it on
  the right modules helps — the ZO-native version of Method B's group reallocation.

---

---

## MATH-VERIFICATION

Notation: groups i = 1..G. Group i gets K_i draws, total budget fixed Σ_i K_i = K. Let the
**per-draw** gradient-estimate covariance of group i be C_i, and define the per-draw total
variance v_i := tr(C_i) = E‖(1-draw estimate) − ḡ_i‖². Averaging K_i i.i.d. draws gives the
group estimate ĝ_i with

    E‖ĝ_i − ḡ_i‖² = v_i / K_i        (the stated "variance ∝ 1/K").

Let σ_i := √v_i (per-draw standard deviation of group i's estimate), ḡ_i the true group gradient.

### (a) Neyman / optimal allocation — stated and verified

**Theorem (Neyman 1934; Cochran §5.5).** To minimize a stratified estimator's variance under a
fixed total sample size, allocate to stratum h in proportion to W_h σ_h (weight × std); with
per-unit cost c_h, in proportion to W_h σ_h / √c_h.

**Applied to DUST (equal-weight groups), Objective A = minimize the total gradient MSE:**

    minimize  Σ_i v_i / K_i   subject to   Σ_i K_i = K.

Lagrangian ∂/∂K_i [ v_i/K_i + λK_i ] = −v_i/K_i² + λ = 0  ⇒  K_i = √(v_i/λ)  ⇒

    ►  K_i  =  K · σ_i / Σ_j σ_j        (draws ∝ per-group standard deviation σ_i)   ✓

Minimum achievable (Cauchy–Schwarz):  Σ_i v_i/K_i |_opt = (Σ_i σ_i)² / K.

**Cost-weighted** (draw for group i costs c_i, budget Σ c_i K_i = C):
    ►  K_i ∝ σ_i / √c_i.
(In DUST a "draw" is one forward pass; if all groups cost the same per draw, c_i = const and the
√c_i term drops — K_i ∝ σ_i stands. Add √c_i only if groups truly differ in per-draw cost.)

**Verdict on the current plan:** "draws ∝ per-group σ" is **exactly correct — for Objective A**
(minimizing the L2 error of the *stacked/total* gradient vector, weighting every coordinate
equally). It is the right default. Two caveats below.

### (b) Variance only, or also gradient magnitude?

Magnitude enters **only through the choice of objective**, not through Objective A:

- **Objective A (above):** absolute total-gradient MSE → K_i ∝ σ_i. No magnitude term. Correct if
  all coordinates' errors matter equally (reasonable for plain SGD-style use).
- **Importance/curvature-weighted** (group i's error matters with weight w_i — e.g. per-group
  curvature h_i, or an optimizer that scales groups differently; minimize Σ_i w_i v_i/K_i):
      ►  K_i ∝ √(w_i · v_i) = √(w_i)·σ_i   (and with cost: √(w_i v_i / c_i)).
- **Relative / SNR-equalizing** (the Bollapragada *norm test*: want per-group
  v_i/K_i ≤ θ²‖ḡ_i‖²; equalize the ratio r_i = v_i/(K_i‖ḡ_i‖²) across groups):
      ►  K_i ∝ v_i / ‖ḡ_i‖² = (σ_i / ‖ḡ_i‖)².
  This gives **more** draws to high-variance *and* low-magnitude groups — magnitude-aware.

So: pure σ-allocation over- or under-serves groups when groups differ a lot in gradient norm or in
how much their error moves the loss. **Recommendation:** ship K_i ∝ σ_i as the baseline; switch to
the SNR / norm-test target K_i ∝ v_i/‖ḡ_i‖² if (i) the optimizer per-coordinate-normalizes (Adam)
so getting each group's *direction* right matters more than its absolute scale, or (ii) group
gradient norms are very unequal. The split-half cosine (next) natively measures the *relative*
quantity, so it pairs with the SNR target; the split-half *disagreement* measures absolute σ_i and
pairs with the plain Neyman target.

### (c) Split-estimate agreement as an SNR signal — valid, and how to compute it

Split group i's K_i draws into two independent halves (K_i/2 each) → estimates a, b. Both unbiased
(E[a]=E[b]=ḡ_i), independent, with Cov(a)=Cov(b)=2C_i/K_i. Then:

    E[a·b]   = ḡ_i·ḡ_i + <Cov(a,b)> = ‖ḡ_i‖²        (independent ⇒ cross-cov 0)
    E‖a‖² = E‖b‖² = ‖ḡ_i‖² + 2 v_i / K_i
    E‖a − b‖² = tr(Cov(a)+Cov(b)) = 4 v_i / K_i.

**Cosine (relative/SNR signal):**
    E[cos(a,b)] ≈ ‖ḡ_i‖² / (‖ḡ_i‖² + 2 v_i/K_i) = ρ/(ρ+1),  ρ := ‖ḡ_i‖² K_i /(2 v_i) = SNR.
  ⇒ cos→1 high SNR (well-estimated, can cut draws); cos→0 noise-dominated (add draws).
  ⇒ implied noise-to-signal  NSR_i := (1−cos)/cos ≈ 2 v_i/(K_i‖ḡ_i‖²)  —  this is exactly the
    **norm-test ratio** (Bollapragada) / **McCandlish B_simple,i / (K_i/2)** for group i.
  So the cosine **is a valid SNR proxy** and is inherently *relative* (magnitude already in it).

**Squared disagreement (absolute variance signal), for the plain-Neyman target:**
    v_i ≈ (K_i / 4) · ‖a − b‖²   ⇒   σ_i ≈ (√K_i / 2) · ‖a − b‖.   ← K_i-invariant estimate of σ_i.

**Critical implementation note:** σ_i means the **per-draw** std. Do NOT compare the raw variances
of the current K_i-averaged estimates across groups — those already carry a 1/K_i factor, so groups
with different K_i are not comparable. Always rescale to the per-draw σ_i (via ‖a−b‖·√K_i/2, or
multiply a current-estimate variance by its own K_i) before allocating.

**Relation to the two tests:** cosine threshold cos ≥ 1/(1+θ²) ⇔ norm test with tolerance θ² on a
half-population estimate. The inner-product test additionally bounds Var(ĝ_iᵀḡ_i)/‖ḡ_i‖⁴ (direction
reliability); a·b/‖ḡ_i‖² − 1 is a one-sample estimate of its error, so split-agreement blends
norm- and inner-product-test information but is most directly the **norm test**.

### (d) Pitfalls

1. **Noisy variance estimates → allocation thrash.** σ̂_i / ĉos_i are estimated from only K_i draws;
   the relative error of a sample std ≈ 1/√(2(K_i−1)), so with small K_i the signal is noisy and the
   allocation oscillates step-to-step. Mitigate: EMA-smooth per-group v_i/SNR across steps; recompute
   only every N steps (cadence); add a dead-band / hysteresis (reallocate only if change exceeds a
   threshold); cap per-recompute movement of K_i (trust region).
2. **Bias from reallocating mid-run.** Sizing K_i from the *same* draws used for the gradient makes
   the sample size a random variable correlated with the realized noise (optional-stopping bias);
   and σ_i, ‖ḡ_i‖ drift over training (noise scale grows — McCandlish). Mitigate: decide the current
   step's allocation from the *previous* step's statistics (one-step delay), so the decision is
   independent of the gradient it sizes; refresh on the cadence.
3. **Minimum-allocation + integer/divisibility rounding.** The continuous optimum K_i ∝ σ_i must be
   projected onto {integer, ≥ K_min, multiple of world×chunk, Σ = K}. Effects: (i) the K_min floor
   reserves budget — groups whose optimum < K_min get over-served and steal from others (the same
   "over-large stratum" capping issue as classic Neyman, where n_h > N_h must be capped and the rest
   re-solved on the remainder); (ii) rounding to a multiple of world×chunk quantizes shares — with
   small K and many groups the quantization is a large fraction of each share. Use largest-remainder
   (Hamilton) apportionment to the required multiple so Σ K_i = K is preserved. Reassuringly, the
   objective Σ v_i/K_i is flat near its optimum, so *small* rounding costs little; large K_min or
   coarse world×chunk relative to K is what actually hurts.
4. **Exploration/exploitation (Hyperband analogy).** Cutting a group's draws degrades the very signal
   used to re-decide it (fewer draws → noisier σ̂ → may never recover). The per-group **minimum** is
   therefore also a *measurement floor*, not just a safety floor — keep it.
