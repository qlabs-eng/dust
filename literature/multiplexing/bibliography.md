# Literature: PERTURBATION MULTIPLEXING (DUST extension — Method C)

Category for the proposed **Structured Perturbation Multiplexing** method: perturb multiple module
groups in a *shared* set of forward passes using sign codes (Hadamard/Rademacher), then decode each
group's contribution by correlating per-pass loss changes against its code column.

All PDFs in this folder verified as real PDFs (>40 KB) with correct titles/authors.

---

## 1. Simultaneous perturbation (basis for perturbing many coordinates at once)

### Spall (1992) — SPSA
- **Citation:** J. C. Spall, "Multivariate Stochastic Approximation Using a Simultaneous Perturbation
  Gradient Approximation," *IEEE Trans. Automatic Control*, 37(3):332–341, 1992.
- **URL:** https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_TAC92.pdf  (file: `spall1992_spsa.pdf`, scanned IEEE copy)
- **Relevance to C:** The ancestor of multiplexed credit assignment. SPSA perturbs *all* coordinates
  simultaneously with a single random vector Δ and still gets a per-coordinate gradient estimate
  `[y(θ+cΔ)−y(θ−cΔ)]/(2cΔ_i)`. The reason it works is exactly method C's reason: Δ_i are i.i.d.
  **±1 (Rademacher), zero-mean with finite inverse moments**, so the cross-coordinate terms
  `E[Δ_j/Δ_i]=0` cancel in expectation. Shows one perturbed pass can probe many sites — and names the
  precise statistical condition (zero-mean symmetric codes) that makes separation unbiased.

### Nesterov & Spokoiny (2017) — Random gradient-free minimization
- **Citation:** Yu. Nesterov and V. Spokoiny, "Random Gradient-Free Minimization of Convex Functions,"
  *Foundations of Computational Mathematics*, 17(2):527–566, 2017 (first as CORE DP 2011/1).
- **URL:** https://doi.org/10.1007/s10208-015-9296-2  (file: `nesterov2017_random_gradient_free.pdf`)
- **Relevance to C:** Defines Gaussian smoothing f_μ(x)=E_u f(x+μu) and the two-point estimator, and
  **quantifies the smoothing bias as O(μ)** (here O(σ)) arising from the second-order Taylor curvature.
  This is the analytical foundation for Method C deliverable (b): the surviving multiplexing bias is a
  second-order (Hessian) effect that → 0 as the perturbation scale σ → 0.

---

## 2. Orthogonal / structured / antithetic directions (variance reduction)

### Salimans et al. (2017) — Evolution Strategies as a Scalable Alternative to RL
- **Citation:** T. Salimans, J. Ho, X. Chen, S. Sidor, I. Sutskever, arXiv:1703.03864, 2017.
- **URL:** https://arxiv.org/abs/1703.03864  (file: `salimans2017_evolution_strategies.pdf`)
- **Relevance to C:** Gives the canonical ES/NES gradient estimator and uses **antithetic ("mirrored")
  sampling to reduce variance** by cancelling odd-order error terms. Method C's *balanced* codes
  (zero column-sum, equal +/− per group) ARE antithetic sampling embedded in the code matrix — this is
  what cancels each group's own self-curvature term at second order.

### Choromanski et al. (2018) — Structured Evolution with Compact Architectures
- **Citation:** K. Choromanski, M. Rowland, V. Sindhwani, R. Turner, A. Weller, arXiv:1804.02395, ICML 2018.
- **URL:** https://arxiv.org/abs/1804.02395  (file: `choromanski2018_structured_evolution.pdf`)
- **Relevance to C:** The central variance-reduction result. **Theorem 3.2**: the orthogonal antithetic
  ES estimator is **unbiased and has STRICTLY LOWER MSE** than the i.i.d. estimator, and they use
  structured (Hadamard-type) orthogonal matrices for cheap orthogonality. This grounds deliverable (c):
  orthogonal/Hadamard codes genuinely *reduce variance* (not merely save passes) — but, as the
  decomposition below shows, only for the part of per-pass variation that is signal-independent.

### Maheswaranathan et al. (2019) — Guided Evolutionary Strategies
- **Citation:** N. Maheswaranathan, L. Metz, G. Tucker, D. Choi, J. Sohl-Dickstein, arXiv:1806.10230, ICML 2019.
- **URL:** https://arxiv.org/abs/1806.10230  (file: `maheswaranathan2019_guided_es.pdf`)
- **Relevance to C:** Formal bias–variance tradeoff when perturbation directions are drawn from an
  informed (correlated) subspace rather than isotropically. Relevant when the groups' base noise
  directions a_g are not independent, and for reasoning about how many multiplexed draws to allocate.

### Rowland et al. (2018) — Geometrically Coupled Monte Carlo Sampling
- **Citation:** M. Rowland, K. Choromanski, F. Chalus, A. Pacchiano, T. Sarlos, R. Turner, A. Weller,
  *NeurIPS* 31, 2018.
- **URL:** https://proceedings.nips.cc/paper/2018/hash/b3e3e393c77e35a4a3f3cbd1e429b5dc-Abstract.html
  (file: `rowland2018_geometrically_coupled_mc.pdf`)
- **Relevance to C:** Proves (via optimal transport) that **orthogonal coupling of MC samples is optimal**
  for variance reduction and analyzes antithetic pairs. The theoretical reason orthogonal sign-codes
  (Hadamard) beat i.i.d. Rademacher codes for clean separation of superimposed perturbations.

---

## 3. Coded / multiplexed measurement

### Tandon et al. (2017) — Gradient Coding
- **Citation:** R. Tandon, Q. Lei, A. G. Dimakis, N. Karampatziakis, arXiv:1612.03301, ICML 2017.
- **URL:** https://arxiv.org/abs/1612.03301  (file: `tandon2017_gradient_coding.pdf`)
- **Relevance to C:** The clean coding-theoretic template. Encode gradients with matrix **B**, decode with
  **A** such that **AB = 1** recovers the exact gradient sum. This is the linear-algebra skeleton of
  Method C's decode `r̂ = (1/M) Cᵀ ΔL`. Important caveat it makes explicit: exact recovery relies on the
  measurement being **linear** in the encoded quantities — precisely the assumption Method C must defend.

### Nijima et al. (2025) — Hadamard Transform Spectral Imaging
- **Citation:** J. Nijima, Z. Ninkov, D. Vorobiev, K. Kearney, "Performance analysis of a Hadamard
  Transform Spectral Imaging system," arXiv:2510.20195, 2025. (Classical source: Harwit & Sloane,
  *Hadamard Transform Optics*, 1979.)
- **URL:** https://arxiv.org/abs/2510.20195  (file: `nijima2025_hadamard_spectral_imaging.pdf`)
- **Relevance to C:** The classical **multiplex (Fellgett) advantage** made concrete: measurements
  η = Hψ are decoded by inverting the Hadamard transform (Method C's exact analog). Crucial finding:
  the √N SNR gain holds **only under signal-independent (read) noise** and shrinks/reverses under
  **signal-dependent (shot) noise**. Direct analogy: Method C's nonlinear cross-terms are
  "signal-dependent" (they scale with the perturbations that are switched on), so multiplexing does
  **not** reduce them — the Fellgett *disadvantage* is the optical mirror of our cross-curvature bias.

---

## 4. Node-perturbation / forward-gradient perturbing multiple sites

### Baydin et al. (2022) — Gradients without Backpropagation
- **Citation:** A. G. Baydin, B. A. Pearlmutter, D. Syme, F. Wood, P. Torr, arXiv:2202.08587, 2022.
- **URL:** https://arxiv.org/abs/2202.08587  (file: `baydin2022_gradients_without_backprop.pdf`)
- **Relevance to C:** Forward-gradient (forward-mode JVP) gives an **unbiased** single-direction gradient
  estimate in one forward pass — the no-backprop, forward-only regime DUST/Method C live in. Method C is,
  in effect, multiplexing several such directional probes into shared passes and de-mixing them.

### Malladi et al. (2023) — MeZO: Fine-Tuning LMs with Just Forward Passes
- **Citation:** S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, S. Arora, arXiv:2305.17333,
  NeurIPS 2023.
- **URL:** https://arxiv.org/abs/2305.17333  (file: `malladi2023_mezo.pdf`)
- **Relevance to C:** SPSA applied to LLM fine-tuning with forward passes only — the DUST-adjacent
  setting. Documents that ZO/SPSA gradient-estimate **variance scales with dimension**, which motivates
  amortizing passes via multiplexing AND is the reason multiplexing does *not* fix the fundamental
  d-dependence (it re-allocates passes across groups; it does not shrink per-estimate dimension cost).

---

# Method C math verification (the key deliverable)

**Setup.** Groups g = 1..G. In forward pass m (m = 1..M) group g's activations are perturbed by
`s_{m,g} · σ a_g`, where `s_{m,g} = C_{m,g} ∈ {+1,−1}` is the sign code and a_g is the group's fixed
base noise field for that block of passes. Code matrix C is M×G with columns c_g; orthogonal codes
satisfy `Cᵀ C = M·I` (Hadamard exactly; Rademacher in expectation). Let L_m be the pass-m loss,
L_0 the unperturbed loss, ΔL_m = L_m − L_0. The per-group "reward" we want is the first-order
directional loss change `r_g = σ a_gᵀ ∇_g L`.

## (a) Linear decoding — exact, unbiased, with 1/M normalization
If L is linear in each group's perturbation, `ΔL_m = Σ_g s_{m,g} r_g + ε_m = (C r)_m + ε_m`.
**Decode estimator:**

    r̂_h = (1/M) Σ_{m=1}^M C_{m,h} (L_m − L_0) = (1/M) c_hᵀ ΔL ,   i.e.  r̂ = (1/M) Cᵀ ΔL.

Then `r̂ = (1/M) Cᵀ C r = (1/M)(M I) r = r` exactly (the matched filter / least-squares solution,
since `(CᵀC)⁻¹Cᵀ = (1/M)Cᵀ`). **The 1/M comes from `‖c_g‖² = M`** (M entries of ±1). With zero-mean
measurement noise, `E[r̂_h] = r_h` (unbiased) and per-group noise variance is `σ_ε²/M` — the ×M
matched-filter / Fellgett SNR gain. Use codes with **zero column sum** (Hadamard minus the all-ones
column) so the baseline/DC term L_0 is rejected automatically: `(1/M) c_hᵀ 1 = 0`.

## (b) Nonlinearity — second-order cross-term bias
Second-order Taylor of the loss in the stacked perturbation `δ_m = Σ_g s_{m,g} σ a_g`:

    ΔL_m = Σ_g s_{m,g} r_g  +  D  +  Σ_{g<h} s_{m,g} s_{m,h} q_{gh}  +  O(σ³)

where `D = Σ_g (½)σ² a_gᵀ H_gg a_g` is a **per-pass constant** (self-curvature; its coefficient is
`s_{m,g}² = 1`, pass-independent) and `q_{gh} = σ² a_gᵀ H_gh a_h` is the **cross-curvature** through the
inter-group Hessian block `H_gh = ∂²L/∂y_g∂y_h`. Decoding group k:

    r̂_k = r_k  +  D·[(1/M)Σ_m s_{m,k}]  +  Σ_{g<h} q_{gh} · T_{k,g,h}  +  O(σ³),
           with triple-product   T_{k,g,h} = (1/M) Σ_m s_{m,k} s_{m,g} s_{m,h}.

- **Self-curvature D is CANCELLED** by any zero-column-sum (balanced/antithetic) code: `(1/M)Σ_m s_{m,k}=0`.
  So even group k's own second-order term does not bias r̂_k. This is the antithetic cancellation
  (Salimans) delivered by the code itself.
- **Cross-curvature q_gh SURVIVES**, weighted by the triple product T_{k,g,h}:
  - **Rademacher codes:** `E[T]=0` for all distinct triples (independent zero-mean columns), so
    multiplexing is **unbiased in expectation**, but T has std `1/√M` → each realization leaks a
    zero-mean nuisance of size `O(σ²‖H_gh‖/√M)`; also first-order cross-talk `(1/M)c_kᵀc_g` is O(1/√M).
  - **Hadamard (Walsh) codes:** the elementwise column product equals another column:
    `s_{·,g}·s_{·,h} = s_{·,g⊕h}`, so `T_{k,g,h} = δ_{k, g⊕h}`. This is **structured aliasing**:
    cross-curvature is cancelled EXACTLY unless the Walsh indices satisfy **k = g⊕h**, in which case
    q_gh leaks into r̂_k at **full strength (coefficient 1)**. Hadamard gives exact first-order
    separation (zero first-order cross-talk) but does NOT generically kill second-order cross-terms —
    it maps each one deterministically onto one decoded group.

**Surviving bias** ∝ σ² · (inter-group Hessian cross-block ‖H_gh‖) · (triple-product coefficient).
**Approximately unbiased iff:** (i) **small σ** — relative bias is `O(σ·‖H_gh‖/‖∇_g L‖) → 0`; and
(ii) **weak inter-group coupling** `H_gh ≈ 0` (groups on disjoint/distant computation paths).
For Hadamard, additionally assign groups to columns forming an **XOR-Sidon / sum-free index set** (no
g⊕h equals any active k) to cancel ALL second-order cross terms exactly; for Rademacher, average over
independent code redraws to drive the O(1/√M) residual down.

## (c) Variance reduction vs. compute saving — they are SEPARATE and must be reported separately
Split per-pass variation into (i) signal-independent noise (minibatch/label/FP noise, same regardless of
which groups are on) and (ii) signal-dependent nonlinearity (the cross-curvature, which scales with the
switched-on perturbations):
- **On (i), multiplexing is genuine variance reduction**, ∝ M (Fellgett/matched-filter advantage),
  and orthogonal/Hadamard codes are provably optimal (Choromanski Thm 3.2 lower MSE; Rowland optimality;
  Salimans antithetic). This is the real win, and it is exactly the regime where Hadamard spectroscopy's
  multiplex advantage holds (Nijima: read-noise-dominated).
- **On (ii), multiplexing does NOT reduce variance — it CREATES error.** The nonlinear cross-terms are
  aliased (Hadamard) or injected as O(σ²/√M) noise (Rademacher). This is the Fellgett *disadvantage*
  (shot-noise regime). Orthogonality cannot remove it because it is signal-dependent.
- **Verdict:** multiplexing = *variance reduction* on the measurement-noise part, *pass-saving-with-
  added-bias* on the nonlinearity part. Do not conflate the two; measure each.

## (d) Net compute saving requires a shared suffix
Perturbing group g at depth d_g forces recomputation of layers d_g..L_max (upstream is unchanged and
reusable in either scheme). Per-pass cost of a multiplexed pass is set by the **shallowest** perturbed
group: `L_max − min_g d_g + 1`. So M multiplexed passes cost `M·(L_max − min_g d_g + 1)`, versus
`Σ_g (draws_g)·(L_max − d_g + 1)` for separate passes.
- Multiplexing **saves compute only when groups share most of their downstream path** (same/adjacent
  layers, so `min_g d_g ≈ d_g` for all). Multiplexing **distant** layers (e.g. embedding d≈1 with head
  d≈L_max) **pays the earliest layer's full suffix for every group**, saving ~nothing on the deep ones.
- **Fundamental tension (key finding):** the shared suffix that makes multiplexing cheap is the SAME
  shared nonlinearity that makes `H_gh` large. So the compute-saving regime (same-layer groups) coincides
  with the high cross-curvature-bias regime, and the low-bias regime (distant, weakly-coupled groups)
  coincides with the no-compute-saving regime. The only escapes are small σ and bias-aware code
  assignment (XOR-Sidon Hadamard).
