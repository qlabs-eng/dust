# GENERAL background bibliography — DUST reimplementation (HPML project)

Category: transformer architecture + backprop and non-backprop credit assignment.
All PDFs in this folder (`dust/literature/general/`). Verified real PDFs, > 40 KB.
Date compiled: 2026-11-06.

---

## 0. The target paper

**DUST** — Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna (2026).
*Dust: Pretraining Transformers Without Backpropagation.* Q Labs. https://qlabs.sh/research/dust
Code: https://github.com/qlabs-eng/dust · BibTeX key `dahal2026backprop`.
File: `dust_qlabs_webpage.html` (web article; no PDF published — saved as raw HTML, 208 KB).
- The paper we reimplement/extend. Zeroth-order **node (activation) perturbation**: jitter each
  linear layer's output per token `y_t -> y_t + sigma a_t`, reward each token's noise by its
  (baseline-centered, gamma-discounted) loss reduction, average reward-weighted noise over K draws to
  get the output error `g_hat_t`, then `G_W = sum_t g_hat_t x_t^T`.
- Core equations: (1) `r_t = sum_{s>=t} gamma^{s-t} c_s`, `c_s = l~_s - l_s`; (2)
  `g_hat_t = -(1/(K sigma)) sum_i r_t^{(i)} a_t^{(i)}`; (3) `G_W = sum_t g_hat_t x_t^T`.
- Baselines it beats: weight-space ES (EGGROLL) by 1e3–1e4x; approaches/exceeds backprop at large K.

---

## 1. Transformer architecture fundamentals (DUST's model = modded-nanoGPT-style decoder)

**Vaswani et al. (2017).** *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
File: `vaswani2017_attention.pdf`
- The transformer decoder DUST trains. Supplies multi-head self-attention, the residual/LayerNorm
  block structure, and the QKV projections that DUST perturbs as separate "attention internals."
- DUST's per-layer credit assignment forms the same `delta · x^T` weight-gradient outer product that
  backprop forms through these linear layers.

**Su et al. (2021).** *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE).*
arXiv:2104.09864 (Neurocomputing 2024). File: `su2021_roformer_rope.pdf`
- RoPE is the positional scheme in the modded-nanoGPT model DUST reimplements (with normalized q/k).
  Needed to match DUST's architecture exactly; rotation is applied inside the attention path that
  DUST jitters.

**Zhang & Sennrich (2019).** *Root Mean Square Layer Normalization (RMSNorm).* NeurIPS 2019.
arXiv:1910.07467. File: `zhang2019_rmsnorm.pdf`
- RMSNorm is DUST's normalization. Relevant to node perturbation because normalization rescales the
  activations being perturbed, affecting the effective noise scale sigma per layer (DUST tunes sigma
  per layer type).

**Brown et al. (2020).** *Language Models are Few-Shot Learners (GPT-3).* NeurIPS 2020.
arXiv:2005.14165. File: `brown2020_gpt3.pdf`
- Canonical decoder-only autoregressive LM pretraining objective (next-token cross-entropy) that
  DUST optimizes without backprop. Establishes the GPT pretraining setup DUST matches against backprop.

**So et al. (2021).** *Primer: Searching for Efficient Transformers for Language Modeling.* NeurIPS 2021.
arXiv:2109.08668. File: `so2021_primer_squared_relu.pdf`
- Canonical source for the **squared-ReLU MLP** activation used in the modded-nanoGPT / DUST model.
  (The other modded-nanoGPT tricks — value embeddings, learned residual mix, QK-norm — originate in
  Keller Jordan's modded-nanoGPT speedrun repo, https://github.com/KellerJordan/modded-nanogpt, which
  is a code artifact, not a paper.)

**Xiao et al. (2023).** *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024.
arXiv:2309.17453. File: `xiao2023_attention_sinks.pdf`
- Canonical source for the **attention-sink** phenomenon (also the motivation for sliding-window
  attention behavior). Relevant to DUST's sliding-window attention and to the special role of the
  first token when crediting attention-internal perturbations.

---

## 2. Backprop foundations (what DUST replaces)

**Rumelhart, Hinton & Williams (1986).** *Learning representations by back-propagating errors.*
Nature 323, 533–536. File: `rumelhart1986_backprop.pdf` (from Hinton's archive, cs.toronto.edu)
- The backprop algorithm DUST abandons. Defines the "error at a layer's output" (delta) and the
  weight gradient as delta outer-producted with the input — exactly the `G_W = sum_t g_hat_t x_t^T`
  form DUST reproduces with a population-estimated delta instead of the chain rule.

**Lillicrap, Santoro, Marris, Akerman & Hinton (2020).** *Backpropagation and the brain.*
Nature Reviews Neuroscience 21, 335–346. File: `lillicrap2020_backprop_and_brain.pdf`
- Review of the biological-plausibility critique of backprop (weight transport, non-locality) that
  motivates forward-only / perturbation learning. Frames why zeroth-order credit assignment like DUST
  is of interest beyond pure compute scaling.

---

## 3. Non-backprop / zeroth-order / evolution-strategy training (the core lineage)

**Sarkar, Fellows, Duque, ... Foerster (2025).** *Evolution Strategies at the Hyperscale (EGGROLL:
Evolution Guided GeneRal Optimisation via Low-rank Learning).* arXiv:2511.16652.
File: `sarkar2025_eggroll.pdf`
- DUST's primary **weight-space-ES baseline**. EGGROLL makes batched ES GPU-efficient via low-rank
  weight perturbations. DUST reports being 1e3–1e4x more population-efficient than an EGGROLL-
  Transformer because it perturbs activations (O(width) dims/token) instead of weights (O(params)).

**Salimans, Ho, Chen, Sidor & Sutskever (2017).** *Evolution Strategies as a Scalable Alternative to
Reinforcement Learning.* arXiv:1703.03864. File: `salimans2017_evolution_strategies.pdf`
- The foundational scalable-ES paper. Supplies the exact Gaussian-smoothing gradient estimator DUST
  instantiates: `grad E[f] = (1/sigma) E_{a~N(0,I)}[ f(theta+sigma a) a ]`, antithetic sampling, and
  the "variance scales with dimension" observation that explains DUST's O(d) sample cost.

**Malladi, Gao, Nichani, Damian, Lee, Chen & Arora (2023).** *Fine-Tuning Language Models with Just
Forward Passes (MeZO).* NeurIPS 2023. arXiv:2305.17333. File: `malladi2023_mezo.pdf`
- Closest prior zeroth-order method for transformer LMs, but in **weight space** and for fine-tuning.
  Uses SPSA/2-point ZO with memory equal to inference. DUST's contribution is doing ZO in
  **activation space** and at **pretraining** scale; MeZO is the natural ZO point of comparison.

**Baydin, Pearlmutter, Syme, Wood & Torr (2022).** *Gradients without Backpropagation (forward
gradients).* arXiv:2202.08587. File: `baydin2022_forward_gradients.pdf`
- Forward-mode automatic differentiation + a random tangent gives an unbiased directional gradient
  in one forward pass. A forward-only alternative to DUST; shares the "unbiased but high-variance,
  variance ~ O(d)" structure. Useful contrast: forward gradients need JVPs; DUST needs only function
  values (true zeroth order).

**Ren, Kornblith, Liao & Hinton (2023).** *Scaling Forward Gradient With Local Losses.* ICLR 2023.
arXiv:2210.03310. File: `ren2023_forward_gradient_local_losses.pdf`
- Shows forward-gradient variance explodes with width and fixes it with **local losses / activity
  perturbation** per group of units — directly analogous to DUST's per-token, per-layer local credit
  assignment that keeps the perturbed dimension (and thus variance) small. Key methodological ancestor.

**Nesterov & Spokoiny (2017).** *Random Gradient-Free Minimization of Convex Functions.* Found. Comput.
Math. 17:527–566. DOI 10.1007/s10208-015-9296-2 (CORE Discussion Paper 2011/1).
**PDF not downloaded** (paywalled; only an auth-gated UCLouvain/Alfresco copy was reachable).
- The canonical **zeroth-order optimization theory**. Proves the Gaussian-smoothed estimator's bias is
  O(sigma) / O(sigma^2) and that random-gradient-free methods need ~O(n) (n = dimension) times more
  iterations than first-order methods — the formal basis for the "variance ~ O(d), so K ~ O(d)"
  claim in the math verification below.

**Lillicrap, Cownden, Tweed & Akerman (2016).** *Random synaptic feedback weights support error
backpropagation for deep learning (feedback alignment).* Nature Communications 7:13276. (arXiv:1411.0247,
"Random feedback weights support learning in deep neural networks.") File: `lillicrap2016_feedback_alignment.pdf`
- Shows learning works without the transposed weight matrix backprop requires — fixed random feedback
  suffices. A different relaxation of backprop's credit assignment; context for why DUST's non-chain-rule
  delta estimate can still train.

**Nøkland (2016).** *Direct Feedback Alignment Provides Learning in Deep Neural Networks (DFA).* NeurIPS
2016. arXiv:1609.01596. File: `nokland2016_direct_feedback_alignment.pdf`
- DFA projects the output error directly to each layer via random feedback, removing sequential backward
  dependency. Like DUST, it supplies each layer a surrogate output error locally; contrast is DFA still
  needs the global output error, DUST estimates each layer's error from the population.

**Millidge, Tschantz & Buckley (2020).** *Predictive Coding Approximates Backprop Along Arbitrary
Computation Graphs.* arXiv:2006.04182 (Neural Computation 2022).
File: `millidge2020_predictive_coding_backprop.pdf`
- Canonical statement that predictive coding (local, forward/iterative inference) can approximate backprop
  gradients. Represents the biologically-motivated "local credit assignment" family that DUST sits beside
  as a search-based (rather than inference-based) alternative.

---

## Foundational references NOT downloaded (noted for completeness)

- **Werfel, Xie & Seung (2003/2005).** *Learning curves for stochastic gradient descent in linear
  feedforward networks.* NeurIPS 2003 / Neural Computation 17:2699–2718. The classical **node-perturbation**
  analysis: perturb unit activities, correlate with the global reward — exactly DUST's mechanism — and the
  reference for node-perturbation's O(d) variance penalty vs backprop. (No open PDF obtained.)
- **Widrow & Lehr (1990).** *30 Years of Adaptive Neural Networks: Perceptron, Madaline, and
  Backpropagation.* Proc. IEEE 78(9):1415–1442. Historical source for weight/node-perturbation (MRI / Madaline
  Rule) learning rules. (Paywalled; no open PDF obtained.)
