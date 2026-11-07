# DUST project: TODOs and parallel work plan

Updated: November 4, 2026

## Starting point and scope

-   The official repository is cloned. GCP, dependencies, data, and experiments are not set up.
-   Starting upstream commit: `b20f7c03eac630b6441ba6f254128c1761dc45ad`.
-   Available budget: \$3,000 in GCP credits, subject to credit eligibility and expiration.
-   Team size: five people. Person numbers below are suggested roles, not assigned names.
-   Research scope: restore relevant functionality explicitly described by the DUST authors, then implement the proposal's two methods: adaptive population allocation and structured perturbation multiplexing.
-   The linear credit scan, compilation, kernel fusion, and attention-backend changes previously suggested are outside this plan.

| Variant | Definition |
|------------------------------------|------------------------------------|
| Upstream reference | Untouched public implementation, retained for reproduction and restoration checks |
| A | DUST with the authors' cached suffix execution restored |
| B | A + adaptive population allocation |
| C | A + structured perturbation multiplexing |
| D | A + both proposal methods |

All A-D comparisons share the same restored baseline. Improvements from restoring suffix execution are reported separately from improvements from the two proposal methods.

## Ordered TODOs

### 1. Set up GCP, billing, and GPU access

Owner: Person 1. Can start immediately.

-   [ ] Create a team GCP project and attach the billing account containing the credits.
-   [ ] Verify credit eligibility for Compute Engine GPU usage, remaining balance, and expiration.
-   [ ] Give teammates the access needed for VMs, storage, and experiment work.
-   [ ] Enable the Compute Engine API.
-   [ ] Choose a supported region and request the required quota for one A100 40GB, including the applicable global GPU quota.
-   [ ] Confirm capacity when provisioning; approved quota does not guarantee availability.
-   [ ] Configure project spending alerts and a routine for stopping idle VMs. Alerts-only budgets do not automatically stop spending.
-   [ ] Reserve approximately \$600 for failures and follow-up experiments.

Done when: the project can provision one GPU VM and the team can see its credit usage and costs.

### 2. Create the VM and reproducible environment

Owner: Person 1. Depends on step 1.

-   [ ] Start with one on-demand A100 40GB VM, such as `a2-highgpu-1g`, for setup and debugging.
-   [ ] Configure SSH, persistent storage, and a Cloud Storage bucket for shared data and results.
-   [ ] Install a compatible NVIDIA driver and the repository's pinned environment: Python 3.10+, PyTorch 2.10.0 with CUDA 12.8, and `requirements.txt`.
-   [ ] Verify `nvidia-smi`, CUDA availability, and BF16 support.
-   [ ] Clone the team repository on the VM and preserve the untouched upstream commit.
-   [ ] Record the VM type, GPU, operating system, driver, Python/PyTorch versions, and installation steps.

Done when: a teammate can reproduce the environment and run a CUDA operation successfully.

### 3. Prepare and preserve the dataset

Owner: Person 1. Depends on a usable Python environment; preparation itself does not require a GPU.

-   [ ] Use the bundled BPE-4096 tokenizer.
-   [ ] Run `python prepare_data.py --tokens 1M` from the repository root.
-   [ ] Preserve the generated training, validation, test, and manifest files, including hashes.
-   [ ] Share one fixed dataset across all methods and people rather than preparing separate copies independently.
-   [ ] Prepare the 10M-token training file for the later study using the official script.

Done when: everyone can load the same data and verify matching hashes. Keep validation/test data fixed across the study.

### 4. Run the unchanged repository and measure its cost

Owner: Person 1, with Person 3. Depends on steps 2-3.

-   [ ] Run a short DUST smoke check:

    ``` bash
    python dust.py --tokens 1M --population 256 --steps 2 --output runs/upstream-smoke
    ```

-   [ ] Run a short backpropagation smoke check:

    ``` bash
    python baselines/backprop.py --tokens 1M --steps 2 --output runs/backprop-smoke
    ```

-   [ ] Use a fresh output directory for each run.

-   [ ] Complete a 1M-token upstream reference run at population 256 and save its configuration, metrics, and checkpoint.

-   [ ] Measure warmed-up training-step time, evaluation/checkpoint overhead, peak GPU memory, and actual cost.

-   [ ] Estimate the cost of the proposed experiment matrix before scheduling it. Repeat the estimate after baseline restoration.

Done when: the unchanged code runs successfully and there is a measured runtime/cost estimate. One reference run is a sanity check, not a claim of reproducing the paper's multi-seed mean.

### 5. Restore the authors' cached suffix execution: baseline A

Owner: Person 2. Implementation can begin immediately; GPU validation depends on steps 2-4 and diagnostic support from step 6.

-   [ ] Cache the clean state needed to restart a direct perturbation evaluation at the affected block.
-   [ ] Recompute that block onward, as described in paper section 2.3 and Appendix E.
-   [ ] Preserve existing perturbation sites, noise scales, draw allocations, normalization, and logical reward-centering groups.
-   [ ] Compare losses and estimated gradients against full replay using matched perturbations and numerical tolerances.
-   [ ] Measure runtime and memory against the untouched reference.
-   [ ] Save the validated restored implementation as baseline A, shared by B, C, and D.

Done when: restoration passes the comparison checks and any timing changes are measured. Keep this implementation change separate from the two proposal contributions.

### 6. Build shared diagnostics, configuration, and accounting

Owner: Person 3. Can start immediately against the upstream implementation; finalize with Person 2's restored baseline.

-   [ ] Save and load identical checkpoints and batches for comparisons across A-D.
-   [ ] Compare estimated gradients with backpropagation gradients, using the same loss normalization.
-   [ ] Measure cosine similarity, gradient norms, variance, and empirical bias over repeated perturbation seeds.
-   [ ] Record direct draws, local-attention draws, head draws, scalar evaluations, suffix-forward evaluations, and estimated FLOPs.
-   [ ] Record training latency, evaluation overhead, peak memory, wall time, GPU-hours, and cost.
-   [ ] Save method, model, optimizer, population, allocation, noise, centering, seed, and source-version settings with every run.
-   [ ] Expose the existing relevant settings through saved configurations so experiments do not depend on manual constant edits.
-   [ ] Implement the needed author-described cosine diagnostics and bounded learning-rate/momentum sweeps. Use shipped winners initially; give compared methods comparable tuning budgets.
-   [ ] Use backpropagation only for diagnostics and separate reference runs, never to update B-D.

Done when: every method can be evaluated through the same harness and its total compute is accounted for. Nominal population K excludes some work in the public recipe; K alone is not a complete compute budget.

### 7. Implement adaptive population allocation: B

Owner: Person 4. Policy development can begin immediately; integrated validation depends on steps 5-6.

-   [ ] Split each module group's perturbation samples into independent subsets and compare their gradient estimates.
-   [ ] Implement the proposal's gradient-agreement policy and variance-based policy.
-   [ ] Redistribute a fixed total budget while maintaining a minimum allocation for every group.
-   [ ] Respect integer counts, draw-chunk/GPU divisibility, normalization, and centering requirements.
-   [ ] Vary how frequently allocation is recomputed and measure its overhead.
-   [ ] Compare B with A at matched budgets, then test smaller budgets for matching baseline quality.
-   [ ] Log allocation changes for the planned visualization.

Done when: B respects its declared budget and has measured gradient-quality and runtime results, including allocation overhead.

### 8. Implement structured perturbation multiplexing: C

Owner: Person 5. Codebook and decoding development can begin immediately; integrated validation depends on steps 5-6.

-   [ ] Implement Hadamard-style signed code sequences and contribution decoding.
-   [ ] Verify code generation and recovery on a controlled linear example.
-   [ ] Start with two multiplexed module groups in shared suffix-forward evaluations.
-   [ ] Compare against separate group evaluations and i.i.d. Rademacher sign codes at matched group-draw counts and noise scales.
-   [ ] Account for every code-sequence evaluation; count each perturbed group's draw separately within a shared evaluation.
-   [ ] Measure empirical bias, variance, cosine similarity, and runtime in the nonlinear transformer.
-   [ ] Increase multiplexing degree only after the two-group case is understood.

Done when: C has verified decoding and measured transformer results. Orthogonal codes do not by themselves guarantee nonlinear separation or a net compute saving.

### 9. Combine the proposal methods: D

Owners: Persons 4-5; Person 2 integrates training-core changes. Depends on independently checked B and C.

-   [ ] Make adaptive allocations compatible with multiplexing group sizes, code lengths, and chunk requirements.
-   [ ] Integrate B and C into D.
-   [ ] Repeat the same budget, normalization, diagnostic, and timing checks.
-   [ ] Compare D with A, B, and C to establish whether combining the methods helps.

Done when: D passes the shared harness and has attributable results against the individual methods.

### 10. Add and validate the smaller model configurations

Owners: Persons 2-3. Configuration work can start early; validated GPU comparisons depend on steps 5-6.

-   [ ] Add the paper's 2-layer/128-width and 4-layer/256-width configurations alongside the released 8-layer/512-width model.
-   [ ] Verify parameter counts, head dimensions, value-embedding placement, and architecture settings.
-   [ ] Map allocation tilt to corresponding relative depth and preserve the intended draw totals, following Appendix E.
-   [ ] Use and document the appropriate optimizer protocol from the paper's model-size study; do not silently copy all main-comparison settings.
-   [ ] Check valid chunk sizes, complete parameter coverage, and population accounting at each size.
-   [ ] Run short baseline and diagnostic checks before expanding to A-D training runs.

Done when: all selected sizes have valid configurations and comparable budget accounting. Changing only width and layer count is insufficient because the released recipes assume eight layers.

### 11. Freeze the experiment matrix, run comparisons, and report

Owners: Person 1 schedules runs; Person 3 aggregates results; everyone contributes method checks and interpretation. Depends on the selected methods and model configurations passing their gates.

-   [ ] Select promising configurations using checkpoint diagnostics before full training.
-   [ ] Freeze matched data, token budgets, seeds, hardware, accounting rules, and comparable tuning budgets.
-   [ ] Run A-D initially at the 1M-token budget and selected small populations.
-   [ ] Repeat key training comparisons over three seeds and summarize variability.
-   [ ] Run population and model-size comparisons sized from measured costs. Target a common 10M-token budget for the model-size study; if unaffordable, apply the same smaller budget across compared sizes and methods.
-   [ ] Include a backpropagation reference at each studied size.
-   [ ] Select settings using validation data and reserve test results for final reporting.
-   [ ] Produce plots for gradient quality versus compute, held-out loss versus wall time/GPU-hours, allocation changes, multiplexing degree, and model size at fixed budgets.
-   [ ] Document negative results and overhead as well as improvements.
-   [ ] Preserve configurations, source versions, logs, checkpoints, and plotting scripts for the report/demo.

Done when: the report can distinguish the effect of baseline restoration, adaptive allocation, multiplexing, and their combination.

## Suggested five-person split

| Person | Primary ownership | Work that can begin now | Later handoff |
|------------------|------------------|------------------|------------------|
| 1: Cloud and runs | Steps 1-4; storage, cost tracking, run scheduling, resume support | GCP account/quota setup, environment recipe, data preparation | Working VM, fixed data, upstream reference, cost pilot |
| 2: Baseline and integration | Step 5; shared training-core interface; step 10 with Person 3 | Suffix replay implementation and equivalence checks | Validated A; integrates B/C/D into the training core |
| 3: Diagnostics and evaluation | Step 6; step 10 with Person 2; result aggregation | Gradient comparison harness, metrics/accounting, configuration and plotting support | Shared acceptance checks, small-model validation, final plots |
| 4: Adaptive allocation | Step 7; step 9 with Person 5 | Split-estimate logic, allocation policies, minimum-budget and rounding checks | B and the allocation component of D |
| 5: Multiplexing | Step 8; step 9 with Person 4 | Hadamard/random codebooks, decoding, controlled examples | C and the multiplexing component of D |

Everyone can start development immediately. Integrated GPU comparisons have dependencies; starting a prototype does not mean it is ready for a training campaign.

## Parallel work waves and dependencies

| Wave | Work that can happen in parallel | Required gate before moving on |
|------------------------|------------------------|------------------------|
| 1: Bootstrap | Person 1 sets up GCP/data; Person 2 develops suffix replay; Person 3 builds diagnostics; Persons 4-5 prototype their separate methods | Agree on the shared estimator interface and accounting rules; obtain GPU access |
| 2: Establish A | Persons 1 and 3 collect the upstream reference/cost pilot; Persons 2 and 3 validate restored A; Persons 4-5 continue isolated method development | A matches full replay within agreed tolerances; diagnostic and compute counters are trusted |
| 3: Validate individual methods | Person 4 validates B; Person 5 validates C; Persons 2-3 add smaller-model support; Person 1 manages runs and cost projections | B and C pass their own checks; model configurations pass before size comparisons |
| 4: Integrate and select | Persons 2, 4, and 5 integrate D; Person 3 prepares result aggregation; Person 1 schedules selected individual-method runs | D passes the shared checks; final experiment matrix fits the remaining budget |
| 5: Final experiments | Independent A-D/seed/size jobs can run separately; plotting and writing can proceed as results arrive | All selected runs complete with comparable settings, saved artifacts, and variability reported |

The main dependency chain is:

`GPU/environment/data + restored A -> shared diagnostic acceptance and cost pilot -> checked B and C -> checked D -> frozen final experiment matrix -> final results`

Small-model work runs alongside B/C development, but must be validated before final size sweeps.

## Coordination rules that keep parallel work useful

1.  Agree early on group identifiers, requested draw counts, estimator outputs, gradient normalization, RNG ownership, logical reward-centering groups, and compute counters.
2.  Keep adaptive allocation and codebook/decoding logic in separate modules and branches. Let Person 2 coordinate edits to the shared training core instead of everyone independently changing `dust.py`.
3.  Preserve the untouched upstream reference and fixed dataset hashes. Save the exact commit and configuration for every result.
4.  Preserve logical reward-centering groups when changing scheduling. Physical draw batching currently affects centering, so changing it silently can change the estimator.
5.  Parallel people do not require simultaneous GPU jobs. With one VM, schedule GPU time; do not benchmark competing jobs together because contention invalidates timing comparisons.
6.  Launch additional VMs only when the cost pilot and remaining budget justify them. Keep hardware constant within comparisons.
7.  Before long or Spot runs, support resume with model weights, optimizer momentum, RNG/generator state, training position, allocator state, and multiplexing schedule/configuration. The existing best-weights file alone is insufficient.

## Proposed spending allocation

These are planning envelopes, not automatic cloud spending caps or measured runtime estimates.

| Work                                             |  Amount |
|--------------------------------------------------|--------:|
| Setup, upstream reference, and profiling         |   \$300 |
| Implementation checks and diagnostic experiments |   \$900 |
| Final seeded comparisons                         | \$1,200 |
| Reserve for failures and follow-up validation    |   \$600 |
| Total                                            | \$3,000 |

Recalculate the matrix after the cost pilot and after suffix restoration. Useful negative results are preferable to exhausting the budget on an oversized sweep.

## Sources

-   [Team proposal](/Users/alt2005/Downloads/Team%2027%20Proposal%202.0%20(2).pdf)
-   [Official repository and setup instructions](https://github.com/qlabs-eng/dust)
-   [DUST paper, especially section 2.3](https://qlabs.sh/research/dust)
-   [DUST appendix: architecture, allocation, and tuning](https://qlabs.sh/research/dust/appendix.html)
-   [GCP GPU quotas](https://docs.cloud.google.com/compute/resource-usage)
-   [Create an A100/A2 instance](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-accelerator-optimized)
-   [GCP billing budgets and alerts](https://docs.cloud.google.com/billing/docs/how-to/budgets)