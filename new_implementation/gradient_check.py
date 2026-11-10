"""Gradient-quality check: compare DUST's forward-only gradient estimate against the true backprop
gradient, on a frozen checkpoint and a fixed batch, without ever updating the weights.

This is the ruler the rest of the study is graded against. Nothing here runs an optimizer step. Scope
is deliberately narrow -- gradient fidelity for one step, not training-loss or held-out-loss curves
over a run (those are the experiment campaign). That focus is why this file is gradient_check.py and
not a catch-all "diagnostics".

- ``reference_gradients``: the true gradient of the mean cross-entropy, by one autograd pass.
- ``estimate_gradients``: DUST's estimate (via DUST.estimate), at the same mean-loss normalization.
- ``compare``: cosine, relative norm and norms, keyed by the shared module groups (groups.py).
- ``estimate_statistics``: mean cosine, cosine spread, estimator variance and empirical bias across
  repeated perturbation seeds -- the split-quality signal the adaptive allocator also wants.
- ``fit_cosine_law`` / ``cosine_curve``: fit cos(K)=c_max/sqrt(1+c/K) over population K. The fitted
  (c_max, c) per group should reproduce the paper's Appendix G -- the cheapest baseline-A fidelity gate.

The reference is computed in fp32 by default for a clean "true gradient"; pass ``precision='match'``
to compute it under the same bf16 autocast DUST sees on CUDA. Cosine is scale-invariant, so it is the
robust metric; relative norm, variance and bias are scale-sensitive and only mean what you intend when
both sides use this same mean-loss normalization (summed estimate divided by the valid-token count).

Example (on a tiny model, any device)::

    ref, _ = reference_gradients(model, x, y)
    est, _ = estimate_gradients(estimator, x, y, seed=0)
    report = compare(ref, est)          # report['per_group']['writers']['cosine'], ...
"""
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

# This module lives in new_implementation/ but reaches back into the original DUST code at the repo
# root for average_ranks. Put the repo root (this file's parent's parent) on the path so `import dust`
# resolves whether this file is run as a script or imported. groups.py sits beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dust import average_ranks  # noqa: E402  (import after the sys.path shim above)
from groups import param_group  # noqa: E402


def reference_gradients(model, x, y, precision='fp32'):
    """True gradient of the mean cross-entropy w.r.t. every parameter, as a name -> tensor dict.

    Temporarily enables autograd on the (otherwise frozen) model, runs one forward/backward, reads
    the grads, then restores the original requires_grad flags and clears the grads. Returns
    (gradients, loss).

    Precision matters for the comparison and bf16 != fp32 (Lecture 3): DUST's clean pass runs under
    bf16 autocast, so cosine against an fp32 reference mixes DUST's algorithm error with the
    fp32-vs-bf16 numerical floor. Report both: ``precision='fp32'`` (default) is the mathematical
    "true gradient" and answers "how close is DUST to the real gradient"; ``precision='match'`` uses
    bf16 autocast on CUDA, matching DUST's own precision, and isolates DUST's algorithm error from the
    dtype gap. On CPU the two are identical (no bf16 autocast), so this only bites on the GPU runs.
    """
    device = next(model.parameters()).device
    was = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    autocast = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if precision == 'match' and device.type == 'cuda' else nullcontext()
    )
    try:
        with torch.enable_grad(), autocast:
            loss = model(x, y)  # default reduction='mean' -> grads are already mean-loss gradients
        loss.backward()
        gradients = {
            name: p.grad.detach().float().clone()
            for name, p in model.named_parameters() if p.grad is not None
        }
    finally:
        for name, p in model.named_parameters():
            p.grad = None
            p.requires_grad_(was[name])
    return gradients, loss.item()


def reseed(estimator, seed):
    """Point the estimator's perturbation RNG at a new seed, so repeated estimates are independent."""
    estimator.generator.manual_seed(seed)


@torch.no_grad()
def estimate_gradients(estimator, x, y, seed=None, sync=False):
    """DUST's forward-only estimate at the mean-loss normalization, as a name -> tensor dict.

    ``seed`` reseeds the perturbation RNG first (use a different seed per sample for statistics).
    ``sync`` all-reduces across ranks (leave False for single-process diagnostics). Returns
    (gradients, clean_loss). Does not touch the optimizer or the weights.
    """
    if seed is not None:
        reseed(estimator, seed)
    gradients, valid_tokens, loss = estimator.estimate(x, y)
    out = {}
    for name, grad in gradients.items():
        g = (grad / valid_tokens).float()
        out[name] = average_ranks(g) if sync else g.clone()
    return out, loss


def group_vectors(gradients):
    """Flatten and concatenate each group's parameters into one vector, params sorted by name.

    Sorting makes the reference and the estimate concatenate in the same order, so their cosine is
    meaningful; the two dicts are built in different orders otherwise.
    """
    buckets = defaultdict(list)
    for name in sorted(gradients):
        buckets[param_group(name)].append(gradients[name].reshape(-1))
    return {group: torch.cat(parts) for group, parts in buckets.items()}


def _cosine(a, b):
    # float64: fp32 accumulation over a multi-million-entry group (the embedding) drifts by ~1e-4,
    # enough to report a cosine above 1 and to blur the high-cosine end of the cosine-law fit.
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0, eps=1e-20).item()


def compare(reference, estimate, per_parameter=False):
    """Per-group cosine / relative norm between a reference and an estimate, plus a group-mean summary.

    ``rel_norm`` is ||estimate|| / ||reference|| per group. ``overall.mean_group_cosine`` is the
    unweighted mean of the per-group cosines -- a single scalar that, unlike concatenating all
    parameters, is not swamped by the embedding's much larger gradient norm.
    """
    ref_groups, est_groups = group_vectors(reference), group_vectors(estimate)
    per_group = {}
    for group in sorted(ref_groups):
        r, e = ref_groups[group], est_groups[group]
        per_group[group] = dict(
            cosine=_cosine(r, e),
            rel_norm=(e.norm() / r.norm().clamp(min=1e-20)).item(),
            ref_norm=r.norm().item(),
            est_norm=e.norm().item(),
        )
    cosines = [m['cosine'] for m in per_group.values()]
    overall = dict(
        mean_group_cosine=sum(cosines) / len(cosines),
        min_group_cosine=min(cosines),
    )
    report = dict(overall=overall, per_group=per_group)
    if per_parameter:
        report['per_parameter'] = {
            name: _cosine(reference[name].reshape(-1), estimate[name].reshape(-1))
            for name in sorted(reference) if name in estimate
        }
    return report


def estimate_statistics(estimator, x, y, seeds, reference=None, sync=False):
    """Variance, bias and cosine spread of the estimate over repeated perturbation seeds, by group.

    Runs DUST.estimate once per seed (no weight updates) and reports, per group: ``est_variance``
    (mean per-coordinate variance of the estimate), and -- when a ``reference`` is given --
    ``bias`` = ||E[estimate] - reference||, plus ``cosine_mean`` / ``cosine_std`` of the per-seed
    cosine to the reference. The cosine spread is exactly the split-quality signal adaptive
    allocation keys on: a noisy group has low, high-variance cosine and should get more draws.
    """
    if len(seeds) < 2:
        raise ValueError('estimate_statistics needs at least 2 seeds; the variance of one sample is undefined')
    samples =[estimate_gradients(estimator, x, y, seed=s, sync=sync)[0] for s in seeds]
    names = list(samples[0])
    mean = {n: torch.stack([s[n] for s in samples]).mean(0) for n in names}
    variance = {n: torch.stack([s[n] for s in samples]).var(0, unbiased=True) for n in names}

    mean_groups, var_groups = group_vectors(mean), group_vectors(variance)
    stats = {group: dict(est_variance=var_groups[group].mean().item()) for group in mean_groups}

    if reference is not None:
        ref_groups = group_vectors(reference)
        per_seed = defaultdict(list)
        for sample in samples:
            sample_groups = group_vectors(sample)
            for group in ref_groups:
                per_seed[group].append(_cosine(sample_groups[group], ref_groups[group]))
        for group in stats:
            cosines = torch.tensor(per_seed[group])
            stats[group].update(
                bias=(mean_groups[group] - ref_groups[group]).norm().item(),
                mean_cosine_to_ref=_cosine(mean_groups[group], ref_groups[group]),
                cosine_mean=cosines.mean().item(),
                cosine_std=cosines.std(unbiased=True).item() if len(cosines) > 1 else 0.0,
            )
    return stats


def fit_cosine_law(populations, cosines):
    """Fit cos(K) = c_max / sqrt(1 + c/K) and return {c_max, c, rmse}.

    Linearized exactly: 1/cos^2 = 1/c_max^2 + (c/c_max^2) * (1/K), a straight line in 1/K, so this is
    an ordinary least-squares fit -- no solver or scipy needed. Compare (c_max, c) against Appendix G
    (e.g. LM head c_max=1.0, c=759; MLP out c_max=0.88, c=31623) to check baseline-A fidelity.
    """
    K = torch.as_tensor(populations, dtype=torch.float64)
    cos = torch.as_tensor(cosines, dtype=torch.float64)
    # The linearization 1/cos^2 assumes cos > 0; a non-positive point (possible at tiny population)
    # would blow up the fit silently, so drop such points and flag it instead.
    positive = cos > 0
    dropped = int((~positive).sum())
    K, cos = K[positive], cos[positive]
    if len(cos) < 2:
        return dict(c_max=float('nan'), c=float('nan'), rmse=float('nan'),
                    note=f'need >=2 positive-cosine points, had {len(cos)}')
    design = torch.stack([torch.ones_like(K), 1.0 / K], dim=1)
    coefficients = torch.linalg.lstsq(design, (1.0 / cos.square()).unsqueeze(1)).solution.squeeze(1)
    a, b = coefficients[0].item(), coefficients[1].item()
    if a <= 0:  # 1/c_max^2 must be positive; a ** -0.5 would silently return a complex number
        return dict(c_max=float('nan'), c=float('nan'), rmse=float('nan'),
                    note=f'fit has no positive ceiling (1/c_max^2 = {a:.3g}); the points do not follow the law')
    c_max, c = a ** -0.5, b / a
    predicted = c_max / torch.sqrt(1.0 + c / K)
    result = dict(c_max=c_max, c=c, rmse=torch.sqrt((predicted - cos).square().mean()).item())
    if dropped:
        result['note'] = f'dropped {dropped} non-positive cosine point(s) before fitting'
    return result


def cosine_curve(make_estimator, populations, x, y, reference, seeds=(0,)):
    """Measure the reference cosine at each population and fit the cosine law per group.

    ``make_estimator(population)`` builds a DUST estimator at that population (it carries the model,
    so pass the frozen checkpoint in). Returns {group: {'cosines': [...per population...],
    'fit': {c_max, c, rmse}}}. Averaging over several ``seeds`` steadies the small-population points.
    """
    by_group = defaultdict(list)
    for population in populations:
        estimator = make_estimator(population)
        try:
            reports = [compare(reference, estimate_gradients(estimator, x, y, seed=s)[0]) for s in seeds]
        finally:
            estimator.close()  # remove this estimator's forward hooks before the next one re-registers
        for group in reports[0]['per_group']:
            mean_cosine = sum(r['per_group'][group]['cosine'] for r in reports) / len(reports)
            by_group[group].append(mean_cosine)
    return {
        group: dict(cosines=cosines, fit=fit_cosine_law(populations, cosines))
        for group, cosines in by_group.items()
    }
