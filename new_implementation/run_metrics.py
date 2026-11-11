"""Training-time metrics: the per-step record dust.py and baselines/backprop.py write to metrics.jsonl.

This wires the measurement stack into a training run; it computes nothing new. Each step's record says
what the update cost and how long it took, cumulatively, so held-out loss can be plotted against compute:

- ``flops`` / ``forward_equivalents``: cumulative analytic cost from accounting.analytic_cost (DUST) or
  3x a forward (backprop), summed over GPUs. Recomputed from the live Recipe every step, so a run whose
  allocation changes (adaptive allocation) is charged what it actually ran. Attention FLOPs follow
  accounting's current convention (see TODO.md, "FLOP convention").
- ``draws`` (cumulative perturbations, every group) and ``population`` (the per-update K).
- ``train_seconds`` / ``gpu_hours``: training-step time only. Evaluation, gradient probes and checkpoint
  writes are timed separately and excluded, so loss-vs-wall-clock does not charge a method for how often
  it was evaluated. ``elapsed_seconds`` is the honest total wall clock, everything included.
- ``step_host_ms`` (and ``step_gpu_ms`` + ``peak_memory_gb`` on CUDA) from accounting.measure.
- ``allocation``: the per-group draw counts, written on the first step and whenever they change.

``gradient_probe`` scores the estimate against backprop on a fixed held-out batch (gradient_check) without
disturbing training: it saves and restores the perturbation generator, so a probed run trains exactly
like an unprobed one.
"""
import json
import time
from contextlib import contextmanager

import torch

from accounting import RunMeters, analytic_cost, block_flops, head_flops, measure, window_sizes

ALLOCATION_FIELDS = ('writers', 'hidden', 'hubs', 'embedding', 'head', 'local', 'nrep')


def allocation(settings):
    """The draw counts of a Recipe (what adaptive allocation changes), as a JSON-ready dict."""
    return {name: getattr(settings, name) for name in ALLOCATION_FIELDS}


def dust_update_cost(settings, config, world=1, batch=8, suffix=True):
    """Cost of one DUST update from the Recipe, population-wide (all GPUs)."""
    cost = analytic_cost(settings, config, world=world, batch=batch, suffix=suffix)
    return dict(flops=cost.flops_total, forward_equivalents=cost.forward_equivalents,
                draws=cost.total_draws, population=cost.population_K)


def forward_flops(config, batch=8):
    """One clean forward over ``batch`` sequences, in accounting's conventions."""
    tokens = batch * config.sequence_len
    return sum(block_flops(tokens, w, config) for w in window_sizes(config)) + head_flops(tokens, config)


def backprop_update_cost(config, batch=8):
    """One backprop update: forward + backward, the backward counted as 2 forwards (the standard 3x)."""
    return dict(flops=3 * forward_flops(config, batch), forward_equivalents=3.0, draws=0, population=0)


class TrainingLog:
    """Cumulative cost and time of a training run, one record per step; resumable via state_dict.

    ``update_cost(settings)`` returns one update's cost dict (dust_update_cost / backprop_update_cost with
    the run's config bound); it is cached per allocation, so it is not recomputed every step.
    """

    def __init__(self, update_cost, world=1, warmup=3):
        self.update_cost, self.world = update_cost, world
        self.meters = RunMeters(warmup=warmup)
        self.totals = dict(flops=0.0, forward_equivalents=0.0, draws=0, train_seconds=0.0,
                           eval_seconds=0.0, probe_seconds=0.0, checkpoint_seconds=0.0)
        self.elapsed_before = 0.0          # wall clock of earlier processes of a resumed run
        self.started = time.monotonic()
        self.last_allocation = None
        self._costs = {}

    def _cost(self, settings):
        key = json.dumps(allocation(settings), sort_keys=True) if settings is not None else None
        if key not in self._costs:
            self._costs[key] = self.update_cost(settings)
        return self._costs[key]

    @contextmanager
    def step(self):
        """Time one training update (CUDA events + host clock, warmup-aware) into the meters."""
        with measure(self.meters, 'step'):
            yield

    @contextmanager
    def timed(self, kind):
        """Time non-training work ('eval', 'probe', 'checkpoint') so it stays out of train_seconds."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.monotonic()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.totals[f'{kind}_seconds'] += time.monotonic() - start

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started

    def record(self, step, train_loss, settings=None):
        """The step's metrics record, after its update was timed with ``step()``."""
        cost = self._cost(settings)
        host_ms = self.meters.step_host_ms[-1]
        self.totals['flops'] += cost['flops']
        self.totals['forward_equivalents'] += cost['forward_equivalents']
        self.totals['draws'] += cost['draws']
        self.totals['train_seconds'] += host_ms / 1000
        record = dict(
            step=step, train_loss=train_loss,
            flops=self.totals['flops'], forward_equivalents=self.totals['forward_equivalents'],
            draws=self.totals['draws'], population=cost['population'],
            train_seconds=self.totals['train_seconds'],
            gpu_hours=self.totals['train_seconds'] * self.world / 3600,
            elapsed_seconds=self.elapsed(), step_host_ms=host_ms,
        )
        if self.meters.step_gpu_ms:
            record['step_gpu_ms'] = self.meters.step_gpu_ms[-1]
            record['peak_memory_gb'] = self.meters.peak_memory_bytes / 1024 ** 3
        if settings is not None:
            current = allocation(settings)
            if current != self.last_allocation:
                record['allocation'] = current
                self.last_allocation = current
        return record

    def summary(self):
        return dict(**self.meters.summary(world=self.world), totals=dict(self.totals),
                    elapsed_seconds=self.elapsed())

    def state_dict(self):
        meters = self.meters
        return dict(
            totals=dict(self.totals), elapsed=self.elapsed(), last_allocation=self.last_allocation,
            meters=dict(warmup=meters.warmup, step_gpu_ms=list(meters.step_gpu_ms),
                        step_host_ms=list(meters.step_host_ms), eval_seconds=list(meters.eval_seconds),
                        peak_memory_bytes=meters.peak_memory_bytes),
        )

    def load_state_dict(self, state):
        self.totals = dict(state['totals'])
        self.elapsed_before, self.started = state['elapsed'], time.monotonic()
        self.last_allocation = state['last_allocation']
        self.meters = RunMeters(**state['meters'])


def gradient_probe(estimator, x, y, seeds=(0,), sync=False):
    """Cosine and relative norm of DUST's estimate vs the backprop gradient, per group, on (x, y).

    With two or more ``seeds`` it also reports the estimator's variance and bias over them
    (gradient_check.estimate_statistics). The perturbation generator is restored afterwards, so the
    training run continues exactly as if the probe had not run. Costs one backprop pass plus one
    DUST estimate per seed.
    """
    import gradient_check as gc  # lazy: gradient_check imports dust, which imports this module

    state = estimator.generator.get_state()
    try:
        reference, _ = gc.reference_gradients(estimator.model, x, y)
        if len(seeds) == 1:
            estimate, _ = gc.estimate_gradients(estimator, x, y, seed=seeds[0], sync=sync)
            report = gc.compare(reference, estimate)
            return dict(**report['overall'], per_group={
                g: dict(cosine=m['cosine'], rel_norm=m['rel_norm']) for g, m in report['per_group'].items()})
        stats = gc.estimate_statistics(estimator, x, y, seeds, reference=reference, sync=sync)
        cosines = [m['cosine_mean'] for m in stats.values()]
        return dict(mean_group_cosine=sum(cosines) / len(cosines), min_group_cosine=min(cosines),
                    seeds=len(seeds), per_group=stats)
    finally:
        estimator.generator.set_state(state)
