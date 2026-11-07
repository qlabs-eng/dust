"""Fair compute accounting for DUST updates.

Two things, both pure arithmetic (no GPU, no run):

- ``analytic_cost(recipe, config, ...)``: the expected FLOPs and forward-equivalents of one update,
  derived from the Recipe. This is what makes A/B/C/D comparable on paper, before any GPU time, and
  what quantifies Track 1's cached-suffix saving exactly (call it with ``suffix=True`` vs ``False``).
- ``RunMeters`` / ``measure``: passive wall-time and peak-memory meters a training loop fills in on a
  real run. Defined here so the interface is fixed now; they need a GPU only when actually used.

Why an analytic model and not just a draw count: the nominal population K counts only the direct-loss
draws (writers + hidden + hubs + embedding). Head, the five attention internals, and the residual
scalars are *extra*, and a direct draw's cost depends on which block it perturbs -- a layer-0 draw
reruns the whole stack, a layer-7 draw almost nothing. Counting draws uniformly misreports every
A-vs-upstream and B-vs-A comparison. See groups.total_draws for the K-vs-total point.

Cost currency: ``forward_equivalents`` = total update FLOPs / one clean forward's FLOPs. Hardware
agnostic, and it matches how the paper talks about cost ("a draw is slightly cheaper than a forward
because the clean forward is cached"). A key subtlety the model captures: every direct draw recomputes
the loss, i.e. a full head (lm_head over the whole vocab) + cross-entropy, which cached-suffix
execution does NOT remove. Suffix caching skips *blocks*; the head floor stays. So with the paper's
front-loaded allocation the suffix saving is real but bounded -- the counter tells you by how much.

FLOP conventions: one multiply-add = 2 FLOPs; matmuls dominate, so norm / rotary / softmax /
elementwise are ignored as lower order. Numbers are estimates for comparison, not profiler truth.
"""
from dataclasses import dataclass, field, asdict

from groups import group_draw_count, suffix_start_block


def window_sizes(config):
    """Left-context window per block, mirroring GPT._compute_window_sizes in model.py.

    Kept in sync with model.py by hand; pattern repeats, and the last block is always full.
    """
    pattern = config.window_pattern.upper()
    long_w, short_w = config.sequence_len, config.sequence_len // 2
    sizes = [long_w if pattern[i % len(pattern)] == 'L' else short_w for i in range(config.n_layer)]
    sizes[-1] = long_w
    return sizes


def _padded_vocab(config, pad_to=64):
    return (config.vocab_size + pad_to - 1) // pad_to * pad_to


def attention_flops(tokens, window, config):
    """The two attention matmuls (scores and value mix) for one block, over ``tokens`` query tokens.

    Each query attends min(window + 1, sequence_len) keys; 2 matmuls, 2 FLOPs per multiply-add.
    """
    head_dim = config.n_embd // config.n_head
    keys = min(window + 1, config.sequence_len)
    return 4 * tokens * keys * head_dim * config.n_head


def block_flops(tokens, window, config):
    """One transformer block's forward FLOPs over ``tokens`` tokens: q/k/v/out projections, MLP, attention."""
    n = config.n_embd
    kv = config.n_kv_head * (config.n_embd // config.n_head)
    projections = 2 * tokens * n * (2 * n + 2 * kv)          # c_q, c_proj (n->n) and c_k, c_v (n->kv)
    mlp = 2 * tokens * (n * 4 * n + 4 * n * n)               # c_fc and squared-ReLU c_proj
    return projections + mlp + attention_flops(tokens, window, config)


def head_flops(tokens, config):
    """lm_head projection FLOPs over ``tokens`` tokens (cross-entropy itself is lower order)."""
    return 2 * tokens * config.n_embd * _padded_vocab(config)


@dataclass
class CostRecord:
    suffix: bool
    world: int
    batch: int
    draws: dict                       # group -> total draws across all GPUs
    total_draws: int
    population_K: int                 # direct-loss draws only (writers + hidden + hubs + embedding)
    flops_total: float                # whole update, summed over all GPUs
    flops_per_group: dict
    flops_clean_forward: float
    forward_equivalents: float        # flops_total / flops_clean_forward
    forward_equivalents_per_gpu: float
    note: str = (
        'forward_equivalents = update FLOPs / one clean forward. population_K excludes head, '
        'attention-internal and scalar draws. Every direct draw pays a full head recompute, which '
        'suffix caching does not remove. FLOPs predict wall-clock only when the step is GPU-bound; '
        'DUST issues many tiny forwards through Python hooks, so confirm with profiling.launch_bound_test '
        'before reading forward_equivalents as time.'
    )

    def as_dict(self):
        return asdict(self)


def analytic_cost(recipe, config, world=1, batch=8, suffix=True):
    """Expected FLOPs and forward-equivalents of one DUST.estimate update, from the Recipe alone.

    ``suffix``: True models cached-suffix execution (a direct draw at block L reruns blocks L..end);
    False models the public code's full replay (every direct draw reruns the whole stack). The ratio
    of the two is Track 1's saving for this exact allocation.

    Costs are population-wide (summed over GPUs). forward_equivalents_per_gpu divides the parallel
    draw work by ``world`` (the clean forward runs once per GPU, so it is not divided).
    """
    tokens = batch * config.sequence_len
    windows = window_sizes(config)
    blocks = [block_flops(tokens, w, config) for w in windows]
    suffix_flops = [sum(blocks[start:]) for start in range(config.n_layer + 1)]  # blocks start..end
    head_full = head_flops(tokens, config)
    clean_forward = sum(blocks) + head_full

    draws, flops_per_group = {}, {}

    def direct_group(group):
        total_draws, total_flops = 0, 0.0
        for layer in range(config.n_layer):
            count = group_draw_count(recipe, group, layer)
            start = suffix_start_block(f'o@{layer}', config.n_layer) if suffix else 0
            total_draws += count
            total_flops += count * (suffix_flops[start] + head_full)  # blocks reran + full head/CE
        return total_draws, total_flops

    for group in ('writers', 'hidden', 'hubs'):
        draws[group], flops_per_group[group] = direct_group(group)

    embedding_draws = group_draw_count(recipe, 'embedding')          # perturbs the input -> whole stack
    draws['embedding'] = embedding_draws
    flops_per_group['embedding'] = embedding_draws * (suffix_flops[0] + head_full)

    # Head slab draws recompute cross-entropy from cached logits over a 256-column slab: cheap, no block
    # or full-head matmul. Estimated as the slab's share of the head projection.
    head_draws = group_draw_count(recipe, 'head')
    slab_fraction = 256 / _padded_vocab(config)
    draws['head'] = head_draws
    flops_per_group['head'] = head_draws * head_full * slab_fraction

    # Attention-internal draws recompute one block's attention from cached q/k/v: no projections,
    # no other block, no head.
    for group in ('q', 'v', 'gate', 'k', 've'):
        total_draws, total_flops = 0, 0.0
        for layer in range(config.n_layer):
            if group in ('gate', 've') and not _has_ve(layer, config.n_layer):
                continue
            count = group_draw_count(recipe, group, layer)
            total_draws += count
            total_flops += count * attention_flops(tokens, windows[layer], config)
        draws[group], flops_per_group[group] = total_draws, total_flops

    # Residual scalars: 8 two-sided draws = 16 full forwards (scalar_gradients reruns the whole model).
    scalar_forwards = group_draw_count(recipe, 'scalars') * 2
    draws['scalars'] = group_draw_count(recipe, 'scalars')
    flops_per_group['scalars'] = scalar_forwards * clean_forward

    parallel_flops = sum(flops_per_group.values())
    total_flops = parallel_flops + clean_forward                     # one clean capture pass per update
    population_k = draws['writers'] + draws['hidden'] + draws['hubs'] + draws['embedding']

    return CostRecord(
        suffix=suffix, world=world, batch=batch,
        draws=draws, total_draws=sum(draws.values()), population_K=population_k,
        flops_total=total_flops, flops_per_group=flops_per_group, flops_clean_forward=clean_forward,
        forward_equivalents=total_flops / clean_forward,
        forward_equivalents_per_gpu=clean_forward / clean_forward + parallel_flops / world / clean_forward,
    )


def suffix_saving(recipe, config, **kwargs):
    """Fraction of update FLOPs that cached-suffix execution saves for this allocation (Track 1's payoff)."""
    full = analytic_cost(recipe, config, suffix=False, **kwargs).flops_total
    cached = analytic_cost(recipe, config, suffix=True, **kwargs).flops_total
    return (full - cached) / full


def _param_bytes(config):
    """Approximate parameter bytes: token and value embeddings in bf16 (2 B), the rest in fp32 (4 B)."""
    pv, n = _padded_vocab(config), config.n_embd
    kv = config.n_kv_head * (n // config.n_head)
    per_block = n * n + 2 * n * kv + n * n + n * 4 * n + 4 * n * n  # q, k, v, o, mlp.c_fc, mlp.c_proj
    ve_layers = sum(1 for i in range(config.n_layer) if _has_ve(i, config.n_layer))
    fp32 = config.n_layer * per_block + ve_layers * 32 * config.n_kv_head + pv * n + 2 * config.n_layer
    bf16 = pv * n + ve_layers * pv * kv  # wte and the value embeddings
    return fp32 * 4 + bf16 * 2


def analytic_memory(recipe, config, batch=8, activation_bytes=2):
    """Rough peak-memory estimate for one update, as a component breakdown and a GB total.

    DUST runs under no_grad, so there is no saved-activation cost for a backward pass (unlike the
    backprop baseline, Lecture 3 slide 26). The transient is dominated by the nrep-replicated batch
    (``direct_errors`` does x.repeat(nrep, 1)) flowing through the MLP hidden width; under no_grad only
    one block's activations are live at a time. Persistent memory is parameters + the estimated
    gradient + the SGD momentum buffer (each roughly parameter-sized). Use it to pick population / nrep
    without an out-of-memory error; it is an estimate, not a profiler reading (use profile_memory=True
    in profiling.profile_estimate for that).
    """
    tokens = recipe.nrep * batch * config.sequence_len
    hidden = tokens * 4 * config.n_embd * activation_bytes     # MLP hidden, the dominant live tensor
    residual = tokens * config.n_embd * activation_bytes
    accumulator = 2 * batch * config.sequence_len * config.n_embd * 4  # fp32 error accumulators
    params = _param_bytes(config)
    peak = hidden + residual + accumulator + params * 3  # params + grad + momentum
    return dict(
        mlp_hidden_bytes=hidden, residual_bytes=residual, accumulator_bytes=accumulator,
        params_bytes=params, grad_bytes=params, optimizer_bytes=params,
        peak_estimate_gb=peak / 1024 ** 3,
        note='rough; no autograd activations (no_grad). nrep*batch*seqlen through 4*n_embd dominates '
             'the transient, so population/nrep, not the model, sets peak memory.',
    )


def _has_ve(layer_idx, n_layer):
    """Value embeddings on every other block, counting back from the last (mirrors model.has_ve)."""
    return layer_idx % 2 == (n_layer - 1) % 2


# --- Measured side: filled in on a real run (needs a GPU only when actually used) ---
#
# Follows the Lecture 3 benchmarking rules: time the GPU with CUDA events (not host wall +
# synchronize); discard warmup steps (the first steps pay allocator and cuDNN autotune); report a
# distribution (median, p10, p90), not a mean, which outliers like an eval step or a GC pause skew.
# Analytic FLOPs tell you the work; these tell you the time. Pair them, and if the step time barely
# moves when you double the batch, the step is launch-bound and the FLOP model will not predict it.

def _percentile(values, q):
    """Nearest-rank percentile of a list (q in [0, 100]); 0.0 for an empty list."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


@dataclass
class RunMeters:
    """Per-step GPU and host time plus peak memory from a real run, to pair with the analytic FLOPs.

    ``warmup`` leading steps are dropped from the summary distribution (but kept for total wall time).
    GPU time per step comes from CUDA events; host time is the wall clock around the step, which is
    larger when the step is launch-bound (the CPU issue time, not the GPU kernels, sets it).
    """
    warmup: int = 3
    step_gpu_ms: list = field(default_factory=list)
    step_host_ms: list = field(default_factory=list)
    eval_seconds: list = field(default_factory=list)
    peak_memory_bytes: int = 0

    def record_step(self, host_ms, gpu_ms=None):
        self.step_host_ms.append(host_ms)
        if gpu_ms is not None:
            self.step_gpu_ms.append(gpu_ms)

    def record_eval(self, seconds):
        self.eval_seconds.append(seconds)

    def _distribution(self, samples, name):
        kept = samples[self.warmup:] or samples
        return {
            f'{name}_median_ms': _percentile(kept, 50),
            f'{name}_p10_ms': _percentile(kept, 10),
            f'{name}_p90_ms': _percentile(kept, 90),
        }

    def summary(self, world=1):
        wall = sum(self.step_host_ms) / 1000 + sum(self.eval_seconds)
        launch_bound = None
        if self.step_gpu_ms:
            gpu = _percentile(self.step_gpu_ms[self.warmup:] or self.step_gpu_ms, 50)
            host = _percentile(self.step_host_ms[self.warmup:] or self.step_host_ms, 50)
            launch_bound = host > 1.3 * gpu  # host well above GPU time => CPU issue sets the step
        return {
            'steps_kept': max(0, len(self.step_host_ms) - self.warmup) or len(self.step_host_ms),
            'warmup_discarded': min(self.warmup, len(self.step_host_ms)),
            **self._distribution(self.step_host_ms, 'host'),
            **(self._distribution(self.step_gpu_ms, 'gpu') if self.step_gpu_ms else {}),
            'likely_launch_bound': launch_bound,
            'wall_seconds': wall,
            'gpu_hours': wall * world / 3600,
            'peak_memory_gb': self.peak_memory_bytes / 1024 ** 3,
            'scope': 'step only; excludes data load and host->device copy unless inside the measured block',
        }


class measure:
    """Time a block with CUDA events (GPU) and the wall clock (host), into a RunMeters.

    Usage::

        with measure(meters, 'step'):
            estimator.step(x, y, optimizer)

    The GPU time comes from events, so it is the kernel time regardless of how far the CPU ran ahead;
    the host time is the wall clock, which reveals a launch-bound step when it sits well above it.
    """

    def __init__(self, meters, kind='step'):
        self.meters, self.kind = meters, kind
        self._cuda = False

    def __enter__(self):
        import time
        try:
            import torch
            self._cuda = torch.cuda.is_available()
        except ImportError:
            self._cuda = False
        if self._cuda:
            import torch
            torch.cuda.synchronize()  # drain prior work so it is not charged to this block
            torch.cuda.reset_peak_memory_stats()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc):
        import time
        host_ms = (time.monotonic() - self._start) * 1000
        gpu_ms = None
        if self._cuda:
            import torch
            self._end_event.record()
            torch.cuda.synchronize()
            gpu_ms = self._start_event.elapsed_time(self._end_event)
            self.meters.peak_memory_bytes = max(
                self.meters.peak_memory_bytes, torch.cuda.max_memory_allocated()
            )
        if self.kind == 'eval':
            self.meters.record_eval(host_ms / 1000)
        else:
            self.meters.record_step(host_ms, gpu_ms)
        return False
