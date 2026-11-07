"""Roofline model of one DUST update: FLOPs *and* bytes per op, placed on a GPU's roofs.

accounting.analytic_cost counts FLOPs only, and a FLOP count predicts time only for compute-bound
work. This adds the co-equal axis (Lecture 2): bytes moved to and from HBM. For every op one update
runs -- the clean pass, each draw's forward, the noise and reward arithmetic around it, the head slabs,
the attention-internal recomputes, the residual-scalar forwards and the SGD step -- it estimates
FLOPs, bytes, kernel launches and host syncs, then gives three time floors on a GPU:

- compute: flops / peak FLOP/s at the op's precision (bf16 tensor cores under autocast, TF32 for the
  fp32 matmuls, CUDA cores for elementwise);
- memory:  bytes / HBM bandwidth;
- launch:  kernels * per-launch latency, the "launch roof" a roofline alone lacks (Lecture 3).

An op's roofline time is max(compute, memory). A phase's predicted time is max(the sum of its ops'
roofline times, its launch time), after T_step ~ max(sum CPU issue, sum GPU work) (L3 s37). Whichever
floor wins names the regime. Phases are DUST.estimate's record_function labels plus 'apply' (the SGD
step and everything outside the labels), so a prediction lines up row by row with
profiling.profile_estimate's table.

Byte conventions: every op reads its inputs from HBM and writes its outputs to HBM once (no L2 reuse
across kernels, no fusion); activations are bf16 under autocast, fp32 where dust.py upcasts. Weights
are fp32 and re-cast to bf16 on every call, because autocast caches casts only for tensors that
require grad and the model runs with requires_grad off. Index and scalar kernels are folded in or
dropped. Kept in sync with model.py and dust.py by hand, like accounting.window_sizes.

Attention is the one op whose cost depends on a choice PyTorch makes at run time. model.attention
passes an explicit boolean mask, which rules out the flash kernel. 'efficient' (the default) models
the memory-efficient kernel: it streams q, k and v once but scores all T keys, masked or not, so its
FLOPs ignore the sliding window. 'math' models the fallback that materialises the [batch, head, T, T]
scores, softmax and probabilities in HBM. The GPU profile decides which one runs.

Kernel counts are rough (one per aten op that launches work) and the 5 us launch latency is a
placeholder until a trace calibrates it. These are predictions for comparison, not profiler truth: a
phase whose measured time is far from its prediction means the model is missing something.
"""
from dataclasses import dataclass, field, asdict, replace

from accounting import window_sizes, _padded_vocab, _has_ve

ACT = 2   # bytes per bf16 activation under autocast
F32 = 4
SLAB = 256  # head columns jittered per draw (DUST.head_error)
GATE_CHANNELS = 32  # block-input channels the value-embedding gate reads (CausalSelfAttention)

PHASES = ('clean', 'direct', 'embedding', 'hubs', 'head', 'local', 'scalars', 'apply')
ATTENTION_KERNELS = ('efficient', 'math')


@dataclass(frozen=True)
class Hardware:
    name: str
    bf16_flops: float        # dense tensor-core peak
    tf32_flops: float        # dense tensor-core peak for fp32 matmuls (set_float32_matmul_precision('high'))
    fp32_flops: float        # CUDA-core peak, used for elementwise ops
    hbm_bytes_per_s: float
    launch_s: float = 5e-6   # per-kernel issue latency; a placeholder until a trace calibrates it

    def peak(self, precision):
        return {'bf16': self.bf16_flops, 'tf32': self.tf32_flops, 'fp32': self.fp32_flops}[precision]

    def ridge(self, precision='bf16'):
        """Arithmetic intensity (FLOP/byte) above which an op at this precision is compute-bound."""
        return self.peak(precision) / self.hbm_bytes_per_s


# Datasheet peaks, dense (no structured sparsity). a2-highgpu-1g is the SXM4 40 GB part.
A100_40GB = Hardware('A100-SXM4-40GB', 312e12, 156e12, 19.5e12, 1.555e12)
H100_SXM = Hardware('H100-SXM5-80GB', 989e12, 494e12, 67e12, 3.35e12)  # the lecture slides' numbers
HARDWARE = {'a100': A100_40GB, 'h100': H100_SXM}


@dataclass
class Op:
    """One kind of work, possibly repeated: its totals are summed over every time it runs."""
    name: str
    kind: str                # 'matmul', 'attention' or 'elementwise'
    flops: float
    nbytes: float
    kernels: int = 1
    precision: str = 'bf16'
    syncs: int = 0           # host waits on the GPU (.item(), boolean indexing, a branch on a tensor)
    weight_bytes: float = 0.0  # the part of nbytes that is parameter traffic, not activations

    def times(self, count):
        return replace(self, flops=self.flops * count, nbytes=self.nbytes * count,
                       kernels=self.kernels * count, syncs=self.syncs * count,
                       weight_bytes=self.weight_bytes * count)

    def plus(self, other):
        return replace(self, flops=self.flops + other.flops, nbytes=self.nbytes + other.nbytes,
                       kernels=self.kernels + other.kernels, syncs=self.syncs + other.syncs,
                       weight_bytes=self.weight_bytes + other.weight_bytes)


class Ledger:
    """Ops of one phase, merged by name so the summary reads 'mlp.c_fc matmul', not one row per call."""

    def __init__(self):
        self.ops = {}

    def add(self, ops, count=1):
        if count <= 0:
            return
        for op in ops if isinstance(ops, list) else [ops]:
            op = op.times(count)
            self.ops[op.name] = self.ops[op.name].plus(op) if op.name in self.ops else op


def _ew(name, elements, bytes_per_element, kernels=1, syncs=0):
    """Elementwise or reduction work: about one FLOP per element, so bandwidth sets its time."""
    return Op(name, 'elementwise', elements, elements * bytes_per_element, kernels, 'fp32', syncs)


def _matmul(name, flops, nbytes, precision='bf16', weight_bytes=0.0):
    return Op(name, 'matmul', flops, nbytes, 1, precision, weight_bytes=weight_bytes)


def _linear(name, rows, fan_in, fan_out):
    """A bias-free nn.Linear under bf16 autocast: re-cast the fp32 weight to bf16, then one GEMM."""
    weight = fan_in * fan_out
    cast = _ew(f'{name} weight cast', weight, F32 + ACT)
    cast.weight_bytes = cast.nbytes
    return [cast, _matmul(f'{name} matmul', 2 * rows * fan_in * fan_out,
                          (rows * fan_in + weight + rows * fan_out) * ACT, weight_bytes=weight * ACT)]


def _attention(config, rows, window, kernel, dtype_bytes=ACT, tag=''):
    """model.attention over `rows` query tokens: the mask it builds, then the SDPA kernel."""
    T, h = config.sequence_len, config.n_head
    head_dim = config.n_embd // h
    q, kv = rows * config.n_embd, rows * config.n_kv_head * head_dim
    # positions[:, None] - positions[None, :] (int64), two comparisons and an and, all [T, T]
    ops = [_ew(f'attention mask{tag}', T * T, 8 + 9 + 9 + 3, kernels=4)]
    dense = 2 * rows * T * head_dim * h  # one of the two matmuls over all T keys; the mask skips nothing
    if kernel == 'efficient':
        ops.append(_ew(f'attention mask to additive{tag}', T * T, 1 + dtype_bytes))
        ops.append(Op(f'attention (efficient){tag}', 'attention', 2 * dense,
                      (q + 2 * kv + q) * dtype_bytes + T * T * dtype_bytes, kernels=1))
    elif kernel == 'math':
        scores = rows // T * h * T * T
        ops += [
            _matmul(f'attention scores{tag}', dense, (q + kv + scores) * dtype_bytes),
            _ew(f'attention mask add{tag}', scores, 3 * dtype_bytes),
            _ew(f'attention softmax{tag}', scores, 2 * dtype_bytes),
            _matmul(f'attention values{tag}', dense, (scores + kv + q) * dtype_bytes),
        ]
    else:
        raise ValueError(f'attention kernel must be one of {ATTENTION_KERNELS}')
    return ops


def _forward(config, rows, start=None, jitter=(), attention='efficient'):
    """Ops of one forward over `rows` tokens (batch * sequence, times nrep for a draw chunk).

    start=None is GPT.forward from the token embedding; start=l is GPT.forward_from(l), the cached
    suffix that reruns blocks l onward. jitter names the sites in the first block run (or 'wte') whose
    forward hook adds a noise tensor: 'o' (attn.c_proj input), 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'.
    """
    n = config.n_embd
    kv = config.n_kv_head * (n // config.n_head)
    act = rows * n
    first = 0 if start is None else start
    windows = window_sizes(config)

    def jitter_add(site, elements):
        return [_ew('jitter add', elements, 3 * ACT)] if site in jitter else []

    ops = []
    if start is None:
        ops += [_ew('token embedding lookup', act, 2 * ACT)] + jitter_add('wte', act)
        ops.append(_ew('rms_norm', act, 2 * ACT))
    for layer in range(first, config.n_layer):
        here = jitter if layer == first else ()
        ops.append(_ew('residual mix', act, 9 * ACT, kernels=3))
        ops.append(_ew('rms_norm', act, 2 * ACT))
        ops += _linear('attn.c_q', rows, n, n) + _linear('attn.c_k', rows, n, kv) + _linear('attn.c_v', rows, n, kv)
        if _has_ve(layer, config.n_layer):
            ops.append(_ew('value embedding lookup', rows * kv, 2 * ACT))
            ops += _linear('attn.ve_gate', rows, GATE_CHANNELS, config.n_kv_head)
            ops.append(_ew('value gate', rows * config.n_kv_head, 4 * ACT, kernels=2))  # sigmoid, * 2
            ops.append(_ew('value mix', rows * kv, 6 * ACT, kernels=2))                 # gate * ve, v + it
        # apply_rotary_emb: four half-width multiplies, two adds and a cat (9 bytes/elt/byte-width), q and k
        ops.append(_ew('rotary', act + rows * kv, 9 * ACT, kernels=14))
        ops.append(_ew('rms_norm', act + rows * kv, 2 * ACT, kernels=2))
        ops += _attention(config, rows, windows[layer], attention)
        ops.append(_ew('attention output copy', act, 2 * ACT))  # transpose(1, 2).contiguous()
        if 'o' in here:
            ops += [_ew('jitter add', act, 3 * ACT)]
        ops += _linear('attn.c_proj', rows, n, n)
        if 'attn.c_proj' in here:
            ops += [_ew('jitter add', act, 3 * ACT)]
        ops.append(_ew('residual add', act, 3 * ACT))
        ops.append(_ew('rms_norm', act, 2 * ACT))
        ops += _linear('mlp.c_fc', rows, n, 4 * n)
        if 'mlp.c_fc' in here:
            ops += [_ew('jitter add', 4 * act, 3 * ACT)]
        ops.append(_ew('relu squared', 4 * act, 4 * ACT, kernels=2))
        ops += _linear('mlp.c_proj', rows, 4 * n, n)
        if 'mlp.c_proj' in here:
            ops += [_ew('jitter add', act, 3 * ACT)]
        ops.append(_ew('residual add', act, 3 * ACT))
    logits = rows * config.vocab_size
    ops.append(_ew('rms_norm', act, 2 * ACT))
    ops += _linear('lm_head', rows, n, _padded_vocab(config))
    ops.append(_ew('logits to fp32', logits, ACT + F32))
    ops.append(_ew('logit soft cap', logits, 6 * F32, kernels=3))     # / 15, tanh, * 15
    ops.append(_ew('cross-entropy', logits, 2 * F32, kernels=2))      # log_softmax, then the target gather
    return ops


def _weight_gradient(tokens, fan_out, fan_in):
    """DUST.reassemble for a Linear: einsum('btd,bti->di') of fp32 error and fp32 input (a TF32 GEMM)."""
    return _matmul('weight-gradient matmul', 2 * tokens * fan_out * fan_in,
                   (tokens * (fan_out + fan_in) + fan_out * fan_in) * F32, precision='tf32')


def _embedding_gradient(config, tokens, width):
    """DUST.reassemble for an Embedding: a zeroed fp32 table, then index_add_ of the token errors."""
    table = _padded_vocab(config) * width
    return [_ew('embedding-gradient init', table, F32),
            _ew('embedding-gradient scatter-add', tokens * width, 3 * F32)]


def _linear_inputs(config):
    """(fan_in) of every Linear one forward runs, in order: the inputs the clean pass stores in fp32."""
    n = config.n_embd
    fans = []
    for layer in range(config.n_layer):
        fans += [n, n, n] + ([GATE_CHANNELS] if _has_ve(layer, config.n_layer) else []) + [n, n, 4 * n]
    return fans + [n]  # lm_head


def _parameters(config):
    """(elements, bytes per element) of every parameter, mirroring GPT.__init__ (embeddings in bf16)."""
    n, padded = config.n_embd, _padded_vocab(config)
    kv = config.n_kv_head * (n // config.n_head)
    params = [(padded * n, ACT), (padded * n, F32), (config.n_layer, F32), (config.n_layer, F32)]
    for layer in range(config.n_layer):
        params += [(n * n, F32), (kv * n, F32), (kv * n, F32), (n * n, F32), (4 * n * n, F32), (4 * n * n, F32)]
        if _has_ve(layer, config.n_layer):
            params += [(config.n_kv_head * GATE_CHANNELS, F32), (padded * kv, ACT)]
    return params


def _direct(config, tokens, nrep, sites, start, suffix, attention):
    """DUST.direct_errors for one group of sites: (once-per-call ops, ops per chunk of nrep draws).

    sites is a list of (jitter site, error width). start is the block the sites live in, or None for
    the token embedding, which always reruns the whole network.
    """
    rows = nrep * tokens
    n = config.n_embd
    once = [_ew('error accumulator init', tokens * width, F32) for _, width in sites]
    once += [_ew('error average', tokens * width, 2 * F32) for _, width in sites]
    resume = start if suffix else None
    if resume is not None:  # streams[start].repeat and streams[0].repeat for the replicated batch
        once.append(Op('suffix stream repeat', 'elementwise', 2 * rows * n, 2 * (tokens + rows) * n * ACT, 2, 'fp32'))
    chunk = []
    for _, width in sites:
        chunk += [_ew('noise', rows * width, ACT), _ew('jitter scale', rows * width, 2 * ACT)]
    chunk += _forward(config, rows, resume, {site for site, _ in sites}, attention)
    chunk.append(_ew('reward', rows, 3 * F32, kernels=3 if nrep > 1 else 1))
    for _, width in sites:
        # einsum('nbt,nbtd->btd') sums the nrep draws per token: 2 FLOPs per noise element read
        chunk.append(_matmul('reward-weighted noise', 2 * rows * width,
                             rows * width * ACT + rows * ACT + tokens * width * ACT))
        chunk.append(_ew('error update', tokens * width, ACT + F32 + 2 * F32 + 3 * F32, kernels=3))
    return once, chunk


def update_ops(recipe, config, world=1, batch=8, suffix=True, attention='efficient'):
    """Per-phase op ledgers of one DUST.estimate followed by DUST.apply, on one GPU.

    Draw counts are divided by world as dust.py does; the clean pass and the scalars' forwards run on
    every GPU. suffix=False models --full-replay. Communication (average_ranks) is not modelled.
    """
    if attention not in ATTENTION_KERNELS:
        raise ValueError(f'attention kernel must be one of {ATTENTION_KERNELS}')
    # The same divisibility DUST.__init__ enforces, so the model never floors a count the run would reject
    counts = recipe.writers + recipe.hidden + recipe.hubs + [recipe.embedding] + list(recipe.local.values())
    slabs = config.vocab_size // SLAB
    if any(k % (world * recipe.nrep) for k in counts) or 8 % world or recipe.head % (world * recipe.nrep * slabs):
        raise ValueError('Population/chunk size is incompatible with this GPU count.')
    T, n, h = config.sequence_len, config.n_embd, config.n_head
    kv = config.n_kv_head * (n // h)
    tokens, nrep = batch * T, recipe.nrep
    rows = nrep * tokens
    ledgers = {phase: Ledger() for phase in PHASES}

    clean = ledgers['clean']
    clean.add(_forward(config, tokens, attention=attention))
    clean.add([_ew('capture input cast', tokens * fan_in, ACT + F32) for fan_in in _linear_inputs(config)])

    def direct_group(ledger, sites, draws, start, gradients):
        once, chunk = _direct(config, tokens, nrep, sites, start, suffix, attention)
        ledger.add(once)
        ledger.add(chunk, draws // nrep)
        ledger.add(gradients)

    for layer in range(config.n_layer):
        direct_group(ledgers['direct'], [('attn.c_proj', n), ('mlp.c_proj', n)], recipe.writers[layer] // world,
                     layer, [_weight_gradient(tokens, n, n), _weight_gradient(tokens, n, 4 * n)])
        direct_group(ledgers['direct'], [('mlp.c_fc', 4 * n)], recipe.hidden[layer] // world,
                     layer, [_weight_gradient(tokens, 4 * n, n)])
    direct_group(ledgers['embedding'], [('wte', n)], recipe.embedding // world, None,
                 _embedding_gradient(config, tokens, n))
    for layer in range(config.n_layer):
        direct_group(ledgers['hubs'], [('o', n)], recipe.hubs[layer] // world, layer, [])

    _head(ledgers['head'], config, tokens, nrep, recipe.head // world)
    for kind, draws in recipe.local.items():
        _local(ledgers['local'], config, tokens, nrep, kind, draws // world, attention)

    scalars = ledgers['scalars']
    draws = 8 // world
    scalars.add(_forward(config, tokens, attention=attention), 2 * draws)
    # copy_ of both parameter vectors per sign, the noise, then the [y != -1] boolean index (a host sync)
    scalars.add(_ew('scalar perturbation', 4 * config.n_layer, 2 * F32, kernels=5), draws)
    scalars.add(_ew('scalar reward', tokens, 4 * F32, kernels=5, syncs=1), draws)

    apply = ledgers['apply']
    apply.add(_ew('train-loss readout', tokens, 2 * F32, kernels=3, syncs=2))  # [y != -1].mean().item()
    for elements, width in _parameters(config):
        apply.add(_ew('gradient normalize', elements, 2 * F32 + (F32 + width if width != F32 else 0)))
        apply.add(_ew('finite check', elements, width + 1, kernels=2, syncs=1))  # isfinite().all() in an if
    # SGD with momentum (foreach): read parameter, gradient and buffer, write parameter and buffer
    apply.add(Op('sgd momentum step', 'elementwise', sum(e for e, _ in _parameters(config)) * 2,
                 sum(e * (3 * w + 2 * F32) for e, w in _parameters(config)), kernels=12, precision='fp32'))
    return ledgers


def _head(ledger, config, tokens, nrep, draws):
    """DUST.head_error: the cap and slab sums of the cached logits once, then one slab per draw chunk."""
    vocab, padded = config.vocab_size, _padded_vocab(config)
    rows, slab_rows = nrep * tokens, nrep * tokens * SLAB
    ledger.add([
        _ew('head logit cap', tokens * vocab, ACT + F32 + 6 * F32, kernels=4),
        _ew('head slab sums', tokens * vocab, 3 * F32, kernels=2),          # exp, then the per-slab sum
        _ew('head error init', tokens * padded, F32),
        _ew('head error average', tokens * padded, 2 * F32),
        _weight_gradient(tokens, padded, config.n_embd),
    ])
    ledger.add([
        _ew('head noise', slab_rows, ACT),
        _ew('head slab repeat', tokens * SLAB, ACT + F32 + nrep * F32, kernels=2),  # slice .float(), repeat
        _ew('head jitter', slab_rows, ACT + F32 + 3 * F32 + 2 * F32, kernels=3),   # a.float(), * sigma, +
        _ew('head slab cap', slab_rows, 6 * F32, kernels=3),
        _ew('head slab partition', slab_rows, 3 * F32, kernels=2),               # exp, sum over the slab
        _ew('head reward', rows, 10 * F32, kernels=12),                          # partition, target, CE, centering
        _matmul('head reward-weighted noise', 2 * slab_rows, slab_rows * ACT + rows * ACT + tokens * SLAB * ACT),
        _ew('head error update', tokens * SLAB, ACT + F32 + 3 * F32, kernels=2),
    ], draws // nrep)


def _local(ledger, config, tokens, nrep, kind, draws, attention):
    """DUST.local_errors for one kind: per block, the clean recompute once, then each draw chunk.

    Per-head scores need the estimated attention-output error, so this follows the hubs. Queries keep
    their own token's score; the other kinds also pay the [T, T] credit matmul (gamma decay).
    """
    T, n, h = config.sequence_len, config.n_embd, config.n_head
    kv = config.n_kv_head * (n // h)
    rows = nrep * tokens
    width = h if kind == 'gate' else n
    layers = [l for l in range(config.n_layer) if kind not in ('ve', 'gate') or _has_ve(l, config.n_layer)]
    windows = window_sizes(config)
    for layer in layers:
        has_ve = _has_ve(layer, config.n_layer)
        once = [
            _ew('local clean casts', tokens * (n + 2 * kv + (kv if has_ve else 0)), ACT + F32, kernels=4),
            _ew('local clean rotary+norm', tokens * (n + kv), 9 * F32 + 2 * F32 + F32 + ACT, kernels=18),
            _ew('local values', tokens * kv, 6 * F32 + F32 + ACT, kernels=5),
            *_attention(config, tokens, windows[layer], attention, tag=' (local)'),
            # lag, the window test and gamma ** lag as one fp32 [T, T] credit matrix
            _ew('local credit matrix', T * T, 74, kernels=8),
            _ew('local error init', tokens * width, F32),
            _ew('local error average', tokens * width, 2 * F32),
        ]
        if kind == 've':
            once += _embedding_gradient(config, tokens, kv)
        else:
            fan_in = GATE_CHANNELS if kind == 'gate' else n
            once.append(_weight_gradient(tokens, width if kind != 'gate' else config.n_kv_head, fan_in))
        ledger.add(once)

        chunk = [_ew('local noise', rows * width, F32)]
        repeat = Op('local repeat', 'elementwise', rows * n, (tokens + rows) * n * ACT, 1, 'fp32')
        if kind in ('q', 'k'):
            chunk += [
                _ew('local jitter', rows * n, 5 * F32, kernels=2),
                _ew('local rotary+norm', rows * n, 9 * F32 + 2 * F32 + F32 + ACT, kernels=18),
                repeat, _ew('local values cast', tokens * kv, F32 + ACT), repeat,
                *_attention(config, rows, windows[layer], attention, tag=' (local)'),
                _ew('local delta', rows * n, 3 * ACT),
            ]
        else:
            if kind == 'gate':
                chunk += [_ew('local gate jitter', rows * h, 8 * F32, kernels=6),
                          _ew('local gated value', rows * n, 2 * F32)]
            else:
                chunk.append(_ew('local jitter', rows * n, (5 if kind == 've' else 2) * F32,
                                 kernels=4 if kind == 've' else 1))
            chunk += [repeat, repeat, _ew('local value cast', rows * n, F32 + ACT),
                      *_attention(config, rows, windows[layer], attention, tag=' (local)')]
        chunk += [
            _ew('local target cast', tokens * n, F32 + ACT),
            _matmul('local attention-output score', 2 * rows * n, (rows + tokens) * n * ACT + rows * h * ACT),
            _ew('local score cast', rows * h, ACT + F32),
        ]
        if kind != 'q':
            chunk.append(_matmul('local credit', 2 * rows * h * T, 2 * rows * h * F32 + T * T * F32, precision='tf32'))
        chunk += [
            _matmul('local score-weighted noise', 2 * rows * width,
                    (rows * h + rows * width + tokens * width) * F32, precision='tf32'),
            _ew('local error update', tokens * width, 5 * F32, kernels=2),
        ]
        ledger.add(chunk, draws // nrep)


@dataclass
class PhaseRoofline:
    name: str
    flops: float
    nbytes: float
    kernels: int
    syncs: int
    compute_s: float         # sum of flops / peak, as if nothing were bandwidth-bound
    memory_s: float          # sum of bytes / bandwidth, as if nothing were compute-bound
    roofline_s: float        # sum over ops of max(compute, memory): the GPU-time floor
    launch_s: float          # kernels * launch latency: the CPU-issue floor
    predicted_s: float       # max(roofline_s, launch_s)
    bandwidth_bound_share: float  # fraction of roofline_s spent in ops left of their ridge
    regime: str              # 'launch', 'bandwidth' or 'compute'
    intensity: float         # flops / bytes
    top_ops: list = field(default_factory=list)  # (name, roofline seconds, 'compute' | 'bandwidth')


def _regime(roofline_s, launch_s, bandwidth_s):
    if launch_s > roofline_s:
        return 'launch'
    return 'bandwidth' if bandwidth_s > roofline_s / 2 else 'compute'


def _summarize(name, ops, hardware, top):
    rows, roofline_s, bandwidth_s = [], 0.0, 0.0
    for op in ops:
        compute, memory = op.flops / hardware.peak(op.precision), op.nbytes / hardware.hbm_bytes_per_s
        roofline_s += max(compute, memory)
        if memory >= compute:
            bandwidth_s += memory
        rows.append((op.name, max(compute, memory), 'bandwidth' if memory >= compute else 'compute'))
    flops, nbytes = sum(op.flops for op in ops), sum(op.nbytes for op in ops)
    kernels, syncs = sum(op.kernels for op in ops), sum(op.syncs for op in ops)
    launch_s = kernels * hardware.launch_s
    return PhaseRoofline(
        name=name, flops=flops, nbytes=nbytes, kernels=kernels, syncs=syncs,
        compute_s=sum(op.flops / hardware.peak(op.precision) for op in ops),
        memory_s=nbytes / hardware.hbm_bytes_per_s,
        roofline_s=roofline_s, launch_s=launch_s, predicted_s=max(roofline_s, launch_s),
        bandwidth_bound_share=bandwidth_s / roofline_s if roofline_s else 0.0,
        regime=_regime(roofline_s, launch_s, bandwidth_s),
        intensity=flops / nbytes if nbytes else 0.0,
        top_ops=sorted(rows, key=lambda row: -row[1])[:top],
    )


def roofline(recipe, config, hardware=A100_40GB, world=1, batch=8, suffix=True, attention='efficient', top=3):
    """Place one DUST update on `hardware`'s roofline, per phase and in total.

    The total's predicted_s is max(sum of GPU floors, sum of launch floors) over the whole update
    (the CPU can run ahead across phases between syncs), so the phases' predicted_s need not sum to it.
    predicted_mfu is the best MFU this update could reach: its FLOPs over predicted_s at the bf16 peak.
    """
    ledgers = update_ops(recipe, config, world, batch, suffix, attention)
    phases = {name: _summarize(name, list(ledger.ops.values()), hardware, top) for name, ledger in ledgers.items()}
    every = [op for ledger in ledgers.values() for op in ledger.ops.values()]
    total = _summarize('total', every, hardware, top)
    total.roofline_s = sum(p.roofline_s for p in phases.values())
    total.predicted_s = max(total.roofline_s, total.launch_s)
    return dict(
        hardware=asdict(hardware),
        settings=dict(world=world, batch=batch, suffix=suffix, attention=attention, nrep=recipe.nrep),
        phases=phases,
        total=total,
        weight_byte_share=sum(op.weight_bytes for op in every) / total.nbytes,
        predicted_mfu=mfu(total.flops, total.predicted_s, hardware),
    )


def mfu(flops, seconds, hardware=A100_40GB, precision='bf16'):
    """Model FLOP utilisation: achieved FLOP/s over the peak. Pass analytic FLOPs and a measured time."""
    return flops / seconds / hardware.peak(precision)


def amdahl(fractions, speedups):
    """Overall speedup when each part of a step (fractions of its time, summing to 1) is sped up by its
    factor in speedups (parts not named keep factor 1). Use measured fractions, e.g. per-phase CUDA time."""
    if abs(sum(fractions.values()) - 1) > 1e-6:
        raise ValueError('fractions must sum to 1')
    return 1 / sum(f / speedups.get(part, 1.0) for part, f in fractions.items())


def amdahl_limit(fraction):
    """The most a step can speed up if only `fraction` of its time is made free (S_max = 1 / (1 - p))."""
    return float('inf') if fraction >= 1 else 1 / (1 - fraction)


def format_report(report):
    """A plain-text table of a roofline() report, one row per phase plus the total."""
    hw = report['hardware']
    lines = [
        f"{hw['name']}: bf16 ridge {hw['bf16_flops'] / hw['hbm_bytes_per_s']:.0f} FLOP/byte, "
        f"launch {hw['launch_s'] * 1e6:.0f} us/kernel; {report['settings']}",
        f"{'phase':<10} {'TFLOP':>8} {'GB':>8} {'FLOP/B':>7} {'kernels':>8} {'syncs':>5} "
        f"{'gpu ms':>8} {'launch ms':>9} {'pred ms':>8} {'bw-bound':>8}  regime     top op",
    ]
    for phase in [*report['phases'].values(), report['total']]:
        name, seconds, bound = phase.top_ops[0] if phase.top_ops else ('-', 0, '')
        lines.append(
            f"{phase.name:<10} {phase.flops / 1e12:>8.2f} {phase.nbytes / 1e9:>8.2f} {phase.intensity:>7.1f} "
            f"{phase.kernels:>8} {phase.syncs:>5} {phase.roofline_s * 1e3:>8.2f} {phase.launch_s * 1e3:>9.2f} "
            f"{phase.predicted_s * 1e3:>8.2f} {phase.bandwidth_bound_share:>8.0%}  {phase.regime:<10} "
            f"{name} ({bound}, {seconds * 1e3:.2f} ms)"
        )
    lines.append(f"weight traffic {report['weight_byte_share']:.1%} of bytes; "
                 f"best-case MFU {report['predicted_mfu']:.1%}")
    return '\n'.join(lines)


def main():
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dust import DEFAULT_CONFIGS, DRAW_CHUNK_SIZES, recipe  # imports torch
    from model import GPTConfig

    parser = argparse.ArgumentParser(description='Predicted roofline of one DUST update.')
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--population', type=int, choices=DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--hardware', choices=HARDWARE, default='a100')
    parser.add_argument('--attention', choices=ATTENTION_KERNELS, default='efficient')
    parser.add_argument('--full-replay', action='store_true', help='model the public code (no cached suffix)')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON')
    args = parser.parse_args()

    report = roofline(recipe(args.tokens, args.population), GPTConfig(), HARDWARE[args.hardware], args.world,
                      suffix=not args.full_replay, attention=args.attention)
    if args.json:
        report = {**report, 'phases': {k: asdict(v) for k, v in report['phases'].items()},
                  'total': asdict(report['total'])}
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == '__main__':
    main()
