"""Minimal DUST implementation: forward-only transformer training with SGD.

No autograd or backward pass is used. One update (DUST.step):
  1. A clean forward caches every layer's input and output and the per-token losses.
  2. Direct draws jitter a layer's output at every token, y + sigma * a with a ~ N(0, I), rerun the
     forward and reward each token's jitter with the loss reduction at that token. The
     reward-weighted noise, averaged over draws, is the estimated error at the layer's output.
     A draw at block l reruns only blocks l onward, from the residual stream the clean pass
     cached at block l (cached-suffix execution; --full-replay reruns the whole network instead).
  3. The attention internals (queries, keys, values, value-embedding gates, value embeddings) are
     credited through the estimated error at the attention output instead,
     recomputing only their block's attention.
  4. The head is jittered on the cached logits, one 256-column slab at a time.
  5. The outer product of each layer's estimated output error with its cached input, summed over
     tokens, is the layer's weight gradient, as in backprop. The 2L residual mixing scalars use
     two-sided weight-space ES. Then one SGD step.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.profiler import record_function

from model import GPT, GPTConfig, attention, apply_rotary_emb, has_ve, norm


@dataclass
class Recipe:
    """Draws per update for each part of the model, summed over GPUs. Lists have one entry per block."""
    writers: list[int]      # draws per block for the attention and MLP output projections (c_proj)
    hidden: list[int]       # draws per block for the MLP's first layer (mlp.c_fc)
    hubs: list[int]         # draws per block for the attention output, before attn.c_proj
    embedding: int          # draws for the token embedding
    head: int               # draws for the output head (lm_head)
    local: dict[str, int]   # draws per block for each attention internal: q, v, gate, k, ve
    nrep: int               # draws run together in one forward pass
    lr: float
    momentum: float
    gamma: float = 0.98  # attention-internal credit decay per token of lag


DEFAULT_CONFIGS = {
    '1M': {
        'steps': 61,
        'eval_interval': 2,
        'optimizer': {
            256: dict(lr=0.25, momentum=0.95),
            1024: dict(lr=0.225, momentum=0.95),
            4096: dict(lr=0.225, momentum=0.98),
            16384: dict(lr=0.225, momentum=0.98),
        },
    },
    '10M': {
        'steps': 610,
        'eval_interval': 10,
        'optimizer': {
            256: dict(lr=0.175, momentum=0.95),
            1024: dict(lr=0.25, momentum=0.95),
            4096: dict(lr=0.25, momentum=0.95),
            16384: dict(lr=0.25, momentum=0.95),
        },
        # At 16,384 draws the estimator constants were tuned with (lr, momentum) at fixed compute:
        # twice the attention-output draws, paid by the writer streams, and credit decay 0.99.
        'allocation': {
            16384: dict(writers=[34, 28, 14, 8, 6, 4, 2, 2], hubs=[12, 10, 8, 4, 4, 4, 4, 4], gamma=0.99),
        },
    },
}

# Draws per forward pass (the batch is repeated this many times).
DRAW_CHUNK_SIZES = {256: 2, 1024: 8, 4096: 8, 16384: 16}


def recipe(tokens, population):
    """Build a saved config, population counts direct-loss draws only. """
    optimizer = DEFAULT_CONFIGS[tokens]['optimizer'][population]
    nrep = DRAW_CHUNK_SIZES[population]
    scale = population // 256
    override = DEFAULT_CONFIGS[tokens].get('allocation', {}).get(population, {})
    writers = [v * scale for v in override.get('writers', [38, 32, 16, 10, 6, 4, 2, 2])]
    hidden = writers.copy()
    hubs = [v * scale for v in override.get('hubs', [6, 6, 4, 2, 2, 2, 2, 2])]
    embedding, local = 10 * scale, [v * scale for v in [2, 20, 6, 12, 8]]

    return Recipe(
        writers, hidden, hubs, embedding, 4 * population,
        dict(zip(('q', 'v', 'gate', 'k', 've'), local)),
        nrep, **optimizer, gamma=override.get('gamma', 0.98),
    )


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


class DUST:
    """
    The actual algorithm.

    Forward hooks on every Linear and Embedding capture its input and output during the clean pass and add the
    current jitter to its output during a draw. Pre-hooks on attn.c_proj add the virtual sites o@{layer}, the
    attention output before the projection. Each GPU draws different noise for the same batch.

    With suffix=True (baseline A) a direct draw at block l restarts the forward at block l from the
    clean pass's cached residual stream; suffix=False reruns the whole network (the public code).
    Both consume the same noise, so for a fixed seed they differ only by floating-point rounding.
    """

    def __init__(self, model, settings, seed=42, rank=0, world=1, suffix=True):
        self.model, self.settings, self.world, self.suffix = model, settings, world, suffix
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.jitter = {}, {}, {}
        self.capturing = False

        counts = settings.writers + settings.hidden + settings.hubs + [settings.embedding]
        counts += list(settings.local.values())
        if any(k % (world * settings.nrep) for k in counts) or 8 % world:
            raise ValueError('Population/chunk size is incompatible with this GPU count.')
        slabs = model.config.vocab_size // 256
        if model.config.vocab_size % 256 or settings.head % (world * settings.nrep * slabs):
            raise ValueError('Head draws must visit each 256-column slab equally on every GPU.')

        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer in range(model.config.n_layer):
            module = self.modules[f'transformer.h.{layer}.attn.c_proj']
            self.hooks.append(module.register_forward_pre_hook(self.input_hook(f'o@{layer}')))

    def autocast(self):
        return torch.autocast('cuda', dtype=self.dtype) if self.device.type == 'cuda' else nullcontext()

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
            if name in self.jitter:
                return output + self.jitter[name].to(output.dtype)

        return hook

    def input_hook(self, name):
        def hook(module, args):
            if name in self.jitter:
                return (args[0] + self.jitter[name].to(args[0].dtype),)

        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def noise(self, *shape, dtype=None):
        return torch.randn(*shape, device=self.device, dtype=dtype or self.dtype, generator=self.generator)

    def losses(self, x, y, resume=None, streams=None):
        """Per-token losses. resume = (start, stream, x0) reruns only blocks start onward."""
        with self.autocast():
            if resume is None:
                loss = self.model(x, y, loss_reduction='none', streams=streams)
            else:
                loss = self.model.forward_from(*resume, x, y, loss_reduction='none')
            return loss.view_as(y).float()

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.streams = []  # streams[l]: the clean residual stream entering block l; streams[0] is x0
        self.capturing = True
        try:
            self.clean_loss = self.losses(x, y, streams=self.streams)
        finally:
            self.capturing = False

    def direct_errors(self, sites, draws, sigma, start=None):
        """One-sided loss drops, centered across each chunk of draws.

        start is the first block the sites can change; with cached-suffix execution each draw reruns
        only blocks start onward. None (the token embedding) always reruns the whole network.
        """
        n = self.settings.nrep
        x, y = self.x.repeat(n, 1), self.y.repeat(n, 1)
        resume = None
        if self.suffix and start is not None:
            resume = start, self.streams[start].repeat(n, 1, 1), self.streams[0].repeat(n, 1, 1)
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        for _ in range(draws // n):
            noise = {s: self.noise(n * self.batch, self.length, width(s)) for s in sites}
            self.jitter = {s: sigma * a for s, a in noise.items()}
            try:
                perturbed_loss = self.losses(x, y, resume).view(n, self.batch, self.length)
            finally:
                self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            for site, a in noise.items():
                errors[site] -= torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype),
                    a.view(n, self.batch, self.length, -1),
                ).float() / sigma

        return {s: error / draws for s, error in errors.items()}

    def head_error(self, draws, sigma=0.05):
        """Perturb one vocabulary slab at a time; recompute CE from its changed logits.

        Each chunk jitters one 256-column slab of the cached logits, and the loss is recomputed from the
        cached sums of the other slabs. This is much cheaper than a forward pass, and each reward only
        reflects 256 jittered columns instead of all 4096.
        """
        n, b, t = self.settings.nrep, self.batch, self.length
        raw = self.outputs['lm_head']
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        logits = 15 * torch.tanh(raw[..., :vocab].float() / 15)
        slab_sums = logits.exp().view(b, t, slabs, slab).sum(-1)
        total = slab_sums.sum(-1)
        target = self.y.clamp(min=0)
        clean_target = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        target = target.repeat(n, 1)
        error = torch.zeros_like(raw, dtype=torch.float32)

        for chunk in range(draws // n):
            block = chunk % slabs
            lo = block * slab
            a = self.noise(n * b, t, slab)
            changed = raw[..., lo:lo + slab].float().repeat(n, 1, 1) + sigma * a.float()
            changed = 15 * torch.tanh(changed / 15)
            partition = (total - slab_sums[..., block]).repeat(n, 1) + changed.exp().sum(-1)
            in_slab = (target >= lo) & (target < lo + slab)
            target_logit = torch.where(
                in_slab,
                changed.gather(
                    -1, (target - lo).clamp(0, slab - 1).unsqueeze(-1),
                ).squeeze(-1),
                clean_target.repeat(n, 1),
            )
            loss = (partition.log() - target_logit).view(n, b, t)
            reward = torch.where(self.y.unsqueeze(0) != -1, self.clean_loss.unsqueeze(0) - loss, 0)
            if b > 1:
                reward = reward - reward.mean(1, keepdim=True)
            error[..., lo:lo + slab] -= torch.einsum(
                'nbt,nbtd->btd', (reward / sigma).to(self.dtype), a.view(n, b, t, slab),
            ).float()

        return error / (draws // slabs)

    def local_errors(self, kind, draws, targets, sigma=0.05):
        """Score attention-output changes against estimated output errors (no derivatives).

        Each draw recomputes the block's attention output from the cached clean activations and scores the
        change per head against the output's estimated error (`targets`, computed by direct_errors on the 
        o@{layer} sites). Queries keep their own token's score; the other kinds sum the scores of later
        tokens, decayed by gamma per token.
        """
        cfg = self.model.config
        n, b, t, h, d = self.settings.nrep, self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {
            i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
            for i in layers
        }
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        def repeat(x):
            return x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)

        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q), processed(k)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
            # credit[t, s]: how much of output s's score goes to the jitter at t, inside the causal window
            credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        for _ in range(draws // n):
            for i in layers:
                a = self.noise(n, b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                divisor = sigma
                if kind in ('q', 'k'):
                    raw = q if kind == 'q' else k
                    perturbed = processed(
                        (raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d)
                    )
                    qa, ka = (perturbed, repeat(kp)) if kind == 'q' else (repeat(qp), perturbed)
                    delta = (
                        attention(qa, ka, repeat(values.to(self.dtype)), window).view(n, b, t, h, d)
                        - baseline
                    )
                else:
                    if kind == 'gate':
                        delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
                        delta_value = delta_gate.unsqueeze(-1) * ve
                        divisor = 2 * sigma
                    else:
                        delta_value = sigma * a.view(n, b, t, h, d)
                        if kind == 've':
                            delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
                    delta = attention(
                        repeat(qp), repeat(kp),
                        delta_value.reshape(n * b, t, h, d).to(self.dtype), window,
                    ).view(n, b, t, h, d)

                score = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                if kind != 'q':
                    score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
                estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
                errors[i] += estimate.reshape(b, t, width) / divisor

        return {sites[i]: errors[i] / draws for i in layers}

    def reassemble(self, site, error):
        """The exact local linear map: error times input, or scatter-add for an embedding. """
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

    def scalar_gradients(self, draws, sigma=0.03):
        """Two-sided weight-space ES for the 2L residual mixing scalars, rewarded by the summed token loss."""
        parameters = [self.model.resid_lambdas, self.model.x0_lambdas]
        original = [p.clone() for p in parameters]
        estimate = torch.zeros(2, self.model.config.n_layer, device=self.device)
        try:
            for _ in range(draws):
                a = self.noise(2, self.model.config.n_layer, dtype=torch.float32)
                losses = []
                for sign in (1, -1):
                    for p, clean, direction in zip(parameters, original, a):
                        p.copy_(clean + sign * sigma * direction)
                    losses.append(self.losses(self.x, self.y))
                difference = (losses[0] - losses[1])[self.y != -1].sum() / (2 * sigma)
                estimate += difference * a / draws
        finally:
            for p, clean in zip(parameters, original):
                p.copy_(clean)
        return dict(zip(('resid_lambdas', 'x0_lambdas'), estimate))

    @torch.no_grad()
    def estimate(self, x, y):
        """One clean pass plus the forward-only gradient estimate, without touching the weights.

        Returns (gradients, valid_tokens, clean_loss). ``gradients`` maps each parameter name to
        its estimate summed over tokens; dividing by ``valid_tokens`` gives the gradient of the
        mean loss, the same normalization the backprop baseline uses. Diagnostics and the adaptive
        allocator call this and inspect the estimate; apply() consumes it to take the SGD step.
        estimate() followed by apply() is identical to the old fused step().
        """
        # record_function labels name each phase in the profiler table and the Perfetto trace, so the
        # trace reads as direct / embedding / hubs / head / local / scalars (Lecture 3). They are inert
        # when no profiler is active, so they stay in the hot path.
        with record_function('dust/clean'):
            self.capture(x, y)
        r, world = self.settings, self.world
        gradients = {}
        with record_function('dust/direct'):
            for layer in range(self.model.config.n_layer):
                prefix = f'transformer.h.{layer}.'
                groups = [
                    ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer]),
                    ([prefix + 'mlp.c_fc'], r.hidden[layer]),
                ]
                for sites, count in groups:
                    for site, error in self.direct_errors(sites, count // world, 0.2, layer).items():
                        gradients[site + '.weight'] = self.reassemble(site, error)

        with record_function('dust/embedding'):
            site = 'transformer.wte'
            error = self.direct_errors([site], r.embedding // world, 0.2)[site]
            gradients[site + '.weight'] = self.reassemble(site, error)

        # Attention outputs, each block alone. The deeper half moves the loss too little at 0.2 for a readable
        # reward, so it uses 0.4. The targets are averaged over GPUs so that every GPU scores the attention
        # internals against the estimate from the whole population.
        with record_function('dust/hubs'):
            targets = {}
            for layer in range(self.model.config.n_layer):
                site = f'o@{layer}'
                sigma = 0.2 if layer < self.model.config.n_layer // 2 else 0.4
                errors = self.direct_errors([site], r.hubs[layer] // world, sigma, layer)
                targets[layer] = average_ranks(errors[site])

        with record_function('dust/head'):
            gradients['lm_head.weight'] = self.reassemble('lm_head', self.head_error(r.head // world))
        with record_function('dust/local'):
            for kind, count in r.local.items():
                for site, error in self.local_errors(kind, count // world, targets).items():
                    gradients[site + '.weight'] = self.reassemble(site, error)
        with record_function('dust/scalars'):
            gradients.update(self.scalar_gradients(8 // world))

        valid_tokens = (y != -1).sum()
        return gradients, valid_tokens, self.clean_loss[y != -1].mean().item()

    @torch.no_grad()
    def apply(self, gradients, valid_tokens, optimizer):
        """Normalize the summed estimate to the mean-loss gradient, sync across ranks, take one SGD step."""
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            gradient = average_ranks((gradients[name] / valid_tokens).to(parameter.dtype))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = gradient
        optimizer.step()

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """One clean pass, forward-only estimate, then an SGD update (estimate followed by apply)."""
        gradients, valid_tokens, loss = self.estimate(x, y)
        self.apply(gradients, valid_tokens, optimizer)
        return loss


def load_sequences(path, count, sequence_len=2048):
    """Read a token matrix or chunked dataset, excluding padded rows."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        rows = [
            chunk.view(data['batch_size'], data['sequence_size'])[:valid]
            for chunk, valid in zip(data['chunks'], data['valid_counts'])
        ]
        data = torch.cat(rows)
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != sequence_len + 1:
        raise ValueError(f'{path}: expected [sequences, {sequence_len + 1}] token IDs')
    if len(data) < count:
        raise ValueError(f'{path}: need {count} sequences, found {len(data)}')
    if data.is_floating_point() or data.is_complex() or data.dtype == torch.bool:
        raise ValueError(f'{path}: token IDs must have an integer dtype')
    data = data[:count].long()
    if data.min() < 0 or data.max() >= 4096:
        raise ValueError(f'{path}: token IDs must be in [0, 4096)')
    return data


@torch.no_grad()
def evaluate(model, rows, device, batch_size=32):
    total, tokens = 0.0, 0
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite evaluation loss')
        total += loss.item() * y.numel()
        tokens += y.numel()
    return total / tokens


def make_model(config, device, seed):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # init output projections at std 1.5 / sqrt(fan_in)
    # instead of the zeros init_weights sets.
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


def make_optimizer(model, settings):
    groups = {}
    for name, parameter in model.named_parameters():
        lr = settings.lr
        if name == 'transformer.wte.weight':
            lr = 1000.0
        elif 'value_embeds' in name:
            lr = 0.3
        elif 'lambdas' in name:
            lr = 0.03
        groups.setdefault(lr, []).append(parameter)
    return torch.optim.SGD(
        [{'params': ps, 'lr': lr} for lr, ps in groups.items()],
        momentum=settings.momentum,
    )


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument(
        '--population', type=int, choices=DRAW_CHUNK_SIZES, default=16384,
        help='Total direct-loss draws across GPUs (default: 16384); each value selects its tuned recipe.',
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/default'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--steps', type=int,
        help='Stop early for a smoke test; the training subset stays unchanged.',
    )
    parser.add_argument(
        '--full-replay', action='store_true',
        help='Rerun the whole network for every direct draw, as the public code does (upstream reference).',
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)

    settings = recipe(args.tokens, args.population)
    defaults = DEFAULT_CONFIGS[args.tokens]
    config = GPTConfig()
    steps = defaults['steps']
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    paths = dict(
        train=args.data_dir / f'train_{args.tokens}.pt',
        val=args.data_dir / 'val.pt',
        test=args.data_dir / 'test.pt',
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            f"Missing data: {', '.join(missing)}. Run python prepare_data.py "
            f"--tokens {args.tokens} --output {args.data_dir}"
        )
    training = load_sequences(paths['train'], steps * 8)
    validation = load_sequences(paths['val'], 544)
    testing = load_sequences(paths['test'], 544)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    # Every rank sees the same batch; only the perturbation population is divided across ranks.
    model = make_model(config, device, args.seed)
    optimizer = make_optimizer(model, settings)
    estimator = DUST(model, settings, args.seed, rank, world, suffix=not args.full_replay)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.output}')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()

    metadata = dict(
        tokens=args.tokens,
        population=args.population,
        seed=args.seed,
        world=world,
        steps=steps,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=544,
        test_sequences=544,
        model=asdict(config),
        recipe=asdict(settings),
        reward_centering='per-token local draw-chunk mean; skipped at nrep=1; no rescaling',
        execution='full-replay' if args.full_replay else 'cached-suffix',
        parameter_groups=[dict(lr=g['lr'], momentum=g['momentum'],
                               parameters=sum(p.numel() for p in g['params']))
                          for g in optimizer.param_groups],
        source_sha256=file_hash(Path(__file__)),
        model_sha256=file_hash(Path(__file__).parent / 'model.py'),
        torch_version=torch.__version__,
    )
    if rank == 0:
        metadata['data_sha256'] = {k: file_hash(v) for k, v in paths.items()}
        (args.output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    best = evaluate(model, validation, device)

    def save_best():
        if rank == 0:
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                args.output / 'best.pt',
            )

    save_best()
    history = [dict(step=0, val_loss=best)]
    interval = defaults['eval_interval']
    start = time.monotonic()
    if rank == 0:
        print(
            f'{args.tokens}: {steps} steps, population={args.population}, GPUs={world}, val={best:.4f}',
            flush=True,
        )

    for step in range(1, steps + 1):
        batch = training[(step - 1) * 8:step * 8].to(device)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        record = dict(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - start)
        if step == 1 or step % interval == 0 or step == steps:
            value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best = value
                save_best()
        history.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')

    estimator.close()
    if rank == 0:
        model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True))
        result = dict(best_val=best, test_at_best_val=evaluate(model, testing, device), history=history)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f"Best validation: {best:.4f}; test: {result['test_at_best_val']:.4f}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
