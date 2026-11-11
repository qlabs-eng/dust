"""A toy DUST setup that trains end to end on a CPU in seconds: tiny model, tiny recipe, learnable data.

For checking the training loop, its logging and checkpoint/resume, not for results. ``python dust.py
--toy`` and ``python baselines/backprop.py --toy`` use it in place of the FineWeb files.

The data is a noisy fixed rule over a 64-token alphabet: each next token is (5 * token + 3) mod 64 with
probability ``follow``, otherwise uniform over the alphabet. A model that learns it beats the uniform
loss ln(4096) = 8.32 by a wide margin: about ln(64) = 4.16 from the unigram alone, about 1.3 with the
rule. So a loss that falls well below 8.32 is a working training loop, not noise.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dust import Recipe  # noqa: E402  (import after the sys.path shim above)
from model import GPTConfig  # noqa: E402

TOY_CONFIG = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
TOY_STEPS = 40
TOY_EVAL_INTERVAL = 5
TOY_BATCH = 8
ALPHABET = 64


def toy_recipe(lr=0.25, momentum=0.95):
    """Every draw count a multiple of nrep; head a multiple of nrep * 16 vocabulary slabs."""
    return Recipe(
        writers=[4, 4, 2, 2], hidden=[4, 4, 2, 2], hubs=[2, 2, 2, 2], embedding=2, head=64,
        local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=lr, momentum=momentum,
    )


def toy_sequences(count, seed, sequence_len=TOY_CONFIG.sequence_len, follow=0.8):
    """[count, sequence_len + 1] token IDs following the noisy rule, the shape load_sequences returns."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.empty(count, sequence_len + 1, dtype=torch.long)
    tokens[:, 0] = torch.randint(0, ALPHABET, (count,), generator=generator)
    for t in range(1, sequence_len + 1):
        rule = (5 * tokens[:, t - 1] + 3) % ALPHABET
        noise = torch.randint(0, ALPHABET, (count,), generator=generator)
        keep = torch.rand(count, generator=generator) < follow
        tokens[:, t] = torch.where(keep, rule, noise)
    return tokens


def toy_entropy(follow=0.8):
    """The best achievable per-token loss on toy data (nats): the noisy rule's conditional entropy."""
    hit = follow + (1 - follow) / ALPHABET
    miss = (1 - follow) / ALPHABET
    return -(hit * math.log(hit) + (ALPHABET - 1) * miss * math.log(miss))


def toy_data(steps=TOY_STEPS, batch=TOY_BATCH, held_out=32, seed=0):
    """Training, validation and test splits, from disjoint seeds."""
    return (toy_sequences(steps * batch, seed), toy_sequences(held_out, seed + 1),
            toy_sequences(held_out, seed + 2))
