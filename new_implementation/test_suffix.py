"""Track 1 gate: cached-suffix execution (baseline A) against full replay (the public code), on CPU.

A direct draw at block l changes nothing before block l, so restarting the forward at block l from the
clean pass's cached residual stream must give the same losses, and so the same gradient estimate, as
rerunning the whole network with the same noise. Checked on a tiny model in fp32 so the comparison is
tight; the bf16 GPU check and the timing comparison are the (GPU) half of the gate.

Run: python -m pytest new_implementation/test_suffix.py   (or python new_implementation/test_suffix.py)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dust import DUST, Recipe, make_model  # noqa: E402  (import after the sys.path shim above)
from model import GPTConfig  # noqa: E402

CONFIG = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
NREP = 2


def tiny_recipe():
    # Every count a multiple of nrep; head a multiple of nrep * 16 slabs. Distinct per-block counts so a
    # wrong start block shows up in the call counts.
    return Recipe(
        writers=[6, 4, 2, 2], hidden=[4, 4, 2, 2], hubs=[2, 2, 2, 4], embedding=2, head=32,
        local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=NREP, lr=0.1, momentum=0.9,
    )


def batch(seed=0, size=2):
    tokens = torch.randint(0, CONFIG.vocab_size, (size, CONFIG.sequence_len + 1),
                           generator=torch.Generator().manual_seed(seed))
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def estimate(model, suffix, seed=7):
    estimator = DUST(model, tiny_recipe(), seed=seed, suffix=suffix)
    try:
        gradients, valid_tokens, loss = estimator.estimate(*batch())
    finally:
        estimator.close()
    return gradients, valid_tokens, loss


@torch.no_grad()
def test_forward_from_matches_forward():
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    x, y = batch()
    streams = []
    full = model(x, y, loss_reduction='none', streams=streams)
    assert len(streams) == CONFIG.n_layer and streams[0].shape == (*x.shape, CONFIG.n_embd)
    for start in range(CONFIG.n_layer):
        resumed = model.forward_from(start, streams[start], streams[0], x, y, loss_reduction='none')
        assert torch.equal(resumed, full), f'forward_from({start}) differs from the full forward'


def test_suffix_matches_full_replay():
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    suffix, valid_a, loss_a = estimate(model, suffix=True)
    replay, valid_b, loss_b = estimate(model, suffix=False)
    assert valid_a == valid_b and loss_a == loss_b
    assert suffix.keys() == replay.keys() == {n for n, _ in model.named_parameters()}
    for name in suffix:
        a, b = suffix[name].float(), replay[name].float()
        scale = b.abs().max().clamp(min=1e-12)
        error = (a - b).abs().max() / scale
        assert error < 1e-4, f'{name}: relative difference {error:.2e}'


def test_suffix_skips_unaffected_blocks():
    """Count block executions: a direct draw at block l must run blocks l.. and none before it."""
    recipe = tiny_recipe()
    chunks = lambda draws: draws // NREP
    calls = {}

    def run(suffix):
        model = make_model(CONFIG, torch.device('cpu'), seed=0)
        counts = [0] * CONFIG.n_layer
        hooks = [block.register_forward_hook(lambda *_, i=i: counts.__setitem__(i, counts[i] + 1))
                 for i, block in enumerate(model.transformer.h)]
        estimate(model, suffix)
        for hook in hooks:
            hook.remove()
        return counts

    calls['suffix'], calls['replay'] = run(True), run(False)

    # Per block: the clean pass, the embedding draws and 16 residual-scalar forwards always run it,
    # plus every direct chunk (writers, hidden, hubs) whose start block is at or before it.
    always = 1 + chunks(recipe.embedding) + 2 * 8
    direct = [chunks(recipe.writers[l] + recipe.hidden[l] + recipe.hubs[l]) for l in range(CONFIG.n_layer)]
    expected = [always + sum(direct[:b + 1]) for b in range(CONFIG.n_layer)]
    assert calls['suffix'] == expected, calls
    assert calls['replay'] == [always + sum(direct)] * CONFIG.n_layer, calls
    assert calls['suffix'][-1] == calls['replay'][-1]


if __name__ == '__main__':
    for test in (test_forward_from_matches_forward, test_suffix_matches_full_replay,
                 test_suffix_skips_unaffected_blocks):
        test()
        print(f'{test.__name__}: ok')
