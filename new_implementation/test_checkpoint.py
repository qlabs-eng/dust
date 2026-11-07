"""Checks for checkpoint.py, on CPU: a resumed run must be bit-for-bit the run that never stopped.

The core test trains a tiny model 4 steps straight, and separately 2 steps, saves, rebuilds everything
from different seeds (so anything not restored shows up), loads and trains 2 more. The negative
controls drop one piece of state each and must diverge, which proves the core test can fail.

Run: python -m pytest new_implementation/test_checkpoint.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import checkpoint_path, load_state, save_state  # noqa: E402  (after the sys.path shim)
from dust import DUST, Recipe, make_model, make_optimizer  # noqa: E402
from model import GPTConfig  # noqa: E402

CONFIG = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)


def tiny_recipe(lr=0.1):
    return Recipe(
        writers=[4, 2, 2, 2], hidden=[2, 2, 2, 2], hubs=[2, 2, 2, 2], embedding=2, head=32,
        local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=lr, momentum=0.9,
    )


def batch(step):
    tokens = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.sequence_len + 1),
                           generator=torch.Generator().manual_seed(1000 + step))
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class Run:
    """A model, its optimizer and its estimator, built from seeds like dust.main builds them."""

    def __init__(self, model_seed=0, estimator_seed=42, lr=0.1):
        self.settings = tiny_recipe(lr)
        self.model = make_model(CONFIG, torch.device('cpu'), model_seed)
        self.optimizer = make_optimizer(self.model, self.settings)
        self.estimator = DUST(self.model, self.settings, seed=estimator_seed)

    def train(self, steps):
        for step in steps:
            self.estimator.step(*batch(step), self.optimizer)

    def save(self, path, **kwargs):
        return save_state(path, model=self.model, optimizer=self.optimizer, estimator=self.estimator,
                          settings=self.settings, **kwargs)

    def load(self, path, **kwargs):
        return load_state(path, model=self.model, optimizer=self.optimizer, estimator=self.estimator, **kwargs)

    def close(self):
        self.estimator.close()


def momentum(optimizer):
    return [optimizer.state[p]['momentum_buffer'] for group in optimizer.param_groups for p in group['params']]


def same_state(a, b):
    weights = all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values()))
    buffers = all(torch.equal(x, y) for x, y in zip(momentum(a.optimizer), momentum(b.optimizer)))
    generator = torch.equal(a.estimator.generator.get_state(), b.estimator.generator.get_state())
    return weights, buffers, generator


@pytest.fixture
def straight():
    """The reference: 4 steps with no interruption."""
    run = Run()
    run.train(range(4))
    yield run
    run.close()


def resumed(path, drop=None):
    """2 steps, save, rebuild from other seeds, load, 2 more steps; `drop` skips restoring one piece."""
    first = Run()
    first.train(range(2))
    first.save(path, step=2)
    first.close()

    second = Run(model_seed=123, estimator_seed=999)
    step, _, _ = second.load(path)
    if drop == 'generator':
        second.estimator.generator.manual_seed(999)
    elif drop == 'momentum':
        second.optimizer = make_optimizer(second.model, second.settings)
    second.train(range(step, 4))
    return second


def test_resume_is_bit_for_bit(straight, tmp_path):
    run = resumed(tmp_path / 'state.pt')
    try:
        assert same_state(straight, run) == (True, True, True)
    finally:
        run.close()


@pytest.mark.parametrize('drop', ['generator', 'momentum'])
def test_resume_without_a_piece_of_state_diverges(straight, tmp_path, drop):
    run = resumed(tmp_path / 'state.pt', drop)
    try:
        weights, _, _ = same_state(straight, run)
        assert not weights, f'resuming without the {drop} still matched, so the core test proves nothing'
    finally:
        run.close()


def test_round_trip_keeps_dtypes_learning_rates_and_metadata(tmp_path):
    source = Run()
    source.train(range(1))
    # bf16 as init_weights stores it on CUDA; cast after training, since CPU has no bf16 autocast
    source.model.transformer.wte.to(dtype=torch.bfloat16)
    path = source.save(tmp_path / 'state.pt', step=1, allocator_state={'counts': [4, 2]}, extra={'best_val': 7.5})
    source.close()

    assert torch.load(path, weights_only=False)['model']['transformer.wte.weight'].dtype == torch.bfloat16
    assert not list(tmp_path.glob('*.tmp')), 'the atomic write left its temporary file behind'

    target = Run(model_seed=5, lr=0.5)  # a different base learning rate, which the load must undo
    target.model.transformer.wte.to(dtype=torch.bfloat16)
    step, allocator_state, extra = target.load(path)
    try:
        assert (step, allocator_state, extra) == (1, {'counts': [4, 2]}, {'best_val': 7.5})
        assert target.model.transformer.wte.weight.dtype == torch.bfloat16
        assert sorted(g['lr'] for g in target.optimizer.param_groups) == [0.03, 0.1, 0.3, 1000.0]
    finally:
        target.close()


def test_per_rank_files_and_world_mismatch(tmp_path):
    base = tmp_path / 'state.pt'
    assert checkpoint_path(base) == base
    assert checkpoint_path(base, rank=1, world=2) == tmp_path / 'state.pt.rank1'

    run = Run()
    try:
        for rank in range(2):
            run.save(base, step=0, rank=rank, world=2)
        assert {p.name for p in tmp_path.iterdir()} == {'state.pt.rank0', 'state.pt.rank1'}
        assert run.load(base, rank=1, world=2)[0] == 0
        with pytest.raises(ValueError, match='world=2'):
            run.load(base, rank=0, world=4)  # state.pt.rank0 exists but was written by a 2-GPU run
    finally:
        run.close()
