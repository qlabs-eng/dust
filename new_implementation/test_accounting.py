"""Checks for accounting.py, on CPU.

The core check runs a real DUST.estimate under torch's FlopCounterMode and requires analytic_cost's
forward FLOPs to equal what PyTorch executed. CPU SDPA reports no FLOPs, so the model's attention is
swapped for an explicit-matmul version that computes the same outputs (asserted) and is counted.

What analytic_cost covers and how it is checked:
- clean pass, direct draws (writers, hidden, hubs, embedding) and the 16 scalar forwards: exactly,
  against the FLOPs FlopCounterMode attributes to the transformer blocks and lm_head;
- head slab and attention-internal draws: not model-module matmuls, so checked by formula only.
Sliding-window blocks are the one known gap: accounting counts the keys inside each window, but
model.attention computes the masked dense scores, so the executed count is higher (pinned below).

Run: python -m pytest new_implementation/test_accounting.py
"""
import re
import sys
import time
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accounting as acc  # noqa: E402  (imports after the sys.path shim above)
import model as model_module  # noqa: E402
from dust import DUST, Recipe, make_model, recipe  # noqa: E402
from groups import population  # noqa: E402
from model import GPTConfig, has_ve  # noqa: E402

FULL_WINDOWS = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64, window_pattern='L')
SLIDING = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)  # SSSL, as shipped
FORWARD_GROUPS = ('writers', 'hidden', 'hubs', 'embedding', 'scalars')


def tiny_recipe():
    return Recipe(
        writers=[6, 4, 2, 2], hidden=[4, 4, 2, 2], hubs=[2, 2, 2, 4], embedding=2, head=32,
        local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=0.1, momentum=0.9,
    )


def matmul_attention(q, k, v, window):
    """model.attention with the scores and value mix as explicit matmuls, so FlopCounterMode sees them."""
    positions = torch.arange(q.shape[1], device=q.device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance <= window)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    return (scores.masked_fill(~allowed, float('-inf')).softmax(-1) @ v).transpose(1, 2)


@pytest.fixture
def counted_attention(monkeypatch):
    monkeypatch.setattr(model_module, 'attention', matmul_attention)


def tokens(config, batch=2, seed=0):
    rows = torch.randint(0, config.vocab_size, (batch, config.sequence_len + 1),
                         generator=torch.Generator().manual_seed(seed))
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def executed_forward_flops(config, settings, suffix, batch=2):
    """FLOPs a real estimate spends inside the transformer blocks and lm_head, minus the value-embedding
    gate (a 32-channel matmul accounting treats as lower order)."""
    model = make_model(config, torch.device('cpu'), seed=0)
    estimator = DUST(model, settings, suffix=suffix)
    try:
        with FlopCounterMode(display=False) as counter:
            estimator.estimate(*tokens(config, batch))
    finally:
        estimator.close()
    counts = {name: sum(ops.values()) for name, ops in counter.get_flop_counts().items()}
    # forward_from calls blocks outside GPT.__call__, so sum the blocks and head rather than 'GPT'.
    blocks = sum(v for k, v in counts.items() if re.fullmatch(r'GPT\.transformer\.h\.\d+|GPT\.lm_head', k))
    gates = sum(v for k, v in counts.items() if k.endswith('.ve_gate'))
    return blocks - gates


def predicted_forward_flops(cost):
    return cost.flops_clean_forward + sum(cost.flops_per_group[g] for g in FORWARD_GROUPS)


def test_matmul_attention_is_the_model_attention():
    q, k, v = (torch.randn(2, 64, 4, 16, generator=torch.Generator().manual_seed(i)) for i in range(3))
    for window in (31, 64):
        assert torch.allclose(matmul_attention(q, k, v, window), model_module.attention(q, k, v, window), atol=1e-5)


@pytest.mark.parametrize('suffix', [True, False], ids=['cached-suffix', 'full-replay'])
def test_forward_flops_match_a_real_estimate(counted_attention, suffix):
    settings = tiny_recipe()
    cost = acc.analytic_cost(settings, FULL_WINDOWS, batch=2, suffix=suffix)
    assert executed_forward_flops(FULL_WINDOWS, settings, suffix) == predicted_forward_flops(cost)


def test_wrong_suffix_mode_is_caught(counted_attention):
    """Negative control: the cached-suffix prediction must not match a full-replay run."""
    settings = tiny_recipe()
    cached = acc.analytic_cost(settings, FULL_WINDOWS, batch=2, suffix=True)
    assert executed_forward_flops(FULL_WINDOWS, settings, suffix=False) != predicted_forward_flops(cached)


def test_sliding_windows_are_counted_as_windowed(counted_attention):
    """Known convention gap: the masked dense kernel executes every key, accounting counts only the window.

    Pinned so a change to either side is noticed. The difference is exactly the out-of-window keys of the
    short-window blocks, over every forward that runs them.
    """
    settings = tiny_recipe()
    cost = acc.analytic_cost(settings, SLIDING, batch=2, suffix=True)
    executed = executed_forward_flops(SLIDING, settings, suffix=True)
    dense = acc.analytic_cost(settings, FULL_WINDOWS, batch=2, suffix=True)
    assert predicted_forward_flops(cost) < executed == predicted_forward_flops(dense)


def test_suffix_saving_and_draw_bookkeeping():
    settings = recipe('1M', 256)
    config = GPTConfig()
    cached = acc.analytic_cost(settings, config, suffix=True)
    replay = acc.analytic_cost(settings, config, suffix=False)
    assert cached.population_K == population(settings) == 256
    assert cached.draws == replay.draws
    # Only the block-rerunning groups get cheaper; embedding draws always rerun the whole stack.
    for group in ('writers', 'hidden', 'hubs'):
        assert cached.flops_per_group[group] < replay.flops_per_group[group]
    for group in ('embedding', 'head', 'scalars', 'q', 'k', 'v', 'gate', 've'):
        assert cached.flops_per_group[group] == replay.flops_per_group[group]
    saving = acc.suffix_saving(settings, config)
    assert saving == pytest.approx((replay.flops_total - cached.flops_total) / replay.flops_total)
    assert 0.1 < saving < 0.25  # TODO.md records 15.5-17.9% across the shipped recipes
    # The per-GPU figure splits the draws but every GPU runs its own clean pass.
    four = acc.analytic_cost(recipe('1M', 4096), config, world=4)
    one = acc.analytic_cost(recipe('1M', 4096), config, world=1)
    assert four.forward_equivalents == one.forward_equivalents
    assert four.forward_equivalents_per_gpu - 1 == pytest.approx((one.forward_equivalents - 1) / 4)


def test_head_and_local_flops_by_formula():
    config, settings = GPTConfig(), recipe('1M', 256)
    cost = acc.analytic_cost(settings, config, batch=8)
    tokens_ = 8 * config.sequence_len
    assert cost.flops_per_group['head'] == settings.head * acc.head_flops(tokens_, config) * 256 / 4096
    windows = acc.window_sizes(config)
    ve_blocks = [i for i in range(config.n_layer) if has_ve(i, config.n_layer)]
    assert cost.flops_per_group['ve'] == settings.local['ve'] * sum(
        acc.attention_flops(tokens_, windows[i], config) for i in ve_blocks)
    assert cost.draws['q'] == settings.local['q'] * config.n_layer


@pytest.mark.parametrize('pattern,n_layer', [('SSSL', 8), ('L', 3), ('SL', 5), ('S', 2)])
def test_window_sizes_match_the_model(pattern, n_layer):
    config = GPTConfig(n_layer=n_layer, window_pattern=pattern)
    with torch.device('meta'):
        model = model_module.GPT(config)
    assert acc.window_sizes(config) == [w for w, _ in model.window_sizes]


@pytest.mark.parametrize('config', [SLIDING, GPTConfig(), GPTConfig(n_layer=3, n_embd=128, n_head=2, n_kv_head=2)],
                         ids=['tiny', 'base', 'odd'])
def test_parameter_bytes_match_the_model(config):
    """Embeddings in bf16 (as init_weights stores them on CUDA), everything else fp32."""
    with torch.device('meta'):
        model = model_module.GPT(config)
    expected = sum(
        p.numel() * (2 if name == 'transformer.wte.weight' or name.startswith('value_embeds') else 4)
        for name, p in model.named_parameters()
    )
    assert acc._param_bytes(config) == expected
    assert [acc._has_ve(i, config.n_layer) for i in range(config.n_layer)] == \
        [has_ve(i, config.n_layer) for i in range(config.n_layer)]


def test_memory_estimate_scales_with_the_draw_chunk():
    config = GPTConfig()
    small, large = acc.analytic_memory(recipe('1M', 256), config), acc.analytic_memory(recipe('1M', 16384), config)
    assert large['mlp_hidden_bytes'] == 8 * small['mlp_hidden_bytes']  # nrep 16 vs 2
    assert large['params_bytes'] == small['params_bytes']
    assert 0 < small['peak_estimate_gb'] < large['peak_estimate_gb']


def test_percentile_is_nearest_rank():
    values = list(range(11, 0, -1))  # unsorted on purpose
    assert [acc._percentile(values, q) for q in (0, 10, 50, 90, 100)] == [1, 2, 6, 10, 11]
    assert acc._percentile([], 50) == 0.0


def test_run_meters_summary():
    meters = acc.RunMeters(warmup=2)
    for host, gpu in [(1000, 900), (500, 400), (10, 9), (20, 10), (30, 11)]:  # two slow warmup steps
        meters.record_step(host, gpu)
    meters.record_eval(4.0)
    summary = meters.summary(world=2)
    assert summary['steps_kept'] == 3 and summary['warmup_discarded'] == 2
    assert summary['host_median_ms'] == 20 and summary['gpu_median_ms'] == 10
    assert summary['likely_launch_bound'] is True  # median host 20 ms > 1.3 x median GPU 10 ms
    wall = (1000 + 500 + 10 + 20 + 30) / 1000 + 4.0  # every step, warmup included, plus eval
    assert summary['wall_seconds'] == pytest.approx(wall)
    assert summary['gpu_hours'] == pytest.approx(wall * 2 / 3600)


def test_measure_records_host_time_and_eval_separately():
    meters = acc.RunMeters(warmup=0)
    with acc.measure(meters, 'step'):
        time.sleep(0.02)
    with acc.measure(meters, 'eval'):
        time.sleep(0.01)
    assert len(meters.step_host_ms) == 1 and meters.step_host_ms[0] >= 20
    assert len(meters.eval_seconds) == 1 and meters.eval_seconds[0] >= 0.01
    if not torch.cuda.is_available():
        assert meters.step_gpu_ms == [] and meters.summary()['likely_launch_bound'] is None
