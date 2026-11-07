"""Checks for roofline.py, all on CPU.

The strongest one runs a real DUST.estimate on a tiny model under torch's FlopCounterMode and requires
the model's matmul FLOPs to match what PyTorch actually executed, exactly: that pins the op inventory
(draw-chunk counts, row counts, which matmuls each phase runs) to dust.py. Bytes and kernel counts
cannot be measured on CPU; they are checked by hand on single ops here and by a GPU profile later.

Run: python -m pytest new_implementation/test_roofline.py
"""
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import roofline as rl  # noqa: E402  (imports after the sys.path shim above)
from accounting import attention_flops, block_flops, head_flops, window_sizes  # noqa: E402
from dust import DUST, Recipe, make_model, recipe  # noqa: E402
from model import GPTConfig  # noqa: E402

TINY = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
FULL = GPTConfig()


def tiny_recipe():
    return Recipe(
        writers=[6, 4, 2, 2], hidden=[4, 4, 2, 2], hubs=[2, 2, 2, 4], embedding=2, head=32,
        local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=0.1, momentum=0.9,
    )


def matmul_flops(ledgers):
    return sum(op.flops for ledger in ledgers.values() for op in ledger.ops.values() if op.kind == 'matmul')


SUFFIX_MODES = [False] + ([True] if 'suffix' in inspect.signature(DUST).parameters else [])


@pytest.mark.parametrize('suffix', SUFFIX_MODES)
def test_matmul_flops_match_a_real_estimate(suffix):
    """CPU SDPA does not report FLOPs, so attention is excluded on both sides; every mm/bmm must match."""
    model = make_model(TINY, torch.device('cpu'), seed=0)
    tokens = torch.randint(0, TINY.vocab_size, (2, TINY.sequence_len + 1), generator=torch.Generator().manual_seed(0))
    kwargs = dict(suffix=suffix) if 'suffix' in inspect.signature(DUST).parameters else {}
    estimator = DUST(model, tiny_recipe(), **kwargs)
    try:
        with FlopCounterMode(display=False) as counter:
            estimator.estimate(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
    finally:
        estimator.close()
    measured = counter.get_total_flops()
    assert matmul_flops(rl.update_ops(tiny_recipe(), TINY, batch=2, suffix=suffix)) == measured


def test_forward_matmuls_and_attention_match_accounting():
    """Same FLOP conventions as accounting.py: projections + MLP + head exactly (plus the tiny ve_gate),
    and dense attention equals accounting's count on the full-window blocks, more on the windowed ones."""
    tokens = 8 * FULL.sequence_len
    ops = rl._forward(FULL, tokens)
    windows = window_sizes(FULL)
    projections = sum(block_flops(tokens, w, FULL) - attention_flops(tokens, w, FULL) for w in windows)
    gates = sum(op.flops for op in ops if op.name == 'attn.ve_gate matmul')
    assert sum(op.flops for op in ops if op.kind == 'matmul') - gates == projections + head_flops(tokens, FULL)
    dense = [op.flops for op in ops if op.kind == 'attention']
    for flops, window in zip(dense, windows):
        expected = attention_flops(tokens, window, FULL)
        assert flops == expected if window >= FULL.sequence_len else flops > expected


def test_single_ops_by_hand():
    rows, n = 32768, 512  # one K=256 draw chunk: nrep 2 * batch 8 * 2048 tokens
    cast, matmul = rl._linear('mlp.c_fc', rows, n, 4 * n)
    assert matmul.flops == 2 * rows * n * 4 * n
    assert matmul.nbytes == (rows * n + n * 4 * n + rows * 4 * n) * 2
    assert cast.nbytes == n * 4 * n * (4 + 2)
    # ~400 FLOP/byte: right of the A100's ~200 bf16 ridge, so the GEMM is compute-bound
    assert matmul.flops / matmul.nbytes > rl.A100_40GB.ridge('bf16') > 150
    noise = rl._ew('noise', rows * n, 2)
    assert noise.nbytes == rows * n * 2 and noise.flops / noise.nbytes < 1


def test_suffix_only_skips_blocks_before_the_start():
    tokens = 8 * FULL.sequence_len
    full = rl._forward(FULL, tokens)
    from_zero = rl._forward(FULL, tokens, start=0)
    # forward_from(0) skips only the token embedding lookup and its norm
    skipped = sum(op.nbytes for op in full) - sum(op.nbytes for op in from_zero)
    assert skipped == tokens * FULL.n_embd * 2 * 2 * 2
    flops = [sum(op.flops for op in rl._forward(FULL, tokens, start=s)) for s in range(FULL.n_layer)]
    assert flops == sorted(flops, reverse=True) and len(set(flops)) == FULL.n_layer
    cached = rl.roofline(recipe('1M', 256), FULL, suffix=True)['total']
    replay = rl.roofline(recipe('1M', 256), FULL, suffix=False)['total']
    assert cached.flops < replay.flops and cached.roofline_s < replay.roofline_s


def test_draw_counts_and_world_split():
    settings = recipe('1M', 4096)  # the paper runs it on 4 GPUs
    chunks = sum(w // settings.nrep * 2 + h // settings.nrep for w, h in zip(settings.writers, settings.hidden))
    one = rl.update_ops(settings, FULL, world=1)
    assert one['direct'].ops['noise'].kernels == chunks  # one randn per site per chunk
    four = rl.update_ops(settings, FULL, world=4)
    assert four['direct'].ops['noise'].kernels * 4 == chunks
    assert four['clean'].ops['lm_head matmul'].flops == one['clean'].ops['lm_head matmul'].flops
    with pytest.raises(ValueError):  # 1024 draws on 2 GPUs leave per-block counts dust.py rejects
        rl.update_ops(recipe('1M', 1024), FULL, world=2)


def test_host_syncs():
    """8 scalar rewards ([y != -1] indexing), the train-loss readout (2), one finite check per parameter."""
    ledgers = rl.update_ops(recipe('1M', 256), FULL)
    syncs = sum(op.syncs for ledger in ledgers.values() for op in ledger.ops.values())
    assert syncs == 8 + 2 + len(rl._parameters(FULL))


def test_parameter_count_matches_the_model():
    model = make_model(TINY, torch.device('cpu'), seed=0)
    assert sum(e for e, _ in rl._parameters(TINY)) == sum(p.numel() for p in model.parameters())
    assert len(rl._parameters(TINY)) == len(list(model.parameters()))


def test_math_attention_moves_more_bytes_for_the_same_flops():
    efficient = rl.roofline(recipe('1M', 256), FULL, attention='efficient')['total']
    math = rl.roofline(recipe('1M', 256), FULL, attention='math')['total']
    assert math.nbytes > 2 * efficient.nbytes
    assert abs(math.flops - efficient.flops) / efficient.flops < 0.01  # mask add + softmax are ~1 FLOP/elt


def test_roofline_invariants_and_report():
    report = rl.roofline(recipe('1M', 256), FULL)
    for phase in [*report['phases'].values(), report['total']]:
        assert max(phase.compute_s, phase.memory_s) <= phase.roofline_s * (1 + 1e-9)
        assert phase.roofline_s <= (phase.compute_s + phase.memory_s) * (1 + 1e-9)
        assert phase.predicted_s == max(phase.roofline_s, phase.launch_s)
        assert phase.regime in ('launch', 'bandwidth', 'compute')
    assert report['phases'].keys() == set(rl.PHASES)
    assert 0 < report['predicted_mfu'] < 1 and 0 < report['weight_byte_share'] < 1
    assert 'total' in rl.format_report(report)
    json.dumps({**report, 'phases': {k: asdict(v) for k, v in report['phases'].items()},
                'total': asdict(report['total'])})


def test_regime_classification():
    assert rl._regime(roofline_s=1.0, launch_s=2.0, bandwidth_s=0.0) == 'launch'
    assert rl._regime(roofline_s=1.0, launch_s=0.1, bandwidth_s=0.8) == 'bandwidth'
    assert rl._regime(roofline_s=1.0, launch_s=0.1, bandwidth_s=0.2) == 'compute'


def test_mfu_and_amdahl():
    assert rl.mfu(312e12, 1.0) == pytest.approx(1.0)
    assert rl.mfu(156e12, 2.0) == pytest.approx(0.25)
    assert rl.amdahl({'a': 0.5, 'b': 0.5}, {'a': 2}) == pytest.approx(1 / 0.75)
    assert rl.amdahl({'a': 0.9, 'b': 0.1}, {'a': float('inf')}) == pytest.approx(10)
    assert rl.amdahl_limit(0.9) == pytest.approx(10)
    with pytest.raises(ValueError):
        rl.amdahl({'a': 0.5}, {})
