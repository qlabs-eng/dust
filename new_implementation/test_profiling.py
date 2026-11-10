"""Checks for profiling.py, on CPU: the harness runs end to end on a tiny model, and the double-batch
test gives the right verdict for steps whose cost is known.

GPU-only numbers (CUDA time, kernel gaps) cannot be checked here; these tests prove the plumbing.

Run: python -m pytest new_implementation/test_profiling.py
"""
import json
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import profiling  # noqa: E402  (imports after the sys.path shim above)
from dust import DUST, Recipe, make_model, make_optimizer  # noqa: E402
from model import GPTConfig  # noqa: E402

CONFIG = GPTConfig(sequence_len=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=32)
PHASES = ('dust/clean', 'dust/direct', 'dust/embedding', 'dust/hubs', 'dust/head', 'dust/local', 'dust/scalars')


def tiny_recipe():
    return Recipe(
        writers=[2, 2], hidden=[2, 2], hubs=[2, 2], embedding=2, head=32,
        local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=0.1, momentum=0.9,
    )


def batch(seed=0):
    rows = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.sequence_len + 1),
                         generator=torch.Generator().manual_seed(seed))
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


@pytest.fixture
def estimator():
    est = DUST(make_model(CONFIG, torch.device('cpu'), seed=0), tiny_recipe())
    yield est
    est.close()


def test_profile_estimate_labels_every_phase(estimator, tmp_path):
    trace = tmp_path / 'trace.json'
    table, prof = profiling.profile_estimate(
        estimator, *batch(), warmup=1, active=1, sort_by='self_cpu_time_total', trace_path=trace)
    names = {event.key for event in prof.key_averages()}
    assert set(PHASES) <= names
    assert isinstance(table, str) and 'Self CPU' in table
    events = json.loads(trace.read_text())['traceEvents']
    assert {e['name'] for e in events} >= set(PHASES)


def test_profile_training_stops_after_its_schedule(estimator, tmp_path):
    optimizer = make_optimizer(estimator.model, estimator.settings)
    calls = []

    def step(data):
        calls.append(1)
        estimator.step(*data, optimizer)

    trace = tmp_path / 'sub' / 'train.json'  # the harness creates the parent
    profiling.profile_training(step, (batch(i) for i in range(100)), skip_first=1, wait=1, warmup=1,
                               active=1, sort_by='self_cpu_time_total', trace_path=trace)
    assert len(calls) == 4  # skip_first + wait + warmup + active, not the whole iterator
    assert trace.is_file()


@pytest.mark.parametrize('cost,verdict', [
    (lambda size: 0.004, 'launch-bound'),          # time does not depend on the data
    (lambda size: 0.002 * size, 'gpu-bound'),     # time doubles with the data
])
def test_launch_bound_verdict(cost, verdict):
    result = profiling.launch_bound_test(lambda data: time.sleep(cost(len(data))), [0] * 2, [0] * 4,
                                         repeats=5, warmup=1)
    assert result['verdict'] == verdict
    assert result['step_ms_2x'] / result['step_ms_1x'] == pytest.approx(result['ratio'])
