"""End-to-end checks of the training loops (dust.train, backprop.train) on the toy setup, on CPU.

Each check compares against an independent truth: a plain loop of DUST.step (the refactor changes
nothing), groups.total_draws and accounting (the logged cost), the step timings themselves (eval is
excluded), an unprobed run (the probe changes nothing), and an uninterrupted run (resume is exact).

Run: python -m pytest new_implementation/test_training.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'baselines'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backprop  # noqa: E402  (imports after the sys.path shim above)
import dust  # noqa: E402
from groups import total_draws  # noqa: E402
from run_metrics import backprop_update_cost, dust_update_cost  # noqa: E402
from toy import TOY_CONFIG, toy_data, toy_recipe  # noqa: E402

CPU = torch.device('cpu')
TIMING = {'train_seconds', 'gpu_hours', 'elapsed_seconds', 'step_host_ms', 'step_gpu_ms', 'peak_memory_gb'}
DATA = toy_data(steps=24)


def run(output, steps=6, eval_interval=2, **kwargs):
    training, validation, testing = DATA
    return dust.train(TOY_CONFIG, toy_recipe(), training, validation, testing, output, steps=steps,
                      eval_interval=eval_interval, seed=42, device=CPU, echo=lambda _: None, **kwargs)


def records(output):
    return [json.loads(line) for line in (Path(output) / 'metrics.jsonl').read_text().splitlines()]


def untimed(rows):
    return [{k: v for k, v in row.items() if k not in TIMING} for row in rows]


def final_state(output):
    return torch.load(Path(output) / 'state.pt', weights_only=False)


def same_weights(a, b):
    return all(torch.equal(a[name], b[name]) for name in a)


def test_train_is_the_plain_step_loop(tmp_path):
    """train() with logging and checkpointing ends at the weights of a bare loop of DUST.step."""
    steps = 4
    run(tmp_path / 'run', steps=steps, checkpoint_interval=steps)
    trained = final_state(tmp_path / 'run')['model']

    def bare_loop(skip=None):
        model = dust.make_model(TOY_CONFIG, CPU, 42)
        settings = toy_recipe()
        optimizer = dust.make_optimizer(model, settings)
        estimator = dust.DUST(model, settings, 42)
        for step in range(1, steps + 1):
            if step == skip:
                continue
            batch = DATA[0][(step - 1) * 8:step * 8]
            estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        estimator.close()
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert same_weights(trained, bare_loop())
    assert not same_weights(trained, bare_loop(skip=2)), 'negative control: a skipped batch must differ'


def test_dust_learns_the_toy_task(tmp_path):
    result = run(tmp_path / 'run', steps=20, eval_interval=5)
    assert result['history'][0]['val_loss'] > 8.0      # uniform over 4096 tokens: ln 4096 = 8.32
    assert result['best_val'] < 4.0                     # below even the unigram floor ln 64 = 4.16
    assert result['test_at_best_val'] < 4.0


def test_backprop_learns_and_logs_on_the_same_axis(tmp_path):
    training, validation, testing = DATA
    result = backprop.train(TOY_CONFIG, training, validation, testing, tmp_path / 'bp', steps=10,
                            eval_interval=5, lr=0.175, momentum=0.98, device=CPU, echo=lambda _: None)
    assert result['best_val'] < 4.0
    rows = records(tmp_path / 'bp')
    per_update = backprop_update_cost(TOY_CONFIG)['flops']
    assert [r['flops'] for r in rows] == [per_update * r['step'] for r in rows]
    assert rows[-1]['forward_equivalents'] == 30.0


def test_logged_cost_and_time_add_up(tmp_path, monkeypatch):
    """Cumulative cost = steps x one update's analytic cost; eval time is excluded from train_seconds."""
    evaluate = dust.evaluate
    pause = 0.25

    def slow_evaluate(*args, **kwargs):
        time.sleep(pause)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(dust, 'evaluate', slow_evaluate)
    run(tmp_path / 'run', steps=4, eval_interval=2)
    rows = records(tmp_path / 'run')
    settings = toy_recipe()
    cost = dust_update_cost(settings, TOY_CONFIG, batch=8)

    assert [r['step'] for r in rows] == [0, 1, 2, 3, 4]
    for row in rows[1:]:
        assert row['flops'] == pytest.approx(row['step'] * cost['flops'])
        assert row['forward_equivalents'] == pytest.approx(row['step'] * cost['forward_equivalents'])
        assert row['draws'] == row['step'] * total_draws(settings)  # from groups, independent of accounting
        assert row['population'] == 34
    assert 'allocation' in rows[1] and all('allocation' not in r for r in rows[2:])
    assert [('val_loss' in r) for r in rows] == [True, True, True, False, True]

    step_seconds = sum(r['step_host_ms'] for r in rows[1:]) / 1000
    assert rows[-1]['train_seconds'] == pytest.approx(step_seconds)
    assert rows[-1]['gpu_hours'] == pytest.approx(step_seconds / 3600)
    # Four evaluations (steps 0, 1, 2, 4) each slept `pause`; the wall clock sees them, training time does not.
    timing = json.loads((tmp_path / 'run' / 'result.json').read_text())['timing']
    assert timing['totals']['eval_seconds'] >= 4 * pause
    assert timing['elapsed_seconds'] >= step_seconds + 4 * pause


def test_probe_reports_cosines_and_leaves_training_alone(tmp_path):
    run(tmp_path / 'plain', steps=4, checkpoint_interval=4)
    run(tmp_path / 'probed', steps=4, checkpoint_interval=4, probe_interval=2, probe_seeds=(0, 1))
    plain, probed = final_state(tmp_path / 'plain'), final_state(tmp_path / 'probed')
    assert same_weights(plain['model'], probed['model'])
    assert torch.equal(plain['generator'], probed['generator'])

    rows = records(tmp_path / 'probed')
    probes = {r['step']: r['probe'] for r in rows if 'probe' in r}
    assert set(probes) == {2, 4}
    for probe in probes.values():
        assert -1 <= probe['min_group_cosine'] <= probe['mean_group_cosine'] <= 1
        assert probe['seeds'] == 2
        assert {'est_variance', 'bias', 'cosine_mean'} <= set(probe['per_group']['head'])


def test_an_estimate_consumes_the_generator():
    """Why the probe must restore the generator: without it, probing would shift every later draw."""
    model = dust.make_model(TOY_CONFIG, CPU, 42)
    estimator = dust.DUST(model, toy_recipe(), 42)
    before = estimator.generator.get_state()
    batch = DATA[1][:8]
    estimator.estimate(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
    estimator.close()
    assert not torch.equal(before, estimator.generator.get_state())


def test_resume_after_a_crash_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    run(tmp_path / 'straight', steps=6, checkpoint_interval=3)

    evaluate, calls = dust.evaluate, []

    def crash_at_step_six(*args, **kwargs):  # evaluations run at steps 0, 1, 2, 4, 6 (interval 2), so the
        calls.append(1)                       # 5th call is step 6: trained, unrecorded, past the checkpoint
        if len(calls) == 5:
            raise KeyboardInterrupt('simulated preemption')
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(dust, 'evaluate', crash_at_step_six)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / 'crashed', steps=6, checkpoint_interval=3)
    monkeypatch.setattr(dust, 'evaluate', evaluate)
    assert [r['step'] for r in records(tmp_path / 'crashed')] == [0, 1, 2, 3, 4, 5]  # past the checkpoint

    result = run(tmp_path / 'crashed', steps=6, checkpoint_interval=3, resume=True)
    straight = json.loads((tmp_path / 'straight' / 'result.json').read_text())

    assert untimed(records(tmp_path / 'crashed')) == untimed(records(tmp_path / 'straight'))
    assert same_weights(final_state(tmp_path / 'crashed')['model'], final_state(tmp_path / 'straight')['model'])
    assert (result['best_val'], result['best_step'], result['test_at_best_val']) == \
        (straight['best_val'], straight['best_step'], straight['test_at_best_val'])
    assert untimed(result['history']) == untimed(straight['history'])


def test_resume_refuses_a_different_run(tmp_path):
    run(tmp_path / 'run', steps=2, checkpoint_interval=2)
    training, validation, testing = DATA
    with pytest.raises(ValueError, match='seed'):
        dust.train(TOY_CONFIG, toy_recipe(), training, validation, testing, tmp_path / 'run', steps=4,
                   eval_interval=2, seed=7, device=CPU, resume=True, echo=lambda _: None)
    with pytest.raises(ValueError, match='recipe'):
        dust.train(TOY_CONFIG, toy_recipe(lr=0.1), training, validation, testing, tmp_path / 'run', steps=4,
                   eval_interval=2, device=CPU, resume=True, echo=lambda _: None)
    with pytest.raises(FileNotFoundError):
        run(tmp_path / 'never-saved', steps=2, resume=True)
    with pytest.raises(FileExistsError):
        run(tmp_path / 'run', steps=2)  # a fresh run must not overwrite an existing one


def cli(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_command_lines(tmp_path):
    first = cli('dust.py', '--toy', '--device', 'cpu', '--steps', '2', '--checkpoint-interval', '1',
                '--output', str(tmp_path / 'dust'))
    assert first.returncode == 0, first.stderr
    resumed = cli('dust.py', '--toy', '--device', 'cpu', '--steps', '3', '--checkpoint-interval', '1',
                  '--resume', '--output', str(tmp_path / 'dust'))
    assert resumed.returncode == 0, resumed.stderr
    assert [r['step'] for r in records(tmp_path / 'dust')] == [0, 1, 2, 3]
    assert json.loads((tmp_path / 'dust' / 'config.json').read_text())['execution'] == 'cached-suffix'

    baseline = cli('baselines/backprop.py', '--toy', '--device', 'cpu', '--steps', '2',
                   '--output', str(tmp_path / 'bp'))
    assert baseline.returncode == 0, baseline.stderr
    assert (tmp_path / 'bp' / 'result.json').is_file()
