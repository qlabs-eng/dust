"""Checks for gradient_check.py, on CPU: first the ruler on known answers, then the ruler on DUST.

Known answers: compare() on scaled and negated copies, reference_gradients() against central finite
differences in float64, fit_cosine_law() on points generated from a known law. Then end to end on a
tiny model: DUST's estimate must point toward backprop and get closer as the population grows, the
paper's central claim (Figure 5), measured here through cosine_curve. Seeds are fixed, so the
end-to-end test is deterministic rather than flaky; its margins come from a measured run.

Run: python -m pytest new_implementation/test_gradient_check.py
"""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradient_check as gc  # noqa: E402  (imports after the sys.path shim above)
from dust import DUST, Recipe, make_model  # noqa: E402
from model import GPTConfig  # noqa: E402

CONFIG = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)


def tiny_recipe(scale=1):
    """Every direct and local group gets 2 * scale draws per block; the head 32 * scale (16 slabs, nrep 2)."""
    local = {kind: 2 * scale for kind in ('q', 'v', 'gate', 'k', 've')}
    return Recipe(writers=[2 * scale] * 4, hidden=[2 * scale] * 4, hubs=[2 * scale] * 4, embedding=2 * scale,
                  head=32 * scale, local=local, nrep=2, lr=0.1, momentum=0.9)


@pytest.fixture(scope='module')
def model():
    return make_model(CONFIG, torch.device('cpu'), seed=0)


@pytest.fixture(scope='module')
def data():
    tokens = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.sequence_len + 1), generator=torch.Generator().manual_seed(0))
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


@pytest.fixture(scope='module')
def reference(model, data):
    return gc.reference_gradients(model, *data)[0]


@pytest.fixture
def estimator(model):
    estimator = DUST(model, tiny_recipe())
    yield estimator
    estimator.close()


def test_compare_on_known_answers(reference):
    same = gc.compare(reference, reference, per_parameter=True)
    doubled = gc.compare(reference, {n: 2 * g for n, g in reference.items()})
    negated = gc.compare(reference, {n: -g for n, g in reference.items()})
    for group in same['per_group']:  # fp32 cosine of long vectors rounds to within ~1e-5 of 1
        assert same['per_group'][group]['cosine'] == pytest.approx(1, abs=1e-4)
        assert same['per_group'][group]['rel_norm'] == pytest.approx(1, abs=1e-6)
        assert doubled['per_group'][group]['rel_norm'] == pytest.approx(2, abs=1e-5)
        assert doubled['per_group'][group]['cosine'] == pytest.approx(1, abs=1e-4)
        assert negated['per_group'][group]['cosine'] == pytest.approx(-1, abs=1e-4)
    assert same['overall']['mean_group_cosine'] == pytest.approx(1, abs=1e-4)
    assert set(same['per_parameter']) == set(reference)
    # every parameter lands in exactly one group, and all eleven DUST groups appear
    assert set(same['per_group']) == {'embedding', 'head', 'q', 'k', 'v', 'gate', 'writers', 'hidden', 've', 'scalars'}


def test_reference_is_the_true_mean_loss_gradient(model, data):
    """A central finite difference along each parameter tensor's own gradient direction u = g / |g|.

    If g is the gradient, the loss slope along u is |g|; a wrong direction or scale g' gives slope
    g.g'/|g'|, which equals |g'| only when g' = g. Moving a whole tensor gives a loss change far above
    the fp32 rounding of GPT.forward's loss (its logits are cast to fp32 whatever the weights' dtype),
    which single-coordinate differences could not resolve. Measured error at step 0.1: under 1% for
    every kind here except the value embeddings (2%). ve_gate is left out: its gradient (~3e-5) moves
    the loss by about one fp32 rounding step.
    """
    copy_ = copy.deepcopy(model)
    x, y = data
    gradients, loss = gc.reference_gradients(copy_, x, y)
    assert loss == pytest.approx(copy_(x, y).item())
    parameters = dict(copy_.named_parameters())
    eps = 0.1
    with torch.no_grad():
        for name in ('lm_head.weight', 'transformer.wte.weight', 'transformer.h.0.mlp.c_fc.weight',
                     'transformer.h.1.attn.c_proj.weight', 'transformer.h.2.attn.c_k.weight',
                     'value_embeds.1.weight', 'resid_lambdas'):
            parameter, norm = parameters[name], gradients[name].norm()
            original, direction = parameter.clone(), gradients[name] / norm
            parameter.copy_(original + eps * direction)
            up = copy_(x, y).item()
            parameter.copy_(original - eps * direction)
            down = copy_(x, y).item()
            parameter.copy_(original)
            assert (up - down) / (2 * eps) == pytest.approx(norm.item(), rel=3e-2), name
    # the same check must reject a wrong gradient: half the true one claims half the slope
    parameter, half = parameters['lm_head.weight'], gradients['lm_head.weight'] / 2
    with torch.no_grad():
        original = parameter.clone()
        parameter.copy_(original + eps * half / half.norm())
        up = copy_(x, y).item()
        parameter.copy_(original - eps * half / half.norm())
        down = copy_(x, y).item()
        parameter.copy_(original)
    assert (up - down) / (2 * eps) != pytest.approx(half.norm().item(), rel=3e-2)


def test_reference_leaves_the_model_as_it_found_it(model, data):
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    weights = {n: v.clone() for n, v in model.state_dict().items()}
    gc.reference_gradients(model, *data)
    assert {n: p.requires_grad for n, p in model.named_parameters()} == before
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(weights[n], v) for n, v in model.state_dict().items())


def test_estimate_is_seeded_and_never_touches_the_weights(model, data, estimator):
    weights = {n: v.clone() for n, v in model.state_dict().items()}
    first, loss = gc.estimate_gradients(estimator, *data, seed=0)
    again, _ = gc.estimate_gradients(estimator, *data, seed=0)
    other, _ = gc.estimate_gradients(estimator, *data, seed=1)
    assert all(torch.equal(weights[n], v) for n, v in model.state_dict().items())
    assert set(first) == {n for n, _ in model.named_parameters()}
    assert all(torch.equal(first[n], again[n]) for n in first)
    assert not torch.equal(first['lm_head.weight'], other['lm_head.weight'])
    assert loss == pytest.approx(model(*data).item(), rel=1e-5)


def test_estimate_statistics(data, estimator, reference):
    repeated = gc.estimate_statistics(estimator, *data, seeds=[0, 0])
    assert all(stats['est_variance'] == 0 for stats in repeated.values())
    varied = gc.estimate_statistics(estimator, *data, seeds=[0, 1, 2], reference=reference)
    for group, stats in varied.items():
        assert stats['est_variance'] > 0, group
        assert stats['bias'] > 0 and stats['cosine_std'] >= 0
        assert -1 <= stats['cosine_mean'] <= 1 and -1 <= stats['mean_cosine_to_ref'] <= 1
    with pytest.raises(ValueError, match='at least 2 seeds'):
        gc.estimate_statistics(estimator, *data, seeds=[0])


def test_fit_cosine_law_recovers_a_known_law():
    populations = [64, 256, 1024, 4096, 16384, 65536, 131072]
    c_max, c = 0.8779, 31623  # Appendix G, MLP out at 100M tokens
    cosines = [c_max / math.sqrt(1 + c / k) for k in populations]
    fit = gc.fit_cosine_law(populations, cosines)
    assert fit['c_max'] == pytest.approx(c_max, rel=1e-6)
    assert fit['c'] == pytest.approx(c, rel=1e-6)
    assert fit['rmse'] < 1e-9 and 'note' not in fit

    dropped = gc.fit_cosine_law([16] + populations, [-0.01] + cosines)
    assert dropped['c'] == pytest.approx(c, rel=1e-6) and 'dropped 1' in dropped['note']
    too_few = gc.fit_cosine_law([64, 256], [0.1, -0.2])
    assert math.isnan(too_few['c_max']) and 'need >=2' in too_few['note']
    # cosine rising faster than the law allows puts 1/c_max^2 below zero; no complex c_max may leak out
    steep = gc.fit_cosine_law([1, 8], [0.01, 0.5])
    assert math.isnan(steep['c_max']) and 'no positive ceiling' in steep['note']


def test_estimate_has_the_expected_scale(model, data, reference):
    """Cosine ignores scale, so this pins estimate_gradients' mean-loss normalization.

    Rewards are centered on a mean that includes the draw itself, so the estimate's expectation is
    (1 - 1/n) times the gradient. The head centers over the batch (n = 2 sequences): its projection
    onto backprop, (e . g) / (g . g), must average 1/2; measured 0.506 +- 0.008 over 6 seeds. Direct
    groups shrink a little further under sigma = 0.2 smoothing (measured writers 0.43, hidden 0.44).
    """
    estimator = DUST(model, tiny_recipe(8))
    try:
        samples = [gc.group_vectors(gc.estimate_gradients(estimator, *data, seed=s)[0]) for s in range(4)]
    finally:
        estimator.close()
    ref = gc.group_vectors(reference)
    projection = {g: sum((s[g] @ ref[g]).item() for s in samples) / len(samples) / (ref[g] @ ref[g]).item()
                  for g in ('head', 'writers', 'hidden')}
    assert projection['head'] == pytest.approx(0.5, abs=0.05)
    assert 0.3 < projection['writers'] < 0.55 and 0.3 < projection['hidden'] < 0.55, projection


def test_estimate_improves_with_population(model, data, reference):
    """The paper's Figure 5 in miniature, measured with cosine_curve at 1x and 8x the tiny recipe.

    Measured on this model and batch (3 seeds): head 0.07 -> 0.18, writers 0.06 -> 0.18, hidden
    0.06 -> 0.15, embedding 0.08 -> 0.23. Asserted with margin, on the groups with the strongest signal.
    """
    curve = gc.cosine_curve(lambda scale: DUST(model, tiny_recipe(scale)), [1, 8], *data, reference, seeds=(0, 1, 2))
    for group in ('head', 'writers', 'hidden', 'embedding'):
        small, large = curve[group]['cosines']
        assert large > 0.1, f'{group}: cosine {large:.3f} at 8x is not clearly toward backprop'
        assert large > small + 0.05, f'{group}: cosine did not rise with population ({small:.3f} -> {large:.3f})'
        fit = curve[group]['fit']
        assert isinstance(fit['c_max'], float) and (fit['c_max'] > 0 or 'note' in fit)
