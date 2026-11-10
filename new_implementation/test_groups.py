"""Checks for groups.py against the real model and the real recipes, on CPU.

groups.py is torch-free and kept in sync with model.py / dust.py by hand, so every test here compares
it to the live code: the parameter names a real GPT registers, model.has_ve, and the draws DUST.estimate
actually runs.

Run: python -m pytest new_implementation/test_groups.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import groups  # noqa: E402  (imports after the sys.path shim above)
from dust import DUST, DRAW_CHUNK_SIZES, Recipe, make_model, recipe  # noqa: E402
from model import GPT, GPTConfig, has_ve  # noqa: E402

TINY = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
CONFIGS = [TINY, GPTConfig(), GPTConfig(n_layer=3, n_embd=64, n_head=4, n_kv_head=4, sequence_len=64)]


def tiny_recipe():
    return Recipe(
        writers=[6, 4, 2, 2], hidden=[4, 4, 2, 2], hubs=[2, 2, 2, 4], embedding=2, head=32,
        local=dict(q=2, v=4, gate=2, k=6, ve=2), nrep=2, lr=0.1, momentum=0.9,
    )


@pytest.mark.parametrize('config', CONFIGS, ids=['tiny', 'base', 'odd-depth'])
def test_every_parameter_has_a_group(config):
    """Every parameter a real GPT registers maps to a draw group, and every draw group except the
    virtual attention-output hubs owns at least one parameter."""
    with torch.device('meta'):  # names only; no weights needed
        model = GPT(config)
    owned = {groups.param_group(name) for name, _ in model.named_parameters()}
    assert owned == set(groups.DRAW_GROUPS) - {'hubs'}


def test_unknown_names_are_rejected():
    for name in ('transformer.h.0.attn.c_q.bias', 'lm_head', 'something.weight'):
        with pytest.raises(KeyError):
            groups.param_group(name)


def test_layer_of_and_suffix_start():
    assert groups.layer_of('transformer.h.11.mlp.c_fc.weight') == 11
    assert groups.layer_of('value_embeds.3.weight') == 3
    assert groups.layer_of('o@5') == 5
    assert groups.layer_of('lm_head.weight') is None
    assert groups.suffix_start_block('transformer.wte', 8) == 0
    assert groups.suffix_start_block('lm_head', 8) == 8
    assert groups.suffix_start_block('transformer.h.6.attn.c_proj', 8) == 6
    with pytest.raises(KeyError):
        groups.suffix_start_block('resid_lambdas', 8)


@pytest.mark.parametrize('n_layer', [1, 2, 3, 4, 8, 16])
def test_value_embedding_layers_match_the_model(n_layer):
    assert [groups.has_value_embedding(i, n_layer) for i in range(n_layer)] == \
        [has_ve(i, n_layer) for i in range(n_layer)]


@pytest.mark.parametrize('tokens', ['1M', '10M'])
@pytest.mark.parametrize('population', sorted(DRAW_CHUNK_SIZES))
def test_population_is_the_direct_draw_count(tokens, population):
    """The README's population K is exactly writers + hidden + hubs + embedding for every shipped recipe."""
    assert groups.population(recipe(tokens, population)) == population


def test_local_totals_are_per_block_times_blocks():
    settings = tiny_recipe()
    with pytest.raises(ValueError, match='n_layer'):
        groups.group_draw_count(settings, 'k')
    assert groups.group_draw_count(settings, 'k', layer=2) == 6
    assert groups.group_draw_count(settings, 'k', n_layer=4) == 6 * 4
    assert groups.group_draw_count(settings, 've', n_layer=4) == 2 * 2  # value embeddings on blocks 1, 3


def count_executed_draws(settings, config):
    """Run a real estimate and count, per attention-internal kind, the draws local_errors samples."""
    model = make_model(config, torch.device('cpu'), seed=0)
    estimator = DUST(model, settings)
    counts, current = {}, {}
    sample, score = estimator.noise, estimator.local_errors

    def noise(*shape, dtype=None):
        if 'kind' in current:  # local_errors samples [nrep, batch, tokens, width] per block per chunk
            counts[current['kind']] = counts.get(current['kind'], 0) + shape[0]
        return sample(*shape, dtype=dtype)

    def local_errors(kind, draws, targets, **kwargs):
        current['kind'] = kind
        try:
            return score(kind, draws, targets, **kwargs)
        finally:
            current.clear()

    estimator.noise, estimator.local_errors = noise, local_errors
    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_len + 1),
                           generator=torch.Generator().manual_seed(0))
    try:
        estimator.estimate(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
    finally:
        estimator.close()
    return counts


def test_local_totals_match_what_estimate_runs():
    """The whole-model attention-internal counts equal the draws DUST.local_errors samples."""
    settings = tiny_recipe()
    executed = count_executed_draws(settings, TINY)
    for group, count in executed.items():
        assert count == groups.group_draw_count(settings, group, n_layer=TINY.n_layer), group


def test_total_draws():
    settings = tiny_recipe()
    local = (2 + 4 + 6) * 4 + (2 + 2) * 2  # q, v, k on 4 blocks; gate, ve on the 2 value-embedding blocks
    expected = groups.population(settings) + settings.head + local + 8
    assert groups.total_draws(settings) == expected
    assert groups.total_draws(settings) > groups.population(settings)
