"""Shared module-group taxonomy for DUST.

Single source of truth, imported by the diagnostics harness, the compute accounting,
and the adaptive-allocation policy so they all speak the same group names. Nothing here
depends on torch or on a live model; it operates on parameter names and the Recipe.

Two related groupings:

- Parameter groups: the group whose draws estimate a given parameter's gradient. Diagnostics
  key cosine / variance / bias by these (use ``param_group``).
- Draw groups: the units a Recipe allocates draws to, i.e. what adaptive allocation moves
  budget between (``DRAW_GROUPS``). A draw group may cover several parameters (``writers`` =
  each block's ``attn.c_proj`` and ``mlp.c_proj``) or a virtual site that is not a parameter
  (``hubs`` = the attention output ``o@L``, which supplies the target error the q/k/v/gate/ve
  gradients are scored against).
"""
import re

# Draw groups, in the order DUST.estimate evaluates them.
DRAW_GROUPS = (
    'writers', 'hidden', 'hubs', 'embedding', 'head',
    'q', 'v', 'gate', 'k', 've', 'scalars',
)

# Groups whose count is per-block (a list in the Recipe) vs a single total.
PER_BLOCK_GROUPS = frozenset({'writers', 'hidden', 'hubs', 'q', 'v', 'gate', 'k', 've'})
SCALAR_GROUPS = frozenset({'embedding', 'head', 'scalars'})

# The five attention-internal groups share Recipe.local; the rest name their own Recipe field.
_LOCAL_GROUPS = frozenset({'q', 'v', 'gate', 'k', 've'})

# Noise scale each group is perturbed with (DUST.estimate / *_errors defaults).
# hubs uses 0.2 on the first half of the blocks and 0.4 on the second half.
NOISE_SIGMA = {
    'writers': 0.2, 'hidden': 0.2, 'embedding': 0.2,
    'hubs': (0.2, 0.4), 'head': 0.05,
    'q': 0.05, 'v': 0.05, 'gate': 0.05, 'k': 0.05, 've': 0.05,
    'scalars': 0.03,
}

_PARAM_GROUP_PATTERNS = (
    (r'transformer\.wte\.weight', 'embedding'),
    (r'lm_head\.weight', 'head'),
    (r'transformer\.h\.\d+\.attn\.c_q\.weight', 'q'),
    (r'transformer\.h\.\d+\.attn\.c_k\.weight', 'k'),
    (r'transformer\.h\.\d+\.attn\.c_v\.weight', 'v'),
    (r'transformer\.h\.\d+\.attn\.ve_gate\.weight', 'gate'),
    (r'transformer\.h\.\d+\.attn\.c_proj\.weight', 'writers'),
    (r'transformer\.h\.\d+\.mlp\.c_fc\.weight', 'hidden'),
    (r'transformer\.h\.\d+\.mlp\.c_proj\.weight', 'writers'),
    (r'value_embeds\.\d+\.weight', 've'),
    (r'(?:resid|x0)_lambdas', 'scalars'),
)
_PARAM_GROUP_RE = [(re.compile(f'^{p}$'), g) for p, g in _PARAM_GROUP_PATTERNS]


def param_group(name):
    """The draw group whose estimate updates parameter ``name`` (e.g. 'transformer.h.3.mlp.c_fc.weight' -> 'hidden')."""
    for pattern, group in _PARAM_GROUP_RE:
        if pattern.match(name):
            return group
    raise KeyError(f'No draw group for parameter {name!r}')


def layer_of(name):
    """Block index embedded in a parameter name or draw site, or None for model-global tensors."""
    match = re.search(r'(?:\.h\.|value_embeds\.|o@)(\d+)', name)
    return int(match.group(1)) if match else None


def suffix_start_block(site, n_layer):
    """First transformer block a direct perturbation at ``site`` changes.

    Blocks before it are unaffected and need not be recomputed (cached-suffix execution,
    Track 1). ``transformer.wte`` -> 0 (whole network); ``lm_head`` -> n_layer (only the
    logits and cross-entropy recompute, no block runs). ``site`` is a draw site as passed to
    direct_errors: a module name, or the virtual ``o@L`` attention-output site.
    """
    if site == 'transformer.wte':
        return 0
    if site == 'lm_head':
        return n_layer
    layer = layer_of(site)
    if layer is None:
        raise KeyError(f'Cannot place site {site!r} in the block order')
    return layer


def group_draw_count(recipe, group, layer=None):
    """Draws a Recipe assigns to ``group`` (at ``layer`` for per-block groups; the sum if layer is None).

    ``scalars`` is fixed at 8 two-sided draws in DUST.estimate and is not carried in the Recipe.
    """
    if group == 'scalars':
        return 8
    if group in _LOCAL_GROUPS:
        return recipe.local[group]
    value = getattr(recipe, group)
    if group in PER_BLOCK_GROUPS:
        return sum(value) if layer is None else value[layer]
    return value  # embedding, head


def total_draws(recipe):
    """Every group's draws summed. Note this exceeds the nominal population K: K counts only the
    direct-loss draws (writers + hidden + hubs + embedding), while head, the five attention
    internals, and the residual scalars are additional. Accounting must use this, not K."""
    return sum(group_draw_count(recipe, group) for group in DRAW_GROUPS)
