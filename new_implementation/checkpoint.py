"""Full-state checkpointing for resumable DUST runs.

The shipped code saves only ``best.pt`` (weights), which cannot resume a run: SGD momentum, the
perturbation RNG, the data position and (for adaptive allocation) the allocator state are all lost.
This saves everything needed to restart a run bit-for-bit, so long or preemptible (Spot) jobs survive.

What is captured:
- model weights (dtype preserved, so bf16 token/value embeddings stay bf16);
- optimizer state (SGD momentum buffers and param-group learning rates);
- the estimator's perturbation generator state, so draws continue the same stream;
- torch CPU and CUDA RNG states;
- the training step, so the data loader resumes at the same batch (the order is derived from the
  seed, so step alone fixes the position);
- an opaque allocator-state blob for adaptive allocation (Track 2), and any ``extra`` metadata.

Multi-GPU: each rank owns its own perturbation generator, so each rank writes and reads its own file
(``checkpoint_path`` adds a rank suffix). Weights and optimizer state are identical across ranks, so
every rank's file carries them; no broadcast needed on resume.
"""
from pathlib import Path

import torch


def checkpoint_path(base, rank=0, world=1):
    """Per-rank checkpoint file: ``base`` itself for a single process, ``base.rankN`` under DDP."""
    base = Path(base)
    return base if world == 1 else base.with_name(f'{base.name}.rank{rank}')


def save_state(base, *, model, optimizer, estimator, step, settings=None,
               allocator_state=None, extra=None, rank=0, world=1):
    """Write a resumable checkpoint for this process. Returns the path written.

    ``settings`` is the Recipe (stored as-is for provenance); ``allocator_state`` is whatever the
    adaptive allocator hands back; ``extra`` is any extra metadata (best val loss, data hashes, ...).
    """
    state = {
        'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'generator': estimator.generator.get_state().cpu(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'step': step,
        'settings': settings,
        'allocator_state': allocator_state,
        'extra': extra,
        'rank': rank,
        'world': world,
    }
    path = checkpoint_path(base, rank, world)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half-written checkpoint
    return path


def load_state(base, *, model, optimizer, estimator, rank=0, world=1, map_location='cpu'):
    """Restore model, optimizer and every RNG in place; return (step, allocator_state, extra).

    The data loader should resume at the returned step (its order is fixed by the seed). Raises if the
    checkpoint's world size differs from the current one, since the per-rank RNG streams would not line up.
    """
    path = checkpoint_path(base, rank, world)
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get('world', world) != world:
        raise ValueError(
            f'Checkpoint was written with world={state.get("world")}, cannot resume with world={world}'
        )
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    # Generator.set_state takes a CPU ByteTensor even for a CUDA generator (its get_state returns one).
    estimator.generator.set_state(state['generator'].cpu())
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'], state.get('allocator_state'), state.get('extra')
