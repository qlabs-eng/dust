"""torch.profiler harness for DUST, plus the launch-bound test.

Lecture 3 workflow: benchmark (CUDA events, warmup, a distribution) -> profile (per-op table and a
trace) -> change one thing -> benchmark again. accounting.RunMeters / accounting.measure is the
benchmark side; this is the profile side. DUST.estimate labels its phases with
record_function('dust/<phase>'), so the table and the Perfetto trace read as clean / direct /
embedding / hubs / head / local / scalars rather than a wall of aten ops.

Two things to read off a DUST profile:

- Per-phase CUDA time vs the analytic FLOPs from accounting.analytic_cost. If a phase's measured share
  does not match its FLOP share, the FLOP model is missing something (launch overhead, a hidden copy).
- CPU time well above CUDA time (sort_by='self_cpu_time_total'), or gaps between kernels in the trace,
  mean the step is launch-bound: it follows CPU issue time, not GPU work, so a FLOP saving (suffix
  caching, reallocation) will not show up in wall-clock until the step is made GPU-bound. The
  launch_bound_test below is the quick check before a full profile.

Everything here needs torch and, for GPU numbers, a CUDA device; it is import-safe without them but
raises a clear error if asked to profile with no profiler available.
"""


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('profiling.py needs torch installed') from exc


def profile_estimate(estimator, x, y, warmup=2, active=1, sort_by='self_cuda_time_total',
                     row_limit=25, trace_path=None, group_by_shape=False):
    """Profile DUST.estimate and return (table_str, prof).

    Runs ``warmup`` estimates first and discards them (allocator and autotune warmup, Lecture 3),
    then profiles ``active`` estimate() calls. ``sort_by='self_cuda_time_total'`` surfaces GPU hot
    spots; use 'self_cpu_time_total' to surface launch and dispatch overhead. ``trace_path`` exports a
    Perfetto / chrome trace. The dust/<phase> rows give the per-phase breakdown.
    """
    torch = _require_torch()
    from torch.profiler import profile, ProfilerActivity

    for _ in range(warmup):
        estimator.estimate(x, y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(active):
            estimator.estimate(x, y)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    table = prof.key_averages(group_by_input_shape=group_by_shape).table(
        sort_by=sort_by, row_limit=row_limit,
    )
    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))
    return table, prof


def profile_training(step_fn, batches, skip_first=5, wait=1, warmup=1, active=3,
                     sort_by='self_cuda_time_total', row_limit=20, trace_path='runs/trace.json'):
    """Profile a training loop with the Lecture 3 schedule and export one trace.

    ``step_fn(batch)`` runs one update. The schedule records only steps after ``skip_first`` + a
    ``wait``/``warmup`` cycle, so start-up never skews the trace. Prints the table at each cycle end
    and writes the trace for Perfetto (ui.perfetto.dev). Returns the prof object.
    """
    torch = _require_torch()
    from torch.profiler import profile, schedule, ProfilerActivity
    from pathlib import Path

    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    def on_ready(prof):
        print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit), flush=True)
        prof.export_chrome_trace(str(trace_path))

    plan = schedule(skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=1)
    total = skip_first + wait + warmup + active
    with profile(activities=activities, schedule=plan, on_trace_ready=on_ready,
                 record_shapes=True, with_stack=True) as prof:
        for index, batch in enumerate(batches):
            step_fn(batch)
            prof.step()
            if index + 1 >= total:
                break
    return prof


def launch_bound_test(make_step, batch, double_batch, repeats=20, warmup=5):
    """Double-batch test for launch-bound vs GPU-bound (Lecture 3 slide 44).

    ``make_step(data)`` runs one update on ``data``. Times the step at ``batch`` and at
    ``double_batch`` with CUDA events, discarding ``warmup`` runs. If doubling the batch barely moves
    the median step time, the GPU had spare capacity and the step is launch-bound (CPU issue sets it);
    if the time roughly doubles, it is GPU-bound. Returns the medians, the ratio and the verdict.
    """
    from accounting import RunMeters, measure, _percentile

    def median_ms(data):
        meters = RunMeters(warmup=warmup)
        for _ in range(repeats + warmup):
            with measure(meters, 'step'):
                make_step(data)
        kept = meters.step_gpu_ms or meters.step_host_ms  # prefer GPU time when available
        return _percentile(kept[warmup:] or kept, 50)

    base = median_ms(batch)
    doubled = median_ms(double_batch)
    ratio = doubled / base if base else float('inf')
    verdict = 'launch-bound' if ratio < 1.3 else ('gpu-bound' if ratio > 1.7 else 'mixed')
    return dict(
        step_ms_1x=base, step_ms_2x=doubled, ratio=ratio, verdict=verdict,
        note='ratio ~1 => launch-bound (FLOP savings will not move wall-clock); '
             '~2 => GPU-bound (FLOP savings do). Run before trusting analytic forward-equivalents.',
    )
