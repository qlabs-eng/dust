"""Minimal SGD backpropagation baseline with the same model, data, and evaluation as DUST."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, evaluate, file_hash, load_sequences, make_model
from run_metrics import TrainingLog, backprop_update_cost  # new_implementation/, put on the path by dust


DEFAULT_CONFIGS = {
    '1M': dict(lr=0.175, momentum=0.98, steps=61, eval_interval=2),
    '10M': dict(lr=0.2, momentum=0.95, steps=610, eval_interval=10),
}


def make_optimizer(model, lr, momentum):
    groups = [
        {'params': [p for name, p in model.named_parameters()
                    if name != 'transformer.wte.weight'], 'lr': lr},
        {'params': [model.transformer.wte.weight], 'lr': 1000.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        loss = model(x, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite backpropagation loss')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Non-finite backpropagation gradient for {name}')
    optimizer.step()
    return loss.item()


def train(config, training, validation, testing, output, *, steps, eval_interval, lr, momentum, seed=42,
          device=torch.device('cpu'), batch_size=8, metadata=None, echo=print):
    """Backprop SGD with the same per-step metrics records as dust.train (run_metrics.TrainingLog).

    Writes config.json, metrics.jsonl (step 0's val loss, then one record per step), best.pt and
    result.json to ``output``; returns result.json's contents. Compute is charged as forward + backward
    = 3 forwards per update (run_metrics.backprop_update_cost), on the same axis as DUST's.
    """
    output = Path(output)
    model = make_model(config, device, seed).requires_grad_(True)
    optimizer = make_optimizer(model, lr, momentum)
    run_log = TrainingLog(lambda _: backprop_update_cost(config, batch_size))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        metadata = dict(metadata, parameter_groups=[
            dict(lr=g['lr'], momentum=g['momentum'], parameters=sum(p.numel() for p in g['params']))
            for g in optimizer.param_groups])
        (output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def save_best():
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            output / 'best.pt',
        )

    def write(record):
        echo(json.dumps(record))
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    with run_log.timed('eval'):
        best = evaluate(model, validation, device)
    best_step = 0
    save_best()
    history = [dict(step=0, val_loss=best)]
    write(dict(step=0, val_loss=best, flops=0.0, forward_equivalents=0.0, draws=0,
               train_seconds=0.0, gpu_hours=0.0))

    for step in range(1, steps + 1):
        batch = training[(step - 1) * batch_size:step * batch_size].to(device)
        with run_log.step():
            loss = train_step(model, optimizer, batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        record = run_log.record(step, loss)
        if step == 1 or step % eval_interval == 0 or step == steps:
            with run_log.timed('eval'):
                value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best, best_step = value, step
                save_best()
        history.append(record)
        write(record)

    model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True))
    result = dict(best_val=best, best_step=best_step, test_at_best_val=evaluate(model, testing, device),
                  history=history, timing=run_log.summary())
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    echo(f"Best validation: {best:.4f} (step {best_step}); test: {result['test_at_best_val']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/backprop'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, help='Override the base learning rate for the selected token budget.')
    parser.add_argument('--momentum', type=float, help='Override SGD momentum for the selected token budget.')
    parser.add_argument('--steps', type=int, help='Stop early; keep the full selected training subset.')
    parser.add_argument('--toy', action='store_true',
                        help='Tiny model and synthetic data (new_implementation/toy.py): a CPU smoke run.')
    parser.add_argument('--device', help='cuda (default), cpu or mps. Only --toy runs are practical off CUDA.')
    args = parser.parse_args()
    defaults = DEFAULT_CONFIGS[args.tokens]
    args.lr = defaults['lr'] if args.lr is None else args.lr
    args.momentum = defaults['momentum'] if args.momentum is None else args.momentum

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() or not args.toy else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU (use --toy for a CPU smoke run).')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        parser.error('This baseline uses one GPU; launch it with python, not torchrun.')
    if args.lr <= 0 or not 0 <= args.momentum < 1:
        parser.error('Use a positive learning rate and momentum in [0, 1).')
    device = torch.device('cuda', 0) if args.device == 'cuda' else torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    if args.toy:
        from toy import TOY_BATCH, TOY_CONFIG, TOY_EVAL_INTERVAL, TOY_STEPS, toy_data
        config, steps, interval = TOY_CONFIG, TOY_STEPS, TOY_EVAL_INTERVAL
        training, validation, testing = toy_data(TOY_STEPS, TOY_BATCH)
        data = dict(source='toy (new_implementation/toy.py)')
    else:
        config, steps, interval = GPTConfig(), defaults['steps'], defaults['eval_interval']
        paths = dict(
            train=args.data_dir / f'train_{args.tokens}.pt',
            val=args.data_dir / 'val.pt',
            test=args.data_dir / 'test.pt',
        )
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            parser.error(
                f"Missing data: {', '.join(missing)}. Run python prepare_data.py "
                f"--tokens {args.tokens} --output {args.data_dir}"
            )
        training = load_sequences(paths['train'], steps * 8)
        validation = load_sequences(paths['val'], 544)
        testing = load_sequences(paths['test'], 544)
        data = dict(sha256={name: file_hash(path) for name, path in paths.items()})
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    metadata = dict(
        method='backprop', tokens='toy' if args.toy else args.tokens, seed=args.seed, steps=steps,
        lr=args.lr, momentum=args.momentum, weight_decay=0.0,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=len(validation), test_sequences=len(testing), model=asdict(config),
        device=str(device),
        source_sha256=file_hash(Path(__file__)),
        model_source_sha256=file_hash(Path(__file__).resolve().parents[1] / 'dust.py'),
        model_sha256=file_hash(Path(__file__).resolve().parents[1] / 'model.py'),
        torch_version=torch.__version__,
        data=data,
    )
    print(f"Backprop SGD: {metadata['tokens']}, {steps} steps, device={device}", flush=True)
    train(config, training, validation, testing, args.output, steps=steps, eval_interval=interval,
          lr=args.lr, momentum=args.momentum, seed=args.seed, device=device, metadata=metadata,
          echo=lambda line: print(line, flush=True))


if __name__ == '__main__':
    main()
