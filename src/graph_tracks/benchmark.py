"""GPU training-step benchmark; separate from quality training and reporting.

Uses prepared real inputs. Warmup/compilation is excluded from measured steps.
The model is discarded; no checkpoints or evaluation decisions are produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from graph_tracks.config import load_config
from graph_tracks.data import fit_vocabulary, load_records, load_text_cache, tensorize
from graph_tracks.model import AttributeGNN, PairScorer
from graph_tracks.preflight import preflight
from graph_tracks.train import load_pairs


def benchmark(config: Path, *, steps=20, warmup=5, compile_model=False,
              bf16=False, fused_optimizer=False, profile: Path | None = None) -> dict:
    from core.common import TRAIN_ROOT
    if not torch.cuda.is_available():
        raise RuntimeError('GPU benchmark requires CUDA; no CPU timing substituted')
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unavailable on this GPU')
    if steps < 1 or warmup < 1:
        raise ValueError('steps and warmup must be positive')
    info = preflight(config, check_device=False)
    cfg = load_config(config)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    resolve = lambda raw: (TRAIN_ROOT / raw).resolve()
    records = load_records(resolve(cfg.listings))
    pairs = load_pairs(resolve(cfg.pairs), records)['train']
    vocabulary = fit_vocabulary(records)
    support_indices = [i for i, r in enumerate(records) if r['split'] == 'train']
    batch = tensorize(records, vocabulary, 'cuda')
    support = tensorize([records[i] for i in support_indices], vocabulary, 'cuda')
    text = None
    if cfg.text_cache:
        vectors, _ = load_text_cache(resolve(cfg.text_cache), [r['sku_id'] for r in records])
        text = torch.tensor(vectors, device='cuda')
    support_text = None if text is None else text[support_indices]
    model = AttributeGNN(vocabulary, cfg.hidden_dim, cfg.output_dim,
                         0 if text is None else text.shape[1], cfg.graph_enabled).cuda()
    scorer = PairScorer(text is not None).cuda()
    parameters = list(model.parameters()) + list(scorer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, fused=fused_optimizer)
    train_pairs = torch.tensor(pairs[0], device='cuda')
    labels = torch.tensor(pairs[1], device='cuda')
    context, encode, score = model.context, model.encode, scorer.forward
    if compile_model:
        # Prime detached static topology eagerly; compilation then reads the
        # immutable metadata without dynamic boolean edge compaction.
        with torch.no_grad(), torch.autocast(
                'cuda', dtype=torch.bfloat16, enabled=bf16):
            model.encode(batch, model.context(support, support_text), text)
        context, encode, score = [torch.compile(f, backend='inductor') for f in (context, encode, score)]

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            states = context(support, support_text)
            vectors = encode(batch, states, text)
            logits = score(vectors, train_pairs, text)
            classification = F.binary_cross_entropy_with_logits(logits, labels)
            a, b = train_pairs.unbind(1)
            cosine = (vectors[a].float() * vectors[b].float()).sum(-1)
            metric = (labels * (1 - cosine) + (1 - labels)
                      * F.relu(cosine - cfg.negative_margin)).mean()
            loss = classification + cfg.metric_weight * metric
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    warmup_started = time.perf_counter()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        losses.append(float(loss))
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError('benchmark produced nonfinite loss')
    if profile:
        profile.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profiler:
            step()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(profile))
    return {'schema': 'er-graph-gpu-benchmark-v1', 'track': cfg.track,
            'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'compile_inductor': compile_model,
            'bf16': bf16, 'fused_adamw': fused_optimizer,
            'warmup_steps': warmup, 'warmup_seconds': warmup_seconds,
            'measured_steps': steps, 'step_seconds': timings,
            'median_step_seconds': float(torch.tensor(timings).median()),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'losses': losses, 'input_preflight': info,
            'includes': 'forward, loss, backward, clipping, optimizer',
            'excludes': 'preparation, dev evaluation, checkpoint I/O, reports, DVC/W&B',
            'quality_or_speedup_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--compile', action='store_true', dest='compile_model')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = benchmark(args.config, steps=args.steps, warmup=args.warmup,
                       compile_model=args.compile_model, bf16=args.bf16,
                       fused_optimizer=args.fused_optimizer, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
