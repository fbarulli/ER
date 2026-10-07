"""Unprofiled twin of scripts/ablation_timing.py — real phase costs.

cProfile inflates call-heavy code (the pretty-JSON encoder issues ~900k
`write` calls for one report), so the harness's `*_seconds_logged` fields and
the cProfile ranking overstate `baseline_ablation._persist_baseline` by ~7x.
This script replays the same frozen smoke_500 workload (forward + complete on
a fresh output folder per iteration) with the SAME trace_step instrumentation
and NO profiler, so the numbers are the ones the mandate is about.

Usage: python artifacts/abl_opt/micro/timing_noprofile.py --out <dir> [--iters 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MICRO = Path(__file__).resolve().parent
ROOT = MICRO.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))
os.environ.setdefault('ER_TIMING_TRACE_PREFIX', 'abl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import ablation_timing as harness  # noqa: E402


def phase_totals(log_path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in log_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or not line.startswith('[timing]'):
            continue
        label = parts[1]
        elapsed = next((float(p.split('=', 1)[1]) for p in parts
                        if p.startswith('elapsed_seconds=')), None)
        if elapsed is not None and 'state=completed' in parts:
            totals[label] = round(totals.get(label, 0.0) + elapsed, 3)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='folder for timings.log/summary.json')
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    fixture = harness.build_suite()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['ER_TIMING_LOG'] = str(folder / 'timings.log')

    import torch
    torch.set_num_threads(4)
    torch.manual_seed(1729)

    from core.step_trace import trace_step
    from model_tracks import baseline_ablation

    walls = []
    phases = []
    for iteration in range(args.iters):
        out = folder / f'iter{iteration}' / 'output'
        link = out / harness.SHARED.name
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(harness.SHARED)
        log = folder / 'timings.log'
        before = len(log.read_text().splitlines()) if log.exists() else 0
        started = time.perf_counter()
        with trace_step('abl.forward'):
            baseline_ablation.forward(out, harness.SUITE, harness.CHECKPOINT, device='cpu')
        with trace_step('abl.complete'):
            baseline_ablation.complete(out, harness.SUITE)
        walls.append(round(time.perf_counter() - started, 3))
        lines = log.read_text().splitlines()
        chunk = '\n'.join(lines[before:]) + '\n'
        totals = phase_totals_from_text(chunk)
        phases.append(totals)
    summary = {'fixture': fixture, 'iterations': args.iters, 'wall_seconds': walls,
               'wall_cold': walls[0], 'wall_warm_mean': round(sum(walls[1:]) / max(1, len(walls[1:])), 3),
               'phases': phases, 'label': args.label}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'wall_seconds': walls, 'wall_warm_mean': summary['wall_warm_mean'],
                      'phases': phases[-1]}, indent=2))
    print('\nPHASE TOTALS (last iteration):')
    for label, value in sorted(phases[-1].items(), key=lambda kv: -kv[1])[:18]:
        print(f'  {value:8.3f}  {label}')


def phase_totals_from_text(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not line.startswith('[timing]'):
            continue
        label = parts[1]
        elapsed = next((float(p.split('=', 1)[1]) for p in parts
                        if p.startswith('elapsed_seconds=')), None)
        if elapsed is not None and 'state=completed' in parts:
            totals[label] = round(totals.get(label, 0.0) + elapsed, 3)
    return totals


if __name__ == '__main__':
    main()
