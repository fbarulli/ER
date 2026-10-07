"""Accelerator-primitive benchmark (CPU-safe, no GPU required).

Runs each primitive in ``core/fast_kernels.py`` plus the optimizer backend in
``core/gpu_execution.py`` and prints a legacy-vs-enabled table. The *legacy*
column is measured in a child process with ``ER_PERF_LEGACY=1``; the *enabled*
column uses the switches' default (all accelerators on). On this host there is
no GPU, so the Triton segment kernel / CUDA graphs / fused AdamW are not
exercised -- everything dispatches to the torch fallback and is asserted
bit-identical to a hand-written reference.

Run from the repository root::

    PYTHONPATH=src .venv/bin/python scripts/accelerator_bench.py

On a CUDA/T4 host (Triton installed), the same command exercises the Triton
segment kernel and fused optimizer; add ``ER_BENCH_COMPILE=1`` to also measure
real ``torch.compile`` (Inductor, ~40 s on CPU, seconds on GPU)::

    PYTHONPATH=src python scripts/accelerator_bench.py
    ER_BENCH_COMPILE=1 PYTHONPATH=src python scripts/accelerator_bench.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402  (after sys.path setup)

from core.execution_policy import resolve_aggregation  # noqa: E402
from core.fast_kernels import (  # noqa: E402
    autocast_context, compile_model, segment_reduce_fast, triton_available,
)
from core.gpu_execution import OptimizerExecution  # noqa: E402
from core.perf_switches import legacy_mode  # noqa: E402


def _best_seconds(fn, *, iters: int = 40, warmup: int = 10) -> float:
    """Best-of-N wall time; robust to CPU contention from parallel jobs."""
    for _ in range(warmup):
        fn()
    best = float('inf')
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def _segment_bench() -> dict:
    torch.manual_seed(0)
    count, edges, features = 256, 4096, 32
    # Last segment is intentionally empty (index never reaches count-1).
    index = torch.randint(0, count - 1, (edges,), dtype=torch.int64)
    values = torch.randn(edges, features)
    sizes = torch.zeros(count)
    sizes.index_add_(0, index, torch.ones(edges))
    sizes = sizes.clamp_min(1).unsqueeze(1)  # exactly what pooling.topology caches

    reference = values.new_zeros((count, features))
    reference.index_add_(0, index, values)
    reference_sum = reference
    reference_mean = reference / sizes

    fast_sum = segment_reduce_fast(values, index, sizes, reduce='sum')
    fast_mean = segment_reduce_fast(values, index, sizes, reduce='mean')
    torch.testing.assert_close(fast_sum, reference_sum, rtol=0, atol=0)
    torch.testing.assert_close(fast_mean, reference_mean, rtol=0, atol=0)
    assert bool(fast_mean[count - 1].abs().max() == 0), 'empty segment must pool to 0'

    return {
        'mean_seconds': _best_seconds(lambda: segment_reduce_fast(values, index, sizes)),
        'sum_seconds': _best_seconds(lambda: segment_reduce_fast(values, index, sizes, reduce='sum')),
        'parity': 'exact',
        'empty_segment_zero': True,
    }


def _optimizer_bench() -> dict:
    torch.manual_seed(0)
    policy = OptimizerExecution(backend='auto')
    weight = torch.randn(1024, 256, requires_grad=True)
    bias = torch.randn(256, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=1e-3,
                                  **policy.kwargs('cpu'))

    def step():
        optimizer.zero_grad(set_to_none=True)
        (weight.square().mean() + bias.square().mean()).backward()
        optimizer.step()

    return {
        'backend': policy.resolved_backend('cpu'),
        'seconds': _best_seconds(step, iters=60),
    }


def _compile_bench() -> dict:
    module = torch.nn.Linear(32, 16)
    inputs = torch.randn(64, 32)

    def forward(x):
        return torch.relu(module(x)).sum()

    if os.environ.get('ER_BENCH_COMPILE') != '1':
        return {'legacy_seconds': _best_seconds(lambda: forward(inputs)), 'enabled_seconds': None,
                'note': 'skipped (set ER_BENCH_COMPILE=1; ~40s CPU, use T4 for compile)'}
    handle = compile_model(forward, name='bench.forward', mode='reduce-overhead')
    torch.testing.assert_close(handle(inputs), forward(inputs))
    return {'legacy_seconds': _best_seconds(lambda: forward(inputs)),
            'enabled_seconds': _best_seconds(lambda: handle(inputs)),
            'note': 'torch.compile'}


def collect() -> dict:
    torch.set_num_threads(1)
    autocast = autocast_context('cpu')
    return {
        'torch': str(torch.__version__),
        'cuda': torch.cuda.is_available(),
        'triton': triton_available(),
        'legacy_mode': legacy_mode(),
        'segment': _segment_bench(),
        'optimizer': _optimizer_bench(),
        'compile': _compile_bench(),
        'autocast': type(autocast).__name__,
        'aggregation_auto': resolve_aggregation('index_add', 'cpu'),
    }


def _fmt(seconds) -> str:
    return 'n/a' if seconds is None else f'{seconds * 1e3:9.4f} ms'


def _legacy_via_subprocess() -> dict:
    with tempfile.NamedTemporaryFile('r', suffix='.json', delete=False) as handle:
        path = handle.name
    try:
        env = {**os.environ, 'ER_PERF_LEGACY': '1'}
        env.pop('ER_BENCH_COMPILE', None)
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--legacy-child', path],
                       env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return json.loads(Path(path).read_text())
    finally:
        Path(path).unlink(missing_ok=True)


def _print_table(enabled: dict, legacy: dict) -> None:
    print(f"accelerator benchmark  torch={enabled['torch']}  cuda={enabled['cuda']}  "
          f"triton={enabled['triton']}  legacy_mode={enabled['legacy_mode']}")
    header = f"{'primitive':<26}{'legacy':>15}{'enabled':>15}   notes"
    print(header)
    print('-' * len(header))
    rows = [
        ('segment_reduce.mean',
         legacy['segment']['mean_seconds'], enabled['segment']['mean_seconds'],
         f"parity={enabled['segment']['parity']} empty->0={enabled['segment']['empty_segment_zero']}"),
        ('segment_reduce.sum',
         legacy['segment']['sum_seconds'], enabled['segment']['sum_seconds'], 'triton+cuda+flag only'),
        ('adamw.step',
         legacy['optimizer']['seconds'], enabled['optimizer']['seconds'],
         f"backend {legacy['optimizer']['backend']} -> {enabled['optimizer']['backend']}"),
        ('compile.forward',
         legacy['compile']['legacy_seconds'], enabled['compile']['enabled_seconds'],
         enabled['compile']['note']),
    ]
    for name, old, new, note in rows:
        print(f'{name:<26}{_fmt(old):>15}{_fmt(new):>15}   {note}')
    print(f"{'autocast':<26}{'nullcontext(cpu)':>15}{enabled['autocast']:>15}")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == '--legacy-child':
        Path(sys.argv[2]).write_text(json.dumps(collect()))
        return 0
    enabled = collect()
    legacy = _legacy_via_subprocess()
    _print_table(enabled, legacy)
    if enabled['segment']['parity'] != 'exact':
        print('FAIL: segment reduce parity', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
