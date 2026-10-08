"""A/B micro-benchmark for baseline_ablation._persist_baseline (r15).

Times the two implementations on a REAL report document lifted from a frozen
round artifact, and asserts the bytes they produce are identical.

BEFORE: write() -> json.dump(..., indent=2) streamed + file_size(read-back)
AFTER:  json.dumps(..., indent=2) + one write_bytes + size of those bytes

Usage: python artifacts/abl_opt/micro/bench_persist.py [--source <report.json>] [--reps 5]
"""
from __future__ import annotations

import argparse
from core.portable_archive import ByteCount
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))

from graph_tracks.data import file_size  # noqa: E402
from model_tracks.ablation import write  # noqa: E402
from model_tracks import baseline_ablation  # noqa: E402


def before(request_path: Path, result) -> None:
    """The pre-r15 implementation."""
    write(request_path.parent / 'report.json', result)
    (request_path.parent / 'report.size').write_text(
        file_size(request_path.parent / 'report.json') + '\n')


def timeit(fn, reps: int) -> float:
    fn()  # warm
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - started) / reps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='artifacts/abl_opt/rounds/round14/iter0/output/ablation/report.json')
    parser.add_argument('--reps', type=int, default=5)
    args = parser.parse_args()
    source = (ROOT / args.source) if not Path(args.source).is_absolute() else Path(args.source)
    document = json.loads(source.read_text())
    print(f'source={source} bytes={source.stat().st_size} keys={len(document)}')

    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / 'old' / 'ablation'
        new = Path(tmp) / 'new' / 'ablation'
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        old_seconds = timeit(lambda: before(old / 'request.json', document), args.reps)
        new_seconds = timeit(lambda: baseline_ablation._persist_baseline(new / 'request.json', document), args.reps)
        old_bytes = (old / 'report.json').read_bytes()
        new_bytes = (new / 'report.json').read_bytes()
        old_sidecar = (old / 'report.size').read_text()
        new_sidecar = (new / 'report.size').read_text()
    result = {
        'source': str(source), 'reps': args.reps,
        'before_seconds': round(old_seconds, 4), 'after_seconds': round(new_seconds, 4),
        'speedup': round(old_seconds / new_seconds, 2),
        'report_bytes': len(new_bytes), 'byte_identical': old_bytes == new_bytes,
        'sidecar_identical': old_sidecar == new_sidecar,
        'sidecar_matches_bytes': new_sidecar.strip() == ByteCount(new_bytes).total,
    }
    print(json.dumps(result, indent=2))
    (ROOT / 'artifacts/abl_opt/micro/bench_persist.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
