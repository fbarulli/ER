"""A/B micro-benchmark for ablation_retrieval.RetrievalComparison.ranks.

Compares the pre-r16 implementation (kept here verbatim) with the current one
on real fixture data, and asserts they return the SAME rank table.

  python artifacts/abl_opt/micro/bench_ranks.py smoke      # 500-candidate smoke fixture
  python artifacts/abl_opt/micro/bench_ranks.py 10k <request.json> <vectors.npz>
                         # real 10k cohort inputs (read-only paths)

The 10k inputs are the ER-abl-opt cohort inputs:
  request.json  = artifacts/abl_opt/results/<digest>/request.json
  vectors.npz   = artifacts/abl_opt/rounds/round1/vectors.npz
Copy them somewhere writable: RetrievalComparison stages its HNSW index in a
TemporaryDirectory under the request path's parent.
"""
from __future__ import annotations

import json
import os
import cProfile
import pstats
from statistics import median
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))

import numpy as np  # noqa: E402

from model_tracks.ablation import Settings, validate_vectors  # noqa: E402
from model_tracks.ablation_retrieval import RetrievalComparison  # noqa: E402


def legacy_ranks(self, queries):
    """The pre-r16 body, verbatim."""
    pair_ranks = [[None, None] for _ in self.pairs]
    targets = {}
    for n, (a, b) in enumerate(self.pairs):
        for side, source, target in ((0, a, b), (1, b, a)):
            targets.setdefault(source, []).append((n, side, target))
    candidate_order = np.arange(len(self.vectors))
    for source, requested in targets.items():
        scores = queries[source] @ self.vectors.T
        scores[self.endpoints[source]] = -np.inf
        for n, side, target in requested:
            target_index = self.endpoints[target]
            value = scores[target_index]
            rank = 1+np.count_nonzero(scores > value)+np.count_nonzero(
                (scores == value)&(candidate_order < target_index))
            pair_ranks[n][side] = int(rank)
    return pair_ranks


def profiled_self(fn, reps: int) -> dict:
    """Self seconds per module function in this file, under cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(reps):
        fn()
    profiler.disable()
    out = {}
    for (file, line, name), (_cc, nc, tottime, _cum, _callers) in pstats.Stats(profiler).stats.items():
        if 'ablation_retrieval' in file or 'bench_ranks' in file:
            out[f'{name}@{line}'] = [round(tottime, 4), nc]
    return out


def timeit(fn, reps: int) -> float:
    fn()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - started) / reps


# Synthetic crossover grid: (candidates, query sources). Sizes bracket the
# smoke fixture (500) and full-catalog cohorts. Each cell builds a normalized
# random catalog/query matrix (seeded), drives the REAL ranks() size dispatch
# and _rank kernel through an __new__ instance (the HNSW index that __init__
# builds is not on this code path), and asserts the legacy and current
# implementations return the same rank table.
SWEEP_GRID = ((500, 33), (1000, 33), (2000, 64), (5000, 64), (5000, 200),
              (10000, 100), (10000, 358), (20000, 358))
SWEEP_SEED = 20261007
SWEEP_DIM = 384


def sweep() -> dict:
    import os as _os
    rng = np.random.default_rng(SWEEP_SEED)
    cells = []
    for candidates, sources in SWEEP_GRID:
        catalog = rng.standard_normal((candidates, SWEEP_DIM), dtype=np.float32)
        catalog /= np.linalg.norm(catalog, axis=1, keepdims=True)
        queries = rng.standard_normal((sources, SWEEP_DIM), dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        retrieval = RetrievalComparison.__new__(RetrievalComparison)
        retrieval.vectors = catalog
        retrieval.endpoints = list(range(sources))
        retrieval.pairs = [(2 * n, 2 * n + 1) for n in range(sources // 2)]
        reps = 3 if candidates * sources > 2_000_000 else 10
        legacy = legacy_ranks(retrieval, queries)
        current = retrieval.ranks(queries)
        legacy_samples, current_samples = [], []
        for _ in range(reps):
            started = time.perf_counter()
            legacy_ranks(retrieval, queries)
            legacy_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            retrieval.ranks(queries)
            current_samples.append(time.perf_counter() - started)
        before, after = median(legacy_samples), median(current_samples)
        cells.append({'candidates': candidates, 'sources': sources, 'pairs': len(retrieval.pairs),
                      'reps': reps,
                      'before_ms': round(before * 1e3, 3), 'after_ms': round(after * 1e3, 3),
                      'speedup': round(before / after, 2),
                      'path': 'production ranks() (exact per-source GEMV)',
                      'ranks_identical': legacy == current})
        print(f"  candidates={candidates:>6} sources={sources:>4} "
              f"{before * 1e3:9.3f} ms -> {after * 1e3:8.3f} ms  {cells[-1]['speedup']:6.2f}x "
              f"{cells[-1]['path']:>14}  identical={cells[-1]['ranks_identical']}", flush=True)
    result = {'kind': 'synthetic ranks equivalence/throughput sweep', 'seed': SWEEP_SEED, 'dim': SWEEP_DIM,
              'dtype': 'float32 (L2-normalized rows)', 'host_cpus': _os.cpu_count(),
              'loadavg': [round(v, 2) for v in _os.getloadavg()],
              'note': ('wall-clock timing on a shared 4-CPU host; both implementations are '
                       'measured in the same process, alternating reps, and must return the '
                       'same rank table'),
              'cells': cells}
    (ROOT / 'artifacts/abl_opt/micro/bench_ranks_sweep.json').write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == '--sweep':
        result = sweep()
        print(json.dumps({k: result[k] for k in ('seed', 'loadavg', 'host_cpus')}, indent=2))
        return
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if mode == 'smoke':
        ablation = ROOT / 'artifacts/abl_opt/rounds/round15/iter0/output/ablation'
        request_path, vectors_path = ablation / 'request.json', ablation / 'vectors.npz'
        reps = 20
    else:
        request_path, vectors_path = Path(sys.argv[2]), Path(sys.argv[3])
        reps = 6
    request = json.loads(request_path.read_text())
    cfg = Settings.model_validate(request['settings'])
    if mode == 'smoke':
        _, vectors, _, candidates = validate_vectors(request_path, vectors_path)
    else:
        # Read the arrays straight out of the npz: validation of a foreign
        # cohort's sources is not what this benchmark measures.
        with np.load(vectors_path, allow_pickle=False) as data:
            vectors, candidates = data['vectors'], data['candidate_vectors']
    queries = vectors[0]
    print(f'mode={mode} candidates={len(request.get("candidate_ids", []))} '
          f'ids={len(request["ids"])} pairs={len(request["pairs"])} reps={reps}')

    retrieval = RetrievalComparison(request.get('candidate_ids', request['ids']),
                                    candidates, request, request_path, cfg)
    legacy = legacy_ranks(retrieval, queries)
    new = retrieval.ranks(queries)
    identical = legacy == new
    # Interleave the two implementations so drift in host load cannot favour
    # whichever one happens to run first.
    legacy_samples, new_samples = [], []
    for rep in range(reps):
        started = time.perf_counter()
        legacy_ranks(retrieval, queries)
        legacy_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        retrieval.ranks(queries)
        new_samples.append(time.perf_counter() - started)
    legacy_seconds = median(legacy_samples)
    new_seconds = median(new_samples)
    # cProfile view of the same calls, for comparison with ranking_lane.csv:
    # the harness profiles this path, and a profiler charges every call, so a
    # helper split shows up here even where the unprofiled call is faster.
    # The profiled view is only taken on the small fixture: profiling the 10k
    # catalog's per-source GEMV loop (a Python-level numpy call per source)
    # costs more than the benchmark itself.
    profiled = ({'legacy': profiled_self(lambda: legacy_ranks(retrieval, queries), reps),
                 'new': profiled_self(lambda: retrieval.ranks(queries), reps)}
                if mode == 'smoke' else {})
    retrieval.close()
    result = {
        'mode': mode, 'reps': reps,
        'candidates': len(request.get('candidate_ids', [])),
        'ids': len(request['ids']), 'pairs': len(request['pairs']),
        'before_seconds': round(legacy_seconds, 4), 'after_seconds': round(new_seconds, 4),
        'speedup': round(legacy_seconds / new_seconds, 2),
        'before_samples': [round(v, 4) for v in legacy_samples],
        'after_samples': [round(v, 4) for v in new_samples],
        'cprofile_self_seconds': profiled,
        'loadavg': [round(v, 2) for v in os.getloadavg()],
        'ranks_identical': bool(identical),
        'rank_entries': sum(1 for row in new for value in row if value is not None),
        'note': 'wall-clock timing on a shared 4-CPU host, interleaved reps, median',
    }
    print(json.dumps(result, indent=2))
    out = ROOT / f'artifacts/abl_opt/micro/bench_ranks_{mode}.json'
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
