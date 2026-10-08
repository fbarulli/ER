"""Ablation-lane timing harness (reusable across optimization rounds).

Workload (identical every round, CPU-first; real 5k cohort inputs):
  1. ablation.prepare      — catalog=dataset_5k.csv (5,000 real rows),
                             pairs=artifacts/abl_opt/inputs/pairs_dev_500.csv
                             (deterministic consecutive-pair derivation from
                             the same catalog, 500 dev pairs, label = same
                             brand, population='real'), checkpoint=the real
                             MiniLM checkpoint, track='text',
                             retrieval_catalog='full' -> 5k candidate texts.
  2. ablation.encode       — device='cpu', saved_text=the seeded MiniLM cache
                             (artifacts/abl_opt/inputs/shared_minilm.csv.npz)
                             so native forwarding covers only variant texts,
                             exactly like the staged baseline path.
  3. ablation.report       — frozen threshold binding under inputs/, HNSW on
                             the 5k candidate catalog, exact ranks + ann
                             hits + paired rows.

Deterministic cohort rule (recorded, fixed forever):
  pairs are consecutive sorted sku_id neighbors (i, i+1) over the first
  order-preserved rows of dataset_5k.csv after sorting by sku_id; label is
  '1' iff both rows share a nonempty brand; split stays 'dev'; population
  column 'real'.

Outputs per round in artifacts/abl_opt/rounds/round<N>/:
  timings.log ([timing] lines behind ER_TIMING_LOG), ranking.csv
  (cProfile per-function wall ranking), profile.prof, fingerprints.json
  (output-content hashes for the byte-identical mandate), phase summary on
  stdout. Regenerable inputs live under artifacts/abl_opt/inputs/ (npz is
  gitignored; rebuild with --rebuild-cache).
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import json
import pstats
import shutil
import sys
import time
from pathlib import Path

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
import os  # noqa: E402

sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))
os.environ.setdefault('ER_TIMING_TRACE_PREFIX', 'abl')
if os.environ.get('TOKENIZERS_PARALLELISM') is None:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if os.environ.get('OMP_NUM_THREADS') is None:
    os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ABL = ROOT / 'artifacts' / 'abl_opt'
INPUTS = ABL / 'inputs'
ROUNDS = ABL / 'rounds'
CATALOG = ROOT / 'dataset_5k.csv'
PAIRS = INPUTS / 'pairs_dev_500.csv'
CONFIG = INPUTS / 'ablation_settings.yaml'
CHECKPOINT = ROOT / 'artifacts' / 'models' / 'all-MiniLM-L6-v2'
SAVED_TEXT = INPUTS / 'shared_minilm__ablation_5k.npz'
BINDING = INPUTS / 'baseline_threshold.json'
PAIRS_N = 500
THRESHOLD_STATE = INPUTS / 'threshold.json'
REPORT_NOTE = ('inputs recorded 2026-10-08: catalog=dataset_5k.csv '
               '(5k real rows); pairs = consecutive sorted-sku_id '
               'neighbors over first 500 rows, label=1 iff both share '
               'a nonempty brand; split=dev; population=real; '
               'coverage=sampled sample_pairs=100; retrieval_catalog=full')


def sha_bytes(value) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                default=lambda o: '{{unserializable:%r}}' % (o,))
                     .encode('utf-8'))


def sha_npz(path: Path) -> dict:
    """Content fingerprint of npz arrays (zip headers vary per write)."""
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            value = data[key]
            shape = getattr(value, 'shape', None)
            if hasattr(value, 'dtype') and shape:
                out[key] = [list(shape), str(value.dtype)]
                raw = np.ascontiguousarray(value).view(np.uint8)
                out[key] += [sha_bytes(raw.tobytes())]
            else:
                out[key] = sha_json(value.tolist() if hasattr(value, 'tolist') else str(value))
    return out


def list_dirs(base: Path, prefix='round') -> list[int]:
    if not base.is_dir():
        return []
    return sorted(int(p.name[len(prefix):]) for p in base.iterdir()
                  if p.is_dir() and p.name.startswith(prefix) and p.name[len(prefix):].isdigit())


def prepare_pairs():
    """Deterministic dev pairs from the real 5k catalog (rule in module doc)."""
    frame = pd.read_csv(CATALOG, dtype=str, keep_default_na=False)
    frame['sku_id'] = frame['sku_id'].astype(str)
    frame = frame[frame.sku_id != ''].sort_values('sku_id', kind='stable').head(PAIRS_N)
    rows = frame.to_dict('records')
    records = []
    for a, b in zip(rows, rows[1:]):
        a_brand, b_brand = a.get('brand', ''), b.get('brand', '')
        records.append({'sku_id1': a['sku_id'], 'sku_id2': b['sku_id'],
                        'label': '1' if a_brand and a_brand == b_brand else '0',
                        'split': 'dev', 'population': 'real'})
    return pd.DataFrame(records)


def ensure_inputs(profile=False, rebuild_cache=False):
    INPUTS.mkdir(parents=True, exist_ok=True)
    if not PAIRS.exists():
        prepare_pairs().to_csv(PAIRS, index=False)
    if not CONFIG.exists():
        from model_tracks.ablation import settings
        cfg = settings().model_dump()
        cfg.update(coverage='sampled', sample_pairs=100, split='dev',
                   output_dir=str(ABL / 'results'), report_path=str(ABL / 'results' / 'report.json'),
                   attributes=[], retrieval_catalog='full',
                   slice_columns=['population'], batch_size=64,
                   uniform_channels=True)
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=True))
    from model_tracks.ablation import checkpoint_identity
    identity = checkpoint_identity(CHECKPOINT)
    if not THRESHOLD_STATE.exists():
        THRESHOLD_STATE.write_text(json.dumps({'threshold': 0.35, 'note': REPORT_NOTE}))
    if not BINDING.exists():
        BINDING.write_text(json.dumps({'track': 'text', 'checkpoint_sha256': identity,
                                       'threshold': json.loads(THRESHOLD_STATE.read_text())['threshold'],
                                       'note': REPORT_NOTE}, sort_keys=True))
    if rebuild_cache or not SAVED_TEXT.exists():
        build_saved_text_cache()
    return identity


def build_saved_text_cache():
    """One-time CPU encode of the 5k-catalog native baseline texts, then
    freeze the shared saved-text cache npz used by encode(saved_text=...)."""
    from model_tracks.ablation import prepare
    from graph_tracks.text_cache import texts_hash
    from core.model_input import model_input_composition
    work = ABL / 'cache_staging'
    config = CONFIG
    print('[ablbm] one-time full CPU encode for seeded cache (a few minutes).', flush=True)
    request_path = prepare(CATALOG, PAIRS, CHECKPOINT, config=config)
    request = json.loads(request_path.read_text())
    out = work / 'vectors.npz'
    if out.exists():
        out.unlink()
    import model_tracks.ablation as ablation
    ablation.encode(request_path, out, device='cpu')
    ids = request['candidate_ids']
    with np.load(out, allow_pickle=False) as data:
        candidates = np.asarray(data['candidate_vectors'], dtype=np.float32)
        vectors = np.asarray(data['vectors'], dtype=np.float32)
    # baseline encode produces variant vectors too; seeded text cache uses
    # the candidate vectors (baseline text of every catalog id).
    metadata = {'checkpoint_sha256': request['sources'].get(request['checkpoint'], ''),
                'composition': request['composition'],
                'text_sha256': texts_hash([request['texts'][i]
                                           for i in request['candidate_text_indices']]),
                'tokenization': request['prepared_inputs']['tokenization'],
                'embedding_dtype': 'float32'}
    SAVED_TEXT.parent.mkdir(parents=True, exist_ok=True)
    with (SAVED_TEXT.with_suffix('.npz.tmp')).open('wb') as handle:
        np.savez_compressed(handle, ids=np.asarray(ids, dtype=str),
                            embeddings=candidates,
                            metadata=json.dumps(metadata, sort_keys=True))
    (SAVED_TEXT.with_suffix('.npz.tmp')).replace(SAVED_TEXT)
    shutil.rmtree(work, ignore_errors=True)
    print(f'[ablbm] seeded cache written; baseline vector block '
          f'{vectors.shape}', flush=True)


PROFILE_PREDICATES = ('model_tracks/', 'core/', 'graph_tracks/', 'training/')


def rank_profile(prof, out_csv: Path, top=40):
    stats = pstats.Stats(prof)
    rows = []
    for (file, line, name), (cc, nc, tottime, cumtime, callers) in stats.stats.items():
        module = file.replace(ROOT.as_posix(), '') + ':' + str(line)
        if not any(part in file for part in PROFILE_PREDICATES):
            continue
        if name.startswith('<') or 'run_colab' in file:
            continue
        rows.append((tottime, nc, cumtime, f'{module}:{name}'))
    rows.sort(reverse=True)
    with out_csv.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['self_seconds', 'ncalls', 'total_seconds', 'function'])
        for row in rows[:top]:
            writer.writerow([f'{row[0]:.3f}', row[1], f'{row[2]:.3f}', row[3]])
    return rows[:top]


volatile = {'implementation_sha256', 'composition', 'sources'}


def fingerprint_prepare(request_path: Path):
    request = json.loads(request_path.read_text())
    cleaned = {k: v for k, v in request.items() if k not in volatile}
    cleaned['prepared_inputs_plan'] = {k: v for k, v in request['prepared_inputs'].items()
                                       if k != 'sha256'}
    return sha_json(cleaned)


def fingerprint_report(path: Path):
    report = json.loads(path.read_text())
    cleaned = {k: v for k, v in report.items() if k not in volatile}
    return sha_json(cleaned)


def run_round(round_no: int, rebuild_cache=False):
    identity = ensure_inputs(rebuild_cache=rebuild_cache)
    ROUNDS.mkdir(parents=True, exist_ok=True)
    folder = ROUNDS / f'round{round_no}'
    folder.mkdir(parents=True, exist_ok=True)
    for stale in (folder/'vectors.npz', folder/'summary.json', folder/'ranking.csv',
                  folder/'timings.log', folder/'profile.prof'):
        if stale.exists():
            stale.unlink()
    import core.step_trace as step_trace
    os.environ['ER_TIMING_LOG'] = str(folder / 'timings.log')

    import torch  # noqa: E402
    torch.set_num_threads(4)
    torch.manual_seed(1729)

    from core.step_trace import trace_step
    from core.run_log import RunLogger
    logger = RunLogger('abl_timing')

    profiler = cProfile.Profile()
    from model_tracks import ablation as ablation

    idle_ok = wait_for_idle(limit=MAX_LOAD, timeout_s=WAIT_TIMEOUT_S)

    summary = {'idle_ok': idle_ok, 'load_avg_at_start': load_avg(), 'notes': REPORT_NOTE, 'cohort': '5k', 'pairs_source': str(PAIRS),
               'checkpoint_sha256': identity, 'torch_threads': 4}
    load_start = load_avg()
    started = time.perf_counter()
    profiler.enable()
    with trace_step('abl.prepare'):
        request_path = ablation.prepare(CATALOG, PAIRS, CHECKPOINT, config=CONFIG)
    with trace_step('abl.encode'):
        result = folder / 'vectors.npz'
        ablation.encode(request_path, result, device='cpu', saved_text=SAVED_TEXT)
    threshold = json.loads(THRESHOLD_STATE.read_text())['threshold']
    with trace_step('abl.report'):
        report = ablation.report(request_path, result, threshold,
                                 threshold_source=str(BINDING), config=CONFIG)
    profiler.disable()
    elapsed = time.perf_counter() - started

    top = rank_profile(profiler, folder / 'ranking.csv')
    profiler.dump_stats(folder / 'profile.prof')
    summary.update({
        'wall_seconds': round(elapsed, 3),
        'prepare_seconds': phase_seconds(folder / 'timings.log', 'abl.prepare'),
        'encode_seconds': phase_seconds(folder / 'timings.log', 'abl.encode'),
        'report_seconds': phase_seconds(folder / 'timings.log', 'abl.report'),
        'fingerprint_request': fingerprint_prepare(request_path),
        'fingerprint_report': fingerprint_report(ABL / 'results' / 'report.json'),
        'load_avg_start': load_start, 'load_avg_end': load_avg(),
    })
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    logger.info(f'round {round_no} wall={elapsed:.1f}s -> {folder}')
    return summary, top


def phase_seconds(log_path: Path, label: str) -> float | None:
    """Parse [timing] <label> state=completed elapsed_seconds=<s> lines."""
    total = 0.0
    seen = False
    for line in log_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == '[timing]' and parts[1] == label \
                and parts[2].startswith('state='):
            state = parts[2].split('=', 1)[1]
            if state == 'completed':
                seen = True
            for part in parts:
                if part.startswith('elapsed_seconds='):
                    total += float(part.split('=', 1)[1])
    return round(total, 3) if seen else None


# A hang anywhere in a round must be diagnosable: `kill -USR1 <pid>` dumps the
# Python stack of every thread. Registered here so it is armed before any work.
import faulthandler as _faulthandler
import signal as _signal

_faulthandler.enable()
try:
    _faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)
except (AttributeError, ValueError, OSError):
    pass


def load_avg() -> float:
    try:
        return float(open('/proc/loadavg').read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def wait_for_idle(limit: float = 2.0, timeout_s: int = 300, log_every_s: int = 30):
    """Other optimization lanes share this host; prefer not to time under load.

    FIXED 2026-10-08: this loop used to accept `timeout_s` and never check it,
    log nothing, and increment an unused counter — so on a box saturated by
    sibling lanes it slept forever with no output and looked like a hang
    (observed: 0.2% CPU in clock_nanosleep for 18+ minutes while holding the
    shared /tmp/opc/.abl_timing.lock, blocking every other lane).

    Now: progress is logged every iteration, `timeout_s` is honoured, and on
    timeout the round PROCEEDS under load rather than blocking indefinitely.
    The observed load is returned and recorded in summary.json, so a loaded
    round can be discounted afterwards instead of silently trusted.
    """
    start = time.monotonic()
    while load_avg() > limit:
        waited = time.monotonic() - start
        if waited >= timeout_s:
            print(f'[wait_for_idle] TIMEOUT after {waited:.0f}s '
                  f'(load={load_avg():.2f} > limit={limit}); proceeding under load — '
                  f'wall numbers from this round are NOT comparable', flush=True)
            return False
        print(f'[wait_for_idle] load={load_avg():.2f} > limit={limit}; '
              f'waited {waited:.0f}s / {timeout_s}s', flush=True)
        time.sleep(max(1, min(log_every_s, timeout_s - waited)))
    print(f'[wait_for_idle] idle after {time.monotonic() - start:.0f}s '
          f'(load={load_avg():.2f})', flush=True)
    return True


def show_rankings():
    for no in list_dirs(ROUNDS):
        folder = ROUNDS / f'round{no}'
        summary = json.loads((folder / 'summary.json').read_text())
        print(f"\nROUND {no}: wall={summary['wall_seconds']}s "
              f"prepare={summary.get('prepare_seconds')} encode={summary.get('encode_seconds')} "
              f"report={summary.get('report_seconds')}")
        ranking = (folder / 'ranking.csv').read_text().splitlines()[1:]
        print('  rank self_s ncalls total_s function')
        for rank, line in enumerate(ranking[:12], 1):
            print(f'  {rank:>4} {line}')
    for no in list_dirs(ROUNDS):
        folder = ROUNDS / f'round{no}'
        print(f"{no}\t{json.loads((folder / 'summary.json').read_text())['wall_seconds']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, default=None,
                        help='round number for artifacts; defaults to last+1')
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--rankings', action='store_true',
                        help='print existing per-round rankings summary')
    parser.add_argument('--max-load', type=float, default=None,
                        help='1-min load average to wait for before timing '
                             '(default 1.5; use a higher value on a busy host)')
    parser.add_argument('--wait-timeout', type=int, default=None,
                        help='seconds to wait for idle before proceeding anyway '
                             '(default 300; the round is then marked loaded)')
    args = parser.parse_args()
    global MAX_LOAD, WAIT_TIMEOUT_S
    if args.max_load is not None:
        MAX_LOAD = args.max_load
    if args.wait_timeout is not None:
        WAIT_TIMEOUT_S = args.wait_timeout
    if args.rankings:
        show_rankings()
        return
    if args.round is None:
        existing = list_dirs(ROUNDS)
        args.round = (existing[-1] + 1) if existing else 0
    summary, top = run_round(args.round, rebuild_cache=args.rebuild_cache)
    print('\nTOP FUNCTIONS (self seconds):')
    for rank, (tot, nc, cum, name) in enumerate(top, 1):
        print(f'  {rank:>2}. {tot:8.2f}s {nc:>8}c {cum:8.2f}s {name}')


if __name__ == '__main__':
    main()
