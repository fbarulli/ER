"""Profile the repo-root ablation.py replacement on real small data.

Runs the three public faces (prepare -> encode -> report) once each over the
smoke_200 real cohort (200 real catalog rows, bounded to the 10 dev pairs,
6 declared attributes, sampled candidate catalog) and ranks wall seconds:

  - per phase, from the [timing] lines bound behind ER_TIMING_OUT
  - per function, from one cProfile capture per phase (tottime ranking)

Every artifact lands under ablation_profile/: rankings CSVs, .prof files,
and the bound timings.log. CPU-only (encode device='cpu'); no test runs,
no optimization — measurement only.
"""
from __future__ import annotations
import cProfile
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'src'))

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import ablation
from core.common import report_thresholds
from core.run_log import RunLogger
from training.prepare_all_trace import timed

_LOG = RunLogger(__name__)

PROFILE_DIR = ROOT / 'ablation_profile'
CONFIG_PATH = PROFILE_DIR / 'config.yaml'
CATALOG = ROOT / 'data/prepared/smoke_200/eligible_catalog.csv'
PAIRS = ROOT / 'data/prepared/smoke_200/listing_pairs.csv'
CHECKPOINT = ROOT / 'artifacts/models/all-MiniLM-L6-v2'
ENCODE_OUTPUT = PROFILE_DIR / 'encode_result.npz'
THRESHOLD_SOURCE = PROFILE_DIR / 'threshold_source.json'
TIMING_LOG = PROFILE_DIR / 'timings.log'
TIMING_LINE = re.compile(r'^\[timing\] (\S+) state=(\w+)(?: elapsed_seconds=([\d.]+))?')


def bind_timing_log(path: Path) -> Path:
    os.environ['ER_TIMING_LOG'] = str(path)
    os.environ['ER_TIMING_OUT'] = str(path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reset_outputs() -> None:
    import shutil
    for stale in (ENCODE_OUTPUT, PROFILE_DIR / 'run', PROFILE_DIR / 'run_report'):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()


def profiled(function, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return function(*args, **kwargs), profiler
    finally:
        profiler.disable()


@timed
def run_prepare() -> Path:
    request_path, profiler = profiled(
        ablation.prepare, CATALOG, PAIRS, CHECKPOINT,
        track='text', config=CONFIG_PATH)
    save_profiler(profiler, 'prepare')
    return request_path


@timed
def run_encode(request_path: Path) -> Path:
    _result, profiler = profiled(
        ablation.encode, request_path, ENCODE_OUTPUT, device='cpu')
    save_profiler(profiler, 'encode')
    return ENCODE_OUTPUT


@timed
def run_report(request_path: Path, result: Path) -> Path:
    threshold = float(report_thresholds()[0])
    pointer, profiler = profiled(
        ablation.report, request_path, result, threshold,
        threshold_source=THRESHOLD_SOURCE, config=CONFIG_PATH)
    save_profiler(profiler, 'report')
    return pointer


def save_profiler(profiler: cProfile.Profile, phase: str) -> None:
    profiler.dump_stats(str(PROFILE_DIR / f'{phase}.prof'))
    stats = pstats_stats(profiler)
    write_function_rankings(stats, PROFILE_DIR / f'{phase}_functions.csv')


def pstats_stats(profiler: cProfile.Profile):
    import pstats
    return pstats.Stats(profiler).stats


def write_function_rankings(stats, path: Path, top: int = 60) -> list:
    rows = [{'file': file, 'line': line, 'function': name,
             'calls': nc, 'tottime_seconds': round(tt, 6),
             'cumtime_seconds': round(ct, 6)}
            for (file, line, name), (_cc, nc, tt, ct, _callers) in stats.items()]
    rows.sort(key=lambda row: row['tottime_seconds'], reverse=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows[:top])
    return rows


def phase_seconds() -> list:
    totals = {}
    for line in TIMING_LOG.read_text(encoding='utf-8').splitlines():
        match = TIMING_LINE.match(line)
        if not match or match.group(2) != 'completed' or match.group(3) is None:
            continue
        label, elapsed = match.group(1), float(match.group(3))
        count, total = totals.get(label, (0, 0.0))
        totals[label] = (count + 1, total + elapsed)
    return [{'phase': label, 'calls': count, 'wall_seconds': round(total, 3)}
            for label, (count, total) in totals.items()]


def write_phase_rankings() -> list:
    rows = sorted(phase_seconds(), key=lambda row: row['wall_seconds'], reverse=True)
    with (PROFILE_DIR / 'phase_rankings.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['phase', 'calls', 'wall_seconds'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    RunLogger.configure_console()
    bind_timing_log(TIMING_LOG)
    reset_outputs()
    _LOG.info(f'ablation profiling start config={CONFIG_PATH} catalog={CATALOG.name} pairs={PAIRS.name}')
    request_path = run_prepare()
    _LOG.info(f'prepare done request={request_path}')
    result = run_encode(request_path)
    _LOG.info(f'encode done result={result}')
    pointer = run_report(request_path, result)
    _LOG.info(f'report done pointer={pointer}')
    rankings = write_phase_rankings()
    print(f'{"phase":55s} {"calls":>5s} {"wall_s":>10s}')
    for row in rankings:
        print(f'{row["phase"]:55s} {row["calls"]:5d} {row["wall_seconds"]:10.3f}')


if __name__ == '__main__':
    main()
