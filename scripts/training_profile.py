"""Wall/CPU profiling harness for the training path (measurement only).

Mirrors scripts/ablation_profile.py: binds the existing [timing] surface behind
ER_TIMING_LOG / ER_TIMING_OUT, cProfiles each profiled entry point, and writes
per-phase and per-function rankings so the optimization work on
perf/training-optimization can be measured before/after:

  training_profile/
    timings.log                 bound [timing] lines
    timings.json                structured core.timing.Timing dump (ER_TIMING_OUT)
    phase_rankings.csv          wall seconds per phase label (parsed from the log)
    <phase>.prof                raw cProfile dump
    <phase>_functions.csv       top-60 functions by tottime (ablation columns)

Faces (CPU, synthetic/smoke inputs, bounded runtime):
  - graph_inputs   : scripts.smoke_graph_tracks.build_synthetic_inputs
                     (catalog -> prepared graph -> local MiniLM text cache)
  - graph_gnn_only : graph_tracks.train.train for track=gnn_only
  - text           : the smallest runnable CPU text lane
                     (training.train_prepared._main on data/prepared/smoke_200)
  - io             : core.portable_archive.write_archive over a tiny tree

Both measurement modes are supported:
  * default            -> the optimized path (ER_PERF_* switches on)
  * --legacy / ER_PERF_LEGACY=1 -> every optimization off (pre-optimization path)

No tests, no source edits, no optimization: measurement only.
"""
from __future__ import annotations
import argparse
import cProfile
import csv
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ER_PERF_LEGACY must reach perf_switches before any src module imports it, so
# the flag is honoured here, ahead of every src-level import below.
LEGACY = os.environ.get('ER_PERF_LEGACY') == '1' or '--legacy' in sys.argv
if LEGACY:
    os.environ['ER_PERF_LEGACY'] = '1'

sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'src'))

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from core.run_log import RunLogger

_LOG = RunLogger(__name__)

PROFILE_DIR = ROOT / 'training_profile'
RESULTS_DIR = PROFILE_DIR / 'results'
#: Contain every generated artifact under training_profile/: the text lane
#: writes RESULTS/_prepared_inputs and fold metrics, and graph_tracks.train
#: honours this override for its run root.
os.environ['EUROMONITOR_RESULTS_DIR'] = str(RESULTS_DIR)
TIMING_LOG = PROFILE_DIR / 'timings.log'
#: core.timing.Timing writes a structured JSON document to ER_TIMING_OUT, so it
#: is bound to a JSON sibling; the human-readable [timing] lines land in
#: ER_TIMING_LOG (timings.log), which phase_rankings parses.
TIMING_JSON = PROFILE_DIR / 'timings.json'
GRAPH_ROOT = PROFILE_DIR / 'graph_smoke'
TEXT_BUNDLE = ROOT / 'data/prepared/smoke_200/text_prepared.pkl.gz'
TEXT_MODEL = ROOT / 'artifacts/models/all-MiniLM-L6-v2'
TIMING_LINE = re.compile(r'^\[timing\] (\S+) state=(\w+)(?: elapsed_seconds=([\d.]+))?')


class _NoopWandb:
    """W&B surface stub: the text lane must not touch the network while timing."""

    run_id = None
    run_url = None

    def log_config(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def set_summary(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass


def bind_timing_log(path: Path, json_path: Path) -> Path:
    os.environ['ER_TIMING_LOG'] = str(path)
    os.environ['ER_TIMING_OUT'] = str(json_path)
    if path.exists():
        path.unlink()
    if json_path.exists():
        json_path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reset_outputs() -> None:
    for stale in (GRAPH_ROOT, PROFILE_DIR / 'io_src', RESULTS_DIR):
        if stale.is_dir():
            shutil.rmtree(stale)
    for pattern in ('*.prof', '*_functions.csv', 'io_archive.tar.zst', 'io_archive.zip',
                    'phase_rankings.csv'):
        for stale in PROFILE_DIR.glob(pattern):
            stale.unlink()


def profiled(function, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return function(*args, **kwargs), profiler
    finally:
        profiler.disable()


def pstats_stats(profiler: cProfile.Profile):
    import pstats
    return pstats.Stats(profiler).stats


def save_profiler(profiler: cProfile.Profile, phase: str) -> None:
    profiler.dump_stats(str(PROFILE_DIR / f'{phase}.prof'))
    write_function_rankings(pstats_stats(profiler), PROFILE_DIR / f'{phase}_functions.csv')


def write_function_rankings(stats, path: Path, top: int = 60) -> list:
    fieldnames = ['file', 'line', 'function', 'calls', 'tottime_seconds', 'cumtime_seconds']
    rows = [{'file': file, 'line': line, 'function': name,
             'calls': nc, 'tottime_seconds': round(tt, 6),
             'cumtime_seconds': round(ct, 6)}
            for (file, line, name), (_cc, nc, tt, ct, _callers) in stats.items()]
    rows.sort(key=lambda row: row['tottime_seconds'], reverse=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[:top])
    return rows


def phase_seconds() -> list:
    totals = {}
    if not TIMING_LOG.is_file():
        return []
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


# --- faces ----------------------------------------------------------------

def run_graph_inputs() -> dict:
    from scripts.smoke_graph_tracks import build_synthetic_inputs
    with _LOG.section('profile.graph_inputs'):
        inputs, profiler = profiled(build_synthetic_inputs, GRAPH_ROOT, text_checkpoint=TEXT_MODEL)
    save_profiler(profiler, 'graph_inputs')
    _LOG.info(f'graph inputs ready prepared={inputs["prepared"]}')
    return inputs


def run_graph_track(track: str, inputs: dict) -> Path:
    from graph_tracks.train import train
    from scripts.smoke_graph_tracks import build_track_config
    config = build_track_config(inputs['root'], inputs['prepared'], track,
                                epochs=1, hidden_dim=8, output_dim=8,
                                postprocess=False, include_inputs=False)
    with _LOG.section(f'profile.graph_{track}'):
        checkpoint, profiler = profiled(train, config, run_tag=f'profile_{track}')
    save_profiler(profiler, f'graph_{track}')
    _LOG.info(f'graph {track} checkpoint={checkpoint}')
    return checkpoint


def _text_args():
    from types import SimpleNamespace
    from core.common import load_config
    cfg = load_config()
    tr = cfg['training']
    return SimpleNamespace(
        bundle=TEXT_BUNDLE, shared_training_data=None, training_binding=None,
        allow_unshared_supervision=True, model=str(TEXT_MODEL),
        epochs=1, lr=float(tr['lr']), train_frac=1.0, split='holdout',
        payload='full', loss=str(tr['loss']), band=str(cfg['mining']['ann']['band']),
        masking_profile=None, collapse_guardrail_profile=None,
        # The smoke bundle's run plan was frozen with sample=True; matching it
        # lets validate_run_plan relax the config digest for a lifecycle smoke.
        sample=1, device='cpu', report_test=False, resume=False, mask_effect=False,
        no_plot=True, attestation=None, run_tag='profile')


def run_text() -> None:
    import training.train_prepared as text_lane
    args = _text_args()
    with _LOG.section('profile.text_train_prepared'):
        _rows, profiler = profiled(text_lane._main, args, _NoopWandb())
    save_profiler(profiler, 'text')
    _LOG.info('text lane complete')


def run_io() -> None:
    from core.portable_archive import write_archive
    source = PROFILE_DIR / 'io_src'
    source.mkdir(parents=True, exist_ok=True)
    (source / 'a.txt').write_text('a' * 10000, encoding='utf-8')
    (source / 'b.bin').write_bytes(b'x' * 50000)
    files = {'a.txt': source / 'a.txt', 'b.bin': source / 'b.bin'}
    output = PROFILE_DIR / 'io_archive.tar.zst'
    with _LOG.section('profile.io_write_archive'):
        try:
            _result, profiler = profiled(write_archive, output, files,
                                         manifest_name='archive_manifest.json',
                                         metadata={'schema': 'training-profile-io'})
        except (ImportError, ValueError):
            # No zstandard in this runtime: fall back to the always-available ZIP path.
            output = PROFILE_DIR / 'io_archive.zip'
            _result, profiler = profiled(write_archive, output, files,
                                         manifest_name='archive_manifest.json',
                                         metadata={'schema': 'training-profile-io'})
    save_profiler(profiler, 'io')
    _LOG.info(f'io archive written {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', action='store_true',
                        help='run the pre-optimization path (sets ER_PERF_LEGACY=1 before imports)')
    parser.parse_args()
    RunLogger.configure_console()
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    bind_timing_log(TIMING_LOG, TIMING_JSON)
    reset_outputs()
    _LOG.info(f'training profiling start legacy={LEGACY} bundle={TEXT_BUNDLE.name}')

    failures: list[tuple[str, str]] = []

    def attempt(name, function, *args):
        try:
            return function(*args)
        except BaseException as error:  # a mid-edit src module must not kill every face
            import traceback
            failures.append((name, f'{type(error).__name__}: {error}'))
            _LOG.error(f'{name} failed: {type(error).__name__}: {error}')
            _LOG.error(traceback.format_exc())

    inputs = attempt('graph_inputs', run_graph_inputs)
    if inputs is None:
        _LOG.error('graph inputs unavailable; skipping graph tracks')
    else:
        attempt('graph_gnn_only', run_graph_track, 'gnn_only', inputs)
    attempt('text', run_text)
    attempt('io', run_io)

    rankings = write_phase_rankings()
    print(f'{"phase":55s} {"calls":>5s} {"wall_s":>10s}')
    for row in rankings:
        print(f'{row["phase"]:55s} {row["calls"]:5d} {row["wall_seconds"]:10.3f}')
    if failures:
        print('\n[failures]')
        for name, detail in failures:
            print(f'{name}: {detail}')


if __name__ == '__main__':
    main()
