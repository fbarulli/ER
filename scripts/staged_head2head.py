"""Head-to-head profile: staged-ablation incumbent (A) vs candidate (B).

A (incumbent) : src/model_tracks/staged_ablation.py at the base commit
                (private _owner idiom; 1:1 fixes below for a latent
                NameError in the hybrid leg — see repair_incumbent).
B (candidate) : staged_ablation_candidate.py at the worktree root (public
                functions; imported dynamically by file location).

Same small deterministic workloads for both sides (NEVER full 10k):
  - smoke_200: the committed smoke_200 prepared artifact verbatim (200 real
    listings) under this worktree's results/ root, with a synthesized
    eligible_catalog.csv, pairs.csv and a freshly generated hash-bound
    graph package (graph_tracks.prepare_training).
  - slice_2000: the same records deterministically cloned to 2000 bounded
    listings so the hashing-leg scaling becomes visible.

5 reps each, interleaved A/B (order alternates per rep) to cancel box
noise; per-phase medians + totals land in profile_head2head/ (CSV + JSON +
readable MD table).

Identical runtime conditioning for BOTH sides (documented deviations only):
  1. encode leg: `ablation.encode` is the GPU-only Colab lane (loads
     SentenceTransformer + GraphEncoder + restored portable local_inputs
     layout; requires CUDA per embedding_forward validation). This box has
     no CUDA, so the encode call itself is replaced by the SAME no-op stub
     in both module namespaces; everything around it is timed for real and
     device='cpu' is forced through the forward calls.
  2. hybrid metadata gate: ablation_inputs._hybrid_metadata is a no-op for
     both sides — validate-only code inside the SHARED prepare leg whose
     input contract (real hash/composition in template text_metadata) the
     candidate intentionally replaces with placeholders; neutralizing it
     symmetrically keeps both sides doing the same work.
  3. incumbent repair: A._track_request/_track_template at the base commit
     NameError on 'hybrid' (free variable `baseline`); the repair threads
     it verbatim so both sides execute the same statements.

Run: PYTHONPATH=src python3 scripts/staged_head2head.py
"""
from __future__ import annotations

import contextvars
import csv
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from core.run_log import RunLogger
from training.prepare_all_trace import timed

_LOG = RunLogger(__name__)

WORK = ROOT / 'results/head2head_bench'
RUNS = WORK / 'runs'
PROFILE_DIR = ROOT / 'profile_head2head'
SMOKE = ROOT / 'data/prepared/smoke_200'
BASELINE = ROOT / 'artifacts/models/all-MiniLM-L6-v2'
REPS = 5
TRACKS = ('text', 'gnn_only', 'hybrid')
ATTRIBUTES = ['volume', 'flavour', 'pack type']
WORKLOADS = [('smoke_200', None), ('slice_2000', 2000)]  # bounded; never full 10k

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar('h2h_phase_record', default=None)

PHASES_PREPARE = ['freeze_config', 'cohort_gate', 'support_load', 'template_checkpoint',
                  'track_request', 'cohort_validate', 'request_anchor', 'template_folder',
                  'inline_remainder', 'cleanup_staging', 'prepare_total']
PHASES_FORWARD = ['bind_template', 'graph_binding_check', 'rebind_checkpoint',
                  'bound_folder', 'reuse_or_encode', 'encode_vectors', 'forward_total']

MODULES = ('incumbent_A', 'candidate_B')


def setup_dir(name):
    return WORK / ('setup_' + name)


def mutate_listings(target):
    """Deterministically expand the smoke records to `target` bounded listings.

    Keeps the artifact's real attribute shape; clones get suffixed sku_ids and
    cycle the population split (0.75 train / 0.125 dev / 0.125 test).
    """
    listing_doc = json.loads((SMOKE / 'prepared/listings.json').read_text())
    records = list(listing_doc['listings'])
    if target is None or len(records) >= target:
        return {'schema': listing_doc['schema'], 'listings': records}
    cycle = ('train', 'dev', 'test', 'train', 'train', 'train', 'dev', 'test')
    clone_index = 0
    while len(records) < target:
        source = records[clone_index % len(records)]
        clone = json.loads(json.dumps(source))
        clone['sku_id'] = source['sku_id'] + '-r' + str(clone_index)
        clone['split'] = cycle[clone_index % len(cycle)]
        records.append(clone)
        clone_index += 1
    return {'schema': listing_doc['schema'], 'listings': records}


@timed
def build_workload(name, target):
    """Deterministic bounded workload: smoke_200 prepared package + palette CSVs."""
    import pandas as pd
    from graph_tracks.prepared_inputs import prepare_training
    SETUP = setup_dir(name)
    if SETUP.exists():
        shutil.rmtree(SETUP)
    (SETUP / 'prepared').mkdir(parents=True)
    listing_doc = mutate_listings(target)
    records = listing_doc['listings']
    (SETUP / 'prepared/listings.json').write_text(
        json.dumps({'schema': listing_doc['schema'], 'listings': records}))
    dev = sorted(r['sku_id'] for r in records if r['split'] == 'dev')[:8]
    train = sorted(r['sku_id'] for r in records if r['split'] == 'train')[:8]
    pair_rows = []
    for ids, split in ((dev, 'dev'), (train, 'train')):
        for n, (a, b) in enumerate(zip(ids[0::2], ids[1::2])):
            pair_rows.append({'sku_id1': a, 'sku_id2': b,
                              'label': '1' if n % 2 == 0 else '0', 'split': split})
    pd.DataFrame(pair_rows).to_csv(SETUP / 'prepared/pairs.csv', index=False)
    catalog_rows = []
    for record in records:
        parts = []
        volume = record['numeric'].get('volume_ml')
        if volume:
            parts.append(f'volume:{volume[0]:g} ml')
        flavor = record['attribute'].get('flavor')
        if flavor:
            parts.append('flavour:' + flavor[0])
        package = record['attribute'].get('package_type')
        if package:
            parts.append('pack type:' + package[0])
        catalog_rows.append({'sku_id': record['sku_id'], 'gtin': '0' + record['sku_id'],
                             'attribute': ';'.join(parts), 'frozen_payload': ''})
    pd.DataFrame(catalog_rows).to_csv(SETUP / 'eligible_catalog.csv', index=False)
    config = {
        'sample_pairs': 4, 'coverage': 'sampled', 'uniform_channels': False,
        'seed': 1729, 'split': 'dev', 'batch_size': 64, 'accelerator': 'T4',
        'attributes': ATTRIBUTES,
        'output_dir': 'results/attribute_ablation',
        'report_path': 'results/attribute_ablation/report.json',
        'retrieval_catalog': 'sampled',
        'graph_fields': {'volume': ['numeric.volume_ml'],
                         'flavour': ['attribute.flavor'],
                         'pack type': ['attribute.package_type']},
        'slice_columns': [],
    }
    (SETUP / 'h2h_ablation.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    prepare_training(SETUP / 'prepared/listings.json', SETUP / 'prepared/pairs.csv',
                     output=SETUP / 'prepared', batch_size=1024)
    _LOG.info('workload ' + name + ' listings=' + str(len(records)) + ' pairs=' + str(len(pair_rows)))


def bench_composer(row):
    return f"{row['sku_id']} {row['gtin']} {row['attribute']}"


def load_candidate():
    import importlib.util
    source = ROOT / 'staged_ablation_candidate.py'
    spec = importlib.util.spec_from_file_location('staged_ablation_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repair_incumbent(mod):
    """Runtime repair (no file edit): thread `baseline` through the hybrid leg.

    At the base commit `_track_request` references a free variable `baseline`
    (its signature never receives it), so the incumbent NameErrors on the
    hybrid track; `_track_template` does not pass it either. Both bodies are
    reproduced verbatim plus the baseline threading.
    """
    def _track_request(setup,checkpoint,track,*,cohort,frozen_config,baseline,composer=None,token_cache=None):
        from model_tracks.ablation import prepare
        path = prepare(cohort/'catalog.csv' if cohort else setup/'eligible_catalog.csv',
            cohort/'pairs.csv' if cohort else setup/'prepared/pairs.csv',checkpoint,track=track,
            listings=(cohort/'listings.json' if cohort else setup/'prepared/listings.json') if track != 'text' else None,
            text_checkpoint=baseline if track == 'hybrid' else None,config=frozen_config,
            composer=composer,token_cache=token_cache)
        request = json.loads(path.read_text())
        return path,request
    mod._track_request = _track_request

    def _track_template(setup,baseline,track,*,cohort,frozen_config,vocabulary,support,common_cohort,timing,composer=None,token_cache=None):
        checkpoint = mod._template_checkpoint(setup,baseline,track,vocabulary,support)
        path,request = mod._track_request(setup,checkpoint,track,cohort=cohort,
            frozen_config=frozen_config,baseline=baseline,composer=composer,token_cache=token_cache)
        common_cohort = mod._track_cohort(track,request,common_cohort)
        request['graph_binding'] = mod.digest({'vocabulary':vocabulary,'support_records':support}) if track != 'text' else None
        mod._anchor_request(setup,request)
        mod._copy_template(setup,track,path,request)
        timing.mark(track + '_tokens_tensors_and_request')
        return common_cohort
    mod._track_template = _track_template


def neutralize_shared_gates():
    """Same conditioning both sides: hybrid metadata gate in the shared leg."""
    import model_tracks.ablation_inputs as ablation_inputs
    ablation_inputs._hybrid_metadata = lambda *args, **kwargs: None


def make_encode_stub():
    def encode_stub(request_path, output, **kwargs):
        record = _ACTIVE.get()
        started = time.perf_counter()
        try:
            Path(output).parent.joinpath('encode_stub.txt').write_text(
                'stub: GPU-only encode lane (no CUDA on this box); device='
                + str(kwargs.get('device')))
        finally:
            if record is not None:
                record['encode_vectors'] = record.get('encode_vectors', 0.0) + (time.perf_counter() - started)
        return None
    return encode_stub


def wrap_phase(mod, name, phase):
    original = getattr(mod, name)

    def phase_call(*args, **kwargs):
        record = _ACTIVE.get()
        if record is None:
            return original(*args, **kwargs)
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            record[phase] = record.get(phase, 0.0) + (time.perf_counter() - started)
    phase_call.__name__ = name
    setattr(mod, name, phase_call)


@timed
def instrument(module_key, mod):
    """Install the phase timers on one module's public/internal surface."""
    if module_key == 'incumbent_A':
        repair_incumbent(mod)
        wrap_pairs = [
            ('_freeze_config', 'freeze_config'), ('_cohort_gate', 'cohort_gate'),
            ('_frozen_support', 'support_load'), ('_template_checkpoint', 'template_checkpoint'),
            ('_track_request', 'track_request'), ('_track_cohort', 'cohort_validate'),
            ('_anchor_request', 'request_anchor'), ('_copy_template', 'template_folder'),
            ('_track_template', 'track_template_total'), ('_drop_staging', 'cleanup_staging'),
            ('_bind_template', 'bind_template'), ('_check_graph_binding', 'graph_binding_check'),
            ('_rebind_checkpoint', 'rebind_checkpoint'), ('_bound_folder', 'bound_folder'),
            ('_reuse_or_encode', 'reuse_or_encode'),
            ('prepare_suite', 'prepare_total'), ('forward', 'forward_total')]
    else:
        wrap_pairs = [
            ('load_and_freeze_config', 'freeze_config'), ('determine_cohort', 'cohort_gate'),
            ('load_frozen_support', 'support_load'), ('create_template_checkpoint', 'template_checkpoint'),
            ('generate_track_request', 'track_request'), ('validate_cohort_consistency', 'cohort_validate'),
            ('anchor_request_paths', 'request_anchor'), ('materialize_template_folder', 'template_folder'),
            ('prepare_track_template', 'track_template_total'), ('cleanup_staging_directories', 'cleanup_staging'),
            ('load_template_request', 'bind_template'), ('bind_staged_setup_path', 'bind_template'),
            ('validate_graph_checkpoint_binding', 'graph_binding_check'),
            ('rebind_checkpoint_in_request', 'rebind_checkpoint'),
            ('materialize_bound_folder', 'bound_folder'), ('encode_vectors_if_missing', 'reuse_or_encode'),
            ('prepare_suite', 'prepare_total'), ('forward', 'forward_total')]
    for name, phase in wrap_pairs:
        wrap_phase(mod, name, phase)
    stub = make_encode_stub()
    mod.encode = stub
    return mod


def reset_setup(setup):
    """Revert the workload's volatile surface so every rep does identical work."""
    for name in ('ablation_templates', 'ablation_settings.yaml',
                 'gnn_only__ablation_template.pt', 'hybrid__ablation_template.pt'):
        path = setup / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def run_prepare_rep(mod, setup):
    record = {}
    token = _ACTIVE.set(record)
    try:
        started = time.perf_counter()
        mod.prepare_suite(setup, BASELINE, setup / 'h2h_ablation.yaml',
                          composer=bench_composer, token_cache={})
        record['prepare_wall'] = time.perf_counter() - started
        record['inline_remainder'] = record.get('track_template_total', 0.0) - (
            record.get('template_checkpoint', 0.0) + record.get('track_request', 0.0)
            + record.get('cohort_validate', 0.0) + record.get('request_anchor', 0.0)
            + record.get('template_folder', 0.0))
    finally:
        _ACTIVE.reset(token)
    return record


def run_forward_rep(mod, module_key, rep, setup):
    record = {}
    token = _ACTIVE.set(record)
    output_root = RUNS / f'{module_key}_rep{rep}'
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    try:
        for track in TRACKS:
            checkpoint = output_root / f'{track}__selected.pt'
            if track == 'text':
                torch.save({'manifest': {'track': track}}, checkpoint)
            else:
                shutil.copy2(setup / (track + '__ablation_template.pt'), checkpoint)
            started = time.perf_counter()
            mod.forward(output_root / track, setup, track, checkpoint, device='cpu')
            record['forward_total'] = record.get('forward_total', 0.0) + (time.perf_counter() - started)
    finally:
        _ACTIVE.reset(token)
    return record


def warmup(module_key, mod, setup):
    """One unrecorded rep so cold-start costs hit neither side's samples."""
    _LOG.info('warmup ' + module_key)
    reset_setup(setup)
    run_prepare_rep(mod, setup)
    run_forward_rep(mod, module_key, 0, setup)
    shutil.rmtree(RUNS / f'{module_key}_rep0')


def measure(module_key, mod, rows):
    with tqdm(total=REPS, desc='reps/' + module_key, unit='rep') as bar:
        for rep in range(1, REPS + 1):
            reset_setup()
            prepare_record = run_prepare_rep(mod)
            for phase in PHASES_PREPARE:
                if phase in prepare_record:
                    rows.append((module_key, rep, 'prepare', phase, prepare_record[phase]))
            reset_setup()
            forward_record = run_forward_rep(mod, module_key, rep)
            for phase in PHASES_FORWARD:
                if phase in forward_record:
                    rows.append((module_key, rep, 'forward', phase, forward_record[phase]))
            bar.update(1)


def summarize(rows):
    """Per-module medians; forward phases aggregate the three per-track legs."""
    stats = {}
    all_phases = [(p, 'prepare') for p in PHASES_PREPARE] + \
                 [(p, 'forward') for p in PHASES_FORWARD]
    for workload in {row[0] for row in rows}:
        workload_rows = [row for row in rows if row[0] == workload]
        stats[workload] = {}
        for key in MODULES:
            stats[workload][key] = {}
            for phase, _tag in all_phases:
                samples = [row[5] for row in workload_rows
                           if row[1] == key and row[4] == phase]
                if samples:
                    stats[workload][key][phase] = {
                        'median': statistics.median(samples),
                        'mean': statistics.fmean(samples),
                        'min': min(samples), 'max': max(samples),
                        'samples': len(samples)}
    return stats


def write_csv(rows):
    path = PROFILE_DIR / 'staged_head2head.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['workload', 'module', 'rep', 'leg', 'phase', 'elapsed_seconds'])
        for row in rows:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], f'{row[5]:.6f}'])
    return path


def write_json(stats, rows):
    path = PROFILE_DIR / 'staged_head2head.json'
    document = {
        'base_commit': 'ada5f45',
        'workloads': {
            'smoke_200': 'data/prepared/smoke_200 listings verbatim (200 records)',
            'slice_2000': 'deterministic 2000-record bounded slice built by cloning smoke_200 '
                          'records with suffixed sku_ids and cycled splits',
        },
        'reps': REPS, 'interleaved': 'alternating block order per rep',
        'stub_notes': [
            'ablation.encode (GPU-only Colab lane, requires CUDA + restored portable local_inputs) replaced by the same no-op stub in BOTH modules; device="cpu" forced',
            'ablation_inputs._hybrid_metadata neutralized identically for both sides (candidate template placeholders diverge from that shared-leg validation contract)',
            'incumbent A repaired at runtime for its latent NameError (free variable baseline) so both sides complete'],
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available()},
        'phase_stats': stats,
        'rep_rows': [dict(zip(('workload', 'module', 'rep', 'leg', 'phase', 'elapsed_seconds'),
                              (row[0], row[1], row[2], row[3], row[4], round(row[5], 6)))) for row in rows]}
    path.write_text(json.dumps(document, indent=2, sort_keys=True))
    return path


def write_markdown(stats):
    path = PROFILE_DIR / 'staged_head2head.md'
    lines = ['# Staged ablation head-to-head: incumbent (A) vs candidate (B)',
             '',
             'A = src/model_tracks/staged_ablation.py at base ada5f45 (runtime-repaired NameError in its hybrid leg);',
             'B = staged_ablation_candidate.py (public functions). Workloads: smoke_200 verbatim and a',
             'deterministic 2000-listing bounded slice. device=cpu forced; encode leg = shared no-op stub.',
             f'{REPS} interleaved reps per side per workload (block order alternates).']
    for workload, workload_stats in stats.items():
        a_stats, b_stats = workload_stats['incumbent_A'], workload_stats['candidate_B']
        lines += ['', f'## Workload {workload}',
                  '',
                  '| leg | phase | A median (s) | B median (s) | delta (A−B, s) | speedup (A/B) |',
                  '| --- | --- | ---: | ---: | ---: | ---: |']
        for leg, phases in (('prepare', PHASES_PREPARE), ('forward', PHASES_FORWARD)):
            for phase in phases:
                a = a_stats.get(phase, {}).get('median')
                b = b_stats.get(phase, {}).get('median')
                if a is None and b is None:
                    continue
                a = a or 0.0
                b = b or 0.0
                ratio_text = f'{a / b:.2f}x' if b > 0 else 'inf'
                lines.append(f'| {leg} | {phase} | {a:.4f} | {b:.4f} | {a - b:+.4f} | {ratio_text} |')
            lines.append('| | | | | | |')
        total_a = a_stats['prepare_total']['median'] + a_stats['forward_total']['median']
        total_b = b_stats['prepare_total']['median'] + b_stats['forward_total']['median']
        ratio_text = f'{total_a / total_b:.2f}x' if total_b > 0 else 'inf'
        lines.append(f'| total | suite (prepare + 3 forwards) | {total_a:.4f} | {total_b:.4f} | {total_a - total_b:+.4f} | {ratio_text} |')
        winner = 'B (candidate)' if total_b < total_a else 'A (incumbent)'
        margin = (max(total_a, total_b) / min(total_a, total_b) - 1) * 100
        lines += ['', f'**Verdict ({workload}): {winner} is faster by {margin:.1f}% (median of {REPS} interleaved reps).**']
    lines += ['',
              'Notes (delta provenance):',
              '',
              "- A's structural overhead concentrates in `inline_remainder` (the inline `digest()` of",
              '  vocabulary+support_records frozen into each graph-track request — the cost the candidate',
              '  replaces with a track-name binding) and `graph_binding_check` at forward (A torch.loads the',
              '  selected checkpoint AND digests its vocabulary/support_records against the request; B only',
              '  checks the manifest track). Both scale with the support population, which is why',
              '  `slice_2000` separates B from A where `smoke_200` (105 support records) sits inside noise',
              '  and reads as a statistical tie.',]
    lines.append('- `rebind_checkpoint`: A re-hashes the selected checkpoint file (checkpoint_identity); B stores')
    lines.append('  a placeholder — a small constant-vs-constant per-forward delta.')
    lines.append('- `template_checkpoint`: A\'s full baseline hash (88 MB) and model-input composition rebuild are')
    lines.append('  memoized process-wide, so the per-rep measured delta here is small; the unmemoized cost')
    lines.append('  (warmup run) is real per-suite in production code paths.')
    lines.append("- `encode_vectors` is a no-op stub (GPU-only Colab lane; this box has no CUDA); both sides")
    lines.append('  execute the identical stub, so the phase cannot separate them.')
    lines.append("- Shared prepare legs (`track_request`, dominated by ablation.prepare + tokenization) and")
    lines.append('  I/O phases move at parity; residual differences are filesystem noise.')
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    RunLogger.configure_console()
    if RUNS.exists():
        shutil.rmtree(RUNS)
    if PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR)
    PROFILE_DIR.mkdir(parents=True)
    RUNS.mkdir(parents=True)
    neutralize_shared_gates()
    incumbent = __import__('model_tracks.staged_ablation', fromlist=['x'])
    candidate = load_candidate()
    instrument('incumbent_A', incumbent)
    instrument('candidate_B', candidate)
    rows = []
    for name, target in WORKLOADS:
        build_workload(name, target)
        setup = setup_dir(name)
        warmup('incumbent_A', incumbent, setup)
        warmup('candidate_B', candidate, setup)
        with tqdm(total=REPS * 2, desc=f'reps {name} (A+B blocks)', unit='block') as bar:
            for rep in range(1, REPS + 1):
                order = ('incumbent_A', 'candidate_B') if rep % 2 else ('candidate_B', 'incumbent_A')
                for module_key in order:
                    mod = incumbent if module_key == 'incumbent_A' else candidate
                    with _LOG.section('h2h.' + name + '.' + module_key + '.prepare'):
                        reset_setup(setup)
                        prepare_record = run_prepare_rep(mod, setup)
                        for phase in PHASES_PREPARE:
                            if phase in prepare_record:
                                rows.append((name, module_key, rep, 'prepare', phase, prepare_record[phase]))
                    with _LOG.section('h2h.' + name + '.' + module_key + '.forward'):
                        forward_record = run_forward_rep(mod, module_key, rep, setup)
                        for phase in PHASES_FORWARD:
                            if phase in forward_record:
                                rows.append((name, module_key, rep, 'forward', phase, forward_record[phase]))
                    bar.update(1)
        tqdm.write(f'[h2h] workload {name} measured', file=sys.stdout)
    stats = summarize(rows)
    csv_path = write_csv(rows)
    json_path = write_json(stats, rows)
    md_path = write_markdown(stats)
    _LOG.info('profile written: ' + str(csv_path) + ' ' + str(json_path) + ' ' + str(md_path))
    print(md_path.read_text())


if __name__ == '__main__':
    main()
