"""Ablation-lane E (baseline) timing harness (reusable across optimization rounds).

Workload — baseline_ablation.complete() on a frozen smoke_500 suite fixture
(deterministic, byte-reproducible, identical every round, CPU-only):

  1. Once (fixtures under artifacts/abl_opt/baseline/, cached byte-exactly):
     - eligible_catalog.csv  = first 500 reviewed rows of dataset_10k.csv
       after dropping empty/duplicate sku_id, sorted stably by sku_id
       (smoke_500; NEVER the full 10k workload).
     - prepared/             = graph_tracks.prepare package (listings.json,
       pairs.csv, manifest; graph tensors not needed by the text lane).
     - ablation_templates/text = the staged text template (tokens/tensors +
       anchored request) for the real MiniLM checkpoint.
     - shared_minilm__embeddings.npz = one-time full CPU encode, saved-text
       cache with ids=candidate catalog ids (per-suite contract).
  2. Per round: 3 consecutive complete() calls (suite session: cold iter 0 +
     warm iters 1-2, each into its own fresh output folder), device='cpu'.

Deterministic cohort rule (recorded, fixed forever):
  catalog = first 500 reviewed, unique-sku_id rows of dataset_10k.csv sorted
  by sku_id. Rows sharing a nonempty gtin form duplicate-product groups
  (sorted by gtin); group 0 is dev, the rest train (positive pairs =
  consecutive gtin matches, label 1). Negative pairs = consecutive sorted
  neighbors with differing gtins (label 0), 32 per split. coverage='sampled'
  sample_pairs=32 split='dev'; slice_columns=[]; retrieval_catalog='full';
  real MiniLM checkpoint.

Outputs per round in artifacts/abl_opt/rounds/round<N>/:
  timings.log ([timing] lines behind ER_TIMING_LOG), ranking.csv (cProfile
  per-function wall ranking) plus ranking_lane.csv restricted to
  src/model_tracks/baseline_ablation.py, profile.prof, summary.json (walls +
  fingerprints). fixtures under artifacts/abl_opt/baseline/ carry their own
  fixtures.json fingerprint.
"""
from __future__ import annotations


ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
import argparse  # noqa: E402
import cProfile  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

SRC = str(ROOT / 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))
os.environ.setdefault('ER_TIMING_TRACE_PREFIX', 'abl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import re  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ABL = ROOT / 'artifacts' / 'abl_opt'
SUITE = ABL / 'baseline'
ROUNDS = ABL / 'rounds'
CATALOG_N = 500
ITERATIONS = 3
SAMPLE_PAIRS = 32
CHECKPOINT = ROOT / 'artifacts' / 'models' / 'all-MiniLM-L6-v2'
SHARED = SUITE / 'shared_minilm__embeddings.npz'
COHORT_NOTE = ('baseline smoke_500 cohort recorded 2026-10-07: catalog = first 500 '
               'reviewed unique-sku_id rows of dataset_10k.csv sorted by sku_id; '
               'gtin duplicate-product groups (sorted by gtin), group 0 -> dev '
               'with the rest -> train; positive pairs are consecutive gtin '
               'matches (label 1), negatives are consecutive sorted neighbors '
               'with differing gtins (label 0), 32 per split; sampled '
               'sample_pairs=32 split=dev; slice_columns=[]; '
               'retrieval_catalog=full; real MiniLM checkpoint')
PROFILE_PREDICATES = ('model_tracks/', 'core/', 'graph_tracks/', 'training/')
LANE_FILE = 'baseline_ablation'
VOLATILE_REPORT_KEYS = {'implementation_sha256', 'composition', 'sources',
                        'request_path', 'result_path', 'result_sha256',
                        'threshold_source'}


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


# ------------------------------------------------------------------ fixtures


def checkpoint_identity() -> str:
    from graph_tracks.text_cache import checkpoint_hash
    return checkpoint_hash(CHECKPOINT)


def build_config() -> Path:
    path = SUITE / 'template_config.yaml'
    if path.exists():
        return path
    from model_tracks.ablation import Settings
    settings = Settings.model_validate({
        'accelerator': 'T4', 'attributes': [], 'batch_size': 64,
        'coverage': 'sampled', 'graph_fields': {}, 'slice_columns': [],
        'retrieval_ks': [1, 5, 10], 'sample_pairs': SAMPLE_PAIRS, 'seed': 1729,
        'split': 'dev', 'uniform_channels': True,
        'output_dir': str(SUITE / 'ablation_templates'),
        'report_path': str(SUITE / 'report.json'),
        'retrieval_catalog': 'full'})
    SUITE.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings.model_dump(), sort_keys=True))
    return path


def build_catalog() -> Path:
    catalog = SUITE / 'eligible_catalog.csv'
    if catalog.exists():
        return catalog
    import pandas as pd
    from core.identity_policy import reviewed_row_mask
    frame = pd.read_csv(ROOT / 'dataset_10k.csv', dtype=str, keep_default_na=False,
                        low_memory=False)
    frame = frame[~reviewed_row_mask(frame)]
    frame = frame[frame.sku_id != ''].drop_duplicates('sku_id')
    frame = frame.sort_values('sku_id', kind='stable').head(CATALOG_N)
    SUITE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(catalog, index=False)
    return catalog


def build_pairs(catalog: Path):
    import pandas as pd
    pairs_raw = SUITE / 'pairs_raw.csv'
    splits_csv = SUITE / 'listing_splits.csv'
    if pairs_raw.exists() and splits_csv.exists():
        return splits_csv, pairs_raw
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False)
    records = [dict(row) for _, row in frame.iterrows()]
    groups: dict[str, list[int]] = {}
    for n, record in enumerate(records):
        if record.get('gtin'):
            groups.setdefault(record['gtin'], []).append(n)
    outline = sorted((sorted(members) for members in groups.values() if len(members) >= 2))
    if len(outline) < 2:
        raise SystemExit('smoke slice has fewer than 2 gtin duplicate-product '
                         'groups; dev/train cannot both carry positive labels')
    split = [''] * len(records)
    for label, members in enumerate(outline):
        target = 'dev' if label == 0 else 'train'
        for member in members:
            split[member] = target
    for n in range(len(records)):
        if not split[n]:
            split[n] = 'dev' if n % 8 == 0 else 'train'
    annotated = [dict(record, split=value) for record, value in zip(records, split)]
    pd.DataFrame({'sku_id': [record['sku_id'] for record in annotated],
                  'split': [record['split'] for record in annotated]}).to_csv(splits_csv, index=False)
    by_split: dict[str, list[int]] = {'dev': [], 'train': []}
    for n, record in enumerate(annotated):
        if record['split'] in by_split:
            by_split[record['split']].append(n)
    pairs = []
    for label_frame in ('dev', 'train'):
        for group in outline:
            in_split = [m for m in group if annotated[m]['split'] == label_frame]
            for a, b in zip(in_split, in_split[1:]):
                pairs.append({'sku_id1': annotated[a]['sku_id'], 'sku_id2': annotated[b]['sku_id'],
                              'label': '1', 'split': label_frame})
        chosen = 0
        for a, b in zip(by_split[label_frame], by_split[label_frame][1:]):
            if chosen >= SAMPLE_PAIRS:
                break
            gtin_a, gtin_b = annotated[a].get('gtin', ''), annotated[b].get('gtin', '')
            if gtin_a and gtin_a == gtin_b:
                continue
            pairs.append({'sku_id1': annotated[a]['sku_id'], 'sku_id2': annotated[b]['sku_id'],
                          'label': '0', 'split': label_frame})
            chosen += 1
    for required in ('dev', 'train'):
        labels = {pair['label'] for pair in pairs if pair['split'] == required}
        if labels != {'0', '1'}:
            raise SystemExit('smoke pairs need both labels in ' + required)
    pd.DataFrame(pairs).to_csv(pairs_raw, index=False)
    return splits_csv, pairs_raw


def build_graph_package(catalog: Path, splits_csv: Path, pairs_raw: Path) -> Path:
    package = SUITE / 'prepared'
    if (package / 'listings.json').exists():
        return package
    from graph_tracks.prepare import prepare as graph_prepare
    graph_prepare(catalog, splits_csv, pairs_raw, package, training_tensors=False)
    return package


def build_template() -> Path:
    template = SUITE / 'ablation_templates' / 'text'
    request = template / 'request.json'
    if request.exists():
        return template
    import model_tracks.ablation as ablation
    import model_tracks.staged_ablation as staged
    cfg = ablation.settings(SUITE / 'template_config.yaml')
    cfg.output_dir = str(SUITE / 'ablation_templates')
    frozen = SUITE / 'ablation_settings.yaml'
    frozen.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=True))
    path, staged_request = staged._track_request(
        SUITE, CHECKPOINT, 'text', cohort=None, frozen_config=frozen)
    staged_request['graph_binding'] = None
    staged._anchor_request(SUITE, staged_request)
    staged._copy_template(SUITE, 'text', path, staged_request)
    staged._drop_staging(SUITE)
    return template


def build_shared_embeddings(template: Path):
    if SHARED.exists():
        return
    import model_tracks.ablation as ablation
    import model_tracks.staged_ablation as staged
    from core.model_input import model_input_composition
    from graph_tracks.text_cache import texts_hash
    staging = ABL / 'shared_staging.npz'
    bound = template / 'request_bound.json'
    if staging.exists():
        staging.unlink()
    payload = json.loads((template / 'request.json').read_text())
    staged._bind_staged_setup(SUITE, payload)
    bound.write_text(json.dumps(payload))
    ablation.encode(bound, staging, device='cpu')
    with np.load(staging, allow_pickle=False) as data:
        candidates = np.asarray(data['candidate_vectors'], dtype=np.float32)
    metadata = {'checkpoint_sha256': checkpoint_identity(),
                'composition': model_input_composition().model_dump(mode='json'),
                'text_sha256': texts_hash([payload['texts'][i]
                                           for i in payload['candidate_text_indices']]),
                'tokenization': payload['prepared_inputs']['tokenization'],
                'embedding_dtype': 'float32'}
    SHARED.parent.mkdir(parents=True, exist_ok=True)
    tmp = SHARED.with_suffix('.npz.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, ids=np.asarray(payload['candidate_ids'], dtype=str),
                            embeddings=candidates,
                            metadata=json.dumps(metadata, sort_keys=True))
    tmp.replace(SHARED)
    staging.unlink()
    bound.unlink()


def build_suite() -> dict:
    fixture_digest = SUITE / 'fixtures.json'
    build_config()
    if not (SUITE / 'prepared' / 'listings.json').exists():
        catalog = build_catalog()
        splits_csv, pairs_raw = build_pairs(catalog)
        build_graph_package(catalog, splits_csv, pairs_raw)
    template = build_template()
    build_shared_embeddings(template)
    if not fixture_digest.exists():
        fixture_digest.write_text(json.dumps(
            {'cohort_note': COHORT_NOTE,
             'catalog_sha256': sha_bytes((SUITE / 'eligible_catalog.csv').read_bytes()),
             'listings_sha256': sha_bytes((SUITE / 'prepared' / 'listings.json').read_bytes()),
             'template_request_sha256': sha_bytes((template / 'request.json').read_bytes()),
             'shared_npz_arrays': sha_npz(SHARED),
             'checkpoint_sha256': checkpoint_identity()}, indent=2, sort_keys=True))
    return {'suite': str(SUITE), 'checkpoint': CHECKPOINT.name, 'cohort': 'smoke_500'}


# ------------------------------------------------------------------ rankings


def rank_profile(prof, out_csv: Path, top=40, lane_only=False):
    import pstats
    rows = []
    for (file, line, name), (cc, nc, tottime, cumtime, callers) in pstats.Stats(prof).stats.items():
        if lane_only and LANE_FILE not in file:
            continue
        if not any(part in file for part in PROFILE_PREDICATES):
            continue
        module = file.replace(ROOT.as_posix(), '') + ':' + str(line)
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


# ------------------------------------------------------------- fingerprints


def volatile_normalize(value):
    """Round/iteration-absolute paths are volatile; their content is not."""
    if isinstance(value, str):
        value = value.replace(str(ROOT) + '/', '')
        return re.sub(r'rounds/round\d+/iter\d+', 'rounds/<round>/<iter>', value)
    if isinstance(value, list):
        return [volatile_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: volatile_normalize(item) for key, item in value.items()}
    return value


def fingerprint_report(path: Path):
    document = json.loads(path.read_text())
    cleaned = {k: v for k, v in document.items() if k not in VOLATILE_REPORT_KEYS}
    for key in ('threshold_provenance', 'threshold_binding'):
        if key in cleaned:
            cleaned[key] = volatile_normalize(cleaned[key])
    return sha_json(cleaned)


def fingerprint_binding(path: Path):
    document = json.loads(path.read_text())
    return sha_json(document)


def phase_total(log_path: Path, label: str) -> float | None:
    total, seen = 0.0, False
    for line in log_path.read_text().splitlines():
        parts = line.split()
        if (len(parts) >= 4 and parts[2] == label
                and line.strip().startswith('[timing]')):
            for part in parts:
                if part == 'state=completed':
                    seen = True
                if part.startswith('elapsed_seconds='):
                    total += float(part.split('=', 1)[1])
    return round(total, 3) if seen else None


# -------------------------------------------------------------------- rounds


def run_round(round_no: int):
    fixture = build_suite()
    ROUNDS.mkdir(parents=True, exist_ok=True)
    folder = ROUNDS / f'round{round_no}'
    folder.mkdir(parents=True, exist_ok=False)
    os.environ['ER_TIMING_LOG'] = str(folder / 'timings.log')

    import torch
    torch.set_num_threads(4)
    torch.manual_seed(1729)

    from core.step_trace import trace_step
    from model_tracks import baseline_ablation

    profiler = cProfile.Profile()
    walls = []
    fingerprints = []
    profiler.enable()
    for iteration in range(ITERATIONS):
        out = folder / f'iter{iteration}' / 'output'
        # Complete() draws the saved vectors from the output tensor, as
        # deployed suites do; the fixture file is linked into each fresh
        # output folder before the measured stages.
        link = out / SHARED.name
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(SHARED)
        started = time.perf_counter()
        with trace_step('abl.forward'):
            baseline_ablation.forward(out, SUITE, CHECKPOINT, device='cpu')
        with trace_step('abl.complete'):
            report_path = baseline_ablation.complete(out, SUITE)
        walls.append(round(time.perf_counter() - started, 3))
        ablation_dir = report_path.parent
        fingerprints.append({
            'report': fingerprint_report(ablation_dir / 'report.json'),
            'binding': fingerprint_binding(ablation_dir / 'baseline_threshold.json'),
            'vectors_npz': sha_npz(ablation_dir / 'vectors.npz'),
            'request_sha256': json.loads((ablation_dir / 'report.json').read_text())['request_sha256'],
        })
    profiler.disable()

    top = rank_profile(profiler, folder / 'ranking.csv')
    lane = rank_profile(profiler, folder / 'ranking_lane.csv', top=20, lane_only=True)
    profiler.dump_stats(folder / 'profile.prof')
    summary = {'fixture': fixture, 'note': COHORT_NOTE, 'iterations': ITERATIONS,
               'wall_seconds': walls,
               'wall_cold': walls[0] if walls else None,
               'wall_warm_mean': round(sum(walls[1:]) / max(1, len(walls[1:])), 3),
               'forward_seconds_logged': phase_total(folder / 'timings.log', 'abl.forward'),
               'complete_seconds_logged': phase_total(folder / 'timings.log', 'abl.complete'),
               'fingerprint_chain': fingerprints,
               'byte_identical_sequence': len({json.dumps(f, sort_keys=True) for f in fingerprints}) == 1}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ('wall_seconds', 'wall_cold', 'wall_warm_mean',
                       'forward_seconds_logged', 'complete_seconds_logged',
                       'byte_identical_sequence')}, indent=2))
    return summary, top, lane


def show_rankings():
    for no in list_dirs(ROUNDS):
        folder = ROUNDS / f'round{no}'
        summary = json.loads((folder / 'summary.json').read_text())
        print(f"\nROUND {no}: wall={summary['wall_seconds']} "
              f"cold={summary.get('wall_cold')} warm={summary.get('wall_warm_mean')} "
              f"identical={summary.get('byte_identical_sequence')}")
        ranking = (folder / 'ranking_lane.csv').read_text().splitlines()[1:]
        print('  lane rank self_s ncalls total_s function')
        for rank, line in enumerate(ranking[:10], 1):
            print(f'  {rank:>4} {line}')
    for no in list_dirs(ROUNDS):
        folder = ROUNDS / f'round{no}'
        summary = json.loads((folder / 'summary.json').read_text())
        print(f"{no}\tcold={summary.get('wall_cold')}\twarm={summary.get('wall_warm_mean')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, default=None,
                        help='round number for artifacts; defaults to last+1')
    parser.add_argument('--rankings', action='store_true',
                        help='print existing per-round rankings summary')
    args = parser.parse_args()
    if args.rankings:
        show_rankings()
        return
    if args.round is None:
        existing = list_dirs(ROUNDS)
        args.round = (existing[-1] + 1) if existing else 0
    summary, top, lane = run_round(args.round)
    print('\nTOP FUNCTIONS (self seconds):')
    for rank, (tot, nc, cum, name) in enumerate(top, 1):
        print(f'  {rank:>2}. {tot:8.2f}s {nc:>8}c {cum:8.2f}s {name}')
    print('\nBASELINE ABLATION FILE FUNCTIONS:')
    for rank, (tot, nc, cum, name) in enumerate(lane, 1):
        print(f'  {rank:>2}. {tot:8.2f}s {nc:>8}c {cum:8.2f}s {name}')


if __name__ == '__main__':
    main()
