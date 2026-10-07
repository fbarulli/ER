"""Micro-benchmark for the identity extractor (sku_identity / product_dimensions).

Replays the real per-row call shape of graph_tracks.prepare / ablation._compose
(`row_identity(row)` over catalog rows, plus `row_dimensions(row)` alone) on
real dataset rows, and digests every produced bundle so a rewrite can be shown
to change no extracted value.

Usage: python artifacts/abl_opt/micro/bench_identity.py [--rows 2000] [--reps 3] [--label X]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))

import pandas as pd  # noqa: E402

from core.product_dimensions import row_dimensions  # noqa: E402
from core.sku_identity import row_identity  # noqa: E402


def load_rows(rows: int) -> list[dict]:
    frame = pd.read_csv(ROOT / 'dataset_10k.csv', dtype=str, keep_default_na=False,
                        nrows=rows, low_memory=False)
    return [dict(row) for _, row in frame.iterrows()]


def digest_of(identities) -> str:
    digest = sha256()
    for identity in identities:
        digest.update(json.dumps({
            'brand': sorted(identity.brand),
            'volume_ml': sorted(identity.volume_ml),
            'pack': sorted(identity.pack),
            'flavor': sorted(identity.flavor),
            'carbonation': sorted(identity.carbonation),
            'sweetener': sorted(identity.sweetener),
            'sweetener_type': sorted(identity.sweetener_type),
            'sweetening': sorted(identity.sweetening),
            'pulp': sorted(identity.pulp),
            'package_type': sorted(identity.package_type),
            'package_material': sorted(identity.package_material),
            'diet_claim': identity.diet_claim,
            'sugar_claim': identity.sugar_claim,
            'gtin_trusted': identity.gtin_trusted,
            'gtin_key': identity.gtin_key,
            'identity_review_reason': identity.identity_review_reason,
            'completeness': identity.completeness,
            'dimensions': None if identity.dimensions is None else {
                'attributes': {k: sorted(v) for k, v in identity.dimensions.attributes.items()},
                'unclassified': list(identity.dimensions.unclassified_keys),
                'malformed': list(identity.dimensions.malformed_parts),
            },
        }, sort_keys=True).encode())
    return digest.hexdigest()


def run_identity(rows) -> tuple[str, int]:
    identities = [row_identity(row) for row in rows]
    return digest_of(identities), len(identities)


def run_dimensions(rows) -> tuple[str, int]:
    digest = sha256()
    for row in rows:
        evidence = row_dimensions(row)
        digest.update(json.dumps({k: sorted(v) for k, v in evidence.attributes.items()},
                                 sort_keys=True).encode())
    return digest.hexdigest(), len(rows)


def timeit(fn, reps: int):
    fn()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return median(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=2000)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    rows = load_rows(args.rows)
    identity_digest, calls = run_identity(rows)
    id_seconds, id_samples = timeit(lambda: run_identity(rows), args.reps)
    dim_digest, dim_calls = run_dimensions(rows)
    dim_seconds, dim_samples = timeit(lambda: run_dimensions(rows), args.reps)
    result = {
        'label': args.label, 'rows': len(rows), 'reps': args.reps,
        'row_identity_seconds': round(id_seconds, 3),
        'row_identity_us_per_call': round(id_seconds / calls * 1e6, 2),
        'row_identity_samples': [round(v, 3) for v in id_samples],
        'row_identity_digest': identity_digest,
        'row_dimensions_seconds': round(dim_seconds, 3),
        'row_dimensions_us_per_call': round(dim_seconds / dim_calls * 1e6, 2),
        'row_dimensions_samples': [round(v, 3) for v in dim_samples],
        'row_dimensions_digest': dim_digest,
    }
    print(json.dumps(result, indent=2))
    out = ROOT / f'artifacts/abl_opt/micro/bench_identity{("_" + args.label) if args.label else ""}.json'
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
