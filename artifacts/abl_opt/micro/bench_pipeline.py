"""Micro-benchmark for the pack-evidence / volume scan family in pipeline.py.

Replays the real call shape of a catalog row over real dataset rows:
`_PackEvidenceReader.read` runs once per text field (title, description, url,
image) and the volume adapter once per text, exactly like
pipeline._VolumeAndPack.resolve_volume_and_pack / harvest_evidence_channels do.
A digest of every returned record proves the rewrite changed no output byte.

Usage: python artifacts/abl_opt/micro/bench_pipeline.py [--rows 2000] [--reps 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))

import pandas as pd  # noqa: E402

from pipeline import (  # noqa: E402
    extract_pack_evidence,
    extract_pack_from_title,
    extract_volume_from_title,
    parse_attribute_volume_pack,
)


def load_texts(rows: int) -> list[tuple[str, str, str, str, str]]:
    frame = pd.read_csv(ROOT / 'dataset_10k.csv', dtype=str, keep_default_na=False,
                        nrows=rows, low_memory=False)
    return [(row.sku_name_eng, row.description_short_eng, row.sku_url,
             row.image_url, row.attribute) for row in frame.itertuples(index=False)]


def run_scan(texts) -> tuple[str, int]:
    digest = hashlib.sha256()
    calls = 0
    for title, description, url, image, attribute in texts:
        for text in (title, description, url, image):
            digest.update(json.dumps(extract_pack_evidence(text), sort_keys=True).encode())
            digest.update(json.dumps(extract_pack_from_title(text)).encode())
            digest.update(json.dumps(extract_volume_from_title(text), sort_keys=True).encode())
            calls += 3
        digest.update(json.dumps(parse_attribute_volume_pack(attribute)).encode())
        calls += 1
    return digest.hexdigest(), calls


def timeit(fn, reps: int) -> float:
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
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    texts = load_texts(args.rows)
    digest, calls = run_scan(texts)
    seconds, samples = timeit(lambda: run_scan(texts), args.reps)
    result = {'label': args.label, 'rows': len(texts), 'calls': calls,
              'seconds': round(seconds, 3),
              'samples': [round(v, 3) for v in samples],
              'per_call_us': round(seconds / calls * 1e6, 3),
              'digest': digest}
    print(json.dumps(result, indent=2))
    out = Path(args.out) if args.out else ROOT / f'artifacts/abl_opt/micro/bench_pipeline{("_" + args.label) if args.label else ""}.json'
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
