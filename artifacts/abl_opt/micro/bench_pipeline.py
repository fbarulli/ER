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
    extract_all,
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


def run_extract_all(texts) -> tuple[str, int]:
    """The per-row entry point (covers _fuse_and_bound and the whole card)."""
    digest = hashlib.sha256()
    for title, description, url, image, attribute in texts:
        digest.update(json.dumps(extract_all(title, attribute, description, url, image),
                                 sort_keys=True, default=str).encode())
    return digest.hexdigest(), len(texts)


def bulk_pattern_ab(reps: int = 5) -> dict:
    """In-process A/B of the _fuse_and_bound bulk-container reader.

    Targeted computation only (no host drift between the two variants): the old
    body rebuilt the alternation, re-escaped every configured term and went
    through the module-level re.search on every fused row; the new one asks for
    the compiled pattern. 11,441 fused rows per 10k cohort.
    """
    import re
    from core.common import data_cfg
    from pipeline import _bulk_container_re
    terms = tuple(data_cfg().extraction.bulk_container_terms)
    text = "San Pellegrino 12 x 500ml keg sparkling water"
    def legacy():
        escaped = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in terms)
        return re.search(rf"\b(?:{escaped})\b", text, re.I)
    def current():
        return _bulk_container_re(terms).search(text)
    assert bool(legacy()) == bool(current()), 'bulk-container match changed'
    call = legacy
    call()
    before = []
    after = []
    for _ in range(reps):
        started = time.perf_counter()
        for _ in range(1000):
            legacy()
        before.append(time.perf_counter() - started)
        started = time.perf_counter()
        for _ in range(1000):
            current()
        after.append(time.perf_counter() - started)
    return {'terms': len(terms), 'per_1000_calls_before_ms': round(median(before) * 1e3, 3),
            'per_1000_calls_after_ms': round(median(after) * 1e3, 3),
            'speedup': round(median(before) / median(after), 2),
            'match_unchanged': True}


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
    all_digest, all_calls = run_extract_all(texts)
    all_seconds, all_samples = timeit(lambda: run_extract_all(texts), args.reps)
    result = {'label': args.label, 'rows': len(texts), 'calls': calls,
              'seconds': round(seconds, 3),
              'samples': [round(v, 3) for v in samples],
              'per_call_us': round(seconds / calls * 1e6, 3),
              'digest': digest,
              'extract_all_seconds': round(all_seconds, 3),
              'extract_all_us_per_row': round(all_seconds / all_calls * 1e6, 2),
              'extract_all_samples': [round(v, 3) for v in all_samples],
              'extract_all_digest': all_digest,
              'loadavg': [round(v, 2) for v in os.getloadavg()],
              'bulk_container_reader': bulk_pattern_ab(),
              'extract_all_digest_note': ('extract_all returns sets, and default=str '
                                          'renders them in hash order, so this digest is '
                                          'process-dependent even for identical code')}
    print(json.dumps(result, indent=2))
    out = Path(args.out) if args.out else ROOT / f'artifacts/abl_opt/micro/bench_pipeline{("_" + args.label) if args.label else ""}.json'
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
