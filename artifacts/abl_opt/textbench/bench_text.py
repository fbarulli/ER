"""Targeted micro-benchmark for the reg2-lane-A owned text hot paths.

WHY THIS EXISTS
---------------
``scripts/ablation_timing.py`` profiles only ``baseline_ablation.forward`` +
``baseline_ablation.complete`` on a *frozen* suite, i.e. the staged request's
``prepared_inputs.npz`` + ``shared_minilm__embeddings.npz`` are reused and no
model text is ever composed.  pstats over ``rounds/round14/profile.prof`` shows
``src/core/text.py`` and ``src/core/declared_identity.py`` are never even
imported inside the timed region, so the round wall time cannot move when these
modules are optimized.  This harness measures them directly instead, on the same
real catalog rows the pipeline feeds them.

It mirrors the real call mix (``src/pipeline.py`` harvest_evidence_channels /
extract_volume_from_title, ``core.critical_attributes`` key folding,
``core.attribute_universe`` field parsing, ``core.declared_identity`` identity
review) over the frozen ``smoke_500`` fixture catalog.

Usage
-----
    /home/opc/ONE/ER/.venv/bin/python artifacts/abl_opt/textbench/bench_text.py \
        --label before --json artifacts/abl_opt/textbench/before.json

Every bench returns a value; the sha256 of a canonical repr of ALL bench results
is printed as ``digest`` — an unchanged digest across two runs proves the
optimization was byte-identical on the real fixture corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
# ER_BENCH_SRC lets the same harness measure another checkout (used to A/B a
# pristine tree); without it the worktree next to this file is measured.
SRC = os.environ.get('ER_BENCH_SRC') or str(ROOT / 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import pandas as pd  # noqa: E402

CATALOG = ROOT / 'artifacts' / 'abl_opt' / 'baseline' / 'eligible_catalog.csv'


def load_rows() -> list[dict]:
    frame = pd.read_csv(CATALOG).head(500)
    return frame.to_dict('records')


def row_metadata(row: dict, key: str):
    """The raw column value for a bench row (mirrors core.common's reader)."""
    value = row.get(key)
    return None if value is None or value != value else value


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None or (isinstance(value, float) and value != value):
        return ''
    return str(value)


def build_benches(rows: list[dict]) -> dict:
    """name -> (callable, human description). Each callable does ONE full pass."""
    from core.text import (
        attribute_fields,
        extract_pack_counts,
        extract_volume_evidence,
        extract_volume_match,
        norm_unit,
        normalize_retailer,
        normalize_text,
        normalized_attribute_text,
        unicode_casefold,
    )
    from core.declared_identity import listing_identity
    from core.product_selection import selected_identity_inputs

    titles = [_text(row, 'sku_name_eng') for row in rows]
    attrs = [_text(row, 'attribute') for row in rows]
    descs = [_text(row, 'description_short_eng') for row in rows]
    urls = [_text(row, 'sku_url') for row in rows]
    imgs = [_text(row, 'image_url') for row in rows]
    brands = [_text(row, 'brand') for row in rows]
    cats = [_text(row, 'category') for row in rows]
    crumbs = [_text(row, 'breadcrumbs_eng') for row in rows]

    def bench_volume_evidence():
        # pipeline.harvest_evidence_channels: title, url, image per row.
        out = []
        for index in range(len(rows)):
            out.append(extract_volume_evidence(titles[index]))
            out.append(extract_volume_evidence(urls[index]))
            out.append(extract_volume_evidence(imgs[index]))
        return out

    def bench_volume_match():
        # pipeline.extract_volume_from_title: title + url + image per row.
        out = []
        for index in range(len(rows)):
            out.append(extract_volume_match(titles[index]))
            out.append(extract_volume_match(urls[index]))
            out.append(extract_volume_match(imgs[index]))
        return out

    def bench_volume_adapter():
        # pipeline.extract_volume_from_title — title, url and image per row
        # (pipeline.py:1023-1036). This is the path that reaches
        # core.text._volume_entry -> _volume_spelling_index.
        from pipeline import extract_volume_from_title
        out = []
        for index in range(len(rows)):
            out.append(extract_volume_from_title(titles[index]))
            out.append(extract_volume_from_title(urls[index]))
            out.append(extract_volume_from_title(imgs[index]))
        return out

    def bench_pack_counts():
        out = []
        for index in range(len(rows)):
            out.append(sorted(extract_pack_counts(titles[index])))
            out.append(sorted(extract_pack_counts(descs[index])))
        return out

    def bench_attribute_fields():
        return [attribute_fields(attrs[index]) for index in range(len(rows))]

    def bench_norm_attr_keys():
        # core.critical_attributes._field_tokens / pipeline.capture_universe_attributes:
        # every raw attribute key folded, plus the row-level (title, attributes) folds.
        out = []
        for index in range(len(rows)):
            for part in attrs[index].split(';'):
                if ':' in part:
                    out.append(normalized_attribute_text(part.split(':', 1)[0]))
            out.append(normalized_attribute_text(titles[index]))
            out.append(normalized_attribute_text(titles[index], attrs[index]))
            out.append(normalized_attribute_text(descs[index]))
        return out

    def bench_casefold():
        out = []
        for index in range(len(rows)):
            out.append(unicode_casefold(titles[index]))
            out.append(unicode_casefold(attrs[index]))
            out.append(unicode_casefold(descs[index]))
        return out

    def bench_normalize_text():
        out = []
        for index in range(len(rows)):
            out.append(normalize_text(titles[index]))
            out.append(normalize_text(urls[index]))
            out.append(normalize_text(descs[index]))
            out.append(normalize_retailer(_text(rows[index], 'retailer')))
            out.append(norm_unit(_text(rows[index], 'brand')[:6]))
        return out

    def bench_identity():
        # core.declared_identity.listing_identity as the review lane calls it.
        return [
            listing_identity(titles[index], attrs[index], descs[index])
            for index in range(len(rows))
        ]

    def bench_selected_inputs():
        return [
            selected_identity_inputs(titles[index], attrs[index])
            for index in range(len(rows))
        ]

    def bench_critical_claims():
        from core.critical_attributes import (
            extract_critical_claims,
            extract_description_claims,
            extract_flavor_tokens,
        )
        out = []
        for index in range(len(rows)):
            title, attributes, _variant = selected_identity_inputs(
                titles[index], attrs[index])
            out.append(extract_critical_claims(title, attributes))
            out.append(extract_description_claims(descs[index]))
            out.append(extract_flavor_tokens(titles[index], attrs[index]))
        return out

    # Built ONCE, at bench-construction time (outside every timed region), so
    # `sku_text` isolates core/model_input.py from structured_features —
    # `sku_info` is ~35x heavier and owned by another lane.
    from core.model_input import _RowProxy, build_sku_text, model_input_info
    from core.structured_features import sku_info as _sku_info
    prepared = []
    for index in range(len(rows)):
        _row = _RowProxy(rows[index])
        _info = model_input_info(_sku_info(
            row_metadata(rows[index], 'sku_name_eng'),
            row_metadata(rows[index], 'attribute'),
            row_metadata(rows[index], 'description_short_eng')))
        prepared.append((_row, _info))

    def bench_sku_text():
        # The per-row encoder text ONLY (core.model_input's own entry point).
        return [build_sku_text(row, info) for row, info in prepared]

    def bench_model_input():
        from core.model_input import build_sku_texts
        from core.structured_features import sku_info
        frame = pd.DataFrame({
            'sku_name_eng': titles,
            'attribute': attrs,
            'description_short_eng': descs,
            'brand': brands,
            'category': cats,
            'breadcrumbs_eng': crumbs,
        })
        texts, infos = build_sku_texts(frame, structured_enabled=True)
        return texts

    def bench_sku_info():
        from core.structured_features import sku_info
        return [sku_info(titles[i], attrs[i], descs[i]) for i in range(len(rows))]

    return {
        'volume_evidence': bench_volume_evidence,
        'volume_match': bench_volume_match,
        'volume_adapter': bench_volume_adapter,
        'pack_counts': bench_pack_counts,
        'attribute_fields': bench_attribute_fields,
        'norm_attr_keys': bench_norm_attr_keys,
        'casefold': bench_casefold,
        'normalize_text': bench_normalize_text,
        'selected_inputs': bench_selected_inputs,
        'identity': bench_identity,
        'critical_claims': bench_critical_claims,
        'sku_info': bench_sku_info,
        'sku_text': bench_sku_text,
        'model_input': bench_model_input,
    }


def _stable(value):
    """JSON-safe, hash-seed-independent view of a bench result.

    Several benches return dicts whose values are sets; ``str(set)`` order
    varies per process (PYTHONHASHSEED), so every set is emitted as a sorted
    list here or the digest would be meaningless across runs.
    """
    from collections.abc import Mapping as _Mapping

    if isinstance(value, _Mapping):
        return {str(key): _stable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_stable(item) for item in value]
        return ['__set__', sorted(items, key=lambda item: json.dumps(item, sort_keys=True))]
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, int)) or value is None or isinstance(value, bool):
        return value
    return repr(value)


def canonical(value) -> str:
    return json.dumps(_stable(value), sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='run')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--json', default=None)
    parser.add_argument('--only', default=None,
                        help='comma-separated bench names')
    args = parser.parse_args()

    rows = load_rows()
    benches = build_benches(rows)
    if args.only:
        wanted = {name.strip() for name in args.only.split(',') if name.strip()}
        benches = {name: fn for name, fn in benches.items() if name in wanted}

    # Warm import/config caches once, outside every timing.
    for fn in benches.values():
        fn()

    digest = hashlib.sha256()
    results = {}
    for name, fn in benches.items():
        samples = []
        value = None
        for _ in range(args.reps):
            started = time.perf_counter()
            value = fn()
            samples.append(time.perf_counter() - started)
        digest.update(name.encode())
        digest.update(hashlib.sha256(canonical(value).encode()).digest())
        results[name] = {
            'best': round(min(samples), 6),
            'median': round(statistics.median(samples), 6),
            'samples': [round(sample, 6) for sample in samples],
        }
        print(f'{name:20s} best={results[name]["best"]:.6f}s '
              f'median={results[name]["median"]:.6f}s')

    total_best = round(sum(r['best'] for r in results.values()), 6)
    print(f'{"TOTAL(best)":20s} {total_best:.6f}s')
    print(f'digest {digest.hexdigest()}')
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'label': args.label, 'reps': args.reps, 'rows': len(rows),
             'total_best': total_best, 'digest': digest.hexdigest(),
             'benches': results}, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
