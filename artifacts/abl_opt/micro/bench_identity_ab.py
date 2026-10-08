"""In-process A/B for the r18 identity changes (no lock needed, no load drift).

Load on this host is shared with other lanes and drifts by more than the
effect sizes involved, so each targeted function is measured against a
verbatim copy of its pre-r18 body inside ONE process, alternating reps.

Legacy bodies below are copied from commit 4dbe661.
Usage: python artifacts/abl_opt/micro/bench_identity_ab.py [--rows 1000] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))

import pandas as pd  # noqa: E402

from core.portable_archive import ByteCount as size  # noqa: E402
import core.product_dimensions as pdims  # noqa: E402
import core.sku_identity as sku  # noqa: E402
from core.text import normalized_attribute_text, unicode_casefold  # noqa: E402


def legacy_row_dimensions(row, *, policy=None):
    """Pre-r18 body, verbatim (registry rebuilt and patterns re-dispatched)."""
    policy = policy or pdims.dimension_policy()
    registry = {normalized_attribute_text(k): k for k in policy.attributes}
    attributes: dict[str, set[str]] = {}
    unknown, malformed = set(), []
    for part in str(row.get("attribute", "") or "").split(";"):
        if not part.strip():
            continue
        if ":" not in part:
            malformed.append(part.strip())
            continue
        key, value = part.split(":", 1)
        normalized_key = normalized_attribute_text(key)
        if not normalized_key:
            malformed.append(part.strip())
            continue
        name = registry.get(normalized_key, normalized_key)
        if name not in policy.attributes:
            unknown.add(name)
        rule = policy.attributes.get(name)
        for item in value.split(","):
            normalized = re.sub(r"\s+", " ", unicode_casefold(item)).strip()
            if not normalized:
                continue
            normalized = rule.aliases.get(normalized, normalized) if rule else normalized
            attributes.setdefault(name, set()).add(normalized)
    columns = {str(k): str(v or "").strip() for k, v in row.items() if k != "attribute"}
    from core.product_context import resolve_context
    frozen_attributes = {k: frozenset(v) for k, v in attributes.items()}
    return pdims.DimensionEvidence(frozen_attributes, columns, tuple(sorted(unknown)),
                                   tuple(malformed), resolve_context(row, frozen_attributes))


def legacy_attr_token_set(attr, key):
    """Pre-r18 body, verbatim (module-level re.search + string cache lookup)."""
    match = re.search(key + r"\s*:\s*([^;]+)", str(attr or ""), re.I)
    if not match:
        return frozenset()
    return frozenset(sku._TOKEN_RE.findall(sku.normalize_text(match.group(1))))


def legacy_completeness(row):
    """Pre-r18 body, verbatim (every descriptor cell read and stringified twice)."""
    get = (lambda k: row.get(k, "")) if isinstance(row, dict) else (
        lambda k: getattr(row, k, "")
    )
    return sum(
        1 for col in sku.DESCRIPTOR_COLUMNS
        if not pd.isna(get(col)) and str(get(col)).strip()
    )


def load_rows(rows: int) -> list[dict]:
    frame = pd.read_csv(ROOT / 'dataset_10k.csv', dtype=str, keep_default_na=False,
                        nrows=rows, low_memory=False)
    return [dict(row) for _, row in frame.iterrows()]


def plain(value):
    """JSON-able projection of whatever a target function returns."""
    if isinstance(value, pdims.DimensionEvidence):
        return {'attributes': {k: sorted(v) for k, v in value.attributes.items()},
                'columns': dict(value.columns),
                'unclassified_keys': list(value.unclassified_keys),
                'malformed_parts': list(value.malformed_parts)}
    if isinstance(value, (frozenset, set, tuple)):
        return sorted(str(v) for v in value)
    return value


def size_of(value) -> int:
    return size(json.dumps(plain(value), sort_keys=True, default=plain).encode()).total


def ab(label: str, legacy, current, rows, reps: int) -> dict:
    legacy_out = [legacy(row) for row in rows]
    current_out = [current(row) for row in rows]
    legacy_bytes = size_of(legacy_out)
    current_bytes = size_of(current_out)
    legacy_samples, current_samples = [], []
    for _ in range(reps):
        started = time.perf_counter()
        for row in rows:
            legacy(row)
        legacy_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        for row in rows:
            current(row)
        current_samples.append(time.perf_counter() - started)
    before, after = median(legacy_samples), median(current_samples)
    return {'function': label, 'calls': len(rows), 'reps': reps,
            'before_seconds': round(before, 4), 'after_seconds': round(after, 4),
            'before_us_per_call': round(before / len(rows) * 1e6, 2),
            'after_us_per_call': round(after / len(rows) * 1e6, 2),
            'speedup': round(before / after, 2),
            'identical_output': legacy_bytes == current_bytes,
            'digest': current_bytes}


def ab_row_identity(rows, reps: int) -> dict:
    """row_identity with its pre-r18 callees swapped back in, vs current."""
    originals = (sku.attr_token_set, sku.completeness, sku.row_dimensions)
    try:
        sku.attr_token_set, sku.completeness, sku.row_dimensions = (
            legacy_attr_token_set, legacy_completeness, legacy_row_dimensions)
        legacy_out = [sku.row_identity(row) for row in rows]
        legacy_samples = []
        for _ in range(reps):
            started = time.perf_counter()
            for row in rows:
                sku.row_identity(row)
            legacy_samples.append(time.perf_counter() - started)
    finally:
        (sku.attr_token_set, sku.completeness, sku.row_dimensions) = originals
    current_out = [sku.row_identity(row) for row in rows]
    current_samples = []
    for _ in range(reps):
        started = time.perf_counter()
        for row in rows:
            sku.row_identity(row)
        current_samples.append(time.perf_counter() - started)
    before, after = median(legacy_samples), median(current_samples)
    key = lambda identity: (sorted(identity.brand), sorted(identity.volume_ml), sorted(identity.pack),
                            sorted(identity.flavor), sorted(identity.carbonation), sorted(identity.sweetener),
                            sorted(identity.sweetener_type), sorted(identity.sweetening), sorted(identity.pulp),
                            sorted(identity.package_type), sorted(identity.package_material),
                            identity.diet_claim, identity.sugar_claim, identity.gtin_trusted,
                            identity.gtin_key, identity.identity_review_reason, identity.completeness,
                            identity.dimensions.attributes if identity.dimensions else None)
    legacy_bytes = size_of([key(value) for value in legacy_out])
    current_bytes = size_of([key(value) for value in current_out])
    return {'function': 'sku_identity.row_identity (with r18 callees)', 'calls': len(rows), 'reps': reps,
            'before_seconds': round(before, 4), 'after_seconds': round(after, 4),
            'before_us_per_call': round(before / len(rows) * 1e6, 2),
            'after_us_per_call': round(after / len(rows) * 1e6, 2),
            'speedup': round(before / after, 2),
            'identical_output': legacy_bytes == current_bytes,
            'digest': current_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--reps', type=int, default=3)
    args = parser.parse_args()
    rows = load_rows(args.rows)
    results = [
        ab_row_identity(rows, args.reps),
        ab('product_dimensions.row_dimensions', legacy_row_dimensions,
           lambda row: pdims.row_dimensions(row), rows, args.reps),
        ab('sku_identity.attr_token_set', lambda row: legacy_attr_token_set(row['attribute'], r'flavou?r'),
           lambda row: sku.attr_token_set(row['attribute'], r'flavou?r'), rows, args.reps),
        ab('sku_identity.completeness', legacy_completeness, lambda row: sku.completeness(row),
           rows, args.reps),
    ]
    print(json.dumps(results, indent=2))
    (ROOT / 'artifacts/abl_opt/micro/bench_identity_ab.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
