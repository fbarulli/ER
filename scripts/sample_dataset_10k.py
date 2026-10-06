"""Sample 10k skus from the raw export preserving population slice proportions.

Method (seeded, deterministic):
  1. Stratify on the joint country x category key; proportional largest-remainder
     allocation with a floor of 1 per observed stratum so every stratum is hit.
  2. Forced floor BEFORE selection: every singleton retailer and every sku of any
     attribute field too rare to survive the draw (expected < 3 in-sample, capped
     at 10 skus) — "hit every slice" is a hard constraint, not a hope.
  3. Within each stratum pick forced skus first, then uncovered retailers
     (round-robin), then attribute fields still under 5 in-sample, then random.
  4. Verify: every slice value present; per-axis population vs sample share drift
     report; closure to exactly the requested row count.

Usage: PYTHONPATH=src python scripts/sample_dataset_10k.py [--rows 10000]
Output: dataset_10k.csv + dataset_10k.coverage.json next to the population.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from core.common import SEED, DATA_PATH, TRAIN_ROOT


def attribute_fields(text: str) -> set[str]:
    from core.text import normalized_attribute_text
    return {normalized_attribute_text(part.split(':', 1)[0])
            for part in str(text).split(';') if ':' in part}


def largest_remainder(total: int, weights: dict) -> dict:
    """Proportional allocation summing exactly to total (floor of 1 + biggest fractions)."""
    weight_sum = sum(weights.values())
    raw = {key: total * value / weight_sum for key, value in weights.items()}
    quota = {key: max(1, int(raw[key])) for key in weights}
    order = sorted(weights, key=lambda key: raw[key] - int(raw[key]), reverse=True)
    for key in order[:max(0, total - sum(quota.values()))]:
        quota[key] += 1
    return quota


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=10_000)
    parser.add_argument('--output', type=Path, default=TRAIN_ROOT / 'dataset_10k.csv')
    arguments = parser.parse_args()
    rng = random.Random(SEED)
    frame = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False, low_memory=False)
    population = len(frame)
    frame = frame.reset_index().rename(columns={'index': 'row_id'})
    frame['field_set'] = frame.attribute.map(attribute_fields)
    lookup = frame.set_index('row_id')

    rows = list(frame.itertuples(index=False))
    country_of = {row.row_id: row.country for row in rows}
    category_of = {row.row_id: row.category for row in rows}
    retailer_of = {row.row_id: row.retailer for row in rows}
    fields_of = {row.row_id: row.field_set for row in rows}
    retailer_counts = Counter(retailer_of.values())

    field_counts = Counter(field for fields in fields_of.values() for field in fields)
    strata: dict[tuple, list[int]] = defaultdict(list)
    for row_id in country_of:
        strata[(country_of[row_id], category_of[row_id])].append(row_id)

    # 1. forced floor: singleton retailers + attribute fields with expected < 3
    forced: set[int] = set()
    rare_fields = {field for field, count in field_counts.items()
                   if count * arguments.rows / population < 3}
    for field in sorted(rare_fields):
        forced.update([row_id for row_id, fields in fields_of.items() if field in fields][:10])
    forced.update(row_id for row_id, value in retailer_of.items()
                  if retailer_counts[value] == 1)

    quota = largest_remainder(arguments.rows, {key: len(ids) for key, ids in strata.items()})
    chosen_by_stratum: dict[tuple, list[int]] = defaultdict(list)
    chosen: set[int] = set()
    selected_retailers: set[str] = set()
    field_hits: Counter = Counter()

    def remember(row_id: int) -> None:
        chosen.add(row_id)
        chosen_by_stratum[(country_of[row_id], category_of[row_id])].append(row_id)
        selected_retailers.add(retailer_of[row_id])
        field_hits.update(fields_of[row_id])

    for row_id in sorted(forced):
        remember(row_id)

    # 2. per-stratum fill: uncovered retailer first, then fields under 5, then random
    pool = {key: [row_id for row_id in ids if row_id not in chosen]
            for key, ids in strata.items()}
    for ids in pool.values():
        rng.shuffle(ids)

    def pick(ids: list[int]) -> int | None:
        for row_id in ids:
            if retailer_of[row_id] not in selected_retailers:
                return row_id
        for row_id in ids:
            if any(field_hits[field] < 5 for field in fields_of[row_id]):
                return row_id
        return ids[0] if ids else None

    for key in sorted(quota):
        for _ in range(max(0, quota[key] - len(chosen_by_stratum[key]))):
            row_id = pick(pool[key])
            if row_id is None:
                break
            pool[key].remove(row_id)
            remember(row_id)

    # 3. exact closure: trim non-forced extras from over-quota strata / top up
    def chosen_total() -> int:
        return sum(len(ids) for ids in chosen_by_stratum.values())

    by_size = sorted(strata, key=lambda key: -len(strata[key]))
    guard = 0
    while chosen_total() > arguments.rows and guard < 10 * arguments.rows:
        guard += 1
        dropped = False
        for key in by_size:
            if len(chosen_by_stratum[key]) > max(1, quota[key]):
                droppable = [row_id for row_id in chosen_by_stratum[key] if row_id not in forced]
                if droppable:
                    row_id = droppable[-1]
                    chosen_by_stratum[key].remove(row_id)
                    chosen.discard(row_id)
                    dropped = True
                    break
        if not dropped:
            break
    guard = 0
    while chosen_total() < arguments.rows and guard < 10 * arguments.rows:
        guard += 1
        for key in by_size:
            if pool[key]:
                row_id = pick(pool[key])
                if row_id is not None:
                    pool[key].remove(row_id)
                    remember(row_id)
                    break
        else:
            break

    sample = lookup.loc[sorted(chosen)].drop(columns=['field_set'])
    sample.to_csv(arguments.output, index=False)

    # 4. verification
    sample_fields = Counter(field for text in sample.attribute for field in attribute_fields(text))
    report = {'population_rows': population, 'requested_rows': arguments.rows,
              'sample_rows': len(sample), 'seed': SEED, 'axes': {}}
    missing = []

    def axis(name: str, population_counter: Counter, sample_counter: Counter) -> None:
        rows = []
        for value, pop_count in population_counter.most_common():
            sample_count = sample_counter.get(value, 0)
            if sample_count == 0:
                missing.append(f'{name}:{value}')
            rows.append({'value': value, 'population': pop_count, 'sample': sample_count,
                         'population_share': round(pop_count / population, 5),
                         'sample_share': round(sample_count / len(sample), 5),
                         'drift': round(sample_count / len(sample) - pop_count / population, 5)})
        report['axes'][name] = rows

    axis('country', Counter(frame.country), Counter(sample.country))
    axis('category', Counter(frame.category), Counter(sample.category))
    axis('retailer', Counter(frame.retailer), Counter(sample.retailer))
    axis('attribute_field', field_counts, sample_fields)
    report['strata'] = {'observed': len(strata),
                        'hit': sum(1 for key in strata if chosen_by_stratum.get(key))}
    report['missing_slices'] = missing
    report['max_drift_major'] = {name: max((abs(row['drift']) for row in rows
                                            if row['population'] >= 200), default=0)
                                 for name, rows in report['axes'].items()}
    report_path = arguments.output.with_name(arguments.output.stem + '.coverage.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')

    print(f'sample rows={len(sample)} -> {arguments.output}')
    print(f"strata hit={report['strata']['hit']}/{len(strata)} missing_slices={len(missing)}")
    for name, rows in report['axes'].items():
        worst = max(rows, key=lambda row: abs(row['drift']))
        print(f"{name}: values={len(rows)} max|drift|={abs(worst['drift']):.4f} ({worst['value']})")
    if missing:
        raise SystemExit(f'FAIL: uncovered slices: {missing[:10]}')


if __name__ == '__main__':
    main()
