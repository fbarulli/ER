#!/usr/bin/env python3
"""Guarded reading sheet for every registered attribute dimension.

For each of the registered dimensions, emit a bounded sample of listings for a
manual right/wrong/missed read:

  * populated  — rows where the parser emitted a value (checks right/wrong);
  * key-present— rows whose source attribute cell carries the dimension key but
                 the parser emitted nothing (checks missed).

Every dimension is guarded first (core.audit_guard): its value vocabulary must
occur in the source text, a row must never compare "different" from itself, and
coverage must not be exactly 0% or 100%. A dimension that fails a hard guard is
excluded; a degenerate dimension is marked unmeasured. Nothing is published for
a dimension whose guard fired.

This script only *produces the sheet*; the verdicts are the human read. It is
read-only over the dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import pandas as pd

from core.audit_guard import guard_dimensions
from core.common import F, RESULTS
from core.product_dimensions import dimension_policy, evaluate_dimensions, row_dimensions
from core.text import normalized_attribute_text


def _key_present(cell: str, key: str) -> bool:
    needle = normalized_attribute_text(key)
    for part in str(cell or "").split(";"):
        if ":" in part and normalized_attribute_text(part.split(":", 1)[0]) == needle:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=F["dataset_deduped"])
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "attribute_readings")
    parser.add_argument("--per-dimension", type=int, default=100)
    parser.add_argument("--missed-per-dimension", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset, dtype=str, keep_default_na=False)
    required = {"sku_id", "retailer", "sku_name_eng", "attribute"}
    if not required <= set(frame.columns):
        raise ValueError(f"dataset missing columns: {sorted(required - set(frame.columns))}")
    policy = dimension_policy()
    records = frame.to_dict("records")
    evidence = [row_dimensions(row) for row in records]

    values: dict[str, set[str]] = defaultdict(set)
    populated_rows: dict[str, list[int]] = defaultdict(list)
    key_present_rows: dict[str, list[int]] = defaultdict(list)
    for index, (record, record_evidence) in enumerate(zip(records, evidence)):
        for name, members in record_evidence.attributes.items():
            values[name].update(members)
            populated_rows[name].append(index)
        for name in policy.attributes:
            if name not in record_evidence.attributes and _key_present(record.get("attribute", ""), name):
                key_present_rows[name].append(index)

    attr_texts = list(frame["attribute"])
    sample = evidence[:200]
    guard_specs = [
        {
            "name": name,
            "values": values.get(name, set()),
            "source_texts": attr_texts,
            "self_compare": (
                lambda a, b, _n=name: evaluate_dimensions(a, b)[_n]["status"] != "different"
            ),
            "self_samples": sample,
            "populated": len(populated_rows.get(name, [])),
            "total": len(frame),
        }
        for name in sorted(policy.attributes)
    ]
    guard_results = guard_dimensions(guard_specs, label="attribute-readings")
    hard_failed = {r.name for r in guard_results if not r.passed}
    unmeasured = {r.name for r in guard_results if r.unmeasured}

    rng = pd.Series(range(len(frame))).sample(frac=1, random_state=args.seed).tolist()
    order = {index: rank for rank, index in enumerate(rng)}
    rows = []
    for name in sorted(policy.attributes):
        if name in hard_failed or name in unmeasured:
            continue
        picked = sorted(populated_rows.get(name, []), key=order.get)[: args.per_dimension]
        for index in picked:
            rows.append(_row(name, records[index], evidence[index], populated=True))
        missed = sorted(key_present_rows.get(name, []), key=order.get)[: args.missed_per_dimension]
        for index in missed:
            rows.append(_row(name, records[index], evidence[index], populated=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet = args.output_dir / "reading_sheet.csv"
    columns = ["dimension", "state", "sku_id", "retailer", "sku_name_eng",
               "attribute", "description_short_eng", "extracted", "verdict", "note"]
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "rows": len(frame),
        "dimensions": len(policy.attributes),
        "guarded": len(guard_results),
        "hard_failed": sorted(hard_failed),
        "unmeasured": sorted(unmeasured),
        "populated_counts": {name: len(populated_rows.get(name, [])) for name in sorted(policy.attributes)},
        "key_present_not_extracted": {name: len(key_present_rows.get(name, [])) for name in sorted(policy.attributes)},
        "sheet_rows": len(rows),
        "per_dimension": args.per_dimension,
        "missed_per_dimension": args.missed_per_dimension,
    }
    (args.output_dir / "reading_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("rows", "dimensions", "guarded", "hard_failed", "unmeasured", "sheet_rows")}))
    print(f"sheet -> {sheet}")


def _flat(value: object) -> str:
    # Whitespace is collapsed for the sheet only (embedded newlines would split
    # one listing across CSV lines). Nothing is truncated.
    return " ".join(str(value or "").split())


def _row(name: str, record: dict, record_evidence, *, populated: bool) -> dict:
    extracted = sorted(record_evidence.attributes.get(name, []))
    return {
        "dimension": name,
        "state": "populated" if populated else "key_present_not_extracted",
        "sku_id": record.get("sku_id", ""),
        "retailer": _flat(record.get("retailer", "")),
        "sku_name_eng": _flat(record.get("sku_name_eng", "")),
        "attribute": _flat(record.get("attribute", "")),
        "description_short_eng": _flat(record.get("description_short_eng", "")),
        "extracted": "|".join(extracted),
        "verdict": "",
        "note": "",
    }


if __name__ == "__main__":
    main()
