#!/usr/bin/env python3
"""Report exact lexical regex captures in the active SKU input columns.

Brand, title, and attributes are the configured cleaned SKU input. Brand hits
are diagnostic text matches, never structured package/claim evidence. Overlap
between regex families is merged before counting captured spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from core.audit_json import csv_to_json
from core.common import F, TRAIN_ROOT
from pipeline import normalize_text
if __package__:
    from .regex_residual_audit import FIELDS, live_patterns
else:
    from regex_residual_audit import FIELDS, live_patterns

ATTRIBUTE_ITEM_RE = re.compile(r"(?:^|;)\s*([^:;]+):\s*([^;]*)")
NUMBER_VALUE_RE = re.compile(
    r"(?<![a-z0-9])(?:\d+\s*[x×]\s*\d+(?:[.,]\d+)?\s*(?:mg|ml|g|l)?|\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*(?:%|mg|ml|g|l)?)(?![a-z0-9])",
    re.IGNORECASE,
)


def attribute_number_captures(raw: str) -> list[dict[str, object]]:
    """Retain raw units/percent signs erased by the normal text cleaner."""
    result = []
    for item in ATTRIBUTE_ITEM_RE.finditer(raw):
        key = item.group(1).strip()
        for number in NUMBER_VALUE_RE.finditer(item.group(2)):
            value = number.group().strip()
            result.append({
                "attribute_field": key,
                "matched_text": value,
                "start": item.start(2) + number.start(),
                "end": item.start(2) + number.end(),
            })
    return result


def captures(text: str, field: str) -> list[tuple[int, int, str, str]]:
    found = sorted(
        (match.start(), match.end(), label)
        for label, pattern in live_patterns(field)
        for match in pattern.finditer(text)
    )
    merged: list[tuple[int, int, set[str]]] = []
    for start, end, label in found:
        if merged and start < merged[-1][1]:
            old_start, old_end, labels = merged[-1]
            labels.add(label)
            merged[-1] = (old_start, max(old_end, end), labels)
        else:
            merged.append((start, end, {label}))
    return [(start, end, "+".join(sorted(labels)), text[start:end])
            for start, end, labels in merged]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=F["dataset_deduped"])
    parser.add_argument("--out", type=Path, default=TRAIN_ROOT / "results" / "regex_captures.csv")
    parser.add_argument("--detail-json", type=Path, help="Optional full row-level JSON; large")
    parser.add_argument("--summary", type=Path, default=TRAIN_ROOT / "results" / "regex_capture_summary.json")
    parser.add_argument("--attributes-summary", type=Path, default=TRAIN_ROOT / "results" / "regex_attribute_captures.json")
    parser.add_argument("--inspect-product-id", help="Write a small per-product attribute capture JSON")
    parser.add_argument("--inspect-out", type=Path, help="Destination for --inspect-product-id")
    args = parser.parse_args()
    if bool(args.inspect_product_id) != bool(args.inspect_out):
        parser.error("--inspect-product-id and --inspect-out must be provided together")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[tuple[str, str, str]] = Counter()
    field_counts: Counter[str] = Counter()
    examples: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    rows = 0
    inspected: dict[str, object] | None = None
    with args.input.open(newline="", encoding="utf-8") as source, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required = {"product_id", *FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing SKU input columns: {sorted(required - set(reader.fieldnames or []))}")
        writer = csv.DictWriter(destination, fieldnames=(
            "product_id", "source_column", "capture_type", "matched_text", "start", "end", "original_text"
        ))
        writer.writeheader()
        for row in reader:
            rows += 1
            for field in FIELDS:
                original = row[field]
                text = normalize_text(original)
                field_captures = captures(text, field)
                if field == "title" and row["product_id"] == args.inspect_product_id:
                    inspected = {
                        "product_id": row["product_id"],
                        "title_original": original,
                        "title_normalized": text,
                        "title_lexical_regex_captures": [
                            {"capture_type": label, "matched_text": phrase,
                             "start": start, "end": end}
                            for start, end, label, phrase in field_captures
                        ],
                        "title_raw_numeric_captures": [
                            {"matched_text": number.group().strip(), "start": number.start(),
                             "end": number.end()}
                            for number in NUMBER_VALUE_RE.finditer(original)
                        ],
                    }
                if field == "attributes" and row["product_id"] == args.inspect_product_id:
                    if inspected is None:
                        inspected = {"product_id": row["product_id"]}
                    inspected.update({
                        "attributes_original": original,
                        "attributes_normalized": text,
                        "lexical_regex_captures": [
                            {"capture_type": label, "matched_text": phrase,
                             "start": start, "end": end}
                            for start, end, label, phrase in field_captures
                        ],
                        "raw_numeric_captures": attribute_number_captures(original),
                        "note": "Lexical and numeric spans are evidence, not necessarily accepted structured parser values. Raw numeric spans preserve units and percent signs lost during normalize_text.",
                    })
                for start, end, label, phrase in field_captures:
                    key = (field, label, phrase)
                    counts[key] += 1
                    field_counts[field] += 1
                    product_id = row["product_id"]
                    if product_id not in examples[key] and len(examples[key]) < 5:
                        examples[key].append(product_id)
                    writer.writerow({
                        "product_id": product_id, "source_column": field,
                        "capture_type": label, "matched_text": phrase,
                        "start": start, "end": end, "original_text": original,
                    })

    summary = {
        "input": str(args.input), "detail_csv": str(args.out), "rows_scanned": rows,
        "fields": list(FIELDS), "capture_spans_by_field": dict(field_counts),
        "method": "All lexical regex spans in the cleaned SKU fields, with overlaps merged. Brand matches are diagnostic only; span counts do not assert structured parser acceptance.",
        "groups_by_field": {
            field: [
                {"capture_type": category, "matched_text": phrase, "count": count,
                 "example_product_ids": examples[(source_col, category, phrase)]}
                for (source_col, category, phrase), count in sorted(
                    ((key, value) for key, value in counts.items() if key[0] == field),
                    key=lambda item: (-item[1], item[0]),
                )
            ] for field in FIELDS
        },
    }
    if args.detail_json:
        csv_to_json(args.out, args.detail_json)
        summary["detail_json"] = str(args.detail_json)
    args.attributes_summary.parent.mkdir(parents=True, exist_ok=True)
    args.attributes_summary.write_text(json.dumps({
        "rows_scanned": rows,
        "capture_spans": field_counts["attributes"],
        "distinct_groups": len(summary["groups_by_field"]["attributes"]),
        "groups": summary["groups_by_field"]["attributes"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.inspect_out:
        if inspected is None:
            raise ValueError(f"product_id not found: {args.inspect_product_id}")
        args.inspect_out.parent.mkdir(parents=True, exist_ok=True)
        args.inspect_out.write_text(json.dumps(inspected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rows={rows:,} detail={args.out} summary={args.summary}")
    for field in FIELDS:
        print(f"\n{field}: {field_counts[field]:,} nonoverlapping captured spans")
        for item in summary["groups_by_field"][field][:15]:
            print(f"{item['count']:>7,}  {item['capture_type']:<25} {item['matched_text']:<35} ids={','.join(item['example_product_ids'][:3])}")


if __name__ == "__main__":
    main()
