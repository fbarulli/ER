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

import pandas as pd

from core.audit_json import csv_to_json
from core.attribute_conflicts import sku_attribute_info
from core.critical_attributes import FLAVOR_ALIASES, normalized_attribute_text
from core.model_input import build_sku_text, model_input_composition, model_input_info
from core.structured_features import sku_info
from core.sweetener_values import SWEETENER_TYPES, declared_sweeteners
from core.common import F, TRAIN_ROOT
from pipeline import normalize_text
if __package__:
    from .regex_residual_audit import FIELDS, ROUNDS, live_patterns, residual
else:
    from regex_residual_audit import FIELDS, ROUNDS, live_patterns, residual

ATTRIBUTE_ITEM_RE = re.compile(r"(?:^|;)\s*([^:;]+):\s*([^;]*)")
NUMBER_VALUE_RE = re.compile(
    r"(?<![a-z0-9])(?:\d+\s*[x×]\s*\d+(?:[.,]\d+)?\s*(?:mg|ml|g|l)?|\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*(?:%|mg|ml|g|l)?)(?![a-z0-9])",
    re.IGNORECASE,
)
SUGAR_INGREDIENT_TYPES = frozenset({
    "sugar", "cane_sugar", "hfcs", "fructose", "glucose", "sucrose", "corn_syrup",
})


def sweetener_type_evidence(raw: str) -> list[dict[str, object]]:
    """Typed declarations, distinct from the no-sugar/sugar claim classes."""
    observations = []
    for item in ATTRIBUTE_ITEM_RE.finditer(raw):
        if item.group(1).strip().casefold() != "sweetener":
            continue
        for part in re.finditer(r"[^,/&]+", item.group(2)):
            value = part.group().strip().casefold()
            if not value:
                continue
            start = item.start(2) + part.start() + len(part.group()) - len(part.group().lstrip())
            observations.append({
                "source_column": "attributes",
                "source_field": "sweetener",
                "raw_span": [start, start + len(part.group().strip())],
                "surface": part.group().strip(),
                "canonical_value": value.replace(" ", "_") if value in SWEETENER_TYPES or value == "unsweetened" else None,
                "assigned_field": "sweetening" if value == "unsweetened" else "sweetener_type",
                "status": "parser_assigned" if value in SWEETENER_TYPES or value == "unsweetened" else "unmapped_or_non_type",
                "swap_eligible": False,
            })
    return observations


def declared_flavor_evidence(raw: str) -> list[dict[str, object]]:
    """Preserve every explicit flavor value, including values outside the lexicon."""
    observations = []
    accepted = sku_attribute_info("", raw)["flavor_set"]
    for item in ATTRIBUTE_ITEM_RE.finditer(raw):
        if item.group(1).strip().casefold() not in {"flavour", "flavor"}:
            continue
        for part in re.finditer(r"[^,/]+", item.group(2)):
            surface = part.group().strip()
            value = normalized_attribute_text(surface)
            if not value:
                continue
            value = " ".join(FLAVOR_ALIASES.get(token, token) for token in value.split())
            start = item.start(2) + part.start() + len(part.group()) - len(part.group().lstrip())
            observations.append({
                "source_column": "attributes",
                "source_field": "flavor",
                "raw_span": [start, start + len(surface)],
                "surface": surface,
                "canonical_value": value.replace(" ", "_"),
                "status": "parser_assigned" if value in accepted else "unmapped",
                "swap_eligible": False,
            })
    return observations



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


def semantic_profile(title: str, attributes: str, numeric_captures: list[dict[str, object]],
                     title_captures: list[dict[str, object]]) -> dict[str, object]:
    """Versioned evidence contract; only parser-accepted slots may be swapped.

    SID assignment still uses embeddings, and training augmentation still uses
    structured payload tokens. This review format does not change either path.
    """
    parsed = sku_attribute_info(title, attributes)
    fields = ("volume", "pack", "package_type", "flavor", "carbonation", "sweetener", "pulp", "sweetener_type", "sweetening")
    trusted = {
        field: sorted(parsed["flavor_set" if field == "flavor" else field])
        for field in fields
    }
    slot_by_field = {
        "caffeine": "caffeine_mg",
        "juice content": "juice_content_pct",
        "volume": "volume_untyped",
        "count per unit": "pack_count",
    }
    observations = []
    for item in numeric_captures:
        surface = str(item["matched_text"])
        values = [float(value.replace(",", ".")) for value in re.findall(r"\d+(?:[.,]\d+)?", surface)]
        unit_match = re.search(r"(%|mg|ml|g|l)\s*$", surface, re.IGNORECASE)
        field_key = str(item["attribute_field"]).casefold()
        observations.append({
            "source_column": "attributes",
            "source_field": field_key,
            "raw_span": [item["start"], item["end"]],
            "surface": surface,
            "candidate_slot": slot_by_field.get(field_key),
            "numbers": values,
            "unit": unit_match.group(1).lower() if unit_match else None,
            "status": "raw_numeric_evidence",
            "swap_eligible": False,
        })
    sweetener_evidence = sweetener_type_evidence(attributes)
    flavor_evidence = declared_flavor_evidence(attributes)
    sweetener_types = trusted["sweetener_type"]
    consistency_flags = sorted(declared_sweeteners(attributes)["consistency_flags"])
    if "no_sugar" in trusted["sweetener"] and SUGAR_INGREDIENT_TYPES.intersection(sweetener_types):
        consistency_flags.append("no_sugar_claim_conflicts_with_sugar_ingredient")
    return {
        "schema_version": "er.attribute_evidence.v1",
        "trusted_structured_values": trusted,
        "swap_compatible_fields": [
            field for field in fields if trusted[field]
            and not (field in {"sweetener", "sweetener_type", "sweetening"} and consistency_flags)
        ],
        "candidate_typed_values": {
            "sweetener_type": sorted(declared_sweeteners(attributes)["unmapped"]),
            "declared_flavor": sorted({item["canonical_value"] for item in flavor_evidence
                                       if item["status"] == "unmapped"}),
        },
        "consistency_flags": consistency_flags,
        "typed_evidence": [*sweetener_evidence, *flavor_evidence],
        "numeric_observations": observations,
        "lexical_only_claims": [
            {"source_column": "title", "surface": item["matched_text"],
             "normalized_span": [item["start"], item["end"]],
             "candidate_slot": "natural_claim", "status": "lexical_only",
             "swap_eligible": False}
            for item in title_captures if item["capture_type"] == "natural_claim_lexical"
        ],
        "note": "Only trusted_structured_values align with current model swap fields. Numeric observations and lexical-only claims are provenance, not SID codes or augmentation instructions.",
    }


def model_payload_review(row: dict[str, str]) -> dict[str, object]:
    """Build the exact active SKU payload and compare it to audit-only dedup."""
    info = model_input_info(sku_info(row["title"], row["attributes"]))
    payload = build_sku_text(pd.Series(row), info)
    plain = [token for token in payload.split() if "_" not in token and not token.startswith("[FIELD_")]
    repeated = {token: count for token, count in Counter(plain).items() if count > 1}
    seen: set[str] = set()
    unique_fields = []
    for field in FIELDS:
        current = normalize_text(row[field])
        for round_name in ROUNDS:
            current, _ = residual(current, field=field, round_name=round_name, seen_tokens=seen)
        unique_fields.append(current)
    return {
        "composition": model_input_composition().model_dump(),
        "payload": payload,
        "token_count": len(payload.split()),
        "repeated_plain_tokens": dict(sorted(repeated.items())),
        "audit_unique_description": " ".join(part for part in unique_fields if part),
        "note": "Audit dedup is diagnostic; it does not rewrite the model payload.",
    }


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
    parser.add_argument("--miss-evidence", type=Path,
                        help="Append source-span evidence from regex_miss_evidence.py to the capture CSV")
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
    review_counts: Counter[tuple[str, str]] = Counter()
    review_span_relations: Counter[str] = Counter()
    inspected: dict[str, object] | None = None
    with args.input.open(newline="", encoding="utf-8") as source, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required = {"product_id", *FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing SKU input columns: {sorted(required - set(reader.fieldnames or []))}")
        writer = csv.DictWriter(destination, fieldnames=(
            "product_id", "source_column", "capture_type", "matched_text", "start", "end",
            "original_text", "span_basis", "capture_origin", "integration_status",
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
                    inspected["semantic_profile"] = semantic_profile(
                        row["title"], original, inspected["raw_numeric_captures"],
                        inspected.get("title_lexical_regex_captures", []),
                    )
                    inspected["model_payload_review"] = model_payload_review(row)
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
                        "span_basis": "normalized", "capture_origin": "live_regex",
                        "integration_status": "see_live_parser",
                    })

        if args.miss_evidence:
            with args.miss_evidence.open(newline="", encoding="utf-8") as review_source:
                for item in csv.DictReader(review_source):
                    start, end = int(item["raw_start"]), int(item["raw_end"])
                    if item["original_text"][start:end] != item["surface"]:
                        raise ValueError(f"invalid review source span: {item['product_id']}")
                    writer.writerow({
                        "product_id": item["product_id"],
                        "source_column": item["source_column"],
                        "capture_type": f"candidate_{item['capture_class']}",
                        "matched_text": item["surface"], "start": start, "end": end,
                        "original_text": item["original_text"],
                        "span_basis": "original", "capture_origin": "miss_evidence",
                        "integration_status": item["integration_status"],
                    })
                    review_counts[(item["capture_class"], item["integration_status"])] += 1
                    review_span_relations[item["span_relation"]] += 1

    summary = {
        "input": str(args.input), "detail_csv": str(args.out), "rows_scanned": rows,
        "fields": list(FIELDS), "capture_spans_by_field": dict(field_counts),
        "method": "All lexical regex spans in the cleaned SKU fields, with overlaps merged. Brand matches are diagnostic only; span counts do not assert structured parser acceptance.",
        "review_candidate_evidence": {
            "input": str(args.miss_evidence) if args.miss_evidence else None,
            "capture_spans": sum(review_counts.values()),
            "span_relations": dict(review_span_relations),
            "groups": [
                {"capture_class": category, "integration_status": status, "count": count}
                for (category, status), count in sorted(review_counts.items())
            ],
            "note": "These additional rows use original-text spans and are not trusted parser outputs.",
        },
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
