#!/usr/bin/env python3
"""Link likely attribute misses to SKU rows and the live parser's output.

Candidates are review leads, not precision/recall labels. A missing structured
value can be deliberate (for example, sweetener ingredients are not one of the
current sweetener claim classes).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

from core.audit_json import csv_to_json
from core.attribute_conflicts import sku_attribute_info
from core.common import TRAIN_ROOT
from core.critical_attributes import FLAVOR_ALIASES, FLAVOR_LEXICON, normalized_attribute_text


ATTRIBUTE_ITEM_RE = re.compile(r"(?:^|;)\s*([^:;]+):\s*([^;]*)")
FLAVOR_CHOICES = tuple(sorted(FLAVOR_LEXICON))
TITLE_SIGNALS = {
    "volume": re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:ml|millilit(?:er|re)s?|cl|lit(?:er|re)s?|l|fl\s*oz|oz)\b"),
    "pack": re.compile(r"\b(?:pack\s+(?:of\s+)?\d+|\d+\s*(?:pack|pk|ct|count))\b"),
    "carbonation": re.compile(r"\b(?:bubbles?|bubbly|fizz|sparkle|effervescent|non\s+sparkling)\b"),
    "sweetener": re.compile(r"\b(?:stevia|sucralose|aspartame|acesulfame|sweeteners?|sweetened|sugar)\b"),
    "pulp": re.compile(r"\b(?:pulp|pulpy)\b"),
}
FIELD_DIMENSIONS = {
    "volume": "volume",
    "count per unit": "pack",
    "pack type": "package_type",
    "flavour": "flavor",
    "flavor": "flavor",
    "carbonization": "carbonation",
    "carbonation": "carbonation",
    "sweetener": "sweetener",
    "pulp": "pulp",
}


@lru_cache(maxsize=None)
def flavor_suggestion(value: str) -> str:
    words = re.findall(r"[a-z]+", normalized_attribute_text(value))
    suggestions = [result[0] for word in words
                   if (result := process.extractOne(word, FLAVOR_CHOICES, scorer=fuzz.ratio, score_cutoff=90))]
    return ", ".join(dict.fromkeys(suggestions))


def parsed_value(info: dict, dimension: str) -> str:
    value = info.get("flavor_set" if dimension == "flavor" else dimension) or set()
    return ", ".join(sorted(str(item) for item in value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRAIN_ROOT / "results" / "regex_residual_rows.csv")
    parser.add_argument("--out", type=Path, default=TRAIN_ROOT / "results" / "regex_miss_candidates.csv")
    parser.add_argument("--detail-json", type=Path, help="Optional full row-level JSON; large")
    parser.add_argument("--summary", type=Path, default=TRAIN_ROOT / "results" / "regex_miss_summary.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    columns = ("product_id", "source", "dimension", "reason", "candidate", "parser_value",
               "fuzzy_flavor_suggestion", "title", "attributes", "live_residual")
    counts: Counter[tuple[str, str, str, str]] = Counter()
    reason_totals: Counter[str] = Counter()
    examples: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    rows_seen = 0
    with args.input.open(newline="", encoding="utf-8") as source, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required = {"product_id", "title", "attributes", "live_regex_title", "live_regex_attributes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing audit columns: {sorted(required - set(reader.fieldnames or []))}")
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in reader:
            rows_seen += 1
            title = row["title"]
            attributes = row["attributes"]
            title_residual = row["live_regex_title"]
            attribute_residual = row["live_regex_attributes"]
            title_hits = [(dimension, match.group()) for dimension, pattern in TITLE_SIGNALS.items()
                          for match in pattern.finditer(title_residual)]
            items = [(normalized_attribute_text(key), value.strip())
                     for key, value in ATTRIBUTE_ITEM_RE.findall(attributes)]
            relevant = [(FIELD_DIMENSIONS[key], key, value) for key, value in items
                        if key in FIELD_DIMENSIONS and value]
            if not title_hits and not relevant:
                continue
            info = sku_attribute_info(title, attributes)

            def emit(source_name: str, dimension: str, reason: str, candidate: str, residual_text: str) -> None:
                candidate = " ".join(candidate.lower().split())
                parsed = parsed_value(info, dimension)
                key = (source_name, dimension, reason, candidate)
                counts[key] += 1
                reason_totals[reason] += 1
                product_id = row["product_id"]
                if product_id not in examples[key] and len(examples[key]) < 5:
                    examples[key].append(product_id)
                writer.writerow({
                    "product_id": product_id, "source": source_name, "dimension": dimension,
                    "reason": reason, "candidate": candidate, "parser_value": parsed,
                    "fuzzy_flavor_suggestion": flavor_suggestion(candidate) if dimension == "flavor" else "",
                    "title": title, "attributes": attributes, "live_residual": residual_text,
                })

            for dimension, phrase in title_hits:
                if not info.get(dimension):
                    emit("title", dimension, "regex_residual_parser_empty", phrase, title_residual)
            for dimension, field_name, value in relevant:
                if dimension == "flavor":
                    # Inspect each declared flavor separately so a known lemon
                    # does not hide a missing guava in a mixed-flavor field.
                    for item in re.split(r"[,/;&]", value):
                        candidate = normalized_attribute_text(item).strip()
                        if not candidate:
                            continue
                        candidate_tokens = {FLAVOR_ALIASES.get(token, token) for token in candidate.split()}
                        if not candidate_tokens & set(info.get("flavor_set") or set()):
                            emit("attributes", dimension, "declared_flavor_unrecognized", candidate, attribute_residual)
                elif not info.get(dimension):
                    reason = "declared_field_parser_empty"
                    if dimension == "sweetener" and value.casefold() not in {"sugar", "diet", "no sugar", "no added sugar"}:
                        reason = "ingredient_outside_claim_classes"
                    emit("attributes", dimension, reason, f"{field_name}: {value}", attribute_residual)

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary = {
        "input": str(args.input), "detail_csv": str(args.out), "rows_scanned": rows_seen,
        "candidate_mentions": sum(counts.values()),
        "reason_counts": dict(reason_totals),
        "method": "Title residual phrase flags when the matching live parser dimension is empty; explicit attribute fields flagged when unrecognized/empty. Fuzzy flavor names are suggestions only.",
        "groups": [
            {"source": key[0], "dimension": key[1], "reason": key[2], "candidate": key[3],
             "count": count, "example_product_ids": examples[key]}
            for key, count in ranked
        ],
    }
    if args.detail_json:
        csv_to_json(args.out, args.detail_json)
        summary["detail_json"] = str(args.detail_json)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rows={rows_seen:,} candidates={sum(counts.values()):,} detail={args.out} summary={args.summary}")
    for item in summary["groups"][:25]:
        print(f"{item['count']:>6,}  {item['source']:<10} {item['dimension']:<13} {item['candidate']:<45} ids={','.join(item['example_product_ids'])}")


if __name__ == "__main__":
    main()
