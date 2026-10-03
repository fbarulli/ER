#!/usr/bin/env python3
"""Capture exact source spans for every remaining regex-miss candidate.

These are evidence records. A captured ingredient or title word is not a
trusted matching claim merely because its source span is known.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from core.common import F, TRAIN_ROOT
from core.critical_attributes import normalized_attribute_text
if __package__:
    from .regex_capture_review import ATTRIBUTE_ITEM_RE, SWEETENER_TYPES
    from .regex_miss_review import REJECTED_PACKAGE_TYPES
else:
    from regex_capture_review import ATTRIBUTE_ITEM_RE, SWEETENER_TYPES
    from regex_miss_review import REJECTED_PACKAGE_TYPES


DESCRIPTION_CUES = {
    "carbonation": (
        ("still", re.compile(r"\b(?:non[- ]?carbonated|uncarbonated|not carbonated|still water)\b", re.I)),
        ("carbonated", re.compile(r"\b(?:(?<!non-)(?<!non )carbonated|sparkling|fizzy|soda pop)\b", re.I)),
        ("tablet_form", re.compile(r"\beffervescent\s+(?:tablets?|tabs?)\b", re.I)),
        ("bubble_tea_style", re.compile(r"\b(?:bubble tea|boba)\b", re.I)),
    ),
    "sweetener": (
        ("no_sugar", re.compile(r"\b(?:no sugar|sugar[- ]free|zero sugar|without sugar)\b", re.I)),
        ("no_added_sugar", re.compile(r"\b(?:no added sugar|without added sugar)\b", re.I)),
        ("low_sugar", re.compile(r"\b(?:low|less|reduced)\s+(?:in\s+)?sugar\b", re.I)),
        ("sugar_ingredient", re.compile(r"\b(?:cane sugar|contains sugar|with sugar|real sugar|pure sugar)(?![- ]free\b)\b", re.I)),
        ("no_sweeteners", re.compile(r"\b(?:no|without)\s+(?:(?:added|artificial)\s+)?sweeteners?\b", re.I)),
        ("sweetened_with", re.compile(r"\bsweetened\s+with\s+[a-z]+(?:\s+[a-z]+)?\b", re.I)),
    ),
    "pulp": (
        ("no_pulp", re.compile(r"\b(?:no pulp|without pulp|pulp[- ]free)\b", re.I)),
        ("with_pulp", re.compile(r"\b(?:with|contains)\s+(?:fruit\s+)?pulp\b|\bjuice\s+w\s+pulp\b", re.I)),
        ("pulp_ingredient", re.compile(r"\bconcentrates\s+and\s+pulps?\b", re.I)),
        ("pulp_ingredient", re.compile(r"\b(?:fruit|aloe|orange)\s+pulp\b", re.I)),
        ("pulp_press_brand", re.compile(r"\bpulp\s+press\b", re.I)),
    ),
}
ATTRIBUTE_CUES = {
    "carbonated", "still", "no_sugar", "no_added_sugar", "low_sugar",
    "sugar_ingredient", "no_sweeteners", "sweetened_with", "no_pulp",
    "with_pulp", "pulp_ingredient",
}


def source_span(row: dict[str, str]) -> tuple[str, str, int, int, str, str]:
    """Locate a candidate in the original title or declared attribute value."""
    candidate = row["candidate"]
    if row["source"] == "sku_name_eng":
        title = row["sku_name_eng"]
        words = normalized_attribute_text(candidate).split()
        pattern = re.compile(
            r"(?<!\w)" + r"[\W_]*".join(map(re.escape, words)) + r"(?!\w)",
            re.IGNORECASE,
        )
        match = pattern.search(title)
        relation = "exact_candidate"
        if match is None:
            # Residual cleanup can remove an intervening unit and create an
            # apparent phrase such as "30 pk" from raw "30 1l PK".
            gap = r"(?:[\W_]*\w+){0,2}[\W_]*"
            match = re.search(
                r"(?<!\w)" + gap.join(map(re.escape, words)) + r"(?!\w)",
                title, re.IGNORECASE,
            )
            relation = "covering_residual_gap"
        if match is None:
            # The cleaner may also split a compact pack/volume token such as
            # raw "6pk32oz" into the surviving candidate "6pk".
            match = re.search(re.escape(candidate), title, re.IGNORECASE)
            relation = "embedded_candidate"
        if match is None and len(words) > 1:
            raw_tokens = [(normalized_attribute_text(item.group()), item.start(), item.end())
                          for item in re.finditer(r"\w+", title)]
            best: tuple[int, int] | None = None
            for index, (word, start, _) in enumerate(raw_tokens):
                if word != words[0]:
                    continue
                next_word = 1
                for later_word, _, end in raw_tokens[index + 1:]:
                    if later_word == words[next_word]:
                        next_word += 1
                        if next_word == len(words):
                            if best is None or end - start < best[1] - best[0]:
                                best = (start, end)
                            break
            if best is not None:
                return "sku_name_eng", "", *best, title[best[0]:best[1]], "synthetic_residual_phrase"
        if match is None:
            raise ValueError(f"title candidate has no raw span: {row['sku_id']} {candidate!r}")
        return "sku_name_eng", "", match.start(), match.end(), match.group(), relation

    if row["source"] == "attribute":
        field_name, separator, value = candidate.partition(":")
        if not separator:
            raise ValueError(f"declared candidate lacks field: {row['sku_id']} {candidate!r}")
        for item in ATTRIBUTE_ITEM_RE.finditer(row["attribute"]):
            if (normalized_attribute_text(item.group(1)) == normalized_attribute_text(field_name)
                    and normalized_attribute_text(item.group(2)) == normalized_attribute_text(value)):
                surface = item.group(2).strip()
                start = item.start(2) + len(item.group(2)) - len(item.group(2).lstrip())
                return "attribute", item.group(1).strip(), start, start + len(surface), surface, "exact_candidate"
            if normalized_attribute_text(item.group(1)) == normalized_attribute_text(field_name):
                for part in re.finditer(r"[^,/&]+", item.group(2)):
                    if normalized_attribute_text(part.group()) == normalized_attribute_text(value):
                        surface = part.group().strip()
                        start = item.start(2) + part.start() + len(part.group()) - len(part.group().lstrip())
                        return "attribute", item.group(1).strip(), start, start + len(surface), surface, "exact_candidate"
        raise ValueError(f"declared candidate has no raw span: {row['sku_id']} {candidate!r}")
    raise ValueError(f"unsupported source column: {row['source']!r}")


def capture_class(row: dict[str, str]) -> tuple[str, list[str], str]:
    """Separate observed values from the matcher semantics they may need."""
    if row["reason"] == "unrecognized_sweetener_value":
        return "unmapped_sweetener_declaration", [row["candidate"].partition(":")[2].strip()], "lexicon_review"
    if row["reason"] == "ingredient_outside_claim_classes":
        values = [normalized_attribute_text(part).replace(" ", "_")
                  for part in row["candidate"].partition(":")[2].split(",")]
        values = [value for value in values if value]
        unknown = {value.replace("_", " ") for value in values} - SWEETENER_TYPES
        if not unknown:
            return "sweetener_ingredient_types", values, "separate_typed_field"
        if unknown == {"unsweetened"}:
            return "unsweetened_declaration", values, "separate_unsweetened_claim"
        return "unmapped_sweetener_declaration", values, "lexicon_review"
    if row["reason"] == "declared_field_parser_empty":
        value = normalized_attribute_text(row["candidate"].partition(":")[2])
        code = REJECTED_PACKAGE_TYPES.get(value)
        if code:
            return "rejected_package_type", [value.replace(" ", "_")], code
        return "declared_package_type", [value.replace(" ", "_")], "package_ontology_review"
    if row["reason"] == "regex_residual_parser_empty":
        return "title_signal", [normalized_attribute_text(row["candidate"])], "context_rule_review"
    raise ValueError(f"unhandled miss reason: {row['reason']!r}")


def description_support(dimension: str, description: str) -> tuple[str, list[dict[str, object]]]:
    """Find explicit supporting phrases while keeping the decision auditable."""
    if not description.strip():
        return "no_description", []
    cues = []
    for label, pattern in DESCRIPTION_CUES.get(dimension, ()):
        match = pattern.search(description)
        if match:
            cues.append({
                "label": label, "start": match.start(), "end": match.end(),
                "surface": match.group(),
                "excerpt": " ".join(description[max(0, match.start() - 55):match.end() + 55].split()),
            })
    labels = {cue["label"] for cue in cues}
    if ({"carbonated", "still"} <= labels or
            {"no_sugar", "sugar_ingredient"} <= labels or
            {"no_pulp", "with_pulp"} <= labels):
        return "conflicting_description_cues", cues
    if labels & ATTRIBUTE_CUES:
        return "explicit_attribute_cue", cues
    return ("context_only" if cues else "no_explicit_cue"), cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRAIN_ROOT / "results" / "regex_miss_candidates.csv")
    parser.add_argument("--dataset", type=Path, default=F["dataset_deduped"])
    parser.add_argument("--out", type=Path, default=TRAIN_ROOT / "results" / "regex_miss_evidence.csv")
    parser.add_argument("--summary", type=Path, default=TRAIN_ROOT / "results" / "regex_miss_evidence_summary.json")
    parser.add_argument("--miss-summary", type=Path, default=TRAIN_ROOT / "results" / "regex_miss_summary.json",
                        help="Add source-span coverage to the existing miss summary")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "sku_id", "reason", "dimension", "candidate", "source_column",
        "source_field", "raw_start", "raw_end", "surface", "original_text",
        "span_relation", "capture_class",
        "canonical_values", "integration_status", "description_status", "description_cues",
    )
    counts: Counter[tuple[str, str]] = Counter()
    span_counts: Counter[str] = Counter()
    description_counts: Counter[str] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows = 0
    with args.input.open(newline="", encoding="utf-8") as source:
        misses = list(csv.DictReader(source))
    wanted_ids = {row["sku_id"] for row in misses if row["source"] == "sku_name_eng"}
    descriptions: dict[str, str] = {}
    with args.dataset.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["sku_id"] in wanted_ids:
                descriptions[row["sku_id"]] = row["description_short_eng"]
    if wanted_ids - descriptions.keys():
        raise ValueError(f"title rows absent from dataset: {sorted(wanted_ids - descriptions.keys())[:5]}")
    with args.out.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in misses:
            column, field, start, end, surface, relation = source_span(row)
            if row[column][start:end] != surface:
                raise AssertionError(f"source span mismatch: {row['sku_id']}")
            category, values, status = capture_class(row)
            description_status, description_cues = (
                description_support(row["dimension"], descriptions[row["sku_id"]])
                if column == "sku_name_eng" else ("not_needed_for_declared_field", [])
            )
            writer.writerow({
                "sku_id": row["sku_id"], "reason": row["reason"],
                "dimension": row["dimension"], "candidate": row["candidate"],
                "source_column": column, "source_field": field,
                "raw_start": start, "raw_end": end, "surface": surface,
                "original_text": row[column],
                "span_relation": relation,
                "capture_class": category,
                "canonical_values": json.dumps(values, ensure_ascii=False),
                "integration_status": status,
                "description_status": description_status,
                "description_cues": json.dumps(description_cues, ensure_ascii=False),
            })
            rows += 1
            key = (category, status)
            counts[key] += 1
            span_counts[relation] += 1
            description_counts[description_status] += 1
            if row["sku_id"] not in examples[key] and len(examples[key]) < 5:
                examples[key].append(row["sku_id"])
    summary = {
        "input": str(args.input), "detail_csv": str(args.out),
        "candidate_mentions": rows, "captured_mentions": sum(counts.values()),
        "unlocated_mentions": rows - sum(counts.values()),
        "span_relations": dict(span_counts),
        "description_statuses": dict(description_counts),
        "groups": [
            {"capture_class": category, "integration_status": status,
             "count": count, "example_sku_ids": examples[(category, status)]}
            for (category, status), count in sorted(counts.items())
        ],
        "note": "All spans refer to original source text. Capture status does not grant matcher or training eligibility.",
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    miss_summary = json.loads(args.miss_summary.read_text(encoding="utf-8"))
    if int(miss_summary["candidate_mentions"]) != rows:
        raise ValueError(
            "miss summary and evidence input have different candidate counts: "
            f"{miss_summary['candidate_mentions']} != {rows}"
        )
    miss_summary.pop("uncaptured_candidate_mentions", None)
    miss_summary.pop("parser_gap_mentions", None)
    miss_summary["capture_review"] = {
        "evidence_csv": str(args.out),
        "evidence_summary": str(args.summary),
        "captured_candidate_mentions": rows,
        "unlocated_candidate_mentions": rows - sum(counts.values()),
        "span_relations": dict(span_counts),
        "evidence_classes": summary["groups"],
        "title_description_statuses": {
            status: count for status, count in description_counts.items()
            if status != "not_needed_for_declared_field"
        },
        "note": "Locating a source span does not resolve a parser gap. candidate_mentions remains the unresolved review count; these are candidates, not confirmed errors.",
    }
    args.miss_summary.write_text(json.dumps(miss_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"candidates={rows:,} captured={sum(counts.values()):,} unlocated={rows - sum(counts.values()):,}")
    for item in summary["groups"]:
        print(f"  {item['count']:>6,} {item['capture_class']}: {item['integration_status']}")


if __name__ == "__main__":
    main()
