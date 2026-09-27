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
from core.common import F, TRAIN_ROOT
from core.critical_attributes import FLAVOR_ALIASES, FLAVOR_LEXICON, normalized_attribute_text
from core.sweetener_values import declared_sweeteners, SWEETENER_CLAIMS
from nltk.util import ngrams


ATTRIBUTE_ITEM_RE = re.compile(r"(?:^|;)\s*([^:;]+):\s*([^;]*)")
FLAVOR_CHOICES = tuple(sorted(FLAVOR_LEXICON))
TITLE_SIGNALS = {
    "volume": re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:ml|millilit(?:er|re)s?|cl|lit(?:er|re)s?|l|fl\s*oz|oz)\b"),
    "pack": re.compile(r"\b(?:pack\s+(?:of\s+)?[1-9]\d*(?![\d.,])|[1-9]\d*\s*(?:pack|pk|ct|count))\b"),
    "carbonation": re.compile(r"\b(?:bubbles?|bubbly|fizz|sparkle|effervescent|non\s+sparkling)\b"),
    "sweetener": re.compile(r"\b(?:cane\s+sugar|stevia|sucralose|aspartame|acesulfame(?:\s+potassium)?|sweeteners?|sweetened|sugar)\b"),
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


def title_signals(title: str) -> list[tuple[str, str]]:
    """Match candidate phrases in one title/residual string."""
    return [(dimension, match.group()) for dimension, pattern in TITLE_SIGNALS.items()
            for match in re.finditer(pattern.pattern, title, re.I)]


def is_contiguous_source_phrase(title: str, phrase: str) -> bool:
    """Reject residual-cleanup phrases assembled from separated source spans."""
    source_words = normalized_attribute_text(title).split()
    phrase_words = normalized_attribute_text(phrase).split()
    return bool(phrase_words) and any(
        source_words[index:index + len(phrase_words)] == phrase_words
        for index in range(len(source_words) - len(phrase_words) + 1)
    )


@lru_cache(maxsize=None)
def flavor_suggestion(value: str) -> str:
    words = re.findall(r"[a-z]+", normalized_attribute_text(value))
    suggestions = [result[0] for word in words
                   if (result := process.extractOne(word, FLAVOR_CHOICES, scorer=fuzz.ratio, score_cutoff=90))]
    return ", ".join(dict.fromkeys(suggestions))


def parsed_value(info: dict, dimension: str) -> str:
    if dimension == "sweetener":
        values = set(info.get("sweetener") or set()) | set(info.get("sweetener_type") or set()) | set(info.get("sweetening") or set())
        return ", ".join(sorted(str(item) for item in values))
    value = info.get("flavor_set" if dimension == "flavor" else dimension) or set()
    return ", ".join(sorted(str(item) for item in value))


def contextual_nonclaim(dimension: str, phrase: str, title: str) -> str | None:
    text = normalized_attribute_text(title)
    if dimension == "carbonation":
        if re.search(r"\b(?:bubble teas?|tea bubbles?|bubble milk tea|boba|tapioca|buddha bubbles|bubble gum)\b", text):
            return "bubble_tea_or_boba_context"
        if phrase.casefold() == "fizz" and re.search(r"\brocket fizz\b", text):
            return "rocket_fizz_brand"
        if phrase.casefold() == "effervescent" and re.search(r"\beffervescent(?:\s+\w+){0,3}\s+tablets?\b", text):
            return "tablet_form"
    if dimension == "pulp" and re.search(r"\bpulp (?:and )?press\b|\bpulp story\b", text):
        return "pulp_press_brand"
    return None


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
    assigned_declarations: Counter[str] = Counter()
    title_ngrams: dict[str, Counter[str]] = defaultdict(Counter)
    flagged_rows: set[str] = set()
    contradictions = 0
    resolved_by_parser: Counter[str] = Counter()
    contextual_exclusions: Counter[str] = Counter()
    synthetic_residual_phrases = 0
    with args.input.open(newline="", encoding="utf-8") as source, args.out.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required = {"product_id", "title", "attributes", "live_regex_title", "live_regex_attributes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing audit columns: {sorted(required - set(reader.fieldnames or []))}")
        candidates = list(reader)
        wanted_ids = {row["product_id"] for row in candidates}
        descriptions: dict[str, str] = {}
        with F["dataset_deduped"].open(newline="", encoding="utf-8") as dataset:
            for item in csv.DictReader(dataset):
                if item["product_id"] in wanted_ids:
                    descriptions[item["product_id"]] = item.get("description", "") or ""
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for row in candidates:
            rows_seen += 1
            title = row["title"]
            attributes = row["attributes"]
            title_residual = row["live_regex_title"]
            attribute_residual = row["live_regex_attributes"]
            title_hits = title_signals(title_residual)
            direct_hits = []
            for dimension, phrase in title_hits:
                if is_contiguous_source_phrase(title, phrase):
                    direct_hits.append((dimension, phrase))
                else:
                    synthetic_residual_phrases += 1
            title_hits = direct_hits
            items = [(normalized_attribute_text(key), value.strip())
                     for key, value in ATTRIBUTE_ITEM_RE.findall(attributes)]
            relevant = [(FIELD_DIMENSIONS[key], key, value) for key, value in items
                        if key in FIELD_DIMENSIONS and value]
            if not title_hits and not relevant:
                continue
            description = descriptions.get(row["product_id"], "")
            info = sku_attribute_info(title, attributes, description)
            sweeteners = declared_sweeteners(attributes)
            contradictions += bool(sweeteners["consistency_flags"])

            def emit(source_name: str, dimension: str, reason: str, candidate: str, residual_text: str) -> None:
                candidate = " ".join(candidate.lower().split())
                parsed = parsed_value(info, dimension)
                key = (source_name, dimension, reason, candidate)
                counts[key] += 1
                reason_totals[reason] += 1
                product_id = row["product_id"]
                flagged_rows.add(product_id)
                if product_id not in examples[key] and len(examples[key]) < 5:
                    examples[key].append(product_id)
                writer.writerow({
                    "product_id": product_id, "source": source_name, "dimension": dimension,
                    "reason": reason, "candidate": candidate, "parser_value": parsed,
                    "fuzzy_flavor_suggestion": flavor_suggestion(candidate) if dimension == "flavor" else "",
                    "title": title, "attributes": attributes, "live_residual": residual_text,
                })

            missing_title_dimensions = set()
            for dimension, phrase in title_hits:
                exclusion = contextual_nonclaim(dimension, phrase, title)
                if exclusion:
                    contextual_exclusions[exclusion] += 1
                    continue
                # A declared ingredient fulfills the same explicit title signal.
                if dimension == "sweetener" and normalized_attribute_text(phrase).replace(" ", "_") in info.get("sweetener_type", set()):
                    resolved_by_parser["sweetener_type"] += 1
                    continue
                present = (bool(info.get("sweetener") or info.get("sweetener_type") or info.get("sweetening"))
                           if dimension == "sweetener" else bool(info.get(dimension)))
                if present:
                    resolved_by_parser[dimension] += 1
                else:
                    emit("title", dimension, "regex_residual_parser_empty", phrase, title_residual)
                    missing_title_dimensions.add(dimension)
            words = re.findall(r"[a-z]+", title.casefold())
            for dimension in missing_title_dimensions:
                for size in (2, 3):
                    title_ngrams[dimension].update({" ".join(parts) for parts in ngrams(words, size)
                                                  if TITLE_SIGNALS[dimension].search(" ".join(parts))})
            for dimension, field_name, value in relevant:
                if dimension == "flavor":
                    # Inspect each declared flavor separately so a known lemon
                    # does not hide a missing guava in a mixed-flavor field.
                    for item in re.split(r"[,/;&]", value):
                        candidate = normalized_attribute_text(item).strip()
                        if not candidate:
                            continue
                        candidate_tokens = {FLAVOR_ALIASES.get(token, token) for token in candidate.split()}
                        parsed_flavors = set(info.get("flavor_set") or set())
                        if candidate not in parsed_flavors and not candidate_tokens & parsed_flavors:
                            emit("attributes", dimension, "declared_flavor_unrecognized", candidate, attribute_residual)
                elif dimension == "sweetener":
                    # Check every value against its actual field. A recognized
                    # claim must not conceal an unknown ingredient in the same row.
                    for item in re.split(r"[,/&]", value):
                        candidate = normalized_attribute_text(item)
                        token = candidate.replace(" ", "_")
                        if token in info.get("sweetener_type", set()):
                            assigned_declarations["sweetener_type"] += 1
                        elif token in info.get("sweetening", set()):
                            assigned_declarations["sweetening"] += 1
                        elif candidate in SWEETENER_CLAIMS and info.get("sweetener"):
                            assigned_declarations["sweetener_claim"] += 1
                        elif candidate:
                            emit("attributes", dimension, "unrecognized_sweetener_value", f"{field_name}: {candidate}", attribute_residual)
                elif not info.get(dimension):
                    reason = "declared_field_parser_empty"
                    emit("attributes", dimension, reason, f"{field_name}: {value}", attribute_residual)

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary = {
        "input": str(args.input), "detail_csv": str(args.out), "rows_scanned": rows_seen,
        "candidate_mentions": sum(counts.values()),
        "products_with_candidates": len(flagged_rows),
        "resolved_residual_signals": dict(resolved_by_parser),
        "excluded_contextual_nonclaims": dict(contextual_exclusions),
        "rejected_synthetic_residual_phrases": synthetic_residual_phrases,
        "assigned_sweetener_value_mentions": dict(assigned_declarations),
        "contradictory_sweetener_declaration_rows": contradictions,
        "reason_counts": dict(reason_totals),
        "method": "Unresolved review candidates, not confirmed errors. Residual signals are verified as contiguous original-title phrases; phrases assembled by cleanup are rejected. N-grams and context exclusions summarize title patterns. Explicit sweetener atoms are checked against ingredient, status, or claim fields. Description cues are recorded by regex_miss_evidence. Fuzzy names are suggestions only.",
        "title_signal_ngrams": {dimension: [{"phrase": phrase, "products": count} for phrase, count in counts.most_common(30)]
                                for dimension, counts in sorted(title_ngrams.items())},
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
