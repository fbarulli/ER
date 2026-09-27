#!/usr/bin/env python3
"""Remove ER's recognized attribute spans and inspect the remaining text.

This is a lexical regex audit, not a claim that every match becomes a trusted
structured value. Title and attribute residuals are reported separately because
the live extractor reads those fields differently.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from nltk.stem import SnowballStemmer

from core.audit_json import csv_to_json
from core.common import F, TRAIN_ROOT
from core.critical_attributes import FLAVOR_ALIASES, FLAVOR_LEXICON
from ner.ner_product_attributes import (
    PACK_COUNT_RES,
    PACKAGE_FORMAT_RE,
    PACKAGE_MATERIAL_RE,
    PACKAGE_TYPE_RE,
    VOLUME_RE,
    WEIGHT_RE,
)
from pipeline import (
    MINIMAL_STOPWORDS,
    VOLUME_PATTERN_METRIC_EXT,
    VOLUME_PATTERN_US_EXT,
    _VOLUME_PACK_RE,
    _MODEL_STOP,
    normalize_text,
)


# These are the phrase branches in core.critical_attributes.extract_critical_claims.
# Keep each entire claim together so a residual cannot turn "no sugar" into
# an apparent positive "sugar" mention.
CLAIM_PATTERNS = (
    re.compile(r"\b(?:no sugar|zero sugar|sugar free|sugarfree|sugarless|without sugar|free of sugar)\b"),
    re.compile(r"\b(?:no|without) added sugar\b"),
    re.compile(r"\b(?:with added sugar|contains sugar|sweetened with sugar|sweetener sugar|sugar sweetened)\b"),
    re.compile(r"\bdiet\b"),
    re.compile(r"\b(?:non carbonated|uncarbonated|not carbonated|still|carbonated|sparkling|fizzy)\b"),
    re.compile(r"\b(?:no pulp|without pulp|pulp free|free of pulp|with (?:extra )?pulp|contains pulp|pulp yes)\b"),
)
# Product-type branches in pipeline.extract_all (searched against title text).
PRODUCT_TYPE_PATTERNS = (
    re.compile(r"\bcoconut\s+water\b"),
    re.compile(r"\bmineral\s+water\b"),
    re.compile(r"\bwater\b"),
    re.compile(r"\bjuice\b"),
    re.compile(r"\b(?:ice\s+)?tea\b"),
    re.compile(r"\benergy\s+(?:drink|water)\b"),
    re.compile(r"\b(?:soda|soft\s+drink)\b"),
    re.compile(r"\btonic\b"),
)
FLAVOR_RE = re.compile(r"\b(?:" + "|".join(sorted(FLAVOR_LEXICON | FLAVOR_ALIASES.keys(), key=len, reverse=True)) + r")\b")
ATTRIBUTE_VOLUME_RE = re.compile(r"\bvolume\s+\d+(?:[.,]\d+)?(?:\s*(?:ml|cl|l|lt|ltr|cc|oz|qt|pt|gal))?\b")
ATTRIBUTE_PACK_RE = re.compile(r"\bcount per unit\s+\d+\b")
ATTRIBUTE_CAFFEINE_RE = re.compile(r"\bcaffeine\s+\d+(?:\s+\d+)?(?:\s+mg)?\b")
ATTRIBUTE_JUICE_CONTENT_RE = re.compile(r"\bjuice content\s+\d+(?:\s+\d+)?\b")
NATURAL_CLAIM_RE = re.compile(r"\b(?:100\s+(?:percent\s+)?natural|all\s+natural|naturally\s+derived\s+natural)\b")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
FIELDS = ("brand", "title", "attributes")
ROUNDS = ("live_regex", "volume_pack_cleanup", "model_stopwords", "candidate_phrases", "unique_tokens")
STEMMER = SnowballStemmer("english")
# Diagnostic candidates suggested by the earlier residual n-grams. These are
# not part of the live extractor, and this audit does not change production.
CANDIDATE_TITLE_RE = re.compile(
    r"\b(?:cold brew|drink mix|ready to drink|iced coffee|sports drink|"
    r"electrolyte drink|smoothie)\b"
)
CANDIDATE_ATTRIBUTES_RE = re.compile(
    r"\b(?:caffeine\s+\d+(?:\s+\d+)?(?:\s*mg)?|juice content\s+\d+(?:\s+\d+)?)\b"
)


def drop_nearby_repeats(text: str, *, window: int, min_words: int) -> tuple[str, int, int]:
    """Keep the first occurrence of an exact phrase, drop later nearby copies.

    The margin is measured in ORIGINAL token positions between phrase starts.
    Only phrases of at least ``min_words`` tokens qualify; the longest available
    nonoverlapping copy is removed. Earlier kept text is never changed.
    """
    matches = list(TOKEN_RE.finditer(text))
    source = [match.group() for match in matches]
    kept: list[str] = []
    original_positions: list[int] = []
    starts: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    removed_runs = 0
    removed_spans: list[tuple[int, int]] = []
    i = 0
    while i < len(source):
        best = 0
        if i + min_words <= len(source):
            key = tuple(source[i:i + min_words])
            candidates = starts.get(key)
            if candidates:
                while candidates and original_positions[candidates[0]] < i - window:
                    candidates.popleft()
                for start in reversed(candidates):
                    length = min(len(kept) - start, len(source) - i)
                    matched = min_words
                    while matched < length and kept[start + matched] == source[i + matched]:
                        matched += 1
                    if matched > best:
                        best = matched
        if best >= min_words:
            removed_runs += 1
            removed_spans.append((matches[i].start(), matches[i + best - 1].end()))
            i += best
            continue
        kept.append(source[i])
        original_positions.append(i)
        if len(kept) >= min_words:
            start = len(kept) - min_words
            starts[tuple(kept[start:])].append(start)
        i += 1
    if not removed_spans:
        return text, 0, 0
    pieces: list[str] = []
    previous = 0
    for start, end in removed_spans:
        pieces.append(text[previous:start])
        previous = end
    pieces.append(text[previous:])
    return " ".join(" ".join(pieces).split()), len(source) - len(kept), removed_runs


def live_patterns(field: str):
    """Yield the lexical patterns audited for a working SKU input field."""
    if field in {"title", "brand"}:
        for pattern in (VOLUME_PATTERN_METRIC_EXT, VOLUME_PATTERN_US_EXT, VOLUME_RE):
            yield "volume", pattern
        yield "weight", WEIGHT_RE
        for pattern in PACK_COUNT_RES:
            yield "pack", pattern
        yield "package_type", PACKAGE_TYPE_RE
        yield "package_format", PACKAGE_FORMAT_RE
        yield "package_material", PACKAGE_MATERIAL_RE
        for pattern in PRODUCT_TYPE_PATTERNS:
            yield "product_type", pattern
    elif field == "attributes":
        yield "attribute_volume", ATTRIBUTE_VOLUME_RE
        yield "attribute_pack", ATTRIBUTE_PACK_RE
        yield "attribute_caffeine", ATTRIBUTE_CAFFEINE_RE
        yield "attribute_juice_content", ATTRIBUTE_JUICE_CONTENT_RE
        yield "package_type_lexical", PACKAGE_TYPE_RE
    else:
        raise ValueError(f"unsupported SKU input field: {field}")
    yield "flavor", FLAVOR_RE
    yield "natural_claim_lexical", NATURAL_CLAIM_RE
    for pattern in CLAIM_PATTERNS:
        yield "claim", pattern


def residual(text: str, *, field: str, round_name: str,
             seen_tokens: set[str] | None = None) -> tuple[str, Counter[str]]:
    spans: list[tuple[int, int]] = []
    hits: Counter[str] = Counter()

    def collect(name: str, pattern: re.Pattern[str]) -> None:
        matches = list(pattern.finditer(text))
        if matches:
            hits[name] += len(matches)
            spans.extend(match.span() for match in matches)

    if round_name == "live_regex":
        for name, pattern in live_patterns(field):
            collect(name, pattern)
    elif round_name == "volume_pack_cleanup":
        collect("model_volume_pack_cleanup", _VOLUME_PACK_RE)
    elif round_name == "model_stopwords":
        stopwords = MINIMAL_STOPWORDS | (_MODEL_STOP if field == "attributes" else set())
        kept = [token for token in text.split() if token not in stopwords and len(token) > 1]
        removed = len(text.split()) - len(kept)
        if removed:
            hits["model_stopwords"] = removed
        return " ".join(kept), hits
    elif round_name == "candidate_phrases":
        collect("candidate_phrase", CANDIDATE_ATTRIBUTES_RE if field == "attributes" else CANDIDATE_TITLE_RE)
    elif round_name == "unique_tokens":
        if seen_tokens is None:
            raise ValueError("unique_tokens requires a per-product seen set")
        kept = []
        for token in text.split():
            key = STEMMER.stem(token) if token.isalpha() else token
            if key not in seen_tokens:
                seen_tokens.add(key)
                kept.append(token)
        removed = len(text.split()) - len(kept)
        if removed:
            hits["duplicate_tokens"] = removed
        return " ".join(kept), hits
    else:
        raise ValueError(round_name)

    if not spans:
        return text, hits
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces = []
    previous = 0
    for start, end in merged:
        pieces.append(text[previous:start])
        previous = end
    pieces.append(text[previous:])
    return " ".join(TOKEN_RE.findall(" ".join(pieces))), hits


def audit(input_path: Path, rows_path: Path | None, *, top: int,
          repeat_window: int = 0, repeat_min_words: int = 4) -> dict[str, object]:
    totals: Counter[str] = Counter()
    grams = {name: {field: {n: Counter() for n in (2, 3, 4)} for field in FIELDS} for name in ROUNDS}
    description_grams = {n: Counter() for n in (2, 3, 4)}
    longest: dict[str, dict[str, list[tuple[int, int, dict[str, object]]]]] = {
        name: {field: [] for field in FIELDS} for name in ROUNDS
    }
    description_longest: list[tuple[int, int, dict[str, object]]] = []
    if rows_path is not None:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
    from contextlib import nullcontext
    destination_context = rows_path.open("w", newline="", encoding="utf-8") if rows_path else nullcontext(None)
    with input_path.open(newline="", encoding="utf-8") as source, destination_context as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=(
            "product_id", *FIELDS,
            *(f"{name}_{field}" for name in ROUNDS for field in FIELDS),
            "unique_description",
        )) if destination else None
        if writer:
            writer.writeheader()
        for row in reader:
            totals["rows"] += 1
            output = {"product_id": row.get("product_id", ""),
                      **{field: row.get(field, "") for field in FIELDS}}
            seen_tokens: set[str] = set()
            for field in FIELDS:
                source_col = field
                current = normalize_text(output[source_col])
                totals[f"{field}_input_tokens"] += len(TOKEN_RE.findall(current))
                if repeat_window:
                    current, removed, runs = drop_nearby_repeats(
                        current, window=repeat_window, min_words=repeat_min_words)
                    totals[f"{field}_repeat_removed_tokens"] += removed
                    totals[f"{field}_repeat_runs"] += runs
                    if runs:
                        totals[f"{field}_rows_with_repeat"] += 1
                for round_name in ROUNDS:
                    previous_count = len(TOKEN_RE.findall(current))
                    current, hits = residual(current, field=field, round_name=round_name,
                                             seen_tokens=seen_tokens)
                    output[f"{round_name}_{field}"] = current
                    totals.update({f"{round_name}_{field}_{key}": value for key, value in hits.items()})
                    if hits:
                        totals[f"{round_name}_{field}_rows_with_match"] += 1
                    tokens = TOKEN_RE.findall(current)
                    totals[f"{round_name}_{field}_removed_tokens"] += previous_count - len(tokens)
                    totals[f"{round_name}_{field}_residual_tokens"] += len(tokens)
                    for n in (2, 3, 4):
                        grams[round_name][field][n].update(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
                    if current:
                        item = {"product_id": output["product_id"], "length": len(current),
                                "residual": current, "original": output[source_col]}
                        heap = longest[round_name][field]
                        entry = (len(current), totals["rows"], item)
                        if len(heap) < top:
                            heapq.heappush(heap, entry)
                        elif len(current) > heap[0][0]:
                            heapq.heapreplace(heap, entry)
            description = " ".join(output[f"unique_tokens_{field}"] for field in FIELDS).strip()
            output["unique_description"] = description
            description_tokens = TOKEN_RE.findall(description)
            totals["unique_description_tokens"] += len(description_tokens)
            for n in (2, 3, 4):
                description_grams[n].update(" ".join(description_tokens[i:i+n])
                                            for i in range(len(description_tokens) - n + 1))
            if description:
                item = {"product_id": output["product_id"], "length": len(description),
                        "description": description}
                entry = (len(description), totals["rows"], item)
                if len(description_longest) < top:
                    heapq.heappush(description_longest, entry)
                elif len(description) > description_longest[0][0]:
                    heapq.heapreplace(description_longest, entry)
            if writer:
                writer.writerow(output)

    report = {
        "input": str(input_path), "rows_csv": str(rows_path) if rows_path else None,
        "method": "Five cumulative rounds over brand/title/attributes: live lexical regexes; legacy volume/pack cleanup regex; model stopword lists; diagnostic candidate phrases; keep each remaining token once by NLTK Snowball stem across the product in brand/title/attributes order. The first original token spelling is retained. Brand matches are diagnostic and are not structured attribute claims. Rank residuals by character length and word n-grams by occurrences. Regex hit counts may overlap; removed token counts do not.",
        "repeat_rule": {"window_tokens": repeat_window, "minimum_phrase_tokens": repeat_min_words,
                        "margin_basis": "original normalized token positions between phrase starts",
                        "action": "drop later exact contiguous phrase; keep first occurrence"} if repeat_window else None,
        "counts": dict(totals),
        "rounds": {
            name: {
                "longest": {field: [entry[2] for entry in sorted(items, key=lambda entry: (-entry[0], entry[2]["product_id"]))]
                            for field, items in longest[name].items()},
                "ngrams": {field: {str(n): [{"text": phrase, "count": count} for phrase, count in grams[name][field][n].most_common(top)]
                                    for n in (2, 3, 4)} for field in grams[name]},
            } for name in ROUNDS
        },
        "unique_descriptions": {
            "longest": [entry[2] for entry in sorted(description_longest,
                                                     key=lambda entry: (-entry[0], entry[2]["product_id"]))],
            "ngrams": {str(n): [{"text": phrase, "count": count}
                                 for phrase, count in description_grams[n].most_common(top)]
                       for n in (2, 3, 4)},
        },
    }
    return report


def print_report(report: dict[str, object]) -> None:
    totals = report["counts"]
    print(f"rows={totals['rows']:,} residuals={report['rows_csv']}")
    for round_name in ROUNDS:
        print(f"\n{round_name}")
        for field in FIELDS:
            print(f"  {field}: removed={totals[f'{round_name}_{field}_removed_tokens']:,} residual={totals[f'{round_name}_{field}_residual_tokens']:,}")
            final = round_name == ROUNDS[-1]
            for item in report["rounds"][round_name]["longest"][field][:5 if final else 3]:
                value = str(item["residual"]) if final else str(item["residual"])[:160]
                print(f"    longest {item['length']:>4} {item['product_id']}: {value}")
            for n in (2, 3, 4):
                print(f"    {n}-grams: " + ", ".join(f"{g['text']} ({g['count']:,})" for g in report["rounds"][round_name]["ngrams"][field][str(n)][:10 if final else 5]))
    print(f"\nunique descriptions: {totals['unique_description_tokens']:,} tokens")
    for item in report["unique_descriptions"]["longest"][:5]:
        print(f"  longest {item['length']:>4} {item['product_id']}: {item['description']}")
    for n in (2, 3, 4):
        print(f"  {n}-grams: " + ", ".join(f"{g['text']} ({g['count']:,})"
                                       for g in report["unique_descriptions"]["ngrams"][str(n)][:10]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=F["dataset_deduped"])
    parser.add_argument("--out", type=Path, default=TRAIN_ROOT / "results" / "regex_residual_audit.json")
    parser.add_argument("--rows-out", type=Path, default=TRAIN_ROOT / "results" / "regex_residual_rows.csv")
    parser.add_argument("--rows-json-out", type=Path, help="Optional full row-level JSON; large")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--compare-repeats", action="store_true", help="Audit both baseline and repeat-reduced text")
    parser.add_argument("--repeat-window", type=int, default=64, help="Maximum original token distance between repeated phrase starts")
    parser.add_argument("--repeat-min-words", type=int, default=4, help="Minimum exact repeated phrase length")
    args = parser.parse_args()
    if args.top < 1 or args.repeat_window < 1 or args.repeat_min_words < 2:
        parser.error("--top, --repeat-window must be positive; --repeat-min-words must be at least 2")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.compare_repeats:
        baseline = audit(args.input, None, top=args.top)
        reduced = audit(args.input, args.rows_out, top=args.top,
                        repeat_window=args.repeat_window, repeat_min_words=args.repeat_min_words)
        report = {"baseline": baseline, "repeat_reduced": reduced}
        if args.rows_json_out:
            reduced["rows_json"] = str(args.rows_json_out)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.rows_json_out:
            csv_to_json(args.rows_out, args.rows_json_out)
        print(f"comparison={args.out} rows={baseline['counts']['rows']:,} margin={args.repeat_window} tokens minimum_repeat={args.repeat_min_words} tokens")
        for field in FIELDS:
            counts = reduced["counts"]
            print(f"{field}: repeated_tokens_dropped={counts.get(f'{field}_repeat_removed_tokens', 0):,} repeated_runs={counts.get(f'{field}_repeat_runs', 0):,} rows={counts.get(f'{field}_rows_with_repeat', 0):,}")
        for round_name in ROUNDS:
            print(f"\n{round_name}")
            for field in FIELDS:
                before = baseline["counts"][f"{round_name}_{field}_residual_tokens"]
                after = reduced["counts"][f"{round_name}_{field}_residual_tokens"]
                print(f"  {field}: baseline={before:,} repeat_reduced={after:,} delta={after-before:+,}")
                for label, scenario in (("before", baseline), ("after", reduced)):
                    for item in scenario["rounds"][round_name]["longest"][field][:3]:
                        print(f"    {label} longest {item['length']:>4} {item['product_id']}: {item['residual'][:220]}")
                    for n in (2, 3, 4):
                        values = scenario["rounds"][round_name]["ngrams"][field][str(n)][:10]
                        print(f"    {label} {n}-grams: " + ", ".join(f"{g['text']} ({g['count']:,})" for g in values))
    else:
        report = audit(args.input, args.rows_out, top=args.top)
        if args.rows_json_out:
            report["rows_json"] = str(args.rows_json_out)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.rows_json_out:
            csv_to_json(args.rows_out, args.rows_json_out)
        print(f"report={args.out}")
        print_report(report)


if __name__ == "__main__":
    main()
