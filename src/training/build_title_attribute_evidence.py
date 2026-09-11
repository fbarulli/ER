"""Build the shared, all-row title-attribute evidence sidecar.

The reconciliation lane owns source schema, GTIN trust, and decisions.  The
NER lane owns rich title spans and packaging vocabulary.  This bridge keeps
both forms of evidence together without changing a gate, label, or model
payload: promotion into a decision requires a separate measured change.

No row is excluded.  Missing title/brand/attribute values are written as
empty evidence and counted in the summary, so coverage loss is visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ner.ner_product_attributes import (
    Candidate,
    extract_title_attributes,
    find_brand_span,
    parse_attribute_details,
    resolve_candidates,
)
from pipeline import clean_sku_text, normalize_text
from core.common import COLUMN_MAPPING, DATA_PATH, F, RESULTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _word_count(text: str) -> int:
    return len(text.split())


def build(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not source.is_file():
        raise FileNotFoundError(
            f"raw export missing: {source}. Pass --input explicitly or restore "
            f"the configured source at {DATA_PATH}; no fallback is used."
        )
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [column for column in COLUMN_MAPPING if column not in raw.columns]
    if missing:
        raise ValueError(f"raw export missing mapped columns {missing}; available: {list(raw.columns)}")
    df = raw.rename(columns=COLUMN_MAPPING)
    product_ids = df["product_id"].fillna("").astype(str).str.strip()
    if product_ids.eq("").any():
        raise ValueError(
            f"{int(product_ids.eq('').sum())} rows have an empty product_id; "
            "the evidence sidecar requires a one-to-one source key"
        )
    if product_ids.duplicated().any():
        raise ValueError(
            f"{int(product_ids.duplicated(keep=False).sum())} rows share a "
            "product_id; resolve source-key duplication before building a "
            "one-row-per-product evidence sidecar"
        )
    rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    removed_tokens: Counter[str] = Counter()
    no_title = 0
    no_brand_span = 0
    for row in df.itertuples(index=False):
        data = row._asdict()
        title = str(data["title"]).strip()
        cleaned_title = clean_sku_text(title)
        original_words = _word_count(title)
        cleaned_words = _word_count(cleaned_title)
        removed_tokens.update(Counter(normalize_text(title).split()) - Counter(cleaned_title.split()))
        brand = data["brand"]
        parsed = extract_title_attributes(title)
        brand_candidate = find_brand_span(title, brand)
        candidates = [
            Candidate(entity["start"], entity["end"], entity["label"], None, "title")
            for entity in parsed["entities"]
        ]
        if brand_candidate is not None:
            candidates.append(brand_candidate)
        entities = [item.entity() for item in resolve_candidates(candidates)]
        for entity in entities:
            start, end = entity["start"], entity["end"]
            if not (0 <= start < end <= len(title)):
                raise ValueError(
                    f"invalid {entity['label']} span ({start}, {end}) for "
                    f"product_id={data['product_id']!r}"
                )
        label_counts.update(entity["label"] for entity in entities)
        attribute = parse_attribute_details(data["attributes"])
        if not title:
            no_title += 1
        if brand_candidate is None:
            no_brand_span += 1
        rows.append({
            "product_id": data["product_id"],
            "retailer": data["retailer"],
            "country": data["country"],
            "gtin_raw": data["barcode"],
            "title": title,
            "cleaned_title": cleaned_title,
            "original_word_count": original_words,
            "cleaned_word_count": cleaned_words,
            "words_removed": original_words - cleaned_words,
            "words_removed_pct": ((original_words - cleaned_words) / original_words * 100) if original_words else 0.0,
            "brand": brand,
            "entities": _json(entities),
            "volume_ml": _json(parsed["volume_ml"]),
            "weight_g": _json(parsed["weight_g"]),
            "pack_count": _json(parsed["pack_count"]),
            "package_types": _json(parsed["package_types"]),
            "package_formats": _json(parsed["package_formats"]),
            "package_materials": _json(parsed["package_materials"]),
            "attribute_volume_ml": _json(attribute.get("attribute_volume_ml", [])),
            "attribute_package_details": _json(attribute.get("attribute_package_details", [])),
            "attribute_raw": _json(attribute.get("attribute_raw", {})),
        })
    evidence = pd.DataFrame(rows)
    if len(evidence) != len(raw):
        raise AssertionError(
            f"evidence row count {len(evidence)} != source row count {len(raw)}"
        )
    summary_rows = [
        {"metric": "source_path", "value": str(source.resolve()), "detail": "explicit input provenance"},
        {"metric": "source_sha256", "value": _sha256(source), "detail": "input bytes fingerprint"},
        {"metric": "rows", "value": len(evidence), "detail": "all raw rows retained"},
        {"metric": "empty_title_rows", "value": no_title, "detail": "written with empty evidence"},
        {"metric": "catalog_brand_not_found_in_title_rows", "value": no_brand_span, "detail": "written without BRAND span"},
        {"metric": "original_title_words", "value": int(evidence["original_word_count"].sum()), "detail": "whitespace-token count before shared cleaning"},
        {"metric": "cleaned_title_words", "value": int(evidence["cleaned_word_count"].sum()), "detail": "whitespace-token count after shared cleaning"},
        {"metric": "title_words_removed", "value": int(evidence["words_removed"].sum()), "detail": "original minus cleaned title tokens"},
        {"metric": "title_words_removed_pct", "value": round(float(evidence["words_removed"].sum() / evidence["original_word_count"].sum() * 100), 3) if evidence["original_word_count"].sum() else 0.0, "detail": "corpus-level percentage; empty titles contribute zero"},
    ]
    summary_rows.extend(
        {"metric": f"entity_{label}", "value": count, "detail": "non-overlapping title span count"}
        for label, count in sorted(label_counts.items())
    )
    removed = pd.DataFrame(sorted(removed_tokens.items(), key=lambda item: (-item[1], item[0])), columns=["token", "occurrences_removed"])
    return evidence, pd.DataFrame(summary_rows), removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    evidence, summary, removed = build(args.input)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for filename, frame in (
        (F["title_attribute_evidence"], evidence),
        (F["title_attribute_summary"], summary),
        (F["title_removed_tokens"], removed),
    ):
        path = RESULTS / filename
        frame.to_csv(path, index=False)
        print(f"[title-attributes] wrote {path} ({len(frame):,} rows)", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
