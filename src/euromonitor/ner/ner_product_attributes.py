"""Weak-label product measurements and packaging for multi-label NER.

The extraction is deliberately conservative: every emitted span is copied from
the product title, spans never overlap, and normalized values are kept as audit
data rather than being substituted into the training text.
"""

from __future__ import annotations

import json
import re
import unicodedata
import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


NUMBER = r"\d+(?:[.,]\d+)?"
VOLUME_RE = re.compile(
    rf"(?P<value>{NUMBER})\s*(?P<unit>"
    r"fl\.?\s*oz|fluid\s+ounces?|millilit(?:er|re)s?|ml|"
    r"centilit(?:er|re)s?|cl|lit(?:er|re)s?|ltr|l|"
    r"gallons?|gal|quarts?|qt|pints?|pt)\b",
    re.IGNORECASE,
)
WEIGHT_RE = re.compile(
    rf"(?P<value>{NUMBER})\s*(?P<unit>kg|kilograms?|grams?|gr|g|"
    r"milligrams?|mg|pounds?|lbs?|lb|ounces?|oz)\b",
    re.IGNORECASE,
)
PACK_COUNT_RES = (
    re.compile(rf"\b(?P<count>\d+)\s*(?=x|×)", re.IGNORECASE),
    re.compile(
        r"\b(?:pack|pk|case|carton|box|tray|bundle|set)\s+(?:of\s+)?"
        r"(?P<count>\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<count>\d+)\s*(?:pack|pk|ct|count|pcs?\.?(?:\b)|"
        r"pieces?|units?|bottles?|cans?|sachets?|sticks?|capsules?|"
        r"tablets?)\b",
        re.IGNORECASE,
    ),
)

PACKAGE_TYPE_RE = re.compile(
    r"\b(?:bottle(?:s)?|can(?:s)?|carton(?:s)?|pouch(?:es)?|jar(?:s)?|"
    r"box(?:es)?|tetra\s*pak|sachet(?:s)?|tub(?:s)?|cup(?:s)?|pot(?:s)?|"
    r"bag(?:s)?|packet(?:s)?|blister(?:s)?|keg(?:s)?|dispenser(?:s)?|"
    r"tin(?:s)?|tube(?:s)?|bucket(?:s)?)\b",
    re.IGNORECASE,
)
PACKAGE_FORMAT_RE = re.compile(
    r"\b(?:multi[-\s]?pack|variety\s+pack|mixed\s+pack|assorted\s+pack|"
    r"value\s+pack|family\s+pack|case|bundle|tray|display\s+pack)\b",
    re.IGNORECASE,
)
PACKAGE_MATERIAL_RE = re.compile(
    r"\b(?:plastic|glass|paper|cardboard|aluminium|aluminum|metal|steel|"
    r"pet|flexible)\b",
    re.IGNORECASE,
)
ATTRIBUTE_ITEM_RE = re.compile(r"\s*([^:;]+):\s*([^;]+)")
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
SPACE_RE = re.compile(r"\s+")

VOLUME_TO_ML = {
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "cl": 10.0,
    "centiliter": 10.0,
    "centiliters": 10.0,
    "centilitre": 10.0,
    "centilitres": 10.0,
    "l": 1000.0,
    "ltr": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "floz": 29.5735295625,
    "fluidounce": 29.5735295625,
    "fluidounces": 29.5735295625,
    "gal": 3785.411784,
    "gallon": 3785.411784,
    "gallons": 3785.411784,
    "qt": 946.352946,
    "quart": 946.352946,
    "quarts": 946.352946,
    "pt": 473.176473,
    "pint": 473.176473,
    "pints": 473.176473,
}
WEIGHT_TO_G = {
    "mg": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
    "g": 1.0,
    "gr": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.349523125,
    "ounce": 28.349523125,
    "ounces": 28.349523125,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}
PACKAGE_TYPE_ALIASES = {
    "bottles": "bottle", "cans": "can", "cartons": "carton",
    "pouches": "pouch", "jars": "jar", "boxes": "box",
    "sachets": "sachet", "tubs": "tub", "cups": "cup", "pots": "pot",
    "bags": "bag", "packets": "packet", "blisters": "blister",
    "kegs": "keg", "dispensers": "dispenser", "tins": "tin",
    "tubes": "tube", "buckets": "bucket", "tetrapak": "tetra pak",
}


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    label: str
    normalized: Any
    source: str

    def entity(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label}


def _unit_key(unit: str) -> str:
    return re.sub(r"[.\s]", "", unit.lower())


def _number(value: str) -> float:
    return float(value.replace(",", "."))


def _canonical_package_type(value: str) -> str:
    compact = SPACE_RE.sub(" ", value.lower()).strip()
    return PACKAGE_TYPE_ALIASES.get(compact.replace(" ", ""), compact)


def _candidates_from_measurements(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for match in VOLUME_RE.finditer(text):
        unit = _unit_key(match.group("unit"))
        candidates.append(Candidate(
            match.start(), match.end(), "VOLUME",
            round(_number(match.group("value")) * VOLUME_TO_ML[unit], 3), "title",
        ))
    for match in WEIGHT_RE.finditer(text):
        unit = _unit_key(match.group("unit"))
        candidates.append(Candidate(
            match.start(), match.end(), "WEIGHT",
            round(_number(match.group("value")) * WEIGHT_TO_G[unit], 3), "title",
        ))
    return candidates


def _candidates_from_package_details(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for pattern in PACK_COUNT_RES:
        for match in pattern.finditer(text):
            start, end = match.span("count")
            candidates.append(Candidate(
                start, end, "PACK_COUNT", int(match.group("count")), "title",
            ))
    for label, pattern in (
        ("PACKAGE_TYPE", PACKAGE_TYPE_RE),
        ("PACKAGE_FORMAT", PACKAGE_FORMAT_RE),
        ("PACKAGE_MATERIAL", PACKAGE_MATERIAL_RE),
    ):
        for match in pattern.finditer(text):
            normalized = _canonical_package_type(match.group()) if label == "PACKAGE_TYPE" else match.group().lower()
            candidates.append(Candidate(
                match.start(), match.end(), label, normalized, "title",
            ))
    return candidates


def resolve_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep deterministic, non-overlapping spans suitable for spaCy NER."""
    priority = {
        "VOLUME": 0, "WEIGHT": 1, "PACK_COUNT": 2,
        "PACKAGE_TYPE": 3, "PACKAGE_FORMAT": 4, "PACKAGE_MATERIAL": 5,
        "BRAND": -1,
    }
    accepted: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.start, priority.get(item.label, 99), -(item.end - item.start)),
    ):
        if any(candidate.start < item.end and item.start < candidate.end for item in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.start, item.end, item.label))


def extract_title_attributes(text: Any) -> dict[str, Any]:
    """Return title-backed NER candidates and normalized audit features."""
    if pd.isna(text):
        return {"entities": [], "volume_ml": [], "weight_g": [], "pack_count": [],
                "package_types": [], "package_formats": [], "package_materials": []}
    title = str(text)
    candidates = resolve_candidates(
        _candidates_from_measurements(title) + _candidates_from_package_details(title)
    )
    return {
        "entities": [item.entity() for item in candidates],
        "volume_ml": [item.normalized for item in candidates if item.label == "VOLUME"],
        "weight_g": [item.normalized for item in candidates if item.label == "WEIGHT"],
        "pack_count": [item.normalized for item in candidates if item.label == "PACK_COUNT"],
        "package_types": [item.normalized for item in candidates if item.label == "PACKAGE_TYPE"],
        "package_formats": [item.normalized for item in candidates if item.label == "PACKAGE_FORMAT"],
        "package_materials": [item.normalized for item in candidates if item.label == "PACKAGE_MATERIAL"],
    }


def parse_attribute_details(attribute: Any) -> dict[str, Any]:
    """Preserve catalog volume and package metadata for audit/QA, not NER spans."""
    if pd.isna(attribute):
        return {}
    items = {
        key.strip().lower(): value.strip()
        for key, value in ATTRIBUTE_ITEM_RE.findall(str(attribute))
    }
    volume_values = extract_title_attributes(items.get("volume", ""))["volume_ml"]
    if not volume_values and items.get("volume"):
        numeric = re.search(NUMBER, items["volume"])
        if numeric:
            volume_values = [_number(numeric.group())]
    package_values = []
    for key in ("pack type", "pack material type", "sustainable packaging"):
        value = items.get(key, "")
        package_values.extend(_canonical_package_type(item) for item in PACKAGE_TYPE_RE.findall(value))
        package_values.extend(item.lower() for item in PACKAGE_MATERIAL_RE.findall(value))
    return {
        "attribute_volume_ml": volume_values,
        "attribute_package_details": sorted(set(package_values)),
        "attribute_raw": items,
    }


def _normalize_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_brand_span(text: str, brand: Any) -> Candidate | None:
    """Find a whole-token catalog brand span without matching inside BOLT24."""
    if pd.isna(brand):
        return None
    brand_text = re.sub(r"^\s*by\s+", "", str(brand), flags=re.IGNORECASE).strip()
    if not brand_text:
        return None
    direct = re.search(re.escape(brand_text), text, flags=re.IGNORECASE)
    if direct and (direct.start() == 0 or not text[direct.start() - 1].isalnum()) and (
        direct.end() == len(text) or not text[direct.end()].isalnum()
    ):
        return Candidate(direct.start(), direct.end(), "BRAND", brand_text, "catalog")
    brand_norm = _normalize_for_match(brand_text)
    tokens = list(TOKEN_RE.finditer(text))
    for start_index in range(len(tokens)):
        for end_index in range(start_index, len(tokens)):
            candidate = " ".join(match.group() for match in tokens[start_index:end_index + 1])
            normalized = _normalize_for_match(candidate)
            if normalized == brand_norm:
                return Candidate(tokens[start_index].start(), tokens[end_index].end(), "BRAND", brand_text, "catalog")
            if len(normalized) >= len(brand_norm):
                break
    return None


def build_enriched_ner_records(
    dataframe: pd.DataFrame,
    text_col: str = "sku_name_eng",
    brand_col: str = "brand",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Create auditable features plus spaCy-compatible multi-label records."""
    if text_col not in dataframe:
        raise KeyError(f"Missing title column: {text_col}")
    audit_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row_id, row in dataframe.iterrows():
        text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
        if not text:
            continue
        parsed = extract_title_attributes(text)
        attribute = parse_attribute_details(row.get("attribute", ""))
        brand = find_brand_span(text, row.get(brand_col, "")) if brand_col in dataframe else None
        candidates = [
            Candidate(item["start"], item["end"], item["label"], None, "title")
            for item in parsed["entities"]
        ]
        if brand:
            candidates.append(brand)
        entities = [item.entity() for item in resolve_candidates(candidates)]
        audit_rows.append({
            "row_id": row_id, "text": text, "entities": json.dumps(entities, ensure_ascii=False),
            **{key: json.dumps(value, ensure_ascii=False) for key, value in parsed.items() if key != "entities"},
            **{key: json.dumps(value, ensure_ascii=False) for key, value in attribute.items()},
        })
        if entities:
            records.append({"text": text, "entities": entities})
    return pd.DataFrame(audit_rows), records


def write_ner_exports(
    audit: pd.DataFrame,
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a review CSV, JSONL training records, and label-count report."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    audit_path = target / "ner_product_attributes_audit.csv"
    dataset_path = target / "ner_dataset_enriched.jsonl"
    stats_path = target / "ner_product_attribute_label_stats.json"
    audit.to_csv(audit_path, index=False)
    with dataset_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = Counter(entity["label"] for record in records for entity in record["entities"])
    stats_path.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n", encoding="utf-8")
    return {"audit": audit_path, "dataset": dataset_path, "stats": stats_path}


def _entity_tuple(entity: Any) -> tuple[int, int, str]:
    if isinstance(entity, dict):
        return int(entity["start"]), int(entity["end"]), str(entity["label"])
    if isinstance(entity, (list, tuple)) and len(entity) == 3:
        return int(entity[0]), int(entity[1]), str(entity[2])
    raise ValueError(f"Unsupported entity format: {entity!r}")


def merge_ner_records(*record_groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union NER entities by exact text while preserving valid, non-overlapping spans.

    The source order is retained for record order only. A BRAND entity wins when
    it overlaps a weak product-attribute entity, preventing a package or number
    label from overwriting a catalog-backed brand annotation.
    """
    grouped: dict[str, list[Candidate]] = {}
    for records in record_groups:
        for record in records:
            text = str(record.get("text", ""))
            if not text:
                continue
            candidates = grouped.setdefault(text, [])
            for entity in record.get("entities", []):
                start, end, label = _entity_tuple(entity)
                if not (0 <= start < end <= len(text)):
                    raise ValueError(f"Invalid {label} span ({start}, {end}) for {text!r}")
                candidates.append(Candidate(start, end, label, None, "merged"))
    return [
        {"text": text, "entities": [item.entity() for item in resolve_candidates(candidates)]}
        for text, candidates in grouped.items()
        if candidates
    ]


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def merge_ner_jsonl(
    brand_dataset: str | Path,
    attribute_dataset: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Merge brand-only and product-attribute JSONL datasets into one dataset."""
    records = merge_ner_records(
        load_jsonl_records(brand_dataset),
        load_jsonl_records(attribute_dataset),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dict(sorted(Counter(
        entity["label"] for record in records for entity in record["entities"]
    ).items()))


def assert_example_extractions() -> None:
    examples = {
        "Soda 12 x 330ml cans": {"VOLUME", "PACK_COUNT", "PACKAGE_TYPE"},
        "Water 1.5 L glass bottle, pack of 6": {"VOLUME", "PACKAGE_MATERIAL", "PACKAGE_TYPE", "PACK_COUNT"},
        "Protein powder 2 lb pouch": {"WEIGHT", "PACKAGE_TYPE"},
        "12 fl oz variety pack": {"VOLUME", "PACKAGE_FORMAT"},
    }
    for text, expected in examples.items():
        actual = {entity["label"] for entity in extract_title_attributes(text)["entities"]}
        assert expected <= actual, (text, actual)


def main() -> None:
    parser = argparse.ArgumentParser(description="NER product-attribute dataset tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge", help="merge brand and product-attribute JSONL datasets")
    merge.add_argument("brand_dataset")
    merge.add_argument("attribute_dataset")
    merge.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "merge":
        counts = merge_ner_jsonl(args.brand_dataset, args.attribute_dataset, args.output)
        print(json.dumps({"output": args.output, "labels": counts}, indent=2))


if __name__ == "__main__":
    main()
