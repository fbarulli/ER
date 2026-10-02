"""Load ER canonical records and labeled pairs, build System One states."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from typing import Iterable

ER_ROOT = Path(__file__).resolve().parents[1]
RECORDS_CSV = ER_ROOT / "data" / "canonical_records.csv"
PAIRS_CSV = ER_ROOT / "data" / "labeled_pairs.csv"


@dataclass(frozen=True)
class Listing:
    title: str
    brand: str
    attributes: str
    category: str
    description: str


def load_record_index(path: Path = RECORDS_CSV) -> dict[str, Listing]:
    index: dict[str, Listing] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                src = json.loads(row["source_rows"])[0]
            except (ValueError, IndexError):
                src = {}
            index[row["gtin"]] = Listing(
                title=(src.get("title") or "").strip(),
                brand=(src.get("brand") or "").strip(),
                attributes=(src.get("attributes") or "").strip(),
                category=(src.get("category") or "").strip(),
                description=(src.get("description") or "").strip(),
            )
    return index


def load_pairs(path: Path = PAIRS_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            {"gtin1": r["gtin1"], "gtin2": r["gtin2"], "label": int(r["true_label"])}
            for r in csv.DictReader(fh)
        ]


def listing_state(gtin: str, listing: Listing) -> dict[str, str]:
    return {
        "gtin": gtin,
        "title": listing.title,
        "brand": listing.brand,
        "attributes": listing.attributes,
        "category": listing.category,
        "description": listing.description,
    }


def pair_state(gtin1: str, a: Listing, gtin2: str, b: Listing) -> dict:
    return {
        "record_a": listing_state(gtin1, a),
        "record_b": listing_state(gtin2, b),
    }


def sample_pairs(pairs: Iterable[dict], limit: int, balanced: bool = True, seed: int = 13) -> list[dict]:
    import random

    pairs = list(pairs)
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    rng = random.Random(seed)
    if balanced:
        half = max(limit // 2, 1)
        return rng.sample(pos, min(half, len(pos))) + rng.sample(neg, min(limit - min(half, len(pos)), len(neg)))
    return rng.sample(pairs, min(limit, len(pairs)))
