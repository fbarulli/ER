#!/usr/bin/env python3
"""Merge manual reading verdicts into the guarded reading sheet.

Verdicts live in a JSON file keyed by ``"<dimension>|<sku_id>"`` with values
``{"verdict": "right"|"wrong"|"missed", "note": "..."}``. This script writes
them into ``reading_sheet.csv`` (verdict/note columns) and prints a per
dimension tally, so the human read is durable and auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from core.common import RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("--sheet", type=Path,
                        default=RESULTS / "attribute_readings" / "reading_sheet.csv")
    args = parser.parse_args()
    verdicts = json.loads(args.verdicts.read_text())
    rows = list(csv.DictReader(args.sheet.open(newline="", encoding="utf-8")))
    fieldnames = list(rows[0].keys()) if rows else []
    tally: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = f"{row['dimension']}|{row['sku_id']}"
        entry = verdicts.get(key)
        if entry:
            row["verdict"] = entry["verdict"]
            row["note"] = entry.get("note", "")
        tally[row["dimension"]][row["verdict"] or "unmarked"] += 1
    with args.sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for dimension in sorted(tally):
        counts = dict(tally[dimension])
        print(f"{dimension:46s} {counts}")


if __name__ == "__main__":
    main()
