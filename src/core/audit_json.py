"""Stream a row-level audit CSV into one complete JSON array."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def csv_to_json(csv_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open(newline="", encoding="utf-8") as source, json_path.open(
        "w", encoding="utf-8"
    ) as destination:
        destination.write("[\n")
        for index, row in enumerate(csv.DictReader(source)):
            if index:
                destination.write(",\n")
            destination.write(json.dumps(row, ensure_ascii=False))
        destination.write("\n]\n")
