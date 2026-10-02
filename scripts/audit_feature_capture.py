"""Fingerprint the complete source-column, attribute-key and date inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import pandas as pd

from core.common import DATA_PATH, data_cfg
from core.manifest import sha256_file
from core.text import normalized_attribute_text
from core.attribute_universe import attribute_registry
from core.date_evidence import extract_date_evidence
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_gate_logic import wiring_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fingerprint = sha256_file(DATA_PATH)
    frame = pd.read_csv(DATA_PATH, **data_cfg().dataset_csv_read.model_dump()).rename(
        columns=data_cfg().column_mapping).fillna("")
    registered = attribute_registry()
    keys, date_keys = Counter(), Counter()
    for cell in frame["attributes"]:
        row_keys = set()
        for part in cell.split(";"):
            if ":" in part:
                key, value = part.split(":", 1)
                if value.strip():
                    row_keys.add(normalized_attribute_text(key))
        keys.update(row_keys)
        date_keys.update(key for key in row_keys if re.search(r"\b(?:date|expiry|expiration|shelf life|best before)\b", key))
    # Screens preserve source strings; numbers and dates alone are not
    # evidence of expiry, product edition, or stable identity.
    patterns = {"calendar_date": r"\b(?:\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}|\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{4})\b",
                "expiry_wording": r"\b(?:best before|best by|use by|expiry|expiration|expires|manufactur(?:e|ing) date|shelf life|tht)\b"}
    screens = {}
    for label, pattern in patterns.items():
        counts, examples = {}, []
        for column in ("title", "attributes", "description", "category_path", "category"):
            selected = frame[frame[column].str.contains(pattern, case=False, regex=True)]
            counts[column] = len(selected)
            examples.extend({"product_id": row.product_id, "column": column,
                             "source": getattr(row, column)}
                            for row in selected.head(10).itertuples())
        screens[label] = {"rows_per_column": counts, "examples": examples}
    date_roles, date_status, date_rows = Counter(), Counter(), set()
    for column in ("title", "attributes", "description", "category_path", "category"):
        for row_index, text in enumerate(frame[column]):
            entries = extract_date_evidence(text)
            if entries:
                date_rows.add(row_index)
            date_roles.update(entry["role"] for entry in entries)
            date_status.update(entry["parse_status"] for entry in entries)
    report = {"scope": "Complete raw source inventory; date screens are lexical observations, not identity labels. Date parser retains ambiguous/reference-only evidence.",
              "source_sha256": fingerprint, "source_rows": len(frame),
              "source_columns": {column: int(frame[column].str.strip().ne("").sum())
                                 for column in data_cfg().column_mapping.values()},
              "registered_keys": {key: keys[key] for key in registered},
              "unregistered_keys": {key: count for key, count in keys.items() if key not in registered},
              "date_attribute_keys": dict(date_keys), "date_screens": screens,
              "date_extraction": {"unique_rows_with_evidence": len(date_rows), "entry_role_counts": dict(date_roles),
                                  "entry_parse_status_counts": dict(date_status)},
              "wiring": wiring_inventory({})}
    assert fingerprint == sha256_file(DATA_PATH), "source changed during audit"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("source_rows", "source_columns", "unregistered_keys", "date_attribute_keys")}))
    print(json.dumps({label: info["rows_per_column"] for label, info in screens.items()}))
    print(json.dumps(report["date_extraction"]))


if __name__ == "__main__":
    main()
