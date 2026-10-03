"""Compare added-sugar title semantics on a fingerprinted source population."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from core.common import DATA_PATH, data_cfg
from core.critical_attributes import extract_critical_claims
from core.manifest import sha256_file
from core.sweetener_values import negated_sweetener_types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fingerprint = sha256_file(DATA_PATH)
    frame = pd.read_csv(DATA_PATH, **data_cfg().dataset_csv_read.model_dump()).rename(
        columns=data_cfg().column_mapping).fillna("")
    rows = []
    for row in frame.to_dict("records"):
        if re.search(r"\b(?:no|zero|0|without)\s+sugars?\s+added\b", row["sku_name_eng"], re.I):
            rows.append({"sku_id": str(row["sku_id"]), "sku_name_eng": row["sku_name_eng"],
                         "claims": sorted(extract_critical_claims(row["sku_name_eng"])["sweetener"]),
                         "negated_ingredients": sorted(negated_sweetener_types(row["sku_name_eng"]))})
    assert fingerprint == sha256_file(DATA_PATH), "source changed during audit"
    report = {"scope": "Title-only semantic extraction; not full canonical or gate replay, source contradictions, or accuracy.",
              "source_sha256": fingerprint, "source_rows": len(frame), "screened_rows": len(rows),
              "no_sugar_rows": sum("no_sugar" in row["claims"] for row in rows),
              "no_added_sugar_rows": sum("no_added_sugar" in row["claims"] for row in rows),
              "negated_sugar_rows": sum("sugar" in row["negated_ingredients"] for row in rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
