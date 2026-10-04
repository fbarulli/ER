"""Screen added-sugar titles with the SSOT regex on a fingerprinted population."""
import argparse
import json
from pathlib import Path

import pandas as pd

from core.audit_guard import (
    assert_not_degenerate,
    assert_vocabulary_overlap,
    self_comparison_control,
)
from core.common import DATA_PATH, data_cfg
from core.critical_attributes import NO_ADDED_SUGAR_RE, extract_critical_claims
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
        if NO_ADDED_SUGAR_RE.search(row["sku_name_eng"]):
            rows.append({"sku_id": str(row["sku_id"]), "sku_name_eng": row["sku_name_eng"],
                         "claims": sorted(extract_critical_claims(row["sku_name_eng"])["sweetener"]),
                         "negated_ingredients": sorted(negated_sweetener_types(row["sku_name_eng"]))})
    assert fingerprint == sha256_file(DATA_PATH), "source changed during audit"
    # Fail-closed guards: a screen that matched nothing (0%) or everything
    # (100%), or a vocabulary absent from the corpus, would make the counts
    # below meaningless. Only `screened_rows` is guarded for degeneracy:
    # no_sugar = 0 and no_added_sugar = 100% of the screen are the INTENDED
    # outcomes of the semantic fix, so a 0%/100% check there would be a false
    # alarm (it fired on the first run and is deliberately not applied).
    # Use the actual regex matches: a separate phrase list drifts on singular,
    # plural, word order and whitespace variants supported by the extractor.
    assert_vocabulary_overlap(
        {match.group(0) for row in rows
         if (match := NO_ADDED_SUGAR_RE.search(row["sku_name_eng"]))},
        frame["sku_name_eng"], label="added-sugar",
    )
    assert_not_degenerate("screened_rows", len(rows), total=len(frame), label="added-sugar")
    self_comparison_control(
        lambda a, b: extract_critical_claims(a)["sweetener"]
        == extract_critical_claims(b)["sweetener"],
        [row["sku_name_eng"] for row in rows], label="added-sugar",
    )
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
