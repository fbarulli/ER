"""Compare legacy gate extraction with the shared title-attribute sidecar.

This is a shadow audit only.  It changes no canonical, gate, label, or model
input; every disagreement is written for review before either extractor is
allowed to affect a matching decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import extract_all
from core.common import F, RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=RESULTS / F["title_attribute_evidence"])
    args = parser.parse_args()
    if not args.evidence.is_file():
        raise FileNotFoundError(f"evidence missing: {args.evidence}; run build_title_attribute_evidence first")
    evidence = pd.read_csv(args.evidence, dtype=str, keep_default_na=False)
    required = {"product_id", "title", "attribute_raw", "volume_ml", "pack_count"}
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"evidence missing columns {sorted(missing)}")
    rows = []
    for row in evidence.itertuples(index=False):
        data = row._asdict()
        attribute = json.loads(data["attribute_raw"])
        raw_attribute = "; ".join(f"{key}: {value}" for key, value in attribute.items())
        legacy = extract_all(data["title"], raw_attribute)
        ner_volumes = json.loads(data["volume_ml"])
        ner_packs = json.loads(data["pack_count"])
        legacy_volume = legacy["volume_ml"] if legacy["volume_ml"] > 0 else None
        volume_agrees = legacy_volume in ner_volumes if legacy_volume is not None and ner_volumes else None
        pack_agrees = legacy["pack_qty"] in ner_packs if ner_packs else None
        if volume_agrees is False or pack_agrees is False:
            rows.append({
                "product_id": data["product_id"], "title": data["title"],
                "legacy_volume_ml": legacy_volume, "ner_volume_ml": json.dumps(ner_volumes),
                "legacy_pack_count": legacy["pack_qty"], "ner_pack_count": json.dumps(ner_packs),
                "volume_agrees": volume_agrees, "pack_agrees": pack_agrees,
                "legacy_volume_status": legacy["volume_status"],
            })
    conflicts = pd.DataFrame(rows)
    total = len(evidence)
    summary = pd.DataFrame([
        {"metric": "evidence_rows", "value": total, "detail": "all sidecar rows compared"},
        {"metric": "conflict_rows", "value": len(conflicts), "detail": "legacy value absent from NER title evidence"},
        {"metric": "volume_conflict_rows", "value": int(conflicts["volume_agrees"].eq(False).sum()) if not conflicts.empty else 0, "detail": "inspect before changing volume gate"},
        {"metric": "pack_conflict_rows", "value": int(conflicts["pack_agrees"].eq(False).sum()) if not conflicts.empty else 0, "detail": "inspect before changing pack gate"},
    ])
    for filename, frame in ((F["attribute_agreement_summary"], summary), (F["attribute_agreement_conflicts"], conflicts)):
        path = RESULTS / filename
        frame.to_csv(path, index=False)
        print(f"[agreement] wrote {path} ({len(frame):,} rows)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
