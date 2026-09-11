"""Shadow-test package type/material constraints against current gate pairs.

No live decision is changed.  Every proposed hard rejection records both
canonical evidence sets and the prior gate reason, making parser-driven drift
reviewable before promotion into ``three_way_gate``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.common import F, RESULTS
from core.gtin import normalize_and_validate_gtin


def _sets(frame: pd.DataFrame, column: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for gtin, group in frame.groupby("gtin_clean", sort=True):
        values: set[str] = set()
        for encoded in group[column]:
            values.update(str(value) for value in json.loads(encoded) if str(value).strip())
        out[str(gtin)] = values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=RESULTS / F["title_attribute_evidence"])
    parser.add_argument("--gates", type=Path, default=RESULTS / F["gate_results"])
    args = parser.parse_args()
    if not args.evidence.is_file() or not args.gates.is_file():
        raise FileNotFoundError("title evidence and gate results must exist before impact auditing")
    evidence = pd.read_csv(args.evidence, dtype=str, keep_default_na=False)
    gate = pd.read_csv(args.gates, dtype={"gtin1": str, "gtin2": str}, keep_default_na=False)
    parsed = normalize_and_validate_gtin(evidence["gtin_raw"])
    evidence = evidence.loc[parsed["gtin_structurally_valid"]].copy()
    # Gate results retain the raw GTIN string as their grouping key.  The
    # normalized form validates trust only; using it here would silently make
    # UPC-12 evidence fail to join a raw-key gate pair.
    evidence["gtin_clean"] = evidence["gtin_raw"].astype(str).str.strip()
    types, materials = _sets(evidence, "package_types"), _sets(evidence, "package_materials")
    changes = []
    for row in gate.itertuples(index=False):
        t1, t2 = types.get(str(row.gtin1), set()), types.get(str(row.gtin2), set())
        m1, m2 = materials.get(str(row.gtin1), set()), materials.get(str(row.gtin2), set())
        reasons = []
        if t1 and t2 and not (t1 & t2): reasons.append("package_type_mismatch")
        if m1 and m2 and not (m1 & m2): reasons.append("package_material_mismatch")
        if reasons:
            changes.append({"gtin1": row.gtin1, "gtin2": row.gtin2, "old_decision": row.gate_decision, "old_reason": row.gate_reason, "proposed_decision": "hard_no", "proposed_reason": " | ".join(reasons), "package_types_1": json.dumps(sorted(t1)), "package_types_2": json.dumps(sorted(t2)), "package_materials_1": json.dumps(sorted(m1)), "package_materials_2": json.dumps(sorted(m2))})
    impact = pd.DataFrame(changes)
    changed_proceed = int(impact["old_decision"].eq("proceed").sum()) if not impact.empty else 0
    summary = pd.DataFrame([{"metric": "gate_pairs", "value": len(gate), "detail": "current decision universe"}, {"metric": "pairs_with_package_evidence_mismatch", "value": len(impact), "detail": "live package constraints applied"}, {"metric": "proceed_to_hard_no", "value": changed_proceed, "detail": "must remain zero after live-gate promotion"}])
    for name, frame in ((F["package_gate_impact_summary"], summary), (F["package_gate_impact_pairs"], impact)):
        frame.to_csv(RESULTS / name, index=False)
        print(f"[package-impact] wrote {RESULTS / name} ({len(frame):,} rows)")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
