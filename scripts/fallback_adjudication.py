#!/usr/bin/env python3
"""Fallback-gate metrics + deterministic adjudication proposal.

Measures the fallback gate pairs (owner ask, 2026-10-01): who they are, their
similarity mass, and — the solvable part — which pairs a CONFIG-FREE doctrine
can adjudicate without guessing:

  positive_if_partial_declaration
      canonical token bags relate as subset (one side's evidence is a
      subset of the other's) AND the populated critical dimensions carry no
      conflict. Doctrine: the same one as the declaration-dropout lane —
      removing evidence cannot contradict identity. Low extraction
      confidence is a capture weakness, not an identity signal.
  variant_conflict
      differing non-subset variant tokens (apro_medium vs apro_classic) —
      stays fallback (a real product question, not a capture weakness).

Two defects this file carried until 2026-10-02, both pinned by the numbers it
now prints:

  * `canon1`/`canon2` are `canonical_records.canonical` — space-joined token
    strings, never Python literals. The old `ast.literal_eval` tokenizer
    therefore returned an empty set for every row, and the whole 41,481-pair
    population it measured was reported as `insufficient_evidence` with
    `resolvable_positive_by_doctrine: {}`. Tokens are whitespace splits, the
    canonical text's own tokenization.
  * "no populated critical attribute conflicts" was never evaluated: the
    doctrine's second clause was proxied by a similarity threshold. Conflicts
    come from the SSOT device the veto lane already uses —
    `core.critical_attribute_evaluation` over
    `core.canonical_attribute_info` — never a second conflict parser.

Outputs results/gate_fallback_metrics.json. NO labels are flipped: the
proposal is written for owner adjudication only.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

GATE = Path("data/gate_results.csv")
CANONICAL = Path("data/canonical_records.csv")
OUT = Path("results/gate_fallback_metrics.json")
EXAMPLES_PER_REASON = 3


def canon_tokens(raw: str) -> set[str]:
    """The canonical text's own tokens: whitespace-separated, lowercased.

    `canon1`/`canon2` are the `canonical` column of canonical_records.csv,
    which is a space-joined token string by construction (pipeline writes it
    that way), so a split IS the tokenization. An empty cell yields an empty
    set, which classifies as insufficient evidence rather than as agreement.
    """
    return {token for token in str(raw or "").split() if token}


def main() -> int:
    from core.attribute_conflicts import (
        canonical_attribute_info,
        critical_attribute_evaluation,
    )
    from core.common import F, canonical_records_frame

    gate_path = F["gate_results"] if F.get("gate_results") else GATE
    if not Path(gate_path).exists():
        print(f"[fallback] no gate artifact at {gate_path}")
        return 1
    g = pd.read_csv(gate_path, dtype=str, keep_default_na=False)
    fb = g[g.gate_decision == "fallback"].copy()
    fb["sim"] = fb.similarity.astype(float)

    records = (
        canonical_records_frame()
        if CANONICAL.exists()
        else pd.read_csv(CANONICAL, dtype=str, keep_default_na=False)
    )
    by_gtin = {str(row["gtin"]): row.to_dict() for _, row in records.iterrows()}
    info_cache: dict[str, dict] = {}

    def info(gtin: str) -> dict:
        if gtin not in info_cache:
            if gtin not in by_gtin:
                return {}
            info_cache[gtin] = canonical_attribute_info(by_gtin[gtin])
        return info_cache[gtin]

    def conflicts(gtin1: str, gtin2: str) -> list[str]:
        left, right = info(gtin1), info(gtin2)
        if not left or not right:
            return []
        return list(
            critical_attribute_evaluation(
                left, right, volume_relative_tolerance=0.0, volume_absolute_tolerance_ml=0.0
            )["conflicts"]
        )

    def classify(t1: set[str], t2: set[str], sim: float, conflict: list[str]) -> str:
        if not t1 or not t2:
            return "insufficient_evidence"
        if t1 == t2:
            return (
                "positive_if_identical_canon"
                if not conflict
                else "identical_canon_but_conflict"
            )
        if t1 <= t2 or t2 <= t1:
            return (
                "positive_if_partial_declaration"
                if not conflict
                else "partial_declaration_but_conflict"
            )
        inter = t1 & t2
        if inter and len(t1 - t2) <= 2 and len(t2 - t1) <= 2 and sim >= 0.7:
            return "variant_conflict"
        return "distinct_stays_fallback"

    fb["conflict"] = [
        conflicts(a, b) for a, b in zip(fb.gtin1, fb.gtin2)
    ]
    fb["proposal"] = [
        classify(canon_tokens(c1), canon_tokens(c2), sim, cf)
        for c1, c2, sim, cf in zip(fb.canon1, fb.canon2, fb.sim, fb.conflict)
    ]

    report = {
        "source": str(gate_path),
        "canonical_source": str(CANONICAL),
        "total_fallback_pairs": len(fb),
        "gate_reason_counts": fb.gate_reason.value_counts().to_dict(),
        "sim_bands_x_reason": {
            reason: dict(Counter(str(b) for b in pd.cut(grp.sim, [0, 0.3, 0.5, 0.7, 0.9, 1.01])))
            for reason, grp in fb.groupby("gate_reason")
        },
        "proposals": fb.proposal.value_counts().to_dict(),
        "resolvable_positive_by_doctrine": {
            name: int(len(grp))
            for name, grp in fb.groupby("proposal")
            if name.startswith("positive_if")
        },
        "critical_conflict_census": dict(
            Counter(
                token
                for tokens in fb.conflict
                for token in tokens
            )
        ),
        "high_priority_review_pool": {
            "sim_ge_0.7": int((fb.sim >= 0.7).sum()),
            "distinct_gtins": int(
                len(set(fb.loc[fb.sim >= 0.7, "gtin1"]) | set(fb.loc[fb.sim >= 0.7, "gtin2"]))
            ),
        },
        "top_examples": [
            {
                "gtin1": r.gtin1,
                "gtin2": r.gtin2,
                "canon1": r.canon1,
                "canon2": r.canon2,
                "sim": round(r.sim, 3),
                "reason": r.gate_reason,
                "proposal": r.proposal,
                "critical_conflicts": ",".join(r.conflict),
            }
            for _, grp in fb.sort_values("sim", ascending=False).groupby("gate_reason")
            for r in grp.head(EXAMPLES_PER_REASON).itertuples()
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"fallback metrics: {len(fb):,} pairs -> {OUT}")
    print("reasons:", report["gate_reason_counts"])
    print("proposals:", report["proposals"])
    print("solvable-positive:", report["resolvable_positive_by_doctrine"])
    print("critical conflicts:", report["critical_conflict_census"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())