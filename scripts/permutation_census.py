"""Permutation census — which attribute combos exist vs don't.

Gate fields (the discriminative ones the model must separate):
  volume_set x pack_set x package_type_set x flavor_set
  universe = 249 x 45 x 17 x 35 = 6,666,975 cells
  observed = 5,884  (0.088%)

Findings so far (from 13,250 canonicals):
  - grid is dominated by 'plain water' (no pack/type/flavor) at various volumes
  - flavor only matters when pack/type is present
  - sweetener/carbonation/pulp/material are low-cardinality but jointly rare
  - many pairwise never-co-occurring combos are structurally impossible
    (e.g. pack='12' + type='can' — a 12-pack of cans is a box, not a can)

This script extends the census with:
  1. pack<->package_type structural rules inferred from the data
  2. per-cell support + reachability flag (structurally possible vs impossible)
  3. the sparse cell list that is the augmentation target
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product

import pandas as pd

from core.common import F, RESULTS

FIELDS = (
    "volume_set", "pack_set", "package_type_set", "flavor_set",
    "carbonation_set", "sweetener_set", "pulp_set", "package_material_set",
)
GATE = ("volume_set", "pack_set", "package_type_set", "flavor_set")

# Inferred structural rules: a pack_count value can only co-occur with
# package_types whose physical form matches that count.
# Rule sources: observed co-occurrence + common-sense packaging logic.
PACK_TYPE_RULES: dict[str, set[str]] = {
    # single units -> any type
    "1": {"<none>", "bottle", "can", "box", "bag", "packet", "tin", "pot"},
    "2": {"<none>", "bottle", "can", "box", "bag", "packet", "tin"},
    "3": {"<none>", "bottle", "can", "box", "bag"},
    "4": {"<none>", "bottle", "can", "box", "bag", "packet"},
    "6": {"<none>", "bottle", "can", "box", "bag"},
    "8": {"<none>", "box", "bag"},
    "12": {"<none>", "box", "bag"},
    "24": {"<none>", "box", "bag"},
}


def parse_set(raw: object) -> set[str]:
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return set()
    return {str(t).strip().lower() for t in parsed if str(t).strip()}


def load_canonicals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [f for f in FIELDS if f not in frame.columns]
    if missing:
        raise SystemExit(f"missing fields: {missing}")
    out = pd.DataFrame({"gtin": frame["gtin"]})
    for field in FIELDS:
        out[field] = [parse_set(v) for v in frame[field]]
    return out


def infer_rules(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Learn pack<->type rules from observed co-occurrence, then tighten."""
    observed: Counter = Counter()
    for row in frame.itertuples(index=False):
        packs = parse_set(row.pack_set) or {"<none>"}
        types = parse_set(row.package_type_set) or {"<none>"}
        for p, t in product(packs, types):
            observed[(p, t)] += 1
    rules: dict[str, set[str]] = defaultdict(set)
    for (p, t), n in observed.items():
        if n >= 3:  # minimum support to count as a rule
            rules[p].add(t)
    # merge learned rules with the hand-coded ones
    merged = {k: set(v) for k, v in PACK_TYPE_RULES.items()}
    for p, types in rules.items():
        merged.setdefault(p, set()).update(types)
    return merged, observed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonicals", type=Path, default=RESULTS / F["canonical_records"])
    ap.add_argument("--out", type=Path, default=RESULTS / "permutation_census.json")
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--singletons", type=int, default=25)
    args = ap.parse_args()

    frame = load_canonicals(args.canonicals)
    marg: dict[str, Counter] = {}
    for f in FIELDS:
        cnt: Counter = Counter()
        for v in frame[f]:
            vals = parse_set(v)
            cnt.update(vals if vals else {"<none>"})
        marg[f] = cnt

    rules, observed_pairs = infer_rules(frame)

    # joint gate-field distribution
    gate_joint: Counter = Counter()
    for row in frame.itertuples(index=False):
        parts = tuple(
            tuple(sorted(parse_set(getattr(row, f))))
            if parse_set(getattr(row, f))
            else ("<none>",)
            for f in GATE
        )
        gate_joint[parts] += 1

    # per-cell reachability: structurally possible?
    structurally_impossible = 0
    structurally_possible_absent = 0
    reachable_cells: list[dict] = []
    thin: list[dict] = []
    for combo in product(*(list(marg[f].keys()) for f in GATE)):
        # gate_joint keys are tuples-of-tuples, e.g. (('500.0',), ('<none>',), ...)
        combo_t = tuple((v,) if isinstance(v, str) else v for v in combo)
        vs, pk, pt, fl = combo
        possible = True
        reasons: list[str] = []
        for p_val, t_val in product(
            (vs,) if vs != ("<none>",) else ("<none>",),
            (pk,) if pk != ("<none>",) else ("<none>",),
        ):
            if p_val in rules and t_val not in rules[p_val]:
                possible = False
                reasons.append(f"pack={p_val} incompatible with type={t_val}")
        for p_val, t_val in product(
            (pk,) if pk != ("<none>",) else ("<none>",),
            (pt,) if pt != ("<none>",) else ("<none>",),
        ):
            if p_val in rules and t_val not in rules[p_val]:
                possible = False
                reasons.append(f"pack={p_val} incompatible with type={t_val}")
        type_mat: dict[str, set[str]] = {
            "bottle": {"pet", "glass", "<none>"},
            "can": {"aluminum", "<none>"},
            "box": {"<none>"},
            "bag": {"plastic", "<none>"},
            "packet": {"<none>"},
            "tin": {"aluminum", "<none>"},
            "pot": {"plastic", "pet", "<none>"},
        }
        for t_val, m_val in product(
            (pt,) if pt != ("<none>",) else ("<none>",),
            (combo[7] if len(combo) > 7 else ("<none>",)),
        ):
            if t_val in type_mat and m_val not in type_mat[t_val]:
                possible = False
                reasons.append(f"type={t_val} incompatible with material={m_val}")

        cell = dict(zip(GATE, combo))
        support = gate_joint.get(combo_t, 0)
        if support == 0:
            if possible:
                structurally_possible_absent += 1
                if len(reasons) == 0 or not any("incompatible" in r for r in reasons):
                    reachable_cells.append(
                        {"cell": cell, "support": 0, "reasons": reasons}
                    )
            else:
                structurally_impossible += 1
        if support < args.singletons and support > 0:
            thin.append({"cell": cell, "support": support, "reasons": []})
    # the real target list: reachable but absent cells, sorted by how
    # 'close' they are to observed cells (sum of marginal frequencies).
    for entry in reachable_cells:
        cell = entry["cell"]
        entry["marginal_score"] = sum(
            marg[f].get(v, 0) for f, v in cell.items()
        )
    reachable_cells.sort(key=lambda d: -d["marginal_score"])

    # pairwise never-co-occurring (same as before, but with structural flags)
    pair_stats: dict[str, dict] = {}
    for a, b in product(GATE, repeat=2):
        if a >= b:
            continue
        joint: Counter = Counter()
        for x, y in zip(frame[a], frame[b]):
            for va in (parse_set(x) or {"<none>"}):
                for vb in (parse_set(y) or {"<none>"}):
                    joint[(va, vb)] += 1
        naive = marg[a] + marg[b]
        unseen = []
        for va in marg[a]:
            for vb in marg[b]:
                if joint[(va, vb)] == 0:
                    exp = (naive[va] / len(frame)) * (naive[vb] / len(frame)) * len(frame)
                    unseen.append(
                        {
                            a: va,
                            b: vb,
                            "expected": round(exp, 2),
                            "a_rate": round(naive[va] / len(frame), 4),
                            "b_rate": round(naive[vb] / len(frame), 4),
                        }
                    )
        pair_stats[f"{a}|{b}"] = {
            "cells": len(joint),
            "unseen_pairs": len(unseen),
            "unseen_fraction": round(len(unseen) / max(1, len(marg[a]) * len(marg[b])), 4),
            "unseen_examples": sorted(unseen, key=lambda d: -d["expected"])[:5],
        }

    report = {
        "canonicals": str(args.canonicals),
        "n_canonicals": len(frame),
        "gate_fields": list(GATE),
        "gate_universe": 1,
        "gate_observed": len(gate_joint),
        "marginals": {f: dict(c.most_common()) for f, c in marg.items()},
        "gate_joint_observed": len(gate_joint),
        "gate_joint_top20": gate_joint.most_common(20),
        "pair_stats": pair_stats,
        "structural": {
            "impossible_cells": structurally_impossible,
            "possible_absent_cells": structurally_possible_absent,
            "pack_type_rules_learned": {k: sorted(v) for k, v in rules.items()},
            "observed_pack_type_pairs": {f"{p}|{t}": n for (p, t), n in observed_pairs.most_common(30)},
        },
        "thin_cells": thin[:500],
        "thin_count": len(thin),
        "reachable_absent_cells": reachable_cells[:500],
        "reachable_absent_count": len(reachable_cells),
        "singleton_threshold": args.singletons,
    }
    report["gate_universe"] = 1
    for f in GATE:
        report["gate_universe"] *= len(marg[f])
    report["gate_coverage"] = round(
        len(gate_joint) / report["gate_universe"], 6
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[census] {len(frame):,} canonicals")
    print(f"[census] gate universe {report['gate_universe']:,} ; observed {len(gate_joint)} ; coverage {report['gate_coverage']:.4%}")
    print(f"[census] thin cells (support<{args.singletons}): {len(thin)}")
    print(f"[census] structurally impossible: {structurally_impossible}")
    print(f"[census] possible but absent (augmentation targets): {structurally_possible_absent}")
    print(f"[census] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())