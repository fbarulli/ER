"""Build the stratified validation CSV.

Design (permutation census):
  - 150 entities covering all strata values with support >5
  - 17 blind-spot augmentations (cells with all components common, zero support)
  - Combined cost: ~1% of training positives
  - Coverage: all strata values + all 17 blind spots

Output: data/validation/stratified_holdout.csv with columns:
  gtin1, gtin2, true_label, role, fold_id,
  slice_volume, slice_pack, slice_sweetener, slice_flavor,
  slice_package_type, slice_carbonation
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.common import F, RESULTS
from training.folds import normalize_gtin as n

FIELDS = (
    "volume_set", "pack_set", "package_type_set", "flavor_set",
    "carbonation_set", "sweetener_set", "pulp_set", "package_material_set",
)
GATE = ("volume_set", "pack_set", "package_type_set", "flavor_set")
SLICE_COLS = [
    "slice_volume", "slice_pack", "slice_sweetener",
    "slice_flavor", "slice_package_type", "slice_carbonation",
]
ROLE = "external_eval"


def parse_set(v: object) -> set[str]:
    try:
        parsed = ast.literal_eval(str(v))
    except (ValueError, SyntaxError):
        return set()
    return {str(t).strip().lower() for t in parsed if str(t).strip()}


def cell_for(row: pd.Series) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(sorted(parse(getattr(row, f)))) if parse(getattr(row, f)) else ("<none>",)
        for f in GATE
    )


def slice_flags(row: pd.Series) -> dict[str, str]:
    out: dict[str, str] = {}
    for f, col in zip(GATE + ("carbonation_set",), SLICE_COLS):
        vals = parse(getattr(row, f))
        out[col] = "|".join(sorted(vals)) if vals else "<none>"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonicals", type=Path, default=RESULTS / F["canonical_records"])
    ap.add_argument("--out", type=Path, default=Path("data/validation/stratified_holdout.csv"))
    # Deliberate per-experiment holdout-construction seed (NOT core.common.SED):
    # it fixes the stratified validation CSV independent of the training seed.
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--support_threshold", type=int, default=5)
    ap.add_argument("--blind_spot_threshold", type=int, default=50)
    args = ap.parse_args()

    random.seed(args.seed)
    can = pd.read_csv(args.canonicals, dtype=str, keep_default_na=False).drop_duplicates("gtin")
    can["norm_gtin"] = can.gtin.map(n)

    # marginals
    marg: dict[str, dict[str, int]] = {}
    for f in GATE:
        cnt: dict[str, int] = defaultdict(int)
        for v in can[f]:
            for vv in parse(v):
                cnt[vv] += 1
        marg[f] = dict(cnt)

    # ── 1. strata coverage: 1 entity per stratum value with support > threshold ──
    print("[holdout] building 150-entity design")
    ents: list[str] = []
    ent_pairs: dict[str, int] = defaultdict(int)
    for f in GATE:
        vals = {vv for vv, c in marg[f].items() if c >= args.support_threshold}
        for v in vals:
            matches = can[can[f].apply(lambda x: v in parse(x))]
            if matches.empty:
                continue
            row = matches.iloc[0]
            ent = row.norm_gtin
            if ent in ents:
                continue
            ents.append(ent)

    # ── 2. blind-spot augmentations ──
    print("[holdout] building blind-spot augmentations")
    r = json.loads(RESULTS.joinpath("permutation_census.json").read_text())
    blind = [
        c
        for c in r["reachable_absent_cells"]
        if all(
            c["cell"][f] in {vv for vv, cnt in marg[f].items() if cnt >= args.blind_spot_threshold}
            for f in GATE
        )
    ]
    # For each blind spot, synthesize a GTIN placeholder (not in catalog).
    # The GTIN is a synthetic key that won't collide with real gtins.
    synthetic_offset = max((int(g.replace("0", "")) for g in can.norm_gtin if g.isdigit()), default=0) + 1
    blind_ents: list[str] = []
    for c in blind:
        synthetic_gtin = f"SYN{synthetic_offset:010d}"
        synthetic_offset += 1
        blind_ents.append(synthetic_gtin)

    all_ents = ents + blind_ents
    print(f"[holdout] total entities: {len(all_ents)} (real: {len(ents)}, synthetic: {len(blind_ents)})")

    # ── 3. build pairs ──
    # Positive pairs: same-product pairs (products with multiple GTINs).
    # Negative pairs: cross-product pairs with different attributes.
    print("[holdout] building pairs")
    # Build a per-entity attribute profile
    ent_profile: dict[str, tuple] = {}
    for _, row in can.iterrows():
        ent = row.norm_gtin
        if ent not in ents:
            continue
        ent_profile[ent] = (cell_for(row), slice_flags(row))
    for ent in blind_ents:
        # Blind-spot entities have the blind-spot cell attributes
        # Find the blind spot cell for this entity
        idx = blind_ents.index(ent)
        c = blind[idx]
        cell = tuple(tuple([v]) if v != "<none>" else ("<none>",) for v in [c["cell"][f][0] if c["cell"][f] != ("<none>",) else "<none>" for f in GATE])
        # Build slice flags from the blind-spot cell
        slices: dict[str, str] = {}
        for i, f in enumerate(GATE):
            vals = c["cell"][f]
            slices[SLICE_COLS[i]] = "|".join(sorted(vals)) if vals != ("<none>",) else "<none>"
        # Carbonation is not in GATE; default to <none>
        slices["slice_carbonation"] = "<none>"
        ent_profile[ent] = (cell, slices)

    # Positive pairs: for each entity, pair with other entities sharing the same product.
    # Since blind-spot entities are synthetic, they have no same-product pairs.
    # Real entities with multiple GTINs get positive pairs.
    rows: list[dict] = []
    for i, e1 in enumerate(all_ents):
        for e2 in all_ents[i + 1 :]:
            if e1 not in ent_profile or e2 not in ent_profile:
                continue
            c1, s1 = ent_profile[e1]
            c2, s2 = ent_profile[e2]
            # Positive pair: same cell (same product attributes)
            if c1 == c2:
                rows.append(
                    {
                        "gtin1": e1,
                        "gtin2": e2,
                        "true_label": 1,
                        "role": ROLE,
                        "fold_id": -1,
                        **s1,
                    }
                )
            # Negative pair: different cells
            else:
                rows.append(
                    {
                        "gtin1": e1,
                        "gtin2": e2,
                        "true_label": 0,
                        "role": ROLE,
                        "fold_id": -1,
                        **{col: s1[col] for col in SLICE_COLS},
                    }
                )

    # ── 4. assign fold_id ──
    # Assign fold_id based on entity index modulo 4, ensuring each fold has
    # entities from all strata. This is a preliminary assignment; the final
    # fold assignment should be done by derive_holdout after the merged graph
    # is built. For now, we assign evenly.
    for i, row in enumerate(rows):
        row["fold_id"] = i % 4

    # ── 5. write CSV ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["gtin1", "gtin2", "true_label", "role", "fold_id"] + SLICE_COLS
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ── 6. verify ──
    print(f"[holdout] wrote {args.out}")
    df = pd.read_csv(args.out, dtype=str)
    pos = df.true_label == "1"
    neg = df.true_label == "0"
    print(f"[holdout] rows: {len(df)}")
    print(f"[holdout] positives: {pos.sum()}, negatives: {neg.sum()}")
    print(f"[holdout] folds: {sorted(df.fold_id.unique())}")
    print(f"[holdout] synhetics: {sum(1 for e in all_ents if e.startswith('SYN'))}")
    print(f"[holdout] blind spots covered: {len(blind_ents)}/{len(blind)}")

    # Verify no straddles: for each fold, check that no positive pair has
    # endpoints in different folds.
    straddles = 0
    for _, row in df[pos].iterrows():
        if row.fold_id != row.fold_id:  # always false, placeholder
            pass
    # Real check: positive pairs should have both endpoints in the same fold.
    # Since we assigned fold_id per row (not per entity), we need to check
    # per-entity fold assignment.
    ent_fold: dict[str, int] = {}
    for _, row in df.iterrows():
        ent_fold[row.gtin1] = row.fold_id
        ent_fold[row.gtin2] = row.fold_id
    straddles = sum(
        1 for _, row in df[pos].iterrows()
        if ent_fold[row.gtin1] != ent_fold[row.gtin2]
    )
    print(f"[holdout] straddles: {straddles}")

    # Verify all strata covered
    for i, f in enumerate(GATE):
        vals = {row[SLICE_COLS[i]] for _, row in df.iterrows()}
        print(f"[holdout] {f}: {len(vals)} values covered")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
