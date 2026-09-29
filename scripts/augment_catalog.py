#!/usr/bin/env python3
"""Augment catalog with synthetic products for weak/blind spots."""
import ast
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from training.folds import normalize_gtin as n

GATE = ["volume_set", "pack_set", "package_type_set", "flavor_set"]


def parse(v):
    try:
        x = ast.literal_eval(str(v))
    except Exception:
        return set()
    return {str(t).strip().lower() for t in x if str(t).strip()}


def cell_key(row):
    return tuple(
        tuple(sorted(parse(getattr(row, f)))) if parse(getattr(row, f)) else ("<none>",)
        for f in GATE
    )


def load_config():
    with open("augment.yaml") as f:
        return yaml.safe_load(f)


def _parse_field(val):
    try:
        return ast.literal_eval(val) if val and val != "[]" else []
    except Exception:
        return []


def _build_canonical_from_gate(row: dict) -> str:
    parts = []
    v = _parse_field(row.get("volume_set", ""))
    if v:
        parts.append(v[0].replace(".0", "") + "ml")
    f = _parse_field(row.get("flavor_set", ""))
    if f:
        parts.extend(f)
    c = _parse_field(row.get("carbonation_set", ""))
    if c:
        parts.extend(c)
    s = _parse_field(row.get("sweetener_set", ""))
    if s:
        parts.extend(s)
    p = _parse_field(row.get("pulp_set", ""))
    if p:
        parts.extend(p)
    pk = _parse_field(row.get("pack_set", ""))
    if pk:
        parts.append(f"pack{pk[0]}")
    pt = _parse_field(row.get("package_type_set", ""))
    if pt:
        parts.append(pt[0])
    pm = _parse_field(row.get("package_material_set", ""))
    if pm:
        parts.append(pm[0])
    return " ".join(parts) if parts else "synthetic product"


def main():
    cfg = load_config()
    random.seed(cfg["seed"])
    target = cfg["target_support"]

    can = pd.read_csv(
        cfg["canonicals_path"],
        dtype=str, keep_default_na=False,
    ).drop_duplicates("gtin")
    can["norm_gtin"] = can.gtin.map(n)

    r = json.load(open("results/permutation_census.json"))
    marg = {f: {vv for vv, cnt in r["marginals"][f].items() if cnt > 50} for f in GATE}
    marg_none = {f for f in GATE if "<none>" in marg[f]}

    def n_common(cell_dict):
        return sum(
            1 for f in GATE
            if cell_dict.get(f, "<none>") in marg[f]
            or (cell_dict.get(f, "<none>") == "<none>" and f in marg_none)
        )

    observed = Counter(cell_key(r) for r in can.itertuples(index=False))

    weak = {}
    for c, s in observed.items():
        if 1 <= s <= 5:
            c_norm = tuple(v[0] if isinstance(v, tuple) and v else "<none>" for v in c)
            if n_common(dict(zip(GATE, c_norm))) >= 4:
                weak[c_norm] = s

    blind = {}
    for c in r["reachable_absent_cells"]:
        cell_t = tuple((c["cell"][f],) for f in GATE)
        if n_common(dict(zip(GATE, cell_t))) >= 4:
            blind[cell_t] = 0

    partial = {}
    for c in r["reachable_absent_cells"]:
        cell_t = tuple((c["cell"][f],) for f in GATE)
        if 3 <= n_common(dict(zip(GATE, cell_t))) < 4:
            partial[cell_t] = 0

    print(f"[augment] blind spots: {len(blind)} cells (0 support)")
    print(f"[augment] partial blind: {len(partial)} cells (0 support)")
    print(f"[augment] weak spots: {len(weak)} cells with support 1-5")

    all_cells = {**blind, **partial, **weak}
    needed = sum(max(0, target - s) for s in all_cells.values())
    print(f"[augment] target support: {target}")
    print(f"[augment] products needed: {needed}")

    max_real = max((int(g.replace("0", "")) for g in can.norm_gtin if g.isdigit()), default=0)
    synthetic = []
    for c, s in all_cells.items():
        need = target - s
        for _ in range(need):
            max_real += 1
            syn_gtin = f"SYN{max_real:010d}"
            # c is tuple of strings (one per GATE field)
            attrs = dict(zip(GATE, c))
            row = {"gtin": syn_gtin, "norm_gtin": syn_gtin}
            for f in GATE:
                v = attrs[f]
                if v == "<none>":
                    row[f] = "[]"
                else:
                    # v is a single string value - store as proper list literal
                    row[f] = "['" + str(v) + "']"
            row["canonical"] = _build_canonical_from_gate(row)
            for col in can.columns:
                if col not in row:
                    row[col] = ""
            synthetic.append(row)

    if synthetic:
        syn_df = pd.DataFrame(synthetic)
        for f in can.columns:
            if f not in syn_df.columns:
                syn_df[f] = ""
        can = pd.concat([can, syn_df], ignore_index=True)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    can.to_csv(out_dir / "canonical_records_augmented.csv", index=False)

    audit = {
        "target_support": target,
        "seed": cfg["seed"],
        "total_synthetic": len(synthetic),
        "blind_spots": len(blind),
        "partial_blind": len(partial),
        "weak_spots": len(weak),
        "cells_augmented": len(all_cells),
        "synthetic_gtins": [f"SYN{i:010d}" for i in range(max_real - len(synthetic) + 1, max_real + 1)],
    }
    (out_dir / "augmentation_audit.json").write_text(json.dumps(audit, indent=2))

    print(f"[augment] added {len(synthetic)} synthetic products")
    print(f"[augment] new catalog size: {len(can)} (was {len(can) - len(synthetic)})")
    print(f"[augment] audit written to {out_dir / 'augmentation_audit.json'}")


if __name__ == "__main__":
    main()