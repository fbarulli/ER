#!/usr/bin/env python3
"""Build the validation set review sample (owner ruling 2026-10-01).

From the ONE validation CSV (``data/final_validation.csv``) grab TWO
positives + TWO negatives per attribute slice VALUE (shared picks: one pair
covers every (field, value) cell both its sides speak, so the total stays
well under the 200-sample budget).

Coverage = value appears on at least one side of the pair (per the
set_bag doctrine — a side IS a bag of values; side equality is explicitly
not the comparison).

Output:
  data/validation/slice_review_sample.csv   (the selected pair rows)
  results/validation/slice_review_manifest.json
    (per-cell realized counts, uncovered cells at the budget cap)

Usage:
  PYTHONPATH=src .venv/bin/python scripts/build_validation_slice_sample.py
  # override inputs / size:
  #   --validation data/final_validation.csv --budget 180
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from core.common import F, RESULTS

FIELDS = ("volume", "pack", "package_type", "sweetener", "flavor", "carbonation")
BUDGET = 300  # review-sample ceiling (raised to 300 by owner, 2026-10-01)
PER_CELL = 2  # 2 positives + 2 negatives per (field, value) cell


def parse_set(raw: object) -> list[str]:
    """Bag-of-values from the stored list literal; deduped, lowercased."""
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return []
    items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    return sorted({str(t).strip().lower() for t in items if str(t).strip()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation", type=Path, default=Path(F["final_validation"]))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--out", type=Path, default=Path("data/validation/slice_review_sample.csv"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("results/validation/slice_review_manifest.json"))
    args = ap.parse_args()

    src = args.validation
    if str(src) == F["final_validation"]:
        src = Path("data") / F["final_validation"]
    out_manifest = args.manifest

    df = pd.read_csv(src, dtype=str)
    df = df.reset_index(drop=True)

    # (field, value) -> {"pos": remaining, "neg": remaining}
    demand: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for f in FIELDS:
        for side in (f"v1_{f}", f"v2_{f}"):
            for raw in df[side].dropna():
                for v in parse_set(raw):
                    demand[(f, v)]
    for cell in demand:
        demand[cell]["pos"] = args.per_cell
        demand[cell]["neg"] = args.per_cell

    cells = sorted(demand)
    coverage: list[set[tuple[str, str]]] = []
    for _, row in df.iterrows():
        cell_set: set[tuple[str, str]] = set()
        for f in FIELDS:
            for col in (f"v1_{f}", f"v2_{f}"):
                for v in parse_set(row[col]):
                    cell_set.add((f, v))
        coverage.append(cell_set)

    support_pos = {c: sum(c in cov for cov in coverage) for c in cells}

    def run_channel(kind: str, budget: int) -> tuple[list[int], Counter]:
        """Greedy cheapest-coverage-first pick inside the shared budget."""
        picked: Counter = Counter()
        order = sorted(cells, key=lambda c: (sum(c in cov for cov in coverage), c))
        picks: list[int] = []
        used: set[int] = set()
        for cell in order:
            while picked[cell] < args.per_cell and budget > 0:
                cand = [
                    i for i, cov in enumerate(coverage)
                    if cell in cov and i not in used
                    and (kind == "pos" and int(df.true_label[i]) == 1
                         or kind == "neg" and int(df.true_label[i]) != 1)
                ]
                if not cand:
                    break
                cand.sort(key=lambda i: (-len(coverage[i]), i))
                i = cand[0]
                used.add(i)
                picks.append(i)
                budget -= 1
                for c in coverage[i]:
                    picked[c] += 1
        return picks, picked

    pos_picks, pos_covered = run_channel("pos", args.budget)
    neg_picks, neg_covered = run_channel("neg", args.budget - len(pos_picks))
    selected = sorted(set(pos_picks) | set(neg_picks))

    out_df = df.iloc[selected].copy()
    out_df["covered_slices"] = [
        json.dumps(sorted(coverage[i]), sort_keys=True) for i in selected
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.drop(columns=["covered_slices"]).assign(covered_slices=out_df["covered_slices"]) \
        .to_csv(args.out, index=False)

    uncovered = {
        f"{f}={v}": {
            "pos": max(0, demand[(f, v)]["pos"] - min(pos_covered[(f, v)], args.per_cell)),
            "neg": max(0, demand[(f, v)]["neg"] - min(neg_covered[(f, v)], args.per_cell)),
        }
        for (f, v) in cells
        if demand[(f, v)]["pos"] - pos_covered[(f, v)] > 0
        or demand[(f, v)]["neg"] - neg_covered[(f, v)] > 0
    }

    manifest = {
        "source": str(src),
        "budget": args.budget,
        "per_cell": {"pos": args.per_cell, "neg": args.per_cell},
        "cells_total": len(cells),
        "cells_fully_covered": len(cells) - len(uncovered),
        "uncovered_cells": uncovered,
        "selected": len(selected),
        "selected_pos": int((out_df.true_label == "1").sum()),
        "selected_neg": int(out_df.true_label.ne("1").sum()),
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[slice-sample] source {src}")
    print(f"[slice-sample] selected {len(selected)} (< budget {args.budget}): "
          f"{manifest['selected_pos']} pos / {manifest['selected_neg']} neg")
    print(f"[slice-sample] cells covered {manifest['cells_fully_covered']}/{len(cells)}; "
          f"uncovered listed in the manifest")
    print(f"[slice-sample] wrote {args.out} + {out_manifest}")
    assert 0 < len(selected) <= args.budget, "review sample must respect the budget"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
