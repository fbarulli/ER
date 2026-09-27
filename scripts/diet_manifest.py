#!/usr/bin/env python3
"""Diet gate for prepared training bundles (config/training.yaml masking:).

Every bundle is gated BEFORE training on the augmentation diet contract:

  neg_aug_views / neg_presentations >= diet_min_neg_aug_frac
  pos_views / neg_views             <= diet_max_pos_neg_view_ratio

Effective views per class = base pairs + augmented copies (the bundle's pair
arrays already contain the copies; the mask audits name them, split by
population / target_mode). Rows without a target_mode predate the swap lane
and count as "random". Labels are never inspected: masked/swapped copies
keep their anchor's label by construction (positives stay 1, negatives 0).

Usage: diet_manifest.py BUNDLE_PATH
Exit 0 PASS, exit 2 FAIL naming the exact violated threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.common import load_config
from training.prepared_bundle import load_prepared_bundle


def _mode(row: dict) -> str:
    """Audit target_mode with the pre-swap default ('random')."""
    return str(row.get("target_mode") or "random")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} BUNDLE_PATH", file=sys.stderr)
        return 2
    masking = load_config()["masking"]
    diet_min_neg_aug_frac = float(masking["diet_min_neg_aug_frac"])
    diet_max_pos_neg_view_ratio = float(masking["diet_max_pos_neg_view_ratio"])

    manifest, data = load_prepared_bundle(Path(argv[1]))
    pos = data["pos"]
    neg = data["neg"]
    train_neg = data["train_neg"]
    mask_audit = list(data.get("mask_audit", []))
    neg_audit = list(data.get("hard_negative_mask_audit", []))

    pos_views = int(len(pos))
    neg_views = int(len(train_neg))
    neg_aug_views = int(len(neg_audit))
    neg_presentations = int(len(train_neg))
    pos_base = pos_views - len(mask_audit)
    neg_base = int(len(neg)) - len(neg_audit)
    if pos_base < 0 or neg_base < 0:
        print(
            "DIET FAIL: audit rows exceed pair rows "
            f"(pos {pos_views}/{len(mask_audit)}, neg {len(neg)}/{len(neg_audit)})",
            flush=True,
        )
        return 2

    def _count(rows: list[dict], population: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            if str(row.get("population", population)) != population:
                continue
            counts[_mode(row)] = counts.get(_mode(row), 0) + 1
        return counts

    pos_modes = _count(mask_audit, "positive")
    neg_modes = _count(neg_audit, "hard_negative")

    print(f"[diet] bundle={argv[1]} profile={manifest.masking_profile}", flush=True)
    print("population     target_mode   views", flush=True)
    print(f"positive       base          {pos_base:,}", flush=True)
    for mode in sorted(pos_modes):
        print(f"positive       {mode:<11} {pos_modes[mode]:,}", flush=True)
    print(f"hard_negative  base          {neg_base:,}", flush=True)
    for mode in sorted(neg_modes):
        print(f"hard_negative  {mode:<11} {neg_modes[mode]:,}", flush=True)
    print(
        f"presentations  train_neg     {neg_presentations:,} "
        f"(pos_views={pos_views:,}, neg_views={neg_views:,})",
        flush=True,
    )

    failures: list[str] = []
    neg_aug_frac = (
        neg_aug_views / neg_presentations if neg_presentations else float("nan")
    )
    pos_neg_ratio = pos_views / neg_views if neg_views else float("nan")
    neg_ok = neg_presentations > 0 and neg_aug_frac >= diet_min_neg_aug_frac
    ratio_ok = neg_views > 0 and pos_neg_ratio <= diet_max_pos_neg_view_ratio
    print(
        f"[diet] neg_aug_views={neg_aug_views:,} / neg_presentations={neg_presentations:,} "
        f"= {neg_aug_frac:.4f} >= diet_min_neg_aug_frac={diet_min_neg_aug_frac:.4f} "
        f"{'OK' if neg_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"[diet] pos_views={pos_views:,} / neg_views={neg_views:,} "
        f"= {pos_neg_ratio:.4f} <= diet_max_pos_neg_view_ratio={diet_max_pos_neg_view_ratio:.4f} "
        f"{'OK' if ratio_ok else 'FAIL'}",
        flush=True,
    )
    if not neg_ok:
        failures.append(
            f"neg_aug_frac {neg_aug_frac:.4f} < "
            f"diet_min_neg_aug_frac {diet_min_neg_aug_frac:.4f}"
        )
    if not ratio_ok:
        failures.append(
            f"pos_neg_view_ratio {pos_neg_ratio:.4f} > "
            f"diet_max_pos_neg_view_ratio {diet_max_pos_neg_view_ratio:.4f}"
        )
    if failures:
        for failure in failures:
            print(f"DIET FAIL: {failure}", flush=True)
        return 2
    print("DIET PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
