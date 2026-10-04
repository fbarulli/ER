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

MNRL accounting uses the frozen objective triples when available. Each actual
triple presents one positive and one negative. Negative augmentation is traced
from the hard-negative audit coordinates, including source-anchored twins;
masked-positive triples do not become augmented-negative presentations. Legacy
bundles without a frozen plan are counted through the production triple builder
and clearly labeled as an unsplit census.

Usage: diet_manifest.py BUNDLE_PATH
Exit 0 PASS, exit 2 FAIL naming the exact violated threshold.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from core.common import load_config, masking_cfg
from training.prepared_bundle import load_prepared_bundle


def project_train_time_neg_views(
    neg_views: int, *, enabled: bool, ratio_to_hard: float, loss: str
) -> int:
    """Upper projection only; actual split-local easy candidates can be absent."""
    import math

    base = int(neg_views)
    if loss != "contrastive" or not enabled or float(ratio_to_hard) <= 0.0 or base <= 0:
        return base
    return base + int(math.ceil(base * float(ratio_to_hard)))


def _mode(row: dict) -> str:
    """Audit target_mode with the pre-swap default ('random')."""
    return str(row.get("target_mode") or "random")


def _negative_anchors(train_neg: np.ndarray) -> set[int]:
    """Anchors that carry at least one explicit negative in the training pool.

    MNRL masked positives train only when their source anchor has a negative
    in the fold; an anchor with none produces no triple (training.py
    _mnrl_training_triples_with_populations). Bundle-level granularity, matching
    the rest of the diet gate (the fold split happens later, at train time).
    """
    anchors: set[int] = set()
    for anchor, _negative in np.asarray(train_neg, dtype=int).reshape(-1, 2):
        anchors.add(int(anchor))
    return anchors


def effective_pos_views(
    pos_views: int, mask_audit: list[dict], train_neg: np.ndarray, *, loss: str
) -> tuple[int, int]:
    """Positive views that actually train, and dead masked-positive copies.

    For MNRL a masked-positive copy only survives to a triple when its source
    anchor has an explicit negative in the training pool; otherwise the copy
    is minted but never trained. Those dead rows are excluded from the
    positive view count so the pos/neg ratio is honest. Other losses train
    every view, so this is a no-op. Returns (surviving_views, dead_copies).
    """
    if loss != "mnrl":
        return int(pos_views), 0
    neg_anchors = _negative_anchors(train_neg)
    dead = sum(
        1
        for row in mask_audit or []
        if _mode(row) != "swap_values"
        and int(row["anchor_payload_idx"]) not in neg_anchors
    )
    return int(pos_views) - dead, dead


def effective_neg_aug_views(
    retained_neg_audit: list[dict], *, loss: str, pos: np.ndarray | None = None
) -> int:
    """Augmented negative views MNRL actually trains.

    A swap hard-negative copy (target_mode="swap_values") is only trained by
    MNRL when it has a COMPATIBLE positive: the transplant rewrites the anchor,
    so the source's unchanged positive no longer matches it and the triple
    builder used to omit the row outright. TIER 1(a) mints that counterpart
    (masking.mint_swap_counterpart_positives) and registers it in ``pos``
    against the copy anchor. So a swap row counts here exactly when its copy
    anchor owns a positive — counted per row, from the bundle itself, never
    assumed from the population label. Swap rows with no counterpart are still
    excluded, which is what keeps neg_aug_frac honest.
    """
    if loss != "mnrl":
        return len(retained_neg_audit)
    positives_by_anchor: set[int] = set()
    if pos is not None:
        for anchor, _positive in np.asarray(pos, dtype=int).reshape(-1, 2):
            positives_by_anchor.add(int(anchor))
    return sum(
        1
        for row in retained_neg_audit
        if _mode(row) != "swap_values"
        or int(row["copy_payload_idx"]) in positives_by_anchor
    )


def mnrl_presentation_counts(data: dict) -> list[dict]:
    """Count actual objective rows, distinguishing negative augmentation lineage."""
    copy_pairs = set()
    twins = set()
    for row in data.get("hard_negative_mask_audit", []):
        if str(row.get("population", "hard_negative")) != "hard_negative":
            continue
        copy = int(row["copy_payload_idx"])
        pair = int(row["pair_payload_idx"])
        if _mode(row) == "counterfactual":
            twins.add((int(row["anchor_payload_idx"]), pair, copy))
        else:
            copy_pairs.add((copy, pair))
    plan = data.get("training_plan")
    if plan is not None:
        if plan.get("identity", {}).get("loss") != "mnrl":
            raise ValueError("frozen objective loss differs from MNRL diet")
        if plan["inputs"].get("skipped") or not plan["inputs"].get("folds"):
            raise ValueError("MNRL diet requires a complete frozen objective")
        objectives = []
        for fold in plan["inputs"]["folds"]:
            objective = fold["objective"]
            triples = objective["triples"]
            dataset = objective["dataset"]
            if any(len(dataset[key]) != len(triples) for key in ("anchor", "positive", "negative")):
                raise ValueError("frozen MNRL triple and dataset rows differ")
            objectives.append((fold["fold_i"], triples, "frozen objective"))
    else:
        from training.training import _mnrl_training_triples_with_populations
        rows = _mnrl_training_triples_with_populations(
            data["pos"], data["train_neg"], mask_audit=data.get("mask_audit", []),
            hard_negative_mask_audit=data.get("hard_negative_mask_audit", []))
        objectives = [("unsplit", [triple for triple, _ in rows], "production triple census (no frozen plan)")]
    return [{"fold": fold, "source": source, "presentations": len(triples),
             "negative_augmented": sum((int(a), int(n)) in copy_pairs or
                                       (int(a), int(p), int(n)) in twins
                                       for a, p, n in triples)}
            for fold, triples, source in objectives]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} BUNDLE_PATH", file=sys.stderr)
        return 2
    manifest, data = load_prepared_bundle(Path(argv[1]))
    masking = masking_cfg(manifest.masking_profile)
    diet_min_neg_aug_frac = float(masking["diet_min_neg_aug_frac"])
    diet_max_pos_neg_view_ratio = float(masking["diet_max_pos_neg_view_ratio"])
    mask_hard_negatives = bool(masking["mask_hard_negatives"])
    hard_negative_frac = float(masking["hard_negative_frac"])

    pos = data["pos"]
    neg = data["neg"]
    train_neg = data["train_neg"]
    mask_audit = list(data.get("mask_audit", []))
    neg_audit = [row for row in data.get("hard_negative_mask_audit", [])
                 if str(row.get("population", "hard_negative")) == "hard_negative"]

    pos_views = int(len(pos))
    neg_views = int(len(train_neg))
    training_cfg = load_config()["training"]
    loss = str(training_cfg["loss"])
    selected_neg = {tuple(map(int, pair)) for pair in train_neg}
    retained_neg_audit = [
        row for row in neg_audit
        if (int(row["copy_payload_idx"]), int(row["pair_payload_idx"])) in selected_neg
    ]
    # MNRL trains a swap hard-negative copy only when TIER 1(a) minted it a
    # compatible counterpart positive; count those, exclude the rest.
    neg_aug_views = effective_neg_aug_views(
        retained_neg_audit, loss=loss, pos=pos
    )
    # MNRL masked positives with no source negative never train; report real
    # survival on the positive side too (dead copies excluded from pos_views).
    surviving_pos_views, dead_masked_pos = effective_pos_views(
        pos_views, mask_audit, train_neg, loss=loss
    )
    easy_cfg = load_config()["training"]["random_easy_negatives"]
    easy_enabled = bool(easy_cfg["enabled"])
    easy_ratio = float(easy_cfg["ratio_to_hard"])
    # Dynamic masking rewrites selected negative presentations IN PLACE
    # (training.py _dynamic_mask_negative_transform — "no static negative
    # copies are added"), so it adds NO views for ANY loss and the old
    # +hard_negative_frac projection double-counted phantom views,
    # biasing neg_aug_frac low and the pos/neg ratio low. Presentations
    # are the bundle's own view counts.
    projected_neg_views = neg_views
    projected_neg_presentations = project_train_time_neg_views(
        projected_neg_views, enabled=easy_enabled, ratio_to_hard=easy_ratio, loss=loss
    )
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
        f"presentations  projected    {projected_neg_presentations:,} "
        f"(pos_views={surviving_pos_views:,}, neg_views={projected_neg_views:,})",
        flush=True,
    )
    if dead_masked_pos:
        print(
            f"[diet] MNRL survival: {dead_masked_pos:,} masked-positive copies "
            "have no source negative in the training pool and never train "
            "(excluded from pos_views)",
            flush=True,
        )

    actual_folds = []
    if loss == "mnrl":
        try:
            actual_folds = mnrl_presentation_counts(data)
        except (KeyError, ValueError) as exc:
            print(f"DIET FAIL: invalid frozen MNRL objective: {exc}", flush=True)
            return 2
        for counts in actual_folds:
            print(f"[diet] MNRL fold={counts['fold']} source={counts['source']} "
                  f"actual_triples={counts['presentations']:,} "
                  f"negative_augmented={counts['negative_augmented']:,}", flush=True)
        surviving_pos_views = projected_neg_views = projected_neg_presentations = sum(
            counts["presentations"] for counts in actual_folds)
        neg_aug_views = sum(counts["negative_augmented"] for counts in actual_folds)
    failures: list[str] = []
    for counts in actual_folds:
        total = counts["presentations"]
        if not total or counts["negative_augmented"] / total < diet_min_neg_aug_frac:
            failures.append(f"MNRL fold {counts['fold']} augmented-negative presentations "
                            f"{counts['negative_augmented']}/{total} below "
                            f"diet_min_neg_aug_frac={diet_min_neg_aug_frac}")
    neg_aug_frac = (
        neg_aug_views / projected_neg_presentations if projected_neg_presentations else float("nan")
    )
    pos_neg_ratio = (
        surviving_pos_views / projected_neg_views if projected_neg_views else float("nan")
    )
    effective_ratio = (
        surviving_pos_views / projected_neg_views if projected_neg_views else float("nan")
    )
    neg_ok = projected_neg_presentations > 0 and neg_aug_frac >= diet_min_neg_aug_frac
    ratio_ok = projected_neg_views > 0 and effective_ratio <= diet_max_pos_neg_view_ratio
    print(
        f"[diet] neg_aug_views={neg_aug_views:,} / neg_presentations={projected_neg_presentations:,} "
        f"= {neg_aug_frac:.4f} >= diet_min_neg_aug_frac={diet_min_neg_aug_frac:.4f} "
        f"{'OK' if neg_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"[diet] bundle-only pos_views={pos_views:,} "
        f"/ neg_views={neg_views:,} = {pos_views / neg_views if neg_views else float('nan'):.4f} (informational)",
        flush=True,
    )
    print(
        f"[diet] projected_neg_views={projected_neg_views:,} "
        f"(easy quota x{easy_ratio:g}, enabled={easy_enabled}, loss={loss}; "
        f"dynamic mask frac={hard_negative_frac:g}, enabled={mask_hard_negatives} "
        "(in-place replacement — adds no views); "
        "easy-projection is an upper bound, not guaranteed)",
        flush=True,
    )
    if not neg_ok:
        failures.append(
            f"neg_aug_frac {neg_aug_frac:.4f} < "
            f"diet_min_neg_aug_frac {diet_min_neg_aug_frac:.4f}"
        )
    if not ratio_ok:
        failures.append(
            f"actual pos_neg_view_ratio {effective_ratio:.4f} > "
            f"diet_max_pos_neg_view_ratio {diet_max_pos_neg_view_ratio:.4f} "
            f"(bundle-only {pos_neg_ratio:.4f})"
        )
    if failures:
        for failure in failures:
            print(f"DIET FAIL: {failure}", flush=True)
        return 3  # Valid inputs, but the presentation diet misses its thresholds.
    print("DIET PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
