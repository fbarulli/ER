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

from core.common import SSOT_LOSS, masking_cfg, runtime
from training.diet_coverage import folded_objectives, masked_positive_coverage
from training.prepared_bundle import load_prepared_bundle


def project_train_time_neg_views(
    neg_views: int, *, enabled: bool, ratio_to_hard: float, loss: str
) -> int:
    """Upper projection only; actual split-local easy candidates can be absent."""
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
    twin_copies = set()
    # Balanced-augmentation bundles record their lineage against the
    # PRE-PROJECTION payload coordinates: augment_balanced stamps
    # anchor_payload_idx / pair_payload_idx before the frozen objective is
    # built, so the (anchor, pair, copy) tuple never matches a frozen triple
    # even though copy_payload_idx is in the triple's own index space. Their
    # payload carries augmentation_coverage; classic-lane bundles do not.
    # Matching the copy index for those bundles restores the measurement
    # without touching the classic lane's arithmetic.
    balanced_lane = "augmentation_coverage" in data
    for row in data.get("hard_negative_mask_audit", []):
        if str(row.get("population", "hard_negative")) != "hard_negative":
            continue
        copy = int(row["copy_payload_idx"])
        pair = int(row["pair_payload_idx"])
        if _mode(row) == "counterfactual":
            twins.add((int(row["anchor_payload_idx"]), pair, copy))
            twin_copies.add(copy)
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
    def _augmented_negative(a: int, p: int, n: int) -> bool:
        if (a, n) in copy_pairs or (a, p, n) in twins:
            return True
        # Balanced lane only: the triple's negative IS the augmented copy.
        return balanced_lane and n in twin_copies

    return [{"fold": fold, "source": source, "presentations": len(triples),
             "negative_augmented": sum(_augmented_negative(int(a), int(p), int(n))
                                       for a, p, n in triples)}
            for fold, triples, source in objectives]


class DietInputs:
    """One gate run's resolved thresholds + bundle arrays (read-only)."""

    def __init__(self, *, manifest, data, diet_min_neg_aug_frac,
                 diet_max_pos_neg_view_ratio, mask_hard_negatives,
                 hard_negative_frac, pos, neg, train_neg, mask_audit,
                 neg_audit, loss) -> None:
        self.manifest = manifest
        self.data = data
        self.diet_min_neg_aug_frac = diet_min_neg_aug_frac
        self.diet_max_pos_neg_view_ratio = diet_max_pos_neg_view_ratio
        self.mask_hard_negatives = mask_hard_negatives
        self.hard_negative_frac = hard_negative_frac
        self.pos = pos
        self.neg = neg
        self.train_neg = train_neg
        self.mask_audit = mask_audit
        self.neg_audit = neg_audit
        self.loss = loss


class DietViews:
    """One gate run's view census (mutated by the MNRL accounting phase)."""

    def __init__(self, *, pos_views, neg_views, retained_neg_audit,
                 neg_aug_views, surviving_pos_views, dead_masked_pos,
                 easy_enabled, easy_ratio, projected_neg_views,
                 projected_neg_presentations) -> None:
        self.pos_views = pos_views
        self.neg_views = neg_views
        self.retained_neg_audit = retained_neg_audit
        self.neg_aug_views = neg_aug_views
        self.surviving_pos_views = surviving_pos_views
        self.dead_masked_pos = dead_masked_pos
        self.easy_enabled = easy_enabled
        self.easy_ratio = easy_ratio
        self.projected_neg_views = projected_neg_views
        self.projected_neg_presentations = projected_neg_presentations
        # Per-fold masked-positive coverage from the frozen objective; None
        # when the bundle has no complete plan (bundle-level fallback used).
        self.fold_coverage = None
        self.pos_base = 0
        self.neg_base = 0


class DietBaseCensus:
    """The audit-vs-pair sanity outcome for one bundle."""

    def __init__(self, *, pos_base, neg_base, exceeds_pairs) -> None:
        self.pos_base = pos_base
        self.neg_base = neg_base
        self.exceeds_pairs = exceeds_pairs


class DietMNRLOutcome:
    """MNRL fold presentation counts (empty outside the MNRL loss)."""

    def __init__(self, *, actual_folds, invalid_frozen=False, exception=None) -> None:
        self.actual_folds = actual_folds
        self.invalid_frozen = invalid_frozen
        self.exception = exception


class DietGate:
    """One bundle's diet gate run: inputs -> verdict (exit code).

    Phases below are single responsibilities, running in ONE fixed order
    inside run(); every print stays byte-identical to the pre-refactor main.

    Phase map:
      read_inputs       — masking thresholds + the bundle's pair/audit arrays
      view_counts       — base/retained views, MNRL survival, easy projection
      base_census       — audit-vs-pair sanity (the DIET FAIL guard)
      population_table  — the views table + MNRL survival note
      mnrl_accounting   — frozen/production triple presentations
      verdict           — threshold fractions/ratios, failures, exit code
    """

    EXIT_USAGE = 2
    EXIT_FAIL = 2
    EXIT_THRESHOLD_MISS = 3  # Valid inputs, but the presentation diet misses its thresholds.
    EXIT_PASS = 0

    def __init__(self, argv, *, prepared=None) -> None:
        self._argv = argv
        self._prepared = prepared

    def run(self) -> int:
        inputs = self.read_inputs()
        if len(self._argv) != 2 or inputs is None:
            print(f"usage: {Path(self._argv[0]).name} BUNDLE_PATH", file=sys.stderr)
            return self.EXIT_USAGE
        views = self.view_counts(inputs)
        base = self.base_census(inputs, views)
        if base.exceeds_pairs:
            print(
                "DIET FAIL: audit rows exceed pair rows "
                f"(pos {views.pos_views}/{len(inputs.mask_audit)}, neg {len(inputs.neg)}/{len(inputs.neg_audit)})",
                flush=True,
            )
            return self.EXIT_FAIL
        self.population_table(inputs, views)
        verdict = self.mnrl_accounting(inputs, views)
        if verdict.invalid_frozen:
            print(f"DIET FAIL: invalid frozen MNRL objective: {verdict.exception}", flush=True)
            return self.EXIT_FAIL
        return self.verdict(inputs, views, verdict)

    # ── phase: inputs ──────────────────────────────────────────────────────

    def read_inputs(self) -> "DietInputs | None":
        """Usage contract + every masking threshold + the bundle arrays."""
        if len(self._argv) != 2:
            return None
        manifest, data = self._prepared if self._prepared is not None else load_prepared_bundle(Path(self._argv[1]))
        masking = masking_cfg(manifest.masking_profile)
        return DietInputs(
            manifest=manifest,
            data=data,
            diet_min_neg_aug_frac=float(masking["diet_min_neg_aug_frac"]),
            diet_max_pos_neg_view_ratio=float(masking["diet_max_pos_neg_view_ratio"]),
            mask_hard_negatives=bool(masking["mask_hard_negatives"]),
            hard_negative_frac=float(masking["hard_negative_frac"]),
            pos=data["pos"],
            neg=data["neg"],
            train_neg=data["train_neg"],
            mask_audit=list(data.get("mask_audit", [])),
            neg_audit=[
                row for row in data.get("hard_negative_mask_audit", [])
                if str(row.get("population", "hard_negative")) == "hard_negative"
            ],
            loss=str(SSOT_LOSS),
        )

    # ── phase: view counts ─────────────────────────────────────────────────

    def view_counts(self, inputs: "DietInputs") -> "DietViews":
        """Base/retained views, MNRL survival, and the easy-negative projection."""
        pos = inputs.pos
        neg = inputs.neg
        train_neg = inputs.train_neg
        mask_audit = inputs.mask_audit
        neg_audit = inputs.neg_audit
        loss = inputs.loss

        pos_views = int(len(pos))
        neg_views = int(len(train_neg))
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
        # PREFER the frozen objective's PER-FOLD truth over that bundle-level
        # heuristic (TODO "compute coverage PER FOLD; prefer
        # generate-only-if-covered over backfill"). The heuristic cannot see
        # the minted-twin negative lineage, so it reported 980/1,200 masked
        # positives dead on the shipped bundle while the frozen objective
        # trains all 1,200 — subtracting those "dead" rows was a
        # backfill-then-remove fiction. Only a complete frozen plan can answer
        # the per-fold question; without one the documented fallback stands.
        fold_coverage = None
        if loss == "mnrl" and mask_audit:
            try:
                frozen_folds = folded_objectives(inputs.data)
            except (KeyError, ValueError):
                frozen_folds = []
            if frozen_folds:
                fold_coverage = masked_positive_coverage(
                    mask_audit, frozen_folds, source_negatives=train_neg
                )
                dead_masked_pos = int(fold_coverage["never_trained"])
                surviving_pos_views = pos_views - dead_masked_pos
        easy_cfg = runtime("random_easy_negatives")
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
        views = DietViews(
            pos_views=pos_views,
            neg_views=neg_views,
            retained_neg_audit=retained_neg_audit,
            neg_aug_views=neg_aug_views,
            surviving_pos_views=surviving_pos_views,
            dead_masked_pos=dead_masked_pos,
            easy_enabled=easy_enabled,
            easy_ratio=easy_ratio,
            projected_neg_views=projected_neg_views,
            projected_neg_presentations=projected_neg_presentations,
        )
        views.fold_coverage = fold_coverage
        return views

    # ── phase: base census ─────────────────────────────────────────────────

    def base_census(self, inputs: "DietInputs", views: "DietViews") -> "DietBaseCensus":
        """Audit-vs-pair sanity: base rows can never go negative."""
        pos_base = views.pos_views - len(inputs.mask_audit)
        neg_base = int(len(inputs.neg)) - len(inputs.neg_audit)
        views.pos_base = pos_base
        views.neg_base = neg_base
        return DietBaseCensus(pos_base=pos_base, neg_base=neg_base,
                              exceeds_pairs=pos_base < 0 or neg_base < 0)

    @staticmethod
    def _mode_counts(rows: list[dict], population: str) -> dict[str, int]:
        """Mode census of one audit population (alphabetical on print)."""
        counts: dict[str, int] = {}
        for row in rows:
            if str(row.get("population", population)) != population:
                continue
            counts[_mode(row)] = counts.get(_mode(row), 0) + 1
        return counts

    def population_table(self, inputs: "DietInputs", views: "DietViews") -> None:
        """The views table + the MNRL survival note (print order pinned)."""
        loss = inputs.loss
        mask_audit = inputs.mask_audit
        neg_audit = inputs.neg_audit
        pos_base, neg_base = views.pos_base, views.neg_base

        pos_modes = self._mode_counts(mask_audit, "positive")
        neg_modes = self._mode_counts(neg_audit, "hard_negative")

        print(f"[diet] bundle={self._argv[1]} profile={inputs.manifest.masking_profile}", flush=True)
        print("population     target_mode   views", flush=True)
        print(f"positive       base          {pos_base:,}", flush=True)
        for mode in sorted(pos_modes):
            print(f"positive       {mode:<11} {pos_modes[mode]:,}", flush=True)
        print(f"hard_negative  base          {neg_base:,}", flush=True)
        for mode in sorted(neg_modes):
            print(f"hard_negative  {mode:<11} {neg_modes[mode]:,}", flush=True)
        print(
            f"presentations  projected    {views.projected_neg_presentations:,} "
            f"(pos_views={views.surviving_pos_views:,}, neg_views={views.projected_neg_views:,})",
            flush=True,
        )
        if views.fold_coverage is not None:
            coverage = views.fold_coverage
            for counts in coverage["folds"]:
                print(
                    f"[diet] MNRL masked-positive coverage fold={counts['fold']} "
                    f"trained={counts['trained']:,}/{coverage['copies']:,} "
                    f"({counts['coverage']:.4f}) of {counts['presentations']:,} objective presentations",
                    flush=True,
                )
            print(
                f"[diet] MNRL masked-positive coverage: {coverage['trained_union']:,}/"
                f"{coverage['copies']:,} copies train in >=1 fold, "
                f"{coverage['never_trained']:,} never train (the rows a "
                "generate-only-if-covered policy must refuse to mint, never backfill); "
                f"bundle-level source-anchor lower bound={coverage['source_anchor_covered']:,} "
                "is NOT survival",
                flush=True,
            )
        elif views.dead_masked_pos:
            print(
                f"[diet] MNRL survival: {views.dead_masked_pos:,} masked-positive copies "
                "have no source negative in the training pool and never train "
                "(excluded from pos_views)",
                flush=True,
            )

    # ── phase: MNRL accounting ─────────────────────────────────────────────

    def mnrl_accounting(self, inputs: "DietInputs", views: "DietViews") -> "DietMNRLOutcome":
        """Actual triple presentations when the loss is MNRL (frozen first)."""
        loss = inputs.loss
        if loss != "mnrl":
            return DietMNRLOutcome(actual_folds=[])
        try:
            actual_folds = mnrl_presentation_counts(inputs.data)
        except (KeyError, ValueError) as exc:
            return DietMNRLOutcome(actual_folds=[], invalid_frozen=True, exception=exc)
        for counts in actual_folds:
            print(f"[diet] MNRL fold={counts['fold']} source={counts['source']} "
                  f"actual_triples={counts['presentations']:,} "
                  f"negative_augmented={counts['negative_augmented']:,}", flush=True)
        views.surviving_pos_views = views.projected_neg_views = views.projected_neg_presentations = sum(
            counts["presentations"] for counts in actual_folds)
        views.neg_aug_views = sum(counts["negative_augmented"] for counts in actual_folds)
        return DietMNRLOutcome(actual_folds=actual_folds)

    # ── phase: verdict ─────────────────────────────────────────────────────

    def verdict(self, inputs: "DietInputs", views: "DietViews",
                outcome: "DietMNRLOutcome") -> int:
        """Threshold fractions/ratios + failures + the exit code (statements
        are the original sequence, division-by-zero and float semantics
        included)."""
        failures: list[str] = []
        for counts in outcome.actual_folds:
            total = counts["presentations"]
            if not total or counts["negative_augmented"] / total < inputs.diet_min_neg_aug_frac:
                failures.append(f"MNRL fold {counts['fold']} augmented-negative presentations "
                                f"{counts['negative_augmented']}/{total} below "
                                f"diet_min_neg_aug_frac={inputs.diet_min_neg_aug_frac}")
        neg_aug_frac = (
            views.neg_aug_views / views.projected_neg_presentations
            if views.projected_neg_presentations else float("nan")
        )
        pos_neg_ratio = (
            views.surviving_pos_views / views.projected_neg_views
            if views.projected_neg_views else float("nan")
        )
        effective_ratio = (
            views.surviving_pos_views / views.projected_neg_views
            if views.projected_neg_views else float("nan")
        )
        neg_ok = views.projected_neg_presentations > 0 and neg_aug_frac >= inputs.diet_min_neg_aug_frac
        ratio_ok = views.projected_neg_views > 0 and effective_ratio <= inputs.diet_max_pos_neg_view_ratio
        print(
            f"[diet] neg_aug_views={views.neg_aug_views:,} / neg_presentations={views.projected_neg_presentations:,} "
            f"= {neg_aug_frac:.4f} >= diet_min_neg_aug_frac={inputs.diet_min_neg_aug_frac:.4f} "
            f"{'OK' if neg_ok else 'FAIL'}",
            flush=True,
        )
        print(
            f"[diet] bundle-only pos_views={views.pos_views:,} "
            f"/ neg_views={views.neg_views:,} = {views.pos_views / views.neg_views if views.neg_views else float('nan'):.4f} (informational)",
            flush=True,
        )
        print(
            f"[diet] projected_neg_views={views.projected_neg_views:,} "
            f"(easy quota x{views.easy_ratio:g}, enabled={views.easy_enabled}, loss={inputs.loss}; "
            f"dynamic mask frac={inputs.hard_negative_frac:g}, enabled={inputs.mask_hard_negatives} "
            "(in-place replacement — adds no views); "
            "easy-projection is an upper bound, not guaranteed)",
            flush=True,
        )
        if not neg_ok:
            failures.append(
                f"neg_aug_frac {neg_aug_frac:.4f} < "
                f"diet_min_neg_aug_frac {inputs.diet_min_neg_aug_frac:.4f}"
            )
        if not ratio_ok:
            failures.append(
                f"actual pos_neg_view_ratio {effective_ratio:.4f} > "
                f"diet_max_pos_neg_view_ratio {inputs.diet_max_pos_neg_view_ratio:.4f} "
                f"(bundle-only {pos_neg_ratio:.4f})"
            )
        if failures:
            for failure in failures:
                print(f"DIET FAIL: {failure}", flush=True)
            return DietGate.EXIT_THRESHOLD_MISS  # Valid inputs, but the presentation diet misses its thresholds.
        print("DIET PASS", flush=True)
        return DietGate.EXIT_PASS


def main(argv: list[str], *, prepared=None) -> int:
    return DietGate(argv, prepared=prepared).run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
