"""MNRL diet accounting: the gate counts only populations the loss trains.

Two accounting fixes for the MNRL/contrastive diet path:

* Swap hard-negative copies are minted but omitted from MNRL triples by
  design (training._mnrl_training_triples_with_populations). Counting them in
  neg_aug_views inflated neg_aug_frac and made the diet gate lenient for
  MNRL; they are now excluded.
* Masked-positive copies only train when their source anchor has an explicit
  negative in the training pool; copies without one (76% on the shipped
  bundle) never see a gradient. The diet reports real survival and excludes
  dead copies from pos_views.

Non-MNRL losses train every retained view and are unchanged.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.diet_manifest import (
    _negative_anchors,
    effective_neg_aug_views,
    effective_pos_views,
)


_SWAP = {
    "anchor_payload_idx": 1,
    "copy_payload_idx": 20,
    "pair_payload_idx": 3,
    "target_mode": "swap_values",
}
_RANDOM = {
    "anchor_payload_idx": 1,
    "copy_payload_idx": 21,
    "pair_payload_idx": 4,
    "target_mode": "random",
}
_COUNTERFACTUAL = {
    "anchor_payload_idx": 1,
    "copy_payload_idx": 22,
    "pair_payload_idx": 5,
    "target_mode": "counterfactual",
}


class EffectiveNegAugViewsTest(unittest.TestCase):
    def test_mnrl_excludes_swap_hard_negatives(self):
        retained = [_SWAP, _RANDOM, _COUNTERFACTUAL]
        self.assertEqual(
            effective_neg_aug_views(retained, loss="mnrl"),
            2,  # random + counterfactual train; swap copy never does
        )

    def test_non_mnrl_counts_every_retained_copy(self):
        retained = [_SWAP, _RANDOM, _COUNTERFACTUAL]
        self.assertEqual(
            effective_neg_aug_views(retained, loss="contrastive"),
            3,
        )
        self.assertEqual(
            effective_neg_aug_views(retained, loss="triplet"),
            3,
        )

    def test_swap_copies_alone_drop_mnrl_frac_below_threshold(self):
        # The shipped bundle sits at 8,828 raw views but only 6,121 train
        # under MNRL (2,707 swap copies excluded) — the honest signal the
        # gate must see.
        retained = [_SWAP] * 2 + [_RANDOM]
        self.assertEqual(effective_neg_aug_views(retained, loss="mnrl"), 1)


class EffectivePosViewsTest(unittest.TestCase):
    def _neg(self) -> np.ndarray:
        # anchor 1 has a negative; anchor 9 has none
        return np.array([[1, 50], [1, 51]], dtype=int)

    def test_mnrl_excludes_masked_positives_without_source_negative(self):
        audit = [
            dict(_RANDOM),  # anchor 1 -> survives
            {
                "anchor_payload_idx": 9,
                "copy_payload_idx": 90,
                "pair_payload_idx": 5,
                "target_mode": "random",
            },  # anchor 9 -> dead
            dict(_SWAP),  # swap positives train; not dead
        ]
        surviving, dead = effective_pos_views(
            100, audit, self._neg(), loss="mnrl"
        )
        self.assertEqual(dead, 1)
        self.assertEqual(surviving, 99)

    def test_non_mnrl_is_a_no_op(self):
        audit = [
            {
                "anchor_payload_idx": 9,
                "copy_payload_idx": 90,
                "pair_payload_idx": 5,
                "target_mode": "random",
            }
        ]
        surviving, dead = effective_pos_views(
            100, audit, self._neg(), loss="contrastive"
        )
        self.assertEqual((surviving, dead), (100, 0))

    def test_swap_positives_are_never_counted_dead(self):
        audit = [dict(_SWAP)]
        surviving, dead = effective_pos_views(
            50, audit, self._neg(), loss="mnrl"
        )
        self.assertEqual(dead, 0)
        self.assertEqual(surviving, 50)


class NegativeAnchorsTest(unittest.TestCase):
    def test_collects_anchors_that_have_a_negative(self):
        train_neg = np.array([[1, 50], [2, 51], [4, 52]], dtype=int)
        self.assertEqual(_negative_anchors(train_neg), {1, 2, 4})

    def test_empty_pool_yields_no_anchors(self):
        self.assertEqual(_negative_anchors(np.empty((0, 2), dtype=int)), set())


def _plan(folds):
    """A complete frozen MNRL plan over (fold_i, triples) pairs."""
    return {
        "identity": {"loss": "mnrl"},
        "inputs": {
            "folds": [
                {
                    "fold_i": label,
                    "objective": {
                        "triples": triples,
                        "dataset": {
                            "anchor": [t[0] for t in triples],
                            "positive": [t[1] for t in triples],
                            "negative": [t[2] for t in triples],
                        },
                    },
                }
                for label, triples in folds
            ]
        },
    }


class MaskedPositiveFoldCoverageTest(unittest.TestCase):
    """TODO "compute coverage PER FOLD": the objective decides, not a heuristic."""

    def test_coverage_is_measured_per_fold_and_unioned(self):
        from training.diet_coverage import folded_objectives, masked_positive_coverage

        # Copy 20 trains in fold 0 only, copy 21 in fold 1 only, copy 22 in
        # neither: a bundle-level count could not tell these apart.
        data = {"training_plan": _plan([
            (0, [[20, 5, 9]]),
            (1, [[21, 6, 9]]),
        ])}
        audit = [
            {"copy_payload_idx": 20, "anchor_payload_idx": 1},
            {"copy_payload_idx": 21, "anchor_payload_idx": 2},
            {"copy_payload_idx": 22, "anchor_payload_idx": 3},
        ]
        coverage = masked_positive_coverage(audit, folded_objectives(data))
        self.assertEqual(
            [(f["fold"], f["trained"], f["coverage"]) for f in coverage["folds"]],
            [(0, 1, 1 / 3), (1, 1, 1 / 3)],
        )
        self.assertEqual(coverage["trained_union"], 2)
        self.assertEqual(coverage["never_trained"], 1)

    def test_twin_lineage_copies_are_not_reported_dead(self):
        """The 980/1,200 mis-measurement: no train_neg anchor, but trained.

        The copy's source anchor owns no explicit negative in ``train_neg``
        (the bundle-level heuristic calls it dead), yet the frozen objective
        trains it through the minted-twin negative lineage. Per-fold coverage
        must report it as trained and NOT subtract it from pos_views.
        """
        from training.diet_coverage import folded_objectives, masked_positive_coverage

        data = {"training_plan": _plan([(0, [[100, 4, 7]])])}
        audit = [{"copy_payload_idx": 100, "anchor_payload_idx": 3}]
        train_neg = np.array([[1, 50], [2, 51]], dtype=int)  # no anchor 3
        coverage = masked_positive_coverage(
            audit, folded_objectives(data), source_negatives=train_neg
        )
        self.assertEqual(coverage["trained_union"], 1)
        self.assertEqual(coverage["never_trained"], 0)
        # The bundle-level lower bound is reported as a LOWER BOUND, not survival.
        self.assertEqual(coverage["source_anchor_covered"], 0)

    def test_incomplete_frozen_objective_fails_loud(self):
        from training.diet_coverage import folded_objectives

        with self.assertRaisesRegex(ValueError, "complete frozen objective"):
            folded_objectives({
                "training_plan": {
                    "identity": {"loss": "mnrl"},
                    "inputs": {"skipped": ["fold-0"], "folds": []},
                }
            })

    def test_missing_or_non_mnrl_plan_returns_no_folds(self):
        """No plan (or a different loss) leaves the bundle-level fallback."""
        from training.diet_coverage import folded_objectives

        self.assertEqual(folded_objectives({}), [])
        self.assertEqual(folded_objectives({"training_plan": None}), [])
        self.assertEqual(
            folded_objectives(
                {"training_plan": {"identity": {"loss": "contrastive"}}}
            ),
            [],
        )

    def test_triple_and_dataset_row_mismatch_fails_loud(self):
        from training.diet_coverage import folded_objectives

        plan = _plan([(0, [[1, 2, 3]])])
        plan["inputs"]["folds"][0]["objective"]["dataset"]["anchor"] = []
        with self.assertRaisesRegex(ValueError, "dataset rows differ"):
            folded_objectives({"training_plan": plan})

    def test_empty_audit_reports_zero_coverage(self):
        from training.diet_coverage import folded_objectives, masked_positive_coverage

        coverage = masked_positive_coverage([], folded_objectives({"training_plan": _plan([(0, [[1, 2, 3]])])}))
        self.assertEqual(coverage["copies"], 0)
        self.assertEqual(coverage["never_trained"], 0)
        self.assertEqual(coverage["folds"][0]["coverage"], 0.0)

    def test_none_source_negatives_leaves_the_lower_bound_at_zero(self):
        from training.diet_coverage import folded_objectives, masked_positive_coverage

        audit = [{"copy_payload_idx": 20, "anchor_payload_idx": 1}]
        coverage = masked_positive_coverage(
            audit, folded_objectives({"training_plan": _plan([(0, [[20, 5, 9]])])})
        )
        self.assertEqual(coverage["source_anchor_covered"], 0)
        self.assertEqual(coverage["trained_union"], 1)


if __name__ == "__main__":
    unittest.main()
