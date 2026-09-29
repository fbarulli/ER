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


if __name__ == "__main__":
    unittest.main()
