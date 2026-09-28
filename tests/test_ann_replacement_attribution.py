"""A3 — ANN re-attribution must apply only to REALIZED replacements.

Defect: ``_dynamic_mask_negative_transform`` relabeled EVERY label-0 pair as
``ann_finetuned`` whenever the ANN refresh lane was active — via
``ann_sources.get(pair_id, "ann_finetuned")`` — even for slots whose
``ann_pairs.get(pair_id)`` is ``None`` (the miner may find FEWER pairs than
the fold's negative slots; training.py assigns the explicit subset
``slot_ids[: len(pairs)]``). A gate slot that never received a replacement
was presented under the ``ann_finetuned`` label.

Telemetry that lied: datapoint_usage_fold{i}.csv (phantom population on
never-replaced rows), datapoint_type_coverage_fold{i}.csv (true sources
undercounted -> spurious ``missing``; ann_finetuned inflated despite its
documented presented_label role with source total 0 by construction),
_negative_source_accounting (present/selected/backprop attributed to
ann_finetuned), mask_visibility.csv population labels, and spurious
ambiguous-pair-attribution warnings.

The registry doctrine (training.py, presented_label role): the label marks
a negative whose TEXT WAS REPLACED. Only realized replacements may carry it.
"""

from __future__ import annotations

import random
import unittest

import training.training as training


def _batch(pair_ids, labels):
    return {
        "label": list(labels),
        "pair_id": list(pair_ids),
        "sentence1": [f"anchor {i}" for i in pair_ids],
        "sentence2": [f"pair {pid}" for pid in pair_ids],
    }


def _run_transform(batch, pair_populations, ann_pairs, ann_sources, counts):
    training._dynamic_mask_negative_transform(
        batch,
        rng=random.Random(0),
        frac=0.0,  # deterministic: never dynamically re-mask
        mask_prob=None,
        mask_lo=0.1,
        mask_hi=0.2,
        counts={},
        counts_by_epoch={},
        stats_by_epoch={},
        epoch_ref={"epoch": 0},
        ann_pairs=ann_pairs,
        ann_structured_features=None,
        ann_sources=ann_sources,
        ann_state={"version": 3},
        pair_populations=pair_populations,
        presentation_counts=counts,
        mask_audit=None,
        fold=0,
    )


class AnnReplacementAttributionTest(unittest.TestCase):
    def test_unreplaced_gate_slot_keeps_its_population(self):
        counts: dict[tuple, int] = {}
        _run_transform(
            batch=_batch(pair_ids=[0, 1], labels=[0, 0]),
            pair_populations=["gate", "gate"],
            ann_pairs={1: ("x", "y")},  # slot 1 realized, slot 0 never replaced
            ann_sources={1: "ann_finetuned"},
            counts=counts,
        )
        attributed = {key[1]: key[2] for key in counts}
        self.assertEqual(
            attributed,
            {0: "gate", 1: "ann_finetuned"},
            f"slot 0 was never replaced but got re-attributed: {counts}",
        )

    def test_realized_replacement_is_attributed_to_ann_finetuned(self):
        counts: dict[tuple, int] = {}
        _run_transform(
            batch=_batch(pair_ids=[1], labels=[0]),
            pair_populations=["gate"],
            ann_pairs={1: ("replaced-a", "replaced-b")},
            ann_sources={1: "ann_finetuned"},
            counts=counts,
        )
        self.assertEqual(
            [key[2] for key in counts],
            ["ann_finetuned"],
            counts,
        )


if __name__ == "__main__":
    unittest.main()
