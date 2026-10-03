"""F1 — the ``<source>+aug`` provenance tag must resolve to its base population.

Defect: train.py mints static masked/value-swapped copies of hard negatives
and joins them into both the eval and training pools with an
``"<source>+aug"`` provenance label (f-string concat, ~6,776 rows/fold on
current bundles). The tag is invisible to the static producer scan (built by
concatenation, not a literal) and absent from DATAPOINT_POPULATION_SPEC, so
any contrastive fold on current bundles dies with
``UnregisteredDatapointPopulationError`` — in BOTH coverage checks:

* ``_training_pair_populations`` passed the source array through verbatim,
  so pair populations contained ``gate+aug`` etc. → the coverage writer
  raised at the unregistered-population guard;
* ``_negative_source_accounting`` enumerated the raw source array → N3
  (registry conformance) raised on the compound tag.

Fix path (chosen over registering the tag): the copy's SOURCE is still the
base population — normalize compound tags to their base wherever a registry
check happens — and the static minting mode (hard_negative_mask_audit
``target_mode``) is carried in the usage row's lineage block, so no
information is lost and the registry never grows a combinatorial
``<base>+aug`` tag family.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import core.common as core_common
import training.training as training

REPO_ROOT = Path(__file__).resolve().parents[1]


class _CapturingVisibilityLog(unittest.TestCase):
    """Capture visibility artifacts instead of writing results/."""

    def setUp(self) -> None:
        self.frames: dict[str, pd.DataFrame] = {}
        self._original = core_common.write_visibility_log

        def capture(df, name, run_tag, sample):  # noqa: ANN001 - test double
            self.frames[name] = df

        core_common.write_visibility_log = capture
        self.addCleanup(
            lambda: setattr(core_common, "write_visibility_log", self._original)
        )


_AUG_AUDIT = [
    {
        "anchor_payload_idx": 0,
        "copy_payload_idx": 10,
        "pair_payload_idx": 5,
        "gtin": "bc0",
        "realized_extent": 0.25,
        "population": "negative",
        "target_mode": "random",
    }
]

_PAYLOAD_METADATA_FIELDS = (
    "point_kind",
    "source_payload_idx",
    "sku_id",
    "gtin",
    "brand",
    "sku_name_eng",
    "attribute",
    "country",
    "category",
    "volume",
    "pack",
    "package_type",
    "flavor",
    "carbonation",
    "sweetener",
    "pulp",
    "volume_confidence",
    "pack_confidence",
)


def _payload_metadata(n: int) -> list[dict]:
    return [
        {key: "" for key in _PAYLOAD_METADATA_FIELDS} for _ in range(n)
    ]


def _fold_inputs():
    train_pos = np.array([[0, 1], [2, 3]], dtype=int)
    train_neg = np.array([[0, 5], [10, 5]], dtype=int)
    sources = np.array(["gate", "gate+aug"], dtype=object)
    return train_pos, train_neg, sources


class AugSourcePopulationTest(_CapturingVisibilityLog):
    def test_pair_populations_strip_aug_suffix_to_registered_base(self):
        train_pos, train_neg, sources = _fold_inputs()
        populations = training._training_pair_populations(
            train_pos,
            train_neg,
            train_neg_sources=sources,
            hp_in_train=None,
            mask_audit=[],
        )
        self.assertEqual(
            populations,
            ["gate_positive", "gate_positive", "gate", "gate"],
        )
        self.assertTrue(
            all(p in training.KNOWN_DATAPOINT_POPULATIONS for p in populations)
        )

    def test_fold_with_masked_hard_negatives_writes_coverage_without_raise(self):
        train_pos, train_neg, sources = _fold_inputs()
        populations = training._training_pair_populations(
            train_pos,
            train_neg,
            train_neg_sources=sources,
            hp_in_train=None,
            mask_audit=[],
        )
        lineage = training._build_pair_lineage(
            train_pos,
            train_neg,
            train_neg_sources=sources,
            mask_audit=[],
            hard_negative_mask_audit=_AUG_AUDIT,
            payload_metadata=_payload_metadata(12),
            gate_lookup={},
        )
        counts = {
            (0, 0, "gate_positive", "none", 0): 1,
            (0, 1, "gate_positive", "none", 0): 1,
            (0, 2, "gate", "none", 0): 1,
            (0, 3, "gate", "none", 0): 1,
        }
        stats = training._write_datapoint_usage(
            fold_i=0,
            pair_populations=populations,
            presentation_counts=counts,
            pair_lineage=lineage,
            dynamic_populations=set(),
            run_tag="test-aug-registration",
            sample=True,
        )
        coverage = self.frames["datapoint_type_coverage_fold0.csv"]
        gate_row = coverage[coverage.population == "gate"].iloc[0]
        self.assertEqual(int(gate_row["expected_pairs"]), 2)
        self.assertEqual(str(gate_row["status"]), "ok")
        self.assertFalse((coverage["status"] == "unregistered").any())
        usage = self.frames["datapoint_usage_fold0.csv"]
        copy_rows = usage[usage["pair_id"] == 3]
        self.assertEqual(len(copy_rows), 1)
        # the static minting mode rides the lineage block of the usage row
        self.assertEqual(copy_rows.iloc[0]["target_mode"], "random")
        self.assertEqual(copy_rows.iloc[0]["is_masked_copy"], 1)
        self.assertEqual(int(stats["n_presented_gate"]), 2)

    def test_negative_source_census_folds_aug_copies_into_base(self):
        coverage = training._negative_source_accounting(
            fold_i=0,
            tr_negs=np.array([[0, 5], [10, 5]], dtype=int),
            tr_neg_sources=np.array(["gate", "gate+aug"], dtype=object),
            usage_rows=[],
        )
        self.assertEqual(coverage["n_train_neg_source_gate"], 2)
        self.assertAlmostEqual(coverage["pct_train_neg_source_gate"], 1.0)
        self.assertEqual(coverage["n_train_neg_source_unregistered"], 0)


if __name__ == "__main__":
    unittest.main()
