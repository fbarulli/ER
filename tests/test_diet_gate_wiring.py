"""Diet-gate arithmetic on minimal censuses.

The bundle diet contract (config/training.yaml ``diet_*``) is enforced by
scripts/diet_manifest.py, which the CPU bundle lane runs: ``prepare_all``
stage ``suite_inputs`` packages through ``model_tracks.package``, whose
``preflight`` (src/model_tracks/preflight.py) executes the gate on the text
bundle. That is where the gate runs now — the legacy local builder that also
ran it, and its "rebuild fails loud" wiring, were deleted with the local CPU
prepping (owner directive: all bundling and prepping goes to the Colab CPU
bundle process).

Drift policy deleted outright (owner ruling 2026-10-13): the loader records
masking/easy provenance and never compares it — no warning, no hard failure,
no strict switch anywhere on the training path.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diet_manifest import main as diet_manifest_main
from training.prepared_bundle import write_prepared_bundle


def _write_bundle(path: Path, *, pos, train_neg, hard_negative_mask_audit):
    """Minimal self-consistent bundle used by both gate directions here."""
    payload = [
        "cola water", "cola aqua", "cola lite", "cola zero", "cola diet",
        "cola gold",
    ]
    return write_prepared_bundle(
        path,
        df=pd.DataFrame({"sku_id": ["p1", "p2"]}),
        payload=payload,
        structured_features=np.zeros((len(payload), 4), dtype=np.float32),
        row_bc=np.asarray(["g1"] * len(payload), dtype=object),
        country=np.asarray(["US"] * len(payload), dtype=object),
        pos=np.asarray(pos, dtype=int),
        hp_pairs=np.empty((0, 2), dtype=int),
        emb0=np.zeros((len(payload), 4), dtype=np.float32),
        neg=np.asarray(train_neg, dtype=int),
        train_neg=np.asarray(train_neg, dtype=int),
        neg_sources=np.asarray(["gate"] * len(train_neg), dtype=object),
        train_neg_sources=np.asarray(["gate"] * len(train_neg), dtype=object),
        mask_audit=[],
        hard_negative_mask_audit=hard_negative_mask_audit,
        labeled_pairs_csv=b"x",
        canonical_records_csv=b"y",
        gate_results_csv=b"z",
        payload_variant="full",
        masking_profile="baseline",
    )


def _passing_census() -> dict:
    """A minimal census that clears both diet clauses under MNRL.

    Both appended copies occupy the payload suffix (4, 5), the swap copy's
    anchor owns a minted positive, and pos:neg stays inside both ceilings.
    """
    train_neg = np.asarray([[4, 3], [5, 3]], dtype=int)
    audit = [
        {
            "anchor_payload_idx": 0,
            "copy_payload_idx": 4,
            "pair_payload_idx": 3,
            "target_mode": "swap_values",
            "fields_hit": [],
        },
        {
            "anchor_payload_idx": 0,
            "copy_payload_idx": 5,
            "pair_payload_idx": 3,
            "target_mode": "random",
        },
    ]
    return {
        "pos": np.asarray([[0, 1], [4, 5]], dtype=int),
        "train_neg": train_neg,
        "hard_negative_mask_audit": audit,
    }


class DietGateVerdictTest(unittest.TestCase):
    """The gate's own arithmetic on minimal censuses drives the wiring."""

    def test_no_neg_presentations_is_a_clause_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "failing.pkl.gz"
            _write_bundle(
                bundle,
                pos=np.asarray([[0, 1]], dtype=int),
                train_neg=np.empty((0, 2), dtype=int),
                hard_negative_mask_audit=[],
            )
            captured = io.StringIO()
            with redirect_stdout(captured):
                verdict = diet_manifest_main([sys.argv[0], str(bundle)])
            # Since 6b328b2 (2026-10-04): 0 pass, 2 integrity/usage failure,
            # 3 valid input whose presentation diet misses its thresholds.
            # Zero negative presentations is the latter.
            self.assertEqual(verdict, 3)

    def test_mnrl_counted_census_passes_both_clauses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "passing.pkl.gz"
            _write_bundle(bundle, **_passing_census())
            captured = io.StringIO()
            with redirect_stdout(captured):
                verdict = diet_manifest_main([sys.argv[0], str(bundle)])
        self.assertEqual(verdict, 0)


if __name__ == "__main__":
    unittest.main()
