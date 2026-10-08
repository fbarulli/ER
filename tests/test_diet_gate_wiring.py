"""Diet-gate wiring: the rebuild path fails loud when the diet gate fails.

P1 STRUCTURAL GAP (TODO.md AUGMENTATION TIERED ACTIONS): the bundle builder
only ran load_prepared_bundle for shape validation — a rebuild "succeeded"
while scripts/diet_manifest.py still exited 2 on the stale bundle
(neg_aug_frac 0.2119 < 0.30 floor). The gate now runs inside
_build_local_training_bundles and raises SystemExit on a violated clause, so
a bundle that misses the diet floor fails loud instead of shipping.

TIER 1(e) rides along: prepared_bundle_drift_strict (required field of the
training config, config/training.yaml declares it; env
PREPARED_BUNDLE_DRIFT_STRICT wins) is a retired, inert policy record —
owner order 2026-10-07 retired the drift checks, so the loader records
masking/easy provenance but never compares it: a drifted bundle loads
with no warning and no hard failure under either polarity.
"""

from __future__ import annotations

import copy
import io
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from cli import colab
import core.common as core_common
from scripts.diet_manifest import main as diet_manifest_main
from training.prepared_bundle import (
    load_prepared_bundle,
    prepared_bundle_drift_strict,
    write_prepared_bundle,
)


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


class DietGateWiringTest(unittest.TestCase):
    def _fake_build(self):
        """Copy the prepared bundle through the --prepare-bundle argument."""

        def fake_build(command, **_kwargs):
            target = Path(command[command.index("--prepare-bundle") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self._bundle.read_bytes())
            target.with_suffix(target.suffix + ".json").write_text(
                self._bundle.with_suffix(self._bundle.suffix + ".json")
                .read_text(encoding="utf-8")
            )

        return fake_build

    def _drive(self, verdict: int):
        """Run the real builder with the gate stubbed to `verdict`.

        The workspace context stays open on the returned ExitStack so the
        caller can still read the produced bundle.
        """
        captured = io.StringIO()
        stack = ExitStack()
        workspace = stack.enter_context(tempfile.TemporaryDirectory())
        # Isolate the diet gate from the independently tested component split gate.
        stack.enter_context(mock.patch.object(colab, "_legacy_validation_sources"))
        stack.enter_context(mock.patch.object(colab, "_validate_legacy_bundle_partitions"))
        stack.enter_context(mock.patch.object(colab, "RESULTS", Path(workspace)))
        stack.enter_context(mock.patch.object(
            colab, "_validation_input_path", return_value=Path(__file__)
        ))
        stack.enter_context(mock.patch.object(
            colab.subprocess, "run", side_effect=self._fake_build()
        ))
        stack.enter_context(mock.patch.object(
            colab, "_run_diet_gate", return_value=verdict
        ))
        stack.enter_context(mock.patch("sys.stdout", captured))
        try:
            bundles = colab._build_local_training_bundles(
                profiles=["baseline"],
                model=None,
                sample=None,
            )
        except BaseException:
            stack.close()
            raise
        return stack, captured, bundles

    def test_rebuild_fails_loud_when_the_diet_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle = Path(tmp) / "failing.pkl.gz"
            _write_bundle(
                self._bundle,
                pos=np.asarray([[0, 1]], dtype=int),
                train_neg=np.empty((0, 2), dtype=int),
                hard_negative_mask_audit=[],
            )
            with self.assertRaisesRegex(SystemExit, "FAILED the diet gate"):
                self._drive(verdict=2)

    def test_rebuild_ships_a_gate_passing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle = Path(tmp) / "passing.pkl.gz"
            _write_bundle(self._bundle, **_passing_census())
            stack, captured, bundles = self._drive(verdict=0)
        try:
            self.assertEqual(len(bundles), 1)
            manifest, _ = load_prepared_bundle(bundles[0])
            self.assertEqual(manifest.n_payload, 6)
        finally:
            stack.close()


class PreparedBundleDriftStrictTest(unittest.TestCase):
    """TIER 1(e): retired policy record; drifted bundles load unconditionally."""

    def setUp(self) -> None:
        os.environ.pop("PREPARED_BUNDLE_DRIFT_STRICT", None)

    def tearDown(self) -> None:
        os.environ.pop("PREPARED_BUNDLE_DRIFT_STRICT", None)

    def _stale_bundle(self, tmp: str) -> Path:
        path = Path(tmp) / "stale.pkl.gz"
        _write_bundle(
            path,
            pos=np.asarray([[0, 1]], dtype=int),
            train_neg=np.empty((0, 2), dtype=int),
            hard_negative_mask_audit=[],
        )
        return path

    def _load_drifted(self, path: Path, *, lenient: bool) -> tuple[object, str]:
        """A bundle recorded under a drifted masking config.

        The drift signal rides the data config the loader used to compare
        against (load_config); the retired strict switch is still driven
        through the documented winning env knob, because its config value
        lives in the training SSOT (training_cfg().prepared_bundle_drift_strict),
        not in the data config this helper mocks."""
        real_config = core_common.load_config()
        drifted = copy.deepcopy(real_config)
        drifted["training"]["random_easy_negatives"]["ratio_to_hard"] = 9.99
        captured = io.StringIO()
        with mock.patch.object(core_common, "load_config", return_value=drifted), \
                mock.patch.dict(os.environ, {"PREPARED_BUNDLE_DRIFT_STRICT": "0" if lenient else "1"}):
            with redirect_stdout(captured):
                try:
                    return load_prepared_bundle(path), captured.getvalue()
                except ValueError as exc:
                    return None, str(exc)

    def test_resolves_the_training_config_value(self) -> None:
        """The knob is a required field of the training config
        (config/training.yaml declares it); the reader returns it verbatim
        for both polarities."""
        from types import SimpleNamespace

        for value in (False, True):
            with mock.patch.object(
                core_common, "training_cfg",
                return_value=SimpleNamespace(prepared_bundle_drift_strict=value),
            ):
                self.assertEqual(prepared_bundle_drift_strict(), value)

    def test_env_flip_is_the_winning_switch(self) -> None:
        os.environ["PREPARED_BUNDLE_DRIFT_STRICT"] = "1"
        self.assertTrue(prepared_bundle_drift_strict())
        os.environ["PREPARED_BUNDLE_DRIFT_STRICT"] = "0"
        self.assertFalse(prepared_bundle_drift_strict())

    def test_drifted_bundle_loads_without_drift_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._stale_bundle(tmp)
            result, warning = self._load_drifted(path, lenient=True)
        self.assertIsNotNone(result)
        self.assertNotIn("bundle-drift", warning)

    def test_strict_flip_no_longer_fails_the_same_bundle(self) -> None:
        os.environ["PREPARED_BUNDLE_DRIFT_STRICT"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._stale_bundle(tmp)
            result, strict_error = self._load_drifted(path, lenient=False)
        self.assertIsNotNone(result)
        self.assertNotIn("bundle-drift", strict_error)


if __name__ == "__main__":
    unittest.main()
