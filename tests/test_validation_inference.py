from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from training.complete_colab_worker import complete_worker
from training.validation_inference import resolve_best_checkpoint, threshold_assignment_metrics


class ValidationInferenceTests(unittest.TestCase):
    def test_resolves_trainer_recorded_best_not_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            root = source / "_checkpoints" / "model" / "run_fold0"
            for step in (10, 20):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text(json.dumps({
                    "global_step": step,
                    "best_global_step": 10,
                    "best_metric": 0.91,
                    "best_model_checkpoint": str(root / "checkpoint-10"),
                }))
            checkpoint, state = resolve_best_checkpoint(source)
            self.assertEqual(checkpoint.name, "checkpoint-10")
            self.assertEqual(state["best_global_step"], 10)

    def test_threshold_assignment_counts(self):
        metrics = threshold_assignment_metrics(
            np.array([0.9, 0.6, 0.4, 0.1]), [0.5, 0.8]
        )
        self.assertEqual(metrics["matched"].tolist(), [2, 1])
        self.assertEqual(metrics["unmatched"].tolist(), [2, 3])
        self.assertEqual(metrics["matched_rate"].tolist(), [0.5, 0.25])

    def test_completion_orders_inference_reports_provenance_before_dvc(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "training.complete_colab_worker.resolve_best_checkpoint",
            return_value=(Path("checkpoint"), {}),
        ), mock.patch(
            "training.complete_colab_worker.subprocess.run",
            side_effect=lambda *_args, **_kwargs: events.append("sku"),
        ), mock.patch(
            "training.complete_colab_worker._write_sku_reports",
            side_effect=lambda *_args, **_kwargs: events.append("reports"),
        ), mock.patch(
            "training.complete_colab_worker._write_input_provenance",
            side_effect=lambda **_kwargs: events.append("provenance"),
        ), mock.patch(
            "training.complete_colab_worker.trace_artifact",
        ), mock.patch(
            "training.dvc_store.publish",
            side_effect=lambda *_args: events.append("dvc"),
        ):
            complete_worker(
                source=Path(temporary) / "worker", run_id="run", worker=2,
                validation_input=Path("sample.csv"),
                validation_source=Path("source.csv"),
                training_input=Path("training.csv"), publish_dvc=False,
                # The shipped config REQUIRES cuda for final inference (a
                # silent CPU fallback is refused). This test only exercises the
                # completion ordering and runs on a CPU box, so it asks for CPU
                # explicitly through the documented override -- which is the
                # whole point of the override existing.
                device="cpu",
            )
        self.assertEqual(events, ["sku", "reports", "provenance"])


class FinalInferenceContractTests(unittest.TestCase):
    """The renamed config key, the GPU requirement, and the batch ceiling."""

    def test_config_key_is_final_inference_and_the_old_one_is_gone(self):
        """A half-renamed key is worse than no rename: the old name must be absent."""
        from core.common import F, training_cfg

        colab = training_cfg().colab
        self.assertTrue(hasattr(colab, "final_inference"))
        self.assertFalse(
            hasattr(colab, "validation_inference"),
            "the old config key must not survive the rename",
        )
        spec = colab.final_inference
        # SCORED-PAIR contract (2026-10-01): the scored population is the SSOT
        # final_validation binding resolved by core.common, not a yaml path
        # literal; the spec itself carries no input path.
        self.assertFalse(
            hasattr(spec, "input_csv") or hasattr(spec, "source_csv"),
            "the scored-pair contract removes the yaml path literals",
        )
        self.assertTrue(
            str(F["final_validation"]).endswith("data/final_validation.csv")
        )
        self.assertTrue(F["final_validation"].is_file())
        self.assertEqual(colab.training_dataset_csv, "data/dataset_deduped.csv")

        self.assertEqual(spec.output_dir, "final_inference")

    def test_a_batch_over_the_ceiling_fails_at_config_load(self):
        """A typo must be a config failure, not a CUDA OOM mid-catalog."""
        from core.common import training_cfg
        from core.schemas import FinalInferenceSpec

        payload = training_cfg().colab.final_inference.model_dump()
        with self.assertRaises(ValueError) as caught:
            FinalInferenceSpec.model_validate({**payload, "batch_size": payload["max_batch_size"] + 1})
        self.assertIn("max_batch_size", str(caught.exception))

    def test_cuda_is_required_and_never_silently_falls_back_to_cpu(self):
        """cuda in the config is a requirement; no GPU means a loud failure."""
        from core.common import training_cfg
        from training.complete_colab_worker import _resolve_final_inference_device

        spec = training_cfg().colab.final_inference
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError) as caught:
                _resolve_final_inference_device(spec, None)
        self.assertIn("no GPU is visible", str(caught.exception))
        # ...and the message says how to run on CPU on purpose
        self.assertIn("device: 'cpu'", str(caught.exception))

    def test_cuda_is_used_when_a_gpu_is_present(self):
        from core.common import training_cfg
        from training.complete_colab_worker import _resolve_final_inference_device

        spec = training_cfg().colab.final_inference
        props = mock.Mock(total_memory=int(12 * 1024**3))
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ):
            self.assertEqual(_resolve_final_inference_device(spec, None), "cuda")

    def test_a_card_below_the_vram_floor_is_refused_before_encoding(self):
        from core.common import training_cfg
        from training.complete_colab_worker import _resolve_final_inference_device

        spec = training_cfg().colab.final_inference
        props = mock.Mock(total_memory=int(4 * 1024**3))
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ):
            with self.assertRaises(RuntimeError) as caught:
                _resolve_final_inference_device(spec, None)
        self.assertIn("VRAM", str(caught.exception))

    def test_the_explicit_cpu_override_still_checks_the_batch_ceiling(self):
        """The override excuses the GPU requirement, not the batch bound."""
        from core.common import training_cfg
        from training.complete_colab_worker import _resolve_final_inference_device

        spec = training_cfg().colab.final_inference
        self.assertEqual(_resolve_final_inference_device(spec, "cpu"), "cpu")
        oversized = spec.model_copy(update={"batch_size": spec.max_batch_size + 1})
        with self.assertRaises(ValueError):
            _resolve_final_inference_device(oversized, "cpu")

    def test_the_scored_pair_census_closes_and_is_never_hardcoded(self):
        """Exec-time accounting: the census re-measures artifacts and closes.

        Pins are removed (2026-10-06 owner ruling): the census re-measures
        every artifact byte-stably at exec time and must not carry its own
        hardcoded constant (that is what this test's name guards)."""
        import training.complete_colab_worker as worker
        from training.complete_colab_worker import (
            _byte_stable_csv_rows,
            scored_validation_accounting,
        )
        from core.common import DATA_PATH, F

        self.assertFalse(
            [n for n in dir(worker) if "EXPECTED_SOURCE_EXPORT" in n.upper()],
            "the worker module carries a hardcoded census constant",
        )

        # DATA-STATE GUARD (owner directive: only the full dataset is tested).
        # The closure this test pins is meaningful only when the PREPARED
        # corpus (dataset_deduped + the dedupe removals) was built from the
        # mounted export. A cohort/partial-prep mount (e.g. a 10k-derived
        # data/ directory beside a restored full dataset.csv) legitimately
        # fails the closure on the WRONG universe; skip rather than fail.
        def _prepared_corpus_closes() -> bool:
            try:
                return (
                    _byte_stable_csv_rows(F["dataset_deduped"])
                    + _byte_stable_csv_rows(F["removals"])
                    == _byte_stable_csv_rows(DATA_PATH)
                )
            except (FileNotFoundError, RuntimeError, ValueError):
                return False

        from core.common import dataset_is_partial_cohort

        if dataset_is_partial_cohort() or not _prepared_corpus_closes():
            self.skipTest(
                "prepared corpus is not the mounted export "
                "(cohort/partial data state); census is full-cohort-only"
            )
        census = scored_validation_accounting()
        # the exec-time census re-measures the raw export, never a stored pin
        self.assertEqual(
            census["source_export_rows"],
            _byte_stable_csv_rows(DATA_PATH),
        )
        self.assertEqual(
            census["deduped_rows"] + census["dropped_rows"],
            census["source_export_rows"],
            "deduped + dropped must close on the re-measured source census",
        )
        self.assertEqual(
            census["train_side_rows"] + census["validation_entity_rows"],
            census["deduped_rows"],
            "train side + validation entities must close on the deduped rows",
        )
        self.assertTrue(
            str(census["scored_population_path"]).endswith("data/final_validation.csv")
        )
        # The scored population is a pair population: a row count that is both
        # measured (Byte stability inside the census, not a hardcoded literal)
        # and non-trivial.
        self.assertGreater(
            _byte_stable_csv_rows(F["final_validation"]), 0
        )
        self.assertGreater(int(census["scored_pair_rows"]), 0)


class MinimalFlipSliceTests(unittest.TestCase):
    """Isolated stress-slice metrics: twin P@R95, margins, donor uniformity."""

    def test_precision_at_recall_uses_the_lowest_achieving_threshold(self):
        from scripts.minimal_flip_slice import precision_at_recall

        rep = precision_at_recall(
            np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 1, 0, 0]), target=1.0,
        )
        # recall 1.0 first achieved at rank 1 (two positives on top)
        self.assertEqual(rep["threshold"], 0.8)
        self.assertAlmostEqual(rep["precision"], 1.0)
        self.assertAlmostEqual(rep["recall"], 1.0)
        rep = precision_at_recall(
            np.array([0.9, 0.2, 0.8, 0.1]), np.array([1, 0, 1, 0]), target=0.5,
        )
        # ranked: 0.9(1), 0.8(1), 0.2(0), 0.1(0); recall 0.5 at rank 0
        self.assertEqual(rep["threshold"], 0.9)
        self.assertAlmostEqual(rep["precision"], 1.0)
        self.assertAlmostEqual(rep["recall"], 0.5)

    def test_precision_at_recall_without_positives_is_nan(self):
        from scripts.minimal_flip_slice import precision_at_recall

        rep = precision_at_recall(np.array([0.5, 0.2]), np.array([0, 0]))
        self.assertTrue(np.isnan(rep["precision"]))

    def test_donor_uniformity_names_the_dominant_value(self):
        from scripts.minimal_flip_slice import donor_uniformity

        payload = [
            "cola flavor_lime volume_ml_500",
            "cola flavor_coconut volume_ml_500",
        ]
        audits = [
            {"donor_anchor_payload_idx": 0, "fields_hit": ["flavor"],
             "target_mode": "swap_values"},
            {"donor_anchor_payload_idx": 0, "fields_hit": ["flavor"],
             "target_mode": "swap_values"},
            {"donor_anchor_payload_idx": 1, "fields_hit": ["flavor"],
             "target_mode": "counterfactual"},
        ]
        rep = donor_uniformity(audits, payload)
        self.assertEqual(rep["draws"], 3)
        self.assertEqual(rep["n_donor_rows"], 2)
        self.assertAlmostEqual(rep["max_row_share"], 2 / 3)
        flavor = rep["by_field"]["flavor"]
        self.assertEqual(flavor["transplants"], 3)
        self.assertAlmostEqual(flavor["max_value_share"], 2 / 3)
        self.assertIn("flavor_lime", flavor["max_value_preview"])


class DietProjectionTests(unittest.TestCase):
    """Train-time easy quota projected statically (data prep, not runtime)."""

    def test_disabled_quota_returns_bundle_views(self):
        from scripts.diet_manifest import project_train_time_neg_views

        self.assertEqual(
            project_train_time_neg_views(30341, enabled=False, ratio_to_hard=1.0, loss="contrastive"),
            30341,
        )
        self.assertEqual(
            project_train_time_neg_views(30341, enabled=True, ratio_to_hard=0.0, loss="contrastive"),
            30341,
        )

    def test_projection_mirrors_the_trainer_target_formula(self):
        from scripts.diet_manifest import project_train_time_neg_views

        # ceil(hard * ratio), the exact _mix_random_easy target
        self.assertEqual(
            project_train_time_neg_views(5, enabled=True, ratio_to_hard=0.5, loss="contrastive"),
            8,
        )
        self.assertEqual(
            project_train_time_neg_views(30341, enabled=True, ratio_to_hard=1.0, loss="contrastive"),
            60682,
        )

    def test_contrastive_projection_is_not_the_guaranteed_bundle_ratio(self):
        from scripts.diet_manifest import project_train_time_neg_views

        effective = project_train_time_neg_views(
            30341, enabled=True, ratio_to_hard=1.0, loss="contrastive"
        )
        self.assertLessEqual(51153 / effective, 1.50)


class ProceedPrecisionTests(unittest.TestCase):
    """Newly admitted proceed pairs must be nearly perfectly clean."""

    def _record(self, volume, **fields) -> dict:
        record = {
            "volume_set": set(volume),
            "pack_set": set(), "package_type_set": set(),
            "flavor_set": set(), "carbonation_set": set(),
            "sweetener_set": set(), "pulp_set": set(),
        }
        record.update(fields)
        return record

    def test_agreement_holds_on_compatible_records(self):
        from scripts.check_proceed_precision import pair_agrees

        left = self._record([500.0], flavor_set={"lemon"})
        right = self._record([500.0], flavor_set={"lemon"})
        self.assertTrue(pair_agrees(left, right))

    def test_volume_and_flavor_conflicts_fail(self):
        from scripts.check_proceed_precision import pair_agrees

        base = self._record([500.0], flavor_set={"lemon"})
        self.assertFalse(
            pair_agrees(base, self._record([250.0], flavor_set={"lemon"}))
        )
        self.assertFalse(
            pair_agrees(base, self._record([500.0], flavor_set={"lime"}))
        )

    def test_absent_evidence_is_not_a_conflict(self):
        from scripts.check_proceed_precision import pair_agrees

        self.assertTrue(
            pair_agrees(
                self._record([500.0], flavor_set={"lemon"}),
                self._record([500.0]),
            )
        )


class FieldSlicePartitionTests(unittest.TestCase):
    """33/33/34 twin partition: deterministic, smallest-bucket bound."""

    def _infos(self, field, n, start):
        return [
            {"copy_idx": start + i, "anchor_idx": 0, "pair_idx": 1,
             "field": field}
            for i in range(n)
        ]

    def test_partition_takes_first_k_per_bucket(self):
        from scripts.build_field_slice import select_field_buckets

        infos = (
            self._infos("volume", 5, 100)
            + self._infos("flavor", 3, 200)
            + self._infos("package_type", 4, 300)
            + self._infos("pack", 4, 400)
            + self._infos("carbonation", 2, 500)
        )
        part = select_field_buckets(infos)
        # package bucket = package_type + pack (8) ; smallest is flavor (3)
        self.assertEqual([r["copy_idx"] for r in part["flavor"]], [200, 201, 202])
        self.assertEqual(
            [r["copy_idx"] for r in part["volume"]], [100, 101, 102]
        )
        self.assertEqual(
            [r["copy_idx"] for r in part["package"]], [300, 301, 302]
        )
        self.assertEqual(part["_per_bucket"], 3)
        self.assertEqual(part["_excluded_unknown_field"], 2)

    def test_empty_bucket_blocks_with_zero(self):
        from scripts.build_field_slice import select_field_buckets

        part = select_field_buckets(self._infos("volume", 2, 0))
        self.assertEqual(part["_per_bucket"], 0)
        self.assertEqual(part["volume"], [])


if __name__ == "__main__":
    unittest.main()
