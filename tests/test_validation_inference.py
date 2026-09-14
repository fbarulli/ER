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
                training_input=Path("training.csv"), publish_dvc=True,
            )
        self.assertEqual(events, ["sku", "reports", "provenance", "dvc"])


if __name__ == "__main__":
    unittest.main()
