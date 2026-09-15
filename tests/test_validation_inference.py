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
                # The shipped config REQUIRES cuda for final inference (a
                # silent CPU fallback is refused). This test only exercises the
                # completion ordering and runs on a CPU box, so it asks for CPU
                # explicitly through the documented override -- which is the
                # whole point of the override existing.
                device="cpu",
            )
        self.assertEqual(events, ["sku", "reports", "provenance", "dvc"])


class FinalInferenceContractTests(unittest.TestCase):
    """The renamed config key, the GPU requirement, and the batch ceiling."""

    def test_config_key_is_final_inference_and_the_old_one_is_gone(self):
        """A half-renamed key is worse than no rename: the old name must be absent."""
        from core.common import training_cfg

        colab = training_cfg().colab
        self.assertTrue(hasattr(colab, "final_inference"))
        self.assertFalse(
            hasattr(colab, "validation_inference"),
            "the old config key must not survive the rename",
        )
        spec = colab.final_inference
        # The name was wrong: this scores the WHOLE catalog, not a sample.
        self.assertEqual(spec.input_csv, "training_data/dataset_deduped.csv")
        self.assertEqual(spec.source_csv, "training_data/dataset_deduped.csv")
        self.assertEqual(spec.output_dir, "final_inference")

    def test_final_inference_batch_is_sized_for_inference_not_finetuning(self):
        """Inference has no gradients or optimiser state, so it batches bigger."""
        from core.common import training_cfg

        spec = training_cfg().colab.final_inference
        finetune_cuda = training_cfg().training.batch_size_cuda
        self.assertGreater(spec.batch_size, finetune_cuda)
        self.assertLessEqual(spec.batch_size, spec.max_batch_size)
        # 512 for a 12-GB card: 8x the finetune batch on a 14.6-GB T4.
        self.assertEqual(spec.batch_size, 512)
        self.assertEqual(spec.max_batch_size, 1024)

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


if __name__ == "__main__":
    unittest.main()
