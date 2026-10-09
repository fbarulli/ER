"""Laya lane local (CPU) held-out eval of a fetched fine-tuned checkpoint.

The offline twin of the ``finetune-eval`` kernel: it loads the checkpoint with
``laya.train.load_checkpoint`` on CPU and runs ``calibration_records`` +
``evaluate_records`` on the corpus split, writing the same ``eval_report.json``
(+ receipt). No network, no training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cli.laya_recipe import (
    FINETUNE_EVAL_DECISION,
    FINETUNE_EVAL_RECEIPT_FILE,
    FINETUNE_EVAL_REPORT_FILE,
    FINETUNE_EVAL_SPLIT_FILES,
    LayaRecipeFactory,
)
from cli.laya_runtime import LayaRuntimeFactory
from core.manifest import atomic_write_json, sha256_file


class LayaLocalEvalRunner:
    """CPU held-out eval for ONE resolved spec + staging runtime."""

    def __init__(self, runtime: LayaRuntimeFactory,
                 recipe: LayaRecipeFactory):
        self._runtime = runtime
        self._recipe = recipe

    @staticmethod
    def fit_eval_calibration(laya_train: Any, records, calibration: dict) -> dict:
        """Fit laya's OWN calibration for the held-out eval path (SSOT selection).

        The CPU ``--local-eval`` twin of the ``finetune-eval`` kernel's
        ``fit_eval_calibration`` (the kernel is a staged string and cannot import
        this module — keep the two in lockstep). Consumes laya's
        ``fit_temperature_map`` / ``fit_abstention_thresholds``, never
        reimplementing either; the defaults reproduce the landed eval exactly.
        """
        level = calibration or {}
        temperature = temperature_by_options = n_by_bucket = None
        if level.get("temperature", True):
            fitted = laya_train.fit_temperature_map(records)
            temperature = fitted.get("temperature")
            temperature_by_options = fitted.get("temperature_by_options")
            n_by_bucket = fitted.get("n_by_bucket")
        thresholds: dict[str, float] = {}
        if level.get("abstention"):
            thresholds = dict(laya_train.fit_abstention_thresholds(
                records, temperature, temperature_by_options or {},
                target_error=level.get("target_error", 0.10),
                min_bucket_n=level.get("min_abstain_n", 10)) or {})
        min_confidence = level.get("min_confidence")
        if min_confidence is not None:
            thresholds["default"] = min_confidence
        return {
            "temperature": temperature,
            "temperature_by_options": temperature_by_options,
            "n_by_bucket": n_by_bucket,
            "abstention_thresholds": thresholds,
            "min_confidence": min_confidence,
        }

    def local_eval_checkpoint(self, checkpoint_dir: Path, *,
                              eval_data: Path | None = None,
                              out_dir: Path | None = None,
                              split: str | None = None,
                              batch_size: int | None = None,
                              limit: int | None = None) -> dict[str, Any]:
        """Local (CPU) held-out eval of a fetched fine-tuned checkpoint."""
        spec = self._runtime.spec
        split = split or spec.finetune_eval_split
        if split not in FINETUNE_EVAL_SPLIT_FILES:
            raise ValueError(
                f"eval split {split!r} is not one of "
                f"{sorted(FINETUNE_EVAL_SPLIT_FILES)}")
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "rl_agent_config.json").is_file():
            raise FileNotFoundError(
                f"checkpoint {checkpoint_dir} carries no rl_agent_config.json")
        if eval_data is None:
            eval_data = (self._runtime.train_root / spec.finetune_corpus_dir
                         / FINETUNE_EVAL_SPLIT_FILES[split])
        eval_data = Path(eval_data)
        if not eval_data.is_file():
            raise FileNotFoundError(f"eval data not found: {eval_data}")
        if out_dir is None:
            out_dir = self._runtime.staging_dir() / "local_eval"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        batch_size = batch_size or spec.finetune_eval_batch_size
        try:
            import torch
            from laya import train as laya_train
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "local eval needs the laya package + torch installed on this "
                f"box (pip install {spec.laya_package}): {error}") from error
        device = torch.device("cpu")
        model, tok, cfg = laya_train.load_checkpoint(str(checkpoint_dir))
        model = model.to(device).eval()
        max_len = int(cfg.get("max_len", 512))
        head_max_len = int(cfg.get("head_max_len", 192))
        parallel = laya_train.uses_parallel_layout(cfg)
        rows = laya_train.read_jsonl(str(eval_data))
        if limit:
            rows = rows[:limit]
        items, skipped = laya_train.items_from_rows(
            tok, rows, max_len, head_max_len, label_smoothing=0.0)
        if not items:
            raise RuntimeError(
                f"eval data {eval_data} produced no usable items "
                f"(skipped: {skipped!r})")
        records = laya_train.calibration_records(
            model, tok, items, device, max_len, head_max_len,
            batch_size=batch_size, parallel=parallel)
        before = laya_train.evaluate_records(records)
        calibration = self._recipe.eval_calibration_config()
        fitted = self.fit_eval_calibration(laya_train, records, calibration)
        after = laya_train.evaluate_records(
            records, fitted["temperature"], fitted["temperature_by_options"])
        report = {
            "eval_mode": "held_out",
            "is_held_out": True,
            "device": "cpu",
            "eval_source": eval_data.name,
            "eval_split": split,
            "rows": len(rows),
            "items": len(items),
            "skipped": skipped,
            "checkpoint": str(checkpoint_dir),
            "before": before,
            "after": after,
            "temperature": fitted["temperature"],
            "temperature_by_options": fitted["temperature_by_options"],
        }
        # Additive: the opt-in knobs alone add keys, so a default config keeps
        # the landed report shape byte-for-byte (mirrors the eval kernel).
        if calibration.get("abstention") or fitted["min_confidence"] is not None:
            report["n_by_bucket"] = fitted["n_by_bucket"] or {}
        if fitted["abstention_thresholds"]:
            report["abstention_thresholds"] = fitted["abstention_thresholds"]
        if fitted["min_confidence"] is not None:
            report["min_confidence"] = fitted["min_confidence"]
        atomic_write_json(report, out_dir / FINETUNE_EVAL_REPORT_FILE)
        receipt = {
            "gpu_kind": FINETUNE_EVAL_DECISION,
            "device": "cpu",
            "eval_split": split,
            "eval_mode": "held_out",
            "is_held_out": True,
            "eval_data": str(eval_data),
            "eval_data_sha256": sha256_file(eval_data),
            "checkpoint": str(checkpoint_dir),
            "eval_calibration": calibration,
            "report": str(out_dir / FINETUNE_EVAL_REPORT_FILE),
        }
        atomic_write_json(receipt, out_dir / FINETUNE_EVAL_RECEIPT_FILE)
        self._runtime.log_lane(
            f"local cpu eval [{split}] items={len(items)} "
            f"accuracy={after['accuracy']} -> {out_dir}")
        return report
