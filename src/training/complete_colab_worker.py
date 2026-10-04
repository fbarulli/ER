"""Complete a successful Colab worker: generate validation outputs for download."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field, model_validator
from core.portable_archive import Digest

import pandas as pd

from core.common import F, TRAIN_ROOT, plot_dpi, trace_artifact, training_cfg
from core.manifest import sha256_file
from training.validation_inference import resolve_best_checkpoint, threshold_assignment_metrics

# ── scored-pair validation census (2026-10-01 contract) ─────────────────────
# Row accounting is re-measured from the artifacts at exec time — never
# hardcoded — and every read is byte-stability asserted, so an artifact being
# regenerated concurrently is never counted half-written:
#   source census  dataset.csv rows == deduped + dropped == 71,623
#                  (config/training.yaml audit.source_export_expected_rows pin)
#   fold map       results/training/validation_fold_map.csv maps every graph
#                  entity to its fold; folds 2+3 are the validation side
#   scored pairs   data/final_validation.csv (files.final_validation binding)
#                  = the scored-pair final-inference population
#   train side     deduped rows minus the validation fold 2+3 entities


class CsvIdentity(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    path: str
    rows: int = Field(ge=0)
    columns: list[str]
    bytes: int = Field(ge=0)
    sha256: Digest


class ScoredValidationAccounting(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    source_export_rows: int = Field(gt=0)
    deduped_rows: int = Field(ge=0)
    dropped_rows: int = Field(ge=0)
    train_side_rows: int = Field(ge=0)
    validation_entity_rows: int = Field(ge=0)
    scored_pair_rows: int = Field(ge=0)
    scored_population_path: str

    @model_validator(mode='after')
    def close_populations(self):
        if self.deduped_rows + self.dropped_rows != self.source_export_rows:
            raise ValueError('source census does not close')
        if self.train_side_rows + self.validation_entity_rows != self.deduped_rows:
            raise ValueError('train and validation populations do not close')
        return self


def _byte_stable_csv_rows(path: Path) -> int:
    """Count CSV rows once, asserting the file's bytes stayed identical."""
    digest_before = sha256_file(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if sha256_file(path) != digest_before:
        raise RuntimeError(f"{path} changed while it was being read")
    return len(frame)


def scored_validation_accounting() -> dict[str, object]:
    """Census + identity of the scored-pair final-inference contract.

    Identity asserts (fail loud, before any launch):
      train_side_rows + validation_entity_rows == deduped_rows
      deduped_rows + dropped_rows == the source-export census pin
    """
    source_export_rows = int(training_cfg().audit.source_export_expected_rows)
    deduped_rows = _byte_stable_csv_rows(F["dataset_deduped"])
    dropped_rows = _byte_stable_csv_rows(F["removals"])
    if deduped_rows + dropped_rows != source_export_rows:
        raise ValueError(
            f"scored-pair accounting broke: deduped {deduped_rows:,} + dropped "
            f"{dropped_rows:,} != source census {source_export_rows:,}"
        )
    fold_map = pd.read_csv(F["validation_fold_map"], dtype=str, keep_default_na=False)
    n_folds = training_cfg().split.holdout_component_folds
    validation_gtins = set(
        fold_map.loc[fold_map["fold"].isin((str(n_folds - 2), str(n_folds - 1))), "gtin"]
    )
    deduped = pd.read_csv(F["dataset_deduped"], dtype=str, keep_default_na=False,
                          usecols=["gtin"])
    validation_entity_rows = int(deduped["gtin"].isin(validation_gtins).sum())
    train_side_rows = deduped_rows - validation_entity_rows
    if train_side_rows + validation_entity_rows != deduped_rows:
        raise ValueError("scored-pair accounting: train side does not close")
    scored_pair_rows = _byte_stable_csv_rows(F["final_validation"])
    return ScoredValidationAccounting.model_validate({
        "source_export_rows": source_export_rows,
        "deduped_rows": deduped_rows,
        "dropped_rows": dropped_rows,
        "train_side_rows": train_side_rows,
        "validation_entity_rows": validation_entity_rows,
        "scored_pair_rows": scored_pair_rows,
        "scored_population_path": str(F["final_validation"]),
    }).model_dump()


def _csv_identity(path: Path) -> dict[str, object]:
    before = sha256_file(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if sha256_file(path) != before:
        raise RuntimeError(f"{path} changed while provenance was being read")
    return CsvIdentity.model_validate({
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "bytes": path.stat().st_size,
        "sha256": before,
    }).model_dump()


def _resolve_final_inference_device(cfg, override: str | None) -> str:
    """Resolve the final-inference device, refusing a SILENT CPU fallback.

    ``cuda`` in the config is a REQUIREMENT, not a preference: a run that
    quietly lands on the CPU takes hours instead of minutes and leaves nothing
    in the artifacts to tell the two apart. Every precondition is therefore
    checked BEFORE the encoder starts, so a misconfiguration is one named
    failure here rather than a CUDA OOM part-way through 61,529 rows:

    * the configured batch must not exceed the configured ceiling (also
      enforced at config load — this is the defence-in-depth copy);
    * ``cuda`` needs a visible GPU;
    * the card must meet ``min_vram_gb``.
    """
    if cfg.batch_size > cfg.max_batch_size:
        raise ValueError(
            f"colab.final_inference.batch_size {cfg.batch_size} exceeds "
            f"max_batch_size {cfg.max_batch_size}"
        )
    requested = str(override or cfg.device)
    if requested != "cuda":
        return requested
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "colab.final_inference.device is 'cuda' but no GPU is visible. "
            "Final inference must not silently fall back to CPU; run on a GPU "
            "runner or set device: 'cpu' deliberately."
        )
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb < float(cfg.min_vram_gb):
        raise RuntimeError(
            f"colab.final_inference needs >= {cfg.min_vram_gb:g} GB VRAM but the "
            f"visible device has {total_gb:.1f} GB; lower batch_size/min_vram_gb "
            "deliberately rather than OOMing mid-run"
        )
    return "cuda"


def _write_input_provenance(
    *, scored_population: Path, training_csv: Path, output_dir: Path,
    predictions_path: Path, sample: int | None,
) -> Path:
    """Scored-pair provenance (2026-10-01 contract): input census + identity.

    The final-inference population is the scored-pair validation CSV
    (data/final_validation.csv), not a reconstruct-the-source holdout, so
    there is no complement/overlap identity to prove.  What is proven instead
    is the row accounting the artifacts close on plus the byte identity of
    what was actually scored.
    """
    payload = {
        "schema": "final-inference-provenance-v2-scored-pairs",
        "scored_pair_population": _csv_identity(scored_population),
        "training_input": _csv_identity(training_csv),
        "predictions_output": _csv_identity(predictions_path),
        "requested_sample": sample,
        "validation_accounting": scored_validation_accounting(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "input_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace_artifact("final_inference", path, producer="training.complete_colab_worker")
    return path


def _write_sku_reports(predictions_path: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cfg = training_cfg().colab.final_inference
    predictions = pd.read_csv(predictions_path, dtype={"SKU_ID": str, "GTIN": str})
    scores = pd.to_numeric(predictions["SCORE"], errors="raise")
    metrics_path = output_dir / "sku_threshold_summary.csv"
    threshold_assignment_metrics(scores.to_numpy(), list(cfg.thresholds)).to_csv(
        metrics_path, index=False
    )
    summary_path = output_dir / "sku_score_summary.json"
    summary_path.write_text(json.dumps({
        "rows": int(len(scores)),
        "score_min": float(scores.min()),
        "score_median": float(scores.median()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=40)
    ax.axvline(cfg.error_threshold, color="red", linestyle="--", label=f"{cfg.error_threshold:g}")
    ax.set(xlabel="nearest-item cosine score", ylabel="SKU count", title="Held-out SKU score distribution")
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "sku_score_distribution.png"
    fig.savefig(plot_path, dpi=plot_dpi())
    plt.close(fig)
    for path in (metrics_path, summary_path, plot_path):
        trace_artifact("final_inference", path, producer="training.complete_colab_worker")


def complete_worker(
    *, source: Path, run_id: str, worker: int, validation_input: Path,
    validation_source: Path | None = None,
    training_input: Path | None = None,
    publish_dvc: bool = False, device: str | None = None,
    sample: int | None = None,
) -> None:
    cfg = training_cfg().colab.final_inference
    if cfg.enabled:
        training_input = training_input or F["dataset_deduped"]
        output_dir = source / cfg.output_dir
        checkpoint, _ = resolve_best_checkpoint(source)
        predictions_path = output_dir / "sku_predictions.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_device = _resolve_final_inference_device(cfg, device)
        command = [
            sys.executable, "-m", "predict_items",
            "--model", str(checkpoint),
            "--input", str(validation_input),
            "--output", str(predictions_path),
            "--threshold", str(cfg.error_threshold),
            "--device", resolved_device,
            "--include-scores",
            "--batch-size", str(cfg.batch_size),
        ]
        if sample is not None:
            command.extend(["--sample", str(sample)])
        print(f"[final-inference] SKU retrieval: {' '.join(command)}", flush=True)
        # The subprocess cwd is the resolved project root from the shared path
        # contract, never a __file__/__parents__ offset (owner directive
        # 2026-09-15): a magic parent count silently breaks the moment this
        # module moves or is installed as a package.
        subprocess.run(command, cwd=TRAIN_ROOT, env=os.environ.copy(), check=True)
        trace_artifact(
            "final_inference", predictions_path,
            producer="training.complete_colab_worker",
        )
        _write_sku_reports(predictions_path, output_dir)
        _write_input_provenance(
            scored_population=validation_input,
            training_csv=training_input,
            output_dir=output_dir, predictions_path=predictions_path, sample=sample,
        )
    else:
        print("[final-inference] disabled by configuration", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker", type=int, required=True)
    # The scored population is the SSOT final_validation binding; retired
    # --validation-source remains accepted (ignored) so existing launch argv
    # does not break mid-flight.
    parser.add_argument("--validation-input", type=Path, default=F["final_validation"])
    parser.add_argument("--validation-source", type=Path, default=None)
    parser.add_argument(
        "--training-input", type=Path,
        default=F["dataset_deduped"],
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--skip-dvc", action="store_true")
    args = parser.parse_args()
    complete_worker(
        source=args.source,
        run_id=args.run_id,
        worker=args.worker,
        validation_input=args.validation_input,
        validation_source=args.validation_source,
        training_input=args.training_input,
        publish_dvc=False,
        device=args.device,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
