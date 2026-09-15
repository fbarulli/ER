"""Complete a successful Colab worker: validate, then publish through DVC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from core.common import TRAIN_ROOT, trace_artifact, training_cfg
from training.validation_inference import resolve_best_checkpoint, threshold_assignment_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_identity(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


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
    *, source_csv: Path, training_csv: Path, sample_csv: Path, output_dir: Path,
) -> Path:
    source_ids = set(pd.read_csv(
        source_csv, usecols=["product_id"], dtype=str, keep_default_na=False
    )["product_id"])
    training_ids = pd.read_csv(
        training_csv, usecols=["product_id"], dtype=str, keep_default_na=False
    )["product_id"]
    sample_ids = pd.read_csv(
        sample_csv, usecols=["product_id"], dtype=str, keep_default_na=False
    )["product_id"]
    training_id_set = set(training_ids)
    sample_id_set = set(sample_ids)
    full_inference = sample_id_set == source_ids
    overlap = sorted(training_id_set & sample_id_set)
    reconstructed = training_id_set | sample_id_set
    if full_inference:
        if not training_id_set <= source_ids:
            raise ValueError("training input contains product IDs absent from the deduped source")
    else:
        if overlap:
            raise ValueError(
                f"training complement overlaps validation sample on {len(overlap)} product IDs"
            )
        if reconstructed != source_ids:
            raise ValueError(
                "training complement plus validation sample does not reconstruct the deduped source: "
                f"missing={len(source_ids - reconstructed)} extra={len(reconstructed - source_ids)}"
            )
    payload = {
        "schema": "validation-input-provenance-v1",
        "deduped_source": _csv_identity(source_csv),
        "training_complement": _csv_identity(training_csv),
        "sku_sample": _csv_identity(sample_csv),
        "training_validation_product_id_overlap": len(overlap),
        "complement_reconstructs_source": not full_inference,
        "full_deduped_inference": full_inference,
        "sku_sample_unique_product_ids": int(sample_ids.nunique()),
        "sku_sample_ids_present_in_source": int(len(sample_ids)),
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
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    for path in (metrics_path, summary_path, plot_path):
        trace_artifact("final_inference", path, producer="training.complete_colab_worker")


def complete_worker(
    *, source: Path, run_id: str, worker: int, validation_input: Path,
    validation_source: Path | None = None,
    training_input: Path | None = None,
    publish_dvc: bool = True, device: str | None = None,
) -> None:
    cfg = training_cfg().colab.final_inference
    if cfg.enabled:
        validation_source = validation_source or Path(cfg.source_csv)
        training_input = training_input or Path(training_cfg().colab.training_dataset_csv)
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
            source_csv=validation_source,
            training_csv=training_input,
            sample_csv=validation_input,
            output_dir=output_dir,
        )
    else:
        print("[final-inference] disabled by configuration", flush=True)
    if publish_dvc:
        from training.dvc_store import publish

        publish(source, run_id, worker)


def main() -> None:
    cfg = training_cfg().colab.final_inference
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--validation-input", type=Path, default=Path(cfg.input_csv))
    parser.add_argument("--validation-source", type=Path, default=Path(cfg.source_csv))
    parser.add_argument(
        "--training-input", type=Path,
        default=Path(training_cfg().colab.training_dataset_csv),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--skip-dvc", action="store_true")
    args = parser.parse_args()
    complete_worker(
        source=args.source,
        run_id=args.run_id,
        worker=args.worker,
        validation_input=args.validation_input,
        validation_source=args.validation_source,
        training_input=args.training_input,
        publish_dvc=not args.skip_dvc,
        device=args.device,
    )


if __name__ == "__main__":
    main()
