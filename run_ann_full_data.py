#!/usr/bin/env python3
"""Launch the pinned ANN training run with fail-fast runtime checks.

This is intentionally separate from the older launcher and downloader.  The
launcher tears the GPU VM down on every outcome: a retained GPU VM consumes
accelerator quota indefinitely, so keep-alive is CPU-only and is deliberately
NOT requested here.  Artifacts are persisted through the DVC remote before
teardown; use manual_download_results.py only for a CPU session that was
deliberately kept alive.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from core.common import training_cfg


ROOT = Path(__file__).resolve().parent
TRAINING_DATA = ROOT / "training_data/dataset_deduped_train_minus_3000.csv"
INFERENCE_DATA = ROOT / "training_data/dataset_deduped_sample_3000.csv"
MODEL_DIR = ROOT / "artifacts/models/all-MiniLM-L6-v2"
EXPECTED_MODEL_BYTES = 91_630_836

EXPECTED_TRAIN_ROWS = 58_529
EXPECTED_INFERENCE_ROWS = 3_000


def data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def check_settings() -> None:
    cfg = training_cfg()
    checks = {
        "training dataset": TRAINING_DATA.is_file(),
        "inference dataset": INFERENCE_DATA.is_file(),
        "model directory": MODEL_DIR.is_dir(),
        "model bundle size": MODEL_DIR.is_dir()
        and sum(path.stat().st_size for path in MODEL_DIR.rglob("*") if path.is_file())
        == EXPECTED_MODEL_BYTES,
        "model key": cfg.training.base_model == "minilm_l6",
        "training input config": cfg.colab.training_dataset_csv
        == "training_data/dataset_deduped_train_minus_3000.csv",
        "inference input config": cfg.colab.final_inference.input_csv
        == "training_data/dataset_deduped_sample_3000.csv",
        "inference source config": cfg.colab.final_inference.source_csv
        == "training_data/dataset_deduped.csv",
        "DVC disabled": cfg.colab.dvc_enabled is False,
        "HPO DVC persistence disabled": cfg.hpo.persistence == "none",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("setting check failed: " + ", ".join(failed))

    actual_train = data_rows(TRAINING_DATA)
    actual_inference = data_rows(INFERENCE_DATA)
    if actual_train != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"training row count changed: expected {EXPECTED_TRAIN_ROWS:,}, "
            f"got {actual_train:,}"
        )
    if actual_inference != EXPECTED_INFERENCE_ROWS:
        raise RuntimeError(
            f"inference row count changed: expected {EXPECTED_INFERENCE_ROWS:,}, "
            f"got {actual_inference:,}"
        )

    print("[preflight] settings verified:", flush=True)
    print(f"  train={TRAINING_DATA} rows={actual_train:,}", flush=True)
    print(f"  inference={INFERENCE_DATA} rows={actual_inference:,}", flush=True)
    print(f"  model=minilm_l6 path={MODEL_DIR}", flush=True)
    print("  gpu=T4 workers=1 train_frac=1.0 epochs=10 keep_alive=false", flush=True)
    print("  dvc=false", flush=True)


def main() -> int:
    check_settings()
    command = [
        sys.executable,
        str(ROOT / "colab_backend.py"),
        "--what",
        "train",
        "--train-frac",
        "1.0",
        "--workers",
        "1",
        "--model",
        "minilm_l6",
        "--gpu",
        "T4",
        "--allow-gpu",
    ]
    print("[launch] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
