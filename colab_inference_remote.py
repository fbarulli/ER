#!/usr/bin/env python3
"""Bootstrap pushed inference code inside a fresh Colab session."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile


REMOTE_ROOT = Path(os.environ.get("REMOTE_ROOT", "/content/EuromonitoR"))
CHECKPOINT_ARCHIVE = Path("/content/checkpoint-114-inference.tar.gz")


def run(command: list[str], *, cwd: Path | None = None, env=None) -> None:
    print(f"[remote] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def extract_checkpoint() -> Path:
    if not CHECKPOINT_ARCHIVE.is_file():
        raise FileNotFoundError(f"checkpoint upload missing: {CHECKPOINT_ARCHIVE}")
    with tarfile.open(CHECKPOINT_ARCHIVE, "r:gz") as archive:
        archive.extractall(REMOTE_ROOT, filter="data")
    checkpoint = REMOTE_ROOT / "checkpoint-114"
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"invalid checkpoint archive: {checkpoint}")
    return checkpoint


def main() -> None:
    import torch

    cuda = torch.cuda.is_available()
    requested_gpu = os.environ.get("REQUESTED_GPU", "CPU")
    if requested_gpu != "CPU" and not cuda:
        raise SystemExit(f"{requested_gpu} requested but CUDA is unavailable")
    device = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"[remote] device={device}", flush=True)

    repo_url = os.environ["REPO_URL"]
    branch = os.environ["REPO_BRANCH"]
    commit = os.environ["REPO_COMMIT"]
    if REMOTE_ROOT.exists():
        raise RuntimeError(f"remote checkout path already exists: {REMOTE_ROOT}")
    run(
        ["git", "clone", "--branch", branch, "--single-branch", repo_url, str(REMOTE_ROOT)]
    )
    run(["git", "checkout", "--detach", commit], cwd=REMOTE_ROOT)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REMOTE_ROOT, text=True
    ).strip()
    if observed != commit:
        raise RuntimeError(f"checkout mismatch: expected {commit}, got {observed}")

    checkpoint = extract_checkpoint()
    packages = [
        "sentence-transformers",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "pydantic",
        "pyyaml",
    ]
    bundled_wheel = (
        REMOTE_ROOT / "artifacts/wheels/hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl"
    )
    if (
        sys.version_info[:2] == (3, 13)
        and platform.machine() == "x86_64"
        and bundled_wheel.is_file()
    ):
        packages.append(str(bundled_wheel))
    else:
        packages.append("hnswlib")
    run([sys.executable, "-m", "pip", "install", "-q", *packages], cwd=REMOTE_ROOT)

    threshold = os.environ.get("INFERENCE_THRESHOLD", "0.61")
    output = "submission/sku_item_submission_original_dataset_calibrated_061.csv"
    command = [
        sys.executable,
        "submission_inference.py",
        "--checkpoint",
        str(checkpoint),
        "--threshold",
        threshold,
        "--output",
        output,
    ]
    batch_size = os.environ.get("INFERENCE_BATCH_SIZE")
    if batch_size:
        command.extend(["--batch-size", batch_size])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_ROOT / "src")
    run(command, cwd=REMOTE_ROOT, env=env)

    result = REMOTE_ROOT / output
    if not result.is_file() or not result.with_suffix(".json").is_file():
        raise RuntimeError(f"inference did not write the result bundle: {result}")
    print(f"[remote] wrote {result}", flush=True)


if __name__ == "__main__":
    main()
