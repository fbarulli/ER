#!/usr/bin/env python3
"""Run only neural embedding inference in a Colab runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


REMOTE_ROOT = Path(os.environ.get("REMOTE_ROOT", "/content/EuromonitoR"))
TEXTS = Path("/content/inference_texts.jsonl")
TEXTS_META = Path("/content/inference_texts.json")
CHECKPOINT_ARCHIVE = Path("/content/checkpoint-114-inference.tar.gz")
EMBEDDINGS = Path("/content/inference_embeddings.npy")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print(f"[remote] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def join_checkpoint() -> Path:
    parts = sorted(Path("/content").glob("checkpoint.part-*"))
    if not parts:
        raise FileNotFoundError("checkpoint upload parts are missing")
    with CHECKPOINT_ARCHIVE.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    with tarfile.open(CHECKPOINT_ARCHIVE, "r:gz") as archive:
        archive.extractall(REMOTE_ROOT, filter="data")
    checkpoint = REMOTE_ROOT / "checkpoint-114"
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError("checkpoint model weights are missing")
    return checkpoint


def checkout() -> None:
    run([
        "git", "clone", "--filter=blob:none", "--no-checkout", "--branch",
        os.environ["REPO_BRANCH"], "--single-branch", os.environ["REPO_URL"],
        str(REMOTE_ROOT),
    ])
    run(["git", "sparse-checkout", "init", "--no-cone"], cwd=REMOTE_ROOT)
    run([
        "git", "sparse-checkout", "set", "/src/", "/config/",
        "/submission_inference.py", "/artifacts/wheels/",
    ], cwd=REMOTE_ROOT)
    run(["git", "checkout", "--detach", os.environ["REPO_COMMIT"]], cwd=REMOTE_ROOT)


def main() -> None:
    import torch

    checkout()
    checkpoint = join_checkpoint()
    metadata = json.loads(TEXTS_META.read_text(encoding="utf-8"))
    if sha256(TEXTS) != metadata["texts_sha256"]:
        raise RuntimeError("prepared text upload failed checksum validation")

    requested_gpu = os.environ.get("REQUESTED_GPU", "CPU")
    if requested_gpu == "CPU":
        print("[remote] CPU validation complete; neural inference intentionally skipped", flush=True)
        return
    if not torch.cuda.is_available():
        raise SystemExit(f"{requested_gpu} requested but CUDA is unavailable")
    print(f"[remote] device={torch.cuda.get_device_name(0)}", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])

    from sentence_transformers import SentenceTransformer
    import numpy as np

    with TEXTS.open("r", encoding="utf-8") as handle:
        texts = [json.loads(line) for line in handle]
    if len(texts) != int(metadata["text_count"]):
        raise RuntimeError("prepared text count changed during transfer")
    model = SentenceTransformer(str(checkpoint), device="cuda")
    embeddings = model.encode(
        texts,
        batch_size=int(os.environ.get("INFERENCE_BATCH_SIZE", "512")),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    np.save(EMBEDDINGS, embeddings.astype(np.float32), allow_pickle=False)
    result_meta = {
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "sha256": sha256(EMBEDDINGS),
    }
    part_size = 16 * 1024 * 1024
    parts = []
    with EMBEDDINGS.open("rb") as source:
        number = 0
        while chunk := source.read(part_size):
            part = Path(f"/content/embedding.part-{number:03d}")
            part.write_bytes(chunk)
            parts.append(part.name)
            number += 1
    result_meta["parts"] = parts
    Path("/content/embedding_manifest.json").write_text(
        json.dumps(result_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[remote] encoded {len(texts):,} texts on GPU", flush=True)


if __name__ == "__main__":
    main()
