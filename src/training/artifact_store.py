"""Publish one completed Colab worker's durable artifacts to Hugging Face Hub.

The worker calls this after training exits successfully and before it writes
its status file.  The completion manifest is uploaded last, so consumers
never mistake a partial upload for a finished run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from core.common import training_cfg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(source: Path, run_id: str, worker: int) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to publish Colab artifacts")
    spec = training_cfg().colab
    api = HfApi(token=token)
    api.create_repo(
        repo_id=spec.artifact_repo_id,
        repo_type="model",
        private=spec.artifact_repo_private,
        exist_ok=True,
    )
    prefix = f"runs/{run_id}/worker_{worker}"
    ignored = ["canonical_records.csv", "gate_results.csv", "mlruns/**", "training.status"]
    api.upload_folder(
        repo_id=spec.artifact_repo_id,
        repo_type="model",
        folder_path=str(source),
        path_in_repo=prefix,
        ignore_patterns=ignored,
        commit_message=f"Upload training artifacts: {run_id} worker {worker}",
    )
    files = [p for p in source.rglob("*") if p.is_file() and p.name not in {"canonical_records.csv", "gate_results.csv", "training.status"} and "mlruns" not in p.parts]
    manifest = {
        "run_id": run_id,
        "worker": worker,
        "files": [
            {"path": str(p.relative_to(source)), "bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(files)
        ],
    }
    local_manifest = source / "hub_manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(local_manifest),
        path_in_repo=f"{prefix}/hub_manifest.json",
        repo_id=spec.artifact_repo_id,
        repo_type="model",
        commit_message=f"Complete training artifacts: {run_id} worker {worker}",
    )
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker", type=int, required=True)
    args = parser.parse_args()
    print(f"[artifact-store] published {publish(args.source, args.run_id, args.worker)}", flush=True)


if __name__ == "__main__":
    main()
