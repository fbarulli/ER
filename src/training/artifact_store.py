"""Publish one completed Colab worker's durable artifacts to Hugging Face Hub.

The worker calls this after training exits successfully and before it writes
its status file.  The completion manifest is uploaded last, so consumers
never mistake a partial upload for a finished run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.common import training_cfg
from core.manifest import sha256_file


# The worker's own bookkeeping files: written for completion/audit, never part
# of a published bundle. The Hub ignore globs and the manifest enumeration both
# derive from this ONE set, so the two spellings cannot drift apart.
UPLOAD_EXCLUDED_FILES = ("canonical_records.csv", "gate_results.csv", "training.status")


def _is_uploadable(path: Path) -> bool:
    """Whether one path is a publishable artifact (a file the worker does not own)."""
    return path.is_file() and path.name not in UPLOAD_EXCLUDED_FILES


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
    ignored = list(UPLOAD_EXCLUDED_FILES)
    api.upload_folder(
        repo_id=spec.artifact_repo_id,
        repo_type="model",
        folder_path=str(source),
        path_in_repo=prefix,
        ignore_patterns=ignored,
        commit_message=f"Upload training artifacts: {run_id} worker {worker}",
    )
    files = [p for p in source.rglob("*") if _is_uploadable(p)]
    manifest = {
        "run_id": run_id,
        "worker": worker,
        "files": [
            {"path": str(p.relative_to(source)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
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
