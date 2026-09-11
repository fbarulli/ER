"""Publish one worker's output through an isolated DVC project."""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path
from core.common import training_cfg

def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def publish(source: Path, run_id: str, worker: int) -> None:
    token = os.environ.get("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required for DagsHub persistence")
    _run(["dvc", "init", "--no-scm"], source)
    remote = training_cfg().colab.dvc_remote_url
    _run(["dvc", "remote", "add", "--default", "dagshub", remote], source)
    _run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], source)
    _run(["dvc", "remote", "modify", "dagshub", "--local", "user", "fbarulli"], source)
    _run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], source)
    paths = [p.name for p in source.iterdir() if p.name not in {".dvc", "mlruns", "canonical_records.csv", "gate_results.csv"}]
    if paths:
        _run(["dvc", "add", *paths], source)
    _run(["dvc", "push"], source)
    (source / "dvc_manifest.json").write_text(f'{{"run_id": {run_id!r}, "worker": {worker}}}\n', encoding="utf-8")
    repo = training_cfg().colab.dagshub_repo
    env = {**os.environ, "DAGSHUB_USER_TOKEN": token}
    subprocess.run(
        ["dagshub", "upload", repo, str(source), f"runs/{run_id}/worker_{worker}"],
        cwd=source, env=env, check=True,
    )

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--run-id", required=True); p.add_argument("--worker", type=int, required=True)
    a = p.parse_args(); publish(a.source, a.run_id, a.worker); print(f"[dvc] published worker {a.worker}", flush=True)

if __name__ == "__main__": main()
