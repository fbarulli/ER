#!/usr/bin/env python3
"""Download a completed/partial Colab training run before teardown.

Usage:
  python manual_download_results.py \
    --session my-highram-session \
    --remote-base /content/EuromonitoR/results/concurrent_train_<run-id>

The script archives and downloads ordinary worker artifacts and checkpoints
separately, then verifies every downloaded file against a remote SHA-256
manifest. It intentionally does not use DVC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_colab(*args: str, timeout: int, config: Path | None = None) -> None:
    command = ["colab"]
    if config is not None:
        command.extend(["--config", str(config)])
    command.extend(args)
    subprocess.run(command, check=True, timeout=timeout)


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def remote_archiver(remote_base: str, script_path: Path) -> None:
    script_path.write_text(
        f'''import hashlib, json, pathlib, tarfile
base = pathlib.Path({remote_base!r})
if not base.is_dir():
    raise FileNotFoundError(base)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build(kind, predicate):
    files = []
    for worker in sorted(base.glob("worker_*")):
        if not worker.is_dir():
            continue
        for path in sorted(worker.rglob("*")):
            if not path.is_file() or not predicate(path):
                continue
            rel = path.relative_to(base).as_posix()
            files.append({{"path": rel, "size": path.stat().st_size, "sha256": digest(path)}})
    archive_path = base / f"manual_download_{{kind}}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for entry in files:
            archive.add(base / entry["path"], arcname=entry["path"])
    manifest = {{"kind": kind, "run_id": base.name, "files": files}}
    manifest_path = base / f"manual_download_{{kind}}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
    print(json.dumps({{"kind": kind, "archive": str(archive_path), "manifest": str(manifest_path), "files": len(files)}}), flush=True)

build("results", lambda p: "_checkpoints" not in p.parts and ".dvc" not in p.parts and p.name not in {{"manual_download_results.tar.gz", "manual_download_results_manifest.json", "manual_download_checkpoints.tar.gz", "manual_download_checkpoints_manifest.json", "canonical_records.csv", "gate_results.csv", "labeled_pairs.csv"}} and "training" not in p.relative_to(base).parts)
build("checkpoints", lambda p: "_checkpoints" in p.parts)
''',
        encoding="utf-8",
    )


def verify_manifest(root: Path, manifest: Path) -> int:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checked = 0
    for entry in payload["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"missing downloaded file: {path}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"checksum/size mismatch: {path}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("colab_cli_state/sessions.json"),
        help="Colab CLI session-state file used by the launcher",
    )
    parser.add_argument(
        "--no-teardown",
        action="store_true",
        help="download and verify, but leave the remote session running",
    )
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="permit teardown even if no checkpoint archive is available",
    )
    args = parser.parse_args()
    run_id = Path(args.remote_base.rstrip("/")).name.removeprefix("concurrent_train_")
    destination = args.destination or Path("training_results") / run_id
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="manual-download-") as temp:
        temp_dir = Path(temp)
        remote_archiver(args.remote_base, temp_dir / "archive.py")
        run_colab(
            "exec", "-s", args.session, "--timeout", "3600",
            "-f", str(temp_dir / "archive.py"), timeout=3700, config=args.config
        )

    for kind in ("results", "checkpoints"):
        remote_archive = f"{args.remote_base}/manual_download_{kind}.tar.gz"
        remote_manifest = f"{args.remote_base}/manual_download_{kind}_manifest.json"
        local_archive = destination / f"manual_download_{kind}.tar.gz"
        local_manifest = destination / f"manual_download_{kind}_manifest.json"
        try:
            run_colab(
                "download", "-s", args.session, remote_archive, str(local_archive),
                timeout=3600, config=args.config
            )
            run_colab(
                "download", "-s", args.session, remote_manifest, str(local_manifest),
                timeout=600, config=args.config
            )
        except subprocess.CalledProcessError:
            if kind == "checkpoints" and args.allow_missing_checkpoints:
                print("No checkpoint archive was available; ordinary results may still be complete.")
                continue
            raise
        extract_root = destination / "_manual_extract"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir()
        safe_extract(local_archive, extract_root)
        checked = verify_manifest(extract_root, local_manifest)
        for path in extract_root.iterdir():
            target = destination / path.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(path), str(target))
        shutil.rmtree(extract_root)
        print(f"Downloaded and verified {kind}: {checked} files -> {destination}")

    if not args.no_teardown:
        print(f"Tearing down remote session: {args.session}")
        try:
            run_colab("stop", "-s", args.session, timeout=120, config=args.config)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Downloads verified, but remote teardown failed; retry `colab stop "
                f"-s {args.session}` manually"
            ) from exc
        print("Remote session teardown complete.")


if __name__ == "__main__":
    main()
