"""Immutable, best-effort durability snapshots for HPO generations.

This module intentionally never reads a live checkpoint tree through DVC and
never raises into training after a persistence failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable


# DVC itself runs in subprocesses, but its helper currently sets a process
# environment variable for its cache directory.  Serialising the short
# publish/verify section prevents sibling model coordinators from changing
# that process-global value underneath one another.  Training never holds it.
_DVC_PUBLISH_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sqlite_backup(source: Path, destination: Path) -> None:
    """Create a consistent SQLite copy while the source remains writable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as reader, sqlite3.connect(destination) as writer:
        reader.backup(writer)


def build_snapshot(
    *,
    generation: Path,
    sequence: int,
    optuna_db: Path | None,
    include: list[Path],
    scope: str | None = None,
) -> Path:
    """Build a self-validating immutable snapshot; READY is written last."""
    if scope is not None and (not scope or Path(scope).name != scope):
        raise ValueError(f"snapshot scope must be one path component: {scope!r}")
    # A scope is a DVC workspace boundary.  Model coordinators never share
    # .dvc/, .resume/, cache, or locks even when they complete simultaneously.
    outbox = generation / "transport" / "outbox"
    if scope:
        outbox /= scope
    outbox.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{sequence}-", dir=outbox))
    snapshot = temporary / "payload"
    snapshot.mkdir()
    files: list[dict[str, str | int]] = []
    try:
        for source in include:
            if not source.exists():
                raise FileNotFoundError(f"required snapshot input missing: {source}")
            if source.is_symlink():
                raise RuntimeError(f"snapshot input must not be a symlink: {source}")
            relative = source.relative_to(generation)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
                for child in sorted(target.rglob("*")):
                    if child.is_symlink():
                        raise RuntimeError(f"snapshot payload contains symlink: {child}")
                    if child.is_file():
                        files.append({"path": str(child.relative_to(snapshot)), "sha256": _sha256(child), "size": child.stat().st_size})
            else:
                shutil.copy2(source, target)
                files.append({"path": str(relative), "sha256": _sha256(target), "size": target.stat().st_size})
        if optuna_db and optuna_db.is_file():
            backup = snapshot / "controller" / "optuna.db"
            sqlite_backup(optuna_db, backup)
            files.append({"path": str(backup.relative_to(snapshot)), "sha256": _sha256(backup), "size": backup.stat().st_size})
        manifest = {
            "generation": generation.name,
            "scope": scope,
            "sequence": sequence,
            "created_at": time.time(),
            "files": files,
        }
        (snapshot / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        (snapshot / "READY").write_text("ready\n", encoding="utf-8")
        final = outbox / str(sequence)
        snapshot.rename(final)
        temporary.rmdir()
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_snapshot(snapshot: Path) -> None:
    """Reject incomplete or corrupted local snapshots before transport."""
    ready = snapshot / "READY"
    manifest_path = snapshot / "manifest.json"
    if not ready.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"snapshot is not complete: {snapshot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {str(entry["path"]): entry for entry in manifest.get("files", [])}
    actual = {
        str(path.relative_to(snapshot))
        for path in snapshot.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "READY"}
    }
    if set(declared) != actual:
        raise RuntimeError("snapshot inventory does not match its manifest")
    for entry in declared.values():
        path = snapshot / str(entry["path"])
        if not path.is_file() or path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"snapshot file missing or changed: {path}")
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"snapshot hash mismatch: {path}")


def write_pointer_registry(path: Path, payload: dict) -> None:
    """Atomically publish a complete worker registry; never write in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def aggregate_pointer_registries(registries: list[Path]) -> dict:
    """Pure, idempotent aggregation over only complete atomic registries."""
    merged: dict[str, object] = {"registries": {}}
    for path in sorted(registries):
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged["registries"][str(path)] = payload
    return merged


def best_effort_dvc_publish(
    snapshot: Path, *, publisher: Callable[[Path, Path], object] | None = None
) -> bool:
    """Publish an already immutable snapshot; failures are visible, never fatal."""
    try:
        verify_snapshot(snapshot)
        if publisher is None:
            from training.dvc_store import publish_checkpoint

            publisher = lambda source, output: publish_checkpoint(
                source, output, resume_name=f"snapshot-{output.name}"
            )
        # See _DVC_PUBLISH_LOCK above.  The lock does not cover snapshot
        # creation or training; it only protects DVC's process-global setup.
        with _DVC_PUBLISH_LOCK:
            publisher(snapshot.parent, snapshot)
        print(f"[hpo-durability] DVC snapshot verified: {snapshot}", flush=True)
        return True
    except BaseException as exc:
        print(f"[hpo-durability] DVC snapshot deferred (training continues): {exc}", flush=True)
        return False
