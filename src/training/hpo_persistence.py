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


class SnapshotBuilder:
    """Build ONE immutable snapshot payload.

    Each method does one job: ``add`` copies one declared input, ``add_sqlite``
    backs up the controller DB, ``_record_*`` build the manifest inventory, and
    ``finalize`` writes the manifest + READY and atomically publishes it.
    """

    def __init__(self, *, generation, sequence, scope=None):
        if scope is not None and (not scope or Path(scope).name != scope):
            raise ValueError(
                f"snapshot scope must be one path component: {scope!r}")
        self.generation = Path(generation)
        self.scope = scope
        self.outbox = self.generation / "transport" / "outbox"
        if scope:
            self.outbox /= scope
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.final = self._reserve(sequence)
        self._temporary = Path(tempfile.mkdtemp(
            prefix=f".{self.final.name}-", dir=self.outbox))
        self.payload = self._temporary / "payload"
        self.payload.mkdir()
        self.files: list[dict[str, str | int]] = []

    def _reserve(self, sequence):
        """The collision-free outbox slot (a same-second race bumps a suffix)."""
        final = self.outbox / str(sequence)
        bump = 1
        while final.exists():
            final = self.outbox / f"{sequence}-{bump}"
            bump += 1
        return final

    def add(self, source):
        """Copy one declared input into the payload and record its hashes."""
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"required snapshot input missing: {source}")
        if source.is_symlink():
            raise RuntimeError(f"snapshot input must not be a symlink: {source}")
        target = self.payload / source.relative_to(self.generation)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
            self._record_tree(target)
        else:
            shutil.copy2(source, target)
            self._record_file(target.relative_to(self.payload), target)
        return self

    def add_sqlite(self, database):
        """Back up the controller SQLite DB consistently (no-op if absent)."""
        database = Path(database)
        if not database.is_file():
            return self
        backup = self.payload / "controller" / "optuna.db"
        sqlite_backup(database, backup)
        self._record_file(backup.relative_to(self.payload), backup)
        return self

    def _record_tree(self, target):
        for child in sorted(target.rglob("*")):
            if child.is_symlink():
                raise RuntimeError(f"snapshot payload contains symlink: {child}")
            if child.is_file():
                self._record_file(child.relative_to(self.payload), child)

    def _record_file(self, relative, path):
        self.files.append({"path": str(relative), "sha256": _sha256(path),
                           "size": path.stat().st_size})

    def _manifest(self):
        return {"generation": self.generation.name, "scope": self.scope,
                "sequence": self.final.name, "created_at": time.time(),
                "files": self.files}

    def finalize(self):
        """Write manifest + READY, then atomically publish."""
        (self.payload / "manifest.json").write_text(
            json.dumps(self._manifest(), sort_keys=True), encoding="utf-8")
        (self.payload / "READY").write_text("ready\n", encoding="utf-8")
        self.payload.rename(self.final)
        self._temporary.rmdir()
        return self.final

    def discard(self):
        shutil.rmtree(self._temporary, ignore_errors=True)

    def build(self, include, optuna_db=None):
        try:
            for source in include:
                self.add(source)
            if optuna_db is not None:
                self.add_sqlite(optuna_db)
            return self.finalize()
        except BaseException:
            self.discard()
            raise


def build_snapshot(
    *,
    generation: Path,
    sequence: int,
    optuna_db: Path | None,
    include: list[Path],
    scope: str | None = None,
) -> Path:
    """Facade: build one immutable snapshot (READY written last)."""
    # A scope is a DVC workspace boundary.  Model coordinators never share
    # .dvc/, .resume/, cache, or locks even when they complete simultaneously.
    builder = SnapshotBuilder(generation=generation, sequence=sequence,
                              scope=scope)
    return builder.build(include, optuna_db)


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


__all__ = [
    "SnapshotBuilder",
    "aggregate_pointer_registries",
    "best_effort_dvc_publish",
    "build_snapshot",
    "sqlite_backup",
    "verify_snapshot",
    "write_pointer_registry",
]
