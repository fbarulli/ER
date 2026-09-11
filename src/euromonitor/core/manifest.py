"""lib/manifest.py — the silent-drop guardrail layer's primitives
(task 1 of SILENT_DROPS.md).

Pipeline outputs were historically written with plain `to_csv` /
`open().write()`, so an interrupt (Ctrl-C, OOM, dead GPU) could leave a
TRUNCATED file on the FINAL path — and a re-run that skips existing
outputs would then treat the partial artifact as good.  Every helper
here removes that class of silent corruption:

  sha256_file      chunked 1 MiB read -> lowercase hex digest (the
                   content fingerprint that manifests record; pattern
                   ported from training/data_quality_audit._sha256)
  atomic_write*    write to a `.tmp-<pid>` SIBLING in the same
                   directory, flush + fsync, then `os.replace` onto the
                   final path — a reader never observes a partial file,
                   and any exception unlinks the temp sibling
  count_drop       the row-accounting atom (before/after/dropped) later
                   tasks pour into manifests and loss guards

Only the pure helpers live here.  The `StageManifest` model plus its
write/read/verify functions are task 3 of SILENT_DROPS.md; no callers
are wired up yet (tasks 4+).

Written-last doctrine: `os.replace` is atomic within a filesystem, so a
manifest published through atomic_write is only ever fully present or
fully absent — its presence with `status: "complete"` IS the completion
marker, and leftover `.tmp-*` residue marks a stage that died mid-write.

Temp-naming choice (documented, one of the two allowed forms): the
sibling is `<name>.tmp-<pid>`, NOT a dotfile `.<name>.tmp-<pid>`, so
that residue stays discoverable by a plain `*.tmp-*` glob — exactly what
`verify_manifest` (task 3) fails on.  Creation is O_EXCL: a stale
`.tmp-<pid>` left by a crashed run colliding with a reused pid fails
loudly (no silent overwrite) per the repo's no-fallbacks doctrine.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from euromonitor.core.common import (
    CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    _path,
    training_cfg,
)
from euromonitor.core.schemas import ManifestFile, StageManifest

# 1 MiB per read — matches data_quality_audit._sha256, keeps the 53MB
# dataset hashable without loading it into memory.
_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Lowercase hex sha256 of a file, read in 1 MiB chunks.

    Raises FileNotFoundError naturally when `path` does not exist.
    """
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _temp_path(final: Path) -> Path:
    """The `<name>.tmp-<pid>` sibling — same directory as the target."""
    return final.with_name(f"{final.name}.tmp-{os.getpid()}")


def _fsync_and_publish(temp: Path, final: Path) -> Path:
    """fsync a fully-written temp file, then os.replace it onto `final`."""
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, final)
    return final


def atomic_write(path: str | Path, data: bytes) -> Path:
    """Write `data` to `path` so the final path never holds a partial file.

    Writes `<name>.tmp-<pid>` in the SAME directory, flushes + fsyncs it,
    then `os.replace`s it onto the target (atomic rename on POSIX and
    Windows).  On any exception — including KeyboardInterrupt — the temp
    sibling is unlinked, leaving at most the old content plus `.tmp-*`
    residue.  Returns the final path.
    """
    final = Path(path)
    temp = _temp_path(final)
    # "xb" = O_WRONLY | O_CREAT | O_EXCL: a stale sibling from a crashed
    # run colliding with a reused pid raises FileExistsError.  Opened
    # BEFORE the try so a collision never unlinks that stale file — the
    # `.tmp-*` residue is the crash evidence verify_manifest fails on.
    stream = temp.open("xb")
    try:
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_and_publish(temp, final)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return final


def atomic_write_text(path: str | Path, text: str) -> Path:
    """`atomic_write` for a str payload, encoded UTF-8."""
    return atomic_write(path, text.encode("utf-8"))


def atomic_write_json(obj: Any, path: str | Path, **json_kwargs: Any) -> Path:
    """`atomic_write` a JSON document.

    Defaults `ensure_ascii=False, indent=2` (UTF-8-native, readable
    diffs); either can be overridden via `json_kwargs`, which are
    forwarded to `json.dumps` verbatim.
    """
    kwargs: dict[str, Any] = {"ensure_ascii": False, "indent": 2}
    kwargs.update(json_kwargs)
    return atomic_write_text(path, json.dumps(obj, **kwargs))


def atomic_write_csv(
    df: pd.DataFrame, path: str | Path, **to_csv_kwargs: Any
) -> Path:
    """`df.to_csv` through the atomic mechanism.

    Serializes to the `<name>.tmp-<pid>` sibling, fsyncs, then
    `os.replace`s onto the target — so a reader or a skip-if-exists
    re-run never sees a truncated CSV.  `to_csv_kwargs` are forwarded
    verbatim (e.g. `index=False`).
    """
    final = Path(path)
    temp = _temp_path(final)
    # to_csv opens "w", so the O_EXCL guarantee atomic_write gets from
    # "xb" is enforced by hand — checked BEFORE the try so a collision
    # never unlinks the stale sibling (the crash evidence, same as above).
    if temp.exists():
        raise FileExistsError(temp)
    try:
        df.to_csv(temp, **to_csv_kwargs)
        _fsync_and_publish(temp, final)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return final


def count_drop(before: int, after: int, reason: str) -> dict[str, int | str]:
    """Row-accounting atom for later tasks — logs nothing itself.

    Returns the drop record manifests and guards aggregate; the caller
    decides what to do with it.
    """
    return {
        "reason": reason,
        "before": before,
        "after": after,
        "dropped": before - after,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-stage manifest (SILENT_DROPS task 3; task 4 wires the first caller)
#
# begin_manifest snapshots inputs; finish_manifest validates the row
# accounting closes, then writes <manifest_dir>/<stage>.json LAST via
# atomic_write_json — the rename IS the completion marker.  verify_manifest
# re-checks everything against disk and fails on any .tmp-* residue.


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _csv_rows(path: Path) -> tuple[int | None, int | None]:
    """(data_rows, cols) for CSV-ish files, (None, None) otherwise.

    Counts by newline without loading the file into memory (the 53MB raw
    export must stay cheap to snapshot).
    """
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None, None
    cols = None
    with path.open("rb") as stream:
        header = stream.readline()
        if header:
            cols = header.count(b",") + 1
        total = sum(1 for line in stream if line.strip())
    return total, cols


def _file_entry(path: Path) -> dict[str, Any]:
    rows, cols = _csv_rows(path)
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "rows": rows,
        "cols": cols,
    }


def _environment(seed: int | None) -> dict[str, str]:
    env: dict[str, str] = {
        "git_sha": "unknown",
        "config_sha256": "unknown",
        "host": platform.node(),
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if head.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            sha = head.stdout.strip()
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha = f"{sha[:12]}-dirty"
            env["git_sha"] = sha
    except (OSError, subprocess.SubprocessError):
        pass  # git absent — "unknown" is the documented fallback
    digests = []
    for cfg_path in (CONFIG_PATH, TRAINING_CONFIG_PATH):
        try:
            digests.append(sha256_file(cfg_path))
        except OSError:
            digests = []
            break
    if digests:
        env["config_sha256"] = hashlib.sha256(
            "|".join(digests).encode()
        ).hexdigest()
    if seed is not None:
        env["seed"] = str(seed)
    return env


def begin_manifest(
    stage: str,
    inputs: list[str | Path],
    seed: int | None = None,
) -> StageManifest:
    """Snapshot the inputs a stage is about to read.

    Hashes each input NOW (cheap-chunked), counts CSV rows, records
    started + environment.  The returned manifest is status "running" —
    incomplete by construction until finish_manifest renames it in.
    """
    return StageManifest(
        schema_version="1",
        stage=stage,
        started=_utc_now(),
        finished=None,
        status="running",
        inputs=[
            ManifestFile.model_validate(_file_entry(Path(p)))
            for p in inputs
        ],
        outputs=[],
        row_accounting={},
        environment=_environment(seed),
        expected_outputs=[],
    )


def _check_closure(row_accounting: dict[str, Any]) -> None:
    input_rows = row_accounting.get("input_rows")
    output_rows = row_accounting.get("output_rows")
    dropped = row_accounting.get("dropped") or {}
    if input_rows is None or output_rows is None:
        return  # partial accounting is allowed until the stage reports
    total_dropped = sum(int(v) for v in dropped.values())
    # aggregation stages (data_prep canonical, dedupe tiers) may record
    # kept-and-collapsed rows outside `dropped`: input == output +
    # collapsed + dropped.  A collapsed bucket is a population that
    # REMAINS represented (one row per group), never one that vanished.
    total_collapsed = int(row_accounting.get("collapsed_same_gtin", 0) or 0)
    if input_rows != output_rows + total_collapsed + total_dropped:
        raise ValueError(
            f"row accounting does not close: input_rows={input_rows} "
            f"!= output_rows={output_rows} + collapsed={total_collapsed} "
            f"+ dropped={total_dropped} "
            f"(dropped by reason: {dropped})"
        )


def finish_manifest(
    manifest: StageManifest,
    outputs: list[str | Path],
    row_accounting: dict[str, Any],
    expected_outputs: list[str] | None = None,
    status: str = "complete",
    manifest_dir: str | Path | None = None,
) -> Path:
    """Validate closure, then write <manifest_dir>/<stage>.json LAST.

    The atomic rename publishes the manifest only after every output was
    hashed and the row accounting closes — so a manifest on disk with
    status "complete" proves the stage finished.  `manifest_dir` defaults
    to the audit knob (training_cfg().audit.manifest_dir resolved through
    lib.common._path); the explicit override exists so tests and smokes
    never touch the real results/ tree.
    """
    _check_closure(row_accounting)
    expected = list(expected_outputs or [])
    entries = []
    for p in outputs:
        entry = _file_entry(Path(p))
        name = Path(p).name
        entry["expected"] = name in expected if expected else None
        entries.append(entry)
    manifest.outputs = [ManifestFile.model_validate(e) for e in entries]
    manifest.row_accounting = row_accounting
    manifest.expected_outputs = expected
    manifest.finished = _utc_now()
    manifest.status = status
    directory = (
        Path(manifest_dir)
        if manifest_dir is not None
        else _path(training_cfg().audit.manifest_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(
        manifest.model_dump(mode="json"), directory / f"{manifest.stage}.json"
    )


def read_manifest(stage: str, manifest_dir: str | Path | None = None) -> StageManifest:
    """Parse <manifest_dir>/<stage>.json; FileNotFoundError propagates
    (a missing manifest IS a stage that never completed)."""
    directory = (
        Path(manifest_dir)
        if manifest_dir is not None
        else _path(training_cfg().audit.manifest_dir)
    )
    return StageManifest.model_validate_json(
        (directory / f"{stage}.json").read_text(encoding="utf-8")
    )


def verify_manifest(
    stage: str,
    manifest_dir: str | Path | None = None,
    check_inputs: bool = False,
) -> None:
    """Re-check a published manifest against disk; raise RuntimeError
    listing EVERY problem (not just the first).

    Checks: manifest exists; status is complete; every output is present
    and hash-matches; every expected_outputs name appears in outputs; no
    .tmp-* residue sits next to any listed file; row accounting closes.
    `check_inputs=True` also re-hashes inputs (slow — off by default
    because the raw export is 53MB).
    """
    problems: list[str] = []
    directory = (
        Path(manifest_dir)
        if manifest_dir is not None
        else _path(training_cfg().audit.manifest_dir)
    )
    path = directory / f"{stage}.json"
    if not path.exists():
        raise RuntimeError(f"manifest missing: {path} — stage never completed")
    manifest = StageManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if manifest.status != "complete":
        problems.append(f"status is {manifest.status!r}, not 'complete'")
    for group, entries, rehash in (
        ("input", manifest.inputs, check_inputs),
        ("output", manifest.outputs, True),
    ):
        for entry in entries:
            f = Path(entry.path)
            if not f.exists():
                problems.append(f"{group} missing on disk: {entry.path}")
                continue
            if rehash:
                actual = sha256_file(f)
                if actual != entry.sha256:
                    problems.append(
                        f"{group} sha256 mismatch: {entry.path} "
                        f"manifest={entry.sha256[:12]} actual={actual[:12]}"
                    )
            residue = list(f.parent.glob(f"{f.name}.tmp-*"))
            if residue:
                problems.append(
                    f"interrupted-write residue next to {entry.path}: "
                    f"{[r.name for r in residue]}"
                )
    for name in manifest.expected_outputs:
        if name not in [Path(e.path).name for e in manifest.outputs]:
            problems.append(f"expected output never produced: {name}")
    try:
        _check_closure(manifest.row_accounting)
    except ValueError as err:
        problems.append(str(err))
    if problems:
        raise RuntimeError(
            f"manifest verification failed for stage {stage!r}:\n  - "
            + "\n  - ".join(problems)
        )


def verify_manifests(
    stages: list[str] | None = None,
    manifest_dir: str | Path | None = None,
) -> None:
    """Verify the registry (audit.manifest_stages when stages is None);
    the FIRST failing stage raises, naming the stage."""
    todo = stages if stages is not None else training_cfg().audit.manifest_stages
    for stage in todo:
        verify_manifest(stage, manifest_dir=manifest_dir)
