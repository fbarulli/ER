"""src/core/manifest.py — the silent-drop guardrail layer's primitives
(task 1 of SILENT_DROPS.md).

Pipeline outputs were historically written with plain `to_csv` /
`open().write()`, so an interrupt (Ctrl-C, OOM, dead GPU) could leave a
TRUNCATED file on the FINAL path — and a re-run that skips existing
outputs would then treat the partial artifact as good.  Every helper
here removes that class of silent corruption:

  file_size_snapshot  the structural identity of a stage's files: present
                   on disk, byte size, plus CSV row/column counts.  No
                   content identity is ever computed.
  atomic_write*    write to a `.tmp-<pid>` SIBLING in the same
                   directory, flush + fsync, then `os.replace` onto the
                   final path — a reader never observes a partial file,
                   and any exception unlinks the temp sibling
  publish_replacing  the publish half alone, for callers that streamed
                   their own payload to the sibling
  atomic_write_stream  the same mechanism for payloads too large to
                   buffer (``json.dump``/``to_csv`` onto the sibling)
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
that residue stays discoverable by a plain `*.tmp-*` glob.  Creation is
O_EXCL: a stale
`.tmp-<pid>` left by a crashed run colliding with a reused pid fails
loudly (no silent overwrite) per the repo's no-fallbacks doctrine.
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

import pandas as pd

from core.common import (
    CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    TRAIN_ROOT,
    _path,
    training_cfg,
)
from core.schemas import ManifestFile, StageManifest


def file_size(path: str | Path) -> int:
    """The ONE structural size accessor, re-exported for manifest consumers.

    Delegates to ``core.portable_archive.file_size``: a regular file reports its
    bytes on disk, a directory reports its summed regular-file bytes. Identity
    is structural (names + byte sizes), so nothing is ever fingerprinted here
    (owner directive 2026-10-08).

    Raises FileNotFoundError naturally when `path` does not exist.
    """
    from core.portable_archive import file_size as _file_size

    return _file_size(path)


def _file_size(path: str | Path) -> int:
    """Byte size of the file at ``path`` (internal alias of :func:`file_size`).

    Raises FileNotFoundError naturally when `path` does not exist. Identity is
    structural: size in bytes, never a content digest.
    """
    return file_size(path)


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
    # `.tmp-*` residue is the crash evidence.
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


def publish_replacing(temp: str | Path, final: str | Path) -> Path:
    """Publish a COMPLETED sibling onto ``final``: fsync, then ``os.replace``.

    The publish half of :func:`atomic_write`, exposed for callers that streamed
    their payload into the sibling themselves (a verified download, a streamed
    JSON dump, an external writer). ``os.replace`` overwrites an existing
    ``final`` atomically; nothing is published while the payload is incomplete.
    """
    return _fsync_and_publish(Path(temp), Path(final))


@contextmanager
def atomic_write_stream(path: str | Path, *, encoding: str = "utf-8",
                        newline: str | None = None) -> Iterator[TextIO]:
    """``atomic_write`` for payloads streamed rather than buffered as bytes.

    Yields the ``<name>.tmp-<pid>`` sibling open for text writing; on clean
    exit it is closed, fsynced and `os.replace`d onto ``path``, and on any
    exception it is unlinked — the same publication, O_EXCL collision and
    ``.tmp-*`` residue semantics as :func:`atomic_write`. Use it when the
    payload is too large to build in memory (``json.dump`` / ``df.to_csv``).
    """
    final = Path(path)
    temp = _temp_path(final)
    stream = temp.open("x", encoding=encoding, newline=newline)
    try:
        with stream:
            yield stream
        _fsync_and_publish(temp, final)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


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
# atomic_write_json — the rename IS the completion marker.  Nothing
# re-checks the recorded values afterwards: data is never checked.


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _csv_rows(path: Path) -> tuple[int | None, int | None]:
    """(data_rows, cols) for CSV-ish files, (None, None) otherwise.

    Counts CSV RECORDS by streaming the file through the csv module —
    newlines quoted inside multi-line fields (product descriptions carry
    them) must not inflate the count as bogus extra rows. Cols from the
    header RECORD for the same reason (a quoted header field could hold
    commas). A malformed final fragment under a strict reader is repaired
    via errors='replace' + strict=False so counting never turns into the
    stage's failure mode.
    """
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None, None
    total = 0
    cols = None
    import csv
    import sys

    # Quoted product descriptions can exceed csv's 128KiB per-field guard;
    # a counting helper must not turn field size into a stage failure.
    csv.field_size_limit(sys.maxsize)

    with path.open("rb") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        rows = csv.reader(text)
        try:
            header = next(rows)
            cols = len(header)
        except StopIteration:
            return 0, None
        for _ in rows:
            total += 1
    return total, cols


def _file_entry(path: Path) -> dict[str, Any]:
    rows, cols = _csv_rows(path)
    return {
        "path": path.resolve().as_posix(),
        "size": _file_size(path),
        "rows": rows,
        "cols": cols,
    }


def _environment(seed: int | None) -> dict[str, str]:
    env: dict[str, str] = {
        "git_sha": "unknown",
        "host": platform.node(),
        "config_files": ",".join(
            sorted(_config_name(path) for path in (CONFIG_PATH, TRAINING_CONFIG_PATH))
        ),
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
    if seed is not None:
        env["seed"] = str(seed)
    return env


def _config_name(path: str | Path) -> str:
    """The declared config file name, repo-relative when it lives in the repo."""
    path = Path(path)
    root = Path(TRAIN_ROOT)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def begin_manifest(
    stage: str,
    inputs: list[str | Path],
    seed: int | None = None,
) -> StageManifest:
    """Snapshot the inputs a stage is about to read.

    Sizes each input NOW (cheap-chunked), counts CSV rows, records
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
    recorded and the row accounting closes — so a manifest on disk with
    status "complete" proves the stage finished.  `manifest_dir` defaults
    to the audit knob (training_cfg().audit.manifest_dir resolved through
    lib.common._path); the explicit override exists so tests and smokes
    never touch the real results/ tree.
    """
    _check_closure(row_accounting)
    # The public helper accepts filesystem paths for convenience, while the
    # manifest schema owns output names. Normalize at this boundary so
    # Pydantic receives the declared ``list[str]`` contract and expected
    # output matching compares the same basename representation.
    expected = [Path(p).name for p in (expected_outputs or [])]
    manifest.outputs = [ManifestFile.model_validate(e)
                        for e in _output_entries(outputs, expected)]
    manifest.row_accounting = row_accounting
    manifest.expected_outputs = expected
    manifest.finished = _utc_now()
    manifest.status = status
    directory = Path(manifest_dir) if manifest_dir is not None \
        else _path(training_cfg().audit.manifest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(
        manifest.model_dump(mode="json"), directory / f"{manifest.stage}.json"
    )


def _output_entries(outputs: list[str | Path], expected: list[str]) -> list[dict]:
    """The publish-time output records, flagged against the expected set."""
    entries = []
    for path in outputs:
        entry = _file_entry(Path(path))
        name = Path(path).name
        entry["expected"] = name in expected if expected else None
        entries.append(entry)
    return entries


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


def _manifest_path(stage: str, manifest_dir: str | Path | None) -> Path:
    """The declared audit manifest directory (or the caller's override)."""
    directory = (Path(manifest_dir) if manifest_dir is not None
                 else _path(training_cfg().audit.manifest_dir))
    return directory / f"{stage}.json"
