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
from pathlib import Path
from typing import Any

import pandas as pd

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
