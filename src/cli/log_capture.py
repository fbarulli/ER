"""One canonical logs root + one progress-frame formatter (owner order A + B).

Every lane/periodic log surface writes under the ONE declared logs root
(``config/paths.yaml`` ``paths.logs_dir``) plus a declared per-lane subdir:
``<logs_root>/<lane>/<name>``. The root resolves against the CANONICAL
(main-worktree) checkout, so a run launched from a linked worktree still
writes the shared transcript to ONE location instead of a per-worktree
``logs/`` (``core.project_root.canonical_root``). Streams that carry tqdm's
carriage-return progress frames write them through progress_frames_to_lines()
so the frames survive every capture surface as plain, grep-able lines — with
the last frame always tagged at the end of the chunk so a training tqdm strip
is visible in any log tail.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.common import TRAIN_ROOT, data_cfg
from core.project_root import canonical_root

# Tag prefixed to the retained last carriage-return frame of a chunk.
LAST_BAR_TAG = "[tqdm]"


def logs_dir_name() -> str:
    """The declared logs-root name (``paths.logs_dir``), the ONE spelling."""
    return data_cfg().paths.logs_dir


def logs_root(base: Path | None = None) -> Path:
    """The canonical logs root, created on demand.

    ``base`` overrides the local project root (tests re-point it); the root is
    otherwise resolved against the canonical (main-worktree) checkout, so a
    linked worktree shares the ONE logs roof. The directory name comes from the
    config SSOT (``paths.logs_dir``) — never a code literal.
    """
    source = TRAIN_ROOT if base is None else Path(base)
    name = Path(logs_dir_name())
    root = (name if name.is_absolute() else canonical_root(source) / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def lane_dir(lane: str, base: Path | None = None) -> Path:
    """The directory for one declared lane under the canonical logs root."""
    if not lane or lane in {".", ".."} or lane != Path(lane).name:
        raise ValueError(f"lane log dir must be a bare lane name: {lane!r}")
    return (logs_root(base) / lane).resolve()


def lane_log(lane: str, name: str, base: Path | None = None) -> Path:
    """One log file under the canonical root: <logs_root>/<lane>/<name>."""
    path = lane_dir(lane, base) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def progress_frames_to_lines(text: str) -> str:
    """Expand a capture chunk so carriage-return progress frames survive.

    tqdm (and every tqdm consumer) separates refresh frames with "\\r", which
    a plain file-appender collapses into one unreadable line. Rewrite at
    write time: "\\r" -> "\\n", and when the chunk actually carried CR frames,
    retain the final frame as one tagged line so the log tail always shows
    the last visible training-progress strip.
    """
    if not text or "\r" not in text:
        return text
    expanded = text.replace("\r\n", "\n").replace("\r", "\n")
    last_bar = next((span.strip() for span in reversed(text.split("\r"))
                     if span.strip()), "")
    if not last_bar:
        return expanded
    expanded = expanded.rstrip("\n") + "\n" + LAST_BAR_TAG + " " + last_bar + "\n"
    return expanded


def main(argv: list[str] | None = None) -> int:
    """Print the SSOT path for one lane/run log (shell entrypoint).

    Wrapper scripts (``scripts/run_colab_bundle.sh`` etc.) tee their output to
    the printed path instead of an ad-hoc top-level ``logs/*.log``, so every
    run log lands in the declared lane subdir via the SAME resolver the lanes
    use.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--lane", required=True, help="declared lane subdir name")
    parser.add_argument("--name", required=True, help="run log basename")
    args = parser.parse_args(argv)
    print(lane_log(args.lane, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
