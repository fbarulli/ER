"""One canonical log root + one progress-frame formatter (owner order A + B).

Every lane/periodic log surface writes under the CANONICAL checkout's
logs/<lane>/ (one roof) — resolved through ProjectRoot.canonical, so a run
launched from a linked worktree logs to the main tree, never the scratch
checkout. Streams that carry tqdm's carriage-return progress frames write them
through progress_frames_to_lines() so the frames survive every capture
surface as plain, grep-able lines — with the last frame always tagged at the
end of the chunk so a training tqdm strip is visible in any log tail.
"""
from __future__ import annotations

from pathlib import Path

from core.common import TRAIN_ROOT
from core.project_root import ProjectRoot

LOGS_DIR_NAME = "logs"
# Tag prefixed to the retained last carriage-return frame of a chunk.
LAST_BAR_TAG = "[tqdm]"


def logs_root() -> Path:
    """The canonical log root, created on demand.

    Resolved through ``ProjectRoot.canonical`` so EVERY lane's logs land in the
    canonical (main) tree at the same roof, whether the run launched from main
    or from a linked ``.worktrees/<name>`` checkout — never the scratch checkout.
    """
    root = (ProjectRoot.canonical(TRAIN_ROOT) / LOGS_DIR_NAME).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def lane_log(lane: str, name: str) -> Path:
    """One log file under the canonical root: logs/<lane>/<name>."""
    return (logs_root() / lane / name).resolve()


def lane_log_at(relative_dir: str, name: str) -> Path:
    """One lane transcript at a fully DECLARED repo-relative directory.

    ``lane_log`` takes the lane subdirectory name; a lane whose transcript roof
    is declared in the config SSOT (``ColabSpec.log_dir``, e.g. ``logs/colab``)
    asks this instead, so the directory is never re-spelled beside the lane's
    declared file name. The parent is created on demand like ``lane_log``.

    Resolved under the CANONICAL checkout (``ProjectRoot.canonical``) like every
    roof here, so a run launched from a linked ``.worktrees/<name>`` checkout
    still transcripts to the main tree — never the scratch checkout.
    """
    directory = Path(relative_dir)
    if directory.is_absolute() or ".." in directory.parts or not directory.parts:
        raise ValueError(f"lane log dir must be repository-relative: {relative_dir!r}")
    path = (ProjectRoot.canonical(TRAIN_ROOT) / directory / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def progress_frames_to_lines(text: str) -> str:
    """Expand a capture chunk so carriage-return progress frames survive.

    tqdm (and every tqdm consumer) separates refresh frames with "\r", which
    a plain file-appender collapses into one unreadable line. Rewrite at
    write time: "\r" -> "\n", and when the chunk actually carried CR frames,
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
