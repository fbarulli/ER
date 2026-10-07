"""One canonical log root + one progress-frame formatter (owner order A + B).

Every lane/periodic log surface writes under TRAIN_ROOT/logs/<lane>/ (one
roof). Streams that carry tqdm's carriage-return progress frames write them
through progress_frames_to_lines() so the frames survive every capture
surface as plain, grep-able lines — with the last frame always tagged at the
end of the chunk so a training tqdm strip is visible in any log tail.
"""
from __future__ import annotations

from pathlib import Path

from core.common import TRAIN_ROOT

LOGS_DIR_NAME = "logs"
# Tag prefixed to the retained last carriage-return frame of a chunk.
LAST_BAR_TAG = "[tqdm]"


def logs_root() -> Path:
    """The canonical log root (TRAIN_ROOT/logs), created on demand."""
    root = (TRAIN_ROOT / LOGS_DIR_NAME).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def lane_log(lane: str, name: str) -> Path:
    """One log file under the canonical root: logs/<lane>/<name>."""
    return (logs_root() / lane / name).resolve()


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
