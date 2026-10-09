"""One canonical log root + one progress-frame formatter (owner order A + B).

Every lane/periodic log surface writes under TRAIN_ROOT/logs/<lane>/ (one
roof). Streams that carry tqdm's carriage-return progress frames write them
through progress_frames_to_lines() so the frames survive every capture
surface as plain, grep-able lines — with the last frame always tagged at the
end of the chunk so a training tqdm strip is visible in any log tail.
"""
from __future__ import annotations

import os
import traceback
from collections.abc import Callable
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


class LaneTranscript:
    """The ONE run transcript every kaggle-family lane appends to.

    The path is declared once in config (``kaggle.logs_dir`` +
    ``kaggle.files.lane_log``); both the ER lane and the laya lane write their
    whole run — staging, watcher status, streamed kernel output, reconnect
    diagnostics, fetch/stop results — into that single file. The first write of
    a process truncates it (one transcript per run); later writes append, and a
    detached child (``ER_KAGGLE_LANE_APPEND=1``) always appends so it never
    wipes its parent's transcript.
    """

    #: Process-wide gate: the first write opens with "w", the rest with "a".
    _started = False

    def __init__(self, path: Path, *, lane: str, stamp: Callable[[], str]):
        self._path = path
        self._lane = lane
        self._stamp = stamp

    @staticmethod
    def _kaggle():
        from core.common import training_cfg

        return training_cfg().kaggle

    @classmethod
    def roof_for(cls, train_root: Path) -> Path:
        """The declared transcript directory (``kaggle.logs_dir``)."""
        return (Path(train_root) / cls._kaggle().logs_dir).resolve()

    @classmethod
    def path_for(cls, train_root: Path) -> Path:
        """The declared single transcript path (``kaggle.files.lane_log``)."""
        return (cls.roof_for(train_root) / cls._kaggle().files.lane_log).resolve()

    @classmethod
    def from_config(cls, train_root: Path, *, lane: str,
                    stamp: Callable[[], str]) -> LaneTranscript:
        return cls(cls.path_for(train_root), lane=lane, stamp=stamp)

    def write(self, line: str) -> None:
        """Timestamped console echo plus the one shared transcript write.

        A log-write failure never masks the operation's own outcome, but it is
        recorded with its FULL traceback (no swallowed logging error).
        """
        stamp = self._stamp()
        print(f"[{self._lane} {stamp}] {line}", flush=True)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            append = os.environ.get("ER_KAGGLE_LANE_APPEND") == "1"
            mode = "a" if (LaneTranscript._started or append) else "w"
            with self._path.open(mode, encoding="utf-8") as handle:
                handle.write(f"{stamp} {line}\n")
            LaneTranscript._started = True
        except OSError as error:
            print(f"[{self._lane}] lane.log write failed ({error}); continuing\n"
                  f"{traceback.format_exc()}", flush=True)


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
