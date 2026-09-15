"""Measure where the Colab launcher's setup wall-clock goes, from its own audit trail.

Evidence sources, no Colab VM and no network needed:

* ``colab_cli_state/history/*.jsonl`` — the CLI's session history.  Every
  remote exec and file operation carries an ISO timestamp, which is what makes
  a per-phase breakdown possible at all.
* ``colab_system.log`` — the launcher's live log, for the ``[local-prepare]``
  and ``[run] remote metadata recorded`` stamp pair that times the local
  prepared-bundle build (that work happens on THIS host, so it never appears
  in the remote history).

Usage::

    PYTHONPATH=src python scripts/profile_colab_setup.py colab_cli_state/history/*.jsonl

Phases are recognised by the distinctive cell the launcher sends for each
step.  Two caveats are reported rather than papered over:

* A single-worker lane sends one long-lived training cell, so its history
  record is written when training FINISHES.  The launch segment is therefore
  only reported when it is under a minute; otherwise it is marked untimed.
* A phase the trail cannot bound is reported as ``untimed``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

# Markers of the launcher's own cells, in the order they run.
_MARKERS = (
    ("provision", "socket.gethostname"),
    ("checkout", "shutil.rmtree"),
    ("deps", "00_deps"),
    ("models", "[models] key="),
    ("runtime_profile", "torch.cuda.get_device_properties"),
)
_TRAIN_MARKERS = ("train-launch", "resume-preflight")
_BUNDLE_MARKER = "prepared_training"
# A launch segment longer than this is a single-worker lane whose training cell
# was recorded at completion, not a real setup phase.
_LAUNCH_SEGMENT_LIMIT_SECONDS = 120.0
_STAMP = re.compile(r"(\d{4}T\d{6}\d{6}Z)")


def _timestamp(event: dict) -> dt.datetime:
    return dt.datetime.fromisoformat(str(event["timestamp"]))


def _outputs(event: dict) -> str:
    outputs = event.get("outputs") or []
    return "".join(item.get("text", "") for item in outputs if isinstance(item, dict))


def _phase_of(event: dict) -> str | None:
    """Name the launcher phase this event opens, or None."""
    if event.get("event_type") != "execution":
        return None
    haystack = f"{event.get('code', '')}\n{_outputs(event)}"
    if any(marker in haystack for marker in _TRAIN_MARKERS):
        return "train_launch"
    for name, marker in _MARKERS:
        if marker in haystack:
            return name
    if _BUNDLE_MARKER in haystack:
        return "local_prepare_end"
    return None


def _is_upload(event: dict) -> bool:
    return event.get("event_type") == "file_operation" and event.get("op") == "upload"


def profile(path: Path) -> list[dict]:
    """Return the setup phases of every training session found in `path`."""
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts = [i for i, e in enumerate(events) if e.get("event_type") == "session_created"]
    rows: list[dict] = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(events)
        window = events[start:stop]
        if not any(_phase_of(e) == "deps" for e in window):
            continue  # not a training lane
        marks = _marks(window)
        uploads = [_timestamp(e) for e in window if _is_upload(e)]
        if not marks.get("deps") or not marks.get("models") or not marks.get("profile"):
            continue
        setup_end = uploads[-1] if uploads else marks["profile"]
        begin = _timestamp(window[0])
        total = (setup_end - begin).total_seconds()
        if total <= 0:
            continue
        # Each name labels the interval that STARTS at that marker and runs to
        # the next one, so [profile, bundles) is the local build and
        # [bundles, setup_end) is the upload burst.
        boundaries = [
            ("provision", begin),
            ("checkout", marks["handshake"] or begin),
            ("launcher_turnaround", marks["checkout"] or begin),
            ("deps", marks["deps"]),
            ("models", marks["models"]),
            ("local_prepare", marks["profile"]),
        ]
        if marks["bundles"] is not None:
            boundaries.append(("uploads", marks["bundles"]))
        else:
            # A lane without prepared bundles uploads its raw inputs instead.
            boundaries.append(("uploads", marks["profile"]))
        for index, (name, at) in enumerate(boundaries):
            following = (
                boundaries[index + 1][1]
                if index + 1 < len(boundaries)
                else setup_end
            )
            rows.append(
                _row(path, window, name, following - at, total)
            )
        if marks["launch"] is not None:
            gap = (marks["launch"] - setup_end).total_seconds()
            if 0 <= gap <= _LAUNCH_SEGMENT_LIMIT_SECONDS:
                rows.append(
                    _row(path, window, "launch", dt.timedelta(seconds=gap), total)
                )
    return rows


def _marks(window: list[dict]) -> dict[str, dt.datetime | None]:
    """First timestamp of each launcher step in one session window."""
    found: dict[str, dt.datetime | None] = {
        "handshake": None, "checkout": None, "deps": None, "models": None,
        "profile": None, "bundles": None, "launch": None,
    }
    for event in window:
        phase = _phase_of(event)
        if phase is None:
            continue
        key = {
            "provision": "handshake", "checkout": "checkout", "deps": "deps",
            "models": "models", "runtime_profile": "profile",
            "local_prepare_end": "bundles", "train_launch": "launch",
        }[phase]
        if found[key] is None:
            found[key] = _timestamp(event)
    return found


def _row(
    path: Path, window: list[dict], phase: str,
    delta: dt.timedelta, total: float,
) -> dict:
    seconds = delta.total_seconds()
    return {
        "history": path.name,
        "session": window[0]["timestamp"][:19],
        "accelerator": window[0].get("accelerator"),
        "phase": phase,
        "seconds": round(seconds, 2),
        "share_pct": round(100 * seconds / total, 1),
        "setup_seconds": round(total, 2),
    }


def _local_prepare_seconds(log: Path) -> float | None:
    """Time the local bundle build from the launcher's own stamp pair.

    This is the same interval the history's `local_prepare` row shows, measured
    on the launcher's clock instead of through an exec round trip, so the two
    together bound it.
    """
    bundle: str | None = None
    recorded: str | None = None
    for line in log.read_text().splitlines():
        match = _STAMP.search(line)
        if match is None:
            continue
        if bundle is None and line.startswith("[local-prepare]") and "bundle=" in line:
            bundle = match.group(1)
        if recorded is None and line.startswith("[run] remote metadata recorded"):
            recorded = match.group(1)
    if bundle is None or recorded is None:
        return None

    def parse(stamp: str) -> dt.datetime:
        return dt.datetime.strptime(stamp, "%m%dT%H%M%S%fZ")

    return (parse(recorded) - parse(bundle)).total_seconds()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="+", type=Path)
    parser.add_argument(
        "--system-log", type=Path, default=Path("colab_system.log"),
        help="launcher live log, for the local prepared-bundle build",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.history:
        rows.extend(profile(path))
    if not rows:
        print("no training session with a deps stage found")
        return
    width = max(len(row["phase"]) for row in rows)
    current = None
    for row in rows:
        if row["session"] != current:
            current = row["session"]
            print(
                f"\n{current}  accelerator={row['accelerator']}  "
                f"setup={row['setup_seconds']:.1f}s  ({row['history']})"
            )
        print(
            f"   {row['phase']:<{width}} {row['seconds']:8.2f}s {row['share_pct']:6.1f}%"
        )
    local = _local_prepare_seconds(args.system_log) if args.system_log.is_file() else None
    if local is None:
        print(
            f"\nlocal bundle build: untimed "
            f"(no [local-prepare]/[run] stamp pair in {args.system_log})"
        )
    else:
        print(f"\nlocal bundle build (launcher clock, {args.system_log}): {local:.2f}s")
    print(
        "launch segments longer than "
        f"{_LAUNCH_SEGMENT_LIMIT_SECONDS:.0f}s are omitted: a single-worker lane "
        "records its training cell at completion, so that gap is training, not setup."
    )


if __name__ == "__main__":
    main()
