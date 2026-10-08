"""Run-result retention — exactly one training run and one smoke locally.

Owner ruling 2026-10-05: as soon as a training run's post-processed results
are downloaded, they are DVC-tracked, committed, and pushed, and every older
local training run is deleted. Smoke results are never DVC-tracked: the most
recent smoke run replaces the previous one locally (overwrite, no history).
This module is part of the training call: the colab download path calls it.

Scope ruling: this covers colab TRAINING/SMOKE lanes (the
download_verified_training_results path). report/sims/HPO lanes publish
through their own flows and are out of retention scope.

No silent fallbacks: a missing DVC remote, a failed add/commit/push, or an
ambiguous prune target raises with the tool's own stderr — a failed
retention must be visible, because it decides which local results survive.

Verified layout facts (independent check 2026-10-05):
- completion markers live in TRACK subdirectories of a run, one or two
  levels below the run root (track_inventory.json or <track>__run_manifest.json
  under <run>/<track>/ or <run>/worker_1/<track>/); a run root itself never
  carries them, so the check recurses no deeper than that.
- .gitignore ignores training_results/ wholesale; retention commits only
  the *.dvc sidecars and the DVC-managed .gitignore additions, enabled by
  explicit negations in .gitignore.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.bundle import bundle_spec
from core.timing import Timing, emit_timing

# Marker files live only in TRACK subdirectories one or two levels below the
# run root: <run>/<track>/<inventory_file> AND (Colab worker layout)
# <run>/worker_1/<track>__<tag>/{<inventory_file>, <track>__run_manifest.json}.
# The inventory name is the bundle contract's (``bundle.inventory_file``), so a
# renamed marker cannot make retention stop recognizing (or start deleting) runs.


def _run_markers() -> tuple[str, ...]:
    """The run-root markers: the bundle inventory name plus the track manifests."""
    return (bundle_spec().inventory_file, "*__run_manifest.json")


class RetentionReceipt(BaseModel):
    """What retention did to one run's results (traceability)."""

    model_config = ConfigDict(extra="forbid")

    run_dir: str
    dvc_tracked: bool
    pruned: list[str]
    seconds: float = Field(ge=0.0)


class RunIndexEntry(BaseModel):
    """One previously published training run in the DVC remote history."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dvc_file: str
    commit: str | None = None


class CalledProcessError(RuntimeError):
    """Tool failure with the tool's own stderr attached to the message."""


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a tool fail-loud; a failure surfaces the tool's own stderr."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CalledProcessError(
            f"{command[0]} failed rc={result.returncode}: "
            f"{' '.join(command[1:])}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git(*arguments: str) -> str:
    """Git runs against the training repo regardless of process cwd."""
    from core.common import TRAIN_ROOT

    return _run(["git", *arguments], cwd=TRAIN_ROOT)


def _dvc(*arguments: str) -> str:
    from core.common import TRAIN_ROOT

    return _run(["dvc", *arguments], cwd=TRAIN_ROOT)


def _looks_like_run(path: Path) -> bool:
    """A run root is recognized by completion markers in its track dirs.

    Markers only ever live in track subdirectories one or two levels below
    the run root (`<run>/<track>/<marker>` or `<run>/worker_1/<track>/<marker>`):
    results/model_tracks/<id>/<track>/<inventory_file>,
    training_results/<run>/worker_1/<track>/<inventory_file> AND
    <track>__run_manifest.json variants all match; nothing deeper and no
    run-root file can fake it.
    """
    for marker in _run_markers():
        for relative in (f"*/{marker}", f"*/*/{marker}"):
            if next(path.glob(relative), None) is not None:
                return True
    return False


def publish_training_run(run_dir: Path) -> RetentionReceipt:
    """DVC-add, commit and push one completed run; prune older local runs.

    The push is the point: the DVC remote becomes the only place older runs
    exist. Local retention keeps exactly this run (and local smokes, which
    belong to a different lane prefix and are never pruned here).
    """
    started = time.monotonic()
    timing = Timing("retention.publish")
    from core.common import TRAIN_ROOT, TRAINING_RESULTS

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or not _looks_like_run(run_dir):
        raise FileNotFoundError(
            f"retention refuses: {run_dir} lacks track-subdir run markers {_run_markers()}"
        )
    relative = run_dir.relative_to(TRAIN_ROOT).as_posix()
    with timing.section("dvc_add"):
        _dvc("add", relative)
    with timing.section("git_commit"):
        # DVC created <run>.dvc beside the data (and its own lane .gitignore
        # additions); both are the only files this commit includes.
        _git("add", f"{relative}.dvc", str(Path(relative).parent / ".gitignore"))
        message = f"results: training run {Path(relative).name} -> dvc; local retention keeps only this run"
        if _git("status", "--porcelain", "--", f"{relative}.dvc"):
            _git("commit", "-m", message)
    with timing.section("dvc_push"):
        _dvc("push", f"{relative}.dvc")
    pruned = _prune_training_runs(keep=run_dir)
    emit_timing(f"[timing] retention published run={run_dir.name} pruned={len(pruned)}")
    return RetentionReceipt(
        run_dir=str(run_dir), dvc_tracked=True, pruned=[str(p) for p in pruned],
        seconds=round(time.monotonic() - started, 3),
    )


def _prune_training_runs(keep: Path) -> list[Path]:
    """Delete every other COMPLETED training run locally; DVC keeps history.

    Smoke runs (smoke_-prefixed dirs) are a different lane: never pruned
    here, they belong to the overwrite rule of replace_smoke.
    """
    from core.common import TRAINING_RESULTS

    pruning: list[Path] = []
    for sibling in sorted(TRAINING_RESULTS.iterdir()):
        if not sibling.is_dir() or sibling == keep or sibling.name == "hpo_runs":
            continue
        if sibling.name.startswith("smoke_"):
            continue
        if not _looks_like_run(sibling):
            continue
        with Timing("retention.prune").section(f"prune:{sibling.name}"):
            _run(["rm", "-rf", str(sibling)])
        pruning.append(sibling)
    return pruning


def replace_smoke(new_smoke_dir: Path) -> Path:
    """Overwrite rule: the newest smoke run is the ONLY training_results/smoke_*.

    Bounded sweep: TRAINING_RESULTS sibling directories with the smoke_
    prefix are removed when they are not `new_smoke_dir` (which already
    carries a smoke_-prefixed id from the colab smoke lane). A deep rglob
    is deliberately NOT used: checkpoint/report directories elsewhere may
    legitimately contain smoke_-named subdirectories.
    """
    from core.common import TRAINING_RESULTS

    new_smoke_dir = Path(new_smoke_dir)
    if not _looks_like_run(new_smoke_dir) or not new_smoke_dir.name.startswith("smoke_"):
        raise FileNotFoundError(
            f"smoke retention refuses: {new_smoke_dir} is not a smoke_-run with track markers"
        )
    timing = Timing("retention.smoke")
    for sibling in sorted(TRAINING_RESULTS.glob("smoke_*")):
        if sibling.is_dir() and sibling != new_smoke_dir:
            with timing.section(f"delete:{sibling.name}"):
                _run(["rm", "-rf", str(sibling)])
    emit_timing(f"[timing] retention smoke overwrite: only {new_smoke_dir.name} remains")
    return new_smoke_dir


def list_runs() -> list[RunIndexEntry]:
    """Enumerate pruned-away runs recoverable from the DVC remote.

    Primary source: the remote directory's .dvc sidecars (dvc list over the
    committed training_results tree). Each entry also carries the retention
    commit that published it, when known.
    """
    from core.common import TRAINING_RESULTS

    base = TRAINING_RESULTS.as_posix()
    entries: list[RunIndexEntry] = []
    for line in (_dvc("list", ".", base).splitlines() or []):
        line = line.strip()
        if not line or not line.endswith(".dvc"):
            # Plain directory/file entries are already covered by their
            # sidecar; only committed *.dvc markers represent published runs.
            continue
        entries.append(RunIndexEntry(run_id=Path(line).stem, dvc_file=f"{base}/{line}"))
    commits = _git("log", "--format=%H %s", "--", base)
    digests = {}
    for row in commits.splitlines():
        head, _, message = row.partition(" ")
        if message.startswith("results: training run "):
            digest_id = message.removeprefix("results: training run ").split(" ", 1)[0]
            digests[digest_id] = head[:10]
    for entry in entries:
        entry.commit = digests.get(entry.run_id)
    return entries


def pull_run(run_id: str) -> Path:
    """Restore one published run locally, on demand (exclusive retrieval).

    Interaction with retention: a pulled-back PREVIOUS run becomes local
    again — it is pruned again when a newer completed run publishes.
    """
    from core.common import TRAINING_RESULTS

    entry = next(
        (candidate for candidate in list_runs() if candidate.run_id == run_id),
        None,
    )
    if entry is None:
        raise FileNotFoundError(
            f"no published training run {run_id!r}; try: "
            f"python -m model_tracks.run_retention list"
        )
    _dvc("pull", entry.dvc_file)
    restored = TRAINING_RESULTS / run_id
    if not _looks_like_run(restored):
        raise FileNotFoundError(f"pulled {run_id!r} but it lacks track markers: {restored}")
    return restored


def main() -> None:
    """CLI: `list` shows published runs; `pull <run_id>` restores one."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "pull"])
    parser.add_argument("run_id", nargs="?", default=None)
    arguments = parser.parse_args()
    if arguments.command == "list":
        entries = list_runs()
        for entry in entries:
            print(f"{entry.run_id}  (publish commit {entry.commit or 'unrecorded'})")
        if not entries:
            print("no published runs yet")
    elif arguments.run_id is None:
        raise SystemExit("pull requires a run id (see `list`)")
    else:
        restored = pull_run(arguments.run_id)
        print(f"restored {restored}")


if __name__ == "__main__":
    main()
