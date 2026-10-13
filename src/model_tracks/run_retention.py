"""Run-result retention — DVC is the store, Git holds only pointers.

Owner ruling 2026-10-05: as soon as a training run's post-processed results
are downloaded, they are DVC-tracked, committed, and pushed, and every older
local training run is deleted. SMOKE runs follow the same rule since the owner
mandate 2026-10-09 removed their exemption: the newest smoke run is published to
DVC and is the only one kept locally (its published siblings are then freed).
This module is part of the training call: the colab download path calls it.

Owner goal 2026-10-09: every run RESULT (the suite result archives, the frozen
input transports and the per-run trees under ``results/``, plus the verified
training-results subtree ``TRAINING_RESULTS`` under it) is published through the SAME flow —
``publish_result_paths`` / ``publish_training_run`` reuse one DVC track, push and
pointer-commit implementation, so the dagshub remote is the only place run
payloads live and Git carries only ``*.dvc`` pointers and small receipts.
``drop_local`` / the prune rule are what keep the working tree free of the big
payloads. DVC is WRITE-ONLY storage (owner mandate 2026-10-09): no lane loads
from it; ``pull_run`` is an OPERATOR-ONLY retrieval tool for the CLI and is
never called by a lane.

Scope ruling: this covers the colab TRAINING/SMOKE lanes (the
download_verified_training_results path) and the all-track/embeddings result
surface; report/sims/HPO lanes publish through their own flows.

No silent fallbacks: a missing DVC remote, a failed add/commit/push, or an
ambiguous prune target raises with the tool's own stderr — a failed
retention must be visible, because it decides which local results survive.

Verified layout facts (independent check 2026-10-05):
- completion markers live in TRACK subdirectories of a run, one or two
  levels below the run root (track_inventory.json or <track>__run_manifest.json
  under <run>/<track>/ or <run>/worker_1/<track>/); a run root itself never
  carries them, so the check recurses no deeper than that.
- .gitignore ignores results/ wholesale; retention commits only
  the *.dvc sidecars and the DVC-managed .gitignore additions, enabled by
  explicit negations in .gitignore.
"""
from __future__ import annotations

import shutil
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


class ResultReceipt(BaseModel):
    """What publishing one batch of run-result paths to DVC did (traceability)."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str]
    pointers: list[str]
    dropped_local: bool
    seconds: float = Field(ge=0.0)


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


def _relative(path: Path) -> str:
    """One repository-relative spelling for a DVC/Git target."""
    from core.common import TRAIN_ROOT

    return Path(path).resolve().relative_to(TRAIN_ROOT).as_posix()


def _authenticate() -> str:
    """Authenticate the repository DVC remote from ``.env``'s DVC_API_KEY.

    ``.dvc/config.local`` is git-ignored, so a fresh clone (and the Colab VM)
    has no credential until this runs; the token is never written anywhere
    tracked.
    """
    from core.common import TRAIN_ROOT
    from core.env_file import EnvFile
    from training import dvc_store

    token = EnvFile.value("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required to publish run results through DVC")
    return dvc_store.configure_repo_remote(TRAIN_ROOT, token)


#: The DVC sidecar suffix. One declaration for pointer construction and lookup.
DVC_POINTER_SUFFIX = ".dvc"


def pointer_for(path: Path) -> Path:
    """The pointer DVC writes beside ``path``."""
    return Path(path).with_name(Path(path).name + DVC_POINTER_SUFFIX)


def _track(paths: list[Path]) -> list[Path]:
    """``dvc add`` each path and return the pointer DVC wrote beside it."""
    from core.common import TRAIN_ROOT

    resolved = [Path(path).resolve() for path in paths]
    for path in resolved:
        path.relative_to(TRAIN_ROOT.resolve())
    _dvc("add", *[_relative(path) for path in resolved])
    pointers = [pointer_for(path) for path in resolved]
    missing = [_relative(pointer) for pointer in pointers if not pointer.is_file()]
    if missing:
        raise RuntimeError(f"DVC created no pointer for: {missing}")
    return pointers


def _commit_pointers(pointers: list[Path], message: str) -> None:
    """Commit exactly the pointers (and the lane .gitignore DVC may amend)."""
    relatives = [_relative(pointer) for pointer in pointers]
    # Results stay git-ignored; the pointers are the deliberate exception.
    _git("add", "-f", "--", *relatives)
    lane_gitignore = pointers[0].parent / ".gitignore"
    if lane_gitignore.is_file():
        _git("add", "--", _relative(lane_gitignore))
    if _git("status", "--porcelain", "--", *relatives):
        _git("commit", "-m", message, "--", *relatives)


def _push_pointers(pointers: list[Path]) -> None:
    """Upload the exact pointers in ONE push and require a clean cloud compare."""
    from core.common import TRAIN_ROOT
    from training import dvc_store

    dvc_store.push_targets(TRAIN_ROOT, [_relative(pointer) for pointer in pointers])


def _drop_local_payload(path: Path) -> None:
    """Delete a pushed payload; its ``*.dvc`` pointer restores it with ``dvc pull``."""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish_result_paths(paths: list[Path], *, drop_local: bool = False) -> ResultReceipt:
    """DVC-track, commit and push run-result paths; optionally free their bytes.

    The suite result archives, the frozen input transports and the per-run trees
    all publish through this single flow: ``dvc add`` stages each payload, the
    objects are pushed and cloud-verified, and only then are the ``*.dvc``
    pointers committed. Git therefore never receives the bytes, and a pointer
    that is in Git is always retrievable with ``dvc pull``.

    ``drop_local`` is the retention choice: a completed run can keep only its
    pointers locally while the payload lives in the dagshub remote.
    """
    from core.common import TRAIN_ROOT

    started = time.monotonic()
    root = TRAIN_ROOT.resolve()
    resolved = [Path(path).resolve() for path in paths]
    if not resolved:
        raise ValueError("publish_result_paths requires at least one path")
    for path in resolved:
        path.relative_to(root)
    _authenticate()
    with Timing("retention.result").section("dvc_add"):
        pointers = _track(resolved)
    with Timing("retention.result").section("dvc_push"):
        _push_pointers(pointers)
    with Timing("retention.result").section("git_commit"):
        names = ", ".join(path.name for path in resolved)
        _commit_pointers(pointers, f"results: {names} -> dvc (payload in the dagshub remote)")
    dropped = bool(drop_local)
    if dropped:
        for path in resolved:
            _drop_local_payload(path)
    emit_timing(f"[timing] retention published {len(resolved)} path(s) dropped_local={dropped}")
    return ResultReceipt(
        paths=[_relative(path) for path in resolved],
        pointers=[_relative(pointer) for pointer in pointers],
        dropped_local=dropped,
        seconds=round(time.monotonic() - started, 3),
    )


#: Small files that stay in Git beside a DVC-managed payload (manifests/receipts,
#: the pointers themselves, and the per-lane .gitignore DVC writes).
GIT_KEPT_SUFFIXES = frozenset({".dvc", ".json", ".gitignore"})


def _payload_units(root: Path) -> list[Path]:
    """The DVC units under ``root``, never overlapping an existing pointer.

    A child is published as ONE unit unless it already contains DVC metadata (a
    run tree that carries its own pointers), in which case the walk descends:
    ``dvc add`` refuses two outputs in the same tracked directory, so the unit
    boundary follows the pointers already committed.
    """
    units: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.suffix in GIT_KEPT_SUFFIXES:
            continue
        if child.is_dir() and next(child.rglob(f"*{DVC_POINTER_SUFFIX}"), None) is not None:
            units.extend(_payload_units(child))
        else:
            units.append(child)
    return units


def publish_output_tree(root: Path, *, drop_local: bool = True) -> ResultReceipt:
    """DVC-publish every payload under ``root`` and free its bytes.

    The one call every lane's post-run step makes: one DVC unit per run/decision
    tree (or per large file), with the ``*.dvc`` pointers and the small
    manifests/receipts left in Git. ``drop_local`` is the retention choice; a
    tree that is already pointers-only publishes nothing.
    """
    from core.common import TRAIN_ROOT

    started = time.monotonic()
    root = Path(root).resolve()
    root.relative_to(TRAIN_ROOT.resolve())
    if not root.is_dir():
        raise FileNotFoundError(f"output root is missing: {root}")
    payloads = _payload_units(root)
    if not payloads:
        return ResultReceipt(paths=[], pointers=[], dropped_local=False,
                             seconds=round(time.monotonic() - started, 3))
    return publish_result_paths(payloads, drop_local=drop_local)


def declared_output_roots() -> list[Path]:
    """The config-declared DVC-managed output roots, resolved under TRAIN_ROOT."""
    from core.common import dvc_output_roots

    return [Path(root) for root in dvc_output_roots()]


def publish_declared_outputs(*, drop_local: bool = True) -> dict[str, ResultReceipt]:
    """Sweep every declared output root; the lane-agnostic post-run publication."""
    receipts: dict[str, ResultReceipt] = {}
    for root in declared_output_roots():
        if root.is_dir():
            receipts[_relative(root)] = publish_output_tree(root, drop_local=drop_local)
    return receipts


def _looks_like_run(path: Path) -> bool:
    """A run root is recognized by completion markers in its track dirs.

    Markers only ever live in track subdirectories one or two levels below
    the run root (`<run>/<track>/<marker>` or `<run>/worker_1/<track>/<marker>`):
    results/model_tracks/<id>/<track>/<inventory_file>,
    TRAINING_RESULTS/<run>/worker_1/<track>/<inventory_file> AND
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
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or not _looks_like_run(run_dir):
        raise FileNotFoundError(
            f"retention refuses: {run_dir} lacks track-subdir run markers {_run_markers()}"
        )
    _authenticate()
    with timing.section("dvc_add"):
        pointers = _track([run_dir])
    with timing.section("dvc_push"):
        # Push before committing the pointer: a pointer that is in Git is then
        # always retrievable from the remote, never a dangling reference.
        _push_pointers(pointers)
    with timing.section("git_commit"):
        _commit_pointers(
            pointers,
            f"results: training run {run_dir.name} -> dvc; local retention keeps only this run",
        )
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
    """DVC-publish the newest smoke run, then keep only it locally.

    Smoke runs are DVC-managed like every other run output (owner mandate
    2026-10-09 removed the exemption): the payload is pushed and its pointer
    committed FIRST, and only then are older local smoke runs whose pointers are
    already published freed. A sibling with no pointer was never published and is
    never deleted, so no local bytes are lost.
    """
    from core.common import TRAINING_RESULTS

    new_smoke_dir = Path(new_smoke_dir)
    if not _looks_like_run(new_smoke_dir) or not new_smoke_dir.name.startswith("smoke_"):
        raise FileNotFoundError(
            f"smoke retention refuses: {new_smoke_dir} is not a smoke_-run with track markers"
        )
    timing = Timing("retention.smoke")
    _authenticate()
    with timing.section("dvc_add"):
        pointers = _track([new_smoke_dir])
    with timing.section("dvc_push"):
        _push_pointers(pointers)
    with timing.section("git_commit"):
        _commit_pointers(pointers, f"results: smoke run {new_smoke_dir.name} -> dvc")
    for sibling in sorted(TRAINING_RESULTS.glob("smoke_*")):
        if sibling.is_dir() and sibling != new_smoke_dir and pointer_for(sibling).is_file():
            with timing.section(f"delete:{sibling.name}"):
                _run(["rm", "-rf", str(sibling)])
    emit_timing(f"[timing] retention smoke published: only {new_smoke_dir.name} remains local")
    return new_smoke_dir


def list_runs() -> list[RunIndexEntry]:
    """Enumerate pruned-away runs recoverable from the DVC remote.

    Primary source: the remote directory's .dvc sidecars (dvc list over the
    committed training-results tree). Each entry also carries the retention
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
    commit_ids = {}
    for row in commits.splitlines():
        head, _, message = row.partition(" ")
        if message.startswith("results: training run "):
            commit_id = message.removeprefix("results: training run ").split(" ", 1)[0]
            commit_ids[commit_id] = head[:10]
    for entry in entries:
        entry.commit = commit_ids.get(entry.run_id)
    return entries


def pull_run(run_id: str) -> Path:
    """OPERATOR-ONLY: restore one published run locally, on demand.

    DVC is write-only storage for every lane; this manual retrieval tool exists
    for the operator CLI (`... list` / `... pull <run_id>`) and no lane calls it.

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
    _authenticate()
    _dvc("pull", entry.dvc_file)
    restored = TRAINING_RESULTS / run_id
    if not _looks_like_run(restored):
        raise FileNotFoundError(f"pulled {run_id!r} but it lacks track markers: {restored}")
    return restored


def main() -> None:
    """CLI: `list`/`pull` published runs; `publish` a path; `publish-outputs` sweep."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["list", "pull", "publish", "publish-outputs"])
    parser.add_argument("run_id", nargs="?", default=None,
                        help="run id for `pull`; a repo-relative path for `publish`")
    arguments = parser.parse_args()
    if arguments.command == "list":
        entries = list_runs()
        for entry in entries:
            print(f"{entry.run_id}  (publish commit {entry.commit or 'unrecorded'})")
        if not entries:
            print("no published runs yet")
    elif arguments.command == "publish-outputs":
        receipts = publish_declared_outputs()
        print(json.dumps({root: receipt.model_dump() for root, receipt in receipts.items()},
                         indent=2), flush=True)
    elif arguments.command == "publish":
        if arguments.run_id is None:
            raise SystemExit("publish requires a repo-relative path")
        receipt = publish_result_paths([Path(arguments.run_id)])
        print(json.dumps(receipt.model_dump(), indent=2), flush=True)
    elif arguments.run_id is None:
        raise SystemExit("pull requires a run id (see `list`)")
    else:
        restored = pull_run(arguments.run_id)
        print(f"restored {restored}")


if __name__ == "__main__":
    main()
