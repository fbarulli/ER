"""Standalone CPU bundle lane: one committed cohort export -> one delivery archive.

CPU LANE of the consolidated Colab lane (src/cli/colab_lane.py): this module
is the CLI facade. The committed-export delivery flow — sparse provisioning,
cohort remap onto dataset.csv, unbuffered file-telemetry prepare, log polling
with transient-probe tolerance, delivery archive, verified download — lives
in ColabCPULane; the GPU lane and the legacy ``--what bundle`` upload lane
stay owned by cli.colab (owner ruling 910ee17).

Provenance of the working implementation (kept verbatim at call
granularity): provisioning order (check_colab_cli -> ensure_session ->
prepare_remote_layout(minimal_runtime=True, sparse_paths=...) ->
install_deps(minimal_runtime=True)); no-upload cohort remap (b1f116f); the
detached POPEN+status-file prepare with local log polling (be6672b/58aa827)
streamed unbuffered (8b41dba) with pathlib-joined PYTHONPATH (1952b71);
delivery member list from the 8ddc614
lane; resume triplet (d264af1); delivery root contract (4d40d1e). The VM is
never stopped: the session stays open by owner directive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.common import TRAIN_ROOT
from core.run_log import RunLogger

import cli.colab as _colab
from cli.colab_lane import (
    ColabCPULane,
    DELIVERY_ARCHIVE_NAME,
    MAX_PARALLEL_PREP_SESSIONS,
    PREPARE_BUDGET_SECONDS,
)

_LANES: dict[str, ColabCPULane] = {}


def lane() -> ColabCPULane:
    """The module's CPU lane instance (one per facade; stateless transport)."""
    if "default" not in _LANES:
        _LANES["default"] = ColabCPULane()
    return _LANES["default"]


def _lane() -> ColabCPULane:
    return lane()


def _poll_prepare_log(deadline_seconds: int) -> None:
    lane().poll_prepare_log(deadline_seconds)


def _provision(dataset_csv: Path | None = None) -> None:
    lane().provision(dataset_csv)


def run_bundle(
    dataset_csv: Path,
    *,
    resume_from: str | None = None,
    resume_run_id: str | None = None,
    resume_state: Path | None = None,
) -> None:
    """Prepare on the VM CPU from the cloned cohort export, download the delivery.

    The chosen committed export rides the sparse checkout and the remote
    launcher remaps it onto dataset.csv (the kaggle lane's contract) — this
    lane uploads no raw export.  With --resume-from/--resume-run-id/
    --resume-state the frozen state tarball is uploaded as
    REMOTE_ROOT/resume_state.tar.zst; the delivery/download flow is unchanged.
    """
    lane().run_delivery(
        dataset_csv,
        resume_from=resume_from,
        resume_run_id=resume_run_id,
        resume_state=resume_state,
    )


def _resume_args(args: argparse.Namespace) -> dict:
    return lane().resume_args(args)


def _plan_only(args: argparse.Namespace) -> None:
    """Print the dry-run plan receipt without provisioning or executing."""
    print(json.dumps({
        "mode": "dry-run",
        "kind": "colab-cpu-bundle",
        "export": str(args.dataset_csv),
        "resume_from": args.resume_from,
        "resume_run_id": args.resume_run_id,
        "resume_state": str(args.resume_state) if args.resume_state else None,
        "archive_name": DELIVERY_ARCHIVE_NAME,
        "prepare_budget_seconds": PREPARE_BUDGET_SECONDS,
        "max_parallel_sessions": MAX_PARALLEL_PREP_SESSIONS,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-csv", type=Path, default=TRAIN_ROOT / "dataset.csv",
                    help="committed cohort export the sparse checkout carries and "
                         "the remote launcher remaps onto dataset.csv (default: "
                         "repo-root dataset.csv; config kaggle.export_csvs entries "
                         "only — no upload exists on this lane)")
    ap.add_argument("--resume-from", default=None,
                    help="prepare_all --resume-from choice for a frozen run "
                         "(dedupe|validation|full_bundle|suite_inputs; e.g. "
                         "validation re-runs the ~45 min token phase onward)")
    ap.add_argument("--resume-state", type=Path, default=None,
                    help="owner-built frozen state tarball (derived CSVs + "
                         "second04 pairs + stage manifests) uploaded and "
                         "extracted at REMOTE_ROOT before the pin re-write")
    ap.add_argument("--resume-run-id", default=None,
                    help="frozen training_prep run id to continue (e.g. "
                         "20261005T162137972376); prepare_all creates the run "
                         "dir itself on the VM")
    ap.add_argument("--preflight-only", action="store_true",
                    help="print the lane plan without contacting Colab "
                         "(kaggle-lane-style dry run; no default flips)")
    args = ap.parse_args()
    if args.preflight_only:
        _plan_only(args)
        return
    _colab.GPU = "CPU"
    os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"
    _provision(args.dataset_csv)
    run_bundle(args.dataset_csv, **_resume_args(args))


if __name__ == "__main__":
    main()
