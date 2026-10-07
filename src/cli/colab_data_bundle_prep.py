"""CPU data-bundle prep lane (owner structural ruling 8).

Data-bundle production is the prep half of the consolidated Colab lane
(src/cli/colab_lane.py), exactly like the committed-export bundle lane
(src/cli/colab_bundle.py): cli.colab — the GPU lane — keeps only
config-gated thin passthroughs and the legacy ``--what bundle`` upload
lifecycle is never edited (owner ruling 910ee17).  The pinned capabilities:

* launch capability  — the high-RAM machine shape (`colab new --high-mem`)
  for a FRESH CPU allocation, config-owned (colab.high_mem); an
  owner-launched named session is re-verified by the shared primitive
  (cli.colab.ensure_session) and never reallocated; a GPU accelerator is
  never reshaped by this lane.
* streaming          — EVERY prep chunk reaches BOTH transcripts (root
  system log + training.log); [done] only on clean completion, [failed]
  otherwise (fail-loud, never retried).
* tqdm passthrough   — strict fd inheritance in the emitted prepare script
  (902689e final state); this lane never captures stderr. tqdm itself lives
  in the preparation code; nothing here duplicates progress rendering.
* cohort tagging     — every lane start prints the cohort tag (50pct/full)
  plus the export's sha256 prefix.
* 2-parallel cap     — exactly TWO owner-launched high-RAM CPU prep VMs may
  exist (50pct + full cohorts); the lane never launches, retries, or
  replaces a VM and prints the cap at start.

ISOLATION: this facade routes through cli.colab_lane's ColabCPULane, which
imports ONLY cli.colab's committed data-bundle production machinery plus
core.common config-path primitives; it never imports GPU-training runtime
modules (training.train, training.train_prepared, model_tracks.*, worker
paths). training.prepare_all runs as a REMOTE subprocess on the prep VM's
own checkout. Its config is the isolated cpu_bundle_prep: block.

The single production entry point is `main` (python -m
cli.colab_data_bundle_prep); `run_cpu_bundle_prep` is also the target of
cli.colab main's --what bundle thin passthrough when
config/training.yaml colab.cpu_data_bundle_lane is true.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import time
from contextlib import contextmanager
from pathlib import Path

from core.common import F, training_cfg

import cli.colab as colab
from cli.colab_lane import ColabCPULane, MAX_PARALLEL_PREP_SESSIONS as MAX_PARALLEL_SESSIONS

_LANES: dict[str, ColabCPULane] = {}


def lane() -> ColabCPULane:
    """The module's CPU prep lane instance (same class the bundle lane uses)."""
    if "prep" not in _LANES:
        _LANES["prep"] = ColabCPULane()
    return _LANES["prep"]


_MAX_PARALLEL_SESSIONS = MAX_PARALLEL_SESSIONS


def cpu_shape_args(accelerator: list[str]) -> tuple[str, ...]:
    """Shape args for the one fresh CPU allocation a lane may make.

    The request is config-owned (training.yaml cpu_bundle_prep.high_mem) and
    applies ONLY to CPU sessions — a GPU accelerator stays governed by its
    own flags.  When the config does not request it the emitted `colab new`
    command stays byte-identical to pre-parity behavior.
    """
    if accelerator or not bool(training_cfg().cpu_bundle_prep.high_mem):
        return ()
    return ("--high-mem",)


def cohort_label(dataset_csv: Path) -> str:
    """Per-VM cohort tag: `full` for the repo-root export, `50pct` for the
    50% cohort, else the sanitized stem."""
    name = Path(dataset_csv).name
    if name == "dataset.csv":
        return "full"
    if "50pct" in name:
        return "50pct"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0])


def _export_digest(dataset_csv: Path) -> str:
    return lane().export_digest(dataset_csv)


@contextmanager
def _dual_transcript_streaming():
    """Forward every streamed chunk to BOTH transcripts for this lane."""
    with lane().dual_transcript_streaming(colab):
        yield


def run_cpu_bundle_prep(dataset_csv: Path | None = None) -> None:
    """One CPU prep run with the parity capabilities layered on top of
    cli.colab.run_bundle (the untouched CSV-to-inputs lifecycle)."""
    lane().run_cpu_prep(dataset_csv)


def _qualify_session_transcripts(session: str) -> None:
    """Per-session root/training transcript paths for the 2-parallel cap."""
    lane().qualify_session_transcripts(F, session)




def _stamp() -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix."""
    return (f"[colab-data-bundle-prep {datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%dT%H:%M:%S %Z}]")




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=None,
                        help="raw export to prepare (default: config/paths.yaml "
                             "dataset binding, the same default cli.colab uses)")
    args = parser.parse_args()
    colab.GPU = "CPU"
    os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"
    session = os.environ.get("EUROMONITOR_COLAB_SESSION", colab.SESSION)
    _qualify_session_transcripts(session)
    colab.start_live_log()
    try:
        run_cpu_bundle_prep(dataset_csv=args.dataset_csv)
        print(_stamp(), "\n[done] cpu prep lane completed and artifacts downloaded locally",
              flush=True)
    except BaseException:
        print(_stamp(), "\n[failed] cpu prep lane did not complete successfully", flush=True)
        raise
    finally:
        colab.close_live_log()


if __name__ == "__main__":
    main()
