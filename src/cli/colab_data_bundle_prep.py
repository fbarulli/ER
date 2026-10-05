"""CPU data-bundle prep lane (owner structural ruling 8).

Data-bundle production lives in its own lane file, exactly like the existing
standalone bundle lane (src/cli/colab_bundle.py): src/cli/colab.py is the
shared colab surface and gains only config-gated thin passthroughs.

This module owns the CPU-prep parity capabilities:

* launch capability  — the high-RAM machine shape (`colab new --high-mem`)
  for a FRESH CPU allocation, config-owned (colab.high_mem); an
  owner-launched named session is re-verified by the shared primitive
  (cli.colab.ensure_session) and never reallocated; a GPU accelerator is
  never reshaped by this lane.
* streaming          — the prep cell streams through cli.colab's proven
  transport with EVERY chunk forwarded to BOTH transcripts (root system
  log + training.log), the 1e18b36 [worker]-forwarding contract applied to
  this lane's own subprocess; [done] only on clean completion, [failed]
  otherwise (fail-loud, never retried).
* tqdm passthrough   — strict fd inheritance in the emitted prepare script
  (cli.colab.run_bundle, final 902689e state); this lane never captures
  stderr, so tqdm bars stream live through run_colab_exec_stream's \r
  handling. tqdm itself lives in the preparation code; nothing here
  duplicates progress rendering.
* cohort tagging     — every lane start prints the cohort tag (50pct/full)
  plus the export's sha256 prefix, so the two owner sessions (50% cohort VM
  and full cohort VM) are identifiable in one transcript.
* 2-parallel cap     — exactly TWO owner-launched high-RAM CPU prep VMs may
  exist (50pct + full cohorts); the lane never launches, retries, or
  replaces a VM and prints the cap at start.

The single production entry point is `main` (python -m
cli.colab_data_bundle_prep); `run_cpu_bundle_prep` is also the target of
cli.colab main's --what bundle thin passthrough when
config/training.yaml colab.cpu_data_bundle_lane is true.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from core.common import DATA_PATH, TRAIN_ROOT, training_cfg

import cli.colab as colab


# Owner cap: exactly two concurrent prep VMs, both owner-launched.
MAX_PARALLEL_SESSIONS = 2


def cpu_shape_args(accelerator: list[str]) -> tuple[str, ...]:
    """Shape args for the one fresh CPU allocation a lane may make.

    The CLI accepts `--high-mem` (requires the Colab Pro entitlement; L4/TPU
    runtimes ignore it).  The request is config-owned (config/training.yaml
    colab.high_mem) and applies ONLY to CPU sessions — a GPU accelerator
    stays governed by its own flags.  When the config does not request it
    the returned args are empty, so the emitted `colab new` command stays
    byte-identical to pre-parity behavior.
    """
    if accelerator or not bool(training_cfg().colab.high_mem):
        return ()
    return ("--high-mem",)


def cohort_label(dataset_csv: Path) -> str:
    """Per-VM cohort tag: `full` for the repo-root export, `50pct` for the
    50% cohort, else the sanitized stem — printed so the two parallel
    sessions are identifiable in one transcript."""
    name = Path(dataset_csv).name
    if name == "dataset.csv":
        return "full"
    if "50pct" in name:
        return "50pct"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0])


def _export_digest(dataset_csv: Path) -> str:
    digest = hashlib.sha256()
    with Path(dataset_csv).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _dual_transcript_streaming():
    """Forward every streamed chunk to BOTH transcripts for this lane.

    Wraps the shared transport (cli.colab.run_colab_exec_stream) for the
    duration of one prep run, forcing its existing training_output opt-in —
    the same forwarding contract the [worker] telemetry uses
    (1e18b36).  The transport itself is never reimplemented, and when the
    lane is not entered nothing in cli.colab changes behavior.
    """
    original = colab.run_colab_exec_stream

    def dual_transcript(*args, **kwargs):
        kwargs["training_output"] = True
        return original(*args, **kwargs)

    colab.run_colab_exec_stream = dual_transcript
    try:
        yield
    finally:
        colab.run_colab_exec_stream = original


def run_cpu_bundle_prep(dataset_csv: Path | None = None) -> None:
    """One CPU prep run with the parity capabilities layered on top of
    cli.colab.run_bundle (the untouched CSV-to-inputs lifecycle)."""
    source = Path(dataset_csv) if dataset_csv is not None else Path(DATA_PATH)
    print(
        f"[cpu-prep] cohort={cohort_label(source)} dataset={source.name} "
        f"sha256={_export_digest(source)[:12]} max_parallel_sessions={MAX_PARALLEL_SESSIONS}",
        flush=True,
    )
    with _dual_transcript_streaming():
        colab.run_bundle(dataset_csv=source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=None,
                        help="raw export to prepare (default: config/paths.yaml "
                             "dataset binding, the same default cli.colab uses)")
    args = parser.parse_args()
    # CPU production lane: the shared surface's accelerator global is set the
    # same way the standalone bundle lane sets it; the owner launches the
    # named high-RAM sessions and this lane reuses them by name.
    colab.GPU = "CPU"
    os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"
    try:
        run_cpu_bundle_prep(args.dataset_csv)
    except BaseException:
        print("\n[failed] cpu prep lane did not complete successfully", flush=True)
        raise
    print("\n[done] cpu prep lane completed and artifacts downloaded locally", flush=True)


if __name__ == "__main__":
    main()
