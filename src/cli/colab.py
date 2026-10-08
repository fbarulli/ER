"""colab_backend.py — run EuromonitoR TRAIN work on a Colab GPU VM.

The Colab CLI (google-colab-cli) provisions a Colab runtime (T4 default —
free-tier GPU, enough for sentence-transformer fine-tuning), pushes code +
data, executes a lane, and pulls results back.

Lanes (post second-series rename — the old second03/second04 scripts are
now the src/training/ module chain):
  train  — GPU training only from LOCALLY PREPARED worker bundles
           (training.train_prepared; no data prep or masking on the VM —
           bundles are built beforehand with training.train
           --prepare-bundle or pulled checkout-native).
  hpo    — masking-enabled Optuna TPE search. Each trial trains on 50%,
           selects on the dev 25%, and does not read the test 25%.
  sims   — the configured zero-shot embedding lane. The current config uses
           minilm_l6 and scores against the same canonical fingerprint
           contract as training.
  smoke  — the chain check (fast verification the remote environment
           reproduces the local results contract). Runs on CPU by default and
           reads the full deduped CSV already in the cloned Colab checkout;
           no CSV or prepared bundle is uploaded. Its config-owned 128-row
           cap is applied only in memory by train.py, never by deleting or
           rewriting source rows.

Every lane reuses the shared bootstrap: the VM clones the configured public
training branch, regenerates all derived CSVs (byte-deterministic: canonicals
and gates reproduce identically), runs the lane, and pulls results back.

Usage:
  python colab_backend.py --what train
  python colab_backend.py --what train --train-frac 0.25 --epochs 2
  python colab_backend.py --what hpo
  python colab_backend.py --what hpo --resume-hpo
  er-colab --what hpo --gpu A100
  python colab_backend.py --what sims
  python colab_backend.py --what smoke
  python colab_backend.py --what stop
  python ... --keep-alive   # keep VM alive for debugging on failure
  python ... --what train --resume-run <run-id>
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache, wraps
import json
import os
import re
import shutil
import signal
import subprocess
import sys
# Capability-module identity anchor (phase-1 split of colab.py, see
# cli/colab_self_watch.py): whichever identity runs this file (`cli.colab`
# import or `__main__` under `python -m cli.colab`) registers the RUNNING
# module here, and split modules resolve it at call time — never importing a
# second copy.
sys.modules["__colab_runtime_self__"] = sys.modules[__name__]
from core.archive_reader import tar_archive
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
# AUDIT FIX (round 2 F15, round 3): RESULTS/DATA come from the config SSOT
# via lib.common (config/paths.yaml paths.results_dir/data_dir) — were
# re-derived inline (HERE / "artifacts" / "results"), a second declaration
# that happened to match today.
from core.common import (
    F,
    RESULTS,
    TRAINING_RESULTS,
    TRAIN_ROOT,
    embedding_model_keys,
    sweep_cfg,
    hpo_cfg,
    load_config,
    resolve_model,
    training_cfg,
)
from core.manifest import sha256_file
from core.bundle import CHECKPOINT_PREFIX
from core.run_log import RunLogger
from core.schemas import StageManifest, canonical_suite_matrix
from training.prepare_all_trace import timed
from cli.colab_lane import (
    DELIVERY_DATA_MEMBERS,
    DELIVERY_PREPARED_DIRS,
    DELIVERY_TRACKED_DIRS,
)
from cli.colab_lane_contracts import _stamp as _lane_stamp
from cli.log_capture import lane_log, progress_frames_to_lines

_LOG = RunLogger(__name__)

# The ONE per-run Colab lane transcript (owner order 2026-10-07): every lane
# surface — stdout/stderr system transcript, trainer/worker view, setup
# timing, and streamed per-stage output — folds into this single file under
# the canonical logs root, exactly like the Kaggle lane's lane.log.
LANE_LOG_NAME = "lane.log"

# Smoke and normal training defaults come from the Colab runtime config.
# Sweep fractions remain exclusive to the sweep lane.
_SMOKE_SAMPLE = int(sweep_cfg()["smoke_sample"])
_TRAIN_FRAC_DEFAULT = float(training_cfg().colab.train_fraction)
_EPOCHS_DEFAULT = int(training_cfg().training.epochs)
_TRAIN_LOSS = str(training_cfg().training.loss)
_RERANK_MODEL = str(sweep_cfg()["rerank_model"])

_COLAB = training_cfg().colab
_SIMS_MODEL = str(_COLAB.sims_model)
# Preparation / bundle archive names (config SSOT); the remote delivery and
# prepared-package discovery spell no path/archive literal.
_PREP_RUN_DIR_BASE = training_cfg().preparation.run_dir_base
_BUNDLE_ARCHIVE = training_cfg().kaggle.files.bundle_archive
REPOSITORY = _COLAB.repository
BRANCH = _COLAB.branch
GIT_REMOTE_NAME = _COLAB.git_remote_name
# Keep the config session as the default, while allowing concurrent launches
# to select an isolated named VM without editing the shared configuration.
SESSION = os.environ.get("EUROMONITOR_COLAB_SESSION", _COLAB.session)
GPU = _COLAB.gpu
REMOTE_ROOT = _COLAB.remote_root
_HPO_MODE = _COLAB.hpo_mode
_HPO_WORKERS = _COLAB.hpo_workers
_HPO_TRIAL_JOBS_DEFAULT = int(hpo_cfg()["n_jobs"])
_HPO_PERSISTENCE = str(hpo_cfg()["persistence"])
_TRAIN_WORKERS = _COLAB.train_workers
_SMOKE_WORKERS = _COLAB.smoke_workers
_MIXED_TRAIN_WORKERS = _COLAB.mixed_train_workers
_MIXED_SIMS_WORKERS = _COLAB.mixed_sims_workers
_MIXED_MINING_PROFILE = _COLAB.mixed_mining_profile
# The VM's asserted distributions and installer preference are config-owned
# (config/training.yaml colab.runtime_packages / colab.prefer_uv_install):
# trimming or re-pinning the remote stack must not require editing this file.
_RUNTIME_PACKAGES = _COLAB.runtime_packages
_PREFER_UV_INSTALL = bool(_COLAB.prefer_uv_install)
# Reusing a prepared bundle whose inputs are byte-identical saves the whole
# local build (531 s measured cold).  Config-owned so it can be turned off
# when a lane needs to prove a bundle was built rather than reused.
_CACHE_PREPARED_BUNDLES = bool(_COLAB.cache_prepared_bundles)
_MASKING_ENABLED = training_cfg().masking.enabled
_MASKING_PROFILE = str(training_cfg().masking.profile)
_COLLAPSE_GUARDRAIL_PROFILE = str(training_cfg().collapse_guardrail.profile)
_LOG_POLL_SECONDS = _COLAB.log_poll_seconds
_LOG_POLL_INITIAL_SECONDS = float(_COLAB.log_poll_initial_seconds)
_PROBE_TIMEOUT_SECONDS = _COLAB.probe_timeout_seconds
_PROBE_RETRIES = _COLAB.probe_retries
_PROBE_RETRY_BACKOFF_SECONDS = _COLAB.probe_retry_backoff_seconds
_MASK_EFFECT_AFTER_TRAIN = _COLAB.mask_effect_after_train
_SMOKE_EPOCHS = _COLAB.smoke_epochs
_WORKER_TIMEOUT_SECONDS = _COLAB.worker_timeout_seconds
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = _COLAB.result_download_timeout_seconds
_RESULT_DOWNLOAD_HEARTBEAT_SECONDS = _COLAB.result_download_heartbeat_seconds
_REMOTE_UPLOAD_RETRIES = _COLAB.remote_upload_retries
_RESULT_ARCHIVE_NAME = _COLAB.result_archive_name
_RESULT_MANIFEST_NAME = _COLAB.result_manifest_name
_RESULT_DOWNLOAD_EXCLUDED_DIRS = frozenset(_COLAB.result_download_excluded_dirs)
_RESULT_EVENTS_FILE = _COLAB.result_events_file
# Poll interval for the incremental result sync that runs during training.
# Small enough that a finished checkpoint is on the laptop well before the run
# ends, large enough that the remote listing does not compete with the trainer
# for the control channel.
_INCREMENTAL_SYNC_SECONDS = _COLAB.incremental_sync_seconds
# The trainer writes this file last inside a checkpoint directory, so its
# presence is what distinguishes a finished checkpoint from one mid-write.
_CHECKPOINT_MANIFEST_NAME = _COLAB.checkpoint_manifest_name
# Checkpoints live at
# ``worker_N/_checkpoints/<model>/<run>_f0/checkpoint-<step>/<file>``, so a
# checkpoint file is four levels below the ``_checkpoints`` root; that is the
# only depth either the step lookup or the file listing needs, because a
# checkpoint can only be identified by listing something inside it.  The walk
# is bounded to stay inside the exec budget however large the run's log and
# wandb trees grow -- an unbounded walk is what timed out on T4.
_CHECKPOINT_LISTING_DEPTH = 4
# The directory holding every checkpoint of a run, directly under a worker.
_CHECKPOINT_ROOT_NAME = training_cfg().bundle.checkpoint_dir
# The ``checkpoint-*/trainer_state.json`` glob fragment (config SSOT).
_CHECKPOINT_STATE_GLOB = f"{CHECKPOINT_PREFIX}*/{training_cfg().bundle.trainer_state_file}"
# Bookkeeping written beside a locally retained checkpoint, recording which
# remote checkpoint it is and the score that won it the slot.
_LATEST_BEST_MARKER = _COLAB.latest_best_marker
_WORKER_MONITOR_SECONDS = _COLAB.worker_monitor_seconds
_FINAL_INFERENCE = _COLAB.final_inference

# ── scored-pair validation row contract (reconciled 2026-10-01) ─────────────
# The row counts are NOT module constants: the scored population is the SSOT
# files.final_validation binding (core.common, data/final_validation.csv)
# plus the fold map, and both are re-measured from the artifacts at exec
# time, byte-stability asserted, by training.complete_colab_worker's census.
# Reconciled live numbers for the 2026-10-01 regen: source census 71,623
# (config audit pin) = deduped 63,079 + dropped 8,544; fold map 14,946
# entities (7,452 fold 0 train side, 3,782 + 3,712 folds 2+3); scored
# population 6,351 pairs (565 pos / 5,786 neg); train side
# 63,079 - 13,927 validation-entity rows = 49,152.

@lru_cache(maxsize=1)

def _scored_validation_census() -> dict[str, object]:
    """The exec-time scored-pair row census and its identity (fail-loud)."""
    from training.complete_colab_worker import scored_validation_accounting

    return scored_validation_accounting()

_HPO_RESUME_DIR = TRAINING_RESULTS / "hpo_resume"
# The installed Colab CLI writes its diagnostic log under $HOME even when a
# config path is supplied. This workspace's home is read-only, so isolate the
# CLI state/history in a visible, root-local folder for every launcher run.
_COLAB_CLI_STATE_DIR = TRAIN_ROOT / "colab_cli_state"
_COLAB_CLI_CONFIG = _COLAB_CLI_STATE_DIR / "sessions.json"
_COLAB_CLI_ENTRYPOINT = Path(__file__).with_name("colab_cli_entry.py")
LIVE_LOG_PATH: Path | None = None
TRAINING_LOG_PATH: Path | None = None
SETUP_TIMING_LOG_PATH: Path | None = None
_setup_timing_active = False
_setup_timing_lock = threading.Lock()
_live_log = None
_training_log = None
_training_log_lock = threading.Lock()
_result_event_lock = threading.Lock()
_original_stdout = None
_original_stderr = None
_SUPPRESS_LIVE_LOG = False


def _suite_config_path() -> Path:
    """The default all-track suite config, resolved from config SSOT.

    ``BundleSpec.suite_config`` (config/training.yaml ``bundle.suite_config``)
    owns the default; a relative value resolves against ``TRAIN_ROOT`` exactly
    as the former hardcoded suite-config path did, so the resolved path is
    unchanged for the shipped config.
    """
    configured = Path(training_cfg().bundle.suite_config)
    return configured if configured.is_absolute() else TRAIN_ROOT / configured


def _legacy_validation_sources() -> dict[str, Path]:
    """Materialize listing partitions from the validated shared component split.

    Training consumes the full prepared catalog and applies prepared_holdout;
    these listing CSVs are solely the final-inference provenance populations.
    """
    import pandas as pd
    from model_tracks.config import load_config as load_suite
    from model_tracks.preflight import preflight as suite_preflight
    from graph_tracks.data import file_hash
    config = _suite_config_path()
    suite_preflight(config)
    suite = load_suite(config)
    setup = (TRAIN_ROOT / suite.setup_dir).resolve()
    layout = training_cfg().preparation.graph_setup
    catalog_path = setup / layout.catalog
    input_manifest = json.loads((setup / layout.prepared_dir / layout.input_manifest).read_text())
    if file_hash(catalog_path) != input_manifest['catalog_sha256']:
        raise ValueError('eligible catalog differs from prepared graph inputs')
    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    splits = pd.read_csv(setup / layout.splits, dtype=str, keep_default_na=False)
    if catalog.sku_id.duplicated().any() or splits.sku_id.duplicated().any():
        raise ValueError('component listing IDs must be unique')
    if set(catalog.sku_id) != set(splits.sku_id):
        raise ValueError('component split must cover the eligible catalog exactly')
    if not set(splits.split) <= {'train', 'dev', 'test'}:
        raise ValueError('unknown component split role')
    roles = splits.set_index('sku_id').split
    assignments = catalog.sku_id.map(roles)
    training = catalog.loc[assignments.eq('train')]
    holdout = catalog.loc[assignments.isin(['dev', 'test'])]
    from training.folds import normalize_gtin
    train_entities = set(training.gtin.map(normalize_gtin))
    held_entities = set(holdout.gtin.map(normalize_gtin))
    if train_entities & held_entities:
        raise ValueError('component train entities overlap inference holdout')
    if training.empty or holdout.empty:
        raise ValueError('component train and inference populations must be nonempty')
    from core.identity_policy import reviewed_row_mask
    if reviewed_row_mask(catalog).any():
        raise ValueError('component inference catalog contains reviewed exclusions')
    folder = TRAIN_ROOT / 'results/prepared_training/component_validation'
    folder.mkdir(parents=True, exist_ok=True)
    sources = {'source': folder / layout.catalog,
               'training': folder / 'component_train.csv',
               'sample': folder / 'component_holdout.csv'}
    for key, frame in [('source', catalog), ('training', training), ('sample', holdout)]:
        # Atomic replacement keeps the prewarm uploader from reading a partial CSV.
        target = sources[key]
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
    return sources

def _validate_legacy_bundle_partitions(bundles: list[Path]) -> None:
    """Reject cached or sampled bundles using a different component holdout."""
    import pandas as pd
    from model_tracks.config import load_config as load_suite
    from training.prepared_bundle import load_prepared_bundle, prepared_holdout
    from training.folds import normalize_gtin
    from core.common import SEED
    suite = load_suite(_suite_config_path())
    setup = TRAIN_ROOT / suite.setup_dir
    layout = training_cfg().preparation.graph_setup
    catalog = pd.read_csv(setup / layout.catalog, dtype=str, keep_default_na=False)
    assignments = pd.read_csv(setup / layout.splits, dtype=str).set_index('sku_id').split
    for path in bundles:
        _, data = load_prepared_bundle(path)
        populations = prepared_holdout(data, dict(training_cfg().split), seed=SEED)
        roles = {normalize_gtin(value): role for role, values in
                 zip(('train', 'dev', 'test'), populations) for value in values}
        for row in catalog.itertuples(index=False):
            if roles.get(normalize_gtin(row.gtin)) != assignments[row.sku_id]:
                raise ValueError(
                    'legacy prepared bundle differs from the shared component split; '
                    'use --tracks-config results/model_tracks/smoke_20261001_128/suite.yaml '
                    'for a sampled CPU smoke, or rebuild full bundles from the current catalog')

def training_lifecycle_preflight(
    *, workers: int, model: str | None, masking_profile: str,
    train_only: bool = False,
) -> dict[str, object]:
    """Validate the scored-pair validation contract without contacting Colab.

    The scored population is the SSOT final_validation binding (the
    merged-graph folds 2+3 scored pairs), so the row accounting here is
    derived from the artifacts at exec time — never hardcoded — and closes
    on the identity `scored_pair_validation_census` asserts before the dict
    is built (train side + validation entities == deduped; deduped + dropped
    == the 71,623 source-export census pin).
    """
    training_path = _validation_input_path(_COLAB.training_dataset_csv)
    profiles = _expand_worker_profiles(masking_profile, workers, "masking")
    model_key = model or str(training_cfg().training.base_model)
    census = _scored_validation_census()
    return {
        "contacts_colab": False,
        "workers": workers,
        "model": model_key,
        "masking_profiles": profiles,
        "split_protocol": "merged component graph (folds 2+3 scored pairs)",
        "training_dataset": str(training_path),
        "training_rows": census["train_side_rows"],
        "validation_entity_rows": census["validation_entity_rows"],
        "deduped_rows": census["deduped_rows"],
        "dropped_rows": census["dropped_rows"],
        "source_census_rows": census["source_export_rows"],
        "inference_dataset": census["scored_population_path"],
        "inference_rows": census["scored_pair_rows"],
        "identity_closes": True,
        "train_only": train_only,
        "final_inference_enabled": not train_only,
        "prepared_train_argv": [
            sys.executable, "-u", "-m", "training.train", "--model", model_key,
            "--dataset", str(training_path), "--prepare-bundle", "<worker-bundle>",
        ],
        "remote_completion_argv": (
            None if train_only else [
                "<remote-python>", "-m", "training.complete_colab_worker",
                "--validation-input", "<final_validation scored population>",
                "--training-input", "<dataset_deduped training census>",
            ]
        ),
        "successful_worker_order": (
            ["train", "write_success_status", "download", "teardown"]
            if train_only else
            ["train", "resolve_best_checkpoint", "heldout_sku_inference",
             "write_success_status", "download", "teardown"]
        ),
    }



def _stamp() -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix for output."""
    return _lane_stamp("colab", now=datetime.now(ZoneInfo('Europe/Paris')))


class _Tee:
    """Mirror launcher output to the terminal and the root live log."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        if not _SUPPRESS_LIVE_LOG:
            # The system tee and the trainer writer share ONE lane handle, so
            # both lock around the file to keep lines from interleaving.
            # tqdm CR frames must survive capture as grep-able lines.
            with _training_log_lock:
                self._log_file.write(progress_frames_to_lines(text))
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        if not _SUPPRESS_LIVE_LOG:
            with _training_log_lock:
                self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

class _LiveLogSuppressed:
    """Temporarily keep streamed worker training out of the system log."""

    def __enter__(self):
        global _SUPPRESS_LIVE_LOG
        self._previous = _SUPPRESS_LIVE_LOG
        _SUPPRESS_LIVE_LOG = True

    def __exit__(self, exc_type, exc_value, traceback_value):
        global _SUPPRESS_LIVE_LOG
        _SUPPRESS_LIVE_LOG = self._previous
        return False

def _write_training_log(text: str) -> None:
    """Write the trainer/worker view into the shared per-run lane transcript.

    The same handle backs the system tee (see start_live_log), so the trainer
    view and the stdout/stderr transcript are one file, never two.
    """
    if _training_log is None or not text:
        return
    with _training_log_lock:
        # CR-separated tqdm frames become grep-able lines (shared formatter).
        _training_log.write(progress_frames_to_lines(text))
        _training_log.flush()

def _result_event(
    run_id: str,
    stage: str,
    state: str,
    *,
    worker: int | None = None,
    **details: object,
) -> None:
    """Persist ordered result-transfer state without measuring duration."""
    root = TRAINING_RESULTS / run_id
    root.mkdir(parents=True, exist_ok=True)
    event = {"stage": stage, "state": state, **details}
    if worker is not None:
        event["worker"] = int(worker)
    event_path = root / _RESULT_EVENTS_FILE
    with _result_event_lock:
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    suffix = f" worker={worker}" if worker is not None else ""
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    print(_stamp(), f"[result-state] {stage} {state}{suffix}" + (f" | {detail_text}" if detail_text else ""), flush=True)

def _record_remote_run(remote_base: str, *, workers: int, lane: str) -> None:
    """Persist the remote location before uploads or training begin."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    root = TRAINING_RESULTS / run_id
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "remote_base": remote_base,
        "run_id": run_id,
        "session": SESSION,
        "workers": workers,
    }
    (root / "remote_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(_stamp(), f"[run] remote metadata recorded -> {root / 'remote_run.json'}", flush=True)

@contextmanager
def _colab_timing(kind: str, name: str):
    """Emit monotonic wall times to stdout and the launcher's durable transcript."""
    started = time.perf_counter()
    def emit(message):
        # Timing is part of the one lane transcript: the flush=True print is
        # captured by the system tee, whose handle IS SETUP_TIMING_LOG_PATH
        # (logs/colab/lane.log).  No second handle can clobber the transcript.
        print(f"{_stamp()} {message}", flush=True)

    emit(f"[timing] {kind}={name} state=started")
    state = "completed"
    stopped = threading.Event()

    def report_wait():
        while not stopped.wait(30):
            emit(
                f"[timing] {kind}={name} state=running "
                f"elapsed_seconds={time.perf_counter() - started:.3f}",
            )

    heartbeat = threading.Thread(target=report_wait, daemon=True)
    heartbeat.start()
    try:
        yield
    except BaseException:
        state = "failed"
        raise
    finally:
        stopped.set()
        heartbeat.join()
        elapsed = time.perf_counter() - started
        emit(f"[timing] {kind}={name} state={state} elapsed_seconds={elapsed:.3f}")
        if kind == 'step' and name == 'initialization':
            _finish_setup_timing()

def _finish_setup_timing():
    global _setup_timing_active
    _setup_timing_active = False

def _timed_colab(kind: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            name = function.__name__
            if name == "colab" and args:
                name = f"cli.{args[0]}"
            elif name == "run_colab_exec_stream":
                name = f"exec.{kwargs.get('log_name') or 'unlabelled'}"
            elif name == "run_detached_stage" and args:
                name = f"detached.{args[0]}"
            elif name == "_upload_with_retries" and args:
                name = f"upload.{Path(args[0]).name}"
            with _colab_timing(kind, name):
                return function(*args, **kwargs)
        return wrapped
    return decorate

@_timed_colab("step")
def check_colab_cli() -> None:
    """Ensure the colab CLI is installed and authenticated."""
    try:
        colab("--help")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "colab CLI not found or not authenticated.\n"
            "Run: uv tool install google-colab-cli\n"
            "Then: colab sessions  (to complete OAuth sign-in)"
        )

# Launcher session lock moved to cli.colab_launch (split phase);
# re-exported so the legacy `from cli import colab` surface is unchanged.
from cli.colab_launch import (  # noqa: E402,F401
    _colab_launch_lock_is_held,
    _colab_launch_lock_path,
    _process_start_ticks,
    _read_colab_launch_owner,
    acquire_colab_launch_lock,
    release_colab_launch_lock,
)


# Colab CLI transport moved to cli.colab_transport (split phase);
# re-exported so the legacy `from cli import colab` surface and its
# monkeypatch needles are unchanged.
from cli.colab_transport import (  # noqa: E402,F401
    _colab_command,
    _download_file_with_visibility,
    _format_bytes,
    _local_file_size,
    _parse_remote_json,
    _serialize_colab_control,
    _upload_with_retries,
    colab,
    run_colab_exec_capture,
    run_colab_exec_stream,
    run_detached_stage,
)


@timed
def run_parallel_train_and_tail(
    args: list[str], workers: int, *, resume_run: str | None = None,
    smoke: bool = False,
    run_labels: list[str] | None = None,
    worker_losses: list[str] | None = None,
    masking_profiles: list[str] | None = None,
    collapse_guardrail_profiles: list[str] | None = None,
    prepared_bundles: list[Path] | None = None,
    final_inference: bool = True,
    inference_sample: int | None = None,
    inference_device: str | None = None,
    remote_checkout_inputs: bool = False,
    remote_checkout_bundles: list[str] | None = None,
    remote_validation_csv: str | None = None,
    incremental_sync: bool = True,
) -> tuple[str, int]:
    """Run isolated full-data trainers concurrently and mirror worker logs."""
    if worker_losses is not None and len(worker_losses) != workers:
        raise ValueError(
            f"worker loss list must contain exactly {workers} values; "
            f"got {len(worker_losses)}"
        )
    stamp = _lane_run_stamp()
    remote_base = (
        f"{REMOTE_ROOT}/results/concurrent_train_{resume_run}"
        if resume_run
        else f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    )
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    _record_remote_run(remote_base, workers=workers, lane="train")
    if prepared_bundles is not None and remote_checkout_bundles is not None:
        raise ValueError("prepared bundles must be uploaded or checkout-native, not both")
    if remote_checkout_bundles is not None:
        if len(remote_checkout_bundles) != workers:
            raise ValueError("checkout bundle list must contain one bundle per worker")
        remote_bundles = []
        for raw in remote_checkout_bundles:
            relative = Path(raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("checkout bundle path must stay inside the checkout")
            remote_bundles.append(str(Path(REMOTE_ROOT) / relative))
    elif prepared_bundles is not None:
        remote_bundles = _upload_prepared_bundles(run_id=run_id, bundles=prepared_bundles)
    else:
        raise ValueError(
            "no prepared bundles supplied: Colab is a GPU-training-only lane — "
            "build worker bundles locally beforehand (training.train "
            "--prepare-bundle) or pass checkout-native bundles; on-the-fly "
            "data prep and masking are no longer permitted on the VM"
        )
    if not final_inference:
        remote_validation_inputs = {"sample": "", "source": "", "training": ""}
    elif remote_validation_csv is not None:
        remote_validation_inputs = {
            "sample": remote_validation_csv,
            "source": remote_validation_csv,
            "training": remote_validation_csv,
        }
    else:
        remote_validation_inputs = _upload_validation_inputs(run_id)
    remote_input_loop = (
        "for name in ():"
        if prepared_bundles is not None or remote_checkout_inputs
        else 'for name in (F["canonical_records"], F["gate_results"], F["labeled_pairs"]):'
    )
    launch = _BOOTSTRAP + _remote_auth_env_script(
        # train_prepared is deliberately remote-only and refuses to run
        # without W&B.  A Git-shipped bundle changes transport, not tracking.
        include_wandb=True
    ) + f"""
import base64, json, os, pathlib, shutil, shlex, subprocess, sys, time, traceback
from core.common import F
from core.tracing import run_trace_env
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
run_id = base.name.removeprefix("concurrent_train_")
base.mkdir(parents=True, exist_ok={bool(resume_run)!r})
run_labels = {run_labels!r}
worker_losses = {worker_losses!r}
masking_profiles = {masking_profiles!r}
collapse_guardrail_profiles = {collapse_guardrail_profiles!r}
remote_bundles = {remote_bundles!r}
inference_sample = {inference_sample!r}
inference_device = {inference_device!r}
started = []
for number in range(1, {workers} + 1):
    worker_profile = (
        run_labels[number - 1]
        if run_labels and number <= len(run_labels)
        else ""
    )
    training_name = f"{{run_id}}-worker_{{number}}" + (f"-{{worker_profile}}" if worker_profile else "")
    worker_profile = (
        worker_profile if worker_profile in ("mining_enabled", "masking_only") else ""
    )
    out = base / f"worker_{{number}}"
    print(f"[resume-preflight] worker {{number}}: preparing {{out}}", flush=True)
    if {bool(resume_run)!r}:
        out.mkdir(exist_ok=True)
        {remote_input_loop}
            relative = name.relative_to(root / "results")
            source = root / "results" / relative
            destination = out / relative
            if not destination.is_file():
                if not source.is_file():
                    raise FileNotFoundError(f"resume worker input missing: {{source}}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        checkpoints = list(out.rglob({_CHECKPOINT_STATE_GLOB!r}))
        if not checkpoints:
            raise FileNotFoundError(f"[resume-preflight] worker {{number}} has no local checkpoint; restore downloaded checkpoint files before resuming")
    else:
        out.mkdir()
        if {remote_bundles is None and not remote_checkout_inputs!r}:
            for name in (
                F["canonical_records"],
                F["gate_results"],
                F["labeled_pairs"],
            ):
                relative = name.relative_to(root / "results")
                source = root / "results" / relative
                if not source.is_file():
                    raise FileNotFoundError(f"worker input missing: {{source}}")
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    worker_args = [sys.executable, *{args!r}]
    if worker_losses is not None:
        loss_index = worker_args.index("--loss") + 1
        worker_args[loss_index] = worker_losses[number - 1]
    worker_args[worker_args.index("training.train")] = "training.train_prepared"
    worker_args.extend(["--bundle", remote_bundles[number - 1]])
    command = " ".join(shlex.quote(part) for part in worker_args)
    # SCORED-PAIR contract (2026-10-01): --validation-input is the VM-side
    # SSOT final_validation binding, not the staged component holdout. These
    # are interpolated verbatim so the remote script resolves ITS F at exec
    # time (a local absolute path would not exist on the VM).
    validation_input_arg = 'str(F["final_validation"])'
    training_input_arg = 'str(F["dataset_deduped"])'
    completion_args = [
        sys.executable, "-m", "training.complete_colab_worker",
        "--source", str(out), "--run-id", run_id, "--worker", str(number),
        "--validation-input", {validation_input_arg},
        "--validation-source", {remote_validation_inputs['source']!r},
        "--training-input", {training_input_arg},
    ]
    if inference_sample is not None:
        completion_args.extend(["--sample", str(inference_sample)])
    if inference_device is not None:
        completion_args.extend(["--device", inference_device])
    completion_args.append("--skip-dvc")
    completion_command = " ".join(shlex.quote(part) for part in completion_args)
    completion_clause = (
        f'if [ "$rc" -eq 0 ]; then echo "[worker-process] training complete; running validation inference"; {{completion_command}}; rc=$?; fi; '
        if {final_inference!r} else ""
    )
    log_path, status_path = out / "training.log", out / "training.status"
    live_status_path = out / "live_status.json"
    wandb_dir = out / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src"), "EUROMONITOR_RESULTS_DIR": str(out),
           "WANDB_DIR": str(wandb_dir),
           "WANDB_RUN_NAME": training_name,
           "EUROMONITOR_RUN_ID": training_name,
           "EUROMONITOR_MINING_PROFILE": worker_profile,
           "EUROMONITOR_REMOTE_TRAINING": "1", "EUROMONITOR_DISABLE_DVC_CHECKPOINTS": "1", **run_trace_env(lane=training_name)}}
    live_status_path.write_text(json.dumps({{
        "updated_at": time.time(), "event": "launched", "step": 0,
        "wandb_run_name": env["WANDB_RUN_NAME"],
    }}) + "\\n", encoding="utf-8")
    process_log = out / "processes.log"
    # Every worker launch owns a fresh diagnostics log.  Resume restores
    # checkpoints, not log history; stale snapshots from an earlier attempt
    # must not be mistaken for this run's lifecycle.
    process_log.write_text("", encoding="utf-8")
    ps_command = f"ps -eo pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args >> {{shlex.quote(str(process_log))}} 2>&1"
    diagnostics = "free -h || true; nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true"
    wrapped = (
        f"echo '[worker-process] starting pid=$$'; {{ps_command}}; "
        f"echo '[worker-process] resource snapshot before training'; {{diagnostics}}; "
        f"timeout --signal=TERM --kill-after=60 {_WORKER_TIMEOUT_SECONDS} {{command}}; rc=$?; "
        f"{{completion_clause}}"
        f"echo '[worker-process] exited rc='$rc; {{ps_command}}; "
        f"echo '[worker-process] resource snapshot after training'; {{diagnostics}}; "
        f"printf '%s\\n' \\"$rc\\" > {{shlex.quote(str(status_path))}}; exit $rc"
    )
    # Resume restores model state only.  Worker training logs always start
    # fresh so a new attempt cannot append to a prior run's output.
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        child = subprocess.Popen(["/bin/bash", "-lc", wrapped], cwd=root, env=env,
            stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True)
    print(f"[train-launch] worker {{number}} started pid={{child.pid}}", flush=True)
    started.append({{"worker": number, "pid": child.pid}})
print(json.dumps({{"base": str(base), "workers": started}}), flush=True)
"""
    print(_stamp(), f"[run] starting {workers} isolated full-data trainers; streaming all worker logs ...", flush=True)
    # Resume validates each worker's local checkpoint before
    # it emits the launch JSON. A full checkpoint pull can legitimately take
    # longer than the short probe budget, so use the configured worker
    # timeout for this one-time preflight.
    launched = _parse_remote_json(
        run_colab_exec_capture(SESSION, launch, timeout=_WORKER_TIMEOUT_SECONDS)
    )
    print(_stamp(), f"[train] remote workers={launched['workers']} base={launched['base']}", flush=True)
    # Fetch finished artifacts from every worker while they train, so the end
    # of the run is a short delta rather than the whole result set.  Stopped
    # before the authoritative download so the two cannot race on one file.
    syncer = (
        _IncrementalResultSync(remote_base, run_id, workers=workers)
        if incremental_sync else None
    )
    if syncer is not None:
        syncer.start()
    offsets = {str(item["worker"]): 0 for item in launched["workers"]}
    live_signatures: dict[str, str] = {}
    try:
        while True:
            probe = _BOOTSTRAP + f"""
import json, pathlib
base = pathlib.Path({remote_base!r})
offsets = {offsets!r}
payload = {{"offsets": {{}}, "chunks": {{}}, "status": {{}}, "resume": {{}}, "live": {{}}}}
for number in range(1, {workers} + 1):
    key = str(number)
    out = base / f"worker_{{number}}"
    log_path, status_path = out / "training.log", out / "training.status"
    live_status_path = out / "live_status.json"
    offset = int(offsets.get(key, 0))
    data = b""
    if log_path.is_file():
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    payload["offsets"][key] = offset + len(data)
    payload["chunks"][key] = data.decode("utf-8", errors="replace")
    payload["status"][key] = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else None
    if live_status_path.is_file():
        try:
            payload["live"][key] = json.loads(live_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
payload["done"] = all(value is not None for value in payload["status"].values())
print(json.dumps(payload), flush=True)
"""
            try:
                payload = _parse_remote_json(
                    run_colab_exec_capture(
                        SESSION, probe, timeout=_PROBE_TIMEOUT_SECONDS,
                        training_output=True,
                    )
                )
            except RuntimeError as exc:
                # The trainer is detached and continues writing remotely.  A
                # transient empty/control-channel reply must not turn a log
                # read into a training failure followed by VM teardown.
                # A lost kernel or missing session is not transient: there
                # can be no remote worker left to poll.  Propagate it so
                # main's finally tears down local state and releases the
                # session lock for the next launch.
                detail = str(exc).lower()
                if (
                    "connection was lost" in detail
                    or f"session '{SESSION}' not found".lower() in detail
                ):
                    raise
                message = f"[probe] log/status unavailable; continuing worker: {exc}"
                _write_training_log(message + "\n")
                print(f"{_stamp()} {message}", flush=True)
                time.sleep(_LOG_POLL_SECONDS)
                continue
            offsets = {str(key): int(value) for key, value in payload["offsets"].items()}
            for worker, live in payload["live"].items():
                signature = json.dumps(live, sort_keys=True)
                if live_signatures.get(worker) == signature:
                    continue
                live_signatures[worker] = signature
                metrics = []
                for key, label in (("train_loss", "train_loss"), ("dev_average_precision", "dev_ap"),
                                   ("dev_accuracy", "dev_acc")):
                    if live.get(key) is not None:
                        metrics.append(f"{label}={float(live[key]):.4f}")
                if live.get("rss_mb") is not None:
                    metrics.append(f"rss={float(live['rss_mb']):.0f}MB")
                if live.get("gpu_free_gb") is not None:
                    metrics.append(
                        f"gpu={float(live.get('gpu_allocated_gb', 0)):.2f}G alloc/"
                        f"{float(live['gpu_free_gb']):.2f}G free"
                    )
                position = f"step {live.get('step', 0)}/{live.get('max_steps', '?')}"
                print(_stamp(), f"[worker {worker}] {live.get('event', 'running')} | {position}" +
                      (" | " + " | ".join(metrics) if metrics else "") +
                      (f" | W&B {live['wandb_url']}" if live.get("wandb_url") else ""), flush=True)
            for worker, chunk in payload["chunks"].items():
                # Forward the complete worker log into the one lane transcript
                # (logs/colab/lane.log): the single chronological record of
                # every Colab stage, including the trainer/worker view.
                for line in str(chunk).splitlines():
                    _write_training_log(f"[worker {worker}] {line}\n")
                    print(f"[worker {worker}] {line}", flush=True)
            if payload["done"]:
                failed = {worker: rc for worker, rc in payload["status"].items() if int(rc) != 0}
                if failed:
                    raise RuntimeError(f"parallel trainers failed: {failed}")
                download_verified_training_results(remote_base, workers, smoke=smoke)
                print(_stamp(), f"[train] all {workers} remote workers completed successfully", flush=True)
                return remote_base, workers
            time.sleep(_LOG_POLL_SECONDS)
    finally:
        if syncer is not None:
            syncer.stop()

# Result transfer + verification moved to cli.colab_result_sync (split
# phase D of colab.py); re-exported so the legacy `from cli import colab`
# surface and its monkeypatch needles are unchanged.
from cli.colab_result_sync import (  # noqa: E402,F401
    _IncrementalResultSync,
    _download_one_remote_file,
    _extract_result_archive,
    _prepare_remote_result_archive,
    _read_remote_text,
    _verify_result_bundle,
    download_verified_training_results,
)
# publish_local_hpo_results moved to cli.colab_retention (phase-1 split of
# colab.py); re-exported so the legacy `from cli import colab` surface and
# its monkeypatch needles are unchanged.
from cli.colab_retention import (  # noqa: E402,F401
    publish_local_hpo_results,
)
def start_live_log() -> None:
    """Start the ONE per-run Colab lane transcript, replacing the prior run's.

    `logs/colab/lane.log` is opened exactly once per run and shared by the
    system stdout/stderr tee, the trainer/worker writer, and setup timing.
    The file is truncated once (write_text) and then held in append mode so a
    detached self-watch child can append its own lines to the same transcript
    without a second "w" open clobbering it.
    """
    global LIVE_LOG_PATH, TRAINING_LOG_PATH, _live_log, _training_log
    global _original_stdout, _original_stderr
    global SETUP_TIMING_LOG_PATH, _setup_timing_active
    if _live_log is not None:
        _live_log.close()
    lane_path = lane_log("colab", LANE_LOG_NAME)
    LIVE_LOG_PATH = lane_path
    TRAINING_LOG_PATH = lane_path
    SETUP_TIMING_LOG_PATH = lane_path
    lane_path.parent.mkdir(parents=True, exist_ok=True)
    lane_path.write_text("", encoding="utf-8")  # exactly one truncation per run
    _setup_timing_active = True
    handle = lane_path.open("a", encoding="utf-8")
    _live_log = handle
    _training_log = handle
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = _Tee(_original_stdout, _live_log)
    sys.stderr = _Tee(_original_stderr, _live_log)
    print(_stamp(), f"[log] capturing Colab output -> {lane_path}", flush=True)

def close_live_log() -> None:
    global _live_log, _training_log, _original_stdout, _original_stderr
    _finish_setup_timing()
    handle = _live_log if _live_log is not None else _training_log
    if handle is not None:
        sys.stdout = _original_stdout or sys.stdout
        sys.stderr = _original_stderr or sys.stderr
        try:
            handle.flush()
        finally:
            handle.close()
    _live_log = None
    _training_log = None
    _original_stdout = None
    _original_stderr = None



@timed
def run_train(
    frac: float, epochs: int, sample: int | None, workers: int = 1,
    *, resume_run: str | None = None, model: str | None = None,
    dataset_csv: str | None = None,
    inference_sample: int | None = None,
    inference_device: str | None = None,
    run_label: str | None = None, masking_profile: str | None = None,
    collapse_guardrail_profile: str | None = None,
    loss: str = _TRAIN_LOSS,
    worker_losses: list[str] | None = None,
    train_only: bool = False,
    remote_dataset_csv: str | None = None,
    remote_prepared_bundles: list[str] | None = None,
    remote_validation_csv: str | None = None,
    incremental_sync: bool = True,
    smoke: bool = False,
) -> tuple[str, int]:
    """Full-chain GPU training on the VM."""
    print(_stamp(), "[run] train.py on the configured VM runtime ...")
    # AUDIT 2026-09-09: --mask-frac 0.15 REMOVED — it hardcoded a value that
    # silently contradicted the SSOT (masking.frac: 1.00 in
    # config/training.yaml). train.py's own default resolves from the config
    # now; the CLI flag remains for explicit overrides.
    args = ["-u", "-m", "training.train",
        "--split", "holdout",
        "--loss", loss,
        "--train-frac", str(frac),
        "--epochs", str(epochs),
        # Reports and plots are intentionally not generated by the Colab lane.
        "--no-plot"]
    if model is not None:
        registry = load_config()["models"]
        if model not in registry:
            raise KeyError(
                f"Colab training model must be a local registry key; "
                f"got {model!r}, expected one of {sorted(registry)}"
            )
        # Resolve inside the remote checkout. A local absolute path would
        # not exist on the VM and would bypass the Git-shipped model contract.
        args.extend(["--model", model])
    if sample is not None:
        args.extend(["--sample", str(sample)])
    # `training.train_prepared` receives the frozen dataframe inside its
    # bundle.  Keep the checkout dataset available for final inference, but
    # do not pass raw-trainer-only `--dataset` to that entrypoint.
    if remote_dataset_csv is not None and remote_prepared_bundles is None:
        remote_dataset = Path(remote_dataset_csv)
        if remote_dataset.is_absolute() or ".." in remote_dataset.parts:
            raise ValueError("remote dataset path must stay inside the checkout")
        args.extend(["--dataset", str(Path(REMOTE_ROOT) / remote_dataset)])
    if workers == 1:
        args.extend(["--masking-profile", masking_profile or _MASKING_PROFILE])
        args.extend([
            "--collapse-guardrail-profile",
            collapse_guardrail_profile or _COLLAPSE_GUARDRAIL_PROFILE,
        ])
    if not _MASK_EFFECT_AFTER_TRAIN:
        args.append("--no-mask-effect")
    if resume_run:
        args.append("--resume")
    profiles = _training_bundle_profiles(masking_profile, workers)
    prepared_bundles: list[Path] | None = None
    if remote_dataset_csv is None and remote_prepared_bundles is None:
        bundle_request = {
            "profiles": profiles,
            "model": model,
            "sample": sample,
        }
        if dataset_csv is not None:
            bundle_request["dataset_csv"] = dataset_csv
        prepared_bundles = _prepare_local_training_bundles(**bundle_request)
    if workers == 1 and resume_run is None:
        if worker_losses is not None:
            raise ValueError("worker_losses requires at least two concurrent workers")
        if remote_prepared_bundles is not None and len(remote_prepared_bundles) != 1:
            raise ValueError("a single-worker run requires exactly one checkout bundle")
        return run_single_train_and_stream(
            args,
            run_label=run_label,
            smoke=smoke,
            prepared_bundle=prepared_bundles[0] if prepared_bundles else None,
            remote_checkout_bundle=(
                remote_prepared_bundles[0] if remote_prepared_bundles else None
            ),
            final_inference=not train_only,
            inference_sample=inference_sample,
            inference_device=inference_device,
            # A remote dataset already shares the checkout's training_data
            # tree; the legacy worker-input copy assumes every input is under
            # RESULTS and is not applicable to this path.
            copy_remote_inputs=remote_dataset_csv is None,
            remote_validation_csv=remote_validation_csv,
            # Calibration pairs are now a versioned checkout input.
            prepare_remote_labeled_pairs=False,
            # Checkout-native prepared bundles still use the configured W&B
            # mirror; the remote dataset flag must not disable credentials.
            include_wandb=True,
            incremental_sync=incremental_sync,
        )
    return run_parallel_train_and_tail(
        args, workers, smoke=smoke, resume_run=resume_run,
        run_labels=(
            _expand_worker_profiles(run_label, workers, "run label")
            if run_label else None
        ),
        worker_losses=worker_losses,
        masking_profiles=profiles,
        collapse_guardrail_profiles=_expand_worker_profiles(
            collapse_guardrail_profile or _COLLAPSE_GUARDRAIL_PROFILE,
            workers,
            "collapse guardrail",
        ),
        prepared_bundles=prepared_bundles,
        final_inference=not train_only,
        inference_sample=inference_sample,
        inference_device=inference_device,
        remote_checkout_inputs=remote_dataset_csv is not None,
        remote_checkout_bundles=remote_prepared_bundles,
        remote_validation_csv=remote_validation_csv,
        incremental_sync=incremental_sync,
    )

# Local prepared-bundle cache + prewarm moved to cli.colab_bundle_prewarm
# (split phase); re-exported so the legacy `from cli import colab` surface
# and its monkeypatch needles are unchanged.  The in-flight prewarm slot
# stays on this module for the same reason.
from cli.colab_bundle_prewarm import (  # noqa: E402,F401
    _BundlePrewarm,
    _build_local_training_bundles,
    _bundle_cache_dir,
    _bundle_manifest,
    _bundle_model_digest,
    _bundle_request_key,
    _cached_bundles,
    _expand_worker_profiles,
    _lane_bundle_request,
    _populate_bundle_cache,
    _prepare_local_training_bundles,
    _run_diet_gate,
    _take_prewarmed_bundles,
    _training_bundle_profiles,
    _tree_digest,
    _upload_prepared_bundles,
    drain_local_bundle_prewarm,
    start_local_bundle_prewarm,
)

_BUNDLE_PREWARM: _BundlePrewarm | None = None


# Validation-input upload + prewarm moved to cli.colab_validation_upload
# (split phase); re-exported so the legacy `from cli import colab` surface
# and its monkeypatch needles are unchanged.  The in-flight upload slot
# stays on this module for the same reason.
from cli.colab_validation_upload import (  # noqa: E402,F401
    _ValidationUploadPrewarm,
    _lane_run_stamp,
    _perform_validation_upload,
    _remote_checkout_copy,
    _upload_validation_inputs,
    _validation_input_path,
    drain_validation_upload_prewarm,
    release_validation_upload_prewarm,
    start_validation_upload_prewarm,
)

_VALIDATION_UPLOAD_PREWARM: _ValidationUploadPrewarm | None = None


@timed
def run_single_train_and_stream(
    args: list[str], *, run_label: str | None = None,
    smoke: bool = False,
    prepared_bundle: Path | None = None, final_inference: bool = True,
    remote_checkout_bundle: str | None = None,
    inference_sample: int | None = None,
    inference_device: str | None = None,
    copy_remote_inputs: bool = True,
    remote_validation_csv: str | None = None,
    prepare_remote_labeled_pairs: bool = False,
    include_wandb: bool = True,
    incremental_sync: bool = True,
) -> tuple[str, int]:
    """Run one worker in the Colab exec stream so W&B is visible immediately."""
    stamp = _lane_run_stamp()
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    _record_remote_run(remote_base, workers=1, lane="train")
    if not final_inference:
        remote_validation_inputs = {"sample": "", "source": "", "training": ""}
    elif remote_validation_csv is not None:
        remote_validation_inputs = {
            "sample": remote_validation_csv,
            "source": remote_validation_csv,
            "training": remote_validation_csv,
        }
    else:
        remote_validation_inputs = _upload_validation_inputs(run_id)
    if prepared_bundle is not None and remote_checkout_bundle is not None:
        raise ValueError("choose either an uploaded or checkout prepared bundle")
    if remote_checkout_bundle is not None:
        relative = Path(remote_checkout_bundle)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkout bundle path must stay inside the checkout")
        remote_bundle = str(Path(REMOTE_ROOT) / relative)
        args = list(args)
        args[args.index("training.train")] = "training.train_prepared"
        args.extend(["--bundle", remote_bundle])
    elif prepared_bundle is not None:
        remote_bundle = _upload_prepared_bundles(
            run_id=Path(remote_base).name.removeprefix("concurrent_train_"),
            bundles=[prepared_bundle],
        )[0]
        args = list(args)
        args[args.index("training.train")] = "training.train_prepared"
        args.extend(["--bundle", remote_bundle])
    remote_input_loop = (
        "for name in ():"
        if remote_checkout_bundle is not None or prepared_bundle is not None or not copy_remote_inputs
        else 'for name in (F["canonical_records"], F["gate_results"], F["labeled_pairs"]):'
    )
    script = _BOOTSTRAP + _remote_auth_env_script(include_wandb=include_wandb) + f"""
import os, pathlib, shutil, subprocess, sys
from core.common import F
from core.tracing import run_trace_env
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
out = base / "worker_1"
base.mkdir(parents=True, exist_ok=False)
out.mkdir()
print(f"[worker] setup complete: results={{out}}", flush=True)
print("[worker] resolving versioned checkout inputs", flush=True)
{remote_input_loop}
    relative = name.relative_to(root / "results")
    source = root / "results" / relative
    if not source.is_file():
        raise FileNotFoundError(f"worker input missing: {{source}}")
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
wandb_dir = out / "wandb"
wandb_dir.mkdir(parents=True, exist_ok=True)
training_name = {f'{Path(remote_base).name.removeprefix("concurrent_train_")}-{run_label}' if run_label else Path(remote_base).name.removeprefix("concurrent_train_")!r}
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src"),
       "EUROMONITOR_RESULTS_DIR": str(out),
       "WANDB_DIR": str(wandb_dir),
       "WANDB_RUN_NAME": {f'{Path(remote_base).name.removeprefix("concurrent_train_")}-{run_label}' if run_label else Path(remote_base).name.removeprefix("concurrent_train_")!r},
       "EUROMONITOR_RUN_ID": {f'{Path(remote_base).name.removeprefix("concurrent_train_")}-{run_label}' if run_label else Path(remote_base).name.removeprefix("concurrent_train_")!r},
       "EUROMONITOR_MINING_PROFILE": {run_label if run_label in ("mining_enabled", "masking_only") else ""!r},
       "EUROMONITOR_REMOTE_TRAINING": "1", "EUROMONITOR_DISABLE_DVC_CHECKPOINTS": "1", **run_trace_env(lane=training_name)}}
if {prepare_remote_labeled_pairs!r}:
    calibration = out / "training" / "labeled_pairs.csv"
    if not calibration.is_file():
        print(f"[data] generating worker calibration input: {{calibration}}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "training.labeled_pairs"],
            cwd=root, env=env, check=True,
        )
command = [sys.executable, *{args!r}]
log_path = out / "training.log"
print("[worker] inputs ready; launching training process", flush=True)
print(f"[train-launch] worker 1 streaming directly: {{' '.join(command)}}", flush=True)
with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
    rc = process.wait()
if rc:
    raise RuntimeError(f"worker 1 failed (rc={{rc}}); log={{log_path}}")
print("[worker] training process finished", flush=True)
run_completion = {final_inference!r}
inference_sample = {inference_sample!r}
inference_device = {inference_device!r}
completion = [
    sys.executable, "-m", "training.complete_colab_worker",
    "--source", str(out), "--run-id", {run_id!r}, "--worker", "1",
        "--validation-input", str(F["final_validation"]),
        "--validation-source", {remote_validation_inputs['source']!r},
    "--training-input", str(F["dataset_deduped"]),
]
if inference_sample is not None:
    completion.extend(["--sample", str(inference_sample)])
if inference_device is not None:
    completion.extend(["--device", inference_device])
completion.append("--skip-dvc")
if run_completion:
    print("[worker] starting final validation inference", flush=True)
    print("[train] training complete; running validation inference", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            completion, cwd=root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        completion_rc = process.wait()
    if completion_rc:
        raise RuntimeError(
            f"worker 1 completion failed (rc={{completion_rc}}); log={{log_path}}"
        )
print(f"[train] worker 1 completed; log={{log_path}}", flush=True)
"""
    print(
        _stamp(),
        "[run] starting one trainer as a detached remote stage; polling its durable log ...",
        flush=True,
    )
    # Stream finished artifacts back while the trainer runs, so the end-of-run
    # download is a short delta instead of the whole result set.  The stream is
    # stopped before the authoritative download so the two never race on the
    # same local file.
    syncer = _IncrementalResultSync(remote_base, run_id, workers=1) if incremental_sync else None
    if syncer is not None:
        syncer.start()
    try:
        # A long-lived ``colab exec`` stream can stall before the kernel begins
        # evaluating the worker cell. Run the exact same script outside the
        # notebook kernel instead; its log, PID, and exit status are then
        # independently visible through the short polling probes.
        run_detached_stage(
            "train",
            ["/usr/bin/python3", "-c", script],
            timeout=_WORKER_TIMEOUT_SECONDS,
        )
    finally:
        if syncer is not None:
            syncer.stop()
    download_verified_training_results(remote_base, 1, smoke=smoke)
    print(
        _stamp(),
        f"[train] single worker completed; results downloaded "
        f"({_format_bytes(syncer.synced_bytes() if syncer is not None else 0)} arrived during the run)",
        flush=True,
    )
    return remote_base, 1


@timed
def run_hpo(
    mode: str | None = None,
    *,
    resume: bool = False,
    trial_jobs: int = _HPO_TRIAL_JOBS_DEFAULT,
    persistence: str | None = None,
    loss: str = _TRAIN_LOSS,
) -> None:
    """Sweep every configured backbone, then evaluate and rerank each winner."""
    mode = mode or _HPO_MODE
    persistence = persistence or _HPO_PERSISTENCE
    if persistence not in {"local", "none"}:
        raise ValueError("Colab HPO supports local/none persistence only")
    print(
        _stamp(),
        f"[run] round-robin HPO (mode={mode}, model_workers={_HPO_WORKERS}, "
        f"trial_jobs={trial_jobs}, resume={resume}, persistence={persistence}) ..."
    )
    run_id = (
        datetime.now(timezone.utc).strftime("hpo_%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    mask_effect_flag = "--mask-effect" if _MASK_EFFECT_AFTER_TRAIN else "--no-mask-effect"
    script = _BOOTSTRAP + _remote_auth_env_script(include_optuna=True) + f"""
import concurrent.futures, json, os, pathlib, shutil, subprocess, sys, time
from datetime import datetime, timezone
from core.common import F, hpo_cfg, resolve_model
from core.tracing import run_trace_env
root = pathlib.Path("{REMOTE_ROOT}")
hpo_root = root / "results" / "hpo_runs" / "{run_id}"
hpo_root.mkdir(parents=True, exist_ok=False)
(hpo_root / "generation.json").write_text(json.dumps({{
    "run_id": "{run_id}", "created_at": datetime.now(timezone.utc).isoformat(),
    "persistence": "{persistence}", "mode": "{mode}", "trial_jobs": {trial_jobs},
}}, indent=2, sort_keys=True), encoding="utf-8")
base = [sys.executable, "-u", "-m", "training.train", "--split", "holdout", "--loss", "{loss}", "--payload", "full", "--no-plot", "{mask_effect_flag}"]
hpo_base = list(base)
hpo_base.extend(["--n-jobs", str({trial_jobs})])
if {resume!r}:
    hpo_base.append("--resume")
model_keys = hpo_cfg()["models"]
required = {{"epochs", "lr", "warmup_ratio", "weight_decay"}}
mode = "{mode}"
workers = {_HPO_WORKERS}

if {resume!r}:
    databases = list((root / "results").rglob("*.optuna.db"))
    if not databases:
        raise FileNotFoundError("[resume-preflight] no local Optuna database; restore downloaded HPO files before resuming")

def run_logged(args, label, extra_env=None):
    log_path = hpo_root / "logs" / (
        f"colab_{{label}}_{{datetime.now(timezone.utc).strftime('%m%dT%H%M%SZ')}}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {{**os.environ, "PYTHONUNBUFFERED": "1"}}
    if extra_env:
        env.update(extra_env)
    print(f"[subprocess] {{' '.join(args)}} -> {{log_path}}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=root,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"{{label}} failed (rc={{rc}}); log={{log_path}}\\n" + "\\n".join(tail)
        )
    return log_path

def worker_setup(model_key):
    out = hpo_root / "models" / model_key
    out.mkdir(parents=True, exist_ok=True)
    for name in (F["canonical_records"], F["gate_results"]):
        source, target = root / "results" / name.name, out / name.name
        if not source.is_file():
            raise FileNotFoundError(f"worker input missing: {{source}}")
        shutil.copy2(source, target)
    return out, {{
        "EUROMONITOR_RESULTS_DIR": str(out),
        "WANDB_RUN_NAME": f"{run_id}_hpo_{{model_key}}",
        "EUROMONITOR_REMOTE_TRAINING": "1",
        "EUROMONITOR_DISABLE_DVC_CHECKPOINTS": "1",
        "EUROMONITOR_HPO_RETENTION_MODE": "1",
        "EUROMONITOR_HPO_GENERATION_ID": "{run_id}",
        "EUROMONITOR_HPO_MODEL_KEY": model_key,
        **run_trace_env(lane=model_key),
    }}

def run_model(model_key):
    model = resolve_model(model_key)
    out, env = worker_setup(model_key)
    model_tag = str(model).rstrip("/").rsplit("/", 1)[-1]
    best_path = out / f"train_{{model_tag}}-dlr_hpo_best.json"
    print(f"== HPO {{model_key}}: {{model}} (dev-selected; test withheld)", flush=True)
    run_logged(hpo_base + ["--model", str(model), "--hpo"], f"hpo_{{model_key}}", env)
    if not best_path.is_file():
        raise RuntimeError(f"missing HPO winner for {{model_key}}: {{best_path}}")
    best = json.loads(best_path.read_text())
    params = best["config"]
    missing = required - set(params)
    if missing:
        raise RuntimeError(f"HPO best config for {{model_key}} lacks {{sorted(missing)}}")
    final = base + [
        "--model", str(model),
        "--epochs", str(params["epochs"]),
        "--lr", str(params["lr"]),
        "--warmup-ratio", str(params["warmup_ratio"]),
        "--weight-decay", str(params["weight_decay"]),
        "--rerank", "{_RERANK_MODEL}",
    ]
    print(f"== FINAL {{model_key}}: selected dev config -> held-out test + rerank", flush=True)
    final_env = dict(env)
    if final_env:
        final_env["WANDB_RUN_NAME"] = f"{run_id}_final_{{model_key}}"
    run_logged(final, f"final_{{model_key}}", final_env)
    return {{"model_key": model_key, "model": str(model), "best": params,
            "results_dir": str(out.relative_to(root / "results"))}}

if mode == "parallel_same_vm":
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(model_keys))) as pool:
        summary = [future.result() for future in [pool.submit(run_model, key) for key in model_keys]]
else:
    summary = [run_model(key) for key in model_keys]
(hpo_root / "hpo_round_robin_summary.json").write_text(
    json.dumps({{"run_id": "{run_id}", "models": summary, "rerank_model": "{_RERANK_MODEL}"}}, indent=2),
    encoding="utf-8",
)
from core.archive_reader import tar_archive
archive = hpo_root.with_suffix('.tar.zst')
with tar_archive(archive, 'w') as bundle:
    bundle.add(hpo_root, arcname=hpo_root.name)
print(f"[hpo-archive] {{archive}}", flush=True)
print(json.dumps({{"hpo_run_id": "{run_id}", "hpo_round_robin": summary, "rerank_model": "{_RERANK_MODEL}"}}, sort_keys=True), flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=8 * 3600 * 3, log_name="training_hpo")
    remote_archive = f"{REMOTE_ROOT}/results/hpo_runs/{run_id}.tar.zst"
    local_archive = TRAINING_RESULTS / "hpo_runs" / f"{run_id}.tar.zst"
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    colab("download", "-s", SESSION, remote_archive, str(local_archive), timeout=3600)
    local_root = TRAINING_RESULTS / "hpo_runs"
    with tar_archive(local_archive) as bundle:
        bundle.extractall(local_root, filter='data')
    print(_stamp(), f"[hpo-archive] preserved -> {local_root / run_id}", flush=True)
    return run_id

def _bundle_delivery_local(run_id: str) -> Path:
    """Local delivery root for one bundle lane delivery archive.

    `_download_file_with_visibility` resolves its display/event paths with
    local.relative_to(TRAINING_RESULTS / run_id) (and `_result_event` writes
    into that same root), so the delivery MUST sit directly under
    TRAINING_RESULTS / <the run_id passed to that call>. run_bundle therefore
    derives one `colab_bundle_`-prefixed run id via this helper and passes
    Path(root).name as the download run_id, keeping the whole transfer inside
    a single TRAINING_RESULTS/colab_bundle_<run_id> root. Retention is
    unaffected: run_retention prunes only track-marker-carrying completed
    runs under TRAINING_RESULTS, and bundle deliveries carry none.
    """
    from cli.colab_lane import ColabCPULane

    return ColabCPULane().delivery_root(run_id)

@timed
def run_bundle(dataset_csv: Path | None = None) -> None:
    """Run the full CSV-to-inputs bundle lifecycle on the VM CPU.

    Fresh-checkout flow: the VM reuses the checkout prepare_remote_layout
    refetches to the configured branch HEAD; the raw export is uploaded on
    top of it because dataset.csv IS git-tracked (commit 1084010 "track the
    five CSVs a clone needs, ignore the rest") — the upload is a freshness
    override that replaces the checkout's committed bytes with the export
    passed via --dataset-csv (or repo-root dataset.csv by default), so an
    uncommitted export still drives the whole run. prepare_all then runs
    on the VM end to end and one delivery archive with the run dir +
    regenerated data artifacts comes back. CPU-only: no GPU allocation, no
    training.

    WHICH export this is, is carried by provenance, not a pin: prepare_all's
    stage manifests + provenance identity record the bytes actually consumed
    (owner ruling 2026-10-06 removed the config pin system), and the run
    archive carries them back for audit. Passing the wrong cohort export
    still produces a complete, internally consistent run — verify the
    uploaded source against the intended cohort BEFORE launching.
    The gate_census stage re-records the measured counts from the freshly
    regenerated gate_results.csv into run_dir/gate_census.json and the stage
    manifest — a measured record, not a tripwire.

    Delivery: the local copy lands under TRAINING_RESULTS/colab_bundle_<id>/
    (see _bundle_delivery_local); the VM-side archive keeps the FIXED name
    REMOTE_ROOT/bundle_delivery.tar.zst, so a rerun on a live VM overwrites
    the previous delivery — acceptable because the download consumes it per
    invocation and the local copy is timestamped per run_id.
    """
    from core.common import DATA_PATH

    source = Path(dataset_csv) if dataset_csv is not None else Path(DATA_PATH)
    if not source.is_file():
        raise FileNotFoundError(f"bundle raw export not found: {source}")
    run_id = datetime.now(timezone.utc).strftime("bundle_%m%dT%H%M%S%fZ")
    print(
        _stamp(),
        f"[bundle] uploading raw export {source} -> {REMOTE_ROOT}/dataset.csv ...",
        flush=True,
    )
    _upload_with_retries(source, f"{REMOTE_ROOT}/dataset.csv", timeout=3600)
    script = _BOOTSTRAP + f"""
import glob, os, subprocess, sys
rc = subprocess.run(
    [sys.executable, "-m", "training.prepare_all"],
    cwd={REMOTE_ROOT!r},
    env={{**os.environ, "WANDB_MODE": "disabled"}},
).returncode
if rc != 0:
    raise RuntimeError(f"prepare_all failed on the VM (rc={{rc}})")
run_dir = sorted(glob.glob({REMOTE_ROOT!r} + "/results/" + {_PREP_RUN_DIR_BASE!r} + "/*"))[-1]
# FIXED delivery name: a rerun on a live VM overwrites the previous archive;
# acceptable because the launcher consumes it per invocation and keeps a
# per-run timestamped copy under TRAINING_RESULTS/colab_bundle_<run_id>/.
delivery = {REMOTE_ROOT!r} + "/bundle_delivery.tar.zst"
from core.archive_reader import tar_archive
with tar_archive(delivery, "w") as tar:
    tar.add(run_dir, arcname={_PREP_RUN_DIR_BASE!r} + "/" + os.path.basename(run_dir))
    for rel in {DELIVERY_DATA_MEMBERS!r}:
        if os.path.exists({REMOTE_ROOT!r} + "/" + rel):
            tar.add({REMOTE_ROOT!r} + "/" + rel, arcname=rel)
    for name in {DELIVERY_TRACKED_DIRS!r}:
        member = {REMOTE_ROOT!r} + "/data/" + name
        if os.path.isdir(member):
            tar.add(member, arcname="data/" + name)
    for name in {DELIVERY_PREPARED_DIRS!r}:
        member = {REMOTE_ROOT!r} + "/data/prepared/" + name
        if os.path.isdir(member):
            tar.add(member, arcname="data/prepared/" + name)
print("[bundle] delivery archive ready", flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=4 * 3600, log_name="bundle")
    delivery_dir = _bundle_delivery_local(run_id)
    delivery_dir.mkdir(parents=True, exist_ok=True)
    # The download's run_id must be the PREFIXED name of the delivery root:
    # _download_file_with_visibility resolves local.relative_to(
    # TRAINING_RESULTS / run_id), so only the exact sibling-free id inside
    # that root satisfies the contract (see _bundle_delivery_local).
    _download_file_with_visibility(
        remote=f"{REMOTE_ROOT}/bundle_delivery.tar.zst",
        local=delivery_dir / "bundle_delivery.tar.zst",
        worker=None,
        index=1,
        total=1,
        run_id=delivery_dir.name,
    )
    print(
        _stamp(),
        f"[bundle] delivered -> {delivery_dir / 'bundle_delivery.tar.zst'}",
        flush=True,
    )


@timed
def run_sims() -> None:
    """Run the configured zero-shot embedding model lane on the VM."""
    print(_stamp(), f"[run] zero_shot_sims --models {_SIMS_MODEL} on the VM ...")
    run_id = datetime.now(timezone.utc).strftime("zero_shot_%m%dT%H%M%S%fZ")
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import os, subprocess, sys
os.environ["EUROMONITOR_RUN_ID"] = {run_id!r}
os.environ["WANDB_RUN_NAME"] = {run_id!r}
rc = subprocess.run(
    [sys.executable, "-m", "training.zero_shot_sims", "--models", {_SIMS_MODEL!r}],
    cwd={REMOTE_ROOT!r},
).returncode
if rc != 0:
    raise RuntimeError(f"zero-shot similarity subprocess failed (rc={{rc}})")
"""
    run_colab_exec_stream(SESSION, script, timeout=2 * 3600, log_name="sims")
    print(_stamp(), "[sims] remote zero-shot completed; downloading verified results ...", flush=True)
    download_results(skip_checkpoints=True, require_manifests=True)


@timed
def run_mixed(
    frac: float,
    epochs: int,
    *,
    model: str | None = None,
    loss: str = _TRAIN_LOSS,
) -> tuple[str, int]:
    """Run one masked trainer and one zero-shot worker on the same VM."""
    model_key = model or str(training_cfg().training.base_model)
    if model_key not in set(embedding_model_keys()):
        raise ValueError(
            "mixed lane requires an embedding model registry key: "
            f"{model_key!r}"
        )
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_mixed_{stamp}"
    _record_remote_run(remote_base, workers=2, lane="mixed")
    train_args = [
        "-u",
        "-m",
        "training.train",
        "--split",
        "holdout",
        "--loss",
        loss,
        "--train-frac",
        str(frac),
        "--epochs",
        str(epochs),
        "--model",
        model_key,
        "--no-plot",
    ]
    sims_args = [
        "-u",
        "-m",
        "training.zero_shot_sims",
        "--models",
        model_key,
    ]
    mixed_workers = _MIXED_TRAIN_WORKERS + _MIXED_SIMS_WORKERS
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import concurrent.futures, json, os, pathlib, shutil, subprocess, sys, threading, time
from core.common import F
from core.tracing import run_trace_env

root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
base.mkdir(parents=True, exist_ok=False)
worker_specs = [
    ("train", {train_args!r}, {_MIXED_MINING_PROFILE!r}, {_MASKING_ENABLED!r}),
    ("zero_shot", {sims_args!r}, {_MIXED_MINING_PROFILE!r}, False),
]

def emit_snapshot(label, out, proc):
    stamp = time.strftime("%m%dT%H%M%SZ", time.gmtime())
    commands = (
        ["ps", "-eo", "pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args", "--forest"],
        ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
    )
    sections = []
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            body = (result.stdout or result.stderr or "").rstrip()
            sections.append("$ " + " ".join(command) + "\\n" + body)
        except Exception as exc:
            sections.append("$ " + " ".join(command) + "\\nERROR " + repr(exc))
    header = (
        "[mixed-monitor] timestamp=" + stamp
        + " worker=" + label
        + " child_pid=" + str(proc.pid)
        + " child_returncode=" + str(proc.poll())
        + "\\n"
    )
    text = header + "\\n".join(sections) + "\\n"
    print(text, end="", flush=True)
    with (out / "processes.log").open("a", encoding="utf-8") as handle:
        handle.write(text)

def monitor_worker(label, out, proc, stop):
    emit_snapshot(label, out, proc)
    while not stop.wait({int(_WORKER_MONITOR_SECONDS)}):
        emit_snapshot(label, out, proc)

def run_worker(number, label, command_args, profile, masking_applied):
    out = base / f"worker_{{number}}"
    out.mkdir()
    (out / "wandb").mkdir()
    for name in (F["canonical_records"], F["gate_results"]):
        source = root / "results" / name.name
        if not source.is_file():
            raise FileNotFoundError(f"worker input missing: {{source}}")
        shutil.copy2(source, out / name.name)
    log_path = out / ("training.log" if label == "train" else "zero_shot.log")
    (out / "worker_spec.json").write_text(json.dumps({{
        "label": label,
        "model_key": {model_key!r},
        "mining_profile": profile,
        "masking_requested": {_MASKING_ENABLED!r},
        "masking_applied": masking_applied,
        "masking_note": (
            "training augmentation is applied by training.train"
            if masking_applied else
            "zero-shot scoring has no training augmentation stage"
        ),
    }}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    env = {{
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(root / "src"),
        "EUROMONITOR_RESULTS_DIR": str(out),
        "WANDB_DIR": str(out / "wandb"),
        "WANDB_RUN_NAME": f"{{base.name}}-{{label}}",
        "EUROMONITOR_RUN_ID": f"{{base.name}}-{{label}}",
        "EUROMONITOR_MINING_PROFILE": profile,
        "EUROMONITOR_REMOTE_TRAINING": "1",
        **run_trace_env(lane=label),
    }}
    command = [sys.executable, *command_args]
    print(f"[mixed] starting {{label}}: {{' '.join(command)}}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            process_group = str(os.getpgid(proc.pid))
        except ProcessLookupError:
            process_group = "exited"
        print(f"[mixed] worker={{label}} pid={{proc.pid}} pgid={{process_group}} monitor_interval={int(_WORKER_MONITOR_SECONDS)}s", flush=True)
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=monitor_worker,
            args=(label, out, proc, monitor_stop),
            name="mixed-monitor-" + label,
            daemon=True,
        )
        monitor.start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(f"[{{label}}] {{line}}", end="", flush=True)
                log.write(line)
        finally:
            monitor_stop.set()
            monitor.join()
            emit_snapshot(label, out, proc)
    rc = proc.wait()
    (out / "worker.status").write_text(f"{{rc}}\\n", encoding="utf-8")
    if rc:
        raise RuntimeError(f"mixed worker {{label}} failed (rc={{rc}}); log={{log_path}}")
    return label

with concurrent.futures.ThreadPoolExecutor(max_workers={mixed_workers}) as pool:
    futures = [
        pool.submit(run_worker, number, label, command, profile, masking_applied)
        for number, (label, command, profile, masking_applied) in enumerate(worker_specs, start=1)
    ]
    completed = [future.result() for future in futures]
print(json.dumps({{"base": str(base), "completed": completed}}), flush=True)
"""
    print(
        _stamp(),
        f"[run] mixed lane: {_MIXED_TRAIN_WORKERS} masked trainer + "
        f"{_MIXED_SIMS_WORKERS} zero-shot worker on {model_key} ...",
        flush=True,
    )
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=_WORKER_TIMEOUT_SECONDS,
        log_name="mixed",
        training_output=True,
    )
    download_verified_training_results(remote_base, mixed_workers)
    print(_stamp(), "[mixed] both workers completed; results downloaded", flush=True)
    return remote_base, 2

def _list_remote(pattern_dir: str, *, max_depth: int | None = None) -> list[str]:
    """List remote files via a stdin-exec (same channel the lanes use).

    ``max_depth`` bounds the walk.  An unbounded ``rglob`` over a live run
    root is what timed out a listing during the 2026-09-15 T4 run: the tree
    holds every checkpoint, ``wandb`` file, and log the run has produced so
    far, and the walk exceeded the 120 s exec budget.  Callers that only need
    a shallow slice pass a depth and get a bounded walk.
    """
    import json as _json

    if max_depth is None:
        collect = "p for p in root.rglob('*')"
    else:
        collect = (
            f"p for p in root.rglob('*') "
            f"if len(p.relative_to(root).parts) <= {int(max_depth)}"
        )
    script = (
        "import pathlib, json\n"
        f"root = pathlib.Path({pattern_dir!r})\n"
        f"files = sorted(str(p) for p in {collect} if p.is_file())\n"
        "print('@@FILES@@' + json.dumps(files))\n"
    )
    proc = subprocess.Popen(
        _colab_command("exec", "-s", SESSION, "--timeout", "120"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(script, timeout=120)
    if proc.returncode != 0:
        raise SystemExit(f"remote listing failed: {err[-500:]}")
    for line in out.splitlines():
        if line.startswith("@@FILES@@"):
            return _json.loads(line[len("@@FILES@@"):])
    raise SystemExit(f"remote listing returned no marker; out={out[-500:]}")

def _download_remote_manifests(*, required: bool = True) -> list[StageManifest]:
    """Pull and validate the completion records produced by remote stages.

    The manifests live outside the normal results-download tree, so they
    must be fetched explicitly before any artifact can be
    trusted.  A lane that produced no completion records is incomplete by
    definition: do not tear down its only copy while claiming success.
    """
    remote_dir = f"{REMOTE_ROOT}/results/manifests"
    names = _list_remote(remote_dir)
    if not names and not required:
        print(_stamp(), "[download] no stage manifests (frozen CSV lane)")
        return []
    if not names:
        raise RuntimeError(
            f"remote manifest directory is empty: {remote_dir}; refusing "
            "to download unverifiable lane results"
        )

    local_dir = RESULTS / "manifests"
    local_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[StageManifest] = []
    for name in names:
        remote = Path(name)
        try:
            rel = remote.relative_to(remote_dir)
        except ValueError as exc:
            raise RuntimeError(f"remote manifest escaped manifest dir: {name}") from exc
        if rel.parent != Path(".") or remote.suffix != ".json":
            raise RuntimeError(f"unexpected remote manifest path: {name}")
        local = local_dir / rel
        print(_stamp(), f"[download] manifest {rel}")
        colab("download", "-s", SESSION, name, str(local), timeout=600)
        try:
            manifest = StageManifest.model_validate_json(
                local.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RuntimeError(f"invalid remote manifest {name}: {exc}") from exc
        if manifest.status != "complete":
            raise RuntimeError(
                f"remote manifest {name} has status {manifest.status!r}; "
                "stage did not complete"
            )
        manifests.append(manifest)
    return manifests

def _local_path_for_remote(remote_path: str) -> Path:
    """Map an absolute path in the mirrored remote repo back to this repo."""
    try:
        rel = Path(remote_path).relative_to(REMOTE_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"manifest output is outside remote project root: {remote_path}"
        ) from exc
    return TRAIN_ROOT / rel

def _verify_manifest_downloads(manifests: list[StageManifest]) -> None:
    """Fail if a manifest-listed expected output is absent or byte-different."""
    problems: list[str] = []
    for manifest in manifests:
        output_names = {Path(entry.path).name for entry in manifest.outputs}
        for expected in manifest.expected_outputs:
            if expected not in output_names:
                problems.append(
                    f"{manifest.stage}: expected output absent from manifest: {expected}"
                )
        for entry in manifest.outputs:
            local = _local_path_for_remote(entry.path)
            if not local.is_file():
                problems.append(f"{manifest.stage}: missing local output: {local}")
                continue
            actual = sha256_file(local)
            if actual != entry.sha256:
                problems.append(
                    f"{manifest.stage}: sha256 mismatch for {local} "
                    f"(remote {entry.sha256[:12]}, local {actual[:12]})"
                )
    if problems:
        raise RuntimeError(
            "Colab download integrity verification failed:\n  - "
            + "\n  - ".join(problems)
        )


@timed
def download_results(
    skip_checkpoints: bool = True, *, require_manifests: bool = False
) -> list[StageManifest]:
    """Pull the result artifacts back to the repo results dir.

    AUDIT FIX 2026-09-08: the generic rglob included _checkpoints (~1.9 GB
    of model weights) for EVERY lane — checkpoints are pulled explicitly by
    download_checkpoints() only when --what train asks for them.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifests = _download_remote_manifests(required=require_manifests)
    files = _list_remote(f"{REMOTE_ROOT}/results")
    for name in files:
        rel = Path(name).relative_to(f"{REMOTE_ROOT}/results")
        if skip_checkpoints and rel.parts[0] == "_checkpoints":
            continue
        local = RESULTS / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(_stamp(), f"[download] {rel}")
        # RULING 2026-09-10 (silent-degradation audit): LOUD-RAISE.
        # This runs after the lane and before stop() destroys the VM: a
        # swallowed failure here is a silent data drop — main() would
        # tear down the only remaining copy and print "[done] artifacts
        # saved" over a partial results dir. Re-raise instead (colab()
        # already printed the command + stderr tail; stop() still runs
        # via main()'s finally unless --keep-alive, so the remote copy
        # survives for a re-pull).
        try:
            colab("download", "-s", SESSION, name, str(local), timeout=600)
        except subprocess.CalledProcessError:
            print(
                _stamp(),
                f"[error] results download failed for {rel} — local copy "
                f"at {local} is absent/partial; refusing to continue "
                f"because teardown would delete the only remote copy "
                f"(re-run the lane, or re-pull from a --keep-alive VM)",
                file=sys.stderr,
            )
            raise
    _verify_manifest_downloads(manifests)
    return manifests


@timed
def download_checkpoints(manifests: list[StageManifest] | None = None) -> None:
    """Pull the trained checkpoints (model weights) back.

    Called after --what train: the trained model IS the deliverable of the
    production run; results CSVs alone don't carry it.
    """
    # Re-fetch and re-verify after this separately downloaded tree too.  A
    # future stage may list a checkpoint as an output; then it receives the
    # same hash gate as ordinary results instead of becoming a blind spot.
    if manifests is None:
        manifests = _download_remote_manifests(required=False)
    print(_stamp(), "[download] checkpoints ...")
    files = _list_remote(f"{REMOTE_ROOT}/results/_checkpoints")
    for name in files:
        rel = Path(name).relative_to(f"{REMOTE_ROOT}/results")
        local = RESULTS / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(_stamp(), f"[download] {rel}")
        # RULING 2026-09-10 (silent-degradation audit): LOUD-RAISE.
        # The trained model weights ARE the deliverable of --what train
        # (results CSVs alone don't carry it, see docstring); a swallowed
        # download here leaves the lane's only artifact on a VM that
        # stop() is about to destroy. Re-raise so the operator can
        # re-pull before teardown (--keep-alive keeps the VM up).
        try:
            colab("download", "-s", SESSION, name, str(local), timeout=1200)
        except subprocess.CalledProcessError:
            print(
                _stamp(),
                f"[error] checkpoint download failed for {rel} — the "
                f"trained weights were NOT pulled local; continuing would "
                f"let stop() destroy the only copy. Re-run the lane or "
                f"re-pull manually before the VM is gone (--keep-alive "
                f"keeps it up)",
                file=sys.stderr,
            )
            raise
    _verify_manifest_downloads(manifests)

def stop_local_launch_owner(*, timeout_seconds: float = 15.0) -> None:
    """Ask a verified launcher owner to exit after its VM is stopped.

    PID reuse makes a bare PID unsafe.  New lock records include Linux's
    process-start ticks, so this only sends SIGTERM when the recorded process
    is demonstrably still the same process that acquired the lock.
    """
    lock_path = _colab_launch_lock_path()
    if not _colab_launch_lock_is_held(lock_path):
        print(_stamp(), "[stop] local launcher lock is already released")
        return
    owner = _read_colab_launch_owner(lock_path)
    if owner is None:
        print(
            _stamp(),
            f"[warn] local launcher lock remains held but has no readable owner metadata: {lock_path}",
            file=sys.stderr,
        )
        return
    pid = owner.get("pid")
    expected_start = owner.get("pid_start_ticks")
    if not isinstance(pid, int) or not isinstance(expected_start, int):
        print(
            _stamp(),
            "[warn] local launcher lock remains held by a legacy or unverifiable owner; "
            f"metadata={owner}. It cannot be signalled safely.",
            file=sys.stderr,
        )
        return
    actual_start = _process_start_ticks(pid)
    if actual_start != expected_start:
        print(
            _stamp(),
            "[warn] local launcher lock remains held, but its recorded owner no longer "
            f"matches pid={pid}; refusing to signal a potentially reused PID.",
            file=sys.stderr,
        )
        return
    if pid == os.getpid():
        print(_stamp(), "[stop] current launcher owns the local session lock")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(_stamp(), f"[stop] recorded launcher pid={pid} has already exited")
    except PermissionError:
        print(
            _stamp(),
            f"[warn] local launcher pid={pid} owns the lock but cannot be signalled",
            file=sys.stderr,
        )
        return
    else:
        print(_stamp(), f"[stop] requested shutdown from local launcher pid={pid}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _colab_launch_lock_is_held(lock_path):
            print(_stamp(), "[stop] local launcher lock released")
            return
        time.sleep(0.2)
    print(
        _stamp(),
        f"[warn] local launcher lock is still held after {timeout_seconds:g}s: {lock_path}",
        file=sys.stderr,
    )

def stop(*, stop_local_owner: bool = False) -> bool:
    confirmed = False
    print(_stamp(), f"[stop] tearing down '{SESSION}'")
    # RULING 2026-09-10 (silent-degradation audit): JUSTIFIED-KEEP.
    # stop() runs in main()'s finally — if the lane itself raised, the
    # lane's exception is the root cause and must stay the error the
    # operator sees; raising here would MASK it with a teardown failure
    # in the finally path. Callers that require confirmed release inspect
    # the returned boolean before beginning CPU postprocessing. Never hide it: an unreleased VM burns
    # Colab GPU quota until manually reaped, so warn loudly with the
    # consequence + the exact recovery command.
    try:
        result = colab("stop", "-s", SESSION, check=False, timeout=30)
        if result.returncode:
            print(
                _stamp(),
                f"[warn] VM release command returned rc={result.returncode}; "
                f"stdout={result.stdout[-2000:]!r} stderr={result.stderr[-2000:]!r}",
                file=sys.stderr,
            )
        status = colab("sessions", check=False, timeout=30)
        if SESSION in (status.stdout or ""):
            print(
                _stamp(),
                f"[warn] teardown verification still lists '{SESSION}'; "
                "the VM may still be live and consuming quota.",
                file=sys.stderr,
            )
        elif status.returncode == 0:
            confirmed = True
            print(_stamp(), "[stop] teardown verified: session is no longer listed")
        else:
            print(
                _stamp(),
                f"[warn] could not verify teardown; sessions command returned "
                f"rc={status.returncode}: {status.stderr[-1000:]!r}",
                file=sys.stderr,
            )
    except (subprocess.SubprocessError, OSError) as exc:
        print(
            _stamp(),
            f"[warn] VM release request failed — the VM '{SESSION}' may "
            f"STILL BE LIVE and burning Colab GPU quota until it times "
            f"out or is reaped. After handling the failure above, reclaim "
            f"it with: colab stop -s {SESSION}   (or 'colab sessions' "
            f"to check). Original error: {exc}",
            file=sys.stderr,
        )
    if stop_local_owner:
        stop_local_launch_owner()
    print(_stamp(), "[stop] VM release requested")
    return confirmed


# Self-watch moved to cli.colab_self_watch (phase-1 split of colab.py);
# re-exported so the legacy `from cli import colab` surface and its
# monkeypatch needles are unchanged.
from cli.colab_self_watch import (  # noqa: E402,F401
    _SELF_WATCH_BUDGET_SECONDS,
    _SELF_WATCH_POLL_SECONDS,
    _SELF_WATCH_TRANSCRIPT_MAX_BYTES,
    _self_watch_delivery_state,
    _self_watch_root,
    _session_listed,
    _write_self_watch_receipt,
    self_watch,
    spawn_self_watch,
)
def _suite_matrix_flip_path(suite_config: Path) -> Path:
    """The scratch cuda clone path for `results/model_tracks/<suite>__gpu/`."""
    matrix = canonical_suite_matrix()
    family = (suite_config.parent.name
              if suite_config.name == 'suite.yaml' else suite_config.stem)
    return RESULTS / matrix.device_flip.tracks_dir / f'{family}{matrix.device_flip.suffix}' / suite_config.name

def _suite_device_flip(suite_config: Path, suite) -> Path:
    """Baked matrix default: a non-CPU cuda request against a tracked
    device-cpu suite immediately generates the scratch cuda variant
    (only the yamls; every data binding stays under data/). The tracks
    gate then validates the clone like any other suite config."""
    pattern = canonical_suite_matrix().device_flip
    source_text = suite_config.read_text(encoding='utf-8')
    flipped, flips = re.subn(r'(?m)^(\s*device:\s*)cpu\s*$', r'\1cuda', source_text, count=1)
    if not flips:
        raise ValueError(
            'suite device is cpu but the suite yaml carries no device line: '+str(suite_config))
    flip_path = _suite_matrix_flip_path(suite_config)
    flip_path.parent.mkdir(parents=True, exist_ok=True)
    setup_dir = (TRAIN_ROOT / suite.setup_dir).resolve()
    layout = training_cfg().preparation.graph_setup
    for name in (layout.track_config('gnn_only'), layout.track_config('cascade'), layout.text_config):
        companion, target = setup_dir / name, flip_path.parent / name
        if companion.is_file():
            shutil.copy2(companion, target)
        elif target.exists():
            target.unlink()
    flip_path.write_text(flipped, encoding='utf-8')
    return flip_path


def prepared_package_candidates() -> list[tuple[Path, dict]]:
    """Known prepared all_tracks_inputs archives with receipt metadata, newest first.

    There are three legitimate homes for the same bundle -- the kaggle lane
    install (``results/kaggle_lane/<cohort>/bundle``), a training_prep run dir,
    and a published model_tracks ``__inputs`` archive -- so anything that helps
    the operator point at "the" bundle must know them all.
    """
    found: list[tuple[Path, dict]] = []

    def add(archive: Path, receipt: Path | None) -> None:
        if not archive.is_file():
            return
        metadata: dict = {}
        if receipt is not None and receipt.is_file():
            try:
                metadata = json.loads(receipt.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        found.append((archive, metadata))

    kaggle_root = RESULTS / 'kaggle_lane'
    if kaggle_root.is_dir():
        for install in sorted(p for p in kaggle_root.iterdir() if p.is_dir()):
            bundle = install / 'bundle'
            add(bundle / _BUNDLE_ARCHIVE, bundle / 'bundle.receipt.json')
    prep_root = RESULTS / _PREP_RUN_DIR_BASE
    if prep_root.is_dir():
        for run in sorted(p for p in prep_root.iterdir() if p.is_dir()):
            add(run / _BUNDLE_ARCHIVE, run / 'manifest.json')
            add(run / 'before' / _BUNDLE_ARCHIVE, None)
    tracks_root = RESULTS / 'model_tracks'
    if tracks_root.is_dir():
        for archive in sorted(tracks_root.glob('*__inputs.tar.zst')):
            add(archive, None)
    return sorted(found, key=lambda item: item[0].stat().st_mtime, reverse=True)


def resolve_prepared_input_package(value: Path) -> Path:
    """Resolve --prepared-input-package to the archive the lane loads.

    Accepts the archive, or a bundle directory resolved through its
    ``bundle.receipt.json`` (the archive it names) or the canonical
    ``all_tracks_inputs.tar.zst``. A path that resolves to nothing fails loud
    with every known bundle and its cohort/revision, so the operator never has
    to guess which of ``results/kaggle_lane``, ``results/training_prep`` or
    ``results/model_tracks`` holds the right copy.
    """
    value = Path(value)
    if value.is_file():
        return value
    if value.is_dir():
        receipt_path = value / 'bundle.receipt.json'
        if receipt_path.is_file():
            try:
                archive_name = json.loads(receipt_path.read_text(encoding='utf-8')).get('archive')
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f'unreadable bundle receipt {receipt_path}: {exc}') from exc
            if archive_name and (value / archive_name).is_file():
                return value / archive_name
        canonical = value / _BUNDLE_ARCHIVE
        if canonical.is_file():
            return canonical
    listing = '\n'.join(
        f'  {archive}'
        + (f"  (cohort={metadata.get('cohort')}, revision={str(metadata.get('revision'))[:12]})"
           if metadata else '')
        for archive, metadata in prepared_package_candidates()) or '  (none found)'
    raise FileNotFoundError(
        f'no prepared all_tracks_inputs package at {value!s}; pass the archive, a '
        f'bundle directory, or one of:\n{listing}')


def default_prepared_input_package() -> Path | None:
    """The canonical full-cohort bundle when the operator names none.

    Policy: every launch trains the full cohort on a GPU from the prebuilt full
    bundle, so an unspecified --prepared-input-package resolves to
    results/kaggle_lane/full/bundle instead of repackaging locally. Unknown
    layout returns None and the lane falls back to on-VM packaging.
    """
    bundle = RESULTS / 'kaggle_lane' / 'full' / 'bundle'
    return bundle if bundle.is_dir() else None


def main() -> None:
    RunLogger.configure_console()
    global GPU
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="tracks",
                    choices=["train", "tracks", "dual-train", "hpo", "sims", "mixed", "smoke", "bundle", "stop", "self-watch"],
                    help="what to run on the VM (default: tracks)")
    ap.add_argument('--tracks-config', type=Path, default=None,
                    help='prepared all-track suite; uses the existing Colab lifecycle')
    ap.add_argument('--prepared-input-package', type=Path, default=None,
                    help='reuse a training.prepare_all all_tracks_inputs package '
                         '(.tar.zst) after freshness validation; may be the archive, '
                         'a bundle directory (resolved via bundle.receipt.json), or a '
                         'results/kaggle_lane/<cohort>/bundle dir')
    ap.add_argument('--dataset-csv', type=Path, default=None,
                    help="raw export to upload for --what bundle "
                         "(default: config/paths.yaml dataset = repo:dataset.csv; "
                         "the VM holds it at dataset.csv so the committed audit "
                         "pins still enforce which export is consumed)")
    ap.add_argument("--train-frac", type=float, default=_TRAIN_FRAC_DEFAULT,
                    help=f"train fraction for --what train (default "
                    f"{_TRAIN_FRAC_DEFAULT:g})")
    ap.add_argument("--epochs", type=int, default=_EPOCHS_DEFAULT,
                    help=f"epochs for --what train (default {_EPOCHS_DEFAULT} = "
                    "config/training.yaml training.epochs)")
    ap.add_argument(
        "--workers", type=int, default=_TRAIN_WORKERS,
        help=f"concurrent full-data trainers for --what train (default {_TRAIN_WORKERS} = "
        "config/training.yaml colab.train_workers; use 1 for a single run)",
    )
    ap.add_argument(
        "--sample", type=int, default=None,
        help="optional smoke cap for --what train; full data when omitted",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="model registry key for --what train (for example minilm_l6)",
    )
    ap.add_argument(
        "--loss",
        choices=["contrastive", "mnrl", "triplet"],
        default=_TRAIN_LOSS,
        help="training loss (default: config/training.yaml training.loss)",
    )
    ap.add_argument(
        "--run-label",
        default=None,
        help="experiment label for W&B (for example mining_enabled or masking_only)",
    )
    ap.add_argument(
        "--masking-profile",
        default=_MASKING_PROFILE,
        help="config/training.yaml masking_profiles entry",
    )
    ap.add_argument(
        "--collapse-guardrail-profile",
        default=_COLLAPSE_GUARDRAIL_PROFILE,
        help="config/training.yaml collapse_guardrail_profiles entry",
    )
    ap.add_argument(
        "--resume-run",
        default=None,
        help="resume this existing concurrent_train_<id> run on the VM",
    )
    ap.add_argument(
        "--resume-hpo",
        action="store_true",
        help="restore the previous HPO Optuna database/checkpoints before the sweep",
    )
    ap.add_argument(
        "--hpo-persistence",
        choices=["local", "none"],
        default=_HPO_PERSISTENCE,
        help="HPO durability backend (default from config/training.yaml)",
    )
    ap.add_argument(
        "--gpu",
        default=GPU,
        help=f"Colab accelerator request (default {GPU}; e.g. A100 when available)",
    )
    ap.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement before a non-CPU runtime can be provisioned",
    )
    ap.add_argument(
        "--hpo-mode",
        choices=["sequential", "parallel_same_vm"],
        default=_HPO_MODE,
        help="HPO scheduling mode (default from config/training.yaml)",
    )
    ap.add_argument(
        "--hpo-jobs",
        type=int,
        default=_HPO_TRIAL_JOBS_DEFAULT,
        help="concurrent Optuna trials per backbone (3 models x 3 jobs = 9 A100 workers)",
    )
    ap.add_argument(
        "--refresh-data",
        action="store_true",
        help="explicitly regenerate frozen CSV inputs before training",
    )
    ap.add_argument(
        "--keep-alive",
        action="store_true",
        help="CPU only: do not tear down the VM on completion/failure (refused for GPU launches)",
    )
    ap.add_argument(
        "--preflight-only", action="store_true",
        help="validate and print the train/inference lifecycle without contacting Colab",
    )
    ap.add_argument(
        "--train-only", action="store_true",
        help="train and collect artifacts without post-training validation inference",
    )
    ap.add_argument(
        "--self-watch-what", default=None,
        help="the lane a spawned detached self-watch is watching (internal)",
    )
    ap.add_argument(
        "--self-watch-run", default=None,
        help="the run identity a detached self-watch watches (internal)",
    )
    args = ap.parse_args()
    if args.what == "hpo" and args.hpo_persistence not in {"local", "none"}:
        raise ValueError("Colab HPO supports local/none persistence only")
    # The detached self-watch child enters here: before any gate could
    # otherwise refuse it.  It provisions nothing; it only proves delivery
    # and release.  Both fed args are set by spawn_self_watch which keeps
    # the baked-in default arg-free for operators.
    if args.what == "self-watch":
        if args.self_watch_run is None:
            raise SystemExit("--self-watch-run is required for --what self-watch")
        plan = self_watch(
            what=args.self_watch_what or "unspecified",
            run_id=args.self_watch_run,
        )
        print(_stamp(), f"[self-watch] receipt released: {plan['receipt']}")
        return

    if args.prepared_input_package is not None:
        args.prepared_input_package = resolve_prepared_input_package(args.prepared_input_package)
    suite_archive = None
    suite_run_tag = None
    suite_git_inputs = None
    if args.prepared_input_package is not None and args.what != 'tracks' and args.tracks_config is None:
        raise ValueError('--prepared-input-package requires an all-track suite')
    if args.what == 'tracks' or args.tracks_config is not None:
        if args.what not in {'tracks','train','smoke'}:
            raise ValueError('--tracks-config applies to train/tracks/smoke only')
        if args.refresh_data or args.train_only:
            raise ValueError('all-track suite requires prepared inputs and full postprocessing')
        from model_tracks.config import load_config as load_suite
        from model_tracks.preflight import preflight as suite_preflight
        from model_tracks.package import package as suite_package
        suite_config = args.tracks_config or _suite_config_path()
        suite = load_suite(suite_config)
        # Policy: the default tracks launch always trains the full cohort from the
        # prebuilt full bundle. An explicit --tracks-config (e.g. the CPU smoke)
        # or an explicit --prepared-input-package still wins.
        if (args.prepared_input_package is None and not args.resume_run
                and args.tracks_config is None):
            canonical = default_prepared_input_package()
            if canonical is not None:
                args.prepared_input_package = resolve_prepared_input_package(canonical)
                print(_stamp(), f'[bundle] using canonical full bundle '
                      f'{args.prepared_input_package}', flush=True)
        if args.preflight_only:
            if args.prepared_input_package is not None:
                from model_tracks.package import verify as verify_package
                checks = verify_package(args.prepared_input_package)['preflight']
            else:
                checks = suite_preflight(suite_config)
            print(json.dumps(checks,indent=2))
            return
        if suite.device != ('cpu' if args.gpu.upper() == 'CPU' else 'cuda'):
            if (args.gpu.upper() != 'CPU' and suite.device == 'cpu'
                    and os.environ.get(canonical_suite_matrix().device_flip.opt_out_env) != '1'):
                matrix = canonical_suite_matrix()
                suite_config = _suite_device_flip(suite_config, suite)
                suite = load_suite(suite_config)
                print(_stamp(), '[suite-matrix] baked device flip -> '
                      f'{suite_config} (opt out with '
                      f'{matrix.device_flip.opt_out_env}=1)', flush=True)
            else:
                raise ValueError('suite device and --gpu must agree')
        suite_run_tag = args.resume_run or _lane_run_stamp()
        suite_archive = RESULTS/'model_tracks'/f'{suite_run_tag}__inputs.{suite.input_archive_format}'
        if args.resume_run:
            if not suite_archive.is_file():
                raise FileNotFoundError(f'resume requires the original prepared input package: {suite_archive}')
            from model_tracks.package import verify as verify_suite_package
            verify_suite_package(suite_archive)
        else:
            if args.prepared_input_package is not None:
                from model_tracks.package import verify as verify_package
                verify_package(args.prepared_input_package)
                suite_archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.prepared_input_package, suite_archive)
            else:
                suite_archive = suite_package(suite_config, suite_archive)
        # Finish immutable input publication before allocating an accelerator.
        # prepare_remote_layout clones the same Git branch below.
        from model_tracks.colab import prepare_git_inputs
        suite_recovery = RESULTS/'model_tracks'/f'{suite_run_tag}.recovery.tar.zst'
        suite_git_inputs = prepare_git_inputs(suite_archive,suite_run_tag,
            resume_archive=suite_recovery if args.resume_run and suite_recovery.is_file() else None)
        args.what = 'tracks'

    if args.what == 'smoke' and suite_archive is None:
        raise ValueError(
            'legacy smoke does not preserve the shared component holdout; '
            'use --what tracks --tracks-config '
            'results/model_tracks/smoke_20261001_128/suite.yaml --gpu CPU')
    if args.what == 'train' and suite_archive is None:
        if args.sample is not None:
            raise ValueError('sampled training requires --tracks-config with frozen parent splits')
        _legacy_validation_sources()
        if (args.workers == 1 and args.model is None and args.resume_run is None):
            _validate_legacy_bundle_partitions([
                TRAIN_ROOT / path for path in _COLAB.full_prepared_bundles])

    GPU = args.gpu
    # GPU/retention boundary gates (allow-gpu acknowledgement; CPU-only
    # --keep-alive) have one SSOT: cli.colab_lane.ColabGPULane.
    from cli.colab_lane import ColabGPULane

    ColabGPULane().enforce_boundary(
        GPU, allow_gpu=args.allow_gpu, keep_alive=args.keep_alive
    )

    # Two different decisions used to be one, and conflating them stopped every
    # GPU lane from provisioning at all:
    #
    #   the DAEMON   is spawned by `colab new` itself and is what provisioning
    #                needs.  It is always allowed, on every lane.
    #   RETENTION    is whether the VM is still running when the work ends.
    #                That is CPU-only, and a GPU lane never keeps it.
    os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"

    if args.preflight_only:
        if args.what not in {"train", "smoke"}:
            raise ValueError("--preflight-only supports train/smoke lanes")
        print(json.dumps(training_lifecycle_preflight(
            workers=args.workers if args.what == "train" else _SMOKE_WORKERS,
            model=args.model,
            masking_profile=args.masking_profile,
            train_only=args.train_only,
        ), indent=2, sort_keys=True))
        return

    if args.what == "train":
        print(
            _stamp(),
            f"[workers] lane=train trainers={args.workers} "
            "result_transport=direct",
            flush=True,
        )
    elif args.what == "dual-train":
        print(
            _stamp(),
            f"[workers] lane=dual-train matcher_loss={args.loss} ann_loss=mnrl "
            "result_transport=direct",
            flush=True,
        )
    elif args.what == "smoke":
        print(
            _stamp(),
            f"[workers] lane=smoke trainers={_SMOKE_WORKERS} "
            "result_transport=direct",
            flush=True,
        )
    elif args.what == "hpo":
        print(
            _stamp(),
            f"[workers] lane=hpo model_workers={_HPO_WORKERS} "
            f"trial_jobs={args.hpo_jobs} result_transport=direct",
            flush=True,
        )
    elif args.what == "mixed":
        print(
            _stamp(),
            f"[workers] lane=mixed train_workers={_MIXED_TRAIN_WORKERS} "
            f"zero_shot_workers={_MIXED_SIMS_WORKERS} "
            "result_transport=direct",
            flush=True,
        )

    if args.what == "stop":
        stop(stop_local_owner=True)
        return

    start_live_log()
    launch_lock = None
    try:
        check_colab_cli()
        launch_lock = acquire_colab_launch_lock()
    except BaseException:
        close_live_log()
        raise
    local_training_run: tuple[str, int] | None = None
    local_mixed_run: tuple[str, int] | None = None
    local_hpo_run: str | None = None
    # Only a clean exit through the end of the try body may announce success.
    # The teardown below runs for every outcome, so without this flag a failed
    # launch would still print [done] and make the transcript read as success.
    completed = False

    try:
        with _colab_timing("step", "initialization"):
            prepared_train_runtime = args.what in {"train", "dual-train", "tracks"}
            # The local bundle build is pure local CPU work over immutable local
            # inputs, so start it before the VM is even provisioned: it then runs
            # under the remote checkout, install, model check, and profile instead
            # of after them.  run_train joins this exact build (or rebuilds when
            # the request differs).
            bundle_request = None if args.what == 'tracks' else _lane_bundle_request(args)
            if bundle_request is not None:
                start_local_bundle_prewarm(**bundle_request)
            # The validation CSVs are read here and pushed to the VM, so they do
            # not have to wait for the dependency install to finish.  A resumed
            # run keeps its existing identity and uploads serially.
            if bundle_request is not None and not args.resume_run:
                start_validation_upload_prewarm()
            # Runtime checkout contract: the VM sparse-checks out only the declared
            # paths, plus this launch's transport and text model. Anything missing
            # or unpushed must fail here -- before ensure_session allocates an
            # accelerator -- not mid-recovery on the VM.
            runtime_paths: tuple[str, ...] = ()
            if suite_git_inputs is not None:
                from core.common import resolve_model
                runtime_paths = tuple(path.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()
                                      for path in (suite_git_inputs, Path(resolve_model(suite.text_model))))
            validate_runtime_checkout(extra_paths=runtime_paths)
            ensure_session()
            # The session exists now, so the prewarmed upload can run for real.  It
            # travels alongside prepare_remote_layout/install_deps below instead of
            # after them, which is the whole point of starting it early.
            release_validation_upload_prewarm()
            if GPU.upper() != "CPU":
                # Backstop, not the primary release: the launcher's own teardown
                # still runs in `finally`.  A GPU VM must never be left held open
                # by its own daemon if this process dies, so the daemon is stopped
                # now that provisioning is done and the launcher owns the run --
                # the VM then idle-terminates instead of burning accelerator quota
                # indefinitely.
                stop_keep_alive_daemon(reason=f"GPU lane ({GPU}) must never be retained")
            if suite_git_inputs is not None:
                prepare_remote_layout(minimal_runtime=True, sparse_paths=runtime_paths)
            else:
                prepare_remote_layout(minimal_runtime=prepared_train_runtime)
            if args.what == 'tracks':
                install_deps(minimal_runtime=True, graph_runtime=True)
            else:
                install_deps(minimal_runtime=prepared_train_runtime)
            if args.what in {"train", "dual-train", "smoke", "mixed"}:
                required_models = [
                    args.model or str(training_cfg().training.base_model)
                ]
            elif args.what == "hpo":
                required_models = list(hpo_cfg()["models"])
                required_models.append(str(sweep_cfg()["rerank_model"]))
            elif args.what == "sims":
                required_models = [_SIMS_MODEL]
            else:
                required_models = []
            if required_models:
                verify_remote_models(required_models)
            log_gpu_profile()
            if args.refresh_data and prepared_train_runtime:
                raise ValueError("--refresh-data is incompatible with local-prepared GPU training")
            if args.refresh_data:
                run_data_prep()
            elif args.what != "smoke" and not prepared_train_runtime:
                verify_training_inputs()
            # Default self-watch spawn point (owner order 2026-10-07): the
            # gates ran pre-provisioning and the first healthy provisioning
            # stream has finished, so this executed run now carries its own
            # detached release+delivery guarantee.  Never spawned on plan
            # paths (they returned above) or on an explicit --keep-alive
            # retention request (the watcher must never keep a VM alive, and
            # must never override the operator's own retention choice).
            if not args.keep_alive:
                print(
                    _stamp(),
                    f"[self-watch] {spawn_self_watch(what=args.what, run_id=_lane_run_stamp())}",
                    flush=True,
                )
        # AUDIT FIX 2026-09-08: --what sims used to run FULL TRAINING first
        # (run_train was unconditional) — hours of unintended GPU quota
        # for a lane that only needs the configured zero-shot scoring.
        if args.what == 'tracks':
            from model_tracks.colab import run as run_suite
            run_suite(suite_archive, suite_run_tag, resume=bool(args.resume_run),
                      resume_archive=suite_recovery if args.resume_run and suite_recovery.is_file() else None,
                      git_inputs=suite_git_inputs)
        elif args.what == "sims":
            run_sims()
        elif args.what == "bundle":
            if bool(training_cfg().cpu_bundle_prep.lane):
                # Owner ruling 8: data-bundle production lives in its own
                # lane file; this forward is the thin passthrough.  With the
                # lane's config flag off, the original direct call runs and
                # behavior is byte-identical.
                from cli.colab_data_bundle_prep import run_cpu_bundle_prep
                run_cpu_bundle_prep(dataset_csv=args.dataset_csv)
            else:
                run_bundle(dataset_csv=args.dataset_csv)
        elif args.what == "mixed":
            local_mixed_run = run_mixed(
                args.train_frac, args.epochs, model=args.model, loss=args.loss,
            )
        elif args.what == "smoke":
            smoke_sample = args.sample if args.sample is not None else _SMOKE_SAMPLE
            local_training_run = run_train(
                args.train_frac, _SMOKE_EPOCHS, sample=smoke_sample,
                workers=_SMOKE_WORKERS,
                smoke=True,
                inference_sample=smoke_sample,
                inference_device="cuda" if GPU.upper() != "CPU" else "cpu",
                run_label=args.run_label,
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
                loss=args.loss,
                # The smoke lane's full CSV is already in the Git checkout.
                # It tests inference against the same file without uploading
                # a separate held-out split.
                train_only=False,
                remote_dataset_csv="data/dataset_deduped.csv",
                # These bundles are committed with the other immutable
                # runtime inputs.  Colab only consumes them on the GPU; no
                # CSV or prepared-data upload is part of this lane.
                remote_prepared_bundles=[
                    "data/prepared/smoke_1000/worker_1_baseline.pkl.gz",
                    "data/prepared/smoke_1000/worker_2_baseline.pkl.gz",
                ] if smoke_sample == 1000 else None,
                remote_validation_csv=f"{REMOTE_ROOT}/data/dataset_deduped.csv",
                # The status/log poll is the only Colab control-channel user
                # during smoke; checkpoint syncing waits for full runs.
                incremental_sync=False,
            )
        elif args.what == "dual-train":
            if args.resume_run:
                raise ValueError("--resume-run is not supported for dual-train")
            # Both workers start from the same shipped base model and prepared
            # data, but write fully isolated checkpoints/results. The ANN
            # encoder is always trained with MNRL; the matcher uses --loss.
            local_training_run = run_train(
                args.train_frac, args.epochs, sample=args.sample, workers=2,
                model=args.model, run_label="matcher,ann_embedding",
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
                loss=args.loss,
                worker_losses=[args.loss, "mnrl"],
                train_only=args.train_only,
            )
        elif args.what == "hpo":
            if args.hpo_jobs < 1:
                raise ValueError("--hpo-jobs must be >= 1")
            local_hpo_run = run_hpo(
                args.hpo_mode,
                resume=args.resume_hpo,
                trial_jobs=args.hpo_jobs,
                persistence=args.hpo_persistence,
                loss=args.loss,
            )
        else:
            checkout_full_bundles = (
                list(_COLAB.full_prepared_bundles)
                if (
                    args.what == "train"
                    and args.workers == 1
                    and args.model is None
                    and args.sample is None
                    and args.resume_run is None
                )
                else None
            )
            local_training_run = run_train(
                args.train_frac, args.epochs, sample=args.sample, workers=args.workers,
                resume_run=args.resume_run, model=args.model,
                run_label=args.run_label,
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
                loss=args.loss,
                train_only=args.train_only,
                remote_dataset_csv=(
                    _COLAB.training_dataset_csv if checkout_full_bundles else None
                ),
                remote_prepared_bundles=checkout_full_bundles,
                inference_device="cuda" if GPU.upper() != "CPU" else "cpu",
            )
        if local_hpo_run is not None:
            print(_stamp(), "[hpo] publishing snapshots on local CPU ...", flush=True)
            publish_local_hpo_results(local_hpo_run, args.hpo_persistence)
        completed = True
    except BaseException:
        print(
            _stamp(),
            "[launcher] traceback before teardown; result transfer is incomplete:",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        # Release the runtime after every outcome.  Retaining a failed VM can
        # consume quota indefinitely; callers that intentionally need live
        # recovery must make that choice explicitly with --keep-alive.
        if not args.keep_alive:
            stop()
        else:
            print(_stamp(), "\n[info] --keep-alive specified. VM is still running.")
        # A lane that failed before its run_train call would otherwise leave
        # the concurrent bundle build writing into a closing log file.
        drain_local_bundle_prewarm()
        drain_validation_upload_prewarm()
        # Announce completion while the tee is still installed; closing first
        # would send this to the terminal only and leave the transcript ending
        # at the last teardown line.  Suppressed on failure, which already
        # reported a traceback above, so [done] stays a success signal.
        if completed:
            print(
                _stamp(),
                "\n[done] compressed results downloaded and completed locally"
                if args.what == "tracks" else "\n[done] artifacts downloaded locally",
                flush=True,
            )
        else:
            print(_stamp(), "\n[failed] launch did not complete successfully", flush=True)
        close_live_log()
        release_colab_launch_lock(launch_lock)

if __name__ == "__main__":
    main()

# Session/runtime/environment moved to cli.colab_runtime (split phase C);
# imported at module end so _timed_colab resolution at decoration time works.
from cli.colab_runtime import (
    _BOOTSTRAP,
    _env_value,
    _forget_cached_session,
    _is_keep_alive_daemon,
    _optuna_env_script,
    _remote_auth_env_script,
    _runtime_install_command,
    _verify_session_handshake,
    _wandb_env_script,
    ensure_session,
    install_deps,
    keep_alive_daemon_pids,
    log_gpu_profile,
    prepare_remote_layout,
    run_data_prep,
    stop_keep_alive_daemon,
    validate_runtime_checkout,
    verify_remote_models,
    verify_training_inputs,
)  # split phase C

