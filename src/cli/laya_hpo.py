"""Laya fine-tune HPO lane (branch laya-hpo).

A remote Kaggle lane that runs Optuna/TPE HPO for the laya fine-tune against a
SHARED PostgreSQL study. One Kaggle session exposes 2xT4; the staged kernel
spawns one worker PER visible device, and every worker (and every concurrent
Kaggle session) claims trials from the SAME study through Optuna's RDBStorage
while ``training.hpo_fencing`` lease epochs fence zombie workers and
``training.hpo_champions`` keeps the best trial transactionally.

Design boundaries:

* Read-only reuse. This module imports the shared HPO control plane
  (``training.hpo_control_plane`` / ``hpo_fencing`` / ``hpo_champions``) and the
  landed laya fine-tune surfaces (``cli.laya_lane``) — it never edits them.
* The remote kernel CANNOT import the repo (the finetune kernel precedent: it
  installs ``laya`` over pip and receives attached datasets). So the shared
  primitives are injected into the staged script via ``inspect.getsource``,
  byte-for-byte, exactly like ``cli.laya_lane`` injects ``core.laya_controls``.
* The search space is YAML/config SSOT (``config/laya_hpo_space.yaml``); no
  bound, type, choice or target is a code literal.
* ``OPTUNA_STORAGE_URL`` is a runtime secret. Staging fails LOUD when it is
  absent, and the URL is baked ONLY into the (gitignored) staged kernel script
  so the remote process can reach the DB. It is NEVER written to the receipt,
  the kernel metadata, the search-space YAML or any other stored manifest
  (``_assert_secret_absent`` guards the receipt).

The objective maximizes DEV accuracy and records dev loss as a secondary
``trial.set_user_attr``. The held-out/test split is never an HPO signal.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# Read-only reuse: the shared control plane, the landed laya fine-tune lane,
# and the pure HPO runtime (injected verbatim into the remote kernel).
from cli import laya_lane
from core.common import TRAIN_ROOT, training_cfg
from core.laya_config import FinetuneSpec
from core.manifest import atomic_write_json
from training import hpo_champions, hpo_control_plane, hpo_fencing, laya_hpo_runtime
from training.hpo_control_plane import (
    create_storage,
    generation_study_name,
    storage_from_environment,
)
from training.laya_hpo_runtime import (
    finished_trial_count,
    objective_value,
    route_dials,
    sample_dials,
)

# ── staging surface ────────────────────────────────────────────────────────
HPO_DECISION = "laya-hpo"
HPO_CODE_FILE = "laya_hpo.py"
HPO_RECEIPT_FILE = "laya-hpo.receipt.json"
SPACE_FILE_NAME = "laya_hpo_space.yaml"
DEFAULT_MODEL_KEY = "laya"
TARGETS = ("config", "control")
TYPES = ("int", "float", "categorical")
_MAX_WORKERS = 2

# The optuna env var the shared control plane reads. One name, one place.
OPTUNA_URL_ENV = "OPTUNA_STORAGE_URL"
GENERATION_ID_ENV = "EUROMONITOR_HPO_GENERATION_ID"

_TOKEN_PATTERN = re.compile(r"@[A-Z][A-Z0-9_]*@")


# ── search-space SSOT (config/laya_hpo_space.yaml) ─────────────────────────
def space_path(path: str | Path | None = None) -> Path:
    """The search-space config path (explicit override wins)."""
    return Path(path) if path else (TRAIN_ROOT / "config" / SPACE_FILE_NAME)


def load_space(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the laya HPO search space (SSOT; fail-loud)."""
    resolved = space_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"laya HPO search space missing: {resolved}")
    space = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise TypeError(f"laya HPO search space must be a mapping: {resolved}")
    validate_space(space)
    return space


def validate_space(space: dict[str, Any]) -> None:
    """Reject a malformed space before a single trial is sampled.

    The dial names are asserted against ``FinetuneSpec`` fields TODAY so a
    rename on the laya training-dials branch fails the loader loudly rather
    than silently sampling a key no trainer reads.
    """
    dials = space.get("dials")
    if not isinstance(dials, dict) or not dials:
        raise ValueError("laya HPO space must declare a non-empty 'dials' mapping")
    known_fields = set(FinetuneSpec.model_fields)
    for name, spec in dials.items():
        if not isinstance(spec, dict):
            raise TypeError(f"laya HPO dial {name!r} must be a mapping")
        if name not in known_fields:
            raise ValueError(
                f"laya HPO dial {name!r} is not a FinetuneSpec field; update "
                "config/laya_hpo_space.yaml when the training dials land")
        target = spec.get("target")
        if target not in TARGETS:
            raise ValueError(
                f"laya HPO dial {name!r} target must be one of {TARGETS}, "
                f"got {target!r}")
        kind = spec.get("type")
        if kind not in TYPES:
            raise ValueError(
                f"laya HPO dial {name!r} type must be one of {TYPES}, "
                f"got {kind!r}")
        if kind in ("int", "float"):
            if "lo" not in spec or "hi" not in spec:
                raise ValueError(f"laya HPO dial {name!r} needs lo and hi")
            lo, hi = spec["lo"], spec["hi"]
            if lo > hi:
                raise ValueError(
                    f"laya HPO dial {name!r} has an invalid range [{lo}, {hi}]")
            if kind == "int" and not (isinstance(lo, int) and isinstance(hi, int)):
                raise ValueError(f"laya HPO dial {name!r} int bounds must be ints")
        else:  # categorical
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"laya HPO dial {name!r} needs a non-empty choices list")
    if int(space.get("n_trials", 0)) <= 0:
        raise ValueError("laya HPO space n_trials must be a positive integer")
    if int(space.get("n_jobs", 0)) <= 0:
        raise ValueError("laya HPO space n_jobs must be a positive integer")
    objective = space.get("objective") or {}
    if objective.get("direction") not in ("maximize", "minimize"):
        raise ValueError("laya HPO space objective.direction must be maximize|minimize")
    if objective.get("primary") != "dev_accuracy":
        raise ValueError("laya HPO objective.primary must be dev_accuracy")
    if objective.get("secondary") != "dev_loss":
        raise ValueError("laya HPO objective.secondary must be dev_loss")
    if objective.get("forbidden") != "test":
        raise ValueError("laya HPO objective.forbidden must be test")


def space_digest(space: dict[str, Any]) -> str:
    """A stable digest of the space (for receipts; never a secret)."""
    payload = json.dumps(space, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# `sample_dials` / `route_dials` live in `training.laya_hpo_runtime` (imported
# above) so the SAME source is injected into the remote kernel. `apply_dials`
# is the host-side alias the tests and receipts use.
apply_dials = route_dials


# ── shared study identity + storage ────────────────────────────────────────
def study_identity(*, space: dict[str, Any] | None = None,
                   generation_id: str | None = None,
                   model_key: str | None = None) -> tuple[str, str, str]:
    """``(generation_id, model_key, study_name)`` for the shared study.

    The generation id is runtime environment (it must change when the corpus or
    recipe generation changes); the model key is space/config SSOT.
    """
    resolved_space = space if space is not None else load_space()
    generation = (generation_id or os.environ.get(GENERATION_ID_ENV, "")).strip()
    if not generation:
        raise RuntimeError(
            f"[laya-hpo] {GENERATION_ID_ENV} is required: the shared study "
            "name is generation-scoped so a new corpus/recipe can never mix "
            "into an old TPE history. Export it before staging.")
    key = (model_key or resolved_space.get("model_key") or DEFAULT_MODEL_KEY).strip()
    return generation, key, generation_study_name(
        generation_id=generation, model_key=key)


def require_optuna_url(read_env: Callable[[str], str | None] | None = None) -> str:
    """Return the shared PostgreSQL Optuna URL or fail LOUD.

    The URL is read through the lane's ``_env_value`` (repo ``.env`` then the
    process env) like ``cli.colab_runtime._optuna_env_script``; a missing or
    non-PostgreSQL URL is a hard error — the lane must never silently run an
    HPO sweep no one can coordinate.
    """
    reader = read_env if read_env is not None else laya_lane._env_value
    url = (reader(OPTUNA_URL_ENV) or "").strip()
    if not url:
        raise RuntimeError(
            f"[laya-hpo] {OPTUNA_URL_ENV} is missing. Concurrent HPO requires "
            "a shared hosted PostgreSQL (Neon/Supabase/RDS) reachable from "
            "Kaggle over the public internet; add the URL to the runtime "
            "secret store / .env and re-stage. It is never written to YAML, "
            "receipts or manifests.")
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError(
            f"[laya-hpo] {OPTUNA_URL_ENV} must be a PostgreSQL URL "
            "(postgresql:// or postgresql+psycopg://); SQLite is not supported "
            "for concurrent HPO.")
    return url


def resolve_study_config(*, space: dict[str, Any] | None = None,
                         generation_id: str | None = None,
                         model_key: str | None = None):
    """``(storage, study_name, storage_config)`` from the shared control plane.

    ``create_storage`` / ``storage_from_environment`` import Optuna lazily, so
    importing this module stays GPU/DB-free.
    """
    storage_config = storage_from_environment()
    _, _, name = study_identity(space=space, generation_id=generation_id,
                                model_key=model_key)
    return create_storage(storage_config), name, storage_config


# ── remote-kernel composition (read-only reuse, injected verbatim) ─────────
def hpo_runtime_source() -> str:
    """The shared HPO primitives, injected verbatim into the staged script.

    The remote kernel attaches datasets and installs ``laya`` over pip; it does
    NOT clone the repo, so it cannot ``import training.hpo_*``. Concatenating
    the real module sources keeps ONE implementation (the
    ``FINETUNE_CONTROL_LOGIC_SOURCE`` precedent). ``from __future__`` lines are
    stripped because the script has exactly one, at the top.
    """
    chunks: list[str] = []
    for module in (hpo_control_plane, hpo_fencing, hpo_champions,
                   laya_hpo_runtime):
        text = inspect.getsource(module)
        cleaned = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("from __future__ import"))
        chunks.append(cleaned.strip("\n"))
    return "\n\n".join(chunks)


def _optuna_env_script(url: str) -> str:
    """The one baked line that injects the URL into the remote process only."""
    return f"os.environ[{OPTUNA_URL_ENV!r}] = {url!r}\n"


def _assert_secret_absent(payload: Any, secret: str) -> None:
    """Fail-loud guard: a secret must never reach a stored manifest."""
    if not secret:
        return
    encoded = json.dumps(payload, default=str)
    if secret in encoded:
        raise RuntimeError(
            "refusing to persist OPTUNA_STORAGE_URL in a stored payload; the "
            "URL belongs only in the staged kernel script / runtime env")


def _kernel_script_gate(script: str) -> None:
    """Every ``@TOKEN@`` replaced and the payload compiles — or never stage."""
    leftovers = sorted(set(_TOKEN_PATTERN.findall(script)))
    if leftovers:
        raise ValueError(
            f"staged laya HPO kernel has unreplaced tokens: {leftovers}; "
            "regenerate the template")
    compile(script, "<laya-hpo-payload>", "exec")


def _current_git_branch() -> str:
    """The branch baked as BRANCH (falls back to the config branch)."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=TRAIN_ROOT,
        capture_output=True, text=True, check=False)
    branch = result.stdout.strip() if result.returncode == 0 else ""
    if not branch or branch == "HEAD":
        return training_cfg().kaggle.branch
    return branch


def stage_laya_hpo_kernel(*, revision: str | None = None,
                          run_tag: str | None = None,
                          generation_id: str | None = None,
                          n_trials: int | None = None,
                          n_jobs: int | None = None,
                          space_config: str | Path | None = None,
                          kernel_slug: str | None = None) -> dict[str, Any]:
    """Stage the Kaggle HPO kernel payload (dry-safe; no push, no run).

    Writes under results/laya_lane/kaggle/laya-hpo/:
      kernel-metadata.json + laya_hpo.py + laya-hpo.receipt.json
      (+ the staged corpus dataset payload, which carries no secret).
    Fail-loud preconditions:
      * ``OPTUNA_STORAGE_URL`` present and PostgreSQL;
      * ``EUROMONITOR_HPO_GENERATION_ID`` set (generation-scoped study);
      * a corpus dataset slug + base-model dataset slug;
      * the published-tip invariant (origin/<branch> == HEAD).

    The URL is baked ONLY into the staged ``laya_hpo.py`` (gitignored); it is
    asserted absent from the receipt.
    """
    spec = training_cfg().laya
    space = load_space(space_config)
    url = require_optuna_url()
    generation, key, study_name = study_identity(
        space=space, generation_id=generation_id)

    slug = kernel_slug or space.get("kernel_slug")
    if not slug:
        raise RuntimeError(
            "laya-hpo kernel slug is unset; set kernel_slug in "
            f"{space_path(space_config)} (owner/slug) before staging")
    dataset_slug = spec.finetune_dataset_slug
    if not dataset_slug:
        raise RuntimeError(
            "config laya.finetune_dataset_slug is unset; the corpus travels "
            "as that dataset (owner/slug) — name it before staging")
    base_dataset = spec.base_model_dataset
    if not base_dataset:
        raise RuntimeError(
            "config laya.base_model_dataset is unset; the base checkpoint "
            "travels as that dataset (owner/slug) — name it before staging")

    repository = training_cfg().kaggle.repository
    branch = _current_git_branch()
    revision = revision or laya_lane._git_revision()
    from core import runtime_inputs

    tip = runtime_inputs.require_published_tip_match(revision, repository, branch)

    dataset_receipt = laya_lane.stage_finetune_dataset_payload(
        dataset_slug=dataset_slug,
        corpus_dir=TRAIN_ROOT / spec.finetune_corpus_dir,
        kind=HPO_DECISION)

    stage = laya_lane.staging_dir() / "kaggle" / HPO_DECISION
    stage.mkdir(parents=True, exist_ok=True)
    tag = run_tag or (spec.run_tag_prefix + "hpo_" + laya_lane.decision_tag())
    budget_trials = int(n_trials if n_trials is not None else space["n_trials"])
    budget_jobs = int(n_jobs if n_jobs is not None else space["n_jobs"])

    preflight = laya_lane._template(laya_lane.FINETUNE_RUNTIME_PREFLIGHT, {
        "TRAIN_JSONL": laya_lane.FINETUNE_CORPUS_FILES[0],
        "DEV_JSONL": laya_lane.FINETUNE_CORPUS_FILES[1],
        "TEST_JSONL": laya_lane.FINETUNE_CORPUS_FILES[2],
    })
    values: dict[str, str] = {
        "LAYA_PACKAGE": spec.finetune_package,
        "RUN_TAG": tag,
        "TRAIN_JSONL": laya_lane.FINETUNE_CORPUS_FILES[0],
        "DEV_JSONL": laya_lane.FINETUNE_CORPUS_FILES[1],
        "BASE_MODEL_ARCHIVE": spec.base_model_archive,
        "BASE_MODEL_DIR": spec.base_model_dir,
        "FINETUNE_DEVICE": spec.finetune.device,
        "BASE_FINETUNE_CONFIG": repr(laya_lane.finetune_config(spec)),
        "BASE_FINETUNE_CONTROL": repr(laya_lane.finetune_control(spec)),
        "HPO_SPACE": repr(space),
        "N_TRIALS": str(budget_trials),
        "N_JOBS": str(budget_jobs),
        "SEED": str(int(space["seed"])),
        "GENERATION_ID": generation,
        "MODEL_KEY": key,
        "WANDB_API_KEY": laya_lane._env_value("WANDB_API_KEY") or "",
        "WANDB_PROJECT": laya_lane._wandb_project(),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
        "HPO_RUNTIME_SOURCE": hpo_runtime_source(),
        "OPTUNA_ENV_SCRIPT": _optuna_env_script(url),
        "RUNTIME_PREFLIGHT": preflight,
    }
    script = laya_lane._template(_HPO_KERNEL_TEMPLATE, values)
    _kernel_script_gate(script)

    metadata: dict[str, Any] = {
        "id": slug,
        "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
        "code_file": HPO_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_slug, base_dataset],
        "kernel_sources": [],
        "competition_sources": [],
        "is_private": True,
    }
    atomic_write_json(metadata, stage / "kernel-metadata.json")
    (stage / HPO_CODE_FILE).write_text(script, encoding="utf-8")

    receipt: dict[str, Any] = {
        "kernel": slug,
        "kind": HPO_DECISION,
        "gpu": "T4 (2x when the session exposes it; one worker per device)",
        "run_tag": tag,
        "staged": str(stage),
        "code_file": HPO_CODE_FILE,
        "dataset": {
            "slug": dataset_slug,
            "payload": dataset_receipt["payload"],
            "files": dataset_receipt["files"],
        },
        "base_model": {
            "dataset": base_dataset,
            "archive": spec.base_model_archive,
            "dir": spec.base_model_dir,
        },
        "study": {"generation_id": generation, "model_key": key,
                  "study_name": study_name},
        "space": {
            "path": str(space_path(space_config)),
            "version": space["space_version"],
            "digest": space_digest(space),
            "dials": sorted(space["dials"]),
        },
        "budget": {"n_trials": budget_trials, "n_jobs": budget_jobs,
                   "seed": int(space["seed"]), "max_workers": _MAX_WORKERS},
        "objective": space["objective"],
        "optuna_storage": {
            "required_env": OPTUNA_URL_ENV,
            "injected_into_kernel": True,
            "url_persisted_to_manifest": False,
        },
        "laya_package": spec.finetune_package,
        "published_pin": {"repository": repository, "branch": branch,
                          "revision": revision},
        "published_tip": tip,
    }
    # The one hard guarantee: the URL is not in the stored receipt.
    _assert_secret_absent(receipt, url)
    atomic_write_json(receipt, stage / HPO_RECEIPT_FILE)
    laya_lane._log_lane(
        f"staged kaggle laya-hpo kernel run_tag={tag} study={study_name} "
        f"-> {stage}")
    return receipt


# ── the embedded Kaggle kernel script ──────────────────────────────────────
# Shape mirrors cli.laya_lane.FINETUNE_KERNEL_SCRIPT: a self-contained script
# that installs laya + the HPO runtime over pip, injects the shared primitives
# verbatim, and runs TPE trials against the shared PostgreSQL study.
_HPO_KERNEL_TEMPLATE = '''\
"""ER laya fine-tune HPO on a Kaggle 2xT4 session (cli.laya_hpo).

Installs laya + Optuna/SQLAlchemy/psycopg over pip, reads the attached JSONL
corpus (train/dev; the held-out test split is NEVER an HPO signal) and the
attached base-model archive, then spawns one worker process PER visible GPU.
Every worker claims trials from the SAME shared PostgreSQL Optuna study (the
URL arrives through the injected OPTUNA_STORAGE_URL environment variable),
issues a fencing lease per trial, and promotes the best DEV accuracy through
the transactional champion registry. Rank-free: no DDP, two independent
trials run concurrently, one per T4.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
TRAIN_JSONL = "@TRAIN_JSONL@"
DEV_JSONL = "@DEV_JSONL@"
BASE_MODEL_ARCHIVE = "@BASE_MODEL_ARCHIVE@"
BASE_MODEL_DIR = "@BASE_MODEL_DIR@"
FINETUNE_DEVICE = "@FINETUNE_DEVICE@"
HPO_SPACE = @HPO_SPACE@
BASE_FINETUNE_CONFIG = @BASE_FINETUNE_CONFIG@
BASE_FINETUNE_CONTROL = @BASE_FINETUNE_CONTROL@
N_TRIALS = @N_TRIALS@
N_JOBS = @N_JOBS@
SEED = @SEED@
GENERATION_ID = "@GENERATION_ID@"
MODEL_KEY = "@MODEL_KEY@"
WANDB_API_KEY = "@WANDB_API_KEY@"
WANDB_PROJECT = "@WANDB_PROJECT@"

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

# The shared HPO primitives (training.hpo_control_plane / hpo_fencing /
# hpo_champions) injected VERBATIM. The kernel cannot import the repo.
@OPTUNA_ENV_SCRIPT@
@HPO_RUNTIME_SOURCE@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")
WANDB_RUN = None
# Per-trial globals the perf patch reads (this worker runs ONE trial at a
# time; each worker is its own process, so the globals never race).
FINETUNE_CONFIG = {}
FINETUNE_CONTROL = {}
FINETUNE_DEV_ROWS = None
FINETUNE_OUTPUT_DIR = None
FINETUNE_CONTROL_RESULT = None


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-hpo " + stamp + "] " + str(line), flush=True)


def wandb_init():
    global WANDB_RUN
    if not WANDB_API_KEY:
        return None
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    try:
        import wandb
        WANDB_RUN = wandb.init(project=WANDB_PROJECT, name=RUN_TAG,
                               config={"hpo_space": HPO_SPACE})
        log("wandb run " + str(getattr(WANDB_RUN, "id", "")))
    except Exception as error:
        log("wandb init skipped: " + type(error).__name__ + ": "
            + str(error)[:200])
        WANDB_RUN = None
    return WANDB_RUN


def wandb_log_epoch(epoch, mean, extra=None):
    if WANDB_RUN is None:
        return
    payload = {"epoch": epoch + 1, "train/mean_loss": mean}
    if isinstance(extra, dict):
        payload.update(extra)
    WANDB_RUN.log(payload, step=epoch)


def wandb_log_control_summary(result):
    if WANDB_RUN is None or not isinstance(result, dict):
        return
    payload = {"select/best_dev_accuracy": result.get("best_dev_accuracy"),
               "early_stop/stopped": 1 if result.get("stopped") else 0}
    WANDB_RUN.log({k: v for k, v in payload.items() if v is not None})


def wandb_finish():
    if WANDB_RUN is not None:
        try:
            WANDB_RUN.finish()
        except Exception:
            pass


@DEVICE_PATCH@

@PERF_PATCH@


def pip_install_runtime():
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check",
               "optuna", "sqlalchemy", "psycopg[binary]", "zstandard"]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pip_install_laya():
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_optuna_url():
    url = os.environ.get("OPTUNA_STORAGE_URL", "").strip()
    if not url:
        raise SystemExit(
            "[laya-hpo] OPTUNA_STORAGE_URL is missing; the shared PostgreSQL "
            "Optuna study cannot be reached. Re-stage with the secret set.")
    # SQLAlchemy's bare postgresql:// defaults to the psycopg2 driver; the
    # session ships psycopg3 (psycopg[binary]), so pin the driver explicitly
    # for BOTH Optuna's RDBStorage and the fencing/champion engines.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
        os.environ["OPTUNA_STORAGE_URL"] = url
    return url


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged laya "
        "corpus / base-model dataset)")


def open_zstd(path):
    try:
        from compression import zstd
        return zstd.open(path, "rb")
    except ImportError:
        pass
    try:
        import zstandard
        return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-input", "--disable-pip-version-check",
                    "zstandard"], check=True)
    import zstandard
    return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))


def extract_base_model(archive):
    destination = WORKING / "base_model"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stream = open_zstd(str(archive))
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            try:
                tar.extractall(destination, filter="data")
            except TypeError:
                tar.extractall(destination)
    finally:
        stream.close()
    candidate = destination / BASE_MODEL_DIR
    if (candidate / "rl_agent_config.json").is_file():
        return candidate
    for found in sorted(destination.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError("base-model archive carried no rl_agent_config.json")


def run_laya_finetune(train_path, dev_path, base_model, out_dir, device):
    apply_perf_patch()
    apply_device_patch()
    from laya import train as laya_train
    config = laya_train.TrainConfig(**FINETUNE_CONFIG,
                                    eval_data=str(dev_path))
    config.validate()
    globals()["FINETUNE_OUTPUT_DIR"] = str(out_dir)
    if FINETUNE_CONTROL.get("eval_dev"):
        try:
            globals()["FINETUNE_DEV_ROWS"] = laya_train.read_jsonl(str(dev_path))
        except Exception as error:
            log("dev rows load skipped: " + str(error)[:200])
    log("TrainConfig: " + json.dumps(FINETUNE_CONFIG, sort_keys=True))
    return laya_train.finetune(data=str(train_path), model_dir=str(base_model),
                               output_dir=str(out_dir), config=config,
                               device=device)


def dev_loss_from_report(out_dir):
    report = Path(out_dir) / "train_report.json"
    if not report.is_file():
        return None
    try:
        parsed = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return None
    block = parsed.get("after") or parsed.get("before") or {}
    value = block.get("loss")
    return None if value is None else float(value)


def run_trial(trial, device, train_path, dev_path, base_model):
    dials = sample_dials(trial, HPO_SPACE)
    config_dict, control_dict = route_dials(
        BASE_FINETUNE_CONFIG, BASE_FINETUNE_CONTROL, dials, HPO_SPACE)
    globals()["FINETUNE_CONFIG"] = config_dict
    globals()["FINETUNE_CONTROL"] = control_dict
    globals()["FINETUNE_CONTROL_RESULT"] = None
    globals()["FINETUNE_DEV_ROWS"] = None
    out_dir = WORKING / ("checkpoint_trial_" + str(int(trial.number)))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("trial %d start device=%s dials=%s"
        % (int(trial.number), device, json.dumps(dials, sort_keys=True)))
    run_laya_finetune(train_path, dev_path, base_model, out_dir, device)
    result = globals().get("FINETUNE_CONTROL_RESULT") or {}
    accuracy = result.get("best_dev_accuracy")
    if accuracy is None:
        raise RuntimeError("trial %d produced no best_dev_accuracy"
                           % int(trial.number))
    dev_loss = dev_loss_from_report(out_dir)
    log("trial %d done dev_accuracy=%s dev_loss=%s"
        % (int(trial.number), accuracy, dev_loss))
    return float(accuracy), dev_loss, out_dir


def make_objective(device, train_path, dev_path, base_model, lease_store,
                   champion_store):
    def objective(trial):
        return objective_value(
            trial,
            lambda trial: run_trial(
                trial, device, train_path, dev_path, base_model),
            lease_store, champion_store, GENERATION_ID, MODEL_KEY)
    return objective


def run_worker(device):
    ensure_optuna_url()
    if os.environ.get("ER_LAYA_HPO_SKIP_INSTALL") != "1":
        pip_install_runtime()
        pip_install_laya()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_config = storage_from_environment()
    storage = create_storage(storage_config)
    study_name = generation_study_name(generation_id=GENERATION_ID,
                                       model_key=MODEL_KEY)
    study = optuna.create_study(
        study_name=study_name,
        direction=str(HPO_SPACE.get("objective", {}).get("direction", "maximize")),
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=storage,
        load_if_exists=True)
    fail_stale_trials(study)
    lease_store = TrialLeaseStore(storage_config.url)
    champion_store = ChampionStore(storage_config.url)

    train_path = resolve_input(TRAIN_JSONL)
    dev_path = resolve_input(DEV_JSONL)
    base_model = extract_base_model(resolve_input(BASE_MODEL_ARCHIVE))
    log("worker device=%s train=%s dev=%s base=%s"
        % (device, train_path.name, dev_path.name, base_model.name))

    try:
        wandb_init()
    except Exception as error:
        log("wandb init failed: " + str(error)[:200])

    objective = make_objective(device, train_path, dev_path, base_model,
                               lease_store, champion_store)
    prior = finished_trial_count(study, ("COMPLETE", "PRUNED", "FAIL"))
    remaining = max(0, int(N_TRIALS) - prior)
    per_worker = max(0, math.ceil(remaining / max(1, int(N_JOBS))))
    log("worker budget prior=%d remaining=%d per_worker=%d study=%s"
        % (prior, remaining, per_worker, study_name))
    if per_worker:
        study.optimize(objective, n_trials=per_worker, n_jobs=1,
                       catch=(Exception,))
    wandb_finish()


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_session_receipt():
    import optuna
    storage_config = storage_from_environment()
    study_name = generation_study_name(generation_id=GENERATION_ID,
                                       model_key=MODEL_KEY)
    study = optuna.load_study(study_name=study_name,
                              storage=create_storage(storage_config))
    complete = [trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None]
    best = max(complete, key=lambda trial: trial.value) if complete else None
    trials = [{
        "number": int(trial.number),
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "dev_accuracy": trial.user_attrs.get("dev_accuracy"),
        "dev_loss": trial.user_attrs.get("dev_loss"),
    } for trial in study.trials]
    receipt = {
        "kernel": "laya-hpo",
        "run_tag": RUN_TAG,
        "study": study_name,
        "generation_id": GENERATION_ID,
        "model_key": MODEL_KEY,
        "space_version": HPO_SPACE.get("space_version"),
        "budget": {"n_trials": int(N_TRIALS), "n_jobs": int(N_JOBS),
                   "seed": int(SEED)},
        "objective": HPO_SPACE.get("objective"),
        "trials": trials,
        "best": None if best is None else {
            "number": int(best.number),
            "value": best.value,
            "params": best.params,
            "dev_loss": best.user_attrs.get("dev_loss"),
            "checkpoint": best.user_attrs.get("checkpoint"),
        },
        "corpus_sha256": {
            TRAIN_JSONL: sha256_of(resolve_input(TRAIN_JSONL)),
            DEV_JSONL: sha256_of(resolve_input(DEV_JSONL)),
        },
        "published_pin": {"repository": REPOSITORY, "branch": BRANCH,
                          "revision": REVISION},
    }
    WORKING.mkdir(parents=True, exist_ok=True)
    path = WORKING / "laya-hpo.receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    log("wrote " + str(path) + " best="
        + (str(best.value) if best is not None else "none"))
    return receipt


def main():
    ensure_optuna_url()
    pip_install_runtime()
    pip_install_laya()
    import torch
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    workers = max(1, min(int(N_JOBS), gpu_count or 1))
    study_name = generation_study_name(generation_id=GENERATION_ID,
                                       model_key=MODEL_KEY)
    log("session gpus=%d workers=%d study=%s budget=%d"
        % (gpu_count, workers, study_name, int(N_TRIALS)))
    script = os.path.abspath(__file__)
    processes = []
    for index in range(workers):
        env = os.environ.copy()
        env["ER_LAYA_HPO_WORKER_DEVICE"] = "cuda:" + str(index)
        env["ER_LAYA_HPO_SKIP_INSTALL"] = "1"
        processes.append(subprocess.Popen([sys.executable, script], env=env))
    codes = [process.wait() for process in processes]
    log("workers exited: " + str(codes))
    receipt = write_session_receipt()
    # Stage ONLY the champion checkpoint (the per-trial checkpoints would make
    # the archive enormous); the receipt is the decision trail.
    WORKING.mkdir(parents=True, exist_ok=True)
    with tarfile.open(WORKING / "laya_hpo.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        tar.add(WORKING / "laya-hpo.receipt.json",
                arcname="laya-hpo.receipt.json")
        champion = ((receipt.get("best") or {}).get("checkpoint") or "")
        if champion and Path(champion).is_dir():
            tar.add(champion, arcname="champion")
            log("staged champion checkpoint " + champion)
    log("staged laya_hpo.tar.gz + receipt in /kaggle/working")
    if any(code != 0 for code in codes):
        raise SystemExit("laya HPO worker failure: exit codes " + str(codes))


if __name__ == "__main__":
    _worker_device = os.environ.get("ER_LAYA_HPO_WORKER_DEVICE")
    if _worker_device:
        run_worker(_worker_device)
    else:
        main()
'''


def main(argv: list[str] | None = None) -> int:
    """Stage (and optionally push) the laya HPO kernel.

    Offline by default: ``stage_laya_hpo_kernel`` writes the payload and prints
    the receipt. ``--execute`` additionally runs ``kaggle kernels push`` through
    the landed laya lane's gated push (which records the session id for the
    verified stop). The owner launches the sweep; this lane never starts one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="push the staged kernel to Kaggle (default: "
                             "offline dry-run staging only)")
    parser.add_argument("--generation-id", default=None,
                        help="shared-study generation id (default: "
                             "EUROMONITOR_HPO_GENERATION_ID)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="cluster TPE budget (default: the space YAML)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="in-session GPU workers (default: the space YAML)")
    parser.add_argument("--revision", default=None,
                        help="published git revision to pin (default: HEAD)")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--space", type=Path, default=None,
                        help="search-space YAML (default: the SSOT config)")
    args = parser.parse_args(argv)
    receipt = stage_laya_hpo_kernel(
        revision=args.revision, run_tag=args.run_tag,
        generation_id=args.generation_id, n_trials=args.n_trials,
        n_jobs=args.n_jobs, space_config=args.space)
    print(json.dumps(receipt, indent=2, default=str), flush=True)
    if args.execute:
        plan = laya_lane.push_kaggle_kernel(Path(receipt["staged"]),
                                            execute=True)
        print(json.dumps(plan, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GENERATION_ID_ENV",
    "HPO_CODE_FILE",
    "HPO_DECISION",
    "HPO_RECEIPT_FILE",
    "OPTUNA_URL_ENV",
    "apply_dials",
    "finished_trial_count",
    "hpo_runtime_source",
    "load_space",
    "objective_value",
    "require_optuna_url",
    "resolve_study_config",
    "route_dials",
    "sample_dials",
    "space_digest",
    "stage_laya_hpo_kernel",
    "study_identity",
    "validate_space",
]
