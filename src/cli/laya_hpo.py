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
  (``assert_secret_absent`` guards the receipt).

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
from training import (
    hpo_champions,
    hpo_control_plane,
    hpo_fencing,
    hpo_observability,
    hpo_persistence,
    hpo_registry,
    laya_hpo_options,
    laya_hpo_runtime,
)
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
COLAB_ENTRY_FILE = "laya_hpo_colab.py"
LANES = ("kaggle", "colab")
SPACE_FILE_NAME = "laya_hpo_space.yaml"
TARGETS = ("config", "control")
TYPES = ("int", "float", "categorical")
# The Colab working/input roots the entry driver overrides; the Kaggle lane
# keeps its defaults (runtime env, never code).
COLAB_WORKING = "/content/laya_hpo/working"
COLAB_INPUT_ROOT = "/content/laya_hpo/input"
# The shared-data cache root baked into the receipt. A fixed literal (not
# ``Path.cwd()``) keeps the stored receipt deterministic across hosts.
HPO_SHARED_CACHE_ROOT = "hpo_shared_cache"

# The optuna env var the shared control plane reads. One name, one place.
OPTUNA_URL_ENV = "OPTUNA_STORAGE_URL"
GENERATION_ID_ENV = "EUROMONITOR_HPO_GENERATION_ID"

_TOKEN_PATTERN = re.compile(r"@[A-Z][A-Z0-9_]*@")


# ── search-space SSOT (config/laya_hpo_space.yaml) ─────────────────────────
def space_path(path: str | Path | None = None) -> Path:
    """The search-space config path (SSOT binding; explicit override wins)."""
    if path:
        return Path(path)
    try:
        from core.common import F
        return Path(F["laya_hpo_space"])
    except Exception:  # noqa: BLE001 - fall back to the conventional path
        return TRAIN_ROOT / "config" / SPACE_FILE_NAME


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
    # The routed channels are the SSOT field tuples (not a second registry):
    # a dial whose name is in neither tuple samples a value no trainer reads.
    config_fields = set(laya_lane.FINETUNE_CONFIG_FIELDS)
    control_fields = set(laya_lane.FINETUNE_CONTROL_FIELDS)
    if not space.get("model_key"):
        raise ValueError("laya HPO space must declare a non-empty 'model_key'")
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
        channel = config_fields if target == "config" else control_fields
        if name not in channel:
            raise ValueError(
                f"laya HPO dial {name!r} targets {target!r} but is not in "
                f"FINETUNE_{target.upper()}_FIELDS; it would be a dead/no-op dial")
        if name in config_fields and name in control_fields:
            raise ValueError(
                f"laya HPO dial {name!r} is routed to BOTH channels; split the "
                "SSOT field tuples")
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
        gate = spec.get("when")
        if gate is not None:
            if not isinstance(gate, dict) or "dial" not in gate:
                raise ValueError(
                    f"laya HPO dial {name!r} 'when' must declare a gate dial")
            gate_name = gate["dial"]
            if gate_name not in dials:
                raise ValueError(
                    f"laya HPO dial {name!r} gates on unknown dial "
                    f"{gate_name!r}")
            gate_spec = dials[gate_name]
            if gate_spec.get("type") != "categorical":
                raise ValueError(
                    f"laya HPO dial {name!r} gate {gate_name!r} must be "
                    "categorical")
            if "equals" in gate and gate["equals"] not in gate_spec.get(
                    "choices", []):
                raise ValueError(
                    f"laya HPO dial {name!r} gate value {gate['equals']!r} is "
                    f"not a choice of {gate_name!r}")
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
    profiler = space.get("profiler")
    if profiler is not None:
        if not isinstance(profiler, dict):
            raise TypeError("laya HPO profiler must be a mapping")
        if not isinstance(profiler.get("enabled"), bool):
            raise ValueError("laya HPO profiler.enabled must be a bool")
        for key in ("wait", "warmup", "active", "repeat", "top_ops"):
            value = profiler.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"laya HPO profiler.{key} must be a non-negative int")
        if profiler["active"] < 1:
            raise ValueError("laya HPO profiler.active must be >= 1")
        if profiler["top_ops"] < 1:
            raise ValueError("laya HPO profiler.top_ops must be >= 1")
    # The option components own their own validation; assemble once so a bad
    # parallelism/sampler/pruner/fidelity/warm-start/ensemble name fails at
    # load, not mid-sweep.
    try:
        laya_hpo_options.build_option_set(space)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"laya HPO options invalid: {exc}") from exc


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
    # The model key is space SSOT (validated non-empty by `load_space`); never a
    # second hardcoded registry.
    key = (model_key or resolved_space["model_key"]).strip()
    registry = hpo_registry.default_registry()
    if key not in registry:
        raise ValueError(
            f"laya HPO model_key {key!r} is not registered; "
            f"expected one of {registry.keys()}")
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
                   laya_hpo_runtime, laya_hpo_options, hpo_observability,
                   hpo_registry, hpo_persistence):
        text = inspect.getsource(module)
        cleaned = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("from __future__ import"))
        chunks.append(cleaned.strip("\n"))
    return "\n\n".join(chunks)


def _optuna_env_script(url: str) -> str:
    """The one baked line that injects the URL into the remote process only."""
    return f"os.environ[{OPTUNA_URL_ENV!r}] = {url!r}\n"


def assert_secret_absent(payload: Any, secret: str) -> None:
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


# The runtime preflight is inline (not the finetune lane's constant) so it reads
# the env-driven INPUTS root and thus works on BOTH Kaggle and Colab.
_HPO_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@TRAIN_JSONL@", "@DEV_JSONL@", "@TEST_JSONL@")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus inputs under the lane's INPUTS root."""
    missing = [name for name in _runtime_files
               if not any(INPUTS.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: " + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


def _compose_hpo_script(*, spec, space, generation: str, key: str, tag: str,
                        budget_trials: int, budget_jobs: int, url: str,
                        repository: str, branch: str,
                        revision: str,
                        optuna_env_script: str | None = None) -> str:
    """Render the ONE HPO kernel script (shared by the Kaggle and Colab lanes).

    Output/input roots are runtime env (ER_LAYA_HPO_WORKING/INPUT) so the same
    text runs on Kaggle (defaults) and Colab (the entry driver overrides them).
    """
    preflight = laya_lane._template(_HPO_RUNTIME_PREFLIGHT, {
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
        "OPTUNA_URL_ENV": OPTUNA_URL_ENV,
        "WANDB_API_KEY": laya_lane._env_value("WANDB_API_KEY") or "",
        "WANDB_PROJECT": laya_lane._wandb_project(),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
        "HPO_RUNTIME_SOURCE": hpo_runtime_source(),
        "OPTUNA_ENV_SCRIPT": (optuna_env_script
                              if optuna_env_script is not None
                              else _optuna_env_script(url)),
        "RUNTIME_PREFLIGHT": preflight,
    }
    script = laya_lane._template(_HPO_KERNEL_TEMPLATE, values)
    _kernel_script_gate(script)
    return script


class HpoStagePlan:
    """The lane-independent, fully-resolved staging inputs (one code path).

    Built once by ``LayaHpoStager.plan`` and consumed by the two thin lane
    envelopes; a slot carrying a value here means that value resolved from its
    single SSOT (config, the search-space YAML or the runtime environment).
    """

    __slots__ = ("lane", "spec", "space", "options", "url", "generation",
                 "key", "study_name", "dataset_slug", "base_dataset", "kernel",
                 "repository", "branch", "revision", "tip", "dataset_receipt",
                 "stage_dir", "tag", "budget_trials", "budget_jobs", "script",
                 "working", "input_root")

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))


class LayaHpoStager:
    """Resolve and stage the laya HPO payload for ONE lane (SRP).

    Kaggle and Colab differ ONLY in the delivery envelope (kernel metadata vs
    the Colab entry driver) and in how the Optuna URL line is injected; the
    space/identity/dataset resolution, the published-tip pin, the staged script
    and the receipt are a single shared path. This replaces the two parallel
    staging functions that previously duplicated that path.
    """

    def __init__(self, lane: str, *, revision: str | None = None,
                 run_tag: str | None = None, generation_id: str | None = None,
                 n_trials: int | None = None, n_jobs: int | None = None,
                 space_config: str | Path | None = None,
                 kernel_slug: str | None = None,
                 working: str = COLAB_WORKING,
                 input_root: str = COLAB_INPUT_ROOT):
        if lane not in LANES:
            raise ValueError(f"unknown laya HPO lane {lane!r}; expected {LANES}")
        self.lane = lane
        self.revision = revision
        self.run_tag = run_tag
        self.generation_id = generation_id
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.space_config = space_config
        self.kernel_slug = kernel_slug
        self.working = working
        self.input_root = input_root

    # ── resolution (shared by both lanes) ──────────────────────────────────
    def _dataset_slugs(self, spec) -> tuple[str, str]:
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
        return dataset_slug, base_dataset

    def _optuna_env_script(self, offline: bool) -> str | None:
        """The lane's URL line, or ``""`` when the study runs offline.

        Colab reuses ``colab_runtime`` to resolve/validate the URL; Kaggle lets
        the shared composer bake ``require_optuna_url``'s value. An offline
        study needs no URL at all: the token is replaced with the empty string
        so the staged kernel never carries a secret it will not use.
        """
        if offline:
            return ""
        if self.lane == "colab":
            return _colab_optuna_env_line()
        return None

    def plan(self) -> HpoStagePlan:
        spec = training_cfg().laya
        space = load_space(self.space_config)
        budget_trials = int(self.n_trials if self.n_trials is not None
                            else space["n_trials"])
        budget_jobs = int(self.n_jobs if self.n_jobs is not None
                          else space["n_jobs"])
        # Build the option set early: its ``session.offline`` decides whether a
        # shared PostgreSQL URL is required at all (and fixes the cache root so
        # the receipt is deterministic).
        options = laya_hpo_options.build_option_set(
            space, root=HPO_SHARED_CACHE_ROOT)
        offline = bool(options.session.offline)
        url = "" if offline else require_optuna_url()
        generation, key, study_name = study_identity(
            space=space, generation_id=self.generation_id)
        dataset_slug, base_dataset = self._dataset_slugs(spec)
        kernel = None
        if self.lane == "kaggle":
            kernel = self.kernel_slug or space.get("kernel_slug")
            if not kernel:
                raise RuntimeError(
                    "laya-hpo kernel slug is unset; set kernel_slug in "
                    f"{space_path(self.space_config)} (owner/slug) before staging")
        repository = training_cfg().kaggle.repository
        branch = _current_git_branch()
        revision = self.revision or laya_lane._git_revision()
        from core import runtime_inputs

        tip = runtime_inputs.require_published_tip_match(
            revision, repository, branch)
        dataset_receipt = laya_lane.stage_finetune_dataset_payload(
            dataset_slug=dataset_slug,
            corpus_dir=TRAIN_ROOT / spec.finetune_corpus_dir, kind=HPO_DECISION)
        stage_dir = laya_lane.staging_dir() / self.lane / HPO_DECISION
        stage_dir.mkdir(parents=True, exist_ok=True)
        tag = self.run_tag or (
            spec.run_tag_prefix + "hpo_" + laya_lane.decision_tag())
        script = _compose_hpo_script(
            spec=spec, space=space, generation=generation, key=key, tag=tag,
            budget_trials=budget_trials, budget_jobs=budget_jobs, url=url,
            repository=repository, branch=branch, revision=revision,
            optuna_env_script=self._optuna_env_script(offline))
        return HpoStagePlan(
            lane=self.lane, spec=spec, space=space, options=options, url=url,
            generation=generation, key=key, study_name=study_name,
            dataset_slug=dataset_slug, base_dataset=base_dataset, kernel=kernel,
            repository=repository, branch=branch, revision=revision, tip=tip,
            dataset_receipt=dataset_receipt, stage_dir=stage_dir, tag=tag,
            budget_trials=budget_trials, budget_jobs=budget_jobs, script=script,
            working=self.working, input_root=self.input_root)

    # ── receipt (one shape; each lane adds only its envelope keys) ─────────
    def _budget(self, plan: HpoStagePlan) -> dict[str, Any]:
        # The GPU-capped worker plan, not the raw pool ceiling: ``n_jobs`` is
        # the in-session GPU worker count (one worker per visible T4), so the
        # scheduler caps the plan exactly as it will at runtime.
        gpu_count = max(1, int(plan.budget_jobs))
        workers = plan.options.scheduler.workers(gpu_count)
        return {"n_trials": plan.budget_trials, "n_jobs": plan.budget_jobs,
                "seed": int(plan.space["seed"]),
                "max_workers": len(workers)}

    def _receipt_common(self, plan: HpoStagePlan) -> dict[str, Any]:
        space = plan.space
        return {
            "lane": plan.lane,
            "kind": HPO_DECISION,
            "run_tag": plan.tag,
            "staged": str(plan.stage_dir),
            "code_file": HPO_CODE_FILE,
            "study": {"generation_id": plan.generation, "model_key": plan.key,
                      "study_name": plan.study_name},
            "space": {"path": str(space_path(self.space_config)),
                      "version": space["space_version"],
                      "digest": space_digest(space),
                      "dials": sorted(space["dials"])},
            "budget": self._budget(plan),
            "objective": space["objective"],
            "profiler": space.get("profiler"),
            "options": plan.options.as_dict(),
            "registry": hpo_registry.default_registry().describe(),
            "observability": hpo_observability.TrialObserver(
                hpo_observability.OBSERVABILITY_DIR,
                offline=bool(plan.options.session.offline)).as_dict(),
            "optuna_storage": {"required_env": OPTUNA_URL_ENV,
                               "injected_into_kernel": True,
                               "offline": bool(plan.options.session.offline),
                               "url_persisted_to_manifest": False},
            "published_pin": {"repository": plan.repository,
                              "branch": plan.branch, "revision": plan.revision},
            "published_tip": plan.tip,
        }

    # ── delivery envelopes (the only lane-specific surface) ────────────────
    def _kernel_metadata(self, plan: HpoStagePlan) -> dict[str, Any]:
        slug = plan.kernel
        return {
            "id": slug,
            "title": slug.rsplit("/", 1)[-1].replace("-", " ").title(),
            "code_file": HPO_CODE_FILE,
            "language": "python",
            "kernel_type": "script",
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [plan.dataset_slug, plan.base_dataset],
            "kernel_sources": [],
            "competition_sources": [],
            "is_private": True,
        }

    def _stage_kaggle(self, plan: HpoStagePlan) -> dict[str, Any]:
        atomic_write_json(self._kernel_metadata(plan),
                          plan.stage_dir / "kernel-metadata.json")
        (plan.stage_dir / HPO_CODE_FILE).write_text(plan.script, encoding="utf-8")
        receipt = self._receipt_common(plan)
        receipt["kernel"] = plan.kernel
        receipt["gpu"] = ("T4 (2x when the session exposes it; one worker per "
                          "device)")
        receipt["dataset"] = {"slug": plan.dataset_slug,
                              "payload": plan.dataset_receipt["payload"],
                              "files": plan.dataset_receipt["files"]}
        receipt["base_model"] = {"dataset": plan.base_dataset,
                                 "archive": plan.spec.base_model_archive,
                                 "dir": plan.spec.base_model_dir}
        receipt["laya_package"] = plan.spec.finetune_package
        return receipt

    def _stage_colab(self, plan: HpoStagePlan) -> dict[str, Any]:
        (plan.stage_dir / HPO_CODE_FILE).write_text(plan.script, encoding="utf-8")
        (plan.stage_dir / COLAB_ENTRY_FILE).write_text(
            _compose_colab_entry(HPO_CODE_FILE, plan.working, plan.input_root),
            encoding="utf-8")
        receipt = self._receipt_common(plan)
        receipt["colab_entry"] = COLAB_ENTRY_FILE
        receipt["working"] = plan.working
        receipt["input_root"] = plan.input_root
        return receipt

    def stage(self) -> dict[str, Any]:
        """Resolve, compose, write the payload + receipt (no push, no run)."""
        plan = self.plan()
        receipt = (self._stage_kaggle(plan) if self.lane == "kaggle"
                   else self._stage_colab(plan))
        # Thread the STAGED kernel slug into the shared dispatch hook so a
        # later `kernel_slug("laya-hpo")` / stop addresses the very kernel this
        # receipt staged (an override wins over the space SSOT).
        if plan.lane == "kaggle" and plan.kernel:
            register_dispatch(plan.kernel)
        # The one hard guarantee: the URL is not in the stored receipt.
        assert_secret_absent(receipt, plan.url)
        atomic_write_json(receipt, plan.stage_dir / HPO_RECEIPT_FILE)
        laya_lane._log_lane(
            f"staged {plan.lane} laya-hpo run_tag={plan.tag} "
            f"study={plan.study_name} -> {plan.stage_dir}")
        return receipt


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
    Fail-loud preconditions: ``OPTUNA_STORAGE_URL`` present and PostgreSQL;
    ``EUROMONITOR_HPO_GENERATION_ID`` set; the corpus + base-model dataset
    slugs; the published-tip invariant. The URL is baked ONLY into the staged
    ``laya_hpo.py`` (gitignored); it is asserted absent from the receipt.
    """
    return LayaHpoStager(
        "kaggle", revision=revision, run_tag=run_tag,
        generation_id=generation_id, n_trials=n_trials, n_jobs=n_jobs,
        space_config=space_config, kernel_slug=kernel_slug).stage()


def _colab_optuna_env_line() -> str:
    """The Colab lane reuses the canonical `cli.colab_runtime._optuna_env_script`.

    It reads OPTUNA_STORAGE_URL exactly like the Colab lane and returns the
    `os.environ[...] = ...` line (or '' when absent); absent is a hard error
    here, matching the lane's fail-loud rule.
    """
    from cli import colab_runtime

    line = colab_runtime._optuna_env_script()
    if not line.strip():
        raise RuntimeError(
            f"[laya-hpo] {OPTUNA_URL_ENV} is missing; the Colab lane needs it "
            "in the runtime secret store / .env before staging")
    return line


def _compose_colab_entry(script_name: str, working: str,
                         input_root: str) -> str:
    """The Colab driver: override the env roots and run the shared kernel.

    It carries NO secret (the kernel script injects OPTUNA_STORAGE_URL itself).
    """
    return (
        '"""ER laya HPO on Colab (cli.laya_hpo).\n\n'
        "Sets the env-driven working/input roots, then runs the SAME staged\n"
        "kernel script the Kaggle lane runs. Attach the corpus + base-model\n"
        "archives under the input root first.\n"
        '"""\n'
        "import os\n"
        "import subprocess\n"
        "import sys\n\n"
        f"os.environ.setdefault('ER_LAYA_HPO_WORKING', {working!r})\n"
        f"os.environ.setdefault('ER_LAYA_HPO_INPUT', {input_root!r})\n"
        "os.makedirs(os.environ['ER_LAYA_HPO_WORKING'], exist_ok=True)\n"
        "os.makedirs(os.environ['ER_LAYA_HPO_INPUT'], exist_ok=True)\n"
        "script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "
        f"{script_name!r})\n"
        "raise SystemExit(subprocess.run([sys.executable, script]).returncode)\n"
    )


def stage_laya_hpo_colab(*, revision: str | None = None,
                         run_tag: str | None = None,
                         generation_id: str | None = None,
                         n_trials: int | None = None,
                         n_jobs: int | None = None,
                         space_config: str | Path | None = None,
                         working: str = COLAB_WORKING,
                         input_root: str = COLAB_INPUT_ROOT
                         ) -> dict[str, Any]:
    """Stage the Colab HPO payload (delivery contract; no session is opened).

    The same ``LayaHpoStager`` path as Kaggle: same script + branch pin + URL
    injection, with the Colab entry driver overriding the working/input roots.
    Writes under results/laya_lane/colab/laya-hpo/: laya_hpo.py +
    laya_hpo_colab.py + laya-hpo.receipt.json.
    """
    return LayaHpoStager(
        "colab", revision=revision, run_tag=run_tag,
        generation_id=generation_id, n_trials=n_trials, n_jobs=n_jobs,
        space_config=space_config, working=working,
        input_root=input_root).stage()


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
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
OPTUNA_URL_ENV = "@OPTUNA_URL_ENV@"
WANDB_API_KEY = "@WANDB_API_KEY@"
WANDB_PROJECT = "@WANDB_PROJECT@"

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"

# Output/input roots are overridable so the SAME kernel runs on Kaggle
# (defaults) and Colab (the entry driver sets ER_LAYA_HPO_WORKING/INPUT).
WORKING = Path(os.environ.get("ER_LAYA_HPO_WORKING") or "/kaggle/working")
INPUTS = Path(os.environ.get("ER_LAYA_HPO_INPUT") or "/kaggle/input")
@RUNTIME_PREFLIGHT@

# The shared HPO primitives (training.hpo_control_plane / hpo_fencing /
# hpo_champions) injected VERBATIM. The kernel cannot import the repo.
@OPTUNA_ENV_SCRIPT@
@HPO_RUNTIME_SOURCE@

WANDB_RUN = None
# Per-trial globals the perf patch reads (this worker runs ONE trial at a
# time; each worker is its own process, so the globals never race).
FINETUNE_CONFIG = {}
FINETUNE_CONTROL = {}
FINETUNE_DEV_ROWS = None
FINETUNE_OUTPUT_DIR = None
FINETUNE_CONTROL_RESULT = None
# The config-selected option components (training.laya_hpo_options), assembled
# once from HPO_SPACE. Model-agnostic; every knob is config SSOT.
OPTION_SET = None
# Per-session shared-data cache manifests (auditable inventory).
SHARED_CACHE_MANIFESTS = []
# The worker's control-plane observer (trial_events CDC + study mirror +
# offline hpo_trials ledger); set in run_worker.
OBSERVER = None


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


def wandb_log_profiler(table):
    """Mirror a per-trial key_averages() top-op table to wandb (best-effort)."""
    if WANDB_RUN is None:
        return
    try:
        WANDB_RUN.log({"profiler/top_ops": table})
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
    url = os.environ.get(OPTUNA_URL_ENV, "").strip()
    if not url:
        # RuntimeError (not SystemExit) so the caller can catch it and fall
        # back to the single-process SQLite study instead of hard-failing.
        raise RuntimeError(
            "[laya-hpo] " + OPTUNA_URL_ENV + " is missing; the shared "
            "PostgreSQL Optuna study cannot be reached. Re-stage with the "
            "secret set.")
    # SQLAlchemy's bare postgresql:// defaults to the psycopg2 driver; the
    # session ships psycopg3 (psycopg[binary]), so pin the driver explicitly
    # for BOTH Optuna's RDBStorage and the fencing/champion engines.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
        os.environ[OPTUNA_URL_ENV] = url
    return url


def offline_marker_path():
    """The sentinel a fallback worker writes so the session end knows to read
    the SQLite study and write the offline ledger."""
    return WORKING / OBSERVABILITY_DIR / "offline.marker"


def mark_offline(reason):
    try:
        path = offline_marker_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(reason), encoding="utf-8")
    except Exception:  # noqa: BLE001,S110 - the marker is best-effort
        pass


def offline_active(options):
    """Configured-offline OR a runtime RDB-outage fallback fired this session."""
    try:
        if bool(options.session.offline):
            return True
    except Exception:  # noqa: BLE001,S110
        pass
    return offline_marker_path().is_file()


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
    import torch
    from laya import train as laya_train
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

    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    fidelity = options.fidelity
    resource = fidelity.resource(int(trial.number))
    if fidelity.dimension == "epochs":
        config_dict["epochs"] = int(resource)
    # Coarse-to-fine: freeze the encoder for `head_only_epochs`, then unfreeze
    # (the perf patch already honours `unfreeze_after_epoch`).
    frozen = fidelity.encoder_frozen_epochs()
    if frozen > 0:
        config_dict["freeze_encoder"] = True
        control_dict["unfreeze_after_epoch"] = int(frozen)
    # Shared-data: a fixed seed keeps every trial's mini-batch/option-order
    # draws identical (fair comparisons); warm start picks the init source.
    config_dict["seed"] = int(HPO_SPACE.get("seed", config_dict.get("seed", 0)))
    start_model = options.warm_start.source() or base_model
    trial_train = (subset_train_json(train_path, resource, fidelity.full)
                   if fidelity.dimension == "subset" else train_path)

    log("trial %d start device=%s resource=%s dim=%s freeze=%s warm=%s dials=%s"
        % (int(trial.number), device, resource, fidelity.dimension, frozen,
           options.warm_start.mode, json.dumps(dials, sort_keys=True)))
    rank0 = True
    try:
        rank0 = bool(globals()["is_rank0"]())
    except Exception:
        rank0 = True
    profiler_config = HPO_SPACE.get("profiler") or {}
    profiler = TrialProfiler(
        torch_module=torch, config=profiler_config,
        trace_path=out_dir / "profiler" / ("trial_" + str(int(trial.number))
                                           + ".json"),
        device_type=("cuda" if str(device).startswith("cuda") else "cpu"),
        laya_train=laya_train, namespace=globals(),
        logger=log, wandb_log=wandb_log_profiler, rank0=rank0)
    # Intermediate reporting for AGGRESSIVE pruning: every per-epoch dev
    # evaluation is reported to Optuna (independent of the fidelity lever), so
    # Hyperband/ASHA can prune early and pair with in-trial early stopping.
    reporter = None
    if options.pruner.kind != "none":
        reporter = FidelityReporter(trial, globals().get("optuna"), globals())
        reporter.install()
    started = time.time()
    # Fail-soft: the profiler never fails the trial (its __exit__ returns
    # False), so a finetune error still propagates unchanged.
    try:
        with profiler:
            run_laya_finetune(trial_train, dev_path, start_model, out_dir,
                              device)
    finally:
        if reporter is not None:
            reporter.uninstall()
        _record_cache_manifest(options)
    epoch_time = max(0.0, time.time() - started)
    result = globals().get("FINETUNE_CONTROL_RESULT") or {}
    accuracy = result.get("best_dev_accuracy")
    if accuracy is None:
        raise RuntimeError("trial %d produced no best_dev_accuracy"
                           % int(trial.number))
    dev_loss = dev_loss_from_report(out_dir)
    trial.set_user_attr("epoch_time_s", float(epoch_time))
    trial.set_user_attr("fidelity_resource", int(resource))
    log("trial %d done dev_accuracy=%s dev_loss=%s epoch_time_s=%.1f"
        % (int(trial.number), accuracy, dev_loss, epoch_time))
    return float(accuracy), dev_loss, out_dir, float(epoch_time)


def subset_train_json(train_path, resource, full):
    """Deterministic training-subset JSONL (cached per resource level)."""
    if not full or int(resource) >= int(full):
        return train_path
    fraction = max(1, int(resource)) / float(full)
    out = WORKING / ("subset_" + str(int(resource)) + ".jsonl")
    if out.is_file():
        return out
    from laya import train as laya_train
    rows = laya_train.read_jsonl(str(train_path))
    keep = max(1, int(len(rows) * fraction))
    with out.open("w", encoding="utf-8") as handle:
        for row in rows[:keep]:
            handle.write(json.dumps(row) + "\\n")
    log("fidelity subset %d/%d rows -> %s" % (keep, len(rows), out.name))
    return out


def _record_cache_manifest(options):
    """Record the shared-data cache manifest for the trial (auditable)."""
    try:
        global SHARED_CACHE_MANIFESTS
        manifest = options.shared_data.manifest({
            "tokenized": {"corpus_present": bool(TRAIN_JSONL)},
            "embeddings": {"encoder_frozen": bool(FINETUNE_CONFIG.get(
                "freeze_encoder"))},
            "dev": {"dev_present": bool(DEV_JSONL)},
        })
        SHARED_CACHE_MANIFESTS.append(manifest)
    except Exception as error:
        log("cache manifest skipped: " + str(error)[:160])


def _sync_observations(study):
    """Observe every COMMITTED trial once (idempotent); return the counts.

    Called after ``study.optimize`` so a trial is never observed before Optuna
    has committed its terminal state, and a resumed session/parallel worker
    cannot re-emit history.
    """
    observer = globals().get("OBSERVER")
    if observer is None:
        return {"written": 0, "skipped": 0}
    try:
        result = observer.sync(study.trials)
        observer.flush()
        log("observer %s" % json.dumps(result))
        return result
    except Exception as error:
        log("observer skipped: " + str(error)[:160])
        return {"written": 0, "skipped": 0}


def make_objective(device, train_path, dev_path, base_model, lease_store,
                   champion_store):
    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    def objective(trial):
        return objective_value(
            trial,
            lambda trial: run_trial(
                trial, device, train_path, dev_path, base_model),
            lease_store, champion_store, GENERATION_ID, MODEL_KEY,
            objective_mode=options.objective_mode)
    return objective


def make_ddp_objective(options, lease_store, champion_store):
    """DDP-per-trial objective: each trial is one torchrun subprocess whose
    rank 0 writes the metric; only this controller calls study.tell."""
    script = os.path.abspath(__file__)
    runner = DdpTrialRunner(options.scheduler, script, result_dir=WORKING,
                            logger=log)

    def objective(trial):
        def run_fn(t):
            dials = sample_dials(t, HPO_SPACE)
            return runner.run(int(t.number), {"dials": dials})
        return objective_value(trial, run_fn, lease_store, champion_store,
                               GENERATION_ID, MODEL_KEY,
                               objective_mode=options.objective_mode)
    return objective


def run_ddp_trial(trial_number):
    """One DDP trial on THIS rank (launched by torchrun via DdpTrialRunner)."""
    ensure_optuna_url()
    payload = json.loads(os.environ.get("ER_LAYA_HPO_DDP_PAYLOAD", "{}"))
    import torch
    from laya import train as laya_train
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    device = ("cuda:" + str(local_rank)) if torch.cuda.is_available() else "cpu"
    train_path = resolve_input(TRAIN_JSONL)
    dev_path = resolve_input(DEV_JSONL)
    base_model = extract_base_model(resolve_input(BASE_MODEL_ARCHIVE))
    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    globals()["OPTION_SET"] = options
    dials = payload.get("dials") or {}
    config_dict, control_dict = route_dials(
        BASE_FINETUNE_CONFIG, BASE_FINETUNE_CONTROL, dials, HPO_SPACE)
    globals()["FINETUNE_CONFIG"] = config_dict
    globals()["FINETUNE_CONTROL"] = control_dict
    globals()["FINETUNE_CONTROL_RESULT"] = None
    globals()["FINETUNE_DEV_ROWS"] = None
    start_model = options.warm_start.source() or base_model
    out_dir = WORKING / ("ddp_checkpoint_trial_" + str(int(trial_number)))
    if is_rank0() and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiler = TrialProfiler(
        torch_module=torch, config=HPO_SPACE.get("profiler") or {},
        trace_path=out_dir / "profiler" / ("trial_" + str(int(trial_number))
                                           + ".json"),
        device_type=("cuda" if device.startswith("cuda") else "cpu"),
        laya_train=laya_train, namespace=globals(), logger=log,
        wandb_log=wandb_log_profiler, rank0=True)
    started = time.time()
    init_distributed()
    try:
        with profiler:
            run_laya_finetune(train_path, dev_path, start_model, out_dir, device)
    finally:
        destroy_if_distributed()
    if is_rank0():
        result = globals().get("FINETUNE_CONTROL_RESULT") or {}
        data = {"accuracy": result.get("best_dev_accuracy"),
                "dev_loss": dev_loss_from_report(out_dir),
                "checkpoint": str(out_dir),
                "epoch_time_s": max(0.0, time.time() - started)}
        path = DdpTrialRunner(options.scheduler, os.path.abspath(__file__),
                              result_dir=WORKING).result_path(trial_number)
        path.write_text(json.dumps(data) + "\\n", encoding="utf-8")


def run_worker(device):
    if MODEL_KEY != "laya":
        # The registry carries a real search space + objective for every model
        # key, but THIS lane's remote worker executes only the laya objective.
        # Fail loud rather than silently score the wrong model.
        registry = default_registry()
        descriptor = registry.objective(MODEL_KEY)
        raise SystemExit(
            "[laya-hpo] model_key %r is registered (metric=%s, runner=%s) but "
            "this lane executes only the 'laya' objective; run that model's own "
            "HPO worker for remote execution" % (
                MODEL_KEY, descriptor.metric, descriptor.runner))
    if os.environ.get("ER_LAYA_HPO_SKIP_INSTALL") != "1":
        pip_install_runtime()
        pip_install_laya()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    globals()["optuna"] = optuna
    import torch
    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    globals()["OPTION_SET"] = options
    options.resource_caps.apply_torch(torch)
    # Read the session policy BEFORE touching the URL: an offline study needs
    # no shared PostgreSQL at all.
    offline = bool(options.session.offline)
    observer = TrialObserver(WORKING / OBSERVABILITY_DIR, offline=offline)
    globals()["OBSERVER"] = observer

    def _sqlite_storage():
        WORKING.mkdir(parents=True, exist_ok=True)
        return optuna.storages.RDBStorage(
            "sqlite:///" + str(WORKING / "hpo_offline.db"))

    study_name = generation_study_name(generation_id=GENERATION_ID,
                                       model_key=MODEL_KEY)
    sampler = options.sampler.create(optuna)
    pruner = options.pruner.create(optuna)
    directions = options.objective_mode.directions()
    study_kwargs = ({"directions": directions} if isinstance(directions, list)
                    else {"direction": directions})

    def _build_study(storage):
        return optuna.create_study(
            study_name=study_name, sampler=sampler, pruner=pruner,
            storage=storage, **options.session.study_kwargs(), **study_kwargs)

    storage_config = None
    if offline:
        # Offline: a single-process local SQLite study + the offline ledger.
        # No shared control plane, so no leases/champions.
        study = _build_study(_sqlite_storage())
    else:
        try:
            ensure_optuna_url()
            storage_config = storage_from_environment()
            study = _build_study(create_storage(storage_config))
        except Exception as error:
            # Graceful RDB outage: fall back to the SQLite study + offline
            # ledger instead of losing the whole session. The marker tells the
            # session end to read SQLite and write hpo_trials.jsonl.
            log("shared Postgres unavailable (%s); falling back to the offline "
                "SQLite study + hpo_trials ledger" % str(error)[:160])
            offline = True
            mark_offline(str(error)[:200])
            observer = TrialObserver(WORKING / OBSERVABILITY_DIR, offline=True)
            globals()["OBSERVER"] = observer
            storage_config = None
            study = _build_study(_sqlite_storage())
    if not offline:
        fail_stale_trials(study)
    # Warm start: enqueue known-good seed configs so TPE starts from them.
    for seed_config in options.warm_start.enqueued_trials():
        try:
            study.enqueue_trial(seed_config)
        except Exception as error:
            log("enqueue_trial skipped: " + str(error)[:160])
    lease_store = None if offline else TrialLeaseStore(storage_config.url)
    champion_store = None if offline else ChampionStore(storage_config.url)

    train_path = resolve_input(TRAIN_JSONL)
    dev_path = resolve_input(DEV_JSONL)
    base_model = extract_base_model(resolve_input(BASE_MODEL_ARCHIVE))
    log("worker device=%s train=%s dev=%s base=%s"
        % (device, train_path.name, dev_path.name, base_model.name))

    try:
        wandb_init()
    except Exception as error:
        log("wandb init failed: " + str(error)[:200])

    if os.environ.get("ER_LAYA_HPO_DDP") == "1":
        objective = make_ddp_objective(options, lease_store, champion_store)
    else:
        objective = make_objective(device, train_path, dev_path, base_model,
                                   lease_store, champion_store)
    prior = finished_trial_count(study, ("COMPLETE", "PRUNED", "FAIL"))
    remaining = max(0, int(N_TRIALS) - prior)
    # DDP serialises trials (torchrun fans each one out); slots parallelise.
    if os.environ.get("ER_LAYA_HPO_DDP") == "1":
        divisor = 1
    else:
        divisor = max(1, int(os.environ.get("ER_LAYA_HPO_WORKER_COUNT")
                             or N_JOBS))
    per_worker = per_worker_budget(remaining, divisor)
    log("worker budget prior=%d remaining=%d per_worker=%d study=%s timeout=%s"
        % (prior, remaining, per_worker, study_name,
           options.session.timeout_s or "none"))
    if per_worker:
        study.optimize(objective, n_trials=per_worker,
                       **options.session.optimize_kwargs())
    # Observe the committed study ONCE (idempotent across workers/sessions) and
    # flush every worker's mirror so a worker that exits never loses it.
    _sync_observations(study)
    wandb_finish()


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_session_receipt():
    import optuna
    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    # Configured offline OR a runtime RDB-outage fallback flagged by a worker.
    offline = offline_active(options)
    study_name = generation_study_name(generation_id=GENERATION_ID,
                                       model_key=MODEL_KEY)

    def _sqlite_storage():
        WORKING.mkdir(parents=True, exist_ok=True)
        return optuna.storages.RDBStorage(
            "sqlite:///" + str(WORKING / "hpo_offline.db"))

    if offline:
        storage = _sqlite_storage()
    else:
        try:
            storage = create_storage(storage_from_environment())
        except Exception as error:
            log("shared Postgres unavailable at session end (%s); reading the "
                "offline SQLite study" % str(error)[:160])
            offline = True
            storage = _sqlite_storage()
    study = optuna.load_study(study_name=study_name, storage=storage)
    directions = options.objective_mode.directions()
    complete = [trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None]
    # Multi-objective-safe: rank by the configured per-objective directions.
    best = (min(complete,
                key=lambda trial: rank_key(_trial_value(trial), directions))
            if complete else None)
    trials = [{
        "number": int(trial.number),
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "dev_accuracy": trial.user_attrs.get("dev_accuracy"),
        "dev_loss": trial.user_attrs.get("dev_loss"),
    } for trial in study.trials]
    # Control-plane observability: CDC events + local study mirror + offline
    # ledger, synced ONCE from the authoritative study (idempotent).
    observer = TrialObserver(WORKING / OBSERVABILITY_DIR, offline=offline)
    observer.sync(study.trials)
    observer.flush()
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
    receipt["options"] = options.as_dict()
    # Ensemble: the top-k trials are directly comparable (fixed data).
    ensemble_trials = options.ensembler.select(
        [SimpleNamespace(
            number=t["number"],
            value=(t["value"][0] if isinstance(t["value"], (list, tuple))
                   else t["value"]))
         for t in trials])
    receipt["ensemble"] = {
        "enabled": options.ensembler.enabled,
        "method": options.ensembler.method,
        "top_k": options.ensembler.top_k,
        "selected": [int(t.number) for t in ensemble_trials],
    }
    receipt["shared_data"] = {"manifests": SHARED_CACHE_MANIFESTS[-8:]}
    receipt["observability"] = observer.as_dict()
    receipt["registry"] = default_registry().describe()
    WORKING.mkdir(parents=True, exist_ok=True)
    path = WORKING / "laya-hpo.receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    log("wrote " + str(path) + " best="
        + (str(best.value) if best is not None else "none"))
    return receipt


def main():
    pip_install_runtime()
    pip_install_laya()
    import torch
    options = build_option_set(HPO_SPACE)
    globals()["OPTION_SET"] = options
    options.resource_caps.apply_torch(torch)
    offline = bool(options.session.offline)
    if not offline:
        ensure_optuna_url()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if options.scheduler.mode == "slots":
        options.mps.start()
    specs = options.scheduler.workers(gpu_count or 1)
    study_name = generation_study_name(generation_id=GENERATION_ID,
                                       model_key=MODEL_KEY)
    log("session gpus=%d cores=%d mode=%s workers=%d study=%s budget=%d "
        "threads_per_worker=%d timeout=%s"
        % (gpu_count, os.cpu_count() or 1, options.scheduler.mode, len(specs),
           study_name, int(N_TRIALS), options.resource_caps.threads_per_worker,
           options.session.timeout_s or "none"))
    script = os.path.abspath(__file__)
    processes = []
    for spec in specs:
        env = os.environ.copy()
        env.update(spec.env)
        env["ER_LAYA_HPO_SKIP_INSTALL"] = "1"
        env["ER_LAYA_HPO_WORKER_COUNT"] = str(len(specs))
        if options.scheduler.mode == "slots":
            # Pin the slot to ONE physical GPU (its own worker process).
            env["CUDA_VISIBLE_DEVICES"] = str(spec.device_index)
            env["ER_LAYA_HPO_WORKER_DEVICE"] = "cuda:0"
        else:
            # DDP-per-trial: one controller process; each OBJECTIVE fans the
            # trial out over torchrun and only rank 0 returns the metric.
            env["ER_LAYA_HPO_WORKER_DEVICE"] = "cuda:0"
            env["ER_LAYA_HPO_DDP"] = "1"
        processes.append(subprocess.Popen([sys.executable, script], env=env))
    codes = [process.wait() for process in processes]
    if options.scheduler.mode == "slots":
        options.mps.stop()
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
    observability = receipt.get("observability") or {}
    if observability.get("events_dropped"):
        log("WARNING observability dropped %s event(s); the snapshot omits them"
            % observability.get("events_dropped"))
    # Durable, self-validating snapshot of the session's decision trail. The
    # effective offline state includes any worker RDB-outage fallback, so the
    # SQLite study travels too.
    try:
        session_offline = offline_active(options)
        include = [p for p in (
            WORKING / "laya-hpo.receipt.json",
            WORKING / OBSERVABILITY_DIR / "trial_events.jsonl",
            WORKING / OBSERVABILITY_DIR / "study_mirror.jsonl",
            WORKING / OBSERVABILITY_DIR / "hpo_trials.jsonl",
        ) if p.exists()]
        snapshot = build_snapshot(
            generation=WORKING, sequence=int(time.time()),
            optuna_db=(WORKING / "hpo_offline.db" if session_offline else None),
            include=include)
        # READY is only trustworthy after an independent verify (never mark a
        # best-effort/partial capture as durable).
        verify_snapshot(snapshot)
        log("hpo snapshot verified -> " + str(snapshot))
    except Exception as error:
        log("hpo snapshot skipped: " + str(error)[:200])
    if any(code != 0 for code in codes):
        raise SystemExit("laya HPO worker failure: exit codes " + str(codes))


if __name__ == "__main__":
    _ddp_trial = os.environ.get("ER_LAYA_HPO_DDP_TRIAL")
    _worker_device = os.environ.get("ER_LAYA_HPO_WORKER_DEVICE")
    if _ddp_trial is not None:
        run_ddp_trial(int(_ddp_trial))
    elif _worker_device:
        run_worker(_worker_device)
    else:
        main()
'''


# ── entry/CLI dispatch registry (one entry per lane) ───────────────────────
STAGE_DISPATCH = {
    "kaggle": "stage_laya_hpo_kernel",
    "colab": "stage_laya_hpo_colab",
}


def main(argv: list[str] | None = None) -> int:
    """Stage (and optionally push) the laya HPO payload for one lane.

    Offline by default: the staged receipts are printed. ``--execute`` (Kaggle
    only) additionally runs ``kaggle kernels push`` through the landed laya
    lane's gated push. The owner launches the sweep; this lane never starts one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, default="kaggle",
                        help="which delivery lane to stage (default: kaggle)")
    parser.add_argument("--execute", action="store_true",
                        help="push the staged kernel to Kaggle (default: "
                             "offline dry-run staging only; kaggle lane)")
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
    stage = globals()[STAGE_DISPATCH[args.lane]]
    receipt = stage(
        revision=args.revision, run_tag=args.run_tag,
        generation_id=args.generation_id, n_trials=args.n_trials,
        n_jobs=args.n_jobs, space_config=args.space)
    print(json.dumps(receipt, indent=2, default=str), flush=True)
    if args.execute:
        if args.lane != "kaggle":
            raise SystemExit("--execute is a kaggle-lane operation")
        plan = laya_lane.push_kaggle_kernel(Path(receipt["staged"]),
                                            execute=True)
        print(json.dumps(plan, indent=2, default=str), flush=True)
    return 0


def register_dispatch(slug: str | None = None) -> str | None:
    """Register the HPO kind with the laya lane's `kernel_slug` dispatch.

    The HPO slug lives in the HPO space SSOT, not LayaSpec, so we push it into
    `laya_lane.EXTERNAL_KIND_SLUGS` instead of duplicating it. ``slug`` threads
    a staged override through so `kernel_slug("laya-hpo")` resolves the SAME
    kernel that was staged; when omitted the space SSOT is used (import-time
    registration).
    """
    effective = slug or load_space().get("kernel_slug")
    if effective:
        laya_lane.EXTERNAL_KIND_SLUGS[HPO_DECISION] = effective
    return effective


def kernel_slug(decision_kind: str = HPO_DECISION) -> str:
    """`laya_lane.kernel_slug` including the HPO kind (delegates)."""
    return laya_lane.kernel_slug(decision_kind)


try:  # registering at import makes `laya_lane.kernel_slug("laya-hpo")` work
    register_dispatch()
except Exception:  # noqa: BLE001,S110 - a bad/missing space must not break import
    pass


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLAB_ENTRY_FILE",
    "GENERATION_ID_ENV",
    "HPO_CODE_FILE",
    "HPO_DECISION",
    "HPO_RECEIPT_FILE",
    "LANES",
    "OPTUNA_URL_ENV",
    "STAGE_DISPATCH",
    "apply_dials",
    "assert_secret_absent",
    "finished_trial_count",
    "hpo_runtime_source",
    "kernel_slug",
    "load_space",
    "objective_value",
    "register_dispatch",
    "require_optuna_url",
    "resolve_study_config",
    "route_dials",
    "sample_dials",
    "space_digest",
    "stage_laya_hpo_colab",
    "stage_laya_hpo_kernel",
    "study_identity",
    "validate_space",
]
