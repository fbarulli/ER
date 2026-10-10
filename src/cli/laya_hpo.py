"""Laya fine-tune HPO lane (branch laya-hpo).

A remote Kaggle lane that runs Optuna/TPE HPO for the laya fine-tune against a
SHARED PostgreSQL study. One Kaggle session exposes 2xT4; the staged kernel
spawns one worker PER visible device, and every worker (and every other
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
* Storage is resolved by ``training.hpo_study.StudyOwner`` from the config
  SSOT: a configured remote ``OPTUNA_STORAGE_URL`` wins; otherwise the
  config-declared local SQLite study is shared by the session's parallel
  worker processes. The URL is baked ONLY into the (gitignored) staged kernel
  script so the remote process can reach the DB. It is NEVER written to the
  receipt, the kernel metadata, the search-space YAML or any other stored
  manifest (``assert_secret_absent`` guards the receipt).

The objective maximizes DEV accuracy and records dev loss as a secondary
``trial.set_user_attr``. The held-out/test split is never an HPO signal.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

# Read-only reuse: the shared control plane, the landed laya fine-tune lane,
# and the pure HPO runtime (injected verbatim into the remote kernel).
from cli import laya_lane
from cli.laya_hpo_baseline import BaselineSeedFactory
from cli.laya_staging import LayaStagingFactory
from cli.laya_transport import LayaTransportFactory
from core.common import TRAIN_ROOT, training_cfg
from core.laya_config import FinetuneSpec
from core.manifest import atomic_write_json
from training import (
    hpo_budget,
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
    HpoStorage,
    create_storage,
    generation_study_name,
)
from training.hpo_study import ResolvedStudy, StudyOwner
from training.laya_hpo_runtime import (
    finished_trial_count,
    objective_value,
    route_dials,
    sample_dials,
    trial_full_value,
    trial_primary_value,
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

# The study-storage owner resolves the remote-URL / generation-id env-var
# NAMES and the local SQLite path from the config SSOT (never a literal here).
_STUDY_OWNER = StudyOwner.from_config()
OPTUNA_URL_ENV = _STUDY_OWNER.url_env
GENERATION_ID_ENV = _STUDY_OWNER.generation_env

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
    generation = _STUDY_OWNER.generation_id(override=generation_id)
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


def resolve_study_storage(*, space: dict[str, Any] | None = None,
                          generation_id: str | None = None,
                          model_key: str | None = None) -> ResolvedStudy:
    """The ONE study storage from the config-driven study owner.

    Remote PostgreSQL when a URL is configured; otherwise the config-declared
    local SQLite study (shared by a single VM's parallel workers). Fails LOUD
    (with the full traceback) only when neither is configured.
    """
    generation, key, name = study_identity(
        space=space, generation_id=generation_id, model_key=model_key)
    return _STUDY_OWNER.resolve(generation_id=generation, model_key=key)


def resolve_study_config(*, space: dict[str, Any] | None = None,
                         generation_id: str | None = None,
                         model_key: str | None = None):
    """``(storage, study_name, storage_config)`` for the resolved study.

    ``create_storage`` imports Optuna lazily, so importing this module stays
    GPU/DB-free.
    """
    resolved = resolve_study_storage(space=space, generation_id=generation_id,
                                     model_key=model_key)
    storage_config = HpoStorage(url=resolved.storage_url())
    return create_storage(storage_config), resolved.study_name, storage_config


# ── remote-kernel composition (read-only reuse, injected verbatim) ─────────
#: A host-only repo import inside an injected module. In the staged kernel the
#: modules share ONE namespace and the SSOT inputs ride baked globals, so these
#: fallbacks never run; they are stripped from the injected text (the kernel must
#: NOT contain repo imports).
_HOST_ONLY_IMPORT = re.compile(
    r"\s*(?:from|import)\s+(?:core|cli|training|graph_tracks|model_tracks)"
    r"(?:[.\s]|$)")


def _toplevel_name(node: ast.stmt) -> str | None:
    """The simple name bound by a top-level def/class/assignment, if any."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _strip_duplicate_definitions(text: str, seen: set[str]) -> str:
    """Drop top-level defs/assignments already provided by an earlier module.

    The injected HPO modules share ONE namespace in the staged kernel
    (``_LEASE_TTL_SECONDS``, the ``_shared`` resolver and ``__all__`` are each
    defined by several modules); the first definition is canonical, so the
    later duplicate is dead weight and removed. ``seen`` accumulates across
    modules; only simple ``NAME = ...`` / ``def NAME`` / ``class NAME`` forms
    participate.
    """
    lines = text.splitlines()
    drop: set[int] = set()
    for node in ast.parse(text).body:
        name = _toplevel_name(node)
        if name is None:
            continue
        if name in seen:
            drop.update(range(node.lineno - 1,
                              getattr(node, "end_lineno", node.lineno)))
        else:
            seen.add(name)
    if not drop:
        return text
    return "\n".join(line for index, line in enumerate(lines)
                     if index not in drop)


def hpo_runtime_source() -> str:
    """The shared HPO primitives, injected verbatim into the staged script.

    The remote kernel attaches datasets and installs ``laya`` over pip; it does
    NOT clone the repo, so it cannot ``import training.hpo_*``. Concatenating
    the real module sources keeps ONE implementation (the
    ``FINETUNE_CONTROL_LOGIC_SOURCE`` precedent). ``from __future__`` lines are
    stripped because the script has exactly one, at the top, and every host-only
    repo import is stripped (the kernel resolves those symbols from the shared
    injected namespace or the baked SSOT globals). Top-level names defined by
    more than one module are kept once (the first, canonical definition).
    """
    chunks: list[str] = []
    seen: set[str] = set()
    for module in (hpo_control_plane, hpo_fencing, hpo_budget, hpo_champions,
                   laya_hpo_runtime, laya_hpo_options, hpo_observability,
                   hpo_registry, hpo_persistence):
        text = inspect.getsource(module)
        cleaned = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("from __future__ import")
            and not _HOST_ONLY_IMPORT.match(line))
        chunks.append(_strip_duplicate_definitions(cleaned, seen).strip("\n"))
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


class HpoArtifactPolicy:
    """What an HPO trial retains: configs + results, never weights it cannot use.

    A trial is scored by its dev metrics and is never resumed (``run_trial``
    wipes its output dir), so the resumable per-epoch checkpoint (model +
    optimizer + scheduler), the per-epoch HF ``checkpoint_latest`` snapshot and
    the ``best.pt`` file have no consumer and only fill the size-capped
    ``/kaggle/working``. ``save_each_epoch=False`` disables all three; the final
    model ``laya.finetune`` writes into the trial dir IS the champion result and
    is kept, and ``keep_best`` still restores the best epoch into it.
    """

    #: Control overrides baked into the HPO trial's FINETUNE_CONTROL (SSOT).
    CONTROL_OVERRIDES = {"save_each_epoch": False, "resume": False}

    @classmethod
    def apply(cls, control: dict[str, Any]) -> dict[str, Any]:
        return {**control, **cls.CONTROL_OVERRIDES}


def _compose_hpo_script(*, spec, space, generation: str, key: str, tag: str,
                        budget_trials: int, budget_jobs: int,
                        storage: ResolvedStudy | None,
                        local_study_file: str | None,
                        repository: str, branch: str,
                        revision: str, cuda_visible_devices: str) -> str:
    """Render the ONE HPO kernel script (shared by the Kaggle and Colab lanes).

    Output/input roots are runtime env (ER_LAYA_HPO_WORKING/INPUT) so the same
    text runs on Kaggle (defaults) and Colab (the entry driver overrides them).
    A remote URL (when configured) is baked into the token; otherwise the
    config-declared local SQLite fragment is baked and the kernel shares that
    file across its parallel worker processes — no secret, no hardcoded URL,
    and no host-absolute path (the kernel joins the fragment to its WORKING).
    """
    url = (storage.storage_url()
           if storage is not None and storage.is_remote else "")
    optuna_env_script = _optuna_env_script(url) if url else ""
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
        "BASE_FINETUNE_CONTROL": repr(HpoArtifactPolicy.apply(
            laya_lane.finetune_control(spec))),
        "HPO_SPACE": repr(space),
        "N_TRIALS": str(budget_trials),
        "N_JOBS": str(budget_jobs),
        "SEED": str(int(space["seed"])),
        "GENERATION_ID": generation,
        "MODEL_KEY": key,
        "SSOT_MODEL_KEYS": repr(list(hpo_registry.ssot_model_keys())),
        "SSOT_MODEL_REGISTRY": repr(hpo_registry.default_registry().describe()),
        "OPTUNA_URL_ENV": OPTUNA_URL_ENV,
        "LOCAL_STUDY_FILE": local_study_file or "",
        "WANDB_API_KEY": laya_lane._env_value("WANDB_API_KEY") or "",
        "WANDB_PROJECT": laya_lane._wandb_project(),
        # The DDP collective-timeout knob (laya.ddp_timeout_minutes, SSOT):
        # the HPO kernel injects the same PERF/DDP patch text as finetune.
        "DDP_TIMEOUT_MINUTES": str(spec.ddp_timeout_minutes),
        "REPOSITORY": repository,
        "BRANCH": branch,
        "REVISION": revision,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "DEVICE_PATCH": laya_lane.FINETUNE_DEVICE_PATCH_SOURCE,
        "PERF_PATCH": laya_lane.FINETUNE_PERF_PATCH_SOURCE,
        "HPO_RUNTIME_SOURCE": hpo_runtime_source(),
        "OPTUNA_ENV_SCRIPT": optuna_env_script,
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

    __slots__ = (
        "base_dataset",
        "branch",
        "budget_jobs",
        "budget_trials",
        "dataset_receipt",
        "dataset_slug",
        "generation",
        "input_root",
        "kernel",
        "key",
        "lane",
        "options",
        "repository",
        "revision",
        "script",
        "space",
        "spec",
        "stage_dir",
        "storage",
        "study_name",
        "tag",
        "tip",
        "working",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))


class HpoReceipt:
    """Build the staged receipt for one resolved plan.

    Each method answers one question; ``kaggle``/``colab`` add only their lane
    envelope to the shared ``common`` body.
    """

    def __init__(self, plan: HpoStagePlan, *, space_config=None):
        self.plan = plan
        self.space_config = space_config

    def budget(self) -> dict[str, Any]:
        # The GPU-capped worker plan, not the raw pool ceiling: ``n_jobs`` is
        # the in-session GPU worker count, so the scheduler caps the plan
        # exactly as it will at runtime.
        gpu_count = max(1, int(self.plan.budget_jobs))
        workers = self.plan.options.scheduler.workers(gpu_count)
        return {"n_trials": self.plan.budget_trials,
                "n_jobs": self.plan.budget_jobs,
                "seed": int(self.plan.space["seed"]),
                "max_workers": len(workers)}

    def _offline(self) -> bool:
        return bool(self.plan.options.session.offline)

    def observability(self) -> dict[str, Any]:
        return hpo_observability.TrialObserver(
            hpo_observability.OBSERVABILITY_DIR,
            offline=self._offline()).as_dict()

    def storage(self) -> dict[str, Any]:
        storage = self.plan.storage
        return {"backend": (storage.backend.value if storage is not None
                            else "offline"),
                "required_env": OPTUNA_URL_ENV,
                "local_file": _STUDY_OWNER.local_file,
                "injected_into_kernel": bool(storage and storage.is_remote),
                "offline": self._offline(),
                "url_persisted_to_manifest": False}

    def _space(self) -> dict[str, Any]:
        space = self.plan.space
        return {"path": str(space_path(self.space_config)),
                "version": space["space_version"],
                "digest": space_digest(space),
                "dials": sorted(space["dials"])}

    def common(self) -> dict[str, Any]:
        plan = self.plan
        return {
            "lane": plan.lane,
            "kind": HPO_DECISION,
            "run_tag": plan.tag,
            "staged": str(plan.stage_dir),
            "code_file": HPO_CODE_FILE,
            "study": {"generation_id": plan.generation, "model_key": plan.key,
                      "study_name": plan.study_name},
            "space": self._space(),
            "budget": self.budget(),
            "objective": plan.space["objective"],
            "profiler": plan.space.get("profiler"),
            "options": plan.options.as_dict(),
            "registry": hpo_registry.default_registry().describe(),
            "observability": self.observability(),
            "optuna_storage": self.storage(),
            "published_pin": {"repository": plan.repository,
                              "branch": plan.branch, "revision": plan.revision},
            "published_tip": plan.tip,
        }

    def kaggle(self) -> dict[str, Any]:
        receipt = self.common()
        plan = self.plan
        receipt["kernel"] = plan.kernel
        receipt["gpu"] = ("T4 (2x when the session exposes it; one worker "
                          "process per device, run in parallel)")
        receipt["dataset"] = {"slug": plan.dataset_slug,
                              "payload": plan.dataset_receipt["payload"],
                              "files": plan.dataset_receipt["files"]}
        receipt["base_model"] = {"dataset": plan.base_dataset,
                                 "archive": plan.spec.base_model_archive,
                                 "dir": plan.spec.base_model_dir}
        receipt["laya_package"] = plan.spec.finetune_package
        return receipt

    def colab(self) -> dict[str, Any]:
        receipt = self.common()
        receipt["colab_entry"] = COLAB_ENTRY_FILE
        receipt["working"] = self.plan.working
        receipt["input_root"] = self.plan.input_root
        return receipt


class LayaHpoStager:
    """Resolve and stage the laya HPO payload for ONE lane (SRP).

    Kaggle and Colab differ ONLY in the delivery envelope (kernel metadata vs
    the Colab entry driver) and in how the Optuna URL line is injected; the
    space/identity/dataset resolution, the published-tip pin, the staged script
    and the receipt are a single shared path. Each private ``_resolve_*``
    helper answers exactly one question; ``plan`` only wires them together.
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

    # ── resolution (each helper answers one question) ──────────────────────
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

    def _options(self, space):
        """Build the option set from the search-space config block."""
        return laya_hpo_options.build_option_set(space)

    def _storage(self, offline: bool, generation: str, key: str
                 ) -> ResolvedStudy | None:
        """The resolved storage, or ``None`` for an explicitly offline study."""
        if offline:
            return None
        return _STUDY_OWNER.resolve(generation_id=generation, model_key=key)

    def _kernel(self, space) -> str | None:
        """The Kaggle kernel slug (override wins over the space SSOT)."""
        if self.lane != "kaggle":
            return None
        kernel = self.kernel_slug or space.get("kernel_slug")
        if not kernel:
            raise RuntimeError(
                "laya-hpo kernel slug is unset; set kernel_slug in "
                f"{space_path(self.space_config)} (owner/slug) before staging")
        return kernel

    def _dataset_receipt(self, spec, dataset_slug) -> dict[str, Any]:
        return laya_lane.stage_finetune_dataset_payload(
            dataset_slug=dataset_slug,
            corpus_dir=TRAIN_ROOT / spec.finetune_corpus_dir, kind=HPO_DECISION)

    def _git_pin(self) -> tuple[str, str, str, Any]:
        """``(repository, branch, revision, published_tip)``."""
        repository = training_cfg().kaggle.repository
        branch = _current_git_branch()
        revision = self.revision or laya_lane._git_revision()
        from core import runtime_inputs

        # HPO stages from the branch it runs on; publish it first when origin
        # does not have it (a never-pushed run branch), so the tip gate and the
        # remote clone below can both resolve it.
        runtime_inputs.publish_run_branch(repository, branch)
        tip = runtime_inputs.require_published_tip_match(
            revision, repository, branch)
        return repository, branch, revision, tip

    def _stage_dir(self) -> Path:
        stage_dir = laya_lane.staging_dir() / self.lane / HPO_DECISION
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def _run_tag(self, spec) -> str:
        return self.run_tag or (
            spec.run_tag_prefix + "hpo_" + LayaStagingFactory.decision_tag())

    def _budget(self, space) -> tuple[int, int]:
        trials = int(self.n_trials if self.n_trials is not None
                     else space["n_trials"])
        jobs = int(self.n_jobs if self.n_jobs is not None else space["n_jobs"])
        return trials, jobs

    def plan(self) -> HpoStagePlan:
        spec = training_cfg().laya
        space = load_space(self.space_config)
        # Seed the SSOT baseline recipe as one enqueued trial (when enabled)
        # BEFORE repr(space) is baked, so the remote worker runs the baseline as
        # one process in parallel with the sweep.
        space = BaselineSeedFactory(spec).apply(space)
        options = self._options(space)
        offline = bool(options.session.offline)
        generation, key, study_name = study_identity(
            space=space, generation_id=self.generation_id)
        storage = self._storage(offline, generation, key)
        dataset_slug, base_dataset = self._dataset_slugs(spec)
        dataset_receipt = self._dataset_receipt(spec, dataset_slug)
        kernel = self._kernel(space)
        repository, branch, revision, tip = self._git_pin()
        stage_dir = self._stage_dir()
        tag = self._run_tag(spec)
        budget_trials, budget_jobs = self._budget(space)
        script = _compose_hpo_script(
            spec=spec, space=space, generation=generation, key=key, tag=tag,
            budget_trials=budget_trials, budget_jobs=budget_jobs,
            storage=storage, local_study_file=_STUDY_OWNER.local_file,
            repository=repository, branch=branch, revision=revision,
            cuda_visible_devices=options.session.cuda_visible_devices)
        return HpoStagePlan(
            lane=self.lane, spec=spec, space=space, options=options,
            storage=storage,
            generation=generation, key=key, study_name=study_name,
            dataset_slug=dataset_slug, base_dataset=base_dataset, kernel=kernel,
            repository=repository, branch=branch, revision=revision, tip=tip,
            dataset_receipt=dataset_receipt, stage_dir=stage_dir, tag=tag,
            budget_trials=budget_trials, budget_jobs=budget_jobs, script=script,
            working=self.working, input_root=self.input_root)

    def _receipt(self, plan: HpoStagePlan) -> dict[str, Any]:
        """The lane envelope's receipt (one shape, one builder)."""
        builder = HpoReceipt(plan, space_config=self.space_config)
        return (builder.kaggle() if self.lane == "kaggle"
                else builder.colab())

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
        return self._receipt(plan)

    def _stage_colab(self, plan: HpoStagePlan) -> dict[str, Any]:
        (plan.stage_dir / HPO_CODE_FILE).write_text(plan.script, encoding="utf-8")
        (plan.stage_dir / COLAB_ENTRY_FILE).write_text(
            _compose_colab_entry(HPO_CODE_FILE, plan.working, plan.input_root),
            encoding="utf-8")
        return self._receipt(plan)

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
        secret = (plan.storage.storage_url()
                  if plan.storage is not None and plan.storage.is_remote else "")
        assert_secret_absent(receipt, secret)
        atomic_write_json(receipt, plan.stage_dir / HPO_RECEIPT_FILE)
        laya_lane._log_lane(
            f"staged {plan.lane} laya-hpo run_tag={plan.tag} "
            f"study={plan.study_name} -> {plan.stage_dir}")
        return receipt


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


# ── the embedded Kaggle kernel script (text home: cli.laya_hpo_kernel_text) ──
from cli.laya_hpo_kernel_text import HPO_KERNEL_TEMPLATE as _HPO_KERNEL_TEMPLATE


def main(argv: list[str] | None = None) -> int:
    """Stage (and optionally publish+push+watch) the laya HPO payload.

    Offline by default: the staged receipt is printed. ``--execute`` (Kaggle
    only) runs the owner's lifecycle — publish the freshly staged corpus,
    ``kaggle kernels push``, then spawn the detached watcher. The owner launches
    the sweep; this lane never starts one.
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
    # The ONE owner composes the study/staging/publish/push/watch lifecycle;
    # this entry point is a thin argparse shell over it (no ad-hoc assembly).
    from cli.laya_training_run import LayaRunKind, LayaTrainingRunFactory

    run = LayaTrainingRunFactory.from_config()
    receipt = run.stage(
        LayaRunKind.HPO, lane=args.lane, revision=args.revision,
        run_tag=args.run_tag, generation_id=args.generation_id,
        n_trials=args.n_trials, n_jobs=args.n_jobs, space_config=args.space)
    print(json.dumps(receipt, indent=2, default=str), flush=True)
    if args.execute:
        if args.lane != "kaggle":
            raise SystemExit("--execute is a kaggle-lane operation")
        # publish the freshly staged corpus -> push -> spawn the watcher: the
        # fine-tune execute path, so the run never attaches a stale dataset.
        plan = run.launch(LayaRunKind.HPO, stage_dir=Path(receipt["staged"]),
                          run_tag=receipt["run_tag"], execute=True)
        print(json.dumps(plan, indent=2, default=str), flush=True)
    return 0


class KernelSlugRegistry:
    """The HPO kind's external-kind descriptor registration.

    The HPO slug lives in the HPO space SSOT, not ``LayaSpec``, so the WHOLE
    descriptor — kernel slug, receipt filename and the attached-dataset attr —
    is registered with the transport factory's external-kind maps. One job: map
    the ``laya-hpo`` kind to the slug actually staged (an explicit override wins
    over the SSOT) plus the constants its kernel/receipt contract pins, so the
    harvest resolves ``laya-hpo.receipt.json`` and the publish resolves the
    fine-tune corpus — never a guess, never the decision-payload fallback.
    """

    #: The LayaSpec field naming the corpus the HPO kernel attaches.
    DATASET_ATTR = "finetune_dataset_slug"

    def __init__(self, kind: str = HPO_DECISION):
        self.kind = kind

    def register(self, slug: str | None = None) -> str | None:
        effective = slug or load_space().get("kernel_slug")
        LayaTransportFactory.register_external_kind(
            self.kind, effective, receipt_name=HPO_RECEIPT_FILE,
            dataset_attr=self.DATASET_ATTR)
        return effective

    def resolve(self) -> str:
        return laya_lane.kernel_slug(self.kind)


def register_dispatch(slug: str | None = None) -> str | None:
    """Register the HPO kind with the laya lane's `kernel_slug` dispatch."""
    return KernelSlugRegistry().register(slug)


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
    "HpoReceipt",
    "HpoStagePlan",
    "KernelSlugRegistry",
    "LayaHpoStager",
    "apply_dials",
    "assert_secret_absent",
    "finished_trial_count",
    "hpo_runtime_source",
    "kernel_slug",
    "load_space",
    "objective_value",
    "register_dispatch",
    "resolve_study_storage",
    "resolve_study_config",
    "route_dials",
    "sample_dials",
    "space_digest",
    "study_identity",
    "trial_full_value",
    "trial_primary_value",
]
