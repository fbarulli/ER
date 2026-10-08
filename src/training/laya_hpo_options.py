"""Config-driven HPO option components for the laya HPO lane (model-agnostic).

Every knob below is selected by name from the ``options:`` block of the
search-space SSOT (``config/laya_hpo_space.yaml``); no option value is a code
literal. The components are dependency-injected (optuna, torch, subprocess and
the filesystem are passed in), so they are unit-tested on the host with stubs —
no GPU, no PostgreSQL, no Optuna install.

Design: one class per responsibility (SRP), a name->builder registry per option
family (``SamplerFactory.KINDS`` etc.), and ``build_option_set`` as the single
assembler. Nothing here imports torch/optuna at module import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

# ── option vocabularies (the registries' keys) ─────────────────────────────
PARALLELISM_MODES = ("slots", "ddp")
SAMPLER_KINDS = ("tpe", "cmaes", "random", "gp", "qmc")
PRUNER_KINDS = ("none", "median", "successive_halving", "hyperband")
FIDELITY_DIMENSIONS = ("epochs", "subset")
WARM_START_MODES = ("scratch", "base", "champion")
ENSEMBLE_METHODS = ("weights", "predictions")

_DEFAULT_SECONDARY_OBJECTIVE = "epoch_time_s"


# ── A) parallelism / resource caps ─────────────────────────────────────────
class ResourceCaps:
    """Per-worker CPU/CUDA fairness caps (no slot starves another).

    ``threads_per_worker`` is the SSOT for thread/core matching: every worker
    process pins the BLAS/OMP thread pools to it (default 1) so N worker
    processes on an N-core box do not oversubscribe.
    """

    def __init__(self, config):
        config = dict(config or {})
        self.threads_per_worker = int(config.get("threads_per_worker", 1))
        self.omp_threads = int(config.get("omp_threads", self.threads_per_worker))
        self.torch_threads = int(config.get("torch_threads",
                                            self.threads_per_worker))
        self.dataloader_workers = int(config.get("dataloader_workers", 2))
        self.cuda_alloc_fraction = float(config.get("cuda_alloc_fraction", 0.0))

    def env(self):
        """The environment a worker process must inherit before torch import."""
        threads = str(self.omp_threads)
        return {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "ER_LAYA_DATALOADER_WORKERS": str(self.dataloader_workers),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        }

    def apply_torch(self, torch_module):
        """Apply the in-process caps (best-effort; never fatal)."""
        try:
            torch_module.set_num_threads(self.torch_threads)
        except Exception:  # noqa: BLE001,S110
            pass
        if self.cuda_alloc_fraction > 0.0:
            try:
                if torch_module.cuda.is_available():
                    torch_module.cuda.set_per_process_memory_fraction(
                        self.cuda_alloc_fraction)
            except Exception:  # noqa: BLE001,S110
                pass

    def as_dict(self):
        return {
            "threads_per_worker": self.threads_per_worker,
            "omp_threads": self.omp_threads,
            "torch_threads": self.torch_threads,
            "dataloader_workers": self.dataloader_workers,
            "cuda_alloc_fraction": self.cuda_alloc_fraction,
        }


class CoreAllocator:
    """Match worker count to cores (documentation + planning, no oversubscribe).

    Two regimes, per the Optuna guidance:
      * many small trials: n_workers ~= cores, 1 thread each (default);
      * few big trials:   1 worker, N threads.
    """

    def __init__(self, threads_per_worker=1):
        self.threads_per_worker = max(1, int(threads_per_worker))

    def regime(self, cores, big_trials=False):
        cores = max(1, int(cores))
        if big_trials:
            return {"regime": "big_trials", "n_workers": 1,
                    "threads_per_worker": cores}
        return {"regime": "many_small_trials",
                "n_workers": cores,
                "threads_per_worker": self.threads_per_worker}

    def as_dict(self):
        return {"threads_per_worker": self.threads_per_worker}


class SessionPolicy:
    """Kaggle-session limits: processes-only, Postgres resume, wall timeout.

    Optuna runs one trial per PROCESS (the slots pool); ``n_jobs`` is pinned to
    1 so ``study.optimize`` never spawns its own threads. The study persists to
    the shared Postgres RDB with ``load_if_exists=True`` so a new session
    resumes; ``timeout_s`` stops cleanly before the Kaggle cutoff.
    """

    def __init__(self, config):
        config = dict(config or {})
        self.timeout_s = int(config.get("timeout_s", 0))
        self.load_if_exists = bool(config.get("load_if_exists", True))
        self.processes_only = bool(config.get("processes_only", True))
        self.n_jobs_threads = 1  # never >1: trials run in their own processes

    def study_kwargs(self):
        return {"load_if_exists": self.load_if_exists}

    def optimize_kwargs(self):
        kwargs = {"n_jobs": self.n_jobs_threads, "catch": (Exception,)}
        if self.timeout_s > 0:
            kwargs["timeout"] = self.timeout_s
        return kwargs

    def as_dict(self):
        return {"timeout_s": self.timeout_s,
                "load_if_exists": self.load_if_exists,
                "processes_only": self.processes_only,
                "n_jobs_threads": self.n_jobs_threads}


class MpsController:
    """CUDA Multi-Process Service lifecycle for the slots scheduler.

    MPS lets several worker processes share one GPU's SM resources instead of
    time-slicing context switches. Start it ONCE per session before spawning
    workers (``nvidia-cuda-mps-control -d``); plain time-slicing works when off.
    """

    def __init__(self, enabled, *, control="nvidia-cuda-mps-control",
                 pipe_dir="/tmp/nvidia-mps", log_dir="/tmp/nvidia-mps-log",
                 runner=None, logger=None):
        self.enabled = bool(enabled)
        self.control = control
        self.pipe_dir = pipe_dir
        self.log_dir = log_dir
        self._runner = runner or subprocess.run
        self._logger = logger

    def env(self):
        if not self.enabled:
            return {}
        return {
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": self.log_dir,
        }

    def commands(self):
        return {
            "start": [self.control, "-d"],
            "stop": [self.control, "-S"],
            "help": [self.control, "-h"],
        }

    def _emit(self, line):
        if self._logger is not None:
            try:
                self._logger(line)
            except Exception:  # noqa: BLE001,S110
                pass

    def start(self):
        """Start the daemon; best-effort (MPS may be unavailable)."""
        if not self.enabled:
            return False
        try:
            Path(self.pipe_dir).mkdir(parents=True, exist_ok=True)
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._runner(self.commands()["start"], check=False)
            self._emit("MPS daemon start requested (" + self.control + " -d)")
            return True
        except Exception as error:  # noqa: BLE001 - fail soft
            self._emit("MPS start skipped: " + str(error)[:160])
            return False

    def stop(self):
        if not self.enabled:
            return False
        try:
            self._runner(self.commands()["stop"], check=False)
            return True
        except Exception:  # noqa: BLE001
            return False

    def as_dict(self):
        return {"enabled": self.enabled, "control": self.control,
                "pipe_dir": self.pipe_dir, "log_dir": self.log_dir}


class WorkerSpec:
    """One worker process: which GPU it owns and its slot on that GPU."""

    __slots__ = ("device_index", "env", "local_rank", "slot_index")

    def __init__(self, device_index, slot_index, local_rank, env):
        self.device_index = int(device_index)
        self.slot_index = int(slot_index)
        self.local_rank = int(local_rank)
        self.env = dict(env)

    def with_caps(self, caps):
        merged = dict(self.env)
        merged.update(caps.env())
        return WorkerSpec(self.device_index, self.slot_index, self.local_rank,
                          merged)

    def as_dict(self):
        return {"device_index": self.device_index, "slot_index": self.slot_index,
                "local_rank": self.local_rank, "env": self.env}


class WorkerPool:
    """Plan the slots-mode worker set: slots_per_gpu processes per GPU."""

    def __init__(self, slots_per_gpu, max_concurrent_trials, resource_caps=None,
                 mps=None):
        self.slots_per_gpu = max(1, int(slots_per_gpu))
        self.max_concurrent_trials = max(1, int(max_concurrent_trials))
        self.resource_caps = resource_caps
        self.mps = mps

    def plan(self, gpu_count):
        gpu_count = max(1, int(gpu_count))
        total = min(gpu_count * self.slots_per_gpu, self.max_concurrent_trials)
        specs = []
        base_env = dict(self.mps.env()) if self.mps is not None else {}
        for index in range(total):
            spec = WorkerSpec(device_index=index % gpu_count,
                              slot_index=index // gpu_count,
                              local_rank=index % gpu_count, env=base_env)
            if self.resource_caps is not None:
                spec = spec.with_caps(self.resource_caps)
            specs.append(spec)
        return specs


class TrialScheduler:
    """Base scheduler; ``create`` is the name->class registry."""

    mode = None

    def __init__(self, config):
        self.config = dict(config or {})

    def workers(self, gpu_count):  # pragma: no cover - interface
        raise NotImplementedError

    def as_dict(self):
        return {"mode": self.mode}

    @classmethod
    def create(cls, config, *, resource_caps=None, mps=None):
        mode = str((config or {}).get("parallelism", "slots"))
        if mode == "slots":
            return SlotsTrialScheduler(config, resource_caps=resource_caps,
                                       mps=mps)
        if mode == "ddp":
            return DdpTrialScheduler(config, resource_caps=resource_caps)
        raise ValueError(
            f"unknown parallelism {mode!r}; expected {PARALLELISM_MODES}")


class SlotsTrialScheduler(TrialScheduler):
    """WORKER-POOL: N processes per GPU, each running its own optimize loop."""

    mode = "slots"

    def __init__(self, config, *, resource_caps=None, mps=None):
        super().__init__(config)
        self.pool = WorkerPool(
            slots_per_gpu=int(self.config.get("slots_per_gpu", 1)),
            max_concurrent_trials=int(self.config.get("max_concurrent_trials", 1)),
            resource_caps=resource_caps, mps=mps)

    def workers(self, gpu_count):
        return self.pool.plan(gpu_count)

    def as_dict(self):
        return {"mode": self.mode, "slots_per_gpu": self.pool.slots_per_gpu,
                "max_concurrent_trials": self.pool.max_concurrent_trials}


class DdpTrialScheduler(TrialScheduler):
    """DDP-PER-TRIAL: each objective runs one trial under torchrun; rank 0 owns
    the metric (never every rank calling ``study.tell``)."""

    mode = "ddp"

    def __init__(self, config, *, resource_caps=None):
        super().__init__(config)
        ddp = dict(self.config.get("ddp") or {})
        self.nproc_per_node = max(1, int(ddp.get("nproc_per_node", 1)))
        self.backend = str(ddp.get("backend", "nccl"))
        self.resource_caps = resource_caps

    def workers(self, gpu_count):
        # A DDP trial spans all GPUs, so there is exactly ONE controller
        # process; it serialises trials (torchrun fans each trial out).
        env = {}
        if self.resource_caps is not None:
            env.update(self.resource_caps.env())
        return [WorkerSpec(device_index=0, slot_index=0, local_rank=0, env=env)]

    def torchrun_argv(self, script, extra=None):
        argv = ["torchrun", "--nproc_per_node", str(self.nproc_per_node),
                "--nnodes", "1", "--node_rank", "0", "--master_addr",
                "127.0.0.1", "--master_port", str(self.config.get(
                    "master_port", 29500)), script]
        if extra:
            argv.extend(extra)
        return argv

    def as_dict(self):
        return {"mode": self.mode, "nproc_per_node": self.nproc_per_node,
                "backend": self.backend}


class DdpTrialRunner:
    """Run ONE DDP trial as a torchrun subprocess; only rank 0 reports.

    The controller samples the trial's dials, hands them to every rank through
    the environment, launches ``torchrun``, and reads the metric rank 0 wrote.
    No rank ever calls ``study.tell`` except the controller's objective.
    """

    DDP_TRIAL_ENV = "ER_LAYA_HPO_DDP_TRIAL"
    DDP_PAYLOAD_ENV = "ER_LAYA_HPO_DDP_PAYLOAD"

    def __init__(self, scheduler, script, *, env=None, runner=None,
                 result_dir=None, logger=None):
        self.scheduler = scheduler
        self.script = str(script)
        self.base_env = dict(env or {})
        self._runner = runner or subprocess.run
        self.result_dir = Path(result_dir) if result_dir else Path.cwd()
        self._logger = logger or (lambda line: None)

    def result_path(self, trial_number):
        return self.result_dir / ("ddp_trial_" + str(int(trial_number)) + ".json")

    def argv(self, trial_number):
        return self.scheduler.torchrun_argv(
            self.script, ["--ddp-trial", str(int(trial_number))])

    def run(self, trial_number, payload):
        env = os.environ.copy()
        env.update(self.base_env)
        env[self.DDP_TRIAL_ENV] = str(int(trial_number))
        env[self.DDP_PAYLOAD_ENV] = json.dumps(payload, sort_keys=True)
        path = self.result_path(trial_number)
        if path.is_file():
            path.unlink()
        self._runner(self.argv(trial_number), env=env, check=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        return (float(data["accuracy"]), data.get("dev_loss"),
                data.get("checkpoint"), float(data.get("epoch_time_s", 0.0)))


# ── B) sampler / pruner / fidelity / objective ─────────────────────────────
class SamplerFactory:
    """Build the Optuna sampler named by config (registry, no literals).

    TPE defaults to ``multivariate=True, constant_liar=True``: the multivariate
    kernel models parameter interactions, and constant_liar makes concurrent
    workers stop proposing points near one another's in-flight trials.
    """

    KINDS = SAMPLER_KINDS

    def __init__(self, kind, seed, *, multivariate=True, constant_liar=True):
        self.kind = str(kind)
        self.seed = int(seed)
        self.multivariate = bool(multivariate)
        self.constant_liar = bool(constant_liar)
        if self.kind not in self.KINDS:
            raise ValueError(
                f"unknown sampler {self.kind!r}; expected {self.KINDS}")

    def create(self, optuna_module):
        samplers = optuna_module.samplers
        if self.kind == "tpe":
            return samplers.TPESampler(
                seed=self.seed, multivariate=self.multivariate,
                constant_liar=self.constant_liar)
        if self.kind == "cmaes":
            return samplers.CmaEsSampler(seed=self.seed)
        if self.kind == "random":
            return samplers.RandomSampler(seed=self.seed)
        if self.kind == "gp":
            return samplers.GPSampler(seed=self.seed)
        if self.kind == "qmc":
            return samplers.QMCSampler(seed=self.seed)
        raise ValueError(f"unknown sampler {self.kind!r}")  # pragma: no cover

    def as_dict(self):
        return {"kind": self.kind, "seed": self.seed,
                "multivariate": self.multivariate,
                "constant_liar": self.constant_liar}


class PrunerFactory:
    """Build the Optuna pruner named by config (registry, no literals)."""

    KINDS = PRUNER_KINDS

    def __init__(self, config):
        config = dict(config or {})
        self.kind = str(config.get("kind", "none"))
        self.n_startup_trials = int(config.get("n_startup_trials", 5))
        self.n_warmup_steps = int(config.get("n_warmup_steps", 1))
        self.min_resource = int(config.get("min_resource", 1))
        self.reduction_factor = int(config.get("reduction_factor", 3))
        # Hyperband's largest resource (epochs/subset): "auto" lets Optuna pick,
        # otherwise pin the full-fidelity resource so brackets are correct.
        self.max_resource = config.get("max_resource", "auto")
        if self.kind not in self.KINDS:
            raise ValueError(
                f"unknown pruner {self.kind!r}; expected {self.KINDS}")

    def create(self, optuna_module):
        pruners = optuna_module.pruners
        if self.kind == "none":
            return pruners.NopPruner()
        if self.kind == "median":
            return pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_warmup_steps)
        if self.kind == "successive_halving":
            return pruners.SuccessiveHalvingPruner(
                min_resource=self.min_resource,
                reduction_factor=self.reduction_factor,
                min_early_stopping_rate=0)
        if self.kind == "hyperband":
            return pruners.HyperbandPruner(
                min_resource=self.min_resource,
                max_resource=self.max_resource,
                reduction_factor=self.reduction_factor)
        raise ValueError(f"unknown pruner {self.kind!r}")  # pragma: no cover

    def as_dict(self):
        return {"kind": self.kind, "n_startup_trials": self.n_startup_trials,
                "n_warmup_steps": self.n_warmup_steps,
                "min_resource": self.min_resource,
                "max_resource": self.max_resource,
                "reduction_factor": self.reduction_factor}


class FidelitySchedule:
    """Multi-fidelity resource assignment (the ASHA/Hyperband lever).

    Trials are interleaved across ``stages``; each stage's resource rises
    geometrically from ``small`` to ``full`` (epochs and/or a training subset
    size). The pruner compares same-resource trials, so the schedule must be a
    pure function of the trial number. ``staged`` coarse-to-fine additionally
    freezes the encoder for the first ``head_only_epochs`` epochs.
    """

    def __init__(self, config, staged=None):
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.dimension = str(config.get("dimension", "epochs"))
        self.small = max(1, int(config.get("small", 1)))
        self.full = max(self.small, int(config.get("full", 1)))
        self.stages = max(1, int(config.get("stages", 1)))
        if self.dimension not in FIDELITY_DIMENSIONS:
            raise ValueError(
                f"unknown fidelity dimension {self.dimension!r}; "
                f"expected {FIDELITY_DIMENSIONS}")
        staged = dict(staged or {})
        self.staged_enabled = bool(staged.get("enabled", False))
        self.head_only_epochs = int(staged.get("head_only_epochs", 0))

    def stage(self, trial_number):
        return int(trial_number) % self.stages

    def resource(self, trial_number):
        """The resource for a trial (epochs or subset size)."""
        if not self.enabled or self.stages == 1:
            return self.full
        stage = self.stage(trial_number)
        if self.stages == 1:
            return self.full
        # geometric small -> full across the stage index.
        fraction = stage / float(self.stages - 1)
        span = self.full - self.small
        return round(self.small + fraction * span)

    def is_full(self, trial_number):
        return self.resource(trial_number) >= self.full

    def stages_list(self):
        return [self.resource(stage) for stage in range(self.stages)]

    def encoder_frozen_epochs(self):
        """Coarse-to-fine: frozen encoder for the head-only warm-up epochs."""
        return self.head_only_epochs if self.staged_enabled else 0

    def as_dict(self):
        return {"enabled": self.enabled, "dimension": self.dimension,
                "small": self.small, "full": self.full, "stages": self.stages,
                "staged": self.staged_enabled,
                "head_only_epochs": self.head_only_epochs}


class ObjectiveMode:
    """Single- vs multi-objective value shaping (directions from config)."""

    def __init__(self, multi_objective, secondary=_DEFAULT_SECONDARY_OBJECTIVE):
        self.multi = bool(multi_objective)
        self.secondary = str(secondary)

    def directions(self):
        if self.multi:
            return ["maximize", "minimize"]  # dev accuracy up, time down
        return "maximize"

    def value(self, metrics):
        """Shape the objective return: a float, or (primary, secondary)."""
        primary = float(metrics["dev_accuracy"])
        if not self.multi:
            return primary
        return (primary, float(metrics.get(self.secondary, 0.0)))

    def as_dict(self):
        return {"multi_objective": self.multi, "secondary": self.secondary}


# ── C) shared-data strategies ──────────────────────────────────────────────
class SharedDataCache:
    """The corpus/dev are identical across trials: cache them ONCE.

    ``tokenized`` is a memory-mapped token cache all slots read (no per-trial
    re-tokenization); ``embeddings`` is the frozen-encoder embedding cache so a
    frozen trial trains only the head; ``dev`` caches dev encodings for fast
    per-epoch evaluation. Paths and the manifest are pure/config-driven.
    """

    KINDS = ("tokenized", "embeddings", "dev")

    def __init__(self, config, root):
        config = dict(config or {})
        self.root = Path(root)
        self.tokenized = bool(config.get("tokenized_cache", True))
        self.embeddings = bool(config.get("embedding_cache", True))
        self.dev = bool(config.get("dev_encoding_cache", True))

    def enabled(self, kind):
        if kind == "tokenized":
            return self.tokenized
        if kind == "embeddings":
            return self.embeddings
        if kind == "dev":
            return self.dev
        raise ValueError(f"unknown cache kind {kind!r}; expected {self.KINDS}")

    def key(self, kind, signature):
        payload = json.dumps({"kind": kind, "signature": signature},
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def path(self, kind, signature):
        return self.root / kind / (self.key(kind, signature) + ".mmap")

    def is_ready(self, kind, signature):
        return self.enabled(kind) and self.path(kind, signature).is_file()

    def manifest(self, entries):
        """An auditable inventory: {kind: {signature, key, path, ready}}."""
        out = {}
        for kind, signature in sorted((entries or {}).items()):
            out[kind] = {
                "signature": signature,
                "key": self.key(kind, signature),
                "path": str(self.path(kind, signature)),
                "ready": self.is_ready(kind, signature),
            }
        return out

    def as_dict(self):
        return {"root": str(self.root), "tokenized": self.tokenized,
                "embeddings": self.embeddings, "dev": self.dev}


class WarmStartPolicy:
    """Where a trial starts from: scratch, the shared base, or the champion.

    Also carries the ``enqueue`` seed configs: known-good parameter sets the
    kernel feeds to ``study.enqueue_trial`` so TPE starts from them rather than
    from cold random points.
    """

    def __init__(self, mode, *, base_model=None, champion_artifact=None,
                 enqueue=None):
        self.mode = str(mode)
        self.base_model = base_model
        self.champion_artifact = champion_artifact
        self.enqueue = [dict(entry) for entry in (enqueue or [])]
        if self.mode not in WARM_START_MODES:
            raise ValueError(
                f"unknown warm_start mode {self.mode!r}; "
                f"expected {WARM_START_MODES}")

    def source(self):
        """The checkpoint dir/None a trial initializes from."""
        if self.mode == "scratch":
            return None
        if self.mode == "base":
            return self.base_model
        if self.mode == "champion":
            return self.champion_artifact or self.base_model
        raise ValueError(f"unknown warm_start mode {self.mode!r}")

    def enqueued_trials(self):
        """The known-good seed configs for ``study.enqueue_trial``."""
        return [dict(entry) for entry in self.enqueue]

    def as_dict(self):
        return {"mode": self.mode, "base_model": self.base_model,
                "champion_artifact": self.champion_artifact,
                "enqueue": len(self.enqueue)}


class TrialEnsembler:
    """Average the top-k trials (weights or predictions).

    The corpus is fixed, so top-k trials are directly comparable: averaging
    their weights (SWA-across-trials) or predictions is a real ensemble, not a
    metric artifact.
    """

    def __init__(self, enabled, *, top_k=3, method="weights"):
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k))
        self.method = str(method)
        if self.method not in ENSEMBLE_METHODS:
            raise ValueError(
                f"unknown ensemble method {self.method!r}; "
                f"expected {ENSEMBLE_METHODS}")

    def rank(self, trials):
        """Sort completed trials by value descending, then trial number."""
        completed = [t for t in trials
                     if getattr(t, "value", None) is not None]
        return sorted(completed, key=lambda t: (-float(t.value), int(t.number)))

    def select(self, trials):
        return self.rank(trials)[:self.top_k] if self.enabled else []

    @staticmethod
    def average_weights(states):
        """Elementwise mean of same-key weight values (numbers or nested lists).

        The kernel converts torch tensors to/from nested lists (or its own
        tensor mean) around this pure helper.
        """
        if not states:
            return {}
        keys = set(states[0])
        for state in states[1:]:
            keys &= set(state)
        return {key: _mean_value([state[key] for state in states])
                for key in sorted(keys)}

    @staticmethod
    def average_predictions(arrays):
        """Mean of same-length prediction arrays."""
        return _mean_value([list(array) for array in arrays]) if arrays else []

    def as_dict(self):
        return {"enabled": self.enabled, "top_k": self.top_k,
                "method": self.method}


def _mean_value(values):
    if isinstance(values[0], (list, tuple)):
        return [_mean_value([value[index] for value in values])
                for index in range(len(values[0]))]
    return sum(float(value) for value in values) / float(len(values))


class OptionSet:
    """The assembled, config-selected option components."""

    def __init__(self, *, scheduler, mps, resource_caps, core_allocator,
                 session, sampler, pruner, fidelity, objective_mode,
                 shared_data, warm_start, ensembler):
        self.scheduler = scheduler
        self.mps = mps
        self.resource_caps = resource_caps
        self.core_allocator = core_allocator
        self.session = session
        self.sampler = sampler
        self.pruner = pruner
        self.fidelity = fidelity
        self.objective_mode = objective_mode
        self.shared_data = shared_data
        self.warm_start = warm_start
        self.ensembler = ensembler

    def as_dict(self):
        return {
            "scheduler": self.scheduler.as_dict(),
            "mps": self.mps.as_dict(),
            "resource_caps": self.resource_caps.as_dict(),
            "core_allocator": self.core_allocator.as_dict(),
            "session": self.session.as_dict(),
            "sampler": self.sampler.as_dict(),
            "pruner": self.pruner.as_dict(),
            "fidelity": self.fidelity.as_dict(),
            "objective_mode": self.objective_mode.as_dict(),
            "shared_data": self.shared_data.as_dict(),
            "warm_start": self.warm_start.as_dict(),
            "ensemble": self.ensembler.as_dict(),
        }


def build_option_set(space, *, root=None, base_model=None,
                     champion_artifact=None):
    """Assemble every option component from the ``options:`` config block."""
    options = dict((space or {}).get("options") or {})
    resources = ResourceCaps(options.get("resources"))
    core_allocator = CoreAllocator(resources.threads_per_worker)
    session = SessionPolicy(options.get("session"))
    mps = MpsController(options.get("mps", False))
    scheduler = TrialScheduler.create(options, resource_caps=resources, mps=mps)
    sampler_cfg = options.get("sampler") or {}
    sampler = SamplerFactory(sampler_cfg.get("kind", "tpe"),
                            (space or {}).get("seed", 0),
                            multivariate=sampler_cfg.get("multivariate", True),
                            constant_liar=sampler_cfg.get("constant_liar", True))
    pruner = PrunerFactory(options.get("pruner"))
    fidelity = FidelitySchedule(options.get("fidelity"),
                                options.get("staged"))
    objective_mode = ObjectiveMode(
        options.get("multi_objective", False),
        options.get("multi_objective_secondary", _DEFAULT_SECONDARY_OBJECTIVE))
    shared_data = SharedDataCache(options.get("shared_data"),
                                  root or (Path.cwd() / "hpo_shared_cache"))
    warm_start_cfg = options.get("warm_start") or {}
    warm_start = WarmStartPolicy(
        warm_start_cfg.get("mode", "base"),
        base_model=base_model, champion_artifact=champion_artifact,
        enqueue=warm_start_cfg.get("enqueue"))
    ensembler = TrialEnsembler(
        (options.get("ensemble") or {}).get("enabled", False),
        top_k=(options.get("ensemble") or {}).get("top_k", 3),
        method=(options.get("ensemble") or {}).get("method", "weights"))
    return OptionSet(
        scheduler=scheduler, mps=mps, resource_caps=resources,
        core_allocator=core_allocator, session=session, sampler=sampler,
        pruner=pruner, fidelity=fidelity, objective_mode=objective_mode,
        shared_data=shared_data, warm_start=warm_start, ensembler=ensembler)
