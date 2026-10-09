"""Head of the embedded HPO Kaggle kernel-script text (cli.laya_hpo)."""
from __future__ import annotations

HPO_KERNEL_TEMPLATE_HEAD = '''\
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
import random
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
# Config-declared visible GPU set (options.session.cuda_visible_devices). Set it
# BEFORE torch is imported by any injected patch or worker: "" keeps every
# device the session exposes (one worker process per device); "0" pins the
# session to one T4 so slots_per_gpu processes share it through MPS.
CUDA_VISIBLE_DEVICES = "@CUDA_VISIBLE_DEVICES@"
if CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
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


class StorageResolver:
    """Open the session's Optuna study on the right storage backend.

    One job: prefer the shared Postgres RDB, fall back to the single-process
    SQLite study (writing the offline marker) when it is missing/unreachable.
    ``offline``/``config`` expose the outcome to the caller.
    """

    def __init__(self, optuna, options, study_name, *, offline=None):
        self.optuna = optuna
        self.options = options
        self.study_name = study_name
        self.offline = (bool(options.session.offline) if offline is None
                        else bool(offline))
        self.config = None

    def sqlite(self):
        WORKING.mkdir(parents=True, exist_ok=True)
        return self.optuna.storages.RDBStorage(
            "sqlite:///" + str(WORKING / "hpo_offline.db"))

    def _create(self, storage, sampler, pruner, study_kwargs):
        return self.optuna.create_study(
            study_name=self.study_name, sampler=sampler, pruner=pruner,
            storage=storage, **self.options.session.study_kwargs(),
            **study_kwargs)

    def open(self, sampler, pruner, study_kwargs):
        """Create or resume the study, degrading to SQLite on an RDB outage."""
        if self.offline:
            return self._create(self.sqlite(), sampler, pruner, study_kwargs)
        try:
            ensure_optuna_url()
            self.config = storage_from_environment()
            return self._create(create_storage(self.config), sampler, pruner,
                                study_kwargs)
        except Exception as error:
            log("shared Postgres unavailable (%s); falling back to the offline "
                "SQLite study + hpo_trials ledger" % str(error)[:160])
            self.offline = True
            self.config = None
            mark_offline(str(error)[:200])
            return self._create(self.sqlite(), sampler, pruner, study_kwargs)

    def load(self):
        """Load an existing study (session end), degrading on an RDB outage."""
        if self.offline:
            return self.optuna.load_study(study_name=self.study_name,
                                          storage=self.sqlite())
        try:
            storage = create_storage(storage_from_environment())
        except Exception as error:
            log("shared Postgres unavailable at session end (%s); reading "
                "the offline SQLite study" % str(error)[:160])
            self.offline = True
            storage = self.sqlite()
        return self.optuna.load_study(study_name=self.study_name,
                                      storage=storage)



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


def exclusive_lock(path):
    """A cross-process exclusive lock (fcntl), no-op where fcntl is absent."""
    import contextlib

    @contextlib.contextmanager
    def held():
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX host
            yield
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    return held()


def locate_base_model_dir(root):
    candidate = root / BASE_MODEL_DIR
    if (candidate / "rl_agent_config.json").is_file():
        return candidate
    for found in sorted(root.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError("base-model archive carried no rl_agent_config.json")


def extract_base_model(archive):
    """Extract the base-model archive ONCE, safely, into shared WORKING.

    Worker processes and DDP ranks all share ``WORKING``; the old
    rmtree+extract raced and could delete a peer's in-progress extraction. The
    lock serialises extraction and an atomic ``os.replace`` publishes a fully
    built staging dir; a ``.extract_ready`` marker makes every later caller
    reuse it without re-extracting.
    """
    destination = WORKING / "base_model"
    ready = destination / ".extract_ready"
    if ready.is_file():
        return locate_base_model_dir(destination)
    WORKING.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(WORKING / ".base_model.lock"):
        if ready.is_file():
            return locate_base_model_dir(destination)
        staging = WORKING / (".base_model.stage." + str(os.getpid()))
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        stream = open_zstd(str(archive))
        try:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                try:
                    tar.extractall(staging, filter="data")
                except TypeError:
                    tar.extractall(staging)
        finally:
            stream.close()
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        (destination / ".extract_ready").write_text("ok", encoding="utf-8")
        return locate_base_model_dir(destination)


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


def apply_fidelity_and_staging(options, trial_number, config_dict, control_dict,
                               train_path):
    """Apply the fidelity/staged/seed/profiler levers for ONE trial.

    ONE implementation shared by the slots and DDP paths, so a config dial can
    never behave differently in the two parallelism modes.
    """
    fidelity = options.fidelity
    resource = fidelity.resource(int(trial_number))
    if fidelity.dimension == "epochs":
        config_dict["epochs"] = int(resource)
    # Coarse-to-fine: freeze the encoder for `head_only_epochs`, THEN unfreeze.
    # The perf patch enables the gradual path only when `freeze_encoder` is
    # False and `unfreeze_after_epoch` is set; setting freeze_encoder=True here
    # (the old bug) disabled gradual and froze the encoder for the whole run.
    frozen = fidelity.encoder_frozen_epochs()
    if frozen > 0:
        config_dict["freeze_encoder"] = False
        control_dict["unfreeze_after_epoch"] = int(frozen)
    # A fixed seed keeps every trial's mini-batch/option-order draws identical.
    config_dict["seed"] = int(HPO_SPACE.get("seed", config_dict.get("seed", 0)))
    # ONE profiler per trial: the HPO TrialProfiler owns profiling; silence the
    # perf patch's nested ProfilerSession so they never double-count.
    control_dict["profile"] = False
    trial_train = (subset_train_json(train_path, resource, fidelity.full,
                                     config_dict["seed"])
                   if fidelity.dimension == "subset" else train_path)
    return resource, frozen, trial_train


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
    resource, frozen, trial_train = apply_fidelity_and_staging(
        options, trial.number, config_dict, control_dict, train_path)
    start_model = options.warm_start.source() or base_model

    log("trial %d start device=%s resource=%s dim=%s freeze=%s warm=%s dials=%s"
        % (int(trial.number), device, resource, options.fidelity.dimension,
           frozen, options.warm_start.mode,
           json.dumps(dials, sort_keys=True)))
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
    finished = started
    # Fail-soft: the profiler never fails the trial (its __exit__ returns
    # False), so a finetune error still propagates unchanged.
    try:
        with profiler:
            run_laya_finetune(trial_train, dev_path, start_model, out_dir,
                              device)
            # Stamp AFTER training but BEFORE profiler teardown/export so the
            # reported epoch_time_s is real work, not trace-export noise.
            finished = time.time()
    finally:
        if reporter is not None:
            reporter.uninstall()
    epoch_time = max(0.0, finished - started)
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


def subset_train_json(train_path, resource, full, seed=0):
    """Deterministic, NESTED training-subset JSONL (atomic + cached).

    A shared seed shuffles the corpus once, so the rung for ``resource`` is a
    prefix of every larger rung and low/high-fidelity ranks stay comparable.
    The write is staged then ``os.replace``d so concurrent workers never read a
    half-written subset.
    """
    if not full or int(resource) >= int(full):
        return train_path
    fraction = max(1, int(resource)) / float(full)
    out = WORKING / ("subset_" + str(int(resource)) + "_s" + str(int(seed))
                     + ".jsonl")
    if out.is_file():
        return out
    from laya import train as laya_train
    rows = laya_train.read_jsonl(str(train_path))
    order = list(range(len(rows)))
    random.Random(int(seed)).shuffle(order)
    keep = max(1, int(len(rows) * fraction))
    staging = out.with_name(out.name + ".tmp." + str(os.getpid()))
    with staging.open("w", encoding="utf-8") as handle:
        for index in order[:keep]:
            handle.write(json.dumps(rows[index]) + "\\n")
    os.replace(staging, out)
    log("fidelity subset %d/%d rows (seed=%d) -> %s"
        % (keep, len(rows), int(seed), out.name))
    return out


def _enqueued_param_key(params):
    try:
        return tuple(sorted((str(k), repr(v)) for k, v in (params or {}).items()))
    except Exception:
        return None


def enqueue_warm_start(study, seeds):
    """Enqueue warm-start configs ONCE across workers/sessions.

    ``study.enqueue_trial`` is not idempotent: every concurrent worker and every
    resumed session would add another WAITING trial with the same fixed params.
    Skip a seed whose fixed params are already waiting.
    """
    if not seeds:
        return 0
    already = set()
    try:
        for existing in study.trials:
            if existing.state.name != "WAITING":
                continue
            key = _enqueued_param_key(
                existing.system_attrs.get("fixed_params"))
            if key is not None:
                already.add(key)
    except Exception:
        already = set()
    enqueued = 0
    for seed_config in seeds:
        key = _enqueued_param_key(seed_config)
        if key is not None and key in already:
            continue
        try:
            study.enqueue_trial(seed_config)
            if key is not None:
                already.add(key)
            enqueued += 1
        except Exception as error:
            log("enqueue_trial skipped: " + str(error)[:160])
    if enqueued:
        log("warm-start enqueued %d seed config(s)" % enqueued)
    return enqueued


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
    rank 0 writes the metric; only this controller calls study.tell.

    Rank 0 streams each epoch's dev accuracy to a sidecar; the controller
    reports it to the active trial and kills the subprocess the moment the
    pruner says stop, so DDP pruning is as real as the slots path.
    """
    script = os.path.abspath(__file__)
    runner = DdpTrialRunner(options.scheduler, script, result_dir=WORKING,
                            logger=log)
    optuna_module = globals().get("optuna")

    def objective(trial):
        def run_fn(t):
            dials = sample_dials(t, HPO_SPACE)

            def on_epoch(step, value):
                trial.report(value, step)

            def should_stop():
                return bool(trial.should_prune())

            try:
                return runner.run(int(t.number), {"dials": dials},
                                  on_epoch=on_epoch, should_stop=should_stop)
            except TrialPrunedSignal:
                raise optuna_module.TrialPruned(
                    "pruned at DDP fidelity stage")
        value = objective_value(trial, run_fn, lease_store, champion_store,
                                GENERATION_ID, MODEL_KEY,
                                objective_mode=options.objective_mode)
        return value
    return objective


def ddp_metric_stream_path(trial_number):
    """The rank0 -> controller per-epoch metric sidecar for ONE DDP trial."""
    return WORKING / ("ddp_trial_" + str(int(trial_number))
                      + DdpTrialRunner.EPOCH_STREAM_SUFFIX)


def ddp_metric_sink(trial_number):
    """Append one ``{"step", "value"}`` line per epoch (rank 0 only)."""
    path = ddp_metric_stream_path(trial_number)

    def sink(step, value):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": int(step), "value": float(value)})
                         + "\\n")
            handle.flush()

    return sink


def run_ddp_trial(trial_number):
    """One DDP trial on THIS rank (launched by torchrun via DdpTrialRunner)."""
    ensure_optuna_url()
    payload = json.loads(os.environ.get("ER_LAYA_HPO_DDP_PAYLOAD", "{}"))
    import torch
    from laya import train as laya_train
    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    globals()["OPTION_SET"] = options
    # Rank processes do NOT pass through run_worker, so apply the CPU/CUDA caps
    # here too (the old path skipped resource caps entirely under DDP).
    options.resource_caps.apply_torch(torch)
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Guard the device: a rank whose cuda:<local_rank> does not exist (1-GPU or
    # CPU session) falls back to CPU instead of crashing on cuda:1.
    if device_count and local_rank < device_count:
        device = "cuda:" + str(local_rank)
    else:
        device = "cpu"
    train_path = resolve_input(TRAIN_JSONL)
    dev_path = resolve_input(DEV_JSONL)
    base_model = extract_base_model(resolve_input(BASE_MODEL_ARCHIVE))
    dials = payload.get("dials") or {}
    config_dict, control_dict = route_dials(
        BASE_FINETUNE_CONFIG, BASE_FINETUNE_CONTROL, dials, HPO_SPACE)
    globals()["FINETUNE_CONFIG"] = config_dict
    globals()["FINETUNE_CONTROL"] = control_dict
    globals()["FINETUNE_CONTROL_RESULT"] = None
    globals()["FINETUNE_DEV_ROWS"] = None
    resource, frozen, trial_train = apply_fidelity_and_staging(
        options, trial_number, config_dict, control_dict, train_path)
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
        wandb_log=wandb_log_profiler, rank0=is_rank0())
    reporter = None
    if options.pruner.kind != "none" and is_rank0():
        reporter = FidelityReporter(None, None, globals(),
                                    sink=ddp_metric_sink(trial_number))
        reporter.install()
    started = time.time()
    finished = started
    # The DDP backend is config SSOT (options.ddp.backend); pass it through so a
    # configured gloo/nccl choice is actually applied, not just recorded.
    init_distributed(backend=getattr(options.scheduler, "backend", None))
    try:
        with profiler:
            run_laya_finetune(trial_train, dev_path, start_model, out_dir,
                              device)
            finished = time.time()
    finally:
        if reporter is not None:
            reporter.uninstall()
        destroy_if_distributed()
    if is_rank0():
        result = globals().get("FINETUNE_CONTROL_RESULT") or {}
        data = {"accuracy": result.get("best_dev_accuracy"),
                "dev_loss": dev_loss_from_report(out_dir),
                "checkpoint": str(out_dir),
                "epoch_time_s": max(0.0, finished - started)}
        path = DdpTrialRunner(options.scheduler, os.path.abspath(__file__),
                              result_dir=WORKING).result_path(trial_number)
        path.write_text(json.dumps(data) + "\\n", encoding="utf-8")


'''
