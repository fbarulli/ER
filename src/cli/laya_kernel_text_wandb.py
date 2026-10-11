"""The ONE shared W&B helpers kernel text for the staged laya kernels (SSOT).

The fine-tune kernel (``cli.laya_kernel_text_finetune``) and the HPO kernel
(``cli.laya_hpo_kernel_text_head``) each carried a near-duplicate wandb helper
block; this module owns the single text, injected at the ``@WANDB_HELPERS@``
marker when each kernel-text module is imported (the same assembly pattern as
``cli.laya_hpo_kernel_text``). Both kernels define ``log()``, ``WANDB_API_KEY``,
``WANDB_PROJECT``, ``RUN_TAG`` and ``WANDB_INIT_CONFIG`` in their own headers;
the helpers resolve those names from the shared kernel namespace at call time.

STEP CONTRACT (the 8146f87 regression fix): EVERY sink logs on wandb's
monotonic auto-increment step — no caller passes an explicit ``step=``. The
epoch number rides the payload as a VALUE, and ``wandb_init`` declares
``epoch`` the step metric of the per-epoch namespaces, so per-epoch charts are
indexed by epoch while the interleaved boot/log-line/event/timing/gpu sinks can
never regress the step again (wandb silently DROPPED every explicit-step epoch
row once the auto-increment sinks had advanced the global step: "Tried to log
to step 0 that is less than the current step N ... this data will be ignored").

TEARDOWN: ``wandb_finish`` re-drives ``wandb sync`` on the run dir — the
documented recovery for "Fatal error while uploading data ... Use wandb sync"
— fail-soft with the full recorded reason, so telemetry never dies silently.
"""
from __future__ import annotations

#: The shared helper block, injected verbatim into BOTH staged kernel texts.
WANDB_HELPERS_SOURCE = '''\
from pydantic import BaseModel, ConfigDict, Field

WANDB_RUN = None

# The per-epoch metric namespaces: `wandb_init` declares `epoch` their step
# metric, so every per-epoch row charts against the epoch value while the run's
# global step stays monotonic for the interleaved sinks (SSOT of the step
# contract; the host-side consolidated plot reads these same namespaces).
EPOCH_STEP_METRICS = ("train/*", "dev/*", "select/*", "early_stop/*",
                      "epoch_time_s")

# `wandb sync` teardown-retry budget (seconds): an already-synced run dir is a
# fast no-op, an interrupted upload gets ONE bounded re-drive.
WANDB_SYNC_TIMEOUT_SECONDS = 900.0
TRAINING_STEP_METRIC = "train_step"

class WandbTrainingMetrics(BaseModel):
    """Measured batch values use training steps, separate from epoch summaries."""

    model_config = ConfigDict(populate_by_name=True)
    step: int = Field(alias=TRAINING_STEP_METRIC)
    loss: float = Field(alias="train/batch_loss")
    learning_rate: float = Field(alias="train/batch_lr")

    @classmethod
    def define(cls, wandb) -> None:
        wandb.define_metric(TRAINING_STEP_METRIC)
        for name, field in cls.model_fields.items():
            if name == "step":
                continue
            if field.alias is not None:
                wandb.define_metric(field.alias, step_metric=TRAINING_STEP_METRIC)

    @classmethod
    def publish(cls, step: int, loss: float, learning_rate: float) -> None:
        if WANDB_RUN is None:
            return
        try:
            payload = cls(step=step, loss=loss, learning_rate=learning_rate)
            WANDB_RUN.log(payload.model_dump(by_alias=True))
        except Exception as error:
            log("[wandb] training metrics failed: " + type(error).__name__
                + ": " + str(error) + "\\n" + traceback.format_exc())

class WandbLogSink:
    """Batch notable log lines into the live W&B text channel `log/line`.

    W&B exposes `output.log` only on flush/end, so `log/line` is the sole
    surface visible while the run is live. Lines are buffered and flushed on N
    lines or the bounded interval; the capped pre-init buffer holds the kernel
    boot lines until `wandb_init` succeeds. Fail-soft: a write error is recorded
    in full, never fatal.
    """

    KEY = "log/line"
    MAX_LINES = 20
    MAX_BUFFER = 400
    FLUSH_SECONDS = 15.0

    def __init__(self):
        self._pending = []
        self._flushed_at = time.monotonic()

    def add(self, line):
        self._pending.append(str(line))
        if len(self._pending) > self.MAX_BUFFER:
            del self._pending[:-self.MAX_BUFFER]
        if WANDB_RUN is not None and (
                len(self._pending) >= self.MAX_LINES
                or time.monotonic() - self._flushed_at >= self.FLUSH_SECONDS):
            self.flush()

    def flush(self):
        if WANDB_RUN is None or not self._pending:
            return
        text = "\\n".join(self._pending)
        self._pending = []
        self._flushed_at = time.monotonic()
        try:
            WANDB_RUN.log({self.KEY: text})
        except Exception as error:
            log("[wandb] log sink skipped: " + type(error).__name__ + ": "
                + str(error) + "\\n" + traceback.format_exc())


LOG_SINK = WandbLogSink()


def wandb_init(name=None, group=None, config=None):
    """Start the optional wandb mirror (project `tracking.wandb.project`).

    The API key travels baked (read from .env at staging, never from the repo);
    with no key the run stays local and artifacts are the record, exactly like
    the ER tracking contract. Rank 0 only (the caller gates it). After init,
    `epoch` is declared the step metric of the per-epoch namespaces: every sink
    then rides the monotonic auto-increment step and no per-epoch row can be
    dropped by an interleaved sink again (the 8146f87 regression class).
    """
    global WANDB_RUN
    if not WANDB_API_KEY:
        log("[wandb] no WANDB_API_KEY; tracking disabled")
        return None
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    try:
        import wandb
        WANDB_RUN = wandb.init(project=WANDB_PROJECT, name=name or RUN_TAG,
                               group=group, reinit=True,
                               config=config if config is not None
                               else WANDB_INIT_CONFIG)
        wandb.define_metric("epoch")
        for pattern in EPOCH_STEP_METRICS:
            wandb.define_metric(pattern, step_metric="epoch")
        WandbTrainingMetrics.define(wandb)
        log("[wandb] run " + str(getattr(WANDB_RUN, "id", "")) + " -> "
            + WANDB_PROJECT)
    except Exception as error:  # tracking is best-effort, never fatal
        log("[wandb] init skipped: " + type(error).__name__ + ": "
            + str(error) + "\\n" + traceback.format_exc())
        WANDB_RUN = None
    LOG_SINK.flush()
    return WANDB_RUN


def wandb_log_epoch(epoch, mean, extra=None):
    """Log ONE per-epoch metric row (epoch, train/mean_loss + dev/lr/grad/time).

    Rides the monotonic auto-increment step — NEVER an explicit ``step=``: the
    explicit 0-based epoch step regressed against the interleaved log/line,
    event, timing and gpu sinks, and wandb silently dropped every epoch row
    ("Tried to log to step 0 that is less than the current step N"). The
    `epoch` value is the declared step metric of the per-epoch namespaces, so
    charts still index by epoch, one row per epoch.
    """
    if WANDB_RUN is None:
        return
    payload = {"epoch": epoch + 1, "train/mean_loss": mean}
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
    try:
        WANDB_RUN.log(payload)
    except Exception as error:  # logging must never crash training
        log("[wandb] epoch log skipped: " + type(error).__name__ + ": "
            + str(error) + "\\n" + traceback.format_exc())
        return
    LOG_SINK.add("epoch %d loss=%.4f" % (epoch + 1, mean))


def wandb_log_control_summary(result):
    """Log the end-of-run selection/early-stop scalars (rank 0 only)."""
    if WANDB_RUN is None or not isinstance(result, dict):
        return
    payload = {
        "early_stop/stopped": 1 if result.get("stopped") else 0,
        "early_stop/stopped_epoch": result.get("stopped_epoch"),
        "select/best_dev_accuracy": result.get("best_dev_accuracy"),
        "select/bad_epochs": result.get("bad_epochs"),
    }
    payload = {key: value for key, value in payload.items()
               if value is not None}
    if payload:
        try:
            WANDB_RUN.log(payload)
        except Exception as error:
            log("[wandb] control summary skipped: " + type(error).__name__
                + ": " + str(error) + "\\n" + traceback.format_exc())


def wandb_log_metrics(report):
    """Log a before/after evaluation report (flat `phase/key` metrics)."""
    if WANDB_RUN is None or not isinstance(report, dict):
        return
    flat = {}
    for phase in ("before", "after"):
        block = report.get(phase) or {}
        for key in ("accuracy", "loss", "ece", "brier", "brier_top1",
                    "mean_confidence"):
            if block.get(key) is not None:
                flat[phase + "/" + key] = block[key]
    if flat:
        try:
            WANDB_RUN.log(flat)
        except Exception as error:  # logging must never crash training
            log("[wandb] metrics log skipped: " + type(error).__name__ + ": "
                + str(error) + "\\n" + traceback.format_exc())


def wandb_sync_unsent(run_dir):
    """Re-drive wandb's upload for this run's dir (the documented recovery for
    "Fatal error while uploading data ... Use wandb sync").

    Runs after EVERY finish(): an already-synced dir is a fast no-op and an
    interrupted upload is retried, so telemetry never dies silently. Fail-soft
    with the FULL recorded reason — the staged artifacts are the durable
    record; a sync failure must never fail the session.
    """
    if not run_dir:
        return
    sync_dir = str(Path(run_dir).parent)
    command = [sys.executable, "-m", "wandb", "sync", sync_dir]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=WANDB_SYNC_TIMEOUT_SECONDS)
    except Exception as error:
        log("[wandb] sync skipped: " + type(error).__name__ + ": "
            + str(error) + "\\n" + traceback.format_exc())
        return
    if completed.returncode != 0:
        log("[wandb] sync failed (exit " + str(completed.returncode) + "): "
            + ((completed.stderr or completed.stdout or "")[-2000:]))


def wandb_finish():
    LOG_SINK.flush()
    if WANDB_RUN is None:
        return
    run_dir = str(getattr(WANDB_RUN, "dir", "") or "")
    try:
        WANDB_RUN.finish()
    except Exception as error:
        log("[wandb] finish skipped: " + type(error).__name__ + ": "
            + str(error) + "\\n" + traceback.format_exc())
    wandb_sync_unsent(run_dir)


def wandb_log_event(name, **fields):
    """Log ONE structured run transition to wandb (no stream of noise).

    The transition name rides ``run/event``; each non-None detail rides
    ``run/event/<key>``. Fail-soft: a tracking error is recorded, never fatal.
    """
    if WANDB_RUN is None:
        return
    payload = {"run/event": str(name)}
    for key, value in fields.items():
        if value is not None:
            payload["run/event/" + key] = value
    try:
        WANDB_RUN.log(payload)
    except Exception as error:
        log("wandb event skipped: " + type(error).__name__ + ": "
            + str(error) + "\\n" + traceback.format_exc())


def wandb_log_profiler(table):
    """Mirror a per-trial key_averages() top-op table to wandb (best-effort)."""
    if WANDB_RUN is None:
        return
    try:
        WANDB_RUN.log({"profiler/top_ops": table})
    except Exception as error:
        log("wandb profiler skipped: " + type(error).__name__ + ": "
            + str(error) + "\\n" + traceback.format_exc())
'''
