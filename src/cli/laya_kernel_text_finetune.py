"""Embedded kernel-script text for the FINE-TUNE / FINE-TUNE-EVAL payloads.

The runtime preflights, the CUDA device patch and the finetune / eval kernel
bodies the laya lane token-substitutes into staged kaggle scripts. Pure text
(no module dependencies), kept apart from the orchestration in ``cli.laya_lane``.
"""
from __future__ import annotations


# The fine-tune corpus travels as its own attached dataset: this preflight
# verifies THE ATTACHED INPUTS (train/dev/test JSONL land under
# /kaggle/input/<slug>/ and rglob finds them by name). It bakes the same
# REPOSITORY/BRANCH/REVISION/_runtime_files inventory the lane push gate
# literal-evals.
FINETUNE_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@TRAIN_JSONL@", "@DEV_JSONL@", "@TEST_JSONL@")
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus inputs (the finetune dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail
    loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# The eval-only kernel attaches the SAME corpus dataset and verifies the ONE
# held-out split it scores (rglob finds the JSONL under /kaggle/input/<slug>/).
# The CHECKPOINT is a separate attached dataset, resolved in-kernel by the
# `rl_agent_config.json` rglob (never vendored here): the push gate's
# `_runtime_files` inventory can only verify files that live in the staged
# dataset_payload, so the checkpoint inventory stays out of it and fails loud
# in `resolve_checkpoint()` instead.
FINETUNE_EVAL_RUNTIME_PREFLIGHT = '''\
_runtime_files = ("@EVAL_JSONL@",)
INPUT_ROOT = Path("/kaggle/input")


def laya_runtime_preflight():
    """Verify the ATTACHED corpus split (the corpus dataset mounts under
    /kaggle/input/<slug>/ and rglob searches recursively by name); fail
    loud before pip touches anything."""
    missing = [name for name in _runtime_files
               if not any(INPUT_ROOT.rglob(name))]
    if missing:
        raise FileNotFoundError(
            "Runtime preflight missing attached inputs: "
            + ", ".join(missing))
    print("[runtime-preflight] verified %d required files"
          % len(_runtime_files), flush=True)


laya_runtime_preflight()
'''


# Runtime device patch for the finetune kernel (laya<=0.4.0). `finetune()`
# evaluates the base checkpoint via `calibration_records()` BEFORE
# `train_model()` calls `model.to(device)`, so `load_checkpoint()`'s CPU
# model meets cuda `input_ids` and `index_select` raises "index is on
# cuda:0, different from other tensors on cpu" on the T4. Injected into the
# kernel below at the `@DEVICE_PATCH@` marker; it leaves the recipe, flags
# and the single-T4 rule untouched.
FINETUNE_DEVICE_PATCH_SOURCE = '''\
def force_model_to_device(model, device):
    # Move every module, and every registered buffer (non-persistent ones
    # included), onto `device` before any forward pass.
    import torch
    device = torch.device(device)
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
        module.to(device)
    return model.to(device)


def apply_device_patch():
    # Wrap the two forward entrypoints so the model is on the training
    # device before any forward pass. `calibration_records` is the crash:
    # it runs the base checkpoint on device inputs while the model is
    # still CPU. `train_model` is wrapped for the same invariant.
    from laya import train as laya_train

    original_calibration_records = laya_train.calibration_records

    def calibration_records(model, tok, items, device, *args, **kwargs):
        force_model_to_device(model, device)
        return original_calibration_records(
            model, tok, items, device, *args, **kwargs)

    laya_train.calibration_records = calibration_records

    original_train_model = laya_train.train_model

    def train_model(model, tok, items, config, device, *args, **kwargs):
        force_model_to_device(model, device)
        return original_train_model(
            model, tok, items, config, device, *args, **kwargs)

    laya_train.train_model = train_model
'''


FINETUNE_KERNEL_SCRIPT = '''\
"""ER laya fine-tune on a Kaggle GPU session (cli.laya_lane).

Distributed data-parallel over the 2xT4 pair: when two CUDA devices are
visible `main()` spawns one process per device (nccl), wraps the model in
`DistributedDataParallel` and shards the training items with a per-rank
`DistributedSampler`; a single GPU / CPU falls back to the original
one-process path unchanged (opt out with ER_LAYA_DDP=0). Installs laya over
pip (pinned `laya>=0.3.29`), reads the attached JSONL corpus (train/dev/test
+ receipt, the er-laya-train dataset), extracts the attached base-model
archive (the er-laya-base dataset; the shipped convaiinnovations/laya
checkpoint) to a local dir, builds the FULL `laya.train.TrainConfig` from
the YAML-driven `FINETUNE_CONFIG` (every trainer knob is config SSOT), and
calls `laya.train.finetune(...)` directly with the extracted DIRECTORY as
the base -- so `resolve_checkpoint_dir` takes the isdir branch and NEVER
calls the Hub. Rank 0 alone scores the held-out split, writes the receipt
and stages the checkpoint tar into /kaggle/working for hash-verified
fetch-back.
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
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
TRAIN_JSONL = "@TRAIN_JSONL@"
DEV_JSONL = "@DEV_JSONL@"
TEST_JSONL = "@TEST_JSONL@"
BASE_MODEL_ARCHIVE = "@BASE_MODEL_ARCHIVE@"
BASE_MODEL_DIR = "@BASE_MODEL_DIR@"
FINETUNE_DEVICE = "@FINETUNE_DEVICE@"
FINETUNE_CONFIG = @FINETUNE_CONFIG@
# YAML-driven training controls for the perf patch (NOT TrainConfig kwargs:
# laya's TrainConfig rejects unknown keys). Baked as ONE repr literal and read
# by `_perf_train_model` from this module global.
FINETUNE_CONTROL = @FINETUNE_CONTROL@
HELD_OUT_BATCH = @HELD_OUT_BATCH@
WANDB_API_KEY = "@WANDB_API_KEY@"
WANDB_PROJECT = "@WANDB_PROJECT@"
# The receipt member `collect_kaggle_result` requires: `laya_<kind>.receipt.json`
# so the CPU smoke's kind and the prod kind each carry their own member.
RECEIPT_NAME = "@RECEIPT_NAME@"

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WANDB_RUN = None
# Per-epoch dev rows + the checkpoint root, set by run_laya_finetune() before
# finetune() so the perf patch (same module namespace) can evaluate/checkpoint.
FINETUNE_DEV_ROWS = None
FINETUNE_OUTPUT_DIR = None
# The SHARED canonical checkpoint dir (all DDP ranks) + the run identity used
# to reject another run's checkpoints on resume.
FINETUNE_CHECKPOINT_DIR = None
FINETUNE_RUN_TAG = None


def wandb_init():
    """Start the optional wandb mirror (project `tracking.wandb.project`).

    The API key travels baked (read from .env at staging, never from the repo);
    with no key the run stays local and artifacts are the record, exactly like
    the ER tracking contract. Rank 0 only (the caller gates it)."""
    global WANDB_RUN
    if not WANDB_API_KEY:
        print("[wandb] no WANDB_API_KEY; tracking disabled", flush=True)
        return None
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    try:
        import wandb
        WANDB_RUN = wandb.init(project=WANDB_PROJECT, name=RUN_TAG,
                               config=FINETUNE_CONFIG)
        print("[wandb] run " + str(getattr(WANDB_RUN, "id", ""))
              + " -> " + WANDB_PROJECT, flush=True)
    except Exception as error:  # tracking is best-effort, never fatal
        print("[wandb] init skipped: " + type(error).__name__ + ": "
              + str(error)[:200], flush=True)
        WANDB_RUN = None
    return WANDB_RUN


def wandb_log_epoch(epoch, mean, extra=None):
    if WANDB_RUN is None:
        return
    payload = {"epoch": epoch + 1, "train/mean_loss": mean}
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
    WANDB_RUN.log(payload, step=epoch)


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
        WANDB_RUN.log(payload)


def wandb_log_metrics(report):
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
        WANDB_RUN.log(flat)


def wandb_finish():
    if WANDB_RUN is not None:
        try:
            WANDB_RUN.finish()
        except Exception:
            pass


@DEVICE_PATCH@

@PERF_PATCH@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    """laya installs over pip, pinned; torch is already on the session."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4).

    `auto`/`cuda` require a live cuda session and resolve to device 0; an
    explicit other device (e.g. `cpu`) is passed through while cuda stays
    pinned to device 0, so a second accelerator is never visible."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if FINETUNE_DEVICE not in ("auto", "cuda"):
        log("device configured: " + FINETUNE_DEVICE
            + " (cuda pinned to device 0)")
        return FINETUNE_DEVICE
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya finetune dataset)")


def open_zstd(path):
    """Open a `.tar.zst` stream with whichever zstd binding the session has.

    The base checkpoint ships as a zstd tar (the project transport); never
    falls back to the network for the checkpoint itself. Python 3.14 exposes
    `compression.zstd`; the Kaggle 3.13 image needs `zstandard` (installed
    on demand only when neither binding is importable)."""
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
    """Extract the attached base-model tar.zst and return the directory that
    carries rl_agent_config.json.

    `--base` then points at a LOCAL dir, so laya's resolve_checkpoint_dir
    takes the isdir branch and NEVER calls snapshot_download (the HF
    dependency is gone from this path)."""
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
    raise FileNotFoundError(
        "base-model archive carried no rl_agent_config.json")


def run_laya_finetune(train_path, dev_path, base_model, out_dir, device):
    """Apply the PERF patch then the device patch, build the FULL
    `TrainConfig` from FINETUNE_CONFIG, and call `laya.train.finetune`
    directly.

    The `laya-train` CLI only exposes a subset of the trainer surface, so
    the non-CLI knobs are set by constructing the config here and calling
    `finetune` in-process (the monkeypatches reach the same
    `train_model`/`calibration_records` entrypoints finetune calls). PERF
    first so the device wrapper closes over the patched train_model (see
    FINETUNE_PERF_PATCH_SOURCE)."""
    apply_perf_patch()
    apply_device_patch()
    from laya import train as laya_train
    config = laya_train.TrainConfig(**FINETUNE_CONFIG,
                                    eval_data=str(dev_path))
    config.validate()
    # Stash the attached dev split + the checkpoint roots for the perf patch
    # (same module namespace): per-epoch dev eval and epoch_<n>.pt resume/
    # checkpointing read these globals. FINETUNE_CHECKPOINT_DIR is the SHARED
    # canonical dir on every rank (rank 0 writes, all ranks resume), so DDP
    # ranks cannot diverge on resume.
    globals()["FINETUNE_OUTPUT_DIR"] = str(out_dir)
    globals()["FINETUNE_CHECKPOINT_DIR"] = str(WORKING / "checkpoint")
    globals()["FINETUNE_RUN_TAG"] = RUN_TAG
    if FINETUNE_CONTROL.get("eval_dev"):
        try:
            globals()["FINETUNE_DEV_ROWS"] = laya_train.read_jsonl(
                str(dev_path))
        except Exception as error:
            log("dev rows load skipped: " + str(error)[:200])
    log("TrainConfig: " + json.dumps(FINETUNE_CONFIG, sort_keys=True))
    log("FINETUNE_CONTROL: " + json.dumps(FINETUNE_CONTROL, sort_keys=True))
    # laya evaluates the dev split before training and prints nothing while it
    # does (27k items here -> several minutes of silence). Say so, so the quiet
    # stretch is not mistaken for a hang.
    log("starting laya.train.finetune: the pre-train dev evaluation runs "
        "silently until the first 'epoch 1/8 step' line (minutes on the full "
        "corpus)")
    return laya_train.finetune(
        data=str(train_path), model_dir=str(base_model),
        output_dir=str(out_dir), config=config, device=device)


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_held_out(test_path, checkpoint, device):
    """Score the just-trained checkpoint on the HELD-OUT test split.

    DEFAULT-ON (owner order): the training-time eval is the dev split and
    overlaps training/calibration, so this is the run's honest generalization
    number. The raw (uncalibrated) metrics are reported — the calibration was
    already fitted on the calibration split during training. Opt out with
    ER_LAYA_HELD_OUT=0.
    """
    import torch
    from laya import train as laya_train

    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(test_path))
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise SystemExit(
            "held-out split " + test_path.name + " produced no usable items "
            "(skipped: " + repr(skipped) + ")")
    log("held-out rows " + str(len(rows)) + " -> items " + str(len(items)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=HELD_OUT_BATCH, parallel=parallel)
    return {
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_source": test_path.name,
        "eval_split": "test",
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint),
        "metrics": laya_train.evaluate_records(records),
    }


def session_env():
    # Self-report the container session identity so the host-side log follower
    # can persist logs/kaggle/<kernel>.session_id for the verified in-place
    # stop. Kaggle sets NO KAGGLE_KERNEL_RUN_ID/KAGGLE_SESSION_ID in the
    # container; the only per-run id is the numeric suffix of
    # KAGGLE_CONTAINER_NAME ("kaggle_<token>-<session_id>-webtier"), which the
    # SDK cancel_kernel_session accepts.
    _container = os.environ.get("KAGGLE_CONTAINER_NAME", "")
    _parts = _container.rsplit("-", 2)
    _session_id = (_parts[1] if len(_parts) == 3 and _parts[1].isdigit()
                   else "")
    print("[kaggle-session] session_id=" + _session_id
          + " container=" + _container, flush=True)
    return {"KAGGLE_CONTAINER_NAME": _container, "session_id": _session_id}


def _finetune_session(distributed, session):
    if distributed:
        local_rank, rank, world_size = dist_env()
        device = "cuda"
        log("rank %d/%d on cuda:%d (DDP)" % (rank, world_size, local_rank))
    else:
        device = pick_device()
    train = resolve_input(TRAIN_JSONL)
    dev = resolve_input(DEV_JSONL)
    test = resolve_input(TEST_JSONL)
    log("corpus: " + train.name + " + " + dev.name + " (+ " + test.name + ")")
    WORKING.mkdir(parents=True, exist_ok=True)
    archive = resolve_input(BASE_MODEL_ARCHIVE)
    log("base-model archive: " + str(archive))
    base_model = extract_base_model(archive)
    log("base model: " + str(base_model))
    out_dir = WORKING / "checkpoint"
    if distributed and not is_rank0():
        # Non-rank-0 ranks must NEVER write the canonical checkpoint: suppress
        # laya's per-epoch + final save_checkpoint and send any incidental
        # byte to a private scratch dir (never /kaggle/working/checkpoint).
        from laya import train as laya_train
        laya_train.save_checkpoint = lambda *args, **kwargs: None
        out_dir = WORKING / ("checkpoint.rank" + str(dist_env()[1]))
    if is_rank0():
        wandb_init()
    gpu_handle = start_gpu_sampler() if is_rank0() else None
    try:
        summary = run_laya_finetune(train, dev, base_model, out_dir, device)
    finally:
        stop_gpu_sampler(gpu_handle)

    def rank0_work():
        receipt = {
            "gpu_kind": "finetune",
            "gpu": "T4 (single)",
            "session_env": session,
            "run_tag": RUN_TAG,
            "laya_package": LAYA_PACKAGE,
            "base_model": str(base_model),
            "base_model_archive": str(archive),
            "device": device,
            "perf_patch_enabled": perf_patch_enabled(),
            "ddp": distributed,
            "world_size": dist_env()[2],
            "gpu_usage": summarize_gpu_usage(WORKING / "gpu_usage.log"),
            "recipe": FINETUNE_CONFIG,
            "control": FINETUNE_CONTROL,
            "output_dir": str(WORKING / "checkpoint"),
            "corpus_sha256": {TRAIN_JSONL: sha256_of(train),
                              DEV_JSONL: sha256_of(dev),
                              TEST_JSONL: sha256_of(test)},
        }
        if isinstance(summary, dict):
            for key in ("train_items", "calibration_items", "eval_items",
                        "temperature", "epoch_loss"):
                if key in summary:
                    receipt[key] = summary[key]
        control_result = globals().get("FINETUNE_CONTROL_RESULT")
        if isinstance(control_result, dict):
            receipt["control_result"] = control_result
        report = WORKING / "checkpoint" / "train_report.json"
        if report.is_file():
            receipt["train_report"] = json.loads(report.read_text())
        # DEFAULT-ON held-out validation (owner order): every fine-tune also
        # scores the just-trained checkpoint on the held-out `test` split, so
        # the receipt always carries the honest generalization number. Opt out
        # with ER_LAYA_HELD_OUT=0; a failure never discards the checkpoint.
        if os.environ.get("ER_LAYA_HELD_OUT", "1").strip().lower() not in (
                "0", "false", "off", "no"):
            try:
                held_out = evaluate_held_out(test, WORKING / "checkpoint",
                                             device)
                (WORKING / "checkpoint" / "held_out_report.json").write_text(
                    json.dumps(held_out, indent=2) + "\\n", encoding="utf-8")
                receipt["held_out"] = held_out
                log("held-out " + held_out["eval_source"] + " items="
                    + str(held_out["items"]) + " accuracy="
                    + str(held_out["metrics"].get("accuracy")))
            except Exception as error:  # keep the checkpoint; surface failure
                receipt["held_out_error"] = (
                    type(error).__name__ + ": " + str(error)[:400])
                log("held-out evaluation FAILED: " + receipt["held_out_error"])
        # Mirror the run to wandb (rank 0 only; no-op without WANDB_API_KEY).
        if isinstance(receipt.get("train_report"), dict):
            wandb_log_metrics(receipt["train_report"])
        if isinstance(receipt.get("held_out"), dict):
            wandb_log_metrics({"after": receipt["held_out"].get("metrics", {})})
        wandb_finish()
        (WORKING / RECEIPT_NAME).write_text(
            json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
        with tarfile.open(WORKING / "laya_finetune.tar.gz", "w:gz",
                          compresslevel=1) as tar:
            for item in sorted(WORKING.iterdir()):
                if item.name not in ("laya_finetune.tar.gz", "base_model"):
                    tar.add(item, arcname=item.name)
        log("staged laya_finetune.tar.gz + receipt in /kaggle/working")

    # Barrier + rank-0-only gate: only rank 0 evaluates the held-out split,
    # writes the receipt and tars /kaggle/working; every rank then tears the
    # process group down.
    run_on_rank0(rank0_work)


def finetune_worker(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    init_distributed()
    try:
        _finetune_session(distributed=True, session=session_env())
    finally:
        destroy_if_distributed()


def main():
    session = session_env()
    pip_install_laya()
    if launch_finetune(finetune_worker) > 1:
        return
    _finetune_session(distributed=False, session=session)


if __name__ == "__main__":
    main()
'''


FINETUNE_EVAL_KERNEL_SCRIPT = '''\
"""ER laya fine-tune EVAL-ONLY on a Kaggle GPU session (cli.laya_lane).

Single T4 per owner ruling: pins one CUDA device, installs laya over pip
(pinned), reads the attached corpus HELD-OUT split + the attached fine-tuned
checkpoint dataset, loads the checkpoint (`laya.train.load_checkpoint`), runs
`calibration_records` + `evaluate_records` on the held-out split, and writes
eval_report.json (before vs after temperature calibration;
eval_mode=held_out, is_held_out=true) + a receipt into /kaggle/working for
hash-verified fetch-back. NO training, NO Hub.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LAYA_PACKAGE = "@LAYA_PACKAGE@"
RUN_TAG = "@RUN_TAG@"
EVAL_JSONL = "@EVAL_JSONL@"
EVAL_SPLIT = "@EVAL_SPLIT@"
CKPT_DIR_HINT = "@CKPT_DIR@"
CHECKPOINT_PATH = "@CHECKPOINT_PATH@"
BATCH_SIZE = @BATCH_SIZE@
# The YAML-driven calibration/abstention selection (`laya.eval_calibration`):
# `temperature` fits laya's per-type temperature map (on by default), the
# opt-in `abstention` fits the per-bucket `min_confidence` gate, and
# `min_confidence` pins the runtime scalar. Baked as ONE repr literal.
EVAL_CALIBRATION = @EVAL_CALIBRATION@

REPOSITORY = "@REPOSITORY@"
BRANCH = "@BRANCH@"
REVISION = "@REVISION@"
@RUNTIME_PREFLIGHT@

WORKING = Path("/kaggle/working")
INPUTS = Path("/kaggle/input")


def log(line):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[laya-lane " + stamp + "] " + line, flush=True)


def pip_install_laya():
    """laya installs over pip, pinned; torch is already on the session."""
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-input",
               "--disable-pip-version-check", LAYA_PACKAGE]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def pick_device():
    """SINGLE T4 ruling: pin the FIRST cuda device only (never 2xT4)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("cuda unavailable: the session is not a T4")
    log("device pinned: " + torch.cuda.get_device_name(0)
        + " (single GPU, never a second one)")
    return "cuda"


def resolve_input(name):
    for candidate in sorted(INPUTS.rglob(name)):
        return candidate
    raise FileNotFoundError(
        "attached inputs carried no " + name + " (expected the staged "
        "laya eval corpus dataset)")


def resolve_checkpoint():
    """The fine-tuned checkpoint dir: an explicit CHECKPOINT_PATH when baked,
    else the CKPT_DIR_HINT match, else the rl_agent_config.json rglob under
    /kaggle/input. Never a Hub fetch."""
    if CHECKPOINT_PATH:
        candidate = Path(CHECKPOINT_PATH)
        if candidate.is_dir() and (candidate / "rl_agent_config.json").is_file():
            return candidate
        raise FileNotFoundError(
            "CHECKPOINT_PATH carries no rl_agent_config.json: "
            + CHECKPOINT_PATH)
    if CKPT_DIR_HINT:
        for found in sorted(INPUTS.rglob(CKPT_DIR_HINT)):
            if (found.is_dir()
                    and (found / "rl_agent_config.json").is_file()):
                return found
    for found in sorted(INPUTS.rglob("rl_agent_config.json")):
        return found.parent
    raise FileNotFoundError(
        "attached inputs carried no fine-tuned checkpoint "
        "(rl_agent_config.json); attach the checkpoint dataset")


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_eval_calibration(laya_train, records):
    """Fit laya's OWN calibration per the baked `EVAL_CALIBRATION` (SSOT).

    Consumes `fit_temperature_map` (per-type temperature sequence + the
    per-bucket map) and, opt-in, `fit_abstention_thresholds` (the per-bucket
    `min_confidence` gate); nothing is reimplemented here. The kernel sets
    `default` on the thresholds map only when `min_confidence` is pinned,
    because an explicit operator pin beats the fit's implied default.
    """
    level = EVAL_CALIBRATION or {}
    temperature = temperature_by_options = n_by_bucket = None
    if level.get("temperature", True):
        fitted = laya_train.fit_temperature_map(records)
        temperature = fitted.get("temperature")
        temperature_by_options = fitted.get("temperature_by_options")
        n_by_bucket = fitted.get("n_by_bucket")
    thresholds = {}
    if level.get("abstention"):
        thresholds = dict(laya_train.fit_abstention_thresholds(
            records, temperature, temperature_by_options or {},
            target_error=level.get("target_error", 0.10),
            min_bucket_n=level.get("min_abstain_n", 10)) or {})
    min_confidence = level.get("min_confidence")
    if min_confidence is not None:
        thresholds["default"] = min_confidence
    return {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
        "abstention_thresholds": thresholds,
        "min_confidence": min_confidence,
    }


def main():
    pip_install_laya()
    device = pick_device()
    import torch
    from laya import train as laya_train
    eval_path = resolve_input(EVAL_JSONL)
    checkpoint = resolve_checkpoint()
    log("checkpoint: " + str(checkpoint))
    log("held-out split: " + str(eval_path) + " (" + EVAL_SPLIT + ")")
    model, tok, cfg = laya_train.load_checkpoint(str(checkpoint))
    model = model.to(torch.device(device)).eval()
    max_len = int(cfg.get("max_len", 512))
    head_max_len = int(cfg.get("head_max_len", 192))
    parallel = laya_train.uses_parallel_layout(cfg)
    rows = laya_train.read_jsonl(str(eval_path))
    items, skipped = laya_train.items_from_rows(
        tok, rows, max_len, head_max_len, label_smoothing=0.0)
    if not items:
        raise SystemExit(
            "eval split " + eval_path.name + " produced no usable items "
            "(skipped: " + repr(skipped) + ")")
    log("held-out rows " + str(len(rows)) + " -> items " + str(len(items)))
    records = laya_train.calibration_records(
        model, tok, items, device, max_len, head_max_len,
        batch_size=BATCH_SIZE, parallel=parallel)
    before = laya_train.evaluate_records(records)
    calibration = fit_eval_calibration(laya_train, records)
    after = laya_train.evaluate_records(
        records, calibration["temperature"],
        calibration["temperature_by_options"])
    comparison = {
        "delta_accuracy": round(
            after["accuracy"] - before["accuracy"], 4),
        "delta_ece": (round(after["ece"] - before["ece"], 4)
                      if after["ece"] is not None
                      and before["ece"] is not None else None),
        "delta_brier": (round(after["brier"] - before["brier"], 4)
                        if after["brier"] is not None
                        and before["brier"] is not None else None),
        "delta_mean_confidence": round(
            after["mean_confidence"] - before["mean_confidence"], 4),
    }
    report = {
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_source": eval_path.name,
        "eval_split": EVAL_SPLIT,
        "rows": len(rows),
        "items": len(items),
        "skipped": skipped,
        "checkpoint": str(checkpoint),
        "run_tag": RUN_TAG,
        "before": before,
        "after": after,
        "comparison": comparison,
        "temperature": calibration["temperature"],
        "temperature_by_options": calibration["temperature_by_options"],
    }
    # Additive: the opt-in knobs alone add keys, so a default config keeps the
    # landed report shape byte-for-byte.
    if EVAL_CALIBRATION.get("abstention") or calibration["min_confidence"] is not None:
        report["n_by_bucket"] = calibration["n_by_bucket"] or {}
    if calibration["abstention_thresholds"]:
        report["abstention_thresholds"] = calibration["abstention_thresholds"]
    if calibration["min_confidence"] is not None:
        report["min_confidence"] = calibration["min_confidence"]
    WORKING.mkdir(parents=True, exist_ok=True)
    report_path = WORKING / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\\n",
                           encoding="utf-8")
    log("wrote " + str(report_path) + " (accuracy before/after "
        + str(before["accuracy"]) + "/" + str(after["accuracy"]) + ")")
    receipt = {
        "gpu_kind": "finetune-eval",
        "gpu": "T4 (single)",
        "run_tag": RUN_TAG,
        "laya_package": LAYA_PACKAGE,
        "eval_split": EVAL_SPLIT,
        "eval_mode": "held_out",
        "is_held_out": True,
        "eval_jsonl_sha256": sha256_of(eval_path),
        "checkpoint": str(checkpoint),
        "eval_calibration": EVAL_CALIBRATION,
        "report_sha256": sha256_of(report_path),
    }
    (WORKING / "laya_finetune-eval.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
    with tarfile.open(WORKING / "laya_finetune_eval.tar.gz", "w:gz",
                      compresslevel=1) as tar:
        for item in sorted(WORKING.iterdir()):
            if item.name != "laya_finetune_eval.tar.gz":
                tar.add(item, arcname=item.name)
    log("staged eval_report.json + receipt in /kaggle/working")


if __name__ == "__main__":
    main()
'''
