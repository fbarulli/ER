"""colab_backend.py — run EuromonitoR TRAIN work on a Colab GPU VM.

The Colab CLI (google-colab-cli) provisions a Colab runtime (T4 default —
free-tier GPU, enough for sentence-transformer fine-tuning), pushes code +
data, executes a lane, and pulls results back.

Lanes (post second-series rename — the old second03/second04 scripts are
now the src/euromonitor/training/ module chain):
  train  — full-chain GPU training: data_prep -> train.py (contrastive,
           OnlineContrastiveLoss, holdout 50/25/25). The production run:
           the CPU lane proved the chain but 15s/step * 740 steps is 3h;
           the T4 does ~1.5-2s/step.
  hpo    — masking-enabled Optuna TPE search. Each trial trains on 50%,
           selects on the dev 25%, and does not read the test 25%.
  sims   — the deberta zero-shot lane (GPU-only: 3.9s/text on CPU — the
           CPU lane leaves its column absent by design, see
           src/euromonitor/training/zero_shot_sims.py). Scores with --models deberta_v3_base
           against the same canonical fingerprint contract.
  smoke  — the 1k chain check on GPU (fast verification the remote
           environment reproduces the local results contract).

Every lane reuses the shared bootstrap: upload the full code tree (src/euromonitor/training/,
lib/, data_pipe.py, config files) + the raw export; regenerate
all derived CSVs on the VM (byte-deterministic: canonicals/gates reproduce
identically — verified in the local worktree replay); run the lane; pull
the results CSVs + per-model stamps back.

Usage:
  python colab_backend.py --what train
  python colab_backend.py --what train --train-frac 0.25 --epochs 2
  python colab_backend.py --what hpo
  er-colab --what hpo --gpu A100
  python colab_backend.py --what sims
  python colab_backend.py --what smoke
  python colab_backend.py --what stop
  python ... --keep-alive   # keep VM alive for debugging on failure
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
# AUDIT FIX (round 2 F15, round 3): RESULTS/DATA come from the config SSOT
# via lib.common (00_config.yaml paths.results_dir/data_dir) — were
# re-derived inline (HERE / "artifacts" / "results"), a second declaration
# that happened to match today.
from euromonitor.core.common import DATA_DIR, RESULTS, load_config, sweep_cfg, training_cfg

DATA = DATA_DIR

# smoke sample size + train defaults: the config SSOT (src/euromonitor/training/training.yaml
# sweep: block via lib.common.sweep_cfg / training_cfg) — were inline
# literals (1000 / 0.25 / 2) that could silently diverge from the configs.
# AUDIT FIX (round 2 F07, round 3): the train-frac default reads
# sweep.train_fracs[0] — the 0.25 literal was the last one still inline.
_SMOKE_SAMPLE = int(sweep_cfg()["smoke_sample"])
_TRAIN_FRAC_DEFAULT = float(sweep_cfg()["train_fracs"][0])
_EPOCHS_DEFAULT = int(training_cfg().training.epochs)
_RERANK_MODEL = str(sweep_cfg()["rerank_model"])

SESSION = "EuromonitoR"
GPU = "T4"
REMOTE_ROOT = "/content/EuromonitoR"
LIVE_LOG_PATH = HERE / "training.log"
_live_log = None

# code tree every lane needs (the TRAIN chain imports lib.* and data_pipe)
# NOTE (config split 2026-09-08, EDA removed 2026-09-10): the monolith
# became 00_config.yaml (root data contract) + src/euromonitor/training/training.yaml (the
# EDA dir is gone — its TRAIN-consumed keys migrated into training.yaml);
# the root stopwords moved to lib/pipe_stopwords.json (matching.py's
# sklearn list renamed to lib/sklearn_stopwords.json) — whole-dir uploads
# carry every config file.
CODE_TARGETS = [
    (HERE / "src", f"{REMOTE_ROOT}/src"),
    (HERE / "00_config.yaml", f"{REMOTE_ROOT}/00_config.yaml"),
]
# derived-data lanes regenerate ON the VM (byte-deterministic) — the ONLY
# uploads are committed SSOT inputs. AUDIT FIX 2026-09-08: the old set
# shipped dataset_deduped.csv alone, but data_prep's chain reads
# artifacts/data/dataset.csv (raw export) and dedupe/build_reference
# regenerate from it; shipping the deduped file without the raw export
# meant the VM lane died at the first loader. number_tokens_reference.csv
# is a COMMITTED input (build_reference --verify reproduces it; without it
# strip_number_tokens degrades to regex-only and the payload drifts).
_RAW_EXPORT = (DATA / load_config()["files"]["dataset"]).resolve()
INPUT_TARGETS = [
    # The source is explicitly configured relative to DATA. Preserve the same
    # resolution remotely rather than create a shadow dataset copy.
    (_RAW_EXPORT, f"{REMOTE_ROOT}/dataset.csv"),
    (DATA / "number_tokens_reference.csv", f"{REMOTE_ROOT}/artifacts/data/number_tokens_reference.csv"),
]


def check_colab_cli() -> None:
    """Ensure the colab CLI is installed and authenticated."""
    try:
        subprocess.run(["colab", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "colab CLI not found or not authenticated.\n"
            "Run: uv tool install google-colab-cli\n"
            "Then: colab sessions  (to complete OAuth sign-in)"
        )


def colab(*args: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a colab CLI subcommand."""
    cmd = ["colab", *args]
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        print(f"\n[error] colab command failed: {' '.join(cmd)}", file=sys.stderr)
        if e.stdout:
            print(f"stdout:\n{e.stdout[-1000:]}", file=sys.stderr)
        if e.stderr:
            print(f"stderr:\n{e.stderr[-1000:]}", file=sys.stderr)
        raise


def run_colab_exec_stream(session: str, script: str, timeout: int | None = None, log_name: str | None = None) -> None:
    """Execute a python script on the colab session via stdin, streaming stdout/stderr.

    log_name labels a stage in the single root training.log. The file is
    opened once in write mode per invocation, line-flushed, and survives VM
    teardown so every Colab stage is inspectable in one chronological log.
    """
    log_file = _live_log
    if log_file and log_name:
        log_file.write(f"\n===== {log_name} =====\n")
        log_file.flush()

    def stream_output(pipe, prefix):
        for line in iter(pipe.readline, ''):
            print(f"{prefix} {line.rstrip()}")
            if log_file:
                log_file.write(f"{prefix} {line}")
                log_file.flush()
        pipe.close()

    process = subprocess.Popen(
        # colab exec has its own 30-second kernel-client timeout.  It must
        # match the caller's legitimate lane timeout; otherwise a live VM
        # computation is reported as failed after 30 seconds.
        ["colab", "exec", "-s", session, "--timeout", str(timeout or 30)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    out_thread = threading.Thread(target=stream_output, args=(process.stdout, "[out]"))
    err_thread = threading.Thread(target=stream_output, args=(process.stderr, "[err]"))
    out_thread.start()
    err_thread.start()

    process.stdin.write(script)
    process.stdin.close()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\n[error] Execution timed out after {timeout}s", file=sys.stderr)
        raise

    out_thread.join()
    err_thread.join()
    if process.returncode != 0:
        raise SystemExit(f"Remote execution failed with return code {process.returncode}")


def start_live_log() -> None:
    """Start a fresh single-file log for one Colab invocation."""
    global _live_log
    if _live_log is not None:
        _live_log.close()
    _live_log = LIVE_LOG_PATH.open("w", encoding="utf-8")


def close_live_log() -> None:
    global _live_log
    if _live_log is not None:
        _live_log.close()
        _live_log = None


def ensure_session() -> None:
    """Provision the session if it does not already exist."""
    r = colab("sessions", check=False)
    if SESSION in (r.stdout or ""):
        print(f"[session] '{SESSION}' already active")
        return
    print(f"[session] provisioning {SESSION} (gpu={GPU}) ...")
    colab("new", "-s", SESSION, "--gpu", GPU, timeout=300)
    print("[session] up")


def _upload_dir(local_dir: Path, remote_dir: str) -> None:
    """Upload a directory tree via per-file colab upload (no tar on VM)."""
    files = sorted(p for p in local_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for f in files:
        rel = f.relative_to(local_dir).as_posix()
        colab("upload", "-s", SESSION, str(f), f"{remote_dir}/{rel}", timeout=600)
    print(f"[upload] {local_dir.name}/ -> {remote_dir} ({len(files)} files)")


def upload_inputs() -> None:
    """Ship code tree + SSOT inputs to the VM."""
    for local_dir, remote_dir in [(t[0], t[1]) for t in CODE_TARGETS if t[0].is_dir()]:
        _upload_dir(local_dir, remote_dir)
    for local, remote in CODE_TARGETS + INPUT_TARGETS:
        if local.is_dir():
            continue
        if not local.exists():
            raise FileNotFoundError(f"Local file not found: {local}")
        print(f"[upload] {local.name} -> {remote}")
        colab("upload", "-s", SESSION, str(local), remote, timeout=600)


def prepare_remote_layout() -> None:
    """Create upload parents before the first file transfer.

    Colab's contents API does not consistently create nested parents on an
    upload request.  Make the layout explicitly, so a 500 cannot masquerade
    as a failed data or training operation.
    """
    paths = [
        REMOTE_ROOT,
        f"{REMOTE_ROOT}/src/euromonitor/training",
        f"{REMOTE_ROOT}/src/euromonitor/core",
        f"{REMOTE_ROOT}/src/euromonitor/ner",
        f"{REMOTE_ROOT}/artifacts/data",
        f"{REMOTE_ROOT}/artifacts/results",
    ]
    script = "import pathlib\n" + "\n".join(
        f"pathlib.Path({path!r}).mkdir(parents=True, exist_ok=True)" for path in paths
    ) + "\nprint('remote upload layout ready')\n"
    run_colab_exec_stream(SESSION, script, timeout=120)


def install_deps() -> None:
    print("[deps] installing dependencies on the VM ...")
    # pip via sys.executable is guaranteed on a Colab VM (uv is NOT installed
    # there by default); streaming shows install progress live.
    # datasets + accelerate + transformers for the HF Trainer-based training
    # lane; sentence-transformers pins its own transformers requirement.
    install_script = (
        "import sys, subprocess\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "                'sentence-transformers', 'datasets', 'accelerate',\n"
        "                'evaluate', 'scikit-learn', 'pandas', 'numpy',\n"
        "                'mlflow', 'optuna', 'wandb'], check=True)\n"
        "print('deps installed')"
    )
    run_colab_exec_stream(SESSION, install_script, timeout=900, log_name="00_deps")


def log_gpu_profile() -> None:
    """Record the exact accelerator and memory budget before training."""
    script = """import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: refusing a CPU HPO run')
p = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print({'name': p.name, 'total_gb': round(total / 1e9, 2), 'free_gb': round(free / 1e9, 2), 'torch': torch.__version__}, flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=120, log_name="gpu_profile")


_BOOTSTRAP = f"""
import sys, runpy, pathlib, os
sys.path.insert(0, "{REMOTE_ROOT}/src")
os.environ["PYTHONPATH"] = "{REMOTE_ROOT}/src" + os.pathsep + os.environ.get("PYTHONPATH", "")
(pathlib.Path("{REMOTE_ROOT}/results")).mkdir(parents=True, exist_ok=True)
(pathlib.Path("{REMOTE_ROOT}/artifacts/data")).mkdir(parents=True, exist_ok=True)
"""


def _env_value(name: str) -> str | None:
    """Read a simple KEY=VALUE entry without printing or uploading .env."""
    env_path = HERE / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _wandb_env_script() -> str:
    """Inject only the API key into the remote process, never remote disk."""
    key = _env_value("WANDB_API_KEY")
    if not key:
        print("[wandb] WANDB_API_KEY absent from .env; run will remain local-only")
        return ""
    print("[wandb] API key loaded from local .env and injected into VM process")
    return f"os.environ['WANDB_API_KEY'] = {key!r}\\n"


def run_data_prep() -> None:
    """Regenerate the derived CSVs on the VM (byte-deterministic replay).

    AUDIT FIX 2026-09-08: 'data_prep only' assumed derived inputs that are
    NOT derived on the VM — the chain is dedupe -> build_reference --verify
    -> data_prep. Running all three keeps the VM replay identical to the
    local worktree replay (byte-comparable outputs).
    """
    print("[run] dedupe + reference-verify + data_prep on the VM ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
for step in ("src/euromonitor/training/dedupe.py", "src/euromonitor/training/build_reference.py --verify", "src/euromonitor/training/data_prep.py"):
    print("== " + step, flush=True)
    rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/" + step.split()[0]] + step.split()[1:]).returncode
    if rc != 0:
        sys.exit(rc)
"""
    # dedupe 1-2 min + reference verify ~3 min + data_prep ~2 min
    run_colab_exec_stream(SESSION, script, timeout=1800, log_name="01_data_prep")


def run_train(frac: float, epochs: int, sample: int | None) -> None:
    """Full-chain GPU training on the VM."""
    print("[run] train.py on the VM (GPU) ...")
    extra = f" --sample {sample}" if sample else ""
    # AUDIT 2026-09-09: --mask-frac 0.15 REMOVED — it hardcoded a value that
    # silently contradicted the SSOT (masking.frac: 1.00 in
    # src/euromonitor/training/training.yaml). train.py's own default resolves from the config
    # now; the CLI flag remains for explicit overrides.
    script = _BOOTSTRAP + _wandb_env_script() + f"""
import subprocess, sys
rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/src/euromonitor/training/train.py",
                     "--split", "holdout",
                     "--loss", "contrastive",
                     "--train-frac", "{frac}",
                     "--epochs", "{epochs}",
                     "--no-plot"{extra}]).returncode
sys.exit(rc)
"""
    # T4 full chain: encode ~1min + 740 steps at ~1.5-2s + eval — allow 4h
    run_colab_exec_stream(SESSION, script, timeout=4 * 3600, log_name="02_train")


def run_hpo() -> None:
    """Search, replay the winner once, then run post-training reranking."""
    print("[run] HPO -> selected final train/test -> post-training rerank on the VM ...")
    script = _BOOTSTRAP + _wandb_env_script() + f"""
import json, pathlib, subprocess, sys
root = pathlib.Path("{REMOTE_ROOT}")
train = root / "src/euromonitor/training/train.py"
base = [sys.executable, str(train), "--split", "holdout", "--loss", "contrastive", "--payload", "full", "--no-plot"]
print("== HPO (dev-selected; test withheld)", flush=True)
if subprocess.run(base + ["--hpo"]).returncode:
    sys.exit(1)
best_paths = sorted((root / "artifacts/results").glob("train_*_hpo_best.json"))
if len(best_paths) != 1:
    raise RuntimeError(f"expected one HPO best-config artifact, found {{best_paths}}")
best = json.loads(best_paths[0].read_text())
params = best["config"]
required = {{"epochs", "lr", "warmup_ratio", "weight_decay"}}
missing = required - set(params)
if missing:
    raise RuntimeError(f"HPO best config lacks {{sorted(missing)}}")
final = base + [
    "--epochs", str(params["epochs"]),
    "--lr", str(params["lr"]),
    "--warmup-ratio", str(params["warmup_ratio"]),
    "--weight-decay", str(params["weight_decay"]),
    "--rerank", "{_RERANK_MODEL}",
]
print("== FINAL selected configuration (test once) + rerank verification", flush=True)
print(json.dumps({{"selected_hpo_config": params, "rerank_model": "{_RERANK_MODEL}"}}, sort_keys=True), flush=True)
sys.exit(subprocess.run(final).returncode)
"""
    run_colab_exec_stream(SESSION, script, timeout=8 * 3600, log_name="training_hpo")


def run_sims_deberta() -> None:
    """The deberta zero-shot lane (GPU-only) on the VM."""
    print("[run] zero_shot_sims --models deberta_v3_base on the VM (GPU) ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/src/euromonitor/training/zero_shot_sims.py",
                     "--models", "deberta_v3_base"]).returncode
sys.exit(rc)
"""
    run_colab_exec_stream(SESSION, script, timeout=2 * 3600, log_name="03_sims_deberta")


def _list_remote(pattern_dir: str) -> list[str]:
    """List remote files via a stdin-exec (same channel the lanes use)."""
    import json as _json

    script = (
        "import pathlib, json\n"
        f"files = sorted(str(p) for p in pathlib.Path('{pattern_dir}').rglob('*') if p.is_file())\n"
        "print('@@FILES@@' + json.dumps(files))\n"
    )
    proc = subprocess.Popen(
        ["colab", "exec", "-s", SESSION],
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


def download_results(skip_checkpoints: bool = True) -> None:
    """Pull the result artifacts back to the repo results dir.

    AUDIT FIX 2026-09-08: the generic rglob included _checkpoints (~1.9 GB
    of model weights) for EVERY lane — checkpoints are pulled explicitly by
    download_checkpoints() only when --what train asks for them.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = _list_remote(f"{REMOTE_ROOT}/artifacts/results")
    for name in files:
        rel = Path(name).relative_to(f"{REMOTE_ROOT}/artifacts/results")
        if skip_checkpoints and rel.parts[0] == "_checkpoints":
            continue
        local = RESULTS / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {rel}")
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
                f"[error] results download failed for {rel} — local copy "
                f"at {local} is absent/partial; refusing to continue "
                f"because teardown would delete the only remote copy "
                f"(re-run the lane, or re-pull from a --keep-alive VM)",
                file=sys.stderr,
            )
            raise


def download_checkpoints() -> None:
    """Pull the trained checkpoints (model weights) back.

    Called after --what train: the trained model IS the deliverable of the
    production run; results CSVs alone don't carry it.
    """
    print("[download] checkpoints ...")
    files = _list_remote(f"{REMOTE_ROOT}/artifacts/results/_checkpoints")
    for name in files:
        rel = Path(name).relative_to(f"{REMOTE_ROOT}/artifacts")
        local = HERE / "artifacts" / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {rel}")
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
                f"[error] checkpoint download failed for {rel} — the "
                f"trained weights were NOT pulled local; continuing would "
                f"let stop() destroy the only copy. Re-run the lane or "
                f"re-pull manually before the VM is gone (--keep-alive "
                f"keeps it up)",
                file=sys.stderr,
            )
            raise


def stop() -> None:
    print(f"[stop] tearing down '{SESSION}'")
    # RULING 2026-09-10 (silent-degradation audit): JUSTIFIED-KEEP.
    # stop() runs in main()'s finally — if the lane itself raised, the
    # lane's exception is the root cause and must stay the error the
    # operator sees; raising here would MASK it with a teardown failure
    # and abort nothing (no local data or pipeline state depends on the
    # VM being gone). But it must not be silent: an unreleased VM burns
    # Colab GPU quota until manually reaped, so warn loudly with the
    # consequence + the exact recovery command.
    try:
        colab("stop", "-s", SESSION, check=False)
    except subprocess.SubprocessError as exc:
        print(
            f"[warn] VM release request failed — the VM '{SESSION}' may "
            f"STILL BE LIVE and burning Colab GPU quota until it times "
            f"out or is reaped. After handling the failure above, reclaim "
            f"it with: colab stop -s {SESSION}   (or 'colab sessions' "
            f"to check). Original error: {exc}",
            file=sys.stderr,
        )
    print("[stop] VM release requested")


def main() -> None:
    global GPU
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", required=True,
                    choices=["train", "hpo", "sims", "smoke", "stop"],
                    help="what to run on the VM")
    ap.add_argument("--train-frac", type=float, default=_TRAIN_FRAC_DEFAULT,
                    help=f"train fraction for --what train (default "
                    f"{_TRAIN_FRAC_DEFAULT:g})")
    ap.add_argument("--epochs", type=int, default=_EPOCHS_DEFAULT,
                    help=f"epochs for --what train (default {_EPOCHS_DEFAULT} = "
                    "src/euromonitor/training/training.yaml training.epochs)")
    ap.add_argument(
        "--gpu",
        default=GPU,
        help=f"Colab accelerator request (default {GPU}; e.g. A100 when available)",
    )
    ap.add_argument("--keep-alive", action="store_true",
                    help="do not tear down the VM on completion/failure")
    args = ap.parse_args()

    GPU = args.gpu

    if args.what == "stop":
        stop()
        return

    start_live_log()
    check_colab_cli()

    # Pre-flight check — gate on the files the lane actually UPLOADS
    # (INPUT_TARGETS), not dataset_deduped.csv: that file is regenerated on
    # the VM by run_data_prep (never uploaded), so requiring it locally was
    # both unnecessary and incomplete — a stale local copy passing the
    # gate while the real inputs (raw export, reference) were missing.
    for local, _ in INPUT_TARGETS:
        if not local.exists():
            raise FileNotFoundError(f"Missing required input: {local}")

    try:
        ensure_session()
        prepare_remote_layout()
        upload_inputs()
        install_deps()
        log_gpu_profile()
        # AUDIT FIX 2026-09-08: --what sims used to run FULL TRAINING first
        # (run_train was unconditional) — hours of unintended GPU quota
        # for a lane that only needs the deberta scoring.
        if args.what == "sims":
            run_data_prep()
            run_sims_deberta()
        elif args.what == "smoke":
            run_data_prep()
            run_train(args.train_frac, args.epochs, sample=_SMOKE_SAMPLE)
        elif args.what == "hpo":
            run_data_prep()
            run_hpo()
        else:
            run_data_prep()
            run_train(args.train_frac, args.epochs, sample=None)
        download_results()
        if args.what == "train":
            download_checkpoints()
    finally:
        # Default behavior is to aggressively teardown to prevent quota burning.
        if not args.keep_alive:
            stop()
        else:
            print("\n[info] --keep-alive specified. VM is still running.")
        close_live_log()

    print(f"\n[done] artifacts saved to {RESULTS}")


if __name__ == "__main__":
    main()
