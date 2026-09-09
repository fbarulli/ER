"""colab_backend.py — run EuromonitoR TRAIN work on a Colab GPU VM.

The Colab CLI (google-colab-cli) provisions a Colab runtime (T4 default —
free-tier GPU, enough for sentence-transformer fine-tuning), pushes code +
data, executes a lane, and pulls results back.

Lanes (post second-series rename — the old second03/second04 scripts are
now the TRAIN/ module chain):
  train  — full-chain GPU training: data_prep -> train.py (contrastive,
           OnlineContrastiveLoss, holdout 50/25/25). The production run:
           the CPU lane proved the chain but 15s/step * 740 steps is 3h;
           the T4 does ~1.5-2s/step.
  sims   — the deberta zero-shot lane (GPU-only: 3.9s/text on CPU — the
           CPU lane leaves its column absent by design, see
           TRAIN/zero_shot_sims.py). Scores with --models deberta_v3_base
           against the same canonical fingerprint contract.
  smoke  — the 1k chain check on GPU (fast verification the remote
           environment reproduces the local results contract).

Every lane reuses the shared bootstrap: upload the full code tree (TRAIN/,
lib/, data_pipe.py, config files) + the raw export; regenerate
all derived CSVs on the VM (byte-deterministic: canonicals/gates reproduce
identically — verified in the local worktree replay); run the lane; pull
the results CSVs + per-model stamps back.

Usage:
  python colab_backend.py --what train
  python colab_backend.py --what train --train-frac 0.25 --epochs 2
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

HERE = Path(__file__).resolve().parent
# AUDIT FIX (round 2 F15, round 3): RESULTS/DATA come from the config SSOT
# via lib.common (00_config.yaml paths.results_dir/data_dir) — were
# re-derived inline (HERE / "artifacts" / "results"), a second declaration
# that happened to match today.
sys.path.insert(0, str(HERE))
from lib.common import DATA_DIR, RESULTS, sweep_cfg, training_cfg

DATA = DATA_DIR

# smoke sample size + train defaults: the config SSOT (TRAIN/training.yaml
# sweep: block via lib.common.sweep_cfg / training_cfg) — were inline
# literals (1000 / 0.25 / 2) that could silently diverge from the configs.
# AUDIT FIX (round 2 F07, round 3): the train-frac default reads
# sweep.train_fracs[0] — the 0.25 literal was the last one still inline.
_SMOKE_SAMPLE = int(sweep_cfg()["smoke_sample"])
_TRAIN_FRAC_DEFAULT = float(sweep_cfg()["train_fracs"][0])
_EPOCHS_DEFAULT = int(training_cfg().training.epochs)

SESSION = "EuromonitoR"
GPU = "T4"
REMOTE_ROOT = "/content/EuromonitoR"

# code tree every lane needs (the TRAIN chain imports lib.* and data_pipe)
# NOTE (config split 2026-09-08, EDA removed 2026-09-10): the monolith
# became 00_config.yaml (root data contract) + TRAIN/training.yaml (the
# EDA dir is gone — its TRAIN-consumed keys migrated into training.yaml);
# the root stopwords moved to lib/pipe_stopwords.json (matching.py's
# sklearn list renamed to lib/sklearn_stopwords.json) — whole-dir uploads
# carry every config file.
CODE_TARGETS = [
    (HERE / "TRAIN", f"{REMOTE_ROOT}/TRAIN"),  # includes TRAIN/training.yaml
    (HERE / "lib", f"{REMOTE_ROOT}/lib"),  # word lists live here now
    (HERE / "data_pipe.py", f"{REMOTE_ROOT}/data_pipe.py"),
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
INPUT_TARGETS = [
    (DATA / "dataset.csv", f"{REMOTE_ROOT}/artifacts/data/dataset.csv"),
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

    log_name: when set, every streamed line is ALSO teed to
    artifacts/logs/colab/{log_name}.log — the owner directive: Colab-run
    logs must exist LOCALLY and be LIVE (readable while the VM works),
    not only in the streaming console. Appends across re-runs of the same
    lane; the file lives after the VM is torn down.
    """
    log_file = None
    if log_name:
        log_dir = HERE / "artifacts" / "logs" / "colab"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (log_dir / f"{log_name}.log").open("a", encoding="utf-8")

    def stream_output(pipe, prefix):
        for line in iter(pipe.readline, ''):
            print(f"{prefix} {line.rstrip()}")
            if log_file:
                log_file.write(f"{prefix} {line}")
                log_file.flush()
        pipe.close()

    process = subprocess.Popen(
        ["colab", "exec", "-s", session],
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
    if log_file:
        log_file.close()

    if process.returncode != 0:
        raise SystemExit(f"Remote execution failed with return code {process.returncode}")


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
        "                'mlflow'], check=True)\n"
        "print('deps installed')"
    )
    run_colab_exec_stream(SESSION, install_script, timeout=900, log_name="00_deps")


_BOOTSTRAP = f"""
import sys, runpy, pathlib
sys.path.insert(0, "{REMOTE_ROOT}")
(pathlib.Path("{REMOTE_ROOT}/artifacts/results")).mkdir(parents=True, exist_ok=True)
(pathlib.Path("{REMOTE_ROOT}/artifacts/data")).mkdir(parents=True, exist_ok=True)
"""


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
for step in ("TRAIN/dedupe.py", "TRAIN/build_reference.py --verify", "TRAIN/data_prep.py"):
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
    # TRAIN/training.yaml). train.py's own default resolves from the config
    # now; the CLI flag remains for explicit overrides.
    script = _BOOTSTRAP + f"""
import subprocess, sys
rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/TRAIN/train.py",
                     "--split", "holdout",
                     "--loss", "contrastive",
                     "--train-frac", "{frac}",
                     "--epochs", "{epochs}",
                     "--no-plot"{extra}]).returncode
sys.exit(rc)
"""
    # T4 full chain: encode ~1min + 740 steps at ~1.5-2s + eval — allow 4h
    run_colab_exec_stream(SESSION, script, timeout=4 * 3600, log_name="02_train")


def run_sims_deberta() -> None:
    """The deberta zero-shot lane (GPU-only) on the VM."""
    print("[run] zero_shot_sims --models deberta_v3_base on the VM (GPU) ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/TRAIN/zero_shot_sims.py",
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
        try:
            colab("download", "-s", SESSION, name, str(local), timeout=600)
        except subprocess.CalledProcessError:
            print(f"[warn] failed to download {rel}", file=sys.stderr)


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
        try:
            colab("download", "-s", SESSION, name, str(local), timeout=1200)
        except subprocess.CalledProcessError:
            print(f"[warn] failed to download {rel}", file=sys.stderr)


def stop() -> None:
    print(f"[stop] tearing down '{SESSION}'")
    try:
        colab("stop", "-s", SESSION, check=False)
    except subprocess.SubprocessError as exc:
        print(f"[warn] stop request failed (VM may still be live): {exc}", file=sys.stderr)
    print("[stop] VM release requested")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", required=True,
                    choices=["train", "sims", "smoke", "stop"],
                    help="what to run on the VM")
    ap.add_argument("--train-frac", type=float, default=_TRAIN_FRAC_DEFAULT,
                    help=f"train fraction for --what train (default "
                    f"{_TRAIN_FRAC_DEFAULT:g})")
    ap.add_argument("--epochs", type=int, default=_EPOCHS_DEFAULT,
                    help=f"epochs for --what train (default {_EPOCHS_DEFAULT} = "
                    "TRAIN/training.yaml training.epochs)")
    ap.add_argument("--keep-alive", action="store_true",
                    help="do not tear down the VM on completion/failure")
    args = ap.parse_args()

    if args.what == "stop":
        stop()
        return

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
        upload_inputs()
        install_deps()
        # AUDIT FIX 2026-09-08: --what sims used to run FULL TRAINING first
        # (run_train was unconditional) — hours of unintended GPU quota
        # for a lane that only needs the deberta scoring.
        if args.what == "sims":
            run_data_prep()
            run_sims_deberta()
        elif args.what == "smoke":
            run_data_prep()
            run_train(args.train_frac, args.epochs, sample=_SMOKE_SAMPLE)
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

    print(f"\n[done] artifacts saved to {RESULTS}")


if __name__ == "__main__":
    main()
