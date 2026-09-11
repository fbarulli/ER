"""Publish one worker's output through an isolated DVC project."""
from __future__ import annotations
import argparse, fcntl, json, os, shutil, subprocess, tempfile, time, traceback
from pathlib import Path
from core.common import training_cfg

def _process_snapshot(label: str) -> None:
    """Put the local process table in the live training/DVC log."""
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout
    for name in (
        "DVC_API_KEY",
        "DAGSHUB_USER_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "WANDB_API_KEY",
    ):
        secret = os.environ.get(name)
        if secret:
            output = output.replace(secret, "<redacted>")
    print(f"[processes:{label}]\n{output.rstrip()}", flush=True)


def _run(command: list[str], cwd: Path) -> None:
    shown = ["<redacted>" if command[i - 1:i] == ["password"] else part for i, part in enumerate(command)]
    print(f"[dvc] running: {' '.join(shown)}", flush=True)
    cfg = training_cfg().colab
    attempts = cfg.dvc_push_retries if command[:2] == ["dvc", "push"] else 1
    for attempt in range(1, attempts + 1):
        _process_snapshot(f"before dvc attempt {attempt}: {command[0]}")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"[processes] started pid={process.pid} command={command[0]}", flush=True)
            _process_snapshot(f"running dvc attempt {attempt}: pid={process.pid}")
            output_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line)
                print(f"[dvc-out] {line.rstrip()}", flush=True)
            process.wait()
            output = "".join(output_lines)
        except BaseException:
            print(
                f"[dvc] traceback while running attempt {attempt}: {' '.join(shown)}",
                flush=True,
            )
            traceback.print_exc()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, output)
        if result.stdout:
            print(result.stdout.rstrip(), flush=True)
        _process_snapshot(f"after dvc attempt {attempt}: rc={result.returncode}")
        if result.returncode == 0:
            return
        if attempt < attempts:
            delay = cfg.dvc_push_backoff_seconds * (2 ** (attempt - 1))
            print(f"[dvc] command failed; retry {attempt}/{attempts - 1} in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"DVC command failed ({result.returncode}): {' '.join(shown)}")


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_outputs(source: Path) -> list[Path]:
    import yaml
    outputs: list[Path] = []
    for pointer in sorted(source.glob("*.dvc")):
        if not pointer.is_file():
            continue
        data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        for entry in data.get("outs", []):
            path = source / str(entry["path"])
            if path.is_file():
                outputs.append(path)
    return outputs


def _verify_clean_pull(source: Path, token: str, remote: str) -> list[dict[str, str]]:
    """Pull into a clean directory; prove the remote is independently readable."""
    tracked = _tracked_outputs(source)
    if not tracked:
        raise RuntimeError("DVC push produced no tracked output pointers")
    with tempfile.TemporaryDirectory(prefix="euromonitor-dvc-verify-") as temp:
        verify = Path(temp)
        os.environ["DVC_SITE_CACHE_DIR"] = str(verify / ".dvc-site-cache")
        _run(["dvc", "init", "--no-scm"], verify)
        _run(["dvc", "config", "cache.dir", str(verify / ".dvc-cache")], verify)
        _run(["dvc", "remote", "add", "--default", "dagshub", remote], verify)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], verify)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "user", "fbarulli"], verify)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], verify)
        for pointer in source.glob("*.dvc"):
            if not pointer.is_file():
                continue
            shutil.copy2(pointer, verify / pointer.name)
        _run(["dvc", "pull", "--force"], verify)
        result = []
        for original in tracked:
            restored = verify / original.relative_to(source)
            if not restored.is_file():
                raise RuntimeError(f"DVC pull did not restore {original.name}")
            expected, actual = _sha256(original), _sha256(restored)
            if expected != actual:
                raise RuntimeError(f"DVC pull hash mismatch for {original.name}")
            result.append({"path": original.name, "sha256": actual})
        return result


def _configure(source: Path, token: str) -> str:
    """Configure an isolated, no-SCM DVC workspace for one worker."""
    os.environ["DVC_SITE_CACHE_DIR"] = str(source / ".dvc-site-cache")
    remote = training_cfg().colab.dvc_remote_url
    if not (source / ".dvc").is_dir():
        _run(["dvc", "init", "--no-scm"], source)
        _run(["dvc", "config", "cache.dir", str(source / ".dvc-cache")], source)
        _run(["dvc", "remote", "add", "--default", "dagshub", remote], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "user", "fbarulli"], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], source)
    return remote


def publish_checkpoint(source: Path, checkpoint_root: Path) -> Path:
    """DVC-push a complete Trainer checkpoint tree after a save event.

    The pointer lives beneath the worker directory and names the tree using a
    path relative to that directory.  Restore therefore uses ``dvc pull`` and
    never assumes a DVC cache layout or remote object URL.
    """
    token = os.environ.get("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required to persist a checkpoint")
    source = source.resolve()
    checkpoint_root = checkpoint_root.resolve()
    relative_root = checkpoint_root.relative_to(source)
    _configure(source, token)
    pointer = source / ".resume" / f"{checkpoint_root.name}.dvc"
    # DVC recursively discovers existing .dvc files.  The durable resume
    # pointers are intentionally kept under source/.resume, but they must not
    # participate in discovery while a new output is added.  Temporarily
    # stage that metadata outside the DVC source, then put it back even when
    # dvc add fails.
    staged_resume = None
    resume_dir = source / ".resume"
    if resume_dir.is_dir():
        staged_resume = Path(tempfile.mkdtemp(prefix=".resume-staging-", dir=source.parent))
        shutil.move(str(resume_dir), str(staged_resume / ".resume"))
    try:
        _run(["dvc", "add", str(relative_root)], source)
    finally:
        if staged_resume is not None:
            shutil.move(str(staged_resume / ".resume"), str(resume_dir))
            shutil.rmtree(staged_resume, ignore_errors=True)
    native_pointer = checkpoint_root.with_name(f"{checkpoint_root.name}.dvc")
    if not native_pointer.is_file():
        raise RuntimeError(f"DVC did not create checkpoint pointer: {native_pointer}")
    # DVC resolves an output path relative to the pointer file's directory.
    # The native pointer is beside the output, but the durable resume pointer
    # lives under ``.resume``.  Copying the YAML byte-for-byte would restore
    # into ``.resume/<name>`` instead of the checkpoint/database path.
    import yaml

    pointer_data = yaml.safe_load(native_pointer.read_text(encoding="utf-8")) or {}
    for entry in pointer_data.get("outs", []):
        native_output = (native_pointer.parent / str(entry["path"])).resolve()
        native_output.relative_to(source)
        entry["path"] = os.path.relpath(native_output, pointer.parent)
    native_relative = native_pointer.relative_to(source)
    lock_path = source.parent / ".dvc-push.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # This is a no-SCM repository and the checkpoint .dvc file is
            # nested below _checkpoints/.  A target-less dvc push can report
            # success while discovering no top-level tracked outputs, which
            # leaves the resume pointer referring to a nonexistent remote
            # object. Push the exact native pointer explicitly.
            print(f"[checkpoint-dvc] pushing target {native_relative}", flush=True)
            _run(["dvc", "push", "--jobs", "1", str(native_relative)], source)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # Expose the durable resume pointer only after the exact target push has
    # succeeded. The launcher mirrors this file while the worker is alive.
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        yaml.safe_dump(pointer_data, sort_keys=False), encoding="utf-8"
    )
    return pointer


def _pointer_outputs(source: Path, pointer: Path) -> list[Path]:
    """Return the output paths declared by a DVC pointer.

    Resume pointers are deliberately kept outside the normal DVC metadata
    tree (under ``.resume``), so DVC itself cannot infer the source directory
    from the pointer's location.  Reading the declared output paths gives the
    preflight code a way to verify that a pull restored the complete object.
    """
    import yaml

    source = source.resolve()
    pointer = pointer.resolve()
    pointer.relative_to(source)
    data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
    outputs: list[Path] = []
    for entry in data.get("outs", []):
        # DVC interprets outs.path relative to the pointer's directory, not
        # the repository root.  This matters because resume pointers are
        # deliberately stored under source/.resume/.
        path = (pointer.parent / str(entry["path"])).resolve()
        path.relative_to(source)
        outputs.append(path)
    if not outputs:
        raise RuntimeError(f"DVC resume pointer has no outputs: {pointer}")
    return outputs


def restore_pointer(source: Path, pointer: Path) -> list[Path]:
    """Pull and verify one previously published resume pointer.

    This is the fail-fast primitive used before a trainer subprocess starts.
    It intentionally verifies every declared output instead of merely
    checking that ``dvc pull`` returned zero: a successful DVC command can
    still leave an incomplete working tree when a pointer is stale or
    malformed.
    """
    try:
        token = os.environ.get("DVC_API_KEY")
        if not token:
            raise RuntimeError("DVC_API_KEY is required to restore a checkpoint")
        source = source.resolve()
        pointer = pointer.resolve()
        pointer.relative_to(source)
        outputs = _pointer_outputs(source, pointer)
        _configure(source, token)
        _run(["dvc", "pull", "--force", str(pointer.relative_to(source))], source)
        missing = [str(path) for path in outputs if not path.is_file() and not path.is_dir()]
        if missing:
            raise RuntimeError(
                f"DVC restore did not materialize outputs for {pointer.name}: "
                + ", ".join(missing)
            )
        return outputs
    except BaseException:
        print(f"[dvc] restore_pointer traceback for {pointer}:", flush=True)
        traceback.print_exc()
        raise


def restore_checkpoint(source: Path, checkpoint_root: Path) -> Path:
    """Restore a previously DVC-pushed checkpoint tree through its pointer."""
    source = source.resolve()
    checkpoint_root = checkpoint_root.resolve()
    checkpoint_root.relative_to(source)
    pointer = source / ".resume" / f"{checkpoint_root.name}.dvc"
    if not pointer.is_file():
        raise FileNotFoundError(f"DVC resume pointer is missing: {pointer}")
    restore_pointer(source, pointer)
    if not checkpoint_root.is_dir() and not checkpoint_root.is_file():
        raise RuntimeError(f"DVC restore did not materialize {checkpoint_root}")
    return checkpoint_root

def publish(source: Path, run_id: str, worker: int) -> None:
    token = os.environ.get("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required for DagsHub persistence")
    remote = _configure(source, token)
    paths = [
        p.name for p in source.iterdir()
        if p.is_file() and p.suffix == ".csv"
        and p.name not in {"canonical_records.csv", "gate_results.csv"}
    ]
    paths.extend(
        p.name for p in source.iterdir()
        if p.is_file() and p.suffix == ".log"
    )
    log_dir = source / "logs"
    if log_dir.is_dir():
        paths.extend(
            str(path.relative_to(source))
            for path in sorted(log_dir.rglob("*.csv"))
            if path.is_file()
        )
    if paths:
        _run(["dvc", "add", *paths], source)
    lock_path = source.parent / ".dvc-push.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        print(f"[dvc] waiting for shared push lock: {lock_path}", flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _run(["dvc", "push", "--jobs", "1"], source)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    outputs = _verify_clean_pull(source, token, remote)
    manifest = {
        "run_id": run_id,
        "worker": worker,
        "remote": remote,
        "verified_download": True,
        "outputs": outputs,
    }
    (source / "dvc_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[dvc] clean pull verified; DagsHub DVC remote is authoritative", flush=True)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--run-id", required=True); p.add_argument("--worker", type=int, required=True)
    a = p.parse_args(); print(f"[dvc] publishing worker {a.worker} from {a.source}", flush=True); publish(a.source, a.run_id, a.worker); print(f"[dvc] published worker {a.worker}", flush=True)

if __name__ == "__main__": main()
