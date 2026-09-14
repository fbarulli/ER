"""Publish one worker's output through an isolated DVC project."""
from __future__ import annotations
import argparse, fcntl, json, os, shutil, subprocess, tempfile, time, traceback
from pathlib import Path
from core import common

# The tracking/verify EXCLUSION set — files/dirs whose path parts hit these
# are skipped in tracking and verification. KEEP the content byte-for-byte.
DVC_EXCLUDED_DIRS = frozenset({
    ".dvc", ".dvc-cache", ".dvc-site-cache", ".resume", "_checkpoints",
    "_checkpoint_upload_staging", "wandb", "mlruns",
})


def _write_dvc_event(source: Path, event: str, **values: object) -> None:
    """Append structured DVC state beside the worker's result bundle."""
    record = {"event": event, **values}
    event_path = source / common.training_cfg().colab.dvc_events_file
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[dvc-state] {event} | {json.dumps(values, sort_keys=True)}", flush=True)

def _run(command: list[str], cwd: Path) -> str:
    if command[:2] == ["dvc", "push"] and "--jobs" not in command:
        jobs = str(common.training_cfg().colab.dvc_jobs)
        command = [command[0], command[1], "--jobs", jobs, *command[2:]]
    shown = ["<redacted>" if command[i - 1:i] == ["password"] else part for i, part in enumerate(command)]
    print(f"[dvc] running: {' '.join(shown)}", flush=True)
    cfg = common.training_cfg().colab
    attempts = cfg.dvc_push_retries if command[:2] == ["dvc", "push"] else 1
    for attempt in range(1, attempts + 1):
        _write_dvc_event(
            cwd,
            "command_started",
            command=shown,
            attempt=attempt,
            attempts=attempts,
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"[dvc] started pid={process.pid}", flush=True)
            _write_dvc_event(
                cwd,
                "process_started",
                command=shown,
                attempt=attempt,
                pid=int(process.pid),
            )
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
        print(f"[dvc] finished rc={result.returncode}", flush=True)
        _write_dvc_event(
            cwd,
            "command_finished",
            command=shown,
            attempt=attempt,
            returncode=int(result.returncode),
        )
        if result.returncode == 0:
            return result.stdout or ""
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


def _dvc_status_is_clean(output: str) -> bool:
    """Interpret DVC's human-readable clean status without hiding dirtiness."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return True
    return all(
        line.endswith("are in sync.")
        and ("remote" in line or "data" in line.lower())
        for line in lines
    )


def _tracked_outputs(source: Path) -> list[Path]:
    import yaml
    outputs: list[Path] = []
    for pointer in sorted(source.rglob("*.dvc")):
        if not pointer.is_file():
            continue
        if ".resume" in pointer.parts or "_checkpoints" in pointer.parts:
            continue
        data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        for entry in data.get("outs", []):
            path = pointer.parent / str(entry["path"])
            if path.is_file():
                outputs.append(path)
            elif path.is_dir():
                outputs.extend(sorted(child for child in path.rglob("*") if child.is_file()))
    return outputs


def _verify_clean_pull(source: Path, token: str, remote: str) -> list[dict[str, str]]:
    """Pull into a clean directory; prove the remote is independently readable."""
    tracked = _tracked_outputs(source)
    if not tracked:
        raise RuntimeError("DVC push produced no tracked output pointers")
    with tempfile.TemporaryDirectory(
        prefix=".dvc-verify-", dir=source.parent
    ) as temp:
        verify = Path(temp)
        os.environ["DVC_SITE_CACHE_DIR"] = str(verify / ".dvc-site-cache")
        _run(["dvc", "init", "--no-scm"], verify)
        _run(["dvc", "config", "cache.dir", str(verify / ".dvc-cache")], verify)
        _run(["dvc", "remote", "add", "--default", "dagshub", remote], verify)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], verify)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "user", "fbarulli"], verify)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], verify)
        for pointer in source.rglob("*.dvc"):
            if not pointer.is_file():
                continue
            if ".resume" in pointer.parts or "_checkpoints" in pointer.parts:
                continue
            target = verify / pointer.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pointer, target)
        # Pull pointers one at a time. A single argv containing every result
        # pointer grows with checkpoint/report count and can exceed the
        # process argument limit on large runs; individual pulls also make
        # the failing pointer explicit in the traceback.
        pointers = [
            pointer
            for pointer in sorted(source.rglob("*.dvc"))
            if (
                pointer.is_file()
                and ".resume" not in pointer.parts
                and "_checkpoints" not in pointer.parts
            )
        ]
        for pointer in pointers:
            _run(["dvc", "pull", "--force", str(pointer.relative_to(source))], verify)
        result = []
        for original in tracked:
            restored = verify / original.relative_to(source)
            if not restored.is_file():
                raise RuntimeError(f"DVC pull did not restore {original.name}")
            expected, actual = _sha256(original), _sha256(restored)
            if expected != actual:
                raise RuntimeError(f"DVC pull hash mismatch for {original.name}")
            result.append({
                "path": str(original.relative_to(source)),
                "sha256": actual,
            })
        return result


def _configure(source: Path, token: str) -> str:
    """Configure an isolated, no-SCM DVC workspace for one worker."""
    os.environ["DVC_SITE_CACHE_DIR"] = str(source / ".dvc-site-cache")
    remote = common.training_cfg().colab.dvc_remote_url
    if not (source / ".dvc").is_dir():
        _run(["dvc", "init", "--no-scm"], source)
        _run(["dvc", "config", "cache.dir", str(source / ".dvc-cache")], source)
        _run(["dvc", "remote", "add", "--default", "dagshub", remote], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "user", "fbarulli"], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], source)
    return remote


def publish_checkpoint(
    source: Path,
    checkpoint_root: Path,
    *,
    resume_name: str | None = None,
    restore_root: Path | None = None,
) -> Path:
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
    _write_dvc_event(
        source,
        "checkpoint_publish_requested",
        checkpoint=str(checkpoint_root),
        resume_name=resume_name or checkpoint_root.name,
    )
    restore_root = (restore_root or checkpoint_root).resolve()
    relative_root = checkpoint_root.relative_to(source)
    restore_root.relative_to(source)
    pointer = common.artifact("resume_pointer", {"name": resume_name or checkpoint_root.name})
    # DVC recursively discovers existing .dvc files.  The durable resume
    # pointers are intentionally kept under source/.resume, but they must not
    # participate in discovery while a new output is added.  Temporarily
    # stage that metadata outside the DVC source, then put it back even when
    # dvc add fails.
    # Separate Optuna trials share one no-SCM DVC workspace.  Serialise its
    # mutable metadata operations, not just the remote push, otherwise two
    # ``dvc add`` calls can observe or rewrite each other's state.
    lock_path = source / ".dvc-push.lock"
    print(f"[dvc] waiting for worker-local metadata lock: {lock_path}", flush=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _configure(source, token)
            staged_resume = None
            resume_dir = pointer.parent
            if resume_dir.is_dir():
                staged_resume = Path(tempfile.mkdtemp(prefix=".resume-staging-", dir=source.parent))
                shutil.move(str(resume_dir), str(staged_resume / resume_dir.name))
            try:
                _run(["dvc", "add", str(relative_root)], source)
            finally:
                if staged_resume is not None:
                    shutil.move(str(staged_resume / resume_dir.name), str(resume_dir))
                    shutil.rmtree(staged_resume, ignore_errors=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
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
        # The upload workspace can be a hard-linked snapshot that is removed
        # once the asynchronous push completes.  The durable pointer must
        # restore to the Trainer's original checkpoint location instead.
        entry["path"] = os.path.relpath(restore_root, pointer.parent)
    native_relative = native_pointer.relative_to(source)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # This is a no-SCM repository and the checkpoint .dvc file is
            # nested below _checkpoints/.  A target-less dvc push can report
            # success while discovering no top-level tracked outputs, which
            # leaves the resume pointer referring to a nonexistent remote
            # object. Push the exact native pointer explicitly.
            print(f"[checkpoint-dvc] pushing target {native_relative}", flush=True)
            _run(["dvc", "push", str(native_relative)], source)
            # A successful push process is not sufficient evidence on its
            # own. Require DVC's cloud comparison to report this exact
            # pointer in sync before making the resume metadata visible.
            cloud_status = _run(["dvc", "status", "--cloud", str(native_relative)], source)
            if not _dvc_status_is_clean(cloud_status):
                raise RuntimeError(
                    f"DVC cloud status is not clean for {native_relative}: "
                    f"{cloud_status.strip()}"
                )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # Expose the durable resume pointer only after the exact target push has
    # succeeded. The launcher mirrors this file while the worker is alive.
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        yaml.safe_dump(pointer_data, sort_keys=False), encoding="utf-8"
    )
    _write_dvc_event(
        source,
        "checkpoint_publish_verified",
        checkpoint=str(checkpoint_root),
        pointer=str(pointer),
    )
    return pointer


def publish_checkpoints(
    source: Path,
    checkpoints: list[tuple[Path, str, Path]],
) -> list[Path]:
    """Publish multiple immutable checkpoint snapshots with one DVC push.

    Each tuple is ``(snapshot, resume_name, restore_root)``.  Snapshots are
    added together, their native pointers are pushed together, and durable
    resume pointers are made visible only after the complete batch is cloud
    clean.  This keeps checkpoint durability while avoiding one network push
    per Trainer save event.
    """
    if not checkpoints:
        return []
    token = os.environ.get("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required to persist checkpoints")
    source = source.resolve()
    resolved = [
        (snapshot.resolve(), name, restore.resolve())
        for snapshot, name, restore in checkpoints
    ]
    for snapshot, name, restore in resolved:
        snapshot.relative_to(source)
        restore.relative_to(source)
        _write_dvc_event(
            source, "checkpoint_publish_requested", checkpoint=str(snapshot),
            resume_name=name,
        )
    lock_path = source / ".dvc-push.lock"
    pointers = [common.artifact("resume_pointer", {"name": name}) for _, name, _ in resolved]
    resume_dir = pointers[0].parent
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _configure(source, token)
            staged_resume = None
            if resume_dir.is_dir():
                staged_resume = Path(tempfile.mkdtemp(prefix=".resume-staging-", dir=source.parent))
                shutil.move(str(resume_dir), str(staged_resume / resume_dir.name))
            try:
                _run(["dvc", "add", *[str(snapshot.relative_to(source)) for snapshot, _, _ in resolved]], source)
            finally:
                if staged_resume is not None:
                    shutil.move(str(staged_resume / resume_dir.name), str(resume_dir))
                    shutil.rmtree(staged_resume, ignore_errors=True)
            native_pointers = [
                snapshot.with_name(f"{snapshot.name}.dvc")
                for snapshot, _, _ in resolved
            ]
            missing = [str(path) for path in native_pointers if not path.is_file()]
            if missing:
                raise RuntimeError(f"DVC did not create checkpoint pointers: {missing}")
            native_relatives = [str(path.relative_to(source)) for path in native_pointers]
            print(f"[checkpoint-dvc] pushing {len(native_relatives)} checkpoint targets together", flush=True)
            _run(["dvc", "push", *native_relatives], source)
            cloud_status = _run(["dvc", "status", "--cloud", *native_relatives], source)
            if not _dvc_status_is_clean(cloud_status):
                raise RuntimeError(f"DVC cloud status is not clean for checkpoint batch: {cloud_status.strip()}")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    import yaml

    for (snapshot, _, restore), pointer, native_pointer in zip(resolved, pointers, native_pointers, strict=True):
        pointer_data = yaml.safe_load(native_pointer.read_text(encoding="utf-8")) or {}
        for entry in pointer_data.get("outs", []):
            entry["path"] = os.path.relpath(restore, pointer.parent)
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(yaml.safe_dump(pointer_data, sort_keys=False), encoding="utf-8")
        _write_dvc_event(source, "checkpoint_publish_verified", checkpoint=str(snapshot), pointer=str(pointer))
    return pointers


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


def _publish_tracked_pointer_index(
    source: Path,
    run_id: str,
    worker: int,
    remote: str,
) -> Path:
    """Expose worker pointers under tracked repository metadata.

    Worker output trees remain ignored and DVC-backed.  The tracked copies are
    only pointer metadata; each copied pointer is rewritten to restore its
    original ``training_results/...`` target from a clean checkout.
    """
    import yaml

    from core.schemas import DvcPublicationManifest, DvcPublicationPointer

    source = source.resolve()
    repo_root = common.TRAIN_ROOT.resolve()
    source.relative_to(repo_root)
    pointer_records: list[DvcPublicationPointer] = []
    pointers = sorted(source.rglob("*.dvc"))
    if not pointers:
        raise RuntimeError(f"no DVC pointers found under {source}")
    for pointer in pointers:
        if not pointer.is_file():
            continue
        data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        outputs = data.get("outs", [])
        if not outputs:
            raise RuntimeError(f"DVC pointer has no outputs: {pointer}")
        output_paths: list[str] = []
        output_sources: list[Path] = []
        for entry in outputs:
            original = (pointer.parent / str(entry["path"])).resolve()
            original.relative_to(repo_root)
            output_rel = original.relative_to(repo_root).as_posix()
            output_paths.append(output_rel)
            output_sources.append(original)
        relative_pointer = pointer.relative_to(source).as_posix()
        tracked_pointer = common.artifact(
            "dvc_publication_pointer",
            {"run_id": run_id, "worker": worker, "pointer": relative_pointer},
        )
        for entry, original in zip(outputs, output_sources, strict=True):
            entry["path"] = os.path.relpath(original, tracked_pointer.parent)
        tracked_pointer.parent.mkdir(parents=True, exist_ok=True)
        tracked_pointer.write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        pointer_records.append(
            DvcPublicationPointer(
                pointer=tracked_pointer.relative_to(repo_root).as_posix(),
                outputs=output_paths,
            )
        )
    if not pointer_records:
        raise RuntimeError(f"no usable DVC pointers found under {source}")
    manifest = DvcPublicationManifest(
        schema_version="1",
        run_id=run_id,
        worker=int(worker),
        remote=remote,
        verified_download=True,
        pointers=pointer_records,
    )
    manifest_path = common.artifact(
        "dvc_publication_manifest", {"run_id": run_id, "worker": worker}
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


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
    pointer = common.artifact("resume_pointer", {"name": checkpoint_root.name})
    resume_dir = pointer.parent
    if not pointer.is_file():
        # New asynchronous publishing writes one durable pointer per immutable
        # ``checkpoint-N`` directory.  Restore the newest published checkpoint
        # beneath this Trainer output root; an unfinished upload has no pointer
        # and therefore can never be selected for resume.
        candidates: list[tuple[int, Path]] = []
        for candidate in sorted(resume_dir.glob("checkpoint-*.dvc")):
            try:
                outputs = _pointer_outputs(source, candidate)
            except (OSError, ValueError):
                continue
            if len(outputs) != 1 or outputs[0].parent != checkpoint_root:
                continue
            try:
                # Concurrent HPO adds a stable output-root suffix to avoid
                # clobbering pointers from trials that share a global step.
                step_text = candidate.stem.removeprefix("checkpoint-").split("--", 1)[0]
                step = int(step_text)
            except ValueError:
                continue
            candidates.append((step, candidate))
        if not candidates:
            raise FileNotFoundError(
                f"DVC resume pointer is missing for {checkpoint_root}"
            )
        _, pointer = max(candidates)
    restore_pointer(source, pointer)
    if not checkpoint_root.is_dir() and not checkpoint_root.is_file():
        raise RuntimeError(f"DVC restore did not materialize {checkpoint_root}")
    return checkpoint_root

def publish(source: Path, run_id: str, worker: int) -> None:
    token = os.environ.get("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required for DagsHub persistence")
    remote = _configure(source, token)
    _write_dvc_event(
        source,
        "worker_publish_started",
        run_id=run_id,
        worker=int(worker),
    )
    tracked_suffixes = {
        ".csv", ".json", ".png", ".log", ".yaml", ".yml", ".txt",
    }
    excluded_dirs = DVC_EXCLUDED_DIRS
    paths = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix not in tracked_suffixes:
            continue
        relative = path.relative_to(source)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        if relative.as_posix() in {"canonical_records.csv", "gate_results.csv"}:
            continue
        paths.append(str(relative))
    if paths:
        _run(["dvc", "add", *paths], source)
    lock_path = source / ".dvc-push.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        print(f"[dvc] waiting for worker-local push lock: {lock_path}", flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _run(["dvc", "push"], source)
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
    publication_manifest = _publish_tracked_pointer_index(
        source, run_id, worker, remote
    )
    _write_dvc_event(
        source,
        "worker_publish_verified",
        run_id=run_id,
        worker=int(worker),
        output_count=len(outputs),
        publication_manifest=str(publication_manifest),
    )
    print("[dvc] clean pull verified; DagsHub DVC remote is authoritative", flush=True)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--run-id", required=True); p.add_argument("--worker", type=int, required=True)
    a = p.parse_args(); print(f"[dvc] publishing worker {a.worker} from {a.source}", flush=True); publish(a.source, a.run_id, a.worker); print(f"[dvc] published worker {a.worker}", flush=True)

if __name__ == "__main__": main()
