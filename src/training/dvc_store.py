"""Publish one worker's output through an isolated DVC project."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, tempfile
from pathlib import Path
from core.common import training_cfg

def _run(command: list[str], cwd: Path) -> None:
    shown = ["<redacted>" if command[i - 1:i] == ["password"] else part for i, part in enumerate(command)]
    print(f"[dvc] running: {' '.join(shown)}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
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

def publish(source: Path, run_id: str, worker: int) -> None:
    token = os.environ.get("DVC_API_KEY")
    if not token:
        raise RuntimeError("DVC_API_KEY is required for DagsHub persistence")
    os.environ["DVC_SITE_CACHE_DIR"] = str(source / ".dvc-site-cache")
    remote = training_cfg().colab.dvc_remote_url
    if not (source / ".dvc").is_dir():
        _run(["dvc", "init", "--no-scm"], source)
        _run(["dvc", "config", "cache.dir", str(source / ".dvc-cache")], source)
        _run(["dvc", "remote", "add", "--default", "dagshub", remote], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "auth", "basic"], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "user", "fbarulli"], source)
        _run(["dvc", "remote", "modify", "dagshub", "--local", "password", token], source)
    paths = [
        p.name for p in source.iterdir()
        if p.is_file() and p.suffix == ".csv"
        and p.name not in {"canonical_records.csv", "gate_results.csv"}
    ]
    log_dir = source / "logs"
    if log_dir.is_dir():
        paths.extend(
            str(path.relative_to(source))
            for path in sorted(log_dir.rglob("*.csv"))
            if path.is_file()
        )
    if paths:
        _run(["dvc", "add", *paths], source)
    _run(["dvc", "push"], source)
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
