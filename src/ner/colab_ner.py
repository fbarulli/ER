"""
colab_ner.py
============

Google Colab runner for NER training.

Configuration policy
--------------------
Persistent project/runtime settings live in ``config.yaml``. This runner does
not import SESSION, GPU, REMOTE_ROOT, or REMOTE_PROJECT_DIR from
the shared reconciliation pipeline and does not use environment variables for them.

Required config structure
-------------------------
The values below must exist in ``config.yaml``. The ``...`` values are only
placeholders here; set them to the values you actually want.

    base_dir: "."
    results_dir: "results"

    colab:
      session: ...
      gpu: ...
      remote_root: ...

    huggingface:
      ner_repo_id: ...
      token_file: ...

    ner_training:
      input_jsonl: "${results_dir}/ner_dataset.jsonl"
      output_dir: "${results_dir}/ner_model"
      model_name: "${base_dir}/models/xlm-roberta-base"
      model_source_repo: ...

``huggingface.token_file`` points to either a local text file containing the
HF token or a ``.env`` file containing ``HF_TOKEN``. The token is read only
when needed and is never printed.

Dry run
-------

    python DONT_TOUCH_ME_PLIS/colab_ner.py --dry-run

The dry run validates:
- config.yaml and required keys
- local ner.py and NER dataset
- Hugging Face token file and repository access
- Colab CLI availability
- whether the configured Colab session currently exists
- whether ``checkpoints/latest_checkpoint.zip`` exists in the HF repo

It does not create a session, upload files, install packages, restore a
checkpoint, download a model, or start training.

Normal run
----------

    python DONT_TOUCH_ME_PLIS/colab_ner.py

Normal execution:
1. validates configuration and local inputs
2. validates HF access
3. reuses or creates the configured Colab session
4. prepares remote directories
5. installs remote dependencies
6. uploads config.yaml, ner.py, optional config_loader.py, and ner_dataset.jsonl
7. downloads the configured base model on Colab if absent
8. restores the rolling HF checkpoint if present
9. launches ner.py detached
10. monitors ner_train.log until training exits
11. downloads final artifacts

Important
---------
``ner.py`` must follow the same configuration policy. If it currently reads
HF_TOKEN or HF_NER_REPO from environment variables, change that file separately
to read the Hugging Face settings from ``config.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from core.common import TRAINING_CONFIG_PATH, TRAIN_ROOT, ner_config

NER_SOURCE_DIR = Path(__file__).resolve().parent
ARTIFACT_MANIFEST_NAME = "ner_artifacts_manifest.json"
EXPECTED_FINAL_ARTIFACTS = (
    "ner_errors.csv",
    "training_metadata.json",
    "ner_model_final.zip",
)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_section(config: dict, name: str) -> dict:
    value = config.get(name)
    if not isinstance(value, dict):
        fail(f"Missing config section: {name}")
    return value


def require_value(section: dict, section_name: str, key: str):
    value = section.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        fail(f"Missing config value: {section_name}.{key}")
    return value


def expand_vars(value: str, variables: dict[str, str]) -> str:
    result = str(value)
    for key, replacement in variables.items():
        result = result.replace("${" + key + "}", replacement)
    return result


def load_settings() -> dict:
    config = ner_config()

    base_dir_raw = str(require_value(config, "root", "base_dir"))
    base_dir = Path(base_dir_raw)
    if not base_dir.is_absolute():
        base_dir = (TRAIN_ROOT / base_dir).resolve()

    results_dir_raw = str(require_value(config, "root", "results_dir"))
    results_dir_text = expand_vars(results_dir_raw, {"base_dir": str(base_dir)})
    results_dir = Path(results_dir_text)
    if not results_dir.is_absolute():
        results_dir = (base_dir / results_dir).resolve()

    variables = {
        "base_dir": str(base_dir),
        "results_dir": str(results_dir),
    }

    colab = require_section(config, "colab")
    hf = require_section(config, "huggingface")
    ner = require_section(config, "ner_training")

    session = str(require_value(colab, "colab", "session")).strip()
    gpu = str(require_value(colab, "colab", "gpu")).strip()
    remote_root = str(require_value(colab, "colab", "remote_root")).rstrip("/")

    hf_repo_id = str(require_value(hf, "huggingface", "ner_repo_id")).strip()
    token_file_raw = expand_vars(
        str(require_value(hf, "huggingface", "token_file")), variables
    )
    token_file = Path(token_file_raw).expanduser()
    if not token_file.is_absolute():
        token_file = (base_dir / token_file).resolve()

    model_source_repo = str(
        require_value(ner, "ner_training", "model_source_repo")
    ).strip()

    model_name = Path(
        expand_vars(str(require_value(ner, "ner_training", "model_name")), variables)
    )
    input_jsonl = Path(
        expand_vars(str(require_value(ner, "ner_training", "input_jsonl")), variables)
    )
    output_dir = Path(
        expand_vars(str(require_value(ner, "ner_training", "output_dir")), variables)
    )

    remote_project_dir = f"{remote_root}/{NER_SOURCE_DIR.name}"
    remote_results_dir = f"{remote_project_dir}/results"

    try:
        model_rel = model_name.resolve().relative_to(base_dir)
        remote_model_name = f"{remote_project_dir}/{model_rel.as_posix()}"
    except ValueError:
        remote_model_name = str(model_name)

    return {
        "base_dir": base_dir,
        "results_dir": results_dir,
        "session": session,
        "gpu": gpu,
        "remote_root": remote_root,
        "remote_project_dir": remote_project_dir,
        "remote_results_dir": remote_results_dir,
        "hf_repo_id": hf_repo_id,
        "token_file": token_file,
        "model_source_repo": model_source_repo,
        "local_input_jsonl": input_jsonl,
        "local_output_dir": output_dir,
        "remote_input_jsonl": f"{remote_results_dir}/{input_jsonl.name}",
        "remote_output_dir": f"{remote_results_dir}/{output_dir.name}",
        "remote_model_name": remote_model_name,
    }


def run(command: list[str], *, check: bool = True, capture: bool = False):
    return subprocess.run(
        command,
        check=check,
        capture_output=capture,
        text=True,
    )


def check_colab_cli() -> None:
    if not shutil.which("colab"):
        fail("'colab' executable not found in PATH")

    result = run(["colab", "version"], check=False, capture=True)
    if result.returncode != 0:
        fail(f"'colab version' failed:\n{result.stderr.strip()}")


def session_exists(session: str) -> bool:
    result = run(
        ["colab", "status", "-s", session],
        check=False,
        capture=True,
    )
    return result.returncode == 0


def ensure_session(settings: dict) -> None:
    session = settings["session"]

    if session_exists(session):
        log(f"[colab] using existing session: {session}")
        return

    log(f"[colab] creating session: {session} | GPU={settings['gpu']}")
    run(["colab", "new", "-s", session, "--gpu", settings["gpu"]])


def colab_exec(settings: dict, code: str, *, capture: bool = False):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as f:
        f.write(code)
        path = Path(f.name)

    try:
        return run(
            ["colab", "exec", "-s", settings["session"], "-f", str(path)],
            capture=capture,
        )
    finally:
        path.unlink(missing_ok=True)


def colab_upload(settings: dict, local_path: Path, remote_path: str) -> None:
    if not local_path.is_file():
        fail(f"Local file not found: {local_path}")

    log(f"[upload] {local_path} -> {remote_path}")
    run(
        [
            "colab",
            "upload",
            "-s",
            settings["session"],
            str(local_path),
            remote_path,
        ]
    )


def read_hf_token(path: Path) -> str:
    if not path.is_file():
        fail(f"Hugging Face token file not found: {path}")

    contents = path.read_text(encoding="utf-8").strip()
    if not contents:
        fail(f"Hugging Face token file is empty: {path}")

    # A plain token file remains supported. For a .env file, extract only
    # HF_TOKEN instead of passing the complete environment file as a token.
    env_entries = [
        line.strip().removeprefix("export ").strip()
        for line in contents.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    if env_entries:
        for entry in env_entries:
            key, _, value = entry.partition("=")
            if key.strip() != "HF_TOKEN":
                continue
            token = value.strip()
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
                token = token[1:-1]
            if token:
                return token
        fail(f"HF_TOKEN is missing or empty in environment file: {path}")

    return contents


def check_local_files(settings: dict) -> None:
    required = [
        TRAINING_CONFIG_PATH,
        NER_SOURCE_DIR / "ner.py",
        settings["local_input_jsonl"],
        settings["token_file"],
    ]

    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        fail(
            "Missing required local files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def check_huggingface(settings: dict) -> dict:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Install local package 'huggingface_hub' before running this file"
        ) from exc

    api = HfApi(token=read_hf_token(settings["token_file"]))
    repo_id = settings["hf_repo_id"]

    log(f"[hf] checking repo: {repo_id}")
    api.repo_info(repo_id=repo_id, repo_type="model")

    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    checkpoint = "checkpoints/latest_checkpoint.zip"

    return {
        "checkpoint_path": checkpoint,
        "has_checkpoint": checkpoint in files,
    }


def prepare_remote_dirs(settings: dict) -> None:
    code = f'''
from pathlib import Path

for path in [
    {settings["remote_project_dir"]!r},
    {settings["remote_results_dir"]!r},
    {str(Path(settings["remote_model_name"]).parent)!r},
]:
    Path(path).mkdir(parents=True, exist_ok=True)
'''
    colab_exec(settings, code)


def install_remote_dependencies(settings: dict) -> None:
    log("[colab] installing NER dependencies")

    code = '''
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "spacy",
    "spacy-transformers",
    "pydantic",
    "pyyaml",
    "huggingface_hub",
])
'''
    colab_exec(settings, code)


def upload_inputs(settings: dict) -> None:
    remote = settings["remote_project_dir"]

    colab_upload(settings, NER_SOURCE_DIR / "ner.py", f"{remote}/ner.py")

    # Render the single authoritative NER section for the remote process.
    # It is an execution input, not another persistent config source.
    remote_config = ner_config()
    remote_config["base_dir"] = settings["remote_project_dir"]
    remote_config["results_dir"] = settings["remote_results_dir"]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as stream:
        json.dump(remote_config, stream)
        runtime_config = Path(stream.name)
    try:
        colab_upload(settings, runtime_config, f"{remote}/ner_runtime_config.json")
    finally:
        runtime_config.unlink(missing_ok=True)

    colab_upload(
        settings,
        settings["local_input_jsonl"],
        settings["remote_input_jsonl"],
    )


def ensure_remote_base_model(settings: dict) -> None:
    log("[model] checking base transformer")

    code = f'''
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path({settings["remote_model_name"]!r})

if (target / "config.json").is_file():
    print("[model] already present:", target)
else:
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id={settings["model_source_repo"]!r},
        local_dir=str(target),
    )
    print("[model] downloaded:", target)
'''
    colab_exec(settings, code)


def restore_latest_checkpoint(settings: dict, hf_state: dict) -> None:
    if not hf_state["has_checkpoint"]:
        log("[checkpoint] none found; starting fresh")
        return

    token = read_hf_token(settings["token_file"])

    code = f'''
from pathlib import Path
import shutil
import zipfile
from huggingface_hub import hf_hub_download

output_dir = Path({settings["remote_output_dir"]!r})

zip_path = hf_hub_download(
    repo_id={settings["hf_repo_id"]!r},
    filename={hf_state["checkpoint_path"]!r},
    repo_type="model",
    token={token!r},
)

if output_dir.exists():
    shutil.rmtree(output_dir)

output_dir.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall(output_dir.parent)

print("[checkpoint] restored:", zip_path)
'''
    colab_exec(settings, code)


def launch_training(settings: dict) -> None:
    remote = settings["remote_project_dir"]
    log_path = f"{settings['remote_results_dir']}/ner_train.log"
    pid_path = f"{settings['remote_results_dir']}/ner_train.pid"

    code = f'''
from pathlib import Path
import subprocess
import sys

project = Path({remote!r})
log_path = Path({log_path!r})
pid_path = Path({pid_path!r})

log_path.parent.mkdir(parents=True, exist_ok=True)

with log_path.open("w", encoding="utf-8") as log_file:
    process = subprocess.Popen(
        [sys.executable, str(project / "ner.py")],
        cwd=str(project),
        env={{
            **os.environ,
            "NER_RUNTIME_CONFIG": str(project / "ner_runtime_config.json"),
            "NER_BASE_DIR": str(project),
            "NER_RESULTS_DIR": {settings["remote_results_dir"]!r},
        }},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

pid_path.write_text(str(process.pid), encoding="utf-8")
print("[train] PID:", process.pid)
'''
    colab_exec(settings, code)


def training_state(settings: dict) -> dict:
    log_path = f"{settings['remote_results_dir']}/ner_train.log"
    pid_path = f"{settings['remote_results_dir']}/ner_train.pid"

    code = f'''
import json
import os
from pathlib import Path

log_path = Path({log_path!r})
pid_path = Path({pid_path!r})

pid = None
alive = False

if pid_path.is_file():
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        alive = False

lines = []
if log_path.is_file():
    lines = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()[-40:]

print(json.dumps({{
    "pid": pid,
    "alive": alive,
    "tail": "\\n".join(lines),
}}))
'''

    result = colab_exec(settings, code, capture=True)

    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass

    fail("Could not parse remote training state")


def monitor_training(settings: dict) -> None:
    last_tail = None

    while True:
        if not session_exists(settings["session"]):
            fail(
                "Colab session disappeared. The latest fully uploaded HF "
                "checkpoint should remain durable."
            )

        state = training_state(settings)
        tail = state.get("tail", "")

        if tail and tail != last_tail:
            print(tail, flush=True)
            last_tail = tail

        if not state.get("alive"):
            break

        time.sleep(15)

    log("[train] remote process finished")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_artifact_manifest(path: Path) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Invalid downloaded NER artifact manifest {path}: {exc}")
    if payload.get("schema_version") != "1" or not isinstance(
        payload.get("artifacts"), dict
    ):
        fail(f"Invalid NER artifact manifest structure: {path}")
    artifacts = payload["artifacts"]
    missing = [name for name in EXPECTED_FINAL_ARTIFACTS if name not in artifacts]
    if missing:
        fail(
            "NER artifact manifest omits required final artifact(s): "
            + ", ".join(missing)
        )
    for name in EXPECTED_FINAL_ARTIFACTS:
        entry = artifacts[name]
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            fail(f"NER artifact manifest has invalid sha256 for {name}")
    return artifacts


def download_if_exists(
    settings: dict,
    remote_path: str,
    local_path: Path,
    *,
    required: bool = True,
) -> None:
    """Download an artifact, failing loudly when a required one is absent."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    result = run(
        [
            "colab",
            "download",
            "-s",
            settings["session"],
            remote_path,
            str(local_path),
        ],
        check=False,
        capture=True,
    )

    if result.returncode == 0:
        log(f"[download] saved: {local_path}")
    else:
        detail = (result.stderr or result.stdout or "unknown colab error").strip()
        message = f"[download] unavailable: {remote_path} ({detail[-500:]})"
        if required:
            fail(message + "; refusing an incomplete NER result set")
        log(message)


def _verify_download(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        fail(f"Downloaded NER artifact is missing locally: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        fail(
            f"NER download hash mismatch for {path}: "
            f"remote={expected_sha256[:12]} local={actual[:12]}"
        )


def download_results(settings: dict) -> None:
    remote = settings["remote_results_dir"]
    local = settings["results_dir"]

    # The manifest is created by ner.py only after all three final artifacts
    # are complete.  Fetch it first, then accept each transfer only if it
    # matches the remote-generated digest.
    manifest_path = local / ARTIFACT_MANIFEST_NAME
    download_if_exists(
        settings,
        f"{remote}/{ARTIFACT_MANIFEST_NAME}",
        manifest_path,
    )
    manifest = _read_artifact_manifest(manifest_path)

    for name in EXPECTED_FINAL_ARTIFACTS:
        local_path = local / name
        download_if_exists(settings, f"{remote}/{name}", local_path)
        _verify_download(local_path, manifest[name]["sha256"])

    # This diagnostic is useful but is not a final deliverable, so a missing
    # log cannot be treated as a successful substitute for model artifacts.
    download_if_exists(
        settings,
        f"{remote}/ner_train.log",
        local / "logs" / "ner" / "training.log",
        required=False,
    )


def extract_final_model(settings: dict) -> None:
    zip_path = settings["results_dir"] / "ner_model_final.zip"
    output_dir = settings["local_output_dir"]

    if not zip_path.is_file():
        log("[model] final ZIP was not downloaded")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)

    log(f"[model] extracted: {output_dir}")


def dry_run(settings: dict) -> None:
    log("[dry-run] checking local files")
    check_local_files(settings)

    log("[dry-run] checking Colab CLI")
    check_colab_cli()

    exists = session_exists(settings["session"])
    log(f"[dry-run] session={settings['session']} exists={exists}")

    log("[dry-run] checking Hugging Face")
    hf_state = check_huggingface(settings)

    print(
        json.dumps(
            {
                "session": settings["session"],
                "gpu": settings["gpu"],
                "remote_root": settings["remote_root"],
                "remote_project_dir": settings["remote_project_dir"],
                "hf_repo_id": settings["hf_repo_id"],
                "token_file": str(settings["token_file"]),
                "model_source_repo": settings["model_source_repo"],
                "remote_model_name": settings["remote_model_name"],
                "input_jsonl": str(settings["local_input_jsonl"]),
                "checkpoint_exists": hf_state["has_checkpoint"],
            },
            indent=2,
        )
    )

    log("[dry-run] OK — no remote changes made")


def train(settings: dict) -> None:
    check_local_files(settings)
    check_colab_cli()
    hf_state = check_huggingface(settings)

    ensure_session(settings)
    prepare_remote_dirs(settings)
    install_remote_dependencies(settings)
    upload_inputs(settings)
    ensure_remote_base_model(settings)
    restore_latest_checkpoint(settings, hf_state)
    launch_training(settings)
    monitor_training(settings)
    download_results(settings)
    extract_final_model(settings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NER training on Google Colab")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and connectivity without modifying Colab",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    if args.dry_run:
        dry_run(settings)
    else:
        train(settings)


if __name__ == "__main__":
    main()
