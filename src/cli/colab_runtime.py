"""Session + runtime + environment (split phase C of cli.colab); cross-references resolve lazily through the colab runtime-self hub."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from core.common import TRAIN_ROOT, load_config, resolve_model, training_cfg
from training.prepare_all_trace import timed

# Byte-equal import-time constant feeding _BOOTSTRAP (config-owned, immutable).
_REMOTE_ROOT = training_cfg().colab.remote_root


def _hub():
    """The RUNNING cli.colab module (never a second import copy)."""
    return sys.modules.get("__colab_runtime_self__") or sys.modules["cli.colab"]


def _timed_colab(kind: str):
    """Lazy decorator shim — cli.colab defines _timed_colab after this module imports."""
    def deco(fn):
        return _hub()._timed_colab(kind)(fn)
    return deco


@_timed_colab("step")
def _verify_session_handshake() -> None:
    """Fail before checkout if the CLI cannot execute on the VM."""
    try:
        heartbeat = _hub().run_colab_exec_capture(
            _hub().SESSION,
            "import os, socket, sys; print({'pid': os.getpid(), 'python': sys.version.split()[0], 'host': socket.gethostname()})",
            timeout=60,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"Colab session '{_hub().SESSION}' failed the control-channel handshake "
            f"before training: {exc}"
        ) from exc
    print(_hub()._stamp(), f"[session] control-channel handshake passed: {heartbeat.strip()}")

@_timed_colab("step")
def _forget_cached_session() -> None:
    """Drop only this launcher's stale local session record before reprovisioning."""
    if not _hub()._COLAB_CLI_CONFIG.is_file():
        return
    try:
        state = json.loads(_hub()._COLAB_CLI_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(state, dict) or _hub().SESSION not in state:
        return
    cached = state.get(_hub().SESSION)
    keep_alive_pid = cached.get("keep_alive_pid") if isinstance(cached, dict) else None
    if isinstance(keep_alive_pid, int) and keep_alive_pid != os.getpid():
        proc_cmdline = Path(f"/proc/{keep_alive_pid}/cmdline")
        try:
            command = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            command = ""
        if _is_keep_alive_daemon(command):
            try:
                os.kill(keep_alive_pid, 15)
                print(_hub()._stamp(), f"[session] stopped stale local keep-alive pid={keep_alive_pid}", flush=True)
            except ProcessLookupError:
                pass
    state.pop(_hub().SESSION, None)
    temporary = _hub()._COLAB_CLI_CONFIG.with_name(_hub()._COLAB_CLI_CONFIG.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, _hub()._COLAB_CLI_CONFIG)
    print(_hub()._stamp(), f"[session] removed stale cached record for '{_hub().SESSION}'", flush=True)

def _is_keep_alive_daemon(command: str) -> bool:
    """Whether a /proc command line is this wrapper's keep-alive daemon."""
    return (
        _hub()._COLAB_CLI_ENTRYPOINT.name in command
        and "keep-alive" in command
    )

def keep_alive_daemon_pids() -> list[int]:
    """PIDs of keep-alive daemons serving THIS session.

    The CLI records the pid of the daemon it spawned, but a backstop must not
    depend on the CLI's bookkeeping being present or correct, so the process
    table is the source of truth and the recorded pid is only a hint.
    """
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except OSError:
            continue
        if _is_keep_alive_daemon(command) and _hub().SESSION in command:
            found.append(int(entry.name))
    return sorted(found)

@_timed_colab("step")
def stop_keep_alive_daemon(*, reason: str) -> int:
    """Stop this session's keep-alive daemon and report how many were stopped.

    The daemon is what provisioning needs, so it is always allowed to start.
    On a lane that must never be retained it is stopped as soon as the launcher
    owns the run: the launcher's own teardown remains the primary release, and
    this is the backstop that keeps a crash from leaving the VM held open by
    its own daemon.  A daemon that cannot be found is reported loudly rather
    than passed over in silence.
    """
    pids = keep_alive_daemon_pids()
    if not pids:
        print(
            f"[session] no keep-alive daemon found for '{_hub().SESSION}' ({reason}); "
            "the VM is released by the launcher's own teardown",
            flush=True,
        )
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:
            print(
                _hub()._stamp(),
                f"[session] could not stop keep-alive pid={pid} ({exc!r}); "
                "the VM is released by the launcher's own teardown",
                flush=True,
            )
            continue
        print(_hub()._stamp(), f"[session] stopped keep-alive daemon pid={pid} ({reason})", flush=True)
    return len(pids)

@_timed_colab("step")
def ensure_session() -> None:
    """Provision and verify the session before any training stage starts."""
    r = _hub().colab("sessions", check=False)
    if r.returncode == 0 and _hub().SESSION in (r.stdout or ""):
        print(_hub()._stamp(), f"[session] '{_hub().SESSION}' already active; verifying control channel ...")
        try:
            _verify_session_handshake()
            return
        except BaseException as exc:
            print(_hub()._stamp(), f"[session] cached session is stale; reprovisioning ({exc})", flush=True)
            _forget_cached_session()
    else:
        # The CLI may retain a named session locally after the VM has been
        # torn down.  Never let that record prevent a fresh allocation.
        _forget_cached_session()
    accelerator = [] if _hub().GPU.upper() == "CPU" else ["--gpu", _hub().GPU]
    print(_hub()._stamp(), f"[session] provisioning {_hub().SESSION} ({'cpu' if not accelerator else f'gpu={_hub().GPU}'}) ...")
    # Owner ruling 8: the CPU high-RAM production shape belongs to its own
    # lane (cli.colab_data_bundle_prep); this line is the thin passthrough.
    # With the lane's config flag off it returns (), byte-identical argv.
    from cli.colab_data_bundle_prep import cpu_shape_args
    _hub().colab("new", "-s", _hub().SESSION, *accelerator, *cpu_shape_args(accelerator), timeout=300)
    print(_hub()._stamp(), "[session] provisioned; running control-channel handshake ...")
    _verify_session_handshake()

@_timed_colab("step")
def prepare_remote_layout(*, minimal_runtime: bool = False, sparse_paths: tuple[str, ...] = ()) -> None:
    """Fetch a shallow prepared runtime, selecting only this suite's inputs."""
    if sparse_paths and not minimal_runtime:
        raise ValueError('sparse checkout requires a prepared runtime')
    patterns = ['/src/', '/config/', '/scripts/', '/artifacts/wheels/',
                '/artifacts/evidence/',
                '/pyproject.toml', '/requirements.txt', '/colab_backend.py']
    for value in sparse_paths:
        path = Path(value)
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or any(char in value for char in '\n\r\\*?[]!')):
            raise ValueError('runtime checkout path must be repository-relative')
        patterns.append('/' + path.as_posix())
    script = f"""
import pathlib, shutil, subprocess, time

def run_git(command, **kwargs):
    name = "git." + command[1]
    started = time.perf_counter()
    print(f"[checkout] event={{name}} state=started", flush=True)
    try:
        result = subprocess.run(command, **kwargs)
    except BaseException:
        print(f"[checkout] event={{name}} state=failed elapsed_seconds={{time.perf_counter() - started:.3f}}", flush=True)
        raise
    print(f"[checkout] event={{name}} state={{'completed' if result.returncode == 0 else 'failed'}} elapsed_seconds={{time.perf_counter() - started:.3f}}", flush=True)
    return result

root = pathlib.Path({_hub().REMOTE_ROOT!r})
remote_name = {_hub().GIT_REMOTE_NAME!r}
sparse_patterns = {patterns if sparse_paths else []!r}
minimal_runtime = {minimal_runtime!r}
def configure_sparse():
    if sparse_patterns:
        run_git(['git', 'sparse-checkout', 'set', '--no-cone', '--stdin'],
                input='\\n'.join(sparse_patterns) + '\\n', text=True, cwd=root, check=True)
    else:
        run_git(['git', 'sparse-checkout', 'disable'], cwd=root, check=False)
if root.exists() and not (root / ".git").is_dir():
    shutil.rmtree(root)
if (root / ".git").is_dir():
    remotes = run_git(
        ["git", "remote"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.split()
    if remote_name not in remotes:
        if remote_name != "origin" and "origin" in remotes:
            run_git(
                ["git", "remote", "rename", "origin", remote_name],
                cwd=root,
                check=True,
            )
        else:
            raise RuntimeError(
                f"configured git remote {{remote_name!r}} is absent in {{root}}; "
                f"available remotes={{remotes}}"
            )
    fetch_options = ['--depth=1', '--filter=blob:none', '--no-tags'] if minimal_runtime else []
    run_git(["git", "fetch", *fetch_options, remote_name, {_hub().BRANCH!r}], cwd=root, check=True)
    configure_sparse()
    run_git(
        ["git", "checkout", "-B", {_hub().BRANCH!r}, 'FETCH_HEAD'],
        cwd=root,
        check=True,
    )
else:
    root.parent.mkdir(parents=True, exist_ok=True)
    clone_options = ['--depth=1', '--single-branch', '--filter=blob:none', '--no-tags'] if minimal_runtime else []
    run_git(["git", "clone", *clone_options, '--no-checkout', "--origin", remote_name,
         "--branch", {_hub().BRANCH!r},
         {_hub().REPOSITORY!r}, str(root)], check=True)
    configure_sparse()
    run_git(['git', 'checkout', {_hub().BRANCH!r}], cwd=root, check=True)
for path in [root / "artifacts" / "data", root / "artifacts" / "results"]:
    path.mkdir(parents=True, exist_ok=True)
print("[repo] ready", {_hub().REPOSITORY!r}, "branch", {_hub().BRANCH!r},
      "prepared_runtime=" + str({minimal_runtime!r}), "at", root)
"""
    _hub().run_colab_exec_stream(_hub().SESSION, script, timeout=600, log_name="checkout", retry_safe=True)

def _runtime_install_command(
    packages: list[str], *, prefer_uv: bool, wheel_paths: list[str],
) -> str:
    """Build the remote command that installs one lane's runtime packages.

    ``uv`` resolves and downloads the same wheels several times faster than
    pip, and the installed Colab CLI already prefers it for its own
    ``colab install`` subcommand.  ``--python sys.executable`` targets exactly
    the interpreter that will import these packages, so the fast path cannot
    land them in a different environment than the pip fallback does.

    A configured prebuilt wheel replaces its distribution and is used only when
    its ABI/platform tag matches the interpreter actually running — a compiled
    extension from another Python or architecture is worthless, and a silent
    mismatch would install nothing while looking successful.

    Which installer ran, which prebuilt wheel was used or rejected, and any
    downgrade to pip are all printed, so the durable stage log never hides the
    slow path (no silent fallbacks).
    """
    program = f"""\
import pathlib, shutil, subprocess, sys, sysconfig

packages = {packages!r}
root = pathlib.Path({_hub().REMOTE_ROOT!r})
tag = "cp{{}}{{}}".format(*sys.version_info[:2])
platform = sysconfig.get_platform().replace("-", "_")
requirements = []
provided = set()
for relative in {wheel_paths!r}:
    wheel = root / relative
    filename = wheel.name
    distribution = filename.split("-")[0].replace("_", "-").lower()
    if not wheel.is_file():
        print(f"[deps] prebuilt wheel missing: {{wheel}}", flush=True)
        continue
    if tag not in filename or platform not in filename:
        print(
            f"[deps] prebuilt wheel {{filename}} does not match {{tag}}/{{platform}}; "
            "building from the index instead",
            flush=True,
        )
        continue
    requirements.append(str(wheel))
    provided.add(distribution)
    print(f"[deps] prebuilt wheel={{wheel}} replaces {{distribution}}", flush=True)
packages = [
    package for package in packages
    if package.split("==")[0].split("[")[0].replace("_", "-").lower() not in provided
]
install = requirements + packages
uv = shutil.which("uv") if {prefer_uv!r} else None
if uv:
    command = [uv, "pip", "install", "--python", sys.executable, *install]
    print("[deps] installer=uv", " ".join(command), flush=True)
    if subprocess.call(command) == 0:
        raise SystemExit(0)
    print("[deps] uv install failed; falling back to pip", flush=True)
elif {prefer_uv!r}:
    print("[deps] uv is absent on the VM; falling back to pip", flush=True)
else:
    print("[deps] uv disabled by configuration; using pip", flush=True)
command = [sys.executable, "-m", "pip", "install", *install]
print("[deps] installer=pip", " ".join(command), flush=True)
raise SystemExit(subprocess.call(command))
"""
    return f"[sys.executable, '-c', {program!r}]"

@_timed_colab("step")
def install_deps(*, minimal_runtime: bool = False, graph_runtime: bool = False) -> None:
    packages = list(
        _hub()._RUNTIME_PACKAGES.prepared if minimal_runtime else _hub()._RUNTIME_PACKAGES.full
    )
    if graph_runtime:
        packages = list(dict.fromkeys([*packages, *_hub()._RUNTIME_PACKAGES.graph]))
    print(
        _hub()._stamp(),
        "[deps] installing "
        + ("prepared training runtime" if minimal_runtime else "full lane dependencies")
        + f" on the VM ({len(packages)} distributions: {', '.join(packages)}) ...",
        flush=True,
    )
    # Run the installer outside the notebook kernel. A kernel disconnect can
    # interrupt the control channel, but the detached process keeps writing a
    # durable log/status pair that the launcher can retrieve before teardown.
    _hub().run_detached_stage(
        "00_deps",
        _runtime_install_command(
            packages,
            prefer_uv=_hub()._PREFER_UV_INSTALL,
            wheel_paths=list(_hub()._RUNTIME_PACKAGES.prebuilt_wheels),
        ),
        timeout=900,
    )

@_timed_colab("step")
def log_gpu_profile() -> None:
    """Record the runtime hardware before training, including CPU smoke runs."""
    script = """import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print({'hardware': 'gpu', 'name': p.name, 'total_gb': round(total / 1e9, 2), 'free_gb': round(free / 1e9, 2), 'torch': torch.__version__}, flush=True)
else:
    print({'hardware': 'cpu', 'threads': torch.get_num_threads(), 'torch': torch.__version__}, flush=True)
"""
    _hub().run_colab_exec_stream(_hub().SESSION, script, timeout=120, log_name="runtime_profile", retry_safe=True)

_BOOTSTRAP = f"""
import sys, runpy, pathlib, os
sys.path.insert(0, "{_REMOTE_ROOT}/src")
os.environ["PYTHONPATH"] = "{_REMOTE_ROOT}/src" + os.pathsep + os.environ.get("PYTHONPATH", "")
(pathlib.Path("{_REMOTE_ROOT}/results")).mkdir(parents=True, exist_ok=True)
(pathlib.Path("{_REMOTE_ROOT}/artifacts/data")).mkdir(parents=True, exist_ok=True)
"""

def _env_value(name: str) -> str | None:
    """Read a simple KEY=VALUE entry without printing or cloning secrets."""
    for env_path in (TRAIN_ROOT / ".env", TRAIN_ROOT.parent / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return os.environ.get(name) or None
    return None

def _wandb_env_script() -> str:
    """Inject only the API key into the remote process, never remote disk."""
    key = _env_value("WANDB_API_KEY")
    if not key:
        print(_hub()._stamp(), "[wandb] WANDB_API_KEY absent from .env; run will remain local-only")
        return ""
    print(_hub()._stamp(), "[wandb] API key loaded from local .env and injected into VM process")
    return f"os.environ['WANDB_API_KEY'] = {key!r}\n"

def _optuna_env_script() -> str:
    """Inject the shared PostgreSQL control-plane URL into the VM only."""
    url = _env_value("OPTUNA_STORAGE_URL")
    if not url:
        print(_hub()._stamp(), "[hpo-control] OPTUNA_STORAGE_URL absent; concurrent HPO is disabled")
        return ""
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("OPTUNA_STORAGE_URL must use a PostgreSQL URL")
    print(_hub()._stamp(), "[hpo-control] PostgreSQL Optuna URL loaded from local .env and injected into VM process")
    return f"os.environ['OPTUNA_STORAGE_URL'] = {url!r}\n"

def _remote_auth_env_script(
    *, include_optuna: bool = False, include_wandb: bool = True,
) -> str:
    """Credential exports used by remote subprocess launch cells only."""
    wandb = _wandb_env_script() if include_wandb else ""
    return ("os.environ['EUROMONITOR_DISABLE_DVC_CHECKPOINTS'] = '1'\n"
            "os.environ['ER_INCREMENTAL_DVC'] = '0'\n"
            + wandb + (_optuna_env_script() if include_optuna else ""))

@_timed_colab("step")
@timed
def run_data_prep() -> None:
    """Regenerate the derived CSVs on the VM (byte-deterministic replay).

    AUDIT FIX 2026-09-08: 'data_prep only' assumed derived inputs that are
    NOT derived on the VM — the chain is dedupe -> build_reference --verify
    -> data_prep. Running all three keeps the VM replay identical to the
    local worktree replay (byte-comparable outputs).
    """
    print(_hub()._stamp(), "[run] dedupe + reference-verify + data_prep on the VM ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
for step in ("src/training/dedupe.py", "src/training/build_second04_pairs.py", "src/training/build_reference.py --verify", "src/training/data_prep.py", "src/training/labeled_pairs.py"):
    print("== " + step, flush=True)
    rc = subprocess.run([sys.executable, "{_hub().REMOTE_ROOT}/" + step.split()[0]] + step.split()[1:]).returncode
    if rc != 0:
        raise RuntimeError(f"data-prep stage failed: {{step}} (rc={{rc}})")
"""
    # dedupe 1-2 min + reference verify ~3 min + data_prep ~2 min
    _hub().run_colab_exec_stream(_hub().SESSION, script, timeout=1800, log_name="data_prep")

@_timed_colab("step")
def verify_remote_models(model_keys: list[str]) -> None:
    """Validate the Git-shipped model bundles before starting any worker."""
    keys = sorted(set(model_keys))
    if not keys:
        return
    config = load_config()
    registry = config["models"]
    unknown = sorted(set(keys) - set(registry))
    if unknown:
        raise KeyError(f"unknown local model registry key(s): {unknown}")
    model_root_relative = str(config["paths"]["models_dir"])
    print(
        _hub()._stamp(),
        f"[models] source=git-shipped requested={keys} status=validation-start",
        flush=True,
    )
    script = _BOOTSTRAP + f"""
from pathlib import Path
from core.common import resolve_model
root = {_hub().REMOTE_ROOT!r}
requested = {keys!r}
model_root = (Path(root) / {model_root_relative!r}).resolve()

def bundle_bytes(path):
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

for key in requested:
    path = Path(resolve_model(key))
    if model_root not in path.parents:
        raise RuntimeError(
            f"resolved Git-shipped model {{key}} escapes {{model_root}}: {{path}}"
        )
    print(
        f"[models] key={{key}} path={{path.relative_to(Path(root))}} "
        f"status=validated bytes={{bundle_bytes(path)}}",
        flush=True,
    )
print(
    f"[models] source=git-shipped requested={{requested}} status=validated",
    flush=True,
)
"""
    _hub().run_colab_exec_stream(
        _hub().SESSION,
        script,
        timeout=600,
        log_name="model_validation",
        retry_safe=False,
    )

@_timed_colab("step")
def verify_training_inputs() -> None:
    """Use frozen CSV inputs and materialize derived calibration input."""
    print(_hub()._stamp(), "[data] validating frozen training CSVs from the cloned branch ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
from core.common import F
required = [
    F["dataset_deduped"],
    F["number_reference"],
    F["canonical_records"],
    F["gate_results"],
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("frozen training CSVs missing: " + ", ".join(missing))
for path in required:
    print(f"[data] {{path}}: {{path.stat().st_size:,}} bytes", flush=True)
calibration_path = F["labeled_pairs"]
if not calibration_path.is_file():
    print(
        f"[data] derived calibration input missing; generating {{calibration_path}}",
        flush=True,
    )
    rc = subprocess.run(
        [sys.executable, "src/training/labeled_pairs.py"],
        cwd={_hub().REMOTE_ROOT!r},
    ).returncode
    if rc != 0:
        raise RuntimeError(f"labeled-pairs generation failed (rc={{rc}})")
if not calibration_path.is_file():
    raise FileNotFoundError(f"derived calibration input missing after generation: {{calibration_path}}")
print(f"[data] {{calibration_path}}: {{calibration_path.stat().st_size:,}} bytes", flush=True)
"""
    _hub().run_colab_exec_stream(_hub().SESSION, script, timeout=120, log_name="01_data_check", retry_safe=True)
