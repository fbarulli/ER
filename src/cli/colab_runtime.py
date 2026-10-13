"""Session + runtime + environment (split phase C of cli.colab); cross-references resolve lazily through cli.colab_hub."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.common import TRAIN_ROOT, load_config, resolve_model, training_cfg
from training.prepare_all_trace import timed
from cli.colab_hub import hub, timed_colab

# Byte-equal import-time constant feeding _BOOTSTRAP (config-owned, immutable).
_REMOTE_ROOT = training_cfg().colab.remote_root


@timed_colab("step")
def _verify_session_handshake() -> None:
    """Fail before checkout if the CLI cannot execute on the VM."""
    try:
        heartbeat = hub().run_colab_exec_capture(
            hub().SESSION,
            "import os, socket, sys; print({'pid': os.getpid(), 'python': sys.version.split()[0], 'host': socket.gethostname()})",
            timeout=60,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"Colab session '{hub().SESSION}' failed the control-channel handshake "
            f"before training: {exc}"
        ) from exc
    print(hub()._stamp(), f"[session] control-channel handshake passed: {heartbeat.strip()}")

@timed_colab("step")
def _forget_cached_session() -> None:
    """Drop only this launcher's stale local session record before reprovisioning."""
    if not hub()._COLAB_CLI_CONFIG.is_file():
        return
    try:
        state = json.loads(hub()._COLAB_CLI_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(state, dict) or hub().SESSION not in state:
        return
    state.pop(hub().SESSION, None)
    temporary = hub()._COLAB_CLI_CONFIG.with_name(hub()._COLAB_CLI_CONFIG.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, hub()._COLAB_CLI_CONFIG)
    print(hub()._stamp(), f"[session] removed stale cached record for '{hub().SESSION}'", flush=True)

@timed_colab("step")
def ensure_session() -> bool:
    """Provision or attach the session; return whether THIS run OWNS the VM.

    False means the session was ALREADY active and merely attached: the
    launcher that provisioned it owns the release. Releasing on attach was the
    bfed863 chain regression — each suite's exit killed the shared VM and
    re-allocated against the single account GPU slot (503/eviction storms).
    """
    r = hub().colab("sessions", check=False)
    if r.returncode == 0 and hub().SESSION in (r.stdout or ""):
        print(hub()._stamp(), f"[session] '{hub().SESSION}' already active; verifying control channel ...")
        try:
            hub()._verify_session_handshake()
            return False
        except BaseException as exc:
            print(hub()._stamp(), f"[session] cached session is stale; reprovisioning ({exc})", flush=True)
            hub()._forget_cached_session()
    else:
        # The CLI may retain a named session locally after the VM has been
        # torn down.  Never let that record prevent a fresh allocation.
        hub()._forget_cached_session()
    accelerator = [] if hub().GPU.upper() == "CPU" else ["--gpu", hub().GPU]
    print(hub()._stamp(), f"[session] provisioning {hub().SESSION} ({'cpu' if not accelerator else f'gpu={hub().GPU}'}) ...")
    # Owner ruling 8: the allocation shape is decided once, in the policy, for
    # CPU and GPU requests alike (training.yaml bundle_prep.high_mem). With the
    # config flag off its argv is empty, byte-identical to pre-parity behavior.
    from cli.colab_data_bundle_prep import AssignShapePolicy
    hub().colab(
        "new",
        "-s",
        hub().SESSION,
        *accelerator,
        *AssignShapePolicy.from_config().argv(accelerator),
        timeout=300,
    )
    print(hub()._stamp(), "[session] provisioned; running control-channel handshake ...")
    hub()._verify_session_handshake()
    return True

# --- Runtime checkout contract -------------------------------------------------
# The prepared Colab lanes sparse-check out only the paths below; the VM never
# clones the full tree. Any repo-root file a remote stage reads at REMOTE_ROOT
# must be listed in RUNTIME_REQUIRED_ROOT_FILES and committed to the pushed
# branch -- the VM sees that branch, not this working tree. A missing or
# uncommitted entry surfaces as a FileNotFoundError on the VM, so
# validate_runtime_checkout() fails loud locally, before any VM is allocated.
#
# ONE HOME (consolidated 2026-10-08): the Colab checkout contract lives in
# ``ColabSpec`` (config/training.yaml colab.checkout_paths) — the declared
# list, its emitted sparse patterns (``ColabSpec.sparse_checkout_patterns``),
# and the path-shape rule every consumer shares (is_checkout_relative_path,
# which the lane's ColabLaneBase.checkout_relative_guard now delegates to).
# The two module attributes below are the PROJECTION the offline checkout test
# patches; the spec remains the only declaration. The Colab list is deliberately
# NOT derivable from kaggle.checkout_paths (this lane needs artifacts/wheels +
# artifacts/evidence and no requirements/artifacts/models), which is why it has
# its own config key.
_COLAB = training_cfg().colab
#: Directory entries carry a trailing slash; the rest are repo-root files.
RUNTIME_DIRECTORY_PATHS = _COLAB.checkout_directory_paths()
RUNTIME_REQUIRED_ROOT_FILES = _COLAB.checkout_root_files()
#: The declared projection, used to tell an unpatched pair from a test override.
_DECLARED_CHECKOUT_LISTS = (RUNTIME_DIRECTORY_PATHS, RUNTIME_REQUIRED_ROOT_FILES)
# Characters that can never appear in a repository-relative checkout path: a
# newline/CR breaks the emitted git pathspec, a backslash escapes it, and
# ``*?[]{}`` turned a declared path into a pattern/expansion in the old
# hand-rolled guards. One set, one rule, both consumers.
CHECKOUT_REJECTED_CHARS = '\n\r\\*?[]!{}'


def is_checkout_relative_path(value: str, *, single_component: bool = False) -> bool:
    """Whether ``value`` is a repository-relative checkout path.

    The ONE checkout-path shape contract: the runtime's per-launch sparse
    patterns (``prepare_remote_layout``) and the lane's single-segment guard
    (``ColabLaneBase.checkout_relative_guard``) both ask this instead of
    re-spelling the traversal/pattern charset. ``single_component`` adds the
    lane's extra requirement: exactly one plain path segment (a run id, not a
    path).
    """
    candidate = Path(value)
    if candidate.is_absolute() or '..' in candidate.parts or not candidate.parts:
        return False
    if single_component and len(candidate.parts) != 1:
        return False
    return not any(char in str(value) for char in CHECKOUT_REJECTED_CHARS)


def runtime_checkout_paths() -> tuple[str, ...]:
    """The no-cone sparse patterns every prepared runtime lane checks out.

    THE declaration is ``ColabSpec.sparse_checkout_paths()``; the module
    attributes above are its projection, and a caller that patches them (the
    offline checkout test) still steers the emitted patterns.
    """
    if (RUNTIME_DIRECTORY_PATHS, RUNTIME_REQUIRED_ROOT_FILES) == _DECLARED_CHECKOUT_LISTS:
        return _COLAB.sparse_checkout_patterns()
    return tuple('/' + name for name in (*RUNTIME_DIRECTORY_PATHS,
                                         *RUNTIME_REQUIRED_ROOT_FILES))


#: The ONE Colab run-id stamp (owner directive: a run id is stamped once here,
#: never re-spelled by a phase that needs the same identity).
_RUN_STAMP_FORMAT = "%m%dT%H%M%S%fZ"


def _lane_run_stamp() -> str:
    """This launch's run id: the same identity every phase of a run adopts."""
    return datetime.now(timezone.utc).strftime(_RUN_STAMP_FORMAT)


def _validation_input_path(configured_value: str) -> Path:
    """Resolve one configured lane input artifact inside the repository.

    The scoped-value resolution the lane report and its callers share; the
    scored-pair final-inference population is not a config literal — it is the
    SSOT final_validation binding the remote worker resolves itself.
    """
    configured = Path(configured_value)
    root = TRAIN_ROOT
    source = configured if configured.is_absolute() else root / configured
    source = source.resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError(
            "the configured lane input CSV must stay inside the repository"
        )
    return source


def validate_runtime_checkout(extra_paths: tuple[str, ...] = ()) -> None:
    """Fail before provisioning when the VM's sparse checkout cannot be complete.

    The VM clones ``<branch>`` and sparse-checks out the declared directories,
    the required root files, and this launch's ``extra_paths`` (the text-model
    directory; the inputs transport is uploaded directly, never checked out).
    Every one must exist in the cloned revision -- the VM sees that branch,
    not this working tree -- and the root files must also be present locally,
    so a contract typo fails with a clear message. Runs locally: nothing is
    allocated when it raises, mirroring the "validate before provisioning"
    launch-lifecycle rule.
    """
    import subprocess

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(['git', *args], cwd=TRAIN_ROOT,
                              capture_output=True, text=True)

    branch = training_cfg().colab.branch
    revision = (f'origin/{branch}'
                if git('rev-parse', '--verify', '--quiet', f'origin/{branch}').returncode == 0
                else 'HEAD')
    for name in RUNTIME_REQUIRED_ROOT_FILES:
        if not (TRAIN_ROOT / name).is_file():
            raise FileNotFoundError(
                f"runtime checkout file {name!r} is missing from the working tree; "
                'the Colab runtime cannot start without it')
    for name in (*RUNTIME_DIRECTORY_PATHS, *RUNTIME_REQUIRED_ROOT_FILES, *extra_paths):
        if git('cat-file', '-e', f'{revision}:{name.rstrip("/")}').returncode != 0:
            raise RuntimeError(
                f'runtime checkout path {name!r} is not in {revision}; the Colab '
                f'VM clones branch {branch!r} and will not see it -- commit and '
                'push it before launching')


@timed_colab("step")
def prepare_remote_layout(*, minimal_runtime: bool = False, sparse_paths: tuple[str, ...] = ()) -> None:
    """Fetch a shallow prepared runtime, selecting only this suite's inputs."""
    if sparse_paths and not minimal_runtime:
        raise ValueError('sparse checkout requires a prepared runtime')
    patterns = list(runtime_checkout_paths())
    for value in sparse_paths:
        if not is_checkout_relative_path(value):
            raise ValueError('runtime checkout path must be repository-relative')
        patterns.append('/' + Path(value).as_posix())
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

root = pathlib.Path({hub().REMOTE_ROOT!r})
remote_name = {hub().GIT_REMOTE_NAME!r}
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
    run_git(["git", "fetch", *fetch_options, remote_name, {hub().BRANCH!r}], cwd=root, check=True)
    configure_sparse()
    run_git(
        ["git", "checkout", "-B", {hub().BRANCH!r}, 'FETCH_HEAD'],
        cwd=root,
        check=True,
    )
else:
    root.parent.mkdir(parents=True, exist_ok=True)
    clone_options = ['--depth=1', '--single-branch', '--filter=blob:none', '--no-tags'] if minimal_runtime else []
    run_git(["git", "clone", *clone_options, '--no-checkout', "--origin", remote_name,
         "--branch", {hub().BRANCH!r},
         {hub().REPOSITORY!r}, str(root)], check=True)
    configure_sparse()
    run_git(['git', 'checkout', {hub().BRANCH!r}], cwd=root, check=True)
for path in [root / "artifacts" / "data", root / "artifacts" / "results"]:
    path.mkdir(parents=True, exist_ok=True)
print("[repo] ready", {hub().REPOSITORY!r}, "branch", {hub().BRANCH!r},
      "prepared_runtime=" + str({minimal_runtime!r}), "at", root)
"""
    hub().run_colab_exec_stream(hub().SESSION, script, timeout=600, log_name="checkout", retry_safe=True)

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
root = pathlib.Path({hub().REMOTE_ROOT!r})
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

@timed_colab("step")
def install_deps(*, minimal_runtime: bool = False, graph_runtime: bool = False) -> None:
    packages = list(
        hub()._RUNTIME_PACKAGES.prepared if minimal_runtime else hub()._RUNTIME_PACKAGES.full
    )
    if graph_runtime:
        packages = list(dict.fromkeys([*packages, *hub()._RUNTIME_PACKAGES.graph]))
    print(
        hub()._stamp(),
        "[deps] installing "
        + ("prepared training runtime" if minimal_runtime else "full lane dependencies")
        + f" on the VM ({len(packages)} distributions: {', '.join(packages)}) ...",
        flush=True,
    )
    # Run the installer outside the notebook kernel. A kernel disconnect can
    # interrupt the control channel, but the detached process keeps writing a
    # durable log/status pair that the launcher can retrieve before teardown.
    hub().run_detached_stage(
        "00_deps",
        _runtime_install_command(
            packages,
            prefer_uv=hub()._PREFER_UV_INSTALL,
            wheel_paths=list(hub()._RUNTIME_PACKAGES.prebuilt_wheels),
        ),
        timeout=900,
    )

@timed_colab("step")
def log_gpu_profile() -> None:
    """Record the runtime hardware before training, including CPU smoke runs.

    Host RAM is logged beside the accelerator: the 1013 ablation OOMs could
    not be root-caused from the transcript because the granted shape was
    never recorded (one read of /proc/meminfo; no work on the VM).
    """
    script = """import pathlib, torch
ram = {}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
    key = line.split(":")[0]
    if key in ("MemTotal", "MemAvailable"):
        ram[key.lower() + "_gb"] = round(int(line.split()[1]) / 1e6, 2)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print({'hardware': 'gpu', 'name': p.name, 'total_gb': round(total / 1e9, 2), 'free_gb': round(free / 1e9, 2), **ram, 'torch': torch.__version__}, flush=True)
else:
    print({'hardware': 'cpu', 'threads': torch.get_num_threads(), **ram, 'torch': torch.__version__}, flush=True)
"""
    hub().run_colab_exec_stream(hub().SESSION, script, timeout=120, log_name="runtime_profile", retry_safe=True)

_BOOTSTRAP = f"""
import sys, runpy, pathlib, os
sys.path.insert(0, "{_REMOTE_ROOT}/{_COLAB.lane_env_spec.remote_src_dir}")
os.environ["{_COLAB.lane_env_spec.python_path_env}"] = "{_REMOTE_ROOT}/{_COLAB.lane_env_spec.remote_src_dir}" + os.pathsep + os.environ.get("{_COLAB.lane_env_spec.python_path_env}", "")
(pathlib.Path("{_REMOTE_ROOT}/results")).mkdir(parents=True, exist_ok=True)
(pathlib.Path("{_REMOTE_ROOT}/artifacts/data")).mkdir(parents=True, exist_ok=True)
"""

def _env_value(name: str) -> str | None:
    """Read a simple KEY=VALUE entry without printing or cloning secrets."""
    from core.env_file import EnvFile

    return EnvFile.value(name)

class SessionKeepAlive:
    """Touch the runtime while the launcher owns it — idle channels starve.

    Evidence (die_probe2 1013): a FRESH `colab exec` after 120s of silence
    hangs forever while the VM is still listed alive server-side; long silent
    gaps therefore kill the lane. This replaces the deleted keep-alive daemon
    as a class owned by the session-holder, at a config-driven cadence
    (``colab.runtime_keepalive_seconds``). A failed beat never kills the run:
    it prints the full reason and the real poller will surface actual death.
    """

    def __init__(self, session: str, seconds: int) -> None:
        self._session = session
        self._interval = max(10, int(seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._beat, daemon=True,
                                        name="colab-keepalive")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _beat(self) -> None:
        while True:
            try:
                hub().run_colab_exec_capture(self._session, "print('keepalive')",
                                             timeout=25)
            except BaseException as exc:
                import traceback
                print(hub()._stamp(), f"[keepalive] touch failed (lane continues): "
                                      f"{exc}\n{traceback.format_exc()}", flush=True)
            if self._stop.wait(self._interval):
                return

def _wandb_env_script() -> str:
    """Inject only the API key into the remote process, never remote disk.

    The variable name and the project/mode the lanes report to come from
    ``ColabSpec.wandb`` (project/mode are projected from ``tracking.wandb``), so
    no surface re-spells ``WANDB_API_KEY``.
    """
    spec = training_cfg().colab.wandb
    key = hub()._env_value(spec.api_key_env)
    if not key:
        print(hub()._stamp(),
              f"[wandb] {spec.api_key_env} absent from .env; run will remain local-only")
        return ""
    print(hub()._stamp(),
          f"[wandb] API key loaded from local .env and injected into VM process "
          f"(project={spec.project} mode={spec.mode})")
    return f"os.environ[{spec.api_key_env!r}] = {key!r}\n"

def _optuna_env_script() -> str:
    """Inject the shared PostgreSQL control-plane URL into the VM only."""
    url = hub()._env_value("OPTUNA_STORAGE_URL")
    if not url:
        print(hub()._stamp(), "[hpo-control] OPTUNA_STORAGE_URL absent; concurrent HPO is disabled")
        return ""
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("OPTUNA_STORAGE_URL must use a PostgreSQL URL")
    print(hub()._stamp(), "[hpo-control] PostgreSQL Optuna URL loaded from local .env and injected into VM process")
    return f"os.environ['OPTUNA_STORAGE_URL'] = {url!r}\n"

def _remote_auth_env_script(
    *, include_optuna: bool = False, include_wandb: bool = True,
) -> str:
    """Credential exports used by remote subprocess launch cells only."""
    wandb = hub()._wandb_env_script() if include_wandb else ""
    return ("os.environ['EUROMONITOR_DISABLE_DVC_CHECKPOINTS'] = '1'\n"
            "os.environ['ER_INCREMENTAL_DVC'] = '0'\n"
            + wandb + (hub()._optuna_env_script() if include_optuna else ""))

@timed_colab("step")
@timed
def run_data_prep() -> None:
    """Regenerate the derived CSVs on the VM (byte-deterministic replay).

    AUDIT FIX 2026-09-08: 'data_prep only' assumed derived inputs that are
    NOT derived on the VM — the chain is dedupe -> build_reference --verify
    -> data_prep. Running all three keeps the VM replay identical to the
    local worktree replay (byte-comparable outputs).
    """
    print(hub()._stamp(), "[run] dedupe + reference-verify + data_prep on the VM ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
for step in ("src/training/dedupe.py", "src/training/build_second04_pairs.py", "src/training/build_reference.py --verify", "src/training/data_prep.py", "src/training/labeled_pairs.py"):
    print("== " + step, flush=True)
    rc = subprocess.run([sys.executable, "{hub().REMOTE_ROOT}/" + step.split()[0]] + step.split()[1:]).returncode
    if rc != 0:
        raise RuntimeError(f"data-prep stage failed: {{step}} (rc={{rc}})")
"""
    # dedupe 1-2 min + reference verify ~3 min + data_prep ~2 min
    hub().run_colab_exec_stream(hub().SESSION, script, timeout=1800, log_name="data_prep")

@timed_colab("step")
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
        hub()._stamp(),
        f"[models] source=git-shipped requested={keys} status=validation-start",
        flush=True,
    )
    script = _BOOTSTRAP + f"""
from pathlib import Path
from core.common import resolve_model
root = {hub().REMOTE_ROOT!r}
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
    hub().run_colab_exec_stream(
        hub().SESSION,
        script,
        timeout=600,
        log_name="model_validation",
        retry_safe=False,
    )

@timed_colab("step")
def verify_training_inputs() -> None:
    """Use frozen CSV inputs and materialize derived calibration input."""
    print(hub()._stamp(), "[data] validating frozen training CSVs from the cloned branch ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
from core.common import F
required = [
    F["dataset_deduped"],
    F["number_reference"],
    F["canonical_records"],
    F["gate_results"],
]
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
        cwd={hub().REMOTE_ROOT!r},
    ).returncode
    if rc != 0:
        raise RuntimeError(f"labeled-pairs generation failed (rc={{rc}})")
print(f"[data] {{calibration_path}}: {{calibration_path.stat().st_size:,}} bytes", flush=True)
"""
    hub().run_colab_exec_stream(hub().SESSION, script, timeout=120, log_name="01_data_check", retry_safe=True)
