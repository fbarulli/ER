"""Read-only-home-safe entry point for the installed Colab CLI.

The launcher invokes this with the Colab CLI's own Python interpreter. Keeping
the wrapper in the repository makes the parent CLI process and its detached
keep-alive child use the same writable state, history, and logging behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the project from stable markers, never a magic parent offset.

    Must stay self-contained: this wrapper is launched with the Colab CLI's
    own interpreter, which has neither the repo on sys.path nor its deps, so
    core.common/TRAIN_ROOT cannot be imported here.
    """
    override = os.environ.get("EUROMONITOR_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "config").is_dir() and (root / "pyproject.toml").is_file():
            return root
        raise RuntimeError(
            "EUROMONITOR_PROJECT_ROOT must contain config/ and pyproject.toml: "
            f"{root}"
        )
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate project root from {Path(__file__).resolve()}"
    )


ENCLOSURE_ROOT = _find_project_root()
STATE_DIR = ENCLOSURE_ROOT / "colab_cli_state"
HISTORY_DIR = STATE_DIR / "history"
ENTRYPOINT = Path(__file__).resolve()


def _colab_cli_python() -> Path | None:
    """Interpreter that owns the installed Colab CLI, if it can be found.

    The CLI is a uv tool, and its dependencies include compiled extensions
    (``pydantic_core._pydantic_core``) built for that tool's interpreter.  They
    cannot be loaded by a different python, even with its site-packages on
    ``sys.path``, so a wrapper started under any other interpreter must hand
    over to this one rather than try to import the package itself.
    """
    override = os.environ.get("COLAB_CLI_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    tools_root = Path.home() / ".local" / "share" / "uv" / "tools"
    for tool_dir in ("google-colab-cli", "colab-cli"):
        for candidate in sorted((tools_root / tool_dir / "bin").glob("python*")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def _reexec_under_colab_cli_python() -> None:
    """Re-run this wrapper with the interpreter that owns the Colab CLI."""
    interpreter = _colab_cli_python()
    if interpreter is None or Path(sys.executable).resolve() == interpreter.resolve():
        # Already the owning interpreter: carrying on here is the point of the
        # handover, so this is the recursion stop.
        return
    try:
        completed = subprocess.run(
            [str(interpreter), str(ENTRYPOINT), *sys.argv[1:]],
            check=False,
        )
    except OSError:
        return
    raise SystemExit(completed.returncode)


def _spawn_keep_alive(endpoint: str, session_name: str, auth_provider=None, config_path=None) -> int:
    """Start keep-alive through this wrapper so it shares CLI state safely.

    The daemon is spawned by the CLI as part of provisioning, so it is never
    denied here: refusing it stops `colab new` from working at all.  Whether a
    VM is RETAINED after the work ends is the launcher's decision, enforced by
    its teardown path and by the operator running `colab stop`.
    """
    command = [sys.executable, str(ENTRYPOINT)]
    if auth_provider is not None:
        command.append(f"--auth={auth_provider.value}")
    if config_path is not None:
        command.extend(["--config", config_path])
    command.extend(["keep-alive", endpoint, session_name])
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def main() -> None:
    # The CLI's native dependencies are built for its own interpreter, so a
    # wrapper started under a different python hands over before importing.
    _reexec_under_colab_cli_python()
    import colab_cli.common as common
    from colab_cli.history import HistoryLogger

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    common.state._history = HistoryLogger(str(HISTORY_DIR))
    # The upstream CLI currently creates ~/.config/colab-cli/colab.log even
    # with --logtostderr. The launcher owns the durable training log instead.
    common.setup_logging = lambda _log_to_stderr: None

    import colab_cli.commands.session as session_commands

    session_commands.spawn_keep_alive = _spawn_keep_alive

    from colab_cli.cli import app

    sys.argv[0] = "colab"
    app()


if __name__ == "__main__":
    main()
