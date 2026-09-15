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


def _spawn_keep_alive(endpoint: str, session_name: str, auth_provider=None, config_path=None) -> int:
    """Start keep-alive through this wrapper so it shares CLI state safely.

    Refused unless the launcher marked this session keep-alive-eligible.  A
    retained GPU VM consumes accelerator quota for its entire lifetime, so the
    default here is deny: only an explicitly CPU launch may leave a VM running.
    """
    if os.environ.get("EUROMONITOR_KEEP_ALIVE_ALLOWED") != "1":
        raise RuntimeError(
            "refusing to start Colab keep-alive: this session was not marked "
            "keep-alive-eligible by the launcher (GPU sessions must never "
            "retain their VM)"
        )
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
