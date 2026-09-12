"""Read-only-home-safe entry point for the installed Colab CLI.

The launcher invokes this with the Colab CLI's own Python interpreter. Keeping
the wrapper in the repository makes the parent CLI process and its detached
keep-alive child use the same writable state, history, and logging behavior.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STATE_DIR = Path(__file__).resolve().parents[2] / "colab_cli_state"
HISTORY_DIR = STATE_DIR / "history"
ENTRYPOINT = Path(__file__).resolve()


def _spawn_keep_alive(endpoint: str, session_name: str, auth_provider=None, config_path=None) -> int:
    """Start keep-alive through this wrapper so it shares CLI state safely."""
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
