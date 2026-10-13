"""Read-only-home-safe entry point for the installed Colab CLI.

The launcher invokes this with the Colab CLI's own Python interpreter. Keeping
the wrapper in the repository makes the CLI process and every detached child it
starts use the same writable state, history, and logging behavior.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
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
def _canonical_state_root(base: Path) -> Path:
    """Canonical checkout home: git's COMMON dir is shared by main + worktrees.

    Mirrors core.project_root.ProjectRoot.canonical, but this entry runs under
    the uv CLI interpreter where `core` is not importable — no dependency here.
    One name->endpoint registry across every lane (visibility fix, 58f76fe).
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--git-common-dir"],
                             cwd=base, capture_output=True, text=True, timeout=10)
        common = out.stdout.strip()
        if out.returncode == 0 and common:
            return Path(common).resolve().parent
    except (OSError, subprocess.SubprocessError):
        pass
    return base

STATE_DIR = _canonical_state_root(ENCLOSURE_ROOT) / "colab_cli_state"
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


def _token_config_path(upstream_path: str) -> str:
    """Reuse existing upstream auth; retain project auth when already present."""
    project_token = STATE_DIR / "token.json"
    if project_token.is_file():
        return str(project_token)
    if Path(upstream_path).is_file():
        return upstream_path
    return str(project_token)


def _attempt(action, description: str) -> None:
    """Run one best-effort teardown step, logging any failure in full."""
    try:
        action()
    except Exception:
        logging.exception("colab-cli entrypoint: %s failed", description)


def _harden_runtime_stop(runtime_module) -> None:
    """Close the kernel client's channels even when its manager is gone.

    Upstream ``ColabRuntime.stop()`` reaches straight into
    ``self._kernel_client._manager.client``.  ``_manager`` can be None (a lost
    connection tears it down), and then the AttributeError is caught and logged
    while EVERY close in the same ``try`` is skipped -- the exec process keeps
    its websocket open, never exits, and the launcher waits out the full stage
    timeout before killing it.  Close whatever still exists, each step guarded
    and logged with its traceback, and never raise from teardown.
    """
    runtime_class = runtime_module.ColabRuntime
    if getattr(runtime_class, "_er_hardened_stop", False):
        return

    def stop(self, shutdown_kernel: bool = False) -> None:
        client = getattr(self, "_kernel_client", None)
        if client is None:
            return
        manager = getattr(client, "_manager", None)
        channel_client = None
        if manager is not None:
            try:
                channel_client = manager.client
            except Exception:
                logging.exception("colab-cli entrypoint: manager.client unavailable")
        if channel_client is not None:
            _attempt(channel_client.stop_channels, "stop_channels")
            kernel_socket = getattr(channel_client, "kernel_socket", None)
            if kernel_socket is not None:
                _attempt(kernel_socket.close, "kernel_socket.close")
        else:
            # No manager-owned channel client to close: fall back to the
            # client's own stop path so a fork with another close route still
            # runs it. ``_own_kernel`` is False, so no kernel is shut down here.
            _attempt(lambda: client.stop(shutdown_kernel=False), "kernel client stop")
        if shutdown_kernel and manager is not None:
            _attempt(lambda: manager.shutdown_kernel(now=True), "shutdown_kernel")

    runtime_class.stop = stop
    runtime_class._er_hardened_stop = True


def _release_session(session_name: str) -> int:
    """Unassign one session server-side, independent of the CLI's own stop()."""
    from colab_cli.common import state

    record = state.store.get(session_name)
    endpoint = getattr(record, "endpoint", None)
    verdict: dict[str, object] = {"session": session_name, "endpoint": endpoint}
    if not endpoint:
        verdict.update(released=False, reason="no local session record")
        print(json.dumps(verdict))
        return 0
    try:
        assigned = {assignment.endpoint for assignment in state.client.list_assignments()}
    except BaseException as exc:
        traceback.print_exc()
        verdict.update(released=False, reason=f"list_assignments failed: {exc!r}")
        print(json.dumps(verdict))
        return 1
    if endpoint in assigned:
        try:
            state.client.unassign(endpoint)
        except BaseException as exc:
            traceback.print_exc()
            verdict.update(released=False, reason=f"unassign failed: {exc!r}")
            print(json.dumps(verdict))
            return 1
        verdict.update(released=True, action="unassigned")
    else:
        verdict.update(released=True, action="already_absent")
    state.store.remove(session_name)
    print(json.dumps(verdict))
    return 0


def _register_release_command(app) -> None:
    """Add the launcher's own stop-by-name command to the CLI app.

    ``colab stop`` depends on the CLI's local record and its kernel-client
    teardown; this command releases the VM server-side from the recorded
    endpoint, so a failed CLI stop cannot leave the VM held open.
    """
    import typer

    @app.command(name="release-session")
    def release_session(
        session: str = typer.Option(..., "-s", "--session", help="Session name"),
    ) -> None:
        """Release a session by name without the CLI's stop() path."""
        raise typer.Exit(_release_session(session))


def main() -> None:
    # The CLI's native dependencies are built for its own interpreter, so a
    # wrapper started under a different python hands over before importing.
    _reexec_under_colab_cli_python()
    import colab_cli.auth as auth
    import colab_cli.auto_update as auto_update
    import colab_cli.common as common
    from colab_cli.history import HistoryLogger

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve an existing login instead of redirecting the CLI to an empty
    # project store. New logins still use the writable project state directory.
    auth.TOKEN_CONFIG_PATH = _token_config_path(auth.TOKEN_CONFIG_PATH)
    common.state._history = HistoryLogger(str(HISTORY_DIR))
    # The upstream CLI currently creates ~/.config/colab-cli/colab.log even
    # with --logtostderr. The launcher owns the durable training log instead.
    common.setup_logging = lambda _log_to_stderr: None

    # Upstream shim (1013): jupyter_kernel_client's KernelClient.stop() reads
    # self._manager, which only the LOCAL subclass sets; the CLI stops plain
    # remote clients on channel teardown and every re-attach spammed
    # AttributeError + traceback lines into the transcript.
    try:
        from jupyter_kernel_client.client import KernelClient as _KC

        def _ensure_manager_attr(self):
            if not hasattr(self, "_manager"):
                self._manager = None

        _orig_stop = _KC.stop

        def _safe_stop(self):
            _ensure_manager_attr(self)
            return _orig_stop(self)

        _KC.stop = _safe_stop
        _orig_close = _KC.close

        def _safe_close(self):
            _ensure_manager_attr(self)
            return _orig_close(self)

        _KC.close = _safe_close
    except (ImportError, AttributeError):
        pass

    # ``colab`` otherwise performs a daily update lookup for every `exec`,
    # `upload`, and `download`.  The launcher makes many short-lived control
    # calls, has just been installed from the upstream Git revision, and pins
    # that revision for a reproducible run; omit only that nonessential check.
    auto_update.run_background_check = lambda: None

    import colab_cli.runtime as runtime_module

    _harden_runtime_stop(runtime_module)

    from colab_cli.cli import app

    _register_release_command(app)

    sys.argv[0] = "colab"
    app()


if __name__ == "__main__":
    main()
