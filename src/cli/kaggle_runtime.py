"""Configuration, credentials, paths, and CLI process execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from cli.log_capture import progress_frames_to_lines

# One fresh lane.log per run: the first write of this process truncates, later
# writes append (owner order 2026-10-07: overwrite, never append-sprawl).
_LANE_LOG_STARTED = False
# Autowatch/supervise call _log_lane from the main AND the follower thread;
# without this lock two racing first writes each open "w" and truncate twice,
# wiping the run's transcript before it starts.
_LANE_LOG_LOCK = threading.Lock()



class KaggleRuntime:
    """Configuration, credentials, paths, and CLI process execution."""

    @staticmethod
    def _spec():
        from cli import kaggle_lane as lane

        return lane.training_cfg().kaggle

    @staticmethod
    def staging_dir() -> Path:
        from cli import kaggle_lane as lane

        return (lane.TRAIN_ROOT / lane._spec().staging_dir).resolve()

    @staticmethod
    def lane_logs_dir() -> Path:
        """The lane's transcript directory under the SSOT logs root.

        Local per-lane logs (live lane log, SSE stream captures, fetched session
        logs) live at ``<logs_root>/<kaggle.logs_dir>`` — never under
        results/kaggle_lane (receipts, zip payloads, staging only). The lane
        name is ``kaggle.logs_dir`` and the root is ``paths.logs_dir``, composed
        by ``cli.log_capture`` (canonical, main-worktree rooted).
        """
        from cli import kaggle_lane as lane
        from cli.log_capture import lane_dir

        return lane_dir(lane._spec().logs_dir, lane.TRAIN_ROOT)

    @staticmethod
    def lane_log_path() -> Path:
        """The ONE run transcript both kaggle-family writers append to.

        The path ``_log_lane`` writes (``lane_logs_dir()``/``LANE_LOG_NAME``,
        both declared once in config: ``kaggle.logs_dir`` under
        ``paths.logs_dir`` plus ``kaggle.files.lane_log``); the monitor's
        streamers take it as their default destination so the follower and the
        transcript are the same file, truncated once by the pusher's first
        ``_log_lane`` and appended thereafter.
        """
        from cli import kaggle_lane as lane

        return (lane.lane_logs_dir() / lane.LANE_LOG_NAME).resolve()

    @staticmethod
    def cohort_label(dataset_csv: Path) -> str:
        """Cohort tag mirroring cli.colab_data_bundle_prep.cohort_label values.

        `full` for the SSOT default export, the export's declared tag for any
        other committed export, else a sanitized stem. The SAME tags keep the
        two lanes' transcripts mutually attributable without importing the colab
        module.
        """
        from cli import kaggle_lane as lane

        name = Path(dataset_csv).name
        spec = lane._spec()
        for tag, export in zip(spec.cohort_tags, spec.export_csvs, strict=True):
            if name == Path(export).name:
                return tag
        return "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0]
        )

    @staticmethod
    def cohort_export_csv(cohort: str) -> str:
        """Root-relative export filename for a cohort tag (inverse cohort_label).

        `full` = the SSOT default export; any other tag must name an export_csvs
        entry (a root-relative, self-describing filename). Fail-loud on unknown.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        exports = dict(zip(spec.cohort_tags, spec.export_csvs, strict=True))
        if cohort in exports:
            return exports[cohort]
        raise RuntimeError(
            f"no export_csvs entry matches cohort {cohort!r}; add it to config "
            "kaggle.export_csvs (root-relative, self-describing filename)")

    @staticmethod
    def _git_revision() -> str:
        from cli import kaggle_lane as lane

        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=lane.TRAIN_ROOT,
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"git rev-parse failed in {lane.TRAIN_ROOT}: {result.stderr.strip()}")
        return result.stdout.strip()

    @staticmethod
    def _stamp() -> str:
        """Bracketed Europe/Paris (CET/CEST) wall-clock prefix for output."""
        from cli import kaggle_lane as lane
        from cli.colab_lane_contracts import _stamp as _lane_stamp

        return _lane_stamp(
            "kaggle-lane",
            now=lane.datetime.now(ZoneInfo(lane._spec().limits.timezone)),
        )

    @staticmethod
    def mark_lane_log_started() -> None:
        """Claim the run's single truncate for a writer that opens the transcript
        before any _log_lane call (the stream follower): its later _log_lane
        writes (reconnect lines) must append, never truncate what was streamed."""
        global _LANE_LOG_STARTED
        with _LANE_LOG_LOCK:
            _LANE_LOG_STARTED = True

    @staticmethod
    def _log_lane(line: str) -> None:
        """Timestamped lane logging: console plus one fresh lane log per run.

        The file is truncated on the first write of this process and appended
        afterwards, so a new run writes over the previous run's transcript
        (owner order 2026-10-07: fresh file per run, never append-sprawl).
        A detached watcher spawned by a push shares the pusher's run transcript:
        the pusher's first write opened it fresh and ER_KAGGLE_LANE_APPEND=1
        tells the child process to append (never truncate again).
        Best-effort on the file side — a log-write failure is printed with its
        full traceback and never allowed to mask the operation's own outcome.
        """
        global _LANE_LOG_STARTED
        from cli import kaggle_lane as lane

        stamp = f"{lane.datetime.now(ZoneInfo(lane._spec().limits.timezone)):%Y-%m-%dT%H:%M:%S %Z}"
        print(f"[kaggle-lane {stamp}] {line}", flush=True)
        try:
            log_dir = lane.lane_logs_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            append = os.environ.get("ER_KAGGLE_LANE_APPEND") == "1"
            with _LANE_LOG_LOCK:
                mode = "a" if (_LANE_LOG_STARTED or append) else "w"
                with (log_dir / lane.LANE_LOG_NAME).open(mode, encoding="utf-8") as handle:
                    handle.write(f"{stamp} {line}\n")
                _LANE_LOG_STARTED = True
        except OSError as error:
            print(lane._stamp(), f"[kaggle-lane] lane.log write failed ({error}); "
                  f"continuing\n{traceback.format_exc()}", flush=True)

    @staticmethod
    def _require_kaggle_executable(executable: str) -> str:
        from cli import kaggle_lane as lane

        if lane.ACCESS_TOKEN_PATH.exists():
            raise RuntimeError(
                "~/.kaggle/access_token exists; the kaggle CLI prefers it over "
                "kaggle.json and it is not this lane's working credential "
                "(every kernels.* call then fails 'kernels.get denied'). Delete "
                "or rename it — kaggle.json is the single credential.")
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(
                f"kaggle executable {executable!r} not found on PATH; install the "
                "kaggle CLI and place credentials at ~/.kaggle/kaggle.json "
                "(never inside this repository)"
            )
        return resolved

    @staticmethod
    def _run_kaggle(command: list[str]) -> tuple[int, str]:
        """Run the kaggle CLI with full logging; never swallow its output.

        stdout and stderr are captured together, echoed line by line, appended
        to the lane log, and — on a failing returncode — embedded verbatim in
        the raised RuntimeError so Kaggle's own diagnostics always surface.
        """
        from cli import kaggle_lane as lane

        printable = " ".join(command)
        lane._log_lane(f"$ {printable}")
        started = time.monotonic()
        result = subprocess.run(command, cwd=lane.TRAIN_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        elapsed = time.monotonic() - started
        output = progress_frames_to_lines(result.stdout or "")
        for line in output.splitlines():
            print(f"[kaggle] {line}", flush=True)
        lane._log_lane(f"rc={result.returncode} seconds={elapsed:.1f}")
        if result.returncode != 0:
            tail = output.strip()[-lane._spec().limits.command_error_tail_chars:] or "(kaggle produced no output)"
            raise RuntimeError(
                f"kaggle command failed (rc={result.returncode}): {printable}\n"
                f"--- kaggle output ---\n{tail}")
        return result.returncode, output

    @staticmethod
    def _env_dot_value(name: str, root: Path | None = None) -> str | None:
        """Read a simple KEY=VALUE from <root>/.env or its parent .env.

        One reader for every ``.env`` lookup in the repo (``core.env_file``): no
        printing (secrets stay out of every log), never cloned into the repo.
        ``root`` defaults to the lane's TRAIN_ROOT (the staging knob tests and
        alternate checkouts re-point); a caller passes the repository root
        explicitly for a value whose lookup must survive that redirection.
        """
        from cli import kaggle_lane as lane
        from core.env_file import EnvFile

        return EnvFile.value(name, root=lane.TRAIN_ROOT if root is None else Path(root))

    @staticmethod
    def write_credentials(*, key_env: str | None = None, execute: bool) -> dict[str, Any]:
        """Materialize ~/.kaggle/kaggle.json from the credential SSOT.

        The token never enters this repository or argv: it is read through the
        canonical owner (core.credentials.CredentialStore) — the config-declared
        env var, process env first, the declared env file second — and written
        to the standard credential path with 0600 permissions. Dry run by
        default; --execute writes the file.
        """
        from cli import kaggle_lane as lane
        from core.credentials import CredentialStore

        spec = lane._spec()
        store = CredentialStore.from_config(root=lane.TRAIN_ROOT)
        resolved_env = key_env or store.spec.keys.kaggle_api_key
        if not spec.username:
            raise RuntimeError(
                "config kaggle.username is unset; name the Kaggle account before "
                "writing credentials")
        token = store.resolve_env_optional(resolved_env)
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "target": str(lane.CREDENTIALS_PATH),
            "username": spec.username,
            "key_env": resolved_env,
            "key_present": token is not None,
        }
        if not execute:
            return plan
        # fail loud with the full traceback when the required key is absent
        secret = store.resolve_env(resolved_env)
        lane.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        lane.CREDENTIALS_PATH.write_text(
            json.dumps({"username": spec.username,
                        "key": secret.get_secret_value()}) + "\n",
            encoding="utf-8")
        lane.CREDENTIALS_PATH.chmod(0o600)
        if lane.ACCESS_TOKEN_PATH.exists():
            # The 2.x CLI prefers the token file over kaggle.json and the CLI
            # token is not this lane's working credential: a stale/hub token
            # here makes every kernels.* call fail with 'kernels.get was
            # denied'. The lane standard is kaggle.json only (owner ruling
            # 2026-10-07 after the determinism fix), so a token file is an
            # operator error, not something the lane restores or overwrites.
            plan["written"] = True
            plan["blocked_by_access_token"] = str(lane.ACCESS_TOKEN_PATH)
            return plan
        if lane.OAUTH_CREDENTIALS_PATH.exists():
            # kaggle CLI 2.x authenticates from the OAuth credentials file FIRST
            # (kagglesdk KaggleCredentials -> ~/.kaggle/credentials.json), so it
            # shadows the kaggle.json just written. Report that the file in force
            # is the OAuth one instead of a silent success.
            plan["written"] = True
            plan["shadowed_by_oauth_credentials"] = str(lane.OAUTH_CREDENTIALS_PATH)
            return plan
        plan["written"] = True
        return plan
