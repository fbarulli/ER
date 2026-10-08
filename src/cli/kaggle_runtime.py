"""Configuration, credentials, paths, and CLI process execution."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from cli.log_capture import progress_frames_to_lines

# One fresh lane.log per run: the first write of this process truncates, later
# writes append (owner order 2026-10-07: overwrite, never append-sprawl).
_LANE_LOG_STARTED = False



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
        """The lane's transcript directory: the config SSOT `kaggle.logs_dir`
        resolved relative to TRAIN_ROOT.

        Local per-lane logs (live lane log, SSE stream captures, fetched session
        logs) live at logs/kaggle/ — never under results/kaggle_lane (receipts,
        zip payloads, staging only). Derived from this module's TRAIN_ROOT so
        tests can re-point the roof; the default and subdir are log_capture's
        one-roof convention.
        """
        from cli import kaggle_lane as lane

        return (lane.TRAIN_ROOT / lane._spec().logs_dir).resolve()

    @staticmethod
    def cohort_label(dataset_csv: Path) -> str:
        """Cohort tag mirroring cli.colab_data_bundle_prep.cohort_label values.

        `full` for the SSOT default export, `50pct` for the half-cohort, else a
        sanitized stem. The SAME tags keep the two lanes' transcripts mutually
        attributable without importing the colab module.
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
        entry whose filename carries the tag (50pct). Fail-loud on unknown.
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

        return (f"[kaggle-lane {lane.datetime.now(ZoneInfo(lane._spec().limits.timezone)):%Y-%m-%dT%H:%M:%S %Z}]")

    @staticmethod
    def _log_lane(line: str) -> None:
        """Timestamped lane logging: console plus one fresh lane log per run.

        The file is truncated on the first write of this process and appended
        afterwards, so a new run writes over the previous run's transcript
        (owner order 2026-10-07: fresh file per run, never append-sprawl).
        A detached watcher spawned by a push shares the pusher's run transcript:
        the pusher's first write opened it fresh and ER_KAGGLE_LANE_APPEND=1
        tells the child process to append (never truncate again).
        Best-effort on the file side — a log-write failure is printed and never
        allowed to mask the operation's own outcome.
        """
        global _LANE_LOG_STARTED
        from cli import kaggle_lane as lane

        stamp = f"{lane.datetime.now(ZoneInfo(lane._spec().limits.timezone)):%Y-%m-%dT%H:%M:%S %Z}"
        print(f"[kaggle-lane {stamp}] {line}", flush=True)
        try:
            log_dir = lane.lane_logs_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            append = os.environ.get("ER_KAGGLE_LANE_APPEND") == "1"
            mode = "a" if (_LANE_LOG_STARTED or append) else "w"
            with (log_dir / lane.LANE_LOG_NAME).open(mode, encoding="utf-8") as handle:
                handle.write(f"{stamp} {line}\n")
            _LANE_LOG_STARTED = True
        except OSError as error:
            print(lane._stamp(), f"[kaggle-lane] lane.log write failed ({error}); continuing",
                  flush=True)

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

        Same semantics as cli.colab._env_value: no printing (secrets stay out of
        every log), env-var override first, never cloned into the repo. ``root``
        defaults to the lane's TRAIN_ROOT (the staging knob tests and alternate
        checkouts re-point); a caller passes the repository root explicitly for
        a value whose lookup must survive that redirection.
        """
        from cli import kaggle_lane as lane

        search_root = lane.TRAIN_ROOT if root is None else Path(root)
        for env_path in (search_root / ".env", search_root.parent / ".env"):
            if not env_path.is_file():
                continue
            for line in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == name:
                    value = value.strip().strip('"').strip("'")
                    if value:
                        return value
        return None

    @staticmethod
    def write_credentials(*, key_env: str | None = None, execute: bool) -> dict[str, Any]:
        """Materialize ~/.kaggle/kaggle.json from the environment.

        The token never enters this repository or argv: it is read from the
        config-named environment variable (overridable with --key-env) and
        written to the standard credential path with 0600 permissions. Dry run
        by default; --execute writes the file.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved_env = key_env or spec.api_key_env
        if not spec.username:
            raise RuntimeError(
                "config kaggle.username is unset; name the Kaggle account before "
                "writing credentials")
        token = os.environ.get(resolved_env, "").strip()
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "target": str(lane.CREDENTIALS_PATH),
            "username": spec.username,
            "key_env": resolved_env,
            "key_present": bool(token),
        }
        if not execute:
            return plan
        if not token:
            raise RuntimeError(
                f"environment variable {resolved_env!r} is empty or unset; export "
                "the Kaggle API token (credentials never live in this repository)")
        lane.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        lane.CREDENTIALS_PATH.write_text(
            json.dumps({"username": spec.username, "key": token}) + "\n",
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
        plan["written"] = True
        return plan

