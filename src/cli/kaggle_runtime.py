"""Configuration, credentials, paths, and CLI process execution."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from cli.log_capture import LaneTranscript, progress_frames_to_lines


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

        Local per-lane state (fetched session logs, session-id handles,
        follower locks) lives here — never under results/kaggle_lane (receipts,
        zip payloads, staging only). Derived from this module's TRAIN_ROOT so
        tests can re-point the roof.
        """
        from cli import kaggle_lane as lane

        return LaneTranscript.roof_for(lane.TRAIN_ROOT)

    @staticmethod
    def lane_log_path() -> Path:
        """The ONE run transcript both kaggle-family lanes append to.

        Path declared once in config (``kaggle.logs_dir`` +
        ``kaggle.files.lane_log``); this is the only resolution site ER reads.
        """
        from cli import kaggle_lane as lane

        return LaneTranscript.path_for(lane.TRAIN_ROOT)

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
        """Timestamped lane logging: console plus the ONE shared run transcript.

        Every kaggle-family writer appends to ``kaggle.files.lane_log`` under
        ``kaggle.logs_dir`` (the laya lane included), truncated once at run
        start and appended thereafter — a detached watcher spawned by a push
        (``ER_KAGGLE_LANE_APPEND=1``) only ever appends.
        """
        from cli import kaggle_lane as lane

        LaneTranscript.from_config(
            lane.TRAIN_ROOT, lane="kaggle-lane",
            stamp=lambda: (f"{lane.datetime.now(ZoneInfo(lane._spec().limits.timezone)):%Y-%m-%dT%H:%M:%S %Z}"),
        ).write(line)

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

