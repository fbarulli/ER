#!/usr/bin/env python3
"""Validate direct and bridged PostgreSQL access plus concurrent Optuna writes.

This script assumes PostgreSQL and the local TCP bridge already exist.  It does
not install packages, start services, create databases, or delete test studies.
Connection strings and credential-like environment values are redacted from all
diagnostics, including child-process tracebacks.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable
from urllib.parse import unquote, urlsplit


DIRECT_URL_ENV = "HPO_POSTGRES_HOST_URL"
BRIDGE_URL_ENV = "HPO_POSTGRES_BRIDGE_URL"
CLIENTS = 3
INITIAL_TRIALS = 5
TRIALS_PER_CLIENT = 3
SECRET_ENV_MARKERS = ("PASSWORD", "PASSWD", "SECRET", "TOKEN", "API_KEY", "AUTH_KEY")


def _secret_values(*urls: str) -> set[str]:
    values = {value for key, value in os.environ.items() if value and any(marker in key.upper() for marker in SECRET_ENV_MARKERS)}
    for url in urls:
        if not url:
            continue
        values.add(url)
        parsed = urlsplit(url)
        if parsed.password:
            values.add(parsed.password)
            values.add(unquote(parsed.password))
    return {value for value in values if len(value) >= 4}


def _redact(text: object, secrets: set[str]) -> str:
    safe = str(text)
    for secret in sorted(secrets, key=len, reverse=True):
        safe = safe.replace(secret, "<redacted>")
    # Catch a DSN embedded in a library error even if it was normalized first.
    safe = re.sub(
        r"(?i)(postgres(?:ql)?(?:\+[a-z0-9_]+)?://)([^\s/@:]+)(?::[^\s/@]*)?@",
        r"\1<redacted>@",
        safe,
    )
    return safe


def _format_exception(secrets: set[str]) -> str:
    return _redact("".join(traceback.format_exception(*sys.exc_info())), secrets)


@dataclass(frozen=True)
class PgTarget:
    host: str
    port: int
    user: str
    database: str
    password: str | None
    sslmode: str | None


def _parse_postgres_url(url: str, env_name: str) -> PgTarget:
    parsed = urlsplit(url)
    if parsed.scheme not in {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}:
        raise ValueError(f"{env_name} must use a PostgreSQL URL scheme")
    if not parsed.hostname or not parsed.username or not parsed.path.strip("/"):
        raise ValueError(f"{env_name} must include host, user, and database")
    query = {}
    for item in parsed.query.split("&") if parsed.query else ():
        key, _, value = item.partition("=")
        query[unquote(key)] = unquote(value)
    return PgTarget(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=unquote(parsed.username),
        database=unquote(parsed.path.lstrip("/")),
        password=unquote(parsed.password) if parsed.password is not None else None,
        sslmode=query.get("sslmode"),
    )


def _psql_scalar(url: str, env_name: str, sql: str, secrets: set[str]) -> str:
    if shutil.which("psql") is None:
        raise RuntimeError("psql is required but was not found on PATH")
    target = _parse_postgres_url(url, env_name)
    command = [
        "psql",
        "--no-password",
        "--no-psqlrc",
        "--quiet",
        "--tuples-only",
        "--no-align",
        "--set=ON_ERROR_STOP=1",
        f"--host={target.host}",
        f"--port={target.port}",
        f"--username={target.user}",
        f"--dbname={target.database}",
        "--command",
        sql,
    ]
    process_env = os.environ.copy()
    process_env["PGCONNECT_TIMEOUT"] = process_env.get("HPO_E2E_CONNECT_TIMEOUT", "15")
    if target.password is not None:
        process_env["PGPASSWORD"] = target.password
    if target.sslmode:
        process_env["PGSSLMODE"] = target.sslmode
    result = subprocess.run(command, env=process_env, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        detail = _redact((result.stderr or result.stdout).strip(), secrets)
        raise RuntimeError(f"psql failed with exit code {result.returncode}: {detail}")
    return result.stdout.strip()


def _objective(trial: object) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    return float((x - 2.0) ** 2)


def _concurrent_client(
    storage_url: str, study_name: str, client_id: int, output: mp.Queue
) -> None:
    secrets = _secret_values(storage_url)
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        study.optimize(_objective, n_trials=TRIALS_PER_CLIENT, catch=())
        output.put({"ok": True, "client": client_id})
    except Exception:
        output.put({"ok": False, "client": client_id, "traceback": _format_exception(secrets)})


def _run_layer(name: str, action: Callable[[], None], failures: list[str], secrets: set[str]) -> None:
    try:
        action()
    except Exception:
        failures.append(name)
        print(f"[FAIL] {name}\n{_format_exception(secrets)}", flush=True)
    else:
        print(f"[PASS] {name}", flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PostgreSQL and Optuna through an existing local bridge."
    )
    parser.add_argument(
        "--phase",
        choices=("all", "host", "bridge"),
        default="all",
        help=(
            "host runs the direct SQL check; bridge runs bridged SQL and Optuna; "
            "all runs every layer in one environment (default)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    direct_url = os.environ.get(DIRECT_URL_ENV, "")
    bridge_url = os.environ.get(BRIDGE_URL_ENV, "")
    secrets = _secret_values(direct_url, bridge_url)
    required = []
    if args.phase in {"all", "host"}:
        required.append((DIRECT_URL_ENV, direct_url))
    if args.phase in {"all", "bridge"}:
        required.append((BRIDGE_URL_ENV, bridge_url))
    missing = [name for name, value in required if not value]
    if missing:
        print(f"[FAIL] configuration: missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    prefix = os.environ.get("HPO_E2E_STUDY_PREFIX", "euromonitor_bridge_e2e")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", prefix):
        print("[FAIL] configuration: HPO_E2E_STUDY_PREFIX must contain only letters, numbers, '_' or '-'", file=sys.stderr)
        return 2
    study_name = ""
    if args.phase in {"all", "bridge"}:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        study_name = f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    failures: list[str] = []

    def host_sql() -> None:
        if _psql_scalar(direct_url, DIRECT_URL_ENV, "SELECT 1;", secrets) != "1":
            raise AssertionError("direct-host SELECT 1 did not return 1")

    def bridge_sql() -> None:
        if _psql_scalar(bridge_url, BRIDGE_URL_ENV, "SELECT 1;", secrets) != "1":
            raise AssertionError("bridged SELECT 1 did not return 1")

    def optuna_five() -> None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(study_name=study_name, storage=bridge_url, direction="minimize")
        study.optimize(_objective, n_trials=INITIAL_TRIALS, catch=())
        reloaded = optuna.load_study(study_name=study_name, storage=bridge_url)
        complete = [trial for trial in reloaded.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if len(reloaded.trials) != INITIAL_TRIALS or len(complete) != INITIAL_TRIALS:
            raise AssertionError(f"expected {INITIAL_TRIALS} complete trials after reload, got {len(complete)}/{len(reloaded.trials)}")
        if "x" not in reloaded.best_params:
            raise AssertionError("reloaded study has no best x parameter")

    def concurrent_optuna() -> None:
        import optuna

        context = mp.get_context("spawn")
        output: mp.Queue = context.Queue()
        processes = [
            context.Process(target=_concurrent_client, args=(bridge_url, study_name, client_id, output))
            for client_id in range(CLIENTS)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                raise TimeoutError("a concurrent Optuna client exceeded 180 seconds")
        try:
            reports = [output.get(timeout=5) for _ in processes]
        except queue.Empty as exc:
            exits = [process.exitcode for process in processes]
            raise RuntimeError(
                f"a concurrent Optuna client exited without a report; exit codes: {exits}"
            ) from exc
        child_failures = [report for report in reports if not report["ok"]]
        if child_failures:
            detail = "\n".join(f"client {item['client']}:\n{item['traceback']}" for item in child_failures)
            raise RuntimeError(f"concurrent Optuna client failure(s):\n{detail}")
        bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
        if bad_exits:
            raise RuntimeError(f"concurrent Optuna client exit codes were not zero: {bad_exits}")

        reloaded = optuna.load_study(study_name=study_name, storage=bridge_url)
        expected = INITIAL_TRIALS + CLIENTS * TRIALS_PER_CLIENT
        complete = [trial for trial in reloaded.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        numbers = [trial.number for trial in reloaded.trials]
        if len(reloaded.trials) != expected or len(complete) != expected:
            raise AssertionError(f"expected {expected} complete trials, got {len(complete)}/{len(reloaded.trials)}")
        if len(numbers) != len(set(numbers)):
            raise AssertionError("duplicate Optuna trial numbers detected")

    if args.phase in {"all", "host"}:
        _run_layer("direct host PostgreSQL SELECT 1", host_sql, failures, secrets)
    if args.phase in {"all", "bridge"}:
        prior_failure_count = len(failures)
        _run_layer("bridged PostgreSQL SELECT 1", bridge_sql, failures, secrets)
        if len(failures) != prior_failure_count:
            print("[SKIP] Optuna layers require bridged PostgreSQL connectivity", flush=True)
        else:
            prior_failure_count = len(failures)
            _run_layer("Optuna create + 5-trial write/read", optuna_five, failures, secrets)
            if len(failures) != prior_failure_count:
                print("[SKIP] concurrent Optuna layer requires the 5-trial layer to pass", flush=True)
            else:
                _run_layer(
                    "3 concurrent Optuna clients on one study",
                    concurrent_optuna,
                    failures,
                    secrets,
                )

    if failures:
        print(f"[RESULT] FAIL ({len(failures)} layer(s)): {', '.join(failures)}", flush=True)
        return 1
    if args.phase in {"all", "bridge"}:
        expected = INITIAL_TRIALS + CLIENTS * TRIALS_PER_CLIENT
        print(f"[RESULT] PASS: study={study_name}; trials={expected}", flush=True)
        print("[NOTE] The test study is intentionally retained for host-side inspection.", flush=True)
    else:
        print("[RESULT] PASS: direct host PostgreSQL is reachable", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[FAIL] interrupted", file=sys.stderr)
        raise SystemExit(130)
