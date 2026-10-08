"""PostgreSQL-backed control-plane primitives for isolated HPO workers.

Trial workers must not use SQLite or own global promotion state.  PostgreSQL
is the canonical Optuna store; artifacts remain per-worker and are carried by
the durability transport separately.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


_HEARTBEAT_SECONDS = 60
_GRACE_SECONDS = 180
# A lease is renewed by the owning worker (``TrialLeaseStore.heartbeat``); if it
# is not renewed for this long it is stale and ``assert_current`` fails closed.
_LEASE_TTL_SECONDS = 300


class HpoInfrastructureError(BaseException):
    """A control-plane failure that must ABORT loud, never be a trial failure.

    It deliberately derives from ``BaseException`` (NOT ``Exception``): every
    worker runs ``study.optimize(..., catch=(Exception,))`` so that an individual
    trial that blows up (OOM, CUDA error, bad dials) is recorded FAIL and the
    sweep continues.  A broken lease/champion/budget/DB must NOT be swallowed
    that way — it would silently consume the whole budget as FAIL trials.  Being
    outside ``Exception`` means Optuna's ``catch`` cannot catch it and it
    propagates to abort the worker.
    """


@dataclass(frozen=True)
class HpoStorage:
    """Validated shared Optuna storage configuration."""

    url: str
    heartbeat_seconds: int = _HEARTBEAT_SECONDS
    grace_seconds: int = _GRACE_SECONDS
    lease_ttl_seconds: int = _LEASE_TTL_SECONDS


def storage_from_environment() -> HpoStorage:
    """Load the only supported concurrent-HPO control-plane endpoint.

    A SQLite URL is rejected deliberately: WAL/retries do not make nine
    competing writers production-safe.  Secrets stay exclusively in the
    environment, never in YAML, results, DVC, or generated manifests.
    """
    url = os.environ.get("OPTUNA_STORAGE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "OPTUNA_STORAGE_URL is required for concurrent HPO; expected a "
            "postgresql+psycopg:// URL supplied through the runtime secret store"
        )
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError(
            "OPTUNA_STORAGE_URL must be PostgreSQL; SQLite is not supported "
            "for concurrent HPO"
        )
    return HpoStorage(url=url)


def create_storage(config: HpoStorage):
    """Create Optuna RDB storage with crash heartbeats enabled."""
    import optuna

    return optuna.storages.RDBStorage(
        url=config.url,
        heartbeat_interval=config.heartbeat_seconds,
        grace_period=config.grace_seconds,
        engine_kwargs={"pool_pre_ping": True},
    )


def ensure_tables(engine, metadata) -> None:
    """Create every table idempotently (``CREATE TABLE IF NOT EXISTS``).

    ``MetaData.create_all`` checks existence then creates, so two workers
    starting concurrently on a fresh database can both pass the check and race
    the ``CREATE TABLE``.  Emitting ``if_not_exists`` DDL removes the race: the
    loser of the race is a no-op rather than a hard ``DuplicateTable`` error.
    """
    from sqlalchemy.schema import CreateTable

    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            conn.execute(CreateTable(table, if_not_exists=True))


def fail_stale_trials(study) -> None:
    """Mark heartbeat-expired RUNNING trials failed before scheduling work."""
    import optuna

    optuna.storages.fail_stale_trials(study)


def reap_stale_trials_for_study(study_name: str, storage) -> bool:
    """Reap stale RUNNING trials ONCE per session, before any worker starts.

    Every worker used to call ``fail_stale_trials`` at startup, so a worker
    could race a sibling's freshly-started trial.  A session's parent calls this
    once instead.  Returns ``False`` when the study does not exist yet (nothing
    to reap); a real storage error propagates loud.
    """
    import optuna

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except (KeyError, ValueError):
        return False
    fail_stale_trials(study)
    return True


def generation_study_name(*, generation_id: str, model_key: str) -> str:
    """No study can cross an HPO generation or a model boundary."""
    if not generation_id or not model_key:
        raise ValueError("generation_id and model_key are required")
    return f"euromonitor::{generation_id}::{model_key}"
