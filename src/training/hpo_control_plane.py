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


@dataclass(frozen=True)
class HpoStorage:
    """Validated shared Optuna storage configuration."""

    url: str
    heartbeat_seconds: int = _HEARTBEAT_SECONDS
    grace_seconds: int = _GRACE_SECONDS


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


def fail_stale_trials(study) -> None:
    """Mark heartbeat-expired RUNNING trials failed before scheduling work."""
    import optuna

    optuna.storages.fail_stale_trials(study)


def generation_study_name(*, generation_id: str, model_key: str) -> str:
    """No study can cross an HPO generation or a model boundary."""
    if not generation_id or not model_key:
        raise ValueError("generation_id and model_key are required")
    return f"euromonitor::{generation_id}::{model_key}"
