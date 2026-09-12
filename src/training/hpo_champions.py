"""Transactional champion registry owned by the PostgreSQL HPO control plane."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Champion:
    generation_id: str
    model_key: str
    trial_number: int
    value: float
    artifact_snapshot: str
    version: int


def _wins(*, value: float, trial_number: int, current: Champion | None) -> bool:
    """Stable ordering: higher score wins; equal score favours lower trial."""
    return current is None or (value, -trial_number) > (current.value, -current.trial_number)


class ChampionStore:
    """Compare-and-swap registry; workers never write champion files directly."""

    def __init__(self, url: str) -> None:
        from sqlalchemy import (
            Column,
            DateTime,
            Float,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
        )

        self._engine = create_engine(url, pool_pre_ping=True)
        metadata = MetaData()
        self._table = Table(
            "euromonitor_hpo_champions",
            metadata,
            Column("generation_id", String, primary_key=True),
            Column("model_key", String, primary_key=True),
            Column("trial_number", Integer, nullable=False),
            Column("value", Float, nullable=False),
            Column("artifact_snapshot", String, nullable=False),
            Column("version", Integer, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        self._leases = Table(
            "euromonitor_hpo_trial_leases",
            metadata,
            Column("generation_id", String, primary_key=True),
            Column("model_key", String, primary_key=True),
            Column("trial_number", Integer, primary_key=True),
            Column("epoch", Integer, nullable=False),
            Column("state", String, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        metadata.create_all(self._engine)

    def read(self, *, generation_id: str, model_key: str) -> Champion | None:
        from sqlalchemy import select

        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._table).where(
                    self._table.c.generation_id == generation_id,
                    self._table.c.model_key == model_key,
                )
            ).mappings().first()
        return None if row is None else Champion(
            generation_id=row["generation_id"], model_key=row["model_key"],
            trial_number=row["trial_number"], value=row["value"],
            artifact_snapshot=row["artifact_snapshot"], version=row["version"],
        )

    def promote(
        self, *, generation_id: str, model_key: str, trial_number: int,
        value: float, artifact_snapshot: str, lease_epoch: int, attempts: int = 8,
    ) -> Champion:
        """Atomically promote only a strictly better deterministic champion.

        The optimistic ``version`` predicate means completion ordering cannot
        overwrite a better result.  Call only after Optuna has committed the
        trial as COMPLETE and the referenced artifact snapshot is READY.
        """
        from sqlalchemy import select, update
        from sqlalchemy.dialects.postgresql import insert

        for _ in range(attempts):
            now = datetime.now(timezone.utc)
            # Fence and champion update share one transaction. A watchdog
            # revoking this epoch either happens before (we reject) or after
            # this commit (the already-accepted completion remains valid).
            with self._engine.begin() as conn:
                lease = conn.execute(select(self._leases).where(
                    self._leases.c.generation_id == generation_id,
                    self._leases.c.model_key == model_key,
                    self._leases.c.trial_number == trial_number,
                ).with_for_update()).mappings().one_or_none()
                if (lease is None or lease["epoch"] != lease_epoch
                        or lease["state"] != "active"):
                    raise RuntimeError(
                        f"stale HPO worker fenced before champion promotion: "
                        f"{model_key}/trial {trial_number} epoch {lease_epoch}"
                    )
                row = conn.execute(select(self._table).where(
                    self._table.c.generation_id == generation_id,
                    self._table.c.model_key == model_key,
                )).mappings().first()
                current = None if row is None else Champion(
                    generation_id=row["generation_id"], model_key=row["model_key"],
                    trial_number=row["trial_number"], value=row["value"],
                    artifact_snapshot=row["artifact_snapshot"], version=row["version"],
                )
                if not _wins(value=value, trial_number=trial_number, current=current):
                    assert current is not None
                    return current
                if current is None:
                    created = conn.execute(insert(self._table).values(
                        generation_id=generation_id, model_key=model_key,
                        trial_number=trial_number, value=value,
                        artifact_snapshot=artifact_snapshot, version=1,
                        updated_at=now,
                    ).on_conflict_do_nothing()).rowcount
                    if created:
                        return Champion(generation_id, model_key, trial_number, value, artifact_snapshot, 1)
                    continue
                changed = conn.execute(update(self._table).where(
                    self._table.c.generation_id == generation_id,
                    self._table.c.model_key == model_key,
                    self._table.c.version == current.version,
                ).values(
                    trial_number=trial_number, value=value,
                    artifact_snapshot=artifact_snapshot, version=current.version + 1,
                    updated_at=now,
                )).rowcount
                if changed:
                    return Champion(generation_id, model_key, trial_number, value, artifact_snapshot, current.version + 1)
            # A competing champion update won the version race; re-read and
            # deterministically compare on the next iteration.
            continue
        raise RuntimeError("champion compare-and-swap exceeded retry budget")
