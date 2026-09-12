"""PostgreSQL lease epochs that fence zombie HPO workers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class TrialLease:
    generation_id: str
    model_key: str
    trial_number: int
    epoch: int


class TrialLeaseStore:
    """The controller issues leases; workers may only act while current."""

    def __init__(self, url: str) -> None:
        from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine

        self._engine = create_engine(url, pool_pre_ping=True)
        metadata = MetaData()
        self._table = Table(
            "euromonitor_hpo_trial_leases", metadata,
            Column("generation_id", String, primary_key=True),
            Column("model_key", String, primary_key=True),
            Column("trial_number", Integer, primary_key=True),
            Column("epoch", Integer, nullable=False),
            Column("state", String, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        metadata.create_all(self._engine)

    def issue(self, *, generation_id: str, model_key: str, trial_number: int) -> TrialLease:
        """Issue a new epoch, invalidating any old worker for this trial."""
        from sqlalchemy import select, update
        from sqlalchemy.dialects.postgresql import insert

        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            row = conn.execute(select(self._table).where(
                self._table.c.generation_id == generation_id,
                self._table.c.model_key == model_key,
                self._table.c.trial_number == trial_number,
            ).with_for_update()).mappings().first()
            if row is None:
                created = conn.execute(insert(self._table).values(
                    generation_id=generation_id, model_key=model_key,
                    trial_number=trial_number, epoch=1, state="active", updated_at=now,
                ).on_conflict_do_nothing()).rowcount
                if created:
                    return TrialLease(generation_id, model_key, trial_number, 1)
                # A competing controller created the row; callers must retry
                # rather than silently accepting an unfenced lease.
                raise RuntimeError("concurrent lease creation; retry issue()")
            epoch = int(row["epoch"]) + 1
            conn.execute(update(self._table).where(
                self._table.c.generation_id == generation_id,
                self._table.c.model_key == model_key,
                self._table.c.trial_number == trial_number,
                self._table.c.epoch == row["epoch"],
            ).values(epoch=epoch, state="active", updated_at=now))
        return TrialLease(generation_id, model_key, trial_number, epoch)

    def assert_current(self, lease: TrialLease) -> None:
        """Fail closed: an expired/stale worker cannot publish anything."""
        from sqlalchemy import select

        with self._engine.connect() as conn:
            row = conn.execute(select(self._table.c.epoch, self._table.c.state).where(
                self._table.c.generation_id == lease.generation_id,
                self._table.c.model_key == lease.model_key,
                self._table.c.trial_number == lease.trial_number,
            )).one_or_none()
        if row is None or row.epoch != lease.epoch or row.state != "active":
            raise RuntimeError(
                f"stale HPO worker fenced: {lease.model_key}/trial {lease.trial_number} "
                f"epoch {lease.epoch}"
            )

    def revoke(self, lease: TrialLease) -> bool:
        """Fence an orphan before retrying/replacing its work."""
        from sqlalchemy import update

        with self._engine.begin() as conn:
            changed = conn.execute(update(self._table).where(
                self._table.c.generation_id == lease.generation_id,
                self._table.c.model_key == lease.model_key,
                self._table.c.trial_number == lease.trial_number,
                self._table.c.epoch == lease.epoch,
                self._table.c.state == "active",
            ).values(state="revoked", updated_at=datetime.now(timezone.utc))).rowcount
        return bool(changed)
