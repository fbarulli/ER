"""PostgreSQL lease epochs that fence zombie HPO workers.

A lease now has a real lifetime.  ``TrialLeaseStore.heartbeat`` renews its
``updated_at`` while the owning worker is alive; ``assert_current`` fails closed
once the lease has not been renewed for ``ttl_seconds`` (or its epoch/state is
superseded).  ``reap_expired`` marks abandoned active leases ``expired`` so a
replacement controller can act on them.  The guarantee this delivers:

    A worker may only publish a result (champion promotion) while it holds a
    lease that is still ACTIVE, still the newest epoch for its trial, AND not
    expired.  A partitioned/zombie worker whose heartbeat lapsed can no longer
    promote, even though Optuna never reuses its trial number.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


_LEASE_TTL_SECONDS = 300


def _shared(name: str):
    """Resolve a symbol from the earlier injected control-plane module."""
    symbol = globals().get(name)
    if symbol is not None:
        return symbol
    from training import hpo_control_plane

    return getattr(hpo_control_plane, name)


def _as_utc(value: datetime) -> datetime:
    """Normalise a DB timestamp to tz-aware UTC (naive values are UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def lease_expired(*, updated_at, now: datetime, ttl_seconds: int) -> bool:
    """True when a lease has not been renewed for longer than its TTL."""
    if updated_at is None:
        return True
    if ttl_seconds <= 0:
        return False
    return _as_utc(now) - _as_utc(updated_at) > timedelta(seconds=int(ttl_seconds))


def lease_status(*, lease, row, now: datetime, ttl_seconds: int):
    """Pure freshness decision shared by the store and its offline tests.

    Returns ``(ok, reason)``; ``reason`` is one of ``"missing"``/``"epoch"``/
    ``"state"``/``"expired"``/``None``.
    """
    if row is None:
        return False, "missing"
    if int(row["epoch"]) != int(lease.epoch):
        return False, "epoch"
    if str(row["state"]) != "active":
        return False, "state"
    if lease_expired(updated_at=row.get("updated_at"), now=now,
                     ttl_seconds=ttl_seconds):
        return False, "expired"
    return True, None


@dataclass(frozen=True)
class TrialLease:
    generation_id: str
    model_key: str
    trial_number: int
    epoch: int
    ttl_seconds: int = _LEASE_TTL_SECONDS


class LeaseHeartbeat:
    """Renews one lease in a daemon thread for the length of a trial.

    Fail-soft: a renewal error (a transient DB/network blip) never fails the
    trial.  If the worker is truly partitioned the renewals keep failing and the
    lease ages out, which is exactly what fences the zombie at promotion time.
    """

    def __init__(self, store, lease, *, interval_seconds=None,
                 clock=time.monotonic):
        self._store = store
        self._lease = lease
        ttl = int(getattr(lease, "ttl_seconds", _LEASE_TTL_SECONDS) or
                  _LEASE_TTL_SECONDS)
        self._interval = max(1, int(interval_seconds or ttl // 3 or 1))
        self._clock = clock
        self._stop = threading.Event()
        self._thread = None

    def renew_once(self) -> bool:
        try:
            return bool(self._store.renew(self._lease))
        except Exception:  # noqa: BLE001 - renewal is best-effort
            return False

    def _loop(self):
        while not self._stop.wait(self._interval):
            self.renew_once()

    def __enter__(self):
        if self._store is None or not hasattr(self._store, "renew"):
            return self
        self._thread = threading.Thread(
            target=self._loop, name="hpo-lease-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1, self._interval))
            self._thread = None
        return False


class TrialLeaseStore:
    """The controller issues leases; workers may only act while current."""

    def __init__(self, url: str, *, ttl_seconds: int = _LEASE_TTL_SECONDS,
                 clock=None, sleep=None) -> None:
        from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine

        self._ttl_seconds = max(1, int(ttl_seconds))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep
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
        _shared("ensure_tables")(self._engine, metadata)

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def _now(self) -> datetime:
        return _as_utc(self._clock())

    def _issue_attempt(self, *, generation_id, model_key, trial_number):
        from sqlalchemy import select, update
        from sqlalchemy.dialects.postgresql import insert

        now = self._now()
        with self._engine.begin() as conn:
            row = conn.execute(select(self._table).where(
                self._table.c.generation_id == generation_id,
                self._table.c.model_key == model_key,
                self._table.c.trial_number == trial_number,
            ).with_for_update()).mappings().first()
            if row is None:
                created = conn.execute(insert(self._table).values(
                    generation_id=generation_id, model_key=model_key,
                    trial_number=trial_number, epoch=1, state="active",
                    updated_at=now,
                ).on_conflict_do_nothing()).rowcount
                if not created:
                    return None  # lost the creation race; caller retries
                return TrialLease(generation_id, model_key, trial_number, 1,
                                  self._ttl_seconds)
            epoch = int(row["epoch"]) + 1
            changed = conn.execute(update(self._table).where(
                self._table.c.generation_id == generation_id,
                self._table.c.model_key == model_key,
                self._table.c.trial_number == trial_number,
                self._table.c.epoch == row["epoch"],
            ).values(epoch=epoch, state="active", updated_at=now)).rowcount
            if not changed:
                return None
        return TrialLease(generation_id, model_key, trial_number, epoch,
                          self._ttl_seconds)

    def issue(self, *, generation_id: str, model_key: str, trial_number: int,
              attempts: int = 8) -> TrialLease:
        """Issue a new epoch, invalidating any old worker for this trial.

        A concurrent creator no longer surfaces an unfenced lease: the losing
        attempt is retried (the winner's row is then locked and bumped), so the
        caller always receives a lease it owns the newest epoch of.
        """
        for _ in range(max(1, int(attempts))):
            lease = self._issue_attempt(
                generation_id=generation_id, model_key=model_key,
                trial_number=trial_number)
            if lease is not None:
                return lease
            if self._sleep is not None:
                self._sleep(0.05)
        raise _shared("HpoInfrastructureError")(
            "lease issue exceeded retry budget for "
            f"{model_key}/trial {trial_number}")

    def renew(self, lease: TrialLease, *, now=None) -> bool:
        """Refresh the lease deadline; only the current active epoch may renew."""
        from sqlalchemy import update

        moment = _as_utc(now) if now is not None else self._now()
        with self._engine.begin() as conn:
            changed = conn.execute(update(self._table).where(
                self._table.c.generation_id == lease.generation_id,
                self._table.c.model_key == lease.model_key,
                self._table.c.trial_number == lease.trial_number,
                self._table.c.epoch == lease.epoch,
                self._table.c.state == "active",
            ).values(updated_at=moment)).rowcount
        return bool(changed)

    def assert_current(self, lease: TrialLease, *, now=None) -> None:
        """Fail closed: an expired/stale worker cannot publish anything."""
        from sqlalchemy import select

        moment = _as_utc(now) if now is not None else self._now()
        with self._engine.connect() as conn:
            row = conn.execute(select(
                self._table.c.epoch, self._table.c.state,
                self._table.c.updated_at).where(
                    self._table.c.generation_id == lease.generation_id,
                    self._table.c.model_key == lease.model_key,
                    self._table.c.trial_number == lease.trial_number,
                )).mappings().one_or_none()
        ok, reason = lease_status(lease=lease, row=row, now=moment,
                                  ttl_seconds=self._ttl_seconds)
        if not ok:
            raise _shared("HpoInfrastructureError")(
                f"stale HPO worker fenced ({reason}): {lease.model_key}/"
                f"trial {lease.trial_number} epoch {lease.epoch}")

    def reap_expired(self, *, now=None) -> list:
        """Mark heartbeat-expired active leases ``expired``; return them.

        A reaper (a session's first worker, or a watchdog) can then fail the
        OFFLINE Optuna trials named by the returned leases.  Leases that are
        still being renewed are untouched.
        """
        from sqlalchemy import select, update

        moment = _as_utc(now) if now is not None else self._now()
        reaped: list = []
        with self._engine.begin() as conn:
            rows = conn.execute(select(self._table).where(
                self._table.c.state == "active",
            ).with_for_update()).mappings().all()
            for row in rows:
                if not lease_expired(updated_at=row["updated_at"], now=moment,
                                     ttl_seconds=self._ttl_seconds):
                    continue
                conn.execute(update(self._table).where(
                    self._table.c.generation_id == row["generation_id"],
                    self._table.c.model_key == row["model_key"],
                    self._table.c.trial_number == row["trial_number"],
                    self._table.c.epoch == row["epoch"],
                ).values(state="expired", updated_at=moment))
                reaped.append(TrialLease(
                    row["generation_id"], row["model_key"],
                    int(row["trial_number"]), int(row["epoch"]),
                    self._ttl_seconds))
        return reaped

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
            ).values(state="revoked", updated_at=self._now())).rowcount
        return bool(changed)

    def heartbeat(self, lease: TrialLease, *, interval_seconds=None):
        """A context manager that renews ``lease`` while the trial runs."""
        return LeaseHeartbeat(self, lease, interval_seconds=interval_seconds)
