"""Atomic, shared trial-budget reservation for concurrent HPO workers.

The old worker budget was ``remaining = N_TRIALS - finished`` computed
independently by every process and then handed to ``study.optimize(n_trials=...)``.
Two concurrent sessions therefore each scheduled up to the WHOLE remaining
budget, and a FAIL/PRUNED trial counted as budget spent, so a crash-loop could
exhaust the budget with zero COMPLETE trials.

``WorkLedger`` replaces that with one transactional row per
``(generation_id, model_key)``.  A worker reserves ONE trial at a time
(``reserve``), then either charges it as COMPLETE (``complete``) or gives it back
(``release``).  Two independent bounds:

* ``budget``     — the maximum number of COMPLETE trials (the goal);
* ``max_trials`` — the maximum number of attempts (a compute safety net so a
  persistently failing configuration cannot loop forever).

The invariants are::

    completed + in_flight <= budget          # COMPLETE can never exceed budget
    attempted              <= max_trials      # finite even if every trial FAILs

Because every transition locks the single row (``SELECT ... FOR UPDATE``), they
hold across any number of concurrent workers and sessions.  A FAIL/PRUNED trial
releases its slot (``attempted`` still counts), so it can never *starve* a
COMPLETE trial — that is the F8 fix.

``BudgetCounter`` is the pure, optuna/SQL-free model of those invariants; it is
the single source of truth for the arithmetic and is unit-tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


def _shared(name: str):
    """Resolve a symbol from the earlier injected control-plane module."""
    symbol = globals().get(name)
    if symbol is not None:
        return symbol
    from training import hpo_control_plane

    return getattr(hpo_control_plane, name)


def default_max_trials(budget: int) -> int:
    """The attempt ceiling for a COMPLETE budget of ``budget``.

    Twice the budget: enough retries for transient failures, still finite if
    every attempt fails.
    """
    budget = max(0, int(budget))
    return budget if budget == 0 else budget * 2


@dataclass
class BudgetCounter:
    """Pure reservation arithmetic.

    ``completed + in_flight <= budget`` and ``attempted <= max_trials`` always.
    """

    budget: int
    max_trials: int = 0
    in_flight: int = 0
    completed: int = 0
    attempted: int = 0

    def __post_init__(self):
        self.budget = max(0, int(self.budget))
        self.max_trials = int(self.max_trials) or default_max_trials(self.budget)
        self.in_flight = max(0, int(self.in_flight))
        self.completed = max(0, int(self.completed))
        self.attempted = max(0, int(self.attempted))

    def remaining(self) -> int:
        """Free COMPLETE slots (the goal budget)."""
        return max(0, int(self.budget) - int(self.completed)
                   - int(self.in_flight))

    def attempts_remaining(self) -> int:
        return max(0, int(self.max_trials) - int(self.attempted))

    def reserve(self, amount: int = 1) -> int:
        """Grant up to ``amount`` slots; the caller MUST settle each one."""
        amount = max(0, int(amount))
        granted = min(amount, self.remaining(), self.attempts_remaining())
        self.in_flight += granted
        self.attempted += granted
        return granted

    def complete(self, amount: int = 1) -> int:
        """Charge in-flight slots as COMPLETE (never exceeds what was held)."""
        amount = max(0, int(amount))
        charged = min(amount, self.in_flight)
        self.in_flight -= charged
        self.completed += charged
        return charged

    def release(self, amount: int = 1) -> int:
        """Return in-flight slots that did not produce a COMPLETE trial.

        ``attempted`` is NOT refunded: a released attempt still counted against
        the compute ceiling.
        """
        amount = max(0, int(amount))
        released = min(amount, self.in_flight)
        self.in_flight -= released
        return released

    def as_dict(self) -> dict:
        return {"budget": int(self.budget), "max_trials": int(self.max_trials),
                "in_flight": int(self.in_flight),
                "completed": int(self.completed), "attempted": int(self.attempted),
                "remaining": self.remaining(),
                "attempts_remaining": self.attempts_remaining()}


class WorkLedger:
    """Postgres-backed transactional reservation of the cluster trial budget."""

    def __init__(self, url: str, *, generation_id: str, model_key: str,
                 budget: int, max_trials: int = 0) -> None:
        from sqlalchemy import (
            Column,
            DateTime,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
        )

        self._generation_id = generation_id
        self._model_key = model_key
        self._budget = max(0, int(budget))
        self._max_trials = int(max_trials) or default_max_trials(self._budget)
        self._engine = create_engine(url, pool_pre_ping=True)
        metadata = MetaData()
        self._table = Table(
            "euromonitor_hpo_budget", metadata,
            Column("generation_id", String, primary_key=True),
            Column("model_key", String, primary_key=True),
            Column("budget", Integer, nullable=False),
            Column("max_trials", Integer, nullable=False),
            Column("in_flight", Integer, nullable=False),
            Column("completed", Integer, nullable=False),
            Column("attempted", Integer, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        _shared("ensure_tables")(self._engine, metadata)
        self._ensure_row()

    def _ensure_row(self) -> None:
        from sqlalchemy import update
        from sqlalchemy.dialects.postgresql import insert

        now = datetime.now(UTC)
        with self._engine.begin() as conn:
            conn.execute(insert(self._table).values(
                generation_id=self._generation_id, model_key=self._model_key,
                budget=self._budget, max_trials=self._max_trials,
                in_flight=0, completed=0, attempted=0, updated_at=now,
            ).on_conflict_do_nothing())
            # A larger budget (a re-staged generation) raises the shared cap;
            # a smaller one never silently shrinks a live generation.
            conn.execute(update(self._table).where(
                self._table.c.generation_id == self._generation_id,
                self._table.c.model_key == self._model_key,
                self._table.c.budget < self._budget,
            ).values(budget=self._budget, max_trials=self._max_trials,
                     updated_at=now))

    def _locked_counter(self, conn):
        from sqlalchemy import select

        row = conn.execute(select(self._table).where(
            self._table.c.generation_id == self._generation_id,
            self._table.c.model_key == self._model_key,
        ).with_for_update()).mappings().one()
        return BudgetCounter(budget=int(row["budget"]),
                             max_trials=int(row["max_trials"]),
                             in_flight=int(row["in_flight"]),
                             completed=int(row["completed"]),
                             attempted=int(row["attempted"]))

    def _write(self, conn, counter: BudgetCounter) -> None:
        from sqlalchemy import update

        conn.execute(update(self._table).where(
            self._table.c.generation_id == self._generation_id,
            self._table.c.model_key == self._model_key,
        ).values(in_flight=int(counter.in_flight),
                 completed=int(counter.completed),
                 attempted=int(counter.attempted),
                 budget=int(counter.budget),
                 max_trials=int(counter.max_trials),
                 updated_at=datetime.now(UTC)))

    def reserve(self, amount: int = 1) -> int:
        with self._engine.begin() as conn:
            counter = self._locked_counter(conn)
            granted = counter.reserve(amount)
            if granted:
                self._write(conn, counter)
            return granted

    def complete(self, amount: int = 1) -> int:
        with self._engine.begin() as conn:
            counter = self._locked_counter(conn)
            charged = counter.complete(amount)
            if charged:
                self._write(conn, counter)
            return charged

    def release(self, amount: int = 1) -> int:
        with self._engine.begin() as conn:
            counter = self._locked_counter(conn)
            released = counter.release(amount)
            if released:
                self._write(conn, counter)
            return released

    def snapshot(self) -> BudgetCounter:
        from sqlalchemy import select

        with self._engine.connect() as conn:
            row = conn.execute(select(self._table).where(
                self._table.c.generation_id == self._generation_id,
                self._table.c.model_key == self._model_key,
            )).mappings().one()
        return BudgetCounter(budget=int(row["budget"]),
                             max_trials=int(row["max_trials"]),
                             in_flight=int(row["in_flight"]),
                             completed=int(row["completed"]),
                             attempted=int(row["attempted"]))
