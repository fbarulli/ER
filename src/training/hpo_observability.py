"""Control-plane observability for the model-agnostic HPO lane.

Three durable, offline-testable surfaces the owner asked for:

* ``TrialEventLog``   — append-only CDC stream of every trial event
  (``trial_events.jsonl``): create/complete/fail/prune/promote rows with the
  trial number, state, value, params and user attrs, plus a stable
  ``event_id`` so a downstream CDC job can tail it and resume/skip
  idempotently without touching Postgres.
* ``StudyMirror``     — a local JSONL snapshot of the study's trials, written
  atomically between sessions so a resumed session (or an offline audit) can
  reconstruct the study without the remote RDB.
* ``OfflineTrialLedger`` — the ``hpo_trials.jsonl`` fallback written when the
  shared PostgreSQL RDB is unavailable; it never blocks a trial, and it carries
  enough to re-emit the decision trail later.

``TrialObserver`` composes the three so a worker only calls ``observe(trial)``
(or ``sync(trials)`` once the trials are committed) and ``flush()``. Every
surface is append-only and idempotent: an ``event_id`` is derived from the
event name and the trial number, so re-observing history (a resumed session, a
second worker reading the same shared study) writes nothing new. Multi-process
appends are serialised with an advisory file lock. Nothing here imports
optuna/torch/sqlalchemy; trials are plain objects, so the surfaces are
unit-tested on the host.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

try:  # advisory multi-process append lock (POSIX only; degrades gracefully)
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX host
    fcntl = None

# The observer root dir name (single source for the host receipt and the
# staged kernel, which both build <root>/trial_events.jsonl etc.).
OBSERVABILITY_DIR = "hpo_observability"

# CDC event names (the log's vocabulary; one place).
EVENT_CREATE = "trial_created"
EVENT_COMPLETE = "trial_completed"
EVENT_FAIL = "trial_failed"
EVENT_PRUNE = "trial_pruned"
EVENT_PROMOTE = "trial_promoted"

# Append outcome sentinels.
_WRITTEN = True
_DUPLICATE = False
_DROPPED = None


def _trial_state_name(trial):
    state = getattr(trial, "state", None)
    return getattr(state, "name", None) or str(state)


def _trial_value(trial):
    value = getattr(trial, "value", None)
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


# ── idempotent, locked JSONL append (one implementation) ───────────────────
@contextmanager
def _file_lock(path):
    """Exclusive advisory lock on ``<path>.lock`` (no-op when fcntl is absent)."""
    lock_path = Path(str(path) + ".lock")
    handle = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+")  # noqa: SIM115 - lifetime spans the lock
    except Exception:  # noqa: BLE001 - locking is best-effort
        handle = None
    try:
        if handle is not None and fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:  # noqa: BLE001,S110
                pass
        yield
    finally:
        if handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001,S110
                pass
            try:
                handle.close()
            except Exception:  # noqa: BLE001,S110
                pass


def _read_event_ids(path):
    """The set of ``event_id`` values already durable in ``path``."""
    ids: set[str] = set()
    try:
        if not Path(path).is_file():
            return ids
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001,S112 - a torn tail line is skipped
                continue
            event_id = row.get("event_id")
            if event_id is not None:
                ids.add(event_id)
    except Exception:  # noqa: BLE001,S110 - observability never raises
        pass
    return ids


def _append_once(path, row):
    """Append ``row`` unless its ``event_id`` is already present.

    Returns ``_WRITTEN``/``_DUPLICATE``/``_DROPPED``. The read-check-append is
    held under ``<path>.lock`` so concurrent processes cannot double-append the
    same logical event.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path):
            event_id = row.get("event_id")
            if event_id is not None and event_id in _read_event_ids(path):
                return _DUPLICATE
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            return _WRITTEN
    except Exception:  # noqa: BLE001 - observability never blocks work
        return _DROPPED


# ── ranking (one implementation for the mirror and the ledger) ─────────────
def _directions(direction):
    """Normalise a direction spec to a tuple of ``maximize``/``minimize``."""
    if isinstance(direction, str):
        return (direction,)
    return tuple(direction)


def rank_key(value, directions):
    """A sort key applying each objective's configured direction.

    ``maximize`` negates (ascending ``min`` == descending objective); a scalar
    value uses the primary direction only, so the legacy ``best("minimize")``
    contract is preserved. Multi-objective tuples are ranked lexicographically
    primary-first with each objective's own direction.
    """
    directions = _directions(directions)
    if isinstance(value, (list, tuple)):
        parts = []
        for index, component in enumerate(value):
            direction = (directions[index] if index < len(directions)
                         else directions[-1])
            number = 0.0 if component is None else component
            parts.append(-number if direction == "maximize" else number)
        return tuple(parts)
    number = 0.0 if value is None else value
    return (-number if directions[0] == "maximize" else number,)


class TrialEventLog:
    """Append-only, idempotent ``trial_events.jsonl`` CDC stream.

    Best-effort (never fatal), but a failed write is counted so the caller can
    surface dropped events.
    """

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self._clock = clock
        self._count = 0
        self._dropped = 0

    @staticmethod
    def event_id_for(event, trial_number):
        """The stable identity of one logical event (idempotency key)."""
        number = "None" if trial_number is None else int(trial_number)
        return f"{event}#{number}"

    def emit(self, event, *, trial_number=None, state=None, value=None,
             params=None, user_attrs=None, extra=None, event_id=None):
        event = str(event)
        number = None if trial_number is None else int(trial_number)
        row = {
            "event_id": event_id or self.event_id_for(event, number),
            "event": event,
            "trial_number": number,
            "state": state,
            "value": value,
            "params": dict(params or {}),
            "user_attrs": dict(user_attrs or {}),
            "at": float(self._clock()),
        }
        if extra:
            row["extra"] = dict(extra)
        status = _append_once(self.path, row)
        if status is _WRITTEN:
            self._count += 1
        elif status is _DROPPED:
            self._dropped += 1
        return row

    def observe(self, trial, event=None):
        """Emit one row for a trial whose state Optuna has committed."""
        state = _trial_state_name(trial)
        if event is None:
            event = {
                "COMPLETE": EVENT_COMPLETE,
                "FAIL": EVENT_FAIL,
                "PRUNED": EVENT_PRUNE,
            }.get(str(state).upper(), EVENT_CREATE)
        return self.emit(event, trial_number=getattr(trial, "number", None),
                         state=state, value=_trial_value(trial),
                         params=getattr(trial, "params", None),
                         user_attrs=getattr(trial, "user_attrs", None))

    def read(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def count(self):
        return self._count

    def dropped(self):
        return self._dropped


class StudyMirror:
    """Local, atomically-written JSONL snapshot of the study's trials.

    ``record`` upserts by trial number, so re-observing history (a resumed
    session) never duplicates a row. ``best`` prefers the in-memory rows (the
    current session's authoritative view) and only falls back to the on-disk
    snapshot when nothing has been recorded yet.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._rows = []
        self._index = {}

    def record(self, trial):
        number = getattr(trial, "number", None)
        row = {
            "number": number,
            "state": _trial_state_name(trial),
            "value": _trial_value(trial),
            "params": dict(getattr(trial, "params", {}) or {}),
            "user_attrs": dict(getattr(trial, "user_attrs", {}) or {}),
        }
        if number is None:
            self._rows.append(row)
        elif number in self._index:
            self._rows[self._index[number]] = row
        else:
            self._index[number] = len(self._rows)
            self._rows.append(row)
        return self

    def rows(self):
        return list(self._rows)

    def _effective_rows(self):
        """In-memory rows when present, else the persisted snapshot."""
        return list(self._rows) if self._rows else self.load()

    def flush(self):
        """Write the snapshot atomically (never a partially-written mirror)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = "\n".join(json.dumps(row, sort_keys=True)
                            for row in self._rows)
        temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        os.replace(temporary, self.path)
        return self.path

    def load(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def best(self, direction="maximize"):
        """The best COMPLETE row under ``direction`` (str or per-objective list).

        ``direction`` may be a single ``maximize``/``minimize`` (applied to the
        primary objective) or a sequence (one direction per objective), so a
        multi-objective study ranks by its configured directions.
        """
        rows = [row for row in self._effective_rows()
                if row.get("value") is not None
                and row.get("state") == "COMPLETE"]
        if not rows:
            return None
        directions = _directions(direction)
        return min(rows, key=lambda row: rank_key(row["value"], directions))


class OfflineTrialLedger:
    """``hpo_trials.jsonl`` fallback for when the shared RDB is unavailable.

    A best-effort local record of every trial so the decision trail survives an
    offline/single-worker fallback. It is append-only and idempotent (one row
    per trial number) and never blocks a trial.
    """

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self._clock = clock
        self._dropped = 0

    @staticmethod
    def event_id_for(trial_number):
        return f"hpo_trials#{int(trial_number)}"

    def append(self, *, trial_number, value, params=None, state="COMPLETE",
               user_attrs=None, event_id=None):
        number = int(trial_number)
        row = {
            "event_id": event_id or self.event_id_for(number),
            "trial_number": number,
            "value": value,
            "params": dict(params or {}),
            "state": str(state),
            "user_attrs": dict(user_attrs or {}),
            "at": float(self._clock()),
        }
        if _append_once(self.path, row) is _DROPPED:
            self._dropped += 1
        return row

    def observe(self, trial):
        return self.append(
            trial_number=getattr(trial, "number", -1),
            value=_trial_value(trial), params=getattr(trial, "params", None),
            state=_trial_state_name(trial),
            user_attrs=getattr(trial, "user_attrs", None))

    def load(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def dropped(self):
        return self._dropped

    def best(self, direction="maximize"):
        rows = [row for row in self.load()
                if row.get("value") is not None
                and row.get("state") == "COMPLETE"]
        if not rows:
            return None
        directions = _directions(direction)
        return min(rows, key=lambda row: rank_key(row["value"], directions))


class TrialObserver:
    """Compose the CDC log, the study mirror and the offline ledger.

    ``observe(trial)`` always writes the CDC event log and the local mirror.
    The ``hpo_trials.jsonl`` ledger is the OFFLINE-ONLY fallback: it is written
    only when ``offline=True`` (no shared PostgreSQL), so a healthy Postgres run
    never double-writes the same trial. ``sync(trials)`` observes a whole
    committed study ONCE (idempotent) and reports how many rows were new.
    ``flush()`` writes the mirror snapshot. Every write is best-effort —
    observability must never fail a trial — but dropped writes are counted.
    """

    def __init__(self, root, *, offline=False):
        root = Path(root)
        self.offline = bool(offline)
        self.events = TrialEventLog(root / "trial_events.jsonl")
        self.mirror = StudyMirror(root / "study_mirror.jsonl")
        self.ledger = OfflineTrialLedger(root / "hpo_trials.jsonl")

    @property
    def mode(self):
        return "offline" if self.offline else "postgres"

    def observe(self, trial, *, promoted=False):
        self.events.observe(
            trial, event=EVENT_PROMOTE if promoted else None)
        self.mirror.record(trial)
        if self.offline:
            self.ledger.observe(trial)
        return self

    def sync(self, trials, *, promoted=()):
        """Observe every committed trial once; return the new/skip counts.

        Idempotent across sessions and processes: a trial whose ``event_id`` is
        already durable is skipped rather than re-emitted.
        """
        promoted_numbers = set(promoted or ())
        written = 0
        skipped = 0
        for trial in trials:
            before = self.events.count()
            self.observe(trial, promoted=getattr(trial, "number", None)
                         in promoted_numbers)
            if self.events.count() > before:
                written += 1
            else:
                skipped += 1
        return {"written": written, "skipped": skipped}

    def flush(self):
        try:
            return self.mirror.flush()
        except Exception:  # noqa: BLE001 - best-effort
            return None

    def as_dict(self):
        try:
            event_rows = len(self.events.read())
        except Exception:  # noqa: BLE001
            event_rows = 0
        return {
            "mode": self.mode,
            "events": str(self.events.path),
            "mirror": str(self.mirror.path),
            "ledger": str(self.ledger.path),
            "ledger_active": self.offline,
            "events_count": event_rows,
            "events_written": self.events.count(),
            "events_dropped": self.events.dropped(),
            "ledger_dropped": self.ledger.dropped(),
        }
