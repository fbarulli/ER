"""Control-plane observability for the model-agnostic HPO lane.

Four cohesive, single-purpose surfaces:

* ``JsonlStore``       — an idempotent, file-locked, drop-counted JSONL append
  store (the watermark/cursor primitive: every row carries a stable
  ``event_id`` and a repeat append is a no-op).
* ``TrialEventLog``    — the ``trial_events.jsonl`` CDC stream (create /
  complete / fail / prune / promote rows) built on a ``JsonlStore``.
* ``StudyMirror``      — an atomically-written JSONL snapshot of the study's
  trials, upserted by trial number.
* ``OfflineTrialLedger`` — the ``hpo_trials.jsonl`` fallback for a Postgres-less
  run, one row per trial, built on a ``JsonlStore``.

``ObjectiveRanker`` is the one place that turns configured per-objective
directions into a comparable sort key. ``TrialObserver`` composes the surfaces
so a caller only calls ``observe(trial)`` / ``sync(trials)`` / ``flush()``.
Nothing here imports optuna/torch/sqlalchemy; trials are plain objects.
"""
from __future__ import annotations

import json
import os
import time
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


class ObjectiveRanker:
    """Turn configured per-objective directions into a comparable sort key.

    ``maximize`` negates (ascending ``min`` == descending objective); a scalar
    value uses the primary direction, preserving the legacy ``best("minimize")``
    contract. A sequence ranks a multi-objective tuple lexicographically,
    primary-first, each component by its own direction.
    """

    def __init__(self, directions="maximize"):
        if isinstance(directions, str):
            self.directions = (directions,)
        else:
            self.directions = tuple(directions) or ("maximize",)

    def key(self, value):
        if isinstance(value, (list, tuple)):
            parts = []
            for index, component in enumerate(value):
                direction = (self.directions[index]
                             if index < len(self.directions)
                             else self.directions[-1])
                number = 0.0 if component is None else component
                parts.append(-number if direction == "maximize" else number)
            return tuple(parts)
        number = 0.0 if value is None else value
        return (-number if self.directions[0] == "maximize" else number,)

    def best(self, items, *, value_of):
        """The best item by ``value_of(item)`` (or ``None`` for an empty set)."""
        items = list(items)
        if not items:
            return None
        return min(items, key=lambda item: self.key(value_of(item)))


def rank_key(value, directions):
    """Free-function form of ``ObjectiveRanker(directions).key``."""
    return ObjectiveRanker(directions).key(value)


class FileLock:
    """Exclusive advisory lock on ``<path>.lock`` (no-op when fcntl is absent)."""

    def __init__(self, path):
        self._lock_path = Path(str(path) + ".lock")
        self._handle = None

    def __enter__(self):
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self._lock_path, "a+")
        except Exception:  # noqa: BLE001 - locking is best-effort
            self._handle = None
        if self._handle is not None and fcntl is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
            except Exception:  # noqa: BLE001,S110
                pass
        return self

    def __exit__(self, *_exc):
        if self._handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001,S110
                pass
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001,S110
                pass
        return False


class JsonlStore:
    """Idempotent append-only JSONL store. One job: durable, deduped rows.

    ``append_once`` reads the durable ``event_id`` set under ``FileLock`` and
    skips a row whose id is already present, so concurrent processes and
    resumed sessions cannot double-append the same logical event. Failed
    appends are counted (``dropped``) instead of raised.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._dropped = 0

    def read(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def read_ids(self):
        """The set of durable ``event_id`` values (the resume cursor)."""
        ids: set[str] = set()
        try:
            if not self.path.is_file():
                return ids
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001,S112 - a torn tail line
                    continue
                event_id = row.get("event_id")
                if event_id is not None:
                    ids.add(event_id)
        except Exception:  # noqa: BLE001,S110 - observability never raises
            pass
        return ids

    def append_once(self, row):
        """Append ``row`` unless its ``event_id`` is already durable."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.path):
                event_id = row.get("event_id")
                if event_id is not None and event_id in self.read_ids():
                    return _DUPLICATE
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                return _WRITTEN
        except Exception:  # noqa: BLE001 - observability never blocks work
            self._dropped += 1
            return _DROPPED

    @property
    def dropped(self):
        return self._dropped


class TrialEventLog:
    """The ``trial_events.jsonl`` CDC stream. One job: emit/read event rows.

    Rows are keyed by a stable ``event_id`` (``<event>#<trial_number>``); a
    duplicate ``emit`` is a no-op, so re-observing history is idempotent.
    """

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self._store = JsonlStore(path)
        self._clock = clock
        self._count = 0

    @staticmethod
    def event_id_for(event, trial_number):
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
        if self._store.append_once(row) is _WRITTEN:
            self._count += 1
        return row

    def observe(self, trial, event=None):
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
        return self._store.read()

    def seen_event_ids(self):
        return self._store.read_ids()

    def count(self):
        return self._count

    def dropped(self):
        return self._store.dropped


class StudyMirror:
    """Local, atomically-written JSONL snapshot of the study's trials.

    ``record`` upserts by trial number, so re-observing history never
    duplicates. ``best`` prefers the in-memory rows (the authoritative current
    view) and only falls back to disk when nothing has been recorded yet.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._rows = []
        self._index = {}

    def record(self, trial):
        row = self._row(trial)
        number = row["number"]
        if number is None:
            self._rows.append(row)
        elif number in self._index:
            self._rows[self._index[number]] = row
        else:
            self._index[number] = len(self._rows)
            self._rows.append(row)
        return self

    def _row(self, trial):
        return {
            "number": getattr(trial, "number", None),
            "state": _trial_state_name(trial),
            "value": _trial_value(trial),
            "params": dict(getattr(trial, "params", {}) or {}),
            "user_attrs": dict(getattr(trial, "user_attrs", {}) or {}),
        }

    def rows(self):
        return list(self._rows)

    def _effective_rows(self):
        return list(self._rows) if self._rows else self.load()

    def flush(self):
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
        rows = [row for row in self._effective_rows()
                if row.get("value") is not None
                and row.get("state") == "COMPLETE"]
        return ObjectiveRanker(direction).best(
            rows, value_of=lambda row: row["value"])


class OfflineTrialLedger:
    """The ``hpo_trials.jsonl`` fallback, one idempotent row per trial."""

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self._store = JsonlStore(path)
        self._clock = clock

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
        self._store.append_once(row)
        return row

    def observe(self, trial):
        return self.append(
            trial_number=getattr(trial, "number", -1),
            value=_trial_value(trial), params=getattr(trial, "params", None),
            state=_trial_state_name(trial),
            user_attrs=getattr(trial, "user_attrs", None))

    def load(self):
        return self._store.read()

    def dropped(self):
        return self._store.dropped

    def best(self, direction="maximize"):
        rows = [row for row in self.load()
                if row.get("value") is not None
                and row.get("state") == "COMPLETE"]
        return ObjectiveRanker(direction).best(
            rows, value_of=lambda row: row["value"])


class TrialObserver:
    """Compose the CDC log, the study mirror and the offline ledger.

    ``observe(trial)`` writes the log + mirror (and the ledger when offline).
    ``sync(trials)`` observes a whole committed study ONCE and reports the new
    row count; ``flush()`` persists the mirror.
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
        """Observe every committed trial once; return the new/skip counts."""
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


__all__ = [
    "EVENT_COMPLETE",
    "EVENT_CREATE",
    "EVENT_FAIL",
    "EVENT_PROMOTE",
    "EVENT_PRUNE",
    "OBSERVABILITY_DIR",
    "FileLock",
    "JsonlStore",
    "ObjectiveRanker",
    "OfflineTrialLedger",
    "StudyMirror",
    "TrialEventLog",
    "TrialObserver",
    "rank_key",
]
