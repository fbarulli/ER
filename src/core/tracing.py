"""tracing.py — the ONE consolidated pipeline trace.

Doctrine (owner directive 2026-09-15): "consolidate all csv's generated and
make sure we have visibility into the new processes we just created, 0 gap
coverage". Traceability used to be scattered across separate CSVs in
``results/logs/`` (gate_visibility, negative_resolution_manifest,
payload_pairs, ...) so answering "why did this pair end at this label?" meant
opening several files and reconstructing the order by hand. Every stage now
appends to ONE csv whose ROW ORDER IS THE DATA FLOW: read it top to bottom and
you read the pipeline.

Row contract — ``TRACE_COLUMNS``:
  run_id                WHICH RUN the row belongs to (see "run identity"
                        below). Every row carries it, so two runs can never be
                        confused for one another in the same file.
  stage                 the pipeline stage that emitted the row — the module
                        entry point ("data_prep", "pairs"), NOT a decision.
  step                  the decision inside that stage, dotted
                        ("gtin_guard.identity_claims_evaluated"), so a stage's
                        steps sort in flow order and stay greppable.
  scope                 ``run`` (whole population), ``entity`` (one object),
                        or ``group`` (one reason/label bucket).
  key                   the object the step is about (gtin, gtin pair,
                        canonical id, query id) — empty for run scope.
  in_count / out_count  the funnel: what entered the step, what survived it.
                        ``dropped_count`` is derived here, never hand-written,
                        so a step can never silently lie about its own
                        attrition. On a group row ``in_count`` is that
                        bucket's exact population and ``out_count`` how many of
                        its entity rows are in the file (a bounded sample, never
                        a different population — see "entity sampling").
  reason                why the remainder was kept/dropped/labelled, in the
                        gate's own words; on a bucket row it is the bucket
                        label itself.
  detail                JSON of the step's own readback (thresholds, parsed
                        values, sub-counts, examples) — the audit triple every
                        other CSV in this tree carries.
  source / producer     which artifact was read, which module wrote the row
  at                    UTC ISO-8601 timestamp of the row's creation

RUN IDENTITY (the duplicate-row policy)
----------------------------------------
The file is a RUN-SCOPED, IDEMPOTENT APPEND — not an overwrite and not an
unqualified append:

1. A run id is attached to every row. It is resolved, in order, from
   ``EUROMONITOR_TRACE_RUN``, then the launcher's ``EUROMONITOR_RUN_ID``
   (train.py's existing immutable-run doctrine), and otherwise from a content
   fingerprint of the run's shared artifacts (``canonical_records.csv`` +
   ``gate_results.csv``). The fingerprint is what makes the TWO-STAGE flow
   work with no orchestrator: stage 1 writes those artifacts, stage 2 READS
   them, so both stages compute the same run id and their rows land in one run.
2. Writing a stage REPLACES that stage's rows for the current run in place
   (same position, so flow order survives a stage re-run) instead of appending
   a second copy. Re-running an identical pipeline therefore leaves the row
   count unchanged — that is the anti-duplication guarantee, and it is exactly
   the case a deterministic pipeline produces.
3. Historical runs are retained for ``TRACE_RUN_HISTORY`` runs and pruned
   oldest-first. A reader always sees whole runs (all stages of a run are
   pruned together), each tagged with its own run_id.

Writers MUST go through :func:`record`/:class:`TraceRun`. The path comes from
``paths.yaml`` layout ``training_trace`` via :func:`core.common.artifact` —
never from ``__file__``/``__parents__``, never from a literal.

ENTITY SAMPLING (why the per-entity rows are capped)
----------------------------------------------------
A census, not a dump: every bucket keeps its EXACT population in a group row
(so "which pairs got which decision and why" is answered by the file alone),
and the entity rows are a bounded, stratified sample of that population with
the literal evidence. Caps live here, once, and are justified at their
definition.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── row contract ───────────────────────────────────────────────────────────
TRACE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "stage",
    "step",
    "scope",
    "key",
    "in_count",
    "out_count",
    "dropped_count",
    "reason",
    "detail",
    "source",
    "producer",
    "at",
)

SCOPE_RUN = "run"
SCOPE_ENTITY = "entity"
SCOPE_GROUP = "group"

PRODUCER = "core.tracing"

# ── run identity ───────────────────────────────────────────────────────────
# The explicit override wins; otherwise the launcher's immutable run id (the
# same one train.py already stamps on every artifact) is reused so the trace
# and the run tree can never disagree.
TRACE_RUN_ENV = "EUROMONITOR_TRACE_RUN"
LAUNCHER_RUN_ENV = "EUROMONITOR_RUN_ID"
# The shared artifacts that DEFINE a data-prep run: stage 1 writes both, stage 2
# reads both, so the fingerprint is the same on both sides of the handoff.
RUN_FINGERPRINT_SOURCES: tuple[str, ...] = ("canonical_records", "gate_results")
RUN_UNBOUND = "run-unbound"
# Runs kept in the file, newest first (whole runs, never a partial one). Five
# is enough to answer "what happened in the run on disk" and "what did the run
# before it do" while keeping a trace that is rewritten on every stage bounded;
# unbounded history is DVC/mlflow's job, not a hand-readable csv's.
TRACE_RUN_HISTORY = 5

# ── entity sampling budget ─────────────────────────────────────────────────
# Why these numbers: the gate emits 135,769 pair decisions and stage 2 emits
# ~38k labelled pairs. Dumping every entity row with its literal readback is a
# ~120MB csv — nobody reads it, and it merely duplicates gate_results.csv /
# the pair bundle, which stay on disk as the full census artifacts. So the
# entity rows are a SAMPLE and exactness is carried by the group rows: with
# ENTITY_SAMPLE_PER_REASON the smallest observed bucket (3 pairs) is fully
# listed and the 7 observed (decision, reason) buckets each get a real slice;
# ENTITY_ROW_CAP keeps the row count (and therefore the file, ~1.5MB) inside
# "openable in a spreadsheet and greppable by hand" while the leftover budget
# is spent on the largest populations, where the mass of the data is.
ENTITY_SAMPLE_PER_REASON = 64
ENTITY_ROW_CAP = 1024


def trace_path() -> Path:
    """Return the consolidated trace destination from its owned layout.

    ``core.common`` is imported lazily: ``common`` imports the config that
    declares this layout, so a module-level import would be circular. The
    authoritative path still comes from the layout registry, not from this
    file's location.
    """
    from core.common import artifact

    return artifact("training_trace")


def resolve_run_id() -> str:
    """Resolve the run id for THIS process (see the module docstring).

    Resolution order: explicit trace override → launcher run id → content
    fingerprint of the run's shared artifacts → ``run-unbound`` (no artifacts
    yet: the row is still labelled, never anonymous).
    """
    for variable in (TRACE_RUN_ENV, LAUNCHER_RUN_ENV):
        value = str(os.environ.get(variable, "")).strip()
        if value:
            return value
    return _artifact_run_fingerprint()


def _artifact_run_fingerprint() -> str:
    """Fingerprint the shared artifacts that define the current run."""
    from core.common import F
    from core.manifest import sha256_file

    parts: list[str] = []
    for key in RUN_FINGERPRINT_SOURCES:
        bound = F.get(key)
        if bound is None:
            continue
        path = Path(bound)
        if not path.exists():
            continue
        parts.append(f"{key}:{path.stat().st_size}:{sha256_file(path)}")
    if not parts:
        return RUN_UNBOUND
    return f"run-{hashlib.sha256('|'.join(parts).encode()).hexdigest()[:12]}"


def _detail_text(detail: object) -> str:
    """Serialize a detail payload deterministically, or pass text through."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, sort_keys=True, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(
    stage: str,
    step: str,
    *,
    scope: str = SCOPE_RUN,
    key: object = "",
    in_count: int | None = None,
    out_count: int | None = None,
    reason: object = "",
    detail: object = None,
    source: object = "",
    run_id: str = "",
    at: str | None = None,
) -> dict[str, object]:
    """Build one trace row. ``dropped_count`` is always derived here."""
    if scope not in (SCOPE_RUN, SCOPE_ENTITY, SCOPE_GROUP):
        raise ValueError(
            f"unknown trace scope {scope!r}; expected one of "
            f"{SCOPE_RUN!r}, {SCOPE_ENTITY!r}, {SCOPE_GROUP!r}"
        )
    if not stage or not step:
        raise ValueError("trace rows require both stage and step")
    incoming = None if in_count is None else int(in_count)
    outgoing = None if out_count is None else int(out_count)
    if incoming is not None and incoming < 0:
        raise ValueError(f"trace in_count must be >= 0, got {incoming}")
    if outgoing is not None and outgoing < 0:
        raise ValueError(f"trace out_count must be >= 0, got {outgoing}")
    dropped = (
        incoming - outgoing
        if incoming is not None and outgoing is not None
        else None
    )
    return {
        "run_id": str(run_id or ""),
        "stage": str(stage),
        "step": str(step),
        "scope": scope,
        "key": "" if key is None else str(key),
        "in_count": incoming,
        "out_count": outgoing,
        "dropped_count": dropped,
        "reason": "" if reason is None else str(reason),
        "detail": _detail_text(detail),
        "source": "" if source is None else str(source),
        "producer": PRODUCER,
        "at": at or _now(),
    }


def read_trace(path: Path | None = None) -> pd.DataFrame:
    """Read the consolidated trace, empty-but-typed when it does not exist.

    A file written before the ``run_id`` axis existed is read with an empty
    ``run_id`` (its rows are unlabelled); :func:`_commit` drops those rows on
    the next write rather than letting anonymous rows share the file with
    labelled ones.
    """
    target = path if path is not None else trace_path()
    if not target.exists():
        return pd.DataFrame(columns=list(TRACE_COLUMNS))
    frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    for column in TRACE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[list(TRACE_COLUMNS)]


def _plan_sample(
    counts: Mapping[str, int],
    *,
    per_reason: int,
    total_cap: int,
) -> dict[str, int]:
    """Allocate the entity-row budget across buckets (deterministic).

    Pass 1 gives every non-empty bucket ``min(count, per_reason)`` rows while
    RESERVING one row for each bucket still to come, so a bucket can never be
    squeezed out by a bigger one ahead of it (with a budget smaller than the
    number of buckets that reservation is impossible; those buckets still get
    their exact census row, which carries sample keys). Pass 2 spends whatever
    budget is left on the largest populations first. Buckets are ordered by
    ``(-count, name)``, so the plan depends only on the data, never on dict
    iteration order.
    """
    buckets = sorted(counts, key=lambda name: (-int(counts[name]), name))
    quota = {name: 0 for name in buckets}
    budget = max(0, int(total_cap))
    if not buckets or budget <= 0:
        return quota

    def spent() -> int:
        return sum(quota.values())

    for position, name in enumerate(buckets):
        if int(counts[name]) <= 0:
            continue
        reserve = sum(
            1 for later in buckets[position + 1 :] if int(counts[later]) > 0
        )
        give = min(
            int(counts[name]), int(per_reason), budget - spent() - reserve
        )
        if give > 0:
            quota[name] = give
    for name in buckets:
        room = int(counts[name]) - quota[name]
        if room <= 0:
            continue
        give = min(room, budget - spent())
        if give > 0:
            quota[name] += give
        if spent() >= budget:
            break
    return quota


def _commit(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    run_id: str,
    stage: str,
    history: int = TRACE_RUN_HISTORY,
) -> pd.DataFrame:
    """Commit one stage's rows into the run-scoped trace (see module docstring).

    Three rules, in order:
      1. unlabelled rows (legacy files, ``run_id == ""``) are dropped — the row
         contract has no anonymous run;
      2. only ``history`` runs are kept, oldest first-seen out, whole runs at a
         time, and the current run is never pruned;
      3. this stage's rows for this run are REPLACED IN PLACE, so re-running a
         stage updates its rows instead of appending a second copy.
    """
    frame = existing
    if frame.empty and not incoming.empty:
        return incoming.reset_index(drop=True)

    if not frame.empty:
        labelled = frame["run_id"].astype(str).str.strip() != ""
        n_unlabelled = int((~labelled).sum())
        if n_unlabelled:
            print(
                f"[trace] dropping {n_unlabelled} legacy row(s) with no run_id "
                f"from {stage}: the row contract has no anonymous run",
                flush=True,
            )
        frame = frame[labelled]

    if not frame.empty:
        order: list[str] = []
        for value in frame["run_id"].astype(str):
            if value not in order:
                order.append(value)
        if run_id not in order:
            order.append(run_id)
        keep = order[-max(1, int(history)) :]
        frame = frame[frame["run_id"].astype(str).isin(keep)]

    if frame.empty:
        return incoming.reset_index(drop=True)

    mine = frame["run_id"].astype(str).eq(run_id) & frame["stage"].astype(str).eq(
        stage
    )
    if not bool(mine.any()):
        parts = [part for part in (frame, incoming) if len(part)]
        if not parts:
            return incoming.reset_index(drop=True)
        return pd.concat(parts, ignore_index=True)

    # Same position as the rows being replaced: flow order survives a re-run
    # even when a later stage's rows are already in the file.
    positions = mine.to_numpy().nonzero()[0]
    first = int(positions[0])
    head = frame.iloc[:first]
    tail = frame.iloc[first:][~mine.iloc[first:].to_numpy()]
    parts = [part for part in (head, incoming, tail) if len(part)]
    if not parts:
        return incoming.reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)


def _as_int_or_none(value: object) -> int | None:
    """Counts stay integral across a re-write (pandas would emit 132.0)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return int(float(text))


def _normalize_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the three count columns integral-or-empty in the written file.

    Reading the file back (strings) and concatenating it with freshly built
    rows (ints) makes pandas infer a float dtype, and the next reader then sees
    ``132.0`` for a row count. Normalizing at the write boundary keeps the
    published cells as integers.
    """
    for column in ("in_count", "out_count", "dropped_count"):
        if column in frame.columns:
            # dtype="object" explicitly: a plain list of ints + None would be
            # inferred as float64 and the file would carry 132.0 again.
            frame[column] = pd.Series(
                [_as_int_or_none(value) for value in frame[column].tolist()],
                dtype="object",
                index=frame.index,
            )
    return frame


class TraceRun:
    """Accumulate one stage's rows and commit them in a single atomic write.

    One writer per stage keeps the file consistent: the stage merges into
    whatever is already there (run-scoped, idempotent — see the module
    docstring) and writes the whole frame once via the shared atomic
    mechanism. A crash mid-stage therefore leaves the previous stage's trace
    intact instead of a half-written file.
    """

    def __init__(self, stage: str, run_id: str | None = None) -> None:
        self.stage = str(stage)
        # An explicit run id is for hermetic callers (tests, smokes). The
        # default is resolved at WRITE time, never at construction: stage 1
        # constructs its writer before it writes the very artifacts the run
        # fingerprint is taken from, so an early resolution would tag stage 1
        # with the PREVIOUS run and split the two stages apart.
        self._run_id = run_id
        self._rows: list[dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def run_id(self) -> str:
        return str(self._run_id or "")

    def add(
        self,
        step: str,
        substep: str,
        /,
        *,
        scope: str = SCOPE_RUN,
        key: object = "",
        in_count: int | None = None,
        out_count: int | None = None,
        reason: object = "",
        detail: object = None,
        source: object = "",
    ) -> dict[str, object]:
        """Append one run/group row under ``step``/``substep``.

        Both are positional-only: a typo'd keyword must not silently invent a
        trace step that nobody can grep for.
        """
        row = record(
            self.stage,
            f"{step}.{substep}",
            scope=scope,
            key=key,
            in_count=in_count,
            out_count=out_count,
            reason=reason,
            detail=detail,
            source=source,
            run_id=self.run_id,
            at=self._stamp(),
        )
        self._rows.append(row)
        return row

    def add_column_contract(
        self,
        frame: pd.DataFrame,
        *,
        contract: str,
        required: Iterable[str],
        note: object = "",
    ) -> dict[str, object]:
        """Record the COLUMN CONTRACT of the frame a stage received.

        Stage 1 consumes the raw export (``gtin``/``sku_name_eng``/
        ``attribute``); stage 2 consumes the deduped dataset
        (``barcode``/``title``/``attributes``). Those two contracts are
        different, and the handoff between them used to be invisible — a frame
        with the wrong column names produced a KeyError far from its cause.
        This row makes the contract, and any missing required column, part of
        the trace.
        """
        columns = [str(column) for column in frame.columns]
        missing = [name for name in required if name not in columns]
        return self.add(
            "column_contract",
            "input_frame",
            in_count=int(len(frame)),
            out_count=int(len(columns)),
            reason=(
                f"stage consumed the {contract} column contract"
                + (f"; MISSING REQUIRED {missing}" if missing else "")
            ),
            detail={
                "contract": contract,
                "columns": columns,
                "required": list(required),
                "missing_required": missing,
                "note": note,
            },
            source=contract,
        )

    def add_entities(
        self,
        step: str,
        records: Iterable[object],
        *,
        key_of: object = None,
        reason_of: object = None,
        detail_of: object = None,
        source: object = "",
        per_reason: int = ENTITY_SAMPLE_PER_REASON,
        total_cap: int = ENTITY_ROW_CAP,
    ) -> dict[str, object]:
        """Census every reason bucket, then emit a bounded sample of entities.

        Emits, in this order:
          * one GROUP row per reason bucket — its EXACT population, the sample
            size chosen for it, and the omitted remainder. Summing these rows
            reproduces the step's whole population, which is what makes the
            trace self-sufficient: no second csv is needed to know how many
            pairs carried each decision and why.
          * the sampled ENTITY rows themselves, with the literal readback.
          * one RUN row announcing the budget actually spent.

        Returns a summary mapping (population/sampled/omitted/per-bucket) so a
        caller can put the same numbers in its own step row.
        """
        rows = list(records)
        stamp = self._stamp()
        buckets: dict[str, list[object]] = {}
        for value in rows:
            label = "" if reason_of is None else str(reason_of(value))
            buckets.setdefault(label, []).append(value)
        counts = {name: len(values) for name, values in buckets.items()}
        quota = _plan_sample(
            counts, per_reason=int(per_reason), total_cap=int(total_cap)
        )

        emitted: list[dict[str, object]] = []
        sampled_rows: list[dict[str, object]] = []
        for name in sorted(buckets, key=lambda key: (-counts[key], key)):
            take = int(quota.get(name, 0))
            chosen = buckets[name][:take]
            for value in chosen:
                sampled_rows.append(
                    {
                        "stage": self.stage,
                        "step": str(step),
                        "scope": SCOPE_ENTITY,
                        "key": "" if key_of is None else str(key_of(value)),
                        "in_count": None,
                        "out_count": None,
                        "dropped_count": None,
                        "reason": name,
                        "detail": _detail_text(
                            None if detail_of is None else detail_of(value)
                        ),
                        "source": "" if source is None else str(source),
                        "producer": PRODUCER,
                        "run_id": self.run_id,
                        "at": stamp,
                    }
                )
            census_detail: dict[str, object] = {
                "population": counts[name],
                "sampled": take,
                "omitted": counts[name] - take,
            }
            if counts[name] > take:
                census_detail["sample_keys"] = [
                    "" if key_of is None else str(key_of(value))
                    for value in buckets[name][:5]
                ]
            emitted.append(
                record(
                    self.stage,
                    f"{step}.reason_census",
                    scope=SCOPE_GROUP,
                    key="",
                    in_count=counts[name],
                    out_count=take,
                    reason=name,
                    detail=census_detail,
                    source=source,
                    run_id=self.run_id,
                    at=stamp,
                )
            )
        emitted.extend(sampled_rows)
        sampled = len(sampled_rows)
        emitted.append(
            record(
                self.stage,
                f"{step}.sample_budget",
                in_count=len(rows),
                out_count=sampled,
                reason=(
                    "every reason bucket is censused exactly above; the entity "
                    "rows are a bounded stratified sample of that census"
                ),
                detail={
                    "population": len(rows),
                    "sampled": sampled,
                    "omitted": len(rows) - sampled,
                    "buckets": len(buckets),
                    "per_reason": int(per_reason),
                    "total_cap": int(total_cap),
                    "full_census": "" if source is None else str(source),
                },
                source=source,
                run_id=self.run_id,
                at=stamp,
            )
        )
        self._rows.extend(emitted)
        return {
            "population": len(rows),
            "sampled": sampled,
            "omitted": len(rows) - sampled,
            "per_reason": {name: counts[name] for name in counts},
            "sampled_per_reason": {
                name: int(quota.get(name, 0)) for name in counts
            },
        }

    def rows(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=list(TRACE_COLUMNS))

    def write(self, path: Path | None = None) -> Path:
        """Commit this stage's rows to the consolidated trace."""
        from core.manifest import atomic_write_csv

        target = path if path is not None else trace_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._run_id is None:
            self._run_id = resolve_run_id()
        incoming = self.rows()
        incoming["run_id"] = self.run_id
        frame = _commit(read_trace(target), incoming, run_id=self.run_id, stage=self.stage)
        frame = _normalize_counts(frame)
        assert_trace_frame(frame, path=target)
        atomic_write_csv(frame, target, index=False)
        return target

    def _stamp(self) -> str:
        """One timestamp per row; stages share it so rows read as a snapshot."""
        return self._rows[-1]["at"] if self._rows else _now()  # type: ignore[return-value]


def count_rows(values: object, *, limit: int | None = None) -> list[str]:
    """Return the most common values as ``"value=n"`` strings for ``detail``.

    A bounded readback by default, but ``limit=None`` returns the WHOLE
    distribution: with a handful of bucket labels that is the exact census the
    trace promises, and it stays a few cells wide.
    """
    series = pd.Series(list(values))
    if series.empty:
        return []
    counts = series.astype(str).value_counts()
    if limit is not None:
        counts = counts.head(int(limit))
    return [f"{name}={int(count)}" for name, count in counts.items()]


def sample_keys(values: object, *, limit: int = 5) -> list[str]:
    """Return a deterministic, bounded sample of keys for ``detail``."""
    unique = sorted({str(value) for value in list(values)})
    return unique[: int(limit)]


def assert_trace_frame(frame: pd.DataFrame, *, path: object = "") -> None:
    """Fail loudly if a trace frame violates the row contract."""
    missing = [column for column in TRACE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"trace frame {path} is missing columns {missing}; "
            f"expected {list(TRACE_COLUMNS)}"
        )
    extra = [column for column in frame.columns if column not in TRACE_COLUMNS]
    if extra:
        raise ValueError(
            f"trace frame {path} carries undeclared columns {extra}; "
            f"expected {list(TRACE_COLUMNS)}"
        )
    if frame.empty:
        return
    blank = frame["stage"].astype(str).str.strip().eq("") | frame["step"].astype(
        str
    ).str.strip().eq("")
    if bool(blank.any()):
        raise ValueError(
            f"trace frame {path} has {int(blank.sum())} rows without stage/step"
        )
    anonymous = frame["run_id"].astype(str).str.strip().eq("")
    if bool(anonymous.any()):
        raise ValueError(
            f"trace frame {path} has {int(anonymous.sum())} rows with no "
            f"run_id — every row must belong to a run"
        )
    unknown = ~frame["scope"].astype(str).isin([SCOPE_RUN, SCOPE_ENTITY, SCOPE_GROUP])
    if bool(unknown.any()):
        raise ValueError(
            f"trace frame {path} has {int(unknown.sum())} rows with an unknown scope"
        )


def detail_json(text: object) -> dict[str, object]:
    """Parse a ``detail`` cell back into a mapping (never raises)."""
    if isinstance(text, Mapping):
        return dict(text)
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def accounting(frame: pd.DataFrame) -> dict[str, object]:
    """The run's own accounting identity, recomputed FROM THE TRACE ALONE.

    ``rows_in == canonical_records + collapsed_same_gtin +
    gtin_missing_or_nan + gs1_checksum_failed`` for the row population and
    ``candidate_pairs == hard_no + fallback + proceed`` for the pair
    population. Reading this back from the file (instead of trusting the
    caller's variables) is what makes the trace independently checkable.
    """
    def row(stage: str, step: str) -> pd.Series | None:
        hit = frame[
            frame["stage"].astype(str).eq(stage)
            & frame["step"].astype(str).eq(step)
        ]
        return None if hit.empty else hit.iloc[-1]

    def census(step_prefix: str) -> dict[str, int]:
        """Every group row whose step starts with ``step_prefix``, by suffix."""
        hit = frame[
            frame["step"].astype(str).str.startswith(step_prefix)
            & frame["scope"].astype(str).eq(SCOPE_GROUP)
        ]
        return {
            str(rec["step"])[len(step_prefix) :]: int(float(rec["out_count"]))
            for _, rec in hit.iterrows()
        }

    guard = row("data_prep", "gtin_guard.identity_claims_evaluated")
    canon = row("data_prep", "canonical.records_built")
    result: dict[str, object] = {}
    if guard is not None:
        detail = detail_json(guard["detail"])
        result["rows_in"] = int(float(guard["in_count"]))
        result["rows_identity_valid"] = int(float(guard["out_count"]))
        result["gtin_missing_or_nan"] = int(detail.get("gtin_missing_or_nan", 0))
        result["gs1_checksum_failed"] = int(detail.get("gs1_checksum_failed", 0))
    if canon is not None:
        detail = detail_json(canon["detail"])
        result["canonical_records"] = int(float(canon["out_count"]))
        result["collapsed_same_gtin"] = int(detail.get("collapsed_same_gtin", 0))
    decisions = census("gate.decision_")
    if decisions:
        result["gate_decisions"] = decisions
        result["gate_pairs"] = sum(decisions.values())
    labels = census("labels.destiny_")
    if labels:
        result["label_destiny"] = labels
        result["label_pairs"] = sum(labels.values())
    return result


__all__ = [
    "ENTITY_ROW_CAP",
    "ENTITY_SAMPLE_PER_REASON",
    "PRODUCER",
    "RUN_UNBOUND",
    "SCOPE_ENTITY",
    "SCOPE_GROUP",
    "SCOPE_RUN",
    "TRACE_COLUMNS",
    "TRACE_RUN_HISTORY",
    "TraceRun",
    "accounting",
    "assert_trace_frame",
    "count_rows",
    "detail_json",
    "read_trace",
    "record",
    "resolve_run_id",
    "sample_keys",
    "trace_path",
]
