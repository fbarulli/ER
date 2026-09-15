"""tracing.py — the ONE consolidated pipeline trace.

Doctrine (owner directive 2026-09-15): "we need to see and inspect every step
of the way just like the rest of the csv's do". Traceability used to be
scattered across separate CSVs in ``results/logs/`` (gate_visibility,
negative_resolution_manifest, payload_pairs, ...) so answering "why did this
pair end at this label?" meant opening several files and reconstructing the
order by hand. Every stage now appends to ONE csv whose ROW ORDER IS THE
DATA FLOW: read it top to bottom and you read the pipeline.

Row contract — ``TRACE_COLUMNS``:
  stage / step          where in the flow (stage = the pipeline function,
                        step = the decision inside it)
  scope                 ``run`` (whole population), ``entity`` (one object),
                        or ``group`` (one reason/split bucket)
  key                   the object the step is about (gtin, gtin pair,
                        canonical id, query id) — empty for run scope
  in_count / out_count  the funnel: what entered the step, what survived it.
                        ``dropped_count`` is derived, never hand-written, so a
                        step can never silently lie about its own attrition.
  reason                why the remainder was kept/dropped, in the gate's own
                        words
  detail                JSON of the step's own readback (thresholds, parsed
                        values, sub-counts, examples) — the audit triple every
                        other CSV in this tree carries.
  source / producer     which artifact was read, which module wrote the row
  at                    UTC ISO-8601 timestamp

Writers MUST go through :func:`record`/:class:`TraceRun`. The path comes from
``paths.yaml`` layout ``training_trace`` via :func:`core.common.artifact` —
never from ``__file__``/``__parents__``, never from a literal.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── row contract ───────────────────────────────────────────────────────────
TRACE_COLUMNS: tuple[str, ...] = (
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


def trace_path() -> Path:
    """Return the consolidated trace destination from its owned layout.

    ``core.common`` is imported lazily: ``common`` imports the config that
    declares this layout, so a module-level import would be circular. The
    authoritative path still comes from the layout registry, not from this
    file's location.
    """
    from core.common import artifact

    return artifact("training_trace")


def _detail_text(detail: object) -> str:
    """Serialize a detail payload deterministically, or pass text through."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, sort_keys=True, default=str)


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
    row = {
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
        "at": datetime.now(timezone.utc).isoformat(),
    }
    # ROW BOUNDARY (pydantic contract, core.schemas.TraceRow): the counted
    # arithmetic lives in exactly one place there, so a row that cannot be
    # reasoned about is rejected here rather than written and discovered later.
    # Imported lazily: core.schemas imports this module's TRACE_COLUMNS.
    from core.schemas import TraceRow

    TraceRow.model_validate(row)
    return row


def read_trace(path: Path | None = None) -> pd.DataFrame:
    """Read the consolidated trace, empty-but-typed when it does not exist."""
    target = path if path is not None else trace_path()
    if not target.exists():
        return pd.DataFrame(columns=list(TRACE_COLUMNS))
    return pd.read_csv(target, dtype=str, keep_default_na=False)


class TraceRun:
    """Accumulate rows for one stage and write them as a single atomic append.

    One writer per stage keeps the file consistent: the stage reads whatever
    is already there, appends its own rows in flow order, and writes the whole
    frame once via the shared atomic mechanism. A crash mid-stage therefore
    leaves the previous stage's trace intact instead of a half-written file.
    """

    def __init__(self, stage: str) -> None:
        self.stage = str(stage)
        self._rows: list[dict[str, object]] = []

    def __len__(self) -> int:
        return len(self._rows)

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
            f"{self.stage}.{step}",
            substep,
            scope=scope,
            key=key,
            in_count=in_count,
            out_count=out_count,
            reason=reason,
            detail=detail,
            source=source,
        )
        self._rows.append(row)
        return row

    def extend(self, rows: object) -> None:
        for row in rows or ():
            self._rows.append(row)

    def add_entities(
        self,
        step: str,
        records: object,
        *,
        key_of: object = None,
        reason_of: object = None,
        detail_of: object = None,
        source: object = "",
        limit: int | None = None,
    ) -> int:
        """Append capped per-entity rows for one step; return how many landed.

        ``records`` is any iterable of objects (usually DataFrame rows).
        ``key_of``/``reason_of``/``detail_of`` are callables; every row shares
        this stage's timestamp so a stage's rows are one coherent snapshot.
        The cap is explicit: rows beyond it are replaced by a ``_truncated``
        marker, so a partial entity census is never silent.
        """
        rows = list(records)
        stamp = (
            self._rows[-1]["at"]
            if self._rows
            else datetime.now(timezone.utc).isoformat()
        )
        emitted: list[dict[str, object]] = []
        effective = len(rows) if limit is None else max(0, int(limit))
        for record_value in rows[:effective]:
            emitted.append(
                {
                    "stage": self.stage,
                    "step": str(step),
                    "scope": SCOPE_ENTITY,
                    "key": "" if key_of is None else str(key_of(record_value)),
                    "in_count": None,
                    "out_count": None,
                    "dropped_count": None,
                    "reason": "" if reason_of is None else str(reason_of(record_value)),
                    "detail": _detail_text(
                        None if detail_of is None else detail_of(record_value)
                    ),
                    "source": "" if source is None else str(source),
                    "producer": PRODUCER,
                    "at": stamp,
                }
            )
        if len(rows) > effective:
            emitted.append(
                {
                    "stage": self.stage,
                    "step": f"{step}_truncated",
                    "scope": SCOPE_RUN,
                    "key": "",
                    "in_count": len(rows),
                    "out_count": effective,
                    "dropped_count": len(rows) - effective,
                    "reason": "entity_rows_capped",
                    "detail": _detail_text(
                        {"cap": effective, "omitted": len(rows) - effective}
                    ),
                    "source": "" if source is None else str(source),
                    "producer": PRODUCER,
                    "at": stamp,
                }
            )
        self._rows.extend(emitted)
        return len(emitted)

    def rows(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=list(TRACE_COLUMNS))

    def write(self, path: Path | None = None) -> Path:
        """Append this stage's rows to the consolidated trace.

        WRITE BOUNDARY: the concatenated frame is validated against the
        pydantic contract BEFORE the atomic write, so a frame that cannot be
        reasoned about never lands in the artifact every reader trusts.
        """
        from core.manifest import atomic_write_csv

        target = path if path is not None else trace_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = read_trace(target)
        frame = pd.concat([existing, self.rows()], ignore_index=True)
        assert_trace_frame(frame, path=target)
        atomic_write_csv(frame, target, index=False)
        return target


def count_rows(
    values: object, *, limit: int = 12
) -> list[str]:
    """Return the most common values as ``"value=n"`` strings for ``detail``.

    A bounded readback: the trace shows the top of every reason distribution
    without turning one step into a million rows.
    """
    series = pd.Series(list(values))
    if series.empty:
        return []
    counts = series.astype(str).value_counts()
    return [f"{name}={int(count)}" for name, count in counts.head(limit).items()]


def sample_keys(values: object, *, limit: int = 5) -> list[str]:
    """Return a deterministic, bounded sample of keys for ``detail``."""
    unique = sorted({str(value) for value in list(values)})
    return unique[:limit]


def assert_trace_frame(frame: pd.DataFrame, *, path: object = "") -> None:
    """Fail loudly if a trace frame violates the pydantic row contract.

    Delegates to :func:`core.schemas.check_trace_frame`, which owns the single
    declaration of the counted arithmetic (``dropped_count == in_count -
    out_count``) and of the row schema. This wrapper only adds the frame's
    path to the message so a violation names the file it came from.
    """
    if frame.empty and not any(column in frame.columns for column in TRACE_COLUMNS):
        return
    from core.schemas import check_trace_frame

    try:
        check_trace_frame(frame)
    except ValueError as exc:
        raise ValueError(f"trace frame {path}: {exc}") from exc


def merge_entity_rows(
    rows: list[dict[str, object]],
    *,
    stage: str,
    step: str,
    scope: str,
    reason: object = "",
    detail: object = None,
    source: object = "",
    limit: int = 200,
) -> list[dict[str, object]]:
    """Cap per-entity rows so an entity trace can never explode the file.

    Returns the rows that were emitted, appends a ``..._truncated`` marker row
    when the cap bites, so a partial entity census is NEVER silent.
    """
    emitted = list(rows[:limit])
    if len(rows) > limit:
        emitted.append(
            record(
                stage,
                f"{step}_truncated",
                scope=SCOPE_RUN,
                in_count=len(rows),
                out_count=limit,
                reason="entity_rows_capped",
                detail={"cap": limit, "omitted": len(rows) - limit},
                source=source,
            )
        )
    return emitted


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


__all__ = [
    "PRODUCER",
    "SCOPE_ENTITY",
    "SCOPE_GROUP",
    "SCOPE_RUN",
    "TRACE_COLUMNS",
    "TraceRun",
    "assert_trace_frame",
    "count_rows",
    "detail_json",
    "merge_entity_rows",
    "read_trace",
    "record",
    "sample_keys",
    "trace_path",
]
