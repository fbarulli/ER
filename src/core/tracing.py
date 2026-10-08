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

ONE TRACE PER RUN (why the destination and the id are pinned)
------------------------------------------------------------
The layout declares ONE destination, but a run has LANES and each lane may own
its own RESULTS subtree: the suite spawns its trained tracks with
``EUROMONITOR_RESULTS_DIR=<run>/<track>`` so each worker's artifacts stay its
own. A lane that re-derives ``RESULTS/logs/training_trace.csv`` therefore writes
a DIFFERENT file, and its per-lane ``EUROMONITOR_RUN_ID`` outranks the run
fingerprint, so its rows can never join the data-prep rows of the same run.
Launchers close that by handing every spawned lane :func:`run_trace_env` — the
run's ONE destination (``EUROMONITOR_TRACE_PATH``), the run's id
(``EUROMONITOR_TRACE_RUN``) and the lane's own name
(``EUROMONITOR_TRACE_LANE``); nothing else about the lane changes. Two lanes of
the SAME module (both trained tracks write stage ``worker``) stay disjoint
because the lane qualifies their rows' ``producer``, and a commit replaces its
own (run, stage, producer) subset: without it the second lane's commit would
replace the first's rows. Concurrent writers of one file serialize their
read -> merge -> write cycle on a sibling lock file (:func:`_trace_lock`).

The run id itself is the run's artifacts' fingerprint, so a stage that runs
BEFORE those artifacts exist cannot know it (resolving then returns the
PREVIOUS run's fingerprint). Such a run pins :data:`TRACE_PENDING_RUN` for the
early stages and calls :func:`adopt_run` once its own artifacts exist, so the
whole run lands under ONE id — the same id the training side reproduces.

HISTORICAL ROWS (write-time validation must not wedge the pipeline)
-------------------------------------------------------------------
Validation at the write boundary is fail-loud for the rows being written and for
the run being written (see :meth:`TraceRun.write`). Rows of OTHER runs already
in the file are only REPORTED (:func:`_report_other_runs`): they are immutable
history, they say nothing about the write being made, and failing on them would
wedge the pipeline on stale data — the prune of old runs happens inside the very
commit that would fail, so no write could ever heal the file. The remedy for an
invalid historical run is to regenerate or prune it (the trace is a regenerated
artifact). The join from an orchestrator stage to its trace rows is declared in
``config/paths.yaml`` (``orchestration_stages``) and read by
:func:`orchestration_trace_stages` / :func:`trace_stages_for`.

ENTITY SAMPLING (why the per-entity rows are capped)
----------------------------------------------------
A census, not a dump: every bucket keeps its EXACT population in a group row
(so "which pairs got which decision and why" is answered by the file alone),
and the entity rows are a bounded, stratified sample of that population with
the literal evidence. Caps live here, once, and are justified at their
definition (ENTITY_SAMPLE_PER_REASON / ENTITY_ROW_CAP, then CENSUS_TOP_N /
DETAIL_CELL_CHARS for the readback cells).

VALIDATION (the writer cannot publish a row its readers reject)
---------------------------------------------------------------
The row contract is declared ONCE, in ``core.schemas.TraceRow`` /
``check_trace_frame`` (which imports TRACE_COLUMNS from here, hence the lazy
imports back). It is enforced at BOTH boundaries: :func:`record` validates the
row it just built (so a bad row dies at its producer), and
:func:`assert_trace_frame` — called by :meth:`TraceRun.write` BEFORE the atomic
write — validates the whole merged frame (so a corrupted stage, or a legacy row
already in the file, can never land in results/logs/training_trace.csv).

THE ACCOUNTING IDENTITY (run-scoped, enforced)
----------------------------------------------
:func:`accounting` recomputes the run's identities FROM THE FILE ALONE and
raises when one does not close:

    rows_in == rows_retained + gtin_missing_or_nan + gs1_checksum_failed
                              + identity_review_quarantined
    rows_retained (guard.out) == canonical_records + collapsed_same_gtin
    gate_pairs == sum(gate_decisions) over the run's group rows

It is RUN-SCOPED: the file holds up to TRACE_RUN_HISTORY runs, so the default
run is the one the artifacts on disk resolve to (the writers' own rule), never
"the last matching row in the file" — see :func:`_rows_of_run`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.run_log import RunLogger
from core.step_trace import timed

_LOG = RunLogger(__name__)

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
# A run's LANES all append to ONE trace. A launcher that spawns per-lane workers
# with a per-lane EUROMONITOR_RESULTS_DIR (model_tracks.parallel / model_tracks.run
# do, because each worker owns its own artifacts) must pin these two, or the
# lanes FRAGMENT the trace: each would derive its own <lane>/logs/training_trace.csv
# from RESULTS, and its per-lane EUROMONITOR_RUN_ID would outrank the run
# fingerprint, so the rows could never join the data-prep rows of the same run.
# The pins are resolved by the launcher (whose RESULTS is the run root) and are
# the only way a child learns the RUN's destination instead of re-deriving a
# lane-local one:
#   EUROMONITOR_TRACE_PATH  the run's ONE trace file, from the training_trace
#                           layout (trace_path() in the launcher) — one
#                           destination for every lane;
#   EUROMONITOR_TRACE_LANE  which lane this process writes for. Two lanes of the
#                           same module (the suite's trained tracks BOTH write
#                           stage "worker") would otherwise claim one another's
#                           rows: the lane qualifies the row's ``producer``, and
#                           a stage commit replaces only its OWN lane's rows.
TRACE_PATH_ENV = "EUROMONITOR_TRACE_PATH"
TRACE_LANE_ENV = "EUROMONITOR_TRACE_LANE"
# The run id a preparation run's rows carry BEFORE the artifacts that define its
# identity exist (core.tracing.run_artifact_fingerprint). resolve_run_id() at
# that moment would return the PREVIOUS run's fingerprint, which is worse than
# "unknown": it attributes this run's rows to another run. prepare_all pins this
# and adopts the rows once the run's own artifacts exist (see adopt_run).
TRACE_PENDING_RUN = "run-pending"
# The shared artifacts that DEFINE a data-prep run: stage 1 writes both, stage 2
# reads both, so the fingerprint is the same on both sides of the handoff.
RUN_FINGERPRINT_SOURCES: tuple[str, ...] = ("canonical_records", "gate_results")
RUN_UNBOUND = "run-unbound"
# Runs kept in the file, newest first (whole runs, never a partial one). Five
# is enough to answer "what happened in the run on disk" and "what did the run
# before it do" while keeping a trace that is rewritten on every stage bounded;
# unbounded history is DVC/W&B's job, not a hand-readable csv's.
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

# ── bounded census cells ───────────────────────────────────────────────────
# A census row's ``detail`` is the audit readback, but a FREE-TEXT or PER-PAIR
# dimension used as a VALUE (gate reasons embed the evidence that produced them,
# "...: mode_flavor:orange|apple") mints one entry per pair. Measured on the
# live census (results/logs/training_trace.csv): a single
# ``count_rows(..., limit=None)`` cell carried 13,067 strings / 1.07 MB and grew
# the file to ~12 MB for three runs, against the ~1.5 MB this design budgets
# (see "entity sampling" above). So every count readback is bounded: the top
# CENSUS_TOP_N values are listed and the remainder is stated as NUMBERS
# (``others=<n>`` / ``others_buckets=<n>``) — nothing hidden, nothing lost, a
# wide distribution costs two integers instead of a megabyte — and the
# serialized detail cell is hard-capped at DETAIL_CELL_CHARS so no single row,
# however it was built, can emit a megabyte.
CENSUS_TOP_N = 24
DETAIL_CELL_CHARS = 4096
DETAIL_VALUE_CHARS = 512

# ── the trace batch-grain budget ───────────────────────────────────────────
# A stage that walks a large population does not emit one row per item: it
# accumulates a BATCH of source rows and emits ONE BATCH row carrying the
# batch's counts, and the remainder is announced in the stage's
# ``batch_census`` row. Both numbers are the trace's own contract — how many
# source rows make a batch, and how many batch rows a stage traces — so the
# producers read them here instead of re-spelling the literals in each module
# (build_final_validation, build_second04_pairs, graph_tracks.setup). A stage
# that traced MORE batches than the budget would grow the file past its design.
TRACE_BATCH_ROWS = 4096
TRACE_MAX_BATCH_ROWS = 16

# ── the accounting identity's terms ────────────────────────────────────────
# The gtin-guard row's fields, BY NAME, because the identity is stated over
# named populations: a term the row does not state cannot be checked, and an
# absent population is NOT "zero of those" (see accounting()).
GUARD_IDENTITY_TERMS: tuple[str, ...] = (
    "gtin_missing_or_nan",
    "gs1_checksum_failed",
    "identity_review_quarantined",
)
CANONICAL_IDENTITY_TERMS: tuple[str, ...] = ("collapsed_same_gtin",)

GUARD_STEP = ("data_prep", "gtin_guard.identity_claims_evaluated")
CANONICAL_STEP = ("data_prep", "canonical.records_built")

# ── the ONE registry of the preparation's orchestration stages (config SSOT) ─
# The preparation orchestrator (``training.prepare_all``) names its stages one
# way and each child writes its rows under its own module's stage
# (``training.data_prep`` -> "data_prep", ``training.build_final_validation`` ->
# "final_validation", ``graph_tracks.setup`` -> "graph_setup"/"graph_prepare"),
# so a reader cannot join a stage's manifest entry to its trace rows without a
# hand map. That join IS ``config/paths.yaml``'s ``orchestration_stages`` block
# (keys = every orchestration stage, in execution order; values = the trace
# stage name(s) the stage's producer writes rows under), with
# ``orchestration_lane_stages`` naming the stages only a negative-supply run tag
# inserts. Config is the ONE home: ``training.prepare_all`` DERIVES its
# ``STAGES``/``_LANE_STAGE_ORDER`` from the accessors below instead of restating
# the names, so the plan can never drift from the join. The block's empty lists
# mark a stage that writes NO trace rows today (a coverage gap a reader must not
# mistake for "no work happened"); the two row-free stages stay explicit there.
#
# The accessors import ``core.common`` lazily: ``common`` imports the config that
# declares this block, so a module-level import would be circular (the same
# reason :func:`trace_path` imports it lazily).


def orchestration_trace_stages() -> dict[str, tuple[str, ...]]:
    """The orchestration stage -> trace stage(s) join, read from config SSOT.

    Names come from ``config/paths.yaml`` (``orchestration_stages``); this
    module holds no copy of them. Keys are orchestration stages in execution
    order, values the trace stage name(s) their producers write. An empty tuple
    means the stage writes no trace rows today (a coverage gap, not a silent
    success).
    """
    from core import common

    declared = common.load_config().get("orchestration_stages", {})
    return {str(stage): tuple(names) for stage, names in declared.items()}


def orchestration_lane_stages() -> tuple[str, ...]:
    """The registry's LANE subset, read from ``config/paths.yaml``.

    These are the stages the orchestrator inserts (before ``validation``) only
    when a negative-supply run tag is requested, and the ones ``prepare_all``
    excludes when it needs the base stage list (state validation, resume
    arithmetic).
    """
    from core import common

    return tuple(common.load_config().get("orchestration_lane_stages", ()))


def trace_stages_for(orchestration_stage: str) -> tuple[str, ...]:
    """The trace stage name(s) one orchestration stage writes rows under.

    Empty means the stage writes no trace rows today (a coverage gap, not a
    silent success). ``training.prepare_all`` DERIVES its stage list from the
    same config registry (its base stages are these keys minus
    :func:`orchestration_lane_stages`), so a key there IS a stage the plan runs
    and there is no second list to drift against. An unknown name is still a
    loud error: a caller asking for a stage the registry never declared is a bug
    on the caller's side, not an empty join.
    """
    registry = orchestration_trace_stages()
    try:
        return registry[str(orchestration_stage)]
    except KeyError as exc:
        raise ValueError(
            f"unknown orchestration stage {orchestration_stage!r}; declared: "
            f"{sorted(registry)}"
        ) from exc


def trace_path() -> Path:
    """Return the consolidated trace destination: the RUN's ONE trace file.

    Normally that is the ``training_trace`` layout rendered against this
    process's RESULTS binding (``core.common`` is imported lazily: ``common``
    imports the config that declares this layout, so a module-level import would
    be circular; the authoritative path still comes from the layout registry,
    never from this file's location).

    When the launcher pinned ``EUROMONITOR_TRACE_PATH`` (see the pin block
    above), that IS the run's destination: a lane whose RESULTS is its own
    subtree must still append to the run's trace, or the run's rows fragment
    across one file per lane. The pin carries the launcher's own
    ``trace_path()``, so the SSOT is still the layout — the child is told the
    run's file instead of re-deriving a lane-local one.
    """
    pinned = str(os.environ.get(TRACE_PATH_ENV, "")).strip()
    if pinned:
        return Path(pinned).expanduser().resolve()
    from core.common import artifact

    return artifact("training_trace")


def run_trace_env(*, lane: str | None = None) -> dict[str, str]:
    """The pins that make ONE run's lanes append to ONE trace.

    Resolved in the LAUNCHER (whose RESULTS is the run root), so every spawned
    lane gets the run's destination, the run's id and its own lane qualifier.
    ``model_tracks.parallel`` / ``model_tracks.run`` spread this into each
    worker's environment next to the per-lane ``EUROMONITOR_RESULTS_DIR`` they
    already set; nothing else changes about those lanes.

    The run id and the destination are resolved HERE, from the two SSOTs
    (``resolve_run_id`` and ``trace_path``), so a launcher never spells either.
    """
    pins = {
        TRACE_PATH_ENV: str(trace_path()),
        TRACE_RUN_ENV: resolve_run_id(),
    }
    if lane:
        pins[TRACE_LANE_ENV] = str(lane)
    return pins


def resolve_run_id() -> str:
    """Resolve the run id for THIS process (see the module docstring)."""
    return str(resolve_run_identity()["run_id"])


def _explicit_run_identity() -> dict[str, object] | None:
    """The first explicitly pinned run id in the environment, if any.

    The provenance matters as much as the value: when a file holds two runs, a
    reader must be able to see WHY they are two runs. The resolution rule plus
    the fingerprint inputs are written into each stage's ``run_identity`` row.
    """
    for variable in (TRACE_RUN_ENV, LAUNCHER_RUN_ENV):
        value = str(os.environ.get(variable, "")).strip()
        if value:
            return {
                "run_id": value,
                "resolution": f"explicit env {variable}",
                "sources": {},
            }
    return None


def _fingerprint_run_identity(
    sources: dict[str, dict[str, object]],
) -> dict[str, object]:
    """The content fingerprint of the run artifacts that define a run."""
    joined = "|".join(
        f"{name}:{entry['size']}:{entry['sha256']}"
        for name, entry in sources.items()
    )
    return {
        "run_id": f"run-{hashlib.sha256(joined.encode()).hexdigest()[:12]}",
        "resolution": "content fingerprint of the run artifacts",
        "sources": sources,
    }


def resolve_run_identity() -> dict[str, object]:
    """Resolve the run id AND how it was resolved (see the module docstring)."""
    explicit = _explicit_run_identity()
    if explicit is not None:
        return explicit
    sources = run_artifact_fingerprint()
    if not sources:
        return {
            "run_id": RUN_UNBOUND,
            "resolution": "no run artifact on disk yet",
            "sources": {},
        }
    return _fingerprint_run_identity(sources)


@timed
def run_artifact_fingerprint() -> dict[str, dict[str, object]]:
    """Size + sha256 of every artifact that defines the current run.

    Runs on every unbound TraceRun write, so its hashing cost is traced; the
    stage's own artifacts are the digest input.
    """
    from core.common import F
    from core.manifest import sha256_file

    found: dict[str, dict[str, object]] = {}
    for key in RUN_FINGERPRINT_SOURCES:
        bound = F.get(key)
        if bound is None:
            continue
        path = Path(bound)
        if not path.exists():
            continue
        found[key] = {
            "path": path.as_posix(),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return found


def _row_accounting(row, result: dict[str, object]) -> None:
    """Augment ``result`` with the row-population identity of the guard row.

    Every term the identity is stated over is read BY NAME and left ``None``
    when the row does not carry it, so :func:`_enforce_row_identity` can tell
    "the producer did not state this population" from "it stated zero".
    """
    if row is None:
        return
    detail = detail_json(row["detail"])
    result["rows_in"] = int(float(row["in_count"]))
    result["rows_retained"] = int(float(row["out_count"]))
    # Legacy name of the same number, kept because readers (and the identity
    # test) already use it: "the rows that kept a valid identity".
    result["rows_identity_valid"] = result["rows_retained"]
    for term in GUARD_IDENTITY_TERMS:
        value = detail.get(term)
        result[term] = None if value is None else int(float(value))
    stated_retained = detail.get("rows_retained")
    if stated_retained is not None and int(
        float(stated_retained)
    ) != result["rows_retained"]:
        raise ValueError(
            f"guard row out_count {result['rows_retained']} contradicts its own "
            f"detail rows_retained {int(float(stated_retained))}"
        )


def _canonical_accounting(row, result: dict[str, object]) -> None:
    """Augment ``result`` with the canonical-records identity of its row."""
    if row is None:
        return
    detail = detail_json(row["detail"])
    result["canonical_records"] = int(float(row["out_count"]))
    for term in CANONICAL_IDENTITY_TERMS:
        value = detail.get(term)
        result[term] = None if value is None else int(float(value))


def _enforce_row_identity(result: dict[str, object]) -> None:
    """The two documented row closures, or a loud failure.

        guard.in  == rows_retained + gtin_missing_or_nan + gs1_checksum_failed
                                  + identity_review_quarantined
        guard.out == canonical.out + collapsed_same_gtin

    LIVE EVIDENCE for the quarantine term (results/logs/training_trace.csv, all
    three runs in it — read off the guard row's detail and the canonical row):

        run-cd483a04c3b2   71,623 == 25,376 + 41,545 + 3,715 + 987
        run-02e06b18fb0f   35,561 == 12,656 + 20,580 + 1,869 + 456
        run-de49ec3e6450   10,000 ==  3,540 +  5,786 +   557 + 117

    and the guard.out closure on the same rows (25,376 = 13,102 + 12,274;
    12,656 = 7,165 + 5,491; 3,540 = 2,941 + 599). Every run MISSES the
    three-term form by exactly its reviewed population (987 / 456 / 117), which
    is why the identity as previously documented never closed on any real run.
    A term the row does not state is NOT read as zero: it is a producer that
    cannot be checked, and that is an error, not a closed identity.
    """
    if "rows_in" not in result:
        return  # no guard row in scope: nothing to close (e.g. a pairs-only frame)
    unstated = [term for term in GUARD_IDENTITY_TERMS if result.get(term) is None]
    if unstated:
        raise ValueError(
            f"the gtin-guard row does not state {unstated}, so the row identity "
            f"({GUARD_STEP[1]}) cannot be checked; an absent drop population is "
            f"NOT zero and must be named on the row"
        )
    accounted = result["rows_retained"] + sum(
        result[term] for term in GUARD_IDENTITY_TERMS
    )
    if accounted != result["rows_in"]:
        terms = " + ".join(
            f"{term} {result[term]}" for term in GUARD_IDENTITY_TERMS
        )
        raise ValueError(
            f"row identity does not close: guard.in {result['rows_in']} != "
            f"rows_retained {result['rows_retained']} + {terms} = {accounted} "
            f"(unaccounted {result['rows_in'] - accounted} rows)"
        )
    if "canonical_records" not in result:
        return
    if result.get("collapsed_same_gtin") is None:
        raise ValueError(
            f"the canonical row ({CANONICAL_STEP[1]}) does not state "
            f"collapsed_same_gtin, so the guard-out identity cannot be checked"
        )
    collapsed = result["canonical_records"] + result["collapsed_same_gtin"]
    if result["rows_retained"] != collapsed:
        raise ValueError(
            f"guard-out identity does not close: guard.out "
            f"{result['rows_retained']} != canonical.out "
            f"{result['canonical_records']} + collapsed_same_gtin "
            f"{result['collapsed_same_gtin']} = {collapsed}"
        )


def _rows_of_run(frame: pd.DataFrame, run_id: str | None) -> tuple[pd.DataFrame, str]:
    """The frame's rows for ONE run, plus which run that is.

    The file holds up to :data:`TRACE_RUN_HISTORY` whole runs, so a reader that
    ignores the run axis does not read "the run" — it reads whichever row the
    file happens to end on. Measured on the live 3-run file, the run-agnostic
    read reported the LAST run's guard row (rows_in 10,000, not the 71,623 of
    the run above it) while the label census collapsed to one arbitrary run
    through a dict-key collision: an identity assembled from two different
    runs, silently.

    The default is therefore the run the artifacts ON DISK resolve to — the same
    rule :func:`resolve_run_id` gives the writers, so reader and writer can
    never disagree about which run is current. A frame that holds exactly ONE
    run is unambiguous and is used as-is (nothing can be picked wrong), and an
    ambiguous frame is a loud failure naming the runs it holds, never a silent
    last-row pick.
    """
    present = list(dict.fromkeys(frame["run_id"].astype(str))) if len(frame) else []
    if run_id is not None:
        selected = str(run_id).strip()
        if selected not in present:
            raise ValueError(
                f"trace has no rows for run_id {selected!r}; present runs: {present}"
            )
        return frame[frame["run_id"].astype(str).eq(selected)], selected
    resolved = resolve_run_id()
    if resolved in present:
        return frame[frame["run_id"].astype(str).eq(resolved)], resolved
    if len(present) == 1:
        return frame[frame["run_id"].astype(str).eq(present[0])], present[0]
    raise ValueError(
        f"the trace holds runs {present} but the run artifacts on disk resolve "
        f"to {resolved!r}; pass run_id= to select one of them explicitly"
    )


def _census_accounting(
    frame: pd.DataFrame, step_prefix: str, *, run_id: str
) -> dict[str, int]:
    """Every group row whose step starts with ``step_prefix``, by suffix.

    The suffix IS the census key, so a repeated one inside one run would make
    the mapping silently keep only the last row (the same last-row-wins bug the
    run axis fixes). It is a failure here instead: two rows for one census key
    in one run are two populations claiming one number.
    """
    hit = frame[
        frame["step"].astype(str).str.startswith(step_prefix)
        & frame["scope"].astype(str).eq(SCOPE_GROUP)
    ]
    keys = [str(value)[len(step_prefix) :] for value in hit["step"]]
    if duplicate := sorted({key for key in keys if keys.count(key) > 1}):
        raise ValueError(
            f"run {run_id} has more than one group row for {step_prefix}{duplicate}; "
            f"the census key must be unique inside a run"
        )
    return {
        key: int(float(record["out_count"]))
        for key, (_, record) in zip(keys, hit.iterrows(), strict=True)
    }


def accounting(
    frame: pd.DataFrame, *, run_id: str | None = None
) -> dict[str, object]:
    """THE RUN's accounting identity, recomputed FROM THE TRACE ALONE.

    Run-scoped: ``run_id`` names the run explicitly; by default the run is the
    one the artifacts on disk resolve to (see :func:`_rows_of_run`). The trace
    holds several runs, and an identity assembled across them is not an
    identity — it is one run's guard row next to another run's census.

    The closures are ENFORCED (a mismatch raises) and stated with their terms:

        rows_in == rows_retained + gtin_missing_or_nan + gs1_checksum_failed
                                  + identity_review_quarantined
        rows_retained (guard.out) == canonical_records + collapsed_same_gtin
        gate_pairs == sum(gate_decisions); label_pairs == sum(label_destiny)

    The first two are the row population (every raw row lands somewhere: kept,
    collapsed into its gtin, or one of the three guard drops — missing/NaN
    gtin, failed GS1 checksum, identity-review hold). The pair lines are the
    pair population: every candidate pair carries exactly one gate decision and
    exactly one label destiny. Reading this back from the file (instead of
    trusting the caller's variables) is what makes the trace independently
    checkable.
    """
    scoped, resolved = _rows_of_run(frame, run_id)

    def row(step: tuple[str, str]) -> pd.Series | None:
        stage, name = step
        hit = scoped[
            scoped["stage"].astype(str).eq(stage)
            & scoped["step"].astype(str).eq(name)
        ]
        return None if hit.empty else hit.iloc[-1]

    result: dict[str, object] = {
        "run_id": resolved,
        "rows_in_scope": int(len(scoped)),
    }
    _row_accounting(row(GUARD_STEP), result)
    _canonical_accounting(row(CANONICAL_STEP), result)
    _enforce_row_identity(result)
    decisions = _census_accounting(scoped, "gate.decision_", run_id=resolved)
    if decisions:
        result["gate_decisions"] = decisions
        result["gate_pairs"] = sum(decisions.values())
    labels = _census_accounting(scoped, "labels.destiny_", run_id=resolved)
    if labels:
        result["label_destiny"] = labels
        result["label_pairs"] = sum(labels.values())
    return result


def _raw_detail_text(detail: object) -> str:
    """Serialize a detail payload deterministically, or pass text through."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, sort_keys=True, default=str)


def _elided_value(value: object) -> object:
    """One detail VALUE, elided IN PLACE with the elided size stated."""
    if isinstance(value, str) and len(value) > DETAIL_VALUE_CHARS:
        elided = len(value) - DETAIL_VALUE_CHARS
        return f"{value[:DETAIL_VALUE_CHARS]}…<elided {elided} chars>"
    if isinstance(value, (list, tuple)) and len(value) > CENSUS_TOP_N:
        elided = len(value) - CENSUS_TOP_N
        return [*value[:CENSUS_TOP_N], f"…<elided {elided} items>"]
    return value


def _bounded_detail_text(detail: object, text: str) -> str:
    """A ``detail`` cell that exceeded :data:`DETAIL_CELL_CHARS`.

    The NAMED numbers are the audit's anchor ("gtin_missing_or_nan=41545"), so a
    cap may cut a value's tail but must never drop its key: a long string is
    truncated in place, a long list keeps its head, and both state how much was
    elided. Only a payload that still cannot fit (thousands of keys) degrades to
    its key list plus the original size — a bounded cell that says so, never a
    megabyte. Always valid JSON: the result is re-measured, never sliced.
    """
    if isinstance(detail, Mapping):
        capped: dict[str, object] = {
            str(key): _elided_value(value) for key, value in detail.items()
        }
        capped["detail_chars_before"] = len(text)
        rendered = json.dumps(capped, sort_keys=True, default=str)
        if len(rendered) <= DETAIL_CELL_CHARS:
            return rendered
        keys: list[str] = sorted(str(key)[:64] for key in detail)[:CENSUS_TOP_N]
    else:
        keys = []
    for budget in (
        DETAIL_CELL_CHARS // 2,
        DETAIL_CELL_CHARS // 4,
        DETAIL_CELL_CHARS // 16,
        0,
    ):
        payload: dict[str, object] = {
            "detail_truncated": True,
            "detail_chars_before": len(text),
            "keys": keys,
            "head": text[:budget],
        }
        rendered = json.dumps(payload, sort_keys=True, default=str)
        if len(rendered) <= DETAIL_CELL_CHARS:
            return rendered
    return json.dumps({"detail_truncated": True, "detail_chars_before": len(text)})


def _detail_text(detail: object) -> str:
    """Serialize a ``detail`` payload deterministically, within the cell cap.

    Every row's detail goes through here, so :data:`DETAIL_CELL_CHARS` is the
    ONE place a runaway readback (a per-pair dimension used as a value, a
    free-text reason list) stops being written.
    """
    text = _raw_detail_text(detail)
    if len(text) <= DETAIL_CELL_CHARS:
        return text
    bounded = _bounded_detail_text(detail, text)
    _LOG.warning(
        f"[trace] detail cell of {len(text)} chars exceeded "
        f"DETAIL_CELL_CHARS={DETAIL_CELL_CHARS}; elided to {len(bounded)} chars "
        f"(keys preserved, values elided — nothing silently dropped)"
    )
    return bounded


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_scope(scope: str) -> None:
    """Only the three declared scopes exist; anything else is a producer bug."""
    if scope not in (SCOPE_RUN, SCOPE_ENTITY, SCOPE_GROUP):
        raise ValueError(
            f"unknown trace scope {scope!r}; expected one of "
            f"{SCOPE_RUN!r}, {SCOPE_ENTITY!r}, {SCOPE_GROUP!r}"
        )


def _require_stage_step(stage: str, step: str) -> None:
    """Both stage and step are mandatory (rows require a grep anchor)."""
    if not stage or not step:
        raise ValueError("trace rows require both stage and step")


def _validated_counts(
    in_count: int | None, out_count: int | None
) -> tuple[int | None, int | None]:
    """Counts become integral-or-None and must never be negative."""
    incoming = None if in_count is None else int(in_count)
    outgoing = None if out_count is None else int(out_count)
    if incoming is not None and incoming < 0:
        raise ValueError(f"trace in_count must be >= 0, got {incoming}")
    if outgoing is not None and outgoing < 0:
        raise ValueError(f"trace out_count must be >= 0, got {outgoing}")
    return incoming, outgoing


def _dropped_count(
    incoming: int | None, outgoing: int | None
) -> int | None:
    """The funnel closes arithmetically: dropped = in − out, derived only."""
    if incoming is not None and outgoing is not None:
        return incoming - outgoing
    return None


def _checked_row(row: dict[str, object]) -> dict[str, object]:
    """Validate a freshly built row against the SHARED contract (core.schemas).

    Lazy import on purpose: ``core.schemas`` imports ``TRACE_COLUMNS`` from this
    module at module scope, so a module-level import would be a cycle. This is
    the ROW boundary from the schemas docstring — a row dies at its producer
    instead of in the csv, so a hand-written scope typo, a whitespace-only
    stage, or a derived-dropped_count disagreement (e.g. in_count=1,
    out_count=5 states dropped_count=-4, which the contract forbids) can never
    reach the file. The contract itself stays declared ONCE, in core.schemas.
    """
    from core.schemas import TraceRow

    TraceRow.model_validate(row)
    return row


def lane_producer(lane: str | None) -> str:
    """The ``producer`` cell for a writer: the module, qualified by its lane.

    ``producer`` is the row's "which module wrote this row" axis, and a run's
    lanes are processes: the suite's two trained tracks BOTH write stage
    ``worker``, so without the qualifier one lane's commit would REPLACE the
    other lane's rows (a stage commit replaces its own (run, stage, producer)
    subset). Unlaned writers keep the historical constant, so every other
    stage's commit behaves exactly as before.
    """
    return PRODUCER if not lane else f"{PRODUCER}:{lane}"


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
    lane: str | None = None,
    at: str | None = None,
) -> dict[str, object]:
    """Build one trace row. ``dropped_count`` is always derived here.

    The row is validated against ``core.schemas.TraceRow`` before it is
    returned (see :func:`_checked_row`), so this function is the fail-loud
    producer boundary for every step/run/group row in the tree.
    """
    _require_scope(scope)
    _require_stage_step(stage, step)
    incoming, outgoing = _validated_counts(in_count, out_count)
    return _checked_row({
        # Default to the resolved run identity rather than "". An unlabelled
        # row cannot survive: TraceRow requires run_id min_length=1, and
        # _commit DROPS rows whose run_id is empty as legacy-unlabelled. So a
        # "" default manufactured rows that were invalid on arrival and
        # silently discarded later — the writer/contract drift the selftest
        # oracle exists to catch. resolve_run_id() already encodes the
        # unattributed case as RUN_UNBOUND.
        "run_id": str(run_id or resolve_run_id()),
        "stage": str(stage),
        "step": str(step),
        "scope": scope,
        "key": "" if key is None else str(key),
        "in_count": incoming,
        "out_count": outgoing,
        "dropped_count": _dropped_count(incoming, outgoing),
        "reason": "" if reason is None else str(reason),
        "detail": _detail_text(detail),
        "source": "" if source is None else str(source),
        "producer": lane_producer(lane),
        "at": at or _now(),
    })


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


def _bucket_order(counts: Mapping[str, int]) -> list[str]:
    """Buckets ordered by ``(-count, name)``: data-determined, never dict order."""
    return sorted(counts, key=lambda name: (-int(counts[name]), name))


def _nonempty_buckets_after(
    counts: Mapping[str, int], buckets: list[str], position: int
) -> int:
    """How many still-to-come buckets need a reservation this pass can't eat."""
    return sum(
        1 for later in buckets[position + 1 :] if int(counts[later]) > 0
    )


def _reserve_pass(
    counts: Mapping[str, int],
    buckets: list[str],
    quota: dict[str, int],
    *,
    per_reason: int,
    budget: int,
) -> int:
    """Pass 1: every non-empty bucket gets ``min(count, per_reason)`` rows.

    RESERVING one row per still-to-come bucket so a bucket can never be
    squeezed out by a bigger one ahead of it (with a budget smaller than the
    number of buckets that reservation is impossible; those buckets still get
    their exact census row, which carries sample keys). Returns the budget
    spent so far; quota only grows, so the running total equals its recomputed
    sum.
    """
    spent = 0
    for position, name in enumerate(buckets):
        if int(counts[name]) <= 0:
            continue
        reserve = _nonempty_buckets_after(counts, buckets, position)
        give = min(
            int(counts[name]), int(per_reason), budget - spent - reserve
        )
        if give > 0:
            quota[name] = give
            spent += give
    return spent


def _leftover_pass(
    counts: Mapping[str, int],
    buckets: list[str],
    quota: dict[str, int],
    *,
    budget: int,
) -> None:
    """Pass 2: spend what is left on the largest populations first."""
    spent = sum(quota.values())
    for name in buckets:
        room = int(counts[name]) - quota[name]
        if room <= 0:
            continue
        give = min(room, budget - spent)
        if give > 0:
            quota[name] += give
            spent += give
        if spent >= budget:
            break


def _plan_sample(
    counts: Mapping[str, int],
    *,
    per_reason: int,
    total_cap: int,
) -> dict[str, int]:
    """Allocate the entity-row budget across buckets (deterministic).

    See :func:`_reserve_pass` for the reservation rule; the plan depends only
    on the data, never on dict iteration order.
    """
    buckets = _bucket_order(counts)
    quota = {name: 0 for name in buckets}
    budget = max(0, int(total_cap))
    if not buckets or budget <= 0:
        return quota
    _reserve_pass(
        counts, buckets, quota, per_reason=int(per_reason), budget=budget
    )
    _leftover_pass(counts, buckets, quota, budget=budget)
    return quota


def _drop_unlabelled_rows(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Rule 1 of commit: unlabelled rows (``run_id == ""``) are dropped.

    The row contract has no anonymous run; a legacy file is silently READ but
    never allowed to masquerade as part of the current run.
    """
    if frame.empty:
        return frame
    labelled = frame["run_id"].astype(str).str.strip() != ""
    n_unlabelled = int((~labelled).sum())
    if n_unlabelled:
        _LOG.warning(
            f"[trace] dropping {n_unlabelled} legacy row(s) with no run_id "
            f"from {stage}: the row contract has no anonymous run"
        )
    return frame[labelled]


def _retain_recent_runs(
    frame: pd.DataFrame, run_id: str, history: int
) -> pd.DataFrame:
    """Rule 2 of commit: only ``history`` runs survive, whole runs at a time.

    First-seen order defines age (oldest out); the current run is never pruned.
    """
    if frame.empty:
        return frame
    order: list[str] = []
    for value in frame["run_id"].astype(str):
        if value not in order:
            order.append(value)
    if run_id not in order:
        order.append(run_id)
    keep = order[-max(1, int(history)) :]
    return frame[frame["run_id"].astype(str).isin(keep)]


def _owns(
    frame: pd.DataFrame, run_id: str, stage: str, producer: str
) -> pd.Series:
    """The rows one writer's commit owns: (run, stage, producer).

    ``producer`` is the writer's identity — the module, and for a parallel lane
    the module qualified by the lane (see :func:`lane_producer`). With the plain
    constant this is exactly the historical (run, stage) key, so every existing
    stage behaves as before; a lane-qualified writer replaces only its own rows,
    which is what lets two lanes of one module share the run's ONE trace without
    claiming one another's rows.
    """
    return (
        frame["run_id"].astype(str).eq(run_id)
        & frame["stage"].astype(str).eq(stage)
        & frame["producer"].astype(str).eq(producer)
    )


def _merge_or_replace_stage_rows(
    frame: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    run_id: str,
    stage: str,
    producer: str,
) -> pd.DataFrame:
    """Rule 3 of commit: this WRITER's rows for this run REPLACE IN PLACE.

    Same position as the rows being replaced, so flow order survives a re-run
    even when a later stage's rows are already in the file; a writer with no
    prior rows is concatenated at the end.
    """
    mine = _owns(frame, run_id, stage, producer)
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


def _commit(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    run_id: str,
    stage: str,
    producer: str = PRODUCER,
    history: int = TRACE_RUN_HISTORY,
) -> pd.DataFrame:
    """Commit one writer's rows into the run-scoped trace (see module docstring).

    Three rules, in order: drop unlabelled legacy rows
    (:func:`_drop_unlabelled_rows`), prune whole runs past
    :func:`_retain_recent_runs`, then replace this writer's own rows in place
    (:func:`_merge_or_replace_stage_rows`).
    """
    if existing.empty and not incoming.empty:
        return incoming.reset_index(drop=True)
    frame = _drop_unlabelled_rows(existing, stage)
    frame = _retain_recent_runs(frame, run_id, history)
    if frame.empty:
        return incoming.reset_index(drop=True)
    return _merge_or_replace_stage_rows(
        frame, incoming, run_id=run_id, stage=stage, producer=producer
    )


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

    Owner-class responsibilities:

      - bucketize a record set and record its exact census -> _bucketize_recordset
      - emit the group (census) rows                        -> _census_rows
      - emit the bounded entity-sample rows                 -> _entity_rows
      - emit the sample-budget run row                      -> _sample_budget_row
      - resolve and stamp the run identity                  -> _resolved_identity
    """

    def __init__(
        self, stage: str, run_id: str | None = None, lane: str | None = None
    ) -> None:
        self.stage = str(stage)
        # An explicit run id is for hermetic callers (tests, smokes). The
        # default is resolved at WRITE time, never at construction: stage 1
        # constructs its writer before it writes the very artifacts the run
        # fingerprint is taken from, so an early resolution would tag stage 1
        # with the PREVIOUS run and split the two stages apart.
        self._run_id = run_id
        # The lane this process writes for: explicit for hermetic callers, else
        # the launcher's pin (see the pin block above). It qualifies the row's
        # ``producer`` so two lanes of the same module own disjoint row sets in
        # the run's ONE trace instead of replacing one another.
        self._lane = str(lane).strip() if lane else self._env_lane()
        self._identity: dict[str, object] | None = None
        self._rows: list[dict[str, object]] = []

    @staticmethod
    def _env_lane() -> str:
        """The launcher's lane pin, or "" for a single-writer stage."""
        return str(os.environ.get(TRACE_LANE_ENV, "")).strip()

    @property
    def lane(self) -> str:
        return self._lane

    @property
    def producer(self) -> str:
        """This writer's identity in the trace (module, lane-qualified)."""
        return lane_producer(self._lane)

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
            lane=self._lane,
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

        Stage 1 consumes the raw export and stage 2 consumes the deduped
        dataset. The two frames are different DATASETS and the handoff between
        them used to be invisible — a frame with the wrong column names produced
        a KeyError far from its cause — so this row makes the contract, and any
        missing required column, part of the trace.

        Since commit c698200 every column carries the raw export's own name
        (``gtin``/``sku_name_eng``/``attribute``), which made COLUMN_MAPPING the
        identity map. The two requirement lists are therefore the SAME list
        today: both are derived from ``DATA_PREP_REQUIRED_COLUMNS``. The rows
        stay per-stage because the frames are still separately loaded and
        separately validated, but they no longer distinguish the vocabularies
        and must not be read as if they did.

        A UNIT CHANGE, not a funnel: ROWS received, COLUMNS in the contract. The
        row therefore states only its output (the contract's size) and carries
        the frame's row count in its detail — an in/out pair here would claim a
        drop of ``rows - columns``, which is not a drop at all, and on a frame
        narrower than its own column set it would be a NEGATIVE drop, which the
        row contract forbids (so the write would reject the stage's own row).
        """
        columns = [str(column) for column in frame.columns]
        missing = [name for name in required if name not in columns]
        return self.add(
            "column_contract",
            "input_frame",
            out_count=len(columns),
            reason=(
                f"stage consumed the {contract} column contract"
                + (f"; MISSING REQUIRED {missing}" if missing else "")
            ),
            detail={
                "contract": contract,
                "rows": int(len(frame)),
                "columns": columns,
                "required": list(required),
                "missing_required": missing,
                "note": note,
            },
            source=contract,
        )

    # ── entity sampling: the three owned row kinds ─────────────────────────

    def _bucketize_recordset(
        self, records: Iterable[object], reason_of: object
    ) -> tuple[int, dict[str, list[object]], dict[str, int]]:
        """Population size, labels -> the exact per-bucket membership and counts."""
        rows = list(records)
        buckets: dict[str, list[object]] = {}
        for value in rows:
            label = "" if reason_of is None else str(reason_of(value))
            buckets.setdefault(label, []).append(value)
        counts = {name: len(values) for name, values in buckets.items()}
        return len(rows), buckets, counts

    def _entity_row(
        self,
        step: str,
        reason: str,
        value: object,
        *,
        key_of: object,
        detail_of: object,
        source: object,
        stamp: str,
    ) -> dict[str, object]:
        """One entity row: the sampled pair carries its bucket as its reason."""
        return {
            "stage": self.stage,
            "step": str(step),
            "scope": SCOPE_ENTITY,
            "key": "" if key_of is None else str(key_of(value)),
            "in_count": None,
            "out_count": None,
            "dropped_count": None,
            "reason": reason,
            "detail": _detail_text(
                None if detail_of is None else detail_of(value)
            ),
            "source": "" if source is None else str(source),
            "producer": self.producer,
            "run_id": self.run_id,
            "at": stamp,
        }

    def _entity_rows(
        self,
        step: str,
        buckets: Mapping[str, list[object]],
        counts: Mapping[str, int],
        quota: Mapping[str, int],
        *,
        key_of: object,
        detail_of: object,
        source: object,
        stamp: str,
    ) -> list[dict[str, object]]:
        """The sampled ENTITY rows, in bucket order, within each bucket's quota."""
        sampled: list[dict[str, object]] = []
        for name in sorted(buckets, key=lambda key: (-counts[key], key)):
            chosen = buckets[name][: int(quota.get(name, 0))]
            for value in chosen:
                sampled.append(
                    self._entity_row(
                        step,
                        name,
                        value,
                        key_of=key_of,
                        detail_of=detail_of,
                        source=source,
                        stamp=stamp,
                    )
                )
        return sampled

    def _census_detail(
        self, bucket: list[object], take: int, *, key_of: object
    ) -> dict[str, object]:
        """A group row's readback: exact population, sample slice, sample keys."""
        population = len(bucket)
        detail: dict[str, object] = {
            "population": population,
            "sampled": take,
            "omitted": population - take,
        }
        if population > take:
            detail["sample_keys"] = [
                "" if key_of is None else str(key_of(value))
                for value in bucket[:5]
            ]
        return detail

    def _census_rows(
        self,
        step: str,
        buckets: Mapping[str, list[object]],
        counts: Mapping[str, int],
        quota: Mapping[str, int],
        *,
        key_of: object,
        source: object,
        stamp: str,
    ) -> list[dict[str, object]]:
        """One GROUP row per bucket: the EXACT census of every reason bucket."""
        rows: list[dict[str, object]] = []
        for name in sorted(buckets, key=lambda key: (-counts[key], key)):
            take = int(quota.get(name, 0))
            rows.append(
                record(
                    self.stage,
                    f"{step}.reason_census",
                    scope=SCOPE_GROUP,
                    key="",
                    in_count=counts[name],
                    out_count=take,
                    reason=name,
                    detail=self._census_detail(
                        buckets[name], take, key_of=key_of
                    ),
                    source=source,
                    run_id=self.run_id,
                    lane=self._lane,
                    at=stamp,
                )
            )
        return rows

    def _sample_budget_row(
        self,
        step: str,
        *,
        population: int,
        sampled: int,
        n_buckets: int,
        per_reason: int,
        total_cap: int,
        source: object,
        stamp: str,
    ) -> dict[str, object]:
        """One RUN row announcing the sampling budget actually spent."""
        return record(
            self.stage,
            f"{step}.sample_budget",
            in_count=population,
            out_count=sampled,
            reason=(
                "every reason bucket is censused exactly above; the entity "
                "rows are a bounded stratified sample of that census"
            ),
            detail={
                "population": population,
                "sampled": sampled,
                "omitted": population - sampled,
                "buckets": n_buckets,
                "per_reason": int(per_reason),
                "total_cap": int(total_cap),
                "full_census": "" if source is None else str(source),
            },
            source=source,
            run_id=self.run_id,
            lane=self._lane,
            at=stamp,
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
        population, buckets, counts = self._bucketize_recordset(records, reason_of)
        quota = _plan_sample(
            counts, per_reason=int(per_reason), total_cap=int(total_cap)
        )
        stamp = self._stamp()
        emitted: list[dict[str, object]] = self._census_rows(
            step, buckets, counts, quota, key_of=key_of, source=source, stamp=stamp
        )
        sampled_rows = self._entity_rows(
            step, buckets, counts, quota,
            key_of=key_of, detail_of=detail_of, source=source, stamp=stamp,
        )
        emitted.extend(sampled_rows)
        sampled = len(sampled_rows)
        emitted.append(
            self._sample_budget_row(
                step,
                population=population,
                sampled=sampled,
                n_buckets=len(buckets),
                per_reason=int(per_reason),
                total_cap=int(total_cap),
                source=source,
                stamp=stamp,
            )
        )
        self._rows.extend(emitted)
        return {
            "population": population,
            "sampled": sampled,
            "omitted": population - sampled,
            "per_reason": {name: counts[name] for name in counts},
            "sampled_per_reason": {
                name: int(quota.get(name, 0)) for name in counts
            },
        }

    def rows(self) -> pd.DataFrame:
        """This stage's rows: the ``run_identity`` row, then the recorded steps."""
        rows = [self._identity_row(), *self._rows]
        return pd.DataFrame(rows, columns=list(TRACE_COLUMNS))

    @timed
    def write(self, path: Path | None = None) -> Path:
        """Commit this writer's rows to the consolidated trace.

        One run has ONE trace, so a run's lanes commit to the same file while
        their siblings are still running: the read -> merge -> write cycle runs
        under a file lock (:func:`_trace_lock`), or two lanes that read before
        either wrote would each publish a frame without the other's rows.

        The row contract is checked at THREE points:

        1. THIS COMMIT's rows, before they are merged — fail-loud, so a producer
           that emits an invalid row (a negative ``dropped_count``, a unit change
           stated as a funnel) is named as the culprit and nothing is written.
        2. The rows of the RUN BEING WRITTEN, before the atomic write —
           fail-loud: the run this commit belongs to must satisfy the contract
           its readers enforce.
        3. Rows of OTHER runs already in the file — reported loudly, NOT fatal.
           A historical run is immutable history: it says nothing about the write
           being made, and failing on it would wedge the pipeline on stale data
           (the prune happens in the very commit that would fail, so no write
           could ever heal the file). The rows are named so the offending run can
           be regenerated; the trace is a regenerated artifact.
        """
        from core.manifest import atomic_write_csv

        target = path if path is not None else trace_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with _trace_lock(target):
            incoming = self._incoming_frame()
            assert_trace_frame(
                incoming,
                path=(
                    f"{target} (producer {self.producer!r}, stage {self.stage!r}, "
                    f"run {self.run_id!r})"
                ),
            )
            frame = _commit(
                read_trace(target),
                incoming,
                run_id=self.run_id,
                stage=self.stage,
                producer=self.producer,
            )
            frame = _normalize_counts(frame)
            assert_trace_frame(
                self._rows_of_this_run(frame),
                path=f"{target} (run {self.run_id!r})",
            )
            _report_other_runs(frame, target=target, run_id=self.run_id)
            atomic_write_csv(frame, target, index=False)
        return target

    def _rows_of_this_run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The frame's rows belonging to the run this writer commits."""
        return frame[frame["run_id"].astype(str).eq(self.run_id)]

    def _incoming_frame(self) -> pd.DataFrame:
        """This stage's rows with the run id forced onto every one of them.

        The force is load-bearing: the ``run_identity`` row (and any row added
        before the resolved id was known) may have been built while the run id
        was still unbound, but after resolution the whole commit belongs to the
        one resolved run.
        """
        incoming = self.rows()
        incoming["run_id"] = self.run_id
        return incoming

    def _resolved_identity(self) -> dict[str, object]:
        """The run identity, resolved once per commit, or the caller's explicit one."""
        if self._run_id is None:
            identity = resolve_run_identity()
            self._run_id = str(identity["run_id"])
            self._identity = identity
        return self._identity or {
            "run_id": self.run_id,
            "resolution": "explicit run id supplied by the caller",
            "sources": {},
        }

    def _identity_row(self) -> dict[str, object]:
        """The stage's first row: which run these rows belong to, and why.

        Written automatically (not by the caller) so every stage commit is
        self-describing: the run id, the rule that produced it, and the size +
        sha256 of each artifact the fingerprint was taken from. Two runs in one
        file are therefore explainable from the file.
        """
        identity = self._resolved_identity()
        return record(
            self.stage,
            "run_identity",
            reason=f"run id resolved by {identity['resolution']}",
            detail={
                "run_id": identity["run_id"],
                "resolution": identity["resolution"],
                "policy": (
                    "run-scoped idempotent append: a stage REPLACES its own rows "
                    "for this run; different runs stay distinguishable by run_id"
                ),
                "sources": identity["sources"],
                "rows_in_stage": len(self._rows),
            },
            source="config/paths.yaml layout training_trace",
            run_id=self.run_id,
            lane=self._lane,
            at=self._stamp(),
        )

    def _stamp(self) -> str:
        """One timestamp per row; stages share it so rows read as a snapshot."""
        return self._rows[-1]["at"] if self._rows else _now()  # type: ignore[return-value]


# ── the ONE lazy stage-writer shim ─────────────────────────────────────────
# Every producer used to carry its own private copy of the same two functions:
# a writer slot (``_TRACE = None`` at module scope), a ``trace()`` that lazily
# constructed ``TraceRun(STAGE)`` on first use, and a ``flush_trace()`` that
# committed it once and was a no-op while empty. Eleven copies of that logic
# meant eleven chances to drift (one of them, ``model_tracks.ablation``, kept a
# whole paragraph of rationale about NOT resetting, and the shim in
# ``training.training`` was the only one that reset). The logic lives here ONCE;
# a producer declares its slot and delegates:
#
#     _TRACE = None                      # the process-lifetime writer slot
#     trace, flush_trace = ...           # (kept as thin delegates so tests can
#                                        #  reset the slot: module._TRACE = None)
#
# The slot stays the producer's own module global so a test (and a re-run in
# one process) can clear it, and so the writer is never shared between stages.
#
# LAZY, and the run id is resolved at WRITE time by :class:`TraceRun` (its
# doctrine): importing a producer module therefore never touches the trace
# layout, and a row recorded before the run artifacts exist still lands in
# whichever run finally commits it.
def stage_trace(
    stage: str, current: TraceRun | None = None, *, pinned: str | None = None
) -> TraceRun:
    """This process's writer for ``stage``, created on first use.

    ``current`` is the caller's module-level slot (``None`` until first use);
    the returned writer is assigned back to it. ``pinned`` optionally names the
    stage on that first creation and REFUSES a later relabel: the training side
    runs its bundle lane under the orchestrator's ``full_bundle`` stage (a
    one-off pin), and rows already recorded carry the old stage, so
    ``core.tracing`` commits a writer's rows as ONE stage and relabelling
    mid-run would publish them under the wrong one.

    The writer is NEVER reset here. A stage commits run-scoped and idempotently
    (a re-write REPLACES that stage's rows for the run), so a producer whose
    entry point runs once per track inside one suite (``model_tracks.ablation``,
    ``post_training_ablation``, ``bundle_steps.finalize``, every worker lane)
    must keep accumulating: resetting would make the second track's commit
    silently replace the first track's rows. The one deliberate variant is
    ``training.training.flush_training_trace``, which releases its slot after a
    commit so the NEXT run starts a fresh writer under its own run id; it does
    that at its own call site (see :func:`flush_stage_trace`).
    """
    if current is not None:
        if pinned is not None and str(current.stage) != str(pinned):
            raise ValueError(
                f"the trace writer already owns stage {current.stage!r}; refusing "
                f"to relabel it {pinned!r} mid-run (recorded rows would be "
                "committed under the wrong stage)"
            )
        return current
    return TraceRun(pinned or stage)


def flush_stage_trace(current: TraceRun | None) -> Path | None:
    """Commit one stage's writer once; a no-op while it is absent or empty.

    Returns the destination path (``None`` when there was nothing to commit), so
    a caller can tell "wrote" from "no-op" without re-reading the writer. The
    writer is deliberately retained (see :func:`stage_trace`): rows added after
    a flush are committed by the next flush instead of being dropped. A caller
    that WANTS to release the slot (the one reset-on-flush variant,
    ``training.training.flush_training_trace``) clears its own module global when
    this returns a path.
    """
    if current is None or len(current) == 0:
        return None
    return current.write()


def _census_cell(
    listed: list[tuple[str, int]], remainder: list[tuple[str, int]]
) -> list[str]:
    """``"value=n"`` entries for ``listed`` plus the explicit remainder as NUMBERS."""
    entries = [f"{name}={count}" for name, count in listed]
    if remainder:
        entries.append(f"others={sum(count for _, count in remainder)}")
        entries.append(f"others_buckets={len(remainder)}")
    return entries


def count_rows(values: object, *, limit: int | None = None) -> list[str]:
    """Return the most common values as ``"value=n"`` strings for ``detail``.

    BOUNDED BY POLICY, always. ``limit`` (default :data:`CENSUS_TOP_N`) is how
    many values are LISTED; everything past it is aggregated into one explicit
    ``others=<n>`` entry plus ``others_buckets=<n>``, and the cell is folded
    further until it fits :data:`DETAIL_CELL_CHARS`. ``limit=None`` therefore
    means "the policy cap", NOT "unbounded" — an unbounded readback of a
    per-pair dimension is what minted a 1.07 MB cell (13,067 strings) on the
    live census. ``value_counts`` is sorted by count, so the listed values are
    the most common ones and the remainder is the long tail; the totals are
    still stated exactly, so a bounded cell hides nothing.

    When the cell must fold, the SMALLEST listed values move into ``others``
    first: the largest populations are the most informative and stay.
    """
    series = pd.Series(list(values))
    if series.empty:
        return []
    couples = [
        (str(name), int(count))
        for name, count in series.astype(str).value_counts().items()
    ]
    listed = couples[: max(0, CENSUS_TOP_N if limit is None else int(limit))]
    remainder = list(couples[len(listed) :])
    while listed and len(json.dumps(_census_cell(listed, remainder))) > (
        DETAIL_CELL_CHARS
    ):
        remainder.insert(0, listed.pop())
    return _census_cell(listed, remainder)


def sample_keys(values: object, *, limit: int = 5) -> list[str]:
    """Return a deterministic, bounded sample of keys for ``detail``."""
    unique = sorted({str(value) for value in list(values)})
    return unique[: int(limit)]


def _reject_missing_columns(frame: pd.DataFrame, path: object) -> None:
    if missing := [column for column in TRACE_COLUMNS if column not in frame.columns]:
        raise ValueError(
            f"trace frame {path} is missing columns {missing}; "
            f"expected {list(TRACE_COLUMNS)}"
        )


def _reject_undeclared_columns(frame: pd.DataFrame, path: object) -> None:
    if extra := [column for column in frame.columns if column not in TRACE_COLUMNS]:
        raise ValueError(
            f"trace frame {path} carries undeclared columns {extra}; "
            f"expected {list(TRACE_COLUMNS)}"
        )


def _reject_blank_stage_or_step(frame: pd.DataFrame, path: object) -> None:
    blank = frame["stage"].astype(str).str.strip().eq("") | frame["step"].astype(
        str
    ).str.strip().eq("")
    if bool(blank.any()):
        raise ValueError(
            f"trace frame {path} has {int(blank.sum())} rows without stage/step"
        )


def _reject_anonymous_rows(frame: pd.DataFrame, path: object) -> None:
    anonymous = frame["run_id"].astype(str).str.strip().eq("")
    if bool(anonymous.any()):
        raise ValueError(
            f"trace frame {path} has {int(anonymous.sum())} rows with no "
            f"run_id — every row must belong to a run"
        )


def _reject_unknown_scopes(frame: pd.DataFrame, path: object) -> None:
    unknown = ~frame["scope"].astype(str).isin([SCOPE_RUN, SCOPE_ENTITY, SCOPE_GROUP])
    if bool(unknown.any()):
        raise ValueError(
            f"trace frame {path} has {int(unknown.sum())} rows with an unknown scope"
        )


def _first_invalid_row_label(frame: pd.DataFrame) -> str:
    """``stage/step/run_id`` of the frame's first invalid row, or ``""``.

    A violation in a multi-run file is otherwise reported as a bare row index
    (7,552 rows, three runs), which names nothing an operator can act on. Runs
    only on the failure path.
    """
    from core.schemas import TraceRow

    for _, row in frame.iterrows():
        try:
            TraceRow.model_validate(dict(row))
        except Exception:
            return (
                f" (first invalid row: run_id={row['run_id']!r} "
                f"stage={row['stage']!r} step={row['step']!r})"
            )
    return ""


def _check_shared_frame_contract(frame: pd.DataFrame, path: object) -> None:
    """Run ``core.schemas.check_trace_frame`` over the frame.

    Lazy import: ``core.schemas`` imports ``TRACE_COLUMNS`` from this module at
    module scope, so a module-level import would be a cycle. The frame checker
    is the ONE boundary contract (it is registered in ``FRAME_CHECKERS``), and
    running it here wires the WRITE boundary: ``TraceRun.write`` calls
    :func:`assert_trace_frame` on the merged frame BEFORE
    ``core.manifest.atomic_write_csv``, so a corrupted stage (or a legacy row
    already in the file) can never land in results/logs/training_trace.csv.
    """
    from core.schemas import check_trace_frame

    try:
        check_trace_frame(frame)
    except ValueError as exc:
        raise ValueError(
            f"trace frame {path}: {exc}{_first_invalid_row_label(frame)}"
        ) from exc


def _report_other_runs(frame: pd.DataFrame, *, target: Path, run_id: str) -> None:
    """Report loudly (never fail) invalid rows that belong to OTHER runs.

    The rows of the run being written are checked fail-loud by the caller. A run
    that is already in the file is history: re-validating it on every commit
    says nothing about the write being made, and failing here would WEDGE the
    pipeline on stale data — the prune of old runs happens in the very commit
    that would fail, so no write could ever heal the file. The rows are named so
    the offending run can be regenerated (the trace is a regenerated artifact).
    """
    others = frame[~frame["run_id"].astype(str).eq(run_id)]
    if others.empty:
        return
    try:
        assert_trace_frame(others, path=f"{target} (rows of other runs)")
    except ValueError as exc:
        _LOG.warning(
            f"[trace] {int(len(others))} row(s) belonging to OTHER runs in "
            f"{target} do not satisfy the row contract; they are left untouched "
            f"(regenerate or prune the offending run to clear them): {exc}"
        )


@contextmanager
def _trace_lock(target: Path) -> Iterator[None]:
    """Serialize the read -> merge -> write cycle of a trace's writers.

    ONE trace per run means a run's lanes commit to the same file CONCURRENTLY
    (the suite's trained tracks both write while their sibling runs). The merge
    is a read-modify-write and ``atomic_write_csv`` only makes the replace
    atomic: two lanes that both read before either wrote would each publish a
    frame without the other's rows, dropping them silently. The lock is a
    SIBLING file, never the trace itself — the atomic replace swaps the trace's
    inode, so a lock held on the trace would guard a file nobody writes to next.
    """
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX runtime
        _LOG.warning(
            f"[trace] no fcntl on this platform: concurrent writers of {target} "
            f"are not serialized"
        )
        yield
        return
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def adopt_run(
    path: Path | None = None, *, from_run_id: str, to_run_id: str
) -> int:
    """Move one run's rows onto its real run id, whole rows at a time.

    A run's identity is its artifacts' fingerprint (``run_artifact_fingerprint``),
    and the stages that run BEFORE those artifacts exist cannot know it: resolving
    then would return the PREVIOUS run's fingerprint and attribute this run's rows
    to another run. ``prepare_all`` therefore pins ``TRACE_PENDING_RUN`` for
    those stages and adopts their rows here, once the run's own artifacts exist,
    so the whole preparation run is ONE run in the file — the same id the
    training side reproduces from the same artifacts.

    Rewritten together with the run's ``run_identity`` rows, whose detail states
    the resolution rule: after adoption it names the adoption rather than the
    pin that has been superseded. Returns the number of rows adopted.
    """
    target = path if path is not None else trace_path()
    if str(from_run_id) == str(to_run_id):
        return 0
    if not target.exists():
        return 0
    from core.manifest import atomic_write_csv

    with _trace_lock(target):
        frame = read_trace(target)
        mine = frame["run_id"].astype(str).eq(str(from_run_id))
        if not bool(mine.any()):
            return 0
        frame.loc[mine, "run_id"] = str(to_run_id)
        for index, row in frame[mine].iterrows():
            if str(row["step"]) != "run_identity":
                continue
            detail = detail_json(row["detail"])
            detail["run_id"] = str(to_run_id)
            detail["resolution"] = (
                f"adopted from {from_run_id!r}: this run's rows are tagged with "
                f"the run identity its own artifacts define"
            )
            frame.loc[index, "detail"] = _detail_text(detail)
        frame = _normalize_counts(frame)
        assert_trace_frame(
            frame[frame["run_id"].astype(str).eq(str(to_run_id))],
            path=f"{target} (adopted run {to_run_id!r})",
        )
        _report_other_runs(frame, target=target, run_id=str(to_run_id))
        atomic_write_csv(frame, target, index=False)
        return int(mine.sum())


def assert_trace_frame(frame: pd.DataFrame, *, path: object = "") -> None:
    """Fail loudly if a trace frame violates the row contract.

    Two layers, ONE contract. The checks below give the terse, trace-specific
    messages the writers rely on (a missing column, a blank stage, an
    anonymous run, an unknown scope), and then the whole frame goes through
    ``core.schemas.check_trace_frame`` / ``TraceRow`` — the same model the
    file's readers use — so the row contract cannot drift between the module
    that writes it and the module that validates it.
    """
    _reject_missing_columns(frame, path)
    _reject_undeclared_columns(frame, path)
    if frame.empty:
        return
    _reject_blank_stage_or_step(frame, path)
    _reject_anonymous_rows(frame, path)
    _reject_unknown_scopes(frame, path)
    _check_shared_frame_contract(frame, path)


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
    "CANONICAL_IDENTITY_TERMS",
    "CANONICAL_STEP",
    "CENSUS_TOP_N",
    "DETAIL_CELL_CHARS",
    "DETAIL_VALUE_CHARS",
    "ENTITY_ROW_CAP",
    "ENTITY_SAMPLE_PER_REASON",
    "GUARD_IDENTITY_TERMS",
    "GUARD_STEP",
    "orchestration_lane_stages",
    "orchestration_trace_stages",
    "PRODUCER",
    "RUN_UNBOUND",
    "SCOPE_ENTITY",
    "SCOPE_GROUP",
    "SCOPE_RUN",
    "TRACE_BATCH_ROWS",
    "TRACE_COLUMNS",
    "TRACE_LANE_ENV",
    "TRACE_MAX_BATCH_ROWS",
    "TRACE_PATH_ENV",
    "TRACE_PENDING_RUN",
    "TRACE_RUN_HISTORY",
    "TraceRun",
    "accounting",
    "adopt_run",
    "assert_trace_frame",
    "count_rows",
    "detail_json",
    "flush_stage_trace",
    "lane_producer",
    "read_trace",
    "record",
    "resolve_run_id",
    "resolve_run_identity",
    "run_artifact_fingerprint",
    "run_trace_env",
    "sample_keys",
    "stage_trace",
    "trace_path",
    "trace_stages_for",
]
