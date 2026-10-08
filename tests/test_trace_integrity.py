"""Trace INTEGRITY tests — the three blockers an independent detective found.

The three properties locked here are the ones that let a reader answer "why did
this row end at this label?" from ``results/logs/training_trace.csv`` alone:

1. ``accounting()`` reads ONE RUN. The file holds up to ``TRACE_RUN_HISTORY``
   runs, so a run-agnostic read assembles an identity from two different runs
   (the last matching guard row next to whichever census row the file ends on).
   It also states the identity with its real terms:
   ``rows_in == rows_retained + gtin_missing_or_nan + gs1_checksum_failed +
   identity_review_quarantined`` and
   ``guard.out == canonical.out + collapsed_same_gtin``.
2. The row contract is enforced at the WRITE boundary, so a bad row (a unit
   change stated as a funnel, i.e. a negative ``dropped_count``) cannot be
   written by its producer and cannot ride along in the file.
3. A census row's readback cells are BOUNDED: ``count_rows`` lists the top-N
   values and states the remainder as numbers, and a ``detail`` cell is capped,
   so no single row can emit a megabyte.

Everything here is the PUBLIC surface (``record``, ``TraceRun``, ``accounting``,
``assert_trace_frame``, ``count_rows``) — the same interface a stage or a
dashboard reads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from core.tracing import (
    CENSUS_TOP_N,
    DETAIL_CELL_CHARS,
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    GUARD_IDENTITY_TERMS,
    SCOPE_GROUP,
    TRACE_COLUMNS,
    TraceRun,
    accounting,
    assert_trace_frame,
    count_rows,
    detail_json,
    read_trace,
    record,
)

# A run whose guard populations close with the four documented terms, and whose
# canonical row closes the guard's output. The pair census is one gate decision
# and one label destiny per candidate pair.
RUN_A = {
    "run_id": "run-a",
    "rows_in": 1000,
    "retained": 400,
    "missing": 500,
    "checksum": 57,
    "quarantined": 43,
    "canonical": 300,
    "collapsed": 100,
    "decisions": {"hard_no": 12, "fallback": 3, "proceed": 25},
    "destinies": {"negative_hard": 12, "proceed_not_a_training_pair": 28},
}
RUN_B = {
    "run_id": "run-b",
    "rows_in": 50,
    "retained": 20,
    "missing": 25,
    "checksum": 4,
    "quarantined": 1,
    "canonical": 14,
    "collapsed": 6,
    "decisions": {"hard_no": 1, "fallback": 0, "proceed": 4},
    "destinies": {"negative_hard": 1, "proceed_not_a_training_pair": 4},
}


def _stage(run: dict) -> TraceRun:
    """One ``data_prep`` stage's rows for a run's identity."""
    stage = TraceRun("data_prep", run_id=run["run_id"])
    stage.add(
        "gtin_guard",
        "identity_claims_evaluated",
        in_count=run["rows_in"],
        out_count=run["retained"],
        reason="rows keep identity only with a present, GS1-valid gtin",
        detail={
            "gtin_missing_or_nan": run["missing"],
            "gs1_checksum_failed": run["checksum"],
            "identity_review_quarantined": run["quarantined"],
            "rows_retained": run["retained"],
        },
        source="raw export",
    )
    stage.add(
        "canonical",
        "records_built",
        in_count=run["retained"],
        out_count=run["canonical"],
        reason="one canonical record per distinct GS1-valid gtin",
        detail={"collapsed_same_gtin": run["collapsed"]},
        source="raw export",
    )
    for name, count in run["decisions"].items():
        stage.add(
            "gate",
            f"decision_{name}",
            scope=SCOPE_GROUP,
            in_count=count,
            out_count=count,
            reason=name,
        )
    for name, count in run["destinies"].items():
        stage.add(
            "labels",
            f"destiny_{name}",
            scope=SCOPE_GROUP,
            in_count=count,
            out_count=count,
            reason=name,
        )
    return stage


@pytest.fixture()
def two_run_file(tmp_path: Path) -> pd.DataFrame:
    """A real trace file holding TWO whole runs, largest first.

    Largest first on purpose: the run-agnostic read ("last matching row in the
    file") picks the SECOND run, so a test that distinguishes the two runs also
    distinguishes "one run" from "last row wins".
    """
    target = tmp_path / "trace.csv"
    _stage(RUN_A).write(target)
    _stage(RUN_B).write(target)
    return read_trace(target)


# ── 1. accounting reads ONE run ────────────────────────────────────────────
def test_accounting_scopes_to_one_run_instead_of_the_last_row(
    two_run_file, monkeypatch
):
    """``run_id`` selects a run; the default is the current run, not the last."""
    frame = two_run_file
    assert list(dict.fromkeys(frame["run_id"])) == ["run-a", "run-b"]

    for run in (RUN_A, RUN_B):
        measured = accounting(frame, run_id=run["run_id"])
        assert measured["run_id"] == run["run_id"]
        assert measured["rows_in"] == run["rows_in"]
        assert measured["rows_retained"] == run["retained"]
        # the run-axis bug: the run-agnostic read gives ONE run's guard row to
        # the other run's census, so the two runs would report the same numbers
        assert measured["gate_pairs"] == sum(run["decisions"].values())
        assert measured["label_pairs"] == sum(run["destinies"].values())

    # a frame is not a run: without the axis the identity is one run's
    pinned = accounting(frame, run_id="run-a")
    assert pinned["rows_in"] != accounting(frame, run_id="run-b")["rows_in"]

    # the default resolves the CURRENT run the way the writers do (env pin)
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-a")
    assert accounting(frame)["run_id"] == "run-a"
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-b")
    assert accounting(frame)["run_id"] == "run-b"


def test_accounting_refuses_an_unknown_or_ambiguous_run(two_run_file, monkeypatch):
    """An unresolvable run is a loud failure, never a silent last-row pick."""
    monkeypatch.delenv("EUROMONITOR_TRACE_RUN", raising=False)
    monkeypatch.delenv("EUROMONITOR_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="no rows for run_id"):
        accounting(two_run_file, run_id="run-does-not-exist")
    # the on-disk artifacts do not name either of these synthetic runs, and two
    # runs are present, so the choice is ambiguous and must be stated
    with pytest.raises(ValueError, match="pass run_id="):
        accounting(two_run_file)


def test_accounting_reports_a_single_run_frame_without_being_pinned(tmp_path):
    """One run in the file is unambiguous: nothing can be picked wrong."""
    target = tmp_path / "trace.csv"
    _stage(RUN_B).write(target)
    measured = accounting(read_trace(target))
    assert measured["run_id"] == "run-b"
    assert measured["rows_in"] == RUN_B["rows_in"]


# ── 1b. the identity is stated with its real terms ─────────────────────────
def test_accounting_states_and_closes_the_row_identity(two_run_file):
    """The four-term row identity and the guard-out identity, from the file."""
    for run in (RUN_A, RUN_B):
        measured = accounting(two_run_file, run_id=run["run_id"])
        for term in GUARD_IDENTITY_TERMS:
            assert term in measured, f"{term} missing from the identity"
        assert measured["rows_in"] == (
            measured["rows_retained"]
            + measured["gtin_missing_or_nan"]
            + measured["gs1_checksum_failed"]
            + measured["identity_review_quarantined"]
        )
        assert measured["rows_retained"] == (
            measured["canonical_records"] + measured["collapsed_same_gtin"]
        )
        # the quarantine term is load-bearing, not decoration
        assert measured["identity_review_quarantined"] > 0


def test_accounting_rejects_a_guard_row_that_omits_the_quarantine_term(tmp_path):
    """The three-term form NEVER closed on a real run; an unnamed population is
    not "zero of those", so the identity fails loudly instead of closing short."""
    target = tmp_path / "trace.csv"
    stage = TraceRun("data_prep", run_id="run-three-term")
    stage.add(
        "gtin_guard",
        "identity_claims_evaluated",
        in_count=100,
        out_count=40,
        detail={"gtin_missing_or_nan": 40, "gs1_checksum_failed": 20},
    )
    stage.add(
        "canonical",
        "records_built",
        in_count=40,
        out_count=30,
        detail={"collapsed_same_gtin": 10},
    )
    stage.write(target)

    with pytest.raises(
        ValueError, match="does not state \\['identity_review_quarantined'\\]"
    ):
        accounting(read_trace(target))


def test_accounting_rejects_a_row_identity_that_does_not_close(tmp_path):
    """Sabotage is caught: a guard row that loses a dropped population."""
    target = tmp_path / "trace.csv"
    _stage(RUN_A).write(target)
    frame = read_trace(target)
    guard = frame["step"].eq("gtin_guard.identity_claims_evaluated")
    frame.loc[guard, "in_count"] = str(RUN_A["rows_in"] + 7)  # 7 rows unaccounted
    with pytest.raises(ValueError, match="row identity does not close"):
        accounting(frame)

    # and the guard-out identity is enforced too
    frame = read_trace(target)
    collapsed = detail_json(
        frame.loc[frame["step"].eq("canonical.records_built"), "detail"].iloc[0]
    )["collapsed_same_gtin"]
    frame.loc[
        frame["step"].eq("canonical.records_built"), "out_count"
    ] = str(RUN_A["canonical"] + 1)
    with pytest.raises(ValueError, match="guard-out identity does not close"):
        accounting(frame)
    assert int(collapsed) == RUN_A["collapsed"]  # the fixture really closes


def test_accounting_rejects_two_census_rows_for_one_census_key(tmp_path):
    """Two group rows for one census key in one run are two populations claiming
    one number — the same last-row-wins ambiguity, refused instead of hidden."""
    target = tmp_path / "trace.csv"
    _stage(RUN_A).write(target)
    frame = read_trace(target)
    extra = frame[frame["step"].eq("gate.decision_hard_no")].copy()
    extra["reason"] = "hard_no_second_bucket"
    frame = pd.concat([frame, extra], ignore_index=True)
    assert_trace_frame(frame)
    with pytest.raises(ValueError, match="more than one group row"):
        accounting(frame)


# ── 2. the write boundary enforces the row contract ────────────────────────
def test_record_rejects_a_unit_change_stated_as_a_funnel():
    """``out_count > in_count`` states a negative drop, which the row contract
    (``dropped_count`` ge 0) forbids: the producer dies, the csv never sees it."""
    with pytest.raises(ValueError, match="greater_than_equal"):
        record("pairs", "payload.materialized", in_count=1, out_count=5)
    # a genuine unit change is stated with only its output
    row = record("pairs", "payload.materialized", out_count=5)
    assert row["in_count"] is None and row["dropped_count"] is None
    # and a legitimate funnel still derives its drop
    assert record("gate", "gated", in_count=5, out_count=3)["dropped_count"] == 2


def test_record_rejects_an_unknown_scope_and_a_blank_identity():
    with pytest.raises(ValueError):
        record("stage", "step", scope="batch")
    with pytest.raises(ValueError):
        record("  ", "step")
    with pytest.raises(ValueError):
        record("stage", "step", run_id="   ")


def test_write_boundary_refuses_a_bad_stage_and_writes_nothing(tmp_path):
    """A stage that emits an invalid row cannot land in the file at all."""
    target = tmp_path / "trace.csv"
    stage = TraceRun("pairs", run_id="run-bad")
    with pytest.raises(ValueError, match="greater_than_equal"):
        stage.add("payload", "materialized", in_count=1, out_count=5)
    assert not target.exists()  # nothing was written


def _poison_another_stage(target) -> pd.DataFrame:
    """Splice one contract-violating row of ``run_id`` into the trace file.

    The row is put under ANOTHER stage of that run, so a writer of stage
    ``data_prep`` neither owns nor replaces it: whatever it sees of that run is
    the poisoned state on disk.
    """
    frame = read_trace(target)
    bad = frame.iloc[[0]].copy()
    bad["stage"] = "pairs"
    bad["step"] = "payload.materialized"
    bad["dropped_count"] = "-1"
    frame = pd.concat([frame, bad], ignore_index=True)
    frame.to_csv(target, index=False)
    return frame


def test_write_boundary_refuses_an_invalid_row_of_the_run_being_written(tmp_path):
    """The run being committed must satisfy the contract its readers enforce."""
    target = tmp_path / "trace.csv"
    _stage(RUN_A).write(target)
    poisoned = _poison_another_stage(target)

    # the SAME run: fail-loud, nothing written, and the offender is named
    with pytest.raises(ValueError) as exc:
        _stage(RUN_A).write(target)
    message = str(exc.value)
    assert "run-a" in message and "payload.materialized" in message
    assert read_trace(target).equals(poisoned)  # the file was not rewritten


def test_an_invalid_historical_run_does_not_wedge_the_next_write(tmp_path):
    """History is reported, never fatal: the prune happens in the same commit
    that would fail, so failing on a stale run would wedge the file forever."""
    target = tmp_path / "trace.csv"
    _stage(RUN_A).write(target)
    _poison_another_stage(target)

    # ANOTHER run commits normally: the stale run is left untouched and named
    _stage(RUN_B).write(target)
    frame = read_trace(target)
    assert set(frame["run_id"]) == {"run-a", "run-b"}
    assert len(frame[frame["run_id"].eq("run-b")]) == len(_stage(RUN_B).rows())
    # the surviving historical rows are still the (invalid) ones on disk: the
    # write did not silently rewrite history either
    stale = frame[frame["run_id"].eq("run-a")]
    assert stale.loc[
        stale["step"].eq("payload.materialized"), "dropped_count"
    ].iloc[0] == "-1"
    # and the accounting of the run that IS clean still closes
    measured = accounting(frame, run_id="run-b")
    assert measured["rows_in"] == RUN_B["rows_in"]


def test_assert_trace_frame_still_rejects_the_shape_contracts(tmp_path):
    """The trace-owned shape checks keep their messages (they run first)."""
    target = tmp_path / "trace.csv"
    _stage(RUN_B).write(target)
    good = read_trace(target)
    assert_trace_frame(good)

    anonymous = good.copy()
    anonymous.loc[0, "run_id"] = ""
    with pytest.raises(ValueError, match="no.*run_id"):
        assert_trace_frame(anonymous)


# ── 3. volume guards ───────────────────────────────────────────────────────
def test_count_rows_is_bounded_and_states_the_remainder():
    population = [f"value-{index}" for index in range(1000)]
    listed = count_rows(population, limit=None)
    # limit=None means the POLICY cap, not "unbounded"
    assert listed[:CENSUS_TOP_N] == [
        f"value-{index}=1" for index in range(CENSUS_TOP_N)
    ]
    assert listed[-2] == f"others={1000 - CENSUS_TOP_N}"
    assert listed[-1] == f"others_buckets={1000 - CENSUS_TOP_N}"
    assert count_rows(population) == listed  # the default IS the policy cap

    # nothing is lost: the listed values plus the remainder are the population
    listed_total = sum(int(entry.rsplit("=", 1)[1]) for entry in listed[:-2])
    assert listed_total + int(listed[-2].rsplit("=", 1)[1]) == len(population)

    # a small distribution is unchanged, and an explicit limit still bounds
    assert count_rows(["a", "a", "b"]) == ["a=2", "b=1"]
    assert count_rows(population, limit=3) == [
        "value-0=1", "value-1=1", "value-2=1",
        f"others={1000 - 3}", f"others_buckets={1000 - 3}",
    ]


def test_count_rows_fits_the_cell_even_with_pathological_value_names():
    """A single wide value can never mint a megabyte cell."""
    population = ["x" * 50_000] * 3 + ["y" * 50_000] * 2
    listed = count_rows(population)
    assert len("".join(listed)) <= DETAIL_CELL_CHARS
    # both values folded into the remainder, and the population is still exact
    assert listed[-2] == f"others={len(population)}"
    assert listed[-1] == "others_buckets=2"


def test_a_wide_detail_cell_is_capped_and_keeps_its_named_numbers():
    """The realistic wide cell: a long per-pair list beside the anchor numbers.

    The named numbers are the audit's anchor, so the cap elides the LIST (with
    its elided item count stated) and leaves every key standing.
    """
    payload = {
        "gtin_missing_or_nan": 20580,
        "gs1_checksum_failed": 1869,
        "identity_review_quarantined": 456,
        "dimension_conflicts": [
            f"conflict-{index}:" + "w" * 60 for index in range(13_067)
        ],
    }
    row = record("data_prep", "gtin_guard.identity_claims_evaluated", detail=payload)
    assert len(row["detail"]) <= DETAIL_CELL_CHARS
    parsed = detail_json(row["detail"])
    assert parsed["gtin_missing_or_nan"] == 20580
    assert parsed["identity_review_quarantined"] == 456
    assert len(parsed["dimension_conflicts"]) <= CENSUS_TOP_N + 1
    assert "elided 13043 items" in parsed["dimension_conflicts"][-1]
    assert list(row) == list(TRACE_COLUMNS)  # still a valid contract row


def test_a_pathologically_wide_detail_payload_stays_bounded_and_parseable():
    """Thousands of wide values: the cell degrades to a stated, bounded form."""
    payload = {"anchor": 41545}
    payload.update({f"wide_{index}": "z" * 400 for index in range(2000)})
    row = record("data_prep", "gtin_guard.identity_claims_evaluated", detail=payload)
    assert len(row["detail"]) <= DETAIL_CELL_CHARS
    parsed = detail_json(row["detail"])
    assert parsed["detail_truncated"] is True
    assert parsed["detail_chars_before"] > DETAIL_CELL_CHARS
    assert parsed["head"]  # the payload's head is kept, not silently dropped


def test_a_census_row_from_add_entities_stays_bounded(tmp_path):
    """End to end: a census row over a wide free-text dimension is bounded."""
    target = tmp_path / "trace.csv"
    stage = TraceRun("pairs", run_id="run-wide")
    stage.add_entities(
        "pair_payload",
        [
            {"key": f"k{index}", "why": f"reason with values: {index}"}
            for index in range(500)
        ],
        key_of=lambda value: value["key"],
        reason_of=lambda value: value["why"],
        detail_of=lambda value: {"key": value["key"], "pad": "p" * 300},
        per_reason=ENTITY_SAMPLE_PER_REASON,
        total_cap=ENTITY_ROW_CAP,
    )
    stage.write(target)
    frame = read_trace(target)
    assert_trace_frame(frame)
    assert set(frame["run_id"]) == {"run-wide"}
    # the guard: no cell anywhere in the file is a megabyte
    assert int(frame["detail"].str.len().max()) <= DETAIL_CELL_CHARS


# ── 4. ONE trace per run across the PARALLEL lanes (G1) ────────────────────
# The suite spawns its trained lanes with a per-lane RESULTS subtree, so each
# lane would derive its OWN trace file and (with its per-lane run id) its rows
# could never join the data-prep rows of the same run. These tests drive the
# REAL launch path (model_tracks.parallel.run_parallel -> worker env ->
# core.tracing.trace_path/TraceRun) so the env contract is what is proven.
LANE_WORKER = '''
import json, os, pathlib
from model_tracks.parallel import wait_for_start
from core.tracing import TraceRun, trace_path

track = os.environ['ER_TRACK_NAME']
wait_for_start(pathlib.Path(os.environ['ER_TRACK_BARRIER']), track, timeout=30)
# the same STAGE for both lanes, exactly like model_tracks.worker does
run = TraceRun('worker')
run.add('training', 'completed', scope='entity', key=track, in_count=1, out_count=1,
        reason='lane reached the end of training')
target = run.write()
out = pathlib.Path(os.environ['EUROMONITOR_RESULTS_DIR'])
out.mkdir(parents=True, exist_ok=True)
(out / 'lane.json').write_text(json.dumps({
    'track': track, 'trace': str(target), 'run_id': run.run_id,
    'configured_trace': str(trace_path()), 'results': str(out)}))
'''


def test_parallel_lanes_append_to_one_run_trace(tmp_path, monkeypatch):
    """G1: env -> trace file -> run id, on the real parallel launch path."""
    import core.common as common
    from core.tracing import TRACE_PATH_ENV, TRACE_RUN_ENV, resolve_run_id
    from model_tracks.parallel import run_parallel
    from model_tracks.resume import TRAINING_TRACKS

    # the run root: what the launcher binds before spawning the lanes
    monkeypatch.setenv("EUROMONITOR_RESULTS_DIR", str(tmp_path))
    for variable in (TRACE_PATH_ENV, TRACE_RUN_ENV, "EUROMONITOR_RUN_ID"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}

    commands = {
        track: [sys.executable, "-c", LANE_WORKER] for track in TRAINING_TRACKS
    }
    run_parallel(commands, tmp_path, env, barrier_timeout=60)

    lanes = [
        json.loads((tmp_path / track / "lane.json").read_text())
        for track in TRAINING_TRACKS
    ]
    expected_trace = str(common.artifact("training_trace"))
    expected_run = resolve_run_id()

    # every lane appended to the RUN's trace file, not to its own subtree
    assert {lane["configured_trace"] for lane in lanes} == {expected_trace}
    assert {lane["trace"] for lane in lanes} == {expected_trace}
    for track in TRAINING_TRACKS:
        assert not (tmp_path / track / "logs" / "training_trace.csv").exists()
    # ONE run id, and it is the run's (never the per-lane "<root>-<track>")
    assert {lane["run_id"] for lane in lanes} == {expected_run}
    assert all(lane["run_id"] != f"{tmp_path.name}-{lane['track']}" for lane in lanes)

    frame = read_trace(Path(expected_trace))
    assert_trace_frame(frame)
    assert set(frame["run_id"]) == {expected_run}
    # every lane's rows are IN the file: the lane-qualified producer keeps two
    # writers of the same stage from replacing one another
    for track in TRAINING_TRACKS:
        rows = frame[frame["producer"].eq(f"core.tracing:{track}")]
        assert list(rows["step"]) == ["run_identity", "training.completed"]
        assert rows.loc[
            rows["step"].eq("training.completed"), "key"
        ].tolist() == [track]
    assert len(frame) == 2 * len(TRAINING_TRACKS)


def test_re_running_one_lane_replaces_only_its_own_rows(tmp_path, monkeypatch):
    """A lane re-run is idempotent for ITS rows and never eats its sibling's."""
    import core.common as common
    from core.tracing import TRACE_PATH_ENV, TRACE_RUN_ENV
    from model_tracks.parallel import run_parallel
    from model_tracks.resume import TRAINING_TRACKS

    monkeypatch.setenv("EUROMONITOR_RESULTS_DIR", str(tmp_path))
    for variable in (TRACE_PATH_ENV, TRACE_RUN_ENV, "EUROMONITOR_RUN_ID"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    first = next(iter(TRAINING_TRACKS))

    run_parallel({t: [sys.executable, "-c", LANE_WORKER] for t in TRAINING_TRACKS},
                 tmp_path, env, barrier_timeout=60)
    # a RE-RUN of one lane: the resume path gives it a fresh barrier
    run_parallel({first: [sys.executable, "-c", LANE_WORKER]},
                 tmp_path, env, barrier_timeout=60, resume=True)

    frame = read_trace(common.artifact("training_trace"))
    assert_trace_frame(frame)
    # the re-run REPLACED its own rows (no duplicate) and the sibling's survived
    assert len(frame) == 2 * len(TRAINING_TRACKS)
    for track in TRAINING_TRACKS:
        rows = frame[frame["producer"].eq(f"core.tracing:{track}")]
        assert list(rows["step"]) == ["run_identity", "training.completed"]
        assert rows.loc[
            rows["step"].eq("training.completed"), "key"
        ].tolist() == [track]


# ── 5. ONE run id for the whole PREPARATION run (G2) ──────────────────────
def test_preparation_pins_the_run_id_per_stage(monkeypatch):
    """G2: no stage tags with the PREVIOUS run's fingerprint, and the run's own
    id is used from the stage that defines it onward."""
    from core.tracing import TRACE_PENDING_RUN, resolve_run_id
    from training import prepare_all

    # stages that run BEFORE the identity artifacts exist
    for stage in ("dedupe", "cross_country_pairs", "number_reference",
                  "verify_reference"):
        assert prepare_all._trace_run_pin(stage) == TRACE_PENDING_RUN
    # the identity-defining stage resolves at WRITE time: it writes the very
    # artifacts the fingerprint is taken from, so an early resolution would tag
    # its rows with the identity they supersede
    assert prepare_all._trace_run_pin("canonical_and_gates") is None
    # every later stage shares the run's own id (the training side's id)
    for stage in ("gate_census", "labeled_pairs", "validation", "graph_inputs",
                  "full_bundle", "suite_inputs", "verify_handoff"):
        assert prepare_all._trace_run_pin(stage) == resolve_run_id()


def test_preparation_children_get_the_run_trace_file(tmp_path, monkeypatch):
    """G2: every prepared child is told the RUN's trace, never a re-derived one."""
    import core.common as common
    from core.tracing import TRACE_PATH_ENV
    from training.prepare_all import _prepare_environment

    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setenv("EUROMONITOR_RESULTS_DIR", str(tmp_path))
    env, _ = _prepare_environment(tmp_path, tmp_path / "run", None)

    run_trace = common.artifact("training_trace")
    assert env[TRACE_PATH_ENV] == str(run_trace)
    assert Path(env[TRACE_PATH_ENV]).parent.parent == tmp_path


def test_pending_rows_are_adopted_onto_the_runs_own_id(tmp_path):
    """G2: pre-identity rows join the run once its identity exists — ONE run."""
    from core.tracing import TRACE_PENDING_RUN, adopt_run

    target = tmp_path / "trace.csv"
    stage = TraceRun("number_reference", run_id=TRACE_PENDING_RUN)
    stage.add("reference", "verdicts_recorded", in_count=9, out_count=3, reason="x")
    stage.write(target)
    _stage(RUN_B).write(target)  # another run: must not be touched

    adopted = adopt_run(target, from_run_id=TRACE_PENDING_RUN, to_run_id="run-abc123")
    assert adopted == 2  # the run_identity row + the step row
    frame = read_trace(target)
    assert set(frame["run_id"]) == {"run-abc123", "run-b"}
    identity = frame[
        frame["step"].eq("run_identity") & frame["run_id"].eq("run-abc123")
    ].iloc[0]
    detail = detail_json(identity["detail"])
    assert detail["run_id"] == "run-abc123"
    assert TRACE_PENDING_RUN in detail["resolution"]  # states where it came from
    assert_trace_frame(frame)
    # idempotent: there is nothing pending left to adopt
    assert adopt_run(target, from_run_id=TRACE_PENDING_RUN, to_run_id="run-abc123") == 0


# ── 6. the orchestration stage -> trace stage join (G4) ────────────────────
def test_orchestration_stage_map_is_the_producers_own_vocabulary():
    """G4: a reader joins prepare_all's manifest to the trace without a hand map."""
    from core import tracing
    from core.tracing import trace_stages_for
    from training import (
        build_final_validation,
        build_reference,
        build_second04_pairs,
        labeled_pairs,
        prepare_all,
    )

    # every orchestration stage is declared (the lane stages are the two the
    # orchestrator inserts when a negative-supply tag is requested); the
    # registry reads config/paths.yaml, so it cannot drift from the join
    declared = set(prepare_all.STAGES) | {"negative_supply", "discriminator"}
    assert declared == set(tracing.orchestration_trace_stages())
    # the mapped names are the PRODUCERS' stage constants, so the map cannot
    # drift from the rows the producers actually write
    assert trace_stages_for("cross_country_pairs") == (build_second04_pairs.STAGE,)
    assert trace_stages_for("validation") == (build_final_validation.STAGE,)
    assert trace_stages_for("labeled_pairs") == (labeled_pairs.STAGE,)
    assert trace_stages_for("number_reference") == (build_reference.STAGE_WRITE,)
    assert trace_stages_for("verify_reference") == (build_reference.STAGE_VERIFY,)
    # the two reference modes are DISTINCT stages: sharing one name made one
    # overwrite the other's rows
    assert build_reference.STAGE_WRITE != build_reference.STAGE_VERIFY
    assert trace_stages_for("canonical_and_gates") == ("data_prep",)
    # an unmapped stage is a loud error, not a silent empty join
    with pytest.raises(ValueError, match="unknown orchestration stage"):
        trace_stages_for("not-a-stage")
