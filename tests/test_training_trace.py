"""tests/test_training_trace.py — the TRAINING process in the ONE trace.

The owner directive ("full traceability: we get to follow each step in the
training and data-processing steps") makes the training lane a producer of the
consolidated ``core.tracing`` trace, at three grains:

  stage/step  run config -> dataset load -> pairs -> split -> folds -> epochs
              -> checkpoint select -> early stop -> final metrics -> handoff;
  batch/step  one row per optimizer step, published through the trace's own
              entity sampling caps so a 38k-step run stays readable;
  entity      the EXACT sample behind a collapse breach, with the responsible
              attribute/token (or an explicit "unknown", never a guess).

These tests drive the real production functions on real inputs (the real
collapse diagnostic, the real batch collector, the real tracing writer) and
read the result back with ``core.tracing.read_trace``, validating every row
with ``core.schemas.TraceRow``. Nothing touches ``results/``: every trace and
every side file goes to ``tmp_path``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core.tracing as tracing
from core.schemas import TraceRow
from core.tracing import TraceRun, assert_trace_frame, read_trace
from training import training as training_mod
from training.training import (
    _BatchStepTrace,
    _emit_batch_rows,
    _emit_collapse_rows,
    _trace_fold_outcome,
    collapse_pair_records,
)
from training.uniformity import collapse_diagnostics

OPERATING_THRESHOLD = 0.90


def _guardrail(**overrides) -> dict:
    """The guardrail shape the diagnostic reads, at readable test values."""
    guardrail = {
        "enabled": True,
        "unrelated_pairs": 1,
        "seed": 1337,
        "max_token_frequency": 0.5,
        "operating_threshold": OPERATING_THRESHOLD,
        "crossing_rate_ceiling": 0.05,
        "median_penalty_start": 0.5,
        "p90_penalty_start": 0.6,
        "cosine_std_floor": 0.05,
        "reject_median": 0.75,
    }
    guardrail.update(overrides)
    return guardrail


def _collapse_catalog() -> tuple[pd.DataFrame, list[str]]:
    """Two rows that can only be paired together and share one token ("500").

    Different brands, different categories, and every non-shared token has
    document frequency 1 (kept for blocking), so the ONLY selectable pair is
    (0, 1) and the only token they share is the above-limit "500".
    """
    frame = pd.DataFrame(
        [
            {
                "sku_id": "sku-a",
                "brand": "brand-a",
                "breadcrumbs_eng": "cat-a",
                "gtin": "1",
                "pack_size": "500 g",
            },
            {
                "sku_id": "sku-b",
                "brand": "brand-b",
                "breadcrumbs_eng": "cat-b",
                "gtin": "2",
                "pack_size": "750 g",
            },
        ]
    )
    payload = ["alpha bravo 500", "charlie delta 500"]
    return frame, payload


class _StubEncoder:
    """A minimal encoder: fixed vectors per payload, so cosine is controlled."""

    def __init__(self, by_text: dict[str, list[float]]):
        self._by_text = by_text
        self.training = True

    def encode(self, texts, **kwargs):
        return np.asarray([self._by_text[text] for text in texts], dtype=float)

    def train(self, mode: bool = True):  # the diagnostic restores train mode
        self.training = bool(mode)
        return self


def _frame_from(trace: TraceRun, path: Path) -> pd.DataFrame:
    trace.write(path)
    frame = read_trace(path)
    assert_trace_frame(frame)
    for row in frame.to_dict("records"):
        TraceRow.model_validate(row)
    return frame


# ── 1. the collapse grain: the exact sample + the responsible attribute ────
def test_collapse_breach_emits_an_entity_row_naming_the_sample_and_cause(
    tmp_path: Path,
):
    """A real breaching pair becomes an entity row whose reason IS the cause."""
    df, payload = _collapse_catalog()
    guardrail = _guardrail()
    # cosine = 0.99 (well above the 0.90 threshold) for the one pair that the
    # real selector can pick.
    encoder = _StubEncoder(
        {
            payload[0]: [1.0, 0.0],
            payload[1]: [0.99, np.sqrt(1 - 0.99**2)],
        }
    )
    pair_csv = tmp_path / "loss_backprop_fold0_collapse_pairs.csv"
    diagnostics = collapse_diagnostics(
        model=encoder,
        df=df,
        payload=payload,
        config={"collapse_guardrail": guardrail},
        batch_size=2,
        trace_path=pair_csv,
        evaluation_step=7,
    )
    assert diagnostics["collapse_guardrail_enabled"] == 1
    assert pair_csv.is_file(), "the diagnostic must have scored the unrelated pair"

    records = collapse_pair_records(
        df, payload, pair_trace_path=pair_csv, guardrail=guardrail, fold=0
    )
    assert len(records) == 1, records
    record = records[0]
    assert record["key"] == "sku-a|sku-b"
    assert record["cosine"] >= OPERATING_THRESHOLD
    assert record["operating_threshold"] == OPERATING_THRESHOLD
    # the responsible token, with its real document frequency and the limit
    assert record["reason"] == "shared_high_frequency_token:500"
    (token,) = record["shared_tokens"]
    assert token["token"] == "500"
    assert token["document_frequency"] == 2
    assert token["frequency_limit"] == pytest.approx(1.0)
    assert token["above_limit"] is True
    # the pair's other surface differs, so nothing else can be blamed
    assert "pack_size" not in record.get("shared_attributes", {})
    assert record["row_index1"] == 0 and record["row_index2"] == 1
    assert record["n_observations"] == 1
    assert record["observations"] == [{"evaluation_step": 7, "cosine": record["cosine"]}]

    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-collapse-probe")
    _emit_collapse_rows(
        trace, records, guardrail=guardrail, source="test collapse diagnostic",
        evidence_folds=1,
    )
    frame = _frame_from(trace, tmp_path / "trace.csv")

    assert set(frame["stage"]) == {training_mod.TRAINING_STAGE}
    assert list(frame["step"]) == [
        "run_identity",
        "collapse.pairs.reason_census",
        "collapse.pairs",
        "collapse.pairs.sample_budget",
        "collapse.breach_summary",
    ]
    entity = frame[frame["scope"] == "entity"].iloc[0]
    assert entity["key"] == "sku-a|sku-b"
    assert entity["reason"] == "shared_high_frequency_token:500"
    # the entity row carries the literal readback; the exact population lives on
    # the bucket census row below (the trace's entity contract)
    assert entity["in_count"] == "" and entity["dropped_count"] == ""
    detail = tracing.detail_json(entity["detail"])
    assert detail["cosine"] >= OPERATING_THRESHOLD
    assert detail["operating_threshold"] == OPERATING_THRESHOLD
    assert detail["crossed"] is True
    assert detail["attribution_reason"] == "shared_high_frequency_token:500"
    assert detail["payload1"] == payload[0] and detail["payload2"] == payload[1]
    # the census is EXACT and the budget that bounded the sample is declared
    census = frame[frame["step"] == "collapse.pairs.reason_census"].iloc[0]
    assert census["scope"] == "group"
    assert census["reason"] == "shared_high_frequency_token:500"
    assert int(census["in_count"]) == 1
    budget = tracing.detail_json(
        frame[frame["step"] == "collapse.pairs.sample_budget"].iloc[0]["detail"]
    )
    assert budget["total_cap"] == tracing.ENTITY_ROW_CAP
    assert budget["per_reason"] == tracing.ENTITY_SAMPLE_PER_REASON


def test_collapse_without_traceable_evidence_is_marked_unknown_not_invented(
    tmp_path: Path,
):
    """Two unrelated rows that share no token/attribute: the cause is UNKNOWN."""
    df, payload = _collapse_catalog()
    # No shared token at all (so the pair would normally be blocked; force the
    # evidence file directly, which is the shape the diagnostic writes).
    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "evaluation_step": "3",
                "pair_number": "0",
                "cosine": "0.97",
                "a_row_index": "0",
                "b_row_index": "1",
                "a_payload": "alpha bravo",
                "b_payload": "charlie delta",
                "a_sku_id": "sku-a",
                "b_sku_id": "sku-b",
                "a_brand": "brand-a",
                "b_brand": "brand-b",
                "a_gtin": "1",
                "b_gtin": "2",
            }
        ]
    ).to_csv(pair_csv, index=False)

    records = collapse_pair_records(
        df, payload, pair_trace_path=pair_csv, guardrail=_guardrail(), fold=1
    )
    assert len(records) == 1
    record = records[0]
    assert record["reason"] == "attribution_unknown"
    assert record["shared_tokens"] == []
    assert "attribution_note" in record and record["attribution_note"]

    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-collapse-unknown")
    _emit_collapse_rows(
        trace, records, guardrail=_guardrail(), source="test", evidence_folds=1
    )
    frame = _frame_from(trace, tmp_path / "trace.csv")
    entity = frame[frame["scope"] == "entity"].iloc[0]
    assert entity["reason"] == "attribution_unknown"
    assert tracing.detail_json(entity["detail"])["attribution_reason"] == (
        "attribution_unknown"
    )


def test_a_non_breaching_pair_is_not_recorded(tmp_path: Path):
    """The entity grain is a breach census, not a dump of every scored pair."""
    df, payload = _collapse_catalog()
    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "evaluation_step": "1",
                "pair_number": "0",
                "cosine": "0.10",
                "a_row_index": "0",
                "b_row_index": "1",
                "a_payload": payload[0],
                "b_payload": payload[1],
                "a_sku_id": "sku-a",
                "b_sku_id": "sku-b",
            }
        ]
    ).to_csv(pair_csv, index=False)

    assert collapse_pair_records(
        df, payload, pair_trace_path=pair_csv, guardrail=_guardrail(), fold=0
    ) == []


# ── 2. the batch/step grain, bounded by the trace's own caps ──────────────
def _control():
    return SimpleNamespace(should_log=False, should_evaluate=False)


def test_batch_rows_are_censused_exactly_and_sampled_under_the_declared_caps(
    tmp_path: Path,
):
    """Every optimizer step is captured; the file keeps the census + a sample."""
    collector = _BatchStepTrace(fold_i=0)

    class _Loss:
        def compute_loss_from_embeddings(self, value):
            return value

    loss = _Loss()
    assert collector.attach(loss) is True
    for step_index, loss_value in enumerate([1.0, 0.8, 0.6, 0.4], start=1):
        assert loss.compute_loss_from_embeddings(loss_value) == loss_value  # unchanged
        collector.on_step_end(
            SimpleNamespace(per_device_train_batch_size=16),
            SimpleNamespace(
                epoch=(1.0 if step_index <= 2 else 2.0),
                global_step=step_index,
                max_steps=4,
            ),
            _control(),
        )
    assert len(collector.rows) == 4
    assert [row["loss"] for row in collector.rows] == [1.0, 0.8, 0.6, 0.4]
    assert {row["epoch_index"] for row in collector.rows} == {0, 1}
    assert all(row["batch_size"] == 16 for row in collector.rows)
    assert all(
        row["loss_source"].startswith("loss.compute_loss_from_embeddings")
        for row in collector.rows
    )

    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-batch-probe")
    summary = _emit_batch_rows(
        trace, collector.rows, source="test batch collector", total_cap=None
    )
    trace.add(
        "batch",
        "capture",
        in_count=len(collector.rows),
        detail={
            "entity_cap": None,
            "sampling_contract": "core.tracing ENTITY_ROW_CAP / ENTITY_SAMPLE_PER_REASON",
        },
    )
    frame = _frame_from(trace, tmp_path / "trace.csv")

    census = frame[frame["step"] == "batch.record.reason_census"]
    assert dict(zip(census["reason"], census["in_count"].astype(int))) == {
        "fold0/epoch0": 2,
        "fold0/epoch1": 2,
    }
    entities = frame[frame["scope"] == "entity"]
    assert len(entities) == 4  # the whole (tiny) population is under the cap
    assert summary["sampled"] == 4 and summary["omitted"] == 0
    budget = tracing.detail_json(
        frame[frame["step"] == "batch.record.sample_budget"].iloc[0]["detail"]
    )
    assert budget["total_cap"] == tracing.ENTITY_ROW_CAP
    assert budget["per_reason"] == tracing.ENTITY_SAMPLE_PER_REASON
    keys = set(entities["key"])
    assert keys == {f"fold0/step{i}" for i in range(1, 5)}
    detail = tracing.detail_json(entities.iloc[0]["detail"])
    assert detail["grad_norm"] is None  # never fabricated at this hook
    assert "grad_norm" in detail["grad_norm_note"]


def test_batch_census_is_kept_but_the_sample_is_withheld_for_a_sweep_lane(
    tmp_path: Path,
):
    """total_cap=0 keeps the exact census and drops the entity sample."""
    rows = [
        {
            "fold": 0,
            "epoch": 1.0,
            "epoch_index": 0,
            "global_step": step,
            "max_steps": 3,
            "batch_size": 8,
            "micro_batches": 1,
            "loss": 0.5,
            "loss_source": "unavailable: loss exposes no embeddings hook",
            "learning_rate": 1e-3,
        }
        for step in (1, 2, 3)
    ]
    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-sweep-probe")
    _emit_batch_rows(trace, rows, source="test sweep lane", total_cap=0)
    frame = _frame_from(trace, tmp_path / "trace.csv")
    census = frame[frame["step"] == "batch.record.reason_census"]
    assert int(census.iloc[0]["in_count"]) == 3  # exact population survives
    assert frame[frame["scope"] == "entity"].empty  # no sample for a sweep lane
    budget = tracing.detail_json(
        frame[frame["step"] == "batch.record.sample_budget"].iloc[0]["detail"]
    )
    assert budget["sampled"] == 0 and budget["total_cap"] == 0


# ── 3. the stage/step grain, and it joins the ONE trace ───────────────────
def test_fold_outcome_emits_the_step_checkpoint_and_early_stop_rows(tmp_path: Path):
    """epoch/step metrics, checkpoint selection and early stop, all real."""
    hist = [
        {"epoch": 1.0, "step": 4, "loss": 0.9, "grad_norm": 1.2, "learning_rate": 1e-3},
        {
            "epoch": 1.0,
            "step": 4,
            "eval_loss": 0.8,
            "eval_dev_cosine_ap": 0.5,
            "eval_dev_cosine_auc": 0.7,
        },
        {"epoch": 2.0, "step": 8, "loss": 0.5, "grad_norm": 0.9, "learning_rate": 5e-4},
        {
            "epoch": 2.0,
            "step": 8,
            "eval_loss": 0.6,
            "eval_dev_cosine_ap": 0.62,
            "eval_dev_cosine_auc": 0.75,
        },
        {"train_runtime": 12.5, "epoch": 2.0},  # no metric: must not become a row
    ]
    state = SimpleNamespace(
        global_step=8,
        max_steps=12,
        epoch=2.0,
        best_model_checkpoint="/checkpoints/checkpoint-8",
        best_metric=0.62,
    )
    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-fold-probe")
    summary = _trace_fold_outcome(
        trace,
        fold_i=0,
        hist=hist,
        trainer_state=state,
        cfg={"epochs": 3, "patience": 2, "es_threshold": 0.01},
        planned_steps=12,
        best_metric_key="eval_dev_cosine_ap",
        guardrail=_guardrail(),
        collapse_records=[],
    )
    frame = _frame_from(trace, tmp_path / "trace.csv")

    steps = frame[frame["step"] == "step.metrics"]
    # one row per real log event (the trainer logs loss and eval separately at
    # the same step); the runtime-only event is NOT a metric row
    assert list(steps["key"]) == [
        "fold0/step4",
        "fold0/step4",
        "fold0/step8",
        "fold0/step8",
    ]
    train_event = tracing.detail_json(steps.iloc[0]["detail"])
    assert train_event["train_loss"] == 0.9 and train_event["grad_norm"] == 1.2
    eval_first = tracing.detail_json(steps.iloc[1]["detail"])
    assert eval_first["eval_dev_cosine_ap"] == 0.5
    assert eval_first["eval_dev_cosine_auc"] == 0.7
    eval_second = tracing.detail_json(steps.iloc[3]["detail"])
    assert eval_second["eval_dev_cosine_ap"] == 0.62

    # early stop: planned 3 epochs, ran 2 -> one epoch saved, stated as such
    early = frame[frame["step"] == "fold.early_stop"].iloc[0]
    assert int(early["in_count"]) == 3 and int(early["out_count"]) == 2
    early_detail = tracing.detail_json(early["detail"])
    assert early_detail["stopped_early"] == 1
    assert early_detail["epochs_saved"] == 1.0
    assert early_detail["metric_for_best_model"] == "eval_dev_cosine_ap"

    select = frame[frame["step"] == "checkpoint.select"].iloc[0]
    assert int(select["out_count"]) == 1
    select_detail = tracing.detail_json(select["detail"])
    assert select_detail["best_model_checkpoint"] == "/checkpoints/checkpoint-8"
    assert select_detail["best_metric"] == 0.62

    # planned vs executed optimizer steps closes arithmetically
    complete = frame[frame["step"] == "fold.complete"].iloc[0]
    assert int(complete["in_count"]) == 12
    assert int(complete["out_count"]) == 8
    assert int(complete["dropped_count"]) == 4
    assert summary["executed_steps"] == 8 and summary["epochs_run"] == 2.0


def test_training_rows_join_the_existing_trace_and_replace_in_place(
    tmp_path: Path, monkeypatch
):
    """One file, one run: training rows land next to the data-prep rows."""
    target = tmp_path / "training_trace.csv"
    run_id = "run-joined"
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", run_id)
    monkeypatch.delenv("EUROMONITOR_RUN_ID", raising=False)
    prep = TraceRun("data_prep", run_id=run_id)
    prep.add("gtin_guard", "identity_claims_evaluated", in_count=10, out_count=8)
    prep.write(target)

    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)
    monkeypatch.setattr(tracing, "trace_path", lambda: target)
    trace = training_mod.training_trace()
    trace.add("folds", "dispatch", in_count=10, out_count=1)
    assert training_mod.flush_training_trace() == target

    frame = read_trace(target)
    assert_trace_frame(frame)
    assert set(frame["run_id"]) == {run_id}
    assert list(dict.fromkeys(frame["stage"])) == ["data_prep", "training"]
    assert set(frame["stage"]) == {"data_prep", "training"}

    # a re-run of the training stage replaces its rows in place, never duplicates
    again = training_mod.training_trace()
    again.add("folds", "dispatch", in_count=12, out_count=1)
    training_mod.flush_training_trace()
    replayed = read_trace(target)
    assert len(replayed) == len(frame)
    assert list(replayed["step"]) == list(frame["step"])
    assert int(replayed[replayed["stage"] == "training"].iloc[-1]["in_count"]) == 12
    assert_trace_frame(replayed)


# ── 4. the graph lane's data-processing steps land in the SAME trace ──────
def test_graph_lane_data_steps_are_traced_into_the_same_stage(tmp_path: Path):
    """The graph worker's census step joins the training trace.

    ``graph_tracks.data`` exposes these as OPT-IN (``trace=``) so the worker
    that already owns a writer can pass it: this pins that the pass-through
    lands the data-processing grain in the SAME stage as the training rows, and
    that its funnel closes (in == out, one split per listing).

    ``fit_vocabulary`` is deliberately NOT passed the trace: its row counts
    train LISTINGS against vocabulary ENTRIES, so the derived dropped_count is
    a unit-change artefact (it went negative on the live run). Reported for the
    owner of ``graph_tracks/data.py``; the vocabulary numbers stay in the
    ``inputs.load`` detail until that row's counts are same-unit.
    """
    from graph_tracks.data import census, fit_vocabulary

    def _record(sku_id, split, brand_values):
        return {
            "sku_id": sku_id,
            "split": split,
            "attribute": {"brand": list(brand_values)},
            "numeric": {"volume_ml": [330.0], "pack": [1.0]},
        }

    records = [
        _record("s1", "train", ["acme"]),
        _record("s2", "train", ["acme"]),
        _record("s3", "dev", ["other"]),
    ]
    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-graph-data-probe")
    vocabulary = fit_vocabulary(records)
    census(records, vocabulary, trace=trace)
    frame = _frame_from(trace, tmp_path / "trace.csv")

    assert set(frame["stage"]) == {training_mod.TRAINING_STAGE}
    assert "vocabulary.fitted" not in set(frame["step"])
    census_row = frame[frame["step"] == "census.representation"].iloc[0]
    assert int(census_row["in_count"]) == 3
    assert int(census_row["out_count"]) == 3  # every listing has one split
    assert int(census_row["dropped_count"]) == 0


# ── 5. the bundle lane owns its own stage (never clobbers "training") ──────
def test_bundle_lane_owns_a_separate_stage_and_survives_the_training_commit(
    tmp_path: Path, monkeypatch
):
    """A stage's rows are replaced in place, so the bundle lane needs its own.

    ``prepare_all`` runs the bundle write as its ``full_bundle`` stage before
    the training stage: if both committed under "training", the later commit
    would erase the earlier one's rows. Pinned: the bundle rows survive a
    training commit, and the stage name IS the orchestrator's own vocabulary.
    """
    from training.prepare_all import STAGES
    from training.train import BUNDLE_STAGE, _trace_bundle_lane

    assert BUNDLE_STAGE in STAGES, (BUNDLE_STAGE, STAGES)
    target = tmp_path / "trace.csv"
    run_id = "run-two-stages"
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", run_id)
    monkeypatch.delenv("EUROMONITOR_RUN_ID", raising=False)
    monkeypatch.setattr(tracing, "trace_path", lambda: target)

    # the real writer may not be relabelled mid-run (a stage's rows are one
    # commit), and the bundle lane pins the process to its own stage
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)
    training_mod.training_trace(BUNDLE_STAGE)
    with pytest.raises(ValueError, match="refusing to relabel"):
        training_mod.training_trace("training")
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)

    # the REAL bundle-lane writer (its own stage, its own commit)
    _trace_bundle_lane(
        SimpleNamespace(n_df=100, n_payload=120, n_pos=40, n_neg=60),
        SimpleNamespace(
            prepare_bundle="bundle.pkl.gz",
            run_tag="two-stages",
            payload="full",
            loss="contrastive",
            train_frac=1.0,
            sample=False,
        ),
    )

    # the training lane is a separate process in production; here it starts
    # with a fresh writer for its own stage
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)
    training = training_mod.training_trace()
    training.add("folds", "dispatch", in_count=3, out_count=1)
    training_mod.flush_training_trace()

    frame = read_trace(target)
    assert_trace_frame(frame)
    assert set(frame["stage"]) == {BUNDLE_STAGE, "training"}
    bundle_rows = frame[frame["stage"] == BUNDLE_STAGE]
    assert list(bundle_rows["step"]) == ["run_identity", "bundle.materialize"]
    assert bundle_rows.iloc[-1]["dropped_count"] == ""  # not a funnel
    assert "folds.dispatch" in set(frame[frame["stage"] == "training"]["step"])


def test_no_emitted_row_can_derive_a_negative_dropped_count(tmp_path: Path):
    """Unit-change 'funnels' must not reach the file (TraceRow requires ge=0).

    ``_frame_from`` validates every row with ``core.schemas.TraceRow``; this
    states the rule explicitly for the two rows that regressed on the live run
    (a grain statement and a resume-safe checkpoint selection) plus the text
    lane's step/epoch rows.
    """
    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-counts-probe")
    # a resume-shaped fold: 0 candidate epochs in THIS invocation, yet a
    # checkpoint is selected (it predates the invocation) -> in>=out was false
    state = SimpleNamespace(
        global_step=0,
        max_steps=0,
        epoch=0.0,
        best_model_checkpoint="/checkpoints/checkpoint-4",
        best_metric=0.9,
    )
    _trace_fold_outcome(
        trace,
        fold_i=0,
        hist=[],
        trainer_state=state,
        cfg={"epochs": 10, "patience": 2, "es_threshold": 0.01},
        planned_steps=0,
        best_metric_key="eval_dev_cosine_ap",
        guardrail=_guardrail(),
        collapse_records=[],
    )
    _emit_batch_rows(
        trace,
        [
            {
                "fold": 0,
                "epoch": 1.0,
                "epoch_index": 0,
                "global_step": 1,
                "max_steps": 10,
                "batch_size": 4,
                "micro_batches": 1,
                "loss": 0.5,
                "loss_source": "test",
                "learning_rate": 1e-3,
            }
        ],
        source="test",
        total_cap=None,
    )
    frame = _frame_from(trace, tmp_path / "trace.csv")  # validates TraceRow
    counted = frame[frame["dropped_count"] != ""]
    assert not counted.empty
    assert (counted["dropped_count"].astype(int) >= 0).all()
    select = frame[frame["step"] == "checkpoint.select"].iloc[0]
    assert int(select["in_count"]) == 1 and int(select["out_count"]) == 1
    assert int(select["dropped_count"]) == 0


# ── 6. the PRODUCTION entry flushes what the trainer recorded ──────────────
def test_train_prepared_entry_flushes_the_training_stage(tmp_path, monkeypatch):
    """``python -m training.train_prepared`` must commit the fold rows.

    This entry (the suite's production training path) does not go through
    train.py, so its flush is the only thing that publishes what
    train_one_config recorded. Driven through the REAL ``main()`` (W&B, arg
    parsing and the trainer body are the only things stubbed), on both the
    success and the failure path.
    """
    from training import train_prepared as tp

    target = tmp_path / "trace.csv"
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-prepared-probe")
    monkeypatch.delenv("EUROMONITOR_RUN_ID", raising=False)
    monkeypatch.setattr(tracing, "trace_path", lambda: target)
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)

    class _NullWandb:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _recording_main(args, wandb_ctx):
        training_mod.training_trace().add("folds", "dispatch", in_count=3, out_count=1)

    monkeypatch.setattr(tp, "WandbCtx", _NullWandb)
    monkeypatch.setattr(
        tp, "_parse_args", lambda: SimpleNamespace(run_tag="prepared-probe")
    )
    monkeypatch.setattr(tp, "_main", _recording_main)
    tp.main()

    frame = read_trace(target)
    assert_trace_frame(frame)
    rows = frame[frame["stage"] == training_mod.TRAINING_STAGE]
    assert list(rows["step"]) == ["run_identity", "folds.dispatch"]

    # the failure path is flushed too: a partial run stays describable
    target.unlink()
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)

    def _failing_main(args, wandb_ctx):
        training_mod.training_trace().add("folds", "dispatch", in_count=1, out_count=1)
        raise RuntimeError("boom")

    monkeypatch.setattr(tp, "_main", _failing_main)
    with pytest.raises(RuntimeError, match="boom"):
        tp.main()
    assert list(read_trace(target)["step"]) == ["run_identity", "folds.dispatch"]


def test_guardrail_rejection_is_recorded_with_the_breaching_aggregate(monkeypatch):
    """The HPO prune decision is traced where the decision is made."""
    trace = TraceRun(training_mod.TRAINING_STAGE, run_id="run-hpo-probe")
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", trace)
    # optuna is an optional dependency of the sweep lane; the reject path only
    # needs its exception type, so a minimal stub keeps this deterministic in
    # an environment where the sweep dependency is absent.
    fake_optuna = types.ModuleType("optuna")

    class TrialPruned(Exception):
        pass

    fake_optuna.TrialPruned = TrialPruned
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    guardrail = {"reject_median": 0.75, "crossing_rate_ceiling": 0.05}
    proxy_rows = [
        {"fold": 0, "collapse_median_cosine": 0.81, "collapse_crossing_rate": 0.01},
        {"fold": 1, "collapse_median_cosine": 0.6, "collapse_crossing_rate": 0.02},
    ]
    with pytest.raises(TrialPruned):
        training_mod.OptunaObjectiveOwner._reject_on_collapse_guardrail(
            proxy_rows, guardrail
        )

    frame = trace.rows()
    assert_trace_frame(frame)
    TraceRow.model_validate(frame.iloc[-1].to_dict())
    row = frame[frame["step"] == "hpo.guardrail_reject"].iloc[0]
    assert int(row["in_count"]) == 2 and int(row["out_count"]) == 0
    detail = tracing.detail_json(row["detail"])
    assert detail["median_max"] == 0.81
    assert detail["median_breach"] == 1
    assert detail["crossing_rate_breach"] == 0
