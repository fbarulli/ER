"""tests/test_trace_writer_consolidation.py — the ONE home for the trace plumbing.

Two consolidations (independent audit findings), both pinned at their PUBLIC
surface — a producer's own entry point and the orchestration registry, never a
class or validator internals:

1. The lazy ``trace()``/``flush_trace()`` shim lives ONCE in :mod:`core.tracing`
   (:func:`~core.tracing.stage_trace` / :func:`~core.tracing.flush_stage_trace`).
   Every producer keeps only its own module-level slot and delegates, so a
   producer's rows are exactly what ``TraceRun(STAGE)`` would have written, and
   no producer restates the lazy-init/flush body (the regression this file
   guards). ``training.training`` is the ONE documented reset-on-flush variant.

2. The orchestration stage registry lives ONCE in :mod:`core.tracing`
   (``ORCHESTRATION_TRACE_STAGES`` + its lane subset). ``training.prepare_all``
   DERIVES its stage lists from it instead of restating the names, and the four
   stages the registry used to declare row-free (``dedupe``, ``suite_inputs``,
   ``verify_handoff``, ``negative_supply``) are pinned to the producer module
   whose ``STAGE`` constant actually writes their rows.

Every test redirects the trace to a tmp file and pins a run id, so nothing
touches the real results tree.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import core.tracing as tracing

#: (module, the stage its writer owns) for every producer that uses the shim.
WRITERS = (
    ("model_tracks.run", "suite_run"),
    ("model_tracks.worker", "worker"),
    ("model_tracks.bundle_steps", "finalize"),
    ("model_tracks.ablation", "ablation"),
    ("model_tracks.ablation_cohort", "ablation_cohort"),
    ("model_tracks.baseline_ablation", "baseline_ablation"),
    ("model_tracks.post_training_ablation", "post_training_ablation"),
    ("model_tracks.staged_ablation", "staged_ablation"),
    ("model_tracks.local_complete", "local_complete"),
    ("model_tracks.snapshot_completion", "snapshot_completion"),
)


@pytest.fixture()
def shim_env(tmp_path, monkeypatch):
    """A pinned run id, a tmp trace destination and a frozen clock."""
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-shim-passthrough")
    monkeypatch.delenv("EUROMONITOR_RUN_ID", raising=False)
    monkeypatch.delenv("EUROMONITOR_TRACE_LANE", raising=False)
    target = tmp_path / "training_trace.csv"
    monkeypatch.setattr(tracing, "trace_path", lambda: target)
    monkeypatch.setattr(tracing, "_now", lambda: "2024-01-01T00:00:00+00:00")
    return target


@pytest.mark.parametrize("module_path,stage", WRITERS)
def test_producer_writer_passes_through_the_shared_shim_byte_for_byte(
    module_path, stage, shim_env, monkeypatch
):
    """A producer's writer is the shared shim and writes the shared row verbatim."""
    module = importlib.import_module(module_path)
    # the SAME shared helpers, imported, not a private copy
    assert module.stage_trace is tracing.stage_trace
    assert module.flush_stage_trace is tracing.flush_stage_trace
    assert module.STAGE == stage

    monkeypatch.setattr(module, "_TRACE", None)
    writer = module.trace()
    assert isinstance(writer, tracing.TraceRun) and writer.stage == stage

    # the row the producer's writer mints is EXACTLY the one core.tracing
    # builds for the same inputs: the shim is a pure pass-through, not a wrapper
    # with its own semantics
    row = writer.add("probe", "passthrough", in_count=3, out_count=2, reason="r")
    assert row == tracing.record(
        stage, "probe.passthrough", in_count=3, out_count=2, reason="r",
        run_id="run-shim-passthrough", at="2024-01-01T00:00:00+00:00",
    )

    # ...and it commits once to the destination, retaining the writer
    assert module.flush_trace() == shim_env
    assert tracing.read_trace(shim_env)["stage"].tolist() == [stage, stage]

    # an absent writer is a no-op, not an error
    monkeypatch.setattr(module, "_TRACE", None)
    assert module.flush_trace() is None


@pytest.mark.parametrize("module_path,_stage", WRITERS)
def test_no_producer_restates_the_lazy_writer_shim(module_path, _stage):
    """The duplicated body is gone: the logic has exactly one home."""
    module = importlib.import_module(module_path)
    assert "if _TRACE is None" not in inspect.getsource(module)


def test_the_lazy_shim_body_exists_in_exactly_one_file_repo_wide():
    """A repo-wide grep guard: only core.tracing may restate the lazy shim."""
    source_root = Path(tracing.__file__).resolve().parents[1]  # <repo>/src
    markers = ("if _TRACE is None", "if _TRAINING_TRACE is None")
    offenders = []
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in markers):
            offenders.append(path.relative_to(source_root).as_posix())
    assert offenders == []


def test_training_writer_is_the_one_reset_on_flush_variant(shim_env, monkeypatch):
    """``training.training`` keeps only its slot + the ONE documented reset."""
    from training import training as training_mod

    assert training_mod.stage_trace is tracing.stage_trace
    assert training_mod.flush_stage_trace is tracing.flush_stage_trace
    assert "if _TRAINING_TRACE is None" not in inspect.getsource(training_mod)

    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)
    assert training_mod.flush_training_trace() is None  # no writer yet: no-op

    writer = training_mod.training_trace()
    assert writer.stage == training_mod.TRAINING_STAGE
    # an EMPTY writer is a no-op that RETAINS the slot (only a commit resets)
    assert training_mod.flush_training_trace() is None
    assert training_mod._TRAINING_TRACE is writer

    # the pin is all-or-nothing, and a relabel is refused mid-run
    monkeypatch.setattr(training_mod, "_TRAINING_TRACE", None)
    assert training_mod.training_trace("full_bundle").stage == "full_bundle"
    with pytest.raises(ValueError, match="refusing to relabel"):
        training_mod.training_trace("training")

    training_mod.training_trace().add("probe", "row", in_count=1, out_count=1)
    assert training_mod.flush_training_trace() == shim_env
    assert training_mod._TRAINING_TRACE is None  # released for the next run id


# ── the ONE orchestration registry ─────────────────────────────────────────
def test_prepare_all_stage_lists_derive_from_the_one_registry():
    """``prepare_all`` derives its stage lists; it never restates the names."""
    from core.tracing import ORCHESTRATION_LANE_STAGES, ORCHESTRATION_TRACE_STAGES
    from training import prepare_all

    registry = tuple(ORCHESTRATION_TRACE_STAGES)
    lane = tuple(ORCHESTRATION_LANE_STAGES)
    assert prepare_all._LANE_STAGE_ORDER == lane
    assert prepare_all._EXTRA_LANE_STAGES == frozenset(lane)
    assert prepare_all.STAGES == tuple(s for s in registry if s not in lane)
    # the registry's keys ARE the orchestrator's plan: base stages plus the
    # lane stages it inserts, nothing more and nothing less
    assert set(prepare_all.STAGES) | set(lane) == set(registry)
    # the child-dispatch table may only name declared stages (the guard that
    # makes _STAGE_MODULES a table, never a second inventory)
    assert set(prepare_all._STAGE_MODULES) <= set(prepare_all.STAGES)


def test_registry_declares_the_producers_that_write_rows_and_only_two_gaps():
    """The four stale row-free declarations are corrected, tied to producers."""
    from core.tracing import ORCHESTRATION_TRACE_STAGES, trace_stages_for
    from model_tracks import package
    from training import dedupe, handoff, negative_supply

    # each declaration IS the producer's own stage constant, so it cannot drift
    assert trace_stages_for("dedupe") == (dedupe.STAGE,)
    assert trace_stages_for("suite_inputs") == (package.STAGE,)
    assert trace_stages_for("verify_handoff") == (handoff.STAGE,)
    assert trace_stages_for("negative_supply") == (negative_supply.STAGE,)
    # every one of those producers is a stage the orchestrator actually runs
    from training.prepare_all import STAGES
    assert {"dedupe", "suite_inputs", "verify_handoff"} <= set(STAGES)

    # only two stages are genuinely row-free: the inline gate census and the
    # inline diagnostic child (neither constructs a TraceRun)
    assert {
        stage for stage, names in ORCHESTRATION_TRACE_STAGES.items() if not names
    } == {"gate_census", "discriminator"}


def test_trace_stages_for_rejects_a_stage_the_registry_does_not_declare():
    with pytest.raises(ValueError, match="unknown orchestration stage"):
        tracing.trace_stages_for("not-a-stage")
