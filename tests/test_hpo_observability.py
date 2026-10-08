"""Offline public-API tests for the HPO control-plane observability surfaces.

No PostgreSQL, no Optuna: trials are plain objects, so the CDC event log, the
local study mirror and the offline ledger are all exercised on the host.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from training import hpo_observability as obs


def _trial(number=0, value=0.5, state="COMPLETE", params=None, attrs=None):
    return SimpleNamespace(number=number, value=value,
                           state=SimpleNamespace(name=state),
                           params=dict(params or {}),
                           user_attrs=dict(attrs or {}))


def test_trial_event_log_emits_cdc_rows(tmp_path):
    log = obs.TrialEventLog(tmp_path / "trial_events.jsonl", clock=lambda: 7.0)
    row = log.observe(_trial(number=2, value=0.9, params={"lr": 1e-4},
                             attrs={"dev_loss": 0.1}))
    assert row["event"] == obs.EVENT_COMPLETE
    assert row["trial_number"] == 2 and row["at"] == 7.0
    rows = log.read()
    assert len(rows) == 1 and rows[0]["params"] == {"lr": 1e-4}
    log.emit("trial_created", trial_number=3)
    assert log.count() == 2
    assert log.read()[-1]["event"] == "trial_created"


def test_trial_event_log_maps_states_and_is_fail_soft(tmp_path):
    log = obs.TrialEventLog(tmp_path / "trial_events.jsonl")
    assert log.observe(_trial(state="FAIL"))["event"] == obs.EVENT_FAIL
    assert log.observe(_trial(state="PRUNED"))["event"] == obs.EVENT_PRUNE
    assert log.observe(_trial(state="RUNNING"))["event"] == obs.EVENT_CREATE
    # A write to an unwritable path must never raise.
    bad = obs.TrialEventLog("/proc/definitely-not-here/trial_events.jsonl")
    assert bad.emit("x", trial_number=1)["event"] == "x"


def test_study_mirror_flush_load_and_best(tmp_path):
    mirror = obs.StudyMirror(tmp_path / "study_mirror.jsonl")
    mirror.record(_trial(number=0, value=0.5))
    mirror.record(_trial(number=1, value=0.9))
    mirror.record(_trial(number=2, value=None, state="FAIL"))
    path = mirror.flush()
    assert path.is_file()
    assert len(mirror.load()) == 3
    assert mirror.best()["number"] == 1
    assert mirror.best("minimize")["number"] == 0
    # No complete rows -> None.
    empty = obs.StudyMirror(tmp_path / "empty.jsonl")
    assert empty.best() is None


def test_offline_trial_ledger_fallback(tmp_path):
    ledger = obs.OfflineTrialLedger(tmp_path / "hpo_trials.jsonl")
    ledger.append(trial_number=0, value=0.4, params={"x": 1})
    ledger.append(trial_number=1, value=0.7)
    assert len(ledger.load()) == 2
    assert ledger.best()["trial_number"] == 1
    assert ledger.best("minimize")["trial_number"] == 0
    ledger.observe(_trial(number=2, value=0.2, state="FAIL"))
    assert ledger.best()["trial_number"] == 1  # FAIL rows are not best


def test_trial_observer_online_mode_skips_the_offline_ledger(tmp_path):
    observer = obs.TrialObserver(tmp_path)  # Postgres mode
    observer.observe(_trial(number=1, value=0.7))
    assert observer.events.count() == 1
    assert observer.mirror.rows() and not observer.ledger.load()
    assert observer.mode == "postgres"
    assert observer.as_dict()["ledger_active"] is False


def test_trial_observer_offline_mode_writes_the_ledger(tmp_path):
    observer = obs.TrialObserver(tmp_path, offline=True)
    trial = _trial(number=4, value=0.81, params={"a": 1})
    observer.observe(trial)
    observer.observe(trial)  # idempotent: the same event is not re-emitted
    assert observer.events.count() == 1
    assert len(observer.ledger.load()) == 1
    assert len(observer.mirror.rows()) == 1
    observer.observe(trial, promoted=True)  # a DIFFERENT event is written
    mirror_path = observer.flush()
    assert mirror_path.is_file()
    assert observer.events.count() == 2
    assert len(observer.ledger.load()) == 1  # one row per trial number
    assert len(observer.mirror.load()) == 1
    payload = observer.as_dict()
    assert payload["mode"] == "offline" and payload["ledger_active"] is True
    assert payload["events_count"] == 2
    assert payload["events_dropped"] == 0
    assert payload["mirror"].endswith("study_mirror.jsonl")
    assert payload["ledger"].endswith("hpo_trials.jsonl")


def test_trial_event_log_has_a_stable_idempotent_event_id(tmp_path):
    path = tmp_path / "trial_events.jsonl"
    first = obs.TrialEventLog(path)
    row = first.observe(_trial(number=2, value=0.9))
    assert row["event_id"] == "trial_completed#2"
    assert first.count() == 1
    # A second observer (a resumed session / another worker) re-observes the
    # same trial: the event_id dedupes it, so the row is skipped.
    resumed = obs.TrialEventLog(path)
    resumed.observe(_trial(number=2, value=0.9))
    assert resumed.count() == 0
    assert len(resumed.read()) == 1
    assert resumed.read()[0]["event_id"] == "trial_completed#2"


def test_trial_event_log_dropped_writes_are_surfaced(tmp_path):
    bad = obs.TrialEventLog("/proc/definitely-not-here/trial_events.jsonl")
    bad.emit("x", trial_number=1)
    assert bad.count() == 0 and bad.dropped() == 1


def test_trial_observer_sync_is_idempotent_across_sessions(tmp_path):
    trials = [_trial(number=n, value=0.5 + n / 10) for n in range(3)]
    first = obs.TrialObserver(tmp_path)
    result = first.sync(trials)
    assert result == {"written": 3, "skipped": 0}
    first.flush()
    # A resumed session over the SAME committed study emits nothing new.
    resumed = obs.TrialObserver(tmp_path)
    assert resumed.sync(trials) == {"written": 0, "skipped": 3}
    assert len(resumed.events.read()) == 3
    assert len(resumed.mirror.load()) == 3


def test_study_mirror_best_uses_the_primary_of_a_multi_objective(tmp_path):
    mirror = obs.StudyMirror(tmp_path / "study_mirror.jsonl")
    mirror.record(_trial(number=0, value=[0.5, 9.9]))
    mirror.record(_trial(number=1, value=[0.9, 1.0]))
    assert mirror.best()["number"] == 1  # primary, not lexicographic tail
    assert mirror.best("minimize")["number"] == 0


def test_study_mirror_best_ranks_by_configured_multi_objective_directions(tmp_path):
    mirror = obs.StudyMirror(tmp_path / "study_mirror.jsonl")
    mirror.record(_trial(number=0, value=[0.9, 9.0]))   # best accuracy, slow
    mirror.record(_trial(number=1, value=[0.9, 1.0]))   # tied accuracy, fast
    mirror.record(_trial(number=2, value=[0.5, 0.1]))   # worse accuracy
    # maximize accuracy, minimize secondary -> the tie on the primary breaks
    # toward the lower secondary (trial 1), not the lexicographic max (trial 0).
    assert mirror.best(["maximize", "minimize"])["number"] == 1


def test_study_mirror_best_prefers_in_memory_rows_on_resume(tmp_path):
    path = tmp_path / "study_mirror.jsonl"
    stale = obs.StudyMirror(path)
    stale.record(_trial(number=0, value=0.1))
    stale.flush()
    resumed = obs.StudyMirror(path)
    resumed.record(_trial(number=0, value=0.1))   # same value...
    resumed.record(_trial(number=1, value=0.99))  # ...then a better new trial
    # best() must see the in-memory row 1, not the stale single-row file.
    assert resumed.best()["number"] == 1


def test_session_artifacts_build_a_verified_snapshot(tmp_path):
    from training import hpo_persistence

    work = tmp_path / "work"
    work.mkdir()
    (work / "laya-hpo.receipt.json").write_text("{}", encoding="utf-8")
    observer = obs.TrialObserver(work / "hpo_observability", offline=True)
    observer.observe(_trial(number=0, value=0.5))
    observer.flush()
    include = [p for p in (
        work / "laya-hpo.receipt.json",
        work / "hpo_observability" / "trial_events.jsonl",
        work / "hpo_observability" / "study_mirror.jsonl",
        work / "hpo_observability" / "hpo_trials.jsonl") if p.exists()]
    snapshot = hpo_persistence.build_snapshot(
        generation=work, sequence=1, optuna_db=None, include=include)
    hpo_persistence.verify_snapshot(snapshot)
    assert (snapshot / "READY").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
