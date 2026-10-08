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


def test_trial_observer_composes_the_three_surfaces(tmp_path):
    observer = obs.TrialObserver(tmp_path)
    trial = _trial(number=4, value=0.81, params={"a": 1})
    observer.observe(trial)
    observer.observe(trial, promoted=True)
    mirror_path = observer.flush()
    assert mirror_path.is_file()
    assert observer.events.count() == 2
    assert len(observer.ledger.load()) == 2
    assert len(observer.mirror.load()) == 2
    payload = observer.as_dict()
    assert payload["events_count"] == 2
    assert payload["mirror"].endswith("study_mirror.jsonl")
    assert payload["ledger"].endswith("hpo_trials.jsonl")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
