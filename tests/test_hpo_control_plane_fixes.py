"""Offline pins for the HPO control-plane fixes (NO PostgreSQL, NO GPU, NO optuna).

Covers the fencing TTL/heartbeat/reaper (F1/F9), the atomic shared budget and
FAIL-release semantics (F2/F8), post-commit champion promotion + champion warm
start (F3), the infrastructure-vs-trial failure split (F6), single post-commit
observation (F4), multi-objective ranking (F5) and the once-per-session reaper
(F7).  The PostgreSQL stores are exercised through their pure decision functions
and via dependency-injected fakes, exactly like ``tests/test_laya_hpo.py``.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from cli import laya_hpo
from training import hpo_budget, hpo_control_plane, hpo_fencing, laya_hpo_runtime


# ── F1: lease TTL / expiry ─────────────────────────────────────────────────
def test_lease_expired_boundaries():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fresh = now - timedelta(seconds=10)
    old = now - timedelta(seconds=400)
    assert hpo_fencing.lease_expired(updated_at=fresh, now=now, ttl_seconds=300) is False
    assert hpo_fencing.lease_expired(updated_at=old, now=now, ttl_seconds=300) is True
    assert hpo_fencing.lease_expired(updated_at=None, now=now, ttl_seconds=300) is True
    # ttl<=0 disables expiry (explicitly requested) but a missing row still fails.
    assert hpo_fencing.lease_expired(updated_at=old, now=now, ttl_seconds=0) is False


def test_lease_expired_normalises_naive_timestamps():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    naive_old = (datetime(2026, 1, 1)  # noqa: DTZ001 - naive is treated as UTC
                 - timedelta(seconds=400))
    assert hpo_fencing.lease_expired(updated_at=naive_old, now=now,
                                     ttl_seconds=300) is True


def _row(epoch=1, state="active", age_seconds=0):
    return {"epoch": epoch, "state": state,
            "updated_at": datetime(2026, 1, 1, tzinfo=UTC)
            - timedelta(seconds=age_seconds)}


def test_lease_status_fails_closed():
    lease = hpo_fencing.TrialLease("g", "laya", 5, 1, 300)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert hpo_fencing.lease_status(lease=lease, row=_row(), now=now,
                                    ttl_seconds=300) == (True, None)
    assert hpo_fencing.lease_status(lease=lease, row=None, now=now,
                                    ttl_seconds=300)[1] == "missing"
    assert hpo_fencing.lease_status(lease=lease, row=_row(epoch=2), now=now,
                                    ttl_seconds=300)[1] == "epoch"
    assert hpo_fencing.lease_status(lease=lease, row=_row(state="revoked"),
                                    now=now, ttl_seconds=300)[1] == "state"
    assert hpo_fencing.lease_status(lease=lease, row=_row(age_seconds=999),
                                    now=now, ttl_seconds=300)[1] == "expired"


def test_trial_lease_carries_its_ttl():
    lease = hpo_fencing.TrialLease("g", "laya", 1, 1)
    assert lease.ttl_seconds == hpo_fencing._LEASE_TTL_SECONDS


# ── F1: heartbeat renews; F9: issue retries ────────────────────────────────
class _RenewingStore:
    def __init__(self, ok=True):
        self.ok = ok
        self.renewed = 0

    def renew(self, lease):
        self.renewed += 1
        return self.ok


def test_lease_heartbeat_renews_and_is_failsafe():
    store = _RenewingStore()
    beat = hpo_fencing.LeaseHeartbeat(store, SimpleNamespace(ttl_seconds=300))
    assert beat.renew_once() is True
    assert store.renewed == 1
    # A renewal error never escapes (the zombie simply ages out).
    class Boom(_RenewingStore):
        def renew(self, lease):
            raise RuntimeError("db down")

    assert hpo_fencing.LeaseHeartbeat(
        Boom(), SimpleNamespace(ttl_seconds=300)).renew_once() is False


def test_lease_heartbeat_noop_store_has_no_thread():
    beat = hpo_fencing.LeaseHeartbeat(None, None)
    with beat as entered:
        assert entered is beat
    assert beat._thread is None


def test_issue_retries_a_lost_creation_race(monkeypatch):
    store = object.__new__(hpo_fencing.TrialLeaseStore)
    store._ttl_seconds = 300
    store._sleep = None
    attempts = []

    def flaky(*, generation_id, model_key, trial_number):
        attempts.append(trial_number)
        if len(attempts) < 3:
            return None  # lost the creation race
        return hpo_fencing.TrialLease(generation_id, model_key, trial_number,
                                      1, 300)

    store._issue_attempt = flaky
    lease = store.issue(generation_id="g", model_key="laya", trial_number=4)
    assert lease.trial_number == 4 and len(attempts) == 3


def test_issue_exhaustion_is_an_infrastructure_error():
    store = object.__new__(hpo_fencing.TrialLeaseStore)
    store._ttl_seconds = 300
    store._sleep = None
    store._issue_attempt = lambda **kwargs: None
    with pytest.raises(hpo_control_plane.HpoInfrastructureError):
        store.issue(generation_id="g", model_key="laya", trial_number=1,
                    attempts=2)


# ── F2 / F8: atomic budget + FAIL-release ──────────────────────────────────
def test_budget_counter_completed_never_exceeds_budget():
    counter = hpo_budget.BudgetCounter(budget=3)
    assert counter.reserve(10) == 3            # capped by the budget
    assert counter.reserve(1) == 0             # nothing free
    counter.complete(3)
    assert counter.completed == 3 and counter.in_flight == 0
    assert counter.reserve(1) == 0             # completing never frees a slot
    # only a RELEASE (a failed trial) frees capacity for another attempt.
    other = hpo_budget.BudgetCounter(budget=3)
    other.reserve(3)
    other.release(1)
    assert other.remaining() == 1 and other.reserve(1) == 1


def test_budget_counter_fail_release_does_not_starve_complete():
    # Budget 2, attempt ceiling 4: four FAILs may run, but they never consume a
    # COMPLETE slot, so the two COMPLETE trials still fit afterwards.
    counter = hpo_budget.BudgetCounter(budget=2, max_trials=4)
    for _ in range(4):
        assert counter.reserve(1) == 1
        counter.release(1)
    assert counter.completed == 0 and counter.attempted == 4
    assert counter.attempts_remaining() == 0   # the compute ceiling bounds it
    assert counter.remaining() == 2            # ...but the GOAL budget is intact
    counter2 = hpo_budget.BudgetCounter(budget=2, max_trials=6)
    for _ in range(4):
        counter2.reserve(1)
        counter2.release(1)
    assert counter2.reserve(1) == 1            # retries still allowed
    counter2.complete(1)
    assert counter2.completed == 1


def test_default_max_trials_is_a_finite_multiple():
    assert hpo_budget.default_max_trials(0) == 0
    assert hpo_budget.default_max_trials(24) == 48


# ── ReservedTrialLoop (F2/F3/F4/F6/F8 wiring) ──────────────────────────────
class _Ledger:
    def __init__(self, budget, max_trials=0):
        self.counter = hpo_budget.BudgetCounter(budget, max_trials)
        self.calls = []

    def reserve(self, amount=1):
        granted = self.counter.reserve(amount)
        self.calls.append(("reserve", granted))
        return granted

    def complete(self, amount=1):
        charged = self.counter.complete(amount)
        self.calls.append(("complete", charged))
        return charged

    def release(self, amount=1):
        released = self.counter.release(amount)
        self.calls.append(("release", released))
        return released


class _Study:
    def __init__(self, outcomes, *, raises=None):
        self._outcomes = list(outcomes)
        self._raises = raises
        self.trials = []

    def optimize(self, objective, n_trials=1, **kwargs):
        if self._raises is not None:
            raise self._raises
        state = self._outcomes.pop(0) if self._outcomes else "FAIL"
        self.trials.append(SimpleNamespace(
            number=len(self.trials),
            state=SimpleNamespace(name=state),
            user_attrs={"dev_accuracy": 0.9, "checkpoint": "/ck",
                        "hpo_lease_epoch": 1}))


class _Observer:
    def __init__(self):
        self.seen = []

    def observe(self, trial):
        self.seen.append(trial.number)


class _Champions:
    def __init__(self):
        self.promotions = []

    def promote(self, **kwargs):
        self.promotions.append(kwargs)
        return SimpleNamespace(**kwargs)


def test_reserved_loop_completes_within_budget_and_observes_once():
    study = _Study(["COMPLETE", "COMPLETE"])
    ledger = _Ledger(budget=2)
    observer = _Observer()
    champions = _Champions()
    loop = laya_hpo_runtime.ReservedTrialLoop(
        study, object(), ledger, observer=observer,
        champion_store=champions, generation_id="g", model_key="laya")
    assert loop.run() == "budget_exhausted"
    assert ledger.counter.completed == 2
    assert observer.seen == [0, 1]                 # exactly once each, post-commit
    assert [p["trial_number"] for p in champions.promotions] == [0, 1]


def test_reserved_loop_releases_failed_trials_then_still_completes():
    study = _Study(["FAIL", "COMPLETE"])
    ledger = _Ledger(budget=1, max_trials=2)
    champions = _Champions()
    loop = laya_hpo_runtime.ReservedTrialLoop(
        study, object(), ledger, champion_store=champions,
        generation_id="g", model_key="laya")
    assert loop.run() == "budget_exhausted"
    assert ledger.counter.completed == 1
    assert ("release", 1) in ledger.calls
    assert [p["trial_number"] for p in champions.promotions] == [1]


def test_reserved_loop_is_bounded_when_every_trial_fails():
    study = _Study(["FAIL", "FAIL", "FAIL", "FAIL"])
    ledger = _Ledger(budget=1, max_trials=2)
    loop = laya_hpo_runtime.ReservedTrialLoop(
        study, object(), ledger, generation_id="g", model_key="laya")
    assert loop.run() == "budget_exhausted"
    assert ledger.counter.attempted == 2          # finite, not an infinite loop
    assert ledger.counter.completed == 0


def test_reserved_loop_releases_and_propagates_infrastructure_error():
    study = _Study([], raises=hpo_control_plane.HpoInfrastructureError("boom"))
    ledger = _Ledger(budget=5)
    loop = laya_hpo_runtime.ReservedTrialLoop(
        study, object(), ledger, generation_id="g", model_key="laya")
    with pytest.raises(hpo_control_plane.HpoInfrastructureError):
        loop.run()
    assert ledger.counter.in_flight == 0          # slot never left pinned


def test_reserved_loop_honours_wall_clock_timeout():
    ticks = iter([0.0, 0.0, 100.0])
    study = _Study(["COMPLETE", "COMPLETE"])
    ledger = _Ledger(budget=5)
    loop = laya_hpo_runtime.ReservedTrialLoop(
        study, object(), ledger, generation_id="g", model_key="laya",
        timeout_s=10, clock=lambda: next(ticks))
    assert loop.run() == "timeout"
    assert ledger.counter.completed == 1


# ── F6: infrastructure error must not be an ordinary Exception ─────────────
def test_infrastructure_error_is_uncatchable_by_optuna_catch():
    assert issubclass(hpo_control_plane.HpoInfrastructureError, BaseException)
    assert not issubclass(hpo_control_plane.HpoInfrastructureError, Exception)


# ── F3: post-commit promotion + champion warm start ────────────────────────
def test_promote_committed_trial_only_promotes_complete():
    champions = _Champions()
    complete = SimpleNamespace(
        number=3, state=SimpleNamespace(name="COMPLETE"),
        user_attrs={"dev_accuracy": 0.8, "checkpoint": "/ck",
                    "hpo_lease_epoch": 2})
    fail = SimpleNamespace(number=4, state=SimpleNamespace(name="FAIL"),
                           user_attrs=dict(complete.user_attrs))
    no_epoch = SimpleNamespace(
        number=5, state=SimpleNamespace(name="COMPLETE"),
        user_attrs={"dev_accuracy": 0.9, "checkpoint": "/ck"})
    assert laya_hpo_runtime.promote_committed_trial(
        champions, fail, generation_id="g", model_key="laya") is None
    assert laya_hpo_runtime.promote_committed_trial(
        champions, no_epoch, generation_id="g", model_key="laya") is None
    assert champions.promotions == []
    promoted = laya_hpo_runtime.promote_committed_trial(
        champions, complete, generation_id="g", model_key="laya")
    assert promoted is not None
    assert champions.promotions[0]["lease_epoch"] == 2
    assert champions.promotions[0]["value"] == 0.8


def test_resolve_champion_artifact_reads_the_registry():
    class _Read:
        def __init__(self, artifact):
            self._artifact = artifact

        def read(self, *, generation_id, model_key):
            if self._artifact is None:
                return None
            return SimpleNamespace(artifact_snapshot=self._artifact)

    # non-champion modes never read.
    assert laya_hpo_runtime.resolve_champion_artifact(
        _Read("/ck"), generation_id="g", model_key="laya", mode="base") is None
    # champion present -> its artifact is used (the missing read side).
    assert laya_hpo_runtime.resolve_champion_artifact(
        _Read("/ck"), generation_id="g", model_key="laya",
        mode="champion") == "/ck"
    # no champion yet -> base fallback.
    assert laya_hpo_runtime.resolve_champion_artifact(
        _Read(None), generation_id="g", model_key="laya",
        mode="champion") is None


# ── F5: multi-objective value helpers ──────────────────────────────────────
def test_trial_value_helpers_handle_single_and_multi_objective():
    single = SimpleNamespace(value=0.7, values=None)
    assert laya_hpo_runtime.trial_primary_value(single) == 0.7
    assert laya_hpo_runtime.trial_full_value(single) == [0.7]
    multi = SimpleNamespace(value=None, values=[0.7, 12.5])
    assert laya_hpo_runtime.trial_primary_value(multi) == 0.7
    assert laya_hpo_runtime.trial_full_value(multi) == [0.7, 12.5]
    empty = SimpleNamespace(value=None, values=None)
    assert laya_hpo_runtime.trial_primary_value(empty) is None
    assert laya_hpo_runtime.trial_full_value(empty) is None


# ── F7: once-per-session reaper + idempotent DDL ───────────────────────────
def test_reap_stale_trials_returns_false_when_study_absent(monkeypatch):
    fake = SimpleNamespace(
        load_study=lambda **kwargs: (_ for _ in ()).throw(KeyError("nope")),
        storages=SimpleNamespace(fail_stale_trials=lambda study: None))
    monkeypatch.setitem(sys.modules, "optuna", fake)
    assert hpo_control_plane.reap_stale_trials_for_study("s", object()) is False


def test_reap_stale_trials_reaps_once_when_present(monkeypatch):
    reaped = []
    fake = SimpleNamespace(
        load_study=lambda **kwargs: SimpleNamespace(name="s"),
        storages=SimpleNamespace(
            fail_stale_trials=lambda study: reaped.append(study)))
    monkeypatch.setitem(sys.modules, "optuna", fake)
    assert hpo_control_plane.reap_stale_trials_for_study("s", object()) is True
    assert len(reaped) == 1


def test_reap_stale_trials_does_not_swallow_storage_errors(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("db unreachable")

    fake = SimpleNamespace(load_study=boom,
                           storages=SimpleNamespace(
                               fail_stale_trials=lambda study: None))
    monkeypatch.setitem(sys.modules, "optuna", fake)
    with pytest.raises(RuntimeError, match="db unreachable"):
        hpo_control_plane.reap_stale_trials_for_study("s", object())


def test_ensure_tables_is_idempotent_ddl(monkeypatch):
    emitted = []

    class _CreateTable:
        def __init__(self, table, *, if_not_exists=False):
            self.table = table
            self.if_not_exists = if_not_exists

    fake_schema = SimpleNamespace(CreateTable=_CreateTable)
    monkeypatch.setitem(sys.modules, "sqlalchemy",
                        SimpleNamespace(schema=fake_schema))
    monkeypatch.setitem(sys.modules, "sqlalchemy.schema", fake_schema)

    class _Conn:
        def execute(self, statement):
            emitted.append(statement)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Engine:
        def begin(self):
            return _Conn()

    class _Meta:
        sorted_tables = ("t1", "t2")

    hpo_control_plane.ensure_tables(_Engine(), _Meta())
    assert [statement.table for statement in emitted] == ["t1", "t2"]
    assert all(statement.if_not_exists for statement in emitted)



# ── structural: the staged kernel wires the fixes ──────────────────────────
def _worker_region():
    template = laya_hpo._HPO_KERNEL_TEMPLATE
    # The worker path is the WorkerSession class plus its run_worker facade
    # (obs's single-purpose-class refactor moved the body out of run_worker).
    return template.split("class WorkerSession", 1)[1].split(
        "def sha256_of", 1)[0]


def test_run_worker_uses_shared_ledger_and_loop():
    region = _worker_region()
    assert "WorkLedger(" in region
    assert "ReservedTrialLoop(" in region
    # fail_stale_trials is reaped ONCE per session by main(), not per worker.
    assert "fail_stale_trials(" not in region
    assert "reap_stale_trials_for_study(" in laya_hpo._HPO_KERNEL_TEMPLATE


def test_objective_no_longer_observes_before_commit():
    assert "_observe(trial)" not in laya_hpo._HPO_KERNEL_TEMPLATE


def test_runtime_source_carries_the_new_helpers():
    source = laya_hpo.hpo_runtime_source()
    for symbol in ("class BudgetCounter", "class WorkLedger",
                   "def default_max_trials", "class ReservedTrialLoop",
                   "def promote_committed_trial",
                   "def resolve_champion_artifact",
                   "def reap_stale_trials_for_study", "def ensure_tables"):
        assert symbol in source, symbol


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
