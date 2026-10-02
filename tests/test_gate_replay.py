"""Replay filters must not hide drift or change global census accounting."""
from concurrent.futures import Future

import pandas as pd

from training import gate_replay as replay


def _fixtures():
    committed = pd.DataFrame([
        ["1", "2", "proceed", "clean", "0.5"],
        ["3", "4", "fallback", "uncertain", "0.4"],
    ], columns=["gtin1", "gtin2", "gate_decision", "gate_reason", "similarity"])
    moved = pd.DataFrame([
        ["1", "2", "proceed", "clean", "hard_no", "mismatch", "0.5"],
        ["3", "4", "fallback", "uncertain", "proceed", "clean", "0.4"],
    ], columns=replay._DIFF_COLUMNS)
    return committed, moved


def test_filtered_display_does_not_hide_global_drift(monkeypatch, tmp_path, capsys):
    committed, moved = _fixtures()
    read_csv = pd.read_csv
    monkeypatch.setattr(replay, "replay", lambda **kwargs: (0, 2, moved))
    monkeypatch.setattr(replay, "fired_stage", lambda reason: reason)
    monkeypatch.setattr(replay, "RESULTS", tmp_path)
    monkeypatch.setattr(replay.pd, "read_csv", lambda *a, **kw: committed)
    reports = []
    def report(**kwargs):
        reports.append(kwargs)
        return {"degraded": {}, "survived": 0}
    monkeypatch.setattr(replay, "gate_census_drift_report", report)
    monkeypatch.setattr("sys.argv", ["gate_replay", "--fired", "absent"])
    assert replay.main() == 1
    assert "FIDELITY PASS" not in capsys.readouterr().out
    assert reports[0]["measured"] == {"total_pairs": 2, "hard_no": 1, "proceed": 1, "fallback": 0}
    assert reports[0]["current_gate"].gate_reason.tolist() == ["mismatch", "clean"]
    assert len(read_csv(tmp_path / "gate_replay_diff.csv")) == 2


def test_empty_candidate_universe(monkeypatch):
    monkeypatch.setattr(replay.pd, "read_csv", lambda *a, **kw: pd.DataFrame(columns=["gtin1", "gtin2"]))
    same, total, moved = replay.replay(workers=1)
    assert (same, total) == (0, 0)
    assert moved.empty
    assert list(moved.columns) == replay._DIFF_COLUMNS


def test_bounded_submission_is_lazy():
    class Pool:
        submitted = 0
        def submit(self, fn, chunk):
            self.submitted += 1
            future = Future()
            future.set_result(chunk)
            return future
    pool = Pool()
    results = replay._bounded_chunks(pool, ([i] for i in range(20)), workers=2)
    next(results)
    assert pool.submitted == 4
    assert len(list(results)) == 19
    assert pool.submitted == 20
