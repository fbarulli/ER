"""Public-behavior pins for the LayaTrainingRun owner (the training-run facade).

One focused test per public behavior, all offline (no network, no kaggle, no
GPU): the owner resolves the config-declared local study and stages the HPO
payload end-to-end, and teardown composes stop-then-delete as a dry run.
"""
from __future__ import annotations

from pathlib import Path

from cli import laya_hpo, laya_lane
from cli.laya_training_run import LayaRunKind, LayaTrainingRunFactory


def _stub_lane_boundary(monkeypatch, tmp_path):
    """Stub only the transport/git boundaries; exercise the real owners."""
    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-run-1")
    monkeypatch.delenv(laya_hpo.OPTUNA_URL_ENV, raising=False)
    monkeypatch.setattr(laya_lane, "TRAIN_ROOT", tmp_path)
    monkeypatch.setattr(laya_lane, "staging_dir", lambda: Path(tmp_path))
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(laya_lane, "_log_lane", lambda line: None)
    monkeypatch.setattr(laya_lane, "stage_finetune_dataset_payload",
                        lambda **kwargs: {"payload": str(tmp_path / "ds"),
                                          "files": {}})
    monkeypatch.setattr(laya_hpo, "_current_git_branch", lambda: "laya-hpo")
    from core import runtime_inputs

    monkeypatch.setattr(runtime_inputs, "require_published_tip_match",
                        lambda rev, repo, branch: rev)


def test_training_run_resolves_local_study_and_stages_hpo(monkeypatch, tmp_path):
    """No OPTUNA_STORAGE_URL: resolve_study is local, and stage writes the payload."""
    _stub_lane_boundary(monkeypatch, tmp_path)
    run = LayaTrainingRunFactory.from_config(train_root=tmp_path)

    study = run.resolve_study()
    assert study.backend.value == "local"

    receipt = run.stage(LayaRunKind.HPO)
    assert receipt["kind"] == laya_hpo.HPO_DECISION
    assert receipt["optuna_storage"]["backend"] == "local"
    assert (Path(receipt["staged"]) / laya_hpo.HPO_CODE_FILE).is_file()


def test_teardown_stops_then_deletes_in_dry_run(monkeypatch, tmp_path):
    """Teardown composes the canonical stop + delete, both dry-run by default."""
    _stub_lane_boundary(monkeypatch, tmp_path)
    run = LayaTrainingRunFactory.from_config(train_root=tmp_path)

    plan = run.teardown("owner/er-laya-finetune")
    assert plan["kernel"] == "owner/er-laya-finetune"
    assert plan["stop"]["mode"] == "dry-run"
    assert plan["delete"]["mode"] == "dry-run"
