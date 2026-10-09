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

    monkeypatch.setattr(runtime_inputs, "publish_run_branch",
                        lambda repo, branch: False)
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


def test_launch_streams_the_run_log_into_the_declared_transcript(
        monkeypatch, tmp_path):
    """The launch path writes the run's live console to the config-declared
    ``logs/laya/lane.log`` (W&B primary), continuously — not only into W&B.

    The path is the ``laya.logs_dir`` / ``laya.lane_log`` SSOT, so the owner can
    ``tail -f`` it while the remote run is live.
    """
    import json

    from cli.kaggle_kernels import KaggleKernels
    from cli.kaggle_monitor import KaggleMonitor
    from cli.kaggle_watcher import KernelLifecycle, KernelWatcher
    from core.wandb_ctx import WandbRunReader

    _stub_lane_boundary(monkeypatch, tmp_path)
    monkeypatch.setenv("ER_KAGGLE_LANE_APPEND", "1")
    monkeypatch.setattr("cli.kaggle_lane.TRAIN_ROOT", tmp_path)
    (tmp_path / "kernel-metadata.json").write_text(
        json.dumps({"id": "owner/er-laya-hpo"}), encoding="utf-8")
    run = LayaTrainingRunFactory.from_config(train_root=tmp_path)
    monkeypatch.setattr(run, "publish", lambda *a, **k: {"mode": "executed"})
    monkeypatch.setattr(run, "push", lambda *a, **k: {"pushed": True})

    # No detached process: run the watcher inline.
    monkeypatch.setattr(KernelWatcher, "spawn",
                        lambda self: self.autowatch(
                            execute=True, slug="owner/er-laya-hpo"))
    # The canonical reader takes the W&B primary source and yields one update.
    monkeypatch.setattr(WandbRunReader, "available",
                        classmethod(lambda cls: True))
    monkeypatch.setattr(
        WandbRunReader, "stream",
        lambda self, *, max_polls: iter([{
            "run": self.path, "state": "running", "metrics": {"loss": 0.1},
            "new_output": "epoch 1 loss=0.1\n", "console_error": None}]))
    # Terminal poll + harvest/release/session-capture are network seams.
    monkeypatch.setattr(KaggleKernels, "kernel_status",
                        staticmethod(lambda *a, **k: {"status": "complete",
                                                      "raw": "COMPLETE"}))
    monkeypatch.setattr(KernelLifecycle, "harvest_and_stop",
                        lambda **k: {"stop": {"stopped": True}})
    monkeypatch.setattr(KaggleMonitor, "capture_kernel_session_id",
                        staticmethod(lambda slug: {}))

    plan = run.launch(LayaRunKind.HPO, stage_dir=tmp_path,
                      run_tag="laya_hpo_1", execute=True)

    assert plan["watch"]["kernel"] == "owner/er-laya-hpo"
    transcript = tmp_path / "logs" / "laya" / "lane.log"
    assert transcript.is_file(), "the launch must create logs/laya/lane.log"
    assert "epoch 1 loss=0.1" in transcript.read_text(encoding="utf-8")
