"""Pins for the D1-D7 neutral instrumentation (telemetry-only, no behavior).

Each test pins that the instrument RECORDS the measured state and never
changes training behavior: no raises flow from the instruments, no RNG or
ordering is touched, and the emitted surfaces (attestation field, [timing]
lines, tracking counters, scan counts) say what the decision rules read.
"""

from __future__ import annotations

from core.portable_archive import ByteCount
import json
from types import SimpleNamespace

import pytest
import torch

from training.attestation import (
    TrainingAttestation,
    record_provenance_verification,
)
from training.prepared_bundle import _timed_lineage_validations


def _attestation(run_dir: str, provenance_size: int | None):
    return TrainingAttestation(
        attestation_schema="er-training-attestation-v1",
        status="pass",
        run_dir=run_dir,
        finished_at="finished",
        bundle_path="bundle.pkl.gz",
        bundle_size=1234,
        provenance_size=provenance_size,
        plan_identity={"loss": "mnrl"},
        checks={},
        attested_at="now",
    )


def _manifest(run_dir, provenance: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "done", "provenance": provenance}), encoding="utf-8"
    )


def _provenance_size(provenance: dict) -> int:
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return ByteCount(canonical.encode()).total


def test_d1_attestation_records_verified_provenance(tmp_path, capsys):
    provenance = {"src/a.py": "a" * 64}
    _manifest(tmp_path, provenance)
    attestation = _attestation(str(tmp_path), _provenance_size(provenance))
    status = record_provenance_verification(attestation)
    assert attestation.provenance_verified == "verified"
    assert '[timing] training.attestation provenance_verified=verified' in capsys.readouterr().out


def test_d1_attestation_records_missing_manifest_without_raising(tmp_path, capsys):
    attestation = _attestation(str(tmp_path), 4321)
    status = record_provenance_verification(attestation)
    assert status == "missing_manifest"
    assert attestation.provenance_verified == "missing_manifest"


def test_d1_attestation_records_mismatch_and_unreadable_without_raising(tmp_path):
    _manifest(tmp_path, {"src/a.py": "a" * 64})
    # The recorded value is the canonical block's BYTE LENGTH (structural
    # identity, owner directive 2026-10-08), so a difference must change the
    # length: a same-length edit is the accepted blind spot of that vocabulary.
    est = _provenance_size({"src/a.py": "c" * 64, "src/extra.py": "d" * 64})
    attestation = _attestation(str(tmp_path), est)
    record_provenance_verification(attestation)
    changed = _attestation(str(tmp_path), est)
    (tmp_path / "manifest.json").write_bytes(b"\xff\xfe{'provenance': 'not json'")
    record_provenance_verification(changed)
    assert attestation.provenance_verified == "mismatch"
    assert changed.provenance_verified == "unreadable_manifest"


def test_d2_late_epoch_lr_records_applied_state_and_resume_flag(monkeypatch, capsys):
    from training.training import LateEpochLrDecayCallback

    lines: list[str] = []
    monkeypatch.setattr("training.training.emit_timing", lines.append)
    callback = LateEpochLrDecayCallback(
        enabled=True, start_epoch_fraction=0.5, multiplier=0.5
    )
    args = SimpleNamespace(num_train_epochs=2)
    state = SimpleNamespace(epoch=1.4, global_step=37, is_world_process_zero=True)
    optimizer = SimpleNamespace(
        param_groups=[{"lr": 0.02, "initial_lr": 0.02}]
    )
    scheduler = SimpleNamespace(base_lrs=[0.02])
    callback.on_train_begin(args, state, None)
    control = callback.on_epoch_begin(args, state, None, optimizer=optimizer, scheduler=scheduler)
    assert control is None
    assert callback.applied is True
    assert callback.resumed_start is True
    assert callback.global_step_at_resume == 37
    assert any(
        "[timing] training.late_epoch_lr applied" in line
        and "resumed_start=True" in line and "global_step_at_resume=37" in line
        for line in lines
    )
    # Idempotence pin: a second boundary pass must NOT re-apply or re-emit.
    control = callback.on_epoch_begin(args, state, None, optimizer=optimizer, scheduler=scheduler)
    assert control is None
    assert sum("training.late_epoch_lr applied" in line for line in lines) == 1


def test_d2_ann_refresh_records_refire_fingerprint(monkeypatch, capsys):
    from training.training import FineTunedAnnRefreshCallback

    lines: list[str] = []
    monkeypatch.setattr("training.training.emit_timing", lines.append)
    callback = FineTunedAnnRefreshCallback(
        df=None,
        payload=None,
        row_gtins=None,
        structured_features=None,
        train_gtins=None,
        existing=None,
        ann_state={"pairs": {}, "version": 0},
        slot_ids=[],
        fold_i=2,
        run_tag="telemetry_test",
        batch_size=8,
        max_seq_length=16,
        model=None,
        wandb_ctx=None,
    )
    state = SimpleNamespace(global_step=42)
    callback._record_ann_refire(state, completed_epoch=6, cadence=3)
    assert any(
        "[timing] training.ann_refresh fired" in line
        and "fold=2" in line and "last_epoch=0" in line
        and "cadence=3" in line and "global_step=42" in line
        for line in lines
    )


def test_d2_resume_scan_counts_v1_pointers_and_trainer_state(tmp_path):
    from scripts.audit_resume_state import scan

    resume = tmp_path / ".resume"
    resume.mkdir()
    (resume / "checkpoint-10--x.dvc").write_text("outs: []\n", encoding="utf-8")
    (resume / "checkpoint-11--y.dvc").write_text(
        "outs: [{'path': 'a'}]\n", encoding="utf-8"
    )
    (tmp_path / "results" / "run" / "checkpoint-1").mkdir(parents=True)
    (tmp_path / "results" / "run" / "checkpoint-1" / "trainer_state.json").write_text(
        "{}", encoding="utf-8"
    )
    report = scan(tmp_path)
    assert report["resume_pointer_v1"] == {
        "v1_pointer_total": 2, "v1_stranded": 1, "v1_outs_listed": 1
    }
    assert len(report["local_trainer_state"]) == 1
    assert report["resume_meta_v2"] == []


def test_d3_lineage_validations_emit_marks_for_both_paths(monkeypatch, capsys):
    from training import prepared_bundle

    calls: list[str] = []
    monkeypatch.setattr(
        prepared_bundle,
        "_validate_augmented_features",
        lambda payload, features, audits: calls.append("augmented"),
    )
    monkeypatch.setattr(
        prepared_bundle,
        "_validate_counterfactual_audits",
        lambda payload, audits: calls.append("counterfactual"),
    )
    _timed_lineage_validations({"payload": [], "structured_features": None,
                                "mask_audit": [], "hard_negative_mask_audit": []})
    out = capsys.readouterr().out
    assert calls == ["augmented", "counterfactual"]
    assert "[timing] prepared_bundle.lineage augmented_features:" in out
    assert "[timing] prepared_bundle.lineage counterfactual_audit:" in out


def test_d4_uniformity_counters_classify_batches_and_drain():
    from training.losses import _tracking_contrastive_loss

    loss = _tracking_contrastive_loss(
        SimpleNamespace(),
        margin=0.7,
        structured_feature_weight=0.3,
        uniformity_weight=0.5,
        uniformity_temperature=0.05,
        uniformity_min_batch_size=6,
        label_smoothing=0.0,
    )
    below = loss._uniformity_penalty([torch.ones(2, 3), torch.ones(2, 3)])
    active = loss._uniformity_penalty([torch.ones(3, 3), torch.eye(3)])
    assert loss._uniformity_below_min_batches == 1
    assert loss._uniformity_active_batches == 1
    assert below.item() == 0.0
    assert loss._uniformity_below_min_batches == 1
    assert active.item() != 0.0
    loss._tracking_batches = 1
    stats = loss.pop_tracking_stats()
    assert stats["uniformity_active_batches"] == 1.0
    assert stats["uniformity_below_min_batches"] == 1.0
    followup = loss.pop_tracking_stats()
    assert followup == {} or (
        followup.get("uniformity_active_batches") == 0.0
        and followup.get("uniformity_below_min_batches") == 0.0
    )


def test_d6_unattested_path_carries_data_digest_mark():
    import inspect
    import training.train_prepared as tp

    main_src = inspect.getsource(tp._main)
    assert "data_size_revalidation" in main_src
    assert "validate_run_plan" in main_src


def test_d7_timed_load_config_is_identical_and_records_seconds():
    from core.common import load_config
    from training.training import _CFG_DEEPCOPY_TOTALS, _timed_load_config

    before = dict(_CFG_DEEPCOPY_TOTALS)
    value = _timed_load_config("telemetry.d7")
    expected = load_config()
    assert type(value) is type(expected)
    assert _CFG_DEEPCOPY_TOTALS.get("telemetry.d7", 0.0) >= 0.0
    for key in before:
        assert key in _CFG_DEEPCOPY_TOTALS

    import inspect
    import training.training as tr

    assert "config_deepcopy fold=" in inspect.getsource(tr.train_one_config)
