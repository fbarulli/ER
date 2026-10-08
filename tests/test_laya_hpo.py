"""Offline pins for the laya HPO lane (NO PostgreSQL, NO GPU).

Covers search-space construction, trial->config mapping, study-name/storage
resolution, objective wiring/failure paths and the staged kernel's secret
hygiene. The shared HPO primitives run on PostgreSQL (psycopg + the postgres
insert dialect), so the fencing/champion stores are stubbed here exactly as
``tests/test_optuna_callback_wiring.py`` pins wiring without a live service.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli import laya_hpo, laya_lane
from training import laya_hpo_runtime


# ── fakes ──────────────────────────────────────────────────────────────────
class FakeTrial:
    def __init__(self, number: int = 0):
        self.number = number
        self.calls: list[tuple] = []
        self.attrs: dict = {}

    def suggest_int(self, name, lo, hi):
        self.calls.append(("int", name, lo, hi))
        return lo

    def suggest_float(self, name, lo, hi, log=False):
        self.calls.append(("float", name, lo, hi, log))
        return lo

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, tuple(choices)))
        return choices[0]

    def set_user_attr(self, key, value):
        self.attrs[key] = value


class FakeLeaseStore:
    def __init__(self, *, stale: bool = False):
        self.stale = stale
        self.issued: list = []
        self.asserted: list = []
        self.revoked: list = []

    def issue(self, *, generation_id, model_key, trial_number):
        lease = SimpleNamespace(epoch=3, trial_number=trial_number)
        self.issued.append((generation_id, model_key, trial_number, lease))
        return lease

    def assert_current(self, lease):
        self.asserted.append(lease)
        if self.stale:
            raise RuntimeError("stale HPO worker fenced")

    def revoke(self, lease):
        self.revoked.append(lease)
        return True


class FakeChampionStore:
    def __init__(self):
        self.promotions: list = []

    def promote(self, **kwargs):
        self.promotions.append(kwargs)
        return SimpleNamespace(**kwargs)


# ── search-space construction ──────────────────────────────────────────────
def test_space_loads_declared_dials_and_metadata():
    space = laya_hpo.load_space()
    assert set(space["dials"]) == {
        "early_stop_patience", "warmup_frac", "lr_scheduler", "encoder_lr",
        "head_lr", "weight_decay", "micro_batch", "grad_accum",
        "label_smoothing", "sigma_start", "sigma_end", "ema", "swa",
        "layer_decay",
    }
    assert space["model_key"] == "laya"
    assert space["objective"] == {
        "direction": "maximize", "primary": "dev_accuracy",
        "secondary": "dev_loss", "forbidden": "test",
    }
    assert space["n_trials"] > 0 and space["n_jobs"] >= 1


def test_space_targets_route_to_the_right_channel():
    dials = laya_hpo.load_space()["dials"]
    assert dials["encoder_lr"]["target"] == "config"
    assert dials["micro_batch"]["target"] == "config"
    assert dials["early_stop_patience"]["target"] == "control"
    assert dials["lr_scheduler"]["target"] == "control"
    assert dials["ema"]["target"] == "control"


@pytest.mark.parametrize("mutation, message", [
    (lambda s: s["dials"].clear(), "non-empty"),
    (lambda s: s["dials"]["encoder_lr"].update(target="nonsense"), "target"),
    (lambda s: s["dials"]["encoder_lr"].update(type="nonsense"), "type"),
    (lambda s: s["dials"]["weight_decay"].update(lo=1.0, hi=0.0), "range"),
    (lambda s: s["dials"]["micro_batch"].update(choices=[]), "choices"),
    (lambda s: s["dials"].update(not_a_field={"type": "float",
                                              "target": "config",
                                              "lo": 0.0, "hi": 1.0}),
     "FinetuneSpec"),
])
def test_validate_space_rejects_malformed(mutation, message):
    space = json.loads(json.dumps(laya_hpo.load_space()))
    mutation(space)
    with pytest.raises(ValueError, match=message):
        laya_hpo.validate_space(space)


def test_validate_space_rejects_bad_objective():
    space = json.loads(json.dumps(laya_hpo.load_space()))
    space["objective"]["primary"] = "test_accuracy"
    with pytest.raises(ValueError, match="dev_accuracy"):
        laya_hpo.validate_space(space)


def test_space_file_carries_no_secret():
    text = laya_hpo.space_path().read_text(encoding="utf-8").lower()
    assert "postgresql://" not in text
    assert "password" not in text


# ── trial -> config mapping ────────────────────────────────────────────────
def test_sample_dials_dispatches_generically_by_type():
    space = laya_hpo.load_space()
    trial = FakeTrial()
    sampled = laya_hpo.sample_dials(trial, space)
    assert set(sampled) == set(space["dials"])
    by_name = {call[1]: call for call in trial.calls}
    assert by_name["early_stop_patience"][0] == "int"
    assert by_name["encoder_lr"][0] == "float" and by_name["encoder_lr"][4] is True
    assert by_name["weight_decay"][0] == "float" and by_name["weight_decay"][4] is False
    assert by_name["lr_scheduler"][0] == "categorical"
    assert by_name["micro_batch"][2] == (4, 8, 16)


def test_apply_dials_routes_by_target():
    space = laya_hpo.load_space()
    base_config = {"encoder_lr": 2.5e-5, "epochs": 8}
    base_control = {"early_stop_patience": 2, "ema": False}
    dials = {"encoder_lr": 1.0e-5, "micro_batch": 16,
             "early_stop_patience": 4, "ema": True}
    config, control = laya_hpo.apply_dials(base_config, base_control, dials,
                                           space)
    assert config["encoder_lr"] == 1.0e-5 and config["micro_batch"] == 16
    assert config["epochs"] == 8  # untouched
    assert control["early_stop_patience"] == 4 and control["ema"] is True
    assert "micro_batch" not in control and "encoder_lr" not in control


def test_route_dials_is_the_shared_runtime_implementation():
    assert laya_hpo.apply_dials is laya_hpo_runtime.route_dials
    assert laya_hpo.sample_dials is laya_hpo_runtime.sample_dials


# ── study-name / storage resolution ────────────────────────────────────────
def test_study_identity_requires_generation(monkeypatch):
    monkeypatch.delenv(laya_hpo.GENERATION_ID_ENV, raising=False)
    with pytest.raises(RuntimeError, match=laya_hpo.GENERATION_ID_ENV):
        laya_hpo.study_identity(space=laya_hpo.load_space(), generation_id=None)


def test_study_identity_builds_generation_scoped_name(monkeypatch):
    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-2026")
    generation, key, name = laya_hpo.study_identity(space=laya_hpo.load_space())
    assert (generation, key) == ("gen-2026", "laya")
    assert name == "euromonitor::gen-2026::laya"


def test_require_optuna_url_missing_is_loud():
    with pytest.raises(RuntimeError, match="OPTUNA_STORAGE_URL"):
        laya_hpo.require_optuna_url(read_env=lambda name: None)


def test_require_optuna_url_rejects_sqlite():
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        laya_hpo.require_optuna_url(read_env=lambda name: "sqlite:///x.db")


def test_require_optuna_url_accepts_postgres():
    url = "postgresql://user:secret@db.example.com:5432/optuna"
    assert laya_hpo.require_optuna_url(read_env=lambda name: url) == url


def test_resolve_study_config_uses_shared_control_plane(monkeypatch):
    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-1")
    sentinel_cfg = SimpleNamespace(url="postgresql://h/db")
    sentinel_storage = object()
    monkeypatch.setattr(laya_hpo, "storage_from_environment",
                        lambda: sentinel_cfg)
    monkeypatch.setattr(laya_hpo, "create_storage", lambda cfg: sentinel_storage)
    storage, name, cfg = laya_hpo.resolve_study_config(space=laya_hpo.load_space())
    assert storage is sentinel_storage and cfg is sentinel_cfg
    assert name == "euromonitor::gen-1::laya"


# ── objective wiring / failure paths ───────────────────────────────────────
def test_objective_value_leases_fences_and_promotes():
    trial = FakeTrial(number=7)
    leases = FakeLeaseStore()
    champions = FakeChampionStore()
    value = laya_hpo_runtime.objective_value(
        trial, lambda t: (0.912, 0.31, Path("/kaggle/working/ckpt_7")),
        leases, champions, "gen-1", "laya")
    assert value == pytest.approx(0.912)
    assert trial.attrs["dev_accuracy"] == pytest.approx(0.912)
    assert trial.attrs["dev_loss"] == pytest.approx(0.31)
    assert trial.attrs["checkpoint"] == "/kaggle/working/ckpt_7"
    assert leases.issued[0][:3] == ("gen-1", "laya", 7)
    assert leases.asserted and not leases.revoked
    assert len(champions.promotions) == 1
    promotion = champions.promotions[0]
    assert promotion["trial_number"] == 7
    assert promotion["value"] == pytest.approx(0.912)
    assert promotion["lease_epoch"] == 3
    assert promotion["artifact_snapshot"] == "/kaggle/working/ckpt_7"


def test_objective_value_can_omit_dev_loss():
    trial = FakeTrial(number=1)
    laya_hpo_runtime.objective_value(
        trial, lambda t: (0.5, None, Path("/ckpt")),
        FakeLeaseStore(), FakeChampionStore(), "gen", "laya")
    assert "dev_loss" not in trial.attrs


def test_objective_value_revokes_lease_on_failure():
    trial = FakeTrial(number=3)
    leases = FakeLeaseStore()
    champions = FakeChampionStore()

    def boom(_trial):
        raise RuntimeError("finetune exploded")

    with pytest.raises(RuntimeError, match="finetune exploded"):
        laya_hpo_runtime.objective_value(trial, boom, leases, champions,
                                         "gen", "laya")
    assert leases.revoked == [item[3] for item in leases.issued]
    assert champions.promotions == []


def test_objective_value_fences_a_stale_lease_before_promotion():
    trial = FakeTrial(number=5)
    leases = FakeLeaseStore(stale=True)
    champions = FakeChampionStore()
    with pytest.raises(RuntimeError, match="stale"):
        laya_hpo_runtime.objective_value(
            trial, lambda t: (0.99, 0.01, Path("/ckpt")),
            leases, champions, "gen", "laya")
    assert leases.asserted  # fenced
    assert leases.revoked   # revoked for a replacement worker
    assert champions.promotions == []  # never published


def test_finished_trial_count_counts_only_terminal_states():
    trials = [SimpleNamespace(state=SimpleNamespace(name=n)) for n in
              ("COMPLETE", "RUNNING", "FAIL", "PRUNED", "WAITING")]
    study = SimpleNamespace(trials=trials)
    assert laya_hpo_runtime.finished_trial_count(
        study, ("COMPLETE", "PRUNED", "FAIL")) == 3


# ── staged kernel composition + secret hygiene ─────────────────────────────
def test_hpo_runtime_source_has_no_future_import_and_carries_primitives():
    source = laya_hpo.hpo_runtime_source()
    assert "from __future__ import" not in source
    for symbol in ("def storage_from_environment", "class TrialLeaseStore",
                   "class ChampionStore", "def objective_value",
                   "def generation_study_name", "def fail_stale_trials"):
        assert symbol in source, symbol


def test_assert_secret_absent_raises_on_leak():
    secret = "postgresql://u:p@host:5432/db"
    with pytest.raises(RuntimeError, match="refusing to persist"):
        laya_hpo._assert_secret_absent({"nested": {"url": secret}}, secret)
    laya_hpo._assert_secret_absent({"ok": True}, secret)  # no raise


def _stage(monkeypatch, tmp_path, url):
    """Stage with the network/git/tip dependencies stubbed."""
    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-stage-1")
    monkeypatch.setattr(laya_lane, "staging_dir", lambda: Path(tmp_path))
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(laya_lane, "_env_value",
                        lambda name: url if name == laya_hpo.OPTUNA_URL_ENV
                        else None)
    monkeypatch.setattr(laya_lane, "_log_lane", lambda line: None)
    monkeypatch.setattr(laya_lane, "stage_finetune_dataset_payload",
                        lambda **kwargs: {"payload": str(tmp_path / "ds"),
                                          "files": {}})
    monkeypatch.setattr(laya_hpo, "_current_git_branch", lambda: "laya-hpo")
    from core import runtime_inputs
    monkeypatch.setattr(runtime_inputs, "require_published_tip_match",
                        lambda rev, repo, branch: rev)
    return laya_hpo.stage_laya_hpo_kernel()


def test_stage_kernel_requires_optuna_url(monkeypatch, tmp_path):
    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-stage-2")
    monkeypatch.setattr(laya_lane, "_env_value", lambda name: None)
    with pytest.raises(RuntimeError, match="OPTUNA_STORAGE_URL"):
        laya_hpo.stage_laya_hpo_kernel()


def test_stage_kernel_bakes_secret_only_into_the_script(monkeypatch, tmp_path):
    secret = "postgresql://hpo_user:s3cret@db.example.com:5432/optuna"
    receipt = _stage(monkeypatch, tmp_path, secret)
    stage = Path(receipt["staged"])
    script = (stage / laya_hpo.HPO_CODE_FILE).read_text(encoding="utf-8")
    compile(script, str(stage / laya_hpo.HPO_CODE_FILE), "exec")
    assert secret in script  # the URL must reach the remote process
    # ...and nowhere a stored manifest/result can carry it.
    serialized_receipt = (stage / laya_hpo.HPO_RECEIPT_FILE).read_text(encoding="utf-8")
    serialized_metadata = (stage / "kernel-metadata.json").read_text(encoding="utf-8")
    assert secret not in serialized_receipt
    assert secret not in serialized_metadata
    assert secret not in json.dumps(receipt)
    assert receipt["optuna_storage"]["url_persisted_to_manifest"] is False


def test_stage_kernel_metadata_and_receipt_shape(monkeypatch, tmp_path):
    receipt = _stage(monkeypatch, tmp_path, "postgresql://u:p@h/db")
    meta = json.loads((Path(receipt["staged"]) / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    assert meta["enable_gpu"] is True and meta["is_private"] is True
    assert meta["code_file"] == laya_hpo.HPO_CODE_FILE
    assert meta["dataset_sources"] == [
        laya_lane.training_cfg().laya.finetune_dataset_slug,
        laya_lane.training_cfg().laya.base_model_dataset,
    ]
    assert receipt["study"]["study_name"] == "euromonitor::gen-stage-1::laya"
    assert receipt["kind"] == laya_hpo.HPO_DECISION
    assert receipt["space"]["dials"]  # sorted dial inventory
    assert receipt["objective"]["primary"] == "dev_accuracy"


def test_stage_kernel_script_has_no_unreplaced_tokens(monkeypatch, tmp_path):
    receipt = _stage(monkeypatch, tmp_path, "postgresql://u:p@h/db")
    script = (Path(receipt["staged"]) / laya_hpo.HPO_CODE_FILE).read_text(
        encoding="utf-8")
    import re
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", script)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
