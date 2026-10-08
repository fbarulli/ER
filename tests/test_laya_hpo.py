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

    def suggest_int(self, name, lo, hi, log=False):
        self.calls.append(("int", name, lo, hi, log))
        return lo

    def suggest_float(self, name, lo, hi, log=False):
        self.calls.append(("float", name, lo, hi, log))
        return lo

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, tuple(choices)))
        return choices[0]

    def set_user_attr(self, key, value):
        self.attrs[key] = value

    @property
    def user_attrs(self):
        return self.attrs


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
        # core dials
        "early_stop_patience", "warmup_frac", "lr_scheduler", "encoder_lr",
        "head_lr", "weight_decay", "micro_batch", "grad_accum",
        "label_smoothing", "sigma_start", "sigma_end", "ema", "swa",
        "layer_decay",
        # optional control dials (default-OFF)
        "no_decay_bias_norm", "optim_state_dtype", "lr_scaling", "base_batch",
        "r_drop", "r_drop_alpha", "drop_path", "drop_path_rate",
        "drop_path_schedule", "dynamic_padding", "pad_to_multiple",
        "batch_size_ramp", "batch_ramp_start_frac", "batch_ramp_epochs",
        "loss_schedule", "w_sph_end", "w_rps_end", "contrastive_margin",
        "contrastive_margin_end",
    }
    assert len(space["dials"]) == 33
    assert space["model_key"] == "laya"
    assert space["objective"] == {
        "direction": "maximize", "primary": "dev_accuracy",
        "secondary": "dev_loss", "forbidden": "test",
    }
    assert space["n_trials"] > 0 and space["n_jobs"] >= 1


def test_optional_control_dials_target_the_laya_control_block():
    dials = laya_hpo.load_space()["dials"]
    optional = [
        "no_decay_bias_norm", "optim_state_dtype", "lr_scaling", "base_batch",
        "r_drop", "r_drop_alpha", "drop_path", "drop_path_rate",
        "drop_path_schedule", "dynamic_padding", "pad_to_multiple",
        "batch_size_ramp", "batch_ramp_start_frac", "batch_ramp_epochs",
        "loss_schedule", "w_sph_end", "w_rps_end", "contrastive_margin",
        "contrastive_margin_end",
    ]
    from core.laya_config import FinetuneSpec
    assert len(optional) == 19
    for name in optional:
        assert dials[name]["target"] == "control"
        assert name in FinetuneSpec.model_fields  # SSOT field
    assert dials["base_batch"]["type"] == "int" and dials["base_batch"]["log"]
    assert dials["optim_state_dtype"]["choices"] == ["fp32", "bf16"]
    assert dials["lr_scaling"]["choices"] == ["none", "linear", "sqrt"]
    assert dials["loss_schedule"]["choices"] == [
        "constant", "linear", "cosine", "laya"]


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


def test_validate_space_rejects_a_bad_gate():
    space = json.loads(json.dumps(laya_hpo.load_space()))
    space["dials"]["r_drop_alpha"]["when"] = {"dial": "no_such_gate"}
    with pytest.raises(ValueError, match="unknown dial"):
        laya_hpo.validate_space(space)

    space = json.loads(json.dumps(laya_hpo.load_space()))
    space["dials"]["r_drop_alpha"]["when"] = {"dial": "encoder_lr",
                                              "equals": True}
    with pytest.raises(ValueError, match="categorical"):
        laya_hpo.validate_space(space)

    space = json.loads(json.dumps(laya_hpo.load_space()))
    space["dials"]["r_drop_alpha"]["when"] = {"dial": "r_drop",
                                              "equals": "maybe"}
    with pytest.raises(ValueError, match="not a choice"):
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
    # FakeTrial's categoricals are False for the gates, so gate dependents are
    # skipped; every non-gated dial is sampled.
    assert set(sampled).issubset(set(space["dials"]))
    assert {"r_drop_alpha", "drop_path_rate", "batch_ramp_epochs",
            "drop_path_schedule", "batch_ramp_start_frac"}.isdisjoint(sampled)
    by_name = {call[1]: call for call in trial.calls}
    assert by_name["early_stop_patience"][0] == "int"
    assert by_name["encoder_lr"][0] == "float" and by_name["encoder_lr"][4] is True
    assert by_name["weight_decay"][0] == "float" and by_name["weight_decay"][4] is False
    assert by_name["lr_scheduler"][0] == "categorical"
    assert by_name["micro_batch"][2] == (4, 8, 16)
    assert by_name["base_batch"][0] == "int" and by_name["base_batch"][4] is True


class _GateTrial:
    """A trial whose gate categoricals return configured values."""

    def __init__(self, gates):
        self.gates = dict(gates)

    def suggest_int(self, name, lo, hi, log=False):
        return lo

    def suggest_float(self, name, lo, hi, log=False):
        return lo

    def suggest_categorical(self, name, choices):
        return self.gates.get(name, choices[0])


def test_gated_optional_dials_are_skipped_when_the_gate_is_off():
    space = laya_hpo.load_space()
    trial = _GateTrial({"r_drop": False, "drop_path": False,
                        "batch_size_ramp": False})
    sampled = laya_hpo.sample_dials(trial, space)
    assert sampled["r_drop"] is False and "r_drop_alpha" not in sampled
    assert sampled["drop_path"] is False and "drop_path_rate" not in sampled
    assert sampled["batch_size_ramp"] is False
    assert "batch_ramp_epochs" not in sampled
    # Non-gated optional dials are always sampled.
    assert "no_decay_bias_norm" in sampled and "loss_schedule" in sampled


def test_gated_optional_dials_are_sampled_when_the_gate_is_on():
    space = laya_hpo.load_space()
    trial = _GateTrial({"r_drop": True, "drop_path": True,
                        "batch_size_ramp": True})
    sampled = laya_hpo.sample_dials(trial, space)
    assert {"r_drop_alpha", "drop_path_rate", "drop_path_schedule",
            "batch_ramp_start_frac", "batch_ramp_epochs"}.issubset(sampled)


def test_off_gate_preserves_the_base_control_default():
    space = laya_hpo.load_space()
    base_control = {name: "SSOT-DEFAULT" for name, spec in space["dials"].items()
                    if spec["target"] == "control"}
    trial = _GateTrial({"r_drop": False, "drop_path": False,
                        "batch_size_ramp": False})
    sampled = laya_hpo.sample_dials(trial, space)
    _, control = laya_hpo.apply_dials({}, base_control, sampled, space)
    # An off gate leaves its dependents at the SSOT default (default-OFF).
    assert control["r_drop_alpha"] == "SSOT-DEFAULT"
    assert control["batch_ramp_epochs"] == "SSOT-DEFAULT"
    assert control["drop_path_rate"] == "SSOT-DEFAULT"
    assert control["r_drop"] is False  # the gate itself is sampled


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
def test_objective_value_leases_fences_and_records_candidate():
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
    assert trial.attrs["hpo_lease_epoch"] == 3
    assert leases.issued[0][:3] == ("gen-1", "laya", 7)
    assert leases.asserted and not leases.revoked
    # F3: NOT promoted before Optuna commits the trial.
    assert champions.promotions == []
    promoted = laya_hpo_runtime.promote_committed_trial(
        champions, trial, generation_id="gen-1", model_key="laya")
    assert promoted is not None
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


def test_objective_value_fences_a_stale_lease_before_recording():
    from training.hpo_control_plane import HpoInfrastructureError

    trial = FakeTrial(number=5)
    leases = FakeLeaseStore(stale=True)
    champions = FakeChampionStore()
    with pytest.raises(HpoInfrastructureError, match="stale"):
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


def test_per_worker_budget_splits_the_remaining_trials():
    assert laya_hpo_runtime.per_worker_budget(24, 2) == 12
    assert laya_hpo_runtime.per_worker_budget(25, 2) == 13  # ceiling
    assert laya_hpo_runtime.per_worker_budget(0, 4) == 0
    assert laya_hpo_runtime.per_worker_budget(24, 0) == 24  # guards /0


def test_objective_value_multi_objective_returns_tuple_and_records_primary():
    trial = FakeTrial(number=9)
    leases = FakeLeaseStore()
    champions = FakeChampionStore()
    mode = SimpleNamespace(multi=True, secondary="epoch_time_s",
                           value=lambda metrics: (metrics["dev_accuracy"],
                                                  metrics["epoch_time_s"]))
    value = laya_hpo_runtime.objective_value(
        trial, lambda t: (0.7, 0.2, Path("/ck"), 12.5),
        leases, champions, "g", "laya", objective_mode=mode)
    assert value == (0.7, 12.5)
    assert champions.promotions == []  # promotion is post-commit
    laya_hpo_runtime.promote_committed_trial(
        champions, trial, generation_id="g", model_key="laya")
    assert champions.promotions[0]["value"] == 0.7  # primary only


def test_fidelity_reporter_reports_per_epoch_and_prunes():
    class _TrialPruned(Exception):
        pass

    class _Trial:
        def __init__(self):
            self.reports = []

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return len(self.reports) >= 2

    calls = {"n": 0}

    def dev_metrics(*args, **kwargs):
        calls["n"] += 1
        return {"accuracy": 0.5 + 0.1 * calls["n"]}

    namespace = {"_dev_metrics": dev_metrics}
    trial = _Trial()
    reporter = laya_hpo_runtime.FidelityReporter(
        trial, SimpleNamespace(TrialPruned=_TrialPruned), namespace)
    reporter.install()
    with pytest.raises(_TrialPruned):
        namespace["_dev_metrics"]()  # step 0 (no prune)
        namespace["_dev_metrics"]()  # step 1 -> prune
    assert [step for _, step in trial.reports] == [0, 1]
    reporter.uninstall()
    assert namespace["_dev_metrics"] is dev_metrics


# ── staged kernel composition + secret hygiene ─────────────────────────────
def test_hpo_runtime_source_has_no_future_import_and_carries_primitives():
    source = laya_hpo.hpo_runtime_source()
    assert "from __future__ import" not in source
    for symbol in ("def storage_from_environment", "class TrialLeaseStore",
                   "class ChampionStore", "def objective_value",
                   "def generation_study_name", "def fail_stale_trials",
                   "class WorkLedger", "class BudgetCounter",
                   "class ReservedTrialLoop", "class HpoInfrastructureError",
                   "def ensure_tables", "def promote_committed_trial",
                   "def resolve_champion_artifact"):
        assert symbol in source, symbol


def test_assert_secret_absent_raises_on_leak():
    secret = "postgresql://u:p@host:5432/db"
    with pytest.raises(RuntimeError, match="refusing to persist"):
        laya_hpo.assert_secret_absent({"nested": {"url": secret}}, secret)
    laya_hpo.assert_secret_absent({"ok": True}, secret)  # no raise


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


def _stage_colab(monkeypatch, tmp_path, url):
    monkeypatch.setenv(laya_hpo.GENERATION_ID_ENV, "gen-colab-1")
    # Reuse of cli.colab_runtime._optuna_env_script reads os.environ too.
    monkeypatch.setenv(laya_hpo.OPTUNA_URL_ENV, url)
    monkeypatch.setattr(laya_lane, "staging_dir", lambda: Path(tmp_path))
    monkeypatch.setattr(laya_lane, "_git_revision", lambda: "b" * 40)
    monkeypatch.setattr(laya_lane, "_env_value",
                        lambda name: url if name == laya_hpo.OPTUNA_URL_ENV
                        else None)
    monkeypatch.setattr(laya_lane, "_log_lane", lambda line: None)
    monkeypatch.setattr(laya_lane, "stage_finetune_dataset_payload",
                        lambda **kwargs: {"payload": "x", "files": {}})
    monkeypatch.setattr(laya_hpo, "_current_git_branch", lambda: "laya-hpo")
    from core import runtime_inputs
    monkeypatch.setattr(runtime_inputs, "require_published_tip_match",
                        lambda rev, repo, branch: rev)
    return laya_hpo.stage_laya_hpo_colab()


def test_stage_colab_writes_entry_without_the_secret(monkeypatch, tmp_path):
    secret = "postgresql://hpo_user:s3cret@db.example.com:5432/optuna"
    receipt = _stage_colab(monkeypatch, tmp_path, secret)
    stage = Path(receipt["staged"])
    script = (stage / laya_hpo.HPO_CODE_FILE).read_text(encoding="utf-8")
    entry = (stage / laya_hpo.COLAB_ENTRY_FILE).read_text(encoding="utf-8")
    compile(script, str(stage / laya_hpo.HPO_CODE_FILE), "exec")
    compile(entry, str(stage / laya_hpo.COLAB_ENTRY_FILE), "exec")
    assert secret in script and secret not in entry
    assert receipt["lane"] == "colab"
    assert receipt["working"].startswith("/content")
    assert {"laya", "text", "gnn", "cascade"}.issubset(receipt["registry"])
    assert "minilm_l6" in receipt["registry"]  # SSOT-sourced key
    assert secret not in json.dumps(receipt)
    assert not (stage / "kernel-metadata.json").exists()  # colab is not a kernel


def test_cli_lane_dispatch_registry_is_complete():
    assert set(laya_hpo.STAGE_DISPATCH) == set(laya_hpo.LANES)
    assert laya_hpo.STAGE_DISPATCH["kaggle"] == "stage_laya_hpo_kernel"
    assert laya_hpo.STAGE_DISPATCH["colab"] == "stage_laya_hpo_colab"


def test_kernel_slug_dispatch_resolves_the_hpo_kind():
    laya_hpo.register_dispatch()
    slug = laya_hpo.kernel_slug("laya-hpo")
    assert slug.endswith("er-laya-hpo")
    assert laya_lane.kernel_slug("laya-hpo") == slug


def test_space_is_bound_in_the_paths_ssot():
    from core.common import F

    assert "laya_hpo_space" in F
    assert str(F["laya_hpo_space"]).endswith("config/laya_hpo_space.yaml")


def test_console_script_is_declared():
    import re

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8")
    assert re.search(r'^er-hpo\s*=\s*"cli\.laya_hpo:main"', text, re.MULTILINE)


def test_stage_kernel_receipt_carries_registry_and_observability(monkeypatch,
                                                                tmp_path):
    receipt = _stage(monkeypatch, tmp_path, "postgresql://u:p@h/db")
    assert {"laya", "text", "gnn", "cascade"}.issubset(receipt["registry"])
    assert "minilm_l6" in receipt["registry"]  # SSOT-sourced key
    observability = receipt["observability"]
    assert observability["mode"] == "postgres"
    assert observability["events"].endswith("trial_events.jsonl")
    assert observability["mirror"].endswith("study_mirror.jsonl")
    assert observability["ledger"].endswith("hpo_trials.jsonl")


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


def test_stage_kernel_embeds_profiler_harness(monkeypatch, tmp_path):
    receipt = _stage(monkeypatch, tmp_path, "postgresql://u:p@h/db")
    script = (Path(receipt["staged"]) / laya_hpo.HPO_CODE_FILE).read_text(
        encoding="utf-8")
    for symbol in ("class TrialProfiler", "def install_phase_hooks",
                   "def profiler_schedule", "def wandb_log_profiler",
                   "PHASE_OPTIMIZER_STEP"):
        assert symbol in script, symbol
    assert receipt["profiler"]["top_ops"] == 15


def test_stage_kernel_receipt_carries_the_option_set(monkeypatch, tmp_path):
    receipt = _stage(monkeypatch, tmp_path, "postgresql://u:p@h/db")
    assert receipt["options"]["scheduler"]["mode"] == "slots"
    assert receipt["options"]["sampler"]["kind"] == "tpe"
    assert receipt["options"]["pruner"]["kind"] == "hyperband"
    assert receipt["options"]["warm_start"]["mode"] == "base"
    assert receipt["options"]["shared_data"]["tokenized"] is True
    assert receipt["options"]["session"]["processes_only"] is True


# ── per-trial profiler coverage (stubbed torch, no GPU) ────────────────────
_RECORDED: list[str] = []


class _FakeRecordFunction:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _RECORDED.append(self.name)
        return self

    def __exit__(self, *exc):
        return False


class _FakeProfilerSession:
    def __init__(self):
        self.started = False
        self.steps = 0
        self.stopped = False
        self.exports: list[str] = []

    def start(self):
        self.started = True

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        self.exports.append(path)
        Path(path).write_text("{}", encoding="utf-8")

    def key_averages(self):
        return SimpleNamespace(table=lambda **kwargs: "TOP OPS")


class _FakeProfilerModule:
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"

    record_function = _FakeRecordFunction

    def __init__(self, *, fail: bool = False):
        self.session = _FakeProfilerSession()
        self.fail = fail
        self.calls = 0
        self.schedule_kwargs = None
        self.profile_kwargs = None

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return "SCHEDULE"

    def profile(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("profiler unavailable")
        self.profile_kwargs = kwargs
        return self.session


class _FakeTensor:
    def backward(self):
        return None


class _FakeOptimizer:
    def step(self):
        return None


class _FakeTorch:
    def __init__(self, *, fail: bool = False):
        self.Tensor = _FakeTensor
        self.profiler = _FakeProfilerModule(fail=fail)


class _FakeLayaTrain:
    def encode_item(self):
        return 1

    def soft_ce_loss(self):
        return 2

    def rlcd_loss(self):
        return 3

    def fit_temperature_map(self):
        return 4


def _fake_namespace():
    return {
        "_forward_dtype": lambda: 5,
        "_dev_metrics": lambda: 6,
        "_save_control_checkpoint": lambda: 7,
        "_make_optimizer": lambda *args, **kwargs: _FakeOptimizer(),
    }


def test_space_declares_profiler_block():
    profiler = laya_hpo.load_space()["profiler"]
    assert profiler["enabled"] is True
    assert profiler["top_ops"] == 15
    assert set(profiler) == {"enabled", "wait", "warmup", "active", "repeat",
                             "top_ops"}


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["profiler"].update(enabled="yes"), "enabled"),
    (lambda s: s["profiler"].update(active=0), "active"),
    (lambda s: s["profiler"].update(top_ops=0), "top_ops"),
    (lambda s: s["profiler"].update(wait=-1), "wait"),
])
def test_validate_space_rejects_bad_profiler(mutate, message):
    space = json.loads(json.dumps(laya_hpo.load_space()))
    mutate(space)
    with pytest.raises(ValueError, match=message):
        laya_hpo.validate_space(space)


def test_profiler_annotates_every_phase_without_gpu(tmp_path):
    """The hook wraps each major phase with record_function and drives the
    bounded schedule from the optimizer step — with a stubbed torch, no GPU."""
    _RECORDED.clear()
    torch_module = _FakeTorch()
    laya = _FakeLayaTrain()
    namespace = _fake_namespace()
    trace = tmp_path / "ckpt" / "profiler" / "trial_0.json"
    original_backward = torch_module.Tensor.backward
    original_forward = namespace["_forward_dtype"]
    profiler = laya_hpo_runtime.TrialProfiler(
        torch_module=torch_module,
        config={"enabled": True, "wait": 1, "warmup": 1, "active": 2,
                "repeat": 1, "top_ops": 5},
        trace_path=trace, device_type="cuda", laya_train=laya,
        namespace=namespace, logger=lambda line: None, rank0=True)
    with profiler:
        laya.encode_item()
        namespace["_forward_dtype"]()
        laya.soft_ce_loss()
        torch_module.Tensor().backward()
        namespace["_make_optimizer"]().step()
        namespace["_dev_metrics"]()
        namespace["_save_control_checkpoint"]()
        laya.fit_temperature_map()

    assert set(_RECORDED) >= {
        "data.encode", "forward", "loss", "backward", "optimizer_step",
        "dev_eval", "checkpoint_save", "calibration"}
    assert torch_module.profiler.schedule_kwargs == {
        "wait": 1, "warmup": 1, "active": 2, "repeat": 1}
    assert torch_module.profiler.profile_kwargs["profile_memory"] is True
    assert torch_module.profiler.profile_kwargs["with_stack"] is False
    assert torch_module.profiler.session.started
    assert torch_module.profiler.session.steps >= 1   # advanced per optimizer step
    assert torch_module.profiler.session.stopped
    assert trace.is_file()
    # Hooks are restored after the trial (no global leakage).
    assert torch_module.Tensor.backward is original_backward
    assert namespace["_forward_dtype"] is original_forward


def test_profiler_fail_soft_when_setup_raises(tmp_path):
    torch_module = _FakeTorch(fail=True)
    profiler = laya_hpo_runtime.TrialProfiler(
        torch_module=torch_module,
        config={"enabled": True, "wait": 1, "warmup": 1, "active": 1,
                "repeat": 1, "top_ops": 5},
        trace_path=tmp_path / "p" / "trial_0.json", device_type="cuda",
        laya_train=_FakeLayaTrain(), namespace=_fake_namespace(),
        logger=lambda line: None, rank0=True)
    with profiler:
        pass  # a profiler error must never escape
    assert profiler.enabled is False
    assert torch_module.profiler.session.started is False


@pytest.mark.parametrize("device_type, config", [
    ("cpu", {"enabled": True}),          # auto-disable on CPU
    ("cuda", {"enabled": False}),        # config knob off
])
def test_profiler_disabled_never_starts(tmp_path, device_type, config):
    torch_module = _FakeTorch()
    profiler = laya_hpo_runtime.TrialProfiler(
        torch_module=torch_module, config=config,
        trace_path=tmp_path / "p" / "trial_0.json", device_type=device_type,
        laya_train=_FakeLayaTrain(), namespace=_fake_namespace(),
        logger=lambda line: None, rank0=True)
    with profiler:
        pass
    assert profiler.enabled is False
    assert torch_module.profiler.calls == 0


def test_profiler_disabled_for_non_rank0(tmp_path):
    torch_module = _FakeTorch()
    profiler = laya_hpo_runtime.TrialProfiler(
        torch_module=torch_module,
        config={"enabled": True, "wait": 1, "warmup": 1, "active": 1,
                "repeat": 1, "top_ops": 5},
        trace_path=tmp_path / "p" / "trial_0.json", device_type="cuda",
        laya_train=_FakeLayaTrain(), namespace=_fake_namespace(),
        logger=lambda line: None, rank0=False)
    with profiler:
        pass
    assert profiler.enabled is False
    assert torch_module.profiler.calls == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
