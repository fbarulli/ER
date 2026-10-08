"""Offline public-API tests for the laya HPO option components.

One test module per option family (A/B/C). Every external dependency (optuna,
torch, subprocess) is stubbed, so these run without a GPU, PostgreSQL or an
Optuna install.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from cli import laya_hpo
from training import laya_hpo_options as opt


# ── fakes ──────────────────────────────────────────────────────────────────
class _Recording:
    """A callable class that records its kwargs and returns itself."""
    calls: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _Recording.calls.append((type(self).__name__, kwargs))


def _fake_optuna():
    def make(name):
        return type(name, (_Recording,), {})

    samplers = SimpleNamespace(
        TPESampler=make("TPESampler"), CmaEsSampler=make("CmaEsSampler"),
        RandomSampler=make("RandomSampler"), GPSampler=make("GPSampler"),
        QMCSampler=make("QMCSampler"))
    pruners = SimpleNamespace(
        NopPruner=make("NopPruner"), MedianPruner=make("MedianPruner"),
        SuccessiveHalvingPruner=make("SuccessiveHalvingPruner"),
        HyperbandPruner=make("HyperbandPruner"))
    return SimpleNamespace(samplers=samplers, pruners=pruners)


# ── A) parallelism / resource caps ─────────────────────────────────────────
def test_resource_caps_env_and_apply_torch():
    caps = opt.ResourceCaps({"omp_threads": 3, "torch_threads": 2,
                             "dataloader_workers": 1, "cuda_alloc_fraction": 0.4})
    env = caps.env()
    assert env["OMP_NUM_THREADS"] == "3" and env["MKL_NUM_THREADS"] == "3"
    assert env["ER_LAYA_DATALOADER_WORKERS"] == "1"
    recorded = {}

    def set_num_threads(n):
        recorded["threads"] = n

    torch_stub = SimpleNamespace(
        set_num_threads=set_num_threads,
        cuda=SimpleNamespace(is_available=lambda: True,
                             set_per_process_memory_fraction=lambda f: recorded.update(fraction=f)))
    caps.apply_torch(torch_stub)
    assert recorded == {"threads": 2, "fraction": 0.4}


def test_mps_controller_commands_and_lifecycle():
    calls = []
    mps = opt.MpsController(True, runner=lambda argv, check=False: calls.append(argv),
                            logger=lambda line: None)
    assert mps.env()["CUDA_MPS_PIPE_DIRECTORY"]
    assert mps.commands()["start"][0] == "nvidia-cuda-mps-control"
    assert mps.start() is True and mps.stop() is True
    assert len(calls) == 2
    off = opt.MpsController(False)
    assert off.env() == {} and off.start() is False


def test_worker_pool_plan_caps_total_and_pins_devices():
    pool = opt.WorkerPool(slots_per_gpu=2, max_concurrent_trials=3,
                          resource_caps=opt.ResourceCaps({"omp_threads": 1}))
    specs = pool.plan(gpu_count=2)
    assert len(specs) == 3  # capped below 2*2
    assert {s.device_index for s in specs} == {0, 1}
    assert all(s.env["OMP_NUM_THREADS"] == "1" for s in specs)


def test_trial_scheduler_registry_selects_slots_and_ddp():
    slots = opt.TrialScheduler.create({"parallelism": "slots",
                                       "slots_per_gpu": 1,
                                       "max_concurrent_trials": 2})
    assert isinstance(slots, opt.SlotsTrialScheduler)
    assert len(slots.workers(2)) == 2
    ddp = opt.TrialScheduler.create({"parallelism": "ddp",
                                     "ddp": {"nproc_per_node": 2}})
    assert isinstance(ddp, opt.DdpTrialScheduler)
    assert len(ddp.workers(2)) == 1  # one controller, torchrun fans out
    with pytest.raises(ValueError, match="parallelism"):
        opt.TrialScheduler.create({"parallelism": "nope"})


def test_ddp_torchrun_argv_has_rendezvous():
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 2},
                                 "master_port": 29511})
    argv = ddp.torchrun_argv("laya_hpo.py", ["--trial", "3"])
    assert argv[0] == "torchrun"
    assert argv[argv.index("--nproc_per_node") + 1] == "2"
    assert "29511" in argv and argv[-2:] == ["--trial", "3"]


def test_ddp_trial_runner_launches_torchrun_and_reads_rank0_result(tmp_path):
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 2}})
    calls = []

    def runner(argv, env=None, check=False):
        calls.append((argv, env))
        (tmp_path / "ddp_trial_3.json").write_text(
            json.dumps({"accuracy": 0.88, "dev_loss": 0.2,
                        "checkpoint": "/ck", "epoch_time_s": 5.0}),
            encoding="utf-8")

    trial_runner = opt.DdpTrialRunner(
        ddp, "/kaggle/working/laya_hpo.py", runner=runner,
        result_dir=tmp_path, logger=lambda line: None)
    result = trial_runner.run(3, {"dials": {"encoder_lr": 1.0e-5}})
    assert result == (0.88, 0.2, "/ck", 5.0)
    argv, env = calls[0]
    assert argv[0] == "torchrun" and argv[-2:] == ["--ddp-trial", "3"]
    assert env[opt.DdpTrialRunner.DDP_TRIAL_ENV] == "3"
    assert json.loads(env[opt.DdpTrialRunner.DDP_PAYLOAD_ENV])["dials"][
        "encoder_lr"] == 1.0e-5


# ── B) sampler / pruner / fidelity / objective ─────────────────────────────
def test_sampler_factory_registry_covers_every_kind():
    _Recording.calls.clear()
    factory = opt.SamplerFactory("tpe", seed=7)
    assert isinstance(factory.create(_fake_optuna()), _Recording)
    assert _Recording.calls[-1][1] == {"seed": 7}
    for kind, cls in (("cmaes", "CmaEsSampler"), ("random", "RandomSampler"),
                      ("gp", "GPSampler"), ("qmc", "QMCSampler")):
        factory = opt.SamplerFactory(kind, seed=1)
        factory.create(_fake_optuna())
        assert _Recording.calls[-1][0] == cls
    with pytest.raises(ValueError, match="sampler"):
        opt.SamplerFactory("bogus", seed=1)


def test_pruner_factory_registry_covers_every_kind():
    _Recording.calls.clear()
    assert opt.PrunerFactory({"kind": "none"}).create(_fake_optuna()).kwargs == {}
    opt.PrunerFactory({"kind": "median", "n_startup_trials": 4,
                       "n_warmup_steps": 2}).create(_fake_optuna())
    assert _Recording.calls[-1] == ("MedianPruner", {"n_startup_trials": 4,
                                                     "n_warmup_steps": 2})
    opt.PrunerFactory({"kind": "successive_halving", "min_resource": 2,
                       "reduction_factor": 4}).create(_fake_optuna())
    assert _Recording.calls[-1][0] == "SuccessiveHalvingPruner"
    opt.PrunerFactory({"kind": "hyperband", "min_resource": 1,
                       "reduction_factor": 3,
                       "n_warmup_steps": 9}).create(_fake_optuna())
    assert _Recording.calls[-1][0] == "HyperbandPruner"
    with pytest.raises(ValueError, match="pruner"):
        opt.PrunerFactory({"kind": "bogus"})


def test_fidelity_schedule_geometric_stages_and_staged_unfreeze():
    fidelity = opt.FidelitySchedule({"enabled": True, "dimension": "epochs",
                                     "small": 1, "full": 9, "stages": 5},
                                    {"enabled": True, "head_only_epochs": 2})
    assert fidelity.stages_list() == [1, 3, 5, 7, 9]
    assert fidelity.resource(0) == 1 and fidelity.resource(4) == 9
    assert fidelity.is_full(4) and not fidelity.is_full(0)
    assert fidelity.encoder_frozen_epochs() == 2
    assert fidelity.stage(7) == 2
    with pytest.raises(ValueError, match="fidelity dimension"):
        opt.FidelitySchedule({"dimension": "bogus"})


def test_objective_mode_single_and_multi():
    single = opt.ObjectiveMode(False)
    assert single.directions() == "maximize"
    assert single.value({"dev_accuracy": 0.8}) == 0.8
    multi = opt.ObjectiveMode(True, "epoch_time_s")
    assert multi.directions() == ["maximize", "minimize"]
    assert multi.value({"dev_accuracy": 0.8, "epoch_time_s": 12.0}) == (0.8, 12.0)


# ── C) shared-data strategies ──────────────────────────────────────────────
def test_shared_data_cache_paths_and_manifest(tmp_path):
    cache = opt.SharedDataCache({"tokenized_cache": True,
                                 "embedding_cache": True,
                                 "dev_encoding_cache": False}, tmp_path)
    sig = {"corpus_sha256": "abc", "max_len": 512}
    token_path = cache.path("tokenized", sig)
    assert token_path.parent.name == "tokenized"
    assert not cache.is_ready("tokenized", sig)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_bytes(b"x")
    assert cache.is_ready("tokenized", sig)
    assert not cache.is_ready("dev", sig)  # disabled
    manifest = cache.manifest({"tokenized": sig})
    assert manifest["tokenized"]["ready"] is True
    assert manifest["tokenized"]["key"] == cache.key("tokenized", sig)
    # different signature -> different cache slot
    assert cache.key("tokenized", sig) != cache.key("tokenized", {"x": 1})


def test_warm_start_policy_modes():
    policy = opt.WarmStartPolicy("base", base_model="/models/base")
    assert policy.source() == "/models/base"
    assert opt.WarmStartPolicy("scratch", base_model="/x").source() is None
    champion = opt.WarmStartPolicy("champion", base_model="/x",
                                   champion_artifact="/ckpt/best")
    assert champion.source() == "/ckpt/best"
    fallback = opt.WarmStartPolicy("champion", base_model="/x")
    assert fallback.source() == "/x"
    with pytest.raises(ValueError, match="warm_start"):
        opt.WarmStartPolicy("bogus")


def test_trial_ensembler_select_and_average():
    trials = [SimpleNamespace(number=n, value=v)
              for n, v in [(0, 0.5), (1, 0.9), (2, 0.7), (3, None)]]
    ensembler = opt.TrialEnsembler(True, top_k=2, method="weights")
    assert [t.number for t in ensembler.select(trials)] == [1, 2]
    averaged = ensembler.average_weights([{"w": [1.0, 3.0]}, {"w": [3.0, 5.0]}])
    assert averaged == {"w": [2.0, 4.0]}
    predictions = ensembler.average_predictions([[0.0, 1.0], [1.0, 1.0]])
    assert predictions == [0.5, 1.0]
    assert opt.TrialEnsembler(False).select(trials) == []
    with pytest.raises(ValueError, match="ensemble method"):
        opt.TrialEnsembler(True, method="bogus")


# ── assembly / SSOT ────────────────────────────────────────────────────────
def test_build_option_set_reads_the_config_block(tmp_path):
    space = laya_hpo.load_space()
    options = opt.build_option_set(space, root=tmp_path, base_model="/base")
    assert isinstance(options, opt.OptionSet)
    assert options.scheduler.mode == "slots"
    assert options.sampler.kind == "tpe"
    assert options.pruner.kind == "none"
    assert options.objective_mode.multi is False
    assert options.warm_start.mode == "base"
    payload = options.as_dict()
    assert payload["scheduler"]["mode"] == "slots"
    json.dumps(payload)  # JSON-serializable


def test_space_declares_the_options_block():
    options = laya_hpo.load_space()["options"]
    assert options["parallelism"] in opt.PARALLELISM_MODES
    assert options["sampler"]["kind"] in opt.SAMPLER_KINDS
    assert options["pruner"]["kind"] in opt.PRUNER_KINDS
    assert options["warm_start"]["mode"] in opt.WARM_START_MODES
    assert set(options["shared_data"]) == {"tokenized_cache", "embedding_cache",
                                           "dev_encoding_cache"}


def test_validate_space_rejects_bad_option_values():
    space = json.loads(json.dumps(laya_hpo.load_space()))
    space["options"]["parallelism"] = "bogus"
    with pytest.raises(ValueError, match="parallelism"):
        laya_hpo.validate_space(space)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
