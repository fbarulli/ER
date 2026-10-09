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
def test_resource_caps_env_and_apply_torch(monkeypatch):
    caps = opt.ResourceCaps({"omp_threads": 3, "torch_threads": 2,
                             "cuda_alloc_fraction": 0.4})
    env = caps.env()
    assert env["OMP_NUM_THREADS"] == "3" and env["MKL_NUM_THREADS"] == "3"
    # No dead knob: the dataloader env var is gone (nothing read it).
    assert "ER_LAYA_DATALOADER_WORKERS" not in env
    recorded = {}
    torch_stub = SimpleNamespace(
        set_num_threads=lambda n: recorded.update(threads=n),
        cuda=SimpleNamespace(is_available=lambda: True,
                             set_per_process_memory_fraction=lambda f: recorded.update(fraction=f)))
    caps.apply_torch(torch_stub)
    assert recorded == {"threads": 2, "fraction": 0.4}
    # A per-slot override (lone worker on a GPU) wins over the raw config.
    monkeypatch.setenv(opt.ResourceCaps.CUDA_FRACTION_ENV, "1.0")
    recorded.clear()
    caps.apply_torch(torch_stub)
    assert recorded == {"threads": 2, "fraction": 1.0}


def test_resource_caps_split_fraction_across_gpu_slots():
    caps = opt.ResourceCaps({"cuda_alloc_fraction": 1.0})
    assert caps.per_process_cuda_fraction(1) == 1.0
    assert caps.per_process_cuda_fraction(2) == 0.5
    assert caps.per_process_cuda_fraction(4) == 0.25
    off = opt.ResourceCaps({"cuda_alloc_fraction": 0.0})
    assert off.per_process_cuda_fraction(2) == 0.0


def test_worker_pool_emits_per_process_cuda_fraction():
    pool = opt.WorkerPool(
        slots_per_gpu=2, max_concurrent_trials=4,
        resource_caps=opt.ResourceCaps({"cuda_alloc_fraction": 1.0}))
    specs = pool.plan(gpu_count=2)
    assert all(s.env[opt.ResourceCaps.CUDA_FRACTION_ENV] == "0.5"
               for s in specs)
    lone = opt.WorkerPool(
        slots_per_gpu=1, max_concurrent_trials=1,
        resource_caps=opt.ResourceCaps({"cuda_alloc_fraction": 1.0})).plan(1)
    assert lone[0].env[opt.ResourceCaps.CUDA_FRACTION_ENV] == "1.0"


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


def test_default_plan_uses_every_visible_gpu_one_process_each():
    """Default space: one worker PROCESS per visible GPU (true parallelism)."""
    options = opt.build_option_set(laya_hpo.load_space())
    two = options.scheduler.workers(2)
    assert len(two) == 2 and {s.device_index for s in two} == {0, 1}
    assert all("CUDA_MPS_PIPE_DIRECTORY" not in s.env for s in two)
    one = options.scheduler.workers(1)
    assert len(one) == 1 and one[0].device_index == 0


def test_worker_pool_plans_two_mps_slots_sharing_one_gpu():
    """One T4, k=2 slots: two processes on device 0, each capped at half + MPS."""
    pool = opt.WorkerPool(
        slots_per_gpu=2, max_concurrent_trials=2,
        resource_caps=opt.ResourceCaps({"cuda_alloc_fraction": 1.0}),
        mps=opt.MpsController(True))
    specs = pool.plan(gpu_count=1)
    assert len(specs) == 2
    assert {s.device_index for s in specs} == {0}
    assert all(s.env[opt.ResourceCaps.CUDA_FRACTION_ENV] == "0.5"
               for s in specs)
    assert all(s.env["CUDA_MPS_PIPE_DIRECTORY"] for s in specs)


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
    # master_port is a ``ddp`` sub-key (SSOT), not a top-level option: reading
    # it from the wrong level silently ignored the configured port.
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 2,
                                         "master_port": 29511}})
    argv = ddp.torchrun_argv("laya_hpo.py", ["--trial", "3"])
    assert argv[0] == "torchrun"
    assert argv[argv.index("--nproc_per_node") + 1] == "2"
    assert argv[argv.index("--master_port") + 1] == "29511"
    assert argv[-2:] == ["--trial", "3"]


def test_ddp_nproc_clamped_to_devices_and_free_port():
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 8,
                                         "master_port": 0}})
    assert len(ddp.workers(1)) == 1          # one controller, as always
    argv = ddp.torchrun_argv("s.py")
    assert argv[argv.index("--nproc_per_node") + 1] == "1"  # clamped to 1 GPU
    port = int(argv[argv.index("--master_port") + 1])
    assert 1 <= port <= 65535
    ddp.workers(2)
    assert ddp.torchrun_argv("s.py")[
        ddp.torchrun_argv("s.py").index("--nproc_per_node") + 1] == "2"


class _FakeProcess:
    """Minimal Popen stand-in for the streaming runner tests."""

    def __init__(self, *, exit_code=0):
        self._done = exit_code is not None
        self.exit_code = exit_code
        self.terminated = 0

    def poll(self):
        return self.exit_code if self._done else None

    def terminate(self):
        self.terminated += 1
        self._done = True

    def kill(self):
        self._done = True

    def wait(self, timeout=None):
        self._done = True
        return self.exit_code


def test_ddp_trial_runner_launches_torchrun_and_reads_rank0_result(tmp_path):
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 2}})
    calls = []

    def popener(argv, env=None):
        calls.append((argv, env))
        (tmp_path / "ddp_trial_3.json").write_text(
            json.dumps({"accuracy": 0.88, "dev_loss": 0.2,
                        "checkpoint": "/ck", "epoch_time_s": 5.0}),
            encoding="utf-8")
        return _FakeProcess()

    trial_runner = opt.DdpTrialRunner(
        ddp, "/kaggle/working/laya_hpo.py", popener=popener,
        result_dir=tmp_path, logger=lambda line: None, poll_interval=0.0)
    result = trial_runner.run(3, {"dials": {"encoder_lr": 1.0e-5}})
    assert result == (0.88, 0.2, "/ck", 5.0)
    argv, env = calls[0]
    assert argv[0] == "torchrun" and argv[-2:] == ["--ddp-trial", "3"]
    assert env[opt.DdpTrialRunner.DDP_TRIAL_ENV] == "3"
    assert env[ddp.BACKEND_ENV] == "nccl"
    assert json.loads(env[opt.DdpTrialRunner.DDP_PAYLOAD_ENV])["dials"][
        "encoder_lr"] == 1.0e-5


def test_ddp_trial_runner_streams_epochs_and_terminates_on_prune(tmp_path):
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 1}})
    seen = []

    def popener(argv, env=None):
        (tmp_path / "ddp_trial_5.epochs.jsonl").write_text(
            json.dumps({"step": 0, "value": 0.5}) + "\n", encoding="utf-8")
        return process

    process = _FakeProcess(exit_code=None)
    trial_runner = opt.DdpTrialRunner(
        ddp, "s.py", popener=popener, result_dir=tmp_path, poll_interval=0.0)
    with pytest.raises(opt.TrialPrunedSignal):
        trial_runner.run(5, {"dials": {}},
                         on_epoch=lambda step, value: seen.append((step, value)),
                         should_stop=lambda: True)
    assert seen == [(0, 0.5)]
    assert process.terminated == 1


def test_ddp_trial_runner_fails_loud_on_nonzero_exit(tmp_path):
    ddp = opt.DdpTrialScheduler({"parallelism": "ddp",
                                 "ddp": {"nproc_per_node": 1}})
    trial_runner = opt.DdpTrialRunner(
        ddp, "s.py", popener=lambda argv, env=None: _FakeProcess(exit_code=1),
        result_dir=tmp_path, poll_interval=0.0)
    with pytest.raises(RuntimeError, match="failed"):
        trial_runner.run(7, {"dials": {}})


# ── B) sampler / pruner / fidelity / objective ─────────────────────────────
def test_sampler_factory_registry_covers_every_kind():
    _Recording.calls.clear()
    factory = opt.SamplerFactory("tpe", seed=7)
    assert isinstance(factory.create(_fake_optuna()), _Recording)
    assert _Recording.calls[-1][1] == {"seed": 7, "multivariate": True,
                                       "constant_liar": True}
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
                       "max_resource": 8,
                       "reduction_factor": 3}).create(_fake_optuna())
    assert _Recording.calls[-1] == ("HyperbandPruner", {
        "min_resource": 1, "max_resource": 8, "reduction_factor": 3})
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


# ── C) warm start / ensembling ─────────────────────────────────────────────
def test_shared_data_cache_family_is_removed():
    """The advertised-but-absent caches must not come back."""
    assert not hasattr(opt, "SharedDataCache")


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
def test_build_option_set_reads_the_config_block():
    space = laya_hpo.load_space()
    options = opt.build_option_set(space, base_model="/base")
    assert isinstance(options, opt.OptionSet)
    assert options.scheduler.mode == "slots"
    assert options.sampler.kind == "tpe"
    assert options.sampler.multivariate is True
    assert options.sampler.constant_liar is True
    assert options.pruner.kind == "none"  # baseline must complete unpruned
    assert options.objective_mode.multi is False
    assert options.warm_start.mode == "base"
    assert options.session.processes_only is True
    assert options.session.cuda_visible_devices == ""
    assert options.resource_caps.cuda_alloc_fraction == 1.0
    payload = options.as_dict()
    assert payload["scheduler"]["mode"] == "slots"
    assert "core_allocator" not in payload
    assert "shared_data" not in payload
    json.dumps(payload)  # JSON-serializable


def test_build_option_set_reads_ddp_backend_and_master_port():
    # Both ddp.backend (applied to init_distributed) and ddp.master_port (passed
    # to torchrun) are nested SSOT keys; the option set must surface them.
    space = {"options": {"parallelism": "ddp",
                         "ddp": {"nproc_per_node": 2, "backend": "gloo",
                                 "master_port": 29600}},
             "seed": 1}
    options = opt.build_option_set(space)
    scheduler = options.scheduler
    assert isinstance(scheduler, opt.DdpTrialScheduler)
    assert scheduler.backend == "gloo"
    assert scheduler.master_port == 29600
    assert scheduler.as_dict()["backend"] == "gloo"
    assert scheduler.as_dict()["master_port"] == 29600
    assert "29600" in scheduler.torchrun_argv("s.py")


def test_space_declares_the_options_block():
    options = laya_hpo.load_space()["options"]
    assert options["parallelism"] in opt.PARALLELISM_MODES
    assert options["sampler"]["kind"] in opt.SAMPLER_KINDS
    assert options["pruner"]["kind"] in opt.PRUNER_KINDS
    assert options["warm_start"]["mode"] in opt.WARM_START_MODES
    assert "shared_data" not in options        # no dead cache toggles
    assert "dataloader_workers" not in options["resources"]


def test_validate_space_rejects_bad_option_values():
    space = json.loads(json.dumps(laya_hpo.load_space()))
    space["options"]["parallelism"] = "bogus"
    with pytest.raises(ValueError, match="parallelism"):
        laya_hpo.validate_space(space)


# ── Optuna best practices (threads/pruner/sampler/warm-start/session) ──────
def test_threads_per_worker_defaults_every_blas_pool():
    caps = opt.ResourceCaps({})  # threads_per_worker default 1
    env = caps.env()
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    recorded = {}
    caps.apply_torch(SimpleNamespace(
        set_num_threads=lambda n: recorded.update(t=n),
        cuda=SimpleNamespace(is_available=lambda: False)))
    assert recorded == {"t": 1}


def test_session_policy_processes_only_and_timeout():
    policy = opt.SessionPolicy({})
    assert policy.optimize_kwargs() == {"n_jobs": 1, "catch": (Exception,)}
    assert policy.study_kwargs() == {"load_if_exists": True}
    assert policy.processes_only is True
    timed = opt.SessionPolicy({"timeout_s": 5400, "load_if_exists": True})
    assert timed.optimize_kwargs()["timeout"] == 5400
    assert timed.optimize_kwargs()["n_jobs"] == 1


def test_sampler_tpe_passes_multivariate_and_constant_liar():
    _Recording.calls.clear()
    opt.SamplerFactory("tpe", seed=3, multivariate=True,
                       constant_liar=True).create(_fake_optuna())
    assert _Recording.calls[-1] == (
        "TPESampler", {"seed": 3, "multivariate": True,
                       "constant_liar": True})


def test_warm_start_enqueued_trials_are_defensively_copied():
    policy = opt.WarmStartPolicy(
        "base", base_model="/b",
        enqueue=[{"encoder_lr": 1.0e-5}, {"head_lr": 1.0e-4}])
    seeds = policy.enqueued_trials()
    assert seeds == [{"encoder_lr": 1.0e-5}, {"head_lr": 1.0e-4}]
    seeds[0]["encoder_lr"] = 9.0
    assert policy.enqueued_trials()[0]["encoder_lr"] == 1.0e-5


def test_space_pruner_is_off_to_protect_the_baseline():
    # The seeded baseline must run to completion; `none` disables pruning so it
    # can never be cut mid-run (the documented tradeoff: no prune savings).
    assert laya_hpo.load_space()["options"]["pruner"]["kind"] == "none"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
