"""Tail of the embedded HPO Kaggle kernel-script text (cli.laya_hpo)."""
from __future__ import annotations

HPO_KERNEL_TEMPLATE_TAIL = '''\
class WorkerSession:
    """One remote worker process: open the study, run its trial budget, observe.

    Each private method does one job; ``run`` only wires them together.
    """

    def __init__(self, device):
        self.device = device
        self.optuna = None
        self.options = None
        self.resolver = None
        self.observer = None

    def _assert_laya(self):
        if MODEL_KEY == "laya":
            return
        # The registry carries a real search space + objective for every model
        # key, but THIS lane's remote worker executes only the laya objective.
        registry = default_registry()
        descriptor = registry.objective(MODEL_KEY)
        raise SystemExit(
            "[laya-hpo] model_key %r is registered (metric=%s, runner=%s) but "
            "this lane executes only the 'laya' objective; run that model's own "
            "HPO worker for remote execution" % (
                MODEL_KEY, descriptor.metric, descriptor.runner))

    def _install(self):
        if os.environ.get("ER_LAYA_HPO_SKIP_INSTALL") != "1":
            pip_install_runtime()
            pip_install_laya()

    def _load_libs(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        globals()["optuna"] = optuna
        import torch
        options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
        globals()["OPTION_SET"] = options
        options.resource_caps.apply_torch(torch)
        self.optuna = optuna
        self.options = options

    def _study_kwargs(self):
        directions = self.options.objective_mode.directions()
        return ({"directions": directions} if isinstance(directions, list)
                else {"direction": directions})

    def _observer(self, offline):
        observer = TrialObserver(WORKING / OBSERVABILITY_DIR, offline=offline)
        globals()["OBSERVER"] = observer
        self.observer = observer
        return observer

    def _open_study(self):
        options = self.options
        self.resolver = StorageResolver(
            self.optuna, options,
            generation_study_name(generation_id=GENERATION_ID,
                                  model_key=MODEL_KEY))
        study = self.resolver.open(
            options.sampler.create(self.optuna),
            options.pruner.create(self.optuna), self._study_kwargs())
        self._observer(self.resolver.offline)
        # Stale-trial reaping happens ONCE in the session controller (a worker
        # must never race a sibling's freshly-started trial).
        return study

    def _warm_start(self, study):
        # Deduped against EXISTING waiting trials: concurrent workers and
        # resumed sessions must not pile up identical enqueued trials.
        enqueue_warm_start(study, self.options.warm_start.enqueued_trials())

    def _stores(self):
        # The fencing/champion stores are PostgreSQL-only; a local SQLite study
        # (single VM, shared file) runs without them.
        if not self.resolver.remote:
            return None, None
        config = self.resolver.config
        return (TrialLeaseStore(config.url,
                                ttl_seconds=config.lease_ttl_seconds),
                ChampionStore(config.url))

    def _warm_start_champion(self, champion_store):
        """Seed from the shared champion registry when configured (F3).

        Reads the champion only AFTER the shared stores are open; a missing
        champion falls back to the base model, loudly.
        """
        if self.resolver.offline or self.options.warm_start.mode != "champion":
            return
        champion_artifact = resolve_champion_artifact(
            champion_store, generation_id=GENERATION_ID, model_key=MODEL_KEY,
            mode=self.options.warm_start.mode)
        if champion_artifact is None:
            log("warm_start=champion: no champion yet; falling back to the "
                "base model")
        else:
            log("warm_start=champion: seeding from " + champion_artifact)
        self.options = build_option_set(
            HPO_SPACE, champion_artifact=champion_artifact)
        globals()["OPTION_SET"] = self.options

    def _inputs(self):
        train_path = resolve_input(TRAIN_JSONL)
        dev_path = resolve_input(DEV_JSONL)
        base_model = extract_base_model(resolve_input(BASE_MODEL_ARCHIVE))
        log("worker device=%s train=%s dev=%s base=%s"
            % (self.device, train_path.name, dev_path.name, base_model.name))
        return train_path, dev_path, base_model

    def _objective(self, lease_store, champion_store, train_path, dev_path,
                   base_model):
        if os.environ.get("ER_LAYA_HPO_DDP") == "1":
            return make_ddp_objective(self.options, lease_store, champion_store)
        return make_objective(self.device, train_path, dev_path, base_model,
                              lease_store, champion_store)

    def _per_worker(self, study):
        prior = finished_trial_count(study, ("COMPLETE", "PRUNED", "FAIL"))
        remaining = max(0, int(N_TRIALS) - prior)
        # DDP serialises trials (torchrun fans each one out); slots parallelise.
        if os.environ.get("ER_LAYA_HPO_DDP") == "1":
            divisor = 1
        else:
            divisor = max(1, int(os.environ.get("ER_LAYA_HPO_WORKER_COUNT")
                                 or N_JOBS))
        per_worker = per_worker_budget(remaining, divisor)
        log("worker budget prior=%d remaining=%d per_worker=%d study=%s timeout=%s"
            % (prior, remaining, per_worker, self.resolver.study_name,
               self.options.session.timeout_s or "none"))
        return per_worker

    def _optimize_offline(self, study, objective):
        """One SQLite study, no shared coordination: the local split applies."""
        per_worker = self._per_worker(study)
        if per_worker:
            study.optimize(objective, n_trials=per_worker,
                           **self.options.session.optimize_kwargs())

    def _optimize_shared(self, study, objective, champion_store):
        """Reserve one shared trial at a time; FAIL/PRUNED releases its slot."""
        ledger = WorkLedger(self.resolver.config.url,
                            generation_id=GENERATION_ID, model_key=MODEL_KEY,
                            budget=int(N_TRIALS))
        log("worker shared budget=%d remaining=%d study=%s timeout=%s"
            % (int(N_TRIALS), ledger.snapshot().remaining(),
               self.resolver.study_name, self.options.session.timeout_s or "none"))
        ReservedTrialLoop(
            study, objective, ledger,
            optimize_kwargs=self.options.session.optimize_kwargs(),
            observer=self.observer, champion_store=champion_store,
            generation_id=GENERATION_ID, model_key=MODEL_KEY,
            timeout_s=self.options.session.timeout_s, log=log).run()

    def run(self):
        self._assert_laya()
        self._install()
        self._load_libs()
        study = self._open_study()
        self._warm_start(study)
        lease_store, champion_store = self._stores()
        self._warm_start_champion(champion_store)
        train_path, dev_path, base_model = self._inputs()
        try:
            wandb_init()
        except Exception as error:
            log("wandb init failed: " + str(error)[:200])
        objective = self._objective(lease_store, champion_store, train_path,
                                    dev_path, base_model)
        if self.resolver.remote:
            self._optimize_shared(study, objective, champion_store)
        else:
            # Local SQLite (no remote URL) or an offline session: no shared
            # ledger, but parallel workers still share the SAME study file.
            self._optimize_offline(study, objective)
        # Observe the committed study ONCE (idempotent across workers/sessions)
        # and flush every worker's mirror so an exiting worker never loses it.
        _sync_observations(study)
        wandb_finish()


def run_worker(device):
    """Entry point for one worker subprocess (thin facade over WorkerSession)."""
    WorkerSession(device).run()


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SessionReceiptWriter:
    """Build and persist the session receipt from the authoritative study.

    One job per method: load the study, project the trial rows, pick the best,
    compute the ensemble, and assemble/write the receipt.
    """

    def __init__(self, optuna, options, *, offline):
        self.optuna = optuna
        self.options = options
        self.offline = bool(offline)
        self.resolver = StorageResolver(
            optuna, options,
            generation_study_name(generation_id=GENERATION_ID,
                                  model_key=MODEL_KEY),
            offline=offline)
        self.observer = TrialObserver(WORKING / OBSERVABILITY_DIR,
                                      offline=self.offline)

    @staticmethod
    def _receipt_value(trial):
        """The scalar for single-objective, the full vector for multi-objective."""
        full = trial_full_value(trial)
        if not full:
            return None
        return full[0] if len(full) == 1 else full

    def _trial_rows(self, study):
        return [{
            "number": int(trial.number),
            "state": trial.state.name,
            "value": self._receipt_value(trial),
            "values": trial_full_value(trial),
            "params": trial.params,
            "dev_accuracy": trial.user_attrs.get("dev_accuracy"),
            "dev_loss": trial.user_attrs.get("dev_loss"),
        } for trial in study.trials]

    def _best(self, study):
        directions = self.options.objective_mode.directions()
        complete = [trial for trial in study.trials
                    if trial.state == self.optuna.trial.TrialState.COMPLETE
                    and trial_primary_value(trial) is not None]
        return ObjectiveRanker(directions).best(
            complete, value_of=trial_primary_value)

    def _best_dict(self, best):
        if best is None:
            return None
        return {"number": int(best.number), "value": self._receipt_value(best),
                "values": trial_full_value(best),
                "params": best.params,
                "dev_loss": best.user_attrs.get("dev_loss"),
                "checkpoint": best.user_attrs.get("checkpoint")}

    def _corpus(self):
        return {TRAIN_JSONL: sha256_of(resolve_input(TRAIN_JSONL)),
                DEV_JSONL: sha256_of(resolve_input(DEV_JSONL))}

    def _ensemble(self, trials):
        options = self.options
        selected = options.ensembler.select(
            [SimpleNamespace(
                number=t["number"],
                value=(t["value"][0] if isinstance(t["value"], (list, tuple))
                       else t["value"]))
             for t in trials])
        return {"enabled": options.ensembler.enabled,
                "method": options.ensembler.method,
                "top_k": options.ensembler.top_k,
                "selected": [int(t.number) for t in selected]}

    def build(self, study):
        trials = self._trial_rows(study)
        best = self._best(study)
        # CDC events + mirror + offline ledger, synced ONCE (idempotent).
        self.observer.sync(study.trials)
        self.observer.flush()
        receipt = {
            "kernel": "laya-hpo",
            "run_tag": RUN_TAG,
            "study": self.resolver.study_name,
            "generation_id": GENERATION_ID,
            "model_key": MODEL_KEY,
            "space_version": HPO_SPACE.get("space_version"),
            "budget": {"n_trials": int(N_TRIALS), "n_jobs": int(N_JOBS),
                       "seed": int(SEED)},
            "objective": HPO_SPACE.get("objective"),
            "trials": trials,
            "best": self._best_dict(best),
            "corpus_sha256": self._corpus(),
            "published_pin": {"repository": REPOSITORY, "branch": BRANCH,
                              "revision": REVISION},
        }
        receipt["options"] = self.options.as_dict()
        receipt["ensemble"] = self._ensemble(trials)
        receipt["observability"] = self.observer.as_dict()
        receipt["registry"] = default_registry().describe()
        return receipt, best

    def write(self, receipt, best):
        WORKING.mkdir(parents=True, exist_ok=True)
        path = WORKING / "laya-hpo.receipt.json"
        path.write_text(json.dumps(receipt, indent=2) + "\\n", encoding="utf-8")
        log("wrote " + str(path) + " best="
            + (str(best.value) if best is not None else "none"))
        return receipt

    def run(self):
        study = self.resolver.load()
        receipt, best = self.build(study)
        return self.write(receipt, best)


def write_session_receipt():
    """Facade: build + write the session receipt (one SessionReceiptWriter)."""
    import optuna
    options = globals().get("OPTION_SET") or build_option_set(HPO_SPACE)
    offline = offline_active(options)
    return SessionReceiptWriter(optuna, options, offline=offline).run()


class SessionArchive:
    """Archive the session's decision trail (tar + snapshot)."""

    def __init__(self, options):
        self.options = options

    def _champion(self, receipt):
        return (receipt.get("best") or {}).get("checkpoint") or ""

    def write_tar(self, receipt):
        """Stage ONLY the champion checkpoint; the receipt is the trail."""
        WORKING.mkdir(parents=True, exist_ok=True)
        with tarfile.open(WORKING / "laya_hpo.tar.gz", "w:gz",
                          compresslevel=1) as tar:
            tar.add(WORKING / "laya-hpo.receipt.json",
                    arcname="laya-hpo.receipt.json")
            champion = self._champion(receipt)
            if champion and Path(champion).is_dir():
                tar.add(champion, arcname="champion")
                log("staged champion checkpoint " + champion)
        log("staged laya_hpo.tar.gz + receipt in /kaggle/working")

    def _warn_dropped(self, receipt):
        dropped = (receipt.get("observability") or {}).get("events_dropped")
        if dropped:
            log("WARNING observability dropped %s event(s); the snapshot omits "
                "them" % dropped)

    def write_snapshot(self):
        """Build the durable decision-trail snapshot."""
        try:
            session_offline = offline_active(self.options)
            include = [p for p in (
                WORKING / "laya-hpo.receipt.json",
                WORKING / OBSERVABILITY_DIR / "trial_events.jsonl",
                WORKING / OBSERVABILITY_DIR / "study_mirror.jsonl",
                WORKING / OBSERVABILITY_DIR / "hpo_trials.jsonl",
            ) if p.exists()]
            builder = SnapshotBuilder(generation=WORKING,
                                      sequence=int(time.time()))
            snapshot = builder.build(
                include, local_study_path() if session_offline else None)
            log("hpo snapshot -> " + str(snapshot))
        except Exception as error:
            log("hpo snapshot skipped: " + str(error)[:200])

    def run(self, receipt):
        self.write_tar(receipt)
        self._warn_dropped(receipt)
        self.write_snapshot()


class SessionOrchestrator:
    """One Kaggle session: prepare, launch one worker per GPU, archive."""

    def __init__(self):
        self.options = None

    def _prepare(self):
        pip_install_runtime()
        pip_install_laya()
        import torch
        options = build_option_set(HPO_SPACE)
        globals()["OPTION_SET"] = options
        options.resource_caps.apply_torch(torch)
        # A remote URL is validated when present; absent, the workers share the
        # config-declared local SQLite study (no hard stop).
        if remote_configured():
            ensure_optuna_url()
        self.options = options
        return torch

    def _specs(self, torch):
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        specs = self.options.scheduler.workers(gpu_count or 1)
        if self.options.session.offline and len(specs) > 1:
            # Offline uses ONE SQLite file; concurrent writers would lock. Keep
            # the documented single-process semantics regardless of the slots plan.
            log("offline mode: collapsing %d workers to 1" % len(specs))
            specs = specs[:1]
        log("session gpus=%d cores=%d mode=%s workers=%d study=%s budget=%d "
            "threads_per_worker=%d timeout=%s"
            % (gpu_count, os.cpu_count() or 1, self.options.scheduler.mode,
               len(specs),
               generation_study_name(generation_id=GENERATION_ID,
                                     model_key=MODEL_KEY),
               int(N_TRIALS),
               self.options.resource_caps.threads_per_worker,
               self.options.session.timeout_s or "none"))
        return specs

    def _launch(self, specs):
        script = os.path.abspath(__file__)
        processes = []
        for spec in specs:
            env = os.environ.copy()
            env.update(spec.env)
            env["ER_LAYA_HPO_SKIP_INSTALL"] = "1"
            env["ER_LAYA_HPO_WORKER_COUNT"] = str(len(specs))
            if self.options.scheduler.mode == "slots":
                # Pin the slot to ONE physical GPU (its own worker process).
                env["CUDA_VISIBLE_DEVICES"] = str(spec.device_index)
                env["ER_LAYA_HPO_WORKER_DEVICE"] = "cuda:0"
            else:
                # DDP-per-trial: one controller fans each trial over torchrun.
                env["ER_LAYA_HPO_WORKER_DEVICE"] = "cuda:0"
                env["ER_LAYA_HPO_DDP"] = "1"
            processes.append(subprocess.Popen([sys.executable, script], env=env))
        return [process.wait() for process in processes]

    def _reap_stale(self):
        """Reap stale RUNNING trials ONCE, before ANY worker starts (F7).

        A storage error is loud (never silently skipped): a worker must never
        race a sibling's freshly-started trial.
        """
        storage = create_storage(storage_from_environment())
        study_name = generation_study_name(generation_id=GENERATION_ID,
                                           model_key=MODEL_KEY)
        reaped = reap_stale_trials_for_study(study_name, storage)
        log("stale-trial reap: " + ("reaped" if reaped else "no study yet"))

    def run(self):
        torch = self._prepare()
        if remote_configured() and not self.options.session.offline:
            self._reap_stale()
        if self.options.scheduler.mode == "slots":
            self.options.mps.start()
        codes = self._launch(self._specs(torch))
        if self.options.scheduler.mode == "slots":
            self.options.mps.stop()
        log("workers exited: " + str(codes))
        receipt = write_session_receipt()
        SessionArchive(self.options).run(receipt)
        if any(code != 0 for code in codes):
            raise SystemExit("laya HPO worker failure: exit codes " + str(codes))


def main():
    SessionOrchestrator().run()


if __name__ == "__main__":
    _ddp_trial = os.environ.get("ER_LAYA_HPO_DDP_TRIAL")
    _worker_device = os.environ.get("ER_LAYA_HPO_WORKER_DEVICE")
    if _ddp_trial is not None:
        run_ddp_trial(int(_ddp_trial))
    elif _worker_device:
        run_worker(_worker_device)
    else:
        main()
'''
