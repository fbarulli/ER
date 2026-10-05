# HANDOFF — PIPE.md preparation refactor (2026-10-05)

Written at end of session for a fresh context. Task: implement the
`PIPE.md` target contract (one owner, config SSOT, typed state, one
interpreter, no per-stage subprocesses, verified handoff boundary).

## 0. UPDATE 2026-10-05 ~11:50 — phase 1 (consolidation) APPLIED; baseline killed twice

- The baseline BEFORE build (pid 1217655) was killed silently at ~11:07
  (agent-session teardown SIGTERM; no traceback, no OOM) mid-`suite_inputs`
  during the local ablation pass (attribute 17/37). A `--resume-from
  suite_inputs` attempt correctly failed twice on the pinned resume contract:
  first on provenance (the colab fixes to `src/model_tracks/run.py` +
  `src/training/prepare_embeddings.py` were edited AFTER run entry — restored
  from `55578b0`, hashes match provenance, then re-applied post-mortem), then
  on `Stale prepared resume input: data/track_setup/eligible_catalog.csv`
  (the killed package build had already mutated setup files after
  `full_bundle` pinned the inventory). A `--resume-from validation` re-run was
  killed the same way (session event) during validation. CONCLUSION: the run
  environment kills even setsid-detached processes on session teardown/abort;
  long builds must run under `systemd-run --user --scope` (verified working,
  linger=yes) and nothing else.
- BEFORE evidence (valid, old code, run dir
  `results/training_prep/20261005T100747497085/`, stages complete):
  dedupe 11.1s, cross_country_pairs 18.4s, number_reference 3.8s,
  verify_reference 0.01s, canonical_and_gates 636.7s, gate_census 0.6s
  (pin 115789/338/19119), labeled_pairs 0.7s, negative_supply 148.3s,
  discriminator 6.1s (SEPARABLE, gate mode unchanged), validation 335.7s,
  graph_inputs 127.3s, full_bundle 578.9s (of which bundle
  objective_and_epoch_plans 272.2s, bundle write 318.4s incl. compress 11.0s,
  native tokens 28.7s). suite_inputs/verify_handoff: NO complete old-code
  record exists (killed mid-ablation; old verify_handoff had no load meter).
- Phase 1 APPLIED (uncommitted, 22/22 pinned tests green:
  test_prepare_all + test_data_gate_attestation + test_training_handoff):
  - `src/training/handoff.py` (NEW): the single offline consumer boundary.
    `verify_training_loads` runs producer checks (provenance re-hash, frozen
    CSV agreement, graph manifest hashes vs sources+checkpoint) then consumer
    checks (`model_tracks.worker.graph_worker_settings` per track when the
    track config exists, loss/batch attestation, `model_tracks.package.verify`,
    smoke-unchanged, final inventory) and persists `run_dir/handoff.json`:
    typed `HandoffReport` with per-input load meter (loads/bytes/seconds/sha),
    `LossBatchAttestation` (loss, epochs, batch_sizes, folds, plan identity,
    "every objective row exactly once per epoch", validated_by chain:
    preflight@suite_inputs -> handoff boundary -> train_prepared@trainer
    start). Bundle is unpickled ONCE per run (run cache); the expensive plan
    data-digest is NOT recomputed at the boundary (validated_by preflight in
    the same provenance-identical process; `plan_identity_revalidate=True`
    recomputes for standalone boundary use). GPU embeddings stay "pending".
  - `src/core/schemas.py`: `PREPARATION_REUSABLE_KEYS`,
    `PreparationGraphSetupSpec` (setup-dir layout contract),
    `PreparationSpec` (run/file layout + reusable_keys + graph_setup;
    validator rejects absolute/traversal paths), `TrainingConfig.preparation`.
  - `config/training.yaml`: `preparation:` block (SSOT; values mirror the
    previously hardcoded names exactly).
  - `src/training/prepare_all.py`: all run/file names now load from
    `training_cfg().preparation` (run dir base, lock, smoke dir, archive
    name, manifest/timings/gate_census/discriminator/handoff file names,
    negative_supply dir, stage log/timing suffixes); `verify_handoff` stage
    is now `handoff.verify_training_loads` + `write_handoff_report`; manifest
    gains `handoff` key; outputs/bundle/suite_package keys unchanged (tests
    pinned green).
- NEXT: phase 2 = static per-function optimization review (in flight; user
  rule: do NOT rely on run timing, reason from code) -> apply safe wins ->
  THEN run the definitive AFTER build over final code via systemd-run scope,
  compare vs the BEFORE numbers above. Also: minimal-proportion SAMPLE build
  across all surfaces (real diet ratios + hard/soft negative mix, see §8)
  to validate the consolidated boundary cheaply before the full AFTER run.

## 0.1 UPDATE 2026-10-05 ~13:20 — independent review PASS; safe-wins applied

- Fresh unseen agent reviewed the whole consolidation + optimization batch:
  **VERDICT: PASS**, 22/22 pinned tests green. Confirmed programmatically:
  config-SSOT values identical to removed literals; one-verified-load
  contract holds (run cache + in-process stages); validated_by chain
  accurate; calibration_rows was a real NameError at 55578b0 (3 uses, 0
  definitions); dedupe hoist pure; memo sound. All findings addressed:
  - package.py dead `import subprocess` removed.
  - reusable_keys SSOT gap fixed: the run now reads
    `training_cfg().preparation.reusable_keys`; module constant stays as the
    defaulted public API only.
  - checkpoint_hash gained `use_memo` (default True); provenance gates pass
    use_memo=False — a re-hash is never a stat comparison (reviewer finding
    3). The pinned provenance-test fake gained **kwargs (signature
    accommodation, pin intent unchanged).
  - composition_fingerprint memo assumption documented.
- Rejected finding (recorded): passing the package-built `shared` object
  into preflight would save a from_bundle recomputation but contradicts the
  explicit ruling at package.py ("Preflight independently reloads and
  validates the bundle. Release the producer's object graph first...") —
  memory + independence win over duplicate work. Keep.
- Check-free trainer landed: src/training/attestation.py
  (TrainingAttestation, verify = ONE streaming bundle sha256 + status;
  attestation_from_handoff; plan_identity carried from the boundary) +
  train_prepared --attestation / ER_TRAINING_ATTESTATION: skips
  validate_run_plan re-verification; STILL cheaply compares
  loss/train_frac/sample against the attested plan identity
  (verify_plan_identity) — invocation drift cannot silently leave the
  attested batch contract. No-attestation path byte-identical to before.
- Fixture golden run in flight: testing.csv grew 16 -> 35 rows through the
  golden loop (each failure = a real contract detail): identical-attribute
  twins on DIFFERENT gtins produce proceed (same-gtin pairs are never
  candidates); labeled negatives need hard_no pairs with similarity >= 0.8
  (volume-flip twins of identical titles); fallback = insufficient agreed
  evidence dimensions; fixed an empty-negative-population pandas crash in
  build_final_validation.negative_policy_evidence (fail-loud, not opaque).
  Remaining: dev-fold calibration carve density (worker agent iterating).


## 1. Task definition

`PIPE.md` (committed in `55578b0`) is the contract. Build order items:
1. Run owner + import-safe stage functions; no per-stage Python subprocesses.
2. Share source/base/bundle objects; de-duplicate diet/preflight work.
3. Clean start every invocation; run-owned state cleared on failure.
4. tar.zst writer/reader/transport complete; no duplicate copies.
5. Connect the ACTUAL training-load boundary; keep failure/timing reports.
6. Representative complete builds before/after; compare outputs, elapsed,
   serialization/load counts, artifact sizes.
Item 6 dominates right now: a BEFORE build is running; the refactor must
land, then an AFTER build must run and be compared.

## 2. State of the world

### Committed: wave 1 (`55578b0` "refactor(preparation): own CSV-to-bundle work in one process")
- `src/training/preparation_run.py`: `TrainingPreparation` Pydantic owner.
  In-interpreter stage runner (`run_stage` via importlib/runpy, saves/restores
  sys.argv/os.environ/cwd, redirects stdout+stderr to the stage log),
  private caches (`_datasets`, `_base`, `_bundles`, `_aliases`, `_objects`),
  `invalidate()`, `alias_bundle()`, ContextVar `active_preparation()`.
- `src/training/prepare_all.py`: the whole lifecycle (481 lines). Stages:
  dedupe, cross_country_pairs, number_reference, verify_reference,
  canonical_and_gates, gate_census, labeled_pairs, [negative_supply,
  discriminator when a run tag is present], validation, graph_inputs,
  full_bundle, suite_inputs, verify_handoff. Typed `PreparationState`
  (persisted manifest.json) + `PreparedFile`, provenance pinning, smoke
  before/after check, `refresh_gate_census` (rewrites the pin in
  config/training.yaml by regex — "expected exactly one configured gate
  census"), `verify_stage_manifest`, `copy_bundle` (FICLONE, never
  hardlinks), `file_inventory` (re-hash every snapshot, dedupe paths).
- `src/model_tracks/data_gate.py` + `tests/test_data_gate_attestation.py`:
  one-shot data-attestation for the suite (sha256 over config digest + all
  input byte digests; workers re-verify unless `ER_DATA_GATE_ENFORCE=1`).
- `src/core/portable_archive.py`: tar.zst streaming (digests stream from the
  Zstandard decoder); ZIP read retained for old artifacts.
  `config/model_tracks.yaml`: `input_archive_format: tar.zst`.
- `tests/test_prepare_all.py` (155 lines added): pins the behavior below.
- `src/training/base_data.py`, `build_reference.py`, `labeled_pairs.py`,
  `prepared_bundle.py`, `graph_tracks/{preflight,prepare,setup}.py`,
  `model_tracks/{colab,config,package,preflight,run,worker}.py`,
  `scripts/diet_manifest.py`, `src/cli/colab.py` — supporting edits for the
  shared-object/attestation paths.

### IN FLIGHT: baseline BEFORE build (item 6)
- Started 2026-10-05 10:07:49 UTC, pid 1217655
  (`.venv/bin/python -m training.prepare_all`).
- Run dir: `results/training_prep/20261005T100747497085/`
  (timings.json, timings.log, manifest.json, per-stage .log/.timing.json,
  before/ archive of replaced artifacts).
- Progress (last confirmed 2026-10-05 ~10:52 UTC, pid still alive,
  ~55 min elapsed): dedupe 11.1s, cross_country_pairs 18.4s,
  number_reference 3.8s, verify_reference 0.01s, canonical_and_gates
  636.7s, gate_census 0.6s (rewrote the pin in config/training.yaml:
  hard_no 115790→115789, fallback 19118→19119 — see working tree
  below), labeled_pairs 0.7s, negative_supply 148.3s (diagnostic lane,
  tag `prep_20261005T100747497085`), discriminator 6.1s (verdict
  SEPARABLE, returncode 1 tolerated in gate mode), validation 335.7s,
  graph_inputs 127.3s, full_bundle 578.9s.
  `suite_inputs` (model_tracks.package) started 10:38:56 and was still
  running; it runs the suite preflight (bundle array validation, token/
  plan validation, diet gate, graph load_inputs, package build). No
  earlier suite_inputs timing exists for reference, so allow generous
  time; if it looks wedged, inspect suite_inputs.log in the run dir
  (the log line count/stage output is the liveness signal).
- Remaining stages after suite_inputs: verify_handoff (in-process: bundle
  load verify_inputs=True, stale-CSV/graph/provenance checks, package verify,
  smoke-unchanged check, output inventory, manifest bundle/suite_package).
- Monitor: `tail -f results/training_prep/20261005T100747497085/timings.log`
  and `ps -p 1217655`. stdout also went to /tmp/opencode/baseline_prepare_all.log
  (tmp path may not survive reboot; the run dir is the durable record).
- The run holds `results/training_prep.lock` (flock, non-blocking) — do not
  start another prepare_all while it runs.
- When it completes: capture timings.json totals, manifest.json
  (reusable_outputs + outputs inventories = artifact sizes/shas), and the
  per-stage seconds as the BEFORE evidence. Keep the run dir intact.

### Uncommitted working tree (do not revert; most are build outputs)
- Build outputs of the running/previous runs:
  `data/canonical_records.csv`, `data/gate_results.csv`,
  `data/prepared/full/worker_1_baseline.pkl.gz{,.json}`,
  `data/track_setup/{eligible_catalog.csv,prepared/*,setup_manifest.json}`,
  and the regenerated `dataset_deduped`/`labeled_pairs`/`final_validation`
  etc. under data/.
- `config/training.yaml`: gate_census_pin updated by the running build's
  gate_census stage (115789/338/19119). Expected; part of the pipeline
  contract (measured census published at its owning boundary).
- Colab-wave fixes from earlier in the session (untested against GPU):
  - `src/cli/colab.py`: worker chunk lines now forward to BOTH the root
    system log and training.log (previously suppressed in root); `[done]`
    now printed only on clean completion (`completed` flag set at end of
    the try body) with `[failed]` otherwise — previously any outcome
    printed [done] during teardown.
  - `src/model_tracks/run.py`: preflight now called with
    `allow_gpu_pending=cfg.post_training_ablation` so the hybrid text cache
    can be declared pending when the suite exports the baseline itself.
  - `src/training/prepare_embeddings.py`: `input_identity` now includes
    `embedding_dtype: 'float32'` (create_cache records the dtype it wrote;
    without the key the request compares None against 'float32' and
    refuses its own output).
  - `tests/test_suite_training_controls.py`: text worker test now stubs
    `PreparedBundleManifest` (sidecar header load) instead of the full
    `load_prepared_bundle` — the worker reads only the bundle sidecar to
    build its command; full verified load happens in the trainer after the
    barrier. (This mirrors PIPE.md "must not unpickle the full bundle just
    to obtain a header" — check the worker itself already does this; the
    test change only speeds the fake.)
  - `COLAB_SURFACES.md`: appended "Principles audit of the Colab/GPU
    training path (2026-10-05)" — table of findings; conclusion: no
    principle 1-4 violations in the colab path; one naming nuance in
    graph_tracks/config.py (`retrieval_ks` field defaulting to the ANN
    recall ladder via SSOT).
  These are NOT part of the PIPE.md commit; decide with the user whether
  they ship in the same commit or a separate one.

### Designed but NOT applied: wave 2 (config-SSOT + typed handoff)
Design drafts in `/tmp/opencode/draft/` (tmp may not survive; treat as
design notes, re-derive against live code if missing):
- `schemas_fragment.py`: `PreparationStageSpec` (name, entrypoint
  module|script|in_process, module/script, args templates, archive,
  post_copies, reusable, invalidate, clear_caches, check),
  `PreparationGraphSetupSpec` (setup-dir file layout: catalog, splits,
  pairs, pair_lineage, census, manifest, text_config, track_config_suffix,
  prepared_dir, shared_training_data/projection, text_training_binding,
  text_export_request, embedding_request, shared_embeddings,
  template_gnn/hybrid/text), `PreparationSpec` (run_dir_base, lock_file,
  smoke_dir, suite_archive_name, archive_dir, base_payload prefix/ext,
  manifest/timings/gate_census/discriminator/handoff file names,
  stage_manifest_dir, negative_supply_dir, suite_config,
  full_bundle_payload, defer_training_tensors, reusable_keys,
  stage_manifests, graph_setup, stages; validators: unique names, required
  resume points, negative_supply+discriminator adjacency before validation,
  only gate_census/verify_handoff may be in_process). Home:
  `config/training.yaml preparation:` + `TrainingConfig.preparation`.
- `prepare_all_part1.py` + `prepare_all_part2.py`: full rewrite of
  prepare_all.py as thin orchestrator + small functions:
  `PreparationState` gains `stage_metrics` (StageRecord: status,
  started_at, detail_path, finished_at, seconds, returncode),
  `stage_seconds`, `shared_base_payload`, `gate_census`, `discriminator`,
  `outputs`, `bundle`, `suite_package`, `handoff`, `failed_stage`,
  `error`; `PreparationRequest`/`PreparationContext` typed;
  `resolve_context`, `hold_preparation_lock`, `begin_state`/`resume_state`,
  `publish`, `stage_command` (renders config args against a declared
  placeholder set {run_dir, tag, suite_config, checkpoint, setup_dir,
  text_bundle, full_bundle, suite_archive, negative_dir, negative_pairs,
  discriminator_out, dataset_deduped, payload, text_model}),
  `stage_environment` (env var names would come from a spec.env sub-spec —
  NOT yet in schemas_fragment; env var names to config-ize:
  EUROMONITOR_SHARED_BASE_DATA, ER_DATA_GATE_ENFORCE, ER_TIMING_OUT,
  ER_TIMING_LOG), `run_stage`, `archive_existing`, `apply_post_copies`,
  `refresh_reusable_outputs`, `apply_stage_side_effects` (declared cache
  invalidations incl. pipeline._VERDICTS_CACHE), in-process handlers
  `run_gate_census_stage` / `run_verify_handoff_stage`, `_execute_stage`,
  `_judging` (discriminator verdict policy), `_finalize_stage`,
  `_fail_stage`.
- `handoff.py`: new module `src/training/handoff.py` —
  `HandoffLoad`/`HandoffReport` (typed, persisted to
  run_dir/handoff.json with a load meter: per-input seconds, bytes, sha,
  load counts — the item-6 serialization/load counts),
  `verify_training_loads(...)` loading through the ACTUAL worker paths:
  text side `load_prepared_bundle(verify_inputs=True)` + frozen CSV
  agreement + `validate_run_plan`/`validate_epoch_batches` +
  `validate_training_tokens`; graph side `model_tracks.worker.
  graph_worker_settings` + `graph_tracks.preflight.load_inputs` per
  executed track; package `model_tracks.package.verify`; smoke-unchanged
  check; final file inventory. CPU only; GPU embeddings explicitly
  "pending" in the report.

## 3. Contract the tests pin (do not break during refactor)
From `tests/test_prepare_all.py` (all currently passing):
- Module API: `training.prepare_all` exposes `prepare_all`,
  `refresh_gate_census`, `verify_stage_manifest`, `PreparedFile`,
  `verify_reusable_outputs`, `file_inventory`, `sha256`, `copy_bundle`,
  `preparation_provenance`, `REUSABLE_KEYS` (13 keys: dataset_deduped,
  sku_to_rep, dedupe_summary, ambiguous_offer_groups, removals,
  dedupe_conflicts, number_reference, second04_pairs_positive,
  canonical_records, gate_results, labeled_pairs, final_validation,
  validation_fold_map).
- Stage commands keep their module identities: `-m training.dedupe`,
  `-m training.build_second04_pairs`, `-m training.build_reference
  [--verify]`, `-m training.data_prep`, `-m training.labeled_pairs`,
  `-m training.negative_supply --run-tag <tag>`,
  `scripts/negative_supply_discriminator.py <pairs> --out <json>`,
  `-m training.build_final_validation`, `-m graph_tracks.setup
  --output <setup> --text-checkpoint <ck> --defer-training-tensors`,
  `-m training.train --dataset <deduped> --payload full --prepare-bundle
  <bundle> --model <text_model> --no-mask-effect --no-plot`,
  `-m model_tracks.package --config <cfg> --output <archive>`.
- Stages end `[..., 'graph_inputs', 'full_bundle', 'suite_inputs',
  'verify_handoff']`; completion requires verified handoff.
- Manifest JSON stays readable with the same keys: status, stages,
  failed_stage, training_started (always False), smoke_updated,
  reusable_outputs, outputs, bundle, suite_package, gate_census,
  discriminator, stage_metrics, stage_seconds, smoke_unchanged_verified.
- Stage env carries `ER_DATA_GATE_ENFORCE=1`; WANDB_API_KEY removed.
- Resume from suite_inputs re-runs ONLY model_tracks.package +
  verify_handoff; any stale reusable output (size or sha) raises
  'Stale prepared resume input'; changed source/config/raw input/
  checkpoint provenance blocks resume; smoke files must be unchanged.
- `copy_bundle`: independent inodes (FICLONE or copy2 fallback; ENOSPC
  etc. propagate; EOPNOTSUPP/EXDEV/EINVAL/ENOTTY/ENOSYS fall back).
- `refresh_gate_census`: preserves surrounding config bytes/comments
  (regex replaces exactly the 4-line pin), rejects self-pairs, empty
  endpoints, duplicates (incl. reversed), unknown decisions.
- Provenance: hashes src/**/*.py, scripts/**/*.py, config **/*.{yaml,
  yml,json} plus the four named paths; training.yaml identity computed
  with rand_matching.gate_census_pin removed (output, not setting).
- `TrainingPreparation.run_stage` is the injection point tests use to
  fake stages; keep it monkeypatchable with signature
  `run_stage(self, arguments, *, root, env, log)`.
- `model_tracks.config.load_config` stays the suite-config entrypoint
  (tests monkeypatch it); suite fields used: setup_dir, text_bundle,
  text_model, epochs, input_archive_format.
Also run the full test file after changes:
`PYTHONPATH=src .venv/bin/python -m pytest tests/test_prepare_all.py
tests/test_data_gate_attestation.py -x -q`

## 4. Known hazards
- The gate_census stage writes to config/training.yaml mid-run (regex
  rewrite). Any refactor must keep the pin the ONLY mutated config and
  keep `preparation_provenance`'s pin-exclusion or resume will fail.
- `resume_from='validation'` inserts negative_supply+discriminator
  before validation; the run tag defaults to `prep_<run_dir name>`.
- full_bundle stage: after publishing the bundle, if text_bundle !=
  bundle it archives the old text bundle and CoW-copies the new one,
  aliasing both in the run's bundle cache, and copies the .json sidecar.
  suite_inputs runs model_tracks.package (verify_inputs via data gate
  attestation; ER_DATA_GATE_ENFORCE=1 forces per-byte re-verification).
- verify_handoff (current inline version) loads the full bundle
  verify_inputs=True ONCE; the handoff.py design keeps that single load
  and reuses the header/arrays — PIPE.md forbids a second unpickle for
  the header.
- In-interpreter stage execution mutates global interpreter state
  (sys.argv, os.environ, cwd, pandas/pipeline caches).
  `apply_stage_side_effects` (draft) is how run-owned caches get cleared
  declaratively; today it's ad-hoc: full_bundle clears `_base`+`_datasets`,
  dedupe invalidates dataset_deduped, number_reference resets
  pipeline._VERDICTS_CACHE/_VERDICTS_LOADED.
- Draft inconsistencies to resolve when applying (noted in drafts):
  `ctx.spec.env` referenced but undefined in schemas_fragment;
  `_fresh_training_view` / `if False` scaffolding;
  `TrainingPreparationOwner.ctx_spec_resume` always 'dedupe' (resume_from
  only reaches _prepare_all via execute()'s model_dump today);
  handoff.py imports a not-yet-existing `training.handoff_checks`
  (split the provenance/inventory helpers out of prepare_all or inline
  them); `graph_inputs.npz`/`listings` inventory names need to match
  what graph_tracks.setup actually publishes.
- Unmeasured performance must not be reported as complete (PIPE.md
  final paragraph). The BEFORE build above is the reference.

## 5. Next steps (in order)
1. Confirm the baseline build finished: manifest.json
   status=='complete', all 14 stage_metrics complete. Save the BEFORE
   evidence (timings.json, manifest.json inventories, artifact sizes) to
   a durable note (e.g. results/training_prep/before_20261005T100747497085/
   or append to this file). If it failed, read the failing stage's
   .log in the run dir first.
2. Run the test suite to confirm green pre-refactor:
   `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prepare_all.py
   tests/test_data_gate_attestation.py -x -q`.
3. Apply wave 2: add the schemas (resolve the `env` sub-spec first),
   add `preparation:` to config/training.yaml mirroring the current
   hardcoded plan EXACTLY (same stage names/order/args so tests and
   behavior are unchanged), rewrite prepare_all.py from the drafts,
   create src/training/handoff.py. Keep the module API pinned in §3.
   Re-run the pinned tests; also run tests/test_suite_training_controls.py
   and anything importing prepare_all (grep for `prepare_all`).
4. Smoke the lifecycle cheaply if possible (dedupe-only or resume
   paths), then run the AFTER full build the same way the BEFORE one was
   started (fresh run dir; it archives replaced artifacts into before/).
5. Compare BEFORE vs AFTER: per-stage seconds, total, reusable/outputs
   inventories (sha/bytes), manifest layout differences, and the new
   handoff.json load counts/seconds. Record in PIPE.md evidence or a
   results note. Only then claim item 6 done.
6. Decide commit grouping with the user: PIPE.md wave-2 commit vs the
   uncommitted colab-wave fixes (§2) — they are separate concerns; the
   colab fixes have no GPU verification yet.

## 6. Commands cheat-sheet
```bash
cd /home/opc/ONE/ER
# monitor build
ps -p 1217655 -o etimes= ; tail -f results/training_prep/20261005T100747497085/timings.log
# tests
PYTHONPATH=src .venv/bin/python -m pytest tests/test_prepare_all.py tests/test_data_gate_attestation.py -x -q
# full build (fresh run dir; takes ~1h; holds results/training_prep.lock)
PYTHONPATH=src .venv/bin/python -m training.prepare_all
# resume from a run dir
PYTHONPATH=src .venv/bin/python -m training.prepare_all \
  --run-dir results/training_prep/<dir> --resume-from suite_inputs
# diagnostics: per-stage logs in the run dir; <stage>.log + <stage>.timing.json
```

## 7. Open questions for the user
- Ship the uncommitted colab-wave fixes in the same commit as wave 2, or
  separately (or hold them back entirely)?
- DVC IS RETIRED (owner decision 2026-10-05): no new DVC runs; a replacement
  publication/transport mechanism is TBD — deliberately NOT being built now.
  When it is, these are the wired surfaces to replace: src/training/dvc_store.py,
  core/schemas dvc fields (publish_dvc/dvc_enabled, DvcPublication*),
  model_tracks.worker dvc.enabled override, preflight runtime_versions dvc
  requirement, artifact publishers (incremental/complete_colab_worker),
  hpo_persistence, dvc.yaml/dvc.lock/dvc_refs/ and the dvc settings in
  config/training.yaml. Publication today still works via publish_git +
  tar.zst package transport.
- The diet-gate blocker in TODO.md (neg_aug_frac 0.2592 < 0.30,
  pos/neg ratio 2.9821 > 1.50) pre-dates this refactor and still blocks
  the suite supervisor; it is OUT of scope for PIPE.md (data-policy,
  owner decision) but will surface again when the suite actually runs.

## 8. Minimal-proportion sample build (user directive, pending)

Validate the consolidated lifecycle on ALL surfaces with a small sample that
keeps the REAL contracts, not a degenerate toy:
- Sample the source listing population minimally but compose the prepared
  inputs so the diet arithmetic matches production expectations: pos/neg view
  ratio (static_view_ratio / effective_train_ratio), hard/soft (train_neg vs
  neg) proportions incl. hard-positive and hard-negative mask audit shares,
  neg_aug_frac style augmentation fractions, and the gate-census decision
  proportions (hard_no/proceed/fallback) scaled from the measured pin.
- Every surface must execute: dedupe -> gates -> census -> labeled pairs ->
  negative supply/discriminator -> validation -> graph_inputs (smoke setup
  manifest) -> full_bundle (tokens + frozen plan, sample plan retention) ->
  suite_inputs package (preflight + diet + shared data) -> verify_handoff
  (handoff.json pass incl. LossBatchAttestation with the sampled batch
  plan).
- Existing levers (to confirm during phase 2): setup_manifest 'smoke' flag +
  source_listing_count, train.py --sample/--train-frac, diet_manifest rc=3
  sampled-smoke warning, training.yaml mining/negative caps, suite sample
  plan relaxation (validate_run_plan sample path). Goal: minutes, not the
  ~1.2h full build, and a green handoff.json as the surface-coverage proof.

### 8.1 Plan-policy experiment matrix (open decision — experiment, don't fork)

Difficulty governance stays an OPEN question, decided by experiment, not
architecture commitment. The planner must treat policy as a config-owned
dimension with shared instrumentation:
- `exact_frozen` (current): pools + exact batches frozen; online = masking/
  inject schedule only. Attested: exact composition.
- `quota_seeded`: pools + slice quotas + coverage window frozen; online =
  seeded member selection within quota. Attested: pools, quotas, window.
- `loss_adaptive`: pools + pairing invariant + hard-pressure floor/ceiling
  frozen; online = slice emphasis within bounds. Attested: bounds + effective
  telemetry.
Constant across ALL arms (non-negotiable): pairing invariant (no dead masked
positives), verified-conflict-only eligibility for hard emphasis, coverage
floor, effective-composition telemetry per epoch, collapse guardrail +
uniformity sized to the EFFECTIVE batch. An arm that cannot state its
effective diet per epoch is not an experiment.
Experiment bed: the minimal-proportion sample build (§8) — same data/slices,
swap policy, compare convergence + collapse telemetry. Decision deferred
until the pipeline is solid (phase 2 applied, sample build green, diet
rebalanced, AFTER build compared and committed). Collapse framing: dynamic
difficulty is anti-collapse machinery; over-softening risks clustering
collapse, unverified hardness risks false-negative shattering — both modes
must be telemetered in every arm.

