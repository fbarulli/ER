# Colab lane

Consolidated CPU + GPU remote-training lanes for Colab. The CPU bundle
generation is happy in the dedicated bundles lane and the kaggle lane is the
network transport for other data. Everything Colab shaped lives here.

The two lanes are declared once in `config/training.yaml` `colab.lanes`:
`bundle` (`--gpu CPU`, the CPU data-bundle lane) and `training` (any other
accelerator, the GPU training lane). `ColabSpec.lane_for(gpu)` resolves the
accelerator request to one of them; nothing else spells the lane names.

## Structure (2026-10-06 consolidation)

| file | role |
|---|---|
| `src/cli/colab_lane.py` | the lane classes: `ColabLaneBase` (transport dial-ins + receipts + shared contracts), `ColabCPULane` (committed-export delivery + CPU prep parity), `ColabGPULane` (accelerator/retention boundary gates) |
| `src/cli/colab_lane_contracts.py` | shared lane constants (`DELIVERY_PREPARED_DIRS`, `MAX_PARALLEL_PREP_SESSIONS`, `DELIVERY_ARCHIVE_NAME`) + the `ColabLaneBase` contracts (delivery root/member list, checkout guard, resume triplet) |
| `src/cli/colab_lane_cpu_provision.py` | CPU lane provisioning order + byte-exact prepare-launch script segments (committed-export membership guard) |
| `src/cli/colab_lane_cpu_delivery.py` | delivery archive, launch/poll/collect/download phases, frozen resume-state upload |
| `src/cli/colab_lane_cpu_poll.py` | VM prepare-log poll (offset probes, transit tolerance, one-lane-transcript fan) |
| `src/cli/colab.py` | GPU launcher surface facade (`train / tracks / dual-train / hpo / sims / mixed / smoke / stop`) plus the legacy `--what bundle` upload lane |
| `src/cli/colab_self_watch.py` | detached session self-watch (spawned default on executed remote runs: poll → delivery proof → release guarantee; receipt under `TRAINING_RESULTS/self_watch_<run_id>/`) |
| `src/cli/colab_retention.py` | local HPO report/snapshot retention (one-roof exported from the launcher surface) |
| `src/cli/colab_bundle.py` | CPU committed-export delivery facade |
| `src/cli/colab_data_bundle_prep.py` | CPU data-bundle prep facade (high-RAM shape, one lane transcript, 2-parallel cap) |
| `src/cli/colab_cli_entry.py` | read-only-home-safe entry point for the installed Colab CLI |
| `src/core/schemas.py` | `ColabBundlePlan` (additive dry-run receipt) + the baked suite-matrix spec: `SuiteMatrixSpec`/`SuiteMatrixEntry`/`SuiteDeviceFlip` with `canonical_suite_matrix()` |

`ColabLaneBase` resolves every transport dial-in through the `cli.colab`
module namespace at call time — one patch surface for the offline fakes, no
captured bindings.

## Commands

GPU lane (`er-colab`, or `colab_backend.py` per the runbook; never
`python -m cli.colab` for a launch — runbook "Do not"):

```bash
er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU
er-colab --what tracks --tracks-config config/model_tracks.yaml --gpu T4 --allow-gpu
er-colab --what stop
```

## Command inventory (SSOT: `cli.colab.main`, src/cli/colab.py:2201–2325)

`--what` choices (default `tracks`): `train | tracks | dual-train | hpo |
sims | mixed | smoke | bundle | stop`. Flags:

| flag | meaning (source anchor) |
|---|---|
| `--tracks-config <yaml>` | all-track suite; applies to train/tracks/smoke only (colab.py:2351); forces `--what tracks` (colab.py:2409) |
| `--prepared-input-package <path>` | reuse a `training.prepare_all` `all_tracks_inputs` package; suite-only; archive-integrity verified then copied to `results/model_tracks/<tag>__inputs.<fmt>`. `<path>` may be the archive, a bundle directory (resolved through its `bundle.receipt.json`'s `archive`, else the canonical `all_tracks_inputs.tar.zst`), so `results/kaggle_lane/full/bundle` works directly. Unspecified on the default `--what tracks` launch, it resolves to that full bundle. A path that resolves to nothing fails before the VM with every known bundle + cohort/revision (`resolve_prepared_input_package`, `prepared_package_candidates`) |
| `--dataset-csv` | raw export for `--what bundle` only (colab.py:2215–2219) |
| `--train-frac`, `--epochs`, `--workers`, `--model`, `--loss`, `--run-label`, `--masking-profile`, `--collapse-guardrail-profile` | plain `train`/`dual-train`/`hpo` knobs (colab.py:2220–2260) |
| `--sample` | legacy sampled-training cap for `--what dual-train` ONLY; refused on a suite (`--tracks-config`) and on plain `--what train` — a sampled suite is PREPARED with `model_tracks.smoke_inputs.prepare_smoke` (writes a child `suite.yaml` over a frozen parent split) then launched by naming that `suite.yaml` (colab.py:2241–2246, suite guard colab.py:2379–2389) |
| `--resume-run <id>` | resume `concurrent_train_<id>` or a prior suite run (needs the original `__inputs` archive, colab.py:2261–2265); not supported by dual-train (colab.py:2644–2645) |
| `--resume-hpo`, `--hpo-persistence {local,none}`, `--hpo-mode`, `--hpo-jobs` | Optuna durability/schedule (colab.py:2266–2298) |
| `--gpu <name>` | accelerator request; `CPU` is the safe default (colab.py:2277–2281) |
| `--allow-gpu` | acknowledgement before any non-CPU runtime (colab.py:2282–2286) |
| `--keep-alive` | CPU-only: no teardown on completion/failure; refused for GPU (colab.py:2304–2308) |
| `--refresh-data` | regenerate frozen CSVs before training; banned with suites (colab.py:2352–2353, 2571–2572) |
| `--preflight-only` | offline lifecycle validation; suite preflight for tracks, train/smoke lanes otherwise (colab.py:2369–2378, 2442–2445) |
| `--train-only` | skip post-training validation inference; banned with suites (colab.py:2352–2353) |

`--what bundle` is the legacy upload lane with a thin passthrough: when
`training.yaml bundle_prep.lane` is set it forwards to
`cli.colab_data_bundle_prep.run_bundle_prep` (owner ruling 8; colab.py:2602–2609).

### Rebuild the full all-track inputs package

To replace an old `all_tracks_inputs.tar.zst` after a packaging-path fix, run
the production preparation on Colab's high-memory CPU lane with the full export:

```bash
er-colab --what bundle --gpu CPU
```

Without `--dataset-csv`, the lane uses `config/paths.yaml` `files.dataset`
(`repo:dataset.csv`). This runs `training.prepare_all` on the remote CPU VM and
delivers a newly created
`results/training_prep/<run-id>/all_tracks_inputs.tar.zst` locally. Use that
archive explicitly with `--prepared-input-package` for the next tracks launch;
the default may select another installed bundle first.

CPU standalone facades (relaunch/production entries, not first launches):

```bash
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --preflight-only   # plan receipt only
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv dataset_3k.csv \
  --resume-from validation --resume-run-id <frozen_id> --resume-state <tar.zst>
PYTHONPATH=src .venv/bin/python -m cli.colab_data_bundle_prep --dataset-csv dataset.csv
```

`cli.colab_bundle` flags (colab_bundle.py:94–118): `--dataset-csv` (default
repo-root `dataset.csv`; must be a `kaggle.export_csvs` entry — no upload
exists on this lane), `--resume-from {dedupe,validation,full_bundle,suite_inputs}`,
`--resume-run-id`, `--resume-state` (triplet all-or-none, colab_lane.py:125–145),
`--preflight-only`. `cli.colab_data_bundle_prep` (colab_data_bundle_prep.py:102)
takes `--dataset-csv` only; it streams every chunk into the ONE per-session
transcript that `ColabSpec.transcript_name` derives from
`EUROMONITOR_COLAB_SESSION` and is the relaunch entry that needs an already-live
session with a checkout (runbook "Do not").

CPU committed-export delivery (offline plan / dry run added 2026-10-06):
```bash
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --preflight-only   # plan receipt only
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv dataset_3k.csv
```

## Capability matrix (likely looked up in `src/cli/colab.py`)

| intent | rule | anchor |
|---|---|---|
| `--sample` on a suite (`--tracks-config`) | rejected: `--sample does not apply to an all-track suite; prepare a sampled suite with model_tracks.smoke_inputs.prepare_smoke and launch its suite.yaml` — the suite trains the frozen inputs it is handed and has no `--sample` reader, so the flag would silently no-op | colab.py:2379–2389 |
| sampled `--what train --sample N` | rejected: `sampled --what train is retired; prepare a sampled suite with model_tracks.smoke_inputs.prepare_smoke and launch it via --tracks-config` | colab.py:2455–2460 |
| `--sample` live reader | only `--what dual-train --sample N` (legacy sampled dual-train); every other lane refuses or ignores it | colab.py:2698 |
| legacy `--sample` bundle rebuild | rejected: `legacy sampled preparation does not preserve the shared component split` | colab_bundle_prewarm.py:335 |
| legacy `--what smoke` (no tracks-config) | raised before provisioning: `legacy smoke does not preserve the shared component holdout; use --what tracks --tracks-config ...` — the smoke body behind this branch is unreachable | colab.py:2413–2416; dispatch 2614 |
| `--tracks-config` on other `--what` | `--tracks-config applies to train/tracks/smoke only` | colab.py:2351 |
| suite device vs `--gpu` | `suite.device and --gpu must agree`: `--gpu CPU` requires `suite.device: cpu`; any accelerator request (e.g. T4) maps to `cuda`. **Baked default (S/M/L matrix): a `--gpu` request that disagrees with the tracked suite AUTO-generates the scratch clone for the requested device** — `results/model_tracks/<suite>__gpu/suite.yaml` from a device-cpu suite, `results/model_tracks/<suite>__cpu/suite.yaml` from a device-cuda suite (only the yamls are copied — every data binding stays under `data/`; the tracks gate then validates the clone), and the launch proceeds with the flipped config. Opt out with `ER_SUITES_KEEP_DEVICE=1` to get the raw must-agree error | colab.py device gate; `_suite_device_flip`; schemas.py `SuiteDeviceFlip.suffix_for` |
| full-cohort bundle default | the default `--what tracks` launch resolves an unspecified `--prepared-input-package` to `results/kaggle_lane/full/bundle` (`default_prepared_input_package`); an explicit `--tracks-config` (CPU smoke) or `--prepared-input-package` wins. **The source/config freshness gate is removed** (no `freshness.json`, no `SuiteFreshnessManifest`, no `ER_SKIP_CONFIG_VERIFY`): a reused package is verified for archive integrity only | colab.py `default_prepared_input_package` / `resolve_prepared_input_package`; model_tracks/package.py `verify` |
| S/M/L dataset matrix | baked SSOT in `canonical_suite_matrix()`: `S=smoke_200` (device cpu), `M=3k` (device cpu; suite config + prepared writer are prep-stage follow-ups), `L=full` (`config/model_tracks.yaml`, device cuda). ADDITIVE ONLY: unknown suite configs keep working — the matrix supplies labels and the device-flip pattern, it never restricts | schemas.py `SuiteMatrixSpec`/`canonical_suite_matrix` |
| `--allow-gpu` | non-CPU `--gpu` without it: `GPU launch requires --allow-gpu` | colab_lane.py:202–203 |
| `--keep-alive` on GPU | refused: `--keep-alive is CPU-only (a retained GPU VM consumes accelerator quota indefinitely)` | colab_lane.py:204–208 |
| `--refresh-data` / `--train-only` with a suite | banned: suites require prepared inputs + full postprocessing | colab.py:2352–2353 |
| bundle export membership | CPU committed-export delivery refuses exports outside `config kaggle.export_csvs` | colab_lane_cpu_provision.py:44–46 |
| smoke parent | `data/prepared/smoke_200` is the established smoke parent: 200 listings frozen, epochs 1, `report_test: false`, publishing off, profiling off (a lifecycle check, not a performance run); post-training ablation stays on so the local finalize fires from the downloaded archive (config binding `colab.smoke_dir`, schemas.py:3127; written by `model_tracks.smoke_inputs.prepare_smoke`) | smoke_inputs.py:72; config/training.yaml:1291 |
| smoke sample floor | the bootstrapped supervision-graph selection must fit the sample: `sample too small for selected smoke supervision` (smoke_inputs.py:158); records show the floor across this parent at ≥ 70 listings |
| smoke retention | smoke results land as `smoke_<run_id>` under TRAINING_RESULTS; `replace_smoke` deletes every older `smoke_*` sibling — the newest smoke is the only local smoke run | colab_result_sync.py:559–561; model_tracks/run_retention.py:166–188 |
| bundle delivery root | delivery archive lands at `TRAINING_RESULTS/colab_bundle_<run_id>/bundle_delivery.tar.zst` | colab_lane_contracts.py:139; colab_lane.py:185 |
| parallel CPU prep cap | exactly two concurrent CPU prep sessions (`MAX_PARALLEL_PREP_SESSIONS = 2`); each isolated session streams into its OWN transcript (`ColabSpec.transcript_name` renders `colab.log_name_template` for `EUROMONITOR_COLAB_SESSION`) — one file per lane, concurrent sessions never truncate one another | colab_lane_contracts.py:42; colab_lane.py:145–156 |
| baked-in self-watch | every executed remote `--what` run that passes its pre-provisioning gates spawns a detached self-watch after the first healthy provisioning stream — no operator arg. It polls the session listing (the `colab sessions` surface main/stop read) until the session is absent, verifies the run's delivery/retention artifacts under TRAINING_RESULTS (captures the lane transcripts on a failed delivery), and runs the lane's own `stop()` release guarantee if the launcher died with the session still listed; release on every terminal outcome; receipts `TRAINING_RESULTS/self_watch_<run_id>/self_watch_receipt.json`, poll log `logs/colab/self_watch_<run_id>.log` | colab.py `spawn_self_watch` / `self_watch` |

## Default self-watch (owner order 2026-10-07)

Session release + result delivery is baked in, mirroring the kaggle lane's
autowatch. Dry-run, `--preflight-only`, `--what stop`, and an explicit
`--keep-alive` retention request never spawn a watcher. One canonical spawn
point: `cli.colab.spawn_self_watch` (tests reference it).

## Self-watch setsid launch contract

Spawned exactly like the kaggle lane's autowatch:

```python
subprocess.Popen(
    [sys.executable, "-m", "cli.colab", "--what", "self-watch",
     "--self-watch-what", what, "--self-watch-run", run_id],
    cwd=TRAIN_ROOT, env=environment, stdout=handle,
    stderr=subprocess.STDOUT, start_new_session=True)
```

`start_new_session=True` (setsid) survives wrapper timeouts and shell
deaths; the runbook's "never `python -m cli.colab` for a launch" rule is
unchanged — the watcher is not a launch and provisions nothing.

Stale-string correction: the legacy-gate error text (colab.py:294, 2415)
still cites `results/model_tracks/smoke_20261001_128/suite.yaml`; that path no
longer exists on disk (results/model_tracks holds only `inputs/`). Sanctioned
CPU-smoke command, exactly as the gate intends:

```bash
er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU
```

## Bundle boundaries (one integrity check per VM crossing)

The Colab lane itself hashes nothing role-related; it drives the shared
boundaries:

| crossing | contract | where |
|---|---|---|
| local → VM prepared inputs | role `inputs` Bundle, ONE `Bundle.load` (manifest + member digests) then a `reader()` install | `model_tracks/package.verify` (package.py:606) and `model_tracks/colab.run` (model_tracks/colab.py:154) + the VM bootstrap `Bundle.load(archive_path, BundleRole.inputs).reader()` (model_tracks/colab.py:216) |
| sealed inputs → GPU trainer | trust boundary, not an integrity crossing: the bytes were checked once above; the trainer asserts the bundle's CPU `seal` (sidecar manifest) and recomputes nothing — no/partial seal fails loud | `training.prepared_bundle.load_prepared_bundle` (prepared_bundle.py), sealed pre-delivery by `model_tracks.package` via `seal_prepared_bundle` |
| resume state | role `recovery` Bundle | `model_tracks/colab.py:177` (opened once, handed to the transport builder) |
| VM → local suite result (tracks flow) | role `result` Bundle, PINNED digest | `model_tracks/colab.verify_result_archive` (model_tracks/colab.py:21–31), used at the download boundary (model_tracks/colab.py:331) |
| CPU delivery archive (prep lane) | transport wrapper, NOT a role Bundle | remote writer `record_digest_script`, local `verify_transport_digest` (cli/colab_lane_cpu_delivery.py:203–227); the sidecar suffix is `training_cfg().bundle.size_sidecar_suffix` |
| legacy multi-worker train result | `ResultBundleManifest` (core.schemas) | `_prepare_remote_result_archive` / `_verify_result_bundle` (cli/colab_result_sync.py:35, 164), which covers the concurrent-worker layout that carries no suite manifest |

The first three are Bundle role crossings; the last two are the documented
alternative shapes (`cli/colab_bundle_transport.py:9–28`), so the rule is "one
integrity check per crossing", not "a role archive for every archive".

## Precondition checklist (Colab)

1. Colab CLI installed/authorized; launch lock available.
2. `origin/main` has every file the VM must see pushed (VMs clone the branch;
   the working tree is invisible to them).
3. Committed exports for the CPU lane must appear in `config kaggle.export_csvs`
   (colab_lane_cpu_provision.py:44–46).
4. Suites require their prepared `__inputs` package (newly packaged, supplied
   via `--prepared-input-package`, or resolvable from `--resume-run`).
5. Do not run any other Colab session while two CPU prep lanes hold the
   two-session cap (runbook).

## Runtime env / edge rules worth pinning

- Prepared-bundle seal (owner order 2026-10-12: no CPU-only process work on GPU
  lanes): the CPU bundle process is the sole preflight owner. `write_prepared_bundle`
  stamps the build seal; `model_tracks.package` recomputes the full contract over the
  serialized bytes (`training.prepared_bundle.seal_prepared_bundle`) and the sealed
  sidecar ships inside the inputs archive. The GPU seam (`load_prepared_bundle`) reads
  the manifest + payload and asserts the seal only — a missing or partial seal fails
  loud naming the absent checks; nothing recomputes on the accelerator. CPU consumers
  (suite preflight, handoff, diet gate, analysis scripts, smoke parent build) call
  `verify_prepared_bundle` (today's full recheck, no seal requirement).
- `PREPARED_BUNDLE_DRIFT_STRICT` (src/training/prepared_bundle.py): retired policy
  record (owner order 2026-10-07) — the env form still parses, nothing acts on it;
  masking/easy provenance is recorded in the manifest and never compared in any lane.
- `ER_SUITES_KEEP_DEVICE=1` (src/cli/colab.py `_suite_device_flip`): opt-out
  for the baked automatic device flip — the suite gate then reports the raw
  `suite device and --gpu must agree` error instead of generating the clone.
  Both directions are baked: `<suite>__gpu/` from a device-cpu suite and
  `<suite>__cpu/` from a device-cuda suite. Unset = flipped (default).
- Full-cohort policy: `--what tracks` with no explicit `--tracks-config`/
  `--prepared-input-package` resolves to the prebuilt
  `results/kaggle_lane/full/bundle` (`default_prepared_input_package`). The
  source/config freshness gate and `ER_SKIP_CONFIG_VERIFY` are gone; a reused
  package is checked for archive integrity only.
- Every Colab status/log line carries a Europe/Paris local stamp (CET/CEST, e.g. `[colab 2026-10-07T09:13:28 CEST]`) and goes through the timestamped live-log opener
  (`start_live_log`, called once in the common dispatch so every `--what` run — CPU and GPU alike — shares it; `src/cli/colab.py`)
  into the ONE declared lane transcript `logs/colab/<log_name>`: the basename is `ColabSpec.transcript_name(session)` and the
  declarations live in the `colab.lanes` block of `config/training.yaml` (each declared lane carries its own `log_name`; the
  configured session keeps `colab.default_log_name`, an isolated `EUROMONITOR_COLAB_SESSION` gets `colab.log_name_template`).
  The retired split (`system.log` / `training.log` / `colab_setup_timing.log`) is gone since 08bb0e4 — one file per lane, opened
  (truncated) once at t=0, everything else appends. One
  canonical logs root (owner order 2026-10-07). All streamed and polled CPU/GPU
  output goes through the single `ColabProgressOutput.forward` entry point
  (`src/cli/colab_progress.py`): carriage-return tqdm frames stay live on the
  terminal and are expanded into grep-able lines in the same lane transcript.

## Sparse checkout (prepared runtime)

The default single-worker `train` lane uploads the locally prepared bundle
listed in `config/training.yaml` under `colab.full_prepared_bundles`; it does
not rebuild training data for each GPU launch. GPU requests that need a new
training bundle, an all-track package, or raw HPO/mixed preparation are refused
before VM setup. Prepare those inputs separately on CPU and configure the
resulting artifact before launching GPU training.

The Colab lane uses a **sparse checkout** (`git sparse-checkout set --no-cone`) to fetch only the files the suite needs — never the full repo. The base patterns are:

```
/src/  /config/  /scripts/  /artifacts/wheels/  /artifacts/evidence/
/pyproject.toml  /requirements.txt  /colab_backend.py  /model_tracks_package.json
```

For a prepared-train launch (`--what tracks --prepared-input-package ...`), two more paths are appended from the suite config: the `suite_git_inputs` archive and the resolved text-model directory (e.g. `artifacts/models/all-MiniLM-L6-v2`). The clone is `--depth=1 --single-branch --filter=blob:none --no-tags` (colab_runtime.py:168–245).

**Runtime checkout contract.** Because the VM never clones the full tree, every
path the remote stage needs from the checkout — the declared directories, every
repo-root file it opens at `REMOTE_ROOT`, and each launch's published inputs
transport and text-model directory — must exist in the pushed branch. Adding a
dependency means adding it to `RUNTIME_REQUIRED_ROOT_FILES` (or the per-launch
`extra_paths`) in the same change that starts reading it.
`validate_runtime_checkout()` runs locally before `ensure_session` and checks
the whole contract against `origin/<branch>` (falling back to `HEAD`), so
anything missing or unpushed fails loud with the path and the fix instead of a
`FileNotFoundError` mid-recovery on the VM. The list is covered by
`tests/test_colab_sparse_checkout.py`.

The patterns are built by `runtime_checkout_paths()`
(`src/cli/colab_runtime.py:174–185`) and applied in `prepare_remote_layout()`
(`:222`, the `git sparse-checkout set --no-cone` call at `:255`). Per-lane
extras are appended by the caller: the CPU bundle lane adds the resolved
base-model directory + `data/prepared/smoke_200` + the committed export
(`colab_lane_cpu_provision.py:51`), the prepared-train lane adds the suite
archive + resolved text-model directory. Branch is `colab.branch`
(`config/training.yaml:347`, currently `main`) — the VM clones `origin/main`,
so committed-and-pushed is the only state the checkout ever sees, and it is the
same branch the Kaggle kernels' sparse checkout fetches (`docs/kaggle-lane.md`
§ Sparse checkout).

A regression to remember: `e1fc9e0` committed the bundle manifest
`model_tracks_package.json` at the repo root assuming "sparse checkout is
disabled"; the prepared lane has always sparse-checked out
(`prepared_runtime=True`), so the manifest stayed off the VM and `tracks_recovery`
read nothing.

## Artifacts and paths

| artifact | path |
|---|---|
| suite input package | `results/model_tracks/<run_tag>__inputs.<fmt>` (tar/format from the suite) |
| suite recovery | `results/model_tracks/<run_tag>.recovery.tar.zst` |
| prepared supply | `data/prepared/{full,smoke_200}` (DELIVERY_PREPARED_DIRS, colab_lane_contracts.py:43) |
| remote prepare log | `prepare_bundle.log` / `prepare_bundle.status` at the VM checkout root; budget `PREPARE_BUDGET_SECONDS = 4h` | 
| delivery archive | `TRAINING_RESULTS/colab_bundle_<run_id>/bundle_delivery.tar.zst` (this lane's pinned run dir + `data/` members + `track_setup` + `prepared/{full,smoke_200}`, composed from the ONE declared manifest — `ColabDeliveryManifest`, missing members refuse the delivery) — auto-promoted after download (`promote_delivery`: installs `data/` + `results/training_prep/<run>`, stages + publishes the hosted `bundle` dataset, both lanes run-ready; laya corpus = named follow-up) |
| suite outputs | `results/model_tracks/<run_tag>/` |
| smoke outputs | `TRAINING_RESULTS/smoke_<run_id>/` (newest-only) |
| lane transcripts | ONE file per lane: `logs/colab/<colab.lanes.<lane>.log_name>` under the canonical logs root — basename via `ColabSpec.transcript_name(session)`, declarations in `config/training.yaml` `colab.lanes`; opened once at t=0 by `cli.colab.start_live_log`. Do not redirect `nohup`/wrapper stdout onto the transcript — the launcher already holds the append fd and a second redirection duplicates every line through the same file |

## What does X run now? (intent → command → surface → artifacts → teardown)

| intent | command | remote surface | artifacts | teardown |
|---|---|---|---|---|
| Colab CPU smoke | `er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU` | CPU VM, sparse checkout + git-published suite inputs | suite outputs under `results/model_tracks/<tag>/`, smoke retention `smoke_<id>`, transcript `logs/colab/` + `colab.lanes.bundle.log_name` | VM released in `finally` |
| Colab GPU smoke | `er-colab --what tracks --tracks-config results/model_tracks/smoke_200__gpu/suite.yaml --prepared-input-package <sealed-inputs.tar.zst> --gpu T4 --allow-gpu` — the device-cpu fixture auto-generates the scratch cuda clone `results/model_tracks/smoke_200__gpu/suite.yaml` on first request (only yamls; data stays under `data/`; `ER_SUITES_KEEP_DEVICE=1` opts out). A GPU launch never packages locally (boundary gate `enforce_data_preparation_boundary`, colab_lane.py:216-232 — b00a8f5): the sealed inputs package comes from the CPU lane first — `--gpu CPU` launch, `er-colab --what bundle --gpu CPU`, or `python -m model_tracks.package --config <suite> --output <archive>` on the CPU box; the package step seals the bundle sidecar (`seal_prepared_bundle`) so the GPU trainer trusts it (see "Prepared-bundle seal") | GPU VM, flipped clone suite + uploaded sealed package | `results/model_tracks/<tag>/` + suite outputs, transcript `logs/colab/lane_<session>.log` | VM released |
| Colab full train | `er-colab --what tracks --tracks-config config/model_tracks.yaml [--prepared-input-package ...] --gpu T4 --allow-gpu` | GPU VM, `model_tracks.run` from git-published inputs | `results/model_tracks/<tag>/` + verified inputs archive | VM released; `--keep-alive` refused |
| Colab mixed / hpo / sims | `er-colab --what mixed|hpo|sims` (+ `--resume-hpo`, `--hpo-jobs` for hpo) | GPU VM legacy lanes (direct result transport) | legacy result bundles under TRAINING_RESULTS | VM released |
| Colab bundle prep (3k/full) | runbook systemd-run block: `python -m cli.colab --what bundle --gpu CPU --keep-alive --dataset-csv dataset_3k.csv` (cap: 2 parallel) | high-RAM CPU VM, sparse checkout, cohort remap, remote `training.prepare_all` | `colab_bundle_<run_id>/bundle_delivery.tar.zst` + training_prep run dir contents | left open by directive (`--keep-alive`); stop via the launcher lock lane |
| Colab committed export | `scripts/run_colab_bundle.sh [dataset.csv] [flags]` → `python -m cli.colab_bundle --dataset-csv ...` (extra flags pass through: `--resume-*`, `--keep-alive`, `--preflight-only`, `--install-delivery`) | high-RAM CPU VM, sparse checkout carries the cohort export (no upload) | `colab_bundle_<run_id>/bundle_delivery.tar.zst` + training_prep run dir, auto-promoted | released after delivery; `--keep-alive` opts out |
| Colab relaunch CPU prep | `python -m cli.colab_data_bundle_prep --dataset-csv ...` (needs a live session with a checkout) | existing CPU VM | same as bundle prep | none (relaunch lane — the next launch consumes the live session) |
| Colab stop | `er-colab --what stop` | — | — | VM release requested; warns if it may still be live (colab.py:2016–2081) |

TBD resolved (owner order 2026-10-07): `colab_bundle_10k_sparse.log` and the
other stray root `*.log` files were moved under `logs/colab/` — the canonical
logs root lane transcripts; the old root copies are deleted.
`scripts/run_colab_smoke.sh` contents beyond its `--what smoke` call
(now legacy — the gate at colab.py:2413 rejects it; prefer the tracks-config
command above).

## Fix inheritance

Delivered behavior (bugfix granularity) is unchanged from the
pre-consolidation lanes. git-history fixes carried into the consolidated
classes: no-upload committed export remap (b1f116f), unbuffered + pathlib-
joined prepare telemetry (8b41dba/1952b71), proven POPEN+status polling with transient-probe tolerance
(be6672b/58aa827/31083da/46e8324), delivery member list (8ddc614), delivery
root contract (4d40d1e), resume triplet (d264af1), recovered step_trace
tracebacks (45547e6), serialized control requests (1943073), both-transcript
worker forwarding (1e18b36), probe destructor-noise suppression (d76f39f),
branch fetch instead of commit (2a3baa9/b370ff1), bounded prewarm waits +
interpreter handover (6892ca2), keep-alive boundary (3fdc039/18ffe7a),
session-qualified transcripts (d5d96d4), lock released on session loss
(378addd), setup timing (33a535b). The preparation pin system is gone by
owner ruling (18c8d9c); eae4d1d was the pin-era re-write fix and no longer
applies.

## References

- [docs/kaggle-lane.md](kaggle-lane.md) — the model for this shape
- [docs/runbook.md](runbook.md) — launch discipline (systemd-run, live-session rules)
- [docs/training.md](training.md) — networks and runbook for training
