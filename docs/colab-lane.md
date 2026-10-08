# Colab lane

Consolidated CPU + GPU remote-training lanes for Colab. The CPU bundle
generation is happy in the dedicated bundles lane and the kaggle lane is the
network transport for other data. Everything Colab shaped lives here.

## Structure (2026-10-06 consolidation)

| file | role |
|---|---|
| `src/cli/colab_lane.py` | the lane classes: `ColabLaneBase` (transport dial-ins + receipts + shared contracts), `ColabCPULane` (committed-export delivery + CPU prep parity), `ColabGPULane` (accelerator/retention boundary gates) |
| `src/cli/colab_lane_contracts.py` | shared lane constants (`DELIVERY_PREPARED_DIRS`, `MAX_PARALLEL_PREP_SESSIONS`, `DELIVERY_ARCHIVE_NAME`) + the `ColabLaneBase` contracts (delivery root/member list, checkout guard, resume triplet) |
| `src/cli/colab_lane_cpu_provision.py` | CPU lane provisioning order + byte-exact prepare-launch script segments (committed-export membership guard) |
| `src/cli/colab_lane_cpu_delivery.py` | delivery archive, launch/poll/collect/download phases, frozen resume-state upload |
| `src/cli/colab_lane_cpu_poll.py` | VM prepare-log poll (offset probes, transit tolerance, dual transcripts) |
| `src/cli/colab.py` | GPU launcher surface facade (`train / tracks / dual-train / hpo / sims / mixed / smoke / stop`) plus the legacy `--what bundle` upload lane |
| `src/cli/colab_self_watch.py` | detached session self-watch (spawned default on executed remote runs: poll → delivery proof → release guarantee; receipt under `TRAINING_RESULTS/self_watch_<run_id>/`) |
| `src/cli/colab_retention.py` | local HPO report/snapshot retention (one-roof exported from the launcher surface) |
| `src/cli/colab_bundle.py` | CPU committed-export delivery facade |
| `src/cli/colab_data_bundle_prep.py` | CPU data-bundle prep facade (high-RAM shape, dual transcripts, 2-parallel cap) |
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
| `--sample` | sampled-training cap; requires `--tracks-config` frozen parent splits (colab.py:2235–2239) |
| `--resume-run <id>` | resume `concurrent_train_<id>` or a prior suite run (needs the original `__inputs` archive, colab.py:2261–2265); not supported by dual-train (colab.py:2644–2645) |
| `--resume-hpo`, `--hpo-persistence {local,none}`, `--hpo-mode`, `--hpo-jobs` | Optuna durability/schedule (colab.py:2266–2298) |
| `--gpu <name>` | accelerator request; `CPU` is the safe default (colab.py:2277–2281) |
| `--allow-gpu` | acknowledgement before any non-CPU runtime (colab.py:2282–2286) |
| `--keep-alive` | CPU-only: no teardown on completion/failure; refused for GPU (colab.py:2304–2308) |
| `--refresh-data` | regenerate frozen CSVs before training; banned with suites (colab.py:2352–2353, 2571–2572) |
| `--preflight-only` | offline lifecycle validation; suite preflight for tracks, train/smoke lanes otherwise (colab.py:2369–2378, 2442–2445) |
| `--train-only` | skip post-training validation inference; banned with suites (colab.py:2352–2353) |

`--what bundle` is the legacy upload lane with a thin passthrough: when
`training.yaml colab.cpu_bundle_prep.lane` is set it forwards to
`cli.colab_data_bundle_prep.run_cpu_bundle_prep` (owner ruling 8; colab.py:2602–2609).

CPU standalone facades (relaunch/production entries, not first launches):

```bash
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --preflight-only   # plan receipt only
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv dataset_50pct.csv \
  --resume-from validation --resume-run-id <frozen_id> --resume-state <tar.zst>
PYTHONPATH=src .venv/bin/python -m cli.colab_data_bundle_prep --dataset-csv dataset.csv
```

`cli.colab_bundle` flags (colab_bundle.py:94–118): `--dataset-csv` (default
repo-root `dataset.csv`; must be a `kaggle.export_csvs` entry — no upload
exists on this lane), `--resume-from {dedupe,validation,full_bundle,suite_inputs}`,
`--resume-run-id`, `--resume-state` (triplet all-or-none, colab_lane.py:125–145),
`--preflight-only`. `cli.colab_data_bundle_prep` (colab_data_bundle_prep.py:102)
takes `--dataset-csv` only; it qualifies per-session transcripts from
`EUROMONITOR_COLAB_SESSION` and is the relaunch entry that needs an already-live
session with a checkout (runbook "Do not").

CPU committed-export delivery (offline plan / dry run added 2026-10-06):
```bash
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --preflight-only   # plan receipt only
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv dataset_50pct.csv
```

## Capability matrix (likely looked up in `src/cli/colab.py`)

| intent | rule | anchor |
|---|---|---|
| sampled `--what train --sample N` | rejected: `sampled training requires --tracks-config with frozen parent splits` — sanctioned path is the suite | colab.py:2235–2239 |
| legacy `--sample` bundle rebuild | rejected: `legacy sampled preparation does not preserve the shared component split` | colab_bundle_prewarm.py:335 |
| legacy `--what smoke` (no tracks-config) | raised before provisioning: `legacy smoke does not preserve the shared component holdout; use --what tracks --tracks-config ...` — the smoke body behind this branch is unreachable | colab.py:2413–2416; dispatch 2614 |
| `--tracks-config` on other `--what` | `--tracks-config applies to train/tracks/smoke only` | colab.py:2351 |
| suite device vs `--gpu` | `suite.device and --gpu must agree`: `--gpu CPU` requires `suite.device: cpu`; any accelerator request (e.g. T4) maps to `cuda`. **Baked default (S/M/L matrix): a non-CPU request against a device-cpu tracked suite AUTO-generates the scratch cuda clone** `results/model_tracks/<suite>__gpu/suite.yaml` (only the yamls are copied — every data binding stays under `data/`; the tracks gate then validates the clone), and the launch proceeds with the flipped config. Opt out with `ER_SUITES_KEEP_DEVICE=1` to get the raw must-agree error | colab.py:2379–2388 device gate; `_suite_device_flip` colab.py:2091; schemas.py `SuiteDeviceFlip` |
| full-cohort bundle default | the default `--what tracks` launch resolves an unspecified `--prepared-input-package` to `results/kaggle_lane/full/bundle` (`default_prepared_input_package`); an explicit `--tracks-config` (CPU smoke) or `--prepared-input-package` wins. **The source/config freshness gate is removed** (no `freshness.json`, no `SuiteFreshnessManifest`, no `ER_SKIP_CONFIG_VERIFY`): a reused package is verified for archive integrity only | colab.py `default_prepared_input_package` / `resolve_prepared_input_package`; model_tracks/package.py `verify` |
| S/M/L dataset matrix | baked SSOT in `canonical_suite_matrix()`: `S=smoke_200` (device cpu), `M=50pct` (device cpu; suite config + prepared writer are prep-stage follow-ups), `L=full` (`config/model_tracks.yaml`, device cuda). ADDITIVE ONLY: unknown suite configs keep working — the matrix supplies labels and the device-flip pattern, it never restricts | schemas.py `SuiteMatrixSpec`/`canonical_suite_matrix` |
| `--allow-gpu` | non-CPU `--gpu` without it: `GPU launch requires --allow-gpu` | colab_lane.py:202–203 |
| `--keep-alive` on GPU | refused: `--keep-alive is CPU-only (a retained GPU VM consumes accelerator quota indefinitely)` | colab_lane.py:204–208 |
| `--refresh-data` / `--train-only` with a suite | banned: suites require prepared inputs + full postprocessing | colab.py:2352–2353 |
| bundle export membership | CPU committed-export delivery refuses exports outside `config kaggle.export_csvs` | colab_lane_cpu_provision.py:44–46 |
| smoke parent | `data/prepared/smoke_200` is the established smoke parent: 200 listings frozen, epochs 1, `report_test: false`, publishing off, profiling on (config binding `colab.smoke_dir`, schemas.py:3127; written by `model_tracks.smoke_inputs.prepare_smoke`) | smoke_inputs.py:72; config/training.yaml:1291 |
| smoke sample floor | the bootstrapped supervision-graph selection must fit the sample: `sample too small for selected smoke supervision` (smoke_inputs.py:158); records show the floor across this parent at ≥ 70 listings |
| smoke retention | smoke results land as `smoke_<run_id>` under TRAINING_RESULTS; `replace_smoke` deletes every older `smoke_*` sibling — the newest smoke is the only local smoke run | colab_result_sync.py:559–561; model_tracks/run_retention.py:166–188 |
| bundle delivery root | delivery archive lands at `TRAINING_RESULTS/colab_bundle_<run_id>/bundle_delivery.tar.zst` | colab_lane_contracts.py:120; colab_lane.py:119 |
| parallel CPU prep cap | exactly two concurrent CPU prep sessions (`MAX_PARALLEL_PREP_SESSIONS = 2`); per-session transcripts `logs/colab/colab_system_<session>.log` / `logs/colab/training_<session>.log` | colab_lane_contracts.py:42; colab_lane.py:150–158 |
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

- `PREPARED_BUNDLE_DRIFT_STRICT` (src/training/prepared_bundle.py:75–89,
  436–466): when a retained bundle's masking/easy-negative config drifted from
  the active config, `1`/strict raises (re-prepare the bundle) and `0`/lenient
  prints `[bundle-drift] WARNING` and proceeds. Use lenient **only** for
  pre-rebuild audit reproducibility; smoke reproduction stays strict.
- `ER_SUITES_KEEP_DEVICE=1` (src/cli/colab.py `_suite_device_flip`): opt-out
  for the baked automatic device flip — the suite gate then reports the raw
  `suite device and --gpu must agree` error instead of generating the
  `results/model_tracks/<suite>__gpu/` cuda clone. Unset = flipped (default).
- Full-cohort policy: `--what tracks` with no explicit `--tracks-config`/
  `--prepared-input-package` resolves to the prebuilt
  `results/kaggle_lane/full/bundle` (`default_prepared_input_package`). The
  source/config freshness gate and `ER_SKIP_CONFIG_VERIFY` are gone; a reused
  package is checked for archive integrity only.
- Every Colab status/log line carries a Europe/Paris local stamp (CET/CEST, e.g. `[colab 2026-10-07T09:13:28 CEST]`) and goes through the timestamped live-log wrapper
  (`start_live_log`) into `logs/colab/system.log` / `logs/colab/training.log`, session-qualified
  to `logs/colab/colab_system_<session>.log` / `logs/colab/training_<session>.log` when
  `EUROMONITOR_COLAB_SESSION` is set (colab.py:119 anchor; colab_lane.py:150–158). One
  canonical logs root (owner order 2026-10-07); tqdm CR frames are expanded to grep-able
  lines at write time by the shared formatter `cli.log_capture.progress_frames_to_lines`.

## Sparse checkout (prepared runtime)

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
| prepared supply | `data/prepared/{full,smoke_200}` (DELIVERY_PREPARED_DIRS, colab_lane_contracts.py:35) |
| remote prepare log | `prepare_bundle.log` / `prepare_bundle.status` at the VM checkout root; budget `PREPARE_BUDGET_SECONDS = 4h` | 
| delivery archive | `TRAINING_RESULTS/colab_bundle_<run_id>/bundle_delivery.tar.zst` (run dir + `data/` members + `track_setup` + `prepared/{full,smoke_200}`) |
| suite outputs | `results/model_tracks/<run_tag>/` |
| smoke outputs | `TRAINING_RESULTS/smoke_<run_id>/` (newest-only) |
| lane transcripts | `logs/colab/system.log`, `logs/colab/training.log`, `logs/colab/colab_setup_timing.log` under the canonical logs root |

## What does X run now? (intent → command → surface → artifacts → teardown)

| intent | command | remote surface | artifacts | teardown |
|---|---|---|---|---|
| Colab CPU smoke | `er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU` | CPU VM, sparse checkout + git-published suite inputs | suite outputs under `results/model_tracks/<tag>/`, smoke retention `smoke_<id>` | VM released in `finally` |
| Colab GPU smoke | now baked: `er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu T4 --allow-gpu` — the lane auto-generates the scratch cuda clone `results/model_tracks/smoke_200__gpu/suite.yaml` (only yamls; data stays under `data/`, `ER_SUITES_KEEP_DEVICE=1` opts out) and validates it with the tracks gates; zero manual scratch prep | GPU VM, flipped clone suite | `results/model_tracks/smoke_200__gpu/` + suite outputs under `results/model_tracks/<tag>/` | VM released |
| Colab full train | `er-colab --what tracks --tracks-config config/model_tracks.yaml [--prepared-input-package ...] --gpu T4 --allow-gpu` | GPU VM, `model_tracks.run` from git-published inputs | `results/model_tracks/<tag>/` + verified inputs archive | VM released; `--keep-alive` refused |
| Colab mixed / hpo / sims | `er-colab --what mixed|hpo|sims` (+ `--resume-hpo`, `--hpo-jobs` for hpo) | GPU VM legacy lanes (direct result transport) | legacy result bundles under TRAINING_RESULTS | VM released |
| Colab bundle prep (50pct/full) | runbook systemd-run block: `python -m cli.colab --what bundle --gpu CPU --keep-alive --dataset-csv dataset_50pct.csv` (cap: 2 parallel) | high-RAM CPU VM, sparse checkout, cohort remap, remote `training.prepare_all` | `colab_bundle_<run_id>/bundle_delivery.tar.zst` + training_prep run dir contents | left open by directive (`--keep-alive`); stop via the launcher lock lane |
| Colab relaunch CPU prep | `python -m cli.colab_data_bundle_prep --dataset-csv ...` (needs a live session with a checkout) | existing CPU VM | same as bundle prep | none (relaunch lane) |
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
