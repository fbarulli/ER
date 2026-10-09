# Kaggle lane

Transport + remote-compute lane. The Colab lanes are consolidated per
[colab-lane.md](colab-lane.md); this lane (like the owner ruling 8
precedent) never imports the colab packages.
Contract + evidence: [kaggle-lane-completion.md](kaggle-lane-completion.md).

## Implementation and error recovery

`cli.kaggle_lane` is the compatibility facade. Operations live in
`KaggleRuntime`, `KaggleDatasets`, `KaggleKernels`, `KaggleOutputs`,
`KaggleMonitor`, and `KaggleChain`, each in its corresponding `kaggle_*.py`
module. `KaggleCLI` dispatches commands; `KernelTemplates` builds standalone
workers; `KernelLifecycle` owns failure archival and download-before-release.

The `kaggle.files`, `kaggle.remote`, and `kaggle.limits` blocks in
`config/training.yaml` own filenames, staging layouts, remote mount paths,
retry/poll limits, and terminal dimensions. Shared output roots come from
`config/paths.yaml`; compression settings come from `archives`. Each staged
worker embeds these validated settings, including the failure finalizer.

On a worker exception (including checkout/setup errors), the worker archives
partial outputs from the configured artifact roots plus its working directory
and logs, using the existing tar.zst writer. If setup failed before the writer
is available, it uses the existing ZIP_DEFLATED transport. Failure archives
have their own SHA-256 manifest and are never published as successful inputs.
The watcher downloads and verifies them before pushing the release stub.
Download and stop failures are retried according to config; exhausted download
failures are recorded and still trigger a release attempt to avoid stranded
compute. Earlier downloads are retained when a new fetch starts.

Every CPU/GPU push starts one watcher. Workers relay stdout/stderr through a
sized terminal so tqdm remains enabled; progress is flushed into `worker.log`
and the live SSE follower's local log without truncation. Newly staged kernels
use this behavior; already-running kernel code is not replaced automatically.

## Ruling (2026-10-06)

**Bundle generation is CPU-only and runs on a Kaggle CPU session — not
locally.** The GPU session only trains.

| Stage | Where |
|---|---|
| Package + upload the raw cohort export | local (this lane) |
| Bundle generation: `prepare_all` → prepared inputs | Kaggle CPU kernel |
| Training + embedding forwards (text / gnn_only; the cascade composes post-training) | Kaggle GPU kernel (tracked, later run) |

## Auth standard (2026-10-07)

`~/.kaggle/kaggle.json` is the single credential. `~/.kaggle/access_token`
must not exist: the kaggle CLI prefers it over kaggle.json and a stale
token there makes every `kernels.*` call fail with `Permission
'kernels.get' denied`. The lane fail-louds (`_require_kaggle_executable`)
until the file is removed; `--what credentials` never writes it.

## Commands

`er-kaggle` (= `PYTHONPATH=src .venv/bin/python -m cli.kaggle_lane`).
Dry-run by default; `--execute` touches the network.

```bash
# 0. one-time: write ~/.kaggle/kaggle.json from the environment
er-kaggle --what credentials --execute

# 1. stage + push the CPU bundle kernel (clones the pinned revision,
#    runs prepare_all, stages the package + receipt)
er-kaggle --what bundle-kernel --execute

# 2. poll
er-kaggle --what kernel-status             # cpu (default)
er-kaggle --what kernel-status --kernel gpu

# 3. fetch the bundle back, sha-verified against the kernel's receipt
er-kaggle --what bundle-fetch --execute

# 4. dataset transport + submission (unchanged)
er-kaggle --what package --dataset-csv dataset_50pct.csv
er-kaggle --what upload --execute
er-kaggle --what download --execute
er-kaggle --what submission --submission-input in.csv --submission-output out.csv
```

## Command inventory (SSOT: `cli.kaggle_lane.main` → `KaggleCLI.run`, src/cli/kaggle_cli.py:13–75)

`er-kaggle` (= `PYTHONPATH=src .venv/bin/python -m cli.kaggle_lane`).
Everything is **dry-run by default**; `--execute` is the only network path.

`--what` choices (kaggle_cli.py:17–22, default `package`):
`package | upload | download | submission | credentials | bundle-kernel |
bundle-fetch | kernel-status | train-kernel | embed-kernel | embed-objective |
finalize-kernel | kernel-logs | fetch-results | stop | supervise | autowatch |
kernel-stream | chain`

Flags (kaggle_cli.py:23–73): `--dataset-csv`, `--config-json`,
`--submission-input`, `--submission-output`, `--execute`, `--no-wait`,
`--kernel {cpu,gpu,embed,finalize}` (default cpu — resolves the configured slug),
`--kind {bundle,train,embed,finalize}` (fetch-results default train;
supervise default ALL THREE when `--kind` is absent, kaggle_cli.py:178),
`--slug`, `--follow`, `--checkpoint`, `--run-tag`,
`--cohort {full,50pct,10k}` (default full),
`--with-embed` (chain: continue into embed-kernel after the train watcher
completes), `--with-finalize` (chain: finish with the remote CPU finalize job),
`--key-env`, `--revision` (pin; default current HEAD).

| operation | what it actually does | preconditions |
|---|---|---|
| `credentials` | writes `~/.kaggle/kaggle.json` (0600) from the env token named by `credentials.keys.kaggle_api_key` (default `KAGGLE_API_KEY`), resolved by `core.credentials.CredentialStore` (process env wins, declared env file second) | `kaggle.username` set; token present; never logged (kaggle_runtime.py) |
| `package` | builds the Kaggle payload (packaged dataset.csv + metadata + `.receipt.json`) into the staging root; hashes census | `--dataset-csv` or the config dataset binding (kaggle_cli.py:23–25; kaggle_datasets.py:36) |
| `upload` / `download` | kaggle CLI dataset upload/fetch-back of the package | `kaggle.slug` configured (null stays fail-loud), executable present (config `kaggle_executable: kaggle`) |
| `submission` | packages the external submission frame keeping only `submission_id_columns` (`sku_id`, `item_id`) | `--submission-input` and `--submission-output` required (kaggle_cli.py:204–205) |
| `bundle-kernel --cohort <c>` | stage CPU kernel (metadata + script + `bundle_kernel.receipt.json`) under `results/kaggle_lane/bundle_kernel`; revision pinned at staging; `--execute` pushes it | `kaggle.cpu_kernel_slug`; the cohort export committed at the repo root and listed in `kaggle.export_csvs` (kaggle_kernels.py:221, 315) |
| `bundle-fetch [--cohort <c>]` | download the CPU kernel output, locate `bundle.receipt.json`, sha256-verify the archive against the receipt/sidecar, install into `results/kaggle_lane/<cohort>/bundle/` with `manifest.json` + `timings.json` sidecars | kernel output available; cohort defaults to the receipt's own cohort, then the config dataset binding (kaggle_outputs.py:207) |
| `train-kernel` / `embed-kernel` | stage GPU kernels (train attaches the CPU kernel as `kernel_sources` + `kaggle.bundle_dataset_slug`; embed attaches `kaggle.embedding_dataset_slug` + the git-shipped checkpoint); `--execute` pushes | `gpu_kernel_slug` / `embedding_kernel_slug`; embed also needs the request dataset uploaded first and `kaggle.checkpoint` in `checkout_paths` (`artifacts/models`) (kaggle_kernels.py:340) |
| `kernel-status [--kernel]` | `kaggle kernels status`; normalizes 2.x `KernelWorkerStatus.` prefixes → `complete/cancelAcknowledged/cancelRequested/running/queued/error` | slug configured; executable present (kaggle_kernels.py:616) |
| `kernel-logs [--kernel --slug --follow]` | poll status at `kaggle.logs_poll_seconds` (15 s); on a terminal state fetch the kernel's own output log into `logs/kaggle` | same as status (kaggle_monitor.py:476) |
| `kernel-stream` | live SSE log follower (kagglesdk `GetKernelSessionLogsStream`); appends decoded payloads to the ONE transcript `logs/kaggle/lane.log` (UTF-8-safe): `/r`-separated tqdm frames are expanded to grep-able lines and the last bar tagged, by the shared formatter `cli.log_capture.progress_frames_to_lines`; reconnects for the WHOLE session, re-truncating its own section because SSE replay restarts at the first line; its reconnect/429 diagnostics ride the same handle | kagglesdk package; returns the `kernel_session_id` the manual kill switch needs (kaggle_monitor.py:230) |
| `fetch-results --kind <k>` | contract fetch + sha256 verification: bundle→`bundle.receipt.json`/`all_tracks_inputs.tar.zst`; train→`result_bundle.manifest.json`/`result_bundle.tar.zst`; embed→`vectors.tar.zst`; installs under `results/kaggle_lane/<cohort>/<kind>/` | the kernel reached output-producing state; `--kind` is singular (kaggle_outputs.py:17) |
| `stop --kernel <k>` | no cancel verb exists on the public CLI: pushes a `cancel_stub.py` version replace — the platform tears the session down to run version N+1 and frees quota | slug configured (kaggle_kernels.py:640, staging `results/kaggle_lane/<which>_stop`); specimens in `results/kaggle_lane/cancel_stub`, `cpu_stop`, `gpu_stop` |
| `supervise --kind <k>` | dependable harvest: one stream thread per kind + status polling to a terminal state; on `complete` fetch + verify; on any other terminal state download and verify partial artifacts plus the session log, and **auto-release the session via the stop-kernel stub replace**; receipt at `results/kaggle_lane/supervise.receipt.json` | slugs configured for every requested kind; polling holds neither session nor quota; idempotent on rerun (kaggle_monitor.py:132) |
| `chain [--cohort <c>] [--with-embed]` | ONE command runs the whole kaggle loop supervised end-to-end: bundle-kernel push (cohort pinned at chain HEAD) → spawned watcher supervises + verifies the fetch + releases → publish default (fresh `fbarulli/er-10k-bundle` version) → train-kernel push via the standard path (single watcher, chain waits on the watcher receipt) → optionally embed-kernel after the train watcher completes; receipt `results/kaggle_lane/chain.receipt.json` with revision pins + fetch shas + dataset version + spawn confirmations | `cpu/gpu/embed` kernel slugs + `kaggle.bundle_dataset_slug` configured; `--execute` (dry-run prints the plan only); no other flags required |
| (any verified fetch) → publishes automatically | after the sha verification of `fetch_kernel_output(kind='bundle'\|'train')` — autowatch, supervise, `bundle-fetch`, `fetch-results`, chain — `publish_bundle_dataset` publishes a fresh `kaggle.bundle_dataset_slug` version from `results/kaggle_lane/<cohort>_bundle_dataset` and records the version + mount pin | verified install present; slug set; cohort slug marker must agree with the fetch's cohort (fail loud otherwise); no hand-invoke |

## Cohorts (kaggle_lane.py:184; kaggle_runtime.py:70; config `kaggle.export_csvs`)

| cohort | export the CPU kernel remaps onto `dataset.csv` | checksum/tag |
|---|---|---|
| `full` | `dataset.csv` | `ER_COHORT_TAG` empty |
| `50pct` | `dataset_50pct.csv` | `ER_COHORT_TAG=50pct` |
| `10k` | `dataset_10k.csv` | `ER_COHORT_TAG=10k` |

The remap is copy-on-checkout — the commit carries the bytes, no upload round
trip (kaggle_kernel_templates.py:146–147). Ablation staging lives in the CPU data bundle:
`training.prepare_all` → `model_tracks.package._prepare_exports` mints the
per-track `ablation_templates/` on the prep machine, so the bundle ships them
and every training suite session (CPU or GPU) forwards from the shipped
templates instead of staging on the accelerator.

## Sparse checkout (remote kernels)

Every pushed kernel clones the configured branch at the pinned revision with
`--filter=blob:none --no-checkout`, then sets `--no-cone` sparse patterns —
never a full tree:

- **bundle kernel**: launch `kaggle_kernel_templates.py:110–121`; patterns
  `checkout_members((*kaggle.checkout_paths, cohort_dataset))`
  (`cli/kaggle_kernels.py:266`).
- **train/embed kernels**: `clone_pinned()`
  `kaggle_kernel_templates.py:229–233`; patterns
  `checkout_members(kaggle.checkout_paths, lane="training")`
  (`cli/kaggle_kernels.py:427`).

`core.runtime_inputs.checkout_members` (`src/core/runtime_inputs.py:22`) is
the SSOT: `src scripts config requirements artifacts/wheels artifacts/evidence
<models_dir> <smoke_dir> pyproject.toml requirements.txt colab_backend.py` plus
the evidence members, then the lane members — `('dataset.csv',)` for
`lane="bundle"` or `('data', 'dataset.csv')` for `lane="training"` (so the
training checkout carries the whole `data/` tree, `data/laya` included).

Branch is `kaggle.branch` (`config/training.yaml:292`, currently `main`): the
kernel clones that branch and fails loud if the tip moved off the pinned
revision (`regenerate the kernel`), so push before staging and re-stage on
drift.

**laya exception.** The laya payloads never clone: `src/cli/laya_lane.py` rides
attached Kaggle datasets (base model, corpus, decisions) and its only git
dependency is the `core.runtime_inputs.require_published_tip_match` gate that
pins `HEAD == origin/<branch>` before it stages.

## Config (SSOT: `config/training.yaml` → `kaggle:`)

`username`, `repository`, `branch`, `cpu_kernel_slug`
(`fbarulli/er-bundle-cpu`), `gpu_kernel_slug` (`fbarulli/er-train-gpu`),
`embedding_kernel_slug` (`fbarulli/er-embed-gpu`),
`embedding_dataset_slug` (`fbarulli/er-embed-requests`),
`bundle_dataset_slug` (`fbarulli/er-10k-bundle`),
`checkout_paths`, `bundle_requirements` (`requirements/graph_tracks.txt`),
`train_suite_config` (`data/model_tracks/suite.yaml`), `checkpoint`
(`artifacts/models`), `logs_poll_seconds` (15.0), `logs_dir` (`logs`),
`run_tag_prefix` (`gpu_`) — plus the transport keys (`slug` — currently `null`,
fail-loud, `export_csvs`, `staging_dir`, `submission_id_columns`,
`kaggle_executable`) (config/training.yaml:182–242).

## Files

| file | role |
|---|---|
| `src/cli/kaggle_lane.py` | the lane (transport + kernels) |
| `kaggle_backend.py` | repo-root shim |
| `src/core/schemas.py` | `KaggleSpec` (additive) |
| `tests/test_kaggle_lane.py` | offline pins |

## Chain op + publish default (owner order 2026-10-07)

`--what chain` runs the whole kaggle loop supervised end-to-end with NO
other flags (only `--cohort` + `--with-embed` shape it; dry-run prints the
entire plan without staging anything or touching the network):

```bash
er-kaggle --what chain --cohort 10k                      # dry run: full plan
er-kaggle --what chain --cohort 10k --execute            # supervised end-to-end
er-kaggle --what chain --cohort 10k --with-embed --execute
```

Steps (every step fails loud with named revision/sha mismatches — the
stale-src standard):

1. **bundle-kernel** — staged + pushed with the cohort pinned at the
   chain's HEAD revision; `push_bundle_kernel`'s own detached autowatch
   supervises the session, sha-verifies the fetch against the kernel
   receipt, releases the session, and drops
   `results/kaggle_lane/autowatch_bundle.receipt.json`.
2. **publish (default, inside the verified fetch)** — see below.
3. **train-kernel** — the SAME revision (any drift between steps fails
   loud with both named); pushed via the exact standard push path
   (`push_kernel` + one `_spawn_autowatch` — never a second spawn; the
   chain waits on `autowatch_train.receipt.json` instead of watching
   again). The train stage attaches the fresh bundle dataset version the
   publish step recorded (`owner/slug/version` pin; unpinned processors
   mount the newest published version automatically).
4. **(--with-embed) embed-kernel** — identical pattern after the train
   watcher reports completion.

The chain receipt (`results/kaggle_lane/chain.receipt.json`) carries: the
per-step revision pin, the fetched archive sha256 per step, the published
dataset version, and the watcher spawn confirmations.

**Publish default**: after ANY successful verified
`fetch_kernel_output(kind='bundle'|'train')` (autowatch, supervise,
`bundle-fetch`, `fetch-results`, or the chain all funnel through it), a
fresh dataset version publishes automatically — no operator hand-invoke.
`publish_bundle_dataset(kind, execute)` builds
`results/kaggle_lane/<cohort>_bundle_dataset` from the VERIFIED install
(the ER 10k bundle precedent: `dataset-metadata.json` + archive + kernel
receipt + `manifest.json`/`timings.json`), runs `kaggle datasets version`
via `_run_kaggle`, then `kaggle datasets status <slug> --format
json(current_version_number)` records the fresh version number plus the
pin-able `owner/slug/<version>` mount form in the plan and the stage's
`publish.receipt.json`. Fail loud on: unset `kaggle.bundle_dataset_slug`,
a cohort marker in the slug that disagrees with the fetched bundle's
cohort (named cohort mismatch), or an install whose archive no longer
matches its own kernel receipt. Train/embed fetch outputs have no bundle
dataset in the SSOT — the plan records a skip note, never a silent
publish to the wrong target.

## Logging convention + artifact paths

- Every lane status line carries a Europe/Paris local stamp (CET/CEST):
  `[kaggle-lane <YYYY-MM-DDTHH:MM:SS CET|CEST>]`;
  every kaggle CLI invocation is echoed and its rc logged (`_run_kaggle`,
  kaggle_runtime.py:154). Console plus the ONE run transcript
  `logs/kaggle/lane.log` (canonical logs root): ER, the laya lane, the watcher
  and the live stream all append to it, truncated once per run.
- Poll cadences: `kaggle.logs_poll_seconds` (15 s) for logs/supervise.

| path | content |
|---|---|
| `logs/kaggle/lane.log` | append-only lane log, Europe/Paris-stamped |
| `results/kaggle_lane/bundle_kernel/`, `train_kernel/`, `embed_kernel/` | staged kernel metadata + scripts + `<kind>_kernel.receipt.json` |
| `results/kaggle_lane/<which>_stop/`, `cancel_stub/` | stop-stub staging |
| `results/kaggle_lane/bundle_fetch/`, `train_fetch/`, `embed_fetch/` | raw CLI-download area before verification |
| `results/kaggle_lane/<cohort>/<kind>/` | VERIFIED installs (archive + manifest + `.sha256`; bundle also `manifest.json`, `timings.json`) |
| `logs/kaggle/lane.log` | the ONE run transcript (ER + laya + watcher + stream) |
| `logs/kaggle/` | separate non-log state/artifacts on the same roof: fetched kernel output logs, `<kernel>.session_id` handles, `*.follower.pid` locks |
| `results/kaggle_lane/supervise.receipt.json` | last supervise plan + history |
| `results/kaggle_lane/autowatch_<bundle\|train\|embed>.receipt.json` | each pushed kernel's own watcher terminal-handler receipt (status, verified fetch with sha + publish plan, stop) |
| `results/kaggle_lane/<cohort>_bundle_dataset/` + `publish.receipt.json` | the publish-default stage dir (dataset-metadata.json + verified archive + receipt) and its dataset version pin record |
| `results/kaggle_lane/chain.receipt.json` | the chain op's plan: per-step revision pins, fetched shas, published dataset version, watcher spawn confirmations |

## Env / edge rules worth pinning

- `PREPARED_BUNDLE_DRIFT_STRICT` (src/training/prepared_bundle.py:75–89,
  436–466): token `'1'` strict (raise on a drifted retained bundle — the
  default for smoke reproduction), `'0'` lenient (`[bundle-drift] WARNING`,
  proceed). Use lenient **only** to keep pre-rebuild audit findings
  reproducible; audits that must regenerate verdicts need strict re-prepared
  bundles.
- The train kernel refuses a moved tip (`branch tip moved: cloned %s but the
  kernel pins %s`) — regenerate the kernel rather than chasing HEAD
  (kaggle_kernel_templates.py:132, 239).
- Stop = version replace, never an in-place cancel: Kaggle's CLI exposes no
  cancel verb (kaggle_kernels.py:640).

## What does X run now? (intent → command → remote surface → artifacts → teardown)

| intent | command | remote surface | artifacts | teardown |
|---|---|---|---|---|
| kaggle CPU bundle (any cohort) | `er-kaggle --what bundle-kernel --cohort full\|50pct\|10k --execute`, then `--what supervise --kind bundle --execute` — or the whole loop via `--what chain --cohort <c> --execute` | CPU kernel: pinned clone → cohort remap → `prepare_all` (stages the per-track ablation templates into the bundle) | `all_tracks_inputs.tar.zst` + `bundle.receipt.json` → verified `results/kaggle_lane/<cohort>/bundle/`; then the publish default (`<cohort>_bundle_dataset` → fresh dataset version) | session ends with the kernel; supervise auto-stops on failure |
| kaggle GPU train chain (train + embed, 10k) | `er-kaggle --what chain --cohort 10k --with-embed --execute` (steps 1–6 manual list replaced by the chain op above) | GPU kernels `er-train-gpu`, `er-embed-gpu` | `train.tar.zst`, `vectors.tar.zst` verified installs; `chain.receipt.json` | watcher release + stop stub on error |
| kaggle status | `er-kaggle --what kernel-status --kernel cpu\|gpu\|embed` | status API only | console JSON | none |
| kaggle logs | `er-kaggle --what kernel-logs --kernel <k> [--follow]` or `--what kernel-stream --kernel <k>` | status polling / SSE stream | `logs/kaggle/*` | stream closes at session teardown |
| kaggle stop | `er-kaggle --what stop --kernel cpu\|gpu\|embed --execute` | version-replace stub push | `results/kaggle_lane/<which>_stop/` | session released |
| dataset transport | `er-kaggle --what package --dataset-csv <csv>` / `--what upload --execute` / `--what download --execute` / `--what submission --submission-input ... --submission-output ...` | kaggle datasets CLI | staging payloads + receipts | none |

## References

- [Kaggle/kagglehub](https://github.com/Kaggle/kagglehub) — Kaggle's Python hub client (dataset/model access)
- [Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api) — the CLI this lane drives (the `kaggle` executable)
- [Kaggle/docker-python](https://github.com/Kaggle/docker-python) — the notebook image; what is preinstalled (torch, transformers, …)
- [abhishekkrthakur/mlframework](https://github.com/abhishekkrthakur/mlframework) — community Kaggle/ML framework reference
