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

# 3. fetch the bundle back, byte-size-verified against the kernel's receipt
er-kaggle --what bundle-fetch --execute

# 4. dataset transport + submission (unchanged)
er-kaggle --what package --dataset-csv dataset_3k.csv
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
`--cohort {full,3k}` (default full),
`--with-embed` (chain: continue into embed-kernel after the train watcher
completes), `--with-finalize` (chain: finish with the remote CPU finalize job),
`--key-env`, `--revision` (pin; default current HEAD).

| operation | what it actually does | preconditions |
|---|---|---|
| `credentials` | writes `~/.kaggle/kaggle.json` (0600) from the env token named by `kaggle.api_key_env` (default `KAGGLE_API_KEY`); also writes the 2.x access-token file | `kaggle.username` set; token present in the env; never logged (kaggle_runtime.py:205) |
| `package` | builds the Kaggle payload (packaged dataset.csv + metadata + `.receipt.json`) into the staging root; byte-size census | `--dataset-csv` or the config dataset binding (kaggle_cli.py:23–25; kaggle_datasets.py:36) |
| `upload` / `download` | kaggle CLI dataset upload/fetch-back of the package | `kaggle.slug` configured (null stays fail-loud), executable present (config `kaggle_executable: kaggle`) |
| `submission` | packages the external submission frame keeping only `submission_id_columns` (`sku_id`, `item_id`) | `--submission-input` and `--submission-output` required (kaggle_cli.py:204–205) |
| `bundle-kernel --cohort <c>` | stage CPU kernel (metadata + script + `bundle_kernel.receipt.json`) under `results/kaggle_lane/bundle_kernel`; revision pinned at staging; `--execute` pushes it | `kaggle.cpu_kernel_slug`; the cohort export committed at the repo root and listed in `kaggle.export_csvs` (kaggle_kernels.py:221, 315) |
| `bundle-fetch [--cohort <c>]` | download the CPU kernel output, locate `bundle.receipt.json`, size-verify the archive against the receipt/sidecar, install into `results/kaggle_lane/<cohort>/bundle/` with `manifest.json` + `timings.json` sidecars | kernel output available; cohort defaults to the receipt's own cohort, then the config dataset binding (kaggle_outputs.py:207) |
| `train-kernel` / `embed-kernel` | stage GPU kernels (train attaches the CPU kernel as `kernel_sources` + the registry's `bundle` dataset; embed attaches the registry's `embeddings` dataset + the git-shipped checkpoint); `--execute` pushes | `gpu_kernel_slug` / `embedding_kernel_slug`; embed also needs the request dataset uploaded first and `kaggle.checkpoint` in `checkout_paths` (`artifacts/models`) (kaggle_kernels.py:391) |
| `finalize-kernel` | stage the remote CPU finalize job (`model_tracks.bundle_steps` role=result) on the bundling CPU slug: one `Bundle.load(..., "result")` boundary on the train kernel's sealed result Bundle, the published inputs bundle attached, `BundlePipeline.finalize` → the sealed `finalized_bundle.tar.zst` + receipt; `--execute` pushes | `cpu_kernel_slug` + `gpu_kernel_slug` + the registry's `bundle` dataset; `--with-finalize` makes the chain do it (kaggle_kernels.py:503) |
| `kernel-status [--kernel]` | `kaggle kernels status`; normalizes 2.x `KernelWorkerStatus.` prefixes → `complete/cancelAcknowledged/cancelRequested/running/queued/error` | slug configured; executable present (kaggle_kernels.py:616) |
| `kernel-logs [--kernel --slug --follow]` | poll status at `kaggle.logs_poll_seconds` (15 s); on a terminal state fetch the kernel's own output log into `logs/kaggle` | same as status (kaggle_monitor.py:476) |
| `kernel-stream` | live SSE log follower (kagglesdk `GetKernelSessionLogsStream`); writes decoded payloads to `logs/kaggle/<slug>.stream.log` (UTF-8-safe): `/r`-separated tqdm frames are expanded to grep-able lines and the last bar tagged, by the shared formatter `cli.log_capture.progress_frames_to_lines`; reconnects ≤5 times, re-truncating because SSE replay restarts at the first line | kagglesdk package; returns the `kernel_session_id` the manual kill switch needs (kaggle_monitor.py:230) |
| `fetch-results --kind <k>` | contract fetch + size verification: bundle→`bundle.receipt.json`/`all_tracks_inputs.tar.zst`; train→`result_bundle.manifest.json`/`result_bundle.tar.zst`; embed→`vectors.tar.zst`; finalize→`finalized_bundle.manifest.json`/`finalized_bundle.tar.zst`. A role kind (`bundle`, `train`, `finalize`) is named by ONE `Bundle.load` at the boundary, so the archive is byte-size-checked once; installs under `results/kaggle_lane/<cohort>/<kind>/` | the kernel reached output-producing state; `--kind` is singular (kaggle_outputs.py:19) |
| `stop --kernel <k>` | session-id-first teardown: the launch-recorded `kernel_session_id` (from every kernel's own self-report line, see below) feeds the SDK's in-place `cancel_kernel_session`; with no recorded id (or an SDK failure) the fallback is a `cancel_stub.py` version replace, where the platform tears the session down to run version N+1 and frees quota. Either path is verified by bounded status polls | slug configured (kaggle_kernels.py:671, staging `results/kaggle_lane/<which>_stop`); specimens in `results/kaggle_lane/cancel_stub`, `cpu_stop`, `gpu_stop` |
| `supervise --kind <k>` | dependable harvest: one stream thread per kind + status polling to a terminal state; on `complete` fetch + verify; on any other terminal state download and verify partial artifacts plus the session log, and **auto-release the session via the stop-kernel stub replace**; receipt at `results/kaggle_lane/supervise.receipt.json` | slugs configured for every requested kind; polling holds neither session nor quota; idempotent on rerun (kaggle_monitor.py:132) |
| `chain [--cohort <c>] [--with-embed] [--with-finalize]` | ONE command runs the whole kaggle loop supervised end-to-end: bundle-kernel push (cohort pinned at chain HEAD) → spawned watcher supervises + verifies the fetch + releases → publish default (fresh `fbarulli/er-10k-bundle` version) → train-kernel push via the standard path (single watcher, chain waits on the watcher receipt) → optionally embed-kernel after the train watcher completes → optionally the remote CPU finalize job (attaches the inputs bundle version the publish recorded + the trained result kernel output). Receipt `results/kaggle_lane/chain.receipt.json` with revision pins + fetch byte sizes + dataset version + spawn confirmations | `cpu/gpu/embed` kernel slugs + the registry's `bundle` dataset configured; `--execute` (dry-run prints the plan only); no other flags required |
| (any verified fetch) → publishes automatically | after the byte-size verification of `fetch_kernel_output(kind='bundle'\|'train')` — autowatch, supervise, `bundle-fetch`, `fetch-results`, chain — `publish_bundle_dataset` publishes a fresh version of the registry's `bundle` dataset from `results/kaggle_lane/<cohort>_bundle_dataset` and records the version + mount pin | verified install present; registry `bundle` role resolvable; cohort slug marker must agree with the fetch's cohort (fail loud otherwise); no hand-invoke |

## Kernel session id (in-place stop)

Kaggle injects no `KAGGLE_KERNEL_RUN_ID`/`KAGGLE_SESSION_ID` into the
container, and the log-stream URL carries no id on the current SDK (probed by
`scripts/kaggle_session_probe.py`). The only per-run id is the numeric suffix
of `KAGGLE_CONTAINER_NAME` (`kaggle_<token>-<session_id>-webtier`), which the
SDK `cancel_kernel_session` accepts. The chain, end to end:

| stage | where |
|---|---|
| self-report | every staged kernel prints `[kaggle-session] session_id=… container=…` at boot: `KernelTemplates.SESSION_REPORT_HELPER` (kaggle_kernel_templates.py:88), injected once per kernel by `render`/stage (kaggle_kernels.py:296, 464, 571) |
| persistence | the SSE follower reads that line off the stream and writes `logs/kaggle/<kernel>.session_id` (`kaggle_monitor.py:357–366`); `capture_kernel_session_id` is the launch-time backup (URL scrape, bounded attempts) |
| consumption | `stop --kernel <k>` feeds the recorded id to the SDK; `KaggleKernels.push_with_session_capture` clears a stale id first, so a fresh push never cancels a previous run's session |

The parse is pinned by `tests/test_kaggle_lane.py`
(`test_kernel_session_report_is_pinned_to_the_host_parser`), so the kernel copy
and the host twin (`cli.laya_lane.container_session_id`) can never drift.

## Finalize step (remote CPU role=result)

The operator box is not a finalize surface: `model_tracks.bundle_steps`
role=result runs as one remote CPU lane job.

1. stage: `stage_finalize_kernel` (`kaggle_kernels.py:503`) attaches the
   published prepared-inputs bundle (the registry's `bundle` dataset, optionally
   `/version`) plus the trained result kernel output (`gpu_kernel_slug`) and
   renders `TRAIN_KERNEL_SHARED + FINALIZE_KERNEL_BODY` under
   `finalize_kernel/finalize_cpu.py` (a second version of the bundling CPU
   slug, one code file per pushed version).
2. run: the kernel reloads the inputs at its `Bundle.load(..., "inputs")`
   boundary (`bundle_steps.finalize`) and the result at ONE
   `Bundle.load(..., "result", expected_size=…)` boundary pinned to the
   manifest byte size the train kernel recorded
   (`kaggle_kernel_templates.py:386–396`), then seals
   `finalized_bundle.tar.zst` + its receipt and `.size` sidecar.
3. fetch: `fetch-results --kind finalize` verifies that archive once more and
   names its role through the same boundary load (`kaggle_outputs.py:176`).

## Ablation templates (GPU inference, never a CPU fallback)

The CPU bundle owns the frozen per-track ablation templates:
`model_tracks.package._prepare_exports` stages them UNCONDITIONALLY when
`post_training_ablation` is on and raises if any `post_training_ablation.ABLATION_TRACKS`
template is missing (package.py:287–303). The train kernel re-checks the
contract at the crossing, from the bundle's own captured member names
(`Bundle.ablation_templates`, kaggle_kernel_templates.py:358–384), and refuses
the session before training when a template is absent; the GPU worker's
`attribute_ablation_export/skipped` branch (worker.py:469) therefore stays a
latent guard instead of a silent local CPU fallback.

**Artifact freshness (checked 2026-10-08).** Every inputs archive already on
this box (`results/model_tracks/*__inputs.tar.zst`,
`results/kaggle_lane/*/bundle/all_tracks_inputs.tar.zst`) predates the cf889bf
ablation fix and carries `post_training_ablation: true` with ZERO
`ablation_templates/` members, so a session started from one of them would skip
the ablation. With the gate above such a session now fails loud at the kernel
instead; the fix is to re-run the CPU bundling kernel (or re-package) so the
bundle carries the templates.

## Cohorts (kaggle_lane.py:184; kaggle_runtime.py:70; config `kaggle.export_csvs`)

| cohort | export the CPU kernel remaps onto `dataset.csv` | checksum/tag |
|---|---|---|
| `full` | `dataset.csv` | `ER_COHORT_TAG` empty |
| `3k` | `dataset_3k.csv` | `ER_COHORT_TAG=3k` |

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

`username`, `api_key_env`, `repository`, `branch`, `cpu_kernel_slug`
(`fbarulli/er-bundle-cpu`), `gpu_kernel_slug` (`fbarulli/er-train-gpu`),
`embedding_kernel_slug` (`fbarulli/er-embed-gpu`),
`checkout_paths`, `bundle_requirements` (`requirements/graph_tracks.txt`),
`train_suite_config` (`data/model_tracks/suite.yaml`), `checkpoint`
(`artifacts/models`), `logs_poll_seconds` (15.0), `logs_dir` (`logs/kaggle`),
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

Steps (every step fails loud with named revision/byte-size mismatches — the
stale-src standard):

1. **bundle-kernel** — staged + pushed with the cohort pinned at the
   chain's HEAD revision; `push_bundle_kernel`'s own detached autowatch
   supervises the session, byte-size-verifies the fetch against the kernel
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
per-step revision pin, the fetched archive size per step, the published
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
`publish.receipt.json`. Fail loud on: the registry `bundle` role being
unresolvable, a cohort marker in the slug that disagrees with the fetched bundle's
cohort (named cohort mismatch), or an install whose archive no longer
matches its own kernel receipt. Train/embed fetch outputs have no bundle
dataset in the SSOT — the plan records a skip note, never a silent
publish to the wrong target.

## Logging convention + artifact paths

- Every lane status line carries a Europe/Paris local stamp (CET/CEST):
  `[kaggle-lane <YYYY-MM-DDTHH:MM:SS CET|CEST>]`;
  every kaggle CLI invocation is echoed and its rc logged (`_run_kaggle`,
  kaggle_runtime.py:154). Console + append-only `logs/kaggle/lane.log` (canonical logs root).
- Poll cadences: `kaggle.logs_poll_seconds` (15 s) for logs/supervise.

| path | content |
|---|---|
| `logs/kaggle/lane.log` | append-only lane log, Europe/Paris-stamped |
| `logs/kaggle/<kernel>.session_id` | launch-recorded `kernel_session_id`: the SDK in-place `cancel_kernel_session` target `stop`/`--session-id` uses |
| `results/kaggle_lane/bundle_kernel/`, `train_kernel/`, `embed_kernel/` | staged kernel metadata + scripts + `<kind>_kernel.receipt.json` |
| `results/kaggle_lane/<which>_stop/`, `cancel_stub/` | stop-stub staging |
| `results/kaggle_lane/bundle_fetch/`, `train_fetch/`, `embed_fetch/` | raw CLI-download area before verification |
| `results/kaggle_lane/<cohort>/<kind>/` | VERIFIED installs (archive + manifest + `.size`; bundle also `manifest.json`, `timings.json`) |
| `logs/kaggle/` | fetched kernel output logs, `<slug>.stream.log` SSE captures (one roof: owner order 2026-10-07) |
| `results/kaggle_lane/supervise.receipt.json` | last supervise plan + history |
| `results/kaggle_lane/autowatch_<bundle\|train\|embed>.receipt.json` | each pushed kernel's own watcher terminal-handler receipt (status, verified fetch with size + publish plan, stop) |
| `results/kaggle_lane/<cohort>_bundle_dataset/` + `publish.receipt.json` | the publish-default stage dir (dataset-metadata.json + verified archive + receipt) and its dataset version pin record |
| `results/kaggle_lane/chain.receipt.json` | the chain op's plan: per-step revision pins, fetched byte sizes, published dataset version, watcher spawn confirmations |

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
- Stop prefers the in-place cancel: every kernel self-reports its
  `kernel_session_id` (see "Kernel session id" above), and `stop` feeds the
  recorded id to the SDK `cancel_kernel_session`; only a missing id (or an SDK
  failure) falls back to the version-replace stub (kaggle_kernels.py:671).

## What does X run now? (intent → command → remote surface → artifacts → teardown)

| intent | command | remote surface | artifacts | teardown |
|---|---|---|---|---|
| kaggle CPU bundle (any cohort) | `er-kaggle --what bundle-kernel --cohort full\|3k --execute`, then `--what supervise --kind bundle --execute` — or the whole loop via `--what chain --cohort <c> --execute` | CPU kernel: pinned clone → cohort remap → `prepare_all` (stages the per-track ablation templates into the bundle) | `all_tracks_inputs.tar.zst` + `bundle.receipt.json` → verified `results/kaggle_lane/<cohort>/bundle/`; then the publish default (`<cohort>_bundle_dataset` → fresh dataset version) | session ends with the kernel; supervise auto-stops on failure |
| kaggle GPU train chain (train + embed, 3k) | `er-kaggle --what chain --cohort 3k --with-embed --execute` (steps 1–6 manual list replaced by the chain op above) | GPU kernels `er-train-gpu`, `er-embed-gpu` | `train.tar.zst`, `vectors.tar.zst` verified installs; `chain.receipt.json` | watcher release + stop stub on error |
| kaggle status | `er-kaggle --what kernel-status --kernel cpu\|gpu\|embed` | status API only | console JSON | none |
| kaggle logs | `er-kaggle --what kernel-logs --kernel <k> [--follow]` or `--what kernel-stream --kernel <k>` | status polling / SSE stream | `logs/kaggle/*` | stream closes at session teardown |
| kaggle stop | `er-kaggle --what stop --kernel cpu\|gpu\|embed --execute` | version-replace stub push | `results/kaggle_lane/<which>_stop/` | session released |
| dataset transport | `er-kaggle --what package --dataset-csv <csv>` / `--what upload --execute` / `--what download --execute` / `--what submission --submission-input ... --submission-output ...` | kaggle datasets CLI | staging payloads + receipts | none |

## References

- [Kaggle/kagglehub](https://github.com/Kaggle/kagglehub) — Kaggle's Python hub client (dataset/model access)
- [Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api) — the CLI this lane drives (the `kaggle` executable)
- [Kaggle/docker-python](https://github.com/Kaggle/docker-python) — the notebook image; what is preinstalled (torch, transformers, …)
- [abhishekkrthakur/mlframework](https://github.com/abhishekkrthakur/mlframework) — community Kaggle/ML framework reference
