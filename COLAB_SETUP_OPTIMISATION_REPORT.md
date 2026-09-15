# Colab setup-path optimisation — measured breakdown and changes

Branch `training`. The substantive commits are **`4f5be42`** (production
change + tests) and **`aaf7838`** (this report + the profiler), pushed as
`3fdc039..aaf7838  training -> training`; any commit after `aaf7838` is a
docs-only correction to this file, so `ER/training` is at or after `aaf7838`.
Baseline before any change: **334 passed, 2 skipped**. After: **357 passed, 2 skipped**
(the 23 extra tests are this change's own; no existing test was modified).

A note on how this was verified: another agent was concurrently editing shared
files in the same checkout (`src/core/schemas.py`, `config/training.yaml`,
`src/core/hard_negatives.py`, `src/pipeline.py`, `src/training/*`) for the
cross-brand mining lane, and at one point their in-flight schema change broke
`training_cfg()` for everyone. Every test run and every file listing reported
here was therefore taken from a throwaway worktree **at my commit**, not from
the live checkout. The shared checkout's uncommitted changes are not mine and
are not in either commit.

The setup path is the launcher work between `session_created` and the first
training cell: `ensure_session` → `prepare_remote_layout` → `install_deps` →
`verify_remote_models` → `log_gpu_profile` → local bundle build → uploads.

---

## 1. Where the wall-clock actually goes

### 1.1 Headline run (the one `colab_system.log` documents)

`my-highram-session.jsonl`, session `2026-09-15T11:31:47`, accelerator T4,
one worker. Reproduce with:

```
PYTHONPATH=src .venv/bin/python scripts/profile_colab_setup.py \
    colab_cli_state/history/my-highram-session.jsonl
```

| # | Phase | Elapsed | Share of setup | Evidence |
|---|---|---|---|---|
| 1 | provision + keep-alive + handshake | 2.44 s | 1.6 % | `session_created` → the `socket.gethostname` exec in the history |
| 2 | checkout (`git fetch`/`clone` + checkout) | 9.30 s | 5.9 % | → the `shutil.rmtree` checkout exec |
| 3 | launcher turnaround | 2.53 s | 1.6 % | → the `00_deps` launch exec |
| 4 | **dependency install (`00_deps`)** | **62.75 s** | **39.8 %** | → the first `[models] key=` exec; the polling loop is collapsed to one interval |
| 5 | remote model validation | 6.41 s | 4.1 % | → the `torch.cuda.get_device_properties` exec |
| 6 | **local prepared-bundle build** | **34.94 s** (33.34 s on the launcher clock) | 22.2 % | → the first `prepared_training` exec; the launcher-clock figure is `[local-prepare] … bundle=…0915T113311296035Z` → `[run] remote metadata recorded …0915T113344635214Z` |
| 7 | **uploads** | **39.13 s** | 24.8 % | → the last `file_operation upload` (bundle manifest, 11:34:25.021) |
| — | **setup total** | **157.5 s** | 100 % | |

Note the 1.6 s gap between rows 6's two figures: the history interval is bounded
by exec round trips (~1.1–1.6 s each, measured in the same run), the
launcher-clock figure is not. Both are reported rather than one being presented
as exact.

### 1.2 Cross-run consistency (every named history file the brief points at)

| History file | Session (UTC) | Accelerator | provision | checkout | deps | models | local prepare | uploads | setup |
|---|---|---|---|---|---|---|---|---|---|
| `gpu-dual-full.jsonl` | 06:32:13 | T4 | 3.8 s | 12.8 s | 78.0 s | 6.9 s | 73.1 s | 136.8 s | 313.8 s |
| `gpu-dual-1k.jsonl` | 06:17:49 | T4 | 3.3 s | 10.7 s | 71.2 s | 6.4 s | 37.0 s | 46.9 s | 177.8 s |
| `cpu-dual-128.jsonl` | 06:24:09 | CPU | 6.6 s | 9.3 s | 84.6 s | 8.1 s | 38.9 s | 40.0 s | 189.2 s |
| `ann-official-0915.jsonl` | 07:43:19 | T4 | 2.6 s | 9.7 s | 76.9 s | 7.5 s | 35.2 s | 39.2 s | 172.7 s |
| `ann-official-full.jsonl` | 07:48:39 | T4 | 2.3 s | 12.2 s | 67.8 s | 6.8 s | 38.4 s | 41.4 s | 170.5 s |
| `ann-gated-3000.jsonl` | 08:55:57 | T4 | 2.6 s | 12.0 s | 72.1 s | 6.8 s | 35.4 s | 44.6 s | 176.0 s |
| `my-highram-session.jsonl` | 11:31:47 | T4 | 2.4 s | 9.3 s | 62.8 s | 6.4 s | 34.9 s | 39.1 s | 157.5 s |

Across **all 60 sessions with a deps stage** in `colab_cli_state/history/*.jsonl`
(medians; the `local_prepare`/`uploads` rows only exist for lanes that use
prepared bundles):

| phase | median | min | max | median share of setup |
|---|---|---|---|---|
| provision | 2.9 s | 2.1 s | 26.8 s | 2.2 % |
| checkout | 9.8 s | −0.3 s | 38.6 s | 7.4 % |
| launcher turnaround | 2.0 s | −6.8 s | 29.1 s | 1.5 % |
| **deps** | **42.0 s** | 7.3 s | 93.8 s | **32.0 %** |
| models | 6.6 s | 1.6 s | 13.0 s | 5.0 % |
| **local prepare** | **21.1 s** | 0.0 s | 364.8 s | **16.1 %** |
| **uploads** | **24.3 s** | 0.0 s | 1146.2 s | **18.5 %** |
| setup total | 131.4 s | 29.2 s | 1539.0 s | |

The aggregate is skewed by short sessions (a handful of deps stages ran in
7–20 s, and two sessions show negative or enormous intervals — an isolated
`--keep-alive` reuse and a resumed lane). That is why §1.1 is the run the
savings are computed against, and the table above is only used to show the
*ordering* of the costs is stable: **deps first, local prepare and uploads
close behind, everything else marginal.**

Two phases the trail cannot bound, stated rather than invented:

* **The launch segment of a single-worker lane.** `run_single_train_and_stream`
  sends one long-lived training cell, so its history record is written when
  training *finishes* (11:53:28 for a launch at ~11:34:25). The profiler omits
  launch gaps over 120 s and says so; there is no usable number here.
* **The internal split of the deps stage** (resolve vs download vs compile).
  The streamed log is not timestamped, so only bounds are available — see §2.

### 1.3 Evidence sources, and the one the brief expected that does not exist

* USED: `colab_cli_state/history/*.jsonl` — per-event ISO timestamps. This is
  the only per-phase timing that exists.
* USED: `colab_system.log` — the streamed stage output and the
  `[local-prepare]`/`[run]` stamp pair.
* **NOT AVAILABLE: `results/logs/colab_stages/*`.** Those files are written on
  the *VM* (`/content/EuromonitoR/results/logs/colab_stages/…`); only the
  streamed copy reaches `colab_system.log`. Verified locally:
  `find . -type d -name colab_stages` returns nothing, and `results/logs/`
  holds only linter/plot directories. So §2 is derived from the 213 streamed
  `[00_deps]` lines, not from the durable remote stage log.
* No timing instrumentation existed in the launcher; nothing was pre-existing
  to read.

---

## 2. What the 62.75 s deps stage is actually spending it on

From the 213 `[00_deps]` lines in `colab_system.log`:

* **35 distributions downloaded, 19.99 MiB total**, of which
  `mlflow` 11.60 MiB + `mlflow-skinny` 3.70 MiB + `mlflow-tracing` 1.80 MiB =
  **17.10 MiB (86 %)**.
* **18 distributions installed**: `hnswlib`, `mlflow{,‑skinny,‑tracing}`, and
  mlflow's transitive `Flask-CORS alembic databricks-sdk docker gitdb
  gitpython graphene graphql-core graphql-relay gunicorn huey
  opentelemetry-proto skops smmap`.
* **124 `Requirement already satisfied` lines** — `sentence-transformers`,
  `datasets`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `wandb` and the
  whole torch stack were already in the image. The launcher asserts nine
  packages; only **two** of them needed real work.
* **One source build**: `hnswlib` ships only an sdist on PyPI, so every fresh
  VM compiles it. Poll offsets bound its build-isolation step at ≈9 s
  (11:32:04 → 11:32:13), and the compile sits inside the 23 s silent window
  between the 11:32:20 and 11:32:44 probes.

Local proxies (same host, `nproc=4`, cold caches, the versions the VM actually
resolved) — clearly *not* VM numbers, used only to size the two items:

| command | wall |
|---|---|
| `pip install --no-cache-dir mlflow==3.16.0 hnswlib==0.8.0` | **73.93 s** |
| `uv pip install --no-cache … mlflow==3.16.0 hnswlib==0.8.0` | **37.16 s** |
| `pip install mlflow==3.16.0` (warm cache) | 42.51 s |
| `pip wheel --no-deps hnswlib==0.8.0` (the compile alone) | 31.15 s |

The two items pip actually installs therefore cost ~74 s here against a 62.75 s
VM stage — the host is slower, which is why §4.4's saving is labelled an
extrapolation.

---

## 3. What changed

### 3.1 Overlap the local bundle build with the VM setup — *estimated 33.3 s*

**What.** `_prepare_local_training_bundles` shells out to
`training.train --prepare-bundle`: pure local CPU over immutable local inputs,
touching nothing on the VM. It used to run *after* the whole remote setup. Now
`main()` starts it before `ensure_session()` and `run_train` joins that exact
build.

**Why it helps.** Measured: on the headline run the build took 33.34 s and the
remote path it can hide under (rows 1–5 plus the upload burst it does not
compete with) is ~118 s. It fits entirely.

**How, without a second implementation.** `_prepare_local_training_bundles`
keeps sole ownership of the build; it first offers the request to
`_take_prewarmed_bundles`, which hands over the in-flight result only when the
request matches exactly (`profiles`, `model`, `sample`, `payload`). The profile
derivation is now `_training_bundle_profiles`, used by both `run_train` and
`main` — the drift that would silently discard a prewarm is designed out, and a
mismatch is *logged and rebuilt* rather than dropped. A build failure is stored
and re-raised inside the owning `run_train` call, and `main`'s `finally` drains
any abandoned build before `close_live_log()`, so a lane that dies before
`run_train` cannot leave a thread writing into a closed log.

**Saving: 33.3 s, estimated.** The build duration and the remote path it
overlaps are both measured; the *overlap itself* is proven offline by a test
that runs the build concurrently with `install_deps` and asserts the thread was
alive during it. It has not yet been observed on a VM.

**Risk: low.** Same bytes, same bundle, same hashes — only the scheduling
changes. Failure modes are a logged rebuild (slow, correct) or a re-raised
error.

### 3.2 Stop re-uploading a 46 MB file the VM already has — *measured 14.7 s, net ≈ 11.9 s*

**What.** `validation_inference.source_csv` is `artifacts/data/dataset_deduped.csv`,
which is **tracked in git** and therefore already at
`/content/EuromonitoR/artifacts/data/dataset_deduped.csv` after
`prepare_remote_layout`. The launcher nevertheless uploaded it into a fresh
`prepared_training/<run_id>/validation/` directory on every run.
`_remote_checkout_copy` now hashes the remote copy and returns it **only when
the digest equals the local file's**.

**Why it helps.** Measured: that upload completed in 14.71 s
(46,428,599 B, 3.16 MB/s) on the headline run. It is the single largest
redundant transfer in setup. The training complement
(`dataset_deduped_train_minus_3000.csv`, 43 MB) stays a genuine upload — it is
gitignored, so the checkout does not have it.

**Saving: 14.7 s measured, ≈11.9 s net.** The probe costs one remote exec per
unmapped input (source and training; `sample` is the same file as `source` and
is mapped by the existing in-run dedupe), measured at ~1.4 s each.

**Risk: low, and content-verified.** Reuse is not assumed from "same branch":
the remote bytes must hash equal. A locally modified input, a missing remote
file, or a probe failure all log and fall back to the normal upload. The
provenance record (`input_provenance.json`) stores `path`, `bytes` and `sha256`,
so the reuse is auditable after the fact — only the recorded path string
changes.

### 3.3 One mkdir exec instead of one per uploaded file — *measured 2.7 s (1 worker), ≈17 s extrapolated (6 workers)*

**What.** `_upload_prepared_bundles` ran one remote `mkdir` exec for the run
directory and then **one more per uploaded file**. All worker directories now
come from a single exec.

**Why it helps.** Exec round trips are not free: measured at 1.09–1.63 s each in
the headline run. For its 1 worker the old code paid 1 + 2 = 3 execs; the new
code pays 1.

**Saving.** 2.7 s measured on the headline run (1 worker). The configured
default is `colab.train_workers: 6`, where the old code paid 1 + 12 = 13 execs
against 1 today: **≈17 s extrapolated** at the same measured per-exec cost.
Labelled extrapolated because no 6-worker history file was available to time.

**Risk: low.** It creates exactly the same directories in one cell, which the
test pins.

### 3.4 Prefer `uv` on the VM, fall back to pip — *measured 2.0× locally; ≈31 s on the VM, extrapolated*

**What.** The detached `00_deps` stage now runs a generated remote program that
tries `uv pip install --python sys.executable <packages>` and falls back to
`python -m pip install <packages>`. Which installer ran — including any
downgrade to pip — is printed into the durable stage log.

**Why it helps.** Two reasons, in order:

1. It is **reuse, not invention**. The installed Colab CLI already does exactly
   this for its own `colab install` subcommand
   (`subprocess.check_call(['uv','pip','install','--system'] + packages)`,
   falling back to pip) — see `commands/automation.py` in
   [google-colab-cli](https://github.com/googlecolab/google-colab-cli). The
   launcher had hand-rolled the pip half of that.
2. Measured on this host, cold caches, the exact pair the VM installs:
   pip **73.93 s** → uv **37.16 s**, i.e. **2.0×, 36.8 s saved**. That ratio is
   *conservative*: ~31 s of both figures is the CPU-bound `hnswlib` compile
   neither tool can avoid, so on resolution+download+install alone uv is ≈7×.

**Saving: ≈31 s on the VM — extrapolated.** Applying the measured 2.0× to the
VM's measured 62.75 s deps stage gives ≈31.5 s. This needs a real run to
confirm; see §5.

**Deviation from upstream, deliberate.** Upstream uses `--system`; this uses
`--python sys.executable`, which is strictly more faithful to today's behaviour:
`sys.executable -m pip install` installs into the kernel's own interpreter, and
`--system` would not if the kernel ever ran inside a virtualenv. The fallback
targets the same interpreter, so the fast path cannot land packages somewhere
the trainer will not import them.

**Also config-driven now.** The two lane package lists were inline string
literals spliced into generated code; they are now `colab.runtime_packages`
(`prepared` / `full`) in `config/training.yaml`, validated by a
`RuntimePackagesSpec` pydantic model with `extra="forbid"` and
non-empty/unique entry checks. `colab.prefer_uv_install` turns the fast path
off. The `[deps] installing …` line now names the distributions, so the durable
log is self-describing.

**Risk: low-moderate, bounded by the fallback.** `uv` presence on the Colab
image is **READ, not verified** — it is inferred from upstream's use of
`uv pip install` as the preferred path. If uv is absent or refuses (for example
an `EXTERNALLY-MANAGED` interpreter), the stage logs it and runs pip, which is
today's behaviour. The config schema change is validated at load; a bad list
fails loudly rather than silently installing nothing.

### 3.5 Net effect on the headline run

| | seconds |
|---|---|
| measured setup before | 157.5 |
| − overlap of the local build (§3.1) | −33.3 |
| − source-CSV re-upload removed (§3.2) | −11.9 |
| − batched mkdir round trips, 1 worker (§3.3) | −2.7 |
| **= measured-only projection** | **109.6 s (−30 %)** |
| − uv deps stage, extrapolated (§3.4) | −31 |
| **= full projection** | **≈78.6 s (−50 %)** |

For the configured 6-worker default, §3.3 contributes ≈17 s instead of 2.7 s,
giving ≈64 s (−59 %). **Only §3.1's and §3.2's and §3.3's inputs are measured
end-to-end; §3.4 is an extrapolation and the last row is a projection, not a
measurement.**

---

## 4. Verified offline vs still needs a real Colab run

**EXECUTED offline (no Colab VM, no GPU, no training):**

* Read the phase breakdown out of the history and log, and reproduced it with
  the shipped profiler.
* Counted the deps stage's downloads/installs/`already satisfied` lines straight
  from `colab_system.log`.
* Ran the generated remote installer program locally against a stubbed `uv`:
  uv succeeds; uv fails (falls back, and says so); uv absent (falls back, and
  says so); `prefer_uv=false` (uv never invoked).
* Ran real `pip`/`uv` installs into throwaway virtualenvs to get the comparison
  in §2 and §3.4.
* 23 new tests in `tests/test_colab_setup_path.py`, including the `main()`
  ordering test that pins "prewarm starts before `install_deps`".
* Full suite on the isolated commit: **357 passed, 2 skipped**.
* `python -m cli.colab --what train --preflight-only`: **could not be used** —
  see §6. It fails on `training/inference overlap contains 58529 product IDs`,
  a pre-existing contradiction between `training_lifecycle_preflight` and
  `complete_colab_worker`'s documented full-inference mode, unrelated to this
  change (the diff does not touch either).

**Needs a real Colab run to confirm (owner's call, not mine):**

1. That `uv` exists on the VM and the log line reads `[deps] installer=uv`.
   Observable: the `[deps] installer=…` line, and the deps stage's wall-clock in
   the next history session.
2. The actual deps-stage duration after §3.4. Observable: the `00_deps`
   interval in `scripts/profile_colab_setup.py` output.
3. That the prewarm overlaps on a real launch and the log shows
   `[local-prepare] building N bundle(s) concurrently with the VM dependency
   install` followed later by `[local-prepare] joining the build started before
   the VM setup` — never the `different request` line.
4. That the source-CSV reuse fires: `[upload] validation source=… reused the
   verified VM checkout copy … (sha256 matches; not uploaded)`, and that
   `input_provenance.json` still records the full-deduped source with an
   unchanged `rows`/`bytes`/`sha256`.
5. That the training lane still trains identically — the bundles are
   byte-identical, but only a run proves the lane end-to-end.

---

## 5. Commits, files, tests

* **`4f5be42`** `perf(colab): overlap local bundle build with VM setup; cut redundant remote work`
  — the production change and its tests.
* **`aaf7838`** `docs(colab): setup-time report and a re-runnable phase profiler`
  — `scripts/profile_colab_setup.py` and this report; plus docs-only
  corrections after it.
* Pushed: `3fdc039..aaf7838  training -> training`, then `aaf7838..412216a`;
  `ER/training` tracks the local `training` branch.
* Files changed: `src/cli/colab.py`, `src/core/schemas.py`,
  `config/training.yaml`, `tests/test_colab_setup_path.py` (new),
  `scripts/profile_colab_setup.py` (new), `COLAB_SETUP_OPTIMISATION_REPORT.md`
  (new).
* Tests: **334 passed / 2 skipped → 357 passed / 2 skipped**. No existing test
  was modified, so no justification for touching one was needed.
* Lint: `ruff 0.16.3` (the version pinned in `requirements.txt`) reports 33
  findings across the two changed modules against 31 at the parent commit. The
  2 new ones are `BLE001` on deliberate broad catches — one in the prewarm
  worker (a build thread that dies silently is exactly the silent drop the
  conventions forbid) and one on the checkout-copy probe (any probe failure
  must fall back to uploading, so the catch is the point). Both modules already
  carry 2 such catches; there is no ruff configuration or lint gate in the
  project (`pyproject.toml` has no `[tool.ruff]`), so these are ruff defaults,
  not a regression against a configured rule set. `ruff` was also used to
  confirm no *other* new finding category was introduced.

---

## 6. Deliberately NOT done, with reasons

| Not done | Why |
|---|---|
| `colab install` instead of the detached stage | It runs in the notebook **kernel** (`run_automation`), losing the durability property this launcher deliberately built: a kernel disconnect kills the install, whereas the detached process keeps writing a log/status pair the launcher can still retrieve. Its uv path *was* adopted — in our detached stage. |
| `mlflow-skinny` instead of `mlflow` | **Investigated and rejected with executed evidence.** It looked ideal (11.6 MB + server deps removed) but `core/mlflow_ctx.py` defaults to a **sqlite** tracking URI and mlflow-skinny raises `UnsupportedModelRegistryStoreURIException: got unsupported URI 'sqlite:///…'` — verified by running the real `MlflowCtx` against a skinny-only venv. It would break the remote lane outright. Switching the remote lane to a file-store URI would contradict the owner's stated local-sqlite decision. |
| `uv` via a config-detected prebuilt environment, or a prebuilt `hnswlib` wheel in the repo / on the DVC remote | The strongest remaining item: `hnswlib` has no wheel on PyPI, so every fresh VM pays a ~30 s C++ compile that a cached build would remove. Rejected for now because producing a CPython 3.13 / linux-x86_64 wheel is impossible on this aarch64 host, so it could be neither built nor verified here, and committing a binary wheel or hosting it on DagsHub is a repo/artifact-lifecycle decision for the owner. Named as the next step in §7. |
| `colab run` (ephemeral job runner) | Its teardown-on-exit is not equivalent to the launcher's `--keep-alive` recovery flow, and the keep-alive guards must not be weakened. |
| Installing `hnswlib` and the mlflow closure as two concurrent pip processes | Would overlap the CPU-bound compile with the network-bound install for maybe 25–30 s, but two pips writing the same `site-packages` with no lock is not a "lowest-risk" change; `uv` gets most of the same overlap safely. |
| `--no-build-isolation` for the hnswlib build | Would remove the ≈9 s build-isolation step, but it depends on `setuptools` being importable in the VM environment. Roughly 9 s for a failure mode that only appears at build time is a bad trade; `uv` removes that step anyway. |
| Making `--preflight-only` work again | It is broken **before** this change: `training_lifecycle_preflight` rejects any training/inference overlap while `complete_colab_worker` explicitly supports the full-deduped-inference mode the current config selects (`input_csv == source_csv == dataset_deduped.csv`), plus it hardcodes `58_529`/`3_000` row counts. Fixing it means changing what a guard asserts about validation semantics — a policy decision, and not a setup-time one. Reported here instead. |
| Unifying `colab._sha256_file` with `core.manifest.sha256_file` | A real latent duplication (`colab.py` uses both), but unrelated to setup time and it has its own blast radius. New code reuses `core.manifest.sha256_file`. |

---

## 7. What remains open

1. **Confirm §3.4 on a real launch** — the only saving that is still an
   extrapolation.
2. **The prepared environment**, the largest remaining structural win. Nothing
   about the VM's environment is cached across runs: a fresh VM re-compiles
   `hnswlib` and re-downloads ~20 MiB every time. A `uv`-built environment, or
   at minimum a prebuilt `hnswlib` wheel, hosted on the existing DagsHub/DVC
   remote (or as a release asset) would remove most of the remaining deps
   stage. This needs CPython 3.13 / linux-x86_64 and a decision about where the
   artifact lives.
3. **The upload burst (~39 s) is the second-largest remaining cost** and is now
   almost entirely the 43 MB gitignored training complement at ~3.4 MB/s. It
   could move to the DVC/DagsHub remote with Direct Data Access instead of the
   Colab upload channel, or be derived on the VM from the source CSV. Both are
   artifact-contract decisions.
4. **A real Colab run** to close 1–5 of §4. That spends accelerator quota, so it
   is the owner's call.

---

## 8. Reused vs newly created

**Reused (no second implementation):**

* `_prepare_local_training_bundles`, `_expand_worker_profiles`,
  `_upload_validation_inputs`, `_upload_prepared_bundles`,
  `_upload_with_retries`, `run_detached_stage`, `run_colab_exec_stream`,
  `run_colab_exec_capture`, `_validation_input_path`, `_BOOTSTRAP`,
  `_VALIDATION_INFERENCE`, `TRAIN_ROOT`, `sha256_file` (from `core.manifest` —
  the documented SSOT for hashing), `training_cfg()`/`ColabSpec` for every new
  knob, and the upstream Colab CLI's own uv-then-pip preference.
* `_prepare_local_training_bundles` keeps sole ownership of the bundle build;
  the prewarm only decides *when* it runs. `_training_bundle_profiles` is the
  single derivation shared by `run_train` and `main`.

**Newly created:**

* `_runtime_install_command` — builds the uv-then-pip remote program. New
  because the launcher previously hardcoded `sys.executable -m pip` and the
  upstream uv invocation is not exposed by the CLI in a way that preserves
  detached durability.
* `_BundlePrewarm`, `start_local_bundle_prewarm`, `drain_local_bundle_prewarm`,
  `_take_prewarmed_bundles`, `_bundle_request_key`, `_lane_bundle_request` —
  the overlap mechanism. New because no scheduling primitive existed; kept
  deliberately small.
* `_remote_checkout_copy` — content-verified reuse of the VM's own checkout.
* `RuntimePackagesSpec` in `core.schemas` + `colab.runtime_packages` /
  `colab.prefer_uv_install` in `config/training.yaml` — moving constants out of
  code, per the conventions.
* `scripts/profile_colab_setup.py` — the measurement tool every number in §1
  comes from, so the owner can re-run it after a real launch.
* `tests/test_colab_setup_path.py` — 23 tests.
* `COLAB_SETUP_OPTIMISATION_REPORT.md` — this file.

**EXECUTED vs READ.** Everything in §1, §2, §3.1–3.3's measurements, §3.4's uv/pip
comparison, the offline installer runs, and the test suites were **executed** on
this host. **Read only:** that the Colab image ships `uv` (inferred from
upstream's preferred path), the claim that the image ships the preinstalled
stack (taken from the launcher's own `already satisfied` log lines), and the
upstream sources cited. No Colab VM was launched, no GPU was requested, no
training was run.

**Blast radius.** `--what train`, `dual-train`, `smoke` gain the prewarm
(behaviour-neutral, scheduling only); `sims`, `mixed`, `hpo` are untouched
(`_lane_bundle_request` returns None). `_upload_validation_inputs` changes for
every lane that runs validation inference; the remote path recorded in
`input_provenance.json` may now name the checkout copy instead of the run
staging directory — the content hashes do not change, so comparisons remain
valid. `_upload_prepared_bundles` changes for every prepared-bundle lane. The
new config keys are required in `config/training.yaml`; a missing key now fails
at config load. Nothing here invalidates a checkpoint, a metric CSV, or a
trained artifact: no input bytes, no model input contract, and no training
hyper-parameter changed.
