# Colab setup-path optimisation — measured breakdown and changes

Branch `training`. Substantive commits: **`4f5be42`** (the optimisation),
**`c3cac78`** (a real defect in it, found by measuring — §2.3), **`aaf7838`**
(report + profiler); docs-only corrections may follow. Pushed to `ER/training`.
Baseline test count before any change: **334 passed, 2 skipped**; after:
**358 passed, 2 skipped** (24 new tests; no existing test modified).

Two measurement campaigns, both reported:

* **Historical trace** (§1) — `colab_cli_state/history/*.jsonl` +
  `colab_system.log` from the T4 runs of 2026-09-15. This is where the costs
  were first located.
* **Real CPU-VM before/after** (§2) — two freshly provisioned **CPU-only**
  Colab VMs, run back to back on 2026-09-15, driving the launcher's own phase
  functions and stopping before training. No GPU or TPU was ever requested.

A note on how this was verified: another agent was concurrently editing shared
files (`src/core/schemas.py`, `config/training.yaml`, `src/core/hard_negatives.py`,
`src/pipeline.py`, `src/training/*`) for the cross-brand mining lane, and at one
point their in-flight schema change broke `training_cfg()` for everyone. Test
runs reported here were taken from a throwaway worktree **at my commit**. The
shared checkout's uncommitted changes are not mine and are in neither commit.

---

## 1. Where the wall-clock goes — historical trace

### 1.1 Headline T4 run (`colab_system.log`, session `2026-09-15T11:31:47`)

```
PYTHONPATH=src .venv/bin/python scripts/profile_colab_setup.py colab_cli_state/history/*.jsonl
```

| phase | elapsed | share | evidence |
|---|---|---|---|
| provision + keep-alive + handshake | 2.44 s | 1.6 % | `session_created` → `socket.gethostname` exec |
| checkout | 9.30 s | 5.9 % | → `shutil.rmtree` exec |
| launcher turnaround | 2.53 s | 1.6 % | → `00_deps` launch exec |
| **dependency install (`00_deps`)** | **62.75 s** | **39.8 %** | → first `[models] key=` exec |
| remote model validation | 6.41 s | 4.1 % | → `torch.cuda` exec |
| **local prepared-bundle build** | **34.94 s** (33.34 s launcher clock) | 22.2 % | → first `prepared_training` exec; clock figure from the `[local-prepare]`→`[run] remote metadata recorded` stamp pair |
| **uploads** | **39.13 s** | 24.8 % | → last `file_operation upload` |
| **setup total** | **157.5 s** | | |

Cross-checked on all six named history files (deps 67.8–84.6 s, local prepare
35.2–73.1 s, uploads 39.2–136.8 s) and over all **60 sessions** with a deps
stage: deps median 42.0 s (32.0 %), local prepare 21.1 s (16.1 %), uploads
24.3 s (18.5 %), everything else ≤7.4 %.

**Could NOT be timed, stated not invented:** (a) the launch segment of a
single-worker lane — `run_single_train_and_stream` sends one long-lived cell
whose history record is written at training *completion* (11:53:28 for an
~11:34:25 launch); the profiler omits gaps over 120 s and says so. (b) The
resolve/download/compile split of the deps stage from the log alone — §2.2 now
measures it directly instead.

**`results/logs/colab_stages/*` does not exist locally** — those files are
written on the VM, and only the streamed copy reaches `colab_system.log`
(`find . -type d -name colab_stages` returns nothing). §1.2 uses the 213
streamed `[00_deps]` lines.

### 1.2 What the 62.75 s deps stage was spending it on

* **35 distributions downloaded, 19.99 MiB**, of which
  `mlflow` 11.60 + `mlflow-skinny` 3.70 + `mlflow-tracing` 1.80 = **17.10 MiB (86 %)**.
* **18 distributions installed** — `hnswlib` plus mlflow's closure.
* **124 `Requirement already satisfied` lines**: the image already ships
  sentence-transformers, datasets, accelerate, scikit-learn, pandas, numpy,
  wandb and the whole torch stack. The launcher asserted nine packages; only
  **two** needed real work.
* **One source build**: `hnswlib` publishes no wheel on PyPI, so every fresh VM
  compiles it.

---

## 2. Real CPU-VM before/after measurement

Two CPU-only Colab VMs, provisioned back to back, driven through the same lane
(1 worker, baseline masking profile, default model) by the launcher's own phase
functions, stopping before training. Both runs used the **same local code
path**; `before` emulates the pre-change behaviour (pip-only install,
unconditional validation uploads via `_remote_checkout_copy` disabled, one
mkdir exec per uploaded file), `after` is the committed code.

| phase | BEFORE (cpu VM, 17:28:21Z) | AFTER (cpu VM, 17:47:23Z) | delta | attribution |
|---|---|---|---|---|
| provision + handshake | 122.54 s | 19.33 s | −103.21 s | **not attributable** — see §2.4 |
| checkout | 15.55 s | 13.97 s | −1.58 s | noise |
| **dependency install** | **81.13 s (pip)** | **42.96 s (uv)** | **−38.17 s (−47 %)** | **§2.1, measured** |
| remote model validation | 4.53 s | 2.57 s | −1.96 s | noise |
| runtime profile | 5.24 s | 4.65 s | −0.59 s | noise |
| local bundle build | 531.13 s, **MainThread** | 598.31 s, **Thread-1** | — | **§2.3, overlap measured** |
| **validation uploads** | **56.47 s** | **36.79 s** | **−19.68 s (−35 %)** | **§2.1, measured** |
| **bundle uploads** | **83.10 s** | **7.57 s** | **−75.53 s** | §2.1, see caveat |
| teardown (verified gone) | 3.59 s | 3.57 s | — | — |
| **setup total (excl. teardown)** | **899.69 s** | **642.66 s** | **−257.03 s (−28.6 %)** | mixed — see below |

Excluding the provision phase, whose difference is environmental (§2.4):
**777.15 s → 623.33 s, −153.8 s (−19.8 %)**. Of that, **57.9 s is cleanly
attributable** (deps −38.2 s, validation uploads −19.7 s); the bundle-upload row
contributes a further 75.5 s that is real but contaminated by a Colab CLI error
(§2.2), so it is not quoted as a saving.

Both VMs were torn down: `colab sessions` → *"No active sessions found on
server"* after each run.

### 2.1 What the real VM confirmed

**`uv` is on the Colab image — verified, not assumed.** The durable stage log
printed:

```
[deps] installer=uv /usr/local/bin/uv pip install --python /usr/bin/python3 \
    sentence-transformers datasets accelerate scikit-learn pandas numpy hnswlib mlflow wandb
Using Python 3.13.15 environment at: /usr
Resolved 123 packages in 2.93s
   Building hnswlib==0.8.0
Downloading mlflow-tracing (1.7MiB) / mlflow (11.0MiB) / mlflow-skinny (3.5MiB)
      Built hnswlib==0.8.0
Prepared 19 packages in 37.19s
Installed 19 packages in 210ms
```

This also settles the deps stage's internal split that the T4 log could not:
**resolution is 2.93 s**, downloads are 16.2 MiB, and **37.19 s of "Prepared" is
dominated by the `hnswlib` compile**. The stage total (42.96 s) is 37.40 s of
work plus ~5 s of detached-stage launch and completion detection.

pip on an identical CPU VM took **81.13 s** for the same nine packages. So
**uv saved 38.17 s, a 1.89× speed-up — measured on real Colab hardware**, not
extrapolated.

**The source-CSV reuse works, and is content-verified.** The after run logged:

```
[upload] validation source=…/artifacts/data/dataset_deduped.csv reused the verified
         VM checkout copy /content/EuromonitoR/artifacts/data/dataset_deduped.csv
         (sha256 matches; not uploaded)
[upload] validation training=…/dataset_deduped_train_minus_3000.csv -> …   (still uploaded)
[upload] validation sample=… reusing /content/EuromonitoR/artifacts/data/dataset_deduped.csv
```

The 45 MB source upload disappeared and the validation phase fell from 56.47 s
to 36.79 s. The training complement (43 MB, gitignored) is still a real upload,
as intended.

**Batched mkdir works**: the after run created every worker directory in one
exec, and the two runs uploaded an identical 18.2 MB bundle.

### 2.2 Caveat on the bundle-upload row

The before run's 83.10 s is **contaminated**: the Colab CLI hit an internal
error during it —

```
File ".../jupyter_kernel_client/client.py", line 434, in stop
  if self._manager and self._manager.has_kernel:
AttributeError: 'KernelClient' object has no attribute '_manager'
```

so part of that 75.5 s gap is a CLI defect, not the per-file mkdirs. The
*structural* difference is unambiguous (3 remote execs → 1); at the 1.09–1.63 s
per-exec cost measured in the T4 history, batching two of them is worth ~2.9 s
for a 1-worker lane and ~17 s for the configured 6-worker default. The 75.5 s
figure is the raw observed pair; it is not the number to quote as the saving.

### 2.3 The overlap, and the defect it exposed

The `after` run's own handles show the mechanism working:

```
build_calls: [{"thread": "Thread-1 (_build)", "seconds": 598.31}]
alive_at_deps_end: true
```

The build ran on the **prewarm thread** (in the before run the same field read
`MainThread`), and it was still running when the VM's dependency install
finished — so the entire VM path (83.5 s in this run, 229.0 s at before-run
speeds) was hidden under it instead of preceding it.

**The first attempt at this did not work at all, and only a real VM run would
have shown it.** `_BundlePrewarm._build` called `_prepare_local_training_bundles`,
which consults `_take_prewarmed_bundles`, which found the prewarm that worker
thread itself belonged to and called `thread.join()` on it: the thread died with
`RuntimeError('cannot join current thread')`, the main thread rebuilt serially,
and the launcher carried on. The 33 s saving was silently zero. Reproduced
offline in one line:

```
[local-prepare] concurrent build failed: RuntimeError('cannot join current thread')
```

Fixed in **`c3cac78`** by splitting the builder from the prewarm lookup
(`_build_local_training_bundles` never consults the prewarm;
`_prepare_local_training_bundles` is its only consumer). The existing tests
missed it because every one of them replaced `_prepare_local_training_bundles`
with a fake, so the real recursion never ran. The new regression test drives the
**real** builder with only its subprocess and manifest reader stubbed, and
asserts the build ran on the prewarm thread rather than the caller's — counting
builds alone cannot tell the two apart, because the buggy path also builds
exactly once. Verified: the test passes on the fix and fails on the
reintroduced bug with the `cannot join current thread` line.

### 2.4 What these two runs do NOT show

* **The provision difference (−103.2 s) is not mine.** The before run's first
  control-channel handshake hit the 60 s `probe_timeout_seconds` ceiling, spent
  5 s backing off, then succeeded (≈57 s) → 122.5 s. The after run passed on the
  first attempt in 19.3 s. CPU-VM handshake latency is variable; the entire
  difference is the retry policy. Provisioning must not be credited to this
  change.
* **The two build durations differ** (531.13 s vs 598.31 s) because another
  agent was running a competing `--prepare-bundle` at ~92 % CPU during both
  runs. The overlap saving is bounded by `min(build, VM path)`, and in both runs
  the build was far longer than the VM path, so the bound was not binding.
* **The local build is not what this change optimises.** Its 531–598 s here is
  an artefact of the shared checkout: `artifacts/embeddings_cache/` (586 MB) is
  gone, so every build now recomputes embeddings. Historically the same build
  took 18–38 s. The overlap's value scales with the build, so it is *larger*
  today than the historical numbers imply — but the honest claim is
  `min(build, VM path)`, not the full build.
* **Only one before/after pair was taken**, on CPU VMs, one worker. A T4 launch
  was not measured (GPU is out of scope), so the historical T4 table in §1
  remains the reference for that lane.

---

## 3. The changes

### 3.1 Prefer `uv` on the VM, fall back to pip — **38.17 s MEASURED**

The detached `00_deps` stage runs a generated remote program that tries
`uv pip install --python sys.executable <packages>` and falls back to
`python -m pip install <packages>`; whichever ran is printed, so a downgrade to
the slow path is never silent. This mirrors what the installed Colab CLI already
does for its own `colab install` subcommand
([google-colab-cli](https://github.com/googlecolab/google-colab-cli),
`commands/automation.py`).

*Measured on identical CPU VMs: pip 81.13 s → uv 42.96 s.* The fast path
targets `--python sys.executable` rather than upstream's `--system`, which is
strictly more faithful to the old behaviour: `sys.executable -m pip install`
installs into the kernel's own interpreter, and `--system` would not if the
kernel ever ran inside a virtualenv.

The package lists and the preference are config-owned
(`colab.runtime_packages`, `colab.prefer_uv_install`) instead of inline string
literals spliced into generated code, validated by a `RuntimePackagesSpec`
pydantic model (`extra="forbid"`, non-empty/unique entries). The
`[deps] installing …` line now names the distributions.

**Risk: low**, and now largely retired: uv's presence is verified, and its
absence or refusal falls back to pip with a log line. Because uv ships on the
image, the `uv is absent` / `uv install failed` branches remain as guards rather
than as the expected path.

### 3.2 Stop re-uploading a 46 MB file the VM already has — **19.68 s MEASURED**

`validation_inference.source_csv` is `artifacts/data/dataset_deduped.csv`, which
is tracked in git and therefore already at
`/content/EuromonitoR/artifacts/data/dataset_deduped.csv` after
`prepare_remote_layout`. `_remote_checkout_copy` hashes the remote copy and
reuses it **only when the digest equals the local file's**.

*Measured: validation uploads 56.47 s → 36.79 s*, with the reuse line quoted in
§2.1. The probe costs one remote exec per unmapped input (source and training;
`sample` is the same file as `source` and is mapped by the existing in-run
dedupe), measured at ~1.4 s each. On the T4 run this removes a measured 14.71 s
upload; on these CPU VMs the phase fell by 19.68 s.

**Risk: low, and content-verified.** Reuse is never assumed from "same branch".
A locally modified input, a missing remote file, or a probe failure logs and
falls back to the normal upload. `input_provenance.json` still records
`path`/`bytes`/`sha256`, so the reuse is auditable — only the recorded path
string changes.

### 3.3 One mkdir exec instead of one per uploaded file — **~2.9 s (1 worker), ~17 s extrapolated (6 workers)**

`_upload_prepared_bundles` ran one exec for the run directory and then one more
per uploaded file; all worker directories now come from a single exec. Remote
execs are not free: 1.09–1.63 s each, measured in the T4 history. The observed
before/after pair on the CPU VMs (83.10 s → 7.57 s) is **not** a clean
measurement of this — see §2.2 — so the quoted saving is the structural one.

**Risk: low.** Same directories, one cell; pinned by a test.

### 3.4 Overlap the local bundle build with the VM setup — **saves `min(build, VM path)`**

`_prepare_local_training_bundles` shells out to
`training.train --prepare-bundle`: pure local CPU over immutable local inputs,
touching nothing on the VM. It used to run strictly *after* the remote setup.
`main()` now starts it before `ensure_session()` and `run_train` joins that
exact build.

*Measured:* the build moved from `MainThread` to `Thread-1 (_build)`, and was
still running when the VM deps stage ended (`alive_at_deps_end: true`), so the
whole VM path was hidden. On the historical T4 run the build was 33.3 s against
a 118 s remote path — it fits entirely. On these CPU VMs the build was
531–598 s, so the saving was capped by the VM path instead (83.5 s this run,
229.0 s at before-run speeds). **The saving is `min(build, VM path)`; it is not
the build duration.**

**Risk: low.** Same bytes, same bundle, same hashes; only the scheduling
changes. A request mismatch is logged and rebuilt rather than silently dropped;
a build failure is stored and re-raised inside the owning `run_train` call; and
`main`'s `finally` drains any abandoned build before `close_live_log()`, so a
lane that dies early cannot leave a thread writing into a closed log.

---

## 4. Verified offline vs verified on a real VM

**EXECUTED offline:** the historical phase breakdown, reproduced with the
shipped profiler; the deps-stage counting; the generated installer run locally
against a stubbed `uv` (success / failure / absent / config-disabled); real
pip-vs-uv installs into throwaway venvs; 24 new tests including the `main()`
ordering test and the prewarm-thread regression test; full suite
**358 passed, 2 skipped** on an isolated worktree at the commit.

**EXECUTED on real CPU Colab VMs (2026-09-15, two runs):** provisioning,
checkout, pip install, uv install, model validation, runtime profile, local
bundle build (serial and concurrent), validation uploads with and without the
verified-checkout reuse, single-exec vs per-file mkdir, and teardown. Both VMs
confirmed gone afterwards. Evidence logs: `/tmp/probe_before.log`,
`/tmp/probe_after.log` (not committed; they contain the raw phase timings).

**Still needs a real run:** only the lanes not exercised here — a T4 (`--what
train` on GPU) launch, `dual-train`/`smoke` with >1 worker, and `hpo`/`sims`/
`mixed`. The code paths for those are shared, and the multi-worker mkdir saving
is extrapolated from a measured per-exec cost rather than observed.

**Could not be used at all:** `--preflight-only`, which is broken **before** this
change — `--what train` dies with `training/inference overlap contains 58529
product IDs`, because `training_lifecycle_preflight` rejects any overlap while
`complete_colab_worker` explicitly supports the full-deduped-inference mode the
current config selects (`input_csv == source_csv == dataset_deduped.csv`), and
it hardcodes 58 529/3 000 row counts. My diff touches neither (verified by grep
on the diff). Not fixed here: it changes what a validation-semantics guard
asserts, which is a policy call.

---

## 5. Commits, files, tests

* **`4f5be42`** `perf(colab): overlap local bundle build with VM setup; cut redundant remote work` — the optimisation and its tests.
* **`c3cac78`** `fix(colab): the bundle prewarm joined its own thread and never overlapped` — the defect from §2.3 and its regression test.
* **`aaf7838`** `docs(colab): setup-time report and a re-runnable phase profiler` — `scripts/profile_colab_setup.py` and this report, plus docs-only corrections after it.
* Pushed to `ER/training`; the branch tip is at or after `aaf7838`.
* Files: `src/cli/colab.py`, `src/core/schemas.py`, `config/training.yaml`,
  `tests/test_colab_setup_path.py` (new), `scripts/profile_colab_setup.py`
  (new), `COLAB_SETUP_OPTIMISATION_REPORT.md` (new).
* Tests: **334 passed / 2 skipped → 358 passed / 2 skipped.** No existing test
  was modified.
* Lint: `ruff 0.16.3` reports 33 findings on the two changed modules against 31
  at the parent commit; the 2 new ones are `BLE001` on deliberate broad catches
  (the prewarm worker must not die silently; the checkout-copy probe must fall
  back to uploading). There is no ruff configuration or lint gate in the
  project, so these are ruff defaults rather than a configured rule set.
* `results/`, `artifacts/`, `dataset.csv` and `training_results/` are untouched:
  `git status --short results/ artifacts/ dataset.csv` is empty. The probe's own
  `results/prepared_training/*` bundle directories were deleted afterwards.

---

## 6. Deliberately NOT done, with reasons

| Not done | Why |
|---|---|
| Fixing the **provisioning blocker** found in §7 | It lives in the keep-alive guards the owner fenced off. Report only. |
| `colab install` instead of the detached stage | Runs in the notebook **kernel** (`run_automation`), losing the durability this launcher deliberately built: a kernel disconnect kills the install, whereas the detached process keeps writing a log/status pair the launcher can still retrieve. Its uv path *was* adopted — inside our detached stage. |
| `mlflow-skinny` instead of `mlflow` | **Investigated and rejected with executed evidence.** `core/mlflow_ctx.py` defaults to a **sqlite** tracking URI and mlflow-skinny raises `UnsupportedModelRegistryStoreURIException: unsupported URI 'sqlite:///…'` — verified by running the real `MlflowCtx` against a skinny-only venv. It would break the lane outright. |
| A prebuilt `hnswlib` wheel / cached environment | Now the single largest remaining item and **precisely measured**: 37.19 s of the 42.96 s deps stage is "Prepared", dominated by the `hnswlib` compile. Not done because producing a CPython 3.13 / linux-x86_64 wheel is impossible on this aarch64 host, so it could be neither built nor verified here, and where it lives (repo, release asset, DagsHub/DVC) is an owner artifact decision. |
| `colab run` (ephemeral job runner) | Its teardown-on-exit is not equivalent to the launcher's `--keep-alive` recovery flow. |
| Concurrent pip processes; `--no-build-isolation` | Largely moot now: uv already overlaps resolution, download and build, and the measured 37.19 s is dominated by a compile neither tool can avoid. |
| Unifying `colab._sha256_file` with `core.manifest.sha256_file` | A real latent duplication (`colab.py` uses both), but unrelated to setup time and with its own blast radius. New code reuses `core.manifest.sha256_file`. |

---

## 7. What remains open

1. **A provisioning blocker, found by running for real and NOT fixed here.**
   `colab_cli_entry.py` refuses to spawn the Colab CLI's keep-alive daemon
   unless `EUROMONITOR_KEEP_ALIVE_ALLOWED=1`, and `src/cli/colab.py` sets that
   to `"1"` **only** when `--keep-alive` is passed *and* the runtime is CPU. But
   `colab new` itself spawns that daemon during provisioning, so with the
   default `--keep-alive` absent the env var is `"0"` and provisioning fails
   outright:

   ```
   colab new -s my-highram-session            → exit status 1
   RuntimeError: refusing to start Colab keep-alive: this session was not marked
   keep-alive-eligible by the launcher (GPU sessions must never retain their VM)
   ```

   For a GPU lane it is worse: `colab.py` refuses `--keep-alive` for GPU, so a
   GPU launch can never satisfy the check and can never provision. The intent —
   never *retain* a GPU VM — is right; the implementation also denies the daemon
   that provisioning itself needs. This is a decision for whoever owns the
   guards (the measurement here used the sanctioned CPU keep-alive path,
   asserting `GPU == "CPU"` first, and still tore the VM down).

2. **The `hnswlib` compile**: 37.19 s of the 42.96 s deps stage, every fresh VM.
   A prebuilt wheel or a cached environment removes most of it (§6).
3. **The upload burst**: still 36.79 s, almost entirely the 43 MB gitignored
   training complement. It could move to the DagsHub/DVC remote via Direct Data
   Access instead of the Colab upload channel, or be derived on the VM. Both are
   artifact-contract decisions.
4. **The local bundle build** is currently 531–598 s because
   `artifacts/embeddings_cache/` was removed from the shared checkout; restoring
   that cache would return it to the historical 18–38 s.

---

## 8. Reused vs newly created

**Reused:** `_upload_validation_inputs`, `_upload_prepared_bundles`,
`_upload_with_retries`, `run_detached_stage`, `run_colab_exec_stream`,
`run_colab_exec_capture`, `_validation_input_path`, `_expand_worker_profiles`,
`_BOOTSTRAP`, `_VALIDATION_INFERENCE`, `TRAIN_ROOT`, `sha256_file` (from
`core.manifest`, the documented hashing SSOT), and `training_cfg()`/`ColabSpec`
for every new knob. The bundle builder keeps sole ownership of building; the
prewarm only decides *when* it runs. `_training_bundle_profiles` is the single
profile derivation shared by `run_train` and `main`. The uv-then-pip order is
the upstream Colab CLI's own preference.

**Newly created:** `_runtime_install_command`; `_build_local_training_bundles`
(the split that fixes §2.3) with `_prepare_local_training_bundles` as its
prewarm-consuming entry point; `_BundlePrewarm`,
`start_local_bundle_prewarm`, `drain_local_bundle_prewarm`,
`_take_prewarmed_bundles`, `_bundle_request_key`, `_lane_bundle_request`;
`_remote_checkout_copy`; `RuntimePackagesSpec` plus `colab.runtime_packages` /
`colab.prefer_uv_install`; `scripts/profile_colab_setup.py`;
`tests/test_colab_setup_path.py`; this report.

**EXECUTED vs READ.** Everything in §1 and §2 was executed: the phase tables come
from the history files, the profiler, and two real CPU Colab VMs (logs in
`/tmp/probe_before.log` and `/tmp/probe_after.log`). The tests were executed.
**Read only:** the upstream Colab CLI sources cited for the uv-then-pip
preference, and the upstream DVC/DagsHub documentation referenced in §7. No GPU
or TPU was ever requested, no training or fine-tuning was run (`--prepare-bundle`
writes the bundle and returns), and both VMs were torn down and verified gone.

**Blast radius.** `--what train`, `dual-train`, `smoke` gain the prewarm
(scheduling only); `sims`, `mixed`, `hpo` are untouched (`_lane_bundle_request`
returns None). `_upload_validation_inputs` changes for every lane that runs
validation inference, and `_upload_prepared_bundles` for every prepared-bundle
lane. The new config keys are required in `config/training.yaml`; a missing key
fails at config load. Nothing here invalidates a checkpoint, a metric CSV or a
trained artifact: no input bytes, no model-input contract and no training
hyper-parameter changed.
