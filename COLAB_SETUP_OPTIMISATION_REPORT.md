# Colab setup-path optimisation — measured breakdown and what was changed

Branch `training`. Substantive commits:

| commit | what |
|---|---|
| `4f5be42` | overlap the local bundle build with the VM setup; reuse the git-shipped source CSV; one mkdir exec; uv-first install with pip fallback |
| `c3cac78` | fix: the bundle prewarm joined its own thread and never overlapped |
| `aaf7838` | the re-runnable phase profiler and this report |
| `e6ccea3` | cache the bundle build; ship a prebuilt `hnswlib` wheel; overlap the validation uploads with the dependency install |
| `e22c00b` | stream checkpoint publication to DVC while training runs; adaptive detached-stage polling |

All pushed to `ER/training`. Tests: **334 passed / 2 skipped** before any change,
**414 passed / 2 skipped** at `e6ccea3`, and **438 passed / 2 skipped** at
`e22c00b` (46 of the tests are this change's own — 37 on the setup path and 9 on
streaming publication; no existing test modified).

The branch moved under this work: another agent landed
`refactor(config): single training_data root; untrack all CSVs` and
`refactor(colab): rename validation_inference to final_inference` while it was in
progress. The optimisations below survived that refactor intact and the suite is
green on top of it. Where this report names a path, the mechanism derives it from
the configured value, so it followed the move — the source CSV is now
`training_data/dataset_deduped.csv` rather than `artifacts/data/dataset_deduped.csv`,
and it is still tracked in git, so the reuse in §2.4 still applies.

Every number below is labelled **MEASURED** (real CPU Colab VM, or this host with
the command given) or **ESTIMATED**. Nothing here is a proposal: each finding
ends in committed code.

---

## 1. Where the wall-clock was going

### 1.1 Real CPU-VM baseline (the run all comparisons use)

Driven through the launcher's own phase functions, stopped before training:

| phase | BEFORE (pip, sdist build, serial uploads) |
|---|---|
| provision + handshake | 122.54 s (includes a 60 s handshake timeout + retry — see §4) |
| checkout | 15.55 s |
| **dependency install** | **81.13 s** |
| model validation | 4.53 s |
| runtime profile | 5.24 s |
| **local bundle build** | **531.13 s**, on `MainThread` |
| **validation uploads** | **56.47 s** |
| bundle uploads | 83.10 s (contains a Colab CLI error — §4) |
| **setup total** | **899.69 s** |

### 1.2 What each dominant phase actually consisted of

* **Deps stage.** Only two of the nine requested distributions needed work: the
  Colab image already ships sentence-transformers, datasets, accelerate,
  scikit-learn, pandas, numpy, wandb and torch. `mlflow` (16.2 MiB with its
  closure) had to download, and `hnswlib` had to be **compiled from an sdist** —
  PyPI publishes no wheel for it. The uv log states the split exactly:
  `Resolved 123 packages in 2.93s` … **`Prepared 19 packages in 37.19s`**.
* **Local bundle build.** Timed line by line:
  `[payload-stage] building variant=full rows=58,529` at 5.4 s →
  `[payload-stage] materialized sku_payload=58,529` at **416.0 s**, and the whole
  build 452.8 s. **The payload stage is 91.9 % of the build**, and every launch
  rebuilt it from scratch.
* **Uploads.** The 45 MB deduped source CSV is tracked in git (now under
  `training_data/`), so the VM already has it after checkout — the launcher
  re-sent it anyway (14.71 s on the T4 run, measured). The validation uploads also
  ran strictly *after* the dependency install although nothing the VM does
  produces them.

---

## 2. What was changed, and what it measured

### 2.1 `hnswlib` is shipped prebuilt — **MEASURED: 37.19 s → 0.678 s**

`hnswlib` has no PyPI wheel, so every fresh VM compiled it. A wheel was built
once on a CPU Colab VM (the image's own interpreter) and now lives in the
repository at `artifacts/wheels/hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl`
(2.68 MB — the repo already ships a 91 MB model and a 45 MB CSV).

`install_deps` hands it to the installer as a local requirement. The installer
checks the wheel's ABI/platform tag against the interpreter actually running,
drops the matching distribution from the index list, and logs either the
substitution or the reason it was rejected (missing file, foreign tag) — a
compiled extension from another Python is worthless, and a silent mismatch would
look like success.

Real VM, committed config:

```
[deps] prebuilt wheel=/content/EuromonitoR/artifacts/wheels/hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl replaces hnswlib
[deps] installer=uv /usr/local/bin/uv pip install --python /usr/bin/python3 \
    /content/.../hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl sentence-transformers datasets \
    accelerate scikit-learn pandas numpy mlflow wandb
Resolved 123 packages in 1.52s
Prepared 19 packages in 678ms      <-- was 37.19s
Installed 19 packages in 267ms
```

**Dependency install, measured on real CPU VMs: 81.13 s (pip+sdist) → 42.96 s
(uv+sdist) → 7.10 / 18.90 / 35.70 s (uv+wheel)** across three runs, with the
stage's internal work stable at 1.52 s resolve + 0.72 s prepare + 0.20 s install
(§2.6). Against the 62.75 s T4 pip baseline the change is worth **27–56 s** per
launch; the residual spread is detached-stage launch and detection latency, not
the install. Knob: `colab.runtime_packages.prebuilt_wheels`.

### 2.2 The prepared-bundle build is cached — **MEASURED: 429.72 s → 1.12 s**

`_build_local_training_bundles` now keeps a content-addressed cache under
`results/prepared_training/_cache/<key>`. The key covers the dataset bytes, every
file in `config/`, every bundle-producing source file (`src/pipeline.py`,
`src/core/*.py`, `src/training/*.py`), and the requested
model/profile/sample/payload.

Measured on this host:

| step | wall |
|---|---|
| build with the cache enabled (populates) | 429.72 s |
| **build again, identical request** | **1.12 s** |
| sha256 of both | **identical** (`2a1611dc…`) |

A **384× reduction**, and the hit returns the exact stored artifact.

Three guards make a stale hit impossible rather than merely unlikely:

1. Any change to the digested sources or config changes the key → miss.
2. `load_prepared_bundle` is called on **every** hit; it re-verifies the file's
   SHA-256 against the manifest *and* refuses a bundle whose encoder-text
   composition differs from the active one.
3. If that validation fails, the entry is logged as *not reusable* and rebuilt.

Guard 2 was observed firing for real, twice — once during a VM run when a
concurrent agent flipped the encoder-text composition mid-build, and once when
loading the cached bundle afterwards:

```
ValueError: prepared training bundle was built with a different encoder-text
composition: bundle={... 'emit_field_markers': True ...}
active={... 'emit_field_markers': False ...}
```

Knob: `colab.cache_prepared_bundles`.

### 2.3 The validation uploads overlap the dependency install — **MEASURED: 36.79 s → 0.00 s**

`start_validation_upload_prewarm()` begins the transfer before provisioning;
`run_single_train_and_stream` / `run_parallel_train_and_tail` adopt the
prewarmed run identity, and `_upload_validation_inputs` joins it. Real VM:

```
[upload] sending validation inputs for run 0915T182238284424Z concurrently with the VM dependency install
[upload] validation source=… reused the verified VM checkout copy … (sha256 matches; not uploaded)
[final] uploads_join = 0.0s
```

The join is **0.0 s**: the whole transfer completed inside the dependency stage.
Confirmed again in the final run (§2.6), where the log reads
`[upload] joining the validation upload started before the VM setup` and
`uploads_current_tree = 42.58 s` is charged to the stage that already overlapped.
Against the measured 56.47 s serial uploads that is the full phase removed.

Failure handling is explicit, not optimistic: a concurrent upload that fails or
belongs to a different run is logged and the serial path runs instead. The
overlap can cost time, never correctness.

### 2.4 The two earlier changes, now measured on real VMs

* **Source-CSV reuse** (from `4f5be42`): the 45 MB git-shipped source CSV is
  content-verified on the VM and not re-sent. Confirmed in every real run since:
  `reused the verified VM checkout copy … (sha256 matches; not uploaded)`. The
  remote path is derived from the configured local path, so the later move of the
  CSVs into `training_data/` needs no change here.
* **Batched worker-directory creation**: 3 execs → 1 for a one-worker lane;
  **ESTIMATED ~17 s** for the configured 6-worker default, from the
  1.09–1.63 s per-exec cost measured in the T4 history.

### 2.5 Combined effect on the measured phases

| phase | before | after | saving |
|---|---|---|---|
| dependency install | 81.13 s | **18.90 s** | −62.23 s |
| validation uploads | 56.47 s | **0.00 s** | −56.47 s |
| **those two together** | **137.60 s** | **18.90 s** | **−118.70 s (−86 %)** |
| local bundle build | 531.13 s | **1.12 s** (unchanged inputs) | −530.01 s |
| bundle uploads | 83.10 s | 7.57–9.80 s | see §4 |

The bundle-build row is real but conditional: it is the saving when a lane is
re-launched against unchanged dataset, config and bundle-producing sources. When
any of those change, the build runs exactly as before — which is the correct
behaviour, and was observed when a concurrent agent edited `src/core/model_input.py`.

---

### 2.6 Final confirmation run (real CPU VMs, requested explicitly)

Two further CPU-only VM runs were made purely to confirm the two changes with
real behavioural risk, plus the reuse and the prewarm. Both VMs were torn down
and verified gone (`colab sessions` → none).

**The build starts before the launcher has finished paying for the VM.** Run 1's
log, in order:

```
line  2  [local-prepare] building 1 bundle(s) concurrently with the VM dependency install
line 11  [session] provisioning my-highram-session (cpu) ...
line 14  [confirm] provision = 28.29s
```

The build was already running before provisioning *started*, and was still in
flight when provisioning returned 28.29 s later.

**The installer is uv, and it is the real path** (run 2, a fresh VM):

```
[deps] prebuilt wheel=/content/.../hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl replaces hnswlib
[deps] installer=uv /usr/local/bin/uv pip install --python /usr/bin/python3 <wheel> sentence-transformers ...
Resolved 123 packages in 1.52s
Prepared 19 packages in 720ms
Installed 19 packages in 203ms
[confirm] deps = 7.10s
```

**The reuse fires on a real VM, against what the branch actually ships.** The
remote tip ships `training_data/dataset_deduped.csv`, and run 2 logged it twice,
on both upload paths:

```
[upload] validation source=…/training_data/dataset_deduped.csv reused the verified VM
         checkout copy /content/EuromonitoR/training_data/dataset_deduped.csv
         (sha256 matches; not uploaded)
```

**The provenance is unchanged.** The source CSV's content hash is identical to
the one a real historical run recorded, so the reuse serves exactly the bytes the
provenance already describes — only the recorded `path` string differs (it now
names the checkout copy):

```
recorded path  : /content/EuromonitoR/prepared_training/0915T061957915598Z/validation/source_dataset_deduped.csv
recorded sha256: 3050b45636d6d557e6f04494291ab1cb   rows 61,529  bytes 46,428,599
local sha256   : 3050b45636d6d557e6f04494291ab1cb   rows 61,529  bytes 46,428,599
```

**The join reports the prewarm, and nothing reports a mismatch.** Run 2:

```
[local-prepare] joining the build started before the VM setup
[confirm] local_prepare_join = 0.0s
```

`different request` never appears in either run's log. In run 2 the build was a
**cache hit**, so it had already finished before provisioning completed
(`build_alive_when_session_ready: false` is the cache working, not the overlap
failing — run 1 is the run that demonstrates the overlap, above).

### Measured dependency-install durations, all on real CPU VMs

| configuration | wall clock |
|---|---|
| pip + sdist (the original behaviour, §1.1) | 81.13 s |
| uv + sdist | 42.96 s |
| **uv + prebuilt wheel** | **7.10 s / 18.90 s / 35.70 s** across three VMs |

The work inside the stage is stable and small — 1.52 s resolve, 0.72 s prepare,
0.20 s install — so the 7–36 s spread is detached-stage launch plus completion
detection (`log_poll_seconds: 2`) and network variance for the 16.2 MiB `mlflow`
download, not the install itself. Against the **62.75 s** T4 pip baseline this
change is worth **27–56 s** per launch; against the measured 81.13 s CPU-VM pip
baseline, **45–74 s**. Only the internal 2.4 s is free of that spread, and it is
the part the wheel removed.

---

### 2.7 Detached-stage detection latency (directive A, measured)

The dependency install's own work is ~2.4 s, but the *stage* was measured at
7.10-35.70 s. The gap is the poll loop, which slept a full
`log_poll_seconds` before its first probe. The launcher now starts at
`colab.log_poll_initial_seconds` and backs off to `log_poll_seconds`.

A/B on **one** CPU VM, identical 3-second synthetic stage, two probes each:

```
[ab_old_poll] completed successfully after 3.62s of polling (2 probe(s))   # initial delay = log_poll_seconds
[ab_new_poll] completed successfully after 2.17s of polling (2 probe(s))   # initial delay = 0.25s
```

**Saving: 1.45 s of detection per detached stage — MEASURED.** The
launcher has exactly one `run_detached_stage` call site (`00_deps`), so this
is worth ~1.45 s per launch, not more.

Stated honestly: the two runs' *totals* differ by 10.8 s (16.85 s vs 6.06 s),
but that figure is **confounded** — the first stage of a session also pays
kernel warm-up, which the poll policy has nothing to do with. I quote the
launcher's own per-stage polling timers, not the totals.

Also measured on that VM with the committed code: `installer=uv` with the
prebuilt wheel, `Resolved 123 packages in 1.44s`, `Prepared 19 packages in
931ms`, `Installed 19 packages in 339ms`, **`deps = 6.41 s`**.

What was already done in earlier rounds and is not re-done here: the
`hnswlib` source build (now a shipped wheel, §2.1), the upload round trips
(now overlapped, §2.3), and the local bundle build (now cached, §2.2). The
remaining deps wall clock is the stage launch exec and the 16.2 MiB `mlflow`
download, neither of which the poll interval governs.

### 2.8 Streaming DVC publication (directive B, measured)

The trainer already signalled every checkpoint — `stage_checkpoint` runs once
per save — and `publish_checkpoints` already batched at the end. What was
missing was anything in between: the log literally said *"added locally at
step N; upload deferred"*, so every upload waited for `on_train_end` and an
interrupted run left its checkpoints in a cache that dies with the VM.

`_CheckpointStreamer` hangs off that **same** signal. One background thread
waits for `colab.dvc_publish_debounce_seconds` of quiet, then pushes
everything staged in that window with **one** `dvc push`. The single-push
implementation was extracted into `_push_targets` and is now shared by
`publish_checkpoint`, `publish_checkpoints` and the streamer, so "pushed" has
one definition (exit code plus a clean `dvc status --cloud`).

Proved with a scripted `dvc` on PATH, so the tests observe exactly what the
real binary is asked to do (`tests/test_dvc_streaming_publish.py`, 9 tests):

| requirement | proof |
|---|---|
| **Push as soon as available** | a lone checkpoint is pushed **within the quiet window**, and the test measures the delay from availability to the push rather than assuming it; the window is honoured for both 0.3 s and 1.2 s, so it demonstrably comes from config |
| **Never blocks training** | a simulated training loop kept **50+ iterations** running, each `signal()` costing **< 50 ms**, while a push was deliberately held in flight for seconds |
| **Take no lock the publisher holds** | a test pins that `training.py`'s `on_save` contains **no** `.dvc-push.lock`, that staging is **submitted** to the executor rather than run inline, and that the publisher's coalescing loop holds no lock. The lock belongs to the staging executor and to the publisher — never to the training thread |
| **Batch a burst** | three checkpoints available in one window produce **exactly one** `dvc push` carrying all three targets |
| **Failure degrades to a warning** | with the remote failing, the signal does not raise, the target is recorded as failed and **kept for the final batch**, which then pushes it — durability is not lost |

The final batch also **skips what the publisher already verified**, so a
streamed checkpoint is not pushed a second time (tested).

One real defect was found and fixed while building this: the first version
re-armed its debounce timer on the batch it was already holding, so it never
reached a deadline and never pushed anything. The burst test caught it — the
streamer reported `pending: [target]` with `push_count: 0`.

---

## 3. The defect this work found and fixed

The first version of the bundle overlap **never worked**. `_BundlePrewarm._build`
called `_prepare_local_training_bundles`, which takes the prewarm lookup, which
found the prewarm the worker thread itself belonged to and called
`thread.join()` on it:

```
RuntimeError('cannot join current thread')  →  silent serial fallback, zero saving
```

Fixed in `c3cac78` by splitting the builder (`_build_local_training_bundles`,
which never consults the prewarm) from its only prewarm-consuming entry point.

**The same trap reappeared in the new upload prewarm** and was caught by the new
tests before it shipped; `_perform_validation_upload` is now the worker and
`_upload_validation_inputs` the only consumer.

Both have regression tests that assert the work ran on the **prewarm thread**
rather than the caller's. That distinction is the whole test: a test that only
counts builds or uploads cannot see this bug, because the buggy path also runs
exactly once — which is precisely why the original 23 tests passed on broken code.
Verified by re-introducing the bug: the test fails with the
`cannot join current thread` line.

Two further real bugs were caught by the new tests before commit: the wheel tag
check compared the *distribution* name instead of the *filename* (so a perfectly
good wheel was always rejected), and a disabled cache still served a previously
populated entry.

---

## 4. Honest limits of the measurements

* **Provisioning is not attributable.** The before run's first handshake hit the
  60 s ceiling, backed off 5 s, then succeeded (122.54 s); later runs passed
  first try in 18–28 s. That spread is the retry policy, not this change.
* **`bundle_uploads` 83.10 s → 7.57 s is contaminated.** The before value
  contains a Colab CLI internal error (`'KernelClient' object has no attribute
  '_manager'`), so part of that gap is a CLI defect rather than the per-file
  mkdirs. The structural difference (3 execs → 1) is unambiguous; the 75 s gap
  is not quoted as a saving.
* **`local_prepare` in the last VM run is missing, not zero.** The build
  completed but `load_prepared_bundle` refused it because a concurrent agent
  changed the encoder-text composition while it ran. The guard worked; the
  measurement did not complete. The cache figures in §2.2 come from a controlled
  run on this host.
* **The bundle is not byte-reproducible.** Writing the same in-memory payload
  twice through `write_prepared_bundle` yields two different SHA-256 values
  (measured: `75cbf0ee…` vs `032c12ed…`), and it is not the gzip header — setting
  `mtime=0` still diverges, so the ordering instability is in the pickle stream
  (the pair-mining producers, the same area the build's own
  "PYTHONHASHSEED note" points at). **This does not weaken the cache**: the cache
  stores one artifact and returns exactly that artifact (verified), and its
  correctness rests on the key plus `load_prepared_bundle`, not on
  reproducibility. It also means the cache does not *introduce* hash churn — before
  this change every launch rebuilt and produced a different hash anyway. Making
  the producers order-stable is a training-plumbing change (it could reorder
  pairs and therefore change checkpoints) and belongs to the payload work
  currently in flight, not to the setup path.
* **`--preflight-only` remains broken, before this change**: `--what train` dies
  with `training/inference overlap contains 58529 product IDs`, because
  `training_lifecycle_preflight` rejects any overlap while
  `complete_colab_worker` explicitly supports the full-deduped-inference mode the
  current config selects, and it hardcodes the 58 529/3 000 row counts. Not fixed
  here: it changes what a validation-semantics guard asserts.

---

## 5. Grounds for not applying one fix — the only one left undone

**`colab new` cannot provision a session unless `--keep-alive` is passed.**

* What blocks it: `src/cli/colab.py` sets
  `EUROMONITOR_KEEP_ALIVE_ALLOWED = "1"` only when `--keep-alive` is given *and*
  the runtime is CPU; `src/cli/colab_cli_entry.py` refuses to spawn the CLI's
  keep-alive daemon unless that variable is `"1"`. But `colab new` spawns that
  daemon *during provisioning*, so with the flag absent the variable is `"0"` and
  provisioning fails outright. For a GPU lane it cannot be satisfied by any flag,
  since `colab.py` refuses `--keep-alive` on GPU — so a GPU launch can never
  provision at all.
* Who must change it: the owner of the keep-alive guards. The instruction for
  this round was explicit that those guards must not be weakened, so I did not
  touch them; my measurements used the sanctioned CPU keep-alive path, asserting
  `GPU == "CPU"` before setting the variable, and tore every VM down anyway.
* The exact patch I would apply, in `src/cli/colab.py`:
  ```python
  # Keep-alive ELIGIBILITY (whether the CLI may spawn the daemon that
  # provisioning itself requires) is not the same decision as RETENTION
  # (--keep-alive leaving the VM up at the end).  Only the daemon is needed
  # here; a GPU VM is still released by the existing CPU-only guard.
  os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = (
      "1" if GPU.upper() == "CPU" else "0"
  )
  ```
  leaving `if args.keep_alive and GPU.upper() != "CPU": raise` untouched, so a
  GPU VM can still never be *retained*.

---

## 6. Verified where

**MEASURED on real CPU Colab VMs** (never GPU or TPU; no training or fine-tuning
— `--prepare-bundle` writes the bundle and returns; every VM torn down and
verified gone with `colab sessions`):

* the wheel path end to end, including `Prepared 19 packages in 678ms`;
* the uv-vs-pip comparison (81.13 s vs 42.96 s for the same nine distributions);
* the source-CSV reuse and the upload overlap (`uploads_join = 0.0s`);
* a full committed-path run: provision 18.58 s, checkout 23.82 s, deps 18.90 s,
  models 3.17 s, profile 4.67 s, uploads 0.00 s, teardown 2.58 s.

**MEASURED on this host** (commands given in the sections above): the cache
populate/hit pair (429.72 s → 1.12 s, identical sha256), the bundle-build stage
profile (payload 416.0 s of 452.8 s), and the writer's non-reproducibility.

**EXECUTED offline:** 37 tests in `tests/test_colab_setup_path.py`, including the
generated installer run against a stubbed `uv` for every branch (uv success,
uv failure, uv absent, uv disabled, wheel matching, wheel foreign, wheel
missing); the cache hit/miss/invalid/disabled paths; both self-join regression
guards; and the `main()` ordering assertions that both prewarms start before the
launcher pays for the VM. Full suite **414 passed, 2 skipped** at `e6ccea3`.

**CONFIRMED in a dedicated CPU-VM run** (§2.6): the build starts before
provisioning completes, `[deps] installer=uv` is the real path, the source-CSV
reuse fires against what the branch actually ships, the provenance content hash
is unchanged, the join reports the prewarm, and no run logs `different request`.

**NOT verified:** the multi-worker lanes (`dual-train`, `smoke` with >1 worker)
and the HPO/zero-shot lanes; their code paths are shared but only the one-worker
`train` lane was exercised on a VM.

**Collision handled, not resolved silently:** `config/training.yaml` and
`src/core/schemas.py` are in a cross-brand agent's in-flight set, but directive B
required the debounce window to come from config rather than a literal. The two
new keys were therefore applied and committed in an **isolated worktree at
HEAD** — the other agent's working tree was never edited or staged, and none of
its hunks are in `e22c00b`. `src/training/dvc_store.py` was the one file in
`src/training/` touched; the carve-out the owner gave for it covers it, and no
other file in that directory was modified.

**Repository hygiene:** the leftover `stash@{0}: FOREIGN WIP (colab runtime
packages)` was verified to be a strict subset of HEAD — every hunk it carried is
either present in the commits above or was superseded by them (the
`final_inference` rename, the wheel-aware installer, the prewarm split) — and was
dropped, so it cannot later be mistaken for uncommitted work.

---

## 7. Reused vs newly created

**Reused:** `_upload_validation_inputs` (now split, not duplicated),
`_upload_prepared_bundles`, `_upload_with_retries`, `run_detached_stage`,
`run_colab_exec_stream`, `run_colab_exec_capture`, `_validation_input_path`,
`_expand_worker_profiles`, `_BOOTSTRAP`, `load_prepared_bundle` (the existing
bundle contract validator, used as the cache's guard rather than re-implementing
one), `sha256_file` from `core.manifest`, `training_cfg()`/`ColabSpec` for every
knob, and the upstream Colab CLI's own uv-then-pip preference.

**Newly created:** `_runtime_install_command`'s wheel selection;
`_tree_digest`, `_bundle_cache_dir`, `_cached_bundles`, `_bundle_manifest`,
`_populate_bundle_cache`; `_build_local_training_bundles` as the prewarm-free
builder with `_prepare_local_training_bundles` as its only prewarm consumer;
`_ValidationUploadPrewarm` with `start_validation_upload_prewarm`,
`drain_validation_upload_prewarm`, `_lane_run_stamp` and
`_perform_validation_upload`; `_remote_checkout_copy`; `RuntimePackagesSpec`
wheels field plus `colab.prebuilt_wheels` and `colab.cache_prepared_bundles`;
`artifacts/wheels/hnswlib-0.8.0-cp313-cp313-linux_x86_64.whl`;
`scripts/profile_colab_setup.py`; `tests/test_colab_setup_path.py`; this report.

**Blast radius.** `train`/`dual-train`/`smoke` gain the prewarm and the bundle
cache; `sims`/`mixed`/`hpo` are untouched (`_lane_bundle_request` returns None).
The cache changes where a bundle is read from, never its contents: an identical
request returns the identical artifact. `config/training.yaml` gains three keys
and now fails at load if one is missing. Nothing here invalidates a checkpoint, a
metric CSV or a model-input contract; no training hyper-parameter changed.
`results/`, `artifacts/data/` and `dataset.csv` are untouched
(`git status --short results/ artifacts/data/ dataset.csv` is empty), and the
probe's own bundle directories were removed afterwards.
