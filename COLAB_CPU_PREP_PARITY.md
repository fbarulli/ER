# COLAB CPU-prep parity contract

Contract bringing the CPU data-bundle prep lane to parity with the bundle
lane (commits d264af1 / 9e7f0e4 / 902689e / 99c7ce8 / 4d40d1e), under
**owner structural ruling 8**: data-bundle production lives in a SEPARATE
lane file, exactly like the existing standalone bundle lane.

## Ownership shape

- **Production lane: `src/cli/colab_data_bundle_prep.py`** (new, dedicated):
  owns launch capability (high-RAM), streaming, tqdm passthrough posture,
  cohort tagging, and the 2-parallel-session cap. Its production entry
  point is `python -m cli.colab_data_bundle_prep`; `run_cpu_bundle_prep` is
  the capability layer over `cli.colab.run_bundle` (never rewrites it).
- **`src/cli/colab.py` keeps only config-gated thin passthroughs**, byte-
  identical when the lane is unused:
  1. `ensure_session` provision argv forwards to `cpu_shape_args`
     (lazy import; returns `()` unless `colab.high_mem` — argv unchanged);
  2. main's `--what bundle` dispatch forwards to `run_cpu_bundle_prep`
     only when `colab.cpu_data_bundle_lane` is true; otherwise the
     original direct `run_bundle` call runs unchanged.
- `colab_bundle.py` (the already-committed standalone bundle lane) is NOT
  extended: it stays the committed CSV→inputs reference implementation.
- `colab_backend.py` remains a 7-line shim delegating to `cli.colab.main`;
  `src/core/schemas.py` gains the two config keys; `src/core/common.py`
  untouched. Original colab code is extended, never rewritten.

## Capability 1 — high-RAM provisioning (owned by the lane)

- Bundle-lane mechanism: the lane inherits config `colab.session`
  (`my-highram-session`) and reuses the owner-allocated high-RAM VM; the
  shared primitive `ensure_session` re-verifies an active session and only
  calls `colab new` when none exists.
- CPU gap (fixed): a fresh lane-driven CPU allocation was standard shape.
- Contract: shape request is config-owned — `colab.high_mem`
  (`ColabSpec.high_mem: bool = False`). Lane fn `cpu_shape_args(accelerator)`
  returns `("--high-mem",)` ONLY for CPU sessions whose config requests it;
  default/False ⇒ `()` ⇒ byte-identical `colab new` argv. GPU accelerators
  are never reshaped by the key. An already-active named session is NEVER
  re-allocated (reuse contract pinned by test).

## Capability 2 — logging via streaming (root system log + training.log)

- Bundle-lane mechanism: the whole cell streams through
  `run_colab_exec_stream` (`[out]/[err]`, `\r` handling) into the root
  system transcript; `cli.colab.main` installs the tee (`start_live_log`)
  and emits `[done]` only on clean completion, `[failed]` otherwise
  (1e18b36 `completed` flag).
- CPU gap (fixed): the prep cell was not mirrored into training.log.
- Contract: the lane's `_dual_transcript_streaming` wraps the EXISTING
  shared transport for the duration of one prep run, forcing its existing
  `training_output=True` opt-in (the `[worker]`-forwarding contract of
  1e18b36 applied to this lane): every streamed chunk lands in BOTH the
  root system log and training.log. The transport is never reimplemented;
  `cli.colab` behavior is unchanged outside the lane context. The lane's
  own `main` prints `[done]` only after a clean run and `[failed]`
  fail-loud on any exception, never retrying (the shared-surface default
  dispatch keeps main's 1e18b36 semantics unchanged).

## Capability 3 — tqdm passthrough (owned by the lane, delivered by fd inheritance)

- Bundle-lane mechanism (902689e, final): the emitted remote script runs
  `subprocess.run([sys.executable, "-m", "training.prepare_all"], cwd=root)`
  with fd inheritance — NO stderr capture, NO log file, NO pump thread; the
  earlier 9e7f0e4 dual-sink pump was reverted because capture hid the bars
  behind the failure tail. tqdm lives in the preparation code (reused,
  never duplicated); `run_colab_exec_stream`'s `stream_output` flushes each
  `\r`-separated unit immediately.
- Contract: the lane drives `run_bundle`'s script unchanged (pinned: no
  `stderr=` capture, prepare_all invocation present) and a staged stream
  fake pins that tqdm-style `\r` bars stream live through the shared
  transport. Nothing collapses stderr.

## 2-parallel-session cap + per-VM cohort isolation

Exactly TWO high-RAM CPU prep VMs may exist at a time, both owner-launched
and owner-owned (`MAX_PARALLEL_SESSIONS = 2` in the lane; the lane never
launches, retries, or replaces a VM and prints the cap at start):

1. 50% cohort VM (`dataset_50pct.csv` cohort, tag `50pct`)
2. full cohort VM (`dataset.csv`, tag `full`)

Isolation per VM: the session name is selected per launch without touching
the shared YAML via the existing `EUROMONITOR_COLAB_SESSION` env override
(src/cli/colab.py:100); the launcher lock is per-session
(`launcher-<session>.lock`); the cohort pins are re-written per VM on the
VM's own config copy (the 99c7ce8 pin filter in `run_bundle`) so each
prepare_all enforces ITS OWN cohort rows+sha; deliveries land per run-id
under `TRAINING_RESULTS/colab_bundle_<id>` (`_bundle_delivery_local`
root contract) — the two cohorts never share a delivery root or a pin. The
lane tags every start with `[cpu-prep] cohort=<tag> ... sha256=<prefix>` so
the parallel transcripts are attributable. `results/training_prep.lock`
bounds only LOCAL token phases; the two owner-VM preps do not contend
locally.

## BLOCKER PATH — exact steps for the owner's two launches

1. session capability (owner, one per cohort, distinct names):
   `colab new -s er-prep-50pct --high-mem`  (50% cohort VM)
   `colab new -s er-prep-full  --high-mem`  (full cohort VM)
2. launcher state (per VM, from this tree; the lane reuses the named
   session and never allocates anything):
   `EUROMONITOR_COLAB_SESSION=er-prep-50pct .venv/bin/python -m cli.colab_data_bundle_prep --dataset-csv dataset_50pct.csv`
   `EUROMONITOR_COLAB_SESSION=er-prep-full  .venv/bin/python -m cli.colab_data_bundle_prep --dataset-csv dataset.csv`
   (equivalently: flip `colab.cpu_data_bundle_lane: true` and use
   `python -m cli.colab --what bundle --gpu CPU --keep-alive --dataset-csv …`
   — the thin passthrough forwards; default config keeps the original path.)
3. on-VM invoke (done by the lane): raw-export upload to
   REMOTE_ROOT/dataset.csv, the on-VM cohort pin re-write (rows+sha), then
   `python -m training.prepare_all` end-to-end with streamed stdout/stderr
   (`[out]/[err]`, tqdm bars live) into BOTH transcripts.
4. bundle artifact: delivery tars at REMOTE_ROOT/bundle_delivery.tar.gz
   (fixed name, consumed per invocation).
5. verified fetch-back: `_download_file_with_visibility` under
   `results/training_results/colab_bundle_<run_id>/bundle_delivery.tar.gz`
   (relative_to root contract); `[done]` prints only after the verified
   download; `[failed]` fail-loud otherwise, no retry.
6. An owner kill of a session is final: a waiting exec is terminated
   fail-loud, never retried, and no replacement VM is launched by lane code.

## Open owner questions

1. Flip `colab.high_mem`/`cpu_data_bundle_lane` to true? Both default false
   so the shared surface stays byte-identical; the flag only matters if a
   lane must self-provision or the shared dispatch should forward.
2. Should the standalone `cli.colab_bundle` lane also honor `cpu_shape_args`
   in its `_provision` (it currently reuses `ensure_session`, which the
   passthrough already covers — zero edits required to benefit).

## Evidence absorbed from the killed `my-highram-session` run

- Setup timing, runtime profile, and the upload/verify path (content-hash
  verify + retry) from the owner-killed session cost nothing structurally:
  the CPU-prep streaming contract never depended on that VM's lifetime; the
  stale `colab exec -s my-highram-session --timeout 14400` wait was
  terminated fail-loud (no retry, no replacement VM).
