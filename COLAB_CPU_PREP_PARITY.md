# COLAB CPU-prep parity contract

Contract bringing the CPU data-bundle prep lane to parity with the bundle
lane (commits d264af1 / 9e7f0e4 / 902689e / 99c7ce8 / 4d40d1e), under
**owner structural ruling 8** (data-bundle production lives in a SEPARATE
lane file, exactly like the existing standalone bundle lane) and the
**owner ruling extension** (FULL separation from the GPU training code —
no common files; additive-only shared-schema additions in their own block).

## Ownership shape

- **Production lane: `src/cli/colab_data_bundle_prep.py`** (new, dedicated):
  owns launch capability (high-RAM), streaming, tqdm passthrough posture,
  cohort tagging, and the 2-parallel-session cap. Its production entry
  point is `python -m cli.colab_data_bundle_prep`; `run_cpu_bundle_prep` is
  the capability layer over `cli.colab.run_bundle` (never rewrites it).
- **`src/cli/colab.py` keeps only config-gated thin passthroughs**, byte-
  identical when the lane is unused (see ISOLATION BOUNDARY below):
  1. `ensure_session` provision argv forwards to `cpu_shape_args`
     (lazy import; returns `()` unless `cpu_bundle_prep.high_mem` — argv unchanged);
  2. main's `--what bundle` dispatch forwards to `run_cpu_bundle_prep`
     only when the ISOLATED root key `cpu_bundle_prep.lane` is true; otherwise the
     original direct `run_bundle` call runs unchanged.
- `colab_bundle.py` (the already-committed standalone bundle lane) is NOT
  extended: it stays the committed CSV→inputs reference implementation.
## ISOLATION BOUNDARY (owner ruling extension — hardens ruling 8)

The CPU data-bundle production code is FULLY SEPARATE from the GPU
training code. Every file the lane touches, and each one's role:

| file | role | GPU-training runtime code? |
|---|---|---|
| `src/cli/colab_data_bundle_prep.py` | the production lane (new) | no — new file |
| `src/cli/colab.py` | thin passthroughs ONLY: lazy `cpu_shape_args` forward in `ensure_session`'s provision argv, config-gated `--what bundle` forwarding; existing behavior byte-identical when the lane is unused | no in-place edits of GPU-lane logic (shared launcher shell, same precedent as `colab_bundle.py` importing it) |
| `src/core/schemas.py` | ADDITIVE: new `CpuBundlePrepSpec` class + one default-factory root field `cpu_bundle_prep` | no — no reshaping, no renames, no shared-validator changes; not read on the GPU runtime path |
| `config/training.yaml` | ADDITIVE: new top-level `cpu_bundle_prep:` block (+ the two keys REMOVED from the shared `colab:` block, restoring it byte-identical to its pre-parity form) | no |
| `tests/test_colab_cpu_prep_parity.py` | lane parity pins (13, staged fakes) | no |
| `COLAB_CPU_PREP_PARITY.md` | this contract | no |

The lane imports ONLY `cli.colab`'s committed data-bundle production
machinery (`run_bundle`, the proven exec transport, upload/retries,
delivery/download, event registry — the `colab_bundle.py`
"copy-assembled from cli.colab's working machinery" precedent) plus
`core.common` config-path primitives (`DATA_PATH`, `training_cfg`).
It NEVER imports or modifies `training.train`, `training.train_prepared`,
`model_tracks.*`, worker paths, or the prepare path in-process:
`training.prepare_all` executes as a REMOTE subprocess on the prep VM's
own checkout, from the code the VM checkout holds, not in this lane's
Python process. No shared function was edited; the two capabilities that
needed a hook (provision shape, dual-transcript forcing, dispatch gate)
are thin config-gated forwards.
- `colab_backend.py` remains a 7-line shim delegating to `cli.colab.main`;
  `src/core/schemas.py` gains the ISOLATED `CpuBundlePrepSpec` class + root
  field; `src/core/common.py` and every GPU-progress module untouched. Original colab code is extended, never rewritten.

## Capability 1 — high-RAM provisioning (owned by the lane)

- Bundle-lane mechanism: the lane inherits config `colab.session`
  (`my-highram-session`) and reuses the owner-allocated high-RAM VM; the
  shared primitive `ensure_session` re-verifies an active session and only
  calls `colab new` when none exists.
- CPU gap (fixed): a fresh lane-driven CPU allocation was standard shape.
- Contract: shape request is config-owned — `cpu_bundle_prep.high_mem`
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
   (equivalently: flip `cpu_bundle_prep.lane: true` and use
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

1. Flip `cpu_bundle_prep.high_mem`/`cpu_bundle_prep.lane` to true? Both default false
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
