# COLAB CPU-prep parity contract

Contract that brings the CPU prep lane (`colab_backend.py` -> `cli.colab main
--what bundle` -> `run_bundle`, src/cli/colab.py) to parity with the bundle
lane (`src/cli/colab_bundle.py`, commits d264af1 / 9e7f0e4 / 902689e /
99c7ce8 / 4d40d1e). `colab_backend.py` is a 7-line shim delegating to
`cli.colab.main`; all CPU-prep behavior lives in `src/cli/colab.py`, so that
is the only runtime file changed. Original colab code is extended, never
rewritten: new behavior is new code/parameters, config-gated, defaulting to
byte-identical previous behavior.

## Capability 1 — high-RAM provisioning

- Bundle-lane mechanism: the lane inherits config `colab.session`
  (`my-highram-session`) and reuses the owner-allocated high-RAM VM; the
  shared primitive `ensure_session` re-verifies an active session and only
  calls `colab new` when none exists.
- CPU gap (fixed): `ensure_session` allocated CPU without `--high-mem`, so a
  fresh lane-driven allocation was standard shape.
- Contract: shape request is config-owned — `config/training.yaml
  colab.high_mem` (schema `ColabSpec.high_mem: bool = False`,
  src/core/schemas.py). `_cpu_high_mem_args(accelerator)` (src/cli/colab.py,
  above `ensure_session`) returns `("--high-mem",)` ONLY for CPU sessions
  whose config requests it; with the key absent/False the emitted
  `colab new` argv is byte-identical to pre-parity. GPU accelerators are
  never reshaped by the key. An already-active named session is NEVER
  re-allocated (reuse contract pinned by test).

## Capability 2 — logging via streaming (root system log + training.log)

- Bundle-lane mechanism: the whole cell streams through
  `run_colab_exec_stream` (kernel transport, `[out]/[err]` lines, `\r`
  handling) — root system transcript; `cli.colab.main` installs the tee
  (`start_live_log`, colab.py:4693) and emits `[done]` only on clean
  completion, `[failed]` otherwise (commit 1e18b36 `completed` flag).
- CPU gap (fixed): `--what bundle` streamed to the root transcript but not
  to training.log.
- Contract: `run_bundle` now passes the EXISTING opt-in
  `training_output=True` to `run_colab_exec_stream` (the `[worker]`
  forwarding contract of 1e18b36 applied to the prep lane's own subprocess):
  every streamed chunk lands in BOTH the root system log and training.log.
  `[done]`/`[failed]` semantics come from main's `completed` flag and apply
  to the CPU prep lane as for every lane (pinned by tests). The standalone
  `cli.colab_bundle` lane is inherited, not duplicated; it keeps its
  behavior (no local training.log tee of its own).

## Capability 3 — tqdm passthrough

- Bundle-lane mechanism (902689e, final): the emitted remote script runs
  `subprocess.run([sys.executable, "-m", "training.prepare_all"], cwd=root)`
  with fd inheritance — NO stderr capture, NO log file, NO pump thread; the
  earlier 9e7f0e4 dual-sink pump was reverted because capture hid the bars
  behind the failure tail. tqdm itself lives in the preparation code
  (reused, never duplicated); `run_colab_exec_stream`'s `stream_output`
  flushes each `\r`-separated unit immediately.
- CPU gap: passthrough existed by construction but was unpinned.
- Contract: `run_bundle`'s emitted script keeps strict fd inheritance (test
  asserts `"stderr=" not in script` and the `training.prepare_all`
  invocation); a staged stream fake pins that tqdm-style `\r` bars stream
  live through the shared transport. Nothing collapses stderr.

## Parallel-session cap: 2 owner-launched VMs

Exactly TWO high-RAM CPU session VMs may exist for prep work at a time,
both launched and owned by the OWNER (lane code never launches, retries, or
replaces them):

1. 50% cohort VM (`dataset_50pct.csv` cohort)
2. full cohort VM (`dataset.csv` full cohort)

Isolation per VM: distinct config session name is selected per launch
without touching the shared YAML via the existing `EUROMONITOR_COLAB_SESSION`
env override (src/cli/colab.py:100); the launcher's control lock is
per-session (`launcher-<session>.lock`, colab.py:508); the cohort pins are
re-written per VM on the VM's own config copy (run_bundle script, the
99c7ce8 pin filter) so each prepare_all enforces ITS OWN cohort rows+sha;
deliveries are per-run-id under `TRAINING_RESULTS/colab_bundle_<id>` (the
`_bundle_delivery_local` root contract), so the two cohorts never share a
delivery root or an audit pin.

## Cohort tagging

- 50% cohort: run tag prefix in the delivery id is a timestamped
  `bundle_<MMDDTHHMMSSffffff>Z`; the cohort identity lives in the VM pin
  re-write (`[pins] source_export_expected_rows/sha256` printed in the
  transcript) — the run is identified by which `--dataset-csv` was uploaded.
- full cohort: same contract with repo-root `dataset.csv`.
- `results/training_prep.lock` botoes only LOCAL token phases
  (`training.prepare_all` single-flight, flocks on this box); the two
  owner-VM preps do not contend locally.

## BLOCKER PATH — exact steps for the owner's two launches

1. session capability (owner): one high-RAM CPU VM per cohort, distinct
   names, e.g.
   `colab new -s er-prep-50pct --high-mem`  (50% cohort VM)
   `colab new -s er-prep-full --high-mem`   (full cohort VM)
2. launcher state (one command per VM, run from this tree; lane reuses the
   named session and never allocates):
   outside: terminal,
   `EUROMONITOR_COLAB_SESSION=er-prep-50pct .venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive --dataset-csv dataset_50pct.csv`
   Full cohort:
   `EUROMONITOR_COLAB_SESSION=er-prep-full .venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive --dataset-csv dataset.csv`
3. on-VM invoke (done by the lane): setup upload of the raw export via
   `--dataset-csv` (git-tracked default) to REMOTE_ROOT/dataset.csv, the
   on-VM pin re-write to the uploaded cohort census (rows+sha), then
   `python -m training.prepare_all` executes end-to-end on the VM with
   stdout+tdERR inherited and streamed live ([out]/[err] with tqdm bars)
   into BOTH transcripts.
4. bundle artifact: delivery tarred on-VM at
   REMOTE_ROOT/bundle_delivery.tar.gz (fixed name, consumed per invocation).
5. verified fetch-back: `_download_file_with_visibility` under
   `results/training_results/colab_bundle_<run_id>/bundle_delivery.tar.gz`
   (local_root relative_to contract); `[done] artifacts downloaded locally`
   prints only after the verified download.
6. An owner kill of a session is final: a waiting exec is terminated
   fail-loud, never retried, and no replacement VM is launched by lane code.

## Open owner questions

1. flip `colab.high_mem` to true? (I left it false; the owner kicks off
   both VMs personally, so the flag only matters if a lane must
   self-provision a CPU runtime.)
2. Should the standalone `cli.colab_bundle` lane also gate `--high-mem`
   through the same config key when it needs a fresh session?  The shared
   primitive already honors it; zero further edits required.

## Evidence absorbed from the killed `my-highram-session` run

- setup timing (~2.5 min to first [deps] event), runtime profile, and the
  upload/verify path (`remote_checkout_copy` content-hash verify + retry)
  performed on the owner-killed session tree: nothing in the CPU-prep
  streaming contract depended on that VM's lifetime; the stale `colab exec -s
  my-highram-session --timeout 14400` wait was terminated fail-loud (no
  retry, no replacement VM).
