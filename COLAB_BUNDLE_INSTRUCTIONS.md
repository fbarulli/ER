# COLAB_BUNDLE_INSTRUCTIONS — how a data-bundle prep launch actually works

Written 2026-10-05 from the CODE, not from docs. Every claim cites
`src/cli/colab.py` (Co) or `src/cli/colab_data_bundle_prep.py` (Prep).
Where earlier docs (COLAB_CPU_PREP_PARITY.md) diverge from code, the code
wins and the divergence is flagged.

## 0. The checkout question (owner asked: "we're using the checkout, correct?")

YES — with one condition. The VM never sees the local working tree. The
launcher clones the PUBLIC remote branch fresh on the VM
(Co 2325–2335: `git clone … --branch {BRANCH} {REPOSITORY}` into
`REMOTE_ROOT = /content/EuromonitoR`; `config/training.yaml colab:`
block: repository `https://github.com/fbarulli/ER.git`, branch `main`,
git_remote_name `ER`). Local checkout patterns are assembled by
`prepare_remote_layout` (Co 2258–2265) and cloned (Co 2325–2335): full
runtime layout for the bundle lane (`minimal_runtime=False`), sparse
patterns `/src/ /config/ /scripts/ /artifacts/wheels/ /pyproject.toml
/requirements.txt /colab_backend.py` (Co 2261–2266).
`_BOOTSTRAP` (Co 2448–2453) does NOT clone — it only sets
`PYTHONPATH=<REMOTE_ROOT>/src` for the remote cell.

So: **the VM runs the pushed origin/main, plus exactly one overlay — the
uploaded raw export.** Anything needed on the VM must be committed and
pushed to `main` FIRST (push-often rule). `--dataset-csv` is the freshness
override: `run_bundle` uploads it to `REMOTE_ROOT/dataset.csv` because the
CSV IS git-tracked in the clone (Co 3819–3827 docstring; upload at
Co 3861–3867 with `_upload_with_retries`).

## 1. Prerequisites (before ANY launch)

1. `colab` CLI present and authorized: `check_colab_cli()` runs first
   (Co 4700–4702); auth default oauth2 (`--auth oauth2`), config file
   default `~/.config/colab-cli/...` (Co `colab new --help`). OAuth consent
   is interactive on FIRST use only.
2. Catch up: the repo's own CLI entrypoint is `_colab_command`
   (Co 620–632): it resolves `shutil.which("colab")`, reads ITS shebang
   python, and runs `<that python> src/cli/colab_cli_entry.py --config
   <TRAIN_ROOT>/colab_cli_state/sessions.json <args>` (Co 197–198).
   Session STATE lives in `colab_cli_state/sessions.json` — NOT in
   `~/.config/colab-cli`.
3. Config keys (`config/training.yaml`):
   - `colab.session` default session name; per-launch override via env
     `EUROMONITOR_COLAB_SESSION` (Co 100) — no YAML edit needed, the two
     parallel cohorts just set the env var.
   - `cpu_bundle_prep.high_mem: true` — REQUIRED for a FRESH high-RAM
     CPU allocation: `cpu_shape_args` (Prep 63–76) returns
     `("--high-mem",)` ONLY when the accelerator is CPU (empty list,
     Co 2268: `accelerator = [] if GPU.upper() == "CPU"`), and the config
     key is true. Without the key a fresh VM allocates the default shape.
   - `cpu_bundle_prep.lane: true` — selects the dedicated lane wrapper
     (dual-transcript streaming + `[cpu-prep]` tag line) at dispatch
     (Co 4783–4792). With `false`, the ORIGINAL `run_bundle` dispatch runs
     — same lifecycle, no training.log mirror, no tag. Both work; `true`
     is the production posture.
4. Flag provenances the main thread prints (Co 4700–4725): `colab
   sessions`, launcher lock, keep-alive environment. Only CPU lanes
   accept `--keep-alive` (Co 4568–4582: refuse for GPU; `--allow-gpu`
   gate for non-CPU at Co 4570–4574).

## 2. THE launch (the way the code does it)

The full lifecycle — session verify/provision, VM checkout to branch HEAD,
dependency install, THEN dispatch — lives in `main()` (Co 4700–4792):
`ensure_session()` (Co 4729) → `prepare_remote_layout()` (Co 4746–4750)
→ `install_deps()` → `--what bundle` dispatch (Co 4783). 

The dedicated lane module alone (`python -m cli.colab_data_bundle_prep`,
Prep 135–151) SKIPS that prologue: it sets `colab.GPU="CPU"` and
`EUROMONITOR_KEEP_ALIVE_ALLOWED=1` (Prep 141–147) and calls
`run_cpu_bundle_prep` → `colab.run_bundle` directly (Prep 122–132) — no
`ensure_session`, no clone install. That path requires an ALREADY-LIVE
session with an EXISTING VM checkout, i.e. it is the RELAUNCH entry, not
the first launch. (Doc divergence: COLAB_CPU_PREP_PARITY.md §BLOCKER PATH
then incorrectly names `colab new -s er-prep-50pct --high-mem` manually —
that skips handshake/lock/checkout, hence the confusion in this session.)
Single launch command per cohort:

```bash
# 50% cohort VM
EUROMONITOR_COLAB_SESSION=er-prep-50pct \
PYTHONPATH=src .venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive \
  --dataset-csv dataset_50pct.csv

# full cohort VM
EUROMONITOR_COLAB_SESSION=er-prep-full \
PYTHONPATH=src .venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive \
  --dataset-csv dataset.csv
```

(`-m cli.colab` = the `er-colab` console script installed in this venv —
`er-colab --what bundle …` is identical; `colab_backend.py --what bundle`
is the shim path. GPU defaults come from `--gpu`; CPU is implied by
`--gpu CPU`.)

Production posture: run each under `systemd-run --user --scope`
(standing rule 6 — session teardown kills bare processes; proven twice):

```bash
systemd-run --user --unit=er-prep-50pct --working-directory=/home/opc/ONE/ER \
  env EUROMONITOR_COLAB_SESSION=er-prep-50pct \
  /home/opc/ONE/ER/.venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive \
  --dataset-csv /home/opc/ONE/ER/dataset_50pct.csv
systemd-run --user --unit=er-prep-full --working-directory=/home/opc/ONE/ER \
  env EUROMONITOR_COLAB_SESSION=er-prep-full \
  /home/opc/ONE/ER/.venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive \
  --dataset-csv /home/opc/ONE/ER/dataset.csv
```

Parallel safety is REAL, not aspirational: the launcher lock is
PER-SESSION `colab_cli_state/launcher-<SESSION>.lock` (Co 4701, 2279–2286
lock docs; path Co 4791+ `_colab_launch_lock_path`), so two distinct
session envs do not contend. What IS shared: `colab_cli_state/sessions.json`
(appends two entries), colab event registry, and the

## 3. What `ensure_session` really does (Co 2233–2274)

1. `colab sessions` via the shared entrypoint (lists LIVE VMs).
2. If `SESSION` is active AND the control handshake verifies → REUSE, no
   allocation (never re-allocates an owner-launched VM).
3. If the named record is STALE (VM died) → `_forget_cached_session()`
   then `colab new -s <SESSION> --high-mem` (fresh high-RAM CPU, because
   accelerator list is empty for CPU and `cpu_shape_args` appends
   `--high-mem` from config). A stale record NEVER blocks a fresh
   allocation (Co 2243–2266 comments).

## 4. Lifecycle timeline after launch (what you will see, in order)

| checkpoint | printed by | meaning |
|---|---|---|
| `[cpu-prep] cohort=50pct dataset=dataset_50pct.csv sha256=<12hex> max_parallel_sessions=2` | Prep 126–129 | cohort tag + export digest + cap echo (lane=true only) |
| `[session] provisioning …` / `'…' already active; verifying control channel` | Co 2252–2271 | VM allocation or reuse |
| `[checkout] …` timeline | Co 2325–2335 | clone of REPO branch main into `/content/EuromonitoR` |
| `[bundle] uploading raw export …` | Co 3832–3835 | dataset.csv upload (retry/backoff at Co 640+) |
| `[bundle] VM audit pins -> rows=… sha=…` | Co 3843–3857 | on-VM rewrite of `audit.source_export_expected_rows/_sha256` to THIS cohort — per-VM cohort isolation, no shared pin |
| `[out]/[err]` streaming incl. tqdm `\r` bars, mirrored into `training.log` | Co `run_colab_exec_stream` + Prep 99–119 dual wrapper | prepare_all stage progress from `results/training_prep/<run_id>/timings.log` on the VM |
| `[done]` / `[failed]` | Co 1e18b36 semantics; Prep 148–151 | `[done]` ONLY on clean completion incl. verified download; anything else fail-loud, NO retry |

## 5. On-VM execution + delivery

- The remote cell runs `python -m training.prepare_all` on REPO/REMOTE_ROOT
  with `WANDB_MODE=disabled` (Co 3866–3874), raising on `rc != 0`
  (Co 3875–3877).
- One delivery archive `REMOTE_ROOT/bundle_delivery.tar.gz` (FIXED name;
  a rerun on a live VM overwrites it — by contract, because the local
  copy is timestamped per run) containing `results/training_prep/<run_id>`,
  the regenerated data CSVs, and `data/track_setup` (Co 3878–3904).
- Verified fetch-back: `results/training_results/colab_bundle_<run_id>/`
  via `_bundle_delivery_local` + `_download_file_with_visibility`
  (Co 3819 docstring; Co 4450+ download). `[done]` prints ONLY after the
  verified download.
- Local artifacts: per-run dir `colab_bundle_<id>/` under
  `results/training_results/`. Full-cohort run dirs are large (bundle +
  CSVs) — check disk before a full-cohort delivery lands.

## 6. Monitoring + recovery

- Transcripts: root system log + `training.log` (dual), per-session
  history `colab_cli_state/history/<SESSION>.jsonl`
  (`execution`/`file_operation`/`session_terminated` events; earlier run:
  `session_terminated reason=pruned` = VM reclaimed).
- Session state: `colab sessions` (CLI), `colab ls`; per-session exec
  process pattern `colab_cli_entry.py --config …/sessions.json exec -s
  <SESSION>`.
- A session the OWNER kills stays dead (Co/Prep fail-loud; NO retry, NO
  replacement allocation): a waiting exec terminates fail-loud (Prep
  contract; Co 4781 comment).
- RELAUNCH on a live VM (same SESSION): the lanes reuse the session and
  re-clone branch HEAD, then run; the module-only relaunch path is
  `python -m cli.colab_data_bundle_prep --dataset-csv …` ONCE THE VM HAS
  A CHECKOUT (see §2 caveat).
- Interruption: the bundle lane resume feature (d264af1) resumes a FROZEN
  full-cohort run from the VM side; partial deliveries re-derive from the
  per-run dir — treat interrupted runs as RE-prepare candidates rather
  than merges into fresh ones.
- Stuck/cancel: `colab exec` processes are plain local processes —
  `systemctl --user stop er-prep-50pct` (and `er-prep-full`) stops the
  launcher; the VM dies per keep-alive teardown unless --keep-alive
  (the 2 cap counts LIVE VMs, so stopped launchers must be GLN-cleaned).

## 7. The 2-parallel cap

`MAX_PARALLEL_SESSIONS = 2` (Prep 60): exactly `er-prep-50pct` + 
`er-prep-full` may exist simultaneously, owner-launched. The lane never
launches, retries, or replaces a VM itself (Prep 27–29) and prints the cap
at start (Prep 126–129). Do not run any OTHER colab session while both
are live (the GPU training traffic must not collide with prep).

## 8. Certainty verdicts

| fact | verdict |
|---|---|
| VM gets fresh clone of pushed branch (+1 uploaded CSV overlay) | verified-by-code (Co 2258–2335, 3819–3827) |
| SESSION env override per launch; per-session launcher lock | verified-by-code (Co 100, 197–198, 2279+) |
| `--high-mem` requires `cpu_bundle_prep.high_mem: true` when CPU is co-located or fresh | verified-by-code (Prep 63–76 + Co 2268) |
| `--what bundle` dispatch gate on `cpu_bundle_prep.lane` (lane wrapper vs original) | verified-by-code (Co 4783–4792) |
| direct lane module SKIPS session/checkout prologue | verified-by-code (Prep 135–151 vs Co 4700–4750) — contra COLAB_CPU_PREP_PARITY.md's manual `colab new` step |
| delivery is one FIXED-name tar + per-run-timestamped local dir | verified-by-code (Co 3878–3904) |
| keep-alive refuse GPU; --allow-gpu gate | verified-by-code (Co 4568–4582) |
