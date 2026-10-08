# Runbook

All commands run from the repo root with the local venv.

## Lane operations (SSOT pointers)

Every remote surface has one landing doc — never dig history or error
strings; if knowledge lives anywhere else, it belongs here:

| surface | doc | one-line truth |
|---|---|---|
| Colab (CPU/GPU sessions, smokes, train) | [colab-lane.md](colab-lane.md) | `er-colab --what <op> …` (dry-run by default planned for remote ops) |
| Kaggle (bundle kernels, train/embed, chain, autowatch) | [kaggle-lane.md](kaggle-lane.md) | `er-kaggle --what <op> …`; `--what chain --cohort <c>` runs bundle→fetch→publish→train (→embed) end-to-end |
| Laya (typed decisions) | [laya-lane.md](laya-lane.md) | `python laya_backend.py --kind kaggle --decision <k> [--execute]` |

Standing rules that live in both lane docs, summarized once here:

- **Auth**: `~/.kaggle/kaggle.json` is the only credential; an existing
  `~/.kaggle/access_token` fail-louds every lane op (delete it).
- **Watchers are default**: kaggle pushes and colab remote runs spawn their
  own detached watcher (same recipe: poll → download → release, always).
  No operator arg; never launch an unwatched session.
- **Long-running commands** (watchers, chains, preps) launch detached with
  `setsid env … < /dev/null &`; plain `nohup` dies with wrapper groups.
- **Sessions close themselves**: a terminal state always ends in a release
  plus receipt; if you find an open session, the lane's `stop` op closes it.
- **Timestamps**: UTC → none. Log stamps are Europe/Paris CET/CEST; logs
  live under `logs/<lane>/` (one roof; receipts stay under `results/`).

## Preparation

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all
```

Flags:

| Flag | Purpose |
|---|---|
| `--run-dir <dir>` | preparation log directory |
| `--tracks-config <yaml>` | override `config/model_tracks.yaml` |
| `--resume-from {dedupe,validation,full_bundle,suite_inputs}` | see [data-prep.md](data-prep.md#resume) |
| `--negative-supply-run-tag <tag>` | diagnostic lane dir; letters, digits, `_`, `-` only |

Run it under `systemd-run` so session teardown cannot kill it:

```bash
systemd-run --user --scope --working-directory=$PWD \
  env PYTHONPATH=src $PWD/.venv/bin/python -m training.prepare_all
```

## Launch

```bash
# full three-track suite
bash scripts/run_full_training.sh \
  --prepared-input-package results/training_prep/<run>/all_tracks_inputs.tar.zst

# CPU smoke
er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU
# (scripts/run_colab_smoke.sh still emits the legacy `--what smoke` lane,
# which the gate at src/cli/colab.py rejects: "legacy smoke does not
# preserve the shared component holdout". The track-config above is the
# sanctioned path.)

# explicit
PYTHONPATH=src .venv/bin/python colab_backend.py --what tracks \
  --tracks-config config/model_tracks.yaml \
  --prepared-input-package results/training_prep/<run>/all_tracks_inputs.tar.zst \
  --gpu T4 --allow-gpu
```

| Env var | Effect |
|---|---|
| `COLAB_GPU` | overrides the full-run runtime |
| `EUROMONITOR_COLAB_SESSION` | per-launch session name, no YAML edit |
| `WANDB_API_KEY` | only needed for online tracking |
| `EUROMONITOR_NEGATIVE_SUPPLY_SPEC` | JSON spec for mining and discriminator thresholds |

## CPU bundle prep lane

**Cohort policy: the full cohort only.** `dataset.csv` (71,623 rows, md5
`0717235b57d059936d5937715b29a4da`) is the only cohort we train. `dataset_10k.csv`
and `dataset_50pct.csv` stay in the repo as historical artifacts and are not
training inputs. Never pass them to `--dataset-csv`.

One session. Run under `systemd-run`.

```bash
systemd-run --user --unit=er-prep-full --working-directory=$PWD \
  env EUROMONITOR_COLAB_SESSION=er-prep-full \
  $PWD/.venv/bin/python -m cli.colab --what bundle --gpu CPU --keep-alive \
  --dataset-csv $PWD/dataset.csv
```

Do not run any other Colab session while it is live. Stop a stuck launcher with
`systemctl --user stop er-prep-full`.

## Regenerating individual artifacts

```bash
PYTHONPATH=src .venv/bin/python -m training.dedupe
PYTHONPATH=src .venv/bin/python -m training.data_prep
PYTHONPATH=src .venv/bin/python -m training.labeled_pairs
PYTHONPATH=src .venv/bin/python -m training.build_final_validation
PYTHONPATH=src .venv/bin/python -m training.build_reference --verify
```

These expect their upstream inputs to exist and to verify.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

Live-data tests skip on a fresh clone until the pipeline regenerates
`data/final_validation.csv` and `results/manifests/final_validation.json`.

## Inspecting a run

```bash
cat results/training_prep/<run>/manifest.json     # stages, provenance, inventory
cat results/training_prep/<run>/timings.json      # per-stage seconds
cat results/training_prep/<run>/timing_offenders.log
cat results/training_prep/<run>/handoff.json      # final verification
cat results/training_prep/<run>/<stage>.log       # stage detail
```

`manifest.json` is the authoritative inventory. It records every output path, size
and SHA-256, plus stage timings and the failure state.

---

# Traps

## Memory

`suite_inputs` is the memory peak. Two causes, both fixed or avoidable:

**Serialization spike.** `ablation.py` used to build a whole 732 MB `request.json`
as one contiguous Python string, twice — once to hash it, once to write it. With the
text track's 7,591 token batches still resident for the next track, that OOM-killed
the process on a 22 GB box. Both now stream through the encoder, so peak memory is
independent of document size and the bytes are unchanged. If you touch those
functions, keep them streaming.

**Genuinely high water.** The ablation cohort is exhaustive — 478k unique texts,
84k candidate texts, 74k graph records per variant. On a small box this can still
OOM. That is a capacity problem, not a bug.

If a run dies with no Python traceback, it was the kernel OOM killer. Check
`free -g` and swap; swap exhaustion is the usual culprit.

**A killed run cannot be resumed.** Preparation hashes all source, so any edit
invalidates it. Start fresh — and expect `archive()` to move ~2 GB of prior state
into the new run dir, so check disk.

## Do not

- **Do not run `colab sessions`/`colab exec` probes while a launcher holds a
  session.** colab-cli 0.7.4's `sessions` refresh cannot match server
  assignments (they carry only the machine endpoint, no session name) and
  rewrites its local `sessions.json` empty — every later `exec -s <name>`
  then fails "not found" while the VM is actually alive. The launcher's own
  transcript is the only visibility source during a live run.
- **Do not use `python -m cli.colab` to launch.** It makes a separate `__main__` and
  can read CPU despite `--gpu`. Use `colab_backend.py` or `er-colab`.
- **Do not use `python -m cli.colab_data_bundle_prep` as a first launch.** It skips
  the session, handshake, lock and checkout prologue. It is the relaunch entry, and
  needs a live session that already has a checkout.
- **Do not edit an artifact a manifest claims to hash.** It poisons resume.
- **Do not relax diet-gate thresholds.**
- **Do not normalize `row_bc`** — fold sets are raw keys, zero-padding drops pairs.
- **Do not flip `negative_supply.mode` from `gate`** without discriminator plus
  stratified eval evidence.
- **Do not run a graph-worker preflight** against a prepared input or frozen cache
  the run has not produced yet. It cannot pass.
- **Do not assume BF16 on a T4.**
- **Do not trust GPU throughput numbers taken with profiling on.** Measure the
  sampling overhead first.

## Fail-loud is the design

A stage that dies is doing its job. Every stage asserts its own row-count closure
and writes a manifest naming the population that failed. The census drift report,
the leak guards, the diet gate and the handoff verification all refuse to continue
rather than produce a quietly wrong artifact.

A `SEPARABLE` discriminator verdict stops lane-mode preparation. `insufficient`
means there was not enough grouped evidence — that is **not** a successful safety
result.

A session you kill stays dead. The launcher never retries or replaces a VM.

## Push often

The Colab VM clones pushed `origin/main`. It never sees your working tree. Anything
the VM needs must be committed and pushed before launch.
