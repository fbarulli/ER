# Lane ops — operator card

Commands + log-checking only. The SSOT docs carry the full contracts —
`docs/colab-lane.md` (structure, capability matrix, bundle boundaries),
`docs/runbook.md` (launch discipline), `docs/kaggle-lane.md`, `docs/laya-lane.md`.
This card points; it does not re-spell.

## Credentials

```bash
set -a; . /home/opc/ONE/.env; set +a   # keys live one dir above the repo (AGENTS.md §7)
```

## Launch

```bash
# CPU bundle lane (committed export -> VM prep -> delivery -> auto-release)
scripts/run_colab_bundle.sh dataset.csv
#   delivers results/training_results/colab_bundle_<run_id>/bundle_delivery.tar.zst
#   transcript: logs/colab/lane_cpu.log (the launcher opens it; the session is
#   released after delivery on every outcome — there is no retention flag; extra
#   flags pass through: --resume-*, --preflight-only, --install-delivery)

# GPU smoke (T4), detached
COLAB_GPU=T4 setsid bash scripts/run_colab_smoke.sh >/dev/null 2>&1 &
#   WARN: never redirect nohup/wrapper stdout onto the lane transcript — the lane
#   already holds the append fd; a second redirection duplicates lines through the
#   same file and a plain `tee` (non -a) punches the zero-filled NUL hole.

# Full E2E, all 3 text-loss suites (mnrl baseline, contrastive, triplet) in
# ONE session on GPU T4: serial launches, sealed inputs reused (no repackage),
# each run releases the VM at the end. One command; --dry-run prints the
# resolved er-colab lines without touching a VM.
bash scripts/run_colab_full.sh
#   transcript: logs/colab/lane.log (full/default session my-highram-session;
#   one launch = one run, each truncates ONCE, so it always holds the CURRENT
#   suite — never tee it; keep this shell's stdout for the full serial history)
```

## How to check logs

Transcript names are the SSOT rule ``ColabSpec.transcript_name`` (config/training.yaml
colab.lanes; the configured session falls to default_log_name) — never hand-map a
launch to a log again. One rule, three names:

```bash
tail -f logs/colab/lane.log              # full E2E (run_colab_full.sh / er-colab --what tracks: default session)
tail -f logs/colab/lane_cpu.log          # bundle lane (--gpu CPU, session smoke-cpu)
tail -f logs/colab/lane_gpu.log          # GPU smoke (--gpu T4, session smoke-gpu)
tr -d '\0' < logs/colab/lane.log | tail  # if a stale NUL-punched log reads as binary

# progress: each captured chunk re-emits its LAST tqdm frame as one tagged line
# "[tqdm] <bar>" — read the newest for position, the count as churn evidence
# (point it at the transcript of the run you launched, per the rule above)
grep -c "\[tqdm\]" logs/colab/lane.log
grep "\[tqdm\]" logs/colab/lane.log | tail -1
# ONE emission mechanism: every line is written once, so a grep hit is one
# event (core.run_log.RunLogger -> core.step_trace.send -> stdout -> the tee).
# There is no second, formatted copy of the same line to de-duplicate by hand.
grep -c "PROBE-LIKE-TEXT" logs/colab/lane.log

# session state + termination reason (user_requested = our release, pruned = Google reaped)
cat colab_cli_state/sessions.json
cat colab_cli_state/history/my-highram-session.jsonl | tail   # history is per-session, same names

# release proof (in the transcript):
#   [stop] teardown verified: session is no longer listed   (colab_release.py:113)
#   [timing] step=release state=completed elapsed_seconds=… (step_trace.py:110)
#   [done] compressed results downloaded and completed locally (colab.py:2797)

# self-watch receipts (spawned on every executed remote run; poll log
# logs/colab/self_watch_<run_id>.log):
ls results/training_results/self_watch_<run_id>/   # self_watch_receipt.json + captured_lane_*.log

# run outputs layout:
#   GPU suite run:  results/model_tracks/<run_id>/  (suite_result.json, per-track dirs,
#                   post_training_ablation.json)
#   CPU deliveries: results/training_results/colab_bundle_<run_id>/
#                   (bundle_delivery.tar.zst + result_events.jsonl — today's stages are
#                   download/download_file; promote events land with the first full
#                   delivery, emitted by promote_delivery, colab_lane_cpu_delivery.py:326)
```

## Data freshness (known gap)

The transcript does NOT print the VM's checkout SHA (nothing to grep — verified).
The VM clones `origin/main` sparse + depth-1, so committed-and-pushed is all it sees:

```bash
git rev-parse origin/main                     # cross-check against the run window
git log -1 --format='%h %cI' origin/main      # timestamps in the transcript are Europe/Paris
```

## Kaggle / laya

One-liner only — the commands and watchers live in `docs/kaggle-lane.md` and
`docs/laya-lane.md`; launch discipline (setsid, watcher-always, self-closing
sessions) in `docs/runbook.md`.
