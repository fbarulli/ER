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
#   transcript: logs/colab/lane_cpu.log (the launcher opens it; session auto-releases
#   after delivery — `--keep-alive` retains, CPU only; extra flags pass through:
#   --resume-*, --preflight-only, --install-delivery)

# GPU smoke (T4), detached
COLAB_GPU=T4 setsid bash scripts/run_colab_smoke.sh >/dev/null 2>&1 &
#   WARN: never redirect nohup/wrapper stdout onto the lane transcript — the lane
#   already holds the append fd; a second redirection duplicates lines through the
#   same file and a plain `tee` (non -a) punches the zero-filled NUL hole.

# Full E2E, all models (the suite matrix L entry)
er-colab --what tracks --tracks-config config/model_tracks.yaml --gpu T4 --allow-gpu
```

## How to check logs

```bash
tail -f logs/colab/lane_cpu.log          # CPU bundle lane transcript
tail -f logs/colab/lane_gpu.log          # GPU training lane transcript
tr -d '\0' < logs/colab/lane_gpu.log | tail   # if a stale NUL-punched log reads as binary

# progress: each captured chunk re-emits its LAST tqdm frame as one tagged line
# "[tqdm] <bar>" — read the newest for position, the count as churn evidence
grep -c "\[tqdm\]" logs/colab/lane_gpu.log
grep "\[tqdm\]" logs/colab/lane_gpu.log | tail -1

# session state + termination reason (user_requested = our release, pruned = Google reaped)
cat colab_cli_state/sessions.json
cat colab_cli_state/history/smoke-gpu.jsonl | tail   # per-session: smoke-cpu / smoke-gpu

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
