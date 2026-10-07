# Colab lane

Consolidated CPU + GPU remote-training lanes for Colab. The CPU bundle
generation is happy in the dedicated bundles lane and the kaggle lane is the
network transport for other data. Everything Colab shaped lives here.

## Structure (2026-10-06 consolidation)

| file | role |
|---|---|
| `src/cli/colab_lane.py` | the lane classes: `ColabLaneBase` (transport dial-ins + receipts + shared contracts), `ColabCPULane` (committed-export delivery + CPU prep parity), `ColabGPULane` (accelerator/retention boundary gates) |
| `src/cli/colab.py` | GPU launcher surface facade (`train / tracks / dual-train / hpo / sims / mixed / smoke / stop`) plus the legacy `--what bundle` upload lane |
| `src/cli/colab_bundle.py` | CPU committed-export delivery facade |
| `src/cli/colab_data_bundle_prep.py` | CPU data-bundle prep facade (high-RAM shape, dual transcripts, 2-parallel cap) |
| `src/cli/colab_cli_entry.py` | read-only-home-safe entry point for the installed Colab CLI |
| `src/core/schemas.py` | `ColabBundlePlan` (additive dry-run receipt) |

`ColabLaneBase` resolves every transport dial-in through the `cli.colab`
module namespace at call time — one patch surface for the offline fakes, no
captured bindings.

## Commands

GPU lane (`er-colab`, or `colab_backend.py` per the runbook):

```bash
er-colab --what smoke --gpu CPU
er-colab --what tracks --tracks-config config/model_tracks.yaml --gpu T4 --allow-gpu
er-colab --what stop
```

CPU committed-export delivery (offline plan / dry run added 2026-10-06):

```bash
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --preflight-only   # plan receipt only
PYTHONPATH=src .venv/bin/python -m cli.colab_bundle --dataset-csv dataset_50pct.csv
```

## Fix inheritance

Delivered behavior (bugfix granularity) is unchanged from the
pre-consolidation lanes. git-history fixes carried into the consolidated
classes: no-upload committed export remap (b1f116f), unbuffered + pathlib-
joined prepare telemetry (8b41dba/1952b71), ablation skipped on CPU bundle
lanes (49fc937), proven POPEN+status polling with transient-probe tolerance
(be6672b/58aa827/31083da/46e8324), delivery member list (8ddc614), delivery
root contract (4d40d1e), resume triplet (d264af1), recovered step_trace
tracebacks (45547e6), serialized control requests (1943073), both-transcript
worker forwarding (1e18b36), probe destructor-noise suppression (d76f39f),
branch fetch instead of commit (2a3baa9/b370ff1), bounded prewarm waits +
interpreter handover (6892ca2), keep-alive boundary (3fdc039/18ffe7a),
session-qualified transcripts (d5d96d4), lock released on session loss
(378addd), setup timing (33a535b). The preparation pin system is gone by
owner ruling (18c8d9c); eae4d1d was the pin-era re-write fix and no longer
applies.

## References

- [docs/kaggle-lane.md](kaggle-lane.md) — the model for this shape
- [docs/runbook.md](runbook.md) — launch discipline (systemd-run, live-session rules)
- [docs/training.md](training.md) — networks and runbook for training
