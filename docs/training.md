# Training

Two halves: build the inputs locally, then train on a Colab GPU.

```bash
# 1. local: rebuild inputs
PYTHONPATH=src .venv/bin/python -m training.prepare_all

# 2. remote: train on a T4
bash scripts/run_full_training.sh \
  --prepared-input-package results/training_prep/<run>/all_tracks_inputs.tar.zst
```

## The three tracks

One runtime, three workers, one control channel.

| Track | Inputs | Trains |
|---|---|---|
| **A** `text` | MiniLM-L6 text + a structured channel | The text encoder. Loss is MNRL. |
| **B** `gnn_only` | Structured attributes, numeric features, typed relations | Graph encoder + pair scorer. No text, no text-derived edges. |
| **C** `hybrid` | The same graph **plus** exact frozen A0 MiniLM vectors | Graph encoder + fusion. Baseline MiniLM stays frozen. |

B and C share one train-only attribute graph and vocabulary. C's frozen vectors come
from the **baseline** checkpoint, not from the concurrently fine-tuned Track A — so A
and C never contaminate each other.

ANN/HNSW retrieves vectors; it is not a fourth model. SID and RQ-VAE are out of scope.

Graph training is one full-graph step per epoch with classification plus cosine
metric loss. Query encoding is separate: each row encodes against a fixed
train-only context, so unused rows never touch the loss.

## Launch

```bash
# full suite on GPU
PYTHONPATH=src .venv/bin/python colab_backend.py --what tracks \
  --tracks-config config/model_tracks.yaml \
  --prepared-input-package results/training_prep/<run>/all_tracks_inputs.tar.zst \
  --gpu T4 --allow-gpu

# CPU smoke
er-colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU
# (scripts/run_colab_smoke.sh uses the legacy `--what smoke` lane, which
# src/cli/colab.py rejects: "legacy smoke does not preserve the shared
# component holdout". The track-config above is the sanctioned path.)
```

`COLAB_GPU` overrides the runtime. Only CPU lanes accept `--keep-alive`.

Use `colab_backend.py` or the installed `er-colab`, both of which delegate to
`cli.colab.main`. **Do not use `python -m cli.colab`** — it creates a separate
`__main__` instance and can make the adapter read the default CPU setting even when
you asked for a GPU.

To reuse an existing package (e.g. `results/model_tracks/<tag>__inputs.tar.zst`) when the local tree has diverged since it was built, set `ER_SKIP_CONFIG_VERIFY=1` to bypass the inline-config and source-hash checks in `verify_current`.

## Launch lifecycle

1. Check the Colab CLI and authorization.
2. Take a per-session launcher lock — two different sessions do not contend.
3. **Validate the package before provisioning an accelerator.**
4. Reuse a live session if the control handshake verifies; never re-allocate an
   owner-launched VM.
5. Clone the pushed `origin/main` at depth one, then restore the package's exact
   source revision.
6. Install the CPU/CUDA PyTorch runtime, then graph requirements.
7. Export the frozen baseline embeddings — before hybrid training is released.
8. Run the three workers in parallel over MPS.
9. After checkpoint selection, each track exports vectors and runs its own
   interventions from its own selected checkpoint.
10. Download a compressed archive, verify SHA-256 and the inventory, shut down.
11. Local CPU postprocessing: reports, calibration, indexes, ablation analysis.

**The VM runs pushed `origin/main`, not your working tree.** Anything it needs must
be committed and pushed first. The only overlay is the uploaded raw export.

GPU workers own optimization and embedding forward passes. Everything else —
scored-pair reports, calibration, HNSW indexes, attribute slices, ablation
analysis — runs locally on CPU after download.

## Freshness

A package is rejected before provisioning if any of these changed since it was
built:

- source bytes (every `.py` under `src/` and `scripts/`)
- config bytes (every file under `config/`)
- the raw input CSV
- the checkpoint
- the smoke inputs

Editing any source file invalidates in-flight resume. That is the intended
behaviour, not a bug.

## Resume

Preparation-stage resume lives in [data-prep.md](data-prep.md#resume).

Training resume is a launcher flag and needs the original package plus a verified
recovery archive:

```bash
colab_backend.py --what tracks --resume-run <tag> ...
```

Text workers restore optimizer, scheduler, RNG and checkpoint paths. Graph workers
can finish postprocessing after their final epoch. Only unfinished workers restart.
Older interrupted runs without resume provenance are rejected rather than guessed at.

Changed config, source or input hashes require a new preparation and a new run.

## Precision and batch sizes

| Setting | Value |
|---|---|
| Text CUDA | native BF16 when supported, else FP16 with gradient scaling (explicitly including T4) |
| Text CPU | float32 |
| CPU smoke batch | 32 |
| GPU training / export / eval / ablation batch | 64 |
| Graph | full batch |

A T4 is not assumed to support BF16 — test FP16 with the scaler first.

## Splits and thresholds

One component split is shared by all three tracks. Train on train labels, select
checkpoints and thresholds on dev, report test only after selection.

`report_test: false` in the suite config means test reporting is suite-controlled.
`fixed_threshold: 0.55` is the lane's operating cosine threshold. Operating targets
come from config, never inline: `operating_precision: 0.95`,
`operating_recall: 0.95`, `operating_thresholds.balanced_review: 0.787`,
`high_precision: 0.85`.

GTIN and identity links may define truth and split components, but they are
excluded from blind model features and from graph edges.

Synthetic or masked probes never substitute for real held-out measurement.

## Profiling

With `profiling: true` the supervisor samples GPU activity every second and
supervisor/descendant CPU time, RSS and disk I/O every two seconds, from preflight
through every forward pass. Samples land in the result archive under
`resource_profile/`.

Sampling does not synchronize CUDA and does not change training behaviour. But it
is not free — measure the overhead before treating profiled throughput as a
production number.

GPU attribution under MPS is device-wide, not per worker. Memory activity is not
allocated VRAM.

## Open items

- **No GPU run has ever been performed on these paths.** Throughput and peak memory
  are unmeasured. CPU smoke cannot establish CUDA or MPS readiness.
- `memory_reservations_gb` is empty, so the launcher does **not** enforce summed
  worker peaks plus headroom. Populate measured peaks before claiming parallel GPU
  readiness.
- No remote interruption/recovery experiment has been run. Local CPU validation
  confirmed all three tracks and correct rejection of mid-run source changes, but
  not a real remote interruption.
- Checkpoints are written every epoch on purpose, to preserve interrupted-run
  resume. Changing that needs profiler evidence.

## Related

- [runbook.md](runbook.md) — every command, and the traps
- [pipeline.md](pipeline.md) — how the data flows
- [audits.md](audits.md) — what the gate audits concluded
