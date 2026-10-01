# Training log transparency

Text, GNN-only, and hybrid workers retain both raw stdout/stderr and durable structured lifecycle events. Logging is independent of W&B, MLflow, and DVC publication.

| Evidence | Location in suite results |
| --- | --- |
| Input validation, configuration, worker selection, barriers, CPU budget, heartbeats, exits, failures | `suite_events.jsonl` |
| Full worker stdout/stderr, training progress, checkpoints, evaluation and report substeps | `<track>__worker.log` |
| Worker stage transitions, command, requested/actual resume, completion inventory, failure traceback | `<track>/worker_events.jsonl` |
| Epoch losses, learning rates, population metrics and gradients | Track training CSV/JSONL artifacts and raw worker logs |
| Selected checkpoint and native trainer state | Track checkpoint manifests and trainer-state files |
| Archive digest, publication result and final suite outcome | Adjacent `<run>.events.jsonl` final collection log |

Events carry UTC timestamps, run/track identity and attempt IDs; worker events also reference the supervisor attempt. Resumed logs append and the live stream begins at the new attempt. UTF-8 characters split across reads are preserved. Event messages redact secret environment values and common credential patterns; credentials are never intentionally included in configuration dumps.

Graph logs show train/dev encoded listing counts, losses, dev metrics, gradient norms, best-checkpoint comparisons, vector export, index/retrieval stages, dev threshold selection, attribute reports, plots, final artifact paths, and explicit skipped test evaluation. Text logs show its existing training/pair/masking/calibration diagnostics plus selected-checkpoint provenance and matching postprocessing stages. Sample calibration unavailability and incomplete quality evidence stay explicit.

The `/training` dashboard exposes raw logs, JSONL diagnostics, live-status files and checkpoint state from local runs or downloaded ZIPs. Previews are bounded and label truncation; downloads stream complete files. Final collection logs are offered separately from the ZIP's earlier event snapshot. Older runs without saved logs are labeled as such. Traversal and symlink paths are rejected.

Reachable failed runtimes attempt checkpoint recovery and separately download checksummed log files, including failures before a suite manifest exists. A lost or terminated VM can make remote files unrecoverable; the launcher reports collection failure explicitly and retains its local streamed stage log. This is not a guarantee against loss of an unreachable runtime.

Validation includes lifecycle failure traces, all three worker identities, preflight failure retention, partial UTF-8 streams, resumed log behavior, archive recovery, bounded dashboard previews/full downloads, and score/gradient parity for efficiency changes. See `GPU_PERFORMANCE_AUDIT.md` for measured performance boundaries.

## CPU smoke evidence

Suite `transparency_cpu_20261001` completed text, GNN-only and hybrid training and postprocessing. The verified result ZIP retained raw logs of 19,152, 19,054 and 18,601 bytes respectively, per-track lifecycle JSONL events, and the suite snapshot. Its separate final event log has 20 events ending in suite completion. Dashboard archive previews and final event-log discovery were verified directly. Receipt: `results/model_tracks/transparency_cpu_20261001.verification.json`. This run is a local CPU smoke; the earlier Colab CPU smoke is documented in `TRAINING_SMOKE_REPORT_20261001.md`.
