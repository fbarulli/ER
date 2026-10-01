# Training performance evidence — 2026-10-01

No paid GPU runtime was provisioned. GPU throughput and peak memory remain unmeasured; CPU measurements below are explicitly a local stage benchmark.

## Implemented graph efficiency changes

Both graph workers previously encoded every prepared listing during training and dev evaluation, then scored only pairs belonging to the respective split. Query encodings are independent against a context constructed exclusively from training listings (`AttributeGNN.context` / `encode`), so those unused rows do not contribute to the training objective or dev scores.

`src/graph_tracks/train.py` now tensorizes and encodes training support for the training pass and dev listings for the dev pass, with pair endpoints remapped to the appropriate local indices. The same vocabulary, training graph context, labels, objective, optimizer, deterministic setting, and selected-checkpoint criterion are retained. Report export still covers the declared complete catalog.

Current prepared population:

| Phase | Previous encoded listings | Current encoded listings |
| --- | ---: | ---: |
| Training | 27,852 | 13,927 |
| Dev evaluation | 27,852 | 6,902 |

Gradient telemetry now stacks parameter gradient norms on the device and transfers the result once, instead of requesting a scalar from the device for every model parameter. This retains the same norm and finite-gradient checks while removing repeated host synchronization requests.

## Measured evidence

Local CPU, two PyTorch threads, current prepared catalog, default GNN dimensions, identical weights and train pairs, no optimizer updates. One warmup and three measured forward/BCE/backward repetitions per variant; medians:

| Variant | Seconds |
| --- | ---: |
| Full-catalog train encoding | 0.880658 |
| Train-only encoding | 0.556954 |

The measured stage was 1.58 times faster on this CPU. This excludes dev evaluation, checkpoint I/O, report generation, and publication; it is not an end-to-end or GPU speed claim.

`tests/test_graph_split_encoding.py` checks both GNN and hybrid logits, complete classification/metric loss, parameter gradients, and dev embeddings against full-catalog encoding. Numerical tolerances account for different matrix batch sizes; they do not imply bitwise identical floating-point kernels. Existing worker/export/resume tests exercise actual selected checkpoints and completion.

## Observability completed in owned modules

Graph logs expose validated input paths and split pair counts, feature dimensions and parameter counts, resume checkpoint and completed epoch count, current/total epochs, loss components, dev metrics, elapsed epoch time, selected checkpoint decisions and comparison with the prior best, checkpoint completion manifests, postprocess configuration, test skips, evaluation summaries, and final artifact paths.

Text postprocessing prints the trainer-recorded selected checkpoint and metric, input pair counts, vector/index start and completion with durations, split evaluation, dev-fitted threshold, explicit test skips, retrieval/report phases, output paths, and completion duration. Supervisor-captured stdout/stderr preserves these messages.

## Remaining investigation boundaries

- GPU speed and memory should be measured on the actual target GPU before changing overlap reservations. The population reduction is demonstrated; a GPU throughput multiplier is not.
- GPU mixed precision, fused optimizer settings, and compilation were not changed in these owned modules. They require hardware-specific measurement and score/gradient validation, especially with deterministic relation pooling.
- Root owns shared worker thread limits and MPS scheduling. CPU oversubscription is a shared-launcher concern; no competing launcher/config edits were made here.
- Graph report substeps now log vector export, scoring, threshold selection, attribute reports, retrieval, plots, and report completion through supervisor-captured stdout.
- Checkpoints remain written each epoch to preserve interrupted-run resume and best-selection behavior. Publication and detailed I/O costs need profiler evidence before changing cadence.
- Text training batch/precision/data-loader internals are owned by another agent. This audit changes only its postprocessing logs.

## Text and supervisor efficiency follow-up

MNRL population telemetry now keeps a fixed detached `[2, number_of_populations]` device buffer for loss sums and counts, with host transfer only at epoch/report boundaries. Pair IDs stay on device; population lookups are cached. Warmup weights use tensor operations. CPU tests preserve exact objective and gradient values, population counts/means, epoch flushing and repeated-report behavior. GPU throughput remains unmeasured; floating-point telemetry reductions can differ in the last rounding digits.

Worker CPU thread budgets respect process CPU affinity and the number of active workers. A resumed single worker receives the available CPU share rather than the fixed one-third allocation. The effective budget is logged. Training batch sizes, precision, objectives, evaluation cadence, and checkpoint cadence remain unchanged.
