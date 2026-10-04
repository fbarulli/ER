# ER training and data generation

This is the single operational guide for preparation, training, and artifact
collection. Entry points and shared contracts are mapped in
[COLAB_SURFACE_MAP.md](COLAB_SURFACE_MAP.md). Model selection and experiment design remain in
[MODEL_TRACKS_PLAN.md](MODEL_TRACKS_PLAN.md). Updated 2026-10-04.

## The three models

| Track | Model inputs | What is trained |
|---|---|---|
| A: text | MiniLM-L6 text plus the configured structured channel | Current MiniLM model; shipped loss is MNRL |
| B: `gnn_only` | Structured attributes, numeric features, typed attribute relations | Graph encoder and pair scorer; no MiniLM vectors or text-derived edges |
| C: `hybrid` | The same structured graph plus exact frozen A0 MiniLM vectors | Graph encoder and fusion/pair scorer; the baseline MiniLM stays frozen |

ANN/HNSW retrieves vectors; it is not a fourth model. SID/RQ-VAE are outside
this plan. The current graph implementation is full-batch typed two-hop mean
aggregation in PyTorch. Neighbor-sampled GraphSAGE and joint MiniLM/GNN
fine-tuning are later experiments, not current preparation requirements.

## One command generates the training inputs

Run from `ER` with the local environment and model files installed:

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all
```

The entry point is [src/training/prepare_all.py](src/training/prepare_all.py).
It rebuilds the CSVs, all three tracks' offline inputs, and one verified
`all_tracks_inputs.zip`. It does not start training or provision Colab.
A timestamped directory under `results/training_prep/` contains stage logs,
`manifest.json`, the package, and backups of replaced setup/bundle/lane files.
The manifest records the exact output paths, sizes, and SHA-256 hashes.

`config/model_tracks.yaml` chooses the setup directory, text bundle, text
model, epochs, ablation preparation, and GPU dispatch settings. Override it
with `--tracks-config <yaml>`. Model and epoch settings must agree with
`config/training.yaml`; preparation rejects incompatible settings before
rebuilding data. `--run-dir <directory>` chooses the preparation log directory.
`--negative-supply-run-tag <tag>` chooses the diagnostic lane output directory;
otherwise preparation uses its run directory's name. Tags must use only
letters, digits, underscores, and hyphens.

### Generation order

1. Deduplicate `dataset.csv`, retaining representative identity and removal accounting.
2. Build cross-country hard-positive pairs and the numeric-token reference; verify reference bytes.
3. Rebuild canonical records and gate results with the current extractors,
   vocabulary, reviewed identity policy, made-from evidence, and source-consistency flags.
4. Refresh the measured gate census in `config/training.yaml`, then rebuild labeled pairs.
5. Generate real-first negative-supply pairs and run the grouped real-vs-minted discriminator.
6. Rebuild final validation and the component fold map.
7. Build the shared eligible catalog, listing split map, clean graph pairs,
   graph features, lineage, and graph census.
8. Build the full text bundle, augmentation lineage, native training tokens,
   fixed objective datasets, and CPU/CUDA presentation plans for every epoch.
9. Prepare graph tensors and query batches, native text export tokens, the
   frozen-baseline embedding request, and configured ablation templates.
10. Validate all three tracks together and package the immutable source/config/input overlay.

The generator reuses one verified base payload within the run in gate mode.
Existing smoke inputs are checked for unchanged bytes. A stage failure stops
the pipeline and records its name and error; inspect its log before resuming.
If only packaging failed after `full_bundle`, reuse that run with
`--run-dir <same-directory> --resume-from suite_inputs`; final handoff checks
still verify the frozen graph, text bundle, CSVs, and checkpoint.
Use `--resume-from full_bundle` only to retry an interrupted bundle stage
with unchanged source, configuration, raw input, and checkpoint. Resume verifies
the saved graph/CSV inventory and the original smoke baseline. Configuration
or source changes require a fresh preparation run; older run manifests without
these provenance fields also require regeneration. Bundle retries use a fresh
shared-payload cache and preserve earlier bundle backups.

### Files produced

Default paths below are resolved through configuration; the run manifest is
the authoritative inventory.

| Artifact | Purpose |
|---|---|
| `data/dataset_deduped.csv`, `data/sku_to_rep.csv` | Catalog and raw-listing-to-representative mapping |
| Configured dedupe summary, offer-group, removal, and conflict CSVs | Closed row accounting and review evidence |
| `results/training/second04_pairs_positive.csv` | Cross-country positive candidates; consumer verifies volume agreement |
| `data/number_tokens_reference.csv` | Number/name interpretation reference |
| `data/canonical_records.csv`, `data/gate_results.csv` | Current extracted attributes and gate evidence |
| `data/labeled_pairs.csv` | Real entity-pair labels under the active threshold policy |
| `results/negative_supply/<tag>/pairs.csv` and `manifest.json` | Real partners, minted top-ups, edited-positive controls, lineage, and coverage |
| Preparation `discriminator.json` | Real-vs-minted separation verdict |
| `data/final_validation.csv`, `results/training/validation_fold_map.csv` | Shared component-safe validation population and split accounting |
| `data/track_setup/eligible_catalog.csv`, `listing_splits.csv`, `listing_pairs.csv` | Retained graph/text-export catalog, listing assignments, clean graph supervision |
| `data/track_setup/prepared/listings.json`, `pairs.csv`, `input_manifest.json` | Graph descriptors, pair rows, and source bindings |
| Graph `pair_lineage.json`, `report_attributes.json`, `graph_census.json` | Provenance and reporting dimensions |
| `data/track_setup/prepared/graph_plan.json`, `graph_inputs.npz` | Train-only vocabulary, full training/dev tensors, pair mappings, and complete query batches |
| `data/prepared/full/worker_1_baseline.pkl.gz` plus sidecar | Full text training bundle, native tokens, fixed datasets, and frozen epoch batches |
| Configured text-bundle path, default `data/track_setup/text_prepared.pkl.gz` plus sidecar | Suite copy of the same text bundle |
| `data/track_setup/prepared_text.npz`, `text_export_request.json` | Locally composed and tokenized text for selected-checkpoint export |
| `data/track_setup/embedding_inputs.json` | Frozen A0 hybrid embedding request, bound to tokens/checkpoint/catalog |
| `data/track_setup/ablation_templates/` | Offline intervention text, tokens, graph tensors, and per-track requests when enabled |
| Preparation `all_tracks_inputs.zip` | Verified portable input package for the three-track run |

### Which negative-generation approach is active?

`config/training.yaml` currently ships `negative_supply.mode: gate`.
The new lane is generated and audited on every preparation run, but that
setting keeps it diagnostic-only: generating a CSV does not switch training.
The active text path retains gate hard negatives and configured additional
miners. Graph tracks start from clean labeled listing pairs plus trusted
same-entity positive chains, as specified by the model plan.

The experimental lane blocks real different-GTIN candidates first, measures
anchors with a real one-attribute-different partner, and mints at most one
whitelisted volume/flavor move for an uncovered anchor. Its JSON spec can be
supplied with `EUROMONITOR_NEGATIVE_SUPPLY_SPEC`; the same spec governs mining
and discriminator thresholds. GTINs are read as strings, preserving zeros.

With `mode: lane`, `pairs_run_tag` is mandatory. The generator rebuilds that
lane before any consumer; text training replaces gate-derived negatives and
keeps minted partners training-only. A `SEPARABLE` discriminator verdict
stops lane-mode preparation. In gate mode it is recorded as a diagnostic
failure while the active gate preparation continues. `insufficient` means
there is not enough grouped evidence; it is not a successful safety result.

Lane activation still requires real-pair quality evidence and graph-specific
supervision work for a comparable all-track lane experiment. Text edits or
minted text partners are not automatically graph augmentations. Do not
claim shared new-lane supervision for B/C from a text-only mode switch.

## Offline batching and hybrid embeddings

Text objective datasets and every epoch's batches are fixed locally for CPU
and CUDA batch sizes. Controlled population weights come from
`training.batch_sampler`; exhausted populations redistribute slots and smaller
batches preserve remaining rows. Every objective row must occur exactly once
per epoch. MNRL duplicate checks inspect text fields, excluding telemetry.
The GPU worker consumes `FrozenBatchSampler` and native token tables; it
rejects missing/repeated/out-of-range indices or mismatched batch settings.

Sampled smoke runs reuse their saved objective rows and the selected device's
saved CPU/CUDA batch size. They permit configuration-hash drift in the row
plan, while still checking its data, loss, split fraction, sample marker,
seed, and valid fold structure. Full training retains strict configuration
and runtime batch-size checks. Smoke still requires a valid prepared plan;
this exception does not waive package freshness or archive integrity checks.

B/C use the same train-only attribute graph and vocabulary. Training runs one
full-graph optimization step per epoch, with classification plus cosine
metric loss. Query/export batching is separate: rows encode independently
against fixed training context, with complete ordered coverage.

Local preparation creates the hybrid embedding request and native tokens,
not the final baseline vectors. At the beginning of the shared GPU run,
`model_tracks.baseline_export.forward` encodes the exact frozen A0 checkpoint
into `shared_minilm__embeddings.npz` before releasing hybrid training. It does
not depend on the concurrently fine-tuned Track A checkpoint. After Track A
selection, its own vectors are exported from the selected checkpoint using
the already prepared tokens.

When ablations are enabled, the frozen baseline embedding interventions run
before the three training workers start. After checkpoint selection, text,
GNN-only, and hybrid each run their prepared interventions using their own
selected best checkpoint. Baseline and trained-text results remain separate.

Both exports enforce checkpoint/input hashes, float32, finite normalized
vectors, ID alignment, and atomic publication. Pydantic prepared-input models
validate token/graph row coverage. Input/config/parser changes require a fresh
preparation; cached vectors must pass current provenance checks.

## Resume and launch

If the CSV stages completed and their current manifests verify, continue the
remaining preparation with a fresh log directory:

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all --resume-from validation
```

This rechecks canonical/label manifests, regenerates the diagnostic lane,
and rebuilds validation, setup, bundles, tensors, requests, and package.
It is not a training-checkpoint resume.

Launch after successful generation:

```bash
PYTHONPATH=src .venv/bin/python colab_backend.py --what tracks \
  --tracks-config config/model_tracks.yaml \
  --prepared-input-package <preparation-run>/all_tracks_inputs.zip \
  --gpu T4 --allow-gpu
```

The default full workflow is GPU baseline embedding export, all three models,
inference exports and configured ablations, Colab DVC publication, shutdown,
local DVC collection, CPU reports, and final publication. Run it with the
configured T4 default using one command:

```bash
bash scripts/run_full_training.sh --prepared-input-package <preparation-run>/all_tracks_inputs.zip
```

`COLAB_GPU` overrides the full-run runtime. `scripts/run_colab_smoke.sh`
continues to request CPU explicitly.

Use the existing `colab_backend.py` entry point (or installed `er-colab`).
Both import `cli.colab.main`, sharing the selected runtime with the track
adapter. `python -m cli.colab` creates a separate `__main__` instance and can
make the adapter read the default CPU setting despite a requested GPU.

The launcher validates the package before accelerator provisioning, saves
immutable Git input transport, and starts isolated text/GNN/hybrid workers on
one runtime with one control channel and NVIDIA MPS. Use `--prepared-input-package` to reuse the completed package without
repeating CPU token/tensor preparation. Freshness checks reject changed
source, configuration, checkpoint, or input bytes before provisioning.
Without that flag, the launcher prepares a new package from existing local
CSVs; CSV generation belongs to `training.prepare_all`. Publishing suites require `DVC_API_KEY` before launch.
W&B credentials are only needed for online mode. Install the target
CPU/CUDA PyTorch runtime and `requirements/graph_tracks.txt` first; the local
MiniLM model directory must contain valid weights and tokenizer files.

CPU-only preparation preflight permits the declared GPU-pending hybrid cache.
A direct hybrid-worker preflight before that cache exists cannot pass.
Do not run the old manual CPU-cache command as a prerequisite for this suite.

For interrupted training, use the launcher's `--resume-run <tag>` with the
original prepared package and verified recovery archive. Changed config,
source, or input hashes require a new preparation/run.

## Outputs and evaluation

New full suites use `result_archive_format: tar.zst`: Zstandard level 1
compresses the training and final result archives. Existing ZIP results remain
readable; immutable prepared inputs and recovery archives retain ZIP format.
SHA-256 inventories and atomic publication apply to both result formats.
Python before 3.14 uses the configured `backports.zstd` runtime dependency.

Train on train labels; select checkpoints and thresholds on dev; report test
only after selection. Keep one component split across tracks. GTIN/identity
links may define truth and split components, but are excluded from blind
model features and graph edges. Synthetic/masked probes do not replace real
held-out quality measurements.

GPU workers own optimization and embedding forward passes. Local CPU
postprocessing owns scored-pair reports, calibration, HNSW indexes,
attribute/sparse-neighborhood slices, and configured ablation analysis.
For publishing suites, Colab pushes the complete training archive to DVC and
verifies a clean remote pull. The launcher validates and saves the small DVC
handoff receipt, stops Colab, then pulls and verifies the training archive from
DVC locally before CPU reporting. Suites with publication disabled collect the
archive directly before shutdown. The same `colab_backend.py` call owns this
entire lifecycle; no separate reporting command is required. Reports run against the input package's frozen source and config,
with a receipt recording any subsequent local changes. After local reports finish, DVC separately publishes the
completed archive containing every retained training checkpoint and the
embedding/ablation artifacts. Text checkpoint retention is unlimited;
inference publication and model ablations use only each track's selected best
checkpoint. Publication completes after CPU reports are included.
Collected artifacts bind vectors/indexes to encoder, catalog, policy, and
schema hashes. Record checkpoint selection, losses, exposure, runtime, memory,
and review failures. Compare model-only behavior and the shadow gate on real
pairs; thresholds and deployment decisions follow the model plan.

### Resource profiling

With suite `profiling: true`, the supervisor records aggregate GPU activity,
VRAM, clocks, power and temperature every second and supervisor/descendant CPU
time, RSS and disk-I/O counters every two seconds, from preflight through all
embedding, training and ablation forwards. Samples stop and flush before sealing
the training archive. GPU attribution is device-wide under MPS; memory activity
is not allocated VRAM or measured DRAM bandwidth. Unsupported GPU fields remain
N/A and CPU monitoring continues when `nvidia-smi` is unavailable.

Archive inventory hashing, compression and verification durations travel in the
DVC handoff receipt, with timestamped DVC command durations, and are retained in
the final local result archive under `resource_profile/`. Existing bounded
PyTorch traces include profiling overhead and cold-start steps. Sampling does
not synchronize CUDA or change training, evaluation, checkpoint or ablation
behavior. Measure sampling overhead before treating profiled throughput as an
unprofiled production estimate.
