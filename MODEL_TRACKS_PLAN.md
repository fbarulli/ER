# ER model development: three tracks

Date: 2026-09-30. Status: development plan; new tracks are not implemented.

Owner direction: develop the current model, GNN-only, and GNN plus the current
model as separate, comparable tracks. This supersedes the earlier TODO decision
to skip GNN development. SID and RQ-VAE are outside this plan.

## Track definitions

ANN is a retrieval algorithm, not an embedding model. All three tracks may use
ANN to search their embeddings.

| Track | Representation | Retrieval | Final scoring |
|---|---|---|---|
| A — current text model | MiniLM-L6 product-text embeddings | Existing HNSW ANN | Existing scoring; optional attribute-aware reranking |
| B — GNN-only | Learned graph embeddings from structured attributes and relationships | HNSW over GNN embeddings | GNN similarity or a learned pair scorer |
| C — text + GNN | MiniLM embeddings plus structured features and graph context | Start with existing text ANN | Learned fusion of text, graph, and attribute evidence |

For the first comparison, **GNN-only means no MiniLM features, scores, or
MiniLM-derived graph edges**. This isolates the contribution of structured
attributes and relationships. It is not a structure-only model: node features
are necessary to distinguish products and support unseen or isolated listings.

## What exists and can be reused

- `config/training_ANN.yaml`: MiniLM-L6/MNRL settings and HNSW settings; current
  configured retrieval budget is top-k 50.
- `src/core/model_input.py`: shared product-text composition.
- `src/training/prepared_bundle.py`, `train_prepared.py`, and `training.py`:
  prepared data and current model-training infrastructure.
- `src/training/hnsw_index.py`: persisted vector indexing.
- `src/training/build_ann_index.py`: current text-model index entry point.
  GNN use requires an adapter because this entry point wraps `RandMatcher`.
- `src/training/ann_refresh.py`: text-embedding refresh workflow to extend or
  mirror for graph artifacts.
- `src/core/product_identity.py`: shared structured identity descriptors and
  conflicts; reuse their extraction rather than introducing another parser.
- `src/training/folds.py`: shared split derivation through `derive_holdout`.
- `src/training/build_final_validation.py` and `evaluate_models.py`: final
  validation artifacts and evaluation infrastructure to extend.

`sid_graph.py` is deterministic constrained clustering, not a trained GNN.
`sid_hybrid.py` combines SID and cosine scores, not learned graph embeddings.
Neither is the implementation of tracks B or C.

## Shared foundations — build first

### 1. Freeze the data and evaluation contract

- Record dataset, canonical attributes, labels, fold map, and configuration
  hashes in one experiment manifest. Use stable listing IDs for vector/node
  alignment and entity IDs for split accounting.
- Finish the final-validation migration: retire the obsolete 3k/5k paths and
  update callers, including the ANN launcher. `run_ann_full_data.py` currently
  pins those files and row counts; do not launch it unchanged for this study.
- Keep one component-safe split for all tracks. Identity links used to derive
  the split are not automatically allowed as model-input edges.
- Train on training labels, select models and thresholds on dev, and report
  test results after selection. Do not change splits between tracks.
- Exclude scored pairs with trained-on endpoints under the shared protocol.
  Resolve the thin dev/test population and negative-straddle policy before
  treating small improvements as conclusive.
- Slice attributes are sets and may differ between feeds. Reconcile bucket
  definitions; do not require raw `v1 == v2`. Report unsupported slices without
  decision floors. Keep augmentation probes separate from clean matching.

### 2. Separate representation, retrieval, and scoring

Define shared interfaces for:

- Encoding listings into vectors, with IDs, dimensionality, and normalization.
- Building/querying an index and returning candidate IDs and scores.
- Scoring pairs and applying the same final identity-conflict policy.
- Persisting checkpoints, vector caches, indexes, and experiment manifests.

Allow numerical pair scores to differ by track. Calibrate each track on dev;
do not compare them at one arbitrary cosine threshold. Report both model-only
results and results after the shared conflict policy.

### 3. Shared configuration and artifact lifecycle

- Register track selection, feature definitions, graph relations, model size,
  losses, sampling, seeds, and artifact paths in validated SSOT configuration.
- Keep outputs in separate per-track/per-run locations and bind every index to
  its encoder/checkpoint, dataset, feature, and graph hashes.
- Persist labels used, training populations, actual gradient participation,
  runtime, memory, and refresh cost. Avoid overwriting baseline artifacts.
- Reuse existing telemetry where applicable; complete MNRL coverage telemetry
  and diet checks before running new text/hybrid training.

## Track A — current MiniLM + ANN

### A0: reproducible baseline

1. Identify and record the actual deployed/current checkpoint; distinguish
   pretrained MiniLM from fine-tuned MiniLM rather than assuming either.
2. Encode the frozen catalog and build the current HNSW index.
3. Measure retrieval recall@k and final matching metrics on the shared split.
4. Reproduce current model training with the corrected bundles, split, diet
   checks, and checkpoint telemetry.

Deliverables: baseline checkpoint reference, embeddings, index, candidate
outputs, scored pairs, and evaluation report.

### A1: attribute-aware extension

Keep one overall embedding for initial retrieval. Add attribute comparisons
to candidate scoring before creating multiple retrieval indexes.

| Field | Initial representation |
|---|---|
| Title/description | Existing MiniLM embedding |
| Flavor/descriptive variant | Normalized sets; experiment with a separate MiniLM field embedding |
| Brand | Normalized identity/alias features |
| Volume and pack count | Numeric values, units, and explicit differences |
| Sweetener, carbonation, package type | Categorical/set features |
| Missing fields | Explicit availability indicators; absence is not disagreement |
| Barcode | Ground truth/identity checks; not an embedding or blind matching feature |

Train a small scorer combining global similarity and attribute evidence using
training pairs. Fit/calibrate on dev. Numeric distinctions and categorical
conflicts must remain visible rather than relying on text cosine alone.

Optional next steps: attribute retrieval if recall@k exposes missing candidates;
a Ditto-style pair cross-encoder as a separate reranking experiment if simpler
scoring leaves ambiguity. Neither is a fourth required track.

## Shared graph builder — required by B and C

### Initial graph schema

- Listing nodes: one per retailer listing with stable IDs.
- Attribute-value nodes: normalized brand, category, flavor, sweetener,
  carbonation, package type, and pack values where available.
- Typed listing-to-attribute edges, with reverse relations for message passing.
- Numeric listing features for volume/pack plus field-availability indicators.
- Candidate listing-to-listing relations are optional later experiments. Keep
  them distinct from verified identity and attribute membership.

Sharing a flavor or brand means shared context, not identity. Avoid turning
every attribute group into a clique of listings. Audit high-degree hubs and
use relation-aware neighbor sampling.

The same base graph construction is used for B and C. Text-derived candidate
edges may be added to C only as a separately reported ablation.

### Leakage and new-listing behavior

- Start with attribute relations available at inference; exclude barcode
  features, GTIN equality edges, and verified match labels from model inputs.
- Labels used to split or supervise pairs must not be visible as scored
  identity edges. Do not union listings into one model node using hidden truth.
- Build training message-passing graphs from training listings only. Fit
  feature vocabularies, numeric scaling, and learned statistics on train.
- At inference, attach unseen listings through observable attributes and use
  unknown-value handling for unseen categories. No evaluation labels or
  gradients enter the graph or encoder.
- Define whether test listings are encoded individually with permitted catalog
  context or jointly as a batch. Use one declared context protocol for B/C;
  report joint/transductive inference separately from strict inductive results.
- Ensure a listing without usable neighbors still gets an embedding from its
  own features; report this population explicitly.

Deliverables: graph tables, relation schema, feature vocabulary, graph manifest,
and census of node/edge counts, missingness, degree, components, conflicts, and
unseen values. Include examples of cross-flavor/cross-pack neighborhoods.

## Track B — GNN-only

### Initial model and training

- Start with a small, two-layer relation-aware GraphSAGE implementation, with
  categorical feature embeddings and numeric feature projections. Use separate
  transformations for different relation types.
- Start with 128–256 output dimensions; choose size using dev results.
- Add the graph library dependency and verify compatibility with the existing
  PyTorch runtime before committing to an implementation.
- Train on verified positive pairs and hard negatives with a metric-learning
  objective; begin with contrastive/triplet loss that directly trains vector
  similarity. Compare a learned pair head later if needed.
- Use seed-controlled neighbor sampling and training-only negative mining.
  Prevent known equivalent entities from becoming false negatives.
- Reuse label provenance and conflict checks. Text-mask augmentations cannot
  simply be copied into this track: graph perturbations need their own
  label-preservation rules. Start from clean pairs and report exposures.

Deliverables: graph encoder, training entry point, checkpoint, graph embeddings,
HNSW adapter/index, inference/refresh workflow, and reports.

Required controls: attribute-only encoder without message passing, one/two-hop
comparison, and relation-preserving shuffled-edge control. These determine
whether graph relationships add value beyond the input attributes.

Pretrained graph weights are an optional experiment after this baseline.
Assess task, relation, feature, checkpoint availability, license, and runtime
compatibility before selecting one. Pretraining is not a prerequisite, and
scratch training is not assumed to be the eventual winner.

## Track C — MiniLM + GNN

### C0: frozen text encoder plus graph training

1. Cache embeddings from the exact A0 checkpoint.
2. Add these vectors to the structured listing features used by B.
3. Train the graph layer using the same base relations and supervised pairs.
4. Retrieve with the existing text ANN first, then train a scorer combining
   text similarity, graph similarity, and the shared attribute features.
5. Evaluate against A0, A1, and B. Attribute scoring alone must not receive
   credit as a graph improvement.

Retain a direct text path in fusion so the model can use text evidence when a
listing has sparse or unhelpful neighbors.

### C1: retrieval and joint-training extensions

- If graph signals recover missed candidates, test the union of text-ANN and
  graph-ANN candidates under the same final candidate budget.
- Compare reranking against a learned combined embedding in one ANN index.
- Unfreeze MiniLM only after the frozen version shows value; this requires
  checkpoint-bound cache/index regeneration and adds compute and attribution
  complexity.

Deliverables: hybrid checkpoint, text/graph caches, fusion scorer, candidate
retrieval policy, indexes, and reports. Include a graph-disabled ablation.

## Evaluation and decision rules

Report two comparisons: fixed candidate pairs to isolate scoring, and complete
retrieval-to-decision evaluation to measure practical matching quality.

| Measurement | Purpose |
|---|---|
| Recall@k at matched candidate budgets | True matches available to the scorer |
| P@R95, PR-AUC, and recall at an agreed precision | Matching quality and false merges |
| Flavor/sweetener/volume/pack/package slices | Critical product distinctions |
| Unseen, sparse-neighborhood, isolated, missing-field slices | Generalization and coverage |
| Encoding/indexing/query latency, memory, refresh time | Operational cost |
| Paired confidence intervals and repeated seeds | Strength and stability of measured gains |

The existing pair CSV alone is insufficient for recall@k: define held-out
queries, eligible catalog targets, and known relevant matches from identity
truth. Mask barcode information from blind inference. Document incomplete
truth and catalog coverage instead of treating unlabeled pairs as negatives.

Choose the primary deployment objective and minimum useful improvement before
examining test results. Select models on dev; report test once. A graph track
must improve the chosen objective without unacceptable retrieval recall,
critical-slice, or runtime regressions. No track is presumed to win.

## Colab training integration

Code review: 2026-09-30. These findings describe the local checkout; no Colab
runtime was launched or inspected as part of this planning review.

### Current lifecycle and reusable components

`colab_backend.py` delegates to `src/cli/colab.py`. The launcher provisions a
session, checks out the configured repository/branch, installs dependencies,
validates model files, and starts detached training workers. Workers have
separate output directories and live logs/status. Results are downloaded in a
verified archive; completed best checkpoints also have an incremental sync
path. The launcher tears down the VM in `finally` unless CPU keep-alive was
explicitly requested. GPU selection requires `--allow-gpu`, and GPU keep-alive
is refused by the current launcher.

There are two prepared-input paths:

- The default single-worker full train, with no model/sample/resume override,
  selects `colab.full_prepared_bundles` from the remote Git checkout.
- Customized training can build bundles locally, cache them by input content,
  and upload them. `_build_local_training_bundles` invokes
  `training.train --prepare-bundle`; Colab consumes them through
  `training.train_prepared`.

Standard remote preparation is disabled in config. Preparing data locally and
training remotely should remain the model for all three tracks. However,
checkout-native bundles bypass a local rebuild: their manifests must also
pass the experiment's freshness and diet checks.

Successful worker completion can invoke `training.complete_colab_worker`, which
resolves a text checkpoint and runs `predict_items` for catalog inference.
Full-catalog inference can include trained-on listings; it is an operational
output, not automatically held-out evaluation. Shared dev/test evaluation
must retain the fold-map exclusions from this plan.

### Findings to address before the three-track launches

| Area | Current behavior | Required work |
|---|---|---|
| Remote source | Fetches/checks out the configured `training` branch | Pin and record an immutable commit for each experiment; ensure local bundle producers and remote consumers are compatible |
| Split assumptions | Lifecycle preflight and config still reference the old 5k sample/complement; default full training has a separate checkout-input path | Migrate preflight, uploads, completion provenance, launchers, and tests to the final split contract |
| Bundle checks | Local rebuild validates bundle shape; inspected builder does not execute the diet gate | Gate newly built, cached, and checkout-native inputs before training; apply text diet rules to A/C, and define separate graph exposure checks for B |
| Track dispatch | Trainer selection and model-file verification assume text models | Add validated track dispatch and track-specific preparation, verification, training, and completion adapters |
| Dependencies | Config owns `prepared`/`full` package lists; graph dependencies are absent | Add a graph runtime profile and test graph-library compatibility against the actual Colab Python/PyTorch/CUDA versions |
| Completion | Calls text-specific `predict_items` and plots cosine scores | Add graph/hybrid inference adapters and model-specific scoring reports; reuse common provenance and evaluation |
| Checkpoint collection | Finds Transformer-style checkpoint directories and `trainer_state.json` | Give graph/hybrid checkpoints a compatible manifest/selection contract or generalize collection; verify the selected model is actually downloaded |
| Resume | Inspected resume restoration uses DVC pointers; config has `dvc_enabled: false` | Provide and test a local-artifact upload/restore path or explicitly reject unsupported resume; persist optimizer, scheduler, RNG, and sampler state |
| Tracking | Prepared trainer continues when W&B is disabled | Keep artifact collection and local records sufficient; fix stale launcher comments claiming W&B is mandatory |

Local uncommitted changes are not transmitted by a remote branch checkout.
Before running an experiment, the intended code must be available at the
pinned remote revision, and artifact manifests must identify that revision.

### Colab workload per track

| Track | Prepare locally | Run on Colab | Retrieve and verify |
|---|---|---|---|
| A | Text bundles, labels, split, augmentation audits | MiniLM training, checkpoint evaluation, final encoding/inference | Text checkpoint, telemetry, vectors/index as configured, scored results |
| B | Typed graph, structured feature vocabulary, labels/split, graph census | GNN training with neighbor sampling, graph inference and evaluation | GNN checkpoint/state, vocabulary/schema, embeddings/index, graph manifest, reports |
| C0 | Same graph plus checkpoint-bound frozen A0 text-vector cache | Graph layer and fusion training, hybrid inference/evaluation | Graph/fusion checkpoint, exact text-checkpoint reference, cache hashes, vectors/indexes, reports |
| C1 | Graph plus text inputs; immutable feature/split manifests | Joint fine-tuning only after C0 is validated | Both encoder states, regenerated vectors/indexes, complete provenance |

Start with one trainer per VM and run tracks sequentially against the frozen
data. Existing `dual-train` means two text-training workers; it is not GNN/text
fusion. Increase concurrency only after measuring host RAM and GPU memory.
Select the GPU from a real smoke profile rather than assuming T4 or A100 is
needed. Text-vector caching for C0 can happen locally or as a separate encoding
job; bind the cache to the exact A0 checkpoint in either case.

### Colab readiness checks

1. Local preflight for each track: validate commit/input compatibility, split,
   manifests, dependencies, and outputs before provisioning.
2. Small real-data runtime smoke: train, select a checkpoint, infer, download,
   verify, and tear down for A, B, and C. Use a graph sample that retains usable
   neighborhoods; the current text smoke population is not automatically a
   representative graph smoke. Smoke inference is not a quality benchmark.
3. Interruption/resume smoke: verify model and optimizer/sampler state restore
   and prevent cross-track or cross-manifest reuse.
4. Verify preservation of the selected checkpoint and required manifests
   before normal teardown. Exercise failure handling and incremental recovery;
   automatic teardown makes artifact collection part of correctness.
5. Full run: publish runtime, peak memory, throughput, graph sampling coverage,
   transfer size/time, and the shared quality report for each track.

Track-specific CLI flags and graph worker entry points are proposed work, not
existing runnable commands. Do not use the old pinned full-training launcher
unchanged for this experiment.

## Build sequence and completion checkpoints

1. **Shared contract + A0:** finish validation migration and telemetry/diet
   blockers, capture the current checkpoint, and publish baseline results.
2. **A1 + graph census:** test attribute scoring; construct and audit the shared
   graph. Do not start expensive graph training on an unaudited edge population.
3. **B baseline:** implement GNN training/inference and HNSW integration; run
   attribute-only and shuffled-edge controls.
4. **C0 baseline:** train the graph layer with frozen A0 embeddings and evaluate
   fusion against the attribute-aware text baseline.
5. **Controlled extensions:** pretrained GNN transfer, additional relations,
   retrieval fusion, or joint fine-tuning only in separate named experiments.
6. **Selection:** publish one comparison report with quality, uncertainty,
   failure examples, resource costs, and a deployment recommendation.

For each implementation step: map affected callers, validate with synthetic
data, run a small real-data smoke, then run the frozen full-data experiment.
Add guard tests for silent failures: node/vector ID alignment, split/label
leakage, unseen/missing features, cache invalidation, and index normalization.
Model development is complete when all three tracks have reproducible
checkpoints, inference artifacts, and comparable evaluation reports.
