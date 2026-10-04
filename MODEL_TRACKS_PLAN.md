# ER model development: three tracks

Date: 2026-09-30. Status: three-track Colab dispatch, prepared inputs,
post-training reports, artifact collection and profiling are wired;
100-listing CPU Colab verification passed for all three tracks, including
postprocessing, existing eight-class attribute reports and profiler traces
(run `0930T214957725080Z`, 231 seconds of parallel worker execution).
The downloaded ZIP passed inventory and SHA verification. Smoke publication
was disabled. Full runs publish the complete ZIP through verified DVC storage
and commit selected inference models plus DVC references for repo clones;
missing DVC credentials fail before launch. Losses and report persistence now
live in shared `training/losses.py` and `training/report_rows.py` modules.
Full suites also upload immutable checkpoint generations during training and
completed postprocessing artifacts per track through isolated background
publishers. Each generation retains durable DVC references; upload failures
propagate before successful completion. The final ZIP remains a consolidated
publication, rather than the first persistence point.
Full-data comparison and
GPU performance measurements remain pending. See
[src/graph_tracks/README.md](src/graph_tracks/README.md) for current runnable
commands, verified tests, W&B/DVC behavior and architecture limits. The initial
model is full-batch typed two-hop aggregation, not sampled GraphSAGE.

Additional Colab CPU verification: `0930T221351242600Z` passed a three-epoch
budget on 100 listings. Graph tracks completed three epochs; native text
finished at epoch two with existing stopping behavior. Thirteen incremental
generations and the final suite ZIP passed DVC clean-pull verification.
All three worker logs contain training output, and each track retained reports
and profiler traces. The suite's parallel workers took 758.7 seconds including
publication. Recovery pointers are committed; smoke models are not deployed.
In-process DVC configuration subsequently passed on Colab in 2.33 seconds,
removing repeated setup command startup. Live per-track forwarding now feeds
the dedicated training log as well as the supervisor stream.

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
- `src/core/sku_identity.py`: shared structured identity descriptors and
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

- Start with attribute relations available at inference; exclude gtin
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
truth. Mask gtin information from blind inference. Document incomplete
truth and catalog coverage instead of treating unlabeled pairs as negatives.

Choose the primary deployment objective and minimum useful improvement before
examining test results. Select models on dev; report test once. A graph track
must improve the chosen objective without unacceptable retrieval recall,
critical-slice, or runtime regressions. No track is presumed to win.

## Preparation and training instructions

Use [TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md) as the single
operational guide. It owns the one-command CSV/input generation, active
negative-supply mode, offline batch preparation, frozen hybrid embedding
request, GPU lifecycle, verification, and artifact inventory.

`training.prepare_all` prepares all three tracks. B/C currently implement
full-batch typed two-hop aggregation; neighbor sampling remains a planned
extension. Hybrid baseline vectors are generated from locally frozen tokens
on the shared GPU before hybrid training. A new negative-supply CSV does not
activate that lane or automatically create graph augmentations.

## Build sequence and completion checkpoints

Use the `colab_backend.py` or installed `er-colab` launch command in
[TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md); it shares the selected
runtime with the all-track adapter.

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
