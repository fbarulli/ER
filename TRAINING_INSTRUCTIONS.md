# ER training instructions — all three model tracks

Consolidated from `MODEL_TRACKS_PLAN.md`, `src/graph_tracks/README.md`,
`DATA_TRAIN_FLOW.md`, `requirements/graph_tracks.txt` and `config/model_tracks.yaml`.
Status: preparation, smoke and dispatch are wired; GPU/full-data runs remain pending.

## Tracks

| Track | Resource | Representation | Retrieval | Final scoring |
|---|---|---|---|---|
| A — current text model | MiniLM-L6 product-text embeddings | Existing HNSW ANN | Existing scoring; optional attribute-aware reranking |
| B — `gnn_only` | Learned graph embeddings from structured attributes | HNSW over GNN embeddings | GNN similarity or learned pair scorer |
| C — hybrid | MiniLM vectors + structured features + graph context | Start with existing text ANN | Learned fusion scorer |

ANN is a retrieval algorithm, not an embedding model; all tracks may use it.
GNN-only means NO MiniLM features/scores/text-derived edges. `sid_graph.py` /
`sid_hybrid.py` are not the implementation of B or C.

## Requirements

### Environment

- Python 3.12/3.14.6, `torch==2.14.0` installed FIRST for the target CPU/CUDA
  runtime (no silent CPU fallback when CUDA is requested).
- Graph profile install:

  ```bash
  .venv/bin/python -m pip install -r requirements/graph_tracks.txt
  ```

  Pins: `hnswlib 0.8.0`, `sentence-transformers`, `wandb`, `dvc`, `matplotlib`.
  No torch-geometric, no MLflow.
- W&B: shipped configs use project `e-r`, mode `offline`. Online needs
  `WANDB_API_KEY`; `disabled` for tests. Credentials are never config values.
- DVC: enabled locally by default. Remote push is opt-in via
  `dvc: {enabled: true, remote: <path>, push: true}`. DagsHub uses
  `DVC_API_KEY`; HTTP basic auth uses `DVC_HTTP_USER`/`DVC_HTTP_PASSWORD`.
  Missing DVC credentials fail BEFORE launch on full runs.
- Run everything from ER with `PYTHONPATH=src` and `.venv/bin/python`.

### Data prerequisites

- Offline pipeline: raw export → `src/training/data_prep.py`
  (`run_within_brand_pipeline`) → `canonical_records.csv` + `gate_results.csv`
  → `pipeline.build_training_data` (positive/negative mining) → masking
  augmentation → component-safe split via `training.folds.derive_holdout`.
- Component-safe split shared across ALL tracks; never change splits between
  tracks. Train on train labels, select on dev, report test once.
- Identity links used to derive the split are not automatically allowed as
  model-input edges; label/gtin leakage is forbidden in graph inputs.
- Track-specific Colab workload (`MODEL_TRACKS_PLAN.md`):

  | Track | Prepare locally | Run on Colab | Retrieve and verify |
  |---|---|---|---|
  | A | Text bundles, labels, split, augmentation audits | MiniLM training, checkpoint evaluation, final encoding/inference | Text checkpoint, telemetry, vectors/index, scored results |
  | B | Typed graph, feature vocabulary, labels/split, graph census | GNN training with sampling, graph inference/eval | GNN checkpoint, vocab/schema, embeddings/index, manifest, reports |
  | C0 | Same graph + checkpoint-bound frozen A0 text cache | Graph layer + fusion training, hybrid inference/eval | Fusion checkpoint, exact checkpoint ref, cache hashes, reports |

## Required input files

All prepared inputs already exist in `data/track_setup/`.

### Tracks B and C (graph workers)

| File | Purpose |
|---|---|
| `data/track_setup/prepared/listings.json` | Listing features/text (graph nodes) |
| `data/track_setup/prepared/pairs.csv` | Labeled pairs: `sku_id1,sku_id2,label,split` (the only required CSV) |
| `data/track_setup/prepared/input_manifest.json` | Hashes binding catalog/splits/pairs/lineage; workers reject stale inputs |
| `data/track_setup/prepared/graph_plan.json` | Binds listing ID order, train-only vocab, numeric schema, pooling topology to source hashes |
| `data/track_setup/prepared/graph_inputs.npz` | Pre-tensorized int64 edges/pairs + float32 numerics — MUST exist for CUDA training/inference; CPU synthetic smoke may tensorize on the fly |
| `data/track_setup/prepared/pair_lineage.json` | Pair provenance (validated in preflight) |
| `data/track_setup/prepared/report_attributes.json` | Attribute slice definitions for reporting |
| `config/graph_tracks_gnn.yaml` / `config/graph_tracks_hybrid.yaml` | Track configs (curated copies: `data/track_setup/{gnn_only,hybrid}.yaml`) |
| `config/identity_reviews.json`, `config/vocabulary.json` | Policy files referenced by caches/manifests |

Track C additionally requires the frozen text cache (below); `gnn_only`
FORBIDS a `text_cache` (`src/graph_tracks/config.py:74-75`).

Generation CSVs (kept for provenance / re-prepare): `eligible_catalog.csv`,
`listing_splits.csv` (`sku_id,split`), `listing_pairs.csv`
(`sku_id1,sku_id2,label,split`). Both pair endpoints must belong to the
declared split; train and dev need both positives and negatives.

### Track C only — frozen MiniLM cache (produce BEFORE training)

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.text_cache \
  --catalog data/track_setup/eligible_catalog.csv \
  --checkpoint artifacts/models/all-MiniLM-L6-v2 \
  --output data/track_setup/shared_minilm__embeddings.npz
```

- Cache must come from the EXACT frozen A0 checkpoint; the NPZ carries
  checkpoint/model-input/catalog/identity-policy hashes.
- Hybrid training checks provenance against the prepared manifest, not just
  vector dimensions. Extra IDs allowed; production training requires the same
  source catalog fingerprint.
- Validated by `graph_tracks.preflight` before training.

### Track A (text/ANN)

- `data/track_setup/text_prepared.pkl.gz` (+ `.json` sidecar) — prepared text
  bundle; upstream: `data/canonical_records.csv`, `data/gate_results.csv`,
  deduped dataset (`load_dataset_deduped`), `config/training_ANN.yaml`.

### 3-track Colab dispatch

`config/model_tracks.yaml` references the above: `setup_dir:
data/track_setup`, `text_bundle: data/track_setup/text_prepared.pkl.gz`;
`epochs: 10`, `device: cuda`, `schedule: parallel`, `max_parallel: 3`,
`gpu_parallel_backend: mps`, `profiling: true`, `report_test: false`,
`publish_git: true`.

## Steps

### Step 0 — shared setup (all tracks, no training)

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.setup --output data/track_setup
PYTHONPATH=src .venv/bin/python -m graph_tracks.preflight \
  --config data/track_setup/gnn_only.yaml
PYTHONPATH=src .venv/bin/python -m graph_tracks.preflight \
  --config data/track_setup/hybrid.yaml
```

Setup builds listing assignments via the shared `derive_holdout` entry point,
records the local text checkpoint, exclusions and graph census, and creates the
per-track configs with test reporting disabled.

### Step 0b — regenerate graph tensors if missing

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.prepared_inputs \
  --listings data/track_setup/prepared/listings.json \
  --pairs data/track_setup/prepared/pairs.csv
```

### Step 1 — embed (Track C prerequisite)

Run the `graph_tracks.text_cache` command above, then re-run prefights.

### Step 2 — train

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.train \
  --config config/graph_tracks_gnn.yaml --run-tag experiment-001
PYTHONPATH=src .venv/bin/python -m graph_tracks.train \
  --config config/graph_tracks_hybrid.yaml --run-tag experiment-001
```

- Architecture: small two-layer relation-aware full-batch GraphSAGE
  (listing → attribute → listing) in plain PyTorch, categorical embeddings +
  numeric projections, 128–256 output dims, metric learning (contrastive/
  triplet) on verified positives and hard negatives, seed-controlled sampling,
  training-only negative mining.
- Track A: MiniLM/MNRL training per `config/training_ANN.yaml` with the
  prepared bundle and controlled batch sampler (optional).
- Each worker outputs to `<output_dir>/<track>__<run_tag>` (or
  `$EUROMONITOR_RESULTS_DIR`); per-epoch losses/dev metrics, gradient norms,
  heartbeats, integrity-marked checkpoints and DVC snapshot generations.
- Resume: `graph_tracks.train --config <updated>.yaml --run-tag
  <tag>-resumed --resume <checkpoint>/<track>__graph_model.pt` (restores
  model/scorer/optimizer/RNG; hashes and config must match).

### Step 3 — Colab 3-track launch (all parallel on one VM)

```bash
PYTHONPATH=src .venv/bin/python -m cli.colab --what tracks \
  --tracks-config config/model_tracks.yaml
```

- One control channel, isolated subprocess workers/outputs, NVIDIA MPS for
  GPU overlap (supervisor rejects unavailable MPS).
- Default path: single-worker full train via `colab.full_prepared_bundles`
  from the pinned remote checkout. Customized training: build bundles
  locally (`training.train --prepare-bundle`) and upload.
- Teardown happens in `finally` unless CPU keep-alive was requested; GPU
  keep-alive is refused; GPU selection requires `--allow-gpu`.

### Step 4 — post-training artifacts (automatic)

- Normalized vectors + HNSW indexes bound to provenance.
- Dev-fitted Youden threshold; test labels never select checkpoints.
- Metrics: ROC-AUC, PR-AUC, P@R95, accuracy, precision/recall/F1, confusion
  counts, pooled pair-ranking stats, attribute-availability slices,
  score/PR plots, within-split ANN retrieval reports.
- Portable packages:

  ```bash
  PYTHONPATH=src .venv/bin/python -m graph_tracks.worker_package \
    --config data/track_setup/gnn_only.yaml --output results/gnn_worker_setup.zip
  PYTHONPATH=src .venv/bin/python -m graph_tracks.bundle \
    --source results/graph_tracks/hybrid__experiment-001 \
    --output /path/hybrid__experiment-001.zip
  ```

- Standalone inference/reporting:

  ```bash
  PYTHONPATH=src .venv/bin/python -m graph_tracks.infer \
    --checkpoint <ckpt>/<track>__graph_model.pt --listings <query_listings.json> \
    --output /path/export --build-index        # hybrid adds: --text-cache <npz>
  PYTHONPATH=src .venv/bin/python -m graph_tracks.report \
    --config config/graph_tracks_hybrid.yaml \
    --checkpoint <ckpt>/<track>__graph_model.pt --output /path/reports
  ```

  Exported vector/index IDs are listing `sku_id`s, not GTINs. HNSW cosine
  search does not reproduce the learned pair scorer.

## Colab blockers to resolve before full launches

- Pin and record an immutable repo commit per experiment (local uncommitted
  changes are not transmitted; local bundle producers and remote consumers
  must be compatible at that revision).
- Migrated split contract everywhere (no old 3k/5k sample paths; do not use
  `run_ann_full_data.py` unchanged).
- Diet/freshness gates on cached AND checkout-native bundles.
- Graph dependencies in the Colab profile; verify against Colab
  Python/PyTorch/CUDA.
- Graph/hybrid completion adapters (text-only `predict_items` is not enough)
  and checkpoint-collection contract for graph checkpoints.
- Resume: persist optimizer/scheduler/RNG/sampler state or explicitly reject.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_graph_tracks.py -q
PYTHONPATH=src WANDB_SILENT=true WANDB_CONSOLE=off .venv/bin/python \
  scripts/smoke_graph_tracks.py --output /tmp/new-graph-smoke
```

Smoke uses the real local MiniLM checkpoint, both CPU workers, real offline
W&B, complete inference/reports, and DVC push + independent clean pull.
Synthetic inputs — perfect metrics are NOT quality evidence.

## Build sequence and completion checkpoints

1. **Shared contract + A0:** validation migration, telemetry/diet blockers,
   current checkpoint captured, baseline results published.
2. **A1 + graph census:** attribute scoring tested; shared graph constructed
   and audited. No expensive graph training on unaudited edges.
3. **B baseline:** GNN training/inference + HNSW integration; attribute-only
   (`graph_enabled: false`), one/two-hop and shuffled-edge controls.
4. **C0 baseline:** graph layer over frozen A0 embeddings; fusion evaluated
   against the attribute-aware text baseline.
5. **Controlled extensions** (pretrained GNN transfer, extra relations,
   retrieval fusion, joint fine-tuning) only as separately named experiments.
6. **Selection:** one comparison report — quality, uncertainty, failure
   examples, resource costs, deployment recommendation.

Done = all three tracks have reproducible checkpoints, inference artifacts and
comparable evaluation reports.
