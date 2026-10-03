# Standalone graph training tracks

Two runnable experimental workers live here. Existing ANN, text training and
Colab launcher code stays unchanged. MLflow is neither imported nor required.

| Track | Inputs | Learned outputs |
|---|---|---|
| `gnn_only` | Structured attributes and training-only graph context | Graph embeddings and calibrated pair scorer |
| `hybrid` | Same graph plus frozen MiniLM vectors | Graph-informed embeddings and graph/text pair scorer |

This is full-batch, relation-specific listing → attribute → listing aggregation
in plain PyTorch. It is not neighbor-sampled GraphSAGE. The initial feature
schema contains eight categorical relations plus volume/pack; preparation
explicitly records this subset, rather than claiming all identity dimensions.

## Runtime

Run from ER with `PYTHONPATH=src`. Install PyTorch for your CPU/CUDA runtime,
then the separate profile:

```bash
.venv/bin/python -m pip install -r requirements/graph_tracks.txt
```

Verified locally with Python 3.14.6, torch 2.14.0, sentence-transformers 6.0.1,
hnswlib 0.8.0, W&B 0.30.0 and DVC 3.67.1. GPU/Colab runtime compatibility is
unverified; no silent CPU fallback occurs when CUDA is requested.

## Shared preparation

### Real ER catalog setup (no training)

The setup command builds listing assignments from the same
`training.folds.derive_holdout` entry point used by text training. It records
the exact local text checkpoint, source hashes, exclusions and graph census.
It creates independent GNN/hybrid configs with test reporting disabled.

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.setup --output data/track_setup
PYTHONPATH=src .venv/bin/python -m graph_tracks.text_cache \
  --catalog data/track_setup/eligible_catalog.csv \
  --checkpoint artifacts/models/all-MiniLM-L6-v2 \
  --output data/track_setup/shared_minilm__embeddings.npz
PYTHONPATH=src .venv/bin/python -m graph_tracks.preflight --config data/track_setup/gnn_only.yaml
PYTHONPATH=src .venv/bin/python -m graph_tracks.preflight --config data/track_setup/hybrid.yaml
```

Entity labels select the lexically first listing per normalized entity.
Additional listings sharing a checksum-valid, unheld gtin form positive chains. Only labeled same-split
negatives are retained; cross-split negatives and labels without listing
endpoints are counted in `setup_manifest.json`. These pair semantics must be
used by a future text comparison too; the existing entity-level text report is
not automatically comparable. Unassigned/empty-gtin listings are excluded
and counted rather than assigned an invented split. The recorded local text
checkpoint is a baseline reference; this does not establish its training history.

Portable worker packages contain prepared inputs, optional frozen text cache,
worker config, graph source overlay and a SHA256 inventory. They contain no
credentials and do not start a worker or provision a VM:

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.worker_package \
  --config data/track_setup/gnn_only.yaml --output results/gnn_worker_setup.zip
PYTHONPATH=src .venv/bin/python -m graph_tracks.worker_package \
  --config data/track_setup/hybrid.yaml --output results/hybrid_worker_setup.zip
```

Extract into the recorded ER checkout revision and follow the package README.
The target runtime must pass its own preflight, including CUDA availability,
before training. These packages support manual workers. The consolidated
three-track Colab route uses the existing entry point:

```bash
PYTHONPATH=src .venv/bin/python -m cli.colab --what tracks \
  --tracks-config config/model_tracks.yaml
```

Set `profiling: true` in the suite YAML for isolated worker traces and operator
summaries, included in the downloaded result archive.

Use the eligible canonical catalog after the shared identity corrections and
exclusions. Preparation rejects quarantined GTINs and scoped held listings. It does not correct a raw
catalog or invent identities/splits. The listing split CSV must contain exactly
`sku_id,split`, cover the retained catalog exactly, and come from the shared
component-safe process. Pair CSV must contain exactly
`sku_id1,sku_id2,label,split`; both endpoints must belong to that split.
Train and dev require both positive and negative pairs.

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.prepare \
  --catalog /path/eligible_catalog.csv --splits /path/listing_splits.csv \
  --pairs /path/pairs.csv --output data/graph_tracks
PYTHONPATH=src .venv/bin/python -m graph_tracks.text_cache \
  --catalog /path/eligible_catalog.csv \
  --checkpoint artifacts/models/all-MiniLM-L6-v2 \
  --output data/graph_tracks/shared_minilm__embeddings.npz
```

The exporter uses `core.sku_identity.row_identity`; there is no second
identity parser. Barcodes and verified-match edges are excluded from graph
features. Catalog/split/pair/listing, identity-policy and dimension-policy hashes
are recorded. Workers require the prepared manifest and reject stale inputs or
policy changes. `allow_unmanifested_inputs: true` is a synthetic-smoke escape
hatch, disabled in shipped configs.

Frozen caches are NPZ with string `ids`, finite nonzero `embeddings` and JSON
`metadata`. They identify the local checkpoint, shared model-input composition,
catalog and identity policies. IDs are aligned explicitly. Hybrid training
checks cache/preparation provenance, not just vector dimensions. A cache may
contain extra IDs, but production training requires the same source catalog
fingerprint. No checkpoint is downloaded implicitly.

Input contracts validate declared splits; they cannot prove that unknown or
unlabeled identities were split safely. The shared identity/split audit remains
necessary before any quality comparison.

## Train and complete

Production CUDA workers consume locally prepared graph tensors. New shared
preparation writes `graph_plan.json` and `graph_inputs.npz` beside listings;
for existing prepared catalogs regenerate them locally before packaging:

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.prepared_inputs \
  --listings data/track_setup/prepared/listings.json \
  --pairs data/track_setup/prepared/pairs.csv
```

The plan binds listing ID order, train-only vocabulary, numeric float32 values,
int64 edges/pairs and pooling topology to source hashes. CUDA training and
inference fail if this local preparation is missing. CPU synthetic smoke may
use record tensorization. Numeric features intentionally retain only min/max
and presence; all missing/unseen categorical values use token zero without
sharing graph context. Census records these representation reductions.

`graph_tracks.infer.forward_outputs` saves normalized float32 listing vectors,
dev/test pair scores and checkpoint/source hashes without HNSW or plots.
`graph_tracks.report.complete(..., saved_inference=...)` validates those saved
outputs and performs local metrics, indexing and plots without a model forward.
New inductive catalogs can use `prepared_inputs.prepare_inference` locally with
the selected checkpoint; their queries retain the checkpoint's training support.

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.train \
  --config config/graph_tracks_gnn.yaml --run-tag experiment-001
PYTHONPATH=src .venv/bin/python -m graph_tracks.train \
  --config config/graph_tracks_hybrid.yaml --run-tag experiment-001
```

Each worker creates `<output_dir>/<track>__<run_tag>`. If
`EUROMONITOR_RESULTS_DIR` is set it supplies the base output directory. The same
run tag can be used for both tracks safely. Checkpoint **filenames**, manifests,
logs, vectors, reports and W&B artifacts carry the track name. The shared HNSW
adapter's internal filenames stay inside a track-prefixed index directory.

For example:

```text
results/graph_tracks/gnn_only__experiment-001/
  gnn_only__run_manifest.json
  gnn_only__training.log
  gnn_only__inputs/gnn_only__listings.json
  _checkpoints/gnn_only/experiment-001_f0/checkpoint-1/
    gnn_only__graph_model.pt
    gnn_only__trainer_state.json
    gnn_only__checkpoint_manifest.json
  gnn_only__completion-epoch-10/
    gnn_only__inference/gnn_only__vectors.npz
    gnn_only__reports/gnn_only__model_evaluation_summary.csv
    gnn_only__training_report.md
  gnn_only__dvc-epoch-10/
```

Every epoch writes loss/dev metrics, listing supervision/support accounting,
parameter gradient norms, worker heartbeats and integrity-marked checkpoints.
The run manifest records configuration, source revision, implementation hashes
and resume lineage. Prepared input copies are preserved by default.

After training, the selected checkpoint automatically produces:

- Normalized vectors and, by default, HNSW indexes bound to its provenance.
- Scored dev/test pairs and a threshold fitted **only on dev** using Youden J.
- ROC-AUC, PR-AUC, P@R95, accuracy, precision, recall, F1 and confusion counts.
- Explicitly pooled pair-ranking statistics through the shared ER metric code.
- Attribute-availability slice metrics, PR/score-distribution plot and Markdown report.
- Actual within-split ANN retrieval reports against direct known-positive pairs.

Test labels do not select checkpoints or thresholds. `report_test: false`
suppresses test scoring when test should stay sealed. Train endpoints are
excluded from held-out pair and retrieval metrics. Full-catalog operational
vectors still include training listings; they are not themselves an evaluation.

Retrieval catalogs contain same-split listings and exclude the query itself.
Unknown pairs are **not** treated as negatives. Reports identify incomplete
truth and budgets that cover the entire catalog. These retrieval statistics
must not be confused with full production-catalog recall. Scores are model-only;
the shared identity conflict policy is not applied automatically.

## W&B and DVC

Shipped configs use W&B project `e-r`, mode `offline`. The real SDK records
config, epochs, final metrics and track-specific artifacts without credentials.
Use `mode: online` with `WANDB_API_KEY` for live tracking, or `disabled` for tests.
Credentials are never configuration values or logged artifacts. MLflow has no
role in these workers.

DVC is enabled locally by default. Each completed epoch generation gets an
isolated `--no-scm` DVC workspace, a result snapshot including **all checkpoints,
prepared inputs, vectors, indexes and reports**, and a SHA256 inventory. A clean
restore is verified before publication succeeds. Repository Git/DVC state is
not modified. W&B's SDK cache/log directory is managed separately.

Remote publication is opt-in:

```yaml
dvc:
  enabled: true
  remote: /absolute/path/to/local-or-mounted-store
  push: true
```

HTTP/DagsHub remotes are also supported. DagsHub uses `DVC_API_KEY`; generic
HTTP basic authentication uses `DVC_HTTP_USER` / `DVC_HTTP_PASSWORD`. Credentials
stay in permission-restricted `.dvc/config.local`, outside snapshots/artifacts.
Credential-bearing remote URLs are rejected. Other backends need their DVC
extras and standard provider configuration. Live remote publication is unverified.

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.dvc \
  --project /path/hybrid__run/hybrid__dvc-epoch-10 --output /path/restored-run
```

Local-only restores need the original local DVC cache. A remote push permits an
independent clean pull using the pointer/config, manifest and current credentials.

## Resume and portable results

Resume restores model, scorer, optimizer and RNG state. Hashes, track and model
configuration must match; extending epochs and relocating input/output paths are
allowed. A moved checkpoint tree must retain the previously selected best epoch
alongside the resumed epoch. Trusted checkpoint files only: optimizer/RNG state
uses Python pickle. Legacy unprefixed experimental files require fresh training.

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.train \
  --config /path/updated-hybrid-config.yaml --run-tag experiment-001-resumed \
  --resume /path/restored-run/_checkpoints/hybrid/experiment-001_f0/checkpoint-10/hybrid__graph_model.pt
PYTHONPATH=src .venv/bin/python -m graph_tracks.bundle \
  --source results/graph_tracks/hybrid__experiment-001 \
  --output /path/hybrid__experiment-001.zip
```

For relocated runs, point `listings`, `pairs`, `input_manifest`, and `text_cache`
at their copied `hybrid__inputs/` files; increase `epochs`. The bundle retains
raw artifacts and DVC pointers but excludes credentials, SDK caches and duplicated
DVC payloads. `--include-dvc-cache` additionally makes a local-only DVC workspace
portable. Online W&B runs persist remotely; offline W&B logs remain in the worker
output and are excluded from the portable ZIP.

## Standalone inference and reporting

```bash
PYTHONPATH=src .venv/bin/python -m graph_tracks.infer \
  --checkpoint /path/checkpoint-N/gnn_only__graph_model.pt \
  --listings /path/query_listings.json --output /path/new-export --build-index
PYTHONPATH=src .venv/bin/python -m graph_tracks.report \
  --config config/graph_tracks_hybrid.yaml \
  --checkpoint /path/checkpoint-N/hybrid__graph_model.pt --output /path/new-reports
```

Hybrid inference additionally needs `--text-cache`. Optional `--pairs` contains
exactly `sku_id1,sku_id2`, without labels. New queries may use
`split: inference`, rejected by training. Query batches cannot communicate;
their graph context is frozen from training support stored in the checkpoint.

ANN cosine search does not reproduce the learned pair scorer, especially the
hybrid's direct text path. Exported vector/index IDs are listing `sku_id`s,
not GTINs. HNSW settings and candidate budgets are configurable. Index metadata
contains source paths; after relocation regenerate indexes through inference
instead of treating old absolute paths as valid.

## Colab/manual GPU worker

The existing Colab launcher still dispatches text workers. For these tracks,
prepare locally, transfer the shared prepared directory/cache, check out an
immutable commit containing this code, install the graph dependency profile, and
run the same standalone worker command. Set `device: cuda` in a separate config,
verify the tiny smoke in that runtime first, and collect the track's complete
result ZIP before teardown. Use W&B online or an explicit remote DVC push for
persistence independent of the VM. No provisioning or Colab run was performed.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_graph_tracks.py -q
PYTHONPATH=src WANDB_SILENT=true WANDB_CONSOLE=off .venv/bin/python \
  scripts/smoke_graph_tracks.py --output /tmp/new-graph-smoke
```

The standalone smoke uses the actual local MiniLM checkpoint, shared preparation,
both CPU workers, real W&B offline runs, complete inference/reports, and DVC
push/independent clean pull to a temporary local remote. Its listings and labels
are synthetic; perfect metrics there are not quality evidence.

Focused guards cover split/label leakage, forbidden gtin features, train-only
vocabularies, unknown-value isolation, query batching, cache alignment, stale
manifests, checkpoint track/hash identity, moved-tree and same-run resume,
portable input/bundle restore, secret exclusion, and real W&B/DVC lifecycle.
Live online tracking, live remote DVC, GPU/Colab, full-catalog memory/quality and
scientific ablations remain unverified. Attribute-only control is available via
`graph_enabled: false`; neighbor sampling and shuffled-edge controls are further
architecture experiments, not requirements for running these initial workers.
