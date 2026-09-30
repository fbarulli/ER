# Experimental graph tracks

Initial implementation: GNN-only and frozen-text hybrid models in new files.
The existing ANN/Colab code is unchanged.

Implemented:

- Strict listing/pair contracts and train-only attribute vocabularies.
- Typed listing–attribute–listing message passing in plain PyTorch.
- Training-only neighborhood context; no query-to-query message passing.
- GNN metric-learning loss and calibrated pair scores; hybrid adds text cosine.
- Validated YAML configurations, existing MLflow/W&B contexts, local training
  logs, epoch metrics, shared worker heartbeats, and checkpoint manifests.
- Prepared input export through shared `row_identity` (no second parser).
- Local-checkpoint frozen MiniLM cache using shared model-input composition.
- Checkpoint-bound batched inference, pair scoring, and vector/HNSW export.
- Listing support/supervision accounting and per-parameter gradient telemetry.
- Hash-validated checkpoint resume, including moved checkpoint trees.

Entry point (requires prepared inputs and existing tracking dependencies):

```bash
PYTHONPATH=src python -m graph_tracks.train \
  --config config/graph_tracks_gnn.yaml --run-tag gnn-baseline
```

For hybrid, use `config/graph_tracks_hybrid.yaml`. Paths are resolved against
the shared project root. `EUROMONITOR_RESULTS_DIR` overrides the output directory;
each independent experiment needs a fresh output directory.

Listing JSON:

```json
{
  "schema": "er-graph-listings-v1",
  "listings": [
    {
      "product_id": "listing-1",
      "split": "train",
      "attributes": {"brand": ["example"], "flavor": ["lemon"]},
      "numeric": {"volume_ml": [330], "pack": [1]}
    }
  ]
}
```

Use existing component-safe assignments for `train`, `dev`, and `test`.
The input omits barcode and text fields. A pair CSV has exactly
`product_id1,product_id2,label,split`; both endpoints must belong to that split.
Train and dev each require positive and negative pairs. Test labels are
validated but are not used for model selection or automatically scored.

Hybrid text caches are NPZ files with `ids` (strings), `embeddings` (finite,
nonzero vectors), and `metadata` (a scalar JSON string containing
`checkpoint_sha256` and `composition`). IDs are aligned explicitly; missing
requested IDs and duplicate cached IDs fail. A full-catalog cache can serve a
subset of listings. Inference checks checkpoint and composition compatibility.

## Prepare and export

These standalone entry points do not alter ANN or Colab dispatch. Run from ER
with `PYTHONPATH=src`. Split input must contain exactly `product_id,split`, cover
the retained catalog exactly, and come from the shared component-safe process.
The exporter checks pair endpoint splits but cannot independently prove that
all unlabeled identities were assigned safely; use the shared split audit.

```bash
python -m graph_tracks.prepare --catalog /path/catalog.csv \
  --splits /path/listing_splits.csv --pairs /path/pairs.csv \
  --output /path/new_prepared_dir
python -m graph_tracks.text_cache --catalog /path/catalog.csv \
  --checkpoint artifacts/models/all-MiniLM-L6-v2 --output /path/text.npz
python -m graph_tracks.infer --checkpoint /path/checkpoint-N/graph_model.pt \
  --listings /path/listings.json --output /path/new_export_dir --build-index
```

Hybrid inference additionally requires `--text-cache /path/text.npz`. Optional
`--pairs` takes exactly `product_id1,product_id2`, without labels, and exports
scores. No quality metrics are automatically computed on test. Inference batches
may contain only dev/test listings; graph support comes from the checkpoint.

Output vectors are normalized graph-informed embeddings. HNSW retrieves by
vector similarity; hybrid's learned pair score also includes direct text cosine,
so the ANN similarity is not the final pair score. HNSW currently uses explicit
prototype settings (`M=16`, `ef_construction=200`, `ef_search=100`); matched-budget
comparisons and configuration integration remain future work.

The model currently uses eight categorical relations and volume/pack numeric
features. The prepared manifest explicitly identifies this subset. It does not
yet consume every dimension in the expanding shared identity registry.

## Verified status

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_graph_tracks.py -q`:
**6 passed**. Checks cover forbidden barcode inputs, split boundary guards,
train-only vocabulary, unknown-value isolation, query batch invariance, cache
alignment/zero vectors, shared-identity preparation, CPU training for both tracks,
pair scoring, normalized vectors, actual HNSW index building, copied-tree resume,
and changed-seed resume rejection. All test outputs are isolated temporary files.

A separate two-listing smoke used the actual local
`artifacts/models/all-MiniLM-L6-v2` checkpoint and produced a validated 384-D
frozen cache through the cleaned model-input composition. This checks runtime
integration, not matching quality.

MLflow is absent in the current venv. Lifecycle tests set
`MLFLOW_TRACKING_URI=off` and remove `WANDB_API_KEY`; live tracking publication is
unverified. Local logs, heartbeats, manifests, usage and gradient telemetry are
verified. Do not infer CUDA/full-catalog readiness from tiny CPU smokes. This is
a full-batch typed two-hop prototype, not neighbor-sampled GraphSAGE. CUDA's
scatter reductions also require a deterministic-runtime compatibility check.

Still needed: all-dimension graph feature integration after identity SSOT lands,
neighbor sampling and runtime/memory profiling, attribute-only/shuffled controls,
retrieval-quality evaluation, Colab dispatch/completion adapters, and a full-data
comparison. The standard Colab launcher does not know these tracks yet. Copied
tree resume needs the prior best epoch alongside the resumed epoch; collecting
only the latest checkpoint is insufficient. No Colab run was launched.

W&B follows the existing project config and environment. Avoid live credentials
when running local validation. PyTorch resume files must come from trusted runs
because they contain optimizer and RNG objects.
