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
- Initial same-filesystem checkpoint resume support.

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
`checkpoint_sha256` and `composition`). IDs are aligned explicitly; unknown
IDs and duplicate IDs fail. No cache generator exists yet.

Validation so far: syntax, both configs, and synthetic GNN/hybrid forward and
backward checks. Full training, tracking publication, resume, and Colab have
not been exercised. This is a full-batch prototype, not the planned
neighbor-sampled GraphSAGE implementation.

Still needed: prepared-input exporters, inference/vector/index export,
comprehensive leakage guards, runtime/memory profiling, Colab dispatch and
completion adapters, and full-data comparison. The standard Colab launcher
does not know these tracks yet. Resume currently requires the prior selected
checkpoint at its recorded path; portable Colab restore needs an adapter.

W&B follows the existing project config and environment. Avoid live credentials
when running local validation. PyTorch resume files must come from trusted runs
because they contain optimizer and RNG objects.
