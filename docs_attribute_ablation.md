# Frozen-checkpoint attribute ablations

Train each track once. This workflow performs inference on frozen checkpoints;
it does not schedule another training run per attribute. Preparation/reporting
run locally; the encoding job uses Colab CUDA and the existing Git clone flow.

Settings are in `config/attribute_ablation.yaml`. The initial sample is 100
held-out dev pairs, stratified across label and supplied difficulty/masking/
generation columns. Missing axes are reported, never invented. An explicit test
split is supported for final reporting; no threshold is fitted during ablation.

Prepare a selected text checkpoint locally:

```bash
PYTHONPATH=src .venv/bin/python -m model_tracks.ablation prepare \
  --catalog data/track_setup/eligible_catalog.csv \
  --pairs data/track_setup/prepared/pairs.csv \
  --checkpoint artifacts/models/tracks/RUN/text \
  --track text
```

For `gnn_only`, add `--listings data/track_setup/prepared/listings.json` and use
its selected graph checkpoint. For `hybrid`, also supply `--text-checkpoint`
matching the frozen text checkpoint in its graph training manifest. Naming and
schema checks reject legacy pair IDs; rebuild prepared inputs using the existing
preparation workflow when needed.

The command prints the request path. Run its GPU evaluation with the existing
saved baseline decision threshold and name its source:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_colab_ablation.py \
  --request results/attribute_ablation/REQUEST_HASH/request.json \
  --threshold SAVED_THRESHOLD --threshold-source SAVED_REPORT
```

Inputs and outputs use `.tar.gz` through the existing Git artifact publisher.
The runtime clones the repository and loads the packaged shared runtime overlay;
there are no direct input uploads. Text checkpoint directories must already be
available in the clone. Existing unrelated staged changes cause the existing
publisher to refuse publication; use the established isolated publication branch
flow. The launcher releases the runtime after encoding, validates downloads
locally and saves the report/result/request and sanitized logs together.

All composed text is deduplicated globally and encoded in one batch sequence per
checkpoint. Every attribute/channel with unchanged inputs reuses its baseline
output. GNN loads once, keeps its frozen training context and vocabulary, and
removes inference endpoint relation/numeric features. Hybrid reports text-only,
graph-only and both interventions, including its actual learned pair scorer.

Text intervention removes the named **declared attribute entry**, then runs the
shared composer and identity extractor. Title/brand evidence may still imply the
attribute. This scope is recorded; unsupported graph fields are explicit no-ops.
This is a measurement of inference influence, not the effect of retraining
without an attribute and not a universal scalar attribute weight.

The local report keeps pair score deltas, decision flips at the frozen threshold,
endpoint embedding changes, both directional ranks and known-positive recall
changes. Retrieval uses the same sampled endpoint catalog for baseline and all
variants; it is not full-catalog ANN recall. Gate/JEV fields and current shared
attribute engine evidence remain distinct. The attribute dashboard filters these
rows by registry attribute and rejects stale provenance.

A downloaded result can be reported locally without allocating Colab again:

```bash
PYTHONPATH=src .venv/bin/python -m model_tracks.ablation report \
  --request REQUEST_JSON --result RESULT_NPZ \
  --threshold SAVED_THRESHOLD --threshold-source SAVED_REPORT
```

Preparation also writes `prepared_inputs.npz`: checkpoint-native token IDs,
attention masks, padding/prompt metadata, graph numeric/relation tensors and
pooling topology, training-support topology, pair indices and the deduplicated
execution plan. The same shared text preparation code serves ordinary embedding
smokes and the hybrid text encoder. Each uses its own checkpoint's tokenizer;
Colab checks the serialized tokenizer fingerprint and consumes the prepared IDs
without tokenizing again. Complete inputs are tokenized with truncation disabled.
Any sequence exceeding the checkpoint's supported positional window fails local
preparation, and embedding-dimension truncation is also rejected. Reports bind
the prepared archive checksum. Encoding logs explicitly record `truncated=0`.

Training completion now enables `post_training_ablation` through
`config/model_tracks.yaml`. After publishing selected inference checkpoints,
it automatically prepares and runs one frozen-checkpoint job per track. Resume
validates completed results and never retrains. The calibration attestation
preserves the original saved dev report and the verified deployment lineage;
a numeric threshold without an attested matching track and checkpoint is
rejected before GPU allocation.

`retrieval_catalog: full` prepares the complete candidate catalog once. Retrieval
ablates query endpoints while keeping candidate vectors fixed, and reports both
exact directional ranks and existing HNSW hits. This avoids a catalog-squared
matrix and repeated candidate encoding. It measures query-side influence;
rebuilding the entire candidate index under each intervention is a different
experiment. Missing historical slice/masking/generation fields remain unknown.
The earlier sampled-only description applies when explicitly configured with
`retrieval_catalog: sampled`.

Shared model loaders now guard bi-encoder evaluation and cross-encoder reranking.
Complete pairs that exceed the cross-encoder window are rejected before predict.
Fine-tuned ANN refresh enables the same bi-encoder guard even for a live model
passed directly to the refresh function.
