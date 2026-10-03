# Training Prep — Commands to Generate All CSVs for Full Train + Smoke 200

## One-call full preparation

Run from `ER`:

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all
```

This rebuilds deduplication, SKU mapping, cross-country pairs, the number
reference, canonical records, gates, labeled pairs, final validation and
its fold map, graph-track inputs, and the full prepared text bundle. It
updates the measured gate census before labeling and verifies that the
bundle embeds the current CSVs and the graph manifest names the current
checkpoint. Existing graph inputs and bundles are archived in the run folder.

Smoke files are left untouched and training is not started. Step logs and
the completion manifest are written under `results/training_prep/`. After
the CLI package is installed, the same command is available as `er-prepare`.

Validation, graph setup, and bundle preparation share one base payload per
run. Reuse verifies the dataframe, input CSVs, configuration, source code,
and payload checksum. Changed inputs stop reuse; a new run builds a fresh payload.

To continue after an already completed CSV rebuild:

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all --resume-from validation
```

This checks the current CSV stage manifests before preparing downstream inputs.
The individual steps below remain available for inspection or explicit smoke work.

## 0. Prerequisites

- `data/track_setup/` must exist with CSV files (eligible_catalog.csv, listing_splits.csv, listing_pairs.csv)
- `data/track_setup/text_prepared.pkl.gz` must exist (frozen payload + pair arrays)
- `data/track_setup/setup_manifest.json` must have correct checkpoint sha256
- `data/track_setup/gnn_only.yaml` and `data/track_setup/hybrid.yaml` must exist

## 1. Track Setup (base CSV files)

Regenerates `data/track_setup/` from raw inputs:
- eligible_catalog.csv
- listing_splits.csv
- listing_pairs.csv
- text_prepared.pkl.gz

```bash
python -m graph_tracks.setup --output data/track_setup --text-checkpoint artifacts/models/all-MiniLM-L6-v2
```

If this times out, the CSV files are created first; then run the prepare step separately:

```bash
# After CSV files exist, run prepare step
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
from graph_tracks.prepare import prepare
from pathlib import Path
prepare(Path('data/track_setup/eligible_catalog.csv'), Path('data/track_setup/listing_splits.csv'), Path('data/track_setup/listing_pairs.csv'), Path('data/track_setup/prepared'))
"
```

## 2. Setup Manifest Checkpoint Hash

The setup_manifest.json needs the correct checkpoint sha256:

```bash
.venv/bin/python -c "
import sys, json
sys.path.insert(0, 'src')
from graph_tracks.text_cache import checkpoint_hash
from pathlib import Path
chk = Path('artifacts/models/all-MiniLM-L6-v2')
actual = checkpoint_hash(chk)
m = json.load(open('data/track_setup/setup_manifest.json'))
m['text_checkpoint_sha256'] = actual
json.dump(m, open('data/track_setup/setup_manifest.json', 'w'), indent=2)
print(f'updated: {actual}')
"
```

## 3. Track Config YAMLs

Create gnn_only.yaml and hybrid.yaml in data/track_setup/:

```bash
# gnn_only.yaml
python -c "
import yaml
from pathlib import Path
setup = Path('data/track_setup').resolve()
cfg = yaml.safe_load(open('config/graph_tracks_gnn.yaml'))
cfg['listings'] = str(setup / 'prepared/listings.json')
cfg['pairs'] = str(setup / 'prepared/pairs.csv')
cfg['input_manifest'] = str(setup / 'prepared/input_manifest.json')
cfg['device'] = 'cpu'
cfg['report_test'] = False
cfg['postprocess'] = False
cfg['build_index'] = False
yaml.dump(cfg, open(setup / 'gnn_only.yaml', 'w'), sort_keys=False)
"

# hybrid.yaml
python -c "
import yaml
from pathlib import Path
setup = Path('data/track_setup').resolve()
cfg = yaml.safe_load(open('config/graph_tracks_hybrid.yaml'))
cfg['listings'] = str(setup / 'prepared/listings.json')
cfg['pairs'] = str(setup / 'prepared/pairs.csv')
cfg['input_manifest'] = str(setup / 'prepared/input_manifest.json')
cfg['text_cache'] = str(setup / 'shared_minilm__embeddings.npz')
cfg['device'] = 'cpu'
cfg['report_test'] = False
cfg['postprocess'] = False
cfg['build_index'] = False
yaml.dump(cfg, open(setup / 'hybrid.yaml', 'w'), sort_keys=False)
"
```

## 4. Smoke 200 Sampled Setup

Creates `data/prepared/smoke_200/` with 200 sampled listings, retaining augmentation lineage and parent splits:

```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from model_tracks.smoke_inputs import prepare_smoke
result = prepare_smoke(Path('data/track_setup'), Path('data/prepared/smoke_200'), sample=200)
print(f'Done: {result}')
"
```

This generates:
- `data/prepared/smoke_200/eligible_catalog.csv` (~527 listings)
- `data/prepared/smoke_200/listing_splits.csv` (200 rows)
- `data/prepared/smoke_200/listing_pairs.csv` (92 pairs)
- `data/prepared/smoke_200/text_prepared.pkl.gz` (sampled bundle)
- `data/prepared/smoke_200/suite.yaml` (device: cpu, epochs: 1)

## 5. Training Bundles (for full train lane)

Builds prepared bundles for the train lane:

```bash
.venv/bin/python -m training.train --dataset data/dataset_deduped.csv --payload full --prepare-bundle data/prepared/full/worker_1_baseline.pkl.gz --no-mask-effect --no-plot
```

## 6. Final Validation (already exists)

`data/final_validation.csv` — 10,359 rows (10,206 + 153 unique slice-review pairs).
Regenerate if needed:

```bash
python -m training.build_final_validation
```

## 7. Run CPU Smoke Test on Colab

```bash
python -m src.cli.colab --what tracks --tracks-config data/prepared/smoke_200/suite.yaml --gpu CPU --keep-alive
```

## 8. Run Full Train on Colab (CPU)

```bash
python -m src.cli.colab --what train --gpu CPU --workers 1 --keep-alive
```

## File Inventory

| File | Source | Used by |
|------|--------|---------|
| data/track_setup/eligible_catalog.csv | graph_tracks.setup | prepare_smoke |
| data/track_setup/listing_splits.csv | graph_tracks.setup | prepare_smoke |
| data/track_setup/listing_pairs.csv | graph_tracks.setup | prepare_smoke |
| data/track_setup/text_prepared.pkl.gz | training.train --prepare-bundle | prepare_smoke |
| data/track_setup/setup_manifest.json | manual (checkpoint hash) | prepare_smoke |
| data/track_setup/gnn_only.yaml | manual (from config) | prepare_smoke |
| data/track_setup/hybrid.yaml | manual (from config) | prepare_smoke |
| data/prepared/smoke_200/* | prepare_smoke | Colab smoke test |
| data/prepared/full/worker_1_baseline.pkl.gz | training.train --prepare-bundle | Colab train |
| data/final_validation.csv | build_final_validation | eval |

GNN and hybrid use AdamW weight decay (`0.0001`), gradient clipping (`1.0`),
and dev PR-AUC for checkpoint selection. Early stopping requires an improvement
greater than `0.001` and stops after three unsuccessful epoch evaluations.
The plateau scheduler halves the learning rate after two unsuccessful evaluations,
with a floor of `0.00001`. Scheduler and stopping state are checkpointed for resume;
completion reports record the actual number of epochs.

### Separate hybrid embedding job

After full preparation, run `notebooks/prepare_hybrid_embeddings.ipynb` on a
GPU runtime. Upload the prepared `colab_embeddings_inputs.zip` when prompted;
the notebook extracts the checkout, prepared inputs, and frozen checkpoint and
installs dependencies. The standalone command from ER is:

```bash
PYTHONPATH=src python -m training.prepare_embeddings --device cuda
```

This creates `data/track_setup/shared_minilm__embeddings.npz` for hybrid only.
Return that file to the prepared setup directory before running the three-track
preflight. Existing caches are reused only after input/checkpoint/composition
validation; stale caches fail rather than silently overwrite. Generation writes
a temporary file and publishes it only after validation. No training is launched.

The automated GPU launcher uses Git for code, the frozen MiniLM checkpoint,
and prepared embedding inputs, with no input ZIP upload:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_colab_embeddings.py
```

It runs the embedding worker on T4 at batch size 256, verifies the downloaded
cache checksum, and releases the runtime. Run it only when starting a new job.
