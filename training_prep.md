# Training Prep — Commands to Generate All CSVs for Full Train + Smoke 200

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
