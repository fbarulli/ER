# Kaggle lane

Transport + remote-compute lane. The Colab lane is untouched.
Contract + evidence: [kaggle-lane-completion.md](kaggle-lane-completion.md).

## Ruling (2026-10-06)

**Bundle generation is CPU-only and runs on a Kaggle CPU session — not
locally.** The GPU session only trains.

| Stage | Where |
|---|---|
| Package + upload the raw cohort export | local (this lane) |
| Bundle generation: `prepare_all` → prepared inputs | Kaggle CPU kernel |
| Training + embedding forwards (text / gnn_only / hybrid) | Kaggle GPU kernel (tracked, later run) |

## Commands

`er-kaggle` (= `PYTHONPATH=src .venv/bin/python -m cli.kaggle_lane`).
Dry-run by default; `--execute` touches the network.

```bash
# 0. one-time: write ~/.kaggle/kaggle.json from the environment
er-kaggle --what credentials --execute

# 1. stage + push the CPU bundle kernel (clones the pinned revision,
#    runs prepare_all, stages the package + receipt)
er-kaggle --what bundle-kernel --execute

# 2. poll
er-kaggle --what kernel-status             # cpu (default)
er-kaggle --what kernel-status --kernel gpu

# 3. fetch the bundle back, sha-verified against the kernel's receipt
er-kaggle --what bundle-fetch --execute

# 4. dataset transport + submission (unchanged)
er-kaggle --what package --dataset-csv dataset_50pct.csv
er-kaggle --what upload --execute
er-kaggle --what download --execute
er-kaggle --what submission --submission-input in.csv --submission-output out.csv
```

## Config (SSOT: `config/training.yaml` → `kaggle:`)

`username`, `api_key_env`, `repository`, `branch`, `cpu_kernel_slug`,
`gpu_kernel_slug`, `checkout_paths`, `bundle_requirements` — plus the
transport keys (`slug`, `export_csvs`, `staging_dir`,
`submission_id_columns`).

## Files

| file | role |
|---|---|
| `src/cli/kaggle_lane.py` | the lane (transport + kernels) |
| `kaggle_backend.py` | repo-root shim |
| `src/core/schemas.py` | `KaggleSpec` (additive) |
| `tests/test_kaggle_lane.py` | offline pins |
