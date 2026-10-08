# ER documentation

Product matching for retail listings. Given many retailer listings of drinks and
foods, decide which ones are the same real product, then learn a model that scores
candidate pairs.

## Start here

| If you want to… | Read |
|---|---|
| Rebuild the training data | [data-prep.md](data-prep.md) |
| Train or launch on Colab | [training.md](training.md) |
| Run a command | [runbook.md](runbook.md) |
| Operate the Colab lane (sessions, smokes, GPU) | [colab-lane.md](colab-lane.md) |
| Operate the Kaggle lane (bundles, kernels, chain) | [kaggle-lane.md](kaggle-lane.md) |
| Operate the Laya lane (typed decisions) | [laya-lane.md](laya-lane.md) |
| Understand the design | [pipeline.md](pipeline.md) |
| Know what the audits concluded | [audits.md](audits.md) |
| Know what JEV (the LLM) found | [jev.md](jev.md) |

## The one-paragraph version

`dataset.csv` (71,623 raw listings, 13 columns) goes through two views:

1. **Raw view** — extract attributes per GTIN, then decide for every candidate
   pair whether it is `proceed`, `hard_no` or `fallback`. This is the **gate**.
2. **Training view** — the deduped catalog becomes model input. Labels come from
   the gate.

One command rebuilds everything and produces a verified input package. Training
itself runs on a Colab GPU; all data preparation stays local.

## Vocabulary

| Term | Meaning |
|---|---|
| **GTIN** | The barcode number. The closest thing to a product identity. |
| **canonical record** | One row per valid GTIN, with attributes extracted from every listing that carried it. |
| **gate** | The rule that labels a candidate pair `proceed` / `hard_no` / `fallback`. |
| **hard negative** | Textually similar but the gate proved the products differ. The hard class. |
| **fallback** | Evidence is contradictory or missing. Never labeled — goes to review. |
| **component** | A group of GTINs connected by verified-same-product edges. The unit of splitting. |
| **MNRL** | MultipleNegativesRankingLoss — the shipped training loss (in-batch ranking). |
| **A / B / C track** | text / `gnn_only` / `cascade`. See [training.md](training.md). |
| **JEV** | An LLM used as an independent second opinion on gate decisions. |

## Rules of the road

- **Config is the source of truth.** No thresholds, ladders or paths hardcoded at
  call sites. `config/training.yaml` and `config/paths.yaml` own them.
- **Fail loud, never silently.** Every stage asserts its own row-count closure and
  writes a manifest. A drift you did not ask for stops the run.
- **Provenance is pinned.** Preparation hashes all source, config, the raw input and
  the checkpoint. Change any of them and you must re-prepare.
- **Never normalize `row_bc`.** Fold sets are keys of the caller's raw strings;
  zero-padding silently drops pairs. See [data-prep.md](data-prep.md#splits).
- **Train / dev / test discipline.** Train on train, select on dev, report test
  only after selection. One component split shared by all three tracks.

## Repo layout

```
dataset.csv              the only raw input; git-tracked, 54 MB
config/                  all thresholds and paths (training.yaml, paths.yaml, model_tracks.yaml)
src/core/                shared primitives: config, manifests, hashing, gtin, schemas
src/training/            data prep + trainer
src/graph_tracks/        GNN-only training + the cascade combinator
src/model_tracks/        three-track suite: launch, workers, packaging, reports
src/cli/                 the Colab launcher
scripts/                 audit and maintenance scripts
data/                    generated: deduped catalog, gates, labeled pairs, prepared inputs
artifacts/evidence/      measured evidence the pipeline consumes as an input
docs/                    these docs
```

`artifacts/models/` holds the local MiniLM checkpoint and
`artifacts/evidence/*.json` are tracked inputs the pipeline hashes. Neither is
regenerable by the pipeline — do not delete them.
