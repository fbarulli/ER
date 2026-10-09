# dataclass.md — Dataset class & hosted-dataset surface (requirements)

Status: requirements capture (owner discussion 2026-10-08/09). This is the spec to
build against; it is **not** an implementation.

## 1. Goal

One owner of the project's data surface, in **classes**, **SSOT**, with **zero
duplicated literals**, covering **both**:

- the **local project dataset** (files/artifacts on disk), and
- the **hosted datasets** (Kaggle datasets the lanes mount / publish).

Data is **naked**: no freshness/validity/hash/existence gates anywhere. The only
integrity surface is an **on-demand content digest** — never a verdict.

## 2. What already exists (confirmed)

`src/core/dataset.py::Dataset` (commit `f893037`), built from
`config/dataset.yaml` (`DatasetSpec` / `DatasetMemberSpec`), owns the **local**
data surface:

- `source` — the raw export (`{via: files, key: dataset}`).
- `members` (line-item, the digest surface via `dataset_csv_read`):
  `dataset_deduped`, `sku_to_rep`, `canonical_records`, `gate_results`,
  `labeled_pairs`, `final_validation`, `number_tokens_reference`.
- `layout` (tree roots, addressing only): `prepared`, `track_setup`.
- API: `member()`, `paths()`, `load()`, `load_source()`, `load_deduped()`,
  `load_canonical_records()`, `identity()`, `as_bundle(path, role)`.
- Frozen, `lru_cache`'d, resolves through `core.common.F` / `artifact`
  (no address duplication), delegates sealing to `core.bundle.Bundle`.
- **No validity gate** (naked data); missing member = absence in the digest,
  never a raise.

**Division of labor with `Bundle`:** `Dataset` owns the data and never seals;
`Bundle` (`src/core/bundle.py`, with `BundleRole` / `BundlePipeline`) is the
sealed transport; `Dataset.as_bundle()` is the hand-off seam and integrity is
checked exactly once at `Bundle.load`'s boundary.

Verdict: **adequate for the local surface.** It is a single `Dataset`, not a
registry, and has **no** notion of hosted/remote datasets.

## 3. Gaps to close

1. **No hosted/remote dataset surface.** Kaggle dataset slugs live scattered in
   `core/laya_config.py` (`LayaSpec`) and `core/schemas.py` (`KaggleSpec`):
   - `LayaSpec`: `base_model_dataset`, `dataset_slug`, `export_dataset_slug`,
     `finetune_dataset_slug`, `finetune_ckpt_dataset`, `holdout_dataset_slug`
   - `KaggleSpec`: its bundle/embedding dataset slugs (now referenced from the
     registry via `KaggleSpec.hosted_slug(role)`)
   A fresh SSOT split; no single home.
2. **No role model.** base / corpus / requests / decisions / holdout / ckpt /
   bundle / embeddings are not expressed.
3. **No direction.** input (mount-in) vs output (publish/push).
4. **No attach model.** which kernel mounts/publishes which dataset.
5. **No mount/local path.** `/kaggle/input/<slug>` and the staged local dir are
   re-spelled at each call site.
6. **No split notion.** train/dev/test, holdout, ckpt are not modeled.
7. **`er-laya-holdout` and `er-laya-finetune-ckpt` do not exist on Kaggle yet.**

## 4. Requirements

### 4.1 Local surface (keep)
Keep `Dataset` as the local project-data owner (section 2). Do not bloat it with
Kaggle transport.

### 4.2 Hosted surface (add) — a sibling, class-based surface
Add a **hosted-dataset registry** (sibling to `Dataset`, **not** mixed into it),
class-based, SRP:

- one class declaring the hosted datasets; each entry carries:
  - **identity** — `slug` / owner + human `name`
  - **role** — reuse `BundleRole` where it fits (base, corpus, requests,
    decisions, holdout, ckpt, bundle, embeddings); no second role enum
  - **direction** — `input` (mount) | `output` (publish)
  - **members** — which files/splits (e.g. `train.jsonl`, `dev.jsonl`,
    `test.jsonl`)
  - **attach** — which kernel kind(s) mount/publish it
  - **mount path** — the `/kaggle/input/<slug>` and the staged local dir, one source
- consumers ask the class; **no call site re-spells a slug**.
- `LayaSpec` / `KaggleSpec` / `ColabSpec` **reference** this class (no second slug).

### 4.3 Datasets to declare (SSOT)

| slug | role | direction | attach |
|---|---|---|---|
| `fbarulli/er-laya-base` | base | input | finetune |
| `fbarulli/er-laya-train` | corpus (train/dev/test) | input | finetune |
| `fbarulli/er-laya-requests` | requests | input | decision |
| `fbarulli/er-laya-decisions` | decisions | output | decision |
| `fbarulli/er-laya-holdout` | holdout | input | holdout-eval |
| `fbarulli/er-laya-finetune-ckpt` | ckpt | output→input | finetune → finetune-eval/holdout-eval |
| `fbarulli/er-10k-bundle` | bundle | input | bundle lanes |
| `fbarulli/er-embed-requests` | embeddings | input | embed |

(Mark which exist on Kaggle today; `er-laya-holdout` and `-finetune-ckpt` do not.)

## 5. Design constraints (standing directives)

- **Classes + SRP.** One owner per concern; every method does one thing; no
  god-functions; helpers private; tests target the **public API**.
- **SSOT.** Every slug/path/role declared once; consumers reference, never
  re-spell. No second registry, no second role enum.
- **Naked data.** No freshness/staleness/hash/version/exists checks. On-demand
  `identity()` digest is allowed; never a validity verdict or a "verify before
  use" gate. (Removed: snapshot verification, membership validation gates,
  etc.)
- **Transport.** Sealing/transport stays `Bundle` / `BundlePipeline` /
  `BundleRole`; the dataset surface never seals.
- **Cross-lane.** One shared path used by **Colab and Kaggle**; align with the
  Colab class (`ColabSpec` on `main`) — reuse, do not duplicate.
- **Tracebacks.** Every caught/handled error records the **full traceback**
  (`traceback.format_exc()` / `exc_info=True`); no truncated one-liners; no
  silent fail-soft.
- **Fail-loud on real runtime errors only** (e.g. an empty fetch must not be a
  silent exit-0 success) — that is error visibility, **not** a data check.

## 6. Open questions

1. Should the hosted registry live in `core/dataset.py` beside `Dataset`, or a
   separate module (e.g. `core/hosted_dataset.py`)?
2. Confirm `BundleRole` covers all roles (add `requests`/`decisions`/`holdout`/
   `ckpt` if missing) vs a hosted-specific role enum.
3. Which branch/worktree: `main` (where the Colab work lives) vs a dedicated
   branch.
4. Do the output datasets (`decisions`, `finetune-ckpt`) get the publish/push
   direction modeled here, or stay with `KaggleDatasets` (upload/download)?
