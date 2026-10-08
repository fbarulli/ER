# Laya lane

Typed-decision surface for the `laya` package. Lives alongside the Colab
lanes ([colab-lane.md](colab-lane.md)) and the Kaggle transport lane
([kaggle-lane.md](kaggle-lane.md)); this lane (like the owner ruling 8
precedent) never imports or edits those packages.

## Contract + evidence

| layer | file |
|---|---|
| lane | `src/cli/laya_lane.py` |
| SSOT spec | `src/core/schemas.py` -> `LayaSpec` (additive; mounted in `TrainingConfig` as `laya:`) |
| question schema | `config/laya.question.json` |
| repo-root shim | `laya_backend.py` (the `er-laya` entry, colab_backend.py pattern) |
| offline pins | `tests/test_laya_lane.py` (offline, no network) |

## Rulings

**Single T4 (2xT4 -> 1xT4).** The staged kernel payload never requests a
double accelerator: the script pins `CUDA_VISIBLE_DEVICES = "0"`, the
receipt records `gpu: "T4 (single)"`, and `LayaSpec.gpu` is a
`Literal["T4"]` — `gpu: 2xT4` fails validation at config load, never on a
session.

**One class, two kinds.** `LayaLane(kind)` accepts only `kaggle` or
`colab`; anything else fails loud. `kind=colab` is a DELIVERY CONTRACT
(payload + receipt text) and nothing else: no `cli.colab` import, no
session call (the colab surface stays in cli.colab / cli.colab_lane
untouched).

**laya installs over pip.** Never vendored and never a local wheel: the
kernel script runs `pip install laya` on the session first.

## Decision kinds

| kind | decision CSV (SSOT binding, `laya.decision_csv_bindings`) | state column | what the run decides |
|---|---|---|---|
| `attribute` | `dataset` (dataset.csv) | `attribute` | typed decisions over the export's attribute text — whether laya's calibrated route agrees with the attributes the task layout fixed. NOT a replacement for the frozen SKU_ITEM/GTIN attribution path |
| `identity` | `final_validation` (data/final_validation.csv) | `attribute_pairs` | typed decision over the frozen P0 validation population. NOT a replacement for candidate-generated + gated + annealed/rank path |
| `laya-cli-eval` | `final_validation` | `attribute_pairs` | the `laya-evals run` harness score over the same identity samples (`expected.identity_claim` builds from true_label) |

## Question schema

`config/laya.question.json` is the SSOT for the typed questions
(`schema: er-laya-questions-v1`). The three headline questions:

| key | type | ask |
|---|---|---|
| `attribute_alignment` | `choice` | aligned / mismatched / obscure verdict on the state's attribute evidence |
| `identity_claim` | `noul` | does the paired attribute evidence support the same-grocery-item claim |
| `package_state` | `noul` | does the state carry explicit package-quantity evidence |

The same file also declares the per-field `field_same:<attribute>` choices
(volume, pack, package_type, sweetener, flavor, carbonation), the pack/format
equivalences (`pack_volume_equal`, `pack_format_equivalent`), and the
gate/counterfactual checks (`gate_verdict`, `gate_reason`, `counterfactual`,
`same_brand_only`, `evidence_sufficient`, `better_match`). The live question
set is whatever the file declares — this table names the headline three, not
an exhaustive or fixed count.

No silent placeholder: a missing schema file fails the stage
(FileNotFoundError); a schema with no `questions` dict fails too
(ValueError).

## Commands

`python laya_backend.py` (= `PYTHONPATH=src .venv/bin/python -m
cli.laya_lane`). Dry-run by default; `--execute` is the only network path.

```bash
# stage the kaggle kernel payload (kernel-metadata.json + laya_decision.py
# + question schema + decision csv + <decision>.receipt.json)
python laya_backend.py --kind kaggle --decision attribute

# stage the colab notebook payload + receipt (delivery contract only;
# nothing to --execute)
python laya_backend.py --kind colab --decision identity

# push the staged payload (needs laya.export_dataset_slug + kaggle json)
python laya_backend.py --kind kaggle --decision attribute --execute
```

## Stop

First-class teardown; dry-run by default, `--execute` cancels. The kernel is
resolved from `--decision` (or an explicit `--slug`), never a borrowed
`cpu`/`gpu` label:

```bash
# what it would stop (--decision finetune resolves fbarulli/er-laya-finetune)
python laya_backend.py --kind kaggle --decision finetune --stop

# cancel the running session
python laya_backend.py --kind kaggle --decision finetune --stop --execute
```

Every push records the kernel session id **at launch**
(`logs/kaggle/<kernel>.session_id`, via `capture_kernel_session_id`), so the
stop cancels the EXACT session through the SDK (`cancel_kernel_session`); with
no recorded id it falls back to the version-replace stub. The stop stages under
`results/kaggle_lane/laya_stop` — the `which='laya'` label, never `cpu`.

## Preflight (ducks in a row, before `--execute`)

Every item, and how to re-verify it (all read-only except the launch):

| duck | check |
|---|---|
| credentials | `~/.kaggle/kaggle.json` present; **no** `~/.kaggle/access_token` (the 403 trap) |
| tip gate | staged `<decision>.receipt.json` `published_tip == HEAD == origin/main` (`core.runtime_inputs.require_published_tip_match`; called from the laya staging path) |
| corpus (full) | `data/laya/{train,dev,test}.jsonl` + `receipt.json`, staged into `results/laya_lane/kaggle/<kind>/dataset_payload/` |
| base-model dataset | `kaggle datasets status fbarulli/er-laya-base` -> `ready` (the fine-tune attaches it; no Hub fetch) |
| corpus dataset | `kaggle datasets status fbarulli/er-laya-train` -> `ready` (versioned on `--execute`) |
| kernel | `kaggle kernels status fbarulli/er-laya-finetune`; metadata `enable_gpu:true`, `enable_internet:true`, `dataset_sources: [er-laya-train, er-laya-base]` |
| recipe | `config/training.yaml` `laya:` (or schema defaults): epochs 8, micro_batch 8, grad_accum 8, seed 1729, base `convaiinnovations/laya` |

The full-cohort corpus is built by `scripts/laya_build_dataset.py` from the full
prepared bundle (`data/prepared/full/worker_1_baseline.pkl.gz`); there is no
partial-corpus variant.

The fine-tune kernel is a **standalone pushed script**: it clones no repo, so a
dirty working tree does not block it — only the tip gate (the commit) matters.
The base model and corpus travel as attached datasets, never the git checkout.

```bash
# full-corpus fine-tune (single T4): version the corpus dataset, push, follow logs
python laya_backend.py --kind kaggle --decision finetune --execute

# fetch the checkpoint + receipt back (sha-verified)
python laya_backend.py --kind kaggle --decision finetune --fetch \
  --slug fbarulli/er-laya-finetune --execute
```

## Validation (default-on)

Every `--decision finetune` run now scores the just-trained checkpoint on the
**held-out `test` split** inside the kernel, so the receipt always carries the
honest generalization number — the training-time eval is the `dev` split and
overlaps training/calibration (`is_held_out: false`).

- result: `receipt["held_out"]` + `checkpoint/held_out_report.json`
  (`{"eval_mode":"held_out","is_held_out":true,"eval_split":"test",
  "metrics":{accuracy,loss,ece,brier,…}}`)
- opt out: `ER_LAYA_HELD_OUT=0` (then the training eval is the only number)
- a held-out failure is recorded as `receipt["held_out_error"]` and never
  discards the trained checkpoint

This bakes the old two-step flow (fine-tune, then a separate
`--decision finetune-eval`) into every run; the eval-only kind still exists for
scoring an arbitrary fetched checkpoint against an arbitrary split.

## Config (SSOT: `config/training.yaml` -> `laya:`)

`staging_dir` (results/laya_lane), `question_schema`
(`config/laya.question.json`), `decision_csv_bindings` (per-kind F
bindings), `dataset_slug` / `export_dataset_slug` (fail-loud when unset —
no silent default account for the kaggle payload bindings; the kaggle-lane
`slug: null` precedent), `run_tag_prefix` (laya_), `gpu` (Literal "T4"),
`laya_decision_batch_size` (8, 1..128), `laya_decision_epochs` (1..8; **0
disables the lane entirely**), `laya_decision_max_rows` (2500),
`min_router_confidence` (0.0 = no gate), `calibration` / `onnx` /
`laya_evals_enabled` toggles, `checkpoint_hub` (`convaiinnovations/laya`),
`laya_package` (`laya`).

Fine-tune surface: `finetune_kernel_slug` (`fbarulli/er-laya-finetune`),
`finetune_dataset_slug` (`fbarulli/er-laya-train`), `base_model_dataset`
(`fbarulli/er-laya-base`, the attached base checkpoint — no Hub fetch),
`finetune_ckpt_dataset` (`fbarulli/er-laya-finetune-ckpt`, for the eval-only
kind), and the `finetune:` recipe block (epochs 8, micro_batch 8, grad_accum 8,
seed 1729, loss `soft-ce`).

The committed config/training.yaml does NOT carry a `laya:` block yet —
the schema default factory (`src/core/laya_config.py`) supplies the slugs and
recipe above and keeps the load byte-identical (additive contract; the same one
`kaggle:` rode at its landing). Add the block only to override a default.

## Logging convention + artifact paths

- Lane status lines carry the same Europe/Paris stamp (CET/CEST) as
  kaggle/colab lanes: `[laya-lane <YYYY-MM-DDTHH:MM:SS CET|CEST>]`
  (kaggle_lane._stamp mirror; the earlier UTC-stamp phrasing predates the
  CET convention that landed there). Console + append-only
  `logs/laya/lane.log` (canonical one-roof TRAIN_ROOT/logs convention).
- Inside a KERNEL (remote, timezone-less) the timestamp is UTC — the
  remote session may not share this box's zone; logs/laya/lane.log stays
  Europe/Paris.

| path | content |
|---|---|
| `logs/laya/lane.log` | append-only lane log, Europe/Paris-stamped |
| `results/laya_lane/kaggle/<decision>/` | staged payload: kernel-metadata.json + laya_decision.py (or laya_evals.py) + laya.question.json + decision csv + `<decision>.receipt.json` |
| `results/laya_lane/colab/<decision>/` | colab delivery payload: laya_decision_colab.py + receipt (no session call) |
| `results/laya_lane/fetch/<decision>/` | fetched-back `kaggle kernels output` payload (verified) |
| `logs/kaggle/<kernel>.session_id` | launch-recorded session id — the SDK `cancel_kernel_session` target `--stop` uses |
| `results/kaggle_lane/laya_stop/` | stop stub staging (only used when no session id was recorded -> version replace) |

## What does X run now? (intent -> command -> surface -> artifacts)

| intent | command | surface | artifacts |
|---|---|---|---|
| stage the kaggle GPU decision kernel | `python laya_backend.py --kind kaggle --decision attribute` | local staging only | `results/laya_lane/kaggle/attribute/*` |
| stage the SAME payload as a colab notebook | `python laya_backend.py --kind colab --decision identity` | local staging only | `results/laya_lane/colab/identity/*` |
| push the staged kernel | `--kind kaggle --decision attribute --execute` | `kaggle kernels push` | session queued on a T4 |
| fetch results back | inside the kernel export | `/kaggle/working tar` + hashed receipt | staged decision outputs |

## Caveats

Known gaps + provenance for the operator (no network claim is made here —
these are recorded as CONSIDERATIONS, not live-probed):

* **laya wheels require python >= 3.10.** The pinned stack per laya's docs
  is torch 2.14 + transformers 5.x + huggingface_hub 1.x; the ER venv
  (python 3.14) carries torch 2.14.0+cu130 locally, so the local stack
  baseline matches laya's floor provably for THIS box. The lane's staged
  payload does NOT claim a wheel for every target: the aarch64 torch 2.14
  availability on the remote session comes from the session image, not
  from here.
* **Runtime availability is checked on the session, not at staging.**
  `pick_device()` fails loud if cuda is not up and the pip install step
  fails loud if the laya wheel (or its deps) cannot resolve. The lane
  deliberately does NOT run a `pip index` probe at staging time — that is
  a network path, and the `--execute` gate exists for exactly that
  boundary.
* **The `onnx` toggle is recorded but not yet wired** into the kernel
  script (the laya ONNX export path needs the `laya[onnx]` extra on the
  session first). The receipt records the toggle state so the wiring is
  visible, never silent.
* **The colab surface is a delivery contract only.** `LayaLane("colab")`
  renders a payload + receipt; nothing in this lane opens a Colab
  session (ownership stays with cli.colab / cli.colab_lane per the owner
  ruling).
* **`kaggle_backend.py`-style kaggle yaml still governs the network
  transport config** (`kaggle:` block) — the laya lane reads its own
  `laya:` block only and touches no colab/preparation keys.

## Provenance

Relaunch of the laya dual-surface mission (the previous session's landing
was lost to an empty final report; `src/cli/laya_lane.py` did not exist
on relaunch, verified by `git status --short` + filesystem). Everything
here was landed fresh, offline-pinned by tests/test_laya_lane.py.
