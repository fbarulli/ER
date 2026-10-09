"""Owner's model-evaluation script (bundle-adapted paths).

Needs embedding_similarities.csv (zero_shot_sims.py) and
labeled_pairs.csv (true_label per pair — generate from the gate's
auto_duplicate rule or a manual review round).

MANIFEST (SILENT_DROPS task 7): the stage snapshots its three input
CSVs (labeled_pairs, embedding_similarities, canonical_records), writes
model_evaluation_summary.csv atomically, and publishes
results/manifests/evaluate_models.json LAST.

ROW ACCOUNTING (code truth — PAIR-level over the labeled universe): an
eval stage consumes labeled pairs, not dataset rows, so the accounting
unit is the labeled pair. input_rows = labeled pairs read; output_rows =
pairs that stay in play (DEV + TEST — both are consumed: DEV to fit the
Youden threshold, TEST to score); dropped = inner-join merge loss (0
rows today, kept loud), straddling pairs (endpoints in different
component folds — unassignable, counted + printed by the split block),
and parked pairs (whole pairs in unused folds, k > 2). The code already
asserts dev + test + straddle + parked == merged rows; the manifest
makes that assert-exact partition MANIFEST-ENFORCED (closure
input == output + sum(dropped) is re-asserted at publish and on every
verify). Model-level facts (models evaluated / skipped for a missing
sweep column) are recorded outside `dropped` — a skipped model is a
missing summary row, not a dropped data row.

HOLDOUT DISCIPLINE (self-fit leak closed 2026-09-14): this lane used to
pick its Youden threshold via roc_curve ON THE VERY LABELED SET IT THEN
SCORED accuracy/F1 on — every zero-shot operating metric was inflated.
Now the labeled pairs are split into DEV/TEST halves along connected
components of the positive-pair gtin graph (src/training/folds.
component_folds), the DEV/TEST fold indices are derived from
split.holdout_component_folds (n_folds-2 / n_folds-1, matching
build_final_validation's P0 fold column), the Youden threshold is fit on
DEV ONLY and applied
verbatim to TEST, and ALL reported metrics — ROC-AUC included — are
TEST-half numbers. youden_thr_test_descriptive refits the argmax ON
TEST as the leak diagnostic only (same convention as src/training/training.py
line 1070); it is never applied.
"""


import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from core.common import (
    RESULTS,
    SEED,
    F,
    ensure_parent,
    plot_dpi,
    set_determinism,
    training_cfg,
)
from core.manifest import atomic_write_csv, begin_manifest, finish_manifest
from core.prediction_export import PredictionExport
from core.schemas import EVAL_SUMMARY_COLUMNS, check_eval_summary_frame
from core.ranking_metrics import ranking_at_k, youden_threshold

# determinism (2026-10-06): this lane re-fits the Youden threshold and
# reads embeddings — pin the global RNGs before any of that. The
# component split below keeps passing seed=SEED explicitly (its own
# deterministic contract, unchanged).
set_determinism(SEED)

LABELED_PAIRS_CSV = F["labeled_pairs"]
EMBED_SIM_CSV = F["embedding_similarities"]
CANON_CSV = F["canonical_records"]

# Stage manifest (SILENT_DROPS task 7) — begin BEFORE the work: all three
# input CSVs are sized now so the record pins exactly what this stage
# read. Seed = the SSOT seed; the component split below consumes RNG
# through it (component_folds(seed=SEED)), so the split this manifest
# certifies is the seeded one.
manifest = begin_manifest(
    "evaluate_models",
    inputs=[LABELED_PAIRS_CSV, EMBED_SIM_CSV, CANON_CSV],
    seed=SEED,
)

# Typed SSOT: consume the validated TrainingConfig directly instead of the
# raw merged dict (training_cfg() is the one validated accessor; the raw
# load_config() view is only for cross-file merges).
_training_cfg = training_cfg()
MODEL_COLUMNS = dict(_training_cfg.sim_columns)
_EV = _training_cfg.evaluation  # EvaluationSpec-validated: retrieval_ks
_split = _training_cfg.split    # SplitSpec-validated: policy + fold arity

labeled = pd.read_csv(LABELED_PAIRS_CSV, dtype={"gtin1": str, "gtin2": str}, keep_default_na=False)
if not EMBED_SIM_CSV.exists():
    raise SystemExit(
        f"{EMBED_SIM_CSV.name} missing — run src/training/zero_shot_sims.py first"
    )
emb_sim = pd.read_csv(EMBED_SIM_CSV, dtype={"gtin1": str, "gtin2": str})
# GTIN keys are read/joined as NORMALIZED STRINGS everywhere in this stage
# (labels, sims, canonical map, fold map). Measured on the live artifacts:
# 8,800/8,800 labeled endpoints resolve against canonical_records (0 rows
# affected TODAY), but the join is dtype-fragile in one direction — a
# zero-prefixed UPC-12 canonical ("012345678905") read through any
# int64-coercing path loses its leading zero and then NaNs out of BOTH
# canon-map joins below (synthetic repro: {'012345678958'} x str-read
# canonical maps; int-read maps to NaN). dtype=str is this file's answer —
# the keys stay byte-spelled and the join needs no GTIN rewriting: rewriting
# keys here is deliberately NOT done because every other lane's key remains
# the RAW gtin spelling (core.gtin module doctrine: "grouping keys stay
# the RAW gtin string"), and folding spellings only in this script would
# fork the identity namespace between stages.
canon = pd.read_csv(CANON_CSV, dtype={"gtin": str}, keep_default_na=False)
# AUDIT FIX (round 2 F14, round 3): inner-merge drop accounting — nothing
# may drop silently (transparency contract). Measured on the real CSVs:
# 19,916 -> 19,916, zero rows lost TODAY; this print keeps any drift loud,
# and an empty merge is a hard stop (upstream sims are missing entirely).
_n_before = len(labeled)
df = labeled.merge(emb_sim, on=["gtin1", "gtin2"], how="inner")
_n_after = len(df)
print(
    f"[merge] labeled x embedding_similarities (inner on gtin1/gtin2): "
    f"{_n_before:,} -> {_n_after:,} rows ({_n_before - _n_after:,} dropped)"
)
if _n_after == 0:
    raise SystemExit(
        "[merge] inner join lost ALL rows — embedding_similarities.csv "
        "does not cover labeled_pairs (rerun src/training/zero_shot_sims.py)"
    )
gtin_to_canon = dict(zip(canon["gtin"].astype(str), canon["canonical"].astype(str), strict=True))
df["canon1"] = df["gtin1"].map(gtin_to_canon)
df["canon2"] = df["gtin2"].map(gtin_to_canon)
# Fail-loud on the silent thinning this stage exists to prevent: an
# unmapped canonical is exactly the NaN-join defect above. 0 today —
# stays 0 or the run stops, never "prints oddly and continues".
_unmapped_canon = df["canon1"].isna().sum() + df["canon2"].isna().sum()
if _unmapped_canon:
    _bad = sorted(set(df.loc[df["canon1"].isna(), "gtin1"]) | set(df.loc[df["canon2"].isna(), "gtin2"]))[:5]
    raise SystemExit(
        f"[canon-map] {_unmapped_canon:,} pair endpoints have no "
        f"canonical mapping in {CANON_CSV.name} (e.g. {_bad}) — the "
        "dtype=str join above makes this impossible for a canonical with "
        "a leading-zero UPC-12 spelling, so a live miss means the canon "
        "file itself is out of date; rebuild it (pipeline data-prep lane) before "
        "scoring"
    )

# ── DEV/TEST component split (holdout discipline) ─────────────────────────
# P0 RETARGET. This block used to build its OWN graph from the labeled census
# ALONE and cut `component_split_k=2` folds over it, which is precisely how the
# leak happened: the evaluation graph contained no training edges, so its folds
# were drawn from a different component structure than the one the model trained
# under. Measured on the current census that protocol contaminates 73.7% of
# pairs (24.0% of positives with BOTH endpoints in train), so the reported
# P@R95 measured memorization as much as generalization — and nothing raised,
# because a leak is not a crash.
#
# The fold assignment now comes from `data/final_validation.csv`, which is cut
# from the MERGED component graph (training positives UNION validation
# positives) by the single entry point `folds.derive_holdout`. Two properties
# are inherited and no longer re-derived:
#   * fold ∈ {0,2,3} where 0 = trained on. A pair with either side in fold 0 is
#     scored on data the model saw, so it can never enter DEV or TEST.
#   * a positive's two endpoints share a fold by construction (it is one graph
#     edge), so `_pos_straddle` below can only fire if the artifact is corrupt.
#
# The P0 fold column uses train=0, dev=n_folds-2, test=n_folds-1, where
# n_folds is split.holdout_component_folds (the ONE arity the 50/25/25
# contract supports; build_final_validation._resolve_quarter_folds assigns it).
# evaluation.component_split_k / dev_fold / test_fold are the LEGACY 2-fold
# protocol (0/1) and are NOT the selectors anymore — reading them here would
# silently fold a trained-on quarter into the scored half.
_FINAL_VALIDATION_CSV = F["final_validation"]
_FOLD_MAP_CSV = F["validation_fold_map"]
_fold_map = pd.read_csv(_FOLD_MAP_CSV, dtype={"gtin": str}, keep_default_na=False)
for _required in ("gtin", "fold"):
    if _required not in _fold_map.columns:
        raise SystemExit(
            f"{_FOLD_MAP_CSV} has no `{_required}` column — it is not a P0 "
            "split artifact. Rebuild both artifacts with "
            "`PYTHONPATH=src python -m src.training.build_final_validation`."
        )
if not os.path.exists(_FINAL_VALIDATION_CSV):
    raise SystemExit(
        f"{_FINAL_VALIDATION_CSV} is missing. Rebuild with "
        "`PYTHONPATH=src python -m src.training.build_final_validation`."
    )
# The validation frame's per-side slice columns, keyed by the SSOT pair_id, so
# every per-sample prediction below can be read on the same row as the slice it
# scored (core.prediction_export). Loaded once, joined per model.
_validation = pd.read_csv(
    _FINAL_VALIDATION_CSV, dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
)
_slices = PredictionExport.slice_frame(_validation)
if "true_label" not in df.columns:
    raise SystemExit("labeled census lost its `true_label` column")

# The fold map covers EVERY graph entity, so each pair's fold is looked up
# rather than inferred from a CSV that only holds the scored half. A pair
# absent from the validation CSV is therefore not an error — it is a pair the
# model trained on, and it is parked below, never scored.
_fold_of = dict(zip(_fold_map["gtin"], _fold_map["fold"], strict=True))
df["fold"] = df["gtin1"].map(_fold_of)
df["fold_2"] = df["gtin2"].map(_fold_of)
_unmapped = int(df["fold"].isna().sum() | 0) + int(df["fold_2"].isna().sum() | 0)
if _unmapped:
    _bad = sorted(
        set(df.loc[df["fold"].isna() | df["fold_2"].isna(), "gtin1"].head(5))
    )
    raise SystemExit(
        f"{_unmapped:,} pair endpoints have no row in {_FOLD_MAP_CSV} "
        f"(e.g. {_bad}). The evaluation population and the P0 split are "
        "derived from different graphs, so no metric computed from them is "
        "comparable. Rebuild with src.training.build_final_validation."
    )

_TRAIN_FOLD = 0
# P0 fold-column indices, DERIVED from split.holdout_component_folds exactly as
# build_final_validation._resolve_quarter_folds assigns them (train=0,
# dev=n_folds-2, test=n_folds-1). evaluation.component_split_k/dev_fold/test_fold
# are the LEGACY 2-fold protocol (0/1) and no longer steer the P0 fold column —
# reading them as selectors would silently fold a trained-on quarter into the
# scored half. Committed config: holdout_component_folds=4 -> DEV=2, TEST=3,
# identical to the previous hardcoded selectors.
_N_FOLDS = int(_split.holdout_component_folds)
_DEV_FOLD = _N_FOLDS - 2
_TEST_FOLD = _N_FOLDS - 1
# ── scored-half negative fold-assignment policy (DECIDED 2026-10-01 — see ──
# the DECISION block in src/training/build_final_validation.py: policy B
# "train_side" is the config default; the negatives that scored nowhere
# before (straddle 4,728) now all score. This is the ONE rule both surfaces
# obey — imported, not re-derived, or the consumer would fork the split a
# second time exactly like its own history (the P0 leak) records. Legal values
# are schema-pinned (SplitSpec.negative_fold_policy is a Literal), so the old
# re-inlined check is gone.
_NEG_POLICY = str(_split.negative_fold_policy)
from training.build_final_validation import negative_pair_fold  # noqa: E402

if _NEG_POLICY == "train_side":
    # apply the SAME assignment rule the artifact used: negatives fold whole
    # (train-side fold when a train-side endpoint exists, else fold_a); the
    # raw endpoint mismatch stays visible as fold_raw (byte-visible evidence)
    _raw = df[["fold", "fold_2", "true_label"]].copy()
    df["fold_raw_only_census"] = _raw["fold"]
    # The rule is computed on the NEGATIVE subset only; assigning a full-frame
    # list into a boolean mask raised "Must have equal len keys and value"
    # (falsified 2026-10-09: the lane could not run under policy B at all, so
    # no prediction was ever emitted on that path).
    _neg_mask = df["true_label"] == 0
    df.loc[_neg_mask, "fold"] = [
        negative_pair_fold(
            _NEG_POLICY, int(f1), int(f2), n_folds=_N_FOLDS
        )
        for f1, f2 in zip(
            _raw.loc[_neg_mask, "fold"], _raw.loc[_neg_mask, "fold_2"]
        )
    ]
    df.loc[_neg_mask, "fold_2"] = df.loc[_neg_mask, "fold"]
    # ATTRIBUTION (measured on the current census, 1,023 positives / 7,713
    # negatives): dropped.straddling_fold_pairs 4,728 -> 0 and
    # parked_fold_pairs 1,927 -> 5,669 under policy B; in-play grows
    # 1,623 -> 2,609 (DEV 976 -> 1,471, TEST 647 -> 1,138). When a future
    # census regen moves these, the pinned-update convention applies: the
    # inline old number is re-recorded in this comment before it changes.
in_dev = (df["fold"] == _DEV_FOLD) & (df["fold_2"] == _DEV_FOLD)
in_test = (df["fold"] == _TEST_FOLD) & (df["fold_2"] == _TEST_FOLD)
straddle = df["fold"] != df["fold_2"]  # endpoints in different quarters — unassignable
# whole pair in a quarter this eval does not score (fold 0 = the model trained
# on it) — withheld, never scored as if it were held out
parked = ~(in_dev | in_test | straddle)
# the Youden threshold is fit on DEV, TEST is never touched until scoring
_pos_straddle = int((straddle & (df["true_label"] == 1)).sum())
if _pos_straddle:
    raise AssertionError(
        f"{_pos_straddle} POSITIVE pairs straddle the fold split — the "
        f"component guarantee is broken ({_FOLD_MAP_CSV} is corrupt, "
        "or a positive is missing its graph edge)"
    )
_trained_on = int((parked & (df["fold"] == _TRAIN_FOLD)).sum())
_pos_parked = int((parked & (df["true_label"] == 1)).sum())
_neg_dev = int((df.loc[in_dev, "true_label"] == 0).sum())
_pos_dev = int((df.loc[in_dev, "true_label"] == 1).sum())
_neg_test = int((df.loc[in_test, "true_label"] == 0).sum())
_pos_test = int((df.loc[in_test, "true_label"] == 1).sum())
if _pos_dev == 0 or _neg_dev == 0:
    raise ValueError(
        f"DEV half must contain BOTH classes for the Youden fit — "
        f"got pos={_pos_dev:,} / hard-neg={_neg_dev:,} (adjust "
        f"split.holdout_component_folds in config/training.yaml)"
    )
if _pos_test == 0 or _neg_test == 0:
    raise ValueError(
        f"TEST half must contain BOTH classes for honest metrics — "
        f"got pos={_pos_test:,} / hard-neg={_neg_test:,} (adjust "
        f"split.holdout_component_folds in config/training.yaml)"
    )

print(
    f"[split] fold assignment from {_FINAL_VALIDATION_CSV} (merged component "
    f"graph, P0): DEV fold {_DEV_FOLD} / TEST fold {_TEST_FOLD} | "
    f"dev pos={_pos_dev:,} neg={_neg_dev:,} | "
    f"test pos={_pos_test:,} neg={_neg_test:,} | "
    f"straddling neg={int((straddle & (df['true_label'] == 0)).sum()):,} "
    f"parked={int(parked.sum()):,} (trained-on pairs are never scored)"
)

print(
    f"[split] DEV  = fold {_DEV_FOLD} (split.holdout_component_folds - 2): "
    f"{int(in_dev.sum()):,} pairs ({_pos_dev:,} pos / {_neg_dev:,} hard-neg)"
)
print(
    f"[split] TEST = fold {_TEST_FOLD} (split.holdout_component_folds - 1): "
    f"{int(in_test.sum()):,} pairs ({_pos_test:,} pos / {_neg_test:,} hard-neg)"
)
print(
    f"[split] straddling pairs dropped (endpoints in different folds): "
    f"{int(straddle.sum()):,} — ALL hard-negatives, 0 positives "
    f"(component guarantee asserted above)"
)
if int(parked.sum()):
    print(
        f"[split] pairs parked whole in unused folds (k > 2): "
        f"{int(parked.sum()):,} ({_pos_parked:,} pos / "
        f"{int((parked & (df['true_label'] == 0)).sum()):,} hard-neg)"
    )
_accounted = int(in_dev.sum()) + int(in_test.sum()) + int(straddle.sum()) + int(parked.sum())
assert _accounted == len(df), (
    f"split accounting {_accounted:,} != merged rows {len(df):,} — rows "
    f"vanished or doubled in the DEV/TEST split"
)
print(
    f"[split] accounting: {int(in_dev.sum()):,} dev + {int(in_test.sum()):,} test "
    f"+ {int(straddle.sum()):,} straddling + {int(parked.sum()):,} parked "
    f"= {_accounted:,} = merged rows — nothing dropped silently"
)

df_dev = df.loc[in_dev]
df_test = df.loc[in_test]


# The Youden cutoff is the core.ranking_metrics.youden_threshold SSOT:
# importing TRAIN.training here would drag transformers and tracking
# contexts into a reporting script, and the old verbatim local copy could
# drift from the training lane on the tie-break.
# HOLDOUT DISCIPLINE: fit on DEV scores only, apply the returned threshold
# verbatim to TEST — never on the scores it rates.


def evaluate_model(
    df_half: pd.DataFrame,
    sim_col: str,
    threshold: float | None = None,
    true_col: str = "true_label",
) -> dict:
    """Score ONE model on ONE half at an EXTERNALLY-SUPPLIED threshold.

    SELF-FIT HAZARD (closed 2026-09-14): threshold=None used to fit the
    Youden threshold via roc_curve on these very scores, then report
    accuracy/F1 on the same set — inflating every operating metric. None
    now raises: the caller must pass a threshold fit on a DIFFERENT half
    (dev-fit Youden, test-scored — see the [split] block above).
    """
    if threshold is None:
        raise ValueError(
            "evaluate_model(threshold=None) is the closed self-fit leak: "
            "it would pick the Youden threshold on the SAME labeled set it "
            "scores and inflate accuracy/F1. Fit the threshold on the DEV "
            "component fold (youden_threshold on df_dev) and pass it in — the "
            "scored half must never choose its own threshold."
        )
    y_true = df_half[true_col].values
    y_scores = df_half[sim_col].values
    roc_auc = roc_auc_score(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # zero_division=0 keeps sklearn from crashing on an empty side; the
    # counter below keeps that substitution LOUD instead of silent.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    if tp + fp == 0:
        print(
            f"  [warn] {sim_col}: 0 predicted positives at thr={threshold:.4f} "
            f"— precision is undefined (0/0), reported as 0 (zero_division=0)"
        )
    if tp + fn == 0:
        print(
            f"  [warn] {sim_col}: 0 actual positives in this half at "
            f"thr={threshold:.4f} — recall is undefined (0/0), reported as 0"
        )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        **ranking_at_k(y_true, y_scores, tuple(_EV.retrieval_ks)),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


summary_rows = []
#: Per-model per-sample prediction dumps (core.prediction_export): the artifact
#: that makes "which pair did this model miss, and on what slice" answerable.
prediction_paths: list = []
# models whose sim column is present in the sweep output (a partial sweep
# is evaluated — never crash on a missing column, say it loudly instead).
# Skipped models are recorded for the manifest (model-level facts, NOT a
# row drop — a skipped model means a missing summary row, which the
# models_evaluated / models_skipped census pins).
_models_skipped: list[str] = []
_missing = [c for c in MODEL_COLUMNS.values() if c not in df.columns]
if _missing:
    print(f"[warn] no similarity column yet (sweep incomplete): {_missing}")
for model_name, sim_col in MODEL_COLUMNS.items():
    if sim_col not in df.columns:
        print(f"\nMODEL: {model_name} — skipped (no {sim_col} in sweep csv)")
        _models_skipped.append(model_name)
        continue
    print(f"\n{'=' * 70}\nMODEL: {model_name}\n{'=' * 70}")
    # HOLDOUT DISCIPLINE: threshold fit on DEV, applied verbatim to TEST.
    thr = youden_threshold(
        df_dev[sim_col].to_numpy(), df_dev["true_label"].to_numpy()
    )
    thr_test_descriptive = youden_threshold(
        df_test[sim_col].to_numpy(), df_test["true_label"].to_numpy()
    )
    metrics = evaluate_model(df_test, sim_col, threshold=thr)
    summary_rows.append(
        {
            "model": model_name,
            "eval_half": "test",
            "threshold_source": "dev_youden",
            "youden_thr_dev": thr,
            **metrics,
            "n_dev": int(in_dev.sum()),
            "n_test": int(in_test.sum()),
            "youden_thr_test_descriptive": thr_test_descriptive,
        }
    )
    print(f"  ROC AUC      : {metrics['roc_auc']:.4f}  (TEST half)")
    print(f"  PR-AUC       : {metrics['pr_auc']:.4f}")
    print(
        f"  P@1/5/10     : {metrics['precision_at_1']:.4f} / "
        f"{metrics['precision_at_5']:.4f} / {metrics['precision_at_10']:.4f}"
    )
    print(
        f"  R@1/5/10     : {metrics['recall_at_1']:.4f} / "
        f"{metrics['recall_at_5']:.4f} / {metrics['recall_at_10']:.4f} | "
        f"Hits@1 {metrics['hits_at_1']:.4f}"
    )
    print(f"  Threshold    : {thr:.4f}  (Youden fit on DEV — applied verbatim to TEST)")
    print(
        f"  [diag] Youden refit ON TEST (leak diagnostic, never applied): "
        f"{thr_test_descriptive:.4f}"
    )
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(
        f"  TP: {metrics['tp']}  TN: {metrics['tn']}  FP: {metrics['fp']}  FN: {metrics['fn']}"
    )
    df_model = df_test.copy()
    df_model["pred"] = (df_model[sim_col] >= thr).astype(int)
    for kind, m in (
        ("False Positives", (df_model.true_label == 0) & (df_model.pred == 1)),
        ("False Negatives", (df_model.true_label == 1) & (df_model.pred == 0)),
    ):
        sub = df_model[m]
        print(
            f"\n  {kind} (predicted duplicate but not): {len(sub)}"
            if kind.startswith("False P")
            else f"\n  {kind} (actual duplicate but missed): {len(sub)}"
        )
        for _, row in (
            sub.sample(min(3, len(sub)), random_state=SEED).iterrows()
            if len(sub)
            else []
        ):
            print(f"    sim={row[sim_col]:.3f}")
            print(f"      GTIN1 {row['gtin1']}: {str(row['canon1'])[:80]}")
            print(f"      GTIN2 {row['gtin2']}: {str(row['canon2'])[:80]}")
    # ── per-sample prediction artifact (was computed then DISCARDED) ──────
    # One row per scored pair, keyed by the SSOT pair_id and joined to the
    # validation slice columns, so a metric can be drilled back to the sample
    # that produced it (core.prediction_export).
    prediction_path = PredictionExport.write(
        PredictionExport.frame(
            df_model, _slices, model=model_name, eval_half="test",
            score_column=sim_col, threshold=thr,
        ),
        RESULTS / PredictionExport.filename(model_name, "test", _TEST_FOLD),
    )
    prediction_paths.append(prediction_path)
    print(
        f"  [traceability] {len(df_model):,} scored pairs -> {prediction_path}"
    )

summary_df = pd.DataFrame(summary_rows)[list(EVAL_SUMMARY_COLUMNS)]
# boundary contract: exact columns, every row an EvalSummaryRow (provenance
# + confusion-counts consistency), and NO NaN/inf anywhere — all loud.
check_eval_summary_frame(summary_df)
print(
    f"[contract] eval summary: {len(summary_df)} rows "
    f"× {len(EVAL_SUMMARY_COLUMNS)} cols — EvalSummaryRow-valid, all values finite"
)
print("\nSUMMARY TABLE (TEST component half; thresholds fit on DEV):")
print(summary_df.to_string(index=False))
# atomic write (SILENT_DROPS task 7): the summary is published through
# the same temp-sibling + os.replace mechanism as every other stage
# output — a crash never leaves a truncated CSV on the final path.
summary_out = F["model_evaluation_summary"]
atomic_write_csv(summary_df, ensure_parent(summary_out), index=False)
print(f"\nSaved summary to {summary_out}")

# ── per-model result plots (zero-shot embedding similarity, TEST half) ────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

models_present = [
    (name, col) for name, col in MODEL_COLUMNS.items() if col in df.columns
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
# (a) ROC curves — all models overlaid, scored half only
for name, col in models_present:
    y = df_test["true_label"].values
    s = df_test[col].values
    fpr, tpr, _ = roc_curve(y, s)
    axes[0].plot(fpr, tpr, lw=1.8, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
axes[0].plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8)
axes[0].set_xlabel("false-positive rate")
axes[0].set_ylabel("true-positive rate")
_n_pos = int((df_test["true_label"] == 1).sum())
_n_neg = int((df_test["true_label"] == 0).sum())
axes[0].set_title(
    f"ROC — TEST component half (n: pos={_n_pos:,} / neg={_n_neg:,})"
)
axes[0].legend(fontsize=9)
# (b) per-model metric bars
x = np.arange(len(summary_df))
w = 0.27
for k, (metric, color) in enumerate(
    (("precision", "#4c72b0"), ("recall", "#55a868"), ("f1", "#c44e52"))
):
    axes[1].bar(x + (k - 1) * w, summary_df[metric], w, color=color, label=metric)
# actual confusion counts per model above each metric triplet
for _i, _r in summary_df.iterrows():
    axes[1].text(
        x[_i] + w,
        1.02,
        f"TP {_r['tp']:,}\nFP {_r['fp']:,}\nFN {_r['fn']:,}",
        ha="center",
        fontsize=7,
    )
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary_df["model"], fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].set_title("P/R/F1 at DEV-fit Youden threshold (TEST half)")
axes[1].legend(fontsize=9)
fig.tight_layout()
out1 = RESULTS / "model_comparison_roc.png"
fig.savefig(out1, dpi=plot_dpi())  # SSOT (audit round 2 F03)
plt.close(fig)
print(f"[plot] {out1}")

# score distributions by class — one panel per model, TEST half
fig, axes = plt.subplots(1, len(models_present), figsize=(4.6 * len(models_present), 4), squeeze=False)
for i, (name, col) in enumerate(models_present):
    ax = axes[0][i]
    pos_s = df_test.loc[df_test.true_label == 1, col]
    neg_s = df_test.loc[df_test.true_label == 0, col]
    ax.hist(pos_s, bins=50, alpha=0.6, color="#55a868", label=f"pos (n={len(pos_s):,})")
    ax.hist(
        neg_s, bins=50, alpha=0.6, color="#c44e52", label=f"hard-neg (n={len(neg_s):,})"
    )
    # FIXED (2026-09-14): this lookup used to match sim-column names against
    # summary model KEYS — the axvline never fired. Match on model key.
    thr_row = summary_df.loc[summary_df.model == name, "youden_thr_dev"]
    if len(thr_row):
        ax.axvline(
            float(thr_row.iloc[0]),
            color="black",
            ls="--",
            lw=1.2,
            label="dev-Youden thr",
        )
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("cosine similarity")
    ax.legend(fontsize=8)
fig.suptitle(
    "zero-shot similarity by class — TEST component half (canonical texts)",
    fontsize=11,
)
fig.tight_layout()
out2 = RESULTS / "model_score_distributions.png"
fig.savefig(out2, dpi=plot_dpi())  # SSOT (audit round 2 F03)
plt.close(fig)
print(f"[plot] {out2}")

# ── row accounting + manifest (SILENT_DROPS task 7; capture-only) ─────────
# PAIR-level over the labeled universe (an eval stage consumes labeled
# pairs, not dataset rows). The partition is exactly the one the split
# block above asserts:
#   in play  = DEV + TEST pairs (DEV fits the Youden threshold, TEST is
#              scored — BOTH are consumed, neither is dropped)
#   dropped  = merge loss (inner join labeled x sims — 0 today, kept loud
#              so drift shows as a number)
#            + straddling pairs (endpoints in different folds —
#              unassignable; all hard-negs, positives are asserted zero)
#            + parked pairs (whole pairs in a fold this eval does not
#              score — trained-on fold 0 today)
# closure: input == output + sum(dropped), asserted by finish_manifest
# before the manifest is published and re-checked by verify_manifest.
row_accounting = {
    "input_rows": _n_before,  # labeled pairs read
    "output_rows": int(in_dev.sum()) + int(in_test.sum()),  # DEV + TEST
    "dropped": {
        "merge_dropped_pair_rows": _n_before - _n_after,
        "straddling_fold_pairs": int(straddle.sum()),
        "parked_fold_pairs": int(parked.sum()),
    },
    # P0: pairs the model was TRAINED on. Excluded from DEV/TEST by
    # construction and reported here rather than left implicit -- the old
    # protocol scored 73.7% of its population as if it were held out, and
    # nothing in the artifacts recorded it.
    "trained_on_pairs_not_scored": _trained_on,
    # population detail (outside `dropped`; not part of the closure)
    "dev_pairs": int(in_dev.sum()),
    "test_pairs": int(in_test.sum()),
    "pos_dev": _pos_dev,
    "hard_neg_dev": _neg_dev,
    "pos_test": _pos_test,
    "hard_neg_test": _neg_test,
    # model-level census (a skipped model = a missing summary row, not a
    # dropped data row)
    "models_evaluated": len(summary_df),
    "models_skipped_no_sweep_column": len(_models_skipped),
    "models_skipped_names": sorted(_models_skipped),
    "summary_rows": len(summary_df),
    # per-sample prediction dumps: one row per scored TEST pair per model
    "prediction_artifacts": len(prediction_paths),
    "prediction_rows_per_model": int(in_test.sum()),
    "holdout_component_folds": int(_split.holdout_component_folds),
    "dev_fold": int(_DEV_FOLD),
    "test_fold": int(_TEST_FOLD),
}
manifest_path = finish_manifest(
    manifest,
    outputs=[summary_out, out1, out2, *prediction_paths],
    row_accounting=row_accounting,
    expected_outputs=[F["model_evaluation_summary"]],
)
print(
    f"[manifest] evaluate_models complete -> {manifest_path} | closure "
    f"{row_accounting['input_rows']:,} == {row_accounting['output_rows']:,} "
    f"in-play (dev {row_accounting['dev_pairs']:,} / test "
    f"{row_accounting['test_pairs']:,}) + "
    f"{sum(row_accounting['dropped'].values()):,} dropped "
    f"(merge {row_accounting['dropped']['merge_dropped_pair_rows']:,} / "
    f"straddle {row_accounting['dropped']['straddling_fold_pairs']:,} / "
    f"parked {row_accounting['dropped']['parked_fold_pairs']:,})"
)
