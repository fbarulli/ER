"""lib/common.py — the ONLY file that reads 00_config.yaml.

Everything else in the tree gets paths, file names, the column mapping, the
seed, and shared helpers through this module. No hardcoded paths or file
names exist anywhere else (owner SSOT directive 2026-09-06).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import pandas as pd
import yaml

from lib.text import extract_volume_ml


def require_keys(config: dict, keys: list[str], context: str) -> None:
    """Fail loudly when config is missing keys (no silent defaults)."""
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"{context}: config missing required key(s): {missing}")


def load_config() -> dict:
    """Load 00_config.yaml (the SSOT). Hard error when missing."""
    if not CONFIG_PATH.exists():
        raise SystemExit(f"config missing: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


TRAIN_ROOT = Path(__file__).resolve().parent.parent  # TRAIN_GPU/
CONFIG_PATH = TRAIN_ROOT / "00_config.yaml"
_CFG = load_config()
require_keys(_CFG, ["paths", "files", "column_mapping", "seed"], "00_config.yaml")


def _path(cfg_value: str) -> Path:
    """Config path strings resolve relative to TRAIN_ROOT unless absolute."""
    p = Path(cfg_value)
    return p if p.is_absolute() else TRAIN_ROOT / p


# ── paths (SSOT) ────────────────────────────────────────────────────────────
DATA_DIR = _path(_CFG["paths"]["data_dir"])
RESULTS = _path(_CFG["paths"]["results_dir"])
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_PATH = DATA_DIR / _CFG["files"]["dataset"]

# ── file names (SSOT) ────────────────────────────────────────────────────────
F = _CFG["files"]

# ── column mapping + seed (SSOT, read once) ──────────────────────────────────
COLUMN_MAPPING = dict(_CFG["column_mapping"])
SEED = int(_CFG["seed"])


def load_dataset() -> pd.DataFrame:
    """Load the ACTIVE dataset as raw strings (no silent coercion).

    THE dataset for this series until further notice: euromonitor. Rename
    DATA_PATH + this loader when the active dataset changes; steps import
    `load_dataset`, never a hardcoded path. Columns are canonicalized
    project-wide via COLUMN_MAPPING (00_config.yaml).
    """
    df = pd.read_csv(DATA_PATH, dtype=str)
    return df.rename(columns=COLUMN_MAPPING)


def load_raw_export() -> pd.DataFrame:
    """The raw export WITHOUT column renames — the data-prep pipeline
    (01_data_prep) works in raw-export column names (gtin, sku_name_eng,
    attribute); the training/eval lane works in canonical ones."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} missing")
    return pd.read_csv(DATA_PATH, dtype=str)


def load_dataset_deduped() -> pd.DataFrame:
    """The DEDUPED dataset (06 tiered dedupe) — the matching-stage input.

    Step 03+ matching consumes this (one row per retailer-product after
    marketplace-listing collapse); the raw export remains the source of
    truth via load_dataset. Columns are already canonical (written by 06).
    """
    path = DATA_DIR / F["dataset_deduped"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run 06_dedupe.py first")
    return pd.read_csv(path, dtype=str)


# Backward-compatible alias for existing steps; prefer load_dataset going
# forward (single active-dataset entry point).
load_euromonitor = load_dataset


# ---------------------------------------------------------------------------
# Shared dataframe/regex helpers used by multiple steps.
# ---------------------------------------------------------------------------


def has_barcode(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: row has a non-empty barcode (GTIN)."""
    return df["barcode"].fillna("").astype(str).str.len() > 0


def multi_retailer_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: known barcode that appears under more than one retailer."""
    return has_barcode(df) & (
        df.groupby("barcode")["retailer"].transform("nunique") > 1
    )


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: non-null count, cardinality, numeric_like, stored dtype.

    numeric_like is the fraction of the first 5k non-null values that parse as
    numbers (loader reads dtype=str, so this shows the real content signal).
    Single source for 01's dtype table and 01b's column scatter.
    """
    rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        cardinality = int(df[col].dropna().nunique())
        numeric_like = 0.0
        if non_null:
            sample = df[col].dropna().head(5000)
            numeric_like = round(
                float(pd.to_numeric(sample, errors="coerce").notna().mean()), 4
            )
        rows.append(
            {
                "column": col,
                "stored_dtype": str(df[col].dtype),
                "non_null": non_null,
                "cardinality": cardinality,
                "numeric_like": numeric_like,
            }
        )
    return pd.DataFrame(rows)


def canonical_volume(series: pd.Series) -> pd.DataFrame:
    """Title series -> (canonical_volume_ml, canonical_volume_ambiguous) frame.

    Canonical volume comes from title ONLY (extract_volume_ml); the ambiguous
    flag marks bare oz/ounce (weight vs fluid). Single source for the
    extract-volume projection used by 01e/01f/01h/02/02b/02c.
    """
    vol = series.map(extract_volume_ml)
    return pd.DataFrame(
        {
            "canonical_volume_ml": vol.map(lambda t: t[0]),
            "canonical_volume_ambiguous": vol.map(lambda t: t[1]),
        }
    )


def barcode_agreement_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    *,
    sample: bool = False,
) -> pd.DataFrame:
    """Per-barcode volume-agreement table over multi-retailer barcode groups.

    `columns` is a list of (column_name, label) pairs. For each pair the result
    has `{label}_volumes` (sorted unique non-null values) and `{label}_agree`
    (True when the group's detected values collapse to one unique value, None
    when none are detected). Empty groups are NOT dropped here — callers
    dropna() the column they validate (the honest denominator: empty groups are
    excluded, never counted as trivially agreeing). `sample=True` adds
    sample_names/sample_retailers (first 5 unique).
    """
    multi = df[multi_retailer_mask(df)]

    def _agg(x: pd.DataFrame) -> pd.Series:
        row: dict = {"retailers": x["retailer"].nunique(), "skus": len(x)}
        for col, label in columns:
            vols = sorted(x[col].dropna().unique().tolist())
            row[f"{label}_volumes"] = vols
            row[f"{label}_agree"] = (len(vols) <= 1) if vols else None
        if sample:
            row["sample_names"] = x["title"].dropna().unique().tolist()[:5]
            row["sample_retailers"] = x["retailer"].unique().tolist()[:5]
        return pd.Series(row)

    return multi.groupby("barcode").apply(_agg, include_groups=False).reset_index()


# ===========================================================================
# SSOT: shared metrics, tokenizers, and split helpers (GATES_MAP.md owner).
# Prior homes of these definitions are noted for traceability; consumers
# import from here now — do NOT reintroduce local copies.
# ===========================================================================
import re as _re

import numpy as _np

# Tokenizer SSOT: one word scheme for the whole series. The three prior
# independent schemes (second01.TOKEN_RE, second02f.WORD_RE,
# second02g._TOK_RE) split "the same word" differently depending on which
# script processed it. This superset keeps 02f's diacritic coverage and
# 02g's intra-word apostrophe/hyphen gluing; second01's plain [a-z0-9]+ is
# a strict subset (see GATES_MAP.md).
TOKEN_RE = _re.compile(
    r"[a-zàâäáéèêëïîôöùûüçñåäöøæé0-9]+(?:[-\'][a-zàâäáéèêëïîôöùûüçñåäöøæé0-9]+)*",
    _re.IGNORECASE,
)


def title_tokens(text: str) -> list[str]:
    """Tokenize with the series' ONE scheme (see TOKEN_RE)."""
    return [w for w in TOKEN_RE.findall(str(text).lower()) if not w.isdigit()]


def jaccard(a: set, b: set) -> float:
    """Jaccard overlap of two token sets; 0.0 on empty input.

    Prior homes: second01e._jaccard, second03._jaccard.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_auc(pos_scores: "_np.ndarray", neg_scores: "_np.ndarray") -> float:
    """AUC of a pos/neg score split (rank-based, no threshold needed).

    Returns NaN when either side is empty (the caller decides how to report).
    Prior homes: second06._auc, second07._auc, second08._auc.
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    y = _np.r_[_np.ones(len(pos_scores)), _np.zeros(len(neg_scores))]
    s = _np.r_[pos_scores, neg_scores]
    return float(roc_auc_score(y, s))


def pair_similarity(emb: "_np.ndarray", pairs_idx: "_np.ndarray") -> "_np.ndarray":
    """Row-pair similarity scores: emb rows paired by (a, b) index columns.

    Prior home: second06._auc_pair. Works on any (N, d) array whose rows are
    L2-normalized (dot product == cosine).
    """
    a = emb[pairs_idx[:, 0]]
    b = emb[pairs_idx[:, 1]]
    return _np.sum(a * b, axis=1)


def bootstrap_auc_ci(
    y_true, scores, n: int = 1000, alpha: float = 0.05, seed: int | None = None
) -> tuple[float, float]:
    """Bootstrap percentile CI for AUC.

    Prior home: second03.bootstrap_auc_ci (was nested in main).
    """
    y_true = _np.asarray(y_true)
    scores = _np.asarray(scores)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    rng = _np.random.default_rng(seed if seed is not None else SEED)
    n_rows = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        aucs.append(pair_auc(scores[idx][pos_mask[idx]], scores[idx][neg_mask[idx]]))
    aucs = _np.array([a for a in aucs if _np.isfinite(a)])
    if len(aucs) == 0:
        return float("nan"), float("nan")
    lo = float(_np.percentile(aucs, 100 * (alpha / 2)))
    hi = float(_np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def kfold_groups(groups: list, k: int, seed: int | None = None) -> list[list[int]]:
    """Group-aware K-fold: same group never straddles folds; returns row-index
    lists per fold. The disjoint-group guard (G8 in GATES_MAP.md) is inherent.

    Prior homes: second06.make_folds, second10.make_folds (brand-grouped),
    07b/second06.kfold_barcodes (barcode-grouped).
    """
    uniq = sorted(set(groups))
    rng = _np.random.RandomState(seed if seed is not None else SEED)
    rng.shuffle(uniq)
    fold_of: dict = {g: i % k for i, g in enumerate(uniq)}
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, g in enumerate(groups):
        folds[fold_of[g]].append(i)
    return folds


def kfold_barcodes(df: pd.DataFrame, k: int, seed: int | None = None) -> list[set[str]]:
    """K barcode sets over multi-retailer barcodes, shuffled and split ~evenly.

    Splits on the barcode (entity) so no product's rows straddle a fold.
    Prior homes: 07b.kfold_barcodes, second06.kfold_barcodes — this is the
    exact strided-permutation implementation both used, so fold membership
    is unchanged for existing callers.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    bcs = _np.array(sorted(multi["barcode"].unique()))
    perm = _np.random.default_rng(seed if seed is not None else SEED).permutation(
        len(bcs)
    )
    return [set(bcs[perm[i::k]]) for i in range(k)]
