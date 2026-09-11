"""lib/common.py — the ONLY module that reads the config files.

Split-SSOT (2026-09-08; EDA removed 2026-09-10): the monolithic
00_config.yaml was broken into domain configs, each in its owning
directory:

  00_config.yaml        DataConfig      — paths/files/column_mapping/seed/models
  src/euromonitor/training/training.yaml   TrainingConfig  — the training lane's knobs

(The EDA dir and its eda.yaml were deleted 2026-09-10 — the lane is
training-only. The five TRAIN-consumed EDA keys — plots.dpi,
pairs.max_pos_per_group/n_neg/neg_oversample, strip_audit_sample —
migrated into src/euromonitor/training/training.yaml blocks of the same names.)

lib/common deep-merges them into ONE view (load_config()) and VALIDATES each
file against its pydantic model in lib/schemas at load — a bad value
crashes at import with a named field error, never mid-run. No script reads
YAML directly (unchanged SSOT doctrine), and every accessor below reads the
merged view, so consumers don't care which physical file a knob lives in.

New accessors:
  training_cfg()  the validated TrainingConfig (typed)
  data_cfg()      the validated DataConfig (typed)
  resolve_model(key)  registry key -> local bundle dir first, hub id fallback
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import os

import pandas as pd
import yaml

from euromonitor.core.schemas import DataConfig, TrainingConfig
from euromonitor.core.text import extract_volume_ml

# require_keys REMOVED (audit 2026-09-09): zero consumers — the pydantic
# validation at load (DataConfig/TrainingConfig) already fails
# loudly on missing keys, with named field errors. This helper duplicated
# that guarantee and was never called.


TRAIN_ROOT = Path(__file__).resolve().parents[3]  # repository root
CONFIG_PATH = TRAIN_ROOT / "00_config.yaml"
TRAINING_CONFIG_PATH = TRAIN_ROOT / "src" / "euromonitor" / "training" / "training.yaml"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config() -> dict:
    """Load + validate the split-SSOT and deep-merge into ONE view.

    Order: root data contract (00_config.yaml) is the base; src/euromonitor/training/training.yaml
    overlays it (the domain file wins on conflicts — a conflict
    is a config bug and the domain file is the authority for its keys).
    Every file is validated against its pydantic model BEFORE merging, so an
    invalid knob crashes here with the file + field named.
    """
    base = dict(_read_yaml(CONFIG_PATH))
    DataConfig.model_validate(base)  # root contract — fail before merge
    merged = dict(base)
    for path, model in (
        (TRAINING_CONFIG_PATH, TrainingConfig),
    ):
        if path.exists():
            overlay = dict(_read_yaml(path))
            model.model_validate(overlay)
            for key, block in overlay.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(block, dict)
                ):
                    merged[key] = {**merged[key], **block}
                else:
                    merged[key] = block
        else:
            raise SystemExit(f"config missing: {path}")
    return merged


# ── validated singletons (read once at import; the merge order above) ───────
_CFG = load_config()
_DATA_CFG = DataConfig.model_validate(_read_yaml(CONFIG_PATH))
_TRAIN_CFG = TrainingConfig.model_validate(_read_yaml(TRAINING_CONFIG_PATH))


def data_cfg() -> DataConfig:
    """The validated root data contract (00_config.yaml)."""
    return _DATA_CFG


def category_macros() -> dict[str, str]:
    """SSOT accessor for the category -> macro bucket taxonomy
    (00_config.yaml category_macros:).

    Was the inline MACRO_MAP dict in lib/text.py — domain data the owner
    may tune (the blocking layer's recall-first rollup of the dataset's
    strict categories), hence config-owned. Validated by DataConfig at
    load; consumers (blocking_audit / report_plots / hard_negatives)
    read THIS, never a module-level copy.
    """
    return dict(_CFG["category_macros"])


def training_cfg() -> TrainingConfig:
    """The validated training-lane config (src/euromonitor/training/training.yaml)."""
    return _TRAIN_CFG


def runtime(key: str, default: Any = None) -> Any:
    """SSOT accessor for every training runtime knob (src/euromonitor/training/training.yaml
    training: block — validated by TrainingSpec at load).

    ONE place to read batch sizes, seq length, eval cadence, dev fraction,
    band edges. Scripts call runtime("batch_size_cpu") etc. — the literal
    lives in src/euromonitor/training/training.yaml only, never in a script.

    The return is typed `Any` deliberately (audit round 2 F17): the
    training: block holds ints, floats, bools and strings; a narrower lie
    would be worse than the honest open type.

    NO FALLBACKS (owner Q27, audit 2026-09-09): the `default` escape hatch
    (silently returning a caller literal when the key was MISSING) is
    CLOSED. A default may only supply a value when the SSOT defines the
    key as an explicit YAML null (opt-in "use my default"). A truly
    missing key always raises — the literal can never diverge from SSOT.
    """
    tr = _CFG.get("training", {})
    if key in tr:
        val = tr[key]
        if val is not None:
            return val
        if default is not None:
            return default  # explicit null in YAML = caller's default, opt-in
        raise KeyError(
            f"training.{key} is explicitly null in src/euromonitor/training/training.yaml — "
            f"either set a value or have the caller pass a default"
        )
    raise KeyError(
        f"training.{key} missing from src/euromonitor/training/training.yaml — the SSOT must "
        f"define it (no per-script literals allowed)"
    )


# no-fallback SSOT scalars (owner directive Q27: NO FALLBACKS — a missing
# config key must crash, never silently default). Consumers import these
# instead of chaining .get(...) with inline literals.
SSOT_LOSS = runtime("loss")
SSOT_CONTRASTIVE_MARGIN = runtime("contrastive_margin")


def plot_dpi() -> int:
    """SSOT accessor for figure DPI (src/euromonitor/training/training.yaml plots.dpi).

    Every fig.savefig in the tree renders at THIS value — dpi=150 was
    inlined at 29 call sites across the plot scripts, a
    second declaration the config could not steer (audit, owner Q27).
    """
    return int(_CFG["plots"]["dpi"])


def strip_ladder_bands() -> list[tuple[float, float]]:
    """SSOT accessor for the strip-audit similarity ladder's Jaccard band
    edges (src/euromonitor/training/training.yaml audit.strip_ladder_bands).

    Was an inline literal list in src/euromonitor/training/strip_audit.py (~:186) — a second
    declaration the config could not steer (same doctrine as plot_dpi).
    Validated by AuditSpec at load (contiguous ascending cover of
    [0, 1+eps]); returns plain (lo, hi) tuples for the ladder loop.
    """
    return [(float(b[0]), float(b[1])) for b in _CFG["audit"]["strip_ladder_bands"]]


# ── HPO / rerank / sweep accessors (validated by lib.schemas at load) ───────
# These expose the hpo:, rerank:, sweep: blocks as PLAIN JSON-able data
# (lists of dicts / tuples) so callers never re-declare the sweep spaces.
def hpo_cfg() -> dict:
    """The hpo: block (grid/quick/tpe_space/n_trials/n_jobs) as plain data.

    grid/quick rows come back as dicts ({epochs, lr, warmup}); tpe_space as
    {knob: (lo, hi)}. Validated by HpoSpec at load — no re-validation here.
    """
    h = dict(_CFG["hpo"])
    h["grid"] = [dict(r) for r in h["grid"]]
    h["quick"] = [dict(r) for r in h["quick"]]
    return h


def rerank_cfg() -> dict:
    """The rerank: block (07e decision rule) as plain data."""
    return dict(_CFG["rerank"])


def sweep_cfg() -> dict:
    """The sweep: block (run_all ablation axes) as plain data."""
    return dict(_CFG["sweep"])


def band(name: str) -> tuple[float, float]:
    """SSOT accessor for cosine bands (src/euromonitor/training/training.yaml bands:).

    name: 'eval_mining' (train_one_config's eval-pool mining band) or
    'rerank_band' (cross-encoder). 'mining_band' was REMOVED (audit
    round 2 F21): it had zero callers — the live in-batch mining band is
    mining.band "lo-hi" (the string form, mined via _band_tuple).
    """
    b = _CFG.get("bands", {}).get(name)
    if not b or len(b) != 2:
        raise KeyError(f"bands.{name} missing/malformed in src/euromonitor/training/training.yaml")
    lo, hi = float(b[0]), float(b[1])
    if not lo < hi:
        raise ValueError(f"bands.{name}: lo must be < hi, got [{lo}, {hi}]")
    return lo, hi


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

# ── pinned census counts (2026-09-12; mirrors src/euromonitor/training/selftest.py's
# oracle_pinned_counts hard pin) ─────────────────────────────────────────────
# The transductive-census gate_results.csv is the universe BOTH
# src/euromonitor/training/labeled_pairs.py and the selftest oracle read. fallback pairs are
# excluded from the labeled set BY CONSTRUCTION (uncertain tier, would
# inject label noise) — this pin makes that exclusion LOUD and COUNTED
# instead of silent. Same audit lineage as the selftest pin (2026-09-08:
# the pack_qty >= 1 zero-guard fixed 26 gate decisions).
# NOT recomputed here: a pinned constant, updated alongside any
# intentional census drift (paired with the selftest oracle update).
PINNED_GATE_FALLBACK_PAIRS = 13_765


def set_determinism(seed: int) -> None:
    """Seed EVERYTHING the training lane touches, loudly and unconditionally.

    One call at each training entrypoint (before any model/data randomness)
    pins: random, numpy, torch (CPU + all CUDA devices) and the cudnn
    flags. The EXISTING SSOT seed is the only source — 00_config.yaml
    `seed:` (read once into lib.common.SEED); no second knob exists or is
    needed, so callers pass exactly that value.

    PYTHONHASHSEED: os.environ is set here for the CURRENT process, but
    hash() randomization is fixed only when the variable is present
    BEFORE the interpreter starts — setting it here cannot retro-fit an
    already-running CPython. It is still exported (harmless, and it makes
    child processes spawned after this call inherit the value). For full
    hash determinism the SAME seed must ALSO be exported at container
    start (Dockerfile `ENV PYTHONHASHSEED=42` — documented here, NOT
    changed by that task; align it with 00_config.yaml `seed:` when the
    Dockerfile is next touched).

    No new config key: cudnn.deterministic=True / benchmark=False are
    unconditional by design (the point of the helper is "always
    reproducible", not "reproducible when configured").
    """
    import random

    random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch is imported LOCALLY: lib/common is imported by data/plot/audit
    # scripts that never touch torch — a module-level import would make
    # every one of them pay torch's multi-second import + CUDA init.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(
        f"[determinism] seed={seed} cudnn.deterministic=True "
        f"(PYTHONHASHSEED note: effective only if set before interpreter "
        f"start)",
        flush=True,
    )

# ── model registry + resolution (shared src/euromonitor/training/run_all) ──────────────────────
MODELS = dict(_CFG["models"])
_MODEL_DIRS = [
    _path(_CFG["paths"]["models_dir"]),
    _path(_CFG["paths"]["models_dir_sibling"]),
]


def resolve_model(key_or_sub: str) -> str:
    """Registry key OR subdir name -> a model id the encoder can load.

    Local bundle dirs (paths.models_dir / models_dir_sibling) win first so
    offline GPU runs never hit the hub; the hub id is the fallback. A key
    not in the registry and not on disk resolves like the old run_all
    helper (sentence-transformers/<sub>, deberta special-cased). Every
    caller must come through here — never a hardcoded hub string.
    """
    sub = MODELS.get(key_or_sub, key_or_sub)
    for d in _MODEL_DIRS:
        cand = d / sub
        if cand.exists():
            return str(cand.resolve())
    if "deberta" in sub:
        return f"microsoft/{sub}"
    return f"sentence-transformers/{sub}"


# ── visibility-log writes (owner directive 2026-09-07) ─────────────────────
# Visibility dumps must survive run collisions: a --sample chain check used
# to overwrite a 3h full run's logs (same name, no run axis). Every dump
# writes BOTH:
#   results/logs/<name>.csv          — the "latest NON-SAMPLE run" copy
#                                      (sample runs never touch it, mirroring
#                                      the fold-metrics pointer discipline)
#   results/logs/<run_tag>/<name>.csv — the run's own copy (every run,
#                                      sample or not)
def write_visibility_log(
    df: pd.DataFrame, name: str, run_tag: str, sample: bool
) -> None:
    """Write a visibility dump under the run-tag dir + latest pointer."""
    logs = RESULTS / "logs"
    (logs / run_tag).mkdir(parents=True, exist_ok=True)
    df.to_csv(logs / run_tag / name, index=False)
    if not sample:
        df.to_csv(logs / name, index=False)


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
    (src/euromonitor/training/data_prep) works in raw-export column names (gtin, sku_name_eng,
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
        raise FileNotFoundError(f"{path} missing — run src/euromonitor/training/dedupe.py first")
    return pd.read_csv(path, dtype=str)


# load_euromonitor alias REMOVED (audit 2026-09-09): zero importers —
# every step already used load_dataset (verified by grep before removal).


# ---------------------------------------------------------------------------
# Shared dataframe/regex helpers used by multiple steps.
# ---------------------------------------------------------------------------


def has_barcode(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: row has a non-empty barcode (GTIN)."""
    return df["barcode"].fillna("").astype(str).str.len() > 0


# multi_retailer_mask REMOVED (audit round 2 F18, round 3): defined, never
# called — zero consumers (grep-verified). The same mask is derived inline
# where actually needed (kfold_barcodes, report_plots' country slice).


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



def kfold_barcodes(df: pd.DataFrame, k: int, seed: int | None = None) -> list[set[str]]:
    """K barcode sets over multi-retailer barcodes, shuffled and split ~evenly.

    Splits on the barcode (entity) so no product's rows straddle a fold.
    Prior homes: 07b.kfold_barcodes, second06.kfold_barcodes — this is the
    exact strided-permutation implementation both used, so fold membership
    is unchanged for existing callers.

    AUDIT 2026-09-09 (DATA DROP, now loud): only MULTI-RETAILER barcodes
    are dealt into folds. Single-retailer barcodes appear in NO fold, so
    under the CV path their positives are silently dropped from every
    test pool by pairs_in_set (measured on test data: a singleton
    barcode's pair vanishes from all k folds). This is a KNOWN, PRINTED
    limitation of the legacy CV mode — the production holdout lane
    (component_folds, src/euromonitor/training/folds.py) does NOT share it: it folds EVERY
    barcode including singletons. Callers must treat the returned folds
    as test-pool keysets, not as dataset coverage.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    multi = known[known.groupby("barcode")["retailer"].transform("nunique") > 1]
    bcs = _np.array(sorted(multi["barcode"].unique()))
    perm = _np.random.default_rng(seed if seed is not None else SEED).permutation(
        len(bcs)
    )
    folds = [set(bcs[perm[i::k]]) for i in range(k)]
    n_single = int(
        known.groupby("barcode")["retailer"].nunique().eq(1).sum()
    )
    print(
        f"[kfold_barcodes] {len(bcs):,} multi-retailer barcodes in {k} folds; "
        f"{n_single:,} single-retailer barcodes are in NO fold "
        f"(legacy CV semantics — use component_folds for full coverage)",
        flush=True,
    )
    return folds
