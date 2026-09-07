"""
lib/cache.py — caching layer for the pipeline.

Caches:
  - GTIN normalized columns (from raw dataset)
  - Canonical records per GTIN (after extraction and n-gram generation)
  - Gate results (candidate pairs with decisions)
  - Embeddings per model for canonical strings

Usage:
    from lib.cache import get_canonical, get_gate_results, get_embeddings_by_model
"""

import hashlib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
# DATASET_PATH + RESULTS via lib.common (SSOT): a CWD-relative literal broke
# portability (the bundle/Colab runs from a different working dir); lib.common
# resolves paths from the file layout, not the CWD. The import moved ABOVE
# the mkdir calls — the original had RESULTS undefined at this point.
from lib.common import DATA_PATH as DATASET_PATH
from lib.common import RESULTS, F

RESULTS.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS / F["cache_dir"]
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default models
MODEL_NAMES = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "microsoft/deberta-v3-base",  # adjust if exact name differs
]

# MODEL_NAME (singular): the cache's PRIMARY embedding model — SSOT for every
# second-series consumer (second06 MODEL, 07, 08, 11). The multilingual L12:
# the cross-country effect is the lane's core claim, so it is the default
# embedding base everywhere.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Dataset fingerprint
# ---------------------------------------------------------------------------
def _dataset_tag() -> str:
    """Hash of the raw dataset file for cache invalidation."""
    if DATASET_PATH.exists():
        h = hashlib.sha256()
        with open(DATASET_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1_000_000), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    else:
        # Fallback: hash a sample of the loaded dataframe
        df = pd.read_csv(DATASET_PATH, dtype=str, nrows=1000)
        payload = pd.util.hash_pandas_object(df, index=False).to_numpy()
        return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


DATASET_TAG = _dataset_tag()
print(f"[cache_v2] dataset tag: {DATASET_TAG}")

# ---------------------------------------------------------------------------
# Cache file paths
# ---------------------------------------------------------------------------
GTIN_CACHE = CACHE_DIR / f"gtin_norm_{DATASET_TAG}.csv"
CANON_CACHE = CACHE_DIR / f"canonical_records_{DATASET_TAG}.csv"
GATE_CACHE = CACHE_DIR / f"gate_results_{DATASET_TAG}.csv"


def embedding_cache_path(model_name: str) -> Path:
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return CACHE_DIR / f"embeddings_{safe_name}_{DATASET_TAG}.npz"


# ---------------------------------------------------------------------------
# 1. Load raw dataset with required columns
# ---------------------------------------------------------------------------
def load_raw_dataset() -> pd.DataFrame:
    """Load the raw dataset with the columns expected by the new pipeline."""
    df = pd.read_csv(DATASET_PATH, dtype=str)
    required = {"gtin", "brand", "sku_name_eng", "attribute"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df["gtin"] = df["gtin"].astype(str).str.strip()
    df["brand"] = df["brand"].astype(str).str.strip()
    df["sku_name_eng"] = df["sku_name_eng"].astype(str)
    df["attribute"] = df["attribute"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 2. Normalized GTIN columns (cached)
# ---------------------------------------------------------------------------
# ── GTIN normalization (inlined from repo second01 — self-contained) ──
def _canonicalize_gtin(x):
    """Canonicalize UPC-12 to GTIN-13 by adding a leading zero."""
    if pd.isna(x):
        return x

    x = str(x)

    if len(x) == 12:
        return "0" + x

    return x


def normalize_and_validate_gtin(series: pd.Series) -> pd.DataFrame:
    """
    Clean raw barcode strings and validate GTIN structure.

    Returns:
        gtin_clean
        gtin_structurally_valid
    """
    # Keep only digit sequences.
    cleaned = series.astype(str).str.extract(r"(\d+)", expand=False)

    # Empty strings -> missing.
    cleaned = cleaned.where(cleaned.ne(""))

    # Remove placeholder all-zero barcodes.
    cleaned = cleaned.where(~cleaned.str.fullmatch(r"0+", na=False))

    # UPC-12 -> GTIN-13 canonicalization.
    cleaned = cleaned.map(_canonicalize_gtin)

    # Keep plausible GTIN lengths.
    valid_length = cleaned.str.len().isin([8, 12, 13, 14])
    cleaned = cleaned.where(valid_length)

    # Validate checksum without using fillna downcasting.
    is_valid = pd.Series(False, index=series.index, dtype=bool)
    notna_mask = cleaned.notna()

    if notna_mask.any():
        is_valid.loc[notna_mask] = (
            cleaned.loc[notna_mask].map(is_valid_gtin_checksum).astype(bool)
        )

    return pd.DataFrame(
        {
            "gtin_clean": cleaned,
            "gtin_structurally_valid": is_valid,
        }
    )


# ---------------------------------------------------------------------------
# 2. Intra-group purity
# ---------------------------------------------------------------------------


def is_valid_gtin_checksum(gtin: str) -> bool:
    """Validate check digit for GTIN-8, GTIN-12, GTIN-13, GTIN-14."""
    if not gtin or not gtin.isdigit():
        return False

    if len(gtin) not in {8, 12, 13, 14}:
        return False

    digits = [int(d) for d in gtin]
    check_digit = digits[-1]
    body = digits[:-1]

    # GS1 spec (all four GTIN lengths use the same rule): from the right-most
    # BODY digit (check digit excluded), weights alternate 3, 1, 3, 1, ...;
    # check digit = (10 - weighted_sum mod 10) mod 10.
    body_reversed = body[::-1]
    total = sum(body_reversed[0::2]) * 3 + sum(body_reversed[1::2])
    expected = (10 - total % 10) % 10

    return check_digit == expected


def get_gtin_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Load or compute normalized GTIN columns (gtin_clean, gtin_structurally_valid)."""
    if GTIN_CACHE.exists():
        cached = pd.read_csv(GTIN_CACHE, dtype={"gtin_clean": str})
        cached["gtin_structurally_valid"] = cached["gtin_structurally_valid"].astype(
            bool
        )
        return cached

    result = normalize_and_validate_gtin(df["gtin"])
    result.to_csv(GTIN_CACHE, index=False)
    print(f"[cache_v2] saved GTIN columns -> {GTIN_CACHE.name}")
    return result


# ---------------------------------------------------------------------------
# 6. Cache status
# ---------------------------------------------------------------------------
def cache_status() -> None:
    print(f"\nDataset tag: {DATASET_TAG}")
    print(f"Cache dir  : {CACHE_DIR}\n")
    files = [
        ("GTIN columns", GTIN_CACHE),
        ("Canonical records", CANON_CACHE),
        ("Gate results", GATE_CACHE),
    ]
    for model in MODEL_NAMES:
        files.append((f"Embeddings ({model})", embedding_cache_path(model)))

    for name, path in files:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name:<25} {path.name:<50} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name:<25} MISSING")
    print()


if __name__ == "__main__":
    cache_status()
