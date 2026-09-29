"""record_linkage.py — reusable entity clustering for rows with no valid GTIN.

Barcodes assert identity only when GS1-checksum VALID (owner ruling, shared
with core.blocking). Rows with no usable barcode are invisible to the GTIN
identity model; this module links them into entity clusters using brand +
normalized-title blocking, so the same product listed by different retailers
can be grouped without a shared barcode.

Match rule (high-precision, cross-source only):
  - consider only rows without a valid GS1 barcode (missing, malformed, or
    checksum-invalid barcodes all have unknown identity),
  - block by normalized brand,
  - link two rows ONLY when they come from DIFFERENT retailers (a same-
    retailer near-duplicate is not trustworthy cross-source identity),
  - a cross-retailer pair is linked when their normalized titles are
    IDENTICAL (exact) or word-set Jaccard overlap >= threshold (fuzzy),
  - connected components of linked pairs become one entity cluster.

The fuzzy primitive is pipeline.jaccard_similarity (the SSOT word-set
Jaccard) — reused, never duplicated.

Reusable: the linkage logic (link_barcode_less) is importable so other
lanes/scripts can consume the same rule without re-implementing it.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

from core.gtin import barcode_validity
from pipeline import jaccard_similarity, normalize_text

DEFAULT_JACCARD_THRESHOLD = 0.7

# Pack-multiplicity tokens that may vary across listings of the SAME product
# (12's / 24's, 6x, x6, 6-pack, 24 count, pack of 6) but must NOT be treated
# as a product-identity difference. Stripped BEFORE similarity so same-product
# listings in different pack sizes still link, while a different flavor keeps
# its discriminating word (mocha vs latte) and does NOT link. Bare "of N" is
# handled conservatively: stripped only when N is a plausible pack count and
# NOT followed by a volume unit, so "of 250" (a volume) is never removed.
_PACK_MULTI_TOKENS = (
    r"\bone\s+(?=\d+\s*[- ]pack\b)",
    r"\b\d+\s+s\b",          # 12's -> "12 s" after normalize_text
    r"\b\d+\s*x\b",           # 6x, 12x
    r"\bx\s*\d+\b",           # x6, x12
    r"\b\d+\s*[- ]pack\b",    # 6-pack, 6 pack
    r"\b\d+\s*(?:ct|count|cans?|bottles?|units?|pcs?|pieces?|cases?)\b",
    # 24 ct/count/cans/bottles/units, 12 pcs, 2 cases
    r"\bpack\s+of\s+\d+\b",   # pack of 6
    r"\bcount\s+of\s+\d+\b",  # count of 6
    r"\b(?:case|carton|box|bundle|set|qty|quantity)\s+of\s+\d+\b",
    r"\bper\s+pack\b",        # per pack
)
_PACK_MULTI_RE = re.compile("|".join(_PACK_MULTI_TOKENS), re.IGNORECASE)
# Bare "of N" — only when N is a plausible pack count (2..72) and the
# following token is not a volume unit.
_OF_N_RE = re.compile(r"\bof\s+(\d{1,2})\b", re.IGNORECASE)
_VOLUME_UNIT_RE = re.compile(r"^(ml|l|cl|oz|lt|ltr|litre|liter|dl|gal)\b", re.IGNORECASE)


def strip_pack_multiplicity(text: str) -> str:
    """Remove pack-quantity tokens from a normalized title.

    Returns the text with unambiguous pack-count markers removed so the
    remaining string captures product identity (brand, flavor, size) without
    pack-size noise. Used by link_barcode_less before similarity so pack
    variants of the same product link while distinct flavors stay separate.
    """
    out = _PACK_MULTI_RE.sub(" ", text)

    def _strip_of_n(match: re.Match) -> str:
        n = int(match.group(1))
        tail = text[match.end():].lstrip()
        if not (2 <= n <= 72):
            return match.group(0)
        # A following volume unit means "of 250" is a size, not a pack count.
        if _VOLUME_UNIT_RE.match(tail):
            return match.group(0)
        return " "

    out = _OF_N_RE.sub(_strip_of_n, out)
    return re.sub(r"\s+", " ", out).strip()


class _UnionFind:
    """Path-compressed union-find over row indices."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self._parent.setdefault(x, x)
        while self._parent[parent] != parent:
            self._parent[parent] = self._parent[self._parent[parent]]
            parent = self._parent[parent]
        return parent

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _norm(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").map(normalize_text)


def link_barcode_less(
    df: pd.DataFrame,
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    title_col: str = "title",
    brand_col: str = "brand",
    barcode_col: str = "barcode",
    retailer_col: str = "retailer",
) -> tuple[dict[int, str], dict]:
    """Return (row_index -> cluster_id, census) for rows without valid GTINs.

    The returned map is keyed by the ORIGINAL row index so callers can join
    back onto the source frame. Rows with a checksum-valid barcode are
    excluded; missing, malformed, and checksum-invalid barcode rows are
    clustered. Every such row lands in a cluster (single rows form
    singleton clusters).
    """
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")
    missing = [
        c for c in (title_col, brand_col, barcode_col, retailer_col) if c not in df
    ]
    if missing:
        raise KeyError(f"missing required linkage columns: {', '.join(missing)}")
    if not df.index.is_unique:
        raise ValueError("link_barcode_less requires a unique dataframe index")

    barcodes = df[barcode_col].fillna("").astype(str).str.strip()
    valid_barcode = barcode_validity(barcodes)
    no_bc = df[~valid_barcode].copy()
    if no_bc.empty:
        return {}, _empty_census(0, jaccard_threshold)
    normalized_titles = no_bc[title_col].where(no_bc[title_col].notna(), "").map(
        normalize_text
    )
    no_bc["_nts"] = normalized_titles.map(strip_pack_multiplicity)
    no_bc["_nb"] = _norm(no_bc[brand_col])
    no_bc["_nr"] = _norm(no_bc[retailer_col])

    uf = _UnionFind()
    for idx in no_bc.index:
        uf.find(idx)
    exact_pairs = 0
    fuzzy_pairs = 0
    checked_pairs = 0

    candidates = no_bc[
        no_bc["_nb"].ne("") & no_bc["_nts"].ne("") & no_bc["_nr"].ne("")
    ]
    for _nb, g in candidates.groupby("_nb"):
        if len(g) < 2:
            continue
        titles = g["_nts"].to_dict()
        retailers = g["_nr"].to_dict()
        indices = list(g.index)
        for i in range(len(indices)):
            a = indices[i]
            uf.find(a)
            for j in range(i + 1, len(indices)):
                b = indices[j]
                uf.find(b)
                if retailers[a] == retailers[b]:
                    continue  # same-retailer near-dup: not cross-source identity
                checked_pairs += 1
                ta, tb = titles[a], titles[b]
                if ta == tb:
                    exact_pairs += 1
                    uf.union(a, b)
                elif jaccard_similarity(ta, tb) >= jaccard_threshold:
                    fuzzy_pairs += 1
                    uf.union(a, b)

    # Stable cluster ids in deterministic (first-seen) order.
    cluster_id: dict[int, str] = {}
    root_counter: dict[int, int] = {}
    for idx in no_bc.index:
        root = uf.find(idx)
        cid = root_counter.setdefault(root, len(root_counter) + 1)
        cluster_id[idx] = f"bl-{cid:06d}"

    clusters: dict[str, list[int]] = defaultdict(list)
    for idx, cid in cluster_id.items():
        clusters[cid].append(idx)
    sizes = Counter(len(v) for v in clusters.values())
    census = {
        "barcode_less_rows": int(len(no_bc)),
        "clustered_rows": int(len(cluster_id)),
        "unclustered_single_rows": int(
            sum(1 for v in clusters.values() if len(v) == 1)
        ),
        "num_clusters": int(len(clusters)),
        "num_multirow_clusters": int(
            sum(1 for v in clusters.values() if len(v) > 1)
        ),
        "rows_in_multirow_clusters": int(
            sum(len(v) for v in clusters.values() if len(v) > 1)
        ),
        "pair_checks": int(checked_pairs),
        "exact_title_pairs_linked": int(exact_pairs),
        "fuzzy_title_pairs_linked": int(fuzzy_pairs),
        "jaccard_threshold": float(jaccard_threshold),
        "cluster_size_distribution": {str(k): int(v) for k, v in sorted(sizes.items())},
    }
    return cluster_id, census


def _empty_census(n_rows: int, jaccard_threshold: float) -> dict:
    return {
        "barcode_less_rows": int(n_rows),
        "clustered_rows": 0,
        "unclustered_single_rows": 0,
        "num_clusters": 0,
        "num_multirow_clusters": 0,
        "rows_in_multirow_clusters": 0,
        "pair_checks": 0,
        "exact_title_pairs_linked": 0,
        "fuzzy_title_pairs_linked": 0,
        "jaccard_threshold": float(jaccard_threshold),
        "cluster_size_distribution": {},
    }
