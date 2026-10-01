"""record_linkage.py — reusable entity clustering for rows with no valid GTIN.

Barcodes assert identity only when GS1-checksum VALID (owner ruling, shared
with core.blocking). Rows with no usable barcode are invisible to the GTIN
identity model; this module links them into entity clusters using brand +
normalized-title blocking, so the same product listed by different retailers
can be grouped without a shared barcode.

Match rule (high-precision, cross-source only):
  - consider only rows without a valid GS1 barcode (missing, malformed, or
    checksum-invalid barcodes all have unknown identity),
  - block by normalized brand (core.text.normalize_retailer handles the
    retailer identity; brand blocks use the same case/accent fold),
  - link two rows ONLY when they come from DIFFERENT retailers (a same-
    retailer near-duplicate is not trustworthy cross-source identity),
  - a cross-retailer pair is linked when their pack-stripped normalized
    titles are IDENTICAL (exact) or IDF-WEIGHTED word-set Jaccard >=
    threshold (fuzzy): block-level token IDF stops generic tokens
    ("water", "ml") from outvoting the rare discriminating token, so
    mocha and latte no longer merge on shared brand/size words,
  - merging is AVERAGE-LINKAGE agglomerative, not raw union-find closure:
    a link only merges two clusters when the MEAN weighted similarity over
    every cross-set pair holds the threshold, so one fuzzy edge cannot
    chain distinct flavors through shared tokens,
  - per-item UNIQUENESS (corpus token-IDF mean) is computed and reported as
    a measured feature; it gates nothing until it demonstrates separation
    on GTIN-verified duplicate pairs.

The fuzzy weighting is block-local IDF over the SSOT word-set Jaccard
(pipeline.jaccard_similarity's set logic, reweighted) — no duplicate
similarity definition lives elsewhere.

Reusable: the linkage logic (link_barcode_less) is importable so other
lanes/scripts can consume the same rule without re-implementing it.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import pandas as pd

from core.gtin import barcode_validity
from core.text import normalize_retailer
from pipeline import normalize_text

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
    r"\bpack_qty_\d+\b",      # structured pack token from the model-input text
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


def finalized_texts(
    df: pd.DataFrame, *, include_attributes: bool = False
) -> pd.Series:
    """Finalized model-input text per row via the core.model_input SSOT.

    The linkage compares the SAME normalized space the gate and the encoder
    see (build_sku_texts: lowercase tokens, fixed [brand][title][attributes]
    order, schema words dropped, word-once reduction, concept folds) — not a
    private re-normalization of raw titles that would fork text semantics
    across lanes.

    Attributes are blanked by default (mirroring the pipeline's own
    title_only payload variant): the canonical IDENTITY surface is the title
    — retailer-specific attribute/nutrition text adds shared tokens that
    shrink the discriminating flavor token's margin (measured: Obsesso
    black-vs-latte 0.711 with attributes vs ~0.59 title-only) and the
    attribute channel differs per retailer for the same product, which broke
    exact cross-retailer matches (2,376 -> 66). structured_features.enabled
    is read from the config SSOT.
    """
    from core.common import training_cfg
    from core.model_input import build_sku_texts

    enabled = bool(training_cfg().training.structured_features.enabled)
    frame = df
    if not include_attributes:
        frame = df.copy()
        for column in ("attributes", "attr", "description", "description_short_eng"):
            if column in frame.columns:
                frame[column] = ""
    texts, _infos = build_sku_texts(frame, structured_enabled=enabled)
    return pd.Series(texts, index=df.index)


def corpus_idf_from_finalized(finalized: pd.Series) -> tuple[dict[str, float], float]:
    """Token IDF over an ALREADY-finalized text series (pack tokens stripped).

    The rarity substrate for item uniqueness: idf(t) = log(1 + N/df(t)).
    Returns (idf_map, unseen_token_weight) — the unseen weight is the
    maximum (log(1+N)), applied to tokens absent from the corpus.
    """
    doc_freq: Counter[str] = Counter()
    for text in finalized:
        doc_freq.update(strip_pack_multiplicity(text).split())
    n_docs = max(len(finalized), 1)
    idf = {tok: math.log(1.0 + n_docs / df_t) for tok, df_t in doc_freq.items()}
    return idf, math.log(1.0 + n_docs)


def corpus_idf(
    df: pd.DataFrame, title_col: str = "title", *, finalized: pd.Series | None = None
) -> tuple[dict[str, float], float]:
    """Corpus-level token IDF over every row's finalized text.

    Pass ``finalized`` (a precomputed _finalized_texts Series) to avoid
    rebuilding the per-row model texts a second time within one run.
    """
    if finalized is None:
        finalized = finalized_texts(df)
    return corpus_idf_from_finalized(finalized)


def item_uniqueness_from_tokens(
    tokens: list[str], idf: dict[str, float], unseen_weight: float
) -> float:
    """Mean IDF of one item's tokens (0.0 for an empty title)."""
    if not tokens:
        return 0.0
    return sum(idf.get(t, unseen_weight) for t in tokens) / len(tokens)


def item_uniqueness(
    df: pd.DataFrame, title_col: str = "title", *, finalized: pd.Series | None = None
) -> dict[int, float]:
    """Per-row uniqueness score (corpus token-IDF mean) for EVERY row.

    Low uniqueness = generic listing where fuzzy text evidence is weak;
    high = distinctive listing. A measured feature and diagnostic — never
    a hard identity rule (two distinct rare products can collide; a
    generic duplicate across retailers is still a true match).
    Pass ``finalized`` (precomputed finalized_texts) to avoid rebuilding.
    """
    if finalized is None:
        finalized = finalized_texts(df)
    idf, unseen = corpus_idf_from_finalized(finalized)
    stripped = finalized.map(strip_pack_multiplicity)
    return {
        idx: item_uniqueness_from_tokens(text.split(), idf, unseen)
        for idx, text in stripped.items()
    }


def item_uniqueness_from_finalized(
    finalized: pd.Series, idf: dict[str, float], unseen_weight: float
) -> dict[int, float]:
    """Per-row uniqueness from an ALREADY-finalized text Series (no rebuild)."""
    return {
        idx: item_uniqueness_from_tokens(
            strip_pack_multiplicity(text).split(), idf, unseen_weight
        )
        for idx, text in finalized.items()
    }


def _norm(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").map(normalize_text)


def _brand_family_key(series: pd.Series) -> pd.Series:
    """Brand cell -> the alias-family block key (empty for a blank cell).

    Blocking keys must be IDENTICAL for alias siblings, and the folded token
    sets are not: "A SHOC" folds to {a, shoc} while "Accelerator" folds to
    {accelerator, shoc} — the fold gives the two sides a SHARED token but not
    an EQUAL set, so groupby would still put them in two blocks. The block
    key therefore collapses a brand to the alias-family canonicals its fold
    reaches (config/vocabulary.json "brand_aliases" targets, e.g. `shoc`);
    a brand no family reaches keeps its full folded spelling, which is the
    pre-alias block key for every alias-free cell, so plain ("Goat Fuel" vs
    "Goa Fuel") blocking is unchanged.

    ONE SSOT call (`core.product_identity.normalize_brand`) plus a blank
    guard, no new normalization: rows with no brand at all must still stay
    OUT of every block (the candidates filter empties on `_nb`, and an empty
    fold must never collapse them into one shared block), so the pre-fold
    block key keeps its blank sentinel.

    Buffered by the veto-asymmetry doctrine: `normalize_brand` only ADDS the
    alias target token, and the family keys are the config-reviewed
    canonicals whose allowed territory the rarity audit measured
    (scripts/seed_brand_aliases.py). A wider block is a RECALL gain — more
    candidate pairs — and the link rule (different retailer + IDF-weighted
    Jaccard + average-linkage) still has to pass; blocking asserts nothing.

    Imported INSIDE the closure: `core.product_identity` imports this
    module's ``strip_pack_multiplicity`` at module level, so a module-level
    import here would make the two modules mutually unimportable no matter
    which module the entry import reaches first. `product_identity`'s own
    import bar (``_barcode_facts``) is a deferral for exactly this shape.
    """
    from core.product_identity import brand_aliases, normalize_brand

    family_tokens = frozenset(brand_aliases().values())

    def _key(value: object) -> str:
        folded = normalize_brand(value)
        if not folded:
            return ""
        families = folded & family_tokens
        return " ".join(sorted(families)) if families else " ".join(sorted(folded))

    return series.where(series.notna(), "").map(_key)


def _norm_retailer(series: pd.Series) -> pd.Series:
    """Retailer identity via the normalize_retailer SSOT (accent-fold, not
    normalize_text's accent-DELETION which splits Voilà -> 'voil')."""
    return series.where(series.notna(), "").map(normalize_retailer)


def link_barcode_less(
    df: pd.DataFrame,
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    title_col: str = "title",
    brand_col: str = "brand",
    barcode_col: str = "barcode",
    retailer_col: str = "retailer",
    finalized: pd.Series | None = None,
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
    # Finalized model-input text (SSOT core.model_input.build_sku_texts) with
    # pack tokens stripped: pack size varies across listings of the SAME
    # product and must never be a similarity signal here.
    if finalized is None:
        finalized = finalized_texts(df)
    no_bc["_nts"] = finalized.loc[no_bc.index].map(strip_pack_multiplicity)
    # Brand blocks run on the product_identity SSOT alias-family key (see
    # `_brand_block_key`), so alias siblings ("A SHOC"/"Accelerator") land in
    # ONE block and stay reachable for candidate generation.
    no_bc["_nb"] = _brand_family_key(no_bc[brand_col])
    # The MATCH RULE needs the folded brand too (veto-asymmetry consumes the
    # same family key): the finalized text carries each spelling's own RAW
    # [brand][title] surface, so an alias pair whose block key is identical
    # still compares as "a shoc ..." vs "accelerator ..." — the fold ADDS the
    # shared family token but never equalizes the spellings, and the
    # exact-title link inside the family block could never fire. The rule's
    # comparison text therefore reads the brand through the SAME family key
    # the block used ("shoc"), and only inside that one block can the family
    # spellings differ — alias-free brands keep their byte-identical
    # normalized surface (the block key is the full folded spelling), so
    # cross-retailer exact titles and flavors-separate verdicts are
    # unchanged. Blocking asserted nothing; the rule still verifies.
    match_frame = no_bc.copy()
    match_frame[brand_col] = no_bc["_nb"]
    no_bc["_nts"] = finalized_texts(match_frame).loc[no_bc.index].map(
        strip_pack_multiplicity
    )
    no_bc["_nr"] = _norm_retailer(no_bc[retailer_col])
    # A finalized text of a title-less row is just the brand string — brand
    # alone asserts NO product identity, so such rows stay singletons even
    # when their (brand-only) texts match exactly.
    no_bc["_has_title"] = (
        no_bc[title_col].where(no_bc[title_col].notna(), "").astype(str).str.strip().ne("")
    )

    # Corpus-level item uniqueness (rarity of the row's tokens against the
    # WHOLE catalog, pack tokens stripped for consistency): mean IDF of the
    # row's tokens. Low uniqueness = generic listing ("pepsi 500 ml") where
    # fuzzy text evidence is weak; high = distinctive listing. A FEATURE and
    # a reported diagnostic — never a hard rule (two distinct rare products
    # can collide; a generic duplicate is still a true match).
    idf_corpus, unseen_weight = corpus_idf(df, title_col=title_col, finalized=finalized)
    uniqueness = {
        idx: item_uniqueness_from_tokens(text.split(), idf_corpus, unseen_weight)
        for idx, text in no_bc["_nts"].items()
    }

    uf = _UnionFind()
    for idx in no_bc.index:
        uf.find(idx)
    exact_pairs = 0
    fuzzy_pairs = 0
    checked_pairs = 0
    rejected_merges = 0

    candidates = no_bc[
        no_bc["_nb"].ne("")
        & no_bc["_nts"].ne("")
        & no_bc["_nr"].ne("")
        & no_bc["_has_title"]
    ]
    for _nb, g in candidates.groupby("_nb"):
        if len(g) < 2:
            continue
        # Block-level IDF: generic tokens ("water", "ml") dominate word-set
        # Jaccard inside a brand block and merge distinct flavors that share
        # brand/size tokens. Weighting by inverse block frequency makes the
        # rare discriminating token (mocha vs latte) decide the match.
        block_tokens = [t for text in g["_nts"] for t in text.split()]
        block_df = Counter(block_tokens)
        n_block = max(len(g), 2)
        idf_block = {
            tok: math.log(1.0 + n_block / df_t) for tok, df_t in block_df.items()
        }
        tokens = {idx: frozenset(text.split()) for idx, text in g["_nts"].items()}
        titles = g["_nts"].to_dict()
        retailers = g["_nr"].to_dict()
        indices = list(g.index)

        def _wj(a: int, b: int) -> float:
            sa, sb = tokens[a], tokens[b]
            if not sa or not sb:
                return 0.0
            inter = sum(idf_block.get(t, 0.0) for t in sa & sb)
            union = sum(idf_block.get(t, 0.0) for t in sa | sb)
            return inter / union if union else 0.0

        links: list[tuple[float, int, int]] = []
        for i in range(len(indices)):
            a = indices[i]
            for j in range(i + 1, len(indices)):
                b = indices[j]
                if retailers[a] == retailers[b]:
                    continue  # same-retailer near-dup: not cross-source identity
                checked_pairs += 1
                if titles[a] == titles[b]:
                    exact_pairs += 1
                    links.append((1.0, a, b))
                    continue
                sim = _wj(a, b)
                if sim >= jaccard_threshold:
                    fuzzy_pairs += 1
                    links.append((sim, a, b))

        # Average-linkage agglomeration over DISTINCT titles (desc
        # similarity): a cross-retailer link merges two clusters ONLY when
        # the MEAN weighted similarity over every pair of DISTINCT
        # finalized texts inside the merged cluster holds the threshold.
        # Raw union-find merged on single edges, so one fuzzy link chained
        # distinct flavors (mocha -> shared brand tokens -> latte) into one
        # cluster. Judging distinct titles (one representative per text)
        # stops same-title duplicates from diluting the mean with their
        # 1.0 pairs and masking a flavor merge behind them.
        members: dict[int, list[int]] = {i: [i] for i in indices}

        def _pair_sim(x: int, y: int) -> float:
            return 1.0 if titles[x] == titles[y] else _wj(x, y)

        for sim, a, b in sorted(links, key=lambda x: -x[0]):
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            merged = members[ra] + members[rb]
            reps: dict[str, int] = {}
            for x in merged:
                reps.setdefault(titles[x], x)
            rep_rows = list(reps.values())
            if len(rep_rows) > 1:
                total = 0.0
                for i in range(len(rep_rows)):
                    for j in range(i + 1, len(rep_rows)):
                        total += _pair_sim(rep_rows[i], rep_rows[j])
                internal_after = total / (len(rep_rows) * (len(rep_rows) - 1) // 2)
            else:
                internal_after = 1.0
            if internal_after < jaccard_threshold:
                rejected_merges += 1
                continue
            uf.union(ra, rb)
            root = uf.find(ra)
            other = rb if root == ra else ra
            members[root] = merged
            members.pop(other, None)

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

    # Cluster-coherence review flag: recompute each multirow cluster's
    # distinct-title mean the same way the agglomeration judged it, and flag
    # clusters sitting within a small margin of the threshold — the zone
    # where a same-brand different-flavor mix can ride in on weak block IDF
    # (measured residual: Wandering Bear cross-flavor pairs 0.71-0.78). The
    # linkage is candidate generation; flagged clusters route to review, the
    # gate/verifier keeps the final say.
    flagged = 0
    for cid, rows in clusters.items():
        if len(rows) <= 1:
            continue
        texts = [no_bc.at[i, "_nts"] for i in rows]
        reps = list(dict.fromkeys(texts))
        if len(reps) < 2:
            continue
        block_df = Counter(t for x in no_bc["_nts"] for t in x.split())
        n_block = max(len(no_bc), 2)
        idf_b = {t: math.log(1.0 + n_block / d) for t, d in block_df.items()}
        tot = 0.0
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                sa, sb = set(reps[i].split()), set(reps[j].split())
                if not sa or not sb:
                    continue
                inter = sum(idf_b.get(t, 0.0) for t in sa & sb)
                union = sum(idf_b.get(t, 0.0) for t in sa | sb)
                tot += (inter / union) if union else 0.0
        mean_sim = tot / (len(reps) * (len(reps) - 1) // 2)
        if mean_sim < jaccard_threshold + 0.05:
            flagged += 1

    linked_uniq = [
        uniqueness[i]
        for cid, rows in clusters.items()
        if len(rows) > 1
        for i in rows
    ]
    single_uniq = [
        uniqueness[idx]
        for cid, rows in clusters.items()
        if len(rows) == 1
        for i in rows
    ]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

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
        "idf_weighted": True,
        "idf_weighted": True,
        "avg_linkage_rejected_merges": int(rejected_merges),
        "low_coherence_clusters_flagged": int(flagged),
        "uniqueness": {
            "mean_linked_rows": round(_mean(linked_uniq), 4),
            "mean_singletons": round(_mean(single_uniq), 4),
            "note": "corpus token-IDF mean per item; measured feature, no gate",
        },
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
        "idf_weighted": True,
        "avg_linkage_rejected_merges": 0,
        "low_coherence_clusters_flagged": 0,
        "uniqueness": {},
        "cluster_size_distribution": {},
    }
