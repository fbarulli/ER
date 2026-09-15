"""Are brands failing to match, and why not? TF-IDF + fuzzy string evidence.

The brand-separation metric says brand does not separate the pair population.
This module answers the NEXT question mechanically: when two brand strings are
compared, what KIND of difference is it — and which kinds are defects?

Method (CPU only, no model, no training):

* **TF-IDF** — brand strings vectorised with character n-grams
  (``char_wb``), which is the right space for short strings where whole-token
  overlap is too sparse. Cosine between two brand vectors is scale- and
  order-insensitive.
* **Fuzzy matching** — ``difflib`` similarity ratio plus the standard
  token-sort and token-set ratios, and a Levenshtein edit distance. All
  stdlib: no new dependency is required, so none is added (``rapidfuzz`` is
  present in the venv but is NOT declared in ``requirements.txt``; relying on
  it would make a fresh install fail).

Normalisation reuses the repo's ONE accent-folding implementation,
``core.critical_attributes.normalized_attribute_text`` — the same function
``core.model_input`` applies to the encoder text, so this analysis and the
model agree on what "the same brand" means.

Every threshold is config (``evaluation.brand_analysis``), and every reported
pair carries the support of the brand it belongs to, so a brand seen twice is
never presented as a defect.
"""
from __future__ import annotations

import argparse
import difflib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.common import F, ensure_parent, load_config
from core.critical_attributes import normalized_attribute_text
from core.schemas import (
    BRAND_PAIR_COLUMNS,
    BrandAnalysisSpec,
    BrandPairRow,
)

# Classification vocabulary. Evidence-driven: these are the distinctions the
# real data actually exhibits, not a pre-imposed taxonomy.
CLASS_EXACT = "exact_match"
CLASS_SURFACE_VARIANT = "surface_variant"
CLASS_MISSING = "missing_brand"
CLASS_TITLE_ONLY = "brand_only_in_title"
CLASS_DIFFERENT = "different_brand"


def brand_spec() -> BrandAnalysisSpec:
    """The validated brand-analysis settings (config SSOT)."""
    return BrandAnalysisSpec.model_validate(load_config()["evaluation"]["brand_analysis"])


def _levenshtein(left: str, right: str) -> int:
    """Edit distance. Stdlib-only: no declared fuzzy dependency exists."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (lch != rch),
                )
            )
        previous = current
    return previous[-1]


def _token_sorted(text: str) -> str:
    return " ".join(sorted(text.split()))


def _token_set(text: str) -> str:
    return " ".join(sorted(set(text.split())))


def _ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def _strip_corporate_suffixes(text: str, suffixes: tuple[str, ...]) -> str:
    """Remove trailing legal-form words so ``Quelle GmbH`` == ``Quelle``."""
    tokens = text.split()
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    return " ".join(tokens)


def brand_features(
    left: str, right: str, *, spec: BrandAnalysisSpec
) -> dict[str, float]:
    """Fuzzy features for one brand pair, on suffix-stripped normal forms."""
    suffixes = tuple(spec.corporate_suffixes)
    a = _strip_corporate_suffixes(normalized_attribute_text(left), suffixes)
    b = _strip_corporate_suffixes(normalized_attribute_text(right), suffixes)
    return {
        "ratio": _ratio(a, b),
        "token_sort_ratio": _ratio(_token_sorted(a), _token_sorted(b)),
        "token_set_ratio": _ratio(_token_set(a), _token_set(b)),
        "edit_distance": float(_levenshtein(a, b)),
        "normalized_equal": float(a == b and bool(a)),
    }


def _tfidf_cosine(brands: list[str], spec: BrandAnalysisSpec) -> dict[tuple[str, str], float]:
    """Cosine between every pair of the given brand strings, in TF-IDF space."""
    vectorizer = TfidfVectorizer(
        analyzer=spec.tfidf_analyzer,
        ngram_range=(spec.tfidf_ngram_min, spec.tfidf_ngram_max),
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(brands)
    similarities = cosine_similarity(matrix)
    return {
        (left, right): float(similarities[i, j])
        for i, left in enumerate(brands)
        for j, right in enumerate(brands)
    }


def classify_pair(
    left: str, right: str, features: dict[str, float], *, spec: BrandAnalysisSpec
) -> str:
    """Name the KIND of difference between two brand strings."""
    left_norm = normalized_attribute_text(left)
    right_norm = normalized_attribute_text(right)
    if not left_norm and not right_norm:
        return CLASS_MISSING
    if not left_norm or not right_norm:
        return CLASS_MISSING
    if features["normalized_equal"]:
        return CLASS_EXACT
    if features["ratio"] >= spec.surface_variant_min_ratio:
        return CLASS_SURFACE_VARIANT
    if features["ratio"] <= spec.different_brand_max_ratio:
        return CLASS_DIFFERENT
    # The band between the two thresholds is deliberately NOT forced into a
    # class: the evidence does not decide, so it is reported as its own value.
    return spec.indeterminate_class


def analyse_brands(
    comparisons: pd.DataFrame, *, spec: BrandAnalysisSpec
) -> pd.DataFrame:
    """Classify every brand-vs-brand comparison, with its support counts.

    ``comparisons`` carries ``brand_left``, ``brand_right``, ``title`` (the
    source title, used for the title-only class), ``support_positive`` and
    ``support_negative`` — the number of labelled pairs each brand participates
    in, so a two-row brand can never be reported as a defect.
    """
    required = {
        "brand_left", "brand_right", "title", "support_positive", "support_negative",
    }
    absent = required - set(comparisons.columns)
    if absent:
        raise ValueError(f"brand comparisons missing columns {sorted(absent)}")

    brands = sorted(
        {
            str(value)
            for value in pd.concat(
                [comparisons["brand_left"], comparisons["brand_right"]]
            )
        }
    )
    cosines = _tfidf_cosine(brands, spec)

    rows: list[dict[str, object]] = []
    for _, row in comparisons.iterrows():
        left, right = str(row["brand_left"]), str(row["brand_right"])
        features = brand_features(left, right, spec=spec)
        cosine = cosines.get((left, right), cosines.get((right, left), 0.0))
        kind = classify_pair(left, right, features, spec=spec)
        if kind == CLASS_MISSING:
            title_norm = normalized_attribute_text(row["title"])
            present = normalized_attribute_text(left) or normalized_attribute_text(right)
            if present and present in title_norm:
                kind = CLASS_TITLE_ONLY
        rows.append(
            BrandPairRow(
                brand_left=left,
                brand_right=right,
                classification=kind,
                similarity_ratio=features["ratio"],
                token_sort_ratio=features["token_sort_ratio"],
                token_set_ratio=features["token_set_ratio"],
                edit_distance=int(features["edit_distance"]),
                tfidf_cosine=cosine,
                n_positive=int(row["support_positive"]),
                n_negative=int(row["support_negative"]),
                reportable=bool(
                    min(int(row["support_positive"]), int(row["support_negative"]))
                    >= spec.min_pair_support
                ),
            ).model_dump()
        )
    return pd.DataFrame(rows, columns=list(BRAND_PAIR_COLUMNS)).sort_values(
        ["classification", "similarity_ratio"], ignore_index=True
    )


def brand_support(labeled_pairs: pd.DataFrame, brands: dict[str, str]) -> pd.DataFrame:
    """Per-brand positive/negative pair counts — the support behind a verdict."""
    left = labeled_pairs["gtin1"].astype(str).map(brands)
    right = labeled_pairs["gtin2"].astype(str).map(brands)
    label = labeled_pairs["true_label"].astype(int)
    counts: dict[tuple[str, str], list[int]] = {}
    for brand, positive in zip(pd.concat([left, right]), pd.concat([label, label])):
        entry = counts.setdefault(str(brand), [0, 0])
        entry[0 if positive == 1 else 1] += 1
    return pd.DataFrame(
        [
            {"brand": brand, "support_positive": pos, "support_negative": neg}
            for brand, (pos, neg) in counts.items()
        ]
    )


def catalog_brand_variants(
    canonical_records: pd.DataFrame, *, spec: BrandAnalysisSpec
) -> pd.DataFrame:
    """Raw brand strings that collapse to ONE normal form, with GTIN support.

    This is the surface-variant class measured on the CATALOG rather than on
    pairs: two GTINs whose brand fields differ only by case, accents,
    punctuation or a corporate suffix are the same brand written twice, and
    the encoder would see them as different.
    """
    brands = canonical_records["mode_brand"].astype(str)
    suffixes = tuple(spec.corporate_suffixes)
    keys = brands.map(
        lambda value: _strip_corporate_suffixes(
            normalized_attribute_text(value), suffixes
        )
    )
    variant_support: dict[str, int] = {}
    for key in keys:
        variant_support[key] = variant_support.get(key, 0) + 1
    rows = []
    for key, raw in pd.DataFrame({"key": keys, "brand": brands}).groupby("key"):
        distinct = sorted(set(raw["brand"]))
        if not key or len(distinct) < 2:
            continue
        rows.append(
            {
                "normalized_brand": key,
                "variants": " | ".join(distinct),
                "n_variants": len(distinct),
                "n_gtins": int(variant_support[key]),
                "reportable": bool(variant_support[key] >= spec.min_pair_support),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "normalized_brand", "variants", "n_variants", "n_gtins", "reportable",
        ],
    ).sort_values(["n_gtins", "normalized_brand"], ignore_index=True)


def review_comparisons(review: pd.DataFrame) -> pd.DataFrame:
    """Brand-vs-brand comparisons from the human-review band population.

    This is the population where brands actually differ (the labelled-pair
    population is ~100% same-brand by construction), so it is where the
    "why did the brand not match" question is answerable.
    """
    required = {
        "source_original_brand", "canonical_brand", "source_original_title",
    }
    absent = required - set(review.columns)
    if absent:
        raise ValueError(f"review frame missing columns {sorted(absent)}")
    support = review.groupby("source_original_brand").size()
    frame = pd.DataFrame(
        {
            "brand_left": review["source_original_brand"].astype(str),
            "brand_right": review["canonical_brand"].astype(str),
            "title": review["source_original_title"].astype(str),
        }
    )
    frame["support_positive"] = (
        frame["brand_left"].map(support).fillna(0).astype(int)
    )
    frame["support_negative"] = frame["support_positive"]
    return frame


def write_brand_analysis(
    comparisons: pd.DataFrame, *, spec: BrandAnalysisSpec | None = None
) -> pd.DataFrame:
    resolved = brand_spec() if spec is None else spec
    frame = analyse_brands(comparisons, spec=resolved)
    frame.to_csv(ensure_parent(F["brand_analysis_pairs"]), index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=F["labeled_pairs"])
    parser.add_argument("--canonicals", default=F["canonical_records"])
    args = parser.parse_args()
    spec = brand_spec()
    if not spec.enabled:
        raise SystemExit("evaluation.brand_analysis.enabled is false")

    canonicals = pd.read_csv(args.canonicals, dtype=str, keep_default_na=False)
    brands = dict(zip(canonicals["gtin"].astype(str), canonicals["mode_brand"].astype(str)))
    labeled = pd.read_csv(args.pairs, dtype={"gtin1": str, "gtin2": str})
    support = brand_support(labeled, brands).set_index("brand")

    comparisons = pd.DataFrame({
        "brand_left": labeled["gtin1"].astype(str).map(brands),
        "brand_right": labeled["gtin2"].astype(str).map(brands),
        "title": "",
    })
    comparisons = comparisons.join(
        support, on="brand_left"
    ).rename(
        columns={"support_positive": "support_positive", "support_negative": "support_negative"}
    )
    comparisons[["support_positive", "support_negative"]] = comparisons[
        ["support_positive", "support_negative"]
    ].fillna(0)

    frame = write_brand_analysis(comparisons, spec=spec)
    print(frame["classification"].value_counts().to_string())
    support_cleared = frame[frame["reportable"]]
    print(f"\n{len(frame):,} comparisons; {len(support_cleared):,} support-cleared")
    for kind in (CLASS_SURFACE_VARIANT, CLASS_MISSING, CLASS_TITLE_ONLY, CLASS_DIFFERENT):
        worst = support_cleared[support_cleared["classification"].eq(kind)].head(5)
        if not worst.empty:
            print(f"\n  worst {kind!r} (ratio, tfidf, support):")
            for _, row in worst.iterrows():
                print(
                    f"    {row['brand_left']!r:28} vs {row['brand_right']!r:28} "
                    f"ratio={row['similarity_ratio']:.3f} tfidf={row['tfidf_cosine']:.3f} "
                    f"n+= {int(row['n_positive'])} n-= {int(row['n_negative'])}"
                )


if __name__ == "__main__":
    main()
