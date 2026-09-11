"""Write a provenance-pinned quality audit for a raw reconciliation export.

This is deliberately a report, not a filter: rows are never removed and no
quality threshold is hidden in code.  It answers whether the source can
support GTIN-derived reconciliation labels and identifies the records that
need review before training.

The configured input is used by default.  A different export requires an
explicit ``--input`` path; there is no fallback to a similarly named file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from euromonitor.pipeline import extract_all, normalize_text
from euromonitor.core.common import COLUMN_MAPPING, DATA_PATH, F, RESULTS, column_profile
from euromonitor.core.gtin import normalize_and_validate_gtin


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def _values(values: pd.Series) -> str:
    return " | ".join(sorted({str(v) for v in values if str(v).strip()}))


def _audit_groups(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[df["gtin_valid"]].copy()
    if valid.empty:
        return pd.DataFrame(columns=[
            "gtin", "rows", "retailers", "countries", "titles", "brands",
            "categories", "volume_ml", "pack_count", "volume_statuses",
            "review_reasons",
        ])

    rows: list[dict[str, object]] = []
    for gtin, group in valid.groupby("gtin_clean", sort=True, dropna=False):
        titles = group["title"].map(normalize_text)
        brands = group["brand"].map(normalize_text)
        categories = group["category"].map(normalize_text)
        volumes = group["volume_ml"].dropna().astype(float)
        packs = group["pack_count"].dropna().astype(int)
        reasons: list[str] = []
        if titles.nunique() > 1:
            reasons.append("title_variation")
        if brands[brands.ne("")].nunique() > 1:
            reasons.append("brand_conflict")
        if categories[categories.ne("")].nunique() > 1:
            reasons.append("category_conflict")
        if volumes.nunique() > 1:
            reasons.append("volume_conflict")
        if packs.nunique() > 1:
            reasons.append("pack_conflict")
        if group["retailer"].nunique() < 2:
            reasons.append("single_retailer_identity_not_cross_source_verified")
        rows.append({
            "gtin": str(gtin),
            "rows": len(group),
            "retailers": _values(group["retailer"]),
            "countries": _values(group["country"]),
            "titles": _values(group["title"]),
            "brands": _values(group["brand"]),
            "categories": _values(group["category"]),
            "volume_ml": _values(volumes),
            "pack_count": _values(packs),
            "volume_statuses": _values(group["volume_status"]),
            "review_reasons": " | ".join(reasons),
        })
    return pd.DataFrame(rows)


def audit(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not source.is_file():
        raise FileNotFoundError(
            f"raw export missing: {source}. Pass --input explicitly or restore "
            f"the configured source at {DATA_PATH}; no fallback is used."
        )
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [column for column in COLUMN_MAPPING if column not in raw.columns]
    if missing:
        raise ValueError(
            f"raw export is missing mapped columns {missing}; available columns: "
            f"{list(raw.columns)}"
        )
    df = raw.rename(columns=COLUMN_MAPPING).copy()
    gtin = normalize_and_validate_gtin(df["barcode"])
    df["gtin_clean"] = gtin["gtin_clean"]
    df["gtin_valid"] = gtin["gtin_structurally_valid"]
    attrs = [extract_all(title, attribute) for title, attribute in zip(df["title"], df["attributes"], strict=True)]
    attr_frame = pd.DataFrame(attrs, index=df.index)
    df["volume_ml"] = attr_frame["volume_ml"]
    df["pack_count"] = attr_frame["pack_qty"]
    df["volume_status"] = attr_frame["volume_status"]

    total = len(df)
    valid_rows = int(df["gtin_valid"].sum())
    valid_groups = df.loc[df["gtin_valid"], "gtin_clean"].nunique()
    multi_groups = (
        df.loc[df["gtin_valid"]]
        .groupby("gtin_clean")["retailer"].nunique()
        .gt(1)
        .sum()
    )
    summary = pd.DataFrame([
        {"metric": "source_path", "value": str(source.resolve()), "detail": "explicit input provenance"},
        {"metric": "source_sha256", "value": _sha256(source), "detail": "input bytes fingerprint"},
        {"metric": "rows", "value": total, "detail": "no rows were filtered"},
        {"metric": "empty_product_id_rows", "value": int(_blank(df["product_id"]).sum()), "detail": "identity-source completeness"},
        {"metric": "duplicate_product_id_rows", "value": int(df["product_id"].duplicated(keep=False).sum()), "detail": "raw export duplication; dedupe remains a separate audited step"},
        {"metric": "empty_title_rows", "value": int(_blank(df["title"]).sum()), "detail": "cannot provide title evidence"},
        {"metric": "empty_brand_rows", "value": int(_blank(df["brand"]).sum()), "detail": "cannot use brand blocking"},
        {"metric": "barcode_present_rows", "value": int((~_blank(df["barcode"])).sum()), "detail": "raw identifier availability"},
        {"metric": "valid_gtin_rows", "value": valid_rows, "detail": "GS1-structurally-valid only; valid is not a claim of semantic correctness"},
        {"metric": "invalid_or_missing_gtin_rows", "value": total - valid_rows, "detail": "retained in source but cannot form GTIN-derived labels"},
        {"metric": "valid_gtin_groups", "value": int(valid_groups), "detail": "provisional canonical identities"},
        {"metric": "cross_retailer_valid_gtin_groups", "value": int(multi_groups), "detail": "groups that can provide cross-source positive evidence"},
        {"metric": "title_volume_extracted_rows", "value": int((df["volume_status"] != "no_volume_mention").sum()), "detail": "attribute parser coverage; no row was dropped when absent"},
    ])
    groups = _audit_groups(df)
    if not groups.empty:
        summary.loc[len(summary)] = {
            "metric": "valid_gtin_groups_requiring_review",
            "value": int(groups["review_reasons"].ne("").sum()),
            "detail": "groups with title, brand, category, volume, pack, or source-coverage variation",
        }
        for reason, count in (
            groups.loc[groups["review_reasons"].ne(""), "review_reasons"]
            .str.split(" | ", regex=False)
            .explode()
            .value_counts()
            .sort_index()
            .items()
        ):
            summary.loc[len(summary)] = {
                "metric": f"review_reason_{reason}",
                "value": int(count),
                "detail": "valid-GTIN group count; groups can contribute to multiple reasons",
            }
    profiles = column_profile(raw)
    return summary, profiles, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    summary, profiles, groups = audit(args.input)
    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        F["data_quality_summary"]: summary,
        F["data_quality_columns"]: profiles,
        F["data_quality_gtin_groups"]: groups,
    }
    for name, frame in outputs.items():
        path = RESULTS / name
        frame.to_csv(path, index=False)
        print(f"[quality] wrote {path} ({len(frame):,} rows)", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
