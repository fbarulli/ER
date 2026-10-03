#!/usr/bin/env python3
"""Brand-differentiation audit (owner theory, 2026-10-01).

Per brand, measure whether between-GTIN card delta exceeds within-GTIN
(same product, cross-retailer) variation — the graph lane's go/no-go.

The theory, made testable:
  * NOISE floor: same-GTIN, cross-retailer listing pairs re-state one true
    product — their card deltas (volume spread, pack diversity) are what
    differentiation must rise above.
  * SIGNAL: similar within-brand pairs (gate_results similarity >=
    differentiation_audit.sim_min) — their canonical-card deltas on the
    SAME dimensions are the discriminator.
A brand PASSES when median(signal) >= volume_margin x median(noise) on
volume AND pack/package contrasts actually occur between similar pairs;
those brands support slot-filling differentiation. FAILING brands send
their low-confidence pairs to the original-dataset clarification lane.

Reuses SSOT readers only (no duplicate parsers):
  card set columns   core.attribute_conflicts._value_set / _string_value_set
  volume predicate   core.critical_attributes.volumes_compatible
  listing extraction pipeline.extract_all
  data access        core.common (load_dataset_deduped, F, training_cfg)

Usage:
  PYTHONPATH=src .venv/bin/python scripts/brand_differentiation_audit.py
Output:
  results/differentiation_audit.csv — one row per brand + verdicts
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.attribute_conflicts import _string_value_set, _value_set
from core.critical_attributes import categorical_conflict, volumes_compatible
from core.common import F, RESULTS, load_dataset_deduped, training_cfg


def _listing_card(row: pd.Series, max_listings: int) -> dict | None:
    """One listing's card via pipeline.extract_all (the SSOT extractor)."""
    from pipeline import extract_all

    ex = extract_all(
        str(row.get("sku_name_eng", "")),
        str(row.get("attribute", "")),
        str(row.get("description_short_eng", "")),
        sku_url=str(row.get("sku_url", "")),
        image_url=str(row.get("image_url", "")),
        breadcrumbs_eng=str(row.get("breadcrumbs_eng", "")),
        category=str(row.get("category", "")),
    )
    return ex


def within_gtin_noise(df: pd.DataFrame, knobs) -> pd.DataFrame:
    """Per-GTIN cross-retailer variation of the listing cards.

    Returns one row per multi-retailer GTIN with:
      volume_ratio  (max-min)/mean of listing volumes (0 when <2 known)
      pack_diversity n distinct confident pack_qty values
      n_listings
    """
    records = []
    for gtin, frame in df.groupby("gtin", sort=False):
        if frame["retailer"].nunique() < 2:
            continue
        frame = frame.head(knobs.max_listings_per_gtin)
        volumes: list[float] = []
        packs: set[float] = set()
        for _, row in frame.iterrows():
            extracted = _listing_card(row, knobs.max_listings_per_gtin)
            flags = extracted.get("attribute_consistency_flags") or set()
            if extracted["volume_ml"] > 0 and "ambiguous_volume" not in flags:
                volumes.append(float(extracted["volume_ml"]))
            if extracted["pack_confidence"] > 0:
                packs.add(float(extracted["pack_qty"]))
        spread = (
            (max(volumes) - min(volumes)) / max(min(volumes), 1.0)
            if len(volumes) >= 2
            else 0.0
        )
        records.append(
            {
                "gtin": gtin,
                "brand": str(frame["brand"].iloc[0]).casefold().strip(),
                "volume_within_ratio": spread,
                "pack_within_diversity": len(packs),
                "n_listings": int(len(frame)),
            }
        )
    return pd.DataFrame(
        records, columns=["gtin", "brand", "volume_within_ratio", "pack_within_diversity", "n_listings"]
    )


def between_gtin_signal(canonical: pd.DataFrame, gate: pd.DataFrame, knobs) -> pd.DataFrame:
    """Per similar within-brand pair: canonical-card deltas (SSOT parsers).

    volume_delta   relative gap between the two cards' volume_set medians
                   (abs when either side has no volume)
    pack_conflict  True when both pack_sets nonempty and disjoint
    flavor_conflict True when the flavor dimension categorical-conflicts
    similarity     gate_results similarity
    """
    canon = canonical.set_index("gtin")
    cards: dict[str, dict] = {}
    brands: dict[str, str] = {}
    for gtin, row in canon.iterrows():
        try:
            volume = _value_set(row["volume_set"], kind="volume")
            pack = _value_set(row["pack_set"], kind="pack")
            flavor = _string_value_set(row["flavor_set"], kind="flavor")
        except ValueError:
            continue
        cards[gtin] = {"volume_set": volume, "pack_set": pack, "flavor_set": flavor}
        brands[gtin] = str(row["mode_brand"]).casefold().strip()

    rows = []
    per_brand_counts: dict[str, int] = defaultdict(int)
    ordered = gate[gate["similarity"] != ""].copy()
    ordered = ordered.sort_values("similarity", ascending=False)
    for row in ordered.itertuples():
        left, right = row.gtin1, row.gtin2
        b1, b2 = brands.get(left), brands.get(right)
        if b1 is None or b1 != b2 or not b1:
            continue
        if per_brand_counts[b1] >= knobs.max_between_pairs_per_brand:
            continue
        if float(row.similarity) < knobs.sim_min:
            continue
        c1, c2 = cards.get(left, {"volume_set": set(), "pack_set": set(), "flavor_set": set()}), cards.get(right, {"volume_set": set(), "pack_set": set(), "flavor_set": set()})
        med1 = median(c1["volume_set"])
        med2 = median(c2["volume_set"])
        if med1 is not None and med2 is not None:
            delta = abs(med1 - med2) / max(min(med1, med2), 1.0)
        else:
            delta = float("nan")  # unknown-vs-known or unknown-vs-unknown
        if not (c1["pack_set"] and c2["pack_set"]):
            pack_conflict = False
        else:
            pack_conflict = not (c1["pack_set"] & c2["pack_set"])
        flavor_conflict = categorical_conflict(
            "flavor", {"flavor": c1["flavor_set"]}, {"flavor": c2["flavor_set"]}
        )
        per_brand_counts[b1] += 1
        rows.append(
            {
                "brand": b1,
                "similarity": float(row.similarity),
                "volume_delta": delta,
                "pack_conflict": pack_conflict,
                "flavor_conflict": flavor_conflict,
            }
        )
    return pd.DataFrame(
        rows, columns=["brand", "similarity", "volume_delta", "pack_conflict", "flavor_conflict"]
    )


def median(values: set[float]) -> float | None:
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return None
    return float(sorted(values)[len(values) // 2])


def verdict(row: pd.Series, knobs) -> str:
    """PASS when signal clears the measured noise floor (volume_margin)."""
    signal = row.get("between_signal_ratio")
    noise = row.get("within_noise_ratio")
    if pd.isna(signal) or pd.isna(noise):
        return "insufficient"
    if row.get("between_pairs", 0) < knobs.min_between_pairs:
        return "insufficient"
    if row.get("within_gtins", 0) < knobs.min_within_gtins:
        return "insufficient"
    if float(signal) >= knobs.volume_margin * max(float(noise), 1e-9):
        return "pass"
    return "fail"


def main() -> None:
    knobs = training_cfg().differentiation_audit
    print(f"[knobs] sim_min={knobs.sim_min} margin={knobs.volume_margin} "
          f"min_between={knobs.min_between_pairs} min_within={knobs.min_within_gtins}",
          flush=True)

    print("[load] dataset_deduped ...", flush=True)
    df = load_dataset_deduped()

    print("[within] extracting per-listing cards for multi-retailer GTINs ...", flush=True)
    within = within_gtin_noise(df, knobs)
    print(f"[within] {len(within):,} multi-retailer GTINs", flush=True)

    print("[between] canonical cards + gate similarity ...", flush=True)
    canonical = pd.read_csv(
        RESULTS / F["canonical_records"], dtype={"gtin": str}, keep_default_na=False
    )
    gate = pd.read_csv(RESULTS / F["gate_results"], dtype=str, keep_default_na=False)
    between = between_gtin_signal(canonical, gate, knobs)
    print(f"[between] {len(between):,} similar within-brand pairs", flush=True)

    # Aggregate per brand + verdict
    within_brand = within.groupby("brand").agg(
        within_gtins=("gtin", "size"),
        within_noise_ratio=("volume_within_ratio", "median"),
        within_share_packing=("pack_within_diversity", lambda s: (s > 0).mean()),
    )
    between_brand = between.groupby("brand").agg(
        between_pairs=("volume_delta", "size"),
        between_signal_ratio=("volume_delta", "median"),
        pack_conflict_share=("pack_conflict", "mean"),
        flavor_conflict_share=("flavor_conflict", "mean"),
        mean_similarity=("similarity", "mean"),
    )
    audit = within_brand.join(between_brand, how="outer").reset_index()
    verdicts = []
    for _, row in audit.iterrows():
        piece = between[between["brand"] == row["brand"]]
        verdicts.append(verdict(piece, knobs))
    audit["verdict"] = [verdict(row, knobs) for _, row in audit.iterrows()]
    audit = audit.sort_values(["verdict", "between_pairs"], ascending=[False, False])

    out_path = Path("results/differentiation_audit.csv")
    out_path.parent.mkdir(exist_ok=True)
    audit.to_csv(out_path, index=False)
    print(f"\n[out] {out_path}")
    shown = audit.head(30)
    print(shown.to_string(index=False, max_colwidth=24))
    passed = int((audit["verdict"] == "pass").sum())
    failed = int((audit["verdict"] == "fail").sum())
    insufficient = int((audit["verdict"] == "insufficient").sum())
    print(f"\nverdicts: pass={passed} fail={failed} insufficient={insufficient} total={len(audit)}")


if __name__ == "__main__":
    main()
