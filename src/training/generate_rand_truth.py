"""Generate reusable, balanced truth CSVs for ``er-rand-match``.

The source of truth is the frozen canonical map.  This command chooses one
representative source SKU for each selected canonical identity, balances the
selection across canonical brand, volume, package, material, and type strata,
and writes disjoint ``SKU_ID,true_item_id`` calibration and holdout files.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from core.common import TRAIN_ROOT, canonical_records_frame, load_dataset_deduped, rand_matching_cfg
from core.manifest import atomic_write_csv


_STRATA_COLUMNS = (
    "mode_brand",
    "volume_set",
    "pack_set",
    "package_type_set",
    "package_material_set",
    "mode_type",
)


def _normalise(value: object) -> str:
    text = str(value).strip()
    return text if text else "<missing>"


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _truth_candidates() -> pd.DataFrame:
    canonicals = canonical_records_frame()
    if canonicals["gtin"].duplicated().any():
        raise ValueError("canonical truth map contains duplicate gtin values")
    source = load_dataset_deduped().copy()
    if "product_id" not in source or "barcode" not in source:
        raise ValueError("deduplicated dataset must contain product_id and barcode")
    source["SKU_ID"] = source["product_id"].astype(str).str.strip()
    source["barcode"] = source["barcode"].astype(str).str.strip()
    if source["SKU_ID"].eq("").any() or source["SKU_ID"].duplicated().any():
        raise ValueError("deduplicated dataset has blank or duplicate product_id values")
    representatives = (
        source.loc[source["barcode"].ne(""), ["SKU_ID", "barcode"]]
        .sort_values(["barcode", "SKU_ID"], kind="stable")
        .drop_duplicates("barcode", keep="first")
    )
    candidates = canonicals.merge(
        representatives,
        left_on="gtin",
        right_on="barcode",
        how="inner",
        validate="one_to_one",
    )
    if candidates["gtin"].duplicated().any() or candidates["SKU_ID"].duplicated().any():
        raise RuntimeError("canonical truth candidate identity is not one-to-one")
    if len(candidates) == 0:
        raise RuntimeError("no canonical truths have a representative SKU")
    return candidates


def _balanced_truths(candidates: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size > len(candidates):
        raise ValueError(
            f"requested {sample_size:,} truth samples but only {len(candidates):,} "
            "canonical identities have representative SKUs"
        )
    frame = candidates.copy()
    for column in _STRATA_COLUMNS:
        frame[column] = frame[column].map(_normalise)
    frame["_stratum"] = frame[list(_STRATA_COLUMNS)].agg("\x1e".join, axis=1)
    queues: dict[str, deque[int]] = {}
    for stratum, group in frame.groupby("_stratum", sort=False):
        ordered = sorted(
            group.index.tolist(),
            key=lambda index: _stable_rank(seed, stratum, str(frame.at[index, "gtin"])),
        )
        queues[stratum] = deque(ordered)

    # Round-robin over the composite strata makes each brand/volume/package
    # combination contribute before any combination contributes a second row.
    # The per-cycle order is seeded but stable, so re-runs are reproducible.
    ordered_strata = sorted(queues, key=lambda value: _stable_rank(seed, "stratum", value))
    selected: list[int] = []
    while len(selected) < sample_size:
        progressed = False
        for stratum in ordered_strata:
            if queues[stratum]:
                selected.append(queues[stratum].popleft())
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:
            raise RuntimeError("balanced canonical selection exhausted unexpectedly")
    return frame.loc[selected].copy()


def _split_truths(selected: pd.DataFrame, calibration_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Assign within every composite stratum in a 2:1 rotating pattern.  This
    # preserves broad stratum coverage in each reusable file while retaining
    # strict canonical disjointness.
    calibration: list[int] = []
    holdout: list[int] = []
    groups = sorted(selected.groupby("_stratum", sort=False), key=lambda item: _stable_rank(seed, "split", item[0]))
    for stratum, group in groups:
        indices = sorted(group.index.tolist(), key=lambda index: _stable_rank(seed, "row", stratum, str(selected.at[index, "gtin"])))
        for position, index in enumerate(indices):
            (calibration if position % 3 else holdout).append(index)

    # Composite singleton strata initially go to holdout.  Deterministically
    # fill the requested sizes without ever moving an identity into both sets.
    all_indices = sorted(selected.index.tolist(), key=lambda index: _stable_rank(seed, "rebalance", str(selected.at[index, "gtin"])))
    while len(calibration) < calibration_size:
        index = next(index for index in all_indices if index in holdout)
        holdout.remove(index)
        calibration.append(index)
    while len(calibration) > calibration_size:
        index = next(index for index in all_indices if index in calibration)
        calibration.remove(index)
        holdout.append(index)
    calibration_frame = selected.loc[calibration, ["SKU_ID", "gtin"]].rename(columns={"gtin": "true_item_id"})
    holdout_frame = selected.loc[holdout, ["SKU_ID", "gtin"]].rename(columns={"gtin": "true_item_id"})
    for name, frame in (("calibration", calibration_frame), ("holdout", holdout_frame)):
        if frame["SKU_ID"].duplicated().any() or frame["true_item_id"].duplicated().any():
            raise RuntimeError(f"{name} truth split has duplicate identities")
    if set(calibration_frame["SKU_ID"]) & set(holdout_frame["SKU_ID"]):
        raise RuntimeError("truth splits overlap in SKU_ID")
    if set(calibration_frame["true_item_id"]) & set(holdout_frame["true_item_id"]):
        raise RuntimeError("truth splits overlap in canonical identity")
    return calibration_frame.sort_values("SKU_ID", kind="stable"), holdout_frame.sort_values("SKU_ID", kind="stable")


def generate_truth_splits(output_dir: Path, sample_size: int, calibration_size: int, seed: int, calibration_name: str, holdout_name: str) -> tuple[Path, Path]:
    selected = _balanced_truths(_truth_candidates(), sample_size, seed)
    calibration, holdout = _split_truths(selected, calibration_size, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / calibration_name
    holdout_path = output_dir / holdout_name
    atomic_write_csv(calibration, calibration_path, index=False)
    atomic_write_csv(holdout, holdout_path, index=False)
    print(
        f"[rand-truth] selected={len(selected):,} calibration={len(calibration):,} "
        f"holdout={len(holdout):,} strata={selected['_stratum'].nunique():,}",
        flush=True,
    )
    print(f"[rand-truth] calibration={calibration_path}", flush=True)
    print(f"[rand-truth] holdout={holdout_path}", flush=True)
    return calibration_path, holdout_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--calibration-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = rand_matching_cfg()["truth_splits"]
    output_dir = Path(args.output_dir or cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = TRAIN_ROOT / output_dir
    generate_truth_splits(
        output_dir.resolve(),
        int(args.sample_size if args.sample_size is not None else cfg["sample_size"]),
        int(
            args.calibration_size
            if args.calibration_size is not None
            else cfg["calibration_size"]
        ),
        int(args.seed if args.seed is not None else cfg["seed"]),
        str(cfg["calibration_output"]),
        str(cfg["holdout_output"]),
    )


if __name__ == "__main__":
    main()
