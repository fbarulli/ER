"""Run calibrated matcher inference and build the original-dataset deliverable.

The checkpoint scores the deduplicated catalog through ``RandMatcher``—the
same retrieval and gate path used during calibration. Predictions are then
expanded to every row in ``dataset.csv`` through ``data/sku_to_rep.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.ann_config import load_ann_config
from core.common import F, TRAIN_ROOT, load_dataset, load_dataset_deduped
from core.manifest import sha256_file
from core.model_input import model_input_spec
from training.rand_matching import RandMatcher, _assignments_with_trace


DEFAULT_CHECKPOINT = (
    "training_results/0916T082923217621Z/worker_1/_checkpoints/"
    "all-MiniLM-L6-v2/r0916T082923217621Z_f0/checkpoint-114"
)
DEFAULT_THRESHOLD = 0.61
UNMATCHED_PREFIX = "UNMATCHED_"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (TRAIN_ROOT / path).resolve()


def _expand_to_original(predictions: pd.DataFrame, *, output: Path) -> dict:
    deduped = load_dataset_deduped()
    expected_skus = deduped["product_id"].astype(str).reset_index(drop=True)
    actual_skus = predictions["SKU_ID"].astype(str).reset_index(drop=True)
    if not actual_skus.equals(expected_skus):
        raise RuntimeError("matcher output does not preserve the deduped population")

    by_rep = predictions[["ITEM_ID"]].reset_index().rename(columns={"index": "rep_id"})
    rep_map = pd.read_csv(F["sku_to_rep"], dtype=str, keep_default_na=False)
    if list(rep_map.columns) != ["product_id", "rep_id"]:
        raise RuntimeError(
            "sku_to_rep columns must be ['product_id', 'rep_id']; got "
            f"{list(rep_map.columns)}"
        )
    rep_map["rep_id"] = pd.to_numeric(rep_map["rep_id"], errors="raise").astype(int)

    raw_skus = load_dataset()["product_id"].astype(str).reset_index(drop=True)
    mapped_skus = rep_map["product_id"].astype(str).reset_index(drop=True)
    if not mapped_skus.equals(raw_skus):
        raise RuntimeError("sku_to_rep does not match dataset.csv identity and order")
    if rep_map["rep_id"].lt(0).any() or rep_map["rep_id"].ge(len(deduped)).any():
        raise RuntimeError("sku_to_rep references a row outside the deduped dataset")

    expanded = rep_map.merge(by_rep, on="rep_id", how="left", validate="many_to_one")
    if expanded["ITEM_ID"].isna().any():
        raise RuntimeError("one or more original SKUs have no matcher assignment")
    deliverable = expanded.rename(
        columns={"product_id": "sku_id", "ITEM_ID": "item_id"}
    )[["sku_id", "item_id"]]
    if deliverable["sku_id"].duplicated().any():
        raise RuntimeError("the original dataset contains duplicate sku_id values")
    if deliverable["item_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("matcher emitted an empty item_id")

    output.parent.mkdir(parents=True, exist_ok=True)
    deliverable.to_csv(output, index=False)
    matched = ~deliverable["item_id"].astype(str).str.startswith(UNMATCHED_PREFIX)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "columns": list(deliverable.columns),
        "deduped_rows": int(len(deduped)),
        "raw_rows": int(len(deliverable)),
        "unique_sku_ids": int(deliverable["sku_id"].nunique()),
        "unique_item_ids": int(deliverable["item_id"].nunique()),
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "dataset_sha256": sha256_file(TRAIN_ROOT / "dataset.csv"),
        "sku_to_rep_sha256": sha256_file(F["sku_to_rep"]),
    }


def run_inference(
    checkpoint: Path,
    *,
    threshold: float,
    output: Path,
    batch_size: int | None,
) -> dict:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    input_spec = model_input_spec()
    if input_spec.profile != "cleaned":
        raise RuntimeError(
            "deliverable requires training.model_input.profile=cleaned; got "
            f"{input_spec.profile!r}"
        )
    ann = load_ann_config()
    effective_batch_size = int(batch_size or ann.embedding.encode_batch_size)
    matcher = RandMatcher(
        checkpoint,
        batch_size=effective_batch_size,
        top_k=int(ann.index.top_k),
        rebuild_ann_index=True,
    )
    candidates = matcher.score_candidates(load_dataset_deduped())
    predictions, trace = _assignments_with_trace(candidates, float(threshold))
    metadata = _expand_to_original(predictions, output=output)

    model_file = checkpoint / "model.safetensors"
    metadata.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": (
                sha256_file(model_file) if model_file.is_file() else None
            ),
            "threshold": float(threshold),
            "threshold_source": "fixed calibrated threshold",
            "matcher": "training.rand_matching.RandMatcher",
            "assignment": "training.rand_matching._assignments_with_trace",
            "model_input_profile": input_spec.profile,
            "ann_top_k": int(ann.index.top_k),
            "encode_batch_size": effective_batch_size,
            "candidate_rows": int(len(candidates)),
            "accepted_candidate_rows": int(trace["accepted"].sum()),
        }
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--output",
        default="submission/sku_item_submission_original_dataset_calibrated_061.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = _resolve(args.output)
    metadata = run_inference(
        _resolve(args.checkpoint),
        threshold=float(args.threshold),
        output=output,
        batch_size=args.batch_size,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"[submission] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
