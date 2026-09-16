"""Prepare matcher text locally and finalize GPU embeddings locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from core.ann_config import load_ann_config
from core.common import (
    F,
    TRAIN_ROOT,
    canonical_records_frame,
    load_config,
    load_dataset,
    load_dataset_deduped,
)
from core.manifest import sha256_file
from core.model_input import build_canonical_text, build_sku_text, model_input_info, model_input_spec
from core.structured_features import (
    canonical_info as canonical_structured_info,
    fuse_numpy,
    sku_info as sku_structured_info,
    vector as structured_vector,
)
from pipeline import load_canonical_map
from training.hnsw_index import PersistentHnswIndex, normalize_embeddings
from training.rand_matching import (
    RandMatcher,
    _assignments_with_trace,
    sku_attribute_info,
)


DEFAULT_THRESHOLD = 0.61
UNMATCHED_PREFIX = "UNMATCHED_"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (TRAIN_ROOT / path).resolve()


def _context() -> tuple[dict, list[str], dict[str, dict], pd.DataFrame, list[dict], list[dict]]:
    config = load_config()
    structured = config["training"]["structured_features"]
    if model_input_spec().profile != "cleaned":
        raise RuntimeError("inference requires training.model_input.profile=cleaned")
    canonical = load_canonical_map()
    item_ids = [str(value) for value in canonical]
    records = canonical_records_frame()
    record_map = {str(row["gtin"]): row.to_dict() for _, row in records.iterrows()}
    if set(item_ids) - set(record_map):
        raise RuntimeError("canonical metadata is incomplete")

    skus = RandMatcher._normalise_skus(load_dataset_deduped())
    gate_infos = [
        sku_attribute_info(row.get("title", ""), row.get("attributes", row.get("attr", "")))
        for _, row in skus.iterrows()
    ]
    model_infos = [
        model_input_info(sku_structured_info(
            row.get("title", ""), row.get("attributes", row.get("attr", ""))
        )) if structured["enabled"] else {"volume": set(), "pack": set()}
        for _, row in skus.iterrows()
    ]
    return structured, item_ids, record_map, skus, gate_infos, model_infos


def prepare_texts(output: Path) -> dict:
    structured, item_ids, record_map, skus, _, sku_infos = _context()
    item_infos = [
        model_input_info(canonical_structured_info(record_map[item_id]))
        if structured["enabled"] else {"volume": set(), "pack": set()}
        for item_id in item_ids
    ]
    texts = [
        build_canonical_text(record_map[item_id], info)
        for item_id, info in zip(item_ids, item_infos, strict=True)
    ]
    texts.extend(
        build_sku_text(row, info)
        for (_, row), info in zip(skus.iterrows(), sku_infos, strict=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for value in texts:
            handle.write(json.dumps(str(value), ensure_ascii=False) + "\n")
    metadata = {
        "profile": "cleaned",
        "item_count": len(item_ids),
        "sku_count": len(skus),
        "text_count": len(texts),
        "texts_sha256": sha256_file(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _feature_matrix(infos: list[dict], structured: dict) -> np.ndarray:
    return np.asarray(
        [
            structured_vector(
                info,
                volume_scale_ml=float(structured["volume_scale_ml"]),
                pack_scale=float(structured["pack_scale"]),
                max_set_size=int(structured["max_set_size"]),
            )
            for info in infos
        ],
        dtype=np.float32,
    )


def _score_candidates(
    matcher: RandMatcher,
    skus: pd.DataFrame,
    gate_infos: list[dict],
    sku_embeddings: np.ndarray,
) -> pd.DataFrame:
    labels, _ = matcher.ann_index.query(sku_embeddings, top_k=matcher.top_k)
    rows = []
    for position, (_, row) in enumerate(skus.iterrows()):
        sku_gtin = matcher._gtin(row.get("barcode", row.get("gtin", "")))
        candidate_indexes = matcher._candidate_indexes(labels[position].tolist(), sku_gtin)
        for index in sorted(candidate_indexes):
            rank, source = candidate_indexes[index]
            rows.append(
                matcher._candidate_row(
                    row, gate_infos[position], sku_embeddings[position], index, rank, source
                )
            )
    candidates = pd.DataFrame(rows)
    if candidates["SKU_ID"].nunique() != len(skus):
        raise RuntimeError("candidate retrieval dropped one or more SKU_ID values")
    return candidates


def _expand(predictions: pd.DataFrame, output: Path) -> dict:
    deduped = load_dataset_deduped()
    if predictions["SKU_ID"].astype(str).tolist() != deduped["product_id"].astype(str).tolist():
        raise RuntimeError("matcher output does not preserve the deduped population")
    by_rep = predictions[["ITEM_ID"]].reset_index().rename(columns={"index": "rep_id"})
    rep_map = pd.read_csv(F["sku_to_rep"], dtype=str, keep_default_na=False)
    rep_map["rep_id"] = pd.to_numeric(rep_map["rep_id"], errors="raise").astype(int)
    raw_skus = load_dataset()["product_id"].astype(str).tolist()
    if rep_map["product_id"].astype(str).tolist() != raw_skus:
        raise RuntimeError("sku_to_rep does not match dataset.csv identity and order")
    expanded = rep_map.merge(by_rep, on="rep_id", how="left", validate="many_to_one")
    if expanded["ITEM_ID"].isna().any():
        raise RuntimeError("one or more original SKUs have no assignment")
    result = expanded.rename(columns={"product_id": "sku_id", "ITEM_ID": "item_id"})[
        ["sku_id", "item_id"]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    matched = ~result["item_id"].astype(str).str.startswith(UNMATCHED_PREFIX)
    return {
        "columns": list(result.columns),
        "deduped_rows": len(deduped),
        "raw_rows": len(result),
        "unique_sku_ids": int(result["sku_id"].nunique()),
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "output_sha256": sha256_file(output),
        "dataset_sha256": sha256_file(TRAIN_ROOT / "dataset.csv"),
    }


def finalize(embeddings_path: Path, output: Path) -> dict:
    structured, item_ids, record_map, skus, gate_infos, sku_infos = _context()
    matrix = np.load(embeddings_path, allow_pickle=False)
    expected = len(item_ids) + len(skus)
    if matrix.shape[0] != expected:
        raise RuntimeError(f"embedding count mismatch: {matrix.shape[0]} != {expected}")
    item_raw = matrix[:len(item_ids)]
    sku_raw = matrix[len(item_ids):]
    item_infos = [
        model_input_info(canonical_structured_info(record_map[item_id]))
        if structured["enabled"] else {"volume": set(), "pack": set()}
        for item_id in item_ids
    ]
    weight = float(structured["embedding_weight"]) if (
        structured["enabled"] and structured["feed_to_loss"]
    ) else 0.0
    item_embeddings = normalize_embeddings(
        fuse_numpy(item_raw, _feature_matrix(item_infos, structured), weight)
    )
    sku_embeddings = fuse_numpy(
        sku_raw, _feature_matrix(sku_infos, structured), weight
    )

    ann = load_ann_config()
    with tempfile.TemporaryDirectory(prefix="submission-ann-") as temp_dir:
        index = PersistentHnswIndex(
            Path(temp_dir),
            ef_construction=ann.index.ef_construction,
            M=ann.index.M,
            ef_search=ann.index.ef_search,
            space=ann.index.space,
        )
        index.build(
            item_embeddings,
            item_ids,
            checkpoint=Path("checkpoint-114"),
            model_name=ann.embedding.model,
            preprocessing_fingerprint="local-prepared-cleaned",
        )
        matcher = RandMatcher.__new__(RandMatcher)
        matcher.checkpoint = Path("checkpoint-114")
        matcher.top_k = int(ann.index.top_k)
        matcher.config = load_config()
        matcher.structured_config = structured
        matcher.canonical = load_canonical_map()
        matcher.item_ids = item_ids
        matcher.item_index = {item_id: i for i, item_id in enumerate(item_ids)}
        matcher.record_map = record_map
        matcher.item_embeddings = item_embeddings
        matcher.ann_index = index
        candidates = _score_candidates(matcher, skus, gate_infos, sku_embeddings)
    predictions, trace = _assignments_with_trace(candidates, DEFAULT_THRESHOLD)
    metadata = _expand(predictions, output)
    metadata.update({
        "threshold": DEFAULT_THRESHOLD,
        "threshold_source": "fixed calibrated threshold",
        "model_input_profile": "cleaned",
        "ann_top_k": int(ann.index.top_k),
        "gpu_embeddings_sha256": sha256_file(embeddings_path),
        "candidate_rows": len(candidates),
        "accepted_candidate_rows": int(trace["accepted"].sum()),
    })
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--embeddings", required=True)
    finish.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_texts(_resolve(args.output))
    else:
        result = finalize(_resolve(args.embeddings), _resolve(args.output))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
