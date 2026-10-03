"""Fine-tuned-model ANN refreshes.

This module deliberately has no base-model/zero-shot entry point.  A refresh
receives the live fine-tuned ``SentenceTransformer`` and encodes the corpus
with that model before mining.  The trainer can therefore warm-start on its
non-ANN population and replace dynamic negative slots only after a checkpoint
exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.hard_negatives import calibrated_ann_band, mine_hard_negatives, pairs_in_set


def encode_finetuned_embeddings(
    model,
    payload: list[str],
    structured_features: np.ndarray | None,
    *,
    batch_size: int,
    max_seq_length: int,
) -> np.ndarray:
    """Encode the supplied payload with the live fine-tuned model exactly once."""
    from core.encoding_inputs import enable_zero_truncation
    enable_zero_truncation(model)
    model.max_seq_length = max_seq_length
    emb = model.encode(
        payload,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if structured_features is not None:
        from core.common import load_config
        from core.structured_features import fuse_numpy

        sf_cfg = load_config()["training"]["structured_features"]
        sf_weight = (
            float(sf_cfg["embedding_weight"])
            if bool(sf_cfg["enabled"]) and bool(sf_cfg["feed_to_loss"])
            else 0.0
        )
        emb = fuse_numpy(emb, np.asarray(structured_features), sf_weight)
    if emb.ndim != 2 or emb.shape[0] != len(payload):
        raise RuntimeError(
            "fine-tuned ANN embedding shape mismatch: "
            f"{emb.shape} != ({len(payload)}, d)"
        )
    return emb


def refresh_finetuned_ann(
    model,
    df: pd.DataFrame,
    payload: list[str],
    row_gtins: np.ndarray,
    structured_features: np.ndarray | None = None,
    *,
    train_gtins: set[str],
    existing: np.ndarray | None,
    step: int,
    epoch: float,
    output_path: Path,
    target: int,
    configured_band: tuple[float, float],
    band_mode: str,
    k: int,
    candidate_multiplier: int,
    score_quantiles: tuple[float, float],
    max_per_canonical: int,
    max_per_brand: int,
    batch_size: int,
    max_seq_length: int,
    exclude_conflicting: bool,
    embeddings: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mine from the current fine-tuned model and rewrite one audit CSV.

    The broad candidate pass is intentionally unbounded by the configured
    mid-band.  Its scores are used only to validate/recenter that band against
    the current model.  The second pass applies the calibrated band and the
    configured per-canonical/per-brand diversity caps.
    """
    if target <= 0:
        return np.empty((0, 2), dtype=int), {"status": "disabled"}
    if not payload or len(payload) < len(df):
        raise ValueError(
            "fine-tuned ANN refresh requires payload rows covering the catalog"
        )

    # The caller may provide this checkpoint's single encoding so ANN and
    # attribute-conflict miners operate on identical fine-tuned embeddings.
    emb = (
        np.asarray(embeddings, dtype=np.float32)
        if embeddings is not None
        else encode_finetuned_embeddings(
            model,
            payload[: len(df)],
            structured_features[: len(df)] if structured_features is not None else None,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )
    )
    if emb.shape[0] < len(df):
        raise RuntimeError(
            f"fine-tuned ANN embeddings cover {emb.shape[0]} rows; need {len(df)}"
        )

    # Search only training entities; heldout rows cannot consume top-k slots,
    # diversity quotas or calibration samples. Keep original payload indices.
    source_rows = np.arange(len(df), dtype=int)
    eligible = pairs_in_set(np.column_stack((source_rows, source_rows)), row_gtins, set(train_gtins))
    selected_rows = source_rows[eligible]
    candidate_df = df.iloc[selected_rows].reset_index(drop=True)
    candidate_pairs, candidate_scores = mine_hard_negatives(
        candidate_df, emb[selected_rows], n_target=max(len(selected_rows) * int(k), 1),
        cosine_lo=-1.0, cosine_hi=1.0, k=k,
        exclude_conflicting=exclude_conflicting,
        max_per_canonical=max(len(selected_rows) * int(k), 1),
        max_per_brand=max(len(selected_rows) * int(k), 1),
    )
    candidate_pairs = selected_rows[candidate_pairs]
    existing_keys = {(min(int(a), int(b)), max(int(a), int(b)))
                     for a, b in (existing if existing is not None else [])}
    novel = np.asarray([(min(int(a), int(b)), max(int(a), int(b))) not in existing_keys
                        for a, b in candidate_pairs], dtype=bool)
    rejected_existing = int((~novel).sum())
    candidate_pairs, candidate_scores = candidate_pairs[novel], candidate_scores[novel]
    broad_target = max(int(target), 1) * max(int(candidate_multiplier), 1)
    band_lo, band_hi, band_stats = calibrated_ann_band(
        candidate_scores[:broad_target], configured_band, score_quantiles, band_mode
    )
    from collections import Counter
    canonical_counts, brand_counts = Counter(), Counter()
    gtins = df["gtin"].fillna("").astype(str).to_numpy()
    brands = df["brand"].fillna("").astype(str).str.strip().str.lower().to_numpy()
    rows_out, scores_out = [], []
    for (a, b), score in zip(candidate_pairs, candidate_scores, strict=True):
        if not band_lo <= score <= band_hi:
            continue
        endpoint_gtins, endpoint_brands = (gtins[a], gtins[b]), (brands[a], brands[b])
        if any(value and canonical_counts[value] >= max_per_canonical for value in endpoint_gtins):
            continue
        if any(value and brand_counts[value] >= max_per_brand for value in endpoint_brands):
            continue
        rows_out.append((int(a), int(b)))
        scores_out.append(float(score))
        canonical_counts.update(value for value in endpoint_gtins if value)
        brand_counts.update(value for value in endpoint_brands if value)
        if len(rows_out) >= target:
            break
    refreshed = np.asarray(rows_out, dtype=int).reshape(-1, 2)
    refreshed_scores = np.asarray(scores_out, dtype=np.float32)
    band_stats.update(eligible_source_rows=int(len(selected_rows)),
                      rejected_existing=rejected_existing,
                      eligible_candidate_count=int(len(candidate_pairs)),
                      requested_count=int(target),
                      shortfall_count=int(max(target - len(refreshed), 0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for pair, score in zip(refreshed.tolist(), refreshed_scores.tolist(), strict=True):
        a, b = (int(x) for x in pair)
        rows.append(
            {
                "step": int(step),
                "epoch": float(epoch),
                "row_a": a,
                "row_b": b,
                "gtin_a": str(row_gtins[a]),
                "gtin_b": str(row_gtins[b]),
                "cosine": float(score),
                "band_lo": float(band_lo),
                "band_hi": float(band_hi),
                "band_mode": band_mode,
                "source": "ann_finetuned",
            }
        )
    pd.DataFrame(rows, columns=["step", "epoch", "row_a", "row_b", "gtin_a", "gtin_b", "cosine",
                                "band_lo", "band_hi", "band_mode", "source"]).to_csv(output_path, index=False, mode="w")
    return refreshed, {
        **band_stats,
        "status": "ok",
        "step": float(step),
        "epoch": float(epoch),
        "configured_band_lo": float(configured_band[0]),
        "configured_band_hi": float(configured_band[1]),
        "band_mode": band_mode,
        "refreshed_count": float(len(refreshed)),
        "scores": refreshed_scores.tolist(),
    }


__all__ = ["encode_finetuned_embeddings", "refresh_finetuned_ann"]
