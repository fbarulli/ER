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


def refresh_finetuned_ann(
    model,
    df: pd.DataFrame,
    payload: list[str],
    row_barcodes: np.ndarray,
    structured_features: np.ndarray | None = None,
    *,
    train_barcodes: set[str],
    existing: np.ndarray | None,
    step: int,
    epoch: float,
    output_path: Path,
    target: int,
    configured_band: tuple[float, float],
    k: int,
    candidate_multiplier: int,
    score_quantiles: tuple[float, float],
    max_per_canonical: int,
    max_per_brand: int,
    batch_size: int,
    max_seq_length: int,
    exclude_conflicting: bool,
) -> tuple[np.ndarray, dict[str, float]]:
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

    # This is the only encoding in this path: the caller supplies the already
    # fine-tuned model. No model ID, cache, or zero-shot encoder is accepted.
    model.max_seq_length = max_seq_length
    emb = model.encode(
        payload[: len(df)],
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
        emb = fuse_numpy(emb, np.asarray(structured_features)[: len(df)], sf_weight)

    broad_target = max(int(target), 1) * max(int(candidate_multiplier), 1)
    broad_pairs, broad_scores = mine_hard_negatives(
        df,
        emb,
        n_target=broad_target,
        cosine_lo=-1.0,
        cosine_hi=1.0,
        k=k,
        exclude_conflicting=exclude_conflicting,
        max_per_canonical=max(len(df), 1),
        max_per_brand=max(len(df), 1),
    )
    band_lo, band_hi, band_stats = calibrated_ann_band(
        broad_scores, configured_band, score_quantiles
    )
    refreshed, refreshed_scores = mine_hard_negatives(
        df,
        emb,
        n_target=target,
        cosine_lo=band_lo,
        cosine_hi=band_hi,
        k=k,
        exclude_conflicting=exclude_conflicting,
        max_per_canonical=max_per_canonical,
        max_per_brand=max_per_brand,
    )
    if len(refreshed):
        keep = pairs_in_set(refreshed, row_barcodes, set(train_barcodes))
        existing_keys = {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in (existing if existing is not None else [])
        }
        keep &= np.asarray(
            [
                (min(int(a), int(b)), max(int(a), int(b))) not in existing_keys
                for a, b in refreshed.tolist()
            ],
            dtype=bool,
        )
        refreshed = refreshed[keep]
        refreshed_scores = refreshed_scores[keep]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for pair, score in zip(refreshed.tolist(), refreshed_scores.tolist()):
        a, b = (int(x) for x in pair)
        rows.append(
            {
                "step": int(step),
                "epoch": float(epoch),
                "row_a": a,
                "row_b": b,
                "barcode_a": str(row_barcodes[a]),
                "barcode_b": str(row_barcodes[b]),
                "cosine": float(score),
                "band_lo": float(band_lo),
                "band_hi": float(band_hi),
                "source": "ann_finetuned",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, mode="w")
    return refreshed, {
        **band_stats,
        "status": "ok",
        "step": float(step),
        "epoch": float(epoch),
        "configured_band_lo": float(configured_band[0]),
        "configured_band_hi": float(configured_band[1]),
        "refreshed_count": float(len(refreshed)),
    }


__all__ = ["refresh_finetuned_ann"]
