"""Checkpoint-level unrelated-pair uniformity diagnostics.

This is deliberately separate from the label-based pair metrics.  A model can
score its mined positives/negatives well while mapping unrelated catalog text
to one narrow high-cosine region.  The audit uses the exact model payload,
requires different brands/categories and zero shared payload tokens, and
records the median, P90, and threshold-crossing rate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def select_unrelated_pairs(
    df: pd.DataFrame,
    payload: list[str],
    *,
    n_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Select deterministic, payload-disjoint unrelated SKU pairs."""
    category_col = "category_path" if "category_path" in df.columns else "category"
    brand = df["brand"].fillna("").astype(str).str.strip().str.lower().tolist()
    category = (
        df[category_col].fillna("").astype(str).str.strip().str.lower().tolist()
    )
    tokens = [_tokens(text) for text in payload[: len(df)]]
    valid = [
        i
        for i in range(len(df))
        if tokens[i] and brand[i] and category[i]
    ]
    order = list(np.random.default_rng(seed).permutation(valid))
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for left in order:
        if left in used:
            continue
        for right in order:
            if right == left or right in used:
                continue
            if brand[left] == brand[right] or category[left] == category[right]:
                continue
            if tokens[left] & tokens[right]:
                continue
            selected.append((left, right))
            used.update((left, right))
            break
        if len(selected) >= n_pairs:
            return selected
    # This is a diagnostic, not a training or publication gate.  Return the
    # available deterministic sample so small/overlapping catalogs can still
    # finalize and record an explicit insufficient-sample result.
    return selected


def _summary(scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    return {
        "n_pairs": int(len(scores)),
        "median_cosine": float(np.median(scores)),
        "p90_cosine": float(np.quantile(scores, 0.90)),
        "threshold": float(threshold),
        "threshold_crossing_rate": float(np.mean(scores >= threshold)),
        "n_above_threshold": int(np.sum(scores >= threshold)),
    }


def run_uniformity_audit(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    df: pd.DataFrame,
    payload: list[str],
    n_pairs: int,
    seed: int,
    threshold: float,
) -> dict:
    """Compare base and fine-tuned cosine scores on the same unrelated pairs."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"uniformity checkpoint missing: {checkpoint}")
    base_model = Path(base_model)
    if not base_model.is_dir():
        raise FileNotFoundError(
            f"uniformity base model is not materialized locally: {base_model}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = select_unrelated_pairs(
        df, payload, n_pairs=n_pairs, seed=seed
    )
    if len(pairs) < n_pairs:
        frame = pd.DataFrame(
            [
                {"pair": i, "row_a": left, "row_b": right}
                for i, (left, right) in enumerate(pairs)
            ]
        )
        frame.to_csv(out / "uniformity_unrelated_pairs.csv", index=False)
        result = {
            "base_model": str(base_model),
            "fine_tuned_checkpoint": str(checkpoint),
            "selection": "different brand, different category, zero shared exact payload tokens",
            "seed": int(seed),
            "status": "insufficient_pairs",
            "required_pairs": int(n_pairs),
            "selected_pairs": int(len(pairs)),
        }
        (out / "uniformity_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    texts = [payload[index] for pair in pairs for index in pair]
    base = SentenceTransformer(str(base_model), device="cpu")
    fine = SentenceTransformer(str(checkpoint), device="cpu")
    base_emb = base.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    fine_emb = fine.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    base_scores = np.sum(base_emb[0::2] * base_emb[1::2], axis=1)
    fine_scores = np.sum(fine_emb[0::2] * fine_emb[1::2], axis=1)
    category_col = "category_path" if "category_path" in df.columns else "category"
    brand = df["brand"].fillna("").astype(str).tolist()
    category = df[category_col].fillna("").astype(str).tolist()
    rows = [
        {
            "pair": pair_index,
            "row_a": left,
            "row_b": right,
            "brand_a": brand[left],
            "brand_b": brand[right],
            "category_a": category[left],
            "category_b": category[right],
            "payload_a": payload[left],
            "payload_b": payload[right],
            "zero_shot_cosine": float(base_scores[pair_index]),
            "fine_tuned_cosine": float(fine_scores[pair_index]),
            "delta": float(fine_scores[pair_index] - base_scores[pair_index]),
            "payload_token_overlap": 0,
        }
        for pair_index, (left, right) in enumerate(pairs)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "uniformity_unrelated_pairs.csv", index=False)
    result = {
        "base_model": str(base_model),
        "fine_tuned_checkpoint": str(checkpoint),
        "selection": "different brand, different category, zero shared exact payload tokens",
        "seed": int(seed),
        "zero_shot": _summary(base_scores, threshold),
        "fine_tuned": _summary(fine_scores, threshold),
        "delta": _summary(fine_scores - base_scores, threshold),
    }
    (out / "uniformity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(base_scores, bins=20, alpha=0.55, label="zero-shot")
    axis.hist(fine_scores, bins=20, alpha=0.55, label="fine-tuned")
    axis.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    axis.set(
        xlabel="cosine similarity",
        ylabel="count",
        title="Unrelated-pair uniformity: zero-shot vs fine-tuned",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "uniformity_cosine_comparison.png", dpi=150)
    plt.close(fig)
    return result
