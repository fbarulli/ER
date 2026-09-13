"""Checkpoint-level unrelated-pair uniformity diagnostics.

This is deliberately separate from the label-based pair metrics.  A model can
score its mined positives/negatives well while mapping unrelated catalog text
to one narrow high-cosine region.  The audit uses the exact model payload,
requires different brands/categories and zero shared payload tokens, and
records the median, P90, std, and threshold-crossing rate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from core.common import load_local_sentence_transformer, plot_dpi

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def select_unrelated_pairs(
    df: pd.DataFrame,
    payload: list[str],
    *,
    n_pairs: int,
    seed: int,
    max_token_frequency: float,
) -> list[tuple[int, int]]:
    """Select deterministic, payload-disjoint unrelated SKU pairs.

    ``max_token_frequency`` excludes overly common tokens (units, connector
    words, numeric fragments shared across most of the catalog) from the
    blocking rule. Without this, a handful of near-universal tokens can
    silently starve selection on catalogs where they're common, producing
    ``insufficient_pairs`` even when the catalog has plenty of genuinely
    unrelated items available.
    """
    if n_pairs < 0:
        raise ValueError("n_pairs must be non-negative")
    if n_pairs == 0:
        return []
    if len(payload) != len(df):
        raise ValueError(
            "uniformity payload/data alignment mismatch: "
            f"rows={len(df)} payload={len(payload)}"
        )
    category_col = "category_path" if "category_path" in df.columns else "category"
    brand = df["brand"].fillna("").astype(str).str.strip().str.lower().tolist()
    category = (
        df[category_col].fillna("").astype(str).str.strip().str.lower().tolist()
    )
    tokens = [_tokens(text) for text in payload]
    valid = [
        i
        for i in range(len(df))
        if tokens[i] and brand[i] and category[i]
    ]
    order = list(np.random.default_rng(seed).permutation(valid))
    order_rank = {row: rank for rank, row in enumerate(order)}
    by_brand: dict[str, set[int]] = {}
    by_category: dict[str, set[int]] = {}
    by_token: dict[str, set[int]] = {}
    for row in valid:
        by_brand.setdefault(brand[row], set()).add(row)
        by_category.setdefault(category[row], set()).add(row)
        for token in tokens[row]:
            by_token.setdefault(token, set()).add(row)

    # Drop overly common tokens from blocking. High-frequency tokens (units,
    # connector words, generic numeric fragments) don't indicate relatedness;
    # they indicate shared boilerplate. Blocking on them starves the sampler
    # without improving pair quality.
    if valid:
        frequency_limit = max_token_frequency * len(valid)
        by_token = {
            tok: rows for tok, rows in by_token.items() if len(rows) <= frequency_limit
        }

    selected: list[tuple[int, int]] = []
    available = set(valid)
    for left in order:
        if left not in available:
            continue
        blocked = {left} | by_brand[brand[left]] | by_category[category[left]]
        for token in tokens[left]:
            blocked.update(by_token.get(token, ()))
        eligible = available - blocked
        if not eligible:
            continue
        right = min(eligible, key=order_rank.__getitem__)
        selected.append((left, right))
        available.difference_update((left, right))
        if len(selected) >= n_pairs:
            return selected
    # This is a diagnostic, not a training or publication gate. Return the
    # available deterministic sample so small/overlapping catalogs can still
    # finalize and record an explicit insufficient-sample result.
    return selected


def _summary(
    scores: np.ndarray,
    operating_threshold: float,
    crossing_rate_ceiling: float,
) -> dict[str, float | int]:
    crossing_rate = float(np.mean(scores >= operating_threshold))
    return {
        "n_pairs": int(len(scores)),
        "median_cosine": float(np.median(scores)),
        "p90_cosine": float(np.quantile(scores, 0.90)),
        "std_cosine": float(np.std(scores)),
        "operating_threshold": float(operating_threshold),
        "crossing_rate_ceiling": float(crossing_rate_ceiling),
        "threshold_crossing_rate": crossing_rate,
        "threshold_crossing_rate_flag": int(crossing_rate > crossing_rate_ceiling),
        "n_above_threshold": int(np.sum(scores >= operating_threshold)),
    }


def _write_pair_trace(
    path: str | Path,
    df: pd.DataFrame,
    payload: list[str],
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    evaluation_step: int | None,
) -> None:
    """Append sampled unrelated pairs with source metadata for live audits."""
    rows: list[dict[str, object]] = []
    for pair_number, ((left, right), score) in enumerate(zip(pairs, scores, strict=True)):
        record: dict[str, object] = {
            "evaluation_step": evaluation_step,
            "pair_number": pair_number,
            "cosine": float(score),
        }
        for side, row in (("a", left), ("b", right)):
            record[f"{side}_row_index"] = int(row)
            record[f"{side}_payload"] = payload[row]
            record.update(
                {
                    f"{side}_{key}": value
                    for key, value in df.iloc[row].to_dict().items()
                }
            )
        rows.append(record)
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        trace_path,
        mode="a",
        header=not trace_path.exists(),
        index=False,
    )


def collapse_diagnostics(
    *,
    model,
    df: pd.DataFrame,
    payload: list[str],
    config: dict,
    batch_size: int,
    requested: bool = True,
    allow_unavailable: bool = True,
    trace_path: str | Path | None = None,
    evaluation_step: int | None = None,
) -> dict[str, float | int | str]:
    """Measure unrelated-pair collapse for a live or calibration checkpoint.

    ``guardrail["operating_threshold"]`` lets this gate reflect the actual
    decision boundary that will be used downstream, not just distributional
    drift in median/P90/std. Median/P90 shifting is a symptom; the
    threshold-crossing rate at the real operating point is what maps directly
    to downstream matching risk (over-merging).
    """
    guardrail = config["collapse_guardrail"]
    if not requested or not bool(guardrail["enabled"]):
        return {
            "collapse_guardrail_enabled": 0,
            "collapse_status": "not_requested" if not requested else "disabled",
            "collapse_requested_pairs": int(guardrail["unrelated_pairs"]),
            "collapse_unrelated_pairs": 0,
            "collapse_operating_threshold": float(guardrail["operating_threshold"]),
            "collapse_crossing_rate_ceiling": float(
                guardrail["crossing_rate_ceiling"]
            ),
            "collapse_diagnostics_available": 0,
            "collapse_healthy": 0,
        }
    requested_pairs = int(guardrail["unrelated_pairs"])
    max_token_frequency = float(guardrail["max_token_frequency"])
    pairs = select_unrelated_pairs(
        df,
        payload,
        n_pairs=requested_pairs,
        seed=int(guardrail["seed"]),
        max_token_frequency=max_token_frequency,
    )
    if not pairs:
        if allow_unavailable:
            return {
                "collapse_guardrail_enabled": 1,
                "collapse_status": "unavailable",
                "collapse_requested_pairs": requested_pairs,
                "collapse_unrelated_pairs": 0,
                "collapse_operating_threshold": float(
                    guardrail["operating_threshold"]
                ),
                "collapse_crossing_rate_ceiling": float(
                    guardrail["crossing_rate_ceiling"]
                ),
                "collapse_diagnostics_available": 0,
                "collapse_healthy": 0,
            }
        raise RuntimeError(
            "collapse guardrail could not construct its configured unrelated-pair "
            f"sample: required={requested_pairs} selected=0"
        )
    sample_status = "ok" if len(pairs) == requested_pairs else "insufficient_pairs"
    if sample_status == "insufficient_pairs":
        print(
            "[collapse-guardrail] insufficient unrelated pairs; "
            f"requested={requested_pairs} selected={len(pairs)}",
            flush=True,
        )
    pair_array = np.asarray(pairs, dtype=int)
    was_training = bool(getattr(model, "training", False))
    try:
        embeddings = model.encode(
            [payload[int(row)] for row in pair_array.ravel()],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    finally:
        model.train(was_training)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, np.finfo(float).eps)
    scores = np.sum(normalized[0::2] * normalized[1::2], axis=1)
    if trace_path is not None:
        _write_pair_trace(
            trace_path,
            df,
            payload,
            pairs,
            scores,
            evaluation_step,
        )
    median = float(np.median(scores))
    p90 = float(np.quantile(scores, 0.90))
    std = float(np.std(scores))

    operating_threshold = float(guardrail["operating_threshold"])
    crossing_rate_ceiling = float(guardrail["crossing_rate_ceiling"])
    crossing_rate = float(np.mean(scores >= operating_threshold))

    healthy = int(
        sample_status == "ok"
        and median <= float(guardrail["median_penalty_start"])
        and p90 <= float(guardrail["p90_penalty_start"])
        and std >= float(guardrail["cosine_std_floor"])
        and crossing_rate <= crossing_rate_ceiling
    )
    return {
        "collapse_guardrail_enabled": 1,
        "collapse_status": sample_status,
        "collapse_requested_pairs": requested_pairs,
        "collapse_unrelated_pairs": int(len(scores)),
        "collapse_median_cosine": median,
        "collapse_p90_cosine": p90,
        "collapse_cosine_std": std,
        "collapse_embedding_norm_mean": float(np.mean(norms)),
        "collapse_embedding_norm_std": float(np.std(norms)),
        "collapse_median_flag": int(median > float(guardrail["median_penalty_start"])),
        "collapse_p90_flag": int(p90 > float(guardrail["p90_penalty_start"])),
        "collapse_operating_threshold": operating_threshold,
        "collapse_crossing_rate": crossing_rate,
        "collapse_crossing_rate_ceiling": crossing_rate_ceiling,
        "collapse_crossing_rate_flag": int(crossing_rate > crossing_rate_ceiling),
        "collapse_diagnostics_available": 1,
        "collapse_healthy": healthy,
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
    operating_threshold: float,
    crossing_rate_ceiling: float,
    max_token_frequency: float,
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
        df,
        payload,
        n_pairs=n_pairs,
        seed=seed,
        max_token_frequency=max_token_frequency,
    )
    # n_pairs == 0 selects nothing (see select_unrelated_pairs), so the strict
    # comparison used to fall through to _summary() with empty score arrays and
    # die in np.quantile; ask for at least one pair to report a selection.
    if len(pairs) < max(1, n_pairs):
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
    base = load_local_sentence_transformer(str(base_model), device="cpu")
    fine = load_local_sentence_transformer(str(checkpoint), device="cpu")
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
        "zero_shot": _summary(
            base_scores, operating_threshold, crossing_rate_ceiling
        ),
        "fine_tuned": _summary(
            fine_scores, operating_threshold, crossing_rate_ceiling
        ),
        "delta": _summary(
            fine_scores - base_scores, operating_threshold, crossing_rate_ceiling
        ),
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
    axis.axvline(
        operating_threshold,
        color="black",
        linestyle="--",
        label=f"threshold={operating_threshold:.2f}",
    )
    axis.set(
        xlabel="cosine similarity",
        ylabel="count",
        title="Unrelated-pair uniformity: zero-shot vs fine-tuned",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "uniformity_cosine_comparison.png", dpi=plot_dpi())
    plt.close(fig)
    return result
