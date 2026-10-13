"""core/prediction_export.py — the per-sample prediction artifact of an eval.

The traceability gap this closes: the identity/holdout eval (``training.
evaluate_models``) *computed* a per-pair prediction for every scored pair and
then threw it away — only the aggregate ``model_evaluation_summary.csv``
survived, plus a hand-drawn sample of three false positives and three false
negatives printed to stdout. No artifact named the pair behind a metric, so
"why did this model miss this sample?" was unanswerable after the fact.

:class:`PredictionExport` owns one row per scored pair, keyed by the SAME
``core.pair_identity.PairIdentity`` id the gate, labeled and validation
artifacts carry, and joined to the validation frame's ``v1_*``/``v2_*`` slice
columns. The file parallels the trained lane's
``train_<model>_<run>_fold<f>_pairs.csv`` dump — one row per scored pair with
``fold`` / ``label`` / ``score`` — under
``eval_<model>_<half>_fold<f>_pairs.csv``.

The frame contract is validated in :meth:`PredictionExport.validate` before the
caller writes it (the repo's ``check_*_frame`` convention): the pair key is
non-empty and unique, every required column is present, and every scored pair
resolved to a slice row. An unscored slice coverage gap fails loud rather than
shipping a predictions file that cannot be joined.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.pair_identity import PairIdentity


class PredictionExport:
    """One model's per-scored-pair predictions, joined by ``pair_id``."""

    #: Every emitted prediction row carries these, in this order.
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "pair_id",
        "model",
        "eval_half",
        "fold",
        "gtin1",
        "gtin2",
        "gtin1_norm",
        "gtin2_norm",
        "label",
        "score",
        "predicted",
        "threshold",
    )
    #: Carried verbatim from the scored frame when it has them (the labeled
    #: census has no SKU columns today; the trained lane's dump does).
    OPTIONAL_COLUMNS: tuple[str, ...] = ("sku_id_a", "sku_id_b")
    #: The validation frame's per-side attribute columns, carried so a slice
    #: can be read off the same row as the prediction.
    SLICE_PREFIXES: tuple[str, ...] = ("v1_", "v2_")

    @classmethod
    def slice_columns(cls, final_validation: pd.DataFrame) -> list[str]:
        """The ``v1_*``/``v2_*`` columns the validation frame declares."""
        return [
            name
            for name in final_validation.columns
            if name.startswith(cls.SLICE_PREFIXES)
        ]

    @classmethod
    def slice_frame(cls, final_validation: pd.DataFrame) -> pd.DataFrame:
        """The validation frame reduced to ``pair_id`` + its slice columns.

        The key is recomputed through the SSOT (never read off the artifact):
        a frame produced before the ``pair_id`` column existed joins exactly
        like a fresh one.
        """
        out = pd.DataFrame(
            {
                "pair_id": PairIdentity.column(
                    final_validation["gtin1"], final_validation["gtin2"]
                ).to_numpy()
            }
        )
        for name in cls.slice_columns(final_validation):
            out[name] = final_validation[name].to_numpy()
        return out

    @classmethod
    def frame(
        cls,
        scored: pd.DataFrame,
        slices: pd.DataFrame,
        *,
        model: str,
        eval_half: str,
        score_column: str,
        threshold: float,
    ) -> pd.DataFrame:
        """The prediction rows of one model on one scored half.

        ``scored`` is the half's frame carrying ``gtin1``/``gtin2``,
        ``true_label``, ``fold`` and ``score_column``; ``slices`` is
        :meth:`slice_frame`'s output.
        """
        out = pd.DataFrame(
            {
                "pair_id": PairIdentity.column(
                    scored["gtin1"], scored["gtin2"]
                ).to_numpy(),
                "model": model,
                "eval_half": eval_half,
                "fold": scored["fold"].to_numpy(),
                "gtin1": scored["gtin1"].to_numpy(),
                "gtin2": scored["gtin2"].to_numpy(),
                "gtin1_norm": scored["gtin1"].map(PairIdentity.endpoint_key).to_numpy(),
                "gtin2_norm": scored["gtin2"].map(PairIdentity.endpoint_key).to_numpy(),
                "label": scored["true_label"].astype(int).to_numpy(),
                "score": scored[score_column].astype(float).to_numpy(),
            }
        )
        out["predicted"] = (out["score"] >= float(threshold)).astype(int)
        out["threshold"] = float(threshold)
        for name in cls.OPTIONAL_COLUMNS:
            if name in scored.columns:
                out[name] = scored[name].to_numpy()
        slice_columns = [
            name for name in slices.columns if name != "pair_id"
        ]
        merged = out.merge(
            slices[["pair_id", *slice_columns]], on="pair_id", how="left"
        )
        return cls.validate(merged, slice_columns=slice_columns)

    @classmethod
    def validate(
        cls, frame: pd.DataFrame, *, slice_columns: list[str]
    ) -> pd.DataFrame:
        """The frame contract, asserted before the write (fail loud)."""
        expected = [
            *cls.REQUIRED_COLUMNS,
            *(c for c in cls.OPTIONAL_COLUMNS if c in frame.columns),
            *slice_columns,
        ]
        if tuple(frame.columns) != tuple(expected):
            raise ValueError(
                f"prediction frame columns {tuple(frame.columns)} != contract "
                f"{tuple(expected)}"
            )
        if len(frame) == 0:
            raise ValueError("prediction frame is empty — nothing was scored")
        if frame["pair_id"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("prediction frame has an empty pair_id")
        duplicates = int(frame["pair_id"].duplicated().sum())
        if duplicates:
            raise ValueError(f"{duplicates} duplicate pair_id rows")
        if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
            raise ValueError("prediction frame has a non-finite score")
        if slice_columns:
            missing = int(frame[slice_columns[0]].isna().sum())
            if missing:
                raise ValueError(
                    f"{missing} scored pairs have no validation slice row — the "
                    "prediction cannot be joined to the sample it scored"
                )
        if not bool(
            (frame["predicted"] == (frame["score"] >= frame["threshold"]).astype(int)).all()
        ):
            raise ValueError("predicted disagrees with score/threshold")
        return frame

    @classmethod
    def filename(cls, model: str, eval_half: str, fold: int) -> str:
        """``eval_<model>_<half>_fold<f>_pairs.csv``, paralleling the trained dump."""
        return f"eval_{model}_{eval_half}_fold{int(fold)}_pairs.csv"

    @classmethod
    def write(cls, frame: pd.DataFrame, path: Path) -> Path:
        """Atomic write of a validated frame (``core.manifest`` mechanism)."""
        from core.manifest import atomic_write_csv

        atomic_write_csv(frame, path, index=False)
        return path
