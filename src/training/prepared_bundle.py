"""Local-prepared training bundles shared with GPU-only workers.

The bundle is intentionally produced by the normal training.train data path,
so payload, pair, masking, country, and calibration inputs keep one
implementation. Colab workers consume the immutable result and never rebuild
those CPU-side inputs.
"""

from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PreparedBundleManifest(BaseModel):
    """Machine-checked identity and shape contract for a prepared bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    payload_variant: str = Field(min_length=1)
    masking_profile: str = Field(min_length=1)
    n_df: int = Field(ge=1)
    n_payload: int = Field(ge=1)
    n_pos: int = Field(ge=1)
    n_neg: int = Field(ge=0)
    n_train_neg: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_prepared_bundle(
    path: Path,
    *,
    df: pd.DataFrame,
    payload: list[str],
    structured_features: np.ndarray,
    row_bc: np.ndarray,
    country: np.ndarray,
    pos: np.ndarray,
    hp_pairs: np.ndarray,
    emb0: np.ndarray,
    neg: np.ndarray,
    train_neg: np.ndarray,
    neg_sources: np.ndarray,
    train_neg_sources: np.ndarray,
    mask_audit: list[dict[str, Any]],
    hard_negative_mask_audit: list[dict[str, Any]],
    payload_variant: str,
    masking_profile: str,
) -> PreparedBundleManifest:
    """Write one compressed, self-contained, locally generated input bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload_data = {
        "df": df,
        "payload": payload,
        "structured_features": structured_features,
        "row_bc": row_bc,
        "country": country,
        "pos": pos,
        "hp_pairs": hp_pairs,
        "emb0": emb0,
        "neg": neg,
        "train_neg": train_neg,
        "neg_sources": neg_sources,
        "train_neg_sources": train_neg_sources,
        "mask_audit": mask_audit,
        "hard_negative_mask_audit": hard_negative_mask_audit,
        "payload_variant": payload_variant,
        "masking_profile": masking_profile,
    }
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump(payload_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = PreparedBundleManifest(
        payload_variant=payload_variant,
        masking_profile=masking_profile,
        n_df=len(df),
        n_payload=len(payload),
        n_pos=len(pos),
        n_neg=len(neg),
        n_train_neg=len(train_neg),
        sha256=_digest(path),
    )
    path.with_suffix(path.suffix + ".json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_prepared_bundle(path: Path) -> tuple[PreparedBundleManifest, dict[str, Any]]:
    """Load and validate a bundle before it crosses into the training lane."""

    if not path.is_file():
        raise FileNotFoundError(f"prepared training bundle missing: {path}")
    manifest_path = path.with_suffix(path.suffix + ".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared bundle manifest missing: {manifest_path}")
    manifest = PreparedBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    actual_digest = _digest(path)
    if actual_digest != manifest.sha256:
        raise ValueError(
            f"prepared bundle SHA-256 mismatch: {path} "
            f"{actual_digest} != {manifest.sha256}"
        )
    with gzip.open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError("prepared training bundle must contain a mapping")
    required = {
        "df", "payload", "structured_features", "row_bc", "country", "pos",
        "hp_pairs", "emb0", "neg", "train_neg", "neg_sources",
        "train_neg_sources", "mask_audit", "hard_negative_mask_audit",
        "payload_variant", "masking_profile",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"prepared training bundle missing fields: {missing}")
    if len(data["df"]) != manifest.n_df or len(data["payload"]) != manifest.n_payload:
        raise ValueError("prepared bundle manifest/data row counts disagree")
    if len(data["pos"]) != manifest.n_pos or len(data["neg"]) != manifest.n_neg:
        raise ValueError("prepared bundle manifest/pair counts disagree")
    if len(data["train_neg"]) != manifest.n_train_neg:
        raise ValueError("prepared bundle manifest/training-negative counts disagree")
    if data["payload_variant"] != manifest.payload_variant:
        raise ValueError("prepared bundle payload variant disagrees with manifest")
    if data["masking_profile"] != manifest.masking_profile:
        raise ValueError("prepared bundle masking profile disagrees with manifest")
    return manifest, data
