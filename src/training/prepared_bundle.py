"""Local-prepared training bundles shared with GPU-only workers.

The bundle is intentionally produced by the normal training.train data path,
so payload, pair, masking, country, and calibration inputs keep one
implementation. Colab workers consume the immutable result and never rebuild
those CPU-side inputs.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.schemas import TrainingSpec


class PreparedBundleManifest(BaseModel):
    """Machine-checked identity and shape contract for a prepared bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "3"
    payload_variant: str = Field(min_length=1)
    masking_profile: str = Field(min_length=1)
    # The encoder-text composition the frozen payload was built with. A bundle
    # is a frozen list of strings, so reusing one after the composition moves
    # would train on text no lane produces any more — the manifest names the
    # contract and load_prepared_bundle refuses a mismatch instead of silently
    # training on the wrong payload. (schema_version 2 bundles have no such
    # field and are rejected loudly by extra="forbid".)
    model_input: TrainingSpec.ModelInputSpec
    n_df: int = Field(ge=1)
    n_payload: int = Field(ge=1)
    n_pos: int = Field(ge=1)
    n_neg: int = Field(ge=0)
    n_train_neg: int = Field(ge=0)
    n_labeled_pairs_bytes: int = Field(ge=1)
    n_canonical_records_bytes: int = Field(ge=1)
    n_gate_results_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Diet/augmentation provenance: the resolved masking profile and the
    # easy-negative quota the bundle was built under. config/ can drift
    # after a build (ratio_to_hard moves the diet verdict without touching
    # a byte), so the manifest pins what was true at build time and the
    # loader warns loudly on drift. Optional: pre-feature bundles load
    # with an empty record and skip the check.
    masking_config: dict = Field(default_factory=dict)
    easy_config: dict = Field(default_factory=dict)
    # Ratio contracts (audit 2026-09-28): the train-time arithmetic the
    # diet gate enforced, frozen into the header so any reader can verify
    # the contract without rerunning the gate. Legacy headers load with
    # an explicit warning; the diet gate recomputes its own verdict.
    ratio_to_hard: float = Field(default=0.0, ge=0.0)
    static_view_ratio: float = Field(default=0.0, ge=0.0)
    effective_train_ratio: float = Field(default=0.0, ge=0.0)
    ratio_contract_note: str = Field(default="legacy ratio metadata; recompute")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_PREPARED_BUNDLE_DRIFT_STRICT_TRUTHY = {"1", "true", "yes", "on"}


def prepared_bundle_drift_strict() -> bool:
    """The prepared_bundle_drift_strict switch (default FALSE).

    Env PREPARED_BUNDLE_DRIFT_STRICT wins when set; otherwise the
    prepared_bundle_drift_strict key in config/training.yaml. TRUE hard-fails
    load_prepared_bundle on a bundle built under deleted/drifted masking
    knobs instead of warning; keep FALSE until the bundle rebuild lands.
    """
    raw = os.environ.get("PREPARED_BUNDLE_DRIFT_STRICT")
    if raw is not None:
        return raw.strip().lower() in _PREPARED_BUNDLE_DRIFT_STRICT_TRUTHY
    from core.common import load_config

    return bool(load_config().get("prepared_bundle_drift_strict", False))


def prepared_holdout(data: dict, split_cfg: dict, *, seed: int):
    """Reuse a frozen parent split for sampled smokes; otherwise derive SSOT."""
    from training.folds import derive_holdout, normalize_gtin
    frozen = data.get('holdout_populations')
    if frozen is None:
        return derive_holdout(data['pos'], data['row_bc'], split_cfg, seed=seed)
    if set(frozen) != {'train','dev','test'}:
        raise ValueError('frozen holdout needs train/dev/test')
    populations = tuple(set(frozen[split]) for split in ('train','dev','test'))
    roles = {}
    for role, values in enumerate(populations):
        for value in values:
            key = normalize_gtin(value)
            if key in roles and roles[key]!=role:
                raise ValueError('frozen holdout contains overlapping entities')
            roles[key] = role
    for value in data['row_bc']:
        key = normalize_gtin(value)
        if key and key not in roles:
            raise ValueError('frozen holdout misses a payload entity')
    for a,b in data['pos']:
        ka,kb = normalize_gtin(data['row_bc'][a]),normalize_gtin(data['row_bc'][b])
        if roles.get(ka)!=roles.get(kb):
            raise ValueError('positive pair crosses frozen holdout')
    return populations


def _validate_augmented_features(payload, features, audit) -> None:
    if len(features) != len(payload):
        raise ValueError("prepared bundle feature/payload row counts disagree")
    if not audit:
        return
    from training.masking import extend_augmented_features

    first_copy = min(int(row["copy_payload_idx"]) for row in audit)
    expected = extend_augmented_features(features[:first_copy], payload, audit)
    if not np.array_equal(np.asarray(features), expected):
        raise ValueError(
            "prepared bundle augmentation features disagree with payload lineage; "
            "re-prepare or repair the bundle before training"
        )


def _validate_counterfactual_audits(payload, audit) -> None:
    """Reject stored twin negatives whose claimed field flip is compatible."""
    twins = [row for row in audit if row.get("target_mode") == "counterfactual"]
    if not twins:
        return
    from training.masking import _field_surfaces, _field_values_conflict

    invalid = []
    surfaces = {}
    for row in twins:
        copy_i = int(row["copy_payload_idx"])
        pair_i = int(row["pair_payload_idx"])
        for idx in (copy_i, pair_i):
            if idx not in surfaces:
                surfaces[idx] = _field_surfaces(payload[idx])
        if not any(
            field in surfaces[copy_i]
            and field in surfaces[pair_i]
            and _field_values_conflict(field, surfaces[copy_i][field], surfaces[pair_i][field])
            for field in (row.get("fields_hit") or [])
        ):
            invalid.append((copy_i, pair_i))
    if invalid:
        raise ValueError(
            "prepared bundle contains counterfactual twins without a verified "
            f"semantic conflict (first pairs: {invalid[:5]}); regenerate the bundle"
        )


def canonical_payload_rows(n_source: int, payload: list[str], row_bc: np.ndarray) -> np.ndarray:
    """Validate the native source/canonical/augmentation layout for retrieval.

    Preparation appends the complete canonical map in sorted order after
    source rows. Augmented copies follow that block and are never candidates.
    """
    from pipeline import load_canonical_map
    canon_map = load_canonical_map()
    end = n_source + len(canon_map)
    if end > len(payload):
        raise ValueError(
            f"canonical block [{n_source}, {end}) exceeds the payload "
            f"({len(payload)} entries) — the payload layout changed"
        )
    rows = np.arange(n_source, end, dtype=int)
    if {str(b) for b in row_bc[rows]} != set(canon_map):
        raise ValueError(
            "canonical payload block does not carry the canonical map's "
            "GTINs — refusing to build the competitor universe from rows "
            "that are not canonicals"
        )
    return rows


def _portable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Freeze values without pandas-version-specific string dtype metadata."""
    result = df.astype(object)
    result.columns = pd.Index(df.columns.to_numpy(dtype=object), dtype=object)
    if isinstance(df.index.dtype, pd.StringDtype):
        result.index = pd.Index(df.index.to_numpy(dtype=object), dtype=object)
    return result


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
    labeled_pairs_csv: bytes,
    canonical_records_csv: bytes,
    gate_results_csv: bytes,
    payload_variant: str,
    masking_profile: str,
    holdout_populations: dict[str, list[str]] | None = None,
    token_checkpoint: str | None = None,
    training_tokens: dict | None = None,
    plan_loss: str | None = None,
    plan_train_frac: float = 1.0,
    plan_sample: bool = False,
) -> PreparedBundleManifest:
    """Write one compressed, self-contained, locally generated input bundle."""

    from core.common import load_config, masking_cfg
    from core.model_input import model_input_spec

    recorded_masking = masking_cfg(str(masking_profile))
    _validate_augmented_features(
        payload, structured_features, mask_audit + hard_negative_mask_audit
    )
    _validate_counterfactual_audits(payload, hard_negative_mask_audit)
    recorded_easy = dict(load_config()["training"]["random_easy_negatives"])
    easy_ratio = float(recorded_easy["ratio_to_hard"])
    easy_enabled = bool(recorded_easy["enabled"])
    static_views = len(pos) / max(len(train_neg), 1)
    # Store the guaranteed bundle-only ratio. Easy-negative settings are
    # recorded separately as a possible contrastive projection, since the
    # split-local sampler may return no candidates.
    effective_views = len(train_neg)
    effective_ratio = len(pos) / max(effective_views, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    portable_df = _portable_dataframe(df)
    payload_data = {
        # Colab and preparation hosts can use different pandas versions.
        # Plain object columns avoid pickling version-specific StringDtype
        # constructors while preserving the frozen values and row order.
        "df": portable_df,
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
        "labeled_pairs_csv": labeled_pairs_csv,
        "canonical_records_csv": canonical_records_csv,
        "gate_results_csv": gate_results_csv,
        "payload_variant": payload_variant,
        "masking_profile": masking_profile,
    }
    if token_checkpoint is not None:
        from core.common import load_local_sentence_transformer
        from training.token_inputs import prepare_training_tokens
        token_model = load_local_sentence_transformer(str(token_checkpoint), device="cpu")
        training_tokens = prepare_training_tokens(token_model, payload)
        del token_model
    if training_tokens is not None:
        from training.token_inputs import validate_training_tokens
        validate_training_tokens(training_tokens)
        payload_data["training_tokens"] = training_tokens
    _validate_bundle_arrays(payload_data)
    if holdout_populations is not None:
        payload_data['holdout_populations'] = holdout_populations
    if token_checkpoint is not None:
        from training.run_plan import prepare_run_plan
        payload_data["training_plan"] = prepare_run_plan(payload_data, loss=plan_loss, train_frac=plan_train_frac, sample=plan_sample)
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump(payload_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = PreparedBundleManifest(
        payload_variant=payload_variant,
        masking_profile=masking_profile,
        model_input=model_input_spec(),
        masking_config=recorded_masking,
        easy_config=recorded_easy,
        ratio_to_hard=easy_ratio,
        static_view_ratio=float(static_views),
        effective_train_ratio=float(effective_ratio),
        ratio_contract_note=(
            f"Guaranteed bundle-only view ratio {static_views:.3f}; easy-negative "
            f"settings ({'enabled' if easy_enabled else 'disabled'}, ratio={easy_ratio:g}) "
            "are a possible contrastive projection and are not guaranteed."
        ),
        n_df=len(df),
        n_payload=len(payload),
        n_pos=len(pos),
        n_neg=len(neg),
        n_train_neg=len(train_neg),
        n_labeled_pairs_bytes=len(labeled_pairs_csv),
        n_canonical_records_bytes=len(canonical_records_csv),
        n_gate_results_bytes=len(gate_results_csv),
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
    _validate_bundle_arrays(data)
    required = {
        "df", "payload", "structured_features", "row_bc", "country", "pos",
        "hp_pairs", "emb0", "neg", "train_neg", "neg_sources",
        "train_neg_sources", "mask_audit", "hard_negative_mask_audit",
        "labeled_pairs_csv", "canonical_records_csv", "gate_results_csv",
        "payload_variant", "masking_profile",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"prepared training bundle missing fields: {missing}")
    _validate_augmented_features(
        data["payload"], data["structured_features"],
        data["mask_audit"] + data["hard_negative_mask_audit"],
    )
    _validate_counterfactual_audits(data["payload"], data["hard_negative_mask_audit"])
    if len(data["df"]) != manifest.n_df or len(data["payload"]) != manifest.n_payload:
        raise ValueError("prepared bundle manifest/data row counts disagree")
    if len(data["pos"]) != manifest.n_pos or len(data["neg"]) != manifest.n_neg:
        raise ValueError("prepared bundle manifest/pair counts disagree")
    if len(data["train_neg"]) != manifest.n_train_neg:
        raise ValueError("prepared bundle manifest/training-negative counts disagree")
    if (
        not isinstance(data["labeled_pairs_csv"], bytes)
        or len(data["labeled_pairs_csv"]) != manifest.n_labeled_pairs_bytes
    ):
        raise ValueError("prepared bundle labeled-pairs bytes disagree with manifest")
    for field, expected in (
        ("canonical_records_csv", manifest.n_canonical_records_bytes),
        ("gate_results_csv", manifest.n_gate_results_bytes),
    ):
        if not isinstance(data[field], bytes) or len(data[field]) != expected:
            raise ValueError(f"prepared bundle {field} bytes disagree with manifest")
    if data["payload_variant"] != manifest.payload_variant:
        raise ValueError("prepared bundle payload variant disagrees with manifest")
    if data["masking_profile"] != manifest.masking_profile:
        raise ValueError("prepared bundle masking profile disagrees with manifest")
    from core.model_input import model_input_spec

    active = model_input_spec()
    if manifest.model_input != active:
        raise ValueError(
            "prepared training bundle was built with a different encoder-text "
            f"composition: bundle={manifest.model_input.model_dump()} "
            f"active={active.model_dump()}. Re-prepare the bundle; the frozen "
            "payload strings are not the text the active composition produces."
        )
    if manifest.masking_config or manifest.easy_config:
        from core.common import load_config, masking_cfg

        drifted: list[str] = []
        if manifest.masking_config and manifest.masking_config != masking_cfg(
            str(manifest.masking_profile)
        ):
            drifted.append("masking")
        if manifest.easy_config and manifest.easy_config != dict(
            load_config()["training"]["random_easy_negatives"]
        ):
            drifted.append("random_easy_negatives")
        if drifted:
            detail = (
                f"{path} was built under different {'/'.join(drifted)} "
                f"config than active (bundle={manifest.masking_config} "
                f"{manifest.easy_config}). Diet verdicts and augmentation "
                "yields may not reproduce."
            )
            if prepared_bundle_drift_strict():
                raise ValueError(
                    f"[bundle-drift] STRICT: {detail} Re-prepare the bundle; "
                    "unset PREPARED_BUNDLE_DRIFT_STRICT only to keep "
                    "pre-rebuild audit findings reproducible."
                )
            print(f"[bundle-drift] WARNING: {detail}", flush=True)
    if (
        "dynamic easy-negative joining at step execution" in manifest.ratio_contract_note
        or manifest.ratio_contract_note == "legacy ratio metadata; recompute"
    ):
        print(
            f"[bundle-drift] WARNING: {path} carries a legacy projected easy-negative "
            "ratio claim; it is not treated as guaranteed. The diet gate recomputes "
            "its verdict from selected bundle pairs.",
            flush=True,
        )
    return manifest, data


def _validate_bundle_arrays(data):
    """Reject malformed CPU inputs before casts or GPU compute hide defects."""
    required = {"payload", "row_bc", "country", "structured_features", "pos", "hp_pairs", "neg", "train_neg", "emb0", "neg_sources", "train_neg_sources"}
    if not required <= set(data):
        raise ValueError(f"prepared bundle missing array fields: {sorted(required - set(data))}")
    size = len(data["payload"])
    for key in ("row_bc", "country"):
        if np.asarray(data[key]).ndim != 1 or len(data[key]) != size:
            raise ValueError(f"prepared bundle {key} must cover every payload row")
    features = np.asarray(data["structured_features"])
    if features.ndim != 2 or len(features) != size or features.dtype.kind != "f" or not np.isfinite(features).all():
        raise ValueError("prepared structured features need finite float rows for every payload")
    for key in ("pos", "hp_pairs", "neg", "train_neg"):
        pairs = np.asarray(data[key])
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
            raise ValueError(f"prepared {key} needs integer (n,2) pairs")
        if np.any(pairs < 0) or np.any(pairs >= size):
            raise ValueError(f"prepared {key} contains out-of-bounds payload indices")
    for key, pairs in (("neg_sources", "neg"), ("train_neg_sources", "train_neg")):
        if np.asarray(data[key]).ndim != 1 or len(data[key]) != len(data[pairs]):
            raise ValueError(f"prepared {key} must align with {pairs}")
    embeddings = np.asarray(data["emb0"])
    if embeddings.ndim != 2 or (embeddings.size and len(embeddings) != size) or not np.isfinite(embeddings).all():
        raise ValueError("prepared initial embeddings need finite aligned rows or an empty matrix")
    if "training_tokens" in data:
        from training.token_inputs import validate_training_tokens
        validate_training_tokens(data["training_tokens"])
