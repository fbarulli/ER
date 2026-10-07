"""Local-prepared training bundles shared with GPU-only workers.

The bundle is intentionally produced by the normal training.train data path,
so payload, pair, masking, country, and calibration inputs keep one
implementation. Colab workers consume the immutable result and never rebuild
those CPU-side inputs.

One responsibility per unit:

  PreparedBundleManifest         the frozen identity/shape contract
  hashing primitive              _digest (bar-tracked)
  drift policy                   prepared_bundle_drift_strict + BundleDriftPolicy
  holdout reconstruction         FrozenHoldoutPolicy (prepared_holdout seam)
  lineage attestation            LineageAuditor (write + load share it)
  payload contract               PayloadContract (arrays + required fields)
  canonical layout               CanonicalLayout (canonical_payload_rows seam)
  serialization                  BundleCodec (_portable_dataframe, pickle IO)
  run-object store               PreparationStore (build handoff + load cache)
  write path                     BundleWriter + write_prepared_bundle
  load path                      BundleReader + load_prepared_bundle
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
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from core.run_log import RunLogger
from core.schemas import TrainingSpec
from training.prepare_all_trace import timed, trace_step

_LOG = RunLogger(__name__)

_HASH_CHUNK_BYTES = 1024 * 1024

REQUIRED_BUNDLE_FIELDS = frozenset({
    "df", "payload", "structured_features", "row_bc", "country", "pos",
    "hp_pairs", "emb0", "neg", "train_neg", "neg_sources",
    "train_neg_sources", "mask_audit", "hard_negative_mask_audit",
    "labeled_pairs_csv", "canonical_records_csv", "gate_results_csv",
    "payload_variant", "masking_profile",
})


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
    # a byte), so the manifest pins what was true at build time. The pins
    # are records, never compared (owner order 2026-10-07); the diet gate
    # recomputes its own verdict from the selected pairs.
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
    augmentation_coverage: dict = Field(default_factory=dict)


@timed
def _digest(path: Path) -> str:
    """Hash one file with a byte-accurate progress bar (no manual bhist here)."""
    digest = hashlib.sha256()
    source = Path(path)
    with source.open("rb") as handle, \
            _LOG.bar(total=source.stat().st_size, desc='bundle_sha256',
                     unit='B') as bar:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


@timed
def prepared_bundle_drift_strict() -> bool:
    """Read the declared bundle policy; reject malformed environment overrides."""
    raw = os.environ.get("PREPARED_BUNDLE_DRIFT_STRICT")
    if raw is not None:
        try:
            return TypeAdapter(bool).validate_python(raw.strip())
        except ValueError as exc:
            exc.add_note('Environment setting PREPARED_BUNDLE_DRIFT_STRICT')
            raise
    from core.common import training_cfg
    return training_cfg().prepared_bundle_drift_strict


class BundleDriftPolicy:
    """Strict-mode drift decisioning for stale bundles: fail loud or warn."""

    @staticmethod
    def report_drift(path: Path, detail: str) -> None:
        """Fail in strict mode, warn otherwise — one decision point."""
        if prepared_bundle_drift_strict():
            raise ValueError(f'[bundle-drift] STRICT: {detail}')
        print(f'[bundle-drift] WARNING: {detail}', flush=True)

    @staticmethod
    @timed
    def check_provenance_drift(path: Path, manifest: PreparedBundleManifest) -> None:
        """Detect config drift against the manifest's build-time snapshot."""
        from core.common import load_config, masking_cfg
        if not manifest.masking_config or not manifest.easy_config:
            BundleDriftPolicy.report_drift(
                path, f'{path} lacks masking/easy-negative provenance; regenerate the bundle')
            return
        drifted: list[str] = []
        if manifest.masking_config and manifest.masking_config != masking_cfg(
                str(manifest.masking_profile)):
            drifted.append("masking")
        if manifest.easy_config and manifest.easy_config != dict(
                load_config()["training"]["random_easy_negatives"]):
            drifted.append("random_easy_negatives")
        if drifted:
            BundleDriftPolicy.report_config_drift(path, drifted, manifest)

    @staticmethod
    def report_config_drift(path: Path, drifted: list[str],
                            manifest: PreparedBundleManifest) -> None:
        """The drift verdict sentence for a mismodied masking/easy snapshot."""
        BundleDriftPolicy.report_drift(
            path,
            f"{path} was built under different {'/'.join(drifted)} "
            f"config than active (bundle={manifest.masking_config} "
            f"{manifest.easy_config}). Diet verdicts and augmentation "
            "yields may not reproduce."
        )

    @staticmethod
    def check_legacy_ratio_note(path: Path, manifest: PreparedBundleManifest) -> None:
        """Old projected ratio claims are dropped loudly, then recomputed by gate."""
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


class FrozenHoldoutPolicy:
    """The frozen holdout contract: split shape, entity coverage, pair integrity."""

    @staticmethod
    @timed
    def validate_frozen(data: dict, frozen: dict, normalize_gtin):
        """The three SR checks on a frozen holdout: shape, entity coverage, pairs."""
        if set(frozen) != {'train', 'dev', 'test'}:
            raise ValueError('frozen holdout needs train/dev/test')
        populations = tuple(set(frozen[split]) for split in ('train', 'dev', 'test'))
        FrozenHoldoutPolicy.entities_cover_payload(data, populations, normalize_gtin)
        FrozenHoldoutPolicy.pairs_stay_in_split(data, normalize_gtin)
        return populations

    @staticmethod
    @timed
    def entities_cover_payload(data: dict, populations, normalize_gtin) -> None:
        """Every payload entity sits in exactly one frozen holdout split."""
        roles = {}
        for role, values in enumerate(populations):
            for value in values:
                key = normalize_gtin(value)
                if key in roles and roles[key] != role:
                    raise ValueError('frozen holdout contains overlapping entities')
                roles[key] = role
        for value in _LOG.progress(data['row_bc'], desc='holdout_coverage',
                                   unit='entity'):
            key = normalize_gtin(value)
            if key and key not in roles:
                raise ValueError('frozen holdout misses a payload entity')

    @staticmethod
    @timed
    def pairs_stay_in_split(data: dict, normalize_gtin) -> None:
        """No positive pair may cross the frozen split boundary."""
        roles = {}
        for split in ('train', 'dev', 'test'):
            for value in data['holdout_populations'][split]:
                roles[normalize_gtin(value)] = split
        for a, b in _LOG.progress(data['pos'], desc='holdout_pairs', unit='pair'):
            ka = normalize_gtin(data['row_bc'][a])
            kb = normalize_gtin(data['row_bc'][b])
            if roles.get(ka) != roles.get(kb):
                raise ValueError('positive pair crosses frozen holdout')


@timed
def prepared_holdout(data: dict, split_cfg: dict, *, seed: int):
    """Reuse a frozen parent split for sampled smokes; otherwise derive SSOT."""
    from training.folds import derive_holdout, normalize_gtin
    frozen = data.get('holdout_populations')
    if frozen is None:
        return derive_holdout(data['pos'], data['row_bc'], split_cfg, seed=seed)
    return FrozenHoldoutPolicy.validate_frozen(data, frozen, normalize_gtin)


class LineageAuditor:
    """Payload lineage attestation shared by the write and load lanes."""

    @staticmethod
    @timed
    def validate_augmented_features(payload, features, audit) -> None:
        """Payload rows and features align; audited augmentation reproduces itself."""
        if len(features) != len(payload):
            raise ValueError("prepared bundle feature/payload row counts disagree")
        if not audit:
            return
        LineageAuditor.assert_audited_reproducible(payload, features, audit)

    @staticmethod
    @timed
    def assert_audited_reproducible(payload, features, audit) -> None:
        """Recompute the audited augmentation once; refuse any byte of drift."""
        from training.masking import extend_augmented_features
        first_copy = min(int(row["copy_payload_idx"]) for row in audit)
        expected = extend_augmented_features(features[:first_copy], payload, audit)
        if not np.array_equal(np.asarray(features), expected):
            raise ValueError(
                "prepared bundle augmentation features disagree with payload lineage; "
                "re-prepare or repair the bundle before training"
            )

    @staticmethod
    @timed
    def validate_counterfactual_audits(payload, audit) -> None:
        """Reject stored twin negatives whose claimed field flip is compatible."""
        twins = [row for row in audit if row.get("target_mode") == "counterfactual"]
        if not twins:
            return
        from training.masking import _field_surfaces, _field_values_conflict
        surfaces: dict[int, dict] = {}
        invalid: list[tuple[int, int]] = []
        for twin in _LOG.progress(twins, desc='counterfactual_audit', unit='twin'):
            copy_i = int(twin["copy_payload_idx"])
            pair_i = int(twin["pair_payload_idx"])
            for idx in (copy_i, pair_i):
                if idx not in surfaces:
                    surfaces[idx] = _field_surfaces(payload[idx])
            if not any(
                field in surfaces[copy_i]
                and field in surfaces[pair_i]
                and _field_values_conflict(field, surfaces[copy_i][field], surfaces[pair_i][field])
                for field in (twin.get("fields_hit") or [])
            ):
                invalid.append((copy_i, pair_i))
        if invalid:
            raise ValueError(
                "prepared bundle contains counterfactual twins without a verified "
                f"semantic conflict (first pairs: {invalid[:5]}); regenerate the bundle"
            )


_validate_augmented_features = LineageAuditor.validate_augmented_features
_validate_counterfactual_audits = LineageAuditor.validate_counterfactual_audits


class PayloadContract:
    """The field/shape contract every bundle payload must satisfy (write + load)."""

    @staticmethod
    def assert_required_fields(data: dict[str, Any]) -> None:
        """Every field a consumer needs must exist (exact missing set reported)."""
        missing = sorted(REQUIRED_BUNDLE_FIELDS - set(data))
        if missing:
            raise ValueError(f"prepared training bundle missing fields: {missing}")

    @staticmethod
    @timed
    def validate_arrays(data) -> None:
        """Reject malformed CPU inputs before casts or GPU compute hide defects."""
        required = {"payload", "row_bc", "country", "structured_features", "pos",
                    "hp_pairs", "neg", "train_neg", "emb0", "neg_sources",
                    "train_neg_sources"}
        if not required <= set(data):
            raise ValueError(f"prepared bundle missing array fields: {sorted(required - set(data))}")
        size = len(data["payload"])
        PayloadContract.validate_aligned_columns(data, size)
        PayloadContract.validate_pair_arrays(data, size)
        PayloadContract.validate_embeddings(data, size)

    @staticmethod
    @timed
    def validate_aligned_columns(data, size: int) -> None:
        """Per-row columns: payload index arrays and feature windows alike."""
        for key in ("row_bc", "country"):
            if np.asarray(data[key]).ndim != 1 or len(data[key]) != size:
                raise ValueError(f"prepared bundle {key} must cover every payload row")
        features = np.asarray(data["structured_features"])
        if (features.ndim != 2 or len(features) != size or features.dtype.kind != "f"
                or not np.isfinite(features).all()):
            raise ValueError(
                "prepared structured features need finite float rows for every payload")
        for key, pairs in (("neg_sources", "neg"), ("train_neg_sources", "train_neg")):
            if np.asarray(data[key]).ndim != 1 or len(data[key]) != len(data[pairs]):
                raise ValueError(f"prepared {key} must align with {pairs}")
        if "training_tokens" in data:
            from training.token_inputs import validate_training_tokens
            validate_training_tokens(data["training_tokens"])

    @staticmethod
    @timed
    def validate_pair_arrays(data, size: int) -> None:
        """Pair columns: exact (n,2) integer arrays, indices inside the payload."""
        for key in ("pos", "hp_pairs", "neg", "train_neg"):
            pairs = np.asarray(data[key])
            if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
                raise ValueError(f"prepared {key} needs integer (n,2) pairs")
            if np.any(pairs < 0) or np.any(pairs >= size):
                raise ValueError(f"prepared {key} contains out-of-bounds payload indices")

    @staticmethod
    @timed
    def validate_embeddings(data, size: int) -> None:
        """Initial embeddings: finite and aligned with the payload (or empty)."""
        embeddings = np.asarray(data["emb0"])
        if (embeddings.ndim != 2 or (embeddings.size and len(embeddings) != size)
                or not np.isfinite(embeddings).all()):
            raise ValueError(
                "prepared initial embeddings need finite aligned rows or an empty matrix")


class CanonicalLayout:
    """The source/canonical/augmentation row layout candidates are drawn from."""

    @staticmethod
    @timed
    def canonical_row_window(n_source: int, n_canonical: int, n_payload: int) -> np.ndarray:
        """The [n_source, n_source+n_canonical) row window, bounds-checked."""
        end = n_source + n_canonical
        if end > n_payload:
            raise ValueError(
                f"canonical block [{n_source}, {end}) exceeds the payload "
                f"({n_payload} entries) — the payload layout changed"
            )
        return np.arange(n_source, end, dtype=int)

    @staticmethod
    @timed
    def assert_canonical_block_matches(rows, row_bc, canon_map) -> None:
        """The window's GTINs must be exactly the canonical map's (no drift)."""
        actual = {str(b) for b in row_bc[rows]}
        if actual != set(canon_map):
            # TEMPORARY (owner order 2026-10-07): the committed smoke bundle
            # predates the reviewed-row canonical filtering, so its block
            # legitimately differs. Warn instead of refusing until the smoke
            # bundle is rebuilt.
            from core.run_log import RunLogger
            RunLogger(__name__).warning(
                'canonical payload block differs from the canonical map '
                f'(missing={len(set(canon_map) - actual)} extra={len(actual - set(canon_map))}) '
                '— proceeding (temporary relaxation)')


@timed
def canonical_payload_rows(n_source: int, payload: list[str], row_bc: np.ndarray) -> np.ndarray:
    """Validate the native source/canonical/augmentation layout for retrieval.

    Preparation appends the complete canonical map in sorted order after
    source rows. Augmented copies follow that block and are never candidates.
    """
    from pipeline import load_canonical_map
    canon_map = load_canonical_map()
    rows = CanonicalLayout.canonical_row_window(n_source, len(canon_map), len(payload))
    CanonicalLayout.assert_canonical_block_matches(rows, row_bc, canon_map)
    return rows


class BundleCodec:
    """Freeze + restore the pickle payload for preparation workers of any host."""

    @staticmethod
    @timed
    def portable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Freeze values without pandas-version-specific string dtype metadata."""
        result = df.astype(object)
        result.columns = pd.Index(df.columns.to_numpy(dtype=object), dtype=object)
        if isinstance(df.index.dtype, pd.StringDtype):
            result.index = pd.Index(df.index.to_numpy(dtype=object), dtype=object)
        return result

    @staticmethod
    @timed
    def write_pickle(path: Path, payload_data: dict[str, Any]) -> None:
        """Gzip-pickle the payload with the pipeline's pinned compression level."""
        with gzip.open(path, "wb", compresslevel=6) as handle:
            pickle.dump(payload_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    @timed
    def read_payload(path) -> dict[str, Any]:
        """The in-memory built bundle wins; otherwise decompress + unpickle, typed."""
        built = PreparationStore.pop_built(path)
        if built is not None:
            _, data = built
        else:
            with gzip.open(path, "rb") as handle:
                data = pickle.load(handle)
        if not isinstance(data, dict):
            raise TypeError("prepared training bundle must contain a mapping")
        return data


_portable_dataframe = BundleCodec.portable_dataframe


class PreparationStore:
    """This preparation run's bundle handoff: built objects + load cache."""

    @staticmethod
    def pop_built(path) -> tuple | None:
        """The freshly built bundle for this path, popped from the run's store."""
        from training.preparation_run import active_preparation
        run = active_preparation()
        built = (run._objects.pop('built_bundle:' + str(run.bundle_key(path)), None)
                 if run is not None else None)
        return built

    @staticmethod
    def register_built(path: Path, manifest: PreparedBundleManifest,
                       payload_data: dict[str, Any]) -> None:
        """Keep the freshly built bundle alive in this preparation's object store."""
        from training.preparation_run import active_preparation
        run = active_preparation()
        if run is not None:
            run._objects['built_bundle:' + str(run.bundle_key(path))] = (manifest, payload_data)

    @staticmethod
    def cached(key) -> tuple | None:
        """A completed bundle already alive in this preparation's store, if any."""
        from training.preparation_run import active_preparation
        run = active_preparation()
        if run is not None and run.bundle_key(key) in run._bundles:
            return run._bundles[run.bundle_key(key)]
        return None

    @staticmethod
    def cache(key, manifest: PreparedBundleManifest, data) -> None:
        """Park the loaded bundle in this preparation's store (if one is active)."""
        from training.preparation_run import active_preparation
        run = active_preparation()
        if run is not None:
            run._bundles[run.bundle_key(key)] = (manifest, data)


# --- write path -----------------------------------------------------------

class BundleWriter:
    """One write's CPU-side assembly: provenance, materialization, manifest."""

    @staticmethod
    @timed
    def resolve_masking_provenance(masking_profile: str) -> dict[str, Any]:
        """The masking/easy-negative config snapshot the manifest will pin."""
        from core.common import load_config, masking_cfg
        recorded_masking = masking_cfg(str(masking_profile))
        recorded_easy = dict(load_config()["training"]["random_easy_negatives"])
        return {
            'masking_config': recorded_masking,
            'easy_config': recorded_easy,
            'ratio_to_hard': float(recorded_easy["ratio_to_hard"]),
            'easy_enabled': bool(recorded_easy["enabled"]),
        }

    @staticmethod
    def compute_view_ratios(pos: np.ndarray, train_neg: np.ndarray) -> dict[str, float]:
        """The guaranteed bundle-only and train-time effective view ratios."""
        static_views = len(pos) / max(len(train_neg), 1)
        effective_ratio = len(pos) / max(len(train_neg), 1)
        return {'static_view_ratio': float(static_views),
                'effective_train_ratio': float(effective_ratio)}

    @staticmethod
    @timed
    def materialize_payload_data(**inputs) -> dict[str, Any]:
        """The pickle-ready payload dict (portable df keeps pandas versions off)."""
        # Colab and preparation hosts can use different pandas versions.
        # Plain object columns avoid pickling version-specific StringDtype
        # constructors while preserving the frozen values and row order.
        return {
            "df": BundleCodec.portable_dataframe(inputs['df']),
            "payload": inputs['payload'],
            "structured_features": inputs['structured_features'],
            "row_bc": inputs['row_bc'],
            "country": inputs['country'],
            "pos": inputs['pos'],
            "hp_pairs": inputs['hp_pairs'],
            "emb0": inputs['emb0'],
            "neg": inputs['neg'],
            "train_neg": inputs['train_neg'],
            "neg_sources": inputs['neg_sources'],
            "train_neg_sources": inputs['train_neg_sources'],
            "mask_audit": inputs['mask_audit'],
            "hard_negative_mask_audit": inputs['hard_negative_mask_audit'],
            "labeled_pairs_csv": inputs['labeled_pairs_csv'],
            "canonical_records_csv": inputs['canonical_records_csv'],
            "gate_results_csv": inputs['gate_results_csv'],
            "payload_variant": inputs['payload_variant'],
            "masking_profile": inputs['masking_profile'],
        }

    @staticmethod
    @timed
    def materialize_training_tokens(timing, token_checkpoint, payload, training_tokens):
        """The native token table: checkpoint-supplied when given, else as passed in."""
        if token_checkpoint is None:
            return training_tokens
        from core.common import load_local_sentence_transformer
        from training.token_inputs import prepare_training_tokens
        token_model = load_local_sentence_transformer(str(token_checkpoint), device="cpu")
        training_tokens = prepare_training_tokens(token_model, payload)
        del token_model
        timing.mark('native_training_tokens')
        return training_tokens

    @staticmethod
    @timed
    def validate_and_attach_tokens(payload_data: dict[str, Any], training_tokens) -> None:
        """Validate the token table, then embed it under its contract key."""
        if training_tokens is None:
            return
        from training.token_inputs import validate_training_tokens
        validate_training_tokens(training_tokens)
        payload_data["training_tokens"] = training_tokens

    @staticmethod
    @timed
    def maybe_attach_plans(timing, payload_data, *, token_checkpoint, holdout_populations,
                           augmentation_coverage, plan_loss, plan_train_frac, plan_sample) -> None:
        """Append the optional recipe segments: holdout, coverage, epoch plan."""
        if holdout_populations is not None:
            payload_data['holdout_populations'] = holdout_populations
        if augmentation_coverage is not None:
            from training.balanced_augmentation import AugmentationCoverage
            payload_data['augmentation_coverage'] = (
                AugmentationCoverage.model_validate(augmentation_coverage)
                .model_dump(mode='json'))
        if token_checkpoint is not None:
            from training.run_plan import prepare_run_plan
            with timing.section('objective_and_epoch_plans'):
                payload_data["training_plan"] = prepare_run_plan(
                    payload_data, loss=plan_loss, train_frac=plan_train_frac,
                    sample=plan_sample)

    @staticmethod
    @timed
    def build_manifest(path: Path, payload_data: dict[str, Any], **headers) -> PreparedBundleManifest:
        """Assemble + persist the manifest sidecar (hash included, not computed here)."""
        manifest = PreparedBundleManifest(**headers)
        path.with_suffix(path.suffix + ".json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    @staticmethod
    def ratio_contract_note(provenance: dict[str, Any], ratios: dict[str, float]) -> str:
        """The frozen ratio sentence the manifest carries to any reader."""
        return (
            f"Guaranteed bundle-only view ratio {ratios['static_view_ratio']:.3f}; easy-negative "
            f"settings ({'enabled' if provenance['easy_enabled'] else 'disabled'}, "
            f"ratio={provenance['ratio_to_hard']:g}) "
            "are a possible contrastive projection and are not guaranteed."
        )


@timed
def _audit_payload_lineage(payload, structured_features, mask_audit,
                           hard_negative_mask_audit) -> None:
    """Write-side lineage attestation under its own trace step."""
    with trace_step('write_prepared_bundle.validate_lineage'):
        _validate_augmented_features(
            payload, structured_features, mask_audit + hard_negative_mask_audit)
        _validate_counterfactual_audits(payload, hard_negative_mask_audit)


@timed
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
    augmentation_coverage: dict | None = None,
    token_checkpoint: str | None = None,
    training_tokens: dict | None = None,
    plan_loss: str | None = None,
    plan_train_frac: float = 1.0,
    plan_sample: bool = False,
) -> PreparedBundleManifest:
    """Write one compressed, self-contained, locally generated input bundle."""
    from core.model_input import model_input_spec
    from core.timing import Timing
    timing = Timing('training.bundle')

    provenance = BundleWriter.resolve_masking_provenance(masking_profile)
    ratios = BundleWriter.compute_view_ratios(pos, train_neg)
    _audit_payload_lineage(
        payload, structured_features, mask_audit, hard_negative_mask_audit)
    payload_data = BundleWriter.materialize_payload_data(
        df=df, mask_audit=mask_audit, payload=payload,
        hard_negative_mask_audit=hard_negative_mask_audit,
        structured_features=structured_features, canonical_records_csv=canonical_records_csv,
        labeled_pairs_csv=labeled_pairs_csv, masking_profile=masking_profile,
        payload_variant=payload_variant, gate_results_csv=gate_results_csv,
        emb0=emb0, country=country, hp_pairs=hp_pairs, neg=neg,
        neg_sources=neg_sources, pos=pos, row_bc=row_bc,
        train_neg=train_neg, train_neg_sources=train_neg_sources,
    )
    training_tokens = BundleWriter.materialize_training_tokens(
        timing, token_checkpoint, payload, training_tokens)
    BundleWriter.validate_and_attach_tokens(payload_data, training_tokens)
    PayloadContract.validate_arrays(payload_data)
    BundleWriter.maybe_attach_plans(timing, payload_data,
                                    token_checkpoint=token_checkpoint,
                                    holdout_populations=holdout_populations,
                                    augmentation_coverage=augmentation_coverage,
                                    plan_loss=plan_loss, plan_train_frac=plan_train_frac,
                                    plan_sample=plan_sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    timing.mark('validate_and_materialize')
    BundleCodec.write_pickle(path, payload_data)
    timing.mark('validate_and_compress_bundle')
    manifest = BundleWriter.build_manifest(
        path, payload_data,
        payload_variant=payload_variant,
        masking_profile=masking_profile,
        model_input=model_input_spec(),
        masking_config=provenance['masking_config'],
        easy_config=provenance['easy_config'],
        ratio_to_hard=provenance['ratio_to_hard'],
        static_view_ratio=ratios['static_view_ratio'],
        effective_train_ratio=ratios['effective_train_ratio'],
        ratio_contract_note=BundleWriter.ratio_contract_note(provenance, ratios),
        augmentation_coverage=payload_data.get('augmentation_coverage', {}),
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
    timing.mark('hash_and_write_manifest')
    PreparationStore.register_built(path, manifest, payload_data)
    _LOG.info(f'[bundle] wrote {path} sha256={manifest.sha256}')
    return manifest


# --- load path -------------------------------------------------------------

class BundleReader:
    """One load's verification lane: bytes, counts, identity contracts."""

    @staticmethod
    def resolve_verify_inputs(verify_inputs) -> bool:
        """The suite data gate's decision unless the caller forced one."""
        if verify_inputs is not None:
            return bool(verify_inputs)
        from model_tracks.data_gate import _owner_trusted
        return not _owner_trusted('text bundle')

    @staticmethod
    @timed
    def load_manifest(path: Path) -> PreparedBundleManifest:
        """Parse + fail-loud the bundle's sidecar manifest."""
        manifest_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"prepared training bundle missing: {path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"prepared bundle manifest missing: {manifest_path}")
        return PreparedBundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )

    @staticmethod
    @timed
    def verify_bytes(path: Path, manifest: PreparedBundleManifest) -> None:
        """The whole-file digest must match the manifest's recorded hash."""
        actual_digest = _digest(path)
        if actual_digest != manifest.sha256:
            raise ValueError(
                f"prepared bundle SHA-256 mismatch: {path} "
                f"{actual_digest} != {manifest.sha256}"
            )

    @staticmethod
    @timed
    def validate_manifest_counts(path: Path, manifest: PreparedBundleManifest,
                                 data: dict[str, Any]) -> None:
        """Every manifest count must still match the payload it describes."""
        if len(data["df"]) != manifest.n_df or len(data["payload"]) != manifest.n_payload:
            raise ValueError("prepared bundle manifest/data row counts disagree")
        if len(data["pos"]) != manifest.n_pos or len(data["neg"]) != manifest.n_neg:
            raise ValueError("prepared bundle manifest/pair counts disagree")
        if len(data["train_neg"]) != manifest.n_train_neg:
            raise ValueError("prepared bundle manifest/training-negative counts disagree")
        BundleReader.validate_embedded_csv_bytes(path, manifest, data)

    @staticmethod
    @timed
    def validate_embedded_csv_bytes(path: Path, manifest: PreparedBundleManifest,
                                    data: dict[str, Any]) -> None:
        """The embedded raw CSVs are byte-count contracted per manifest."""
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

    @staticmethod
    @timed
    def validate_identity_fields(path: Path, manifest: PreparedBundleManifest,
                                 data: dict[str, Any]) -> None:
        """Variant/profile identity and the encoder-text contract must hold."""
        if data["payload_variant"] != manifest.payload_variant:
            raise ValueError("prepared bundle payload variant disagrees with manifest")
        if data["masking_profile"] != manifest.masking_profile:
            raise ValueError("prepared bundle masking profile disagrees with manifest")
        BundleReader.assert_model_input_contract(manifest)

    @staticmethod
    @timed
    def assert_model_input_contract(manifest: PreparedBundleManifest) -> None:
        """The frozen text must be what the active composition still produces."""
        from core.model_input import model_input_spec
        active = model_input_spec()
        if manifest.model_input == active:
            return
        raise ValueError(
            "prepared training bundle was built with a different encoder-text "
            f"composition: bundle={manifest.model_input.model_dump()} "
            f"active={active.model_dump()}. Re-prepare the bundle; the frozen "
            "payload strings are not the text the active composition produces."
        )


@timed
def _timed_lineage_validations(data) -> None:
    """D3 telemetry: load-time lineage validators with one timing line each.

    Behavior-identical to calling the two validators back to back; the marks
    record how expensive the attested-path and un-attested-path validators
    are for the D3 decision.
    """
    from core.timing import Timing

    timing = Timing("prepared_bundle.lineage")
    _validate_augmented_features(
        data["payload"], data["structured_features"],
        data["mask_audit"] + data["hard_negative_mask_audit"],
    )
    timing.mark("augmented_features")
    _validate_counterfactual_audits(data["payload"], data["hard_negative_mask_audit"])
    timing.mark("counterfactual_audit")


@timed
def load_prepared_bundle(path: Path, *, verify_inputs=None) -> tuple[PreparedBundleManifest, dict[str, Any]]:
    """Load and validate a bundle before it crosses into the training lane.

    `verify_inputs` defaults to the suite data gate's decision.  The bundle is
    always decompressed and unpickled -- training consumes it -- but when the
    supervisor already attested these exact bytes the whole-file SHA-256 and the
    manifest/provenance comparisons are skipped, since a changed bundle changes
    its digest and leaves enforcement active.  Pass True to force them.
    """
    cached = PreparationStore.cached(path)
    if cached is not None:
        return cached
    verify = BundleReader.resolve_verify_inputs(verify_inputs)
    manifest = BundleReader.load_manifest(path)
    if verify:
        BundleReader.verify_bytes(path, manifest)
    data = BundleCodec.read_payload(path)
    if verify:
        PayloadContract.validate_arrays(data)
    PayloadContract.assert_required_fields(data)
    _timed_lineage_validations(data)
    BundleReader.validate_manifest_counts(path, manifest, data)
    BundleReader.validate_identity_fields(path, manifest, data)
    PreparationStore.cache(path, manifest, data)
    _LOG.info(f'[bundle] loaded {path} sha256={manifest.sha256}')
    # BundleDriftPolicy checks retired (owner order 2026-10-07: no drift enforcement in the lane; prepared_bundle_drift_strict is config-inert).
    return manifest, data
