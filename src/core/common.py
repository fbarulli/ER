"""src/core/common.py — the ONLY module that reads the config files.

Split-SSOT (2026-09-08; EDA removed 2026-09-10): the monolithic
config/paths.yaml was broken into domain configs, each in its owning
directory:

  config/paths.yaml        DataConfig      — paths/files/column_mapping/seed/models
  config/training.yaml   TrainingConfig  — the training lane's knobs

(The EDA dir and its eda.yaml were deleted 2026-09-10 — the lane is
training-only. The five TRAIN-consumed EDA keys — plots.dpi,
pairs.max_pos_per_group/n_neg/neg_oversample, strip_audit_sample —
migrated into config/training.yaml blocks of the same names.)

src/core/common deep-merges them into ONE view (load_config()) and VALIDATES each
file against its pydantic model in src/core/schemas at load — a bad value
crashes at import with a named field error, never mid-run. No script reads
YAML directly (unchanged SSOT doctrine), and every accessor below reads the
merged view, so consumers don't care which physical file a knob lives in.

New accessors:
  training_cfg()  the validated TrainingConfig (typed)
  data_cfg()      the validated DataConfig (typed)
  resolve_model(key)  registry key -> materialized local bundle, no Hub fallback
"""

import json
import copy
from decimal import Decimal
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Sequence

def _find_project_root() -> Path:
    """Locate the project from stable markers, never a magic parent offset."""
    from core.project_root import find_project_root

    return find_project_root(Path(__file__).resolve())


TRAIN_ROOT = _find_project_root()
CONFIG_DIR = TRAIN_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "paths.yaml"
TRAINING_CONFIG_PATH = CONFIG_DIR / "training.yaml"
VOCABULARY_CONFIG_PATH = CONFIG_DIR / "vocabulary.json"

# Keep matplotlib's cache inside the workspace for local and remote runs;
# importing this module must not fall back to a transient /tmp cache.
_MPLCONFIGDIR = TRAIN_ROOT / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot import

import pandas as pd
import yaml

from core.schemas import (
    DataConfig,
    LayoutSpec,
    TrainingConfig,
    check_canonical_records_frame,
    upgrade_canonical_records_frame,
)
from core.text import extract_volume_ml, normalize_retailer


def metadata_text(value: object) -> str:
    """Serialize source metadata without converting valid false-y values."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def row_metadata_text(row, primary: str, *aliases: str) -> str:
    """Read a source field by any of its names, retaining missing as unknown.

    VARIADIC, and that is the point: the set of names a column answers to is
    declared once (config/paths.yaml column_mapping / column_aliases, read via
    core.columns.alias_names). This used to accept exactly one alias, so a
    caller could not hand it the real set — and the callers that hardcoded
    ("attribute", "attr") or ("sku_name_eng", "title") inline were re-declaring
    column_mapping. First name present wins, in the order given, so
    canonical-first ordering resolves the raw/canonical preference in config.
    """
    for name in (primary, *aliases):
        if name in row.index:
            return metadata_text(row[name])
    return ""

# require_keys REMOVED (audit 2026-09-09): zero consumers — the pydantic
# validation at load (DataConfig/TrainingConfig) already fails loudly with
# named field errors. The former duplicate helper was never called.


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_vocabulary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config missing: {path}")
    try:
        vocabulary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed vocabulary config {path}: {exc}") from exc
    required_lists = ("STOPWORDS", "MINIMAL_STOPWORDS", "ENGLISH_STOP_WORDS")
    for key in required_lists:
        values = vocabulary.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise SystemExit(f"vocabulary.{key} must be a non-empty list of strings")
    folds = vocabulary.get("CONCEPT_FOLDS")
    if not isinstance(folds, dict) or not folds or not all(
        isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        for key, value in folds.items()
    ):
        raise SystemExit("vocabulary.CONCEPT_FOLDS must be a non-empty string mapping")
    macros = vocabulary.get("category_macros")
    if not isinstance(macros, dict) or not macros:
        raise SystemExit("vocabulary.category_macros must be a non-empty mapping")
    for category, macro in macros.items():
        if not isinstance(category, str) or not category.strip() or not isinstance(macro, str) or not macro.strip():
            raise SystemExit("vocabulary.category_macros keys and values must be non-empty strings")
        if category.strip() == macro.strip():
            raise SystemExit(f"vocabulary.category_macros maps {category!r} to itself")
    from core.attribute_vocabulary import validated_attribute_vocabulary

    validated_attribute_vocabulary(vocabulary)
    return vocabulary


@lru_cache(maxsize=1)
def _load_config_cached() -> dict:
    """Load + validate the split-SSOT and deep-merge into ONE view.

    Order: paths.yaml is the base; training.yaml
    overlays it (the domain file wins on conflicts — a conflict
    is a config bug and the domain file is the authority for its keys).
    Every file is validated against its pydantic model BEFORE merging, so an
    invalid knob crashes here with the file + field named.
    """
    base = dict(_read_yaml(CONFIG_PATH))
    DataConfig.model_validate(base)  # root contract — fail before merge
    vocabulary = _read_vocabulary(VOCABULARY_CONFIG_PATH)
    merged = dict(base)
    for path, model in (
        (TRAINING_CONFIG_PATH, TrainingConfig),
    ):
        if path.exists():
            overlay = dict(_read_yaml(path))
            model.model_validate(overlay)
            for key, block in overlay.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(block, dict)
                ):
                    merged[key] = {**merged[key], **block}
                else:
                    merged[key] = block
        else:
            raise SystemExit(f"config missing: {path}")
    hpo_models = set(merged.get("hpo", {}).get("models", []))
    registry_models = set(base.get("models", {}))
    base_model = merged["training"]["base_model"]
    if base_model not in registry_models:
        raise SystemExit(
            "training.base_model must be a model registry key: "
            f"{base_model!r} not in {sorted(registry_models)}"
        )
    unknown_hpo_models = sorted(hpo_models - registry_models)
    if unknown_hpo_models:
        raise SystemExit(
            "hpo.models contains unknown config/paths.yaml registry key(s): "
            + ", ".join(unknown_hpo_models)
        )
    rerank_model = merged.get("sweep", {}).get("rerank_model")
    if rerank_model not in registry_models:
        raise SystemExit(
            "sweep.rerank_model must be a config/paths.yaml model registry key: "
            f"{rerank_model!r} not in {sorted(registry_models)}"
        )
    sims_model = merged["colab"]["sims_model"]
    if sims_model not in registry_models:
        raise SystemExit(
            "colab.sims_model must be a config/paths.yaml model registry key: "
            f"{sims_model!r} not in {sorted(registry_models)}"
        )
    embedding_models = set(base["embedding_model_keys"])
    if sims_model not in embedding_models:
        raise SystemExit(
            "colab.sims_model must be listed in embedding_model_keys: "
            f"{sims_model!r} not in {sorted(embedding_models)}"
        )
    mixed_profile = merged["colab"]["mixed_mining_profile"]
    if mixed_profile not in merged["mining_profiles"]:
        raise SystemExit(
            "colab.mixed_mining_profile must name a configured mining profile: "
            f"{mixed_profile!r} not in {sorted(merged['mining_profiles'])}"
        )
    merged["category_macros"] = vocabulary["category_macros"]
    return merged


def load_config() -> dict:
    """Return a copy of the validated merged config from the SSOT cache."""
    return copy.deepcopy(_load_config_cached())


_DEFAULT_CONFIG_LOADER = load_config


def config_section(*keys, loader=None):
    """Copy one requested section while preserving cache and override semantics."""
    selected_loader = load_config if loader is None else loader
    source = (_load_config_cached() if selected_loader is _DEFAULT_CONFIG_LOADER
              else selected_loader())
    for key in keys:
        source = source[key]
    return copy.deepcopy(source)


# ── validated singletons (read once at import; the merge order above) ───────
_CFG = load_config()
_DATA_CFG = DataConfig.model_validate(_read_yaml(CONFIG_PATH))
_TRAIN_CFG = TrainingConfig.model_validate(_read_yaml(TRAINING_CONFIG_PATH))
_VOCABULARY = _read_vocabulary(VOCABULARY_CONFIG_PATH)


def data_cfg() -> DataConfig:
    """The validated root data contract (config/paths.yaml)."""
    return _DATA_CFG


def category_macros() -> dict[str, str]:
    """SSOT accessor for the category -> macro bucket taxonomy
    (config/paths.yaml category_macros:).

    Was the inline MACRO_MAP dict in src/core/text.py — domain data the owner
    may tune (the blocking layer's recall-first rollup of the dataset's
    strict categories), hence config-owned. Validated by DataConfig at
    load; consumers (blocking_audit / report_plots / hard_negatives)
    read THIS, never a module-level copy.
    """
    return dict(_VOCABULARY["category_macros"])


def embedding_model_keys() -> tuple[str, ...]:
    """Return the config-owned registry keys valid for bi-encoder lanes."""
    return tuple(_CFG["embedding_model_keys"])


def vocabulary() -> dict[str, Any]:
    """Return a copy of the validated centralized vocabulary data."""
    return dict(_VOCABULARY)


def training_cfg() -> TrainingConfig:
    """The validated training-lane config (config/training.yaml)."""
    return _TRAIN_CFG


def masking_cfg(profile: str | None = None) -> dict[str, Any]:
    """Return the validated base masking config with one named profile applied."""
    base = dict(_CFG["masking"])
    selected = str(profile or base["profile"])
    profiles = _CFG["masking_profiles"]
    if selected not in profiles:
        raise KeyError(
            f"unknown masking profile {selected!r}; "
            f"expected one of {sorted(profiles)}"
        )
    overrides = {
        key: value
        for key, value in dict(profiles[selected]).items()
        if value is not None
    }
    base.update(overrides)
    base["profile"] = selected
    return base


def collapse_guardrail_cfg(profile: str | None = None) -> dict[str, Any]:
    """Return the validated collapse guardrail with one named profile applied."""
    base = dict(_CFG["collapse_guardrail"])
    selected = str(profile or base["profile"])
    profiles = _CFG["collapse_guardrail_profiles"]
    if selected not in profiles:
        raise KeyError(
            f"unknown collapse guardrail profile {selected!r}; "
            f"expected one of {sorted(profiles)}"
        )
    base.update(dict(profiles[selected]))
    base["profile"] = selected
    return base


def ner_config() -> dict[str, Any]:
    """Return the NER training settings from the centralized training SSOT.

    NER callers must use this accessor rather than constructing a path to a
    private YAML file. A deep copy prevents a consumer from mutating the
    process-wide validated configuration.
    """
    required = {
        "base_dir",
        "results_dir",
        "columns",
        "data_prep",
        "semantic_training",
        "semantic_evaluation",
        "ner_training",
        "colab",
        "huggingface",
    }
    missing = sorted(required - set(_TRAIN_CFG.ner))
    if missing:
        raise KeyError(f"training.ner missing required section(s): {missing}")
    return copy.deepcopy(_TRAIN_CFG.ner)


def runtime(key: str, default: Any = None) -> Any:
    """SSOT accessor for every training runtime knob (config/training.yaml
    training: block — validated by TrainingSpec at load).

    Values come from the validated Pydantic TrainingSpec, including legal
    nullable fields. ONE place to read batch sizes, seq length, eval cadence, dev fraction,
    band edges. Scripts call runtime("batch_size_cpu") etc. — the literal
    lives in config/training.yaml only, never in a script.

    The return is typed `Any` deliberately (audit round 2 F17): the
    training: block holds ints, floats, bools and strings; a narrower lie
    would be worse than the honest open type.

    NO FALLBACKS (owner Q27, audit 2026-09-09): the `default` escape hatch
    (silently returning a caller literal when the key was MISSING) is
    CLOSED. A default may only supply a value when the SSOT defines the
    key as an explicit YAML null. Without a caller default, a valid nullable
    field retains None. A truly
    missing key always raises — the literal can never diverge from SSOT.
    """
    tr = _CFG.get("training", {})
    if key in tr:
        val = getattr(_TRAIN_CFG.training, key)
        if hasattr(val, 'model_dump'):
            val = val.model_dump(mode='python')
        if val is not None:
            return val
        if default is not None:
            return default  # explicit null in YAML = caller's default, opt-in
        return None  # Only schema-approved nullable values reach this point.
    raise KeyError(
        f"training.{key} missing from config/training.yaml — the SSOT must "
        f"define it (no per-script literals allowed)"
    )


# no-fallback SSOT scalars (owner directive Q27: NO FALLBACKS — a missing
# config key must crash, never silently default). Consumers import these
# instead of chaining .get(...) with inline literals.
SSOT_LOSS = runtime("loss")
SSOT_CONTRASTIVE_MARGIN = runtime("contrastive_margin")


def plot_dpi() -> int:
    """SSOT accessor for figure DPI (config/training.yaml plots.dpi).

    Every fig.savefig in the tree renders at THIS value — dpi=150 was
    inlined at 29 call sites across the plot scripts, a
    second declaration the config could not steer (audit, owner Q27).
    """
    return int(_CFG["plots"]["dpi"])


def recall_column_suffix(target_recall: float) -> str:
    """Return a collision-free SSOT label for a recall target.

    Whole percentages retain the compact historical spelling (``0.90`` is
    ``90pct``). Fractional percentage points use ``p`` as the decimal marker,
    so ``0.904`` becomes ``90p4pct`` instead of colliding with ``0.90``.
    """
    percent = Decimal(str(target_recall)) * Decimal("100")
    if not percent.is_finite() or percent < 0:
        raise ValueError(f"target_recall must be a finite non-negative value, got {target_recall!r}")
    text = format(percent, "f").rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return f"{text.replace('.', 'p')}pct"


def strip_ladder_bands() -> list[tuple[float, float]]:
    """SSOT accessor for the strip-audit similarity ladder's Jaccard band
    edges (config/training.yaml audit.strip_ladder_bands).

    Was an inline literal list in src/training/strip_audit.py (~:186) — a second
    declaration the config could not steer (same doctrine as plot_dpi).
    Validated by AuditSpec at load (contiguous ascending cover of
    [0, 1+eps]); returns plain (lo, hi) tuples for the ladder loop.
    """
    return [(float(b[0]), float(b[1])) for b in _CFG["audit"]["strip_ladder_bands"]]


# ── HPO / rerank / sweep accessors (validated by lib.schemas at load) ───────
# These expose the hpo:, rerank:, sweep: blocks as PLAIN JSON-able data
# (lists of dicts / tuples) so callers never re-declare the sweep spaces.
def hpo_cfg() -> dict:
    """The hpo: block as plain data.

    grid/quick rows come back as dicts ({epochs, lr, warmup}); tpe_space as
    {knob: (lo, hi)}. Validated by HpoSpec at load — no re-validation here.
    """
    h = dict(_CFG["hpo"])
    h["models"] = list(h["models"])
    h["grid"] = [dict(r) for r in h["grid"]]
    h["quick"] = [dict(r) for r in h["quick"]]
    return h


def rerank_cfg() -> dict:
    """The rerank: block (07e decision rule) as plain data."""
    return dict(_CFG["rerank"])


def sweep_cfg() -> dict:
    """The sweep: block (07-series ablation axes; src/cli/colab.py derives its
    smoke-sample / train-frac / rerank defaults from it) as plain data."""
    return dict(_CFG["sweep"])


def rand_matching_cfg() -> dict:
    """The final direct SKU-to-canonical matching contract."""
    return copy.deepcopy(_CFG["rand_matching"])


def band(name: str) -> tuple[float, float]:
    """SSOT accessor for cosine bands (config/training.yaml bands:).

    name: 'eval_mining' (train_one_config's eval-pool mining band) or
    'rerank_band' (cross-encoder). 'mining_band' was REMOVED (audit
    round 2 F21): it had zero callers — the live in-batch mining band is
    mining.band "lo-hi" (the string form, mined via _band_tuple).
    """
    b = _CFG.get("bands", {}).get(name)
    if not b or len(b) != 2:
        raise KeyError(f"bands.{name} missing/malformed in config/training.yaml")
    lo, hi = float(b[0]), float(b[1])
    if not lo < hi:
        raise ValueError(f"bands.{name}: lo must be < hi, got [{lo}, {hi}]")
    return lo, hi


def _path(cfg_value: str) -> Path:
    """Config path strings resolve relative to TRAIN_ROOT unless absolute."""
    p = Path(cfg_value)
    return p if p.is_absolute() else TRAIN_ROOT / p


# ── paths (SSOT) ────────────────────────────────────────────────────────────
DATA_DIR = _path(_CFG["paths"]["data_dir"])
_results_override = os.environ.get("EUROMONITOR_RESULTS_DIR")
RESULTS = (
    Path(_results_override).expanduser().resolve()
    if _results_override
    else _path(_CFG["paths"]["results_dir"])
)
RESULTS.mkdir(parents=True, exist_ok=True)
TRAINING_RESULTS = _path(_CFG["paths"]["training_results_dir"])
TRAINING_RESULTS.mkdir(parents=True, exist_ok=True)

# ── artifact binding roots (SSOT) ────────────────────────────────────────────
# Every files./layouts. entry is a "root:name" binding. roots resolve here —
# ONE rule per root, no guessing, no per-lane inference:
#   repo             → TRAIN_ROOT/name
#   data             → DATA_DIR/name
#   results          → RESULTS/name      (honors EUROMONITOR_RESULTS_DIR —
#                     on a Colab worker this is the WORKER dir, not the repo
#                     results root — never change this to TRAIN_ROOT-based)
#   results_training → RESULTS/training/name
#   results_hpo      → RESULTS/hpo/name
_BINDING_ROOTS = {
    "repo": TRAIN_ROOT,
    "data": DATA_DIR,
    # The current, regenerated training inputs. Its OWN root on purpose: these
    # files must never be confused with the stale copies they replaced.
    "training_data": TRAIN_ROOT / _CFG["paths"]["training_data_dir"],
    "results": RESULTS,
    "results_training": RESULTS / "training",
    "results_hpo": RESULTS / "hpo",
}
_KNOWN_ROOT_TOKENS = tuple(_BINDING_ROOTS)


def _resolve_file(binding: str | Path) -> Path:
    """Resolve a "root:name" SSOT binding to an absolute Path.

    Unknown root or empty name CRASHES here (fail-loud at import, never
    mid-run).  No fallback, no relative joins downstream.
    """
    if isinstance(binding, Path):
        return binding.resolve()
    root, sep, name = str(binding).partition(":")
    if not sep or root not in _BINDING_ROOTS:
        raise ValueError(
            f"files/layouts binding {binding!r} must be 'root:name' with root "
            f"in {sorted(_KNOWN_ROOT_TOKENS)}"
        )
    return (_BINDING_ROOTS[root] / name).resolve()


# ── file names (SSOT → resolved absolute Paths) ─────────────────────────────
F = {name: _resolve_file(value) for name, value in _CFG["files"].items()}
DATA_PATH = F["dataset"]


@lru_cache(maxsize=1)
def _canonical_records_frame() -> pd.DataFrame:
    """Read and validate the shared canonical-record artifact exactly once."""
    records = pd.read_csv(
        F["canonical_records"],
        dtype={"gtin": str},
        keep_default_na=False,
    )
    records = upgrade_canonical_records_frame(records)
    check_canonical_records_frame(records)
    return records


def canonical_records_frame() -> pd.DataFrame:
    """Return an isolated view of the validated canonical-record artifact."""
    return _canonical_records_frame().copy(deep=True)

# ── owned layout templates for generated artifacts (SSOT) ───────────────────
# paths.yaml `layouts:` entries are validated into LayoutSpec here (import
# crash on unknown root, bad field type, or undeclared field) — no raw dicts.
LAYOUTS: dict[str, LayoutSpec] = {
    name: LayoutSpec.model_validate(spec)
    for name, spec in (_CFG.get("layouts") or {}).items()
}
_UNSET = object()


def artifact(key: str, fields: dict[str, object] | None = None) -> Path:
    """Render a generated-artifact destination from an owned layout template.

    Template placeholders are filled ONLY from the declared fields set; a
    missing or unexpected field crashes.  Callers must wrap the returned
    path in the layout's OWNER module (declared in paths.yaml layouts:.owner).
    """
    spec = LAYOUTS.get(key)
    if spec is None:
        raise KeyError(f"unknown layout {key!r}; declared layouts: {sorted(LAYOUTS)}")
    declared = set(spec.fields)
    provided = dict(fields or {})
    if set(provided) != declared:
        raise ValueError(
            f"layout {key!r} requires fields {sorted(declared)}, got {sorted(provided)}"
        )
    coerced: dict[str, object] = {}
    for name, typ in spec.fields.items():
        raw = provided[name]
        if typ == "int":
            coerced[name] = int(raw)
        elif typ == "float":
            coerced[name] = float(raw)
        else:
            coerced[name] = str(raw)
    return (_BINDING_ROOTS[spec.root] / spec.template.format(**coerced)).resolve()


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def trace_artifact(key: str, path: Path, producer: str = "") -> None:
    """Stamp a generated-artifact write into the artifacts trace manifest."""
    import json as _json
    from datetime import datetime as _datetime, timezone as _timezone

    trace_dir = RESULTS / "manifests"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "artifacts_trace.json"
    record = {
        "layout": key,
        "path": str(path),
        "producer": producer or "unknown",
        "at": _datetime.now(_timezone.utc).isoformat(),
    }
    rows = []
    if trace_path.exists():
        try:
            rows = _json.loads(trace_path.read_text(encoding="utf-8"))
            if isinstance(rows, dict):
                rows = rows.get("artifacts", [])
        except Exception:
            rows = []
    rows.append(record)
    tmp = trace_path.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps({"artifacts": rows}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(trace_path)


# ── column mapping + seed (SSOT, read once) ──────────────────────────────────
# COLUMN_MAPPING moved to core.columns, which derives BOTH vocabularies (raw
# and canonical) and the per-title evidence-capture field list from
# config/paths.yaml. It stays bound here for its many existing importers; the
# declaration itself lives in core.columns so that "where is a column named"
# has exactly one answer. Imported lazily through a module __getattr__ so the
# two modules can reference each other without an import cycle.
SEED = int(_CFG["seed"])


def __getattr__(name: str):
    if name == "COLUMN_MAPPING":
        from core.columns import COLUMN_MAPPING as mapping

        return mapping
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── gate census tripwire ────────────────────────────────────────────────────
# Single declared source: config/rand_matching.gate_census_pin (schema-
# validated, sum-checked). Consumed by src/training/labeled_pairs.py
# (fallback exclusion) and selftest oracle_pinned_counts. Update the CONFIG
# only, on a measured intentional gate change; no code-side literal exists.
PINNED_GATE_FALLBACK_PAIRS = int(
    training_cfg().rand_matching.gate_census_pin.fallback
)


def gate_census_drift_report(
    *,
    measured: dict[str, int],
    previous_labeled_csv: Path | None = None,
    current_gate: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Per-sample map of what a gate-census drift DID to the labeled data.

    Fired by the census consumers (training.labeled_pairs, selftest
    oracle_pinned_counts) when the measured gate census breaks the config
    rand_matching.gate_census_pin pin. Diffing the CURRENT gate_results.csv
    against the PREVIOUS data/labeled_pairs.csv (on disk, else git HEAD),
    it classifies every labeled pair the census change moved:

      lost_true_block   label 1 whose pair is now gated hard_no
      lost_true_review  label 1 whose pair now lands in the fallback tier
                        (excluded from labels — silent drop, counted here)
      new_merge_risk    label 0 whose pair now passes the gate (proceed)
      lost_neg_review   label 0 whose pair now lands in fallback (excluded)
      survived          labeled pairs whose source outcome is unchanged

    Writes the full per-sample report to results/gate_census_drift.json
    (results/ is gitignored scratch) and returns the summary + sample lists.
    """
    pin = training_cfg().rand_matching.gate_census_pin.model_dump()
    g = current_gate if current_gate is not None else pd.read_csv(
        RESULTS / F["gate_results"],
        dtype=str,
        keep_default_na=False,
    )
    outcome = {
        (str(r.gtin1), str(r.gtin2)): (str(r.gate_decision), str(r.gate_reason))
        for r in g.itertuples(index=False)
    }
    labeled_path = previous_labeled_csv or (DATA_DIR / F["labeled_pairs"])
    if labeled_path.is_file():
        labeled = pd.read_csv(
            labeled_path,
            dtype=str,
            keep_default_na=False,
        )
    else:
        import subprocess
        from io import BytesIO

        try:
            git_path = labeled_path.resolve().relative_to(TRAIN_ROOT.resolve())
        except ValueError as exc:
            raise FileNotFoundError(
                f"Previous labeled artifact is missing outside the repository: {labeled_path}"
            ) from exc
        head = subprocess.run(
            ["git", "show", f"HEAD:{git_path.as_posix()}"],
            cwd=TRAIN_ROOT,
            capture_output=True,
        )
        if head.returncode != 0:
            raise FileNotFoundError(
                f"No previous labeled artifact to diff against: neither "
                f"{labeled_path.as_posix()} on disk nor in git HEAD — the "
                "per-sample drift map needs SOME previous labeled set"
            )
        labeled = pd.read_csv(
            BytesIO(head.stdout),
            dtype=str,
            keep_default_na=False,
        )
    classes: dict[str, list[dict[str, str]]] = {
        key: []
        for key in (
            "lost_true_block",
            "lost_true_review",
            "new_merge_risk",
            "lost_neg_review",
            "survived",
        )
    }
    missing_from_gate = 0
    for row in labeled.itertuples(index=False):
        state = outcome.get((str(row.gtin1), str(row.gtin2)))
        if state is None:
            missing_from_gate += 1
            continue
        decision, reason = state
        label = int(float(row.true_label))
        bucket = (
            {
                "proceed": "survived",
                "hard_no": "lost_true_block",
                "fallback": "lost_true_review",
            }[decision]
            if label == 1
            else {
                "hard_no": "survived",
                "proceed": "new_merge_risk",
                "fallback": "lost_neg_review",
            }[decision]
        )
        classes[bucket].append(
            {
                "gtin1": str(row.gtin1),
                "gtin2": str(row.gtin2),
                "label": label,
                "new_decision": decision,
                "new_reason": reason,
            }
        )
    summary = {
        "config_pin": pin,
        "measured_census": measured,
        "deltas": {
            key: int(measured.get(key, 0)) - int(pin.get(key, 0))
            for key in pin
        },
        "degraded": {
            key: len(items) for key, items in classes.items() if key != "survived"
        },
        "survived": len(classes["survived"]),
        "pairs_absent_from_current_gate": missing_from_gate,
        "samples": classes,
    }
    report_path = RESULTS / "gate_census_drift.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def set_determinism(seed: int) -> None:
    """Seed EVERYTHING the training lane touches, loudly and unconditionally.

    One call at each training entrypoint (before any model/data randomness)
    pins: random, numpy, torch (CPU + all CUDA devices) and the cudnn
    flags. The EXISTING SSOT seed is the only source — config/paths.yaml
    `seed:` (read once into lib.common.SEED); no second knob exists or is
    needed, so callers pass exactly that value.

    PYTHONHASHSEED: os.environ is set here for the CURRENT process, but
    hash() randomization is fixed only when the variable is present
    BEFORE the interpreter starts — setting it here cannot retro-fit an
    already-running CPython. It is still exported (harmless, and it makes
    child processes spawned after this call inherit the value). For full
    hash determinism the SAME seed must ALSO be exported at container
    start (Dockerfile `ENV PYTHONHASHSEED=42` — documented here, NOT
    changed by that task; align it with config/paths.yaml `seed:` when the
    Dockerfile is next touched).

    No new config key: cudnn.deterministic=True / benchmark=False are
    unconditional by design (the point of the helper is "always
    reproducible", not "reproducible when configured").
    """
    import random

    random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch is imported LOCALLY: src/core/common is imported by data/plot/audit
    # scripts that never touch torch — a module-level import would make
    # every one of them pay torch's multi-second import + CUDA init.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(
        f"[determinism] seed={seed} cudnn.deterministic=True "
        f"(PYTHONHASHSEED note: effective only if set before interpreter "
        f"start)",
        flush=True,
    )

# ── model registry + resolution (shared by every model-loading lane) ────────
MODELS = dict(_CFG["models"])
_MODEL_ROOT = _path(_CFG["paths"]["models_dir"])


def _validate_materialized_model(path: Path, reference: str) -> str:
    """Return one owned model directory, rejecting non-materialized inputs."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"model {reference!r} is not materialized locally: {resolved}. "
            "The Git-shipped project-owned model bundle is missing; "
            "external model downloads are disabled."
        )
    if not any((resolved / marker).is_file() for marker in ("modules.json", "config.json")):
        raise ValueError(
            f"model {reference!r} is not a supported local transformer bundle: "
            f"{resolved} lacks modules.json or config.json"
        )
    return str(resolved)


def resolve_model(key_or_sub: str) -> str:
    """Resolve a registry key, configured subdirectory, or local path.

    Registry values are accepted deliberately because older callers and
    persisted run metadata store the configured subdirectory rather than its
    key. Both forms remain strictly project-local and never trigger a Hub
    download.
    """
    reference = str(key_or_sub).strip()
    if not reference:
        raise ValueError("model reference must be non-empty")

    registry_keys = [key for key, value in MODELS.items() if value == reference]
    if reference in MODELS:
        relative = Path(MODELS[reference])
        candidates = [_MODEL_ROOT / relative]
    elif len(registry_keys) == 1:
        relative = Path(reference)
        candidates = [_MODEL_ROOT / relative]
    elif len(registry_keys) > 1:
        raise ValueError(
            f"model registry value {reference!r} is ambiguous; matching keys: "
            f"{sorted(registry_keys)}"
        )
    else:
        direct = Path(reference).expanduser()
        if direct.is_absolute():
            candidates = [direct]
        elif (TRAIN_ROOT / direct).exists():
            candidates = [TRAIN_ROOT / direct]
        else:
            raise KeyError(
                f"unknown model registry key {reference!r}; "
                f"expected one of {sorted(MODELS)} or an existing local path"
            )

    for candidate in candidates:
        if candidate.is_dir():
            return _validate_materialized_model(candidate, reference)

    checked = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        f"model {reference!r} is not materialized locally; checked: {checked}. "
        "The Git-shipped project-owned model bundle is missing; "
        "external model downloads are disabled."
    )


def load_local_sentence_transformer(
    key_or_path: str,
    *,
    device: str,
    **kwargs: Any,
):
    """Load every bi-encoder through the local registry/DVC contract."""
    from sentence_transformers import SentenceTransformer

    resolved = resolve_model(key_or_path)
    model = SentenceTransformer(resolved, device=device, **kwargs)
    from core.encoding_inputs import enable_zero_truncation
    return enable_zero_truncation(model)


def load_local_cross_encoder(
    key_or_path: str,
    *,
    device: str,
    **kwargs: Any,
):
    """Load every cross-encoder through the local registry/DVC contract."""
    from sentence_transformers import CrossEncoder

    resolved = resolve_model(key_or_path)
    from core.encoding_inputs import enable_cross_encoder_zero_truncation
    return enable_cross_encoder_zero_truncation(CrossEncoder(resolved, device=device, **kwargs))


# ── visibility-log writes (owner directive 2026-09-07) ─────────────────────
# Visibility dumps must survive run collisions: a --sample chain check used
# to overwrite a 3h full run's logs (same name, no run axis). Every dump
# writes BOTH:
#   results/logs/<name>.csv          — the "latest NON-SAMPLE run" copy
#                                      (sample runs never touch it, mirroring
#                                      the fold-metrics pointer discipline)
#   results/logs/<run_tag>/<name>.csv — the run's own copy (every run,
#                                      sample or not)
def write_visibility_log(
    df: pd.DataFrame, name: str, run_tag: str, sample: bool
) -> None:
    """Write a visibility dump under the run-tag dir + latest pointer."""
    run_path = artifact("visibility_run", {"run_tag": run_tag, "name": name})
    ensure_parent(run_path)
    df.to_csv(run_path, index=False)
    trace_artifact("visibility_run", run_path)
    if not sample and not os.environ.get("EUROMONITOR_HPO_RETENTION_MODE"):
        latest_path = artifact("visibility", {"name": name})
        ensure_parent(latest_path)
        df.to_csv(latest_path, index=False)
        trace_artifact("visibility", latest_path)


def _validate_source_export(
    df: pd.DataFrame,
    path: Path,
    *,
    audit: Any | None = None,
) -> None:
    """Fail before downstream work if the configured raw export drifted.

    Both public raw-export loaders call this after parsing.  The row census
    catches additions/removals, while the SSOT-held SHA-256 catches a
    same-sized substitution.  Keep the hash helper as a local import: the
    manifest module imports this config module, so a module-level import
    would create a cycle.
    """
    spec = audit if audit is not None else training_cfg().audit
    observed_rows = len(df)
    expected_rows = spec.source_export_expected_rows
    drift_pct = abs(observed_rows - expected_rows) / expected_rows * 100
    if drift_pct > spec.source_drift_threshold_pct:
        raise SystemExit(
            "source export row-count drift: "
            f"path={path} observed_rows={observed_rows} "
            f"expected_rows={expected_rows} "
            f"observed_drift_pct={drift_pct:.6f} "
            f"allowed_drift_pct={spec.source_drift_threshold_pct:.6f}"
        )

    from core.manifest import sha256_file

    observed_sha256 = sha256_file(path)
    if observed_sha256 != spec.source_export_expected_sha256:
        raise SystemExit(
            "source export sha256 drift: "
            f"path={path} observed_sha256={observed_sha256} "
            f"expected_sha256={spec.source_export_expected_sha256}"
        )


def _read_dataset_csv(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read every dataset lane with the configured string/NA contract."""
    if columns is not None and not columns:
        raise ValueError("source export projection requires at least one column")
    return pd.read_csv(path, usecols=columns, **data_cfg().dataset_csv_read.model_dump())


def _load_source_export(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Parse selected raw columns while retaining the source census/hash guard."""
    df = _read_dataset_csv(DATA_PATH, columns=columns)
    _validate_source_export(df, DATA_PATH)
    return df


def load_dataset(*, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the active source with string dtype and pandas missing-value semantics.

    THE dataset for this series until further notice: euromonitor. Rename
    DATA_PATH + this loader when the active dataset changes; steps import
    `load_dataset`, never a hardcoded path. Columns are canonicalized
    project-wide via COLUMN_MAPPING (config/paths.yaml). Optional ``columns``
    uses canonical names and projects at CSV parsing time; the complete source
    row-count and hash checks still run. Pandas missing-value semantics remain
    the same as the full loader.
    """
    # Imported HERE, not as a module global: COLUMN_MAPPING is reachable as
    # a module attribute only through the lazy __getattr__ below (which
    # fires for attribute access, NOT for a bare global name lookup inside a
    # function). Referencing the bare name here raised NameError, so
    # load_dataset() was dead for every caller — src/training/build_reference,
    # dedupe, zero_shot_sims and the selftest oracle. The lazy re-export is
    # kept for `from core.common import COLUMN_MAPPING` consumers.
    from core.columns import COLUMN_MAPPING as _column_mapping

    raw_columns = None
    if columns is not None:
        canonical_to_raw = {canonical: raw for raw, canonical in _column_mapping.items()}
        unknown = sorted(set(columns) - set(canonical_to_raw))
        if unknown:
            raise ValueError(f"unknown canonical source columns: {unknown}")
        raw_columns = [canonical_to_raw[column] for column in columns]
    df = _load_source_export(raw_columns)
    return df.rename(columns=_column_mapping)


def load_raw_export(*, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """The raw export WITHOUT column renames — the data-prep pipeline
    (src/training/data_prep) works in raw-export column names (gtin, sku_name_eng,
    attribute); the training/eval lane works in canonical ones. Optional
    ``columns`` uses raw names and retains the complete source validation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} missing")
    return _load_source_export(columns)


def load_dataset_deduped(path: Path | None = None) -> pd.DataFrame:
    """The DEDUPED dataset (06 tiered dedupe) — the matching-stage input.

    Step 03+ matching consumes this (one row per retailer-product after
    marketplace-listing collapse); the raw export remains the source of
    truth via load_dataset. Columns are already canonical (written by 06).
    An explicit path uses the same parsing, column, and reviewed-identity
    contracts as the default catalog; it need not depend on a second CSV.
    """
    path = Path(path) if path is not None else F["dataset_deduped"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run src/training/dedupe.py first")
    from core.identity_policy import exclude_reviewed_rows
    from core.columns import CANONICAL_COLUMNS

    df = _read_dataset_csv(path)
    missing_columns = sorted(set(CANONICAL_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"deduped dataset lacks canonical columns: {missing_columns}")
    return exclude_reviewed_rows(df)


# load_euromonitor alias REMOVED (audit 2026-09-09): zero importers —
# every step already used load_dataset (verified by grep before removal).


# ---------------------------------------------------------------------------
# Shared dataframe/regex helpers used by multiple steps.
# ---------------------------------------------------------------------------


def has_gtin(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: row has a non-empty gtin (GTIN)."""
    return df["gtin"].fillna("").astype(str).str.len() > 0


# multi_retailer_mask REMOVED (audit round 2 F18, round 3): defined, never
# called — zero consumers (grep-verified). The same mask is derived inline
# where actually needed (kfold_gtins, report_plots' country slice).


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: non-null count, cardinality, numeric_like, stored dtype.

    numeric_like is the fraction of the first 5k non-null values that parse as
    numbers (loader reads dtype=str, so this shows the real content signal).
    Single source for 01's dtype table and 01b's column scatter.
    """
    rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        cardinality = int(df[col].dropna().nunique())
        numeric_like = 0.0
        if non_null:
            sample = df[col].dropna().head(5000)
            numeric_like = round(
                float(pd.to_numeric(sample, errors="coerce").notna().mean()), 4
            )
        rows.append(
            {
                "column": col,
                "stored_dtype": str(df[col].dtype),
                "non_null": non_null,
                "cardinality": cardinality,
                "numeric_like": numeric_like,
            }
        )
    return pd.DataFrame(rows)


def canonical_volume(series: pd.Series) -> pd.DataFrame:
    """Title series -> (canonical_volume_ml, canonical_volume_ambiguous) frame.

    Canonical volume comes from title ONLY (extract_volume_ml); the ambiguous
    flag marks bare oz/ounce (weight vs fluid). Single source for the
    extract-volume projection used by 01e/01f/01h/02/02b/02c.
    """
    vol = series.map(extract_volume_ml)
    return pd.DataFrame(
        {
            "canonical_volume_ml": vol.map(lambda t: t[0]),
            "canonical_volume_ambiguous": vol.map(lambda t: t[1]),
        }
    )



# ===========================================================================
# SSOT: shared metrics, tokenizers, and split helpers (GATES_MAP.md owner).
# Prior homes of these definitions are noted for traceability; consumers
# import from here now — do NOT reintroduce local copies.
# ===========================================================================
import re as _re

import numpy as _np

# Tokenizer SSOT: one word scheme for the whole series. The three prior
# independent schemes (second01.TOKEN_RE, second02f.WORD_RE,
# second02g._TOK_RE) split "the same word" differently depending on which
# script processed it. This superset keeps 02f's diacritic coverage and
# 02g's intra-word apostrophe/hyphen gluing; second01's plain [a-z0-9]+ is
# a strict subset (see GATES_MAP.md).
TOKEN_RE = _re.compile(
    r"[a-zàâäáéèêëïîôöùûüçñåäöøæé0-9]+(?:[-\'][a-zàâäáéèêëïîôöùûüçñåäöøæé0-9]+)*",
    _re.IGNORECASE,
)


def pair_auc(pos_scores: "_np.ndarray", neg_scores: "_np.ndarray") -> float:
    """AUC of a pos/neg score split (rank-based, no threshold needed).

    Returns NaN when either side is empty (the caller decides how to report).
    Prior homes: second06._auc, second07._auc, second08._auc.
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    y = _np.r_[_np.ones(len(pos_scores)), _np.zeros(len(neg_scores))]
    s = _np.r_[pos_scores, neg_scores]
    return float(roc_auc_score(y, s))


def pair_similarity(emb: "_np.ndarray", pairs_idx: "_np.ndarray") -> "_np.ndarray":
    """Row-pair similarity scores: emb rows paired by (a, b) index columns.

    Prior home: second06._auc_pair. Works on any (N, d) array whose rows are
    L2-normalized (dot product == cosine).
    """
    a = emb[pairs_idx[:, 0]]
    b = emb[pairs_idx[:, 1]]
    return _np.sum(a * b, axis=1)



def kfold_gtins(df: pd.DataFrame, k: int, seed: int | None = None) -> list[set[str]]:
    """K gtin sets over multi-retailer gtins, shuffled and split ~evenly.

    Splits on the gtin (entity) so no product's rows straddle a fold.
    Prior homes: 07b.kfold_gtins, second06.kfold_gtins — this is the
    exact strided-permutation implementation both used, so fold membership
    is unchanged for existing callers.

    AUDIT 2026-09-09 (DATA DROP, now loud): only MULTI-RETAILER gtins
    are dealt into folds. Single-retailer gtins appear in NO fold, so
    under the CV path their positives are silently dropped from every
    test pool by pairs_in_set (measured on test data: a singleton
    gtin's pair vanishes from all k folds). This is a KNOWN, PRINTED
    limitation of the legacy CV mode — the production holdout lane
    (component_folds, src/training/folds.py) does NOT share it: it folds EVERY
    gtin including singletons. Callers must treat the returned folds
    as test-pool keysets, not as dataset coverage.
    """
    gtins = df["gtin"].fillna("").astype(str)
    known = df[gtins.str.len() > 0]
    # Retailer identity through the normalize_retailer SSOT: raw spellings
    # alias across exports and would count one alias group as multi-retailer.
    known = known.assign(_retailer_key=known["retailer"].map(normalize_retailer))
    multi = known[known.groupby("gtin")["_retailer_key"].transform("nunique") > 1]
    bcs = _np.array(sorted(multi["gtin"].unique()))
    perm = _np.random.default_rng(seed if seed is not None else SEED).permutation(
        len(bcs)
    )
    folds = [set(bcs[perm[i::k]]) for i in range(k)]
    n_single = int(
        known.groupby("gtin")["retailer"].nunique().eq(1).sum()
    )
    print(
        f"[kfold_gtins] {len(bcs):,} multi-retailer gtins in {k} folds; "
        f"{n_single:,} single-retailer gtins are in NO fold "
        f"(legacy CV semantics — use component_folds for full coverage)",
        flush=True,
    )
    return folds
