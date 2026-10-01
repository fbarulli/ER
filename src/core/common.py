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
from typing import Any

def _find_project_root() -> Path:
    """Locate the project from stable markers, never a magic parent offset."""
    override = os.environ.get("EUROMONITOR_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "config").is_dir() and (root / "pyproject.toml").is_file():
            return root
        raise RuntimeError(
            "EUROMONITOR_PROJECT_ROOT must contain config/ and pyproject.toml: "
            f"{root}"
        )
    source_file = Path(__file__).resolve()
    for candidate in source_file.parents:
        if (candidate / "config").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not locate project root from {source_file}")


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
    ("attributes", "attr") or ("barcode", "gtin") inline were re-declaring
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

    ONE place to read batch sizes, seq length, eval cadence, dev fraction,
    band edges. Scripts call runtime("batch_size_cpu") etc. — the literal
    lives in config/training.yaml only, never in a script.

    The return is typed `Any` deliberately (audit round 2 F17): the
    training: block holds ints, floats, bools and strings; a narrower lie
    would be worse than the honest open type.

    NO FALLBACKS (owner Q27, audit 2026-09-09): the `default` escape hatch
    (silently returning a caller literal when the key was MISSING) is
    CLOSED. A default may only supply a value when the SSOT defines the
    key as an explicit YAML null (opt-in "use my default"). A truly
    missing key always raises — the literal can never diverge from SSOT.
    """
    tr = _CFG.get("training", {})
    if key in tr:
        val = tr[key]
        if val is not None:
            return val
        if default is not None:
            return default  # explicit null in YAML = caller's default, opt-in
        raise KeyError(
            f"training.{key} is explicitly null in config/training.yaml — "
            f"either set a value or have the caller pass a default"
        )
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

# ── pinned census counts (2026-09-12; mirrors src/training/selftest.py's
# oracle_pinned_counts hard pin) ─────────────────────────────────────────────
# The transductive-census gate_results.csv is the universe BOTH
# src/training/labeled_pairs.py and the selftest oracle read. fallback pairs are
# excluded from the labeled set BY CONSTRUCTION (uncertain tier, would
# inject label noise) — this pin makes that exclusion LOUD and COUNTED
# instead of silent. Same audit lineage as the selftest pin (2026-09-08:
# the pack_qty >= 1 zero-guard fixed 26 gate decisions).
# NOT recomputed here: a pinned constant, updated alongside any
# intentional census drift (paired with the selftest oracle update).
#
# RE-PINNED 2026-09-30: 46,791 -> 41,928. The 46,791 figure was measured
# 2026-09-15 against the gate census committed at 5132e9d
# (proceed 1,737 / hard_no 87,241 / fallback 46,791). The GATE LOGIC has
# moved since (7 commits touch src/pipeline.py after 5132e9d, including
# 4a39fdc "all audit gaps resolved" and 1d146d3), and the current code
# reproducibly yields proceed 1,506 / hard_no 92,335 / fallback 41,928.
# `data/training/data_prep.py` was re-run from the raw export on 2026-09-30
# and reproduced data/gate_results.csv BYTE-IDENTICALLY, so the 4,863-pair
# shortfall is a real gate change, not a corrupt artifact.
# CAVEAT, deliberately not papered over: the shift is large in absolute
# terms (proceed -231, hard_no +5,094). It is NOT attributed here to a
# specific commit — 4a39fdc/033c4ab/1d146d3 all touch gate paths and were
# never individually bisected against the census. This pin now records what
# the CURRENT code produces so the lane is consistent again; auditing WHY
# the gate distribution moved is a separate, open question. It matters
# because this census is the population P0's validation split is drawn from.
#
# RE-PINNED 2026-09-30 (packaging level): 41,928 -> 42,039. Cause is
# UNDERSTOOD, unlike the 46,791->41,928 shift above, which remains
# unattributed. `packaging_level_set` is a new canonical field: 217 of 13,250
# records assert a case-level listing, and the gate now routes a one-sided
# level claim to `fallback` instead of letting it merge silently. Effect is
# proceed -111 (1,506 -> 1,395), fallback +111, and hard_no UNCHANGED at
# 92,335 — a hard negative always outranks the review flag, verified by
# control run with the rule disabled reproducing 92,335/1,506/41,928 exactly.
# See the ordering note in three_way_gate.
#
#
# RE-PINNED 2026-10-01 (wave-1/2 evidence wiring): 42,039 -> 41,748
# (-291). Measured, per-clause, from the gate lane's OWN trace
# (results/logs/training_trace.csv run-abbcd6e6ee2c = the pinned
# 135,769/42,039 generation; run-c60abb62a6cb = the current census):
#   universe 135,769 -> 135,246 (-523 candidates) — the dedupe chain
#     rebuilt dataset_deduped.csv (SHA 73a94016, 63,079 rows; re-run
#     byte-identical, 207 conflicts, closure + identity invariants PASS)
#   Low raw pack confidence   33,833 -> 33,564 (-269)
#   Low raw volume confidence  8,028 -> 8,016 (-12)
#   Packaging level one-sided    113 ->   105 (-8)
#   Ambiguous volume evidence     37 ->    36 (-1)
#   Overlap but low consistency   28 ->    27 (-1)
#   fallback total           42,039 -> 41,748 (-291)
# Cascades (measured, same two trace runs): proceed 1,395 -> 1,239
# (-156; the wave-1/2 evidence captures — pack material from the
# attribute cells, juice-content bands, carbonization/prose,
# description-alias — now resolve pairs the old census left for review)
# and hard_no 92,335 -> 92,259 (-76): Package-material-mismatch 3 -> 253
# (+250 NEW, the attribute-side material capture) and flavor mismatch
# 922 -> 821 (-101), vs Pack blocker 91,410 -> 91,185 (-225, riding the
# -523 candidate universe). CLOSURE: -523 universe = -291 fallback -156
# proceed -76 hard_no exactly. NOT a thinning artifact: the same
# census reproduces the labeled closure below, and labeled_pairs.py's
# assert fired FIRST (excluded 41,748 vs pin 42,039) before this pin
# moved. ALWAYS update this pin with src/training/selftest.py's
# oracle_pinned_counts (same universe).
PINNED_GATE_FALLBACK_PAIRS = 41_748


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
    return SentenceTransformer(resolved, device=device, **kwargs)


def load_local_cross_encoder(
    key_or_path: str,
    *,
    device: str,
    **kwargs: Any,
):
    """Load every cross-encoder through the local registry/DVC contract."""
    from sentence_transformers import CrossEncoder

    resolved = resolve_model(key_or_path)
    return CrossEncoder(resolved, device=device, **kwargs)


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


def load_dataset() -> pd.DataFrame:
    """Load the ACTIVE dataset as raw strings (no silent coercion).

    THE dataset for this series until further notice: euromonitor. Rename
    DATA_PATH + this loader when the active dataset changes; steps import
    `load_dataset`, never a hardcoded path. Columns are canonicalized
    project-wide via COLUMN_MAPPING (config/paths.yaml).
    """
    df = pd.read_csv(DATA_PATH, dtype=str)
    _validate_source_export(df, DATA_PATH)
    return df.rename(columns=COLUMN_MAPPING)


def load_raw_export() -> pd.DataFrame:
    """The raw export WITHOUT column renames — the data-prep pipeline
    (src/training/data_prep) works in raw-export column names (gtin, sku_name_eng,
    attribute); the training/eval lane works in canonical ones."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} missing")
    df = pd.read_csv(DATA_PATH, dtype=str)
    _validate_source_export(df, DATA_PATH)
    return df


def load_dataset_deduped() -> pd.DataFrame:
    """The DEDUPED dataset (06 tiered dedupe) — the matching-stage input.

    Step 03+ matching consumes this (one row per retailer-product after
    marketplace-listing collapse); the raw export remains the source of
    truth via load_dataset. Columns are already canonical (written by 06).
    """
    path = F["dataset_deduped"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run src/training/dedupe.py first")
    from core.identity_policy import exclude_reviewed_rows
    return exclude_reviewed_rows(pd.read_csv(path, dtype=str))


# load_euromonitor alias REMOVED (audit 2026-09-09): zero importers —
# every step already used load_dataset (verified by grep before removal).


# ---------------------------------------------------------------------------
# Shared dataframe/regex helpers used by multiple steps.
# ---------------------------------------------------------------------------


def has_barcode(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: row has a non-empty barcode (GTIN)."""
    return df["barcode"].fillna("").astype(str).str.len() > 0


# multi_retailer_mask REMOVED (audit round 2 F18, round 3): defined, never
# called — zero consumers (grep-verified). The same mask is derived inline
# where actually needed (kfold_barcodes, report_plots' country slice).


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



def kfold_barcodes(df: pd.DataFrame, k: int, seed: int | None = None) -> list[set[str]]:
    """K barcode sets over multi-retailer barcodes, shuffled and split ~evenly.

    Splits on the barcode (entity) so no product's rows straddle a fold.
    Prior homes: 07b.kfold_barcodes, second06.kfold_barcodes — this is the
    exact strided-permutation implementation both used, so fold membership
    is unchanged for existing callers.

    AUDIT 2026-09-09 (DATA DROP, now loud): only MULTI-RETAILER barcodes
    are dealt into folds. Single-retailer barcodes appear in NO fold, so
    under the CV path their positives are silently dropped from every
    test pool by pairs_in_set (measured on test data: a singleton
    barcode's pair vanishes from all k folds). This is a KNOWN, PRINTED
    limitation of the legacy CV mode — the production holdout lane
    (component_folds, src/training/folds.py) does NOT share it: it folds EVERY
    barcode including singletons. Callers must treat the returned folds
    as test-pool keysets, not as dataset coverage.
    """
    barcodes = df["barcode"].fillna("").astype(str)
    known = df[barcodes.str.len() > 0]
    # Retailer identity through the normalize_retailer SSOT: raw spellings
    # alias across exports and would count one alias group as multi-retailer.
    known = known.assign(_retailer_key=known["retailer"].map(normalize_retailer))
    multi = known[known.groupby("barcode")["_retailer_key"].transform("nunique") > 1]
    bcs = _np.array(sorted(multi["barcode"].unique()))
    perm = _np.random.default_rng(seed if seed is not None else SEED).permutation(
        len(bcs)
    )
    folds = [set(bcs[perm[i::k]]) for i in range(k)]
    n_single = int(
        known.groupby("barcode")["retailer"].nunique().eq(1).sum()
    )
    print(
        f"[kfold_barcodes] {len(bcs):,} multi-retailer barcodes in {k} folds; "
        f"{n_single:,} single-retailer barcodes are in NO fold "
        f"(legacy CV semantics — use component_folds for full coverage)",
        flush=True,
    )
    return folds
