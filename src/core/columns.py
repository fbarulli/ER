"""core.columns — THE column-name SSOT (owner ruling 2026-10-01).

Every column name in this project is declared once, in
config/paths.yaml, and read from here. No module declares a column-name
tuple of its own.

WHY THIS EXISTS (the same failure, three times now):

* Volume tolerance. Five lanes each carried a ``vol_tolerance`` literal and
  each defaulted the ABSOLUTE tolerance to 0.0. The relative cut is the
  stricter rule at small volumes, so the census and the veto lane answered
  differently about the same pair. One tolerance, five declarations.
* Stage-7 clarification. ``AttributeDecisionEngine._fallback_reparse`` reads
  ``attributes``/``title``/``description``; ``three_way_gate`` handed it a
  canonical record carrying none of them. The mechanism existed, the column
  names existed, and the two were never joined — so it never fired.
* Column tuples. ``RAW_EXPORT_REQUIRED_COLUMNS``,
  ``CANONICAL_DATASET_REQUIRED_COLUMNS``, and the source-row field list were
  each written out by hand while ``config/paths.yaml`` already declared every
  column on both sides as a verified bijection.

The pattern is one declaration per concept, read everywhere. A second copy
cannot be steered by the config and drifts silently. This module makes the
config the only place a column name may come from, and fails loudly at load
when the declarations disagree (see ``DataConfig`` validators in
core.schemas).

VOCABULARIES. The raw export and the canonical dataset name the SAME
thirteen columns differently (``sku_name_eng`` vs ``title``). Both vocabularies
are derived from the single ``column_mapping`` rather than typed out, so a
rename is a one-line config edit that every lane follows.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "CANONICAL_COLUMNS",
    "COLUMN_ALIASES",
    "CANONICAL_DATASET_REQUIRED_COLUMNS",
    "COLUMN_MAPPING",
    "DATA_PREP_REQUIRED_COLUMNS",
    "DESCRIPTOR_COLUMNS",
    "NON_DESCRIPTOR_COLUMNS",
    "RAW_EXPORT_COLUMNS",
    "SOURCE_ROW_FIELDS",
    "SOURCE_ROW_FIELD_NAMES",
    "SOURCE_ROW_RAW_NAMES",
    "alias_names",
    "canonical_of",
    "raw_of",
    "read_column",
    "require_canonical_columns",
    "require_raw_columns",
    "source_row_pairs",
]


def _cfg():
    # Imported lazily: core.common imports this module for COLUMN_MAPPING, so a
    # module-level import would close a cycle.
    from core.common import data_cfg

    return data_cfg()


def _build() -> tuple[
    Mapping[str, str],
    tuple[str, ...],
    tuple[str, ...],
    Mapping[str, str],
    tuple[str, ...],
    tuple[str, ...],
]:
    cfg = _cfg()
    mapping = MappingProxyType(dict(cfg.column_mapping))
    raw = tuple(sorted(mapping))
    canonical = tuple(sorted(set(mapping.values())))

    # The capture is the subset of column_evidence with capture: true; the
    # excluded ones keep their reason in the config beside the ruling, so an
    # exclusion is auditable rather than merely absent.
    source_fields = MappingProxyType(
        {
            name: spec.column
            for name, spec in cfg.column_evidence.items()
            if spec.capture
        }
    )
    return (
        mapping,
        raw,
        canonical,
        source_fields,
        tuple(cfg.data_prep_required_columns),
        tuple(cfg.data_prep_required_columns),
    )


(
    COLUMN_MAPPING,
    RAW_EXPORT_COLUMNS,
    CANONICAL_COLUMNS,
    SOURCE_ROW_FIELDS,
    DATA_PREP_REQUIRED_COLUMNS,
    _CANONICAL_DATASET_REQUIRED_COLUMNS,
) = _build()

# The canonical-dataset lane consumes the RENAMED frame, so its requirement is
# the same list mapped across. Derived, not retyped.
CANONICAL_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    sorted({COLUMN_MAPPING[raw] for raw in DATA_PREP_REQUIRED_COLUMNS})
)

# Field names as they appear in the source_rows JSON (== canonical column
# names; see _build).
SOURCE_ROW_FIELD_NAMES: tuple[str, ...] = tuple(
    sorted(SOURCE_ROW_FIELDS.values())
)


# Which columns the capture REJECTS, with the reasons, read from the same
# declaration (column_evidence capture: false). Derived so nothing retypes the
# exclusion list.
#
# NOTE: there is deliberately NO descriptor/non-descriptor list here. That
# concept belongs to core.product_identity (DESCRIPTOR_COLUMNS), which already
# declares it; restating it in a second module is precisely the duplication
# this file exists to remove, and a test cross-checking the two copies would
# only make the duplication look justified.
EXCLUDED_SOURCE_ROW_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        name: spec.reason
        for name, spec in _cfg().column_evidence.items()
        if not spec.capture
    }
)

# Extra read names per column (config/paths.yaml column_aliases).
COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {column: tuple(aliases) for column, aliases in _cfg().column_aliases.items()}
)


def alias_names(canonical: str) -> tuple[str, ...]:
    """Every name a column answers to, canonical first.

    A read path that accepts "either name" — ("attributes", "attr"),
    ("description", "description_short_eng"), {"title": "sku_name_eng"} — was
    hardcoding a fact column_mapping and column_aliases already declare. Those
    literals are a second declaration that drifts on rename, and the drift is
    silent: the reader simply stops finding the column.
    """
    names = [canonical]
    raw = raw_of(canonical)
    if raw and raw != canonical:
        names.append(raw)
    names.extend(COLUMN_ALIASES.get(canonical, ()))
    return tuple(names)


def read_column(row: Mapping[str, object], canonical: str, default: object = "") -> object:
    """First non-empty value among ``canonical``'s names, else ``default``."""
    for name in alias_names(canonical):
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def raw_of(canonical: str) -> str | None:
    """Canonical column name -> its raw-export name, or None if unmapped."""
    for raw, name in COLUMN_MAPPING.items():
        if name == canonical:
            return raw
    return None


def canonical_of(raw: str) -> str | None:
    """Raw-export column name -> its canonical name, or None if unmapped."""
    return COLUMN_MAPPING.get(raw)


def source_row_pairs() -> tuple[tuple[str, str], ...]:
    """The per-title capture as (JSON key, raw column) pairs.

    Ordered by RAW column name so the writer is byte-deterministic. The JSON
    key is the canonical name (what downstream reads), the raw column is where
    the value actually lives in the export — the writer must not have to know
    that those two vocabularies differ, and must not guess.
    """
    pairs = []
    for canonical in SOURCE_ROW_FIELDS.values():
        raw = raw_of(canonical)
        if raw is None:  # unreachable: DataConfig validates this at load
            raise KeyError(
                f"source_row_fields names {canonical!r}, which column_mapping "
                f"does not map to a raw column"
            )
        pairs.append((canonical, raw))
    return tuple(sorted(pairs, key=lambda pair: pair[1]))


# The raw columns the writer actually reads. Derived, never retyped: the two
# vocabularies differ (sku_name_eng vs title) and the writer must not have to
# carry that knowledge in a literal.
SOURCE_ROW_RAW_NAMES: tuple[str, ...] = tuple(raw for _, raw in source_row_pairs())


def require_raw_columns(frame: object, required: tuple[str, ...] | None = None) -> None:
    """Fail loudly when a RAW-export frame is missing a required column."""
    required = DATA_PREP_REQUIRED_COLUMNS if required is None else required
    present = set(getattr(frame, "columns", ()))
    missing = [column for column in required if column not in present]
    if missing:
        raise ValueError(
            f"raw-export frame missing required columns {missing}; "
            f"required={list(required)}"
        )


def require_canonical_columns(
    frame: object, required: tuple[str, ...] | None = None
) -> None:
    """Fail loudly when a CANONICAL frame is missing a required column."""
    required = (
        CANONICAL_DATASET_REQUIRED_COLUMNS if required is None else required
    )
    present = set(getattr(frame, "columns", ()))
    missing = [column for column in required if column not in present]
    if missing:
        raise ValueError(
            f"canonical frame missing required columns {missing}; "
            f"required={list(required)}"
        )