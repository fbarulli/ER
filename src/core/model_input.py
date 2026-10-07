"""The single source of truth for the text the encoder sees.

Both lanes — training candidate retrieval (``training.rand_matching``) and
scoring (``predict_items``) — plus the payload/audit stage in ``pipeline``
build their model input HERE.  Before this module the same composition was
copy-pasted in all three places, which is precisely how the two lanes drifted
apart: the source side read the raw ``description``/``category``/``breadcrumbs``
columns while the target side read ``description_evidence``/``breadcrumb_evidence``
from the canonical record, so one product produced two unrelated strings.

Profile selection lives in ``config/training.yaml`` (``training.model_input``)
and is validated by ``core.schemas.TrainingSpec.ModelInputSpec``:

``cleaned``
    The shipped default.  ``[Brand] [Title] [Attributes]`` composition — those
    three field groups, in that order, and nothing else.  Underscore compounds
    are split so they can lexically match source text, discriminative numbers
    survive into the model input, and the description/breadcrumb evidence
    channel is excluded.
``legacy``
    The pre-change composition, reproduced byte for byte.  Kept selectable
    from config so going back is a config edit rather than a code revert;
    ``tests/test_model_input_contract.py`` pins it against fixtures captured
    from the unmodified code.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache

import pandas as pd  # frame access in build_sku_texts (consolidated loop)

from core.columns import alias_names
from core.common import load_config, row_metadata_text, config_section
from core.run_log import RunLogger
from core.schemas import TrainingSpec

_LOG = RunLogger(__name__)

__all__ = [
    "build_canonical_text",
    "build_sku_text",
    "implicit_pack_qty",
    "model_input_composition",
    "model_input_info",
    "model_input_spec",
    "token_budget_report",
]

# Percentage evidence, captured before normalize_text removes the sign.
# "0-2%" is a range (juice content bands), "100%" a single value.
_PERCENT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# the [FIELD_*] groups the structured channel can emit
_FIELD_GROUP_RE = re.compile(r"\[FIELD_([A-Z_]+)\]")


def model_input_spec() -> TrainingSpec.ModelInputSpec:
    """The validated model-input composition settings (config SSOT)."""
    return TrainingSpec.ModelInputSpec.model_validate(
        config_section('training', 'model_input', loader=load_config)
    )


def model_input_composition() -> TrainingSpec.ModelInputComposition:
    """The ACTIVE composition, for fingerprints, traces, manifests, bundles.

    Any artifact whose contents depend on the encoder text — a persisted
    embedding index, a checkpoint, a frozen payload bundle above all — must be
    able to name the exact input contract that produced it.  Changing the
    profile changes the text but NOT the catalog, the checkpoint or the code
    path, so without this the artifact looks reusable when it is not.
    """
    return TrainingSpec.ModelInputComposition.from_spec(model_input_spec())


def implicit_pack_qty() -> float:
    """The configured implicit pack count for an unobserved pack (SSOT)."""
    return float(config_section('training', 'structured_features', 'implicit_pack_qty', loader=load_config))


def token_budget_report(
    texts: Sequence[str], *, tokenizer, max_seq_length: int
):
    """Count what the encoder window KEEPS, and name every dropped field group.

    The ``[FIELD_*]`` groups are appended last, so at ``max_seq_length`` they
    are truncated first. Measured before this guard existed: 11.9 % of target
    texts lost the whole structured tail, 27.7 % exceeded the window, and none
    of it was visible anywhere except a shorter string.

    Returns ``core.schemas.TokenBudgetReport``: one record per payload, plus a
    count per dropped field group, so a lost group is a named number rather
    than an absence. Never raises on a long payload — the caller decides
    whether to reorder fields or raise the budget.
    """
    from core.schemas import TokenBudgetReport

    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")

    n_over = 0
    dropped: dict[str, int] = {}
    for text in texts:
        encoded = tokenizer(
            str(text), add_special_tokens=True, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
        if len(offsets) <= max_seq_length:
            continue
        n_over += 1
        kept_end = offsets[max_seq_length - 1][1]
        for group in _FIELD_GROUP_RE.findall(str(text)):
            marker = f"[FIELD_{group}]"
            if str(text).find(marker) >= kept_end:
                dropped[group] = dropped.get(group, 0) + 1
    return TokenBudgetReport(
        max_seq_length=int(max_seq_length),
        n_records=len(texts),
        n_over_budget=n_over,
        n_field_groups_dropped=sum(dropped.values()),
        dropped_groups=dict(sorted(dropped.items())),
    )


def _structured_text_enabled() -> bool:
    """Whether the structured token channel is appended to the model text.

    Owned here so the three call sites stop recomputing the same
    ``enabled and append_to_text`` pair from their own config reads.
    """
    cfg = config_section('training', 'structured_features', loader=load_config)
    return bool(cfg["enabled"]) and bool(cfg["append_to_text"])


@lru_cache(maxsize=65536)
def _normalized_stream(text: str) -> str:
    """The percent-protected, normalized, underscore-split stream (memo).

    OPTIMIZATION (redundant-across-rows): identical brand strings recur on
    thousands of listing rows (and the same title+attributes on the 4.7
    same-canonical listings). The stream is a pure function of the input
    string ONLY — the schema strip runs on top, live (see _normalized_tokens),
    so audit-time _MODEL_STOP changes are always honored.
    """
    from pipeline import normalize_text

    protected = _PERCENT_RANGE_RE.sub(
        lambda m: f" pct{m.group(1)}to{m.group(2)} ", text
    )
    protected = _PERCENT_RE.sub(lambda m: f" pct{m.group(1)} ", protected)
    return " ".join(
        part
        for token in normalize_text(protected).split()
        for part in token.split("_")
        if part
    )


def _normalized_tokens(text: object, *, drop_schema_words: bool) -> list[str]:
    """One normalizer for BOTH lanes.

    Underscore compounds are split into their parts so a canonical token such
    as ``bcaa_pear`` can match the source words ``bcaa`` and ``pear``.  Digits
    are deliberately preserved: ``6000mg`` is discriminative, and no
    number-stripping runs on this path.

    Percentage values become ``pct*`` tokens BEFORE ``normalize_text`` deletes
    the ``%`` sign.  92.6% of the review-band source rows carry a juice-content
    percentage (``100%``, ``0-2%``, ...), and once the sign is gone the bare
    number collides with the ``MINIMAL_STOPWORDS`` volume entries (``100``,
    ``2``), so the attribute was silently discarded on the legacy path.

    Diacritics are NOT folded here (owner ruling): brand-string normalisation
    is owned by the dedicated brand-analysis work, and the measured evidence
    for it is recorded in MODEL_INPUT_FIX_REPORT.md section 24 as input to that
    deliberate decision rather than being applied ahead of it.
    """
    from pipeline import MINIMAL_STOPWORDS, strip_schema_words

    # Diacritics are deliberately NOT folded here (owner ruling — see the
    # docstring). normalize_text turns every non-ASCII letter into a word
    # break, so an accented brand arrives split: "Brämhults" -> "br mhults"
    # and "Reál"/"Réal" -> "re"/"al". The measured consequence is recorded in
    # MODEL_INPUT_FIX_REPORT.md 20.1 — 3 catalog clusters / 28 GTINs / 0
    # labelled pairs, and 378 of the 388 review-band brand mismatches are
    # outright different brands — so this is a retrieval problem, not a
    # composition one, and brand-string normalisation is owned by the
    # dedicated brand-analysis work (commit 38358bf reverted the interim fix).
    split = _normalized_stream(str(text))
    if drop_schema_words:
        # Live read: strip_schema_words is the audit-pinned seam — never
        # memoized, so _MODEL_STOP changes are always honored here.
        split = strip_schema_words(split)
    return [t for t in split.split() if len(t) > 1 and t not in MINIMAL_STOPWORDS]


# The structured channel prefixes that can carry a plain token's twin, e.g.
# "carbonation_still" beside "still". Used by the redundancy removal below.
_STRUCTURED_PREFIXES = (
    "carbonation_",
    "sweetener_",
    "pulp_",
    "flavor_",
    "package_type_",
)


def _reduce_redundancy(text: str, *, spec: TrainingSpec.ModelInputSpec) -> str:
    """Apply the configured redundancy removals to ONE composed text.

    Measured on the real 13,250 canonicals (209,134 tokens, 15.8/text); each
    removal is independently selectable so it can be A/B'd and rolled back by
    config alone, and all three default to today's bytes:

    * ``emit_field_markers`` — the ``[FIELD_*]`` group markers are 55,100 tokens
      = 26.35% of the payload and carry structure only.
    * ``keep_redundant_attribute_words`` — a plain token whose structured twin
      is present in the SAME text is pure duplication: ``still``/
      ``carbonation_still`` co-occur on 100.0% of the rows that carry ``still``,
      ``carbonated``/``carbonation_carbonated`` on 99.8%. The twin is emitted
      from the same attribute set, so dropping the plain token loses no
      attribute. Measured coverage is deliberately used instead of a hardcoded
      pair list: the corpus also carries ``sweetener_diet_sugar``, whose suffix
      is ``diet_sugar``, so plain ``sugar`` is NOT covered and is NOT dropped.
    * ``emit_singleton_pack_token`` — ``pack_qty_1`` sits in 75.8% of texts. The
      numeric structured vector records pack presence and value independently,
      so the text token is not the only record of it; a real multi-pack
      (``pack_qty_6``) is untouched.
    """
    if (
        spec.emit_field_markers
        and spec.keep_redundant_attribute_words
        and spec.emit_singleton_pack_token
    ):
        return text  # nothing selected: leave the string byte-identical
    tokens = text.split()
    if not spec.emit_field_markers:
        tokens = [t for t in tokens if not t.startswith("[FIELD_")]
    if not spec.keep_redundant_attribute_words:
        structured = {t for t in tokens if t.startswith(_STRUCTURED_PREFIXES)}
        twins = {done.split("_", 1)[1] for done in structured}
        tokens = [t for t in tokens if not (t in twins and t not in structured)]
    if not spec.emit_singleton_pack_token:
        tokens = [t for t in tokens if t != "pack_qty_1"]
        # A group marker must not survive with no value left in its group.
        if not any(t.startswith("pack_qty_") for t in tokens):
            tokens = [t for t in tokens if t != "[FIELD_PACK_SIZE]"]
    return " ".join(tokens)


def _cleaned_sku_text(
    row, info: Mapping[str, object], *, spec: TrainingSpec.ModelInputSpec
) -> str:
    """[Brand] [Title] [Attributes], in that order, with no other field.

    The three blocks are emitted as plain tokens in a fixed order rather than
    behind literal ``[BRAND]``-style markers: measured on both fixture
    populations, the markers are constant mass shared by every pair, so they
    lifted cross-pair token overlap (0.084 -> 0.164 on the singleton-GTIN
    group) further than true-pair overlap and lowered the true-vs-cross
    margin.  Field SELECTION and ORDER carry the user's requested structure;
    the markers cost discrimination.
    """
    from core.structured_features import append_text

    tokens: list[str] = []
    tokens += _normalized_tokens(row_metadata_text(row, "brand"), drop_schema_words=False)
    tokens += _normalized_tokens(row_metadata_text(row, "sku_name_eng"), drop_schema_words=False)
    tokens += _normalized_tokens(
        row_metadata_text(row, *alias_names("attribute")), drop_schema_words=True
    )
    symmetric = model_input_info(info, spec=spec)
    return _reduce_redundancy(
        append_text(" ".join(tokens), symmetric, enabled=_structured_text_enabled()),
        spec=spec,
    )


def _cleaned_canonical_text(
    record: Mapping[str, object],
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec,
) -> str:
    """[Brand] [Title] [Attributes] for the canonical side, same normalizer."""
    from core.structured_features import append_text

    brand = _normalized_tokens(record.get("mode_brand", ""), drop_schema_words=False)
    brand_words = {word.casefold() for word in brand}
    canonical = [
        word
        for word in _normalized_tokens(
            record.get("canonical", ""), drop_schema_words=True
        )
        if word.casefold() not in brand_words
    ]
    tokens = [
        *brand,
        *canonical,
        *_normalized_tokens(record.get("mode_type", ""), drop_schema_words=True),
    ]
    symmetric = model_input_info(info, spec=spec)
    return _reduce_redundancy(
        append_text(" ".join(tokens), symmetric, enabled=_structured_text_enabled()),
        spec=spec,
    )


def _legacy_sku_text(row, info: Mapping[str, object]) -> str:
    from core.critical_attributes import extract_flavor_tokens
    from core.structured_features import append_text
    from ner.ner_product_attributes import extract_title_attributes
    from pipeline import clean_sku_text, strip_schema_words

    base = strip_schema_words(clean_sku_text(
        row_metadata_text(row, "sku_name_eng"),
        row_metadata_text(row, *alias_names("attribute")),
        row_metadata_text(row, "brand"),
        row_metadata_text(row, *alias_names("description_short_eng")),
        row_metadata_text(row, *alias_names("category")),
        row_metadata_text(row, *alias_names("breadcrumbs_eng")),
    ))
    # The legacy profile is a byte-for-byte rollback contract. The active
    # cleaned profile now accepts declared Pack Type when the title has none,
    # but that new evidence must not silently alter old legacy payloads.
    legacy_info = dict(info)
    legacy_info.pop("sweetener_type", None)
    legacy_info.pop("sweetening", None)
    # A title-only ablation supplies a precomputed full-row ``info`` while
    # blanking attributes in the row; keep that supplied structured channel.
    if row_metadata_text(row, *alias_names("attribute")).strip():
        legacy_info["package_type"] = set(extract_title_attributes(
            row_metadata_text(row, "sku_name_eng")
        )["package_types"])
        legacy_info["flavor"] = set(extract_flavor_tokens(
            row_metadata_text(row, "sku_name_eng"),
            row_metadata_text(row, *alias_names("attribute")),
        ))
    return append_text(base, legacy_info, enabled=_structured_text_enabled())


def _legacy_canonical_text(
    record: Mapping[str, object], info: Mapping[str, object], *, evidence: bool
) -> str:
    from core.structured_features import append_text
    from pipeline import (
        canonical_evidence_text,
        canonical_model_text,
        strip_schema_words,
    )

    parts = [
        str(record.get("canonical", "")),
        str(record.get("mode_brand", "")),
        str(record.get("mode_type", "")),
    ]
    if evidence:
        parts.append(canonical_evidence_text(record.get("description_evidence", "")))
        parts.append(canonical_evidence_text(record.get("breadcrumb_evidence", "")))
    base = strip_schema_words(canonical_model_text(" ".join(parts)))
    legacy_info = {key: value for key, value in info.items()
                   if key not in {"sweetener_type", "sweetening"}}
    return append_text(base, legacy_info, enabled=_structured_text_enabled())


def model_input_info(
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec | None = None,
) -> dict[str, set[float] | set[str]]:
    """The structured info as the ACTIVE composition sees it.

    Callers use this once per row and feed the result to BOTH the text builder
    and the numeric vector, so an attribute's unobserved treatment can never
    differ between the two channels.

    ``legacy`` returns the info unchanged (its historical, one-sided treatment
    is what the golden fixtures pin). ``cleaned`` applies the universal
    implicit-default rule, which is the shipped default.
    """
    resolved = _resolve(spec)
    if resolved.profile != "cleaned":
        return dict(info)
    from core.structured_features import symmetric_info

    return symmetric_info(info, implicit_pack_qty=implicit_pack_qty())


def _resolve(
    spec: TrainingSpec.ModelInputSpec | None,
) -> TrainingSpec.ModelInputSpec:
    return model_input_spec() if spec is None else spec


def build_sku_text(
    row,
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec | None = None,
) -> str:
    """Model text for one SOURCE sku row.

    ``row`` is a pandas row/Series so field fallbacks go through the shared
    ``core.common.row_metadata_text`` reader; ``info`` is the structured
    attribute mapping the caller already built for the numeric channel.
    """
    resolved = _resolve(spec)
    if resolved.profile == "cleaned":
        return _cleaned_sku_text(row, info, spec=resolved)
    return _legacy_sku_text(row, info)


class _RowProxy:
    """A Mapping-backed row object exposing the Series access surface
    (`name in row.index`, `row[name]`) so a worker task's row can be a plain
    dict — same returns as the frame's own row Series for these readers."""

    __slots__ = ('_record', 'index')

    def __init__(self, record: dict) -> None:
        self._record = record
        self.index = record.keys()

    def __getitem__(self, name):
        return self._record[name]


_SKU_STATE: dict[str, object] = {'records': None, 'structured': True}


def sku_row_payload(record: dict) -> tuple[dict[str, set], str]:
    """One row's payload from its plain record dict (pure, worker-reusable)."""
    from core.structured_features import sku_info as sku_structured_info

    row = _RowProxy(record)
    info = model_input_info(sku_structured_info(
        row_metadata_text(row, "sku_name_eng"),
        row_metadata_text(row, *alias_names("attribute")),
        row_metadata_text(row, *alias_names("description_short_eng")),
    ))
    return info, build_sku_text(row, info)


def sku_row_task(idx: int) -> tuple[dict[str, set], str]:
    """One row's payload from the fork-inherited record table (worker entry).

    The parent ships ONE int per row; the record table is inherited through
    fork COW and memoized in the worker on its first task (never re-shipped)."""
    records = _SKU_STATE['records']
    if records is None:  # pragma: no cover — fork COW inheritance publishes them
        raise RuntimeError('sku pool worker started without the inherited record table')
    return sku_row_payload(records[idx])


class SkuTextPool:
    """Fork-parallel builder of one frame's sku payload (texts + infos).

    One SR owner of the worker lifecycle, mirroring pipeline.CanonicalCardPool:
    each task is a pure function of ONE row record (a plain dict from the
    frame's columns), fork-inherited module state supplies the config-driven
    extractors, and pool.map yields results in submission order — so both
    the infos and the texts are byte-identical to the sequential composition,
    whichever path the row count selects.

    Phase map:
      bind           — publish the variant frame's row records (fork COW)
      row_task       — one row: structured info + encoder text, ordered
      build          — sequential below the threshold (small frames/ tests),
                       fork-parallel across cores otherwise
    """

    _INLINE_THRESHOLD = 4096
    _CHUNKSIZE = 64

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._structured_enabled = True

    def bind(self, rows: list[dict], *, structured_enabled: bool) -> None:
        self._rows = rows
        self._structured_enabled = structured_enabled


    def build(self, frame, *, structured_enabled: bool) -> tuple[list[str], list[dict[str, set]]]:
        rows: list[dict] = frame.to_dict("records")
        _SKU_STATE['records'] = rows
        _SKU_STATE['structured'] = structured_enabled
        if len(rows) < self._INLINE_THRESHOLD:
            replies = [
                sku_row_payload(row)
                for row in _LOG.progress(rows, desc='sku-structured', unit='row')
            ]
            return [text for _info, text in replies], [info for info, _text in replies]
        _LOG.info(
            f"payload: fork-parallel sku-text compose over {len(rows):,} rows"
        )
        import concurrent.futures
        from multiprocessing import get_context
        infos, texts = [], []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(2, os.cpu_count() - 1), mp_context=get_context('fork'),
        ) as pool:
            bar = _LOG.bar(total=len(rows), desc='sku-text-pool', unit='row')
            try:
                for info, text in pool.map(sku_row_task, range(len(rows)), chunksize=self._CHUNKSIZE):
                    infos.append(info)
                    texts.append(text)
                    bar.update()
            finally:
                bar.close()
        return texts, infos


_SKU_TEXT_POOL = SkuTextPool()


class SkuPayloadComposer:
    """One frame's per-row payload composition (the SSOT loop).

    SR phases, ONE fixed order in compose(); statements are the original
    build_sku_texts body verbatim, so (texts, infos) are byte-identical and
    the bars keep their labels.

    Phase map:
      resolve_columns — title/attribute defaults + the description alias
                        diplomacy (canonical name wins, alias fills NaN,
                        same-index so duplicate indexes cannot misalign)
      structured_infos — the per-row numeric-channel info (empty sets when
                        structured features are disabled)
      encode_texts    — the per-row encoder text via the documented call
    """

    def __init__(self, frame: pd.DataFrame, *, structured_enabled: bool) -> None:
        self._frame = frame
        self._structured_enabled = structured_enabled

    # ── phase: column diplomacy ─────────────────────────────────────────────

    def resolve_columns(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        frame = self._frame
        from core.structured_features import sku_info as sku_structured_info

        title = frame["sku_name_eng"].fillna("") if "sku_name_eng" in frame else pd.Series([""] * len(frame))
        attrs = frame["attribute"].fillna("") if "attribute" in frame else pd.Series([""] * len(frame))
        # Description column resolved by bounded alias: the raw Euromonitor
        # export names the column ``description_short_eng``, which is also the
        # canonical name (config/paths.yaml column_mapping is the identity);
        # frames from the transition may still carry the old ``description``
        # alias. Before this resolution the renamed lane silently lost its
        # description (sku_info saw ""), because the reader only knew one name.
        # The canonical name wins when both exist so a canonical-frame path can
        # never change behavior simply because an alias twin happened to ride
        # along; only a NaN in the canonical column is filled from the alias.
        # The fill uses the frame's own index ( .where with a same-index
        # series, not a positional fillna against a fresh RangeIndex series)
        # so duplicated / non-default row indexes cannot misalign. No lane's
        # output changes when the canonical name is present and observed —
        # the NaN cases were already coerced to "" downstream — only the
        # previously empty alias lane regains its description.
        if "description_short_eng" in frame:
            raw = frame["description_short_eng"]
            renamed = frame["description"] if "description" in frame else pd.Series(
                [""] * len(frame), index=frame.index
            )
            descriptions = raw.where(raw.notna(), renamed)
        elif "description" in frame:
            descriptions = frame["description"]
        else:
            descriptions = pd.Series([""] * len(frame), index=frame.index)
        del sku_structured_info
        return title, attrs, descriptions

    # ── phase: structured infos ─────────────────────────────────────────────

    def structured_infos(self, title: pd.Series, attrs: pd.Series,
                         descriptions: pd.Series) -> list[dict[str, set]]:
        from core.structured_features import sku_info as sku_structured_info

        if self._structured_enabled:
            infos = [
                model_input_info(sku_structured_info(t, a, d))
                for t, a, d in _LOG.progress(
                    zip(title, attrs, descriptions, strict=True),
                    total=len(title), unit="row", desc="sku-structured",
                )
            ]
        else:
            empty_info: dict[str, set] = {
                "volume": set(),
                "pack": set(),
                "package_type": set(),
            }
            infos = [
                {k: set(v) for k, v in empty_info.items()}
                for t, a, d in _LOG.progress(
                    zip(title, attrs, descriptions, strict=True),
                    total=len(title), unit="row", desc="sku-structured",
                )
            ]
        return infos

    # ── phase: encoder texts ────────────────────────────────────────────────

    def encode_texts(self, infos: list[dict[str, set]]) -> list[str]:
        texts = [
            build_sku_text(row, info)
            for (_, row), info in _LOG.progress(
                zip(self._frame.iterrows(), infos, strict=True),
                total=len(self._frame),
                unit="row",
                desc="sku-text",
            )
        ]
        return texts

    # ── orchestration ───────────────────────────────────────────────────────

    def compose(self) -> tuple[list[str], list[dict[str, set]]]:
        """(texts, infos): text + numeric channel built once, never disagreeing.

        Fork-parallel when the frame is worth a pool: the per-row
        composition is a pure function of the row record, so the pool's
        results are the sequential bytes with N-fold wall-clock.
        """
        return _SKU_TEXT_POOL.build(self._frame, structured_enabled=self._structured_enabled)


def build_sku_texts(
    frame: pd.DataFrame,
    *,
    structured_enabled: bool,
) -> tuple[list[str], list[dict[str, set]]]:
    """Finalized encoder text + structured info for EVERY source row.

    THE single per-row composition. This loop (sku_info -> model_input_info
    -> build_sku_text per row) previously existed in FOUR copies
    (pipeline.build_training_data payload, predict_items, rand_matching, and
    the record-linkage lane) — a text-normalization fork in waiting: any
    fix here had to be replicated four times or the lanes silently diverged.
    One definition now; callers must not rebuild it inline. The phases run
    on :class:`SkuPayloadComposer`.

    ``structured_enabled`` is the caller's resolved structured_features.enabled
    (pipeline/rand_matching read the same config knob); when False the info
    side degrades to empty sets exactly as the payload builder always did.
    """
    return SkuPayloadComposer(frame, structured_enabled=structured_enabled).compose()
def build_canonical_text(
    record: Mapping[str, object],
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec | None = None,
) -> str:
    """Model text for one canonical (target) record."""
    resolved = _resolve(spec)
    if resolved.profile == "cleaned":
        return _cleaned_canonical_text(record, info, spec=resolved)
    return _legacy_canonical_text(record, info, evidence=resolved.include_evidence)
