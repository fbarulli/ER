"""Model-input composition contract: one builder, two config-selectable profiles.

``core.model_input`` is the single source of truth for the encoder text in both
lanes.  These tests pin the two things that make the config switch safe:

* the ``legacy`` profile still reproduces the committed byte stream exactly
  (the rollback contract, checked against fixtures captured from the
  unmodified code before any change), and
* the ``cleaned`` profile actually delivers what it claims — compound
  splitting, number preservation, boilerplate exclusion and source/target
  symmetry on real data, without collapsing different products together.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from core.model_input import (
    _normalized_tokens,
    build_canonical_text,
    build_sku_text,
    implicit_pack_qty,
    model_input_composition,
    model_input_info,
    model_input_spec,
    token_budget_report,
)
from core.schemas import TrainingSpec
from core.structured_features import canonical_info, sku_info

FIXTURE = Path(__file__).parent / "fixtures" / "model_input_golden.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECORDS = GOLDEN["records"]

LEGACY = TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=True)
LEGACY_NO_EVIDENCE = TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=False)
CLEANED = TrainingSpec.ModelInputSpec(profile="cleaned", include_evidence=False)


def _group(name: str) -> list[dict]:
    rows = [r for r in RECORDS if r["group"] == name]
    assert rows, f"fixture group {name!r} is empty"
    return rows


def _pair(record: dict, spec: TrainingSpec.ModelInputSpec) -> tuple[str, str]:
    row = pd.Series(record["sku"])
    return (
        build_sku_text(row, sku_info(row["title"], row["attributes"]), spec=spec),
        build_canonical_text(
            record["canonical_record"], canonical_info(record["canonical_record"]), spec=spec
        ),
    )


def _jaccard(left: str | set[str], right: str | set[str]) -> float:
    a = left.split() if isinstance(left, str) else left
    b = right.split() if isinstance(right, str) else right
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _cross_jaccard(sources: list[str], targets: list[str], per_row: int = 12) -> float:
    """Mean Jaccard of each source against OTHER rows' targets."""
    step = max(1, len(targets) // per_row)
    values = [
        _jaccard(sources[i], targets[(i + k + 1) % len(targets)])
        for i in range(len(sources))
        for k in range(0, len(targets), step)
    ]
    return sum(values) / len(values)


# ── the switch itself ──────────────────────────────────────────────────────


def test_shipped_config_defaults_to_the_cleaned_profile() -> None:
    """Pin the shipped default so it can never change silently.

    Every other test in this module selects its profile explicitly, so
    without this one a default change would be invisible to the suite.
    """
    spec = model_input_spec()
    assert spec.profile == "cleaned"
    assert spec.include_evidence is False


def test_default_selection_is_reachable_without_any_config_argument() -> None:
    """Omitting ``spec`` must use the config, not an implicit constant."""
    assert model_input_spec() == CLEANED


def test_cleaned_profile_refuses_to_also_request_the_evidence_channel() -> None:
    """One profile plus one granular flag must not admit a contradiction."""
    with pytest.raises(ValidationError, match="excludes the"):
        TrainingSpec.ModelInputSpec(profile="cleaned", include_evidence=True)


def test_legacy_profile_remains_selectable_as_the_fallback() -> None:
    """Legacy must stay expressible from config alone (no code revert)."""
    fallback = TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=True)
    assert fallback.profile == "legacy"
    assert fallback != model_input_spec()


def test_model_input_block_is_declared_in_the_config_contract() -> None:
    """No silent default: the block is part of the validated config shape."""
    assert "model_input" in TrainingSpec.model_fields


# ── legacy profile: the rollback contract ──────────────────────────────────


def test_legacy_profile_reproduces_golden_bytes() -> None:
    """Every fixture row must round-trip byte for byte on BOTH sides.

    The fixtures were captured from the committed composition before this
    module existed, so a mismatch here means the rollback path changed.
    """
    checked = 0
    for record in RECORDS:
        row = pd.Series(record["sku"])
        got_sku = build_sku_text(
            row, sku_info(row["title"], row["attributes"]), spec=LEGACY
        )
        got_canonical = build_canonical_text(
            record["canonical_record"],
            canonical_info(record["canonical_record"]),
            spec=LEGACY,
        )
        assert got_sku == record["legacy_sku_text"], f"sku text changed: {record['sku_id']}"
        assert (
            got_canonical == record["legacy_canonical_text"]
        ), f"canonical text changed: {record['nearest_item_id']}"
        checked += 1
    assert checked == len(RECORDS) > 800


def test_legacy_no_evidence_profile_removes_only_the_evidence_channel() -> None:
    """The granular flag is a real ablation, not a no-op."""
    record = next(
        r
        for r in _group("singleton_gtin")
        if set(r["legacy_canonical_text"].split()) - set(
            _pair(r, LEGACY_NO_EVIDENCE)[1].split()
        )
    )
    kept = set(_pair(record, LEGACY)[1].split())
    dropped = set(_pair(record, LEGACY_NO_EVIDENCE)[1].split())

    removed = kept - dropped
    assert removed, "the evidence channel must actually be removable"
    assert not dropped - kept, "disabling evidence must not ADD tokens"

    from pipeline import normalize_text

    normalized_evidence = normalize_text(" ".join(
        record["canonical_record"][field].strip("[]'\"")
        for field in ("description_evidence", "breadcrumb_evidence")
    ))
    for token in removed:
        assert token in normalized_evidence, f"{token!r} is not evidence text"


# ── cleaned profile: the claimed behaviour ─────────────────────────────────


def test_cleaned_profile_splits_underscore_compounds() -> None:
    """A canonical compound must become lexically matchable source words."""
    record = next(
        r for r in _group("review_band_585") if "_" in r["canonical_record"]["canonical"]
    )
    compound = next(
        t for t in record["canonical_record"]["canonical"].split() if "_" in t
    )
    _, legacy = _pair(record, LEGACY)
    _, cleaned = _pair(record, CLEANED)

    assert compound in legacy.split(), "precondition: legacy keeps the compound"
    assert compound not in cleaned.split()
    for part in compound.split("_"):
        assert part in cleaned.split()


def test_cleaned_profile_preserves_discriminative_numbers() -> None:
    """Numbers destroyed by the legacy canonical cleaner must survive."""
    record = next(
        r for r in _group("review_band_585") if "6000mg" in r["canonical_record"]["canonical"]
    )
    _, legacy = _pair(record, LEGACY)
    _, cleaned = _pair(record, CLEANED)

    assert "6000mg" not in legacy.split(), "precondition: legacy strips the number"
    assert "6000mg" in cleaned.split()


def test_cleaned_profile_preserves_percentage_evidence() -> None:
    """Juice-content percentages must survive instead of colliding with stops.

    Bare ``100`` and ``2`` are volume entries in ``MINIMAL_STOPWORDS``, so
    before this the whole ``Juice Content`` attribute was discarded.
    """
    assert _normalized_tokens("Juice Content: 100%", drop_schema_words=True) == ["pct100"]
    assert _normalized_tokens("Juice Content: 0-2%", drop_schema_words=True) == ["pct0to2"]
    assert "pct5.5" in _normalized_tokens("Alcohol: 5.5%", drop_schema_words=True)

    carried = sum(
        1
        for record in _group("review_band_585")
        if any(t.startswith("pct") for t in _pair(record, CLEANED)[0].split())
    )
    assert carried > 400, f"only {carried} rows carry percentage evidence"


def test_cleaned_profile_excludes_the_evidence_channel() -> None:
    """No description/breadcrumb evidence token may reach the cleaned text."""
    checked = 0
    for record in _group("singleton_gtin"):
        canonical_record = record["canonical_record"]
        sku_source, canonical_target = _pair(record, CLEANED)
        allowed_target = set(
            _normalized_tokens(canonical_record["canonical"], drop_schema_words=True)
            + _normalized_tokens(canonical_record["mode_brand"], drop_schema_words=False)
            + _normalized_tokens(canonical_record["mode_type"], drop_schema_words=True)
        )
        allowed_source = set(
            _normalized_tokens(record["sku"]["brand"], drop_schema_words=False)
            + _normalized_tokens(record["sku"]["title"], drop_schema_words=False)
            + _normalized_tokens(record["sku"]["attributes"], drop_schema_words=True)
        )
        evidence_only = {
            token
            for field in ("description_evidence", "breadcrumb_evidence")
            for token in _normalized_tokens(canonical_record[field], drop_schema_words=True)
        } - allowed_target
        assert not (evidence_only & set(canonical_target.split())), record["nearest_item_id"]

        raw_only = {
            token
            for field in ("description", "category", "category_path")
            for token in _normalized_tokens(record["sku"][field], drop_schema_words=True)
        } - allowed_source
        assert not (raw_only & set(sku_source.split())), record["sku_id"]
        checked += 1
    assert checked == len(_group("singleton_gtin"))


def test_cleaned_profile_emits_no_literal_block_markers() -> None:
    """Field blocks are ordered, not marked: markers measured as constant mass."""
    record = _group("review_band_585")[0]
    for text in _pair(record, CLEANED):
        assert "[BRAND]" not in text
        assert "[TITLE]" not in text
        assert "[ATTRIBUTES]" not in text


def test_cleaned_profile_uses_one_normalizer_for_both_lanes() -> None:
    """Identical input text must yield identical tokens whichever lane builds it."""
    text = "bcaa_6000mg Pear CAN 100%"
    sku_row = pd.Series(
        {"title": text, "attributes": "", "brand": "", "description": "",
         "category": "", "category_path": ""}
    )
    sku_tokens = build_sku_text(sku_row, {}, spec=CLEANED).split()
    canonical_tokens = build_canonical_text(
        {"canonical": text, "mode_brand": "", "mode_type": ""}, {}, spec=CLEANED
    ).split()

    assert sku_tokens == canonical_tokens
    assert "6000mg" in sku_tokens and "pct100" in sku_tokens
    assert "bcaa_6000mg" not in sku_tokens


def test_cleaned_profile_improves_the_true_vs_cross_margin() -> None:
    """On the review band, true pairs must be closer to their target than to others."""
    records = _group("review_band_585")
    margins = {}
    for name, spec in (("legacy", LEGACY), ("cleaned", CLEANED)):
        pairs = [_pair(r, spec) for r in records]
        sources = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]
        true = sum(_jaccard(s, t) for s, t in zip(sources, targets)) / len(records)
        margins[name] = true - _cross_jaccard(sources, targets)

    assert margins["cleaned"] > margins["legacy"], margins


def test_cleaned_profile_keeps_true_pairs_far_above_cross_pairs() -> None:
    """Symmetry must not be bought by making every product look alike.

    ``singleton_gtin`` rows are one-row GTINs: the canonical is a pure function
    of the source row, so a true pair is genuinely the same product while the
    cross pairs are genuinely different products.
    """
    records = _group("singleton_gtin")
    pairs = [_pair(r, CLEANED) for r in records]
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    true = sum(_jaccard(s, t) for s, t in zip(sources, targets)) / len(records)
    cross = _cross_jaccard(sources, targets)

    assert true > 0.5, true
    assert cross < 0.2, cross
    assert true - cross > 0.35, (true, cross)


def test_different_brands_do_not_collapse_to_one_string() -> None:
    """Different products must stay distinguishable in the cleaned profile."""
    records = _group("singleton_gtin")
    left, right = records[0], records[1]
    assert left["sku"]["brand"] != right["sku"]["brand"]

    left_text = _pair(left, CLEANED)[0]
    right_text = _pair(right, CLEANED)[0]
    assert left_text != right_text
    left_brand = _normalized_tokens(left["sku"]["brand"], drop_schema_words=False)
    right_brand = _normalized_tokens(right["sku"]["brand"], drop_schema_words=False)
    assert left_brand and right_brand
    assert set(left_brand) <= set(left_text.split())
    assert set(right_brand) <= set(right_text.split())
    assert _jaccard(left_text, right_text) < 1.0


def test_structured_token_channel_is_identical_across_profiles() -> None:
    """The structured channel is profile-independent EXCEPT for the pack default.

    Every field both extractors already treat alike (volume, package type,
    flavour, carbonation, sweetener, pulp) must reach the encoder unchanged by
    the profile switch.  ``pack`` is the one deliberate exception: ``cleaned``
    applies the implicit-default rule to an UNOBSERVED pack on BOTH sides, so
    the source and the target of one product stop disagreeing for a reason that
    has nothing to do with the product (measured on the review band: 425 of 585
    rows carried ``pack_qty_1`` on the source side and nothing on the target
    side under ``legacy``).  ``legacy`` keeps its historical one-sided sentinel
    because the golden fixtures pin its bytes.
    """
    record = _group("review_band_585")[0]
    info = canonical_info(record["canonical_record"])
    legacy = build_canonical_text(record["canonical_record"], info, spec=LEGACY)
    cleaned = build_canonical_text(record["canonical_record"], info, spec=CLEANED)

    def structured_tail(text: str) -> list[str]:
        return [t for t in text.split() if t.startswith("[FIELD_")]

    def without_pack(tail: list[str]) -> list[str]:
        return [t for t in tail if t != "[FIELD_PACK_SIZE]"]

    # Profile-independent: every shared field group keeps its marker and order.
    assert without_pack(structured_tail(legacy)) == without_pack(structured_tail(cleaned))
    # Profile-dependent by design: only the pack group may differ.
    assert set(structured_tail(legacy)) ^ set(structured_tail(cleaned)) <= {"[FIELD_PACK_SIZE]"}

    # The asymmetry the cleaned profile closes, on a row whose canonical pack
    # set is empty, is visible on the TEXT: legacy emits no pack token on the
    # target side, cleaned emits the configured implicit default.
    assert info["pack"] == set(), "fixture row is expected to have an unobserved pack"
    assert "[FIELD_PACK_SIZE] pack_qty_1" not in legacy
    assert f"[FIELD_PACK_SIZE] pack_qty_{implicit_pack_qty():g}" in cleaned


def test_cleaned_profile_makes_an_unobserved_pack_symmetric_on_both_sides() -> None:
    """The implicit pack default must reach text AND numeric vector alike.

    Both lanes normalize the info ONCE through ``model_input_info`` and feed
    the result to the text builder and the structured vector, so an unobserved
    pack cannot be represented one way in the text and another in the vector.
    """
    from core.model_input import model_input_info
    from core.structured_features import vector

    record = _group("review_band_585")[0]
    info = canonical_info(record["canonical_record"])
    assert info["pack"] == set()

    normalized = model_input_info(info, spec=CLEANED)
    assert normalized["pack"] == {implicit_pack_qty()}
    # Idempotent: applying it twice must not change the representation again.
    assert model_input_info(normalized, spec=CLEANED)["pack"] == {implicit_pack_qty()}
    # legacy is a no-op, which is what keeps the golden bytes valid.
    assert model_input_info(info, spec=LEGACY)["pack"] == set()

    empty_block = vector(
        info, volume_scale_ml=10000.0, pack_scale=100.0, max_set_size=8
    )[5:10]
    filled_block = vector(
        normalized, volume_scale_ml=10000.0, pack_scale=100.0, max_set_size=8
    )[5:10]
    assert empty_block[0] == 0.0, "an unobserved pack has no presence bit"
    assert filled_block[0] == 1.0, "the implicit default must register as present"


def test_title_only_payload_variant_still_blanks_attributes_and_description() -> None:
    """The payload ablation variant must keep its documented field selection.

    ``title_only`` was the payload stage's own composition before the
    consolidation; blanking the attribute/description columns must reproduce it.
    """
    from core.structured_features import append_text
    from pipeline import clean_sku_text, strip_schema_words

    record = _group("singleton_gtin")[0]
    row = pd.Series(record["sku"])
    blanked = row.copy()
    for column in ("attributes", "attr", "description", "description_short_eng"):
        if column in blanked.index:
            blanked[column] = ""
    info = sku_info(row["title"], row["attributes"])

    expected = append_text(
        strip_schema_words(clean_sku_text(
            row["title"], "", row["brand"], "", row["category"], row["category_path"]
        )),
        info,
    )
    assert build_sku_text(blanked, info, spec=LEGACY) == expected


def test_both_lanes_call_the_shared_builder() -> None:
    """Guard against the composition being re-duplicated at a call site."""
    import inspect

    import predict_items
    from training import rand_matching

    for module, expected in (
        (predict_items, 2),
        (rand_matching, 2),
    ):
        source = inspect.getsource(module)
        calls = source.count("build_sku_text(") + source.count("build_canonical_text(")
        assert calls == expected, f"{module.__name__}: {calls} shared-builder calls"


# ── blast radius: the composition is provenance, not just behaviour ────────


def test_composition_provenance_tracks_the_config() -> None:
    """The provenance an artifact records must be the ACTIVE selection."""
    composition = model_input_composition()
    assert (composition.profile, composition.include_evidence) == ("cleaned", False)
    assert composition == TrainingSpec.ModelInputComposition.from_spec(
        model_input_spec()
    )
    # The digest separates the two selectable compositions, so an artifact can
    # name its contract without carrying the text.
    other = TrainingSpec.ModelInputComposition.from_spec(
        TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=True)
    )
    assert other.fingerprint != composition.fingerprint
    assert len(composition.fingerprint) == 64


def test_ann_fingerprint_inputs_include_the_composition() -> None:
    """A persisted ANN index must not survive a composition change.

    Chain: the composition is part of the fingerprint inputs (here), and a
    fingerprint mismatch makes ``PersistentHnswIndex.load`` raise
    (tests/test_hnsw_index.py). Without the first link a profile switch would
    silently reuse an index whose embeddings came from the other text.
    """
    from training.rand_matching import preprocessing_fingerprint_inputs

    inputs = preprocessing_fingerprint_inputs({"enabled": True})
    assert inputs["model_input"] == model_input_composition().model_dump()
    assert set(inputs) >= {
        "structured_features",
        "model_input",
        "unit_canonicalization",
    }

    # Same config twice -> same fingerprint; a different composition -> different one.
    import hashlib
    import json

    def digest(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    assert digest(inputs) == digest(preprocessing_fingerprint_inputs({"enabled": True}))
    other = dict(
        inputs,
        model_input=TrainingSpec.ModelInputComposition.from_spec(
            TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=True)
        ).model_dump(),
    )
    assert digest(other) != digest(inputs)


def test_ann_fingerprint_inputs_cover_the_normalisation_vocabulary() -> None:
    """A vocabulary edit changes the encoder text, so it must move the fingerprint.

    ``MINIMAL_STOPWORDS`` and the schema-word strip both come from
    ``config/vocabulary.json`` and are applied inside the composition, so
    editing that file changes the text while leaving every other fingerprint
    input identical. Without the vocabulary in the reuse contract a persisted
    ANN index built before the edit would be silently reused.
    """
    from core.common import VOCABULARY_CONFIG_PATH
    from core.manifest import sha256_file
    from training.rand_matching import preprocessing_fingerprint_inputs

    inputs = preprocessing_fingerprint_inputs({"enabled": True})
    assert inputs["vocabulary"] == sha256_file(VOCABULARY_CONFIG_PATH)
    assert len(inputs["vocabulary"]) == 64

    # The vocabulary really is an input to the composed text: a stopword that
    # appears in the corpus changes what the builder emits.
    import pipeline

    record = _group("review_band_585")[0]
    info = canonical_info(record["canonical_record"])
    before = build_canonical_text(record["canonical_record"], info, spec=CLEANED)
    original = list(pipeline.MINIMAL_STOPWORDS)
    try:
        pipeline.MINIMAL_STOPWORDS = original + ["carbonated"]
        after = build_canonical_text(record["canonical_record"], info, spec=CLEANED)
    finally:
        pipeline.MINIMAL_STOPWORDS = original
    assert after != before, "the fixture row must carry a token the vocabulary controls"


def test_ann_fingerprint_inputs_cover_the_composition_code() -> None:
    """Editing composition CODE must invalidate the index, not just config/data.

    ``pipeline.SCHEMA_WORDS`` / ``_MODEL_STOP`` and
    ``core.model_input._normalized_tokens`` produce the encoder text but are
    neither configuration nor data, so without their digests a code edit would
    silently keep a persisted index valid — the same silent-reuse seam, one
    level up from the vocabulary.

    The granularity is deliberately COARSE (the whole module, not the single
    symbol): an unrelated edit inside either file costs one visible rebuild,
    whereas under-invalidation serves a stale index that nobody sees.
    """
    import shutil
    import tempfile

    import pipeline
    import core.model_input as model_input_module
    from core.manifest import sha256_file
    from training.rand_matching import preprocessing_fingerprint_inputs

    inputs = preprocessing_fingerprint_inputs({"enabled": True})
    code = inputs["composition_code"]
    assert set(code) == {"core.model_input", "pipeline"}
    assert code["pipeline"] == sha256_file(pipeline.__file__)
    assert code["core.model_input"] == sha256_file(model_input_module.__file__)
    assert len(code["pipeline"]) == 64
    # Stable across calls: the same tree must not churn the index.
    assert preprocessing_fingerprint_inputs({"enabled": True})["composition_code"] == code

    # Content-sensitive: a file edit changes the digest that gates reuse.
    with tempfile.TemporaryDirectory(prefix="fpcode-") as tmp:
        copy = Path(tmp) / "pipeline_copy.py"
        shutil.copy2(pipeline.__file__, copy)
        copy.write_text(copy.read_text(encoding="utf-8") + "\n# simulated edit\n", encoding="utf-8")
        assert sha256_file(copy) != sha256_file(pipeline.__file__)

    # ...and the symbol really does drive the text the index was built from.
    # strip_schema_words reads _MODEL_STOP, so patching it is exactly what a
    # SCHEMA_WORDS edit in the file amounts to.
    probe = "cola type carbonization auditmarker"
    before = _normalized_tokens(probe, drop_schema_words=True)
    saved = pipeline._MODEL_STOP
    try:
        pipeline._MODEL_STOP = saved | {"auditmarker"}
        after = _normalized_tokens(probe, drop_schema_words=True)
    finally:
        pipeline._MODEL_STOP = saved
    assert after != before, "the schema-stop symbol must affect the composed text"


def test_run_trace_records_the_active_composition() -> None:
    """The payload stage must stamp the composition onto the run trace.

    Static guard: the payload stage writes a run-scope trace row naming the
    composition before it builds any text, so a downstream artifact can be
    traced back to the input contract that produced it.
    """
    import inspect

    import pipeline

    source = inspect.getsource(pipeline)
    assert '"model_input_composition"' in source
    assert "detail=model_input_composition().model_dump()" in source

    from core.tracing import TraceRun

    trace = TraceRun("pairs", run_id="unit")
    row = trace.add(
        "payload", "model_input_composition", detail=model_input_composition().model_dump()
    )
    assert row["step"] == "payload.model_input_composition"
    assert row["scope"] == "run"
    assert "cleaned" in str(row["detail"])


def test_checkpoint_manifest_records_the_active_composition(tmp_path: Path) -> None:
    """A checkpoint must name the input contract its weights were trained on."""
    import json
    from types import SimpleNamespace

    from training.training import _write_checkpoint_manifest

    class _AutoModel:
        config = SimpleNamespace(pad_token_id=0, bos_token_id=None, eos_token_id=None)
        generation_config = None

    class _Model(list):
        tokenizer = None

    model = _Model([SimpleNamespace(auto_model=_AutoModel())])
    _write_checkpoint_manifest(
        tmp_path,
        epoch=1,
        global_step=1,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        trainer_state=SimpleNamespace(log_history=[]),
        trainer_control=None,
        training_args=None,
    )
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert manifest["model_input"] == model_input_composition().model_dump()
    assert manifest["format"] == "euromonitor-hf-resume-v1"


# ── universal symmetry: an unobserved attribute is treated the same on both sides


def test_an_explicitly_observed_pack_is_never_overwritten() -> None:
    """Implicit 1.0 fills a GAP; it does not replace real evidence."""
    info = canonical_info({"pack_set": "[6]"})
    assert model_input_info(info, spec=CLEANED).get("pack") == {6.0}
    assert model_input_info(info, spec=LEGACY).get("pack") == {6.0}


def test_no_other_attribute_carries_a_one_sided_implicit_default() -> None:
    """The universal audit: pack was the ONLY one-sided default.

    An empty record must yield empty sets for every other attribute on BOTH
    sides, so the only thing the symmetry rule adds is the implicit pack.
    """
    empty_source = sku_info("", "")
    empty_target = canonical_info({})
    others = ("volume", "package_type", "flavor", "carbonation", "sweetener", "pulp")
    for attribute in others:
        assert not empty_source.get(attribute), attribute
        assert not empty_target.get(attribute), attribute
    # pack is the exception, and the rule closes it on BOTH sides
    assert empty_source.get("pack") == {1.0}
    assert not empty_target.get("pack")
    assert model_input_info(empty_target, spec=CLEANED).get("pack") == {1.0}
    assert model_input_info(empty_source, spec=CLEANED).get("pack") == {1.0}


def test_the_symmetry_rule_is_profile_scoped_and_idempotent() -> None:
    """Legacy keeps its historical one-sided treatment; cleaned is idempotent."""
    assert model_input_info({}, spec=LEGACY).get("pack") in (None, set())
    once = model_input_info({}, spec=CLEANED)
    assert model_input_info(once, spec=CLEANED) == once


def test_cleaned_profile_pack_token_presence_agrees_on_every_row() -> None:
    """Presence agreement: 27.4% before, and it must now be total."""
    agree = 0
    for record in RECORDS:
        row = pd.Series(record["sku"])
        source = build_sku_text(
            row, model_input_info(sku_info(row["title"], row["attributes"]), spec=CLEANED),
            spec=CLEANED,
        )
        target = build_canonical_text(
            record["canonical_record"],
            model_input_info(canonical_info(record["canonical_record"]), spec=CLEANED),
            spec=CLEANED,
        )
        agree += ("[FIELD_PACK_SIZE]" in source) == ("[FIELD_PACK_SIZE]" in target)
    assert agree == len(RECORDS)


def test_the_text_and_the_numeric_vector_cannot_disagree() -> None:
    """Both channels read model_input_info, so a divergence is impossible."""
    from core.structured_features import vector

    record = next(r for r in RECORDS if not r["canonical_record"]["pack_set"].strip("[]"))
    row = pd.Series(record["sku"])
    info = model_input_info(sku_info(row["title"], row["attributes"]), spec=CLEANED)
    text = build_sku_text(row, info, spec=CLEANED)
    numbers = vector(info, volume_scale_ml=10000.0, pack_scale=100.0, max_set_size=8)
    assert "[FIELD_PACK_SIZE]" in text
    assert numbers[5] == 1.0, "pack presence block must agree with the text channel"


# ── accent folding: spelling variants of one brand must collapse ───────────


# ── the symmetry invariant, as a corpus-wide property ─────────────────────


def test_symmetry_invariant_where_the_same_evidence_feeds_both_sides() -> None:
    """Same underlying evidence in, byte-identical text and vector out.

    SCOPE — the trap this avoids. A raw SKU listing and a canonical record for
    the same product legitimately differ in WORDING (the listing carries the
    raw attribute blob, the record a compressed extraction), so a blanket
    "identical products imply identical strings" rule would fail on correct
    input and be ignored. The invariant is therefore applied where the SAME
    data demonstrably feeds both sides: the canonical record's own fields,
    pushed through the source builder.

    LEGITIMATE EXCEPTION, encoded explicitly rather than by loosening the
    assertion: the target builder drops canonical tokens that merely repeat
    the brand it already emitted in the brand block. The mirror below removes
    exactly those tokens and nothing else, so the brand de-duplication is the
    only permitted difference.
    """
    from core.structured_features import vector

    mismatched_text: list[str] = []
    mismatched_vector: list[str] = []
    for record in RECORDS:
        canonical_record = record["canonical_record"]
        info = model_input_info(
            canonical_info(canonical_record), spec=CLEANED
        )
        brand_tokens = {
            token.casefold()
            for token in _normalized_tokens(
                canonical_record["mode_brand"], drop_schema_words=False
            )
        }
        attributes = " ".join([
            *(
                token
                for token in _normalized_tokens(
                    canonical_record["canonical"], drop_schema_words=True
                )
                if token.casefold() not in brand_tokens
            ),
            *_normalized_tokens(
                canonical_record["mode_type"], drop_schema_words=True
            ),
        ])
        mirrored_row = pd.Series({
            "title": "",
            "attributes": attributes,
            "brand": canonical_record["mode_brand"],
            "description": "",
            "category": "",
            "category_path": "",
        })
        source = build_sku_text(mirrored_row, info, spec=CLEANED)
        target = build_canonical_text(canonical_record, info, spec=CLEANED)
        if source != target:
            mismatched_text.append(record["nearest_item_id"])
        # both lanes read the SAME info object, so the numeric channel cannot
        # diverge either
        if vector(info, volume_scale_ml=10000.0, pack_scale=100.0, max_set_size=8) != vector(
            info, volume_scale_ml=10000.0, pack_scale=100.0, max_set_size=8
        ):
            mismatched_vector.append(record["nearest_item_id"])

    assert not mismatched_text, f"{len(mismatched_text)} rows disagreed: {mismatched_text[:5]}"
    assert not mismatched_vector
    assert len(RECORDS) > 800


# ── token budget: a dropped field group must be a named number ─────────────


class _CharTokenizer:
    """Deterministic character tokenizer: one token per character.

    Hermetic on purpose — the guard is about COUNTING, so the test must not
    depend on a model artifact being present.
    """

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=True):
        assert return_offsets_mapping
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_token_budget_report_is_clean_when_everything_fits() -> None:
    report = token_budget_report(
        ["water [FIELD_VOLUME] volume_ml_500"],
        tokenizer=_CharTokenizer(),
        max_seq_length=1000,
    )
    assert report.n_records == 1
    assert report.n_over_budget == 0
    assert report.n_field_groups_dropped == 0
    assert report.dropped_groups == {}


def test_token_budget_report_names_every_dropped_field_group() -> None:
    """The silent-drop defect: the tail is appended last and cut first."""
    text = "x" * 40 + " [FIELD_VOLUME] volume_ml_500 [FIELD_CARBONATION] carbonation_still"
    report = token_budget_report([text], tokenizer=_CharTokenizer(), max_seq_length=20)
    assert report.n_over_budget == 1
    assert set(report.dropped_groups) == {"VOLUME", "CARBONATION"}
    assert report.n_field_groups_dropped == 2
    assert report.max_seq_length == 20


def test_token_budget_report_keeps_groups_that_survive_the_window() -> None:
    text = "water [FIELD_VOLUME] volume_ml_500" + " y" * 200
    report = token_budget_report([text], tokenizer=_CharTokenizer(), max_seq_length=40)
    assert report.n_over_budget == 1
    # VOLUME sits inside the window, so it is NOT reported as dropped
    assert report.dropped_groups == {}


def test_token_budget_report_rejects_a_nonsense_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        token_budget_report(["x"], tokenizer=_CharTokenizer(), max_seq_length=0)


def test_token_budget_report_counts_match_their_parts() -> None:
    """The schema enforces the identity, so a drifting counter cannot ship."""
    from core.schemas import TokenBudgetReport

    with pytest.raises(ValidationError, match="must equal the sum"):
        TokenBudgetReport(
            max_seq_length=128, n_records=1, n_over_budget=1,
            n_field_groups_dropped=5, dropped_groups={"VOLUME": 1},
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        TokenBudgetReport(
            max_seq_length=128, n_records=1, n_over_budget=9,
            n_field_groups_dropped=0, dropped_groups={},
        )
