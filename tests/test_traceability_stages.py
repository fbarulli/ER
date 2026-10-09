"""tests/test_traceability_stages.py — the STAGE/BATCH/ENTITY trace rows of the
data-processing stages this directive instrumented.

One decisive test per instrumented stage, at the public entry point (or the
stage's own recorder where the entry point needs the live export and the frozen
results tree). Every test:

  * redirects the trace to a tmp file (``core.tracing.trace_path``) and pins a run
    id (``EUROMONITOR_TRACE_RUN``), so nothing touches the real results tree;
  * runs the stage;
  * reads the file back with ``core.tracing.read_trace`` and validates EVERY row
    against ``core.schemas.TraceRow`` (the frame boundary contract);
  * asserts the accounting that stage owes: the exact reason census, the named
    ENTITY behind a dropped/quarantined row, and the BATCH rows' caps.

Covered: build_reference, build_second04_pairs, labeled_pairs,
build_final_validation, graph_tracks/setup (pairing + the generated cascade
config), graph_tracks/prepare (incl. the quarantine exception),
graph_tracks/data (opt-in), data_prep/pipeline (single stage writer),
dedupe (prepare stage 1), negative_supply (the lane's block/mine/mint funnels).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import core.tracing as tracing
from core.pair_identity import PairIdentity
from core.schemas import TraceRow
from core.gtin import is_valid_gtin_checksum

VALID_GTIN = "4006381333931"  # GS1 checksum valid


def _with_check_digit(body: str) -> str:
    """GS1 check digit for a body (the same rule core.gtin validates)."""
    reversed_body = body[::-1]
    total = (
        sum(int(digit) for digit in reversed_body[0::2]) * 3
        + sum(int(digit) for digit in reversed_body[1::2])
    )
    return str((10 - total % 10) % 10)


def valid_gtins(count: int, base: str = VALID_GTIN) -> list[str]:
    """``count`` distinct checksum-valid gtins, computed (never hard-coded)."""
    body = base[:-1]
    out: list[str] = []
    for index in range(count):
        mutated = body[:-2] + f"{index:02d}"
        gtin = mutated + _with_check_digit(mutated)
        assert is_valid_gtin_checksum(gtin), gtin
        out.append(gtin)
    return out


# ── hermetic trace harness ─────────────────────────────────────────────────
@pytest.fixture()
def trace_target(tmp_path, monkeypatch) -> Path:
    """The consolidated trace redirected into a tmp file, with a pinned run id."""
    target = tmp_path / "training_trace.csv"
    monkeypatch.setattr(tracing, "trace_path", lambda: target)
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-test-traceability")
    return target


def read_validated(target: Path) -> pd.DataFrame:
    """Read the trace back and prove every row satisfies the row contract."""
    frame = tracing.read_trace(target)
    assert list(frame.columns) == list(tracing.TRACE_COLUMNS)
    tracing.assert_trace_frame(frame)
    for row in frame.to_dict("records"):
        TraceRow.model_validate(row)
    return frame


def rows_of(frame: pd.DataFrame, step: str) -> pd.DataFrame:
    return frame[frame["step"].astype(str) == step]


def one(frame: pd.DataFrame, step: str) -> pd.Series:
    hit = rows_of(frame, step)
    assert len(hit) == 1, f"expected exactly one {step!r} row, got {len(hit)}"
    return hit.iloc[0]


def detail_of(row: pd.Series) -> dict:
    return tracing.detail_json(row["detail"])


def census(frame: pd.DataFrame, step: str) -> dict[str, int]:
    """The exact per-reason census of an add_entities step."""
    hit = rows_of(frame, f"{step}.reason_census")
    return {str(r["reason"]): int(r["in_count"]) for _, r in hit.iterrows()}


def entity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["scope"] == "entity"]


# ══════════════════════════════════════════════════════════════════════════
# build_reference
# ══════════════════════════════════════════════════════════════════════════
def reference_frame(verdicts: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "token": token,
                "class": "numeric_brand",
                "n_occurrences": 3,
                "verdict": verdict,
                "rule": verdict,
            }
            for token, verdict in verdicts.items()
        ]
    )


def test_build_reference_passes_are_distinct_stages_and_neither_is_lost(trace_target):
    """The rebuild and the verify are TWO prepare_all stages, so they must NOT
    share a trace stage: ``_commit`` replaces a stage's rows for the run, and a
    shared name let the verify pass DESTROY the rebuild pass's rows (reproduced:
    corpus.texts_censused vanished, only verify.* left). This test runs both
    passes under one run id and asserts both survive.
    """
    from training import build_reference as ref

    ref_frame = reference_frame(
        {"000": "strip", "1": "strip", "250": "keep_name_embedded"}
    )
    # ── pass 1: number_reference (the rebuild + publication) ──
    rewrite = tracing.TraceRun(ref.STAGE_WRITE)
    rewrite.add(
        "corpus",
        "texts_censused",
        in_count=2,
        out_count=2,
        reason="the shape the rebuild pass emits before its verdict census",
        source="dataset_deduped (core.common.load_dataset_deduped)",
    )
    ref._record_reference(rewrite, ref_frame, ["text one", "text two"])
    rewrite.write(trace_target)

    # the funnel: occurrences -> DISTINCT verdict rows (the collapse is the drop)
    frame = read_validated(trace_target)
    assert set(frame["stage"]) == {ref.STAGE_WRITE}
    recorded = one(frame, "reference.verdicts_recorded")
    assert int(recorded["in_count"]) == 9  # three tokens x three occurrences
    assert int(recorded["out_count"]) == 3
    assert int(recorded["dropped_count"]) == 6

    # the census is EXACT and carries the caps it spent
    assert census(frame, "verdict") == {
        "strip": 2,
        "keep_name_embedded": 1,
    }
    budget = one(frame, "verdict.sample_budget")
    budget_detail = detail_of(budget)
    assert budget_detail["per_reason"] == tracing.ENTITY_SAMPLE_PER_REASON
    assert budget_detail["total_cap"] == tracing.ENTITY_ROW_CAP
    # every bucket fits its cap, so nothing was omitted on this population
    assert budget_detail["omitted"] == 0
    assert set(entity_rows(frame)["key"]) == {"000", "1", "250"}

    # ── pass 2: verify_reference (its own stage, same run id) ──
    committed = reference_frame(
        {"000": "strip", "1": "keep_brand", "250": "keep_name_embedded"}
    )
    rebuilt = reference_frame(
        {"000": "strip", "1": "strip", "250": "keep_name_embedded"}
    )
    with pytest.raises(SystemExit):
        ref.ReferenceVerifier.verify(committed, rebuilt)

    frame = read_validated(trace_target)
    assert set(frame["stage"]) == {ref.STAGE_WRITE, ref.STAGE_VERIFY}
    assert frame["run_id"].nunique() == 1  # ONE run, TWO stages
    # THE DEFECT: the verify commit must not have replaced the rebuild's rows
    assert rows_of(frame, "corpus.texts_censused").shape[0] == 1
    assert rows_of(frame, "reference.verdicts_recorded").shape[0] == 1
    assert census(frame, "verdict")["strip"] == 2

    compared = one(frame, "verify.verdicts_compared")
    assert int(compared["in_count"]) == 3
    assert int(compared["out_count"]) == 2
    assert int(compared["dropped_count"]) == 1
    assert census(frame, "verify.drift") == {
        "committed 'keep_brand' -> rebuilt 'strip'": 1
    }
    drifted = entity_rows(frame)
    drifted = drifted[drifted["step"] == "verify.drift"]
    assert list(drifted["key"]) == ["1"]
    evidence = detail_of(drifted.iloc[0])
    assert evidence["committed_verdict"] == "keep_brand"
    assert evidence["rebuilt_verdict"] == "strip"


# ══════════════════════════════════════════════════════════════════════════
# build_second04_pairs
# ══════════════════════════════════════════════════════════════════════════
def test_build_second04_pairs_traces_exclusions_batches_and_named_rows(trace_target):
    """Every source row's exclusion is censused AND named; the pairing pass
    reports its batch boundaries and the pairs they produced."""
    from training import build_second04_pairs as second04

    frame = pd.DataFrame(
        {
            "sku_id": ["a", "b", "c", "d", "e", "", "g"],
            "gtin": [VALID_GTIN, VALID_GTIN, "", VALID_GTIN, VALID_GTIN, VALID_GTIN, "4000"],
            "country": ["DE", "FR", "DE", "", "DE", "FR", "FR"],
        }
    )
    manifest = second04.build_manifest(frame)
    assert len(manifest) == 2  # (a, b) and (e, b): a and e share DE

    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"build_second04_pairs"}

    classified = one(trace, "source.rows_classified")
    assert int(classified["in_count"]) == 7
    assert int(classified["out_count"]) == 3
    assert int(classified["dropped_count"]) == 4
    assert census(trace, "source.exclusion") == {
        "accepted": 3,
        "missing_sku_id": 1,
        "missing_gtin": 1,
        "missing_country": 1,
        "invalid_gtin": 1,
    }
    # the census closes over the whole population
    assert sum(census(trace, "source.exclusion").values()) == len(frame)

    # each dropped row is NAMED with its exact reason
    named = entity_rows(trace)
    by_key = dict(zip(named["key"], named["reason"]))
    assert by_key[""] == "missing_sku_id"
    assert by_key["c"] == "missing_gtin"
    assert by_key["d"] == "missing_country"
    assert by_key["g"] == "invalid_gtin"

    built = detail_of(one(trace, "pairs.cross_country_built"))
    assert built["accepted_rows"] == 3
    assert built["pairs"] == 2
    assert built["gtins"] == 1

    # BATCH grain: the batches are censused and their caps are stated
    second04_batch_census = detail_of(one(trace, "pairs.batch_census"))
    assert second04_batch_census["gtin_groups"] == 1
    assert second04_batch_census["batches"] == 1
    assert second04_batch_census["batches_traced"] == 1
    assert second04_batch_census["batches_omitted"] == 0
    assert second04_batch_census["pairs"] == 2
    assert second04._BATCH_GTINS == second04_batch_census["batch_gtins"]
    batch = detail_of(one(trace, "pairs.batch_0000"))
    assert batch["gtin_groups"] == 1
    assert batch["pairs"] == 2


# ══════════════════════════════════════════════════════════════════════════
# labeled_pairs
# ══════════════════════════════════════════════════════════════════════════
def test_labeled_pairs_partition_census_joins_the_manifest_buckets(trace_target):
    """The trace's partition census IS the manifest's four-bucket partition, per
    row, with the dropped pairs named by bucket."""
    from training import labeled_pairs as lp

    gates = pd.DataFrame(
        {
            "gtin1": [VALID_GTIN, VALID_GTIN, VALID_GTIN, VALID_GTIN, VALID_GTIN],
            "gtin2": ["1", "2", "3", "4", "5"],
            "gate_decision": ["proceed", "hard_no", "proceed", "fallback", "mystery"],
            "similarity": [0.9, 0.9, 0.5, 0.9, 0.5],
            "gate_reason": ["ok", "ok", "weak", "uncertain", "unknown"],
        }
    )
    pos_sim = neg_sim = 0.8
    pos, neg, is_fallback, is_kept = lp.GateSplit.partition_masks(gates, pos_sim, neg_sim)
    out = lp.GateSplit.labeled_frame(gates, pos, neg)
    accounting = lp.SplitLedger.row_accounting(
        gates, out, is_fallback, is_kept, pos_sim, neg_sim
    )

    run = tracing.TraceRun("labeled_pairs")
    lp._record_partition(
        run, gates, out, pos, neg, is_fallback, is_kept,
        accounting, pos_sim, neg_sim, "gate_results.csv",
    )
    run.write(trace_target)

    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"labeled_pairs"}

    split = one(trace, "gate_rows.split")
    assert int(split["in_count"]) == 5
    assert int(split["out_count"]) == 2
    assert int(split["dropped_count"]) == 3

    measured = census(trace, "partition")
    assert measured == {
        lp.BUCKET_POS: 1,
        lp.BUCKET_NEG: 1,
        lp.BUCKET_FALLBACK: 1,
        lp.BUCKET_BELOW: 1,
        lp.BUCKET_OTHER: 1,
    }
    assert sum(measured.values()) == len(gates)
    # the JOIN the contract promises: the census's dropped buckets are the
    # manifest's own `dropped` keys, and the kept buckets are its labeled counts
    assert measured[lp.BUCKET_FALLBACK] == accounting["dropped"][lp.BUCKET_FALLBACK]
    assert measured[lp.BUCKET_BELOW] == accounting["dropped"][lp.BUCKET_BELOW]
    assert measured[lp.BUCKET_OTHER] == accounting["dropped"][lp.BUCKET_OTHER]
    assert measured[lp.BUCKET_POS] == accounting["pos_labeled"]
    assert measured[lp.BUCKET_NEG] == accounting["hard_neg_labeled"]
    # the labeled frame is untouched by the tracing
    assert len(out) == 2 and int((out.true_label == 1).sum()) == 1

    # the fallback pair is NAMED, with its decision and similarity
    named = entity_rows(trace)
    fallback = named[named["reason"] == lp.BUCKET_FALLBACK]
    assert list(fallback["key"]) == [PairIdentity.of(VALID_GTIN, "4")]
    evidence = detail_of(fallback.iloc[0])
    assert evidence["gate_decision"] == "fallback"
    assert evidence["similarity"] == 0.9


# ══════════════════════════════════════════════════════════════════════════
# build_final_validation
# ══════════════════════════════════════════════════════════════════════════
def test_final_validation_assembler_traces_every_pairs_outcome(trace_target):
    """Labeled pairs -> rows, with BOTH non-row outcomes named at entity grain."""
    from training.build_final_validation import (
        OUTCOME_BOTH_IN_TRAIN,
        OUTCOME_SCORED,
        OUTCOME_UNRESOLVED,
        FoldResolver,
        ValidationRowAssembler,
    )

    fold_of = {"g1": 0, "g5": 0, "g6": 0, "g3": 2, "g4": 2, "g2": 3}
    assembler = ValidationRowAssembler(
        FoldResolver(fold_of), comp_of={}, slice_values={}, n_folds=4
    )
    labeled = pd.DataFrame(
        {
            "gtin1": ["g1", "g2", "g1", "g5"],
            "gtin2": ["g3", "g4", "outside", "g6"],
            "true_label": [1, 0, 0, 0],
        }
    )
    run = tracing.TraceRun("final_validation")
    out, outcomes = assembler.assemble_all_with_trace(labeled, run)
    # the stage's own census recorder (build() calls it right after the assembly)
    from training.build_final_validation import _record_census

    _record_census(run, labeled, out, outcomes, assembler)
    run.write(trace_target)
    # the counters the recorder read, captured before the public-face probes below
    unresolved, both_in_train = assembler.unresolved, assembler.both_in_train

    # the public row-only face is unchanged
    assert assembler.assemble("g2", "g4", 0) is not None
    assert assembler.assemble("g5", "g6", 0) is None

    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"final_validation"}

    assembled = one(trace, "labeled_census.assembled")
    assert int(assembled["in_count"]) == 4
    assert int(assembled["out_count"]) == 2
    assert int(assembled["dropped_count"]) == 2
    assembled_detail = detail_of(assembled)
    assert unresolved == 1
    assert assembled_detail["unresolved_endpoints"] == unresolved
    assert both_in_train == 1
    assert assembled_detail["both_endpoints_in_train"] == both_in_train

    assert census(trace, "labeled_census") == {
        OUTCOME_SCORED: 2,
        OUTCOME_UNRESOLVED: 1,
        OUTCOME_BOTH_IN_TRAIN: 1,
    }
    assert sum(census(trace, "labeled_census").values()) == len(labeled)

    named = entity_rows(trace)
    by_key = dict(zip(named["key"], named["reason"]))
    assert by_key["g1|outside"] == OUTCOME_UNRESOLVED
    assert by_key["g5|g6"] == OUTCOME_BOTH_IN_TRAIN
    # the batch census closes over the whole walk
    batch = detail_of(one(trace, "labeled_census.batch_census"))
    assert batch["pairs"] == 4 and batch["rows"] == len(out) == 2
    assert batch["batches"] == 1 and batch["batches_omitted"] == 0
    # every labeled pair has exactly one outcome record
    assert len(outcomes) == len(labeled)


# ══════════════════════════════════════════════════════════════════════════
# graph_tracks/setup
# ══════════════════════════════════════════════════════════════════════════
def test_graph_setup_traces_retention_supervision_and_untrusted_listings(trace_target):
    """Catalog retention, label application and pair provenance are traced, and a
    listing dropped from its chain because its gtin cannot be trusted is NAMED."""
    from graph_tracks import setup as graph_setup

    catalog = pd.DataFrame(
        {
            "sku_id": ["a", "b", "c", "c2", "z"],
            "gtin": [VALID_GTIN, VALID_GTIN, "2", "2", "3"],
        }
    )
    labels = pd.DataFrame({"gtin1": [], "gtin2": [], "true_label": []})
    run = tracing.TraceRun("graph_setup")
    frame, assignments, pairs, accounting = graph_setup.listing_contract(
        catalog, labels, {"train": {VALID_GTIN, "2"}}, run
    )
    # the stage's own stage rows (setup() calls this right after the contract)
    graph_setup._record_supervision(run, catalog, frame, labels, pairs, accounting)
    run.write(trace_target)

    assert set(frame.sku_id) == {"a", "b", "c", "c2"}  # z is unassigned
    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"graph_setup"}

    retained = one(trace, "catalog.listings_retained")
    assert int(retained["in_count"]) == 5
    assert int(retained["out_count"]) == 4
    assert detail_of(retained)["excluded_unassigned_listings"] == 1

    # the chain pass: entity groups walked, pairs built, batches censused
    chains = detail_of(one(trace, "chains.batch_0000"))
    assert chains["entity_groups"] == 2  # two normalized gtins
    assert chains["pairs"] == 1  # only the trusted gtin forms a chain
    chain_census = detail_of(one(trace, "chains.batch_census"))
    assert chain_census["untrusted_chain_listings"] == 2
    assert chain_census["batches_omitted"] == 0

    # each untrusted listing is NAMED with the exact reason
    untrusted = entity_rows(trace)
    untrusted = untrusted[untrusted["step"] == "chains.untrusted_listing"]
    assert set(untrusted["key"]) == {"c", "c2"}
    assert all("GS1" in reason for reason in untrusted["reason"])

    # the supervision funnel closes and the provenance census is exact
    supervision = one(trace, "pairs.supervision_built")
    assert int(supervision["in_count"]) == int(supervision["out_count"]) + int(
        supervision["dropped_count"]
    )
    assert int(supervision["out_count"]) == len(pairs) == 1
    assert census(trace, "pairs") == {"trusted_same_entity_chain": 1}
    named_pairs = entity_rows(trace)
    named_pairs = named_pairs[named_pairs["step"] == "pairs"]
    assert list(named_pairs["key"]) == ["a|b"]
    assert detail_of(named_pairs.iloc[0])["split"] == "train"

    # the label side is traced even when it is empty
    applied = one(trace, "labels.source_rows_applied")
    assert int(applied["in_count"]) == 0 and int(applied["out_count"]) == 0


def test_generated_cascade_config_names_the_trained_tracks_artifacts(tmp_path):
    """The generated cascade.yaml points at the TRAINED tracks' artifact names.

    It used to point into the setup tree (``<setup>/text_index``), which is not
    where a trained artifact ever lands, so the cascade lane died with
    FileNotFoundError before ranking anything. The names come from the artefact
    SSOT (graph_tracks.artifacts.name) under the declared results root, and match
    the shipped template the worker is documented against.
    """
    import yaml

    from core.common import TRAIN_ROOT
    from graph_tracks.artifacts import name
    from graph_tracks.config import load_config as load_graph_config, load_text_config
    from graph_tracks.setup import (
        _load_setup_templates,
        _write_text_config,
        _write_track_configs,
    )

    templates = _load_setup_templates(Path(TRAIN_ROOT))
    listings = tmp_path / "prepared" / "listings.json"
    listings.parent.mkdir(parents=True)
    listings.write_text("{}")
    _write_track_configs(tmp_path, templates, listings, "baseline-sha")
    _write_text_config(tmp_path, templates)

    # the held-out test switch follows the SUITE config's own switch (the SSOT
    # preflight compares the prepared lanes against), never a literal here
    from core.common import artifact
    from model_tracks.config import load_config as load_suite_config
    from model_tracks.preflight import prepared_report_test

    suite = load_suite_config(Path(artifact("model_tracks_config")))
    switches = {track: bool(suite.report_test) for track in ("gnn_only", "cascade", "text")}
    assert prepared_report_test(tmp_path) == switches

    lane = load_graph_config(tmp_path / "cascade.yaml", expected_track="cascade")
    results_root = Path(lane.output_dir)
    assert Path(lane.text_index) == results_root / name("text", "index")
    assert Path(lane.gnn_checkpoint) == results_root / name(
        "gnn_only", "best_checkpoint.json"
    )
    # the bug being fixed: nothing points into the setup tree
    assert not Path(lane.text_index).is_absolute()
    assert str(tmp_path) not in str(lane.text_index)
    assert str(tmp_path) not in str(lane.gnn_checkpoint)

    # the shipped template carries the SAME config-owned names
    shipped = yaml.safe_load(
        (Path(TRAIN_ROOT) / "config" / "graph_tracks_cascade.yaml").read_text()
    )
    assert lane.text_index == shipped["text_index"]
    assert lane.gnn_checkpoint == shipped["gnn_checkpoint"]

    # gnn_only still forbids cascade artifact references
    gnn = load_graph_config(tmp_path / "gnn_only.yaml", expected_track="gnn_only")
    assert gnn.text_index is None and gnn.gnn_checkpoint is None
    assert gnn.report_test is switches["gnn_only"]
    assert load_text_config(tmp_path / "text.yaml").report_test is switches["text"]


# ══════════════════════════════════════════════════════════════════════════
# graph_tracks/prepare
# ══════════════════════════════════════════════════════════════════════════
def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    catalog, splits, pairs = [
        tmp_path / f"{stem}.csv" for stem in ("catalog", "splits", "pairs")
    ]
    pd.DataFrame(
        [
            {"sku_id": f"{split}-{index}", "sku_name_eng": "Lemon drink 330 ml",
             "gtin": ""}
            for split in ("train", "dev", "test")
            for index in range(3)
        ]
    ).to_csv(catalog, index=False)
    pd.DataFrame(
        [
            {"sku_id": f"{split}-{index}", "split": split}
            for split in ("train", "dev", "test")
            for index in range(3)
        ]
    ).to_csv(splits, index=False)
    pd.DataFrame(
        [
            {"sku_id1": f"{split}-0", "sku_id2": f"{split}-{index}",
             "label": int(index == 1), "split": split}
            for split in ("train", "dev", "test")
            for index in (1, 2)
        ]
    ).to_csv(pairs, index=False)
    return catalog, splits, pairs


def test_graph_prepare_traces_read_scrape_and_publication(trace_target, tmp_path):
    """The prepare stage records the input handoff, the 1:1 scrape with its
    per-split census and sample, and every published artifact."""
    from graph_tracks.prepare import prepare

    catalog, splits, pairs = _prepare_inputs(tmp_path)
    listing_path = prepare(catalog, splits, pairs, tmp_path / "prepared")

    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"graph_prepare"}

    inputs = one(trace, "inputs.read")
    assert int(inputs["in_count"]) == 9
    assert int(inputs["out_count"]) == 9

    scraped = one(trace, "listing_scrape.scraped")
    assert int(scraped["in_count"]) == 9
    assert int(scraped["out_count"]) == 9
    assert detail_of(scraped)["report_attribute_rows"] == 9
    # the exact per-split census, and the sampled listings behind each split
    assert census(trace, "listing_scrape") == {"train": 3, "dev": 3, "test": 3}
    assert set(entity_rows(trace)["reason"]) == {"train", "dev", "test"}

    assert one(trace, "manifest.published")["source"].endswith("input_manifest.json")
    published = one(trace, "output.published")
    assert int(published["out_count"]) == 9
    assert detail_of(published)["records"] == 9
    assert one(trace, "training_tensors.prepared")["step"] == "training_tensors.prepared"
    assert listing_path.exists()


def test_graph_prepare_names_the_quarantined_listings_before_it_refuses(
    trace_target, monkeypatch, tmp_path
):
    """The EXCEPTION path is traced: which listings are quarantined, and why."""
    from graph_tracks import prepare as graph_prepare

    catalog, splits, pairs = _prepare_inputs(tmp_path)
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False)
    mask = pd.Series(False, index=frame.index)
    mask.loc[0] = True  # train-0 carries a quarantined identity group

    monkeypatch.setattr(
        "core.identity_policy.reviewed_row_mask", lambda _frame: mask
    )
    with pytest.raises(ValueError, match="quarantined identity groups"):
        graph_prepare.prepare(catalog, splits, pairs, tmp_path / "prepared")
    # the stage refused before publishing anything
    assert not (tmp_path / "prepared").exists()

    trace = read_validated(trace_target)
    refusal = one(trace, "catalog.quarantined_identity")
    assert int(refusal["in_count"]) == 9
    assert int(refusal["dropped_count"]) == 1
    refusal_detail = detail_of(refusal)
    assert refusal_detail["quarantined_rows"] == 1
    assert refusal_detail["sample_sku_ids"] == ["train-0"]
    assert "REFUSES" in refusal["reason"]


# ══════════════════════════════════════════════════════════════════════════
# graph_tracks/data (opt-in)
# ══════════════════════════════════════════════════════════════════════════
def test_graph_data_records_the_listing_contract_on_request(trace_target, tmp_path):
    """load_records / fit_vocabulary / census trace only when given a writer, and
    then their rows close over the population they describe."""
    import json

    from graph_tracks.data import census, fit_vocabulary, load_records

    listings = tmp_path / "listings.json"
    records = [
        {"sku_id": f"{split}-{index}", "split": split, "attribute": {}, "numeric": {}}
        for split in ("train", "dev", "test")
        for index in range(2)
    ]
    listings.write_text(json.dumps({"schema": "er-graph-listings-v1", "listings": records}))

    # no writer, no file
    assert load_records(listings) == records
    assert not trace_target.exists()

    run = tracing.TraceRun("graph_data")
    loaded = load_records(listings, trace=run)
    vocabulary = fit_vocabulary(loaded, trace=run)
    census(loaded, vocabulary, trace=run)
    run.write(trace_target)

    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"graph_data"}
    validated = one(trace, "listings.validated")
    assert int(validated["in_count"]) == 6
    assert int(validated["out_count"]) == 6
    assert detail_of(validated)["splits"] == {"train": 2, "dev": 2, "test": 2}
    fitted = one(trace, "vocabulary.fitted")
    assert int(fitted["in_count"]) == 2  # train-split listings only
    assert int(fitted["out_count"]) == 0  # no attribute values in this fixture
    assert int(fitted["dropped_count"]) == 2  # a real funnel here: no values
    representation = one(trace, "census.representation")
    assert int(representation["in_count"]) == int(representation["out_count"]) == 6
    assert detail_of(representation)["splits"] == {"train": 2, "dev": 2, "test": 2}

    # ── the UNIT CHANGE: 3 train listings -> 4 distinct values ──
    # One listing carries several distinct values, so the vocabulary is larger
    # than the listing population. That is not a funnel, and an in/out pair here
    # made dropped_count NEGATIVE (live 3 -> 4), which the row contract forbids.
    unit_listings = tmp_path / "unit_listings.json"
    unit_records = [
        {"sku_id": f"train-{index}", "split": "train",
         "attribute": {"flavor": flavors}, "numeric": {}}
        for index, flavors in enumerate((["a"], ["b"], ["c", "d"]))
    ]
    unit_listings.write_text(
        json.dumps({"schema": "er-graph-listings-v1", "listings": unit_records})
    )
    unit_run = tracing.TraceRun("graph_data")
    unit_loaded = load_records(unit_listings, trace=unit_run)
    unit_vocabulary = fit_vocabulary(unit_loaded, trace=unit_run)
    unit_run.write(trace_target.parent / "unit_trace.csv")

    assert sum(len(values) for values in unit_vocabulary.values()) == 4
    unit_trace = read_validated(trace_target.parent / "unit_trace.csv")
    unit_fitted = one(unit_trace, "vocabulary.fitted")
    assert unit_fitted["in_count"] == ""  # no in/out pair: a unit change
    assert unit_fitted["dropped_count"] == ""
    assert int(unit_fitted["out_count"]) == 4
    unit_detail = detail_of(unit_fitted)
    assert unit_detail["unit_change"] is True
    assert unit_detail["train_listings"] == 3
    assert unit_detail["vocabulary_entries"] == 4


# ══════════════════════════════════════════════════════════════════════════
# dedupe (prepare stage 1, previously untraced)
# ══════════════════════════════════════════════════════════════════════════
def test_dedupe_traces_the_collapse_and_names_every_removed_sku(trace_target, tmp_path, monkeypatch):
    """The dedupe stage had NO trace rows, so "which rows did it remove, and why"
    needed the removals CSV by hand. The stage row carries the manifest's closure
    and the census names the removed SKUs, using the manifest's own keys."""
    from training import dedupe

    removed = pd.DataFrame(
        [
            {"sku_id": "s1", "rep_id": "s9", "tier": dedupe._TIER_T1},
            {"sku_id": "s2", "rep_id": "s9", "tier": dedupe._TIER_T1},
            {"sku_id": "s3", "rep_id": "s8", "tier": dedupe._TIER_T2},
            {"sku_id": "s4", "rep_id": "s7", "tier": dedupe._TIER_T3},
        ]
    )
    removals_path = tmp_path / "removals.csv"
    removed.to_csv(removals_path, index=False)
    monkeypatch.setattr(dedupe, "CSV_REMOVALS", removals_path)
    monkeypatch.setattr(dedupe, "CSV_SUMMARY", tmp_path / "summary.csv")
    row_accounting = {
        "input_rows": 10,
        "output_rows": 6,
        "dropped": {
            "t1_retailer_gtin": 2,
            "t1_5_retailer_malformed_gtin_same_product": 0,
            "t2_retailer_title_gtin": 1,
            "t3_retailer_title_identity_partition": 1,
        },
        "skipped_checksum_invalid": 1,
        "deferred_to_t3": 0,
        "ambiguous_offer_groups": 2,
        "unresolved_identity_review_rows": 0,
    }
    summary = [
        {"tier": dedupe._TIER_T1, "dropped_rows": 2},
        {"tier": dedupe._TIER_T2, "dropped_rows": 1},
        {"tier": dedupe._TIER_T3, "dropped_rows": 1},
    ]
    run = tracing.TraceRun(dedupe.STAGE)
    dedupe._record_dedupe(
        run,
        n0=10,
        deduped=pd.DataFrame({"sku_id": list("abcdef")}),
        ambiguous_out=pd.DataFrame({"group": [1, 2]}),
        conflicts=pd.DataFrame({"tier": ["conflict"]}),
        summary=summary,
        row_accounting=row_accounting,
    )
    run.write(trace_target)

    frame = read_validated(trace_target)
    assert set(frame["stage"]) == {dedupe.STAGE}
    collapsed = one(frame, "rows.collapsed")
    assert int(collapsed["in_count"]) == 10
    assert int(collapsed["out_count"]) == 6
    assert int(collapsed["dropped_count"]) == 4
    detail = detail_of(collapsed)
    assert detail["output_rows"] == 6
    assert len(detail["tier_summary"]) == len(summary)

    # the census is the manifest's OWN vocabulary, zero buckets omitted
    measured = census(frame, "removal")
    assert measured == {
        key: value for key, value in row_accounting["dropped"].items() if value
    }
    assert sum(measured.values()) == int(collapsed["dropped_count"])

    # every removed SKU is NAMED, with its surviving representative
    named = entity_rows(frame)
    assert set(named["key"]) == {"s1", "s2", "s3", "s4"}
    by_key = {row["key"]: row for _, row in named.iterrows()}
    assert by_key["s3"]["reason"] == "t2_retailer_title_gtin"
    assert by_key["s4"]["reason"] == "t3_retailer_title_identity_partition"
    assert detail_of(by_key["s3"])["rep_id"] == "s8"
    assert detail_of(by_key["s1"])["rep_id"] == "s9"


# ══════════════════════════════════════════════════════════════════════════
# pipeline volume caps (the 12 MB trace)
# ══════════════════════════════════════════════════════════════════════════
def test_pipeline_bounds_wide_cells_and_nulls_counts_on_unit_change(trace_target, monkeypatch):
    """The three volume defects, each fixed at its own helper:

    * a UNIT-CHANGE step (one block -> its candidate pairs, one pair -> its two
      directions, one row -> row + canonical) emits more than it received, so it
      must NOT state an in/out pair: a negative dropped_count is not a legal row;
    * a wide distribution is a top-N census plus NUMERIC remainder counts, not a
      megabyte cell (the dimension-conflict cell was 1.07 MB / 13,067 strings);
    * a free-text reason embeds the values that produced it, so it cannot be a
      bucket label (589 rows / 257 singleton buckets): the CATEGORY is the label
      and the full reason stays in the sampled row's detail.

    Every capped value also has to survive the row contract.
    """
    import json

    import pipeline

    # ── unit changes: no funnel pair, never a negative drop ──
    for expansion, contraction in (
        ((685, 1370), False),
        ((13102, 1555018), False),
        ((100, 40), True),
        ((40, 40), True),
    ):
        incoming, outgoing, unit_change = pipeline.unit_change_counts(*expansion)
        assert unit_change is (not contraction)
        if contraction:
            assert (incoming, outgoing) == expansion
        else:
            assert incoming is None and outgoing == expansion[1]

    # the same rule guards the other vocabulary row (build_reference): ONE brand
    # string can yield several tokens, so the token set can exceed the brands
    from training import build_reference as reference_module

    monkeypatch.setattr(
        reference_module,
        "load_dataset",
        lambda columns=None: pd.DataFrame({"brand": ["one two three"]}),
    )
    reference_run = tracing.TraceRun("build_reference")
    reference_module.BrandVocabulary.build(reference_run)
    reference_run.write(trace_target.parent / "vocab_trace.csv")
    vocab_row = one(
        read_validated(trace_target.parent / "vocab_trace.csv"),
        "vocabulary.tokens_built",
    )
    assert vocab_row["in_count"] == "" and vocab_row["dropped_count"] == ""
    assert int(vocab_row["out_count"]) == 3  # one brand string -> three tokens
    assert detail_of(vocab_row)["unit_change"] is True

    # ── wide cells: top-N + exact numeric remainder ──
    census = pipeline._bounded_census([f"v{index % 5000}" for index in range(20000)])
    assert census["distinct"] == 5000
    assert len(census["top"]) == pipeline.CENSUS_TOP_N
    assert census["others_buckets"] == 5000 - pipeline.CENSUS_TOP_N
    assert census["others"] + sum(int(entry.rsplit("=", 1)[1]) for entry in census["top"]) == 20000
    assert pipeline._bounded_census([])["distinct"] == 0

    # ── a huge cell is truncated IN PLACE and says what it elided ──
    wide = pipeline._capped_detail(
        {"conflict_paired": [f"conflict-{index}" for index in range(13067)]}
    )
    assert wide["detail_truncated"] is True
    assert wide["detail_bytes_before"] > pipeline.CENSUS_CELL_BYTES
    assert len(json.dumps(wide).encode()) <= pipeline.CENSUS_CELL_BYTES
    long_text = pipeline._capped_detail({"conflict_paired": "x" * 300_000})
    assert len(json.dumps(long_text).encode()) <= pipeline.CENSUS_CELL_BYTES
    assert "elided" in long_text["conflict_paired"]
    assert pipeline._capped_detail({"pairs": 3}) == {"pairs": 3}  # untouched

    # ── the reason label is the category; the values stay in the detail ──
    embedded = "Declared product identity differs or is incomplete: flavor,organic"
    assert pipeline._reason_category(embedded) == (
        "Declared product identity differs or is incomplete"
    )
    assert pipeline._reason_category("Pack blocker: pack size, package type") == "Pack blocker"

    # ── the rows these helpers produce are legal trace rows ──
    incoming, outgoing, unit_change = pipeline.unit_change_counts(685, 1370)
    run = tracing.TraceRun("pairs")
    run.add(
        "payload",
        "materialized",
        in_count=incoming,
        out_count=outgoing,
        detail={"unit_change": unit_change},
    )
    run.add("attribute_gate", "pair_dimension_census", detail=wide)
    run.add(
        "gate",
        "decision_hard_no",
        scope="group",
        in_count=5000,
        out_count=5000,
        detail={"reasons": census},
    )
    run.write(trace_target)
    trace = read_validated(trace_target)
    materialized = one(trace, "payload.materialized")
    assert materialized["in_count"] == "" and materialized["dropped_count"] == ""
    assert detail_of(materialized)["unit_change"] is True
    gate = one(trace, "gate.decision_hard_no")
    assert detail_of(gate)["reasons"]["others_buckets"] == 5000 - pipeline.CENSUS_TOP_N
    assert detail_of(one(trace, "attribute_gate.pair_dimension_census"))["detail_truncated"] is True


# ══════════════════════════════════════════════════════════════════════════
# data_prep / pipeline (the stage-1 single writer)
# ══════════════════════════════════════════════════════════════════════════
def _tiny_raw_export() -> pd.DataFrame:
    """A raw-export-shaped frame: four brands, two valid gtins each, one
    collapsed duplicate per gtin.

    The population is deliberately wider than the raw export's column count:
    the pre-existing ``column_contract.input_frame`` row counts the FRAME as its
    in_count and the COLUMNS as its out_count, so a frame narrower than its own
    column set would make that (already published) row an invalid trace row.
    """
    gtins = valid_gtins(8)
    body = [
        {
            "gtin": gtin,
            "brand": f"brand {index // 2}",
            "sku_name_eng": f"drink {gtin} 330 ml",
            "attribute": "type water",
            "description_short_eng": "",
            "breadcrumbs_eng": "",
        }
        for index, gtin in enumerate(gtins)
    ]
    rows = [dict(row, sku_id=f"s{index}{suffix}")
            for index, row in enumerate(body)
            for suffix in ("", "dup")]
    return pd.DataFrame(rows)


@pytest.fixture()
def redirected_results(monkeypatch, tmp_path):
    """Point the stage-1 artifacts at a tmp dir, so the pipeline writes nothing
    into the frozen results/ tree (mirrors tests/test_consolidated_trace.py)."""
    import core.common as common
    import pipeline

    monkeypatch.setattr(pipeline, "RESULTS", tmp_path)
    monkeypatch.setitem(common._BINDING_ROOTS, "results", tmp_path)
    monkeypatch.setitem(common.F, "canonical_records", tmp_path / "canonical_records.csv")
    monkeypatch.setitem(common.F, "gate_results", tmp_path / "gate_results.csv")
    return tmp_path


def test_data_prep_owns_one_stage_writer_for_pipeline_and_manifest(
    trace_target, redirected_results
):
    """The pipeline writes NO trace when handed an external writer; data_prep then
    commits the pipeline's rows and its own manifest accounting as ONE stage."""
    import training.data_prep as data_prep
    from pipeline import run_within_brand_pipeline

    raw = _tiny_raw_export()
    run = tracing.TraceRun("data_prep")
    pairs, canon = run_within_brand_pipeline(raw, run)
    # the external writer defers the commit, so nothing is on disk yet
    assert not trace_target.exists()

    accounting = data_prep._dp_manifest_accounting(raw, pairs, canon)
    accounting["flags_census"] = data_prep._flag_census(canon)
    data_prep._record_stage_rows(run, raw, canon, accounting)
    run.write(trace_target)

    trace = read_validated(trace_target)
    assert set(trace["stage"]) == {"data_prep"}  # ONE stage, one commit
    # the invariant is IN the trace: exactly one writer claims the stage
    ownership = one(trace, "stage_ownership.single_writer")
    assert detail_of(ownership)["writer"] == "training.data_prep"
    assert rows_of(trace, "column_contract.input_frame").shape[0] == 1
    assert rows_of(trace, "gtin_guard.identity_claims_evaluated").shape[0] == 1

    # the manifest's closure, restated in the trace, closes arithmetically
    manifest_row = one(trace, "manifest.row_accounting")
    acc = detail_of(manifest_row)
    assert int(manifest_row["in_count"]) == (
        int(manifest_row["out_count"])
        + sum(acc["dropped"].values())
        + acc["collapsed_same_gtin"]
    )
    assert acc["collapsed_same_gtin"] == 8  # one crafted duplicate per gtin
    assert int(acc["gate_pairs"]) == len(pairs)

    # the guard readback and the batch-grain statement are present
    guard = one(trace, "guard.recomputed_census")
    assert int(guard["in_count"]) == len(raw)
    assert int(guard["dropped_count"]) == 0
    assert one(trace, "batch_grain.not_applicable")["reason"].startswith(
        "this stage reads ONE"
    )
    # the flag census is a census even when no flag is present
    assert rows_of(trace, "flags.sample_budget").shape[0] == 1


# ══════════════════════════════════════════════════════════════════════════
# negative_supply (the lane, previously untraced)
# ══════════════════════════════════════════════════════════════════════════
def _supply_frames():
    """A tiny lane catalog that exercises every supply outcome.

    Two soda anchors whose canonical records differ in ONE whitelisted
    dimension (a REAL partner), a same-title partner whose canonical record
    differs in three (a rejection), a same-title partner with NO canonical
    record (a rejection), and two rows of ONE gtin no real partner covers (one
    mints, the other is skipped as same_entity). Returns the frames plus the
    duplicated gtin the skip must name.
    """
    covered, partner, other, missing, duplicated = valid_gtins(5)
    soda = "zesty lemon soda 330ml can"
    df = pd.DataFrame({
        "sku_id": ["covered", "partner", "other", "missing", "dup", "dup2"],
        "retailer": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "gtin": [covered, partner, other, missing, duplicated, duplicated],
        "sku_name_eng": [
            soda, "zesty lemon soda 500ml can", soda, soda,
            "still lemon water 1l bottle", "still lemon water 1l bottle",
        ],
    })

    def record(gtin, volume, pack):
        return {"gtin": gtin, "canonical": "x", "volume_set": volume,
                "flavor_set": "{'lemon'}", "pack_set": pack,
                "package_type_set": pack, "package_material_set": "frozenset()",
                "carbonation_set": "{'carbonated'}", "sweetener_set": "frozenset()"}

    canonical = pd.DataFrame([
        record(covered, "{'355'}", "{'can'}"),
        record(partner, "{'500'}", "{'can'}"),
        record(other, "{'500'}", "{'bottle'}"),
        record(duplicated, "{'1000'}", "{'bottle'}"),
    ])
    labeled = pd.DataFrame({
        "gtin1": [covered, duplicated], "gtin2": [partner, covered],
        "true_label": [0, 1],
    })
    gates = pd.DataFrame({
        "gtin1": [covered], "gtin2": [partner],
        "gate_decision": ["hard_no"], "gate_reason": ["volume"],
    })
    return df, canonical, labeled, gates, duplicated


def test_negative_supply_traces_its_funnels_and_names_every_rejection(
    trace_target, tmp_path, monkeypatch
):
    """The lane's attrition had NO trace row: block -> mine -> mint is a named
    funnel, every rejected candidate and skipped anchor keeps its exact reason,
    and the censuses carry the lane's OWN funnel counts (never the sample's)."""
    from core.portable_archive import ByteCount

    import core.common as common
    from training.negative_supply import NegativeSupply, NegativeSupplySpec

    monkeypatch.setattr(common, "RESULTS", tmp_path / "results")
    df, canonical, labeled, gates, duplicated = _supply_frames()
    supply = NegativeSupply(spec=NegativeSupplySpec(), df=df, canonical=canonical,
                            gates=gates, labeled=labeled)
    manifest = supply.emit("trace-test")

    frame = read_validated(trace_target)
    assert set(frame["stage"]) == {"negative_supply"}
    funnel = manifest["funnel"]

    # the blocker is a FAN-OUT census (one anchor -> up to top_k candidates),
    # so it states both counts and no in/out pair
    block = one(frame, "block.candidates_blocked")
    assert str(block["in_count"]) == "" and str(block["out_count"]) == ""
    block_detail = detail_of(block)
    assert block_detail["anchors"] == funnel["block"]["anchors"]
    assert block_detail["candidates"] == funnel["block"]["candidates"]
    assert block_detail["top_k"] == NegativeSupplySpec().blocker.top_k

    # the mine funnel closes over its own rejection census
    mine = funnel["mine_real"]
    mined = one(frame, "mine_real.partners_mined")
    assert int(mined["in_count"]) == (
        mine["no_canonical_record"] + mine["diff_count_1"] + mine["diff_count_other"]
    )
    assert int(mined["out_count"]) == mine["real_partners"] == mine["diff_count_1"]
    assert mine["real_partners"] >= 1
    measured = census(frame, "mine_real.rejected")
    assert measured == {"no_canonical_record": mine["no_canonical_record"],
                        "diff_count_other": mine["diff_count_other"]}
    assert int(mined["dropped_count"]) == sum(measured.values())
    assert measured["no_canonical_record"] > 0 and measured["diff_count_other"] > 0

    # every rejected candidate is NAMED with the exact gate that rejected it
    rejected = entity_rows(frame)
    rejected = rejected[rejected["step"] == "mine_real.rejected.sampled"]
    assert len(rejected) == sum(measured.values())  # everything fits the cap here
    reasons = {str(row["reason"]) for _, row in rejected.iterrows()}
    assert reasons == set(measured)
    others = [detail_of(row) for _, row in rejected.iterrows()
              if row["reason"] == "diff_count_other"]
    assert others and all(int(d["diff_count"]) >= 2 for d in others)
    assert all(len(d["differing"]) == d["diff_count"] for d in others)
    absent = [detail_of(row) for _, row in rejected.iterrows()
              if row["reason"] == "no_canonical_record"]
    assert absent and all(d["missing_canonical"] in ("anchor", "partner",
                                                    "anchor, partner")
                          for d in absent)
    # the census rows carry their own sampling statement
    for _, row in frame[frame["step"] == "mine_real.rejected.reason_census"].iterrows():
        evidence = detail_of(row)
        assert evidence["omitted"] == evidence["population"] - evidence["sampled"]
        assert evidence["cap_per_reason"] == tracing.ENTITY_SAMPLE_PER_REASON

    # the mint funnel closes over its anchor-level skips; empty_pool is counted
    # per move ATTEMPT and is never folded into the row funnel
    mint = funnel["mint"]
    minted = one(frame, "mint.partners_minted")
    assert int(minted["in_count"]) == mint["uncovered_sku_rows"]
    assert int(minted["out_count"]) == mint["minted"]
    skipped = census(frame, "mint.skipped")
    assert skipped == {name: mint[name] for name in
                       ("same_entity", "target_cap", "no_move_surface",
                        "below_blocker_floor", "empty_pool")}
    anchor_skips = sum(skipped[name] for name in
                       ("same_entity", "target_cap", "no_move_surface",
                        "below_blocker_floor"))
    assert int(minted["dropped_count"]) == anchor_skips
    assert skipped["same_entity"] > 0
    named_skip = entity_rows(frame)
    named_skip = named_skip[named_skip["step"] == "mint.skipped.sampled"]
    same_entity = named_skip[named_skip["reason"] == "same_entity"]
    assert list(same_entity["key"]) == [duplicated]
    assert detail_of(same_entity.iloc[0])["entity_level"] == "gtin"

    # the emitted table and the coverage funnel agree with the lane's manifest,
    # and the trace pins the size of the bytes actually published
    emitted = detail_of(one(frame, "pairs.emitted"))
    assert emitted["populations"] == manifest["populations"]
    assert emitted["pairs"] == sum(manifest["populations"].values())
    pairs_csv = tmp_path / "results" / "negative_supply" / "trace-test" / "pairs.csv"
    assert emitted["pairs_size"] == ByteCount(pairs_csv.read_bytes()).total
    covered = one(frame, "coverage.anchors_covered")
    assert int(covered["in_count"]) == manifest["coverage"]["anchors_total"]
    assert int(covered["out_count"]) == manifest["coverage"]["anchors_with_real_partner"]
