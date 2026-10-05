"""Run-history layer semantics: list/facts/compare on fabricated tiny runs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from model_tracks.run_history import (
    bundle_facts, compare, facts, list_bundles, list_training_runs)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _manifest(path: Path, *, status: str, stages: dict, gate_census: dict | None = None,
              bundle: dict | None = None) -> Path:
    document: dict = {"status": status, "stage_metrics": stages}
    if gate_census is not None:
        document["gate_census"] = gate_census
    if bundle is not None:
        document["bundle"] = bundle
    return _write(path, json.dumps(document))


def _make_bundle_run(root: Path, run_id: str, *, status: str = "running",
                     dataset_rows: int | None = None, stages: dict | None = None,
                     gate_census: dict | None = None, labeled: str | None = None,
                     minted: int | None = None, handoff: dict | None = None,
                     offenders: list[tuple[str, float]] | None = None,
                     bundle: dict | None = None) -> Path:
    run_dir = root / run_id
    _manifest(run_dir / "manifest.json", status=status,
              stages=stages or {}, gate_census=gate_census, bundle=bundle)
    if dataset_rows is not None:
        _write(run_dir / "dedupe.log",
               f"wrote /x/sku_to_rep.csv ({dataset_rows:,} rows)\n")
    if labeled is not None:
        _write(run_dir / "labeled_pairs.log",
               f"labeled_pairs.csv: {labeled.split('/')[0]} rows "
               f"({labeled.split('/')[1]} pos / {labeled.split('/')[2]} hard-neg)\n")
    if minted is not None:
        _write(run_dir / "discriminator.json", json.dumps({"minted_rows": minted}))
    if handoff is not None:
        _write(run_dir / "handoff.json", json.dumps(handoff))
    if offenders is not None:
        lines = ["# timing offenders"] + [
            f"{index:3d}. {label:<40.40} {seconds:8.3f}s  {10.0 * index:4.1f}%"
            for index, (label, seconds) in enumerate(offenders, 1)]
        _write(run_dir / "timing_offenders.log", "\n".join(lines) + "\n")
    return run_dir


def _stages(**seconds: float) -> dict:
    return {name: {"status": "complete", "returncode": 0, "seconds": value}
            for name, value in seconds.items()}


def test_list_bundles_requires_manifest_and_training_runs_use_markers(tmp_path):
    prep = tmp_path / "results" / "training_prep"
    good = _make_bundle_run(prep, "20261005T100000000001", status="running")
    (prep / "20261005T100000000002").mkdir()  # no manifest.json: not a bundle run
    training = tmp_path / "training_results"
    marker_run = training / "20261005T110000000000Z" / "worker_1" / "text__tag"
    _write(marker_run / "track_inventory.json", "{}")
    (training / "not_a_run").mkdir()

    bundles = list_bundles([prep])
    runs = list_training_runs([training])

    assert [entry.name for entry in bundles] == [good.name]
    assert [entry.name for entry in runs] == ["20261005T110000000000Z"]


def test_facts_reads_run_own_records_and_never_guesses(tmp_path):
    root = tmp_path / "prep"
    run = _make_bundle_run(
        root, "20261005T120000000000", status="running",
        dataset_rows=71_623,
        stages=_stages(dedupe=12.225, validation=368.734),
        gate_census={"total_pairs": 135_246, "hard_no": 115_789,
                     "proceed": 338, "fallback": 19_119},
        labeled="805/199/606", minted=4_279,
        handoff={"status": "pass", "inputs": [{"input": "a"}, {"input": "b"}],
                 "loss_batch_correctness": {"loss": "cosine", "epochs": 10}},
        offenders=[("stage/canonical_and_gates", 653.185),
                   ("stage/full_bundle", 420.008)])

    run_facts = facts(str(run))

    assert run_facts.source == "bundle"
    assert run_facts.dataset_rows == 71_623
    assert run_facts.created == "2026-10-05T12:00:00+00:00"
    assert run_facts.census.total_pairs == 135_246
    assert run_facts.labeled.kept == 805 and run_facts.labeled.pos == 199
    assert run_facts.minted_rows == 4_279
    assert run_facts.handoff.metered_inputs == 2
    assert run_facts.handoff.loss_batch_attested is True
    assert run_facts.offenders_top[0].label == "stage/canonical_and_gates"
    assert run_facts.stage_seconds_per_1k("dedupe") == pytest.approx(
        12.225 * 1000 / 71_623)
    assert run_facts.stage_seconds_per_1k("missing") is None


def test_facts_without_dataset_record_holds_none(tmp_path):
    run = _make_bundle_run(tmp_path / "prep", "r1", status="failed")
    assert bundle_facts(run).dataset_rows is None


def test_facts_missing_manifest_fails_loud(tmp_path):
    empty = tmp_path / "prep" / "run_without_manifest"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="run_without_manifest"):
        bundle_facts(empty)


def test_compare_scale_normalized_surfaces_growth_not_cohort_size(tmp_path):
    root = tmp_path / "prep"
    # Same cohort shape, different scale: b processed half the rows.
    full = _make_bundle_run(
        root, "20261005T130000000000", status="running", dataset_rows=71_623,
        stages=_stages(heavy=100.0, light=40.0),
        labeled="800/200/600")
    half = _make_bundle_run(
        root, "20261005T130000000001", status="running", dataset_rows=35_561,
        stages=_stages(heavy=100.0, light=18.0),
        labeled="400/100/300")  # counts fell, per-1k rates did not
    report = compare(bundle_facts(full), bundle_facts(half))

    assert report.pair_kind == "complete_pair"
    assert report.scale_normalized is True
    heavy = next(row for row in report.stages if row.stage == "heavy")
    light = next(row for row in report.stages if row.stage == "light")
    assert heavy.regression is True  # same seconds over half the rows: grew per-1k
    assert heavy.per_1k_ratio == pytest.approx(2.0, rel=0.01)
    assert light.regression is False  # b is faster per-1k
    pos = next(row for row in report.outputs if row.name == "labeled_pos")
    assert pos.regression is None or pos.regression is False
    # absolute count halved but per-1k rate unchanged -> no false regression
    assert pos.per_1k_ratio == pytest.approx(1.0, rel=0.01)
    assert all(item.kind != "output_rate" for item in report.regressions)
    assert [item.kind for item in report.regressions] == ["stage_time"]


def test_compare_output_rate_regression_and_no_size_false_positive(tmp_path):
    root = tmp_path / "prep"
    a = _make_bundle_run(root, "a", status="running", dataset_rows=1_000,
                         stages=_stages(dedupe=2.0), labeled="800/200/600")
    b = _make_bundle_run(root, "b", status="running", dataset_rows=1_000,
                         stages=_stages(dedupe=2.0), labeled="600/150/450")
    report = compare(bundle_facts(a), bundle_facts(b))
    pos = next(row for row in report.outputs if row.name == "labeled_pos")
    assert pos.regression is True  # same scale, rate genuinely fell
    assert any(item.kind == "output_rate" for item in report.regressions)

    bigger = _make_bundle_run(root, "c", status="running", dataset_rows=2_000,
                              stages=_stages(dedupe=4.0), labeled="1600/400/1200")
    report_same_rate = compare(bundle_facts(a), bundle_facts(bigger))
    pos_same_rate = next(row for row in report_same_rate.outputs
                         if row.name == "labeled_pos")
    assert pos_same_rate.a == 200 and pos_same_rate.b == 400
    assert pos_same_rate.regression is not True  # bigger cohort, same rate


def test_compare_different_status_is_incomplete_pair(tmp_path):
    root = tmp_path / "prep"
    a = _make_bundle_run(root, "ok", status="complete", dataset_rows=1_000,
                         stages=_stages(dedupe=2.0))
    b = _make_bundle_run(root, "broken", status="failed", dataset_rows=1_000,
                         stages=_stages(dedupe=99.0))
    report = compare(bundle_facts(a), bundle_facts(b))

    assert report.pair_kind == "incomplete_pair"
    assert report.regressions == []
    assert all(row.regression is None for row in report.stages)


def test_compare_attested_flag_flip_is_contract_regression(tmp_path):
    root = tmp_path / "prep"
    attested = {"status": "pass", "inputs": [{"input": "bundle"}],
                "loss_batch_correctness": {"loss": "cosine", "epochs": 10,
                                           "batch_sizes": {"cpu": 64}}}
    a = _make_bundle_run(root, "attested", status="running", handoff=attested)
    b = _make_bundle_run(root, "unattested", status="running")
    report = compare(bundle_facts(a), bundle_facts(b))

    handoff = next(row for row in report.contracts if row.name == "handoff")
    assert handoff.a == "pass" and handoff.b == "absent"
    assert handoff.switched_pass_to_fail is True
    assert any(item.kind == "contract" for item in report.regressions)

    reverse = compare(bundle_facts(b), bundle_facts(a))
    handoff_reverse = next(row for row in reverse.contracts if row.name == "handoff")
    assert handoff_reverse.switched_pass_to_fail is False
    assert all(item.kind != "contract" for item in reverse.regressions)


def test_compare_stage_contract_switch_and_offender_drift(tmp_path):
    root = tmp_path / "prep"
    a = _make_bundle_run(root, "a", status="running", dataset_rows=10_000,
                         stages=_stages(dedupe=10.0, full_bundle=50.0),
                         offenders=[("stage/full_bundle", 50.0),
                                    ("stage/dedupe", 10.0)])
    b = _make_bundle_run(root, "b", status="running", dataset_rows=10_000,
                         stages={"dedupe": {"status": "failed", "returncode": 1,
                                            "seconds": 20.0}},
                         offenders=[("stage/dedupe", 20.0)])
    report = compare(bundle_facts(a), bundle_facts(b))

    dedupe = next(row for row in report.stages if row.stage == "dedupe")
    assert dedupe.regression is True  # per-1k doubled at fixed scale
    assert report.worst_offender_drift is True
    assert report.worst_offender_a.label == "stage/full_bundle"
    assert report.worst_offender_b.label == "stage/dedupe"
    switched = next(row for row in report.contracts if row.name == "stage/dedupe")
    assert switched.switched_pass_to_fail is True
    assert [item.kind for item in report.regressions] == ["stage_time", "contract"]


def test_offenders_fall_back_to_timings_json_when_log_absent(tmp_path):
    run = _make_bundle_run(tmp_path / "prep", "r", status="running")
    _write(run / "timings.json", json.dumps({"stages": {
        "dedupe": {"seconds": 1.0}, "validation": {"seconds": 9.0}}}))
    offenders = bundle_facts(run).offenders_top
    assert [offender.label for offender in offenders] == ["stage/validation",
                                                          "stage/dedupe"]


def test_facts_unknown_id_fails_loud_with_searched_paths(tmp_path):
    prep = tmp_path / "prep"
    _make_bundle_run(prep, "known", status="running")
    training = tmp_path / "training_results"
    training.mkdir()
    with pytest.raises(FileNotFoundError, match="known run"):
        facts("nope", prep_roots=[prep], training_roots=[training])


def test_compare_with_training_run_side_is_treated_incomplete(tmp_path):
    # Training runs carry no stage/census/output facts on this layer, so a
    # training-vs-bundle (or training-vs-training) pair must never emit
    # regressions even with equal (None) status.
    prep = tmp_path / "prep"
    bundle = bundle_facts(_make_bundle_run(
        prep, "20261005T130000000000", status="running", dataset_rows=1_000,
        stages=_stages(heavy=50.0), labeled="800/200/600", minted=500))
    training_root = tmp_path / "training_results"
    marker_run = training_root / "20261005T140000000000" / "worker_1" / "text__tag"
    _write(marker_run / "track_inventory.json", "{}")
    training = facts("20261005T140000000000", prep_roots=[prep],
                     training_roots=[training_root])
    assert training.source == "training"
    assert training.stages == []

    report = compare(training, training)
    assert report.pair_kind == "incomplete_pair"
    assert report.regressions == []

    report = compare(bundle, training)
    assert report.pair_kind == "incomplete_pair"
    assert report.regressions == []


def test_runs_table_pairs_older_neighbor_as_baseline(tmp_path):
    # Dashboard pairing direction pin: each run's badge compares the run
    # against its CHRONOLOGICALLY OLDER neighbor (compare(older, run)), so a
    # REGRESSION badge means "this run got slower per-1k-rows than the run
    # before it" — never "the run after it was faster".
    import re

    from dashboard.app import _runs_table

    prep = tmp_path / "results" / "training_prep"
    training = tmp_path / "training_results"
    training.mkdir()
    _make_bundle_run(prep, "r1", status="running", dataset_rows=1_000,
                     stages=_stages(dedupe=1.0), labeled="800/200/600",
                     minted=300)
    _make_bundle_run(prep, "r2", status="running", dataset_rows=1_000,
                     stages=_stages(dedupe=3.0), labeled="800/200/600",
                     minted=300)   # same rates, 3x stage seconds: REGRESSION
    _make_bundle_run(prep, "r3", status="running", dataset_rows=1_000,
                     stages=_stages(dedupe=3.0), labeled="1,600/400/1,200",
                     minted=600)   # same per-1k rate as r2: improvement

    rows, training_rows = _runs_table(prep, training)

    badges = {}
    for row in rows:
        run_id = re.search(r'/runs/compare\?a=([^&"]+)&b=\1">', row).group(1)
        pair = re.search(r'<a href="/runs/compare\?a=([^&"]+)&b=([^&"]+)"[^>]*>(?:REGRESSION \((\d+)\)|pair\?)', row)
        badges[run_id] = (pair.group(3) if pair else None,
                          (pair.group(1), pair.group(2)) if pair else None)
    # 'r2' regressed vs 'r1' (3x per-1k time); 'r3' improved vs 'r2'
    assert badges["r2"][0] == "1" and badges["r2"][1] == ("r1", "r2")
    assert badges["r3"][0] is None   # rate unchanged vs r2: no badge

    # table rendering order matches the CLI's older->newer compare order
    assert list(badges) == ["r1", "r2", "r3"]
    assert training_rows == []   # no training runs under the fabricated root


def test_runs_table_renders_unparseable_run_row_instead_of_crashing(tmp_path):
    from dashboard.app import _runs_table

    prep = tmp_path / "results" / "training_prep"
    training = tmp_path / "training_results"
    training.mkdir()
    good = _make_bundle_run(prep, "good", status="running", dataset_rows=1_000,
                            stages=_stages(dedupe=1.0))
    _write(prep / "broken" / "manifest.json", json.dumps({"status": "running"}))
    _write(prep / "broken" / "labeled_pairs.log", "no summary line\n")

    rows, training_rows = _runs_table(prep, training)

    rendered_with_broken = ''.join(rows)
    assert 'unreadable run record' in rendered_with_broken
    assert 'labeled_pairs.cs' in rendered_with_broken
    good_row = next(row for row in rows if '/runs/compare?a=good&b=good' in row)
    assert 'unreadable run record' not in good_row   # the good row is a normal run row
    broken_row = next(row for row in rows if 'broken' in row)
    assert 'unreadable run record' in broken_row     # the broken row carries the context
    assert training_rows == []


def test_handoff_without_status_records_missing_status(tmp_path):
    run = _make_bundle_run(tmp_path / "prep", "h", status="running")
    _write(run / "handoff.json", json.dumps({"inputs": [{"input": "a"}]}))
    handoff = bundle_facts(run).handoff
    assert handoff.status == "missing_status"
    report = compare(bundle_facts(_make_bundle_run(
        tmp_path / "prep", "hp", status="running",
        handoff={"status": "pass", "inputs": [],
                 "loss_batch_correctness": {"loss": "cosine"}})), bundle_facts(run))
    handoff_row = next(row for row in report.contracts if row.name == "handoff")
    assert handoff_row.a == "pass" and handoff_row.b == "missing_status"
    assert handoff_row.switched_pass_to_fail is True   # pass -> anything-but-pass IS the drop


def test_offenders_hashlog_counts_skipped_lines_and_still_parses(tmp_path):
    run = _make_bundle_run(tmp_path / "prep", "sk", status="running",
                           offenders=[("stage/full_bundle", 30.0)])
    _write(run / "timing_offenders.log",
           "# timing offenders\n"
           "  1. stage/full_bundle                            30.000s  75.0%\n"
           "garbage line without a seconds column\n"
           "  2. bad-tail 12.5s\n")
    run_facts = bundle_facts(run)
    assert [o.label for o in run_facts.offenders_top] == ["stage/full_bundle"]
    assert run_facts.offenders_skipped == 2   # both malformed lines counted, not dropped silently
