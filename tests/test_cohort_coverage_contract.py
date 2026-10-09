"""The GENERAL coverage contract, adopted by the real cohort producer.

The inputs here are the REAL ones, not a synthetic stand-in: the committed
``smoke_200`` prepared fixture (its setup tree + its prepared bundle) is driven
through the real ``model_tracks.ablation_cohort.prepare_cohort``, and the frame
it emits is what the contract is validated against.

Three decisive properties:

* the contract PASSES on the real cohort and agrees with the frozen
  ``coverage.json`` strata it is adopted beside;
* adopting it moves NO byte of the emitted artifacts (proved against the same
  producer run with the adoption call neutered, not against a golden blob);
* an untagged, unaccounted or mis-declared input is REJECTED.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

import model_tracks.ablation_cohort as ablation_cohort
from core.coverage_contracts import CohortCoverage

REPO = Path(__file__).resolve().parents[1]
SETUP_NAME = 'smoke_200'
EMITTED = ('pairs.csv', 'catalog.csv', 'listings.json', 'coverage.json')


def _run_real_cohort(work: Path) -> Path:
    """Copy the real fixture and run the real producer on it. -> cohort folder."""
    setup = work / SETUP_NAME
    shutil.copytree(REPO / 'data/prepared' / SETUP_NAME, setup)
    from training.prepared_bundle import load_prepared_bundle

    _, bundle = load_prepared_bundle(setup / 'text_prepared.pkl.gz')
    return ablation_cohort.prepare_cohort(setup, bundle)


def _emitted_bytes(folder: Path) -> dict[str, bytes]:
    return {name: (folder / name).read_bytes() for name in EMITTED}


def _cohort_frame(**columns) -> pd.DataFrame:
    """A small frame with the REAL emitted cohort's dimension columns."""
    return pd.DataFrame({
        'cohort_id': ['clean:0', 'bundle:pos:0'],
        'population': ['real', 'real_bundle'],
        'difficulty_slice': ['unknown', 'easy'],
        'evaluation_scope': ['heldout', 'bundle_diagnostic'],
        'split': ['dev', 'train'],
        **columns,
    })


def test_the_real_cohort_passes_the_contract_and_the_adoption_moves_no_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Record every adoption call while still running the REAL validation, so
    # "wired end-to-end" is observed rather than assumed.
    calls: list[int] = []
    adopt = ablation_cohort.adopt_cohort_coverage

    def recording(frame: pd.DataFrame, coverage: CohortCoverage, folder: Path):
        calls.append(len(frame))
        return adopt(frame, coverage, folder)

    monkeypatch.setattr(ablation_cohort, 'adopt_cohort_coverage', recording)
    adopted_folder = _run_real_cohort(tmp_path / 'adopted')
    adopted = _emitted_bytes(adopted_folder)
    emitted = pd.read_csv(adopted_folder / 'pairs.csv', dtype=str, keep_default_na=False)
    coverage = json.loads((adopted_folder / 'coverage.json').read_text())
    assert calls == [len(emitted)]  # validated once, over the whole real cohort

    # the validated GENERAL contract travels WITH the cohort (one new member)
    persisted = json.loads(
        (adopted_folder / ablation_cohort.COHORT_CONTRACT_FILE).read_text())
    from core.coverage_contracts import ReportCoverageContract

    reparsed = ReportCoverageContract.model_validate(persisted)
    assert reparsed.records_total == len(emitted)

    # Pre-adoption behaviour is this same producer with the adoption neutered to
    # a build that persists NOTHING, so byte-identity is a direct proof the
    # wiring emits none of the four FROZEN artifacts (the contract member is the
    # only addition).
    monkeypatch.setattr(
        ablation_cohort, 'adopt_cohort_coverage',
        lambda frame, coverage, folder: ablation_cohort.cohort_report_coverage(frame))
    before = _emitted_bytes(_run_real_cohort(tmp_path / 'pre_adoption'))
    assert adopted == before
    assert not (tmp_path / 'pre_adoption' / SETUP_NAME / 'ablation_cohort'
                / ablation_cohort.COHORT_CONTRACT_FILE).exists()

    contract = ablation_cohort.cohort_report_coverage(emitted)
    names = ablation_cohort.COHORT_DIMENSION_NAMES

    # the contract covers the very records the artifact carries ...
    assert contract.records_total == len(emitted) == coverage['pair_rows']
    assert set(ablation_cohort.COHORT_TAGGED_DIMENSIONS) <= set(emitted.columns)
    # ... and re-derives exactly the strata coverage.json reports, under the ONE
    # shared dimension name (the frame's ``difficulty_slice`` column is the
    # ``difficulty`` axis, never a private second axis)
    derived = contract.derived_counts()
    assert derived[names['population']] == coverage['by_population']
    assert derived[names['evaluation_scope']] == coverage['by_scope']
    assert derived[names['difficulty_slice']] == {
        name: count for name, count in coverage['by_difficulty'].items() if count
    }
    assert 'difficulty' in contract.dimensions and 'difficulty_slice' not in contract.dimensions
    # the cohort frame carries no attribute column, so that axis is declared
    # not_applicable with a reason instead of being silently omitted ...
    assert contract.dimensions['attribute'].policy == 'not_applicable'
    assert contract.dimensions['attribute'].reason
    # ... while every dimension it DOES carry is an explicit partition
    assert all(contract.dimensions[names[name]].policy == 'partition'
               for name in ablation_cohort.COHORT_TAGGED_DIMENSIONS)


def test_untagged_unaccounted_and_mis_declared_cohorts_are_rejected() -> None:
    accepted = ablation_cohort.cohort_report_coverage(_cohort_frame())
    assert accepted.records_total == 2
    assert accepted.dimensions['population'].counts == {'real': 1, 'real_bundle': 1}

    # a NaN tag is NOT a tag (it must never be read back as the string 'nan')
    with pytest.raises(ValidationError, match='blank tag'):
        ablation_cohort.cohort_report_coverage(
            _cohort_frame(population=['real', float('nan')]))
    # neither is an invisible-only tag
    with pytest.raises(ValidationError, match='blank tag'):
        ablation_cohort.cohort_report_coverage(_cohort_frame(split=['dev', '\u200b']))
    # a dimension the frame does not carry at all
    with pytest.raises(ValueError, match='carries no'):
        ablation_cohort.cohort_report_coverage(_cohort_frame().drop(columns=['split']))
    # zero pair rows cannot claim coverage
    with pytest.raises(ValidationError):
        ablation_cohort.cohort_report_coverage(_cohort_frame().iloc[0:0])

    # when the frame DOES carry the attribute axis it is a measured dimension,
    # and a blank tag there fails like every other dimension
    tagged = ablation_cohort.cohort_report_coverage(_cohort_frame(attribute=['brand', 'volume']))
    assert tagged.dimensions['attribute'].policy == 'overlap'
    assert tagged.derived_counts()['attribute'] == {'brand': 1, 'volume': 1}
    with pytest.raises(ValidationError, match='blank tag'):
        ablation_cohort.cohort_report_coverage(_cohort_frame(attribute=['brand', '']))

    # a declared stratum set that disagrees with the carried tags is rejected
    mis_declared = CohortCoverage.model_validate({
        'cohort_size': '0' * 64, 'pair_rows': 2,
        'minted_endpoints_total': 0, 'minted_endpoints_covered': 0,
        'by_scope': {'heldout': 1, 'bundle_diagnostic': 1},
        'by_population': {'real': 2},
        'by_difficulty': {'easy': 1, 'medium': 0, 'hard': 0, 'unknown': 1},
        'unknown_difficulty_policy': 'retain unknown',
    })
    with pytest.raises(ValueError, match='disagrees with the coverage contract'):
        ablation_cohort.validate_cohort_coverage(_cohort_frame(), mis_declared)
