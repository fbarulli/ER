import pytest
from pydantic import ValidationError
from training.difficulty import DifficultyEndpoint, DifficultySpec, PairDifficulty, measure_pair


def endpoint(prose, flavor='apple', volume=250):
    return DifficultyEndpoint.from_text(f'{prose} flavor_{flavor} volume_ml_{volume}')


def test_same_evidence_has_label_specific_difficulty():
    a, b = endpoint('apple juice drink'), endpoint('apple juice drink')
    assert measure_pair(a, b, 1).difficulty == 'easy'
    negative = measure_pair(a, b, 0)
    assert negative.difficulty == 'hard'
    assert not negative.conflicts


def test_single_attribute_lookalike_is_hard_negative():
    result = measure_pair(endpoint('juice drink bottle', 'apple'), endpoint('juice drink bottle', 'cherry'), 0)
    assert result.difficulty == 'hard'
    assert result.conflicts == ['flavor']


def test_positive_conflict_flags_review_without_changing_label():
    result = measure_pair(endpoint('juice drink bottle', 'apple'), endpoint('juice drink bottle', 'cherry'), 1)
    assert result.label == 1
    assert result.reason == 'positive_attribute_conflict_review'


def test_missing_evidence_is_unknown_not_an_easy_negative():
    a = DifficultyEndpoint.from_text('juice flavor_apple')
    b = DifficultyEndpoint.from_text('water volume_ml_250')
    result = measure_pair(a, b, 0)
    assert result.difficulty == 'unknown'
    assert not result.conflicts
    with pytest.raises(ValidationError, match='account for every attribute'):
        PairDifficulty(**{**result.model_dump(), 'neither_observed': []})


def test_rejects_inverted_thresholds():
    with pytest.raises(ValidationError, match='ordered'):
        DifficultySpec(low_text_overlap=.7, high_text_overlap=.5)
