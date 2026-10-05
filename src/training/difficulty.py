"""Structural pair difficulty measured from the exact encoder input.

This is a pre-training proxy, not an assertion about model performance.
Labels determine which evidence makes the decision harder; provenance does not.
"""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common import training_cfg
from core.schemas import DifficultySpec
from training.masking import _FIELD_PREFIXES, _field_surfaces, _field_values_conflict, field_of

Difficulty = Literal['easy', 'medium', 'hard', 'unknown']



class DifficultyEndpoint(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    prose_tokens: frozenset[str]
    attributes: dict[str, list[str]]

    @model_validator(mode='after')
    def complete(self):
        if set(self.attributes) != set(_FIELD_PREFIXES):
            raise ValueError('difficulty endpoint must cover the complete masking attribute registry')
        return self

    @classmethod
    def from_text(cls, text):
        fields = _field_surfaces(text)
        return cls(prose_tokens=frozenset(t.casefold() for t in text.split()
            if field_of(t) is None and not t.startswith('[FIELD_')),
            attributes={name: fields.get(name, []) for name in _FIELD_PREFIXES})


class PairDifficulty(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    label: Literal[0, 1]
    difficulty: Difficulty
    reason: str
    text_overlap: float = Field(ge=0, le=1)
    identical_prose: bool
    both_observed: list[str]
    one_observed: list[str]
    neither_observed: list[str]
    conflicts: list[str]

    @model_validator(mode='after')
    def complete(self):
        groups = self.both_observed + self.one_observed + self.neither_observed
        if len(groups) != len(set(groups)) or set(groups) != set(_FIELD_PREFIXES):
            raise ValueError('pair difficulty must account for every attribute exactly once')
        if not set(self.conflicts) <= set(self.both_observed):
            raise ValueError('missing evidence cannot constitute conflict')
        return self


def measure_pair(left: DifficultyEndpoint, right: DifficultyEndpoint, label: int,
                 spec: DifficultySpec | None = None) -> PairDifficulty:
    spec = spec or training_cfg().difficulty
    both, one, neither, conflicts = [], [], [], []
    for field in _FIELD_PREFIXES:
        a, b = left.attributes[field], right.attributes[field]
        if a and b:
            both.append(field)
            if _field_values_conflict(field, a, b):
                conflicts.append(field)
        elif a or b:
            one.append(field)
        else:
            neither.append(field)
    union = left.prose_tokens | right.prose_tokens
    overlap = len(left.prose_tokens & right.prose_tokens) / len(union) if union else 0.
    identical = bool(union) and left.prose_tokens == right.prose_tokens
    if label not in (0, 1):
        raise ValueError('difficulty requires a binary pair label')
    if not union or len(both) < spec.min_both_observed:
        difficulty, reason = 'unknown', 'insufficient_comparable_evidence'
    elif label == 1:
        if conflicts:
            difficulty, reason = 'hard', 'positive_attribute_conflict_review'
        elif overlap <= spec.low_text_overlap:
            difficulty, reason = 'hard', 'positive_low_text_overlap'
        elif len(one) / max(len(both) + len(one), 1) >= spec.positive_missing_share:
            difficulty, reason = 'hard', 'positive_asymmetric_evidence'
        elif overlap >= spec.high_text_overlap:
            difficulty, reason = 'easy', 'positive_high_text_overlap'
        else:
            difficulty, reason = 'medium', 'positive_moderate_text_overlap'
    elif overlap >= spec.high_text_overlap and len(conflicts) <= 1:
        difficulty, reason = 'hard', 'negative_lookalike_few_conflicts'
    elif overlap <= spec.low_text_overlap or len(conflicts) >= spec.easy_negative_conflicts:
        difficulty, reason = 'easy', 'negative_low_overlap_or_many_conflicts'
    else:
        difficulty, reason = 'medium', 'negative_moderate_distinction'
    return PairDifficulty(label=label, difficulty=difficulty, reason=reason,
        text_overlap=overlap, identical_prose=identical, both_observed=both,
        one_observed=one, neither_observed=neither, conflicts=conflicts)
