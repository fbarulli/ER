"""Controlled batch sampler — deterministic per-population batch composition.

Gives the offline prep full control over what each batch contains, for
all loss functions.  The sampler reads the ``pair_population`` column on
the dataset (written by training.py from ``_training_pair_populations``)
and constructs each batch from a configurable template of population
counts.

Config (``config/training.yaml`` ``training.batch_sampler``):
    enabled: true
    composition:
        gate_positive: 4
        masked_positive: 4
        hard_negative: 8
    seed: 42                     # optional, defaults to SEED
    drop_last: false

Each batch = sum(composition.values()) rows.  Within each epoch each
population's indices are drawn in order, then drawn in template
order until one group runs out.

Text-level deduplication: before a row is placed in a batch, its text
content is checked against rows already in that batch.  Duplicate texts
are deferred to later batches (or dropped when a group is exhausted).
This prevents the same product appearing as both anchor and negative in
one batch — critical for MNRL's in-batch ranking loss.

Why this exists (rationale, EXP-03 / TODO TIER 0):
    - MNRL is an in-batch loss: every other row is a negative for each
      anchor, so batch composition *is* the negative pool.
    - Twin warmup needs twin rows in every batch to down-weight them;
      without twins present the warmup has no effect.
    - Per-population telemetry can only attribute loss for populations
      that appear in the batch.
    - OnlineContrastiveLoss mines hard pos/neg within the batch; both
      must be present.
    - NoDuplicatesBatchSampler (used by MNRL when uncontrolled) prevents
      duplicate texts colliding as in-batch negatives — the controlled
      sampler inherits that discipline.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Iterator

import numpy as np


def _row_text_hash(dataset, idx: int) -> str:
    """Hash the text content of one row for duplicate detection.

    Checks all string columns; non-string values are stringified.
    """
    hasher = hashlib.sha256()
    for col in dataset.column_names:
        val = dataset[idx][col]
        if isinstance(val, (list, tuple)):
            val = str(val)
        hasher.update(str(val).encode())
    return hasher.hexdigest()


class ControlledBatchSampler:
    """Deterministic per-population batch sampler with text dedup.

    Parameters
    ----------
    dataset:
        HuggingFace Dataset with a ``pair_population`` column (str per row).
    batch_size:
        Must equal sum(composition.values()).  Enforced at init.
    composition:
        Mapping population_label → count per batch, e.g.
        ``{"gate_positive": 4, "hard_negative": 8}``.
    seed:
        Base RNG seed; epoch offset is added (seed + epoch).
    drop_last:
        Drop the final incomplete batch instead of yielding it.
    population_column:
        Name of the column holding population labels.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        composition: dict[str, int],
        seed: int = 0,
        drop_last: bool = False,
        population_column: str = "pair_population",
    ):
        if not composition:
            raise ValueError("composition must be non-empty")
        if any(v <= 0 for v in composition.values()):
            raise ValueError("composition counts must be positive")
        expected = sum(composition.values())
        if batch_size != expected:
            raise ValueError(
                f"batch_size ({batch_size}) != sum(composition) ({expected})"
            )
        self.composition = composition
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        # Accept either column name (MNRL uses "population").
        populations = None
        for col in (population_column, "population"):
            if col in dataset.column_names:
                populations = dataset[col]
                break
        if populations is None:
            raise ValueError(
                f"Dataset missing population column (tried "
                f"{population_column}, population). Columns: "
                f"{dataset.column_names}"
            )
        if len(populations) != len(dataset):
            raise ValueError(
                "population column length mismatch with dataset"
            )
        self.groups: dict[str, list[int]] = defaultdict(list)
        for idx, pop in enumerate(populations):
            self.groups[str(pop)].append(idx)
        missing = set(composition) - set(self.groups)
        if missing:
            raise ValueError(
                f"composition references populations not in dataset: {missing}"
            )
        extra = set(self.groups) - set(composition)
        if extra:
            raise ValueError(
                f"dataset populations not in composition (would be silently "
                f"excluded from every batch): {extra}. "
                f"Add them to composition or remove them from the dataset."
            )
        # Flattened template in deterministic insertion order.
        self._template: list[str] = []
        for pop, count in composition.items():
            self._template.extend([pop] * count)
        self._epoch = 0
        # Pre-compute text hashes for duplicate detection.
        self._text_hashes = [_row_text_hash(dataset, i) for i in range(len(dataset))]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        queues: dict[str, list[int]] = {
            pop: list(indices)
            for pop, indices in self.groups.items()
        }
        while all(queues.values()):
            batch: list[int] = []
            batch_hashes: set[str] = set()
            for pop in self._template:
                # Draw from this population's queue, skipping duplicates.
                drawn = False
                while queues[pop]:
                    candidate = queues[pop].pop()
                    h = self._text_hashes[candidate]
                    if h in batch_hashes:
                        continue  # duplicate text — defer to later batch
                    batch.append(candidate)
                    batch_hashes.add(h)
                    drawn = True
                    break
                if not drawn:
                    # This population's queue is exhausted (all remaining
                    # texts are duplicates already in the batch).
                    # Yield what we have so far if drop_last=False,
                    # then stop the epoch.
                    if batch and not self.drop_last:
                        yield batch
                    return
            yield batch
        # Exhausted at least one group — epoch done.

    def __len__(self) -> int:
        complete = min(
            len(q) // self.composition[pop]
            for pop, q in self.groups.items()
        )
        if self.drop_last:
            return complete
        has_remainder = any(
            len(q) % self.composition[pop] > 0
            for pop, q in self.groups.items()
        )
        return complete + (1 if has_remainder else 0)
