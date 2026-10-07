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

Each batch follows relative population weights while groups have eligible
rows. Exhausted groups and duplicate texts produce smaller batches; every
index is presented once when drop_last=False. Duplicate texts are deferred.
Deduplication reads frozen untransformed text only: runtime masking and ANN
replacement are explicitly outside this fixed-text batching guarantee.

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

from collections import defaultdict, deque
from typing import Iterator

import numpy as np

from core.perf_switches import perf_enabled

# Share one text-hash pass between the cpu and cuda plans packed for a fold.
_SHARE_TEXT_HASHES = perf_enabled("text.share_sampler_hashes")


def _text_hash_columns(dataset) -> list[str]:
    columns = {"sentence1", "sentence2", "anchor", "positive", "negative"}
    return [col for col in dataset.column_names if col in columns]


def _row_text_hashes(dataset) -> list[frozenset[str]]:
    """Native text values only; telemetry must not defeat deduplication.

    Column-wise materialization decodes each text column once instead of one
    arrow row per index; every per-row frozenset is identical to the row-wise
    formulation, so packing and RNG order are unchanged.
    """
    names = _text_hash_columns(dataset)
    columns = {name: list(dataset[name]) for name in names}
    return [
        frozenset(str(columns[name][index]) for name in names)
        for index in range(len(dataset))
    ]


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
        text_hashes: list[frozenset[str]] | None = None,
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
        dataset = dataset.with_format(None)
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
        from core.coverage_contracts import BatchPopulationCoverage
        self.coverage = BatchPopulationCoverage(
            batch_size=batch_size, composition=composition,
            observed_counts={key: len(rows) for key, rows in self.groups.items()},
            dataset_rows=len(dataset),
        )
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
        self._packed_epoch = None
        self._packed_batches = None
        # Pre-compute text hashes for duplicate detection, or reuse a hash pass
        # already computed for the same dataset by a sibling device plan.
        if text_hashes is not None:
            if len(text_hashes) != len(dataset):
                raise ValueError("shared text hashes do not align with dataset rows")
            self._text_hashes = text_hashes
        else:
            self._text_hashes = _row_text_hashes(dataset)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _pack(self) -> list[list[int]]:
        batches = []
        rng = np.random.default_rng(self.seed + self._epoch)
        queues = {pop: deque(rng.permutation(indices)) for pop, indices in self.groups.items()}
        while any(queues.values()):
            batch = []
            texts = set()
            exhausted = set()
            for pop in self._template:
                if pop in exhausted:
                    continue
                queue = queues[pop]
                for _ in range(len(queue)):
                    index = queue.popleft()
                    if self._text_hashes[index] & texts:
                        queue.append(index)
                        continue
                    batch.append(index)
                    texts.update(self._text_hashes[index])
                    break
                else:
                    # Texts only accumulate within this batch: a population
                    # with no eligible row cannot fill another slot here.
                    exhausted.add(pop)
            # A duplicate may prevent full composition. It stays in its queue
            # for a later batch; every index is eventually presented once.
            if not batch:
                raise RuntimeError("controlled sampler could not make progress")
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches

    def __iter__(self):
        if self._packed_epoch != self._epoch:
            self._packed_batches = self._pack()
            self._packed_epoch = self._epoch
        return iter(self._packed_batches)

    def __len__(self) -> int:
        # Deduplication changes packing; derive the exact deterministic count.
        if self._packed_epoch != self._epoch:
            self._packed_batches = self._pack()
            self._packed_epoch = self._epoch
        return len(self._packed_batches)


def resolve_composition(weights, populations, batch_size):
    """Scale relative template weights over present populations, without loss."""
    from collections import Counter
    present = set(populations)
    missing = present - set(weights)
    if missing:
        raise ValueError(f"sampler template misses observed populations: {sorted(missing)}")
    active = {key: int(value) for key, value in weights.items() if key in present}
    if not active or any(value < 1 for value in active.values()):
        raise ValueError("sampler requires positive weights for present populations")
    if batch_size < len(active):
        raise ValueError("batch size cannot represent every sampler population")
    # Give every active group one slot, distribute the rest proportionally.
    total = sum(active.values())
    quotas = {key: batch_size * value / total for key, value in active.items()}
    counts = {key: max(1, int(value)) for key, value in quotas.items()}
    while sum(counts.values()) > batch_size:
        key = max((key for key in counts if counts[key] > 1), key=lambda key: counts[key] - quotas[key])
        counts[key] -= 1
    while sum(counts.values()) < batch_size:
        key = max(counts, key=lambda key: quotas[key] - counts[key])
        counts[key] += 1
    absent = set(weights) - present
    if absent:
        print(f"[sampler] absent populations={sorted(absent)}; redistributed template={counts}", flush=True)
    return counts


class FrozenBatchSampler:
    """GPU worker presentation order prepared locally for every epoch."""
    def __init__(self, epochs, *, expected_rows=None, batch_size=None):
        if not epochs:
            raise ValueError('local presentation plan has no training epochs')
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError('local presentation plan has invalid batch size')
        for batches in epochs:
            if not batches or any(not batch for batch in batches):
                raise ValueError('local presentation plan has an empty training epoch or batch')
            if batch_size is not None and any(len(batch) > batch_size for batch in batches):
                raise ValueError('local presentation batch exceeds configured batch size')
            indices = [index for batch in batches for index in batch]
            if any(not isinstance(index, (int, np.integer)) or isinstance(index, bool) or index < 0
                   for index in indices):
                raise ValueError('local presentation plan has invalid row indices')
            if len(set(indices)) != len(indices):
                raise ValueError('local presentation plan repeats training rows')
            if expected_rows is not None and sorted(indices) != list(range(expected_rows)):
                raise ValueError('local presentation plan must cover every training row exactly once')
        self.epochs = epochs
        self.epoch = 0
    def set_epoch(self, epoch):
        if epoch < 0 or epoch >= len(self.epochs):
            raise ValueError('training epoch absent from local presentation plan')
        self.epoch = epoch
    def __iter__(self):
        return iter(self.epochs[self.epoch])
    def __len__(self):
        return len(self.epochs[self.epoch])
