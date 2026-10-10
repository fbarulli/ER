"""src/cli/laya_smoke.py — the tiny CPU end-to-end smoke corpus for the finetune kernel.

The finetune kernel's new training surface (dials + profiler + early-stop /
dev-eval + the fail-loud Kaggle fetchers) is only validated end-to-end when the
REAL kernel actually trains and evaluates. This module owns the one thing that
validation needs and nothing else: cutting a small, deterministic subset of the
committed corpus while carrying a receipt that records exactly what the subset
is (source digest, row counts, seed), so the smoke's input is honest and
reproducible.

The selection lives in :class:`core.laya_config.FinetuneSmokeSpec`; the staging
+ push reuse the lane's production functions (``stage_finetune_kernel`` /
``push_kaggle_kernel``). This module never stages or pushes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from core.laya_config import FinetuneSmokeSpec
from core.manifest import sha256_file

# The three split names the finetune corpus contract names (train/dev/test).
SMOKE_SPLIT_SPECS = ("train", "dev", "test")
#: The corpus receipt member the finetune dataset payload requires.
SMOKE_RECEIPT = "receipt.json"
#: The corpus row tag the subset is stratified by (the builder's traceability
#: tag; a row without it falls into the explicit "unspecified" stratum).
SMOKE_STRATUM_FIELD = "difficulty_slice"
SMOKE_UNSPECIFIED_STRATUM = "unspecified"


class FinetuneSmokeCorpus:
    """Cut the tiny STRATIFIED smoke corpus from a committed source corpus.

    ``source_dir`` is the SSOT corpus root (``laya.finetune_corpus_dir``);
    ``spec`` names the row caps and the destination. Each emitted split is a
    deterministic, proportional sample of the WHOLE source split, stratified by
    ``difficulty_slice`` — not its first N lines. A prefix cut is biased: the
    builder emits the state/mask rows LAST, so the first N train lines are all
    front-of-file identity pairs and the smoke never sees a ``single_state``
    (package_state) example. Sampling the whole split (RNG seeded by the
    training seed) keeps every stratum present in proportion, so the tiny test
    exercises the same question mix as the full corpus. The receipt records the
    source digests, the measured counts AND the per-stratum census, so the
    staged smoke dataset's ``files`` inventory is anchored to a real source.
    """

    def __init__(self, spec: FinetuneSmokeSpec, *, source_dir: Path,
                 dest_dir: Path) -> None:
        self._spec = spec
        self._source_dir = Path(source_dir)
        self._dest_dir = Path(dest_dir)

    def _row_cap(self, split: str) -> int:
        return {
            "train": self._spec.train_rows,
            "dev": self._spec.dev_rows,
            "test": self._spec.test_rows,
        }[split]

    def _stratum(self, line: str) -> str:
        """A source row's stratification tag (its ``difficulty_slice``)."""
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"smoke source row is not JSON: {line[:80]!r}") from error
        return str(row.get(SMOKE_STRATUM_FIELD) or SMOKE_UNSPECIFIED_STRATUM)

    def _quota(self, sizes: dict[str, int], target: int) -> dict[str, int]:
        """Proportional (largest-remainder) allocation of ``target`` by size.

        Deterministic; a stratum is never asked for more rows than it has, and
        a shortfall is offered to the remaining strata in largest-fraction
        order, so the allocation sums to ``min(target, sum(sizes))``.
        """
        total = sum(sizes.values())
        target = max(0, min(target, total))
        if total == 0:
            return {key: 0 for key in sizes}
        shares = {key: target * size / total for key, size in sizes.items()}
        quota = {key: min(sizes[key], int(shares[key])) for key in sizes}
        remaining = target - sum(quota.values())
        for key in sorted(sizes, key=lambda k: (-(shares[k] - quota[k]), k)):
            if remaining <= 0:
                break
            take = min(sizes[key] - quota[key], remaining)
            quota[key] += take
            remaining -= take
        return quota

    def _sample(self, lines: list[str], cap: int,
                seed: int) -> tuple[list[int], dict[str, int]]:
        """Indices of the stratified sample + its per-stratum census."""
        strata: dict[str, list[int]] = {}
        for index, line in enumerate(lines):
            strata.setdefault(self._stratum(line), []).append(index)
        quota = self._quota(
            {key: len(members) for key, members in strata.items()}, cap)
        rng = random.Random(seed)
        chosen: list[int] = []
        census: dict[str, int] = {}
        for key in sorted(strata):
            take = quota.get(key, 0)
            if take:
                chosen.extend(rng.sample(strata[key], take))
                census[key] = take
        return sorted(chosen), census

    def _subset(self, split: str,
                seed: int) -> tuple[Path, int, dict[str, int]]:
        """Write one split's stratified subset; return (path, kept, census)."""
        source = self._source_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(
                f"smoke source split not found: {source} (build the corpus "
                "with scripts/laya_build_dataset.py)")
        lines = [line if line.endswith("\n") else line + "\n"
                 for line in source.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not lines:
            raise ValueError(f"smoke source split {source} yielded no rows")
        chosen, census = self._sample(lines, self._row_cap(split), seed)
        destination = self._dest_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as writer:
            for index in chosen:
                writer.write(lines[index])
        return destination, len(chosen), census

    def build(self, *, seed: int) -> dict[str, Any]:
        """Materialize the stratified subset + its receipt.

        ``seed`` is the training seed the receipt records AND the sampler's RNG
        (the lane passes ``FinetuneSpec.seed``, the SSOT), never a second
        literal here.
        """
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        strata: dict[str, dict[str, int]] = {}
        files: dict[str, str] = {}
        for split in SMOKE_SPLIT_SPECS:
            path, kept, census = self._subset(split, seed)
            counts[split] = kept
            strata[split] = census
            files[path.name] = sha256_file(path)
        receipt = {
            "seed": int(seed),
            "smoke": True,
            "sampling": f"stratified:{SMOKE_STRATUM_FIELD}",
            "source_dir": str(self._source_dir),
            "source_sha256": {
                f"{split}.jsonl": sha256_file(self._source_dir / f"{split}.jsonl")
                for split in SMOKE_SPLIT_SPECS},
            "requested_rows": {
                split: self._row_cap(split) for split in SMOKE_SPLIT_SPECS},
            "counts": counts,
            "strata": strata,
            "epochs": self._spec.epochs,
            "micro_batch": self._spec.micro_batch,
            "grad_accum": self._spec.grad_accum,
        }
        destination = self._dest_dir / SMOKE_RECEIPT
        destination.write_text(json.dumps(receipt, indent=2) + "\n",
                               encoding="utf-8")
        files[destination.name] = sha256_file(destination)
        receipt["files"] = files
        return receipt
