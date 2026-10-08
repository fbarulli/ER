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
from pathlib import Path
from typing import Any

from core.laya_config import FinetuneSmokeSpec
from core.manifest import sha256_file

# The three split names the finetune corpus contract names (train/dev/test).
SMOKE_SPLIT_SPECS = ("train", "dev", "test")
#: The corpus receipt member the finetune dataset payload requires.
SMOKE_RECEIPT = "receipt.json"


class FinetuneSmokeCorpus:
    """Cut the tiny smoke corpus from a committed source corpus (deterministic).

    ``source_dir`` is the SSOT corpus root (``laya.finetune_corpus_dir``);
    ``spec`` names the row caps and the destination. Every emitted split is the
    first ``<split>_rows`` lines of the source split, byte-for-byte, so the
    subset is reproducible and needs no sampler state. The receipt records the
    source digests and the measured counts, so the staged smoke dataset's
    ``files`` inventory is anchored to a real source.
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

    def _subset(self, split: str) -> tuple[Path, int]:
        """Write the first ``cap`` lines of one source split; return (path, n)."""
        source = self._source_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(
                f"smoke source split not found: {source} (build the corpus "
                "with scripts/laya_build_dataset.py)")
        cap = self._row_cap(split)
        kept = 0
        destination = self._dest_dir / f"{split}.jsonl"
        with source.open("r", encoding="utf-8") as reader, \
                destination.open("w", encoding="utf-8") as writer:
            for line in reader:
                if kept >= cap:
                    break
                if not line.strip():
                    continue
                writer.write(line if line.endswith("\n") else line + "\n")
                kept += 1
        if kept == 0:
            raise ValueError(f"smoke source split {source} yielded no rows")
        return destination, kept

    def build(self, *, seed: int) -> dict[str, Any]:
        """Materialize the subset + its receipt under the destination dir.

        ``seed`` is the training seed the receipt records (the lane passes
        ``FinetuneSpec.seed``, the SSOT), never a second literal here.
        """
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        files: dict[str, str] = {}
        for split in SMOKE_SPLIT_SPECS:
            path, kept = self._subset(split)
            counts[split] = kept
            files[path.name] = sha256_file(path)
        receipt = {
            "seed": int(seed),
            "smoke": True,
            "source_dir": str(self._source_dir),
            "source_sha256": {
                f"{split}.jsonl": sha256_file(self._source_dir / f"{split}.jsonl")
                for split in SMOKE_SPLIT_SPECS},
            "requested_rows": {
                split: self._row_cap(split) for split in SMOKE_SPLIT_SPECS},
            "counts": counts,
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
