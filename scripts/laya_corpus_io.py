"""scripts/laya_corpus_io.py — thin file-format boundaries for the corpus.

CSV reads, sha256 digests, JSONL serialization and the prepared text bundle
(gzip pickle) are the corpus builder's only I/O shapes. They live here so the
rule/case/growth/builder layers stay free of format details and no two modules
re-implement a reader.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


class CsvSource:
    """Read a CSV into (header, rows-of-dicts)."""

    @staticmethod
    def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)


class Digest:
    """Content digests for the receipt and determinism pins."""

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class JsonLine:
    """One corpus case -> one JSON line (UTF-8, never ASCII-escaped)."""

    @staticmethod
    def dump(record: dict[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False)


class PreparedBundle:
    """The prepared text bundle: a gzip pickle of plain dict/ndarray/DataFrame
    values (no custom classes)."""

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        with gzip.open(Path(path), "rb") as handle:
            return pickle.load(handle)
