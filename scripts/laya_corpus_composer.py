"""scripts/laya_corpus_composer.py — the reused pair-state composer boundary.

`scripts/laya_metrics_pairs.py` owns the six-field side literals and the
side-by-side state string the metrics lane emits. The corpus builder must be
byte-identical to that composition, so it REUSES that composer by file path
(never re-implements it). This module is the one place the dynamic import
lives; every corpus module imports the composed symbols from here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


class PairsComposerLoader:
    """Loads scripts/laya_metrics_pairs.py and exposes its composer verbatim."""

    MODULE_NAME = "laya_metrics_pairs"
    SOURCE_FILENAME = "laya_metrics_pairs.py"

    @classmethod
    def load(cls) -> ModuleType:
        path = Path(__file__).resolve().parent / cls.SOURCE_FILENAME
        spec = importlib.util.spec_from_file_location(cls.MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pair composer from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_PAIRS_BUILDER: ModuleType = PairsComposerLoader.load()
compose_side = _PAIRS_BUILDER.compose_side
compose_state = _PAIRS_BUILDER.compose_state
PAIR_FIELDS: tuple[str, ...] = _PAIRS_BUILDER.SLICE_FIELDS
