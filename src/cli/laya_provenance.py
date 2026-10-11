"""Read structural input provenance from current and historical lane receipts."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from core.eval_trace import ProvenanceValue


class LayaCorpusProvenance(BaseModel):
    corpus_size: ProvenanceValue | dict[str, ProvenanceValue] | None = None
    eval_jsonl_size: ProvenanceValue | None = None
    eval_data_size: ProvenanceValue | None = None

    @classmethod
    def build(cls, receipt: dict[str, Any]) -> ProvenanceValue:
        from cli.laya_lane import FINETUNE_CORPUS_FILES

        record = cls.model_validate(receipt)
        corpus = record.corpus_size
        values = ([corpus[name] for name in FINETUNE_CORPUS_FILES if name in corpus]
                  if isinstance(corpus, dict) else [corpus])
        for value in (*values, record.eval_jsonl_size, record.eval_data_size):
            if value is not None:
                return value
        raise KeyError("receipt carries no corpus digest (corpus_size / "
                       "eval_jsonl_size / eval_data_size)")
