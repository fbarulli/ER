"""Locally prepared fixed-text tokens consumed by native training and encoding."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import time
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from core.encoding_inputs import prepare_text_features, tokenization_policy

TEXT_COLUMNS = {"sentence1", "sentence2", "anchor", "positive", "negative"}


class PreparedTaskContract(BaseModel):
    """Tasks that preserve the native features in a frozen token table."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    input_module: str
    task_independent: bool

    @classmethod
    def from_model(cls, model):
        from sentence_transformers.models import Transformer
        module = model[0]
        neutral = (type(module) is Transformer
                   and getattr(module, "transformer_task", None) == "feature-extraction"
                   and set(getattr(module, "modality_config", {})) == {"text"}
                   and all(getattr(module, name, None) is None for name in
                           ("query_length", "document_length", "query_expansion")))
        return cls(input_module=type(module).__module__ + "." + type(module).__qualname__,
                   task_independent=neutral)

    def validate_task(self, task):
        if task is not None and (not self.task_independent or task not in {"query", "document"}):
            raise ValueError("prepared training tokens do not support task routing; prepare that task explicitly")


def model_card_text_dataset(dataset):
    """Report model inputs without tokenizing population or routing telemetry."""
    if dataset is None:
        return None
    if isinstance(dataset, dict):
        return type(dataset)({name: model_card_text_dataset(value) for name, value in dataset.items()})
    columns = [name for name in dataset.column_names if name in TEXT_COLUMNS or name == "label"]
    return dataset.select_columns(columns)


def checkpoint_policy(model):
    return tokenization_policy(model)


def prepare_training_tokens(model, payload, *, batch_size=256):
    """Store unpadded native features once per unique text and prompt."""
    payload = list(payload)
    policy = checkpoint_policy(model)
    texts = list(dict.fromkeys(payload))
    prompts = list(dict.fromkeys(["", policy.get("prompt") or "", *getattr(model, "prompts", {}).values()]))
    variants = {}
    last_progress = time.monotonic()
    for prompt in prompts:
        rows = []
        constants = None
        for start in range(0, len(texts), batch_size):
            features = prepare_text_features(model, texts[start:start + batch_size],
                                             policy={**policy, "prompt": prompt or None})
            tensors = {key: value for key, value in features.items() if isinstance(value, torch.Tensor)}
            current = {key: value for key, value in features.items() if key not in tensors}
            if constants is not None and current != constants:
                raise ValueError("native preprocessing constants vary across local token batches")
            constants = current
            for index, mask in enumerate(tensors["attention_mask"]):
                attended = torch.nonzero(mask, as_tuple=False).flatten()
                positions = slice(int(attended[0]), int(attended[-1]) + 1)
                row = {}
                for key, values in tensors.items():
                    if values.ndim != 2 or values.shape != tensors["input_ids"].shape:
                        raise ValueError(f"unsupported native training token feature: {key} {values.shape}")
                    if values.dtype not in (torch.int64, torch.int32, torch.bool):
                        raise ValueError(f"native token feature has unexpected dtype: {key} {values.dtype}")
                    row[key] = values[index, positions].cpu().numpy()
                rows.append(row)
            if time.monotonic() - last_progress >= 10 or start + batch_size >= len(texts):
                print(f"[training tokens/local] prompt={prompt!r} prepared={min(start + batch_size, len(texts)):,}/{len(texts):,} truncated=0", flush=True)
                last_progress = time.monotonic()
        variants[prompt] = {"rows": rows, "constants": constants or {}}
    return {"version": 1, "policy": policy, "texts": texts, "variants": variants,
            "task_contract": PreparedTaskContract.from_model(model).model_dump(),
            "payload_sha256": hashlib.sha256(json.dumps(list(payload), ensure_ascii=False).encode()).hexdigest()}


class PreparedTokenLookup:
    """No fixed-text tokenizer calls; explicitly registered dynamic text only."""
    def __init__(self, model, table, payload):
        if table.get("version") != 1 or table["policy"] != checkpoint_policy(model):
            raise ValueError("prepared training tokenizer/checkpoint policy mismatch; rebuild locally")
        digest = hashlib.sha256(json.dumps(list(payload), ensure_ascii=False).encode()).hexdigest()
        if table["payload_sha256"] != digest:
            raise ValueError("prepared training token payload mismatch; rebuild locally")
        self.table = table
        validate_training_tokens(table)
        self.indices = {text: index for index, text in enumerate(table["texts"])}
        if any(text not in self.indices for text in payload):
            raise ValueError("prepared training tokens miss a fixed payload text")
        self.model = model
        self.task_contract = PreparedTaskContract.from_model(model)
        if "task_contract" in table and PreparedTaskContract.model_validate(table["task_contract"]) != self.task_contract:
            raise ValueError("prepared training task contract differs from the native input module")
        self.original = model.preprocess
        self.generated = Counter()
        self.model.preprocess = self.preprocess

    def register_generated(self, text):
        if text not in self.indices:
            self.generated[text] += 1

    def preprocess(self, inputs, *args, prompt=None, task=None, **kwargs):
        if args:
            raise ValueError("prepared training tokens do not support task routing; prepare that task explicitly")
        if task is not None and PreparedTaskContract.from_model(self.model) != self.task_contract:
            raise ValueError("native task preprocessing changed after prepared-token binding")
        self.task_contract.validate_task(task)
        prompt = prompt or ""
        if prompt not in self.table["variants"]:
            raise ValueError(f"unprepared native training prompt: {prompt!r}")
        variant = self.table["variants"][prompt]
        dynamic_positions = [index for index, text in enumerate(inputs) if text not in self.indices]
        requested = Counter(inputs[index] for index in dynamic_positions)
        if any(count > self.generated[text] for text, count in requested.items()):
            raise ValueError("fixed training/evaluation text missing local tokens; rebuild locally")
        dynamic_rows = {}
        if dynamic_positions:
            generated_texts = [inputs[index] for index in dynamic_positions]
            features = self.original(generated_texts, prompt=prompt, **kwargs)
            generated_constants = {name: value for name, value in features.items() if not isinstance(value, torch.Tensor)}
            if generated_constants != variant["constants"]:
                raise ValueError("native generated/fixed preprocessing constants mismatch")
            positions_by_row = []
            for mask in features["attention_mask"]:
                attended = torch.nonzero(mask, as_tuple=False).flatten()
                positions_by_row.append(slice(int(attended[0]), int(attended[-1]) + 1))
            for dynamic_index, position in enumerate(dynamic_positions):
                dynamic_rows[position] = {
                    name: value[dynamic_index, positions_by_row[dynamic_index]].cpu().numpy()
                    for name, value in features.items() if isinstance(value, torch.Tensor)
                }
            for text, count in requested.items():
                self.generated[text] -= count
                if not self.generated[text]:
                    del self.generated[text]
        rows = [variant["rows"][self.indices[text]] if text in self.indices else dynamic_rows[index]
                for index, text in enumerate(inputs)]
        if not rows:
            return self.original(inputs, prompt=prompt, **kwargs)
        keys = set(rows[0])
        if any(set(row) != keys for row in rows):
            raise ValueError("native generated/fixed token feature mismatch")
        width = max(len(row["input_ids"]) for row in rows)
        if width > self.table["policy"]["input_token_limit"]:
            raise ValueError("zero truncation required: prepared token input exceeds checkpoint limit")
        result = dict(variant["constants"])
        for name in keys:
            pad = self.model.tokenizer.pad_token_id if name == "input_ids" else 0
            array = np.full((len(rows), width), pad, dtype=np.int64)
            for index, row in enumerate(rows):
                length = len(row[name])
                offset = width - length if self.table["policy"]["padding_side"] == "left" else 0
                array[index, offset:offset + length] = row[name]
            result[name] = torch.from_numpy(array)
        return result


def validate_training_tokens(table):
    texts = table.get("texts", [])
    if len(set(texts)) != len(texts) or not texts:
        raise ValueError("prepared training tokens need unique nonempty text rows")
    if "" not in table.get("variants", {}):
        raise ValueError("prepared training tokens miss the native unprompted variant")
    limit = table["policy"]["input_token_limit"]
    for variant in table["variants"].values():
        rows = variant["rows"]
        if len(rows) != len(texts):
            raise ValueError("prepared training token/text row counts disagree")
        expected_keys = set(rows[0])
        for row in rows:
            if set(row) != expected_keys or not {"input_ids", "attention_mask"} <= set(row):
                raise ValueError("prepared native token feature keys disagree")
            width = len(row["input_ids"])
            if width < 1 or width > limit:
                raise ValueError("zero truncation required: invalid prepared training token length")
            for key, value in row.items():
                if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != width:
                    raise ValueError("prepared token feature shapes disagree")
                if value.dtype.kind not in "iu" and not (key == "attention_mask" and value.dtype.kind == "b"):
                    raise ValueError("prepared training token feature must have integer dtype")
                if key == "input_ids" and np.any(value < 0):
                    raise ValueError("prepared token IDs must be nonnegative")
            mask = row["attention_mask"]
            if not np.isin(mask, [0, 1]).all() or mask[0] != 1 or mask[-1] != 1:
                raise ValueError("prepared ragged token rows need binary masks and no outer padding")


from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator

class ObjectiveDataCollator(SentenceTransformerDataCollator):
    """Keep telemetry and structured features out of tokenization."""

    def __call__(self, features):
        text_features = [
            {
                key: value
                for key, value in row.items()
                if key in {"sentence1", "sentence2", "anchor", "positive", "negative", "label", "dataset_name"}
            }
            for row in features
        ]
        batch = super().__call__(text_features)
        # Training rows carry pair_id; evaluator rows do not.
        # Detect the field structurally rather than treating a
        # list of optional values as a valid batch. If a training
        # row is malformed, direct indexing raises loudly.
        for name in ("pair_id", "structured_features"):
            present = [name in row for row in features]
            if any(present) and not all(present):
                raise ValueError(f"objective batch has inconsistent {name} metadata")
        if features and "pair_id" in features[0]:
            batch["pair_id"] = torch.tensor(
                [row["pair_id"] for row in features], dtype=torch.long
            )
        if features and "structured_features" in features[0]:
            batch["structured_features"] = torch.tensor(
                [row["structured_features"] for row in features],
                dtype=torch.float32,
            )
        return batch
