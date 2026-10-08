"""Locally prepared fixed-text tokens consumed by native training and encoding."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import time
from collections.abc import Sequence

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from sentence_transformers.sentence_transformer.data_collator import SentenceTransformerDataCollator

from core.encoding_inputs import prepare_text_features, tokenization_policy
from core.perf_switches import perf_enabled, perf_int
from core.run_log import RunLogger
from core.timing import Timing

_LOG = RunLogger(__name__)

# Cache the bound native task contract once at lookup construction instead of
# re-introspecting the model on every preprocess call.
_CACHE_BOUND_CONTRACT = perf_enabled("text.bound_task_contract")

# Reuse the padded native batch for a repeated *ordered* fixed-text batch. The
# frozen batch sampler replays the exact same epochs, so steady-state epochs
# skip re-padding (and any re-tokenization). The key is the ordered text tuple,
# so option-order shuffles produce distinct entries; batches containing any
# dynamic (generated) text are never cached and keep their quota bookkeeping.
# Bounded FIFO: a shuffled (non-frozen) sampler would otherwise retain one
# padded batch per distinct batch ever seen.
_CACHE_PREPROCESSED = perf_enabled("text.cache_preprocessed_batches")
_PREPROCESSED_CACHE_SIZE = max(1, perf_int("text.preprocessed_cache_size", 8192))

TEXT_COLUMNS = {"sentence1", "sentence2", "anchor", "positive", "negative"}
COLLATOR_TEXT_COLUMNS = set(TEXT_COLUMNS) | {"label", "dataset_name"}


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


def _native_prompt_variants(model, policy) -> list[str]:
    """The distinct native prompts the checkpoint may ask for, canonical first."""
    return list(dict.fromkeys(["", policy.get("prompt") or "", *getattr(model, "prompts", {}).values()]))


def _raveled_attention_slice(mask) -> slice:
    """The slice of one row between its first and last attended position."""
    attended = torch.nonzero(mask, as_tuple=False).flatten()
    return slice(int(attended[0]), int(attended[-1]) + 1)


def _require_native_feature(key: str, value: torch.Tensor, shape: tuple) -> None:
    """2D integer-or-bool tensors shaped like the row batch, or reject loudly."""
    if value.shape != shape:
        raise ValueError(f"unsupported native training token feature: {key} {value.shape}")
    if value.dtype not in (torch.int64, torch.int32, torch.bool):
        raise ValueError(f"native token feature has unexpected dtype: {key} {value.dtype}")


def _cropped_row(tensors: dict, shape: tuple, index: int, positions: slice) -> dict:
    """One unpadded native row: every native feature cropped to its attention span."""
    row = {}
    for key, values in tensors.items():
        _require_native_feature(key, values, shape)
        row[key] = values[index, positions].cpu().numpy()
    return row


def _reconcile_constants(current, constants) -> dict:
    """Native non-tensor preprocessing constants must not drift between batches."""
    if constants is not None and current != constants:
        raise ValueError("native preprocessing constants vary across local token batches")
    return current


def _prepare_prompt_variant(
    model, prompt: str, texts: Sequence[str], policy: dict,
    *, batch_size: int, last_progress: float,
) -> tuple[dict, float]:
    """Tokenize `texts` under one native prompt, unpadded rows, one live bar.

    Returns the variant dict ``{"rows": ..., "constants": ...}`` and the
    carried progress clock (the throttled `[training tokens/local]` line is
    shared across prompt variants, as before).
    """
    rows: list[dict] = []
    constants: dict | None = None
    batches = _LOG.progress(
        range(0, len(texts), batch_size),
        desc="native_tokens", unit="batch",
        total=(len(texts) + batch_size - 1) // batch_size or 1,
    )
    for start in batches:
        features = prepare_text_features(model, texts[start:start + batch_size],
                                         policy={**policy, "prompt": prompt or None})
        tensors = {key: value for key, value in features.items() if isinstance(value, torch.Tensor)}
        current = {key: value for key, value in features.items() if key not in tensors}
        constants = _reconcile_constants(current, constants)
        shape = tensors["input_ids"].shape
        for index, mask in enumerate(tensors["attention_mask"]):
            positions = _raveled_attention_slice(mask)
            rows.append(_cropped_row(tensors, shape, index, positions))
        if time.monotonic() - last_progress >= 10 or start + batch_size >= len(texts):
            print(f"[training tokens/local] prompt={prompt!r} prepared={min(start + batch_size, len(texts)):,}/{len(texts):,} truncated=0", flush=True)
            last_progress = time.monotonic()
    return {"rows": rows, "constants": constants or {}}, last_progress


def prepare_training_tokens(model, payload, *, batch_size=256):
    """Store unpadded native features once per unique text and prompt."""
    timing = Timing('training.native_tokens')
    payload = list(payload)
    policy = checkpoint_policy(model)
    texts = list(dict.fromkeys(payload))
    prompts = _native_prompt_variants(model, policy)
    variants: dict[str, dict] = {}
    last_progress = time.monotonic()
    for position, prompt in enumerate(prompts):
        variants[prompt], last_progress = _prepare_prompt_variant(
            model, prompt, texts, policy, batch_size=batch_size, last_progress=last_progress,
        )
        timing.mark('prompt_' + str(position))
    return {"version": 1, "policy": policy, "texts": texts, "variants": variants,
            "task_contract": PreparedTaskContract.from_model(model).model_dump(),
            "payload_sha256": payload_sha256(payload)}


def payload_sha256(payload) -> str:
    """Stable digest over the frozen payload text list (write- and read-side)."""
    return hashlib.sha256(json.dumps(list(payload), ensure_ascii=False).encode()).hexdigest()


class PreparedTokenLookup:
    """No fixed-text tokenizer calls; explicitly registered dynamic text only."""

    def __init__(self, model, table, payload, *, payload_digest: str | None = None):
        self._require_policy(model, table)
        digest = payload_sha256(payload) if payload_digest is None else payload_digest
        self._require_payload(table, payload, digest)
        self.table = table
        validate_training_tokens(table)
        self.indices = {text: index for index, text in enumerate(table["texts"])}
        self._require_fixed_membership(payload)
        self.model = model
        self.task_contract = self._require_task_contract(model, table)
        self._bound_task_contract = self.task_contract
        self.original = model.preprocess
        self.generated = Counter()
        self._preprocessed_cache: dict[tuple, dict] = {}
        self.model.preprocess = self.preprocess

    def _require_policy(self, model, table) -> None:
        if table.get("version") != 1 or table["policy"] != checkpoint_policy(model):
            raise ValueError("prepared training tokenizer/checkpoint policy mismatch; rebuild locally")

    def _require_payload(self, table, payload, digest: str) -> None:
        if table["payload_sha256"] != digest:
            raise ValueError("prepared training token payload mismatch; rebuild locally")

    def _require_fixed_membership(self, payload) -> None:
        if any(text not in self.indices for text in payload):
            raise ValueError("prepared training tokens miss a fixed payload text")

    def _require_task_contract(self, model, table):
        contract = PreparedTaskContract.from_model(model)
        if "task_contract" in table and PreparedTaskContract.model_validate(table["task_contract"]) != contract:
            raise ValueError("prepared training task contract differs from the native input module")
        return contract

    def register_generated(self, text):
        if text not in self.indices:
            self.generated[text] += 1

    def _require_registered_dynamic(self, requested: Counter) -> None:
        if any(count > self.generated[text] for text, count in requested.items()):
            raise ValueError("fixed training/evaluation text missing local tokens; rebuild locally")

    def _consume_generated(self, requested: Counter) -> None:
        """Borrowed quota only balances back once the dynamic rows exist."""
        for text, count in requested.items():
            self.generated[text] -= count
            if not self.generated[text]:
                del self.generated[text]

    def _require_task_safe(self, args, task) -> None:
        """No positional args, no post-binding drift, only prepared task routes."""
        if args:
            raise ValueError("prepared training tokens do not support task routing; prepare that task explicitly")
        if task is not None:
            current = (
                self._bound_task_contract
                if _CACHE_BOUND_CONTRACT
                else PreparedTaskContract.from_model(self.model)
            )
            if current != self.task_contract:
                raise ValueError("native task preprocessing changed after prepared-token binding")
        self.task_contract.validate_task(task)

    def _variant_for(self, prompt: str) -> dict:
        """The prepared row table for one prompt, or an explicit rejection."""
        if prompt not in self.table["variants"]:
            raise ValueError(f"unprepared native training prompt: {prompt!r}")
        return self.table["variants"][prompt]

    def _require_matched_constants(self, features: dict, variant: dict) -> None:
        """Generated texts must preprocess to the generic non-tensor constants."""
        generated_constants = {name: value for name, value in features.items() if not isinstance(value, torch.Tensor)}
        if generated_constants != variant["constants"]:
            raise ValueError("native generated/fixed preprocessing constants mismatch")

    def _dynamic_rows(self, features: dict, dynamic_positions: list) -> dict:
        """Input position -> unpadded native row for every dynamic text."""
        positions_by_row = [_raveled_attention_slice(mask) for mask in features["attention_mask"]]
        return {
            position: {
                name: value[dynamic_index, positions_by_row[dynamic_index]].cpu().numpy()
                for name, value in features.items() if isinstance(value, torch.Tensor)
            }
            for dynamic_index, position in enumerate(dynamic_positions)
        }

    def _require_same_feature_keys(self, rows) -> set:
        keys = set(rows[0])
        if any(set(row) != keys for row in rows):
            raise ValueError("native generated/fixed token feature mismatch")
        return keys

    def _padded_rows(self, rows, keys: set, constants: dict) -> dict:
        """Pad the (possibly mixed fixed/dynamic) rows into one batch."""
        width = max(len(row["input_ids"]) for row in rows)
        if width > self.table["policy"]["input_token_limit"]:
            raise ValueError("zero truncation required: prepared token input exceeds checkpoint limit")
        result = dict(constants)
        for name in keys:
            pad = self.model.tokenizer.pad_token_id if name == "input_ids" else 0
            array = np.full((len(rows), width), pad, dtype=np.int64)
            for index, row in enumerate(rows):
                length = len(row[name])
                offset = width - length if self.table["policy"]["padding_side"] == "left" else 0
                array[index, offset:offset + length] = row[name]
            result[name] = torch.from_numpy(array)
        return result

    def preprocess(self, inputs, *args, prompt=None, task=None, **kwargs):
        self._require_task_safe(args, task)
        prompt = prompt or ""
        variant = self._variant_for(prompt)
        if not inputs:
            return self.original(inputs, prompt=prompt, **kwargs)
        dynamic_positions = [index for index, text in enumerate(inputs) if text not in self.indices]
        if _CACHE_PREPROCESSED and not dynamic_positions:
            key = (prompt, tuple(inputs))
            cached = self._preprocessed_cache.get(key)
            if cached is not None:
                # Hand out a fresh dict: callers may add/pop batch keys, and the
                # cached entry must not be mutated by any of them.
                return dict(cached)
            rows = [variant["rows"][self.indices[text]] for text in inputs]
            result = self._padded_rows(
                rows, self._require_same_feature_keys(rows), variant["constants"]
            )
            if len(self._preprocessed_cache) >= _PREPROCESSED_CACHE_SIZE:
                self._preprocessed_cache.pop(next(iter(self._preprocessed_cache)))
            self._preprocessed_cache[key] = result
            return dict(result)
        requested = Counter(inputs[index] for index in dynamic_positions)
        self._require_registered_dynamic(requested)
        dynamic_rows: dict[int, dict] = {}
        if dynamic_positions:
            generated_texts = [inputs[index] for index in dynamic_positions]
            features = self.original(generated_texts, prompt=prompt, **kwargs)
            self._require_matched_constants(features, variant)
            dynamic_rows = self._dynamic_rows(features, dynamic_positions)
            self._consume_generated(requested)
        rows = [variant["rows"][self.indices[text]] if text in self.indices else dynamic_rows[index]
                for index, text in enumerate(inputs)]
        if not rows:
            return self.original(inputs, prompt=prompt, **kwargs)
        return self._padded_rows(rows, self._require_same_feature_keys(rows), variant["constants"])


def validate_training_tokens(table):
    texts = table.get("texts", [])
    if len(set(texts)) != len(texts) or not texts:
        raise ValueError("prepared training tokens need unique nonempty text rows")
    if "" not in table.get("variants", {}):
        raise ValueError("prepared training tokens miss the native unprompted variant")
    limit = table["policy"]["input_token_limit"]
    for variant in table["variants"].values():
        _validate_variant(variant, texts, limit)


def _require_row_value(key: str, value, width: int) -> None:
    """One feature vector: array, 1D, right length, integer dtype, valid IDs."""
    if not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) != width:
        raise ValueError("prepared token feature shapes disagree")
    if value.dtype.kind not in "iu" and not (key == "attention_mask" and value.dtype.kind == "b"):
        raise ValueError("prepared training token feature must have integer dtype")
    if key == "input_ids" and np.any(value < 0):
        raise ValueError("prepared token IDs must be nonnegative")


def _require_row_features(row: dict, expected_keys: set, limit: float) -> None:
    """One row's features agree on keys, stay inside the zero-truncation limit."""
    if set(row) != expected_keys or not {"input_ids", "attention_mask"} <= set(row):
        raise ValueError("prepared native token feature keys disagree")
    width = len(row["input_ids"])
    if width < 1 or width > limit:
        raise ValueError("zero truncation required: invalid prepared training token length")
    for key, value in row.items():
        _require_row_value(key, value, width)
    mask = row["attention_mask"]
    if not np.isin(mask, [0, 1]).all() or mask[0] != 1 or mask[-1] != 1:
        raise ValueError("prepared ragged token rows need binary masks and no outer padding")


def _validate_variant(variant: dict, texts: list, limit: float) -> None:
    """One prompt's rows: exact text alignment plus per-row feature checks."""
    rows = variant["rows"]
    if len(rows) != len(texts):
        raise ValueError("prepared training token/text row counts disagree")
    expected_keys = set(rows[0])
    for row in rows:
        _require_row_features(row, expected_keys, limit)


class ObjectiveDataCollator(SentenceTransformerDataCollator):
    """Keep telemetry and structured features out of tokenization."""

    def __call__(self, features):
        text_features = [
            {key: value for key, value in row.items() if key in COLLATOR_TEXT_COLUMNS}
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
