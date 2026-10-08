"""A2: the prepared-token batch cache is order-aware, bounded and private.

The text trainer's native ``preprocess`` is called once per column per step. The
cache reuses the padded native batch for a repeated *ordered* fixed-text batch;
these tests pin the three properties that make that safe: an option-order change
is a distinct entry (never a stale reused batch), batches containing generated
text are never cached (their quota accounting still runs), and the cached entry
is never handed out by reference.
"""
import json
from types import SimpleNamespace

import torch

import training.token_inputs as token_inputs
from training.token_inputs import PreparedTokenLookup, prepare_training_tokens


class Encoder:
    """Minimal native-tokenizer stand-in (same shape as the trainer's model)."""

    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(
            backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps({'vocab': 'native'})),
            special_tokens_map={'pad_token': '[PAD]'}, padding_side='right', pad_token_id=0)
        self.module = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=8), preprocess=self.preprocess,
            get_config_dict=lambda: {'do_lower_case': False, 'max_seq_length': 4})
        self.truncate_dim = None
        self.default_prompt_name = 'query'
        self.prompts = {'query': 'prefix '}

    def __getitem__(self, index):
        return self.module

    def preprocess(self, texts, prompt=None, processing_kwargs=None, **kwargs):
        self.calls.append((list(texts), prompt))
        rows = [[101] + [10 + len(word) for word in ((prompt or '') + text).split()] + [102]
                for text in texts]
        width = max(map(len, rows))
        return {'input_ids': torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                'attention_mask': torch.tensor([[1] * len(row) + [0] * (width - len(row))
                                                for row in rows]),
                'prompt_length': 1 if prompt else 0, 'modality': 'text'}


PAYLOAD = ['one', 'one two', 'two']


def _lookup(*, cache=True, monkeypatch, batch_size=2):
    monkeypatch.setattr(token_inputs, '_CACHE_PREPROCESSED', cache)
    model = Encoder()
    table = prepare_training_tokens(model, PAYLOAD, batch_size=batch_size)
    lookup = PreparedTokenLookup(model, table, PAYLOAD)
    model.calls.clear()
    return model, lookup


def test_repeated_ordered_batch_is_reused_without_re_tokenizing(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    first = model.preprocess(['one two', 'one'], prompt='prefix ')
    assert len(lookup._preprocessed_cache) == 1
    second = model.preprocess(['one two', 'one'], prompt='prefix ')
    assert model.calls == []
    torch.testing.assert_close(first['input_ids'], second['input_ids'])
    torch.testing.assert_close(first['attention_mask'], second['attention_mask'])
    assert second is not first     # the caller gets its own dict


def test_cached_batch_is_bit_identical_to_the_uncached_batch(monkeypatch):
    for order in (['one two', 'one'], ['one', 'one two'], ['two', 'one two']):
        cached_model, _ = _lookup(cache=True, monkeypatch=monkeypatch)
        cached = cached_model.preprocess(order, prompt='prefix ')
        fresh_model, _ = _lookup(cache=False, monkeypatch=monkeypatch)
        fresh = fresh_model.preprocess(order, prompt='prefix ')
        for key, value in fresh.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(cached[key], value), (order, key)
            else:
                assert cached[key] == value, (order, key)


def test_option_order_is_a_distinct_entry_never_a_stale_reuse(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    direct = model.preprocess(['one two', 'one'], prompt='prefix ')
    swapped = model.preprocess(['one', 'one two'], prompt='prefix ')
    assert len(lookup._preprocessed_cache) == 2
    assert torch.equal(swapped['input_ids'][0], direct['input_ids'][1])
    assert torch.equal(swapped['input_ids'][1], direct['input_ids'][0])
    assert torch.equal(swapped['attention_mask'][0], direct['attention_mask'][1])


def test_prompt_variants_do_not_share_cache_entries(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    model.preprocess(['one two', 'one'], prompt='prefix ')
    model.preprocess(['one two', 'one'], prompt='')
    assert len(lookup._preprocessed_cache) == 2
    assert {key[0] for key in lookup._preprocessed_cache} == {'prefix ', ''}


def test_mutating_a_returned_batch_does_not_poison_the_cache(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    first = model.preprocess(['one two', 'one'], prompt='prefix ')
    first.pop('modality', None)
    first['injected'] = 'by-the-caller'
    third = model.preprocess(['one two', 'one'], prompt='prefix ')
    assert 'injected' not in third
    assert third['modality'] == 'text'


def test_generated_text_is_never_cached_and_still_consumes_its_quota(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    lookup.register_generated('fresh')
    model.calls.clear()
    batch = model.preprocess(['one', 'fresh'], prompt='prefix ')
    assert lookup._preprocessed_cache == {}
    assert model.calls == [(['fresh'], 'prefix ')]
    assert not lookup.generated          # quota subtracted, entry dropped


def test_cache_respects_the_legacy_switch(monkeypatch):
    model, lookup = _lookup(cache=False, monkeypatch=monkeypatch)
    model.preprocess(['one two', 'one'], prompt='prefix ')
    model.preprocess(['one two', 'one'], prompt='prefix ')
    assert lookup._preprocessed_cache == {}
    assert model.calls == []             # prepared rows still avoid tokenizing


def test_cache_is_bounded_and_evicts_in_insertion_order(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    monkeypatch.setattr(token_inputs, '_PREPROCESSED_CACHE_SIZE', 2)
    for order in (['one', 'one two'], ['one two', 'two'], ['two', 'one']):
        model.preprocess(order, prompt='prefix ')
    assert len(lookup._preprocessed_cache) == 2
    assert ('prefix ', ('one', 'one two')) not in lookup._preprocessed_cache
    assert ('prefix ', ('two', 'one')) in lookup._preprocessed_cache


def test_cache_size_knob_of_one_keeps_exactly_one_entry(monkeypatch):
    # The knob is clamped to >= 1 in the source, so eviction can never pop an
    # empty dict when an operator sets it to zero.
    model, lookup = _lookup(monkeypatch=monkeypatch)
    monkeypatch.setattr(token_inputs, '_PREPROCESSED_CACHE_SIZE', 1)
    for order in (['one', 'one two'], ['one two', 'two']):
        model.preprocess(order, prompt='prefix ')
    assert len(lookup._preprocessed_cache) == 1


def test_prepared_rows_never_touch_the_native_tokenizer(monkeypatch):
    model, lookup = _lookup(monkeypatch=monkeypatch)
    for order in (['one', 'one two'], ['one two', 'one']):
        batch = model.preprocess(order, prompt='prefix ')
        assert set(batch) >= {'input_ids', 'attention_mask'}
    assert model.calls == []
    assert all(row['input_ids'].dtype == torch.long for row in [batch])
