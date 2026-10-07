"""Byte-equivalence dump for the reg2-lane-A owned text functions.

Runs the same corpus through whichever ``src`` tree PYTHONPATH points at and
writes one JSON document with every function's output.  Two trees are then
compared with ``diff``: an identical document proves the optimization changed
no returned byte on the corpus, which is far stronger evidence than a timing
run (and it covers branches the fixture never reaches).

    PYTHONPATH=/tmp/opc/reg2-A-pristine/src python equivalence_dump.py --out pristine.json
    PYTHONPATH=/tmp/opc/ER-reg2-A/src        python equivalence_dump.py --out optimized.json
    diff pristine.json optimized.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
CATALOG = ROOT / 'artifacts' / 'abl_opt' / 'baseline' / 'eligible_catalog.csv'

os.environ.setdefault('EUROMONITOR_PROJECT_ROOT', str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def stable(value):
    if isinstance(value, Mapping):
        return {str(key): stable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [stable(item) for item in value]
        return ['__set__', sorted(items, key=lambda item: json.dumps(item, sort_keys=True))]
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


# Synthetic edge cases the fixture corpus does not contain: combining marks,
# NFKD-only decompositions, lookaround-adjacent word boundaries, fraction
# notation, EU decimals, glued product codes, every unit spelling.
EDGE_STRINGS = [
    '', '   ', None, 0, 3.5, float('nan'), True,
    'Café', 'CAFÉ ÜNÏCODE', 'Ǻǽ', 'ﬁne ﬂavour', '½ ⅓ ¾', 'Ⅻ',
    'привет', 'ΚΑΛΗΜΈΡΑ', '日本語', 'Ａｕｒａ ２４',
    'Voilà', 'Reál Réal Brämhults', 'a\u0301\u0302\u0303b', 'e\u0300\u0301',
    'x\u00a0\u2028\u2029y', 'tab\there\nnewline\r\n', 'nul\x00byte',
    '100% ÀÇ', '0-2%', '12,5 l', '1 000 ml', '0 8l', '8 12 fl. oz.',
    '24, 500ml', '24 500ml', 'case of 12', '6x1.5l', '3 x 2 l',
    'pack of 6', '12 Pack', '48 pk', '10 Packets', '2/3 oz', '24 / 2oz',
    '1 3/4 cup', '1/2 l', 'per 100 ml', 'per serving', 'makes 8 l',
    'contains 10% juice in concentrate', 'total of 500 ml',
    '0.33l', '.14 oz', '1.000 ml', '500.0 ml', 'BG14980 L', 'chinotto1 l',
    'burst850ml', 'makes 128 gal', '300 cc', '1 dl', '2 qt', '1 gal',
    'Dry Mix powder tea bags', 'per 1 g', 'kcal per serving',
    'Grape White Grape Black', 'apple juice gala braeburn russet golden delicious cox',
    'protein boost bcaa collagen glow c mix multi v', 'battery black rise pearberry',
    'cola kola soda birch beer energy drink regular original light lite diet',
    'turnip ginger ale spicy hot plain simple', 'zero caffeine decaf caffeine free',
    'juice content: 100%', 'Juice Content: 0-2 %', 'Juice Content: 40 - 60%',
    'water slight lightly light medium strong strongly',
    'cream soda dr. bob dr bob lemon tea chai concentrate gingerbread syrup',
    'arishta guduchi', 'not as hot',
    'orange flavoured drink', 'flavour of mango', 'taste of  lemon',
    'Description: Kiwi Flavored', 'x' * 300, '  spaced   out  ',
]

EDGE_ATTRIBUTES = [
    '', 'Brand: Coca Cola; Volume: 1.5 l; Count per Unit: 6',
    'Flavor: Orange; Juice Content: 100%;', '  ; ; no colon here ;',
    'Key:;Empty: ', 'Case: 12 x 330ml; Pack Type: Can',
    'Flavour: Mango; Volume: 0,33 l', 'TYPE: SPARKLING; Sugar: 0g',
]


def _call(function, *args, **kwargs):
    """Call and record the result, or the exception TYPE (also compared).

    Several of these helpers only accept ``str``; the corpus deliberately
    includes non-strings so the two trees are compared on the failure branch
    as well.  Exceptions are recorded by type name, never swallowed.
    """
    try:
        return stable(function(*args, **kwargs))
    except Exception as error:  # noqa: BLE001 — the type IS the compared value
        return ['__exc__', type(error).__name__]


def collect() -> dict:
    import pandas as pd

    from core.text import (
        attribute_fields,
        extract_pack_counts,
        extract_volume_evidence,
        extract_volume_match,
        extract_volume_measurement,
        extract_volume_ml,
        bucket_ml,
        norm_unit,
        normalize_retailer,
        normalize_text,
        normalized_attribute_text,
        unicode_casefold,
        _volume_entry,
        _volume_spelling_index,
        _VOLUME_SEARCH,
        extract_volume_evidence as _eve,
    )
    from core.declared_identity import listing_identity, record_identity, identity_review_dimensions
    from core.model_input import _normalized_stream, _normalized_tokens

    frame = pd.read_csv(CATALOG).head(500)
    rows = frame.to_dict('records')
    texts = []
    for row in rows:
        for key in ('sku_name_eng', 'attribute', 'description_short_eng',
                    'sku_url', 'image_url', 'brand', 'category',
                    'breadcrumbs_eng', 'retailer'):
            value = row.get(key)
            if isinstance(value, str):
                texts.append(value)
    corpus = texts + EDGE_STRINGS

    out: dict = {}
    out['corpus_size'] = len(corpus)
    out['unicode_casefold'] = [_call(unicode_casefold, value) for value in corpus]
    out['normalize_text'] = [_call(normalize_text, value) for value in corpus]
    out['normalize_retailer'] = [_call(normalize_retailer, value) for value in corpus]
    out['norm_unit'] = [_call(norm_unit, value) for value in corpus]
    out['normalized_attribute_text'] = [
        [normalized_attribute_text(value),
         normalized_attribute_text(value, value),
         normalized_attribute_text(value, 'Brand: X', value)] for value in corpus]
    out['normalized_attribute_text_raw'] = [
        _call(normalized_attribute_text, value) for value in corpus]
    out['attribute_fields'] = [
        [attribute_fields(value), attribute_fields(value, include_empty=True)]
        for value in corpus + EDGE_ATTRIBUTES]
    out['extract_pack_counts'] = [_call(lambda v: sorted(extract_pack_counts(v)), value)
                                  for value in corpus]
    out['extract_volume_ml'] = [_call(lambda v: list(extract_volume_ml(v)), value) for value in corpus]
    out['extract_volume_measurement'] = [
        _call(lambda v: list(extract_volume_measurement(v)), value) for value in corpus]
    out['extract_volume_match'] = [_call(lambda v: list(extract_volume_match(v)), value) for value in corpus]
    out['extract_volume_evidence'] = [_call(_eve, value) for value in corpus]
    def _search_span(value):
        match = _VOLUME_SEARCH(value)
        return None if match is None else [match.group(0), list(match.span())]
    out['_VOLUME_SEARCH'] = [_call(_search_span, value) for value in corpus]
    out['bucket_ml'] = [bucket_ml(value) for value in
                        (0.0, 0.5, 1.0, 5.0, 7.5, 12.0, 100.0, 999.9, 1000.0, 12345.6)]
    spelling_index = _volume_spelling_index()
    out['volume_spelling_index'] = sorted(spelling_index.keys())
    out['volume_entry'] = [
        None if _volume_entry(unit) is None else _volume_entry(unit).pattern
        for unit in ['ml', 'l', 'oz', 'fl oz', 'g', 'kg', 'dl', 'cl', 'cc',
                     'ltr', 'lt', 'gal', 'qt', 'pt', 'nope', '']]
    out['normalized_stream'] = [_call(_normalized_stream, value) for value in corpus]
    out['normalized_tokens'] = [
        [_call(_normalized_tokens, value, drop_schema_words=False),
         _call(_normalized_tokens, value, drop_schema_words=True)] for value in corpus]

    title_attr = [(row.get('sku_name_eng') or '', row.get('attribute') or '',
                   row.get('description_short_eng') or '') for row in rows]
    title_attr += [(value, value, value) for value in EDGE_STRINGS
                   if isinstance(value, str)]
    out['listing_identity'] = [
        _call(listing_identity, title, attrs, desc) for title, attrs, desc in title_attr]
    out['record_identity'] = [
        stable(record_identity({'declared_identity': {'flavor': ['a']},
                                'attribute_consistency_flags': [
                                    'volume_sources_disagree',
                                    'categorical_source_conflict:pack',
                                    'other']})),
        stable(record_identity({})),
        stable(record_identity({'source_rows': '[{"sku_name_eng": "Coke Zero"}]'})),
        stable(record_identity({'evidence_ledger': [{'field': 'declared_identity',
                                                     'value': {'flavor': ['cola']}}]})),
    ]
    out['identity_review_dimensions'] = [
        identity_review_dimensions({'declared_identity': {'flavor': ['a']}},
                                   {'declared_identity': {'flavor': ['b']}}),
        identity_review_dimensions({}, {}),
    ]

    # Wide-corpus gate. The 500-row fixture above is NOT enough on its own:
    # r21 introduced a real behaviour difference that appeared ONLY on the 5k
    # cohort — 'Tang Orange Powdered Drink Mix ( Makes 6 Quarts), 20-ounce
    # Canister' — where a latched dry-product probe turned a net_weight role
    # into package_volume. 24,025 texts over five real columns, stored as
    # per-function digests so the document stays small.
    cohort = ROOT / 'dataset_5k.csv'
    if cohort.exists():
        wide: list[str] = []
        wide_frame = pd.read_csv(cohort)
        for column in ('sku_name_eng', 'sku_url', 'image_url',
                       'description_short_eng', 'attribute'):
            if column in wide_frame:
                wide += [value for value in
                         wide_frame[column].fillna('').astype(str).tolist() if value]
        out['wide_corpus_size'] = len(wide)
        for label, function in (
            ('extract_volume_evidence', _eve),
            ('extract_volume_match', extract_volume_match),
            ('extract_pack_counts', lambda text: sorted(extract_pack_counts(text))),
            ('extract_volume_ml', extract_volume_ml),
            ('normalize_text', normalize_text),
            ('normalized_attribute_text', normalized_attribute_text),
            ('unicode_casefold', unicode_casefold),
            ('attribute_fields', attribute_fields),
            ('_normalized_stream', _normalized_stream),
        ):
            digest = hashlib.sha256()
            for text in wide:
                digest.update(json.dumps(_call(function, text), sort_keys=True).encode())
                digest.update(b'\x00')
            out[f'wide_{label}'] = digest.hexdigest()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    document = collect()
    Path(args.out).write_text(json.dumps(stable(document), indent=1, sort_keys=True))
    print(f'wrote {args.out} ({Path(args.out).stat().st_size} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
