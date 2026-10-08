"""Local preparation/reporting and a GPU-only frozen-checkpoint worker.

Intervention removes a declared attribute cell, then reuses the model composer.
Title/brand evidence and the checkpoint's training graph context remain fixed.
No optimization, threshold fitting, synthetic labels or implicit cache reuse.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
import tempfile
import contextvars
import functools
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from core.bundle import bundle_spec
from core.common import TRAIN_ROOT, retrieval_ks
from core.encoding_inputs import load_token_features, tokenization_policy
from core.eval_trace import AttributeAttributionRow, decision_flip
from core.model_input import build_sku_text, model_input_info
from core.portable_archive import cached_file_digest
from core.run_log import RunLogger
from core.sku_identity import row_identity
from core.step_trace import timed
from core.text import normalized_attribute_text
from core.tracing import SCOPE_ENTITY, flush_stage_trace, stage_trace
from graph_tracks.data import file_hash as _raw_file_hash, load_records, RELATIONS, NUMERIC
from graph_tracks.prepared_inputs import load_batch
from graph_tracks.text_cache import checkpoint_hash, composition_fingerprint
from model_tracks.embedding_forward import validate_embedding_device
from model_tracks.resume import TRAINING_TRACKS
from training.masking import field_of

_LOG = RunLogger(__name__)

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "ablation"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout. Deliberately NEVER reset, unlike
#: ``training.training.flush_training_trace``: ``prepare``/``encode``/``report``
#: run once per track inside one suite, and ``core.tracing`` commits a stage
#: run-scoped and idempotently (a re-write REPLACES that stage's rows for the
#: run). A reset would therefore make the second track's commit silently replace
#: the first track's rows; keeping the one writer accumulating is what makes the
#: whole ablation track followable from the single file.
_TRACE = None


def trace():
    """The ONE writer for the ``ablation`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's ablation rows once; a no-op while empty.

    Idempotent: the writer is retained, so rows added after a flush (the next
    track's ``report`` call) are committed by the next flush and never dropped.
    """
    return flush_stage_trace(_TRACE)


def file_hash(path):
    """The ablation lane's file digest: the ONE shared memoized policy by name.

    The algorithm lives once (``core.portable_archive.raw_file_digest``) and the
    per-process memo policy lives once (``cached_file_digest``, keyed on
    abspath/mtime_ns/size). This wrapper only keeps the lane's directory case --
    a checkpoint directory is hashed by its composition, not as one file -- on
    top of that home; the local ``_HASH_MEMO`` copy is gone.
    """
    path = Path(path)
    if path.is_dir():
        return _raw_file_hash(path)
    return cached_file_digest(path)


def _raw_identity(path):
    path = Path(path)
    return checkpoint_hash(path) if path.is_dir() else _raw_file_hash(path)


def _default_retrieval_ks() -> tuple[int, ...]:
    """Inherit evaluation.retrieval_ks unless the lane declares an override."""
    return retrieval_ks()


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True)
    sample_pairs: int = Field(default=100, ge=1)
    coverage: str = 'sampled'
    uniform_channels: bool = False
    seed: int = 1729
    split: str = 'dev'
    batch_size: int = Field(default=256, ge=1)
    accelerator: str = 'T4'
    attributes: list[str] = Field(default_factory=list)
    output_dir: str = 'results/attribute_ablation'
    report_path: str = 'results/attribute_ablation/report.json'
    retrieval_catalog: str = 'full'
    hnsw_m: int = Field(default=16,ge=1)
    hnsw_ef_construction: int = Field(default=200,ge=1)
    hnsw_ef_search: int = Field(default=100,ge=1)
    retrieval_ks: list[StrictInt] = Field(
        default_factory=lambda: list(_default_retrieval_ks()), min_length=1
    )
    graph_fields: dict[str, list[str]] = Field(default_factory=dict)
    slice_columns: list[str] = Field(default_factory=list)


    @model_validator(mode='after')
    def check_retrieval(self):
        if self.coverage not in {'sampled', 'all'}:
            raise ValueError('coverage must be sampled or all')
        if any(k < 1 for k in self.retrieval_ks) or len(set(self.retrieval_ks)) != len(self.retrieval_ks):
            raise ValueError('retrieval_ks must contain unique positive integers')
        return self


def settings(path=None):
    return Settings.model_validate(yaml.safe_load((path or TRAIN_ROOT/'config/attribute_ablation.yaml').read_text()))


_PORTABLE_CONTEXT = contextvars.ContextVar('ablation_portable_context',default=None)


@contextmanager
def request_context(request_path):
    request = json.loads(request_path.read_text())
    context = None
    if request.get('portable_setup'):
        anchor = Path(request['portable_setup'])
        if anchor.is_absolute() or '..' in anchor.parts:
            raise ValueError('unsafe portable setup anchor')
        # Bound jobs always live at <suite>/<track>/ablation/request.json.
        suite = request_path.parent.parent.parent
        from core.bundle import bundle_spec
        setup = suite / bundle_spec().prepared_inputs_dir / request['portable_setup']
        if not setup.exists():
            setup = TRAIN_ROOT/request['portable_setup']
        context = {'@setup':setup,'@suite':suite}
    token = _PORTABLE_CONTEXT.set(context)
    try:
        yield
    finally:
        _PORTABLE_CONTEXT.reset(token)


def scoped_request(function):
    @functools.wraps(function)
    def wrapped(request_path,*args,**kwargs):
        with request_context(request_path):
            return function(request_path,*args,**kwargs)
    return wrapped


def resolve(path):
    path = Path(path)
    if path.parts and path.parts[0] in {'@setup','@suite'}:
        context = _PORTABLE_CONTEXT.get()
        if context is None:
            raise ValueError('portable ablation source requires verified request context')
        root = context[path.parts[0]]
        result = root.joinpath(*path.parts[1:]).resolve()
        if not result.is_relative_to(root.resolve()):
            raise ValueError('unsafe portable ablation source')
        return result
    return path if path.is_absolute() else TRAIN_ROOT/path


_JSON_ENCODER = json.JSONEncoder(sort_keys=True, ensure_ascii=False)


def digest(value):
    # STREAMED, never materialized. `json.dumps` builds the whole document as
    # one contiguous string before hashing; an exhaustive-cohort request is
    # ~1 GB of JSON (732 MB measured on the 2026-10-06 text track) and the
    # gnn_only digest ran while the text track's token batches were still
    # resident, so the kernel OOM-killed the run. iterencode is the SAME
    # encoder with the SAME kwargs, so the emitted bytes — and therefore every
    # cohort_sha256 / content-addressed directory name derived from them — are
    # byte-identical to the previous implementation; only peak memory drops.
    hasher = hashlib.sha256()
    for chunk in _JSON_ENCODER.iterencode(value):
        hasher.update(chunk.encode())
    return hasher.hexdigest()


def source_name(path):
    path = Path(path).resolve()
    return path.relative_to(TRAIN_ROOT).as_posix() if path.is_relative_to(TRAIN_ROOT) else str(path)


def checkpoint_identity(path):
    return checkpoint_hash(path) if path.is_dir() else file_hash(path)


def write(path, value):
    # STREAMED for the same reason as digest(): a request of this size must
    # never exist as one in-memory string. json.dump writes incrementally
    # through the encoder's iterencode, so the bytes on disk are unchanged.
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2,
                  allow_nan=False)
        handle.write('\n')


def _field_surfaces(text: str) -> dict[str, set[str]]:
    """Declared attribute -> set of raw ';'-parts, one pass per endpoint."""
    surfaces: dict[str, set[str]] = {}
    for part in str(text).split(';'):
        if ':' not in part:
            surfaces.setdefault('', set()).add(part)
            continue
        field, cell = part.split(':', 1)
        surfaces.setdefault(normalized_attribute_text(field), set()).add(field + ':' + cell)
    return surfaces


def declaration_removed(row, attribute):
    result = dict(row)
    if result.get('frozen_payload'):
        fields = {'volume': {'volume'}, 'count per unit': {'pack'},
            'flavour': {'flavor'}, 'carbonization': {'carbonation'},
            'sweetener': {'sweetener', 'sweetener_type', 'sweetening'},
            'pack type': {'package_type'}, 'pack material type': {'package_material'},
            'juice content': {'juice_content'}}.get(attribute, set())
        result['frozen_payload'] = ' '.join(token for token in result['frozen_payload'].split()
                                             if field_of(token) not in fields)
    result['attribute'] = ';'.join(part for part in str(row.get('attribute', '')).split(';')
        if ':' not in part or normalized_attribute_text(part.split(':', 1)[0]) != attribute)
    return result


def graph_removed(record, fields):
    result = copy.deepcopy(record)
    for field in fields:
        channel, key = field.split('.', 1)
        allowed = RELATIONS if channel == 'attribute' else NUMERIC if channel == 'numeric' else ()
        if key not in allowed:
            raise ValueError(f'unsupported graph field: {field}')
        result[channel].pop(key, None)
    return result


def sample_pairs(frame, cfg):
    required = {'sku_id1', 'sku_id2', 'label', 'split'}
    if not required.issubset(frame.columns):
        raise ValueError(f'pairs require {sorted(required)}')
    if cfg.coverage == 'all':
        if frame.empty or not frame.label.isin(['0', '1']).all():
            raise ValueError('full ablation requires binary labeled pairs')
        return frame.to_dict('records')
    if cfg.split not in {'dev', 'test'}:
        raise ValueError('ablation requires an explicit held-out dev or test split')
    frame = frame[frame.split == cfg.split].copy()
    if frame.empty or not frame.label.isin(['0', '1']).all():
        raise ValueError('selected split needs binary labeled pairs')
    # Round-robin across observed joint strata, deterministic within each.
    # An all-empty column (empty CSV cells read as '' with keep_default_na=False)
    # is a degenerate single stratum, not a real axis: exclude it like an
    # absent column so prepare() reports it in missing_axes instead.
    axes = ['label'] + [x for x in cfg.slice_columns if x in frame and x != 'lineage_id'
                       and not x.startswith(('gate_', 'jev_'))
                       and not (frame[x].dtype == object and frame[x].eq('').all())]
    rng = np.random.default_rng(cfg.seed)
    groups = [list(rng.permutation(group.index)) for _, group in frame.groupby(axes, sort=True, dropna=False)]
    chosen = []
    while groups and len(chosen) < cfg.sample_pairs:
        for group in groups:
            if group and len(chosen) < cfg.sample_pairs:
                chosen.append(group.pop())
        groups = [group for group in groups if group]
    return frame.loc[chosen].to_dict('records')


def validate_sources(request):
    for path, expected in request['sources'].items():
        if path.endswith('.py'):
            continue
        source = resolve(path)
        if not source.exists() or _raw_identity(source) != expected:
            trace().add(
                "sources", "changed",
                scope=SCOPE_ENTITY, key=path,
                reason='a frozen ablation source no longer matches the identity the request '
                       'pinned; the lane refuses rather than reporting on different inputs',
                detail={'path': path, 'expected': expected,
                        'present': source.exists(),
                        'observed': _raw_identity(source) if source.exists() else None},
                source=path,
            )
            flush_trace()
            raise ValueError(f'ablation source changed: {path}')


class _TextPool:
    """Interned model texts shared by the baseline, variant and candidate phases."""

    def __init__(self):
        self.texts, self.lookup = [], {}

    def intern(self, text):
        if text not in self.lookup:
            self.lookup[text] = len(self.texts)
            self.texts.append(text)
        return self.lookup[text]


def _compose(row, composer):
    if row.get('frozen_payload'):
        return row['frozen_payload']
    if composer is not None:
        return composer(row)
    return build_sku_text(pd.Series(row), model_input_info(row_identity(row).as_mapping()))


@timed
def _prepared_sources(track, listings, text_checkpoint, catalog, pairs, checkpoint, config):
    cfg = settings(config)
    if track not in set(TRAINING_TRACKS):
        raise ValueError('unknown track')
    if (track != 'text') != bool(listings):
        raise ValueError('graph tracks require listings; the text track must not pass any')
    if text_checkpoint:
        # Kept as an accepted keyword so the frozen staged-ablation caller does
        # not change shape, but the fused hybrid checkpoint is retired.
        raise ValueError('text_checkpoint belonged to the retired hybrid track')
    inputs = [catalog, pairs, checkpoint] + ([listings] if listings else [])
    config = config or TRAIN_ROOT/'config/attribute_ablation.yaml'
    inputs += [config, TRAIN_ROOT/'src/core/encoding_inputs.py',TRAIN_ROOT/'src/model_tracks/ablation_inputs.py',
               TRAIN_ROOT/'src/graph_tracks/infer.py',TRAIN_ROOT/'src/graph_tracks/pooling.py',
               TRAIN_ROOT/'src/model_tracks/ablation_retrieval.py',TRAIN_ROOT/'src/training/hnsw_index.py']
    sources = {source_name(p): checkpoint_identity(Path(p)) for p in inputs}
    return cfg, sources


@timed
def _selected_pairs(pairs, cfg):
    frame = pd.read_csv(pairs, dtype=str, keep_default_na=False)
    chosen = sample_pairs(frame, cfg)
    # A present-but-empty slice column reads as '' (not None); normalize
    # all-empty axes to None so they are reported in missing_axes and the
    # report rows carry null instead of a silent empty-string stratum.
    for axis in cfg.slice_columns:
        if chosen and all(p.get(axis) is None or p.get(axis) == '' for p in chosen):
            for p in chosen:
                p[axis] = None
    trace().add(
        "prepare",
        "pairs_selected",
        in_count=int(len(frame)),
        out_count=len(chosen),
        reason=(
            "coverage="
            + cfg.coverage
            + ": the cohort is "
            + ("the whole labeled population" if cfg.coverage == "all"
               else f"the held-out {cfg.split!r} split")
            + f", sampled deterministically at seed={int(cfg.seed)}"
        ),
        detail={
            "coverage": cfg.coverage,
            "split": cfg.split,
            "seed": int(cfg.seed),
            "sample_pairs": int(cfg.sample_pairs),
            "pairs_in_file": int(len(frame)),
            "pairs_chosen": len(chosen),
            "slice_columns": list(cfg.slice_columns),
        },
        source=source_name(Path(pairs)),
    )
    return chosen


@timed
def _catalog_rows(catalog):
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False)
    if 'sku_id' not in frame or frame.sku_id.duplicated().any() or (frame.sku_id == '').any():
        raise ValueError('catalog requires unique nonempty sku_id')
    rows = frame.set_index('sku_id', drop=False).to_dict('index')
    trace().add(
        "prepare", "catalog_rows",
        in_count=int(len(frame)), out_count=len(rows),
        reason='the catalog requires a unique nonempty sku_id; one ablation row per sku_id',
        detail={'catalog_rows': int(len(frame)), 'unique_sku_ids': len(rows)},
        source=source_name(Path(catalog)),
    )
    return rows


@timed
def _pair_endpoints(chosen):
    ids = sorted({p[k] for p in chosen for k in ('sku_id1', 'sku_id2')})
    trace().add(
        "prepare", "endpoints",
        in_count=2 * len(chosen), out_count=len(ids),
        reason='both endpoints of every selected pair, deduplicated to distinct catalog ids',
        detail={'pairs': len(chosen), 'endpoint_slots': 2 * len(chosen),
                'distinct_endpoints': len(ids)},
        source='selected ablation pairs',
    )
    return ids


@timed
def _attributes(cfg):
    from core.attribute_universe import attribute_registry
    registry = attribute_registry()
    attributes = cfg.attributes or sorted(registry)
    if len(set(attributes)) != len(attributes) or set(attributes) - registry.keys():
        raise ValueError('attributes must be unique registry keys')
    trace().add(
        "prepare", "attributes",
        in_count=len(cfg.attributes) if cfg.attributes else len(registry),
        out_count=len(attributes),
        reason=('the lane declared an explicit attribute list'
                if cfg.attributes else
                'the lane declared no attributes; the whole registry is ablated'),
        detail={'declared': list(cfg.attributes), 'registry_keys': len(registry),
                'ablated': len(attributes)},
        source='config/attribute_ablation.yaml (Settings.attributes)',
    )
    return attributes


@timed
def _listing_records(listings, ids, cfg):
    records = {r['sku_id']: r for r in load_records(listings)} if listings else {}
    if listings and any(i not in records or (cfg.coverage != 'all' and records[i]['split'] != cfg.split) for i in ids):
        raise ValueError('graph endpoints must belong to the selected held-out split')
    trace().add(
        "prepare", "graph_records",
        # Two different populations (pair endpoints vs every listing record):
        # a load/validation row, not a funnel.
        in_count=None, out_count=len(records),
        reason=('graph lane: one listing record per distinct pair endpoint, each checked '
                'against the selected held-out split' if listings else
                'text lane: no graph listings are consumed'),
        detail={'listings': source_name(Path(listings)) if listings else None,
                'endpoints': len(ids), 'records': len(records)},
        source=source_name(Path(listings)) if listings else 'no listings (text lane)',
    )
    return records


@timed
def _pair_evidence(chosen, rows, cfg):
    from core.attribute_conflicts import canonical_attribute_info
    from core.attribute_decision import engine
    decision = engine()
    for pair in chosen:
        a, b = (rows[pair[k]] for k in ('sku_id1', 'sku_id2'))
        pair['gtin1'], pair['gtin2'] = a.get('gtin'), b.get('gtin')
        pair['current_attribute_evidence'] = ({} if a.get('frozen_payload') or b.get('frozen_payload') else
            decision.evaluate(canonical_attribute_info(a), canonical_attribute_info(b), left_raw=a, right_raw=b).as_dict())
        if a.get('frozen_payload') or b.get('frozen_payload'):
            pair['evidence_scope'] = 'frozen payload; raw-row evidence unavailable'
        for axis in cfg.slice_columns:
            pair.setdefault(axis, None)
    return chosen


def _baseline_and_variants(pool, rows, ids, attributes, records, cfg, track, listings, composer, started):
    print(f'[ablation/local] composing baseline endpoints={len(ids)} attributes={len(attributes)}',flush=True)
    baseline_text = []
    if track != 'gnn_only':
        for n,i in enumerate(ids):
            baseline_text.append(pool.intern(_compose(rows[i], composer)))
            if (n+1) % 25 == 0 or n+1 == len(ids):
                print(f'[ablation/local] baseline={n+1}/{len(ids)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    baseline_records = [records[i] for i in ids] if listings else []
    variants = [{'attribute':None, 'channel':'baseline', 'text_indices':baseline_text,
                 'records':baseline_records, 'changed_listings':0}]
    endpoint_surfaces = {i: _field_surfaces(rows[i].get('attribute', '')) for i in ids}
    endpoint_parts = {i: str(rows[i].get('attribute', '')).split(';') for i in ids}
    for attr_index, attribute in enumerate(attributes,1):
        altered_text = []
        if baseline_text:
            for n,i in enumerate(ids):
                row_surfaces = endpoint_surfaces[i]
                frozen = rows[i].get('frozen_payload')
                if attribute in row_surfaces or frozen:
                    kept_parts = [
                        part for part in endpoint_parts[i]
                        if part not in row_surfaces.get(attribute, ())
                    ]
                    kept_attribute = ';'.join(kept_parts)
                    changed_row = ({**rows[i], 'attribute': kept_attribute}
                        if not frozen else declaration_removed(rows[i], attribute))
                    altered_text.append(baseline_text[n] if changed_row == rows[i] else pool.intern(_compose(changed_row, composer)))
                else:
                    altered_text.append(baseline_text[n])
        altered_records = [graph_removed(r, cfg.graph_fields.get(attribute, [])) for r in baseline_records]
        for channel in (['text', 'graph', 'both'] if cfg.uniform_channels else
                        ['text'] if track == 'text' else ['graph'] if track == 'gnn_only' else ['text', 'graph', 'both']):
            ti = altered_text if channel in {'text', 'both'} else baseline_text
            gr = altered_records if channel in {'graph', 'both'} else baseline_records
            changed = sum((bool(ti) and ti[n] != baseline_text[n]) or
                          (bool(gr) and gr[n] != baseline_records[n]) for n in range(len(ids)))
            variants.append({'attribute':attribute, 'channel':channel, 'text_indices':ti,
                             'records':gr, 'changed_listings':changed})
        print(f'[ablation/local] attribute={attr_index}/{len(attributes)} {attribute} unique_texts={len(pool.texts)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    channels_per_attribute: dict[str, int] = {}
    for variant in variants[1:]:
        channels_per_attribute[variant['attribute']] = channels_per_attribute.get(variant['attribute'], 0) + 1
    trace().add(
        "prepare", "variants",
        # A DERIVATION, not a funnel: attributes fan out into one variant per
        # channel, so there is no single input population to state.
        in_count=None, out_count=len(variants) - 1,
        reason='one baseline variant plus one variant per ablated attribute and channel',
        detail={'attributes': len(attributes), 'variants_including_baseline': len(variants),
                'channels_per_attribute': channels_per_attribute,
                'changing_variants': sum(1 for v in variants if v['changed_listings'] > 0),
                'unique_texts': len(pool.texts)},
        source='ablation interventions (declared attribute removed per channel)',
    )
    # The EXACT per-effect census is a GROUP row per bucket; the entity rows are a
    # bounded sample of it. A variant that changes no model input is the honest
    # exception set: the intervention is declared but this lane cannot express it.
    trace().add_entities(
        "prepare.variant_effect", variants,
        key_of=lambda v: f"{v['attribute']}:{v['channel']}",
        reason_of=lambda v: 'changes_model_input' if v['changed_listings'] else 'no_changed_input',
        detail_of=lambda v: {'attribute': v['attribute'], 'channel': v['channel'],
                             'changed_listings': int(v['changed_listings']),
                             'texts': len(v['text_indices']), 'records': len(v['records'])},
        source='composed ablation variants',
    )
    return baseline_text, variants


def _candidate_catalog(pool, rows, ids, baseline_text, records, cfg, track, listings, composer, started):
    if cfg.retrieval_catalog not in {'full','sampled'}:
        raise ValueError('retrieval_catalog must be full or sampled')
    candidate_ids = sorted(rows) if cfg.retrieval_catalog == 'full' else ids
    candidate_text = []
    if track != 'gnn_only':
        baseline_lookup = dict(zip(ids,baseline_text))
        last_progress = time.monotonic()
        for n,i in enumerate(candidate_ids,1):
            candidate_text.append(baseline_lookup[i] if i in baseline_lookup else pool.intern(_compose(rows[i], composer)))
            if n == len(candidate_ids) or time.monotonic()-last_progress >= 10:
                print(f'[ablation/local] candidate texts={n}/{len(candidate_ids)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                last_progress = time.monotonic()
    candidate_records = [records[i] for i in candidate_ids] if listings else []
    trace().add(
        "prepare", "candidates",
        in_count=len(rows) if cfg.retrieval_catalog == 'full' else len(ids),
        out_count=len(candidate_ids),
        reason=(f"retrieval_catalog={cfg.retrieval_catalog}: "
                + ('the trained catalog is the fixed candidate set'
                   if cfg.retrieval_catalog == 'full' else
                   'the pair endpoints are the fixed candidate set')),
        detail={'catalog_mode': cfg.retrieval_catalog, 'catalog_rows': len(rows),
                'candidate_ids': len(candidate_ids),
                'candidate_text_indices': len(candidate_text),
                'candidate_records': len(candidate_records)},
        source='ablation retrieval catalog',
    )
    return candidate_ids, candidate_text, candidate_records


def _request_document(cfg, track, checkpoint_role, sources, checkpoint, text_checkpoint,
                      candidate_ids, candidate_text, candidate_records, ids, pool, chosen, variants, attributes):
    return {'schema':'er-attribute-ablation-v2','track':track,'checkpoint_role':checkpoint_role, 'settings':cfg.model_dump(),
        'sources':sources, 'composition':composition_fingerprint(), 'implementation_sha256':file_hash(Path(__file__)),
        'checkpoint':source_name(checkpoint), 'text_checkpoint':source_name(text_checkpoint) if text_checkpoint else None,
        'candidate_ids':candidate_ids,'candidate_text_indices':candidate_text,'candidate_records':candidate_records,
        'ids':ids, 'texts':pool.texts, 'pairs':chosen, 'variants':variants,
        'cohort_sha256':digest(chosen),
        'coverage':{'mode':cfg.coverage, 'pair_rows':len(chosen),
            'by_scope':pd.Series([p.get('evaluation_scope', p['split']) for p in chosen]).value_counts().to_dict(),
            'by_label':pd.Series([p['label'] for p in chosen]).value_counts().to_dict(),
            'by_population':pd.Series([p.get('population') or 'real' for p in chosen]).value_counts().to_dict(),
            'attributes':attributes, 'uniform_channels':cfg.uniform_channels},
        'intervention':'declared attribute removed; title/brand and training graph context fixed',
        'retrieval_scope':f'fixed {cfg.retrieval_catalog} catalog; query-only interventions; incomplete known-positive truth',
        'missing_axes':[a for a in cfg.slice_columns if all(p.get(a) is None or p.get(a) == '' for p in chosen)]}


@timed
def _persist_prepared(request, cfg, token_cache):
    from model_tracks.ablation_inputs import prepare_inputs
    out_dir = resolve(cfg.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        prepared = Path(tmp)/'prepared_inputs.npz'
        request['prepared_inputs'] = prepare_inputs(request,prepared,token_cache=token_cache)
        validate_sources(request)
        # One digest, two uses: the content-addressed staging dir AND the trace
        # row. A second digest() here would stream the (up to ~1GB) request twice.
        request_sha = digest(request)
        output = out_dir/request_sha[:24]
        output.mkdir(parents=True, exist_ok=True)
        destination = output/prepared.name
        if destination.exists():
            if file_hash(destination) != request['prepared_inputs']['sha256']:
                raise ValueError('prepared tensors differ')
        else:
            prepared.replace(destination)
        path = output/bundle_spec().ablation_request_file
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError('existing request differs')
        write(path,request)
        text_slots = (sum(len(variant['text_indices']) for variant in request['variants'])
                      + len(request['candidate_text_indices']))
        trace().add(
            "prepare", "request_persisted",
            in_count=text_slots, out_count=len(request['texts']),
            reason='every intervention text slot is interned down to its distinct native text; '
                   'the request freezes cohort, interventions and texts',
            detail={'request_path': source_name(path), 'request_sha256': request_sha,
                    'prepared_inputs_sha256': request['prepared_inputs']['sha256'],
                    'cohort_sha256': request.get('cohort_sha256'),
                    'variants': len(request['variants']),
                    'pairs': len(request['pairs']),
                    'candidate_ids': len(request['candidate_ids']),
                    'unique_texts': len(request['texts']),
                    'attributes': len(request.get('coverage', {}).get('attributes', [])),
                    'missing_axes': request.get('missing_axes')},
            source=source_name(path),
        )
    return path


@timed
def prepare(catalog, pairs, checkpoint, *, track='text', listings=None, text_checkpoint=None, config=None, checkpoint_role='selected',composer=None,token_cache=None):
    with _LOG.section('ablation.prepare.sources'):
        cfg, sources = _prepared_sources(track, listings, text_checkpoint, catalog, pairs, checkpoint, config)
    with _LOG.section('ablation.prepare.cohort'):
        chosen = _selected_pairs(pairs, cfg)
        rows = _catalog_rows(catalog)
        ids = _pair_endpoints(chosen)
        if set(ids)-rows.keys():
            raise ValueError('pair endpoint absent from catalog')
        attributes = _attributes(cfg)
        records = _listing_records(listings, ids, cfg)
        _pair_evidence(chosen, rows, cfg)
    pool = _TextPool()
    started = time.monotonic()
    with _LOG.section('ablation.prepare.baseline_variants'):
        baseline_text, variants = _baseline_and_variants(pool, rows, ids, attributes, records, cfg, track, listings, composer, started)
    with _LOG.section('ablation.prepare.candidates'):
        candidate_ids, candidate_text, candidate_records = _candidate_catalog(pool, rows, ids, baseline_text, records, cfg, track, listings, composer, started)
    with _LOG.section('ablation.prepare.persist'):
        request = _request_document(cfg, track, checkpoint_role, sources, checkpoint, text_checkpoint,
                                    candidate_ids, candidate_text, candidate_records, ids, pool, chosen, variants, attributes)
        path = _persist_prepared(request, cfg, token_cache)
    print(f'[ablation/local] pairs={len(chosen)} endpoints={len(ids)} unique_texts={len(pool.texts)} '
          f'variants={len(variants)-1} changed={sum(v["changed_listings"] > 0 for v in variants[1:])}', flush=True)
    flush_trace()
    return path


@timed
def load_prepared(request_path, request):
    plan = request.get('prepared_inputs')
    if request.get('schema') != 'er-attribute-ablation-v2' or not plan:
        raise ValueError('locally prepared model inputs required; prepare again')
    path = request_path.parent/'prepared_inputs.npz'
    if _raw_file_hash(path) != plan['sha256']:
        trace().add(
            "prepared_inputs", "checksum_mismatch",
            scope=SCOPE_ENTITY, key=str(path),
            reason='the prepared tensors beside the request do not match the digest the request '
                   'pinned; the lane refuses rather than encoding different inputs',
            detail={'path': str(path), 'expected_sha256': plan['sha256'],
                    'request_path': source_name(request_path)},
            source=source_name(request_path),
        )
        flush_trace()
        raise ValueError('prepared input checksum mismatch')
    return np.load(path,allow_pickle=False)


@timed
def _validated_device(device):
    return validate_embedding_device(device)


@timed
def _prepared_text_vectors(request, arrays, plan, device, track, text_model, saved_text):
    if track == 'gnn_only':
        trace().add(
            "encode", "text_vectors", in_count=0, out_count=0,
            reason='the gnn_only decider consumes no text vectors',
            detail={'track': track, 'device': device}, source='none',
        )
        return None
    from sentence_transformers import SentenceTransformer
    checkpoint = request['checkpoint'] if track == 'text' else request['text_checkpoint']
    model = text_model
    expected_checkpoint = checkpoint_identity(resolve(checkpoint))
    if model is not None and getattr(model,'_er_checkpoint_sha256',None) != expected_checkpoint:
        raise ValueError('shared text model checkpoint differs from frozen request')
    if model is not None and model.device.type != device:
        raise ValueError('shared text model device differs from frozen request')
    if model is None:
        model = SentenceTransformer(str(resolve(checkpoint)),device=device,local_files_only=True)
    model.eval()
    if tokenization_policy(model) != plan['tokenization']:
        raise ValueError('worker tokenizer/checkpoint policy differs from local preparation')
    covered = np.zeros(len(request['texts']),dtype=bool)
    seeded = None
    if saved_text is not None:
        from graph_tracks.data import load_text_cache
        from graph_tracks.text_cache import texts_hash
        from core.model_input import model_input_composition
        with np.load(saved_text,allow_pickle=False) as cache:
            saved_ids = cache['ids'].astype(str).tolist()
        candidates,metadata = load_text_cache(saved_text,request['candidate_ids'])
        mapping = dict(zip(request['candidate_ids'],request['candidate_text_indices']))
        if set(saved_ids) != set(mapping):
            raise ValueError('baseline text export catalog differs from prepared ablation')
        if metadata.get('checkpoint_sha256') != expected_checkpoint or metadata.get('tokenization') != plan['tokenization'] or metadata.get('composition') != model_input_composition().model_dump(mode='json') or metadata.get('text_sha256') != texts_hash([request['texts'][mapping[key]] for key in saved_ids]):
            raise ValueError('baseline text export differs from prepared native text/checkpoint')
        seeded = np.empty((len(request['texts']),candidates.shape[-1]),dtype=np.float32)
        for row,index in enumerate(request['candidate_text_indices']):
            if covered[index] and not np.allclose(seeded[index],candidates[row],atol=1e-5):
                raise ValueError('identical baseline texts have inconsistent exported vectors')
            seeded[index] = candidates[row]
            covered[index] = True
    reused_from_export = int(np.count_nonzero(covered))
    chunks = []
    with torch.no_grad():
        for n,batch in enumerate(plan['token_batches'],1):
            features = load_token_features(arrays,batch,device)
            selected = np.flatnonzero(~covered[batch['start']:batch['start']+batch['count']])
            if not len(selected):
                continue
            features = {key:value[torch.as_tensor(selected,device=device)] if isinstance(value,torch.Tensor) and value.ndim and len(value)==batch['count'] else value for key,value in features.items()}
            vectors = model(features)['sentence_embedding']
            vectors = torch.nn.functional.normalize(vectors,p=2,dim=1).cpu().numpy().astype(np.float32)
            if seeded is None:
                seeded = np.empty((len(request['texts']),vectors.shape[-1]),dtype=np.float32)
            positions = batch['start']+selected
            seeded[positions] = vectors
            covered[positions] = True
            print(f'[ablation/gpu] changed native texts={len(selected)} batch={n}/{len(plan["token_batches"])}; truncated=0',flush=True)
    if not covered.all():
        raise ValueError('prepared ablation text vectors miss native inputs')
    text_vectors = seeded
    trace().add(
        "encode", "text_vectors",
        in_count=len(request['texts']), out_count=int(covered.sum()),
        reason='the baseline text export seeds every catalog text it already covers; the '
               'remaining native texts are encoded forwards on the frozen checkpoint',
        detail={'texts': len(request['texts']), 'reused_from_export': reused_from_export,
                'encoded_on_device': int(covered.sum()) - reused_from_export,
                'token_batches': len(plan['token_batches']), 'device': device,
                'checkpoint': request['checkpoint'],
                'saved_text': None if saved_text is None else source_name(saved_text)},
        source=str(request['checkpoint']),
    )
    del model
    return text_vectors


@timed
def _prepared_graph_encoder(request, arrays, plan, device, track, graph_encoder):
    encoder = None
    graph_batches = {}
    if track != 'text':
        from graph_tracks.infer import GraphEncoder
        vocabulary = plan['vocabulary']
        encoder = graph_encoder
        if encoder is None:
            support = load_batch(arrays,'support',device,vocabulary)
            encoder = GraphEncoder(resolve(request['checkpoint']),device,prepared_support=support)
        elif encoder.checkpoint_sha256 != file_hash(resolve(request['checkpoint'])) or encoder.device != device:
            raise ValueError('shared graph encoder differs from frozen checkpoint/device')
        if encoder.vocabulary != vocabulary:
            raise ValueError('prepared vocabulary differs from checkpoint')
        graph_batches = {key:[load_batch(arrays,prefix,device,vocabulary) for prefix in prefixes]
                         for key,prefixes in plan['graph_batches'].items()}
    trace().add(
        "encode", "graph_encoder",
        # A UNIT row: either the lane has an encoder to encode with or it does not.
        in_count=None, out_count=1 if encoder is not None else 0,
        reason=('the prepared graph batches are encoded through the frozen checkpoint encoder'
                if encoder is not None else
                'the text ranker consumes no graph encoder'),
        detail={'track': track, 'device': device, 'graph_batch_groups': len(graph_batches),
                'shared_encoder_supplied': graph_encoder is not None},
        source=str(request['checkpoint']) if track != 'text' else 'none (text track)',
    )
    return encoder, graph_batches


@timed
def _prepared_candidates(request, arrays, plan, device, text_vectors, encoder, saved_candidates):
    candidate_vectors = saved_candidates
    if candidate_vectors is not None and (candidate_vectors.dtype != np.float32 or candidate_vectors.shape[0] != len(request['candidate_ids']) or not np.isfinite(candidate_vectors).all() or not np.allclose(np.linalg.norm(candidate_vectors,axis=1),1,atol=1e-4)):
        raise ValueError('saved graph candidate vector contract mismatch')
    if candidate_vectors is None and request.get('candidate_ids'):
        candidate_text = text_vectors[arrays['candidate_text_indices']] if text_vectors is not None else None
        candidate_vectors = candidate_text if encoder is None else encoder.encode_prepared(
            [load_batch(arrays,prefix,device,plan['vocabulary']) for prefix in plan['candidate_batches']],candidate_text)
    trace().add(
        "encode", "candidates",
        # A lane may legitimately arrive with saved candidate vectors and no
        # candidate_ids in the request, so the two counts are stated, not linked.
        in_count=None,
        out_count=0 if candidate_vectors is None else int(candidate_vectors.shape[0]),
        reason=('the saved graph candidate vectors are validated and reused'
                if saved_candidates is not None else
                'the fixed catalog candidates are encoded from the prepared tensors'),
        detail={'candidate_ids': len(request.get('candidate_ids', [])),
                'reused_saved': saved_candidates is not None,
                'dim': None if candidate_vectors is None else int(candidate_vectors.shape[-1])},
        source='prepared candidates',
    )
    return candidate_vectors


@timed
def _prepared_jobs(request, arrays, plan, device, text_vectors, encoder, graph_batches, saved_candidates):
    indices = arrays['pair_indices']
    results = []
    for n,job in enumerate(plan['jobs'],1):
        text = text_vectors[arrays[job['text_indices_key']]] if text_vectors is not None else None
        if encoder is None:
            vec = text
        elif saved_candidates is not None and n-1 == plan['variant_jobs'][0]:
            candidate_lookup = {key:i for i,key in enumerate(request['candidate_ids'])}
            vec = saved_candidates[[candidate_lookup[key] for key in request['ids']]]
        else:
            vec = encoder.encode_prepared(graph_batches[job['graph_key']],text)
        if encoder is None:
            score = (vec[indices[:,0]]*vec[indices[:,1]]).sum(-1)
        else:
            with torch.no_grad():
                score = encoder.scorer(torch.as_tensor(vec,device=device),torch.as_tensor(indices,device=device),
                    None if text is None else torch.as_tensor(text,device=device)).sigmoid().cpu().numpy()
        results.append((vec,score))
        print(f'[ablation/{device}] prepared inference job={n}/{len(plan["jobs"])}',flush=True)
    vectors = [results[job][0] for job in plan['variant_jobs']]
    scores = [results[job][1] for job in plan['variant_jobs']]
    trace().add(
        "encode", "jobs",
        in_count=len(plan['jobs']), out_count=len(results),
        reason='one forward pass per prepared job; the baseline job reuses saved candidate '
               'vectors when the lane supplied them',
        detail={'jobs': len(plan['jobs']), 'scored_pairs': len(request['pairs']),
                'variants': len(plan['variant_jobs']), 'device': device,
                'uses_saved_candidates': saved_candidates is not None},
        source='prepared inference jobs',
    )
    return vectors, scores


@timed
def _persist_outputs(output, request_path, arrays, vectors, scores, candidate_vectors):
    arrays.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    request_sha = file_hash(request_path)
    with output.open('xb') as handle:
        np.savez_compressed(handle, vectors=np.asarray(vectors,dtype=np.float32), scores=np.asarray(scores,dtype=np.float32),
                            request_sha256=request_sha,embedding_dtype='float32',
                            **({'candidate_vectors':np.asarray(candidate_vectors,dtype=np.float32)} if candidate_vectors is not None else {}))
    output_sha = file_hash(output)
    output.with_suffix('.sha256').write_text(output_sha)
    trace().add(
        "encode", "persisted",
        # A UNIT row: the whole job fan-in lands in one export artifact.
        in_count=None, out_count=1,
        reason='one float32 vectors/scores export and its sha256 sidecar for the whole request',
        detail={'output': source_name(output), 'output_sha256': output_sha,
                'request_sha256': request_sha,
                'vectors_shape': list(np.asarray(vectors).shape),
                'scores_shape': list(np.asarray(scores).shape),
                'candidate_vectors': candidate_vectors is not None,
                'embedding_dtype': 'float32'},
        source=source_name(request_path),
    )


@timed
@scoped_request
def encode(request_path, output, *, device='cuda',saved_text=None,text_model=None,saved_candidates=None,graph_encoder=None):
    """Colab inference only; all interventions and texts arrive prepared."""
    device = _validated_device(device)
    if output.exists():
        raise FileExistsError(output)
    with _LOG.section('ablation.encode.load'):
        request = json.loads(request_path.read_text())
        # Sources are relocated by the launcher but expected hashes stay frozen.
        validate_sources(request)
        arrays = load_prepared(request_path,request)
        plan = request['prepared_inputs']
        track = request['track']
    trace().add(
        "encode", "plan",
        # A DERIVATION (one job per variant, baseline included), not a funnel.
        in_count=None, out_count=len(plan['jobs']),
        reason='one forward pass per prepared job; a variant whose inputs are unchanged '
               'reuses the baseline vectors instead of a second pass',
        detail={'track': track, 'device': device, 'variants': len(request['variants']),
                'jobs': len(plan['jobs']), 'variant_jobs': list(plan['variant_jobs']),
                'token_batches': len(plan.get('token_batches') or []),
                'pairs': len(request['pairs']), 'ids': len(request['ids']),
                'candidate_ids': len(request.get('candidate_ids', []))},
        source=source_name(request_path),
    )
    with _LOG.section('ablation.encode.text_vectors'):
        text_vectors = _prepared_text_vectors(request, arrays, plan, device, track, text_model, saved_text)
    with _LOG.section('ablation.encode.graph_encoder'):
        encoder, graph_batches = _prepared_graph_encoder(request, arrays, plan, device, track, graph_encoder)
    with _LOG.section('ablation.encode.candidates'):
        candidate_vectors = _prepared_candidates(request, arrays, plan, device, text_vectors, encoder, saved_candidates)
    with _LOG.section('ablation.encode.jobs'):
        vectors, scores = _prepared_jobs(request, arrays, plan, device, text_vectors, encoder, graph_batches, saved_candidates)
    with _LOG.section('ablation.encode.persist'):
        _persist_outputs(output, request_path, arrays, vectors, scores, candidate_vectors)
    flush_trace()


@timed
def _threshold_from_csv(path, value, track, checkpoint):
    frame = pd.read_csv(path)
    values = frame['threshold'].tolist() if 'threshold' in frame else []
    if 'threshold' in frame:
        hits = frame[frame['threshold'].apply(
            lambda x: isinstance(x,(int,float)) and np.isfinite(x) and float(x) == value)]
        for column,key in (('model','track'),('checkpoint','checkpoint')):
            if column in frame:
                found = sorted({str(v) for v in hits[column].tolist() if isinstance(v,str) and v})
                if len(found) > 1:
                    raise ValueError(f'threshold source attests conflicting {column} values')
                if found:
                    if key == 'track':
                        track = found[0]
                    else:
                        checkpoint = found[0]
    return values, track, checkpoint


@timed
def _threshold_from_json(path, value):
    document = json.loads(path.read_text())
    claimed_sha256 = None
    track = None
    checkpoint = None
    if isinstance(document,dict):
        claimed_sha256 = document.get('checkpoint_sha256') or document.get('vectors_metadata',{}).get('checkpoint_sha256')
    values = []
    attested = {'track':set(), 'checkpoint':set()}
    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get('threshold'),(int,float)) and np.isfinite(obj['threshold']):
                values.append(obj['threshold'])
                for column,key in (('model','track'),('track','track'),('checkpoint','checkpoint')):
                    witness = obj.get(column)
                    if isinstance(witness,str) and witness:
                        attested[key].add(witness)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(document)
    # manifests pin identity at the top level while thresholds nest in summaries
    for column,key in (('model','track'),('track','track'),('checkpoint','checkpoint')):
        witness = document.get(column) if isinstance(document,dict) else None
        if isinstance(witness,str) and witness:
            attested[key].add(witness)
    for key,seen in attested.items():
        if len(seen) > 1:
            raise ValueError(f'threshold source attests conflicting {key} values')
        if seen:
            if key == 'track':
                track = next(iter(seen))
            else:
                checkpoint = next(iter(seen))
    return values, track, checkpoint, claimed_sha256


@timed
def frozen_threshold(source, value):
    path = resolve(source)
    if not path.is_file():
        raise ValueError('threshold source must be an existing saved report')
    before = _raw_file_hash(path)
    if path.suffix == '.csv':
        values, track, checkpoint = _threshold_from_csv(path, value, None, None)
        claimed_sha256 = None
    else:
        values, track, checkpoint, claimed_sha256 = _threshold_from_json(path, value)
    if not any(isinstance(x, (int,float)) and np.isfinite(x) and float(x) == value for x in values):
        raise ValueError('threshold differs from the saved baseline report')
    if before != _raw_file_hash(path):
        raise ValueError('threshold report changed while reading')
    return {'path':source_name(path),'sha256':before,'selection':'saved baseline; never refitted during ablation',
            'track':track, 'checkpoint':checkpoint, 'checkpoint_sha256':claimed_sha256}


@timed
def verify_threshold_binding(request, provenance):
    if provenance.get('track') != request['track']:
        raise ValueError('threshold source track differs or is missing')
    checkpoint = request.get('checkpoint')
    expected = request.get('sources',{}).get(checkpoint)
    if not checkpoint or not expected:
        raise ValueError('threshold requires a checkpoint identity in the request')
    claimed = provenance.get('checkpoint_sha256')
    if claimed:
        if claimed != expected:
            raise ValueError('threshold source checkpoint identity differs')
    else:
        named = provenance.get('checkpoint')
        if not named:
            raise ValueError('threshold source checkpoint identity is missing')
        if Path(named).name != Path(checkpoint).name:
            raise ValueError('threshold source checkpoint differs')
        candidates = [resolve(named),resolve(provenance['path']).parent/named,
                      resolve(provenance['path']).parent/Path(named).name]
        located = next((path for path in candidates if path.exists()),None)
        if located is None or checkpoint_identity(located) != expected:
            raise ValueError('threshold source checkpoint identity differs or cannot be verified')
    return {'track':request['track'],'checkpoint_sha256':expected,'verified':True}


@timed
@scoped_request
def validate_vectors(request_path, result):
    """Cheap integrity validation before closing the GPU; no metrics or ANN."""
    request = json.loads(request_path.read_text())
    validate_sources(request)
    if 'prepared_inputs' in request:
        load_prepared(request_path,request).close()
    with np.load(result,allow_pickle=False) as data:
        if str(data['request_sha256'].item()) != _raw_file_hash(request_path):
            raise ValueError('ablation result belongs to another request')
        vectors, scores = data['vectors'],data['scores']
        candidates = data['candidate_vectors'] if 'candidate_vectors' in data else None
    expected = (len(request['variants']),len(request['ids']))
    if vectors.dtype != np.float32 or scores.dtype != np.float32 or (candidates is not None and candidates.dtype != np.float32):
        raise ValueError('ablation exported vectors/scores require persisted float32')
    if vectors.ndim != 3 or vectors.shape[:2] != expected or scores.shape != (expected[0],len(request['pairs'])):
        raise ValueError('ablation result shape mismatch')
    for matrix in (vectors,candidates):
        if matrix is not None and (not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=-1),1,atol=1e-4)):
            raise ValueError('ablation vectors must be finite and normalized')
    if not np.isfinite(scores).all():
        raise ValueError('ablation scores must be finite')
    if request.get('candidate_ids') and (candidates is None or candidates.ndim != 2 or candidates.shape != (len(request['candidate_ids']),vectors.shape[-1])):
        raise ValueError('full catalog candidate vectors missing or shape mismatch')
    trace().add(
        "vectors", "validated",
        # Two different populations (catalog ids vs comparison pairs): this is a
        # validation row, not a funnel.
        in_count=None, out_count=len(request['pairs']),
        reason='cheap integrity validation of one ablation export before it is reused or reported',
        detail={'request_path': source_name(request_path), 'result': source_name(result),
                'variants': expected[0], 'ids': expected[1], 'pairs': len(request['pairs']),
                'candidate_vectors': candidates is not None,
                'vectors_shape': list(vectors.shape), 'scores_shape': list(scores.shape)},
        source=source_name(result),
    )
    return request,vectors,scores,candidates


@timed
def _comparison_rows(request, vectors, scores, threshold, cfg, retrieval, id_lookup, baseline_ranks, ann_baseline, comparison_cache):
    rows = []
    variants = request['variants']
    pairs = request['pairs']
    baseline_variant = variants[0]
    def ranks(vec):
        return retrieval.ranks(vec)
    for n, variant in enumerate(variants[1:], 1):
        if not variant['changed_listings'] and (not np.array_equal(vectors[n],vectors[0]) or not np.array_equal(scores[n],scores[0])):
            trace().add(
                "report", "noop_variant_changed_output",
                scope=SCOPE_ENTITY, key=f"{variant['attribute']}:{variant['channel']}",
                reason='an intervention that changed no model input still produced different '
                       'vectors/scores; the comparison is refused rather than reported',
                detail={'attribute': variant['attribute'], 'channel': variant['channel'],
                        'changed_listings': int(variant['changed_listings']),
                        'cohort_sha256': request.get('cohort_sha256')},
                source='composed ablation variants',
            )
            flush_trace()
            raise ValueError('no-op ablation changed model output')
        key = hashlib.sha256(vectors[n].tobytes()).hexdigest()
        if key not in comparison_cache:
            comparison_cache[key] = (ranks(vectors[n]),retrieval.ann_hits(vectors[n]))
        rank,ann_ablated = comparison_cache[key]
        for p, pair in enumerate(pairs):
            endpoints = [id_lookup[pair[k]] for k in ('sku_id1','sku_id2')]
            # The baseline variant ablates no attribute: carry the pair's
            # full evidence map instead of a meaningless {None: None}
            # (JSON "null" key); variant rows keep the ablated attribute's entry.
            evidence = pair.get('current_attribute_evidence',{})
            if variant['attribute'] is not None:
                evidence = {variant['attribute']:evidence.get(variant['attribute'])}
            rows.append({**pair, 'current_attribute_evidence':evidence,
                'attribute':variant['attribute'],'channel':variant['channel'],
                'endpoint_input_changed':[(bool(variant.get('text_indices')) and variant['text_indices'][i] != baseline_variant['text_indices'][i]) or
                    (bool(variant.get('records')) and variant['records'][i] != baseline_variant['records'][i]) for i in endpoints],
                'changed_listings':variant['changed_listings'], 'baseline_score':float(scores[0,p]),
                'ablated_score':float(scores[n,p]), 'score_delta':float(scores[n,p]-scores[0,p]),
                'decision_flip':bool((scores[n,p]>=threshold)!=(scores[0,p]>=threshold)),
                'baseline_error':bool((scores[0,p]>=threshold) != (pair['label']=='1')),
                'ablated_error':bool((scores[n,p]>=threshold) != (pair['label']=='1')),
                'embedding_cosine_delta':[float(1-np.dot(vectors[0,i],vectors[n,i])) for i in endpoints],
                'baseline_ranks':baseline_ranks[p], 'ablated_ranks':rank[p],
                'ann_baseline_hits':ann_baseline[p], 'ann_ablated_hits':ann_ablated[p],
                'known_positive_recall_change':{str(k):[(int(rank[p][e]<=k)-int(baseline_ranks[p][e]<=k))
                    if pair['label']=='1' else None for e in (0,1)] for k in cfg.retrieval_ks}})
    return rows


def assert_ablation_rows(rows, threshold):
    """Validate the emitted comparison rows; return the ORIGINAL list, untouched.

    The contract is a PREDICATE over the emitter's own dicts: each row is
    validated verbatim against ``AttributeAttributionRow`` and its
    ``decision_flip`` is checked against the FROZEN threshold this run already
    bound -- never against a hardcoded ``0.0`` and never with a chained
    comparison, because the row model owns no threshold and the lane that does
    is the only authority on the verdict (the D3 false rejection).

    Nothing is renamed, rebuilt or re-serialized, so a caller still hands the
    ORIGINAL dicts to the report writer and the emitted bytes are unchanged.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError('ablation row validation requires a finite frozen threshold')
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'ablation row {index} is {type(row).__name__}, not a dict')
        model = AttributeAttributionRow.model_validate(row)
        expected = decision_flip(model.baseline_score, model.ablated_score, threshold)
        if model.decision_flip is not expected:
            raise ValueError(
                f'ablation row {index} decision_flip={model.decision_flip} disagrees '
                f'with the frozen threshold {threshold!r}: baseline='
                f'{model.baseline_score!r} ablated={model.ablated_score!r} -> {expected}')
    return rows


def _report_document(request, request_path, result, cfg, rows, npairs, threshold, threshold_source, threshold_provenance, threshold_binding, candidate_ids):
    return {'schema':'er-attribute-ablation-report-v1', 'track':request['track'],'checkpoint_role':request.get('checkpoint_role','selected'),
        'request_path':source_name(request_path), 'request_sha256':file_hash(request_path),
        'result_path':source_name(result),'result_sha256':file_hash(result),
        'sources':request['sources'],'composition':request['composition'],
        'implementation_sha256':request['implementation_sha256'], 'embedding_dtype':'float32', 'threshold':threshold,
        'threshold_source':str(threshold_source), 'threshold_provenance':threshold_provenance, 'threshold_binding':threshold_binding, 'split':cfg.split, 'sample_pairs':npairs,
        'intervention':request['intervention'],'retrieval_scope':request['retrieval_scope'],
        'missing_axes':request['missing_axes'], 'retrieval_catalog_count':len(candidate_ids),
        'cohort_sha256':request.get('cohort_sha256'), 'coverage':request.get('coverage'),
        'retrieval_intervention':'query only; fixed candidates', 'rows':rows}


@timed
@scoped_request
def report(request_path, result, threshold, *, threshold_source, config=None, save=True):
    """Paired local comparisons at a supplied, already selected threshold."""
    with _LOG.section('ablation.report.validate'):
        if not np.isfinite(threshold) or not threshold_source:
            raise ValueError('frozen threshold and its source are required')
        threshold_provenance = frozen_threshold(threshold_source, threshold)
        request,vectors,scores,candidate_vectors = validate_vectors(request_path,result)
        threshold_binding = verify_threshold_binding(request,threshold_provenance)
        nv, ni, npairs = len(request['variants']),len(request['ids']),len(request['pairs'])
        cfg = Settings.model_validate(request['settings'])
    trace().add(
        "report", "validated",
        # Two different populations (comparison pairs vs ablated variants).
        in_count=None, out_count=nv - 1,
        reason='every ablated variant is compared against the baseline at the frozen threshold '
               'that was selected before this lane ran',
        detail={'track': request['track'], 'variants': nv, 'ids': ni, 'pairs': npairs,
                'threshold': float(threshold), 'threshold_source': str(threshold_source),
                'threshold_sha256': threshold_provenance.get('sha256'),
                'vectors_shape': list(vectors.shape), 'scores_shape': list(scores.shape),
                'candidate_vectors': candidate_vectors is not None,
                'request_path': source_name(request_path)},
        source=source_name(result),
    )
    with _LOG.section('ablation.report.comparison'):
        if not cfg.retrieval_ks or any(k < 1 for k in cfg.retrieval_ks):
            raise ValueError('retrieval ks must be positive')
        id_lookup = {i:n for n,i in enumerate(request['ids'])}
        from model_tracks.ablation_retrieval import RetrievalComparison
        candidate_ids = request.get('candidate_ids',request['ids'])
        if request.get('candidate_ids') and candidate_vectors is None:
            raise ValueError('full catalog candidate vectors missing')
        candidates = vectors[0] if candidate_vectors is None else candidate_vectors
        retrieval = RetrievalComparison(candidate_ids,candidates,request,request_path,cfg)
        def ranks(vec):
            return retrieval.ranks(vec)
        baseline_ranks = ranks(vectors[0])
        ann_baseline = retrieval.ann_hits(vectors[0])
        comparison_cache = {hashlib.sha256(vectors[0].tobytes()).hexdigest():(baseline_ranks,ann_baseline)}
        rows = _comparison_rows(request, vectors, scores, threshold, cfg, retrieval, id_lookup,
                                baseline_ranks, ann_baseline, comparison_cache)
        retrieval.close()
        # The comparison funnel plus its EXACT flip census. ``add_entities``
        # buckets the whole row population once (references only) and emits a
        # capped stratified sample, so the trace stays bounded even at
        # exhaustive-coverage size.
        trace().add(
            "report", "comparisons",
            in_count=(nv - 1) * npairs, out_count=len(rows),
            reason='one row per ablated variant and pair; the census below buckets them by '
                   'decision flip at the frozen threshold',
            detail={'variants_excluding_baseline': nv - 1, 'pairs': npairs, 'rows': len(rows),
                    'threshold': float(threshold), 'track': request['track'],
                    'retrieval_catalog_count': len(candidate_ids)},
            source=source_name(result),
        )
        trace().add_entities(
            "report.decision_flip", rows,
            key_of=lambda row: f"{row['attribute']}:{row['channel']}:{row['sku_id1']}~{row['sku_id2']}",
            reason_of=lambda row: 'flip' if row['decision_flip'] else 'no_flip',
            detail_of=lambda row: {'attribute': row['attribute'], 'channel': row['channel'],
                                   'label': row['label'], 'changed_listings': row['changed_listings'],
                                   'baseline_score': row['baseline_score'],
                                   'ablated_score': row['ablated_score'],
                                   'score_delta': row['score_delta'],
                                   'baseline_error': row['baseline_error'],
                                   'ablated_error': row['ablated_error']},
            source=source_name(result),
        )
    with _LOG.section('ablation.report.document'):
        # The emitted rows are the report's evidence; validate them against the
        # contract and the FROZEN threshold in scope BEFORE anything is saved.
        assert_ablation_rows(rows, threshold)
        output = _report_document(request, request_path, result, cfg, rows, npairs, threshold, threshold_source, threshold_provenance, threshold_binding, candidate_ids)
        validate_sources(request)
        if frozen_threshold(threshold_source, threshold) != threshold_provenance:
            raise ValueError('threshold report changed during comparison')
    if not save:
        flush_trace()
        return output
    report_path = save_report(request_path, output, config=config)
    flush_trace()
    return report_path


def save_report(request_path, output, *, config=None):
    """Persist an already-computed report; returns the dashboard pointer path."""
    path = resolve(settings(config).report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, output)
    # Keep each checkpoint's report alongside its inputs; dashboard pointer is latest.
    saved = request_path.parent/'report.json'
    if saved != path:
        write(saved, output)
    rows, threshold = output['rows'], output['threshold']
    print(f'[ablation/local] report={path} rows={len(rows)} threshold frozen={threshold}', flush=True)
    trace().add(
        "report", "persisted",
        in_count=None, out_count=1 if saved == path else 2,
        reason='identical report bytes land on the dashboard pointer and beside the request',
        detail={'dashboard_pointer': source_name(path), 'beside_request': source_name(saved),
                'single_path': saved == path, 'rows': len(rows),
                'threshold': threshold, 'track': output.get('track'),
                'reported_rows': len(rows)},
        source=source_name(request_path),
    )
    return path


def _cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for key in ('catalog','pairs','checkpoint'):
        prep.add_argument('--'+key,type=Path,required=True)
    prep.add_argument('--track', choices=TRAINING_TRACKS,default='text')
    for key in ('listings','config'):
        prep.add_argument('--'+key,type=Path)
    worker = sub.add_parser('encode')
    worker.add_argument('--request',type=Path,required=True)
    worker.add_argument('--output',type=Path,required=True)
    post = sub.add_parser('report')
    post.add_argument('--request',type=Path,required=True)
    post.add_argument('--result',type=Path,required=True)
    post.add_argument('--threshold',type=float,required=True)
    post.add_argument('--threshold-source',required=True)
    post.add_argument('--config',type=Path)
    return parser


@timed
def main():
    RunLogger.configure_console()
    parser = _cli_parser()
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'prepare':
        print(prepare(**args))
    elif action == 'encode':
        encode(args['request'],args['output'])
    else:
        args['request_path'] = args.pop('request')
        report(**args)


if __name__ == '__main__':
    main()
