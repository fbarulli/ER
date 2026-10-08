"""Project real listings and prepared text views for a CPU smoke.

Retains parent split assignments and materialized augmentation lineage. This
is a lifecycle smoke, not a quality experiment or a new training diet.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml


def _checkout_path(value, *, train_root: Path) -> Path:
    """One config path value as an absolute checkout path.

    Suite configs name their inputs either absolutely (a prepared tree) or
    checkout-relative (the committed fixtures); the relative form is the one
    that survives transport, so both are resolved from the checkout root.
    """
    path = Path(value)
    return path.resolve() if path.is_absolute() else (train_root / path).resolve()


def _portable_path(path: Path, *, train_root: Path) -> str:
    """A written config value that survives transport.

    A fixture must not embed the machine it was built on, so anything inside
    the checkout is recorded relative to it; a genuinely external path stays
    absolute rather than being silently rewritten.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(train_root).as_posix()
    except ValueError:
        return str(resolved)


#: The lane-config keys that name an input the smoke must repoint at its own tree.
_SMOKE_INPUT_KEYS = ('listings', 'pairs', 'input_manifest', 'text_cache',
                     'text_index', 'gnn_checkpoint')


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def _repoint_smoke_paths(settings: dict, *, setup: Path, output: Path,
                         train_root: Path) -> dict:
    """Rebase one graph lane's parent inputs onto this smoke's own tree.

    The parent may record absolute paths (a live prepared tree) or
    checkout-relative ones (a committed fixture); both are resolved from the
    checkout root before the member's path below the parent setup is re-rooted
    under ``output``. The written value stays portable, so a checked-in smoke
    fixture never names the machine that produced it.
    """
    for key in _SMOKE_INPUT_KEYS:
        if not settings.get(key):
            continue
        source = _checkout_path(settings[key], train_root=train_root)
        try:
            relative = source.relative_to(setup)
        except ValueError as exc:
            raise ValueError(
                f'{key} points outside the smoke parent setup: {settings[key]}') from exc
        settings[key] = _portable_path(output / relative, train_root=train_root)
    return settings


def prepare_smoke(setup: Path, output: Path, *, sample: int = 100, suite_config: Path | None = None):
    setup, output = setup.resolve(), output.resolve()
    from core.common import TRAIN_ROOT
    train_root = Path(TRAIN_ROOT).resolve()
    from model_tracks.config import load_config
    parent = load_config(suite_config or TRAIN_ROOT/'config/model_tracks.yaml')
    if (TRAIN_ROOT/parent.setup_dir).resolve() != setup:
        raise ValueError('smoke parent suite differs from requested prepared setup')
    from graph_tracks.config import load_config as load_graph_config, load_text_config
    from graph_tracks.setup import (
        write_setup_frames,
        write_text_config,
        write_track_config,
    )
    layout = _setup_layout()
    graph_settings = {track: load_graph_config(setup/layout.track_config(track), expected_track=track).model_dump()
                      for track in ('gnn_only', 'cascade')}
    text_settings = load_text_config(setup/layout.text_config).model_dump()
    from core.common import SEED, training_cfg
    from graph_tracks.data import file_hash
    from graph_tracks.prepare import prepare
    from graph_tracks.train import write_json
    from training.folds import normalize_gtin
    from training.prepared_bundle import load_prepared_bundle, prepared_holdout, write_prepared_bundle
    if output.exists():
        raise FileExistsError(output)
    catalog=pd.read_csv(setup/layout.catalog,dtype=str,keep_default_na=False,low_memory=False)
    splits=pd.read_csv(setup/layout.splits,dtype=str)
    pairs=pd.read_csv(setup/layout.pairs,dtype={'sku_id1':str,'sku_id2':str})
    _,bundle=load_prepared_bundle(setup/'text_prepared.pkl.gz')
    populations=prepared_holdout(bundle,dict(training_cfg().split),seed=SEED)
    roles={normalize_gtin(key):role for role,values in enumerate(populations) for key in values}
    ids=set(catalog.sku_id)
    representatives={}
    for row in catalog.itertuples(index=False):
        representatives.setdefault(normalize_gtin(row.gtin),row.sku_id)
    chosen=set()
    for split in ('train','dev','test'):
        for label in (0,1):
            for row in pairs[(pairs.split==split)&(pairs.label==label)].head(4).itertuples(index=False):
                chosen.update((row.sku_id1,row.sku_id2))
    ndf=len(bundle['df'])
    # Keep complete transplant lineage, including donors. A prefix of plain
    # negatives can omit every counterfactual/swap view and fail the real diet.
    for role in range(3):
        for mode in ('counterfactual', 'swap_values'):
            count = 0
            for row in bundle['hard_negative_mask_audit']:
                if row.get('target_mode') != mode:
                    continue
                anchor, target = int(row['anchor_payload_idx']), int(row['pair_payload_idx'])
                if roles.get(normalize_gtin(bundle['row_bc'][anchor])) != role or roles.get(normalize_gtin(bundle['row_bc'][target])) != role:
                    continue
                needed = set()
                for key, value in row.items():
                    if not key.endswith('_payload_idx') or key in {'copy_payload_idx', 'copy_pair_payload_idx'} or value is None:
                        continue
                    index = int(value)
                    listing = (str(bundle['df'].iloc[index].sku_id) if index < ndf
                               else representatives.get(normalize_gtin(bundle['row_bc'][index])))
                    if listing not in ids:
                        break
                    needed.add(listing)
                else:
                    if len(chosen | needed) > sample:
                        continue
                    chosen.update(needed)
                    count += 1
                    if count == 3:
                        break
    for role in range(3):
        count=0
        for a,b in bundle['train_neg']:
            if a>=ndf:
                continue
            ka,kb=normalize_gtin(bundle['row_bc'][a]),normalize_gtin(bundle['row_bc'][b])
            source=str(bundle['df'].iloc[a].sku_id)
            target=representatives.get(kb)
            if roles.get(ka)==roles.get(kb)==role and source in ids and target:
                if len(chosen | {source, target}) > sample:
                    continue
                chosen.update((source,target))
                count+=1
                if count==4:
                    break
    if len(chosen)>sample:
        raise ValueError('sample too small for selected smoke supervision')
    for listing in catalog.sku_id:
        if len(chosen)==sample:
            break
        chosen.add(listing)
    if len(chosen)!=sample:
        raise ValueError('not enough listings for requested sample')
    frame=catalog[catalog.sku_id.isin(chosen)].copy()
    assignment=splits[splits.sku_id.isin(chosen)].copy()
    pair_frame=pairs[pairs.sku_id1.isin(chosen)&pairs.sku_id2.isin(chosen)].copy()
    output.mkdir(parents=True)
    write_setup_frames(output,catalog=frame,splits=assignment,pairs=pair_frame)
    listings=prepare(output/layout.catalog,output/layout.splits,output/layout.pairs,
                     output/layout.prepared_dir)
    selected_keys={normalize_gtin(v) for v in frame.gtin}
    source_indices=[i for i,p in enumerate(bundle['df'].sku_id.astype(str)) if p in chosen]
    first_copy=min((int(r['copy_payload_idx']) for r in bundle['mask_audit']+bundle['hard_negative_mask_audit']
                    if r.get('copy_payload_idx') is not None), default=len(bundle['payload']))
    # Native retrieval reads the complete frozen canonical corpus immediately
    # after source rows. Retain that layout, but supervise only sampled pairs.
    supervised=set(source_indices+[i for i in range(ndf,first_copy) if normalize_gtin(bundle['row_bc'][i]) in selected_keys])
    indices=source_indices+list(range(ndf,first_copy))
    retained=set(indices)
    audits={field: [] for field in ('mask_audit','hard_negative_mask_audit')}
    pending={field: list(bundle[field]) for field in audits}
    progress=True
    while progress:
        progress=False
        for field in audits:
            deferred=[]
            for row in pending[field]:
                sources=[int(v) for k,v in row.items() if k.endswith('_payload_idx') and k not in {'copy_payload_idx', 'copy_pair_payload_idx'} and v is not None]
                if not all(i in supervised for i in sources):
                    deferred.append(row)
                    continue
                audits[field].append(dict(row))
                copies={int(v) for k,v in row.items() if k in {'copy_payload_idx', 'copy_pair_payload_idx'} and k.endswith('_payload_idx') and v is not None}
                retained.update(copies)
                supervised.update(copies)
                progress=True
            pending[field]=deferred
    indices=source_indices+sorted(retained-set(source_indices))
    remap={old:new for new,old in enumerate(indices)}
    for rows in audits.values():
        for row in rows:
            for key,value in row.items():
                if key.endswith('_payload_idx') and value is not None:
                    row[key]=remap[int(value)]
    arrays={}
    masks={}
    for field in ('pos','hp_pairs','neg','train_neg'):
        source=np.asarray(bundle[field],dtype=int).reshape(-1,2)
        mask=np.array([int(a) in supervised and int(b) in supervised for a,b in source],dtype=bool)
        arrays[field]=np.array([[remap[int(a)],remap[int(b)]] for a,b in source[mask]],dtype=int).reshape(-1,2)
        masks[field]=mask
    parent_manifest = json.loads((setup/layout.manifest).read_text())
    # The parent records its baseline checkpoint absolutely (a live tree) or
    # checkout-relative (a committed fixture), and the child re-records it
    # portably so a checked-in smoke never names the machine that built it.
    checkpoint = _checkout_path(parent_manifest['text_checkpoint'], train_root=train_root)
    write_prepared_bundle(output/'text_prepared.pkl.gz',df=bundle['df'].iloc[source_indices].reset_index(drop=True),
        payload=[bundle['payload'][i] for i in indices],structured_features=np.asarray(bundle['structured_features'])[indices],
        row_bc=np.asarray(bundle['row_bc'])[indices],country=np.asarray(bundle['country'])[indices],
        emb0=(np.asarray(bundle['emb0'])[indices] if np.asarray(bundle['emb0']).size
              else bundle['emb0']),**arrays,neg_sources=np.asarray(bundle['neg_sources'])[masks['neg']],
        train_neg_sources=np.asarray(bundle['train_neg_sources'])[masks['train_neg']],**audits,
        **{key:bundle[key] for key in ('labeled_pairs_csv','canonical_records_csv','gate_results_csv','payload_variant','masking_profile')},
        plan_sample=True,
        token_checkpoint=_portable_path(checkpoint, train_root=train_root),
        holdout_populations={s:sorted(values) for s,values in zip(('train','dev','test'),populations)})
    manifest=dict(parent_manifest)
    manifest.update(smoke=True,source_listing_count=sample,
                    parent_setup_sha256=file_hash(setup/layout.manifest),
                    text_checkpoint=_portable_path(checkpoint, train_root=train_root))
    write_json(output/layout.manifest,manifest)
    from graph_tracks.text_cache import checkpoint_hash
    if checkpoint_hash(checkpoint) != manifest['text_checkpoint_sha256']:
        raise ValueError('smoke baseline differs from the parent checkpoint')
    # Bind the CPU cache to the same prepared-token request used by the suite.
    from core.common import runtime
    from model_tracks.text_export import prepare as prepare_text_export
    from model_tracks.baseline_export import prepare as prepare_baseline
    prepare_text_export(output, checkpoint, batch_size=runtime('batch_size_embed'))
    prepare_baseline(output, checkpoint)
    for track in ('gnn_only','cascade'):
        settings = _repoint_smoke_paths(graph_settings[track], setup=setup,
                                        output=output, train_root=train_root)
        settings.update(device='cpu',epochs=1,report_test=False)
        write_track_config(output, track, settings)
    text_settings.update(report_test=False)
    write_text_config(output, text_settings)
    cfg = parent.model_dump()
    # A smoke is a lifecycle check, not a performance experiment: the suite
    # config keeps resource profiling off (matching the committed S fixture),
    # while post-training ablation stays inherited from the parent so the local
    # finalize still fires from the downloaded result archive.
    cfg.update(setup_dir=_portable_path(output, train_root=train_root),
               text_bundle=_portable_path(output/'text_prepared.pkl.gz', train_root=train_root),
               epochs=1,device='cpu',report_test=False,publish_git=False,
               publish_dvc=False,profiling=False)
    (output/'suite.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    return output/'suite.yaml'
