"""Validate the prepared populations before provisioning a shared VM."""
from pathlib import Path
import hashlib
import json
import subprocess
import pandas as pd

from model_tracks.config import load_config
from model_tracks.resume import TRAINING_TRACKS


def _payload_digest(payload):
    """Preserve json.dumps(list(payload)) bytes without its full JSON allocation."""
    digest = hashlib.sha256(b'[')
    for index, text in enumerate(payload):
        if index:
            digest.update(b', ')
        digest.update(json.dumps(text, ensure_ascii=False).encode())
    digest.update(b']')
    return digest.hexdigest()


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


def scored_support_floor() -> int:
    """The configured minimum negatives per scored half (config SSOT).

    ``training.evaluation.robust_validation.min_test_negatives`` already owns
    the "a fold with fewer negatives than this cannot be scored" contract;
    the publish gate reuses it instead of inventing a second number.
    """
    from core.common import training_cfg
    return int(training_cfg().evaluation.robust_validation.min_test_negatives)


def assert_scored_support(pairs: dict, *, floor: int) -> dict:
    """Refuse a suite whose scored halves have no support to measure on.

    The graph pairs census is the population both the graph scorer and the
    cascade are scored on. A scored half with a handful of negatives (the
    shipped census carried single digits) cannot fit a Youden threshold or
    report a false-positive rate, and publishing it would present a
    near-degenerate metric as a result. Both classes must be present and the
    negative count must clear the configured floor.
    """
    problems: list[str] = []
    census: dict[str, dict[str, int]] = {}
    for split in ('dev', 'test'):
        counts = pairs.get(split) or {}
        negatives = int(counts.get('negative', 0))
        positives = int(counts.get('positive', 0))
        census[split] = {'positives': positives, 'negatives': negatives}
        if negatives < floor:
            problems.append(f'{split} has {negatives} negatives < floor {floor}')
        if positives <= 0:
            problems.append(f'{split} has no positives')
    if problems:
        raise ValueError(
            'scored-half support floor violated (refusing to publish): '
            + '; '.join(problems)
            + '. Balanced negatives come from config/training.yaml '
            'split.negative_fold_policy (see training.build_final_validation); '
            'min_test_negatives='
            + str(floor) + ' is the floor.'
        )
    return census


def prepared_report_test(root: Path) -> dict[str, bool]:
    """The report_test switch every prepared lane config actually carries.

    The suite config asks for the held-out test split; packaging renders each
    lane's runnable config, and ``graph_tracks/setup.py`` currently forces
    ``report_test=False`` into gnn_only/cascade/text when it writes them. That
    silent override is what kept the test split unscored, so preflight reads
    the PREPARED configs back and refuses instead of reporting a test split it
    will never score.
    """
    from graph_tracks.config import load_config as load_graph_config
    from graph_tracks.config import load_text_config
    layout = _setup_layout()
    prepared = {
        track: bool(load_graph_config(root / layout.track_config(track), expected_track=track).report_test)
        for track in ('gnn_only', 'cascade')
    }
    prepared['text'] = bool(load_text_config(root / layout.text_config).report_test)
    return prepared


def preflight(config: Path, *, allow_gpu_pending=False,native_token_model=None) -> dict:
    from core.common import F, SEED, TRAIN_ROOT, resolve_model, training_cfg
    from graph_tracks.data import file_hash
    from graph_tracks.preflight import preflight as graph_preflight, runtime_versions
    from graph_tracks.text_cache import checkpoint_hash
    from training.prepared_bundle import canonical_payload_rows, load_prepared_bundle, prepared_holdout
    from training.folds import normalize_gtin
    cfg = load_config(config)
    layout = _setup_layout()
    root = (TRAIN_ROOT / cfg.setup_dir).resolve()
    setup = json.loads((root / layout.manifest).read_text())
    is_smoke = setup.get('smoke', False)
    source_hash = file_hash(Path(F['dataset_deduped']))
    if not is_smoke and setup.get('source_catalog_sha256') != source_hash:
        raise ValueError('graph setup is stale: source catalog; rebuild locally before launch')
    labels_hash = file_hash(Path(F['labeled_pairs']))
    if not is_smoke and setup.get('labeled_pairs_sha256') != labels_hash:
        raise ValueError('graph setup is stale: labeled pairs; rebuild locally before launch')
    from graph_tracks.config import load_text_config, load_config as load_graph_config
    for track in ('gnn_only', 'cascade'):
        load_graph_config(root / layout.track_config(track), expected_track=track)
    load_text_config(root / layout.text_config)
    model = Path(resolve_model(cfg.text_model))
    if checkpoint_hash(model) != setup['text_checkpoint_sha256']:
        raise ValueError('text baseline differs from frozen text checkpoint')
    checks = {'gnn_only':graph_preflight(root/layout.track_config('gnn_only'),check_device=False,require_dvc=False)}
    # ── report_test consistency (loud, 2026-10-08) ─────────────────────────
    # The suite asks for the held-out test split; graph_tracks/setup.py forces
    # ``report_test=False`` into every prepared lane config, so the request was
    # silently dropped and the test split was never scored. Read the prepared
    # configs back and refuse instead of reporting a switch that does not
    # survive packaging. Smokes keep their saved switches.
    prepared_switches = prepared_report_test(root)
    report_test_mismatch = sorted(
        track for track, value in prepared_switches.items()
        if bool(value) != bool(cfg.report_test)
    )
    if report_test_mismatch and not is_smoke:
        raise ValueError(
            'prepared lane configs do not honour the suite report_test='
            f'{bool(cfg.report_test)}: {report_test_mismatch} carry '
            + ', '.join(f'{track}={prepared_switches[track]}' for track in report_test_mismatch)
            + ' (graph_tracks/setup.py hardcodes report_test=False when it '
            'renders the runnable configs); the held-out test split would '
            'never be scored'
        )
    # ── scored-half support floor (TODO "Support-floor gate") ──────────────
    # Refuse to publish a suite whose dev/test halves cannot support the
    # metric they are scored with (dev 1286/9, test 1276/7 shipped). Smokes
    # sample the same population and are exempt, like every other
    # stale/thin-population check here.
    support_floor = scored_support_floor()
    scored_support = (
        assert_scored_support(checks['gnn_only']['pairs'], floor=support_floor)
        if not is_smoke else {'exempt': 'smoke'}
    )
    # The cascade is a combinator, not a trainer: validate its declared config
    # now and confirm it consumes the same frozen graph population as gnn_only.
    # The trained text ANN + gnn_only scorer artifacts it composes are validated
    # by the worker once those tracks have completed.
    cascade = load_graph_config(root / layout.track_config('cascade'), expected_track='cascade')
    gnn_lane = load_graph_config(root / layout.track_config('gnn_only'), expected_track='gnn_only')
    for key in ('listings', 'pairs', 'input_manifest'):
        if getattr(cascade, key) != getattr(gnn_lane, key):
            raise ValueError('cascade must consume the shared manifested gnn_only ' + key)
    if cascade.allow_unmanifested_inputs or not cascade.input_manifest:
        raise ValueError('cascade requires manifested frozen graph inputs')
    if allow_gpu_pending and not (root/layout.shared_embeddings).exists():
        from model_tracks.baseline_export import validate_pending
        pending = validate_pending(root,model,native_model=native_token_model)
        checks['cascade'] = {
            'track': 'cascade', 'listings': checks['gnn_only']['listings'],
            'pairs': checks['gnn_only']['pairs'], 'device': cascade.device,
            'report_test': cascade.report_test, 'runtime': runtime_versions(cascade,require_dvc=False),
            'text_dimension': None, 'text_prerequisite': pending,
            'input_population_source': 'shared manifested gnn_only population',
            'composed_from': ['text ranker (ANN candidates)', 'gnn_only pair scorer (decisions)'],
        }
    else:
        checks['cascade'] = {
            'track': 'cascade', 'listings': checks['gnn_only']['listings'],
            'pairs': checks['gnn_only']['pairs'], 'device': cascade.device,
            'report_test': cascade.report_test, 'runtime': runtime_versions(cascade,require_dvc=False),
            'text_dimension': None,
            'composed_from': ['text ranker (ANN candidates)', 'gnn_only pair scorer (decisions)'],
        }
    manifest, bundle = load_prepared_bundle((TRAIN_ROOT / cfg.text_bundle).resolve())
    from training.run_plan import validate_run_plan, validate_epoch_batches
    from training.token_inputs import validate_training_tokens
    if 'training_tokens' not in bundle or 'training_plan' not in bundle:
        raise ValueError('text bundle lacks fixed native tokens/training row plan; rebuild locally')
    validate_training_tokens(bundle['training_tokens'])
    export_request = json.loads((root/layout.text_export_request).read_text())
    if bundle['training_tokens']['policy'] != export_request['plan']['tokenization']:
        raise ValueError('training native tokenizer differs from prepared suite export')
    payload_digest = _payload_digest(bundle['payload'])
    if bundle['training_tokens']['payload_sha256'] != payload_digest or not set(bundle['payload']).issubset(bundle['training_tokens']['texts']):
        raise ValueError('training native tokens differ from frozen payload')
    validate_run_plan(bundle,bundle['training_plan'],loss=training_cfg().training.loss,train_frac=1.,sample=bool(is_smoke),seed=SEED)
    from core.common import runtime
    batch_sizes = {device: int(runtime('batch_size_' + device)) for device in ('cpu', 'cuda')}
    if is_smoke:
        # Lifecycle smokes retain their saved batch settings, like the worker.
        saved = bundle['training_plan']['inputs']['folds'][0]['objective']['sampler']
        batch_sizes = {device: saved[device]['batch_size'] for device in batch_sizes}
    validate_epoch_batches(bundle['training_plan'], epochs=cfg.epochs, batch_sizes=batch_sizes)
    from model_tracks.training_data import SharedTrainingData, TrackTrainingBinding, from_bundle
    from model_tracks.shared_graph_data import validate_projection
    # Finish and release the reconstructed population before parsing its disk copy.
    expected_shared = from_bundle(bundle).fingerprint
    shared = SharedTrainingData.model_validate_json((root / layout.shared_training_data).read_bytes())
    if expected_shared != shared.fingerprint:
        raise ValueError('suite shared training data differs from frozen text objective')
    text_binding = TrackTrainingBinding.model_validate_json((root / layout.text_training_binding).read_text())
    if text_binding.track != 'text':
        raise ValueError('text training binding has wrong track')
    text_binding.validate_data(shared)
    for track in ('gnn_only',):
        validate_projection(root, shared, track=track)
    shared_summary = {'sha256': shared.fingerprint, 'examples': len(shared.examples),
                      'endpoints': len(shared.endpoints), 'graph_pair_rows': 2 * len(shared.examples),
                      'tracks': ['text', 'gnn_only', 'cascade']}
    del shared, text_binding
    from core.schemas import DataTuple
    DataTuple(n_df=len(bundle['df']), **{key: bundle[key] for key in
              ('payload', 'structured_features', 'row_bc', 'country', 'pos', 'hp_pairs', 'emb0')})
    n_payload = len(bundle['payload'])
    import numpy as np
    for pairs_key, sources_key in (('neg', 'neg_sources'), ('train_neg', 'train_neg_sources')):
        pairs = np.asarray(bundle[pairs_key])
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError(f'{pairs_key} must contain integer endpoint pairs')
        if pairs.size and (pairs.min() < 0 or pairs.max() >= n_payload):
            raise ValueError(f'{pairs_key} endpoints are outside the payload')
        if len(pairs) != len(bundle[sources_key]):
            raise ValueError(f'{sources_key} does not align with {pairs_key}')
    if not is_smoke:
        for key in ('labeled_pairs', 'canonical_records', 'gate_results'):
            if hashlib.sha256(bundle[f'{key}_csv']).hexdigest() != file_hash(F[key]):
                raise ValueError(f'text bundle is stale: {key}; rebuild locally before launch')
    canonical_payload_rows(len(bundle['df']), bundle['payload'], bundle['row_bc'])
    train, dev, test = prepared_holdout(bundle, dict(training_cfg().split), seed=SEED)
    roles = {normalize_gtin(key): split for split, values in
             [('train', train), ('dev', dev), ('test', test)] for key in values}
    catalog = pd.read_csv(root / layout.catalog, dtype=str, keep_default_na=False, low_memory=False)
    splits = pd.read_csv(root / layout.splits, dtype=str).set_index('sku_id').split
    for row in catalog.itertuples(index=False):
        if roles.get(normalize_gtin(row.gtin)) != splits[row.sku_id]:
            raise ValueError('text/graph prepared split mismatch')
    from core.identity_policy import reviewed_row_mask
    if reviewed_row_mask(bundle['df']).any():
        raise ValueError('text bundle contains held identity listings')
    # Run the same read-only diet contract locally and on the remote snapshot.
    # Strict drift checks prevent launch under a different augmentation config.
    # Reuse the already validated object; diet checking is CPU logic, not a
    # second interpreter that imports models and decompresses the same bundle.
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO
    import importlib.util
    spec = importlib.util.spec_from_file_location('er_diet_manifest', TRAIN_ROOT / 'scripts/diet_manifest.py')
    diet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diet_module)
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = diet_module.main(['diet_manifest.py', str(TRAIN_ROOT / cfg.text_bundle)],
                                prepared=(manifest, bundle))
    diet = subprocess.CompletedProcess([], code, out.getvalue(), err.getvalue())
    diet_warning = is_smoke and diet.returncode == 3
    if diet_warning:
        print('[preflight] WARNING: sampled smoke diet misses training thresholds:\n' + diet.stdout, flush=True)
    if diet.returncode and not diet_warning:
        raise ValueError('text bundle diet preflight failed:\n' + diet.stdout + diet.stderr)
    return {'shared_training_data': shared_summary,
            'text': {'bundle_sha256': manifest.sha256, 'payload': manifest.payload_variant,
                     'masking_profile': manifest.masking_profile, 'rows': manifest.n_df,
                     'diet': {'status': 'warning' if diet_warning else 'pass', 'log': diet.stdout}},
            **checks, 'cascade_mode': 'text ranker retrieves, gnn_only scorer decides; no fused embedding',
            'report_test': bool(cfg.report_test),
            'prepared_report_test': prepared_switches,
            'scored_support': scored_support,
            'scored_support_floor': support_floor,
            'source_catalog_sha256': source_hash, 'labeled_pairs_sha256': labels_hash,
            'parallel_workers': len(TRAINING_TRACKS), 'colab_sessions': 1, 'colab_control_channels': 1}
