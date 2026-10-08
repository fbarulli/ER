"""Two phases: local interventions/tensors, selected-weight binding and GPU forward.

Single-responsibility phases (behaviour pinned, statements split verbatim):
  - :func:`prepare_suite`      — the suite orchestrator (timed)
  - :func:`_cohort_gate`       — the exhaustive-coverage cohort, or None
  - :func:`_freeze_config`     — settings pointed at the root + frozen yaml
  - :func:`_frozen_support`    — train-population support records and vocabulary
  - :func:`_template_checkpoint` — the text baseline or a template tensor file
  - :func:`_track_request`     — prepare() tokens/tensors, emitted request read
  - :func:`_track_cohort`      — suite-cohort freeze on text, equality elsewhere
  - :func:`_anchor_request`    — prepared sources anchored to the portable package
  - :func:`_copy_template`     — the fixed template folder materialized
  - :func:`_drop_staging`      — generated content-addressed staging cleanup
  - :func:`_track_template`    — the per-track phase orchestrator
  - :func:`_bind_template`     — the staged setup bound onto a track request
  - :func:`_read_template` / :func:`_bind_staged_setup` — request read + staged root
  - :func:`_check_graph_binding` — selected checkpoint vs frozen support/vocabulary
  - :func:`_rebind_checkpoint` — selected/baseline checkpoint role resolution
  - :func:`_bound_folder`      — the bound request and local tensors materialized
  - :func:`_saved_text_default` — saved-vector default for the full local catalog
  - :func:`_encode_vectors`    — the ONLY device-executing leg (CPU migrations land here)
  - :func:`_reuse_or_encode`   — validated existing vectors vs the device call
  - :func:`forward`            — the forward orchestrator (timed)
"""
import json
from pathlib import Path
import shutil
import torch
import yaml
from core.bundle import bundle_spec
from core.model_input import model_input_composition
from core.run_log import RunLogger
from core.tracing import SCOPE_ENTITY, flush_stage_trace, stage_trace
from core.timing import Timing
from training.prepare_all_trace import timed
from graph_tracks.data import load_records
from graph_tracks.prepared_inputs import load_plan
from graph_tracks.text_cache import checkpoint_hash
from model_tracks.ablation import prepare, settings, write, resolve, digest, checkpoint_identity, encode, request_context, validate_vectors, file_hash, source_name
from model_tracks.ablation_cohort import prepare_cohort
from model_tracks.package import package_member

_LOG = RunLogger(__name__)

_BINDING_UNSET = object()


def _templates_dir(setup: Path) -> Path:
    """The declared ablation-template directory (``bundle.ablation_templates_dir``).

    The templates ship inside the prepared inputs bundle under this member, so
    the name is the bundle contract's and every surface (producer here, the
    packaged worker's skip gate, the baseline forward) reads it from there
    rather than re-spelling the literal.
    """
    return setup / bundle_spec().ablation_templates_dir


def _request_name() -> str:
    """The declared ablation request member name (``bundle.ablation_request_file``)."""
    return bundle_spec().ablation_request_file


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "staged_ablation"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout.
_TRACE = None


def trace():
    """The ONE writer for the ``staged_ablation`` stage of the current run."""
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's staged-ablation rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


@timed
def _freeze_suite(setup,config,bundle):
    """Resolve ablation settings, gate the cohort and freeze the template root."""
    cfg = settings(config)
    cohort = _cohort_gate(setup,cfg,bundle)
    frozen_config = _freeze_config(setup,cfg)
    return cohort,frozen_config


@timed
def _cohort_gate(setup,cfg,bundle):
    """The exhaustive-coverage cohort, or None; loud when 'all' lacks a bundle."""
    cohort = None
    if cfg.coverage == 'all':
        if bundle is None:
            raise ValueError('exhaustive ablation requires the prepared training bundle')
        cohort = prepare_cohort(setup, bundle)
    trace().add(
        "prepare_suite", "cohort_gate",
        in_count=1, out_count=1 if cohort is not None else 0,
        reason=("coverage='all' freezes ONE suite cohort over the prepared training bundle"
                if cohort is not None else
                "coverage='sampled' samples each lane's own dev pairs, so no suite cohort is frozen"),
        detail={'coverage': cfg.coverage, 'bundle_supplied': bundle is not None,
                'cohort_folder': None if cohort is None else source_name(cohort)},
        source='config attribute_ablation.yaml (Settings.coverage)',
    )
    return cohort


@timed
def _freeze_config(setup,cfg):
    """Point the settings at the template root and write the frozen yaml."""
    cfg.output_dir = str(_templates_dir(setup))
    frozen_config = setup/'ablation_settings.yaml'
    write_config = yaml.safe_dump(cfg.model_dump())
    frozen_config.write_text(write_config)
    return frozen_config


@timed
def _frozen_support(setup):
    """The training-population support records and prepared vocabulary."""
    with _LOG.section('ablation_support.load'):
        layout = _setup_layout()
        records = load_records(setup/layout.prepared_dir/layout.listings)
        graph_plan,graph_arrays = load_plan(setup/layout.prepared_dir/layout.listings,setup/layout.prepared_dir/'pairs.csv')
        graph_arrays.close()
        support = [records[n] for n in graph_plan['populations']['train']]
        vocabulary = graph_plan['vocabulary']
    trace().add(
        "prepare_suite", "support_vocabulary",
        in_count=len(records), out_count=len(support),
        reason='the frozen templates carry the TRAIN-population support and its vocabulary, fixed '
               'before any model is trained',
        detail={'listing_records': len(records), 'train_support': len(support),
                'vocabulary': len(vocabulary)},
        source=source_name(setup / layout.prepared_dir / layout.listings),
    )
    return support,vocabulary


@timed
def _template_checkpoint(setup,baseline,track,vocabulary,support):
    """The text baseline checkpoint, or the other tracks' template tensor file."""
    checkpoint = baseline
    if track != 'text':
        checkpoint = setup/(track+'__ablation_template.pt')
        payload = {'schema':'er-graph-checkpoint-v1','manifest':{'track':track,
            'text_metadata':{'checkpoint_sha256':checkpoint_hash(baseline),'composition':model_input_composition().model_dump(mode='json')}},
            'vocabulary':vocabulary,'support_records':support}
        torch.save(payload,checkpoint)
    return checkpoint


@timed
def _track_request(setup,checkpoint,track,*,cohort,frozen_config,baseline,composer=None,token_cache=None):
    """prepare() the track's tokens/tensors and read back its emitted request."""
    with _LOG.section('ablation_template.request'):
        layout = _setup_layout()
        path = prepare(cohort/'catalog.csv' if cohort else setup/layout.catalog,
            cohort/'pairs.csv' if cohort else setup/layout.prepared_dir/'pairs.csv',checkpoint,track=track,
            listings=(cohort/layout.listings if cohort else setup/layout.prepared_dir/layout.listings) if track != 'text' else None,
            text_checkpoint=None,config=frozen_config,
            composer=composer,token_cache=token_cache)
        request = json.loads(path.read_text())
    return path,request


@timed
def _track_cohort(track,request,common_cohort):
    """Freeze the suite cohort on the text track; every other must match it."""
    if track == 'text':
        return (request['cohort_sha256'], request['coverage'])
    if (request['cohort_sha256'], request['coverage']) != common_cohort:
        raise ValueError('all models must ablate exactly the same cohort and attributes')
    return common_cohort


@timed
def _anchor_request(setup,request):
    """Anchor prepared sources and shared inputs to the portable package."""
    # Anchor prepared sources to the package setup; checkpoint binding later
    # introduces a suite-relative selected weight, preserving frozen inputs.
    def anchor(name):
        source = resolve(name).resolve()
        if source.is_relative_to(setup.resolve()):
            return '@setup/'+source.relative_to(setup).as_posix()
        return name
    request['sources'] = {anchor(k):v for k,v in request['sources'].items()}
    request['checkpoint'] = anchor(request['checkpoint'])
    request['text_checkpoint'] = anchor(request['text_checkpoint']) if request['text_checkpoint'] else None
    request['portable_setup'] = package_member('suite_package_shared')


@timed
def _copy_template(setup,track,path,request):
    """Materialize the fixed template folder: tensors copy + frozen request."""
    target = _templates_dir(setup)/track
    target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path.parent/'prepared_inputs.npz',target/'prepared_inputs.npz')
    write(target/_request_name(),request)
    return target


@timed
def _track_template(setup,baseline,track,*,cohort,frozen_config,vocabulary,support,common_cohort,timing,composer=None,token_cache=None,graph_binding=_BINDING_UNSET):
    """One track's template: checkpoint, prepared request, anchors, folder copy."""
    with _LOG.section('ablation_template.checkpoint'):
        checkpoint = _template_checkpoint(setup,baseline,track,vocabulary,support)
    with _LOG.section('ablation_template.prepared_request'):
        path,request = _track_request(setup,checkpoint,track,cohort=cohort,
            frozen_config=frozen_config,baseline=baseline,composer=composer,token_cache=token_cache)
    with _LOG.section('ablation_template.cohort_validate'):
        common_cohort = _track_cohort(track,request,common_cohort)
    with _LOG.section('ablation_template.graph_binding'):
        if graph_binding is _BINDING_UNSET and track != 'text':
            graph_binding = digest({'vocabulary':vocabulary,'support_records':support})
        request['graph_binding'] = graph_binding if track != 'text' else None
    with _LOG.section('ablation_template.anchor_and_copy'):
        _anchor_request(setup,request)
        _copy_template(setup,track,path,request)
    timing.mark(track + '_tokens_tensors_and_request')
    return common_cohort


@timed
def _drop_staging(setup):
    """Remove generated content-addressed staging dirs; fixed templates stay."""
    # Generated content-addressed staging directories are temporary; retain one
    # fixed template per track and avoid shipping duplicate tensors.
    staging = [path for path in _templates_dir(setup).iterdir()
               if path.is_dir() and path.name not in {'text','gnn_only'}]
    for path in _LOG.progress(staging,desc='ablation_staging_cleanup',unit='dir'):
        shutil.rmtree(path)
    trace().add(
        "prepare_suite", "cleanup_staging",
        in_count=len(staging), out_count=0,
        reason='generated content-addressed staging dirs are removed; one fixed template per '
               'trained track is retained',
        detail={'staging_dirs': len(staging), 'retained': ['text', 'gnn_only'],
                'sample_removed': [source_name(path) for path in staging[:5]]},
        source=source_name(_templates_dir(setup)),
    )
    trace().add_entities(
        "prepare_suite.removed_staging", staging,
        key_of=lambda path: path.name,
        reason_of=lambda path: 'content_addressed_staging_dir',
        detail_of=lambda path: {'path': source_name(path)},
        source=source_name(_templates_dir(setup)),
    )


@timed
def prepare_suite(setup,baseline,config,*,composer=None,token_cache=None,bundle=None):
    """Fix native tokens and vocabulary/support topology before training exists."""
    timing = Timing('model_tracks.ablation_prepare')
    with _LOG.section('ablation_suite.freeze'):
        cohort,frozen_config = _freeze_suite(setup,config,bundle)
    with _LOG.section('ablation_suite.support_vocabulary'):
        support,vocabulary = _frozen_support(setup)
    timing.mark('load_support_and_vocabulary')
    with _LOG.section('ablation_suite.track_templates'):
        common_cohort = None
        tracks = ('text','gnn_only')
        graph_binding = digest({'vocabulary':vocabulary,'support_records':support})
        for track in _LOG.progress(tracks,desc='ablation_templates',unit='track',total=len(tracks)):
            _LOG.info('ablation template building track=' + track)
            common_cohort = _track_template(setup,baseline,track,cohort=cohort,
                frozen_config=frozen_config,vocabulary=vocabulary,
                support=support,common_cohort=common_cohort,timing=timing,
                composer=composer,token_cache=token_cache,graph_binding=graph_binding)
    with _LOG.section('ablation_suite.cleanup_staging'):
        _drop_staging(setup)
    timing.mark('cleanup_staging')
    trace().add(
        "prepare_suite", "completed",
        in_count=len(tracks), out_count=len(tracks),
        reason='one frozen template per trained track; the cascade trains nothing and ships none',
        detail={'tracks': list(tracks), 'templates': source_name(_templates_dir(setup)),
                'common_cohort': common_cohort},
        source=source_name(_templates_dir(setup)),
    )
    flush_trace()
    return _templates_dir(setup)


@timed
def _read_template(setup,track):
    """The track's frozen template request file."""
    template = _templates_dir(setup)/track
    return template,json.loads((template/_request_name()).read_text())


@timed
def _bind_staged_setup(setup,request):
    """Point the template request's shared inputs at the staged setup root."""
    from core.common import TRAIN_ROOT
    request['portable_setup'] = setup.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()


@timed
def _bind_template(setup,track):
    """The track's frozen template request, bound to the staged setup root."""
    template,request = _read_template(setup,track)
    _bind_staged_setup(setup,request)
    return template,request


@timed
def _check_graph_binding(checkpoint,track,request):
    """Reject a selected graph checkpoint that differs from frozen support."""
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    actual = digest({'vocabulary':payload['vocabulary'],'support_records':payload['support_records']})
    if actual != request['graph_binding'] or payload['manifest']['track'] != track:
        raise ValueError('selected graph checkpoint differs from frozen local support/vocabulary')


@timed
def _rebind_checkpoint(request,output,track,checkpoint,checkpoint_role):
    """Resolve the selected/baseline checkpoint role onto the request."""
    if checkpoint_role not in {'selected','baseline'}:
        raise ValueError('unknown ablation checkpoint role')
    old_checkpoint = request['checkpoint']
    identity = checkpoint_identity(checkpoint)
    if checkpoint_role == 'baseline':
        if track != 'text' or request['sources'][old_checkpoint] != identity:
            raise ValueError('baseline ablation differs from frozen text checkpoint')
        bound = old_checkpoint
    else:
        selected = '@suite/'+checkpoint.relative_to(output.parent).as_posix()
        request['sources'].pop(old_checkpoint)
        request['checkpoint'] = selected
        request['sources'][selected] = identity
        bound = selected
    request['checkpoint_role'] = checkpoint_role
    trace().add(
        "forward", "checkpoint_select",
        scope=SCOPE_ENTITY, in_count=1, out_count=1, key=track,
        reason=('the frozen baseline checkpoint is bound, never a trained one'
                if checkpoint_role == 'baseline' else
                'the trained selected checkpoint is bound as the ablated model'),
        detail={'role': checkpoint_role, 'checkpoint': source_name(checkpoint),
                'checkpoint_sha256': identity, 'bound_source': bound,
                'replaced_source': old_checkpoint},
        source=source_name(checkpoint),
    )


@timed
def _bound_folder(output,template,request):
    """Materialize the bound request and local tensors into the output folder."""
    folder = output/'ablation';folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(template/'prepared_inputs.npz',folder/'prepared_inputs.npz')
    path = folder/_request_name()
    write(path,request)
    return path,folder


@timed
def _saved_text_default(request,*,output,setup,track,saved_text):
    """Default to the suite's saved vectors for the full local retrieval catalog."""
    if saved_text is None and request['settings']['retrieval_catalog'] == 'full' and request['settings'].get('coverage') != 'all':
        saved_text = (output/'text__vectors.npz' if track == 'text' else None)
    return saved_text


@timed
def _encode_vectors(path,vectors,*,device,saved_text,text_model,graph_encoder):
    """The lane's ONLY device-executing call: the GPU-vector encode surface.

    A later CPU migration replaces the device plumbing here alone; callers
    and the device parameter threading upstream stay untouched.
    """
    encode(path,vectors,device=device,saved_text=saved_text,text_model=text_model,graph_encoder=graph_encoder)


@timed
def _reuse_or_encode(path,folder,request,*,output,setup,track,saved_text,text_model,graph_encoder,device):
    """Validated existing vectors win; otherwise the device owner encodes."""
    vectors = folder/'vectors.npz'
    existed = vectors.exists()
    if existed:
        validate_vectors(path,vectors)
    else:
        saved_text = _saved_text_default(request,output=output,setup=setup,
            track=track,saved_text=saved_text)
        _encode_vectors(path,vectors,device=device,saved_text=saved_text,
            text_model=text_model,graph_encoder=graph_encoder)
    # ``vectors_present`` is read AFTER the call: a lane whose encoder silently
    # produced nothing shows up here as present=false instead of a missing row.
    present = vectors.is_file()
    trace().add(
        "forward", "vectors",
        scope=SCOPE_ENTITY, in_count=1, out_count=1, key=track,
        reason=('an existing export was validated against its request and reused'
                if existed else
                'no valid export existed, so the lane encoded it on the frozen checkpoint'),
        detail={'vectors': source_name(vectors),
                'sha256': file_hash(vectors) if present else None,
                'bytes': vectors.stat().st_size if present else 0,
                'reused_existing_export': existed, 'vectors_present': present,
                'device': device,
                'checkpoint_role': request.get('checkpoint_role')},
        source=source_name(vectors),
    )


@timed
def forward(output,setup,track,checkpoint,*,device,text_model=None,checkpoint_role='selected',saved_text=None,graph_encoder=None):
    """Bind the selected/baseline checkpoint onto its template and encode vectors."""
    with _LOG.section('ablation_forward.bind_template'):
        template,request = _bind_template(setup,track)
    trace().add(
        "forward", "template",
        scope=SCOPE_ENTITY, in_count=1, out_count=1, key=track,
        reason='the track template frozen before training supplies the interventions and tensors',
        detail={'track': track, 'template': source_name(template),
                'portable_setup': request.get('portable_setup'),
                'cohort_sha256': request.get('cohort_sha256'),
                'variants': len(request.get('variants', []))},
        source=source_name(template / _request_name()),
    )
    if track != 'text':
        with _LOG.section('ablation_forward.graph_binding'):
            _check_graph_binding(checkpoint,track,request)
    with _LOG.section('ablation_forward.rebind_checkpoint'):
        _rebind_checkpoint(request,output,track,checkpoint,checkpoint_role)
    _LOG.info('ablation forward bound track=' + track + ' role=' + checkpoint_role)
    with _LOG.section('ablation_forward.write_bound_request'):
        path,folder = _bound_folder(output,template,request)
    with _LOG.section('ablation_forward.vectors'):
        _reuse_or_encode(path,folder,request,output=output,setup=setup,track=track,
            saved_text=saved_text,text_model=text_model,graph_encoder=graph_encoder,device=device)
    trace().add(
        "forward", "completed",
        # A UNIT row, not a funnel: one bound request comes out of this step, so
        # there is no in-vs-out attrition to state (a track with no variants
        # would otherwise report dropped_count=-1).
        scope=SCOPE_ENTITY, in_count=None, out_count=1, key=track,
        reason='the bound request and its encoded vectors are the lane output the report consumes',
        detail={'request_path': source_name(path), 'folder': source_name(folder),
                'device': device, 'checkpoint_role': checkpoint_role},
        source=source_name(path),
    )
    flush_trace()
    return path
