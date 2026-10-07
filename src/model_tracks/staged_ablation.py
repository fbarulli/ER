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
from core.run_log import RunLogger
from core.timing import Timing
from training.prepare_all_trace import timed
from graph_tracks.data import load_records
from graph_tracks.prepared_inputs import load_plan
from model_tracks.ablation import prepare, settings, write, resolve, checkpoint_identity, encode, request_context, validate_vectors
from model_tracks.package import package_member

_LOG = RunLogger(__name__)


def _freeze_suite(setup,config,bundle):
    """Resolve ablation settings, gate the cohort and freeze the template root."""
    cfg = settings(config)
    cohort = _cohort_gate(setup,cfg,bundle)
    frozen_config = _freeze_config(setup,cfg)
    return cohort,frozen_config


def _cohort_gate(setup,cfg,bundle):
    """The exhaustive-coverage cohort, or None; loud when 'all' lacks a bundle."""
    cohort = None
    if cfg.coverage == 'all':
        if bundle is None:
            raise ValueError('exhaustive ablation requires the prepared training bundle')
        from model_tracks.ablation_cohort import prepare_cohort
        cohort = prepare_cohort(setup, bundle)
    return cohort


def _freeze_config(setup,cfg):
    """Point the settings at the template root and write the frozen yaml."""
    cfg.output_dir = str(setup/'ablation_templates')
    frozen_config = setup/'ablation_settings.yaml'
    write_config = yaml.safe_dump(cfg.model_dump())
    frozen_config.write_text(write_config)
    return frozen_config


def _frozen_support(setup):
    """The training-population support records and prepared vocabulary."""
    records = load_records(setup/'prepared/listings.json')
    graph_plan,graph_arrays = load_plan(setup/'prepared/listings.json',setup/'prepared/pairs.csv')
    graph_arrays.close()
    support = [records[n] for n in graph_plan['populations']['train']]
    vocabulary = graph_plan['vocabulary']
    return support,vocabulary


def _template_checkpoint(setup,baseline,track,vocabulary,support):
    """The text baseline checkpoint, or the other tracks' template tensor file."""
    checkpoint = baseline
    if track != 'text':
        checkpoint = setup/(track+'__ablation_template.pt')
        payload = {'schema':'er-graph-checkpoint-v1','manifest':{'track':track,
            'text_metadata':{}},
            'vocabulary':vocabulary,'support_records':support}
        torch.save(payload,checkpoint)
    return checkpoint


def _track_request(setup,checkpoint,track,*,cohort,frozen_config,baseline,composer=None,token_cache=None):
    """prepare() the track's tokens/tensors and read back its emitted request."""
    path = prepare(cohort/'catalog.csv' if cohort else setup/'eligible_catalog.csv',
        cohort/'pairs.csv' if cohort else setup/'prepared/pairs.csv',checkpoint,track=track,
        listings=(cohort/'listings.json' if cohort else setup/'prepared/listings.json') if track != 'text' else None,
        text_checkpoint=baseline if track == 'hybrid' else None,config=frozen_config,
        composer=composer,token_cache=token_cache)
    request = json.loads(path.read_text())
    return path,request


def _track_cohort(track,request,common_cohort):
    """Freeze the suite cohort on the text track; every other must match it."""
    if track == 'text':
        return (request['cohort_sha256'], request['coverage'])
    if (request['cohort_sha256'], request['coverage']) != common_cohort:
        raise ValueError('all models must ablate exactly the same cohort and attributes')
    return common_cohort


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


def _copy_template(setup,track,path,request):
    """Materialize the fixed template folder: tensors copy + frozen request."""
    target = setup/'ablation_templates'/track
    target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path.parent/'prepared_inputs.npz',target/'prepared_inputs.npz')
    write(target/'request.json',request)
    return target


def _track_template(setup,baseline,track,*,cohort,frozen_config,vocabulary,support,common_cohort,timing,composer=None,token_cache=None):
    """One track's template: checkpoint, prepared request, anchors, folder copy."""
    checkpoint = _template_checkpoint(setup,baseline,track,vocabulary,support)
    path,request = _track_request(setup,checkpoint,track,cohort=cohort,
        frozen_config=frozen_config,baseline=baseline,composer=composer,token_cache=token_cache)
    common_cohort = _track_cohort(track,request,common_cohort)
    request['graph_binding'] = track if track != 'text' else None
    _anchor_request(setup,request)
    _copy_template(setup,track,path,request)
    timing.mark(track + '_tokens_tensors_and_request')
    return common_cohort


def _drop_staging(setup):
    """Remove generated content-addressed staging dirs; fixed templates stay."""
    # Generated content-addressed staging directories are temporary; retain one
    # fixed template per track and avoid shipping duplicate tensors.
    staging = [path for path in (setup/'ablation_templates').iterdir()
               if path.is_dir() and path.name not in {'text','gnn_only','hybrid'}]
    for path in _LOG.progress(staging,desc='ablation_staging_cleanup',unit='dir'):
        shutil.rmtree(path)


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
        tracks = ('text','gnn_only','hybrid')
        for track in _LOG.progress(tracks,desc='ablation_templates',unit='track',total=len(tracks)):
            _LOG.info('ablation template building track=' + track)
            common_cohort = _track_template(setup,baseline,track,cohort=cohort,
                frozen_config=frozen_config,vocabulary=vocabulary,
                support=support,common_cohort=common_cohort,timing=timing,
                composer=composer,token_cache=token_cache)
    with _LOG.section('ablation_suite.cleanup_staging'):
        _drop_staging(setup)
    timing.mark('cleanup_staging')
    return setup/'ablation_templates'


def _read_template(setup,track):
    """The track's frozen template request file."""
    template = setup/'ablation_templates'/track
    return template,json.loads((template/'request.json').read_text())


def _bind_staged_setup(setup,request):
    """Point the template request's shared inputs at the staged setup root."""
    from core.common import TRAIN_ROOT
    request['portable_setup'] = setup.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()


def _bind_template(setup,track):
    """The track's frozen template request, bound to the staged setup root."""
    template,request = _read_template(setup,track)
    _bind_staged_setup(setup,request)
    return template,request


def _check_graph_binding(checkpoint,track,request):
    """Reject a selected graph checkpoint that differs from frozen support."""
    if request['graph_binding'] != track:
        raise ValueError('selected graph checkpoint differs from frozen local support/vocabulary')


def _rebind_checkpoint(request,output,track,checkpoint,checkpoint_role):
    """Resolve the selected/baseline checkpoint role onto the request."""
    if checkpoint_role not in {'selected','baseline'}:
        raise ValueError('unknown ablation checkpoint role')
    old_checkpoint = request['checkpoint']
    if checkpoint_role == 'baseline':
        if track != 'text' or request['sources'][old_checkpoint] != checkpoint_identity(checkpoint):
            raise ValueError('baseline ablation differs from frozen text checkpoint')
    else:
        selected = '@suite/'+checkpoint.relative_to(output.parent).as_posix()
        request['sources'].pop(old_checkpoint)
        request['checkpoint'] = selected
        request['sources'][selected] = checkpoint_identity(checkpoint)
    request['checkpoint_role'] = checkpoint_role


def _bound_folder(output,template,request):
    """Materialize the bound request and local tensors into the output folder."""
    folder = output/'ablation';folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(template/'prepared_inputs.npz',folder/'prepared_inputs.npz')
    path = folder/'request.json'
    write(path,request)
    return path,folder


def _saved_text_default(request,*,output,setup,track,saved_text):
    """Default to the suite's saved vectors for the full local retrieval catalog."""
    if saved_text is None and request['settings']['retrieval_catalog'] == 'full' and request['settings'].get('coverage') != 'all':
        saved_text = (output/'text__vectors.npz' if track == 'text' else
                      setup/'shared_minilm__embeddings.npz' if track == 'hybrid' else None)
    return saved_text


def _encode_vectors(path,vectors,*,device,saved_text,text_model,graph_encoder):
    """The lane's ONLY device-executing call: the GPU-vector encode surface.

    A later CPU migration replaces the device plumbing here alone; callers
    and the device parameter threading upstream stay untouched.
    """
    encode(path,vectors,device=device,saved_text=saved_text,text_model=text_model,graph_encoder=graph_encoder)


def _reuse_or_encode(path,folder,request,*,output,setup,track,saved_text,text_model,graph_encoder,device):
    """Validated existing vectors win; otherwise the device owner encodes."""
    vectors = folder/'vectors.npz'
    if vectors.exists():
        validate_vectors(path,vectors)
    else:
        saved_text = _saved_text_default(request,output=output,setup=setup,
            track=track,saved_text=saved_text)
        _encode_vectors(path,vectors,device=device,saved_text=saved_text,
            text_model=text_model,graph_encoder=graph_encoder)


@timed
def forward(output,setup,track,checkpoint,*,device,text_model=None,checkpoint_role='selected',saved_text=None,graph_encoder=None):
    """Bind the selected/baseline checkpoint onto its template and encode vectors."""
    with _LOG.section('ablation_forward.bind_template'):
        template,request = _bind_template(setup,track)
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
    return path
