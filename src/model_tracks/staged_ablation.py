"""Two phases: local interventions/tensors, selected-weight binding and GPU forward."""
import json
from pathlib import Path
import shutil
import torch
from graph_tracks.data import load_records
from graph_tracks.text_cache import checkpoint_hash
from model_tracks.ablation import prepare, settings, write, resolve, digest, checkpoint_identity, encode, request_context


def prepare_suite(setup,baseline,config,*,composer=None,token_cache=None,bundle=None):
    """Fix native tokens and vocabulary/support topology before training exists."""
    from core.model_input import model_input_composition
    from core.timing import Timing
    timing = Timing('model_tracks.ablation_prepare')
    cfg = settings(config)
    cohort = None
    if cfg.coverage == 'all':
        if bundle is None:
            raise ValueError('exhaustive ablation requires the prepared training bundle')
        from model_tracks.ablation_cohort import prepare_cohort
        cohort = prepare_cohort(setup, bundle)
    cfg.output_dir = str(setup/'ablation_templates')
    frozen_config = setup/'ablation_settings.yaml'
    write_config = __import__('yaml').safe_dump(cfg.model_dump())
    frozen_config.write_text(write_config)
    records = load_records(setup/'prepared/listings.json')
    from graph_tracks.prepared_inputs import load_plan
    graph_plan,graph_arrays = load_plan(setup/'prepared/listings.json',setup/'prepared/pairs.csv')
    graph_arrays.close()
    support = [records[n] for n in graph_plan['populations']['train']]
    vocabulary = graph_plan['vocabulary']
    timing.mark('load_support_and_vocabulary')
    for track in ('text','gnn_only','hybrid'):
        checkpoint = baseline
        if track != 'text':
            checkpoint = setup/(track+'__ablation_template.pt')
            payload = {'schema':'er-graph-checkpoint-v1','manifest':{'track':track,
                'text_metadata':{'checkpoint_sha256':checkpoint_hash(baseline),'composition':model_input_composition().model_dump(mode='json')}},
                'vocabulary':vocabulary,'support_records':support}
            torch.save(payload,checkpoint)
        path = prepare(cohort/'catalog.csv' if cohort else setup/'eligible_catalog.csv',
            cohort/'pairs.csv' if cohort else setup/'prepared/pairs.csv',checkpoint,track=track,
            listings=(cohort/'listings.json' if cohort else setup/'prepared/listings.json') if track != 'text' else None,
            text_checkpoint=baseline if track == 'hybrid' else None,config=frozen_config,
            composer=composer,token_cache=token_cache)
        request = json.loads(path.read_text())
        if track == 'text':
            common_cohort = (request['cohort_sha256'], request['coverage'])
        elif (request['cohort_sha256'], request['coverage']) != common_cohort:
            raise ValueError('all models must ablate exactly the same cohort and attributes')
        request['graph_binding'] = digest({'vocabulary':vocabulary,'support_records':support}) if track != 'text' else None
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
        request['portable_setup'] = 'data/model_tracks/shared'
        target = setup/'ablation_templates'/track
        target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path.parent/'prepared_inputs.npz',target/'prepared_inputs.npz')
        write(target/'request.json',request)
        timing.mark(track + '_tokens_tensors_and_request')
    # Generated content-addressed staging directories are temporary; retain one
    # fixed template per track and avoid shipping duplicate tensors.
    for path in (setup/'ablation_templates').iterdir():
        if path.is_dir() and path.name not in {'text','gnn_only','hybrid'}:
            shutil.rmtree(path)
    timing.mark('cleanup_staging')
    return setup/'ablation_templates'


def forward(output,setup,track,checkpoint,*,device,text_model=None,checkpoint_role='selected',saved_text=None,graph_encoder=None):
    from core.common import TRAIN_ROOT
    template = setup/'ablation_templates'/track
    request = json.loads((template/'request.json').read_text())
    # Bind the actual staged setup for both direct suites and portable workers.
    request['portable_setup'] = setup.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()
    if track != 'text':
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        actual = digest({'vocabulary':payload['vocabulary'],'support_records':payload['support_records']})
        if actual != request['graph_binding'] or payload['manifest']['track'] != track:
            raise ValueError('selected graph checkpoint differs from frozen local support/vocabulary')
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
    folder = output/'ablation';folder.mkdir(parents=True,exist_ok=True)
    shutil.copy2(template/'prepared_inputs.npz',folder/'prepared_inputs.npz')
    path = folder/'request.json'
    write(path,request)
    vectors = folder/'vectors.npz'
    if vectors.exists():
        from model_tracks.ablation import validate_vectors
        validate_vectors(path,vectors)
    else:
        if saved_text is None and request['settings']['retrieval_catalog'] == 'full' and request['settings'].get('coverage') != 'all':
            saved_text = (output/'text__vectors.npz' if track == 'text' else
                          setup/'shared_minilm__embeddings.npz' if track == 'hybrid' else None)
        encode(path,vectors,device=device,saved_text=saved_text,text_model=text_model,graph_encoder=graph_encoder)
    return path
