import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from model_tracks.parallel import run_parallel


def test_all_workers_overlap_and_outputs_are_isolated(tmp_path):
    code = '''import json, os, pathlib, time
from model_tracks.parallel import wait_for_start
track=os.environ['ER_TRACK_NAME']
root=pathlib.Path(os.environ['EUROMONITOR_RESULTS_DIR'])
wait_for_start(pathlib.Path(os.environ['ER_TRACK_BARRIER']),track,timeout=10)
start=time.time()
time.sleep(.4)
(root/'result.json').write_text(json.dumps({'track':track,'start':start,'end':time.time(),'wandb':os.environ['WANDB_DIR']}))
'''
    commands={track:[sys.executable,'-c',code] for track in ('text','gnn_only','hybrid')}
    result=run_parallel(commands,tmp_path,{**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},barrier_timeout=10)
    rows=[json.loads((tmp_path/track/'result.json').read_text()) for track in commands]
    assert max(row['start'] for row in rows)<min(row['end'] for row in rows)
    assert len({row['wandb'] for row in rows})==3
    assert result['mode']=='parallel'


def test_failed_worker_aborts_whole_suite(tmp_path):
    code='''import os,pathlib,time
from model_tracks.parallel import wait_for_start
track=os.environ['ER_TRACK_NAME']
wait_for_start(pathlib.Path(os.environ['ER_TRACK_BARRIER']),track,timeout=10)
if track=='gnn_only': raise SystemExit(3)
time.sleep(30)
'''
    commands={track:[sys.executable,'-c',code] for track in ('text','gnn_only','hybrid')}
    with pytest.raises(RuntimeError,match='workers failed'):
        run_parallel(commands,tmp_path,{**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},barrier_timeout=10)


def test_existing_auth_is_reused_without_copying_tokens(tmp_path,monkeypatch):
    entry=Path(__file__).resolve().parents[1]/'src/cli/colab_cli_entry.py'
    spec=importlib.util.spec_from_file_location('entry_under_test',entry)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project=tmp_path/'project'
    project.mkdir()
    upstream=tmp_path/'global_token.json'
    upstream.write_text('test token placeholder')
    monkeypatch.setattr(module,'STATE_DIR',project)
    assert module._token_config_path(str(upstream))==str(upstream)
    assert not (project/'token.json').exists()
    (project/'token.json').write_text('project placeholder')
    assert module._token_config_path(str(upstream))==str(project/'token.json')
