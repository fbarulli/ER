import importlib.util
import json
import sys
from pathlib import Path

def test_resume_distinguishes_input_formats_and_skips_only_successes(tmp_path, monkeypatch):
    root=Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root/'jev'))
    spec=importlib.util.spec_from_file_location('jev_audit_runner_test',root/'jev/run_audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checkpoint=tmp_path/'results.jsonl'
    checkpoint.write_text('\n'.join(json.dumps(x) for x in [
        {'gtin1':'1','gtin2':'2','input_scope':'gate_data','status':'ok'},
        {'gtin1':'1','gtin2':'2','input_scope':'original_data','status':'error'},
        {'gtin1':'2','gtin2':'1','input_scope':'original_data','status':'ok'},
    ]))
    done=module.load_done(checkpoint)
    assert ('gate_data','1','2') in done
    assert ('original_data','1','2') not in done
    assert ('original_data','2','1') in done
