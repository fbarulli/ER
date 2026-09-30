import json

import torch

from core.training_profiler import TrainingProfiler


def test_cpu_profile_exports_trace_and_operator_summary(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_TRAINING_PROFILE', '1')
    with TrainingProfiler(tmp_path / 'profile', 'cpu') as profiler:
        for _ in range(3):
            value = torch.ones(4, requires_grad=True)
            with profiler.section('smoke_backward'):
                value.square().sum().backward()
            profiler.step()
    profiler.close()
    trace = json.loads((tmp_path / 'profile/training_trace.json').read_text())
    assert any(event.get('name') == 'smoke_backward' for event in trace['traceEvents'])
    assert 'aten::' in (tmp_path / 'profile/operator_summary.txt').read_text()
    assert json.loads((tmp_path / 'profile/profile_manifest.json').read_text())['device'] == 'cpu'


def test_disabled_profile_creates_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv('ER_TRAINING_PROFILE', raising=False)
    output = tmp_path / 'profile'
    with TrainingProfiler(output, 'cpu') as profiler:
        profiler.call('operation', torch.ones, 2)
        profiler.step()
    assert not output.exists()
