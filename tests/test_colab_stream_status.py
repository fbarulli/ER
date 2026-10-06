import io
from types import SimpleNamespace

import pytest

from cli import colab


def _process(stdout, stderr):
    return SimpleNamespace(
        stdin=io.StringIO(), stdout=io.StringIO(stdout), stderr=io.StringIO(stderr),
        returncode=0, wait=lambda timeout=None: 0,
    )


def test_successful_notebook_ignores_local_cli_destructor_traceback(monkeypatch):
    process = _process('[repo] ready\n', 'ERROR:jupyter_kernel_client\nTraceback (most recent call last):\nAttributeError: KernelClient\n')
    monkeypatch.setattr(colab.subprocess, 'Popen', lambda *a, **kw: process)
    colab.run_colab_exec_stream('smoke', 'print("ready")', timeout=5)


def test_notebook_exception_fails_even_when_cli_returns_zero(monkeypatch):
    process = _process('Traceback (most recent call last):\nValueError: checkout failed\n', '')
    monkeypatch.setattr(colab.subprocess, 'Popen', lambda *a, **kw: process)
    with pytest.raises(RuntimeError, match='checkout failed'):
        colab.run_colab_exec_stream('smoke', 'raise ValueError()', timeout=5)


def test_recovered_step_trace_traceback_completes_with_zero_rc(monkeypatch):
    process = _process(
        '[traceback] copy_bundle.ioctl_clone\n'
        'Traceback (most recent call last):\n'
        '  File "src/core/step_trace.py", line 103, in trace_step\n'
        'OSError: [Errno 95] Operation not supported\n',
        '')
    monkeypatch.setattr(colab.subprocess, 'Popen', lambda *a, **kw: process)
    colab.run_colab_exec_stream('smoke', 'print("recovered")', timeout=5)


def test_unannotated_traceback_after_recovered_one_still_fails(monkeypatch):
    process = _process(
        '[traceback] copy_bundle.ioctl_clone\n'
        'Traceback (most recent call last):\n'
        'OSError: [Errno 95] Operation not supported\n'
        '\n'
        'fatal stage output\n'
        'Traceback (most recent call last):\n'
        'ValueError: packaging failed\n',
        '')
    monkeypatch.setattr(colab.subprocess, 'Popen', lambda *a, **kw: process)
    with pytest.raises(RuntimeError, match='packaging failed'):
        colab.run_colab_exec_stream('smoke', 'raise ValueError()', timeout=5)
