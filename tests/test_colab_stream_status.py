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
