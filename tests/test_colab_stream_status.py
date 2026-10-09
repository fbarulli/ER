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


def test_notebook_exception_on_stderr_fails_even_when_cli_returns_zero(monkeypatch):
    # Upstream `colab exec` writes a notebook cell's `output_type == "error"`
    # block to the client's stderr (colab_cli/commands/execution.py
    # display_output), so a stdout-only scan missed the real remote failure and
    # the launcher went on to download an archive the failed stage never made.
    process = _process(
        '[bundle] uploading raw export\n',
        '\x1b[0;31m---------------------------------------------------------------------------\x1b[0m'
        '\x1b[0;31mRuntimeError\x1b[0m                              Traceback (most recent call last)\n'
        '\x1b[0;32m/tmp/ipykernel_1848/2958785101.py\x1b[0m in \x1b[0;36m<cell line: 0>\x1b[0;34m()\x1b[0m\n'
        '\x1b[0;31mRuntimeError\x1b[0m: prepare_all failed on the VM (rc=1)\n',
    )
    monkeypatch.setattr(colab.subprocess, 'Popen', lambda *a, **kw: process)
    with pytest.raises(RuntimeError, match='prepare_all failed on the VM'):
        colab.run_colab_exec_stream('smoke', 'raise RuntimeError()', timeout=5)


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
