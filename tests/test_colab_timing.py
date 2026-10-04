"""Timing stays observable on successful and interrupted initialization."""
from unittest import mock

import pytest

from cli import colab


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_timing_reports_elapsed_and_preserves_failure(failure):
    with mock.patch.object(colab.time, "perf_counter", side_effect=[10.0, 12.5]), mock.patch("builtins.print") as output:
        if failure:
            with pytest.raises(failure):
                with colab._colab_timing("step", "initialization"):
                    raise failure()
        else:
            with colab._colab_timing("step", "initialization"):
                pass
    state = "failed" if failure else "completed"
    assert output.call_args_list == [
        mock.call("[timing] step=initialization state=started", flush=True),
        mock.call(f"[timing] step=initialization state={state} elapsed_seconds=2.500", flush=True),
    ]
