"""Timing stays observable on successful and interrupted initialization."""
from datetime import datetime as real_datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pytest
import re

from cli import colab


def _fixed_paris_datetime():
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 7, 9, 13, 28, tzinfo=ZoneInfo("Europe/Paris"))

    return FixedDatetime


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_timing_reports_elapsed_and_preserves_failure(failure):
    with mock.patch.object(colab.time, "perf_counter", side_effect=[10.0, 12.5]), \
            mock.patch.object(colab, "datetime", _fixed_paris_datetime()), \
            mock.patch("builtins.print") as output:
        if failure:
            with pytest.raises(failure):
                with colab._colab_timing("step", "initialization"):
                    raise failure()
        else:
            with colab._colab_timing("step", "initialization"):
                pass
    state = "failed" if failure else "completed"
    assert output.call_args_list == [
        mock.call("[colab 2026-10-07T09:13:28 CEST] "
                  "[timing] step=initialization state=started", flush=True),
        mock.call("[colab 2026-10-07T09:13:28 CEST] [timing] step=initialization "
                  f"state={state} elapsed_seconds=2.500", flush=True),
    ]


def test_stamp_matches_paris_local_format():
    assert re.fullmatch(
        r"\[colab \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} (CET|CEST)\]",
        colab._stamp())
