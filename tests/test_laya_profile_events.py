"""Pin reporting against real PyTorch event objects, not CUDA-name stubs."""
from types import SimpleNamespace

import torch

from core.laya_controls import ProfilerSession


class TestProfilerEvents:
    def test_device_time_is_reported_and_sorted(self) -> None:
        events = []
        for name, duration in (("small", 1000.0), ("large", 9000.0)):
            event = torch.autograd.profiler_util.FunctionEventAvg()
            event.key = name
            event.count = 1
            event.self_device_time_total = duration
            event.self_cpu_time_total = 2000.0
            events.append(event)
        session = ProfilerSession(torch, enabled=False, trace_dir=None,
                                  schedule=None)
        rows = session.top_ops(SimpleNamespace(key_averages=lambda: events))
        assert rows == [("large", 9.0, 2.0, 1), ("small", 1.0, 2.0, 1)]
