"""Bounded-cost suite resource monitoring; GPU samples are aggregate under MPS."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


class ResourceProfile:
    def __init__(self, output, enabled=False):
        self.output = Path(output)
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.process = None
        self.thread = None
        self.gpu_handle = None

    def start(self):
        if not self.enabled:
            return self
        self.output.mkdir(parents=True, exist_ok=True)
        binary = shutil.which('nvidia-smi')
        status = 'unavailable'
        if binary:
            self.gpu_handle = (self.output / 'gpu.csv').open('a')
            try:
                self.process = subprocess.Popen([binary,
                    '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                    '--format=csv,nounits', '--loop=1'], stdout=self.gpu_handle,
                    stderr=subprocess.DEVNULL)
                status = 'aggregate samples; unsupported hardware fields may be N/A'
            except OSError:
                self.gpu_handle.close()
                self.gpu_handle = None
        (self.output / 'manifest.json').write_text(json.dumps({
            'gpu_interval_seconds': 1, 'cpu_interval_seconds': 2,
            'gpu_attribution': 'whole device; MPS clients are not attributed',
            'gpu_status': status, 'cpu_scope': 'supervisor and descendants',
            'overhead': 'sampling and file writes; no CUDA synchronization'}, indent=2))
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    @staticmethod
    def processes(root_pid):
        records = {}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                raw = (path / 'stat').read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                record = {'pid': int(path.name), 'ppid': int(fields[1]),
                          'cpu_ticks': int(fields[11]) + int(fields[12]),
                          'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
                try:
                    record['io'] = {key: int(value) for key, value in
                                    (line.split(':', 1) for line in (path / 'io').read_text().splitlines())
                                    if key in {'read_bytes', 'write_bytes'}}
                except (OSError, ValueError):
                    pass
                records[record['pid']] = record
            except (OSError, ValueError, IndexError):
                continue
        owned = {root_pid}
        while True:
            expanded = owned | {pid for pid, record in records.items() if record['ppid'] in owned}
            if expanded == owned:
                break
            owned = expanded
        return [records[pid] for pid in sorted(owned) if pid in records]

    def _sample(self):
        with (self.output / 'cpu.jsonl').open('a', buffering=1) as handle:
            while True:
                try:
                    sample = {'timestamp_unix': time.time(), 'monotonic_seconds': time.monotonic(),
                              'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'),
                              'processes': self.processes(os.getpid())}
                    handle.write(json.dumps(sample) + '\n')
                except (OSError, ValueError):
                    pass
                if self.stop_event.wait(2):
                    break

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.gpu_handle:
            self.gpu_handle.close()
            self.gpu_handle = None
