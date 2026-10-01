"""Forward durable worker logs without replaying already streamed lines."""
from pathlib import Path
import codecs


class WorkerLogs:
    def __init__(self, root: Path, tracks, *, from_end=False):
        self.paths = {track: root / f'{track}__worker.log' for track in tracks}
        self.offsets = {track: path.stat().st_size if from_end and path.exists() else 0
                        for track, path in self.paths.items()}
        self.pending = {track: '' for track in tracks}
        self.decoders = {track: codecs.getincrementaldecoder('utf-8')(errors='replace')
                         for track in tracks}

    def drain(self, *, final=False):
        for track, path in self.paths.items():
            if not path.is_file():
                continue
            with path.open('rb') as handle:
                handle.seek(self.offsets[track])
                chunk = handle.read()
                self.offsets[track] = handle.tell()
            text = self.pending[track] + self.decoders[track].decode(chunk, final=final)
            lines = text.splitlines(keepends=True)
            self.pending[track] = ''
            if lines and not lines[-1].endswith(('\n', '\r')) and not final:
                self.pending[track] = lines.pop()
            for line in lines:
                line = line.rstrip('\r\n')
                if line:
                    category = 'artifact' if line.startswith('[dvc') else 'track'
                    print(f'[{category}/{track}] {line}', flush=True)
