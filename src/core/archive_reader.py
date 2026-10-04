"""Read inventoried ZIP and Zstandard tar archives through one interface."""
from contextlib import contextmanager
import shutil
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import zipfile


def archive_sidecar(path, suffix):
    path = Path(path)
    for ending in ('.tar.zst', '.zip'):
        if path.name.endswith(ending):
            return path.with_name(path.name[:-len(ending)] + suffix)
    return path.with_suffix(suffix)


def zstd_module():
    try:
        from compression import zstd
    except ImportError:
        from backports import zstd
    return zstd


class TarReader:
    def __init__(self, archive):
        self.archive = archive

    def namelist(self):
        return self.archive.getnames()

    def getinfo(self, name):
        member = self.archive.getmember(name)
        return self._info(member)

    @staticmethod
    def _info(member):
        return SimpleNamespace(filename=member.name,
            external_attr=(0o100644 if member.isfile() else 0o120777) << 16,
            is_dir=lambda: member.isdir())

    def infolist(self):
        return [self._info(member) for member in self.archive.getmembers()]

    def open(self, name):
        member = self.archive.getmember(name)
        if not member.isfile():
            raise ValueError('archive member must be a regular file')
        return self.archive.extractfile(member)

    def read(self, name):
        with self.open(name) as handle:
            return handle.read()

    def extract(self, name, destination):
        target = Path(destination) / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe archive path')
        if not target.resolve().is_relative_to(Path(destination).resolve()):
            raise ValueError('unsafe archive extraction target')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError('archive extraction target is a symbolic link')
        with self.open(name) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)
        return str(target)

    def extractall(self, destination):
        for name in self.namelist():
            self.extract(name, destination)


@contextmanager
def open_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            yield archive
        return
    # Inflate once to a disk-backed seekable tar. Repeated reads then avoid
    # replaying the compressed stream for every checkpoint and report.
    with tempfile.TemporaryFile() as spool:
        zstd = zstd_module()
        try:
            with zstd.open(path, 'rb') as compressed:
                shutil.copyfileobj(compressed, spool)
        except zstd.ZstdError as error:
            raise ValueError(f'invalid Zstandard archive: {path}') from error
        spool.seek(0)
        with tarfile.open(fileobj=spool, mode='r:') as archive:
            yield TarReader(archive)
