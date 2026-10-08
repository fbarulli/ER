"""Read inventoried ZIP and Zstandard tar archives through one interface."""
from contextlib import contextmanager
import shutil
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import zipfile


def archive_settings():
    from core.common import training_cfg
    return training_cfg().archives


@contextmanager
def tar_archive(path, mode='r', *, settings=None):
    """Zstandard for new tar writers; spool once for repeated legacy/new reads."""
    settings = archive_settings() if settings is None else settings
    if mode in {'w', 'x'}:
        with zstd_module().open(path, mode + 'b', level=settings.compression_level) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', dereference=True,
                              bufsize=settings.copy_buffer_bytes,
                              copybufsize=settings.copy_buffer_bytes) as archive:
                yield archive
        return
    if mode != 'r':
        raise ValueError('tar archive mode must be r, w or x')
    with Path(path).open('rb') as handle:
        magic = handle.read(4)
    if magic != b'\x28\xb5\x2f\xfd':
        # Historical deliveries use gzip or an uncompressed tar.
        with tarfile.open(path, 'r:*') as archive:
            yield archive
        return
    with open_archive(path, settings=settings) as reader:
        yield reader.archive


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
    def __init__(self, archive, settings):
        self.archive = archive
        self.settings = settings

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
            shutil.copyfileobj(source, output, length=self.settings.copy_buffer_bytes)
        return str(target)

    def extractall(self, destination, *, filter=None):
        """Extract every surviving member through the safe per-member path.

        `filter` mirrors tarfile's extraction-filter protocol: a callable
        (tarfile.TarInfo) -> member-or-None; a falsy return drops the member.
        None keeps the unfiltered loop byte-identical. Caller class: the
        kaggle GPU train kernel's src-exclusion unpack of the verified bundle.
        """
        if filter is None:
            for name in self.namelist():
                self.extract(name, destination)
            return
        for member in self.archive.getmembers():
            if not filter(member):
                continue
            self.extract(member.name, destination)


@contextmanager
def open_archive(path, *, settings=None):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            yield archive
        return
    # Inflate once to a disk-backed seekable tar. Repeated reads then avoid
    # replaying the compressed stream for every checkpoint and report.
    settings = archive_settings() if settings is None else settings
    with tempfile.TemporaryFile() as spool:
        zstd = zstd_module()
        try:
            with zstd.open(path, 'rb') as compressed:
                shutil.copyfileobj(compressed, spool, length=settings.copy_buffer_bytes)
        except (zstd.ZstdError, EOFError) as error:
            raise ValueError(f'invalid Zstandard archive: {path}') from error
        spool.seek(0)
        with tarfile.open(fileobj=spool, mode='r:') as archive:
            yield TarReader(archive, settings)
