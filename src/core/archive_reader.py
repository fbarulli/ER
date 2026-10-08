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


class _HashingWriter:
    """Binary write-through wrapper recording the SHA256 of exactly the bytes written.

    A sealed archive's whole-file digest is the transport token (the ``.sha256``
    sidecar). Computing it here means the bytes are hashed as they are written,
    so no caller re-reads a multi-GB archive to produce that token.
    """

    def __init__(self, handle, digest):
        self._handle = handle
        self._digest = digest

    def write(self, data):
        self._digest.update(data)
        return self._handle.write(data)

    def flush(self):
        return self._handle.flush()

    def close(self):
        return self._handle.close()

    def fileno(self):
        return self._handle.fileno()

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return self._handle.seekable()

    def tell(self):
        return self._handle.tell()

    def __getattr__(self, name):
        return getattr(self._handle, name)


@contextmanager
def tar_archive(path, mode='r', *, settings=None, digest=None):
    """Zstandard for new tar writers; spool once for repeated legacy/new reads.

    ``digest`` (a ``hashlib``-style object) updates with the compressed bytes
    exactly as they are written, so the caller gets the sealed archive's
    whole-file SHA256 without reading it back.
    """
    settings = archive_settings() if settings is None else settings
    if mode in {'w', 'x'}:
        if digest is None:
            with zstd_module().open(path, mode + 'b', level=settings.compression_level) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', dereference=True,
                                  bufsize=settings.copy_buffer_bytes,
                                  copybufsize=settings.copy_buffer_bytes) as archive:
                    yield archive
            return
        with open(path, mode + 'b') as raw:
            with zstd_module().open(_HashingWriter(raw, digest), 'wb',
                                    level=settings.compression_level) as compressed:
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


#: The compressed-archive endings the ONE sidecar rule strips before appending
#: a companion suffix. Every real companion on disk is the stripped shape --
#: ``results/model_tracks/<run_tag>.sha256`` sits beside ``<run_tag>.tar.zst``
#: -- never ``<run_tag>.tar.zst.sha256``.
ARCHIVE_ENDINGS = ('.tar.zst', '.zip')


def archive_sidecar(path, suffix):
    """THE sidecar path rule: the ONE home for an archive's companion name.

    A companion of ``<name>.tar.zst`` (or ``.zip``) is ``<name><suffix>``:
    the declared archive ending (``ARCHIVE_ENDINGS``) is STRIPPED and ``suffix``
    appended in its place (``<run_tag>.tar.zst`` -> ``<run_tag>.sha256``). A
    name with no declared ending keeps its stem and replaces any other suffix
    (``Path.with_suffix``). Lanes, transports, kernel scripts and the
    fetched-output reader all import this rather than re-deriving
    ``name + suffix``, because the append shape names a file that exists
    beside no real archive.
    """
    path = Path(path)
    for ending in ARCHIVE_ENDINGS:
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
