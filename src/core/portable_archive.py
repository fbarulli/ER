"""Shared name+size inventories for prepared-input and result archives."""
from __future__ import annotations
import json
import os
import uuid
import time
import io
import tarfile
from pathlib import Path
import zipfile
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator
from core.archive_reader import zstd_module, archive_sidecar, archive_settings, tar_archive
from core.perf_switches import perf_enabled
from core.progress import tracked
from core.step_trace import timed, trace_step


def inventory_key_home() -> str:
    """The ONE sealed-archive inventory key: config's ``bundle.files_key``.

    Every archive writer/verifier below resolves its default ``inventory_key``
    through here instead of the old hardcoded ``'files'``, so the key is
    config-owned: changing ``bundle.files_key`` steers both the manifest a seal
    writes and the key a boundary verify reads, and the two can never disagree.
    ``core.common`` is imported lazily (it imports the config that declares the
    value, so a module-level import would be circular); a caller that needs a
    different key (a legacy manifest shape) still passes ``inventory_key``
    explicitly, which wins.
    """
    from core.common import training_cfg
    return training_cfg().bundle.files_key


class ByteCount:
    """Accumulates the number of bytes fed to it. NEVER a content digest.

    The structural stand-in for the retired streaming accumulator: callers that
    used to feed bytes into a digest object and read a fixed-width token now
    feed the same bytes here and read the byte total. Identity in this
    repository is names + byte sizes (owner directive 2026-10-08), so a total
    is exactly what those call sites are allowed to carry.
    """

    __slots__ = ('total',)

    def __init__(self, initial: bytes | str = b'') -> None:
        self.total = len(initial)

    def update(self, data) -> None:
        self.total += len(data)

    def __int__(self) -> int:
        return self.total

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f'ByteCount({self.total})'


def file_size(path: Path | str) -> int:
    """The ONE structural size accessor: bytes on disk.

    A regular file reports its ``st_size``. A directory reports the summed
    ``st_size`` of the regular files under it (the graph/checkpoint identity
    the lanes need without reading a byte). Content is NEVER read or
    fingerprinted anywhere in the repository (owner directive 2026-10-08).
    """
    path = Path(path)
    if path.is_dir():
        return sum(member.stat().st_size for member in path.rglob('*')
                   if member.is_file() and not member.is_symlink())
    return path.stat().st_size


class RuntimeSnapshot(BaseModel):
    """Source/config files shared by packaging and recovery identity."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    files: dict[str, Path] = Field(min_length=1)

    @model_validator(mode='after')
    def check_files(self):
        for relative, path in self.files.items():
            member = Path(relative)
            if not relative or member.is_absolute() or '..' in member.parts:
                raise ValueError(f'unsafe runtime snapshot member: {relative}')
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'runtime snapshot requires a regular source file: {path}')
        return self

    def inventory(self) -> dict[str, int]:
        return {relative: path.stat().st_size for relative, path in self.files.items()}


# Recompressing these containers wastes CPU and rarely saves meaningful space.
_COMPRESSED_SUFFIXES = frozenset({'.gz', '.bz2', '.xz', '.zst', '.zip', '.npz',
                                  '.png', '.jpg', '.jpeg', '.webp', '.parquet'})


#: Repository neighborhoods owned by the pinned checkout. A prepared bundle is
#: built at bundle time and its embedded snapshot can predate the checkout's
#: revision; installing those members over the checkout clobbers code/config and
#: crashes on a layout key added after the bundle was built
#: (KeyError: 'source_code_dir'). Only bundle DATA may be installed.
CHECKOUT_AUTHORITATIVE_PREFIXES = ('src/', 'config/', 'scripts/')


def is_checkout_authoritative(name: str) -> bool:
    """True when installing ``name`` would overwrite checkout-owned code/config.

    Matches both the directory member itself (``config``) and every descendant
    (``config/paths.yaml``) so a bundle can never shadow a neighborhood.
    """
    member = name.rstrip('/')
    return any(member == prefix.rstrip('/') or name.startswith(prefix)
               for prefix in CHECKOUT_AUTHORITATIVE_PREFIXES)


def install_data_members(archive, root) -> list[str]:
    """Install only an archive's DATA members under ``root``.

    Members inside :data:`CHECKOUT_AUTHORITATIVE_PREFIXES` are skipped so the
    pinned checkout stays authoritative for code, config and scripts; every
    other member (data, artifacts, inline configs under the declared layouts)
    is extracted, with directory members created explicitly. Returns the
    installed file member names so a caller can log or pin what the bundle
    contributed. Dependency-free: works for the ZIP and tar readers alike.
    """
    root = Path(root)
    installed: list[str] = []
    for info in archive.infolist():
        name = info.filename
        if is_checkout_authoritative(name):
            continue
        if info.is_dir():
            (root / name).mkdir(parents=True, exist_ok=True)
            continue
        archive.extract(name, root)
        installed.append(name)
    return installed


def _write_zip(candidate, files, inline, manifest_name, manifest):
    """Stream standard ZIP64 with fast deflate; copy compressed payloads as-is."""
    with zipfile.ZipFile(candidate, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=archive_settings().legacy_zip_level, allowZip64=True) as archive:
        for target, source in tracked(files.items(), desc='archive.zip_files'):
            compression = (zipfile.ZIP_STORED if source.suffix.lower() in _COMPRESSED_SUFFIXES
                           else zipfile.ZIP_DEFLATED)
            archive.write(source, target, compress_type=compression, compresslevel=archive_settings().legacy_zip_level)
        for target, value in inline.items():
            archive.writestr(target, value)
        archive.writestr(manifest_name, manifest)


class _CountingReader:
    """File-like tee that counts exactly the bytes read."""

    def __init__(self, handle, counter):
        self._handle = handle
        self._counter = counter

    def read(self, size=-1):
        data = self._handle.read(size)
        if data:
            self._counter['bytes'] += len(data)
        return data

    def close(self):
        self._handle.close()


def _count_member_bytes(handle, chunk_bytes: int) -> int:
    """Bytes actually readable from one archive member (no content identity)."""
    total = 0
    while chunk := handle.read(chunk_bytes):
        total += len(chunk)
    return total


def _write_tar(candidate, files, inline, manifest_name, manifest, *,
               sizes: dict[str, int] | None = None, counter=None):
    """Stream a zstd tar. When ``sizes`` is given, record each member's byte count
    from the bytes handed to the writer, so the caller need not re-read the
    archive to prove it matches its frozen inventory. ``counter`` records the
    whole-file byte count of the compressed archive as it is written."""
    with tar_archive(candidate, 'x', counter=counter) as archive:
        for target, source in tracked(files.items(), desc='archive.tar_files'):
            if sizes is None:
                archive.add(source, arcname=target, recursive=False)
                continue
            info = archive.gettarinfo(str(source), arcname=target)
            written = {'bytes': 0}
            with source.open('rb') as handle:
                archive.addfile(info, _CountingReader(handle, written))
            sizes[target] = written['bytes']
        for target, value in inline.items():
            _add_tar_text(archive, target, value)
        _add_tar_text(archive, manifest_name, manifest)


def _add_tar_text(archive, name, value):
    payload = value.encode()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def source_inventory(files: dict[str, Path], inline: dict[str, str]) -> dict[str, int]:
    """The ONE name -> byte-size inventory builder for archive sources.

    Regular-file check then size: every source once, inline text byte counts too.
    ``write_archive`` inventories sources through here while it writes, and callers
    that must mirror that inventory for a legacy manifest (the graph worker
    package's historical ``files_size`` key) reuse it instead of re-stat-ing
    the same sources with a second loop.
    """
    inventory: dict[str, int] = {}
    for target, source in tracked(files.items(), desc='archive.size_sources'):
        if source.is_symlink() or not source.is_file():
            raise ValueError('archive requires regular files, not symbolic links')
        inventory[target] = source.stat().st_size
    inventory.update({target: len(value.encode())
                      for target, value in inline.items()})
    return inventory


def _candidate_path(output: Path) -> Path:
    """One unique unpublished staging sibling for an atomic publish."""
    return output.with_name(f'{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}')


def _publish_atomically(candidate: Path, output: Path) -> None:
    """Link the completed staging sibling; the name never races with a reader."""
    with candidate.open('rb') as handle:
        os.fsync(handle.fileno())
    # Linking the completed sibling publishes atomically without replacing
    # an existing immutable generation, even when two writers race.
    os.link(candidate, output)
    candidate.unlink(missing_ok=True)


def _profile_sidecar(output: Path, timings: dict[str, float],
                     files: dict[str, Path]) -> None:
    """The run-side cost profile (size/compress/verify seconds + sizes)."""
    archive_sidecar(output, ".profile.json").write_text(json.dumps({
        **timings, "timestamp_unix": time.time(),
        "archive_bytes": output.stat().st_size,
        "source_bytes": sum(source.stat().st_size for source in files.values()),
        "file_count": len(files)}, indent=2) + "\n")


@timed
def write_archive(output: Path, files: dict[str, Path], *, manifest_name: str,
                  metadata: dict[str, Any], inline: dict[str, str] | None = None,
                  inventory_key: str | None = None, profile: bool = False) -> Path:
    """Publish one size-inventoried archive (staging → verify → atomic link).

    ``inventory_key`` defaults to the ONE config home (``bundle.files_key``, see
    :func:`inventory_key_home`).
    """
    if output.exists():
        raise FileExistsError(output)
    inventory_key = inventory_key or inventory_key_home()
    inline = inline or {}
    if set(files) & set(inline) or manifest_name in files or manifest_name in inline:
        raise ValueError('archive member collision')
    timings = {}
    for target in files:
        _check_member(target)
    for target in inline:
        _check_member(target)
    _check_member(manifest_name)
    started = time.monotonic()
    # Reject malformed inventories before expensive source sizing.
    inventory = source_inventory(files, inline)
    timings["inventory_seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = _candidate_path(output)
    try:
        started = time.monotonic()
        manifest = json.dumps({**metadata, inventory_key: inventory}, indent=2) + '\n'
        stream_sizes: dict[str, int] | None = (
            {} if perf_enabled('archive.write_size') else None)
        if output.name.endswith('.tar.zst'):
            with trace_step('archive.zstandard', files=len(files),
                            compression_level=archive_settings().compression_level):
                _write_tar(candidate, files, inline, manifest_name, manifest,
                           sizes=stream_sizes)
        else:
            # Explicit ZIP outputs remain available for historical callers.
            stream_sizes = None
            with trace_step('archive.zip', files=len(files)):
                _write_zip(candidate, files, inline, manifest_name, manifest)
        # Sources can change while being archived (e.g. checkpoint rotation).
        # Never publish an archive whose bytes disagree with its frozen inventory.
        timings["compression_seconds"] = time.monotonic() - started
        started = time.monotonic()
        if stream_sizes is not None:
            # The writer counted every member payload as it wrote it; comparing
            # those byte counts to the frozen inventory is the whole integrity
            # check, so the archive is not read and inflated a second time.
            with trace_step('archive.verify_written'):
                for target in files:
                    if stream_sizes.get(target) != inventory[target]:
                        raise ValueError(f'archive integrity mismatch: {target}')
        else:
            with trace_step('archive.verify'):
                verify_archive(candidate, manifest_name, inventory_key=inventory_key)
        timings["verification_seconds"] = time.monotonic() - started
        _publish_atomically(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)
    if profile:
        _profile_sidecar(output, timings, files)
    return output



def _check_member(name: str, *, regular: bool = True) -> None:
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('unsafe archive path')
    if not regular:
        raise ValueError('archive member must be a regular file (no symbolic links)')


def compare_inventory(inventory: dict[str, int], actual: dict[str, int], *,
                      mismatch: str = 'archive integrity mismatch') -> None:
    """The ONE inventory comparison: exact member set, then per-member byte size.

    ``mismatch`` names the surface in the size-mismatch error so callers that
    verify a different shape (the graph worker package's installed tree) keep
    their own message while sharing this comparison.
    """
    if set(actual) != set(inventory):
        raise ValueError('archive has undeclared or missing members')
    for target, expected in inventory.items():
        if actual[target] != expected:
            raise ValueError(f'{mismatch}: {target}')


def _validated_inventory(raw) -> dict[str, int]:
    """The declared member inventory: every value a non-negative byte size."""
    if not isinstance(raw, dict):
        raise ValueError('archive inventory must be a member -> size mapping')
    inventory: dict[str, int] = {}
    for name, size in raw.items():
        if (not isinstance(name, str) or isinstance(size, bool)
                or not isinstance(size, int) or size < 0):
            raise ValueError(f'archive inventory entry is not name+size: {name!r}')
        inventory[name] = size
    return inventory


def _check_inventory(metadata, actual, manifest_name, inventory_key):
    inventory = _validated_inventory(metadata[inventory_key])
    compare_inventory(inventory, actual)
    return metadata


def _count_members(archive, names, manifest_name, chunk_bytes) -> dict[str, int]:
    """Read every declared member once and record its byte count."""
    sizes: dict[str, int] = {}
    for name in tracked(names, desc='archive.verify_members'):
        if name == manifest_name:
            continue
        with archive.open(name) as handle:
            sizes[name] = _count_member_bytes(handle, chunk_bytes)
    return sizes


def verify_open_archive(archive, manifest_name: str, *, inventory_key: str | None = None) -> dict[str, Any]:
    """Verify a caller-owned reader so subsequent reads need no second inflation."""
    inventory_key = inventory_key or inventory_key_home()
    names = archive.namelist()
    if len(set(names)) != len(names):
        raise ValueError('duplicate archive members')
    for member in archive.infolist():
        _check_member(member.filename, regular=not member.is_dir() and
                      (member.external_attr >> 16) & 0o170000 != 0o120000)
    with archive.open(manifest_name) as handle:
        metadata = json.load(handle)
    actual = _count_members(archive, names, manifest_name,
                            archive_settings().copy_buffer_bytes)
    return _check_inventory(metadata, actual, manifest_name, inventory_key)


def verify_archive(path: Path, manifest_name: str, *, inventory_key: str | None = None,
                   names: list[str] | None = None) -> dict[str, Any]:
    """Verify an archive's manifest and member byte sizes.

    When ``names`` is given, the verified member names are appended to it in one
    pass, so a boundary can hand a trusted member list to later stages instead
    of re-parsing (and for a tar, re-inflating) the archive.
    """
    inventory_key = inventory_key or inventory_key_home()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            metadata = verify_open_archive(archive, manifest_name, inventory_key=inventory_key)
            if names is not None:
                names.extend(archive.namelist())
        return metadata
    # Verification is sequential: do not inflate a multi-GB tar to a temporary
    # disk file just to read it once. Count members directly from the zstd stream.
    actual, seen, metadata = {}, set(), None
    zstd = zstd_module()
    try:
        with Path(path).open('rb') as raw:
            with zstd.open(raw, 'rb') as compressed:
                with tarfile.open(fileobj=compressed, mode='r|',
                                  bufsize=archive_settings().copy_buffer_bytes) as archive:
                    for member in tracked(archive, desc='archive.verify_stream'):
                        _check_member(member.name, regular=member.isfile())
                        if member.name in seen:
                            raise ValueError('duplicate archive members')
                        seen.add(member.name)
                        if names is not None:
                            names.append(member.name)
                        with archive.extractfile(member) as handle:
                            if member.name == manifest_name:
                                metadata = json.load(handle)
                            else:
                                actual[member.name] = _count_member_bytes(
                                    handle, archive_settings().copy_buffer_bytes)
                # Consume the frame trailer as well; truncated zstd streams must fail.
                while compressed.read(archive_settings().copy_buffer_bytes):
                    pass
    except (zstd.ZstdError, EOFError) as error:
        raise ValueError(f'invalid Zstandard archive: {path}') from error
    if metadata is None:
        raise ValueError('archive manifest missing')
    return _check_inventory(metadata, actual, manifest_name, inventory_key)
