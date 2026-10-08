"""Shared SHA256 inventories for prepared-input and result archives."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
import uuid
import time
import io
import tarfile
from pathlib import Path
import zipfile
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from core.archive_reader import open_archive, zstd_module, archive_sidecar, archive_settings, tar_archive
from core.perf_switches import perf_enabled
from core.progress import tracked
from core.step_trace import timed, trace_step


Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
INVENTORY = TypeAdapter(dict[str, Digest])

#: Per-process digest cache keyed by (absolute path, mtime_ns, size). Repeated
#: validation of the SAME bytes in one process (the suite gate, resume
#: inventories, archive inventories) reuses the digest instead of re-reading
#: the file. A content change that preserves both mtime and size is not seen,
#: so the switch is off under ER_PERF_LEGACY=1 and individually via
#: ER_PERF_DIGEST_CACHE=0.
_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


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


def raw_file_digest(path: Path | str) -> str:
    """The ONE uncached whole-file SHA256 implementation.

    Every other file digest in the repository is a *policy* wrapper over this
    one (``cached_file_digest`` adds the process memo below; ``core.manifest
    .sha256_file`` and ``graph_tracks.data.FileDigest.of`` forward here), so a
    digest computed anywhere fingerprints the same bytes the same way.
    """
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cached_file_digest(path: Path | str) -> str:
    """SHA256 of one regular file, memoized on (abspath, mtime_ns, size).

    The memoized half of the one digest home: ``raw_file_digest`` owns the
    algorithm, this owns the per-process cache policy described above.
    """
    path = Path(path)
    if not perf_enabled('digest.cache'):
        return raw_file_digest(path)
    stat = path.stat()
    key = (os.path.abspath(os.fspath(path)), stat.st_mtime_ns, stat.st_size)
    value = _DIGEST_CACHE.get(key)
    if value is None:
        value = raw_file_digest(path)
        _DIGEST_CACHE[key] = value
    return value


class _HashingReader:
    """File-like tee that records the SHA256 of exactly the bytes read."""

    def __init__(self, handle, digest):
        self._handle = handle
        self._digest = digest

    def read(self, size=-1):
        data = self._handle.read(size)
        if data:
            self._digest.update(data)
        return data

    def close(self):
        self._handle.close()


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

    def inventory(self) -> dict[str, str]:
        return {relative: cached_file_digest(path) for relative, path in self.files.items()}


def is_result_archive_member(relative: str,
                             selected_checkpoints: frozenset[str] = frozenset()) -> bool:
    """Whether a checkpoint-relative path belongs in the RESULT archive.

    Thin compatibility re-export over :meth:`core.bundle.Bundle.is_result_member`
    (the predicate now lives with the bundle role it enforces). Imported by
    :mod:`model_tracks.resume`, :mod:`model_tracks.run` and
    :mod:`model_tracks.local_complete`; the lazily imported delegation avoids
    the ``core.bundle`` <-> ``core.portable_archive`` import cycle.
    """
    from core.bundle import Bundle
    return Bundle.is_result_member(relative, selected_checkpoints=selected_checkpoints)


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


def _write_tar(candidate, files, inline, manifest_name, manifest, *,
               digests: dict[str, str] | None = None, digest=None):
    """Stream a zstd tar. When ``digests`` is given, record each member's SHA256
    from the bytes handed to the writer, so the caller need not re-read the
    archive to prove it matches its frozen inventory. ``digest`` records the
    whole-file SHA256 of the compressed archive as it is written."""
    with tar_archive(candidate, 'x', digest=digest) as archive:
        for target, source in tracked(files.items(), desc='archive.tar_files'):
            if digests is None:
                archive.add(source, arcname=target, recursive=False)
                continue
            info = archive.gettarinfo(str(source), arcname=target)
            hasher = hashlib.sha256()
            with source.open('rb') as handle:
                archive.addfile(info, _HashingReader(handle, hasher))
            digests[target] = hasher.hexdigest()
        for target, value in inline.items():
            _add_tar_text(archive, target, value)
        _add_tar_text(archive, manifest_name, manifest)


def _add_tar_text(archive, name, value):
    payload = value.encode()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def source_inventory(files: dict[str, Path], inline: dict[str, str]) -> dict[str, str]:
    """The ONE name -> SHA256 inventory builder for archive sources.

    Regular-file check then hash: every source once, inline text bytes too.
    ``write_archive`` hashes sources through here while it writes, and callers
    that must mirror that inventory for a legacy manifest (the graph worker
    package's historical ``files_sha256`` key) reuse it instead of re-hashing
    the same sources with a second loop.
    """
    inventory: dict[str, str] = {}
    for target, source in tracked(files.items(), desc='archive.hash_sources'):
        if source.is_symlink() or not source.is_file():
            raise ValueError('archive requires regular files, not symbolic links')
        inventory[target] = cached_file_digest(source)
    inventory.update({target: hashlib.sha256(value.encode()).hexdigest()
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
    """The run-side cost profile (hash/compress/verify seconds + sizes)."""
    archive_sidecar(output, ".profile.json").write_text(json.dumps({
        **timings, "timestamp_unix": time.time(),
        "archive_bytes": output.stat().st_size,
        "source_bytes": sum(source.stat().st_size for source in files.values()),
        "file_count": len(files)}, indent=2) + "\n")


@timed
def write_archive(output: Path, files: dict[str, Path], *, manifest_name: str,
                  metadata: dict[str, Any], inline: dict[str, str] | None = None,
                  inventory_key: str | None = None, profile: bool = False,
                  digest=None) -> Path:
    """Publish one SHA256-inventoried archive (staging → verify → atomic link).

    ``digest`` (a ``hashlib``-style object) receives the sealed archive's
    whole-file SHA256 during the write, so the transport token costs no extra
    read of the published bytes. ``inventory_key`` defaults to the ONE config
    home (``bundle.files_key``, see :func:`inventory_key_home`).
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
    # Reject malformed inventories before expensive source hashing.
    inventory = source_inventory(files, inline)
    timings["inventory_hash_seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = _candidate_path(output)
    try:
        started = time.monotonic()
        manifest = json.dumps({**metadata, inventory_key: inventory}, indent=2) + '\n'
        stream_digests: dict[str, str] | None = (
            {} if perf_enabled('archive.write_digest') else None)
        if output.name.endswith('.tar.zst'):
            with trace_step('archive.zstandard', files=len(files),
                            compression_level=archive_settings().compression_level):
                _write_tar(candidate, files, inline, manifest_name, manifest,
                           digests=stream_digests, digest=digest)
        else:
            # Explicit ZIP outputs remain available for historical callers.
            stream_digests = None
            with trace_step('archive.zip', files=len(files)):
                _write_zip(candidate, files, inline, manifest_name, manifest)
        # Sources can change while being archived (e.g. checkpoint rotation).
        # Never publish an archive whose bytes disagree with its frozen inventory.
        timings["compression_seconds"] = time.monotonic() - started
        started = time.monotonic()
        if stream_digests is not None:
            # The writer hashed every member payload as it wrote it; comparing
            # those bytes to the frozen inventory is the whole integrity check,
            # so the archive is not read and inflated a second time.
            with trace_step('archive.verify_written'):
                for target in files:
                    if stream_digests.get(target) != inventory[target]:
                        raise ValueError(f'archive integrity mismatch: {target}')
        else:
            with trace_step('archive.verify'):
                verify_archive(candidate, manifest_name, inventory_key=inventory_key)
        timings["verification_seconds"] = time.monotonic() - started
        _publish_atomically(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)
    if digest is not None and not output.name.endswith('.tar.zst'):
        # The legacy ZIP writer cannot tee its own bytes; the digest token still
        # costs exactly one sequential read of the published archive.
        with output.open('rb') as handle:
            while chunk := handle.read(archive_settings().copy_buffer_bytes):
                digest.update(chunk)
    if profile:
        _profile_sidecar(output, timings, files)
    return output



def _check_member(name: str, *, regular: bool = True) -> None:
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('unsafe archive path')
    if not regular:
        raise ValueError('archive member must be a regular file (no symbolic links)')


def compare_inventory(inventory: dict[str, str], actual: dict[str, str], *,
                      mismatch: str = 'archive integrity mismatch') -> None:
    """The ONE inventory comparison: exact member set, then per-member digest.

    ``mismatch`` names the surface in the digest-mismatch error so callers that
    verify a different shape (the graph worker package's installed tree) keep
    their own message while sharing this comparison.
    """
    if set(actual) != set(inventory):
        raise ValueError('archive has undeclared or missing members')
    for target, expected in inventory.items():
        if actual[target] != expected:
            raise ValueError(f'{mismatch}: {target}')


def _check_inventory(metadata, actual, manifest_name, inventory_key):
    inventory = INVENTORY.validate_python(metadata[inventory_key])
    compare_inventory(inventory, actual)
    return metadata


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
    actual = {}
    for name in tracked(names, desc='archive.verify_members'):
        if name != manifest_name:
            with archive.open(name) as handle:
                actual[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return _check_inventory(metadata, actual, manifest_name, inventory_key)


@contextmanager
def verified_archive(path: Path, manifest_name: str, *, inventory_key: str | None = None):
    """Keep the verified reader open for extraction or configuration inspection."""
    inventory_key = inventory_key or inventory_key_home()
    with open_archive(path) as archive:
        yield archive, verify_open_archive(archive, manifest_name, inventory_key=inventory_key)


def verify_archive(path: Path, manifest_name: str, *, inventory_key: str | None = None,
                   digest=None, names: list[str] | None = None) -> dict[str, Any]:
    """Verify an archive's manifest and member digests.

    When ``digest`` (a ``hashlib``-style object) is given it is updated with the
    archive's whole-file bytes. For a Zstandard tar the hash is folded into the
    same pass that verifies members, so callers that need both (the VM->local
    handoff) read the multi-GB archive exactly once.

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
        if digest is not None:
            with Path(path).open('rb') as handle:
                while chunk := handle.read(archive_settings().copy_buffer_bytes):
                    digest.update(chunk)
        return metadata
    # Verification is sequential: do not inflate a multi-GB tar to a temporary
    # disk file just to read it once. Hash members directly from the zstd stream.
    actual, seen, metadata = {}, set(), None
    zstd = zstd_module()
    try:
        with Path(path).open('rb') as raw:
            source = _HashingReader(raw, digest) if digest is not None else raw
            with zstd.open(source, 'rb') as compressed:
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
                                actual[member.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
                # Consume the frame trailer as well; truncated zstd streams must fail.
                while compressed.read(archive_settings().copy_buffer_bytes):
                    pass
    except (zstd.ZstdError, EOFError) as error:
        raise ValueError(f'invalid Zstandard archive: {path}') from error
    if metadata is None:
        raise ValueError('archive manifest missing')
    return _check_inventory(metadata, actual, manifest_name, inventory_key)


def verify_archive_digest(path: Path, manifest_name: str, *,
                          inventory_key: str | None = None,
                          names: list[str] | None = None) -> tuple[dict[str, Any], str]:
    """Verify an archive and return its metadata plus whole-file SHA256."""
    digest = hashlib.sha256()
    metadata = verify_archive(path, manifest_name, inventory_key=inventory_key,
                              digest=digest, names=names)
    return metadata, digest.hexdigest()
