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
from core.archive_reader import open_archive, zstd_module, archive_sidecar


Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
INVENTORY = TypeAdapter(dict[str, Digest])


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
        from core.manifest import sha256_file
        return {relative: sha256_file(path) for relative, path in self.files.items()}


RESULT_ARCHIVE_EXCLUDED_DIRS = frozenset({
    '.dvc', '.dvc-cache', '.dvc-site-cache', '.git', '.resume',
    '_checkpoint_upload_staging', 'wandb', 'mlruns', 'mps_pipe', 'mps_log',
})


def write_archive(output: Path, files: dict[str, Path], *, manifest_name: str,
                  metadata: dict[str, Any], inline: dict[str, str] | None = None,
                  inventory_key: str = 'files', profile: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(output)
    inline = inline or {}
    if set(files) & set(inline) or manifest_name in files or manifest_name in inline:
        raise ValueError('archive member collision')
    timings = {}
    started = time.monotonic()
    inventory = {}
    for target, source in files.items():
        if source.is_symlink():
            raise ValueError('archive must not include symbolic links')
        with source.open('rb') as handle:
            inventory[target] = hashlib.file_digest(handle, 'sha256').hexdigest()
    inventory.update({target:hashlib.sha256(value.encode()).hexdigest() for target,value in inline.items()})
    for target in [*inventory, manifest_name]:
        if Path(target).is_absolute() or '..' in Path(target).parts:
            raise ValueError('unsafe archive path')
    timings["inventory_hash_seconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = output.with_name(f'{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}')
    try:
        started = time.monotonic()
        manifest = json.dumps({**metadata, inventory_key:inventory}, indent=2)+'\n'
        if output.name.endswith('.tar.zst'):
            with zstd_module().open(candidate, 'xb', level=1) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', dereference=True) as archive:
                    for target, source in files.items():
                        archive.add(source, arcname=target, recursive=False)
                    for target, value in {**inline, manifest_name: manifest}.items():
                        payload = value.encode()
                        member = tarfile.TarInfo(target)
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
        else:
            with zipfile.ZipFile(candidate, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                for target, source in files.items():
                    archive.write(source, target)
                for target, value in inline.items():
                    archive.writestr(target, value)
                archive.writestr(manifest_name, manifest)
        # Sources can change while being archived (e.g. checkpoint rotation).
        # Never publish an archive whose bytes disagree with its frozen inventory.
        timings["compression_seconds"] = time.monotonic() - started
        started = time.monotonic()
        verify_archive(candidate, manifest_name, inventory_key=inventory_key)
        timings["verification_seconds"] = time.monotonic() - started
        with candidate.open('rb') as handle:
            os.fsync(handle.fileno())
        # Linking the completed sibling publishes atomically without replacing
        # an existing immutable generation, even when two writers race.
        os.link(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)
    if profile:
        archive_sidecar(output, ".profile.json").write_text(json.dumps({**timings,
            "timestamp_unix": time.time(), "archive_bytes": output.stat().st_size,
            "source_bytes": sum(source.stat().st_size for source in files.values()),
            "file_count": len(files)}, indent=2) + "\n")
    return output



def _check_member(name: str, *, regular: bool = True) -> None:
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('unsafe archive path')
    if not regular:
        raise ValueError('archive member must be a regular file (no symbolic links)')


def _check_inventory(metadata, actual, manifest_name, inventory_key):
    inventory = INVENTORY.validate_python(metadata[inventory_key])
    if set(actual) != set(inventory):
        raise ValueError('archive has undeclared or missing members')
    for target, expected in inventory.items():
        if actual[target] != expected:
            raise ValueError(f'archive integrity mismatch: {target}')
    return metadata


def verify_open_archive(archive, manifest_name: str, *, inventory_key: str = 'files') -> dict[str, Any]:
    """Verify a caller-owned reader so subsequent reads need no second inflation."""
    names = archive.namelist()
    if len(set(names)) != len(names):
        raise ValueError('duplicate archive members')
    for member in archive.infolist():
        _check_member(member.filename, regular=not member.is_dir() and
                      (member.external_attr >> 16) & 0o170000 != 0o120000)
    metadata = json.loads(archive.read(manifest_name))
    actual = {}
    for name in names:
        if name != manifest_name:
            with archive.open(name) as handle:
                actual[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return _check_inventory(metadata, actual, manifest_name, inventory_key)


@contextmanager
def verified_archive(path: Path, manifest_name: str, *, inventory_key: str = 'files'):
    """Keep the verified reader open for extraction or configuration inspection."""
    with open_archive(path) as archive:
        yield archive, verify_open_archive(archive, manifest_name, inventory_key=inventory_key)


def verify_archive(path: Path, manifest_name: str, *, inventory_key: str = 'files') -> dict[str, Any]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return verify_open_archive(archive, manifest_name, inventory_key=inventory_key)
    # Verification is sequential: do not inflate a multi-GB tar to a temporary
    # disk file just to read it once. Hash members directly from the zstd stream.
    actual, seen, metadata = {}, set(), None
    zstd = zstd_module()
    try:
        with zstd.open(path, 'rb') as compressed:
            with tarfile.open(fileobj=compressed, mode='r|') as archive:
                for member in archive:
                    _check_member(member.name, regular=member.isfile())
                    if member.name in seen:
                        raise ValueError('duplicate archive members')
                    seen.add(member.name)
                    with archive.extractfile(member) as handle:
                        if member.name == manifest_name:
                            metadata = json.load(handle)
                        else:
                            actual[member.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
            # Consume the frame trailer as well; truncated zstd streams must fail.
            while compressed.read(1024 * 1024):
                pass
    except zstd.ZstdError as error:
        raise ValueError(f'invalid Zstandard archive: {path}') from error
    if metadata is None:
        raise ValueError('archive manifest missing')
    return _check_inventory(metadata, actual, manifest_name, inventory_key)
