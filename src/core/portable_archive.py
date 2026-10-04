"""Shared SHA256 inventories for prepared-input and result ZIPs."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from pathlib import Path
import zipfile
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


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
                  inventory_key: str = 'files') -> Path:
    if output.exists():
        raise FileExistsError(output)
    inline = inline or {}
    if set(files) & set(inline) or manifest_name in files or manifest_name in inline:
        raise ValueError('archive member collision')
    inventory = {}
    for target, source in files.items():
        if source.is_symlink():
            raise ValueError('archive must not include symbolic links')
        inventory[target] = hashlib.sha256(source.read_bytes()).hexdigest()
    inventory.update({target:hashlib.sha256(value.encode()).hexdigest() for target,value in inline.items()})
    for target in [*inventory, manifest_name]:
        if Path(target).is_absolute() or '..' in Path(target).parts:
            raise ValueError('unsafe archive path')
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = output.with_name(f'{output.name}.partial-{os.getpid()}-{uuid.uuid4().hex}')
    try:
        with zipfile.ZipFile(candidate, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for target, source in files.items():
                archive.write(source, target)
            for target, value in inline.items():
                archive.writestr(target, value)
            archive.writestr(manifest_name, json.dumps({**metadata, inventory_key:inventory}, indent=2)+'\n')
        # Sources can change while being archived (e.g. checkpoint rotation).
        # Never publish a ZIP whose bytes disagree with its frozen inventory.
        verify_archive(candidate, manifest_name, inventory_key=inventory_key)
        with candidate.open('rb') as handle:
            os.fsync(handle.fileno())
        # Linking the completed sibling publishes atomically without replacing
        # an existing immutable generation, even when two writers race.
        os.link(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)
    return output



def verify_archive(path: Path, manifest_name: str, *, inventory_key: str = 'files') -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('duplicate archive members')
        for member in archive.infolist():
            if Path(member.filename).is_absolute() or '..' in Path(member.filename).parts:
                raise ValueError('unsafe archive path')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('archive symbolic link')
        metadata = json.loads(archive.read(manifest_name))
        inventory = INVENTORY.validate_python(metadata[inventory_key])
        if set(archive.namelist()) != set(inventory) | {manifest_name}:
            raise ValueError('archive has undeclared or missing members')
        for target, expected in inventory.items():
            if hashlib.sha256(archive.read(target)).hexdigest() != expected:
                raise ValueError(f'archive integrity mismatch: {target}')
        return metadata
