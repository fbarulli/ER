"""Shared SHA256 inventories for prepared-input and result ZIPs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile


def write_archive(output: Path, files: dict[str, Path], *, manifest_name: str,
                  metadata: dict, inline: dict[str, str] | None = None,
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
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for target, source in files.items():
            archive.write(source, target)
        for target, value in inline.items():
            archive.writestr(target, value)
        archive.writestr(manifest_name, json.dumps({**metadata, inventory_key:inventory}, indent=2)+'\n')
    return output


def verify_archive(path: Path, manifest_name: str, *, inventory_key: str = 'files') -> dict:
    with zipfile.ZipFile(path) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('duplicate archive members')
        for member in archive.infolist():
            if Path(member.filename).is_absolute() or '..' in Path(member.filename).parts:
                raise ValueError('unsafe archive path')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('archive symbolic link')
        metadata = json.loads(archive.read(manifest_name))
        for target, expected in metadata[inventory_key].items():
            if hashlib.sha256(archive.read(target)).hexdigest() != expected:
                raise ValueError(f'archive integrity mismatch: {target}')
        return metadata
