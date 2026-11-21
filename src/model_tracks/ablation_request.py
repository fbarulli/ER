"""The ONE reader/writer of the attribute-ablation request document.

The request used to carry its whole cohort inline (~722 MB: ``pairs`` half of
the bytes, ``variants`` another quarter).  ``request.json`` is now the SMALL
manifest; the two bulk members travel as typed columnar sidecars beside it
(``pairs.parquet`` / ``variants.parquet``).  Every consumer still receives the
same ``dict`` with the same ``list[dict]`` members, so no caller changes shape.

Two boundaries:

* :class:`ColumnarRequestTable` — the ONE encode/decode seam between a
  ``list[dict]`` and a Parquet table.  Scalars stay native (bool/int/float/
  string), a nested or mixed-type cell is JSON-encoded into a string column and
  decoded on read, and a ragged row (a different key set) keeps its own key
  order, so the reconstructed rows are deep-equal to the originals.
* :class:`AblationRequest` — the manifest + sidecar layout.  A manifest that
  declares a sidecar which is missing fails loud; it never degrades to an empty
  list.  A legacy inline request (the sealed packages already on disk) is read
  as it is.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

#: The request members stored column-wise, and the sidecar file each one owns.
SIDECARS: dict[str, str] = {'pairs': 'pairs.parquet', 'variants': 'variants.parquet'}

#: Manifest member declaring which sidecars the document carries.  It is a
#: layout marker only and is never returned to a caller.
SIDECAR_DECLARATION = 'columnar_sidecars'

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


class ColumnarRequestTable:
    """Exact ``list[dict]`` <-> Parquet table codec (the ONE boundary codec)."""

    #: Per-row key order, written only for a ragged table.
    _KEYS = '__columnar_keys__'
    _KEY_ORDER = b'er.ablation.key_order'
    _JSON_COLUMNS = b'er.ablation.json_columns'
    _RAGGED = b'er.ablation.ragged'

    @classmethod
    def write(cls, rows: Iterable[dict[str, Any]], path: Path) -> None:
        """Persist ``rows`` as one compressed Parquet table."""
        pq.write_table(cls.encode(rows), path, compression='zstd')

    @classmethod
    def read(cls, path: Path) -> list[dict[str, Any]]:
        """Reconstruct the exact list of dicts a prior :meth:`write` received."""
        return cls.decode(pq.read_table(path))

    @classmethod
    def encode(cls, rows: Iterable[dict[str, Any]]) -> pa.Table:
        """Encode rows column-wise; nested/mixed cells become JSON strings."""
        records = list(rows)
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f'ablation request rows must be dicts, got {type(record).__name__}')
        key_order = list(dict.fromkeys(key for record in records for key in record))
        if cls._KEYS in key_order:
            raise ValueError(f'ablation request rows use the reserved column {cls._KEYS!r}')
        key_sets = [tuple(record) for record in records]
        ragged = bool(records) and any(keys != key_sets[0] for keys in key_sets)
        columns: dict[str, pa.Array] = {}
        json_columns: list[str] = []
        for key in key_order:
            values = [record.get(key) for record in records]
            kind = cls._kind(values)
            if kind == 'json':
                json_columns.append(key)
            columns[key] = cls._array(values, kind)
        if ragged:
            columns[cls._KEYS] = pa.array(
                [json.dumps(list(keys), ensure_ascii=False) for keys in key_sets],
                type=pa.string())
        table = pa.table(columns)
        return table.replace_schema_metadata({
            cls._KEY_ORDER: json.dumps(key_order, ensure_ascii=False).encode(),
            cls._JSON_COLUMNS: json.dumps(json_columns, ensure_ascii=False).encode(),
            cls._RAGGED: b'1' if ragged else b'0',
        })

    @classmethod
    def decode(cls, table: pa.Table) -> list[dict[str, Any]]:
        """Rebuild rows in their original key order, decoding JSON columns."""
        metadata = table.schema.metadata or {}
        required = {cls._KEY_ORDER, cls._JSON_COLUMNS, cls._RAGGED}
        if not required <= set(metadata):
            raise ValueError('not an ablation columnar table: layout metadata is missing')
        key_order = json.loads(metadata[cls._KEY_ORDER])
        json_columns = set(json.loads(metadata[cls._JSON_COLUMNS]))
        columns = {name: table.column(name).to_pylist()
                   for name in table.column_names if name != cls._KEYS}
        if metadata[cls._RAGGED] == b'1':
            key_sets = [json.loads(value) for value in table.column(cls._KEYS).to_pylist()]
        else:
            key_sets = [key_order] * table.num_rows
        rows: list[dict[str, Any]] = []
        for index, keys in enumerate(key_sets):
            row: dict[str, Any] = {}
            for key in keys:
                value = columns[key][index]
                row[key] = (json.loads(value)
                            if key in json_columns and value is not None else value)
            rows.append(row)
        return rows

    @staticmethod
    def _kind(values: list[Any]) -> str:
        """The native column kind, or ``json`` for nested/mixed/oversized cells."""
        kinds = {type(value) for value in values if value is not None}
        if not kinds or kinds == {str}:
            return 'string'
        if kinds == {bool}:
            return 'bool'
        if kinds == {int}:
            return 'int' if all(_INT64_MIN <= value <= _INT64_MAX
                                for value in values if value is not None) else 'json'
        if kinds == {float}:
            return 'float'
        return 'json'

    @staticmethod
    def _array(values: list[Any], kind: str) -> pa.Array:
        """One native Arrow array, or a compact JSON string array for ``json``."""
        if kind == 'bool':
            return pa.array(values, type=pa.bool_())
        if kind == 'int':
            return pa.array(values, type=pa.int64())
        if kind == 'float':
            return pa.array(values, type=pa.float64())
        if kind == 'string':
            return pa.array(values, type=pa.string())
        return pa.array(
            [None if value is None else json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                allow_nan=False)
             for value in values],
            type=pa.string())


class AblationRequest:
    """The manifest + sidecar layout of one ablation request document."""

    @classmethod
    def save(cls, path: Path | str, request: dict[str, Any]) -> None:
        """Write the small JSON manifest and the declared columnar sidecars."""
        path = Path(path)
        document = dict(request)
        declared: list[str] = []
        for key, filename in SIDECARS.items():
            if key not in document:
                continue
            ColumnarRequestTable.write(document.pop(key), path.parent / filename)
            declared.append(key)
        if declared:
            document[SIDECAR_DECLARATION] = declared
        cls._write_manifest(path, document)

    @classmethod
    def load(cls, path: Path | str) -> dict[str, Any]:
        """The full request document: manifest plus every declared sidecar."""
        path = Path(path)
        manifest = json.loads(path.read_text(encoding='utf-8'))
        sidecars = cls.sidecars(path, manifest)
        manifest.pop(SIDECAR_DECLARATION, None)
        for key, sidecar in sidecars.items():
            if key in manifest:
                raise ValueError(
                    f'ablation request {path} carries {key} both inline and in {sidecar.name}')
            manifest[key] = ColumnarRequestTable.read(sidecar)
        return manifest

    @classmethod
    def sidecars(cls, path: Path | str, manifest: dict[str, Any] | None = None
                 ) -> dict[str, Path]:
        """The sidecar files this request declares, failing loud on a missing one.

        A manifest whose declaration is read here is left untouched; a caller
        holding a parsed manifest passes it so the bytes are parsed only once.
        """
        path = Path(path)
        if manifest is None:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        declared = manifest.get(SIDECAR_DECLARATION, ())
        unknown = [key for key in declared if key not in SIDECARS]
        if unknown:
            raise ValueError(f'ablation request {path} declares unknown sidecars {unknown}')
        files: dict[str, Path] = {}
        for key in declared:
            sidecar = path.parent / SIDECARS[key]
            if not sidecar.is_file():
                raise FileNotFoundError(
                    f'ablation request {path} declares the {key} sidecar, '
                    f'but {sidecar} is missing')
            files[key] = sidecar
        return files

    @staticmethod
    def _write_manifest(path: Path, document: dict[str, Any]) -> None:
        """Stream the manifest through the ONE canonical request writer."""
        from model_tracks.ablation import write
        write(path, document)


def save(path: Path | str, request: dict[str, Any]) -> None:
    """Write one ablation request (manifest + columnar sidecars)."""
    AblationRequest.save(path, request)


def load(path: Path | str) -> dict[str, Any]:
    """Read one ablation request (manifest + columnar sidecars, or legacy inline)."""
    return AblationRequest.load(path)
