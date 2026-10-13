"""The Colab lane's transport boundary: the sidecar token of a VM crossing.

Every archive the Colab lane stages onto the VM or pulls back is handled here,
so the token rule the lane follows has one home.

A **transport wrapper** such as the CPU lane's delivery archive
(``bundle_delivery.tar.zst``) has its whole-archive byte size recorded by
:func:`record_digest_script` (called from the delivery segment itself, so the
writer has ONE implementation) and its measured delivered size read back with
:func:`transport_archive_size`.

The token is the delivered file's byte size, recorded under the `.size`
companion suffix; it is a recorded receipt, never a refusal (owner directive
2026-10-09: data is never checked).
"""
from __future__ import annotations

from pathlib import Path

from core.archive_reader import archive_sidecar
from core.common import training_cfg
from core.manifest import file_size


def digest_suffix() -> str:
    """The transport token's sidecar suffix (a byte size, never a digest)."""
    return ".size"


def digest_sidecar(archive: Path | str) -> Path:
    """``<archive><suffix>`` — where a crossing's whole-archive token lives.

    The ONE sidecar rule (``core.archive_reader.archive_sidecar``): the
    archive's compressed ending is stripped, so ``bundle_delivery.tar.zst``
    records its token at ``bundle_delivery.size``.
    """
    return archive_sidecar(archive, digest_suffix())


def record_digest_script(archive_expression: str, *, label: str) -> str:
    """Remote-side source that records one archive's byte size (stdlib, no read).

    Returns the source lines the lane embeds in its remote script, so the
    writer side of the transport token has exactly one implementation.
    ``archive_expression`` is either a remote variable name (the delivery
    segment passes one) or a literal path, which is quoted here so a caller
    can never emit invalid remote source. The token's path is resolved by the
    ONE sidecar rule (``core.archive_reader.archive_sidecar``), imported by
    the emitted source, never re-spelled as ``path + suffix``.

    No content is fingerprinted anywhere (owner directive 2026-10-08): the
    transport token is the delivered file's ``st_size``.
    """
    suffix = digest_suffix()
    expression = (archive_expression if archive_expression.isidentifier()
                  else repr(str(archive_expression)))
    return (
        "import os as _transport_os\n"
        f"_transport_token = str(_transport_os.path.getsize({expression}))\n"
        "from core.archive_reader import archive_sidecar as _transport_sidecar\n"
        f"with open(str(_transport_sidecar({expression}, {suffix!r})), 'w', encoding='utf-8') as _transport_handle:\n"
        "    _transport_handle.write(_transport_token + '\\n')\n"
        f"print({('[' + label + '] digest size=')!r} + _transport_token, flush=True)\n"
    )


def transport_archive_size(archive: Path | str) -> int:
    """Return the delivered archive's byte size for the transport receipt.

    The writer records the same byte size in a sidecar; the caller owns reading
    that recorded token back and reporting it. Neither copy refuses anything
    (owner directive 2026-10-09: data is never checked).
    """
    return file_size(Path(archive))
