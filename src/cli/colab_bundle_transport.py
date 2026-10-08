"""The Colab lane's transport boundary: ONE integrity check per VM crossing.

Every archive the Colab lane stages onto the VM or pulls back is handled here,
so the rule ``core.bundle`` sets for Colab is the rule the lane follows: an
artifact is verified exactly once, at the boundary it crosses, and the verified
handle is then trusted (no stage re-measures members, no stage re-parses an
archive).

Two shapes cross this boundary:

* a **role Bundle** (a sealed archive carrying a role manifest) — the writer
  seals it through :meth:`core.bundle.Bundle.seal_archive`, which sizes each
  source exactly once while writing AND freezes the member inventory in the
  sealed manifest, and the reader loads it once through
  :meth:`core.bundle.Bundle.load`, which re-checks that inventory. The Kaggle
  transport is the worked example (the kernel templates seal/load ``all_tracks_inputs.tar.zst``
  and the sealed result bundle this way); a Colab stage that ships a role Bundle
  must do the same rather than measure members itself.
* a **transport wrapper** — the CPU lane's delivery archive
  (``bundle_delivery.tar.zst``), whose members are a ``training_prep`` run plus
  data artifacts and which therefore is NOT a role tree. Its writer records the
  whole-archive byte size with :func:`record_digest_script` (called from the
  delivery segment itself, so the writer has ONE implementation), and its reader
  verifies that token once with :func:`verify_transport_digest`.

The token is the delivered file's byte size, recorded under the `.size`
companion suffix; no content identity is written or compared
anywhere (owner directive 2026-10-08).
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


def verify_transport_digest(archive: Path | str, expected: str | None) -> int:
    """Verify one crossing's token, ONCE, and return the observed byte size.

    ``expected`` is what the writer recorded (the remote sidecar, the fetched
    kernel receipt, the archive manifest). A mismatch is fatal: the partial is
    kept, never installed. ``expected`` unset/empty means the writer recorded no
    token (an older lane script); the caller owns making that absence explicit.

    No content identity exists anywhere (owner directive 2026-10-08): the token
    is the delivered file's byte size, recorded as text by the remote script.
    """
    archive = Path(archive)
    observed = file_size(archive)
    if expected and str(observed) != expected.strip():
        raise ValueError(
            f"transport token mismatch for {archive}: observed size={observed} "
            f"but the writer recorded {expected.strip()}")
    return observed
