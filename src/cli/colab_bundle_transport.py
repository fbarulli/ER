"""The Colab lane's transport boundary: ONE integrity check per VM crossing.

Every archive the Colab lane stages onto the VM or pulls back is handled here,
so the rule ``core.bundle`` sets for Colab is the rule the lane follows: an
artifact is verified exactly once, at the boundary it crosses, and the verified
handle is then trusted (no stage re-hashes members, no stage re-parses an
archive).

Two shapes cross this boundary:

* a **role Bundle** (a sealed archive carrying a role manifest) — the writer
  seals it through :meth:`core.bundle.Bundle.seal_archive`, which hashes each
  source exactly once while writing AND returns the sealed archive's whole-file
  digest from that same pass, and the reader loads it once through
  :meth:`core.bundle.Bundle.load` with that digest pinned. The Kaggle transport
  is the worked example (the kernel templates seal/load ``all_tracks_inputs.tar.zst``
  and the sealed result bundle this way); a Colab stage that ships a role Bundle
  must do the same rather than hash members itself.
* a **transport wrapper** — the CPU lane's delivery archive
  (``bundle_delivery.tar.zst``), whose members are a ``training_prep`` run plus
  data artifacts and which therefore is NOT a role tree. Its writer records the
  whole-archive digest with :func:`record_digest_script` (called from the
  delivery segment itself, so the writer has ONE implementation), and its reader
  verifies that token once with :func:`verify_transport_digest`.

The token is the config SSOT suffix (``training_cfg().bundle
.sha256_sidecar_suffix``), the same token Kaggle's fetched-kernel contract uses,
so both lanes speak one transport identity.
"""
from __future__ import annotations

from pathlib import Path

from core.common import training_cfg
from core.manifest import sha256_file


def digest_suffix() -> str:
    """The transport digest sidecar suffix (config SSOT, never a literal here)."""
    return training_cfg().bundle.sha256_sidecar_suffix


def digest_sidecar(archive: Path | str) -> Path:
    """``<archive><suffix>`` — where a crossing's whole-archive token lives."""
    archive = Path(archive)
    return archive.with_name(archive.name + digest_suffix())


def record_digest_script(archive_expression: str, *, label: str) -> str:
    """Remote-side source that records one archive's digest (stdlib, one pass).

    Returns the source lines the lane embeds in its remote script, so the
    writer side of the transport token has exactly one implementation.
    ``archive_expression`` is either a remote variable name (the delivery
    segment passes one) or a literal path, which is quoted here so a caller
    can never emit invalid remote source.
    """
    suffix = digest_suffix()
    expression = (archive_expression if archive_expression.isidentifier()
                  else repr(str(archive_expression)))
    return (
        "import hashlib as _transport_hashlib\n"
        f"_transport_digest = _transport_hashlib.sha256()\n"
        f"with open({expression}, 'rb') as _transport_handle:\n"
        "    for _transport_chunk in iter(lambda: _transport_handle.read(1024 * 1024), b''):\n"
        "        _transport_digest.update(_transport_chunk)\n"
        "_transport_token = _transport_digest.hexdigest()\n"
        f"with open({expression} + {suffix!r}, 'w', encoding='utf-8') as _transport_handle:\n"
        "    _transport_handle.write(_transport_token + '\\n')\n"
        f"print({('[' + label + '] digest sha256=')!r} + _transport_token, flush=True)\n"
    )


def verify_transport_digest(archive: Path | str, expected: str | None) -> str:
    """Verify one crossing's token, ONCE, and return the observed digest.

    ``expected`` is what the writer recorded (the remote sidecar, the fetched
    kernel receipt, the archive manifest). A mismatch is fatal: the partial is
    kept, never installed. ``expected`` unset/empty means the writer recorded no
    token (an older lane script); the caller owns making that absence explicit.
    """
    archive = Path(archive)
    observed = sha256_file(archive)
    if expected and observed != expected.strip():
        raise ValueError(
            f"transport digest mismatch for {archive}: observed sha256={observed} "
            f"but the writer recorded {expected.strip()}")
    return observed
