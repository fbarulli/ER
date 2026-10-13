"""The GitHub-release transport of the sealed inputs package (CPU publish + GPU fetch).

The Colab GPU lane no longer pushes the sealed ``all_tracks_inputs`` archive
(~427MB) through Colab's chunked ``upload`` API.  The crossing is split into the
two ends this module owns:

* :class:`BundleReleasePublisher` — run by the CPU bundle lane on the operator's
  machine after sealing: it publishes the archive as one GitHub Release asset via
  the ``gh`` CLI and enforces the ONE BUNDLE rule (every earlier ``<prefix>-*``
  release+asset is deleted, so exactly one bundle exists anywhere).  Credential
  resolution is the config's: ``gh`` is authed from the environment; a missing
  ``gh auth status`` fails loud, never a silent skip.

* :class:`BundleReleaseFetcher` — renders the VM-side one-shot fetch: a single
  ``curl -L`` of the release asset's ``browser_download_url``.  For a private
  repository the bearer token is read AT VM RUNTIME from the env var named in
  config (``ColabBundleReleaseSpec.fetch_token_env``) — never baked into a URL,
  never printed into the transcript.  A missing release/asset is a loud failure:
  there is no production fallback to the slow chunked upload.

Both ends consume the SAME typed seam (:class:`core.schemas.PublishedBundle`),
whose names derive from the bundle-declaration SSOT
(``ColabSpec.bundle_release_reference``), so publish and fetch can never disagree
on a tag, an asset name or a URL.  Collaborators (the subprocess runner, the
reference, the release spec) are injected, never instantiated inside a method, so
tests drive every branch offline with ``gh``/network MOCKED.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from core.common import training_cfg
from core.schemas import ColabBundleReleaseSpec, ColabBundleTransport, PublishedBundle
from training.prepare_all_trace import timed


class BundleReleaseError(RuntimeError):
    """The GitHub-release transport could not complete: fail loud, never silent."""


class BundleReleasePublisher:
    """Publish the sealed archive as the ONE ``<prefix>-*`` GitHub Release."""

    def __init__(self, *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 spec: ColabBundleReleaseSpec | None = None) -> None:
        self._runner = runner
        self._spec = spec if spec is not None else training_cfg().colab.bundle_release

    @classmethod
    def from_config(cls, *, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run
                    ) -> "BundleReleasePublisher":
        """Factory: the release spec is the config SSOT, never a hand-typed value."""
        return cls(runner=runner, spec=training_cfg().colab.bundle_release)

    def _gh(self, *args: str, capture: bool = True) -> subprocess.CompletedProcess:
        """Run one ``gh`` subcommand; a non-zero exit is reported, never swallowed."""
        command = ["gh", *args]
        result = self._runner(command, capture_output=capture, text=True, check=False)
        if getattr(result, "returncode", 0):
            stdout = (getattr(result, "stdout", "") or "").strip()
            stderr = (getattr(result, "stderr", "") or "").strip()
            raise BundleReleaseError(
                f"gh {' '.join(args)} failed (rc={result.returncode}): "
                f"{stderr or stdout}")
        return result

    @timed
    def require_auth(self) -> None:
        """Refuse to publish when ``gh`` is not authenticated: fail loud, full reason."""
        try:
            self._gh("auth", "status")
        except BundleReleaseError as error:
            raise BundleReleaseError(
                "the GitHub-release bundle transport needs an authenticated gh CLI; "
                f"run 'gh auth login' (or export a GH_TOKEN) before publishing: {error}"
            ) from error

    @timed
    def publish(self, reference: PublishedBundle, archive: Path) -> PublishedBundle:
        """Create the ``<tag_prefix>-<run-tag>`` release carrying ``archive``.

        The asset is uploaded whole (the sealed ``tar.zst`` fits the 2GB asset
        ceiling, so no splitting); ``target_commitish`` is left to gh's default
        (the repo's default branch) because GitHub validates it as a GIT REF —
        the run tag is not one, and the fetch resolves the release by its TAG,
        never by the commitish.
        """
        self.require_auth()
        self._gh("release", "create", reference.release_tag,
                 f"{archive}#{reference.asset_name}",
                 "--title", reference.release_tag,
                 "--notes", f"Colab sealed inputs bundle {reference.run_tag}")
        self.prune_previous(keep=reference.release_tag)
        return reference

    @timed
    def prune_previous(self, *, keep: str) -> list[str]:
        """ONE BUNDLE rule: delete every other ``<prefix>-*`` release+asset.

        Returns the deleted tags.  ``--cleanup-tag`` removes the release's git tag
        too, so a stale bundle cannot be re-discovered.  The release just
        published (``keep``) is never touched.
        """
        prefix = self._spec.tag_prefix
        listed = self._gh("release", "list", "--limit", "100", "--json", "tagName")
        try:
            releases = json.loads(listed.stdout or "[]")
        except json.JSONDecodeError as error:
            raise BundleReleaseError(
                f"gh release list returned unreadable JSON: {error}") from error
        pruned: list[str] = []
        for release in releases:
            tag = release.get("tagName") or release.get("tag")
            if tag and tag != keep and tag.startswith(f"{prefix}-"):
                self._gh("release", "delete", tag, "--cleanup-tag", "--yes")
                pruned.append(tag)
        return pruned


class BundleReleaseFetcher:
    """Render the GPU VM's single authenticated fetch of the published bundle."""

    def __init__(self, *, spec: ColabBundleReleaseSpec | None = None) -> None:
        self._spec = spec if spec is not None else training_cfg().colab.bundle_release

    @classmethod
    def from_config(cls) -> "BundleReleaseFetcher":
        return cls(spec=training_cfg().colab.bundle_release)

    def curl_command(self, reference: PublishedBundle, destination: str) -> list[str]:
        """The unauthenticated ``curl`` argv for one release asset.

        The release spec's budget (``curl_max_time_seconds`` / ``curl_retries``)
        is read from config; nothing here re-spells it.  The URL carries no secret
        for a public repo; the bearer header is attached at VM RUNTIME only when
        the VM actually holds a token (see :meth:`fetch_argv`).  ``-f`` makes a
        missing release/asset exit non-zero — a loud failure, never a fallback.
        """
        return [
            "curl", "-fsSL",
            "--retry", str(self._spec.curl_retries),
            "--retry-delay", "2",
            "--max-time", str(self._spec.curl_max_time_seconds),
            "-o", destination,
            reference.browser_download_url,
        ]

    def fetch_script(self, reference: PublishedBundle, destination: str) -> str:
        """The VM-side source that fetches ``reference`` into ``destination``.

        Self-contained stdlib (no ``_BOOTSTRAP``): it reads the bearer token from
        the VM's own environment at RUNTIME and attaches the ``Authorization``
        header only when that variable is present, so a public release is fetched
        anonymously and no secret is ever baked into a URL or printed.  ``curl -f``
        exits non-zero on a missing release/asset and ``check=True`` propagates it
        into the lane's fail-fast channel with the full remote output.
        """
        argv = self.curl_command(reference, destination)
        return (
            "import os, subprocess\n"
            f"_bundle_cmd = {argv!r}\n"
            f"_bundle_token = os.environ.get({self._spec.fetch_token_env!r})\n"
            "if _bundle_token:\n"
            "    _bundle_cmd[1:1] = ['-H', 'Authorization: bearer ' + _bundle_token]\n"
            "subprocess.run(_bundle_cmd, check=True)\n"
        )


def bundle_transport_release() -> bool:
    """Whether the config selects the GitHub-release transport for the bundle."""
    return training_cfg().colab.bundle_release.transport is ColabBundleTransport.RELEASE
