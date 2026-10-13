"""GitHub-release bundle transport: publish, reference, fetch, one-bundle prune.

Public behavior of the route that replaced the ~20-minute chunked upload of the
sealed ``all_tracks_inputs`` package:

* the published-bundle reference is the SAME typed seam both ends share, derived
  from the bundle-declaration SSOT (tag/asset/URL), and round-trips through JSON;
* the fetch script embeds exactly the release asset's ``browser_download_url``
  and compiles, and carries no baked secret (a public release is anonymous);
* publish calls ``gh`` (auth-gated, fails loud when unauthenticated) and, on a
  successful create, prunes every earlier ``<prefix>-*`` release so exactly one
  bundle exists anywhere.

``gh`` and the network are MOCKED: no test hits GitHub or provisions a VM.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.common import training_cfg
from core.schemas import ColabBundleTransport, PublishedBundle
from cli.colab_bundle_release import (
    BundleReleaseError,
    BundleReleaseFetcher,
    BundleReleasePublisher,
)

_SEALED = Path("/repo/results/training_prep/bundle_1013T083245944884Z/all_tracks_inputs.tar.zst")


def _reference() -> PublishedBundle:
    return training_cfg().colab.bundle_release_reference(_SEALED)


def test_published_bundle_reference_roundtrips_from_the_bundle_ssot():
    reference = _reference()
    assert reference.release_tag == "bundle-1013T083245944884Z"
    # asset name is the data-bundle SSOT, not a re-spelled literal
    assert reference.asset_name == training_cfg().colab.data_bundle.archive_name
    assert reference.run_tag == "bundle_1013T083245944884Z"
    assert reference.browser_download_url == (
        "https://github.com/fbarulli/ER/releases/download/"
        "bundle-1013T083245944884Z/all_tracks_inputs.tar.zst")
    # the compose→decompose pair shares ONE type across the seam
    assert PublishedBundle.model_validate_json(reference.model_dump_json()) == reference


def test_fetch_script_embeds_the_release_url_and_carries_no_secret():
    reference = _reference()
    script = BundleReleaseFetcher.from_config().fetch_script(
        reference, "/content/ER/p.tar.zst")
    compile(script, "<fetch>", "exec")  # the VM stage must be valid Python
    assert reference.browser_download_url in script
    spec = training_cfg().colab.bundle_release
    # budget + fail-loud flags come from config, and the token is read at RUNTIME
    assert "--max-time" in script and str(spec.curl_max_time_seconds) in script
    assert spec.fetch_token_env in script          # read from the VM env
    assert "ghp_" not in script and "Authorization: bearer $" not in script


class _FakeGh:
    """Records the gh argv and answers list/delete from a canned release set."""

    def __init__(self, *, auth_ok: bool = True, existing: list[str] | None = None):
        self.auth_ok = auth_ok
        self.existing = existing if existing is not None else ["bundle-old"]
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        sub = command[1]  # gh <sub>
        out = ""
        rc = 0
        if sub == "auth":
            rc = 0 if self.auth_ok else 1
        elif sub == "release" and command[2] == "list":
            out = _releases_json([t for t in self.existing if t.startswith("bundle-")])
        return _Completed(list(command), rc, out)


def _releases_json(tags: list[str]) -> str:
    import json
    return json.dumps([{"tagName": tag} for tag in tags])


class _Completed:
    def __init__(self, cmd, rc, stdout=""):
        self.args, self.returncode, self.stdout, self.stderr = cmd, rc, stdout, ""


def test_publish_gh_authenticates_then_prunes_every_earlier_bundle():
    gh = _FakeGh(auth_ok=True, existing=["bundle-old", "bundle-older", "release-not-a-bundle"])
    publisher = BundleReleasePublisher(runner=gh)
    reference = _reference()
    published = publisher.publish(reference, _SEALED)
    assert published == reference
    joined = [" ".join(cmd) for cmd in gh.commands]
    assert any("gh auth status" in line for line in joined)
    assert any(f"gh release create {reference.release_tag} "
               f"{_SEALED}#{reference.asset_name}" in line for line in joined)
    deleted = {line.split()[3] for line in joined if "gh release delete" in line}
    assert deleted == {"bundle-old", "bundle-older"}       # one-bundle rule
    assert reference.release_tag not in deleted            # the new one survives
    assert not any("release-not-a-bundle" in line for line in joined)


def test_publish_fails_loud_when_gh_is_not_authenticated():
    gh = _FakeGh(auth_ok=False)
    with pytest.raises(BundleReleaseError, match="authenticated gh"):
        BundleReleasePublisher(runner=gh).publish(_reference(), _SEALED)


def test_dev_upload_transport_is_selected_explicitly():
    """The route is a config enum: release is production, upload the dev fallback."""
    assert training_cfg().colab.bundle_release.transport is ColabBundleTransport.RELEASE
