"""Owner directive 2026-10-08: ZERO freshness/staleness checks repo-wide.

Data stays naked. The ONE permitted integrity check on a bundle is its DIGEST at
the boundary (``core.bundle.Bundle.load`` / ``verify_archive_digest``): that
answers identity, never freshness. Everything else that compared a recorded
value against a freshly derived one to decide whether cached data still "applies"
is removed, and this guard fails if any of it reappears in ``src/`` or
``scripts/``.

Design rule applied here: the policy is baked into one owner,
:class:`NoFreshnessPolicy`, which owns the forbidden patterns, the boundary
allowlist and the scan; the tests only call it. ``test_policy_flags_a_synthetic
_offender`` pins that the policy actually flags enforcement, so a silently
neutered pattern set fails loudly.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class NoFreshnessPolicy:
    """The no-freshness rule as one owner: patterns + allowlist + scan.

    A *freshness check* is any re-derivation of a recorded value (hash, digest,
    mtime, TTL, "not yet produced" allowance) used to decide whether cached data
    is still valid. The identifiers below name that concept directly; the
    patterns catch it in prose-shaped enforcement (``"... is stale"``,
    ``"stale prerequisite"``, ``stale_cols``).
    """

    #: Exact identifiers/literals that ARE a freshness comparison.
    FORBIDDEN_LITERALS: tuple[str, ...] = (
        "ER_SKIP_CONFIG_VERIFY",
        "SuiteFreshnessManifest",
        "freshness.json",
        "request_size=",
        "allow_gpu_pending",
        "GPU_PENDING",
        "ER_DATA_GATE_GPU_PENDING",
        "freshness_checks",
        "verifies_freshness",
        "is_stale",
        "stale_hash_check",
        "_fail_on_stale",
        "stale_cols",
    )

    #: Noun set that follows ``stale`` in an enforcement verdict.
    _DATA_NOUNS: str = (
        "prerequisite|input|resume|frozen|catalog|prepared|bundle|columns?|sims?"
        "|metadata|schema|index|plan|vectors?|scores?|cache|snapshots?|artifacts?"
        "|tokens?|payload|results?|export|request"
    )

    #: Enforcement vocabulary beyond the exact identifiers.
    FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bstaleness\b", re.IGNORECASE),
        re.compile(rf"\bstale\s+(?:{_DATA_NOUNS})\b", re.IGNORECASE),
        re.compile(
            r"\b(?:contains?|is|are|was|were|discarded|re-?scor\w*)\s+stale\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bstale[_a-z]+\b", re.IGNORECASE),
    )

    #: Paths where a pattern is permitted, each with the reason it is allowed.
    #: The digest boundary and the ratified ruling pin may legitimately name the
    #: concept; the parallel ColabSpec change retains its own files untouched.
    ALLOWED: dict[str, str] = {
        "src/core/schemas.py": (
            "owner-ratified ruling pin: freshness_checks is Literal[False] and "
            "verifies_freshness() returns False, so no consumer can turn one on"
        ),
        "src/core/bundle.py": "the permitted digest boundary (Bundle.load)",
        "src/core/portable_archive.py": "the permitted digest primitives",
        "src/cli/colab.py": "retained by the parallel ColabSpec change",
        "src/cli/colab_lane.py": "retained by the parallel ColabSpec change",
        "src/cli/colab_launch.py": "retained by the parallel ColabSpec change",
        "src/cli/colab_runtime.py": "retained by the parallel ColabSpec change",
        "src/cli/colab_self_watch.py": "retained by the parallel ColabSpec change",
        "src/cli/log_capture.py": "retained by the parallel ColabSpec change",
    }

    def __init__(self, root: Path = REPO_ROOT, roots: tuple[str, ...] = ("src", "scripts")) -> None:
        self.root = Path(root)
        self.roots = roots

    def sources(self) -> list[Path]:
        """Every scanned ``*.py`` file, relative to the repo root."""
        found = [
            path
            for base in self.roots
            for path in (self.root / base).rglob("*.py")
            if "__pycache__" not in path.parts
        ]
        return sorted(found)

    def violations_in(self, text: str) -> list[str]:
        """The forbidden matches in one source text (empty = clean)."""
        found = [literal for literal in self.FORBIDDEN_LITERALS if literal in text]
        for pattern in self.FORBIDDEN_PATTERNS:
            found.extend(match.group(0) for match in pattern.finditer(text))
        return found

    def scan(self) -> dict[str, list[str]]:
        """Offending relative paths -> matches, skipping the allowlist."""
        offenders: dict[str, list[str]] = {}
        for path in self.sources():
            relative = path.relative_to(self.root).as_posix()
            if relative in self.ALLOWED:
                continue
            found = self.violations_in(path.read_text(encoding="utf-8"))
            if found:
                offenders[relative] = found
        return offenders

    def is_allowed(self, relative: str) -> bool:
        """Whether a path is a documented exception (boundary/pin/retained)."""
        return relative in self.ALLOWED


class NoFreshnessCheckTest(unittest.TestCase):
    def test_no_freshness_enforcement_reappears(self) -> None:
        offenders = NoFreshnessPolicy().scan()
        self.assertEqual(
            {},
            offenders,
            "freshness/staleness enforcement reappeared (data must stay naked; "
            "the only permitted check is the bundle digest at the boundary): "
            f"{offenders}",
        )

    def test_policy_flags_a_synthetic_offender(self) -> None:
        policy = NoFreshnessPolicy()
        for sample in (
            "if digest != recorded:\n    raise ValueError('stale prerequisite')",
            "allow_gpu_pending = True",
            "def is_stale(e): ...",
            "if fp != stamp:\n    stale_cols.append(c)",
            "ER_SKIP_CONFIG_VERIFY",
            "value = compute(request_size=x)",
        ):
            self.assertTrue(
                policy.violations_in(sample),
                f"policy failed to flag a freshness sample: {sample!r}",
            )
        for clean in (
            "raise ValueError('incompatible embedding shape')",
            "if embeddings.shape != (rows, dim): rebuild()",
            "digest = Bundle.load(path).digest",
        ):
            self.assertEqual(
                [], policy.violations_in(clean),
                f"policy flagged an integrity-only sample: {clean!r}",
            )

    def test_only_the_boundary_and_the_pin_are_allowed(self) -> None:
        policy = NoFreshnessPolicy()
        self.assertTrue(policy.is_allowed("src/core/schemas.py"))
        self.assertTrue(policy.is_allowed("src/core/bundle.py"))
        self.assertTrue(policy.is_allowed("src/core/portable_archive.py"))
        # The digest boundary is real and is where integrity lives.
        from core.bundle import Bundle
        self.assertTrue(hasattr(Bundle, "load"))
        # No allowlisted path may hide an unbounded region of the tree.
        for relative in policy.ALLOWED:
            self.assertTrue(relative.endswith(".py"), relative)

    def test_scan_covers_the_python_tree(self) -> None:
        sources = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in NoFreshnessPolicy().sources()
        }
        self.assertIn("src/training/prepare_all.py", sources)
        self.assertIn("src/training/handoff.py", sources)
        self.assertIn("scripts/run_colab_embeddings.py", sources)
        self.assertGreater(len(sources), 200)


if __name__ == "__main__":
    unittest.main()
