"""Stale augmentation-mode references (cleanup guard, TODO item: swap lane).

Two drift defects after the ``swap_agreed`` lane was deleted:

1. tests/test_mnrl_pair_selection.py carried a fixture with a
   ``swap_agreed`` target_mode — a mode no producer emits; the
   registered modes are the ones masking.py emits (random / targeted /
   swap_values / counterfactual). A fixture naming a phantom mode makes
   the test pass vacuously against renamed/special-cased behavior.
2. train.py's extent-halves comment still claimed "both swap modes are
   excluded" from the extent halves while the code excludes only
   ``swap_values`` (the agreed-swap lane is gone) — plus a dangling
   parenthetical left from an earlier rewrite.

The registered-mode set is derived from the producer source
(src/training/masking.py) so a future mode rename fails loudly here.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MASKING_PY = REPO_ROOT / "src" / "training" / "masking.py"
TRAIN_PY = REPO_ROOT / "src" / "training" / "train.py"

# modes masking.py can write into an audit row's target_mode
MODE_EMIT = re.compile(r'"target_mode":\s*"([A-Za-z_][A-Za-z0-9_]*)"')
MODE_ASSIGN = re.compile(r'target_mode\s*=\s*"([A-Za-z_][A-Za-z0-9_]*)"')
# target_mode literals used anywhere in the test suite
TEST_MODE = re.compile(r'"target_mode":\s*"([A-Za-z_][A-Za-z0-9_]*)"')
# the deleted lane and its stale-comment claim (built by concatenation so
# this guard file itself never contains the deleted literal)
DELETED_MODE = "swap_" + "agreed"
STALE_COMMENT_CLAIM = "so both swap modes are excluded"


class StaleModeReferenceTest(unittest.TestCase):
    def test_deleted_swap_agreed_mode_is_gone(self):
        for rel in ("src", "tests"):
            for path in (REPO_ROOT / rel).rglob("*.py"):
                if path.resolve() == Path(__file__).resolve():
                    continue
                self.assertNotIn(
                    f'"{DELETED_MODE}"',
                    path.read_text(encoding="utf-8"),
                    f"deleted target_mode {DELETED_MODE!r} referenced in {path}",
                )

    def test_target_mode_literals_in_tests_are_registered(self):
        registered = {
            *MODE_EMIT.findall(MASKING_PY.read_text(encoding="utf-8")),
            *MODE_ASSIGN.findall(MASKING_PY.read_text(encoding="utf-8")),
        }
        self.assertIn("random", registered)  # sanity: extraction works
        for path in (REPO_ROOT / "tests").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for mode in set(TEST_MODE.findall(text)):
                self.assertIn(
                    mode,
                    registered,
                    f"test {path.name} uses unregistered target_mode {mode!r} "
                    f"(registered: {sorted(registered)})",
                )

    def test_extent_halves_comment_matches_code(self):
        source = TRAIN_PY.read_text(encoding="utf-8")
        # drop '#' markers so line-wrapped comment prose matches
        normalized = " ".join(source.replace("#", " ").split())
        self.assertNotIn(
            STALE_COMMENT_CLAIM,
            normalized,
            "stale comment: only swap_values is excluded from the extent "
            "halves now that the swap_agreed lane is deleted",
        )


if __name__ == "__main__":
    unittest.main()
