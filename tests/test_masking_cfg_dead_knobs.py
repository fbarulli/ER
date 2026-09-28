"""Dead-knob guard for the masking config (SSOT rule).

SSOT rule violated by audit finding "dead knob
hard_negative_swap_frac": a knob shipped in config/training.yaml and
validated into MaskingSpec implies live behavior; when its only consumer
was deleted (the agreed-surface swap lane), the leftover read
(train.py ``hard_negative_swap_frac = float(mask_cfg["hard_negative_swap_frac"])``)
bound a local that nothing used — config implies behavior that does not
exist. Two guards:

1. CONFIG-TO-CODE — every ``X = ... mask_cfg["Y"]`` read in train.py must
   have its local ``X`` referenced again downstream; a bound-but-never-used
   masking knob fails here instead of shipping phantom behavior.
2. KNOB PIN — ``hard_negative_swap_frac`` must not exist anywhere in src/
   or config/ (field, key, or read).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "src" / "training" / "train.py"

MASK_CFG_READ = re.compile(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:float\()?mask_cfg\[\"([A-Za-z_][A-Za-z0-9_]*)\"\]",
    re.MULTILINE,
)
DEAD_KNOB = "hard_negative_swap_frac"


class MaskingDeadKnobTest(unittest.TestCase):
    def test_every_mask_cfg_binding_is_consumed(self):
        source = TRAIN_PY.read_text(encoding="utf-8")
        # strip the mask_cfg["key"] reads themselves: a binding whose only
        # other occurrence is the config key it was read from is dead.
        code = MASK_CFG_READ.sub("", source)
        dead = [
            local
            for local in {
                m.group(2) for m in MASK_CFG_READ.finditer(source)
            }
            if len(re.findall(rf"\b{re.escape(local)}\b", code)) < 1
        ]
        self.assertEqual(
            dead,
            [],
            "mask_cfg knob read into a local that is never used "
            "(config implies behavior that does not exist): "
            + ", ".join(dead),
        )

    def test_hard_negative_swap_frac_is_gone_from_code_and_config(self):
        for rel in (
            "src/training/train.py",
            "src/core/schemas.py",
            "config/training.yaml",
        ):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn(
                "hard_negative_swap_frac",
                text,
                f"dead knob hard_negative_swap_frac still present in {rel}",
            )


if __name__ == "__main__":
    unittest.main()
