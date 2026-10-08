import sys
from pathlib import Path

# The checkout's OWN src/ must win over a shared editable install so tests
# exercise the working tree (worktrees share the venv, whose .pth may point at
# a sibling checkout).
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))
