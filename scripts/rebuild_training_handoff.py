"""Continue the already verified CSV rebuild through the one-call preparer."""
import json
from pathlib import Path
import subprocess
import sys

from core.project_root import find_project_root

ROOT = find_project_root(Path(__file__))
OUT = ROOT/'jev/full_evidence'

if __name__ == '__main__':
    subprocess.run([sys.executable,'-m','training.prepare_all','--resume-from','validation',
                    '--run-dir',str(OUT/'training_prep')],cwd=ROOT,check=True)
    manifest=json.loads((OUT/'training_prep/manifest.json').read_text())
    (OUT/'training_handoff.json').write_text(json.dumps(manifest,indent=2)+'\n')
