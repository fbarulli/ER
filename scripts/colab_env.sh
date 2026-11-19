#!/usr/bin/env bash
# Shared Colab launcher resolver — the ONE source run_colab_full.sh and
# run_colab_model.sh read: session/GPU/transcript/sealed-package resolve from
# config/training.yaml (colab block) through cli.colab, and the suite
# catalogue derives from the config/model_tracks*.yaml promotion naming
# contract (the canonical MNRL manifest first, then every promoted loss
# variant). Nothing here re-spells a config value.

# Emit shlex-quoted assignments: COLAB_GPU, COLAB_SESSION, RUN_LOG, PKG.
# Fails loud (non-zero) when no sealed all_tracks_inputs package exists.
resolve_colab_env() {
  PYTHONPATH=src .venv/bin/python - <<'PY'
import shlex
from pathlib import Path
import yaml
from cli.colab import (default_prepared_input_package, lane_transcript_path,
                           resolve_prepared_input_package, SESSION)

colab = yaml.safe_load(Path('config/training.yaml').read_text(encoding='utf-8'))['colab']
package = default_prepared_input_package()
if package is None:
    raise SystemExit('no sealed all_tracks_inputs package found: run the CPU bundle '
                     'lane (scripts/run_colab_bundle.sh) first; the full E2E never '
                     're-packages inputs')
print('COLAB_GPU=' + shlex.quote(str(colab['gpu'])))
print('COLAB_SESSION=' + shlex.quote(str(SESSION)))
print('RUN_LOG=' + shlex.quote(str(lane_transcript_path())))
print('PKG=' + shlex.quote(str(resolve_prepared_input_package(package))))
PY
}

# One suite path per line, in promotion order (canonical first).
resolve_colab_suites() {
  local suite
  printf '%s\n' config/model_tracks.yaml
  for suite in config/model_tracks__*.yaml; do printf '%s\n' "$suite"; done
}

# The model label a suite file is chosen by: its stem without the prefix.
suite_label() {
  local stem
  stem="$(basename "$1" .yaml)"
  printf '%s' "${stem#model_tracks__}"
}
