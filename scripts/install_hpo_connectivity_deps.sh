#!/usr/bin/env bash
set -Eeuo pipefail
python -m pip install --quiet 'optuna==4.4.0' 'psycopg[binary]==3.3.5' 'sqlalchemy>=2.0'
python - <<'PY'
import optuna
import psycopg
print(f"[deps] optuna={optuna.__version__} psycopg={psycopg.__version__}")
PY
