"""
config_loader.py

Loads config.yaml, expands ${variable} references, and resolves configured
filesystem paths relative to base_dir.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Fields whose values represent filesystem paths.
PATH_KEYS = {
    "base_dir",
    "results_dir",
    "input_csv",
    "output_csv",
    "input_jsonl",
    "model_dir",
    "output_dir",
    "metrics_csv",
    "fp_csv",
    "fn_csv",
    "predictions_csv",
}

VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_variables(value: Any, variables: dict[str, Any]) -> Any:
    """Recursively expand ${variable} references in strings."""
    if isinstance(value, dict):
        return {k: _expand_variables(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_variables(v, variables) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return str(variables[name])
        if name in os.environ:
            return os.environ[name]
        raise KeyError(f"Unknown config variable: ${{{name}}}")

    return VARIABLE_PATTERN.sub(replace, value)


def _resolve_path(value: str | Path, base_dir: Path) -> str:
    """Resolve one filesystem path against base_dir."""
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_paths(obj: Any, base_dir: Path, parent_key: str | None = None) -> Any:
    """Recursively resolve values belonging to known path fields."""
    if isinstance(obj, dict):
        return {k: _resolve_paths(v, base_dir, parent_key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_paths(v, base_dir, parent_key=parent_key) for v in obj]
    if isinstance(obj, str) and parent_key in PATH_KEYS:
        return _resolve_path(obj, base_dir)
    return obj


def load_config(path: str | Path | None = None) -> dict:
    """
    Load config.yaml.
    - Resolves base_dir relative to config.yaml.
    - Expands ${variable} references.
    - Resolves configured filesystem paths relative to base_dir.
    """
    config_path = Path(path).resolve() if path else CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")

    # 1. Resolve base_dir
    raw_base_dir = raw.get("base_dir", ".")
    base_dir = Path(raw_base_dir)
    if not base_dir.is_absolute():
        base_dir = config_path.parent / base_dir
    base_dir = base_dir.resolve()

    # 2. Resolve results_dir
    results_dir = Path(raw.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = base_dir / results_dir
    results_dir = results_dir.resolve()

    # 3. Expand variables
    variables = {
        "base_dir": str(base_dir),
        "results_dir": str(results_dir),
    }
    expanded = _expand_variables(raw, variables)
    expanded["base_dir"] = str(base_dir)

    # 4. Resolve all known path fields
    resolved = _resolve_paths(expanded, base_dir)
    resolved["base_dir"] = str(base_dir)

    return resolved


if __name__ == "__main__":
    import pprint
    pprint.pp(load_config())