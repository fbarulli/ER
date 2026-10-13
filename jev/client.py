"""System One client for JEV with two adapters: OpenCode Zen and OpenRouter.

Zen:  POST https://opencode.ai/zen/v1/systemone   Bearer OPENCODE_API_KEY   model jev-1.13
OR:   POST https://openrouter.ai/api/v1/systemone Bearer OPENROUTER_API_KEY model typesafe/jev-1.13

Both send the same System One payload {model, state, questions}. The response
parser is tolerant across answer shapes (answers[name].noul/.probability or
nouls[name].noul).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"

ADAPTERS = {
    "zen": {
        "sku_url": "https://opencode.ai/zen/v1/systemone",
        "key_var": "OPENCODE_API_KEY",
        "model": "jev-1.13",
    },
    "openrouter": {
        "sku_url": "https://openrouter.ai/api/v1/systemone",
        "key_var": "OPENROUTER_API_KEY",
        "model": "typesafe/jev-1.13",
    },
}

RETRYABLE = (429, 500, 502, 503, 504)
TIMEOUT = 30.0
MAX_RETRIES = 2


class EnvKeyNotFound(RuntimeError):
    pass


def _load_env_file(path: Path = ROOT_ENV) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def get_api_key(adapter: str) -> str:
    _load_env_file()
    key = os.environ.get(ADAPTERS[adapter]["key_var"], "")
    if not key:
        raise EnvKeyNotFound(f"no key for adapter {adapter!r}: set {ADAPTERS[adapter]['key_var']} (env or {ROOT_ENV})")
    return key


def build_questions() -> dict[str, dict]:
    return {
        "is_same_product": {
            "type": "noul",
            "instructions": (
                "Do record_a and record_b describe the exact same retail product "
                "(same brand, product type, flavor/variant, and pack size), accounting for "
                "typos, abbreviations, and wording differences?"
            ),
        },
    }


def _extract_noul(data: dict, name: str) -> float:
    answers = data.get("answers") or {}
    ans = answers.get(name)
    if isinstance(ans, dict):
        for field in ("noul", "probability", "p_yes"):
            v = ans.get(field)
            if isinstance(v, (int, float)):
                return float(v)
    if name in (data.get("nouls") or {}):
        v = data["nouls"][name]
        v = v.get("noul", v.get("probability")) if isinstance(v, dict) else v
        if isinstance(v, (int, float)):
            return float(v)
    if isinstance(ans, (int, float)):
        return float(ans)
    raise ValueError(f"cannot extract noul {name!r} from response keys: {sorted(data)}")


class JevClient:
    def __init__(self, adapter: str, api_key: str | None = None):
        if adapter not in ADAPTERS:
            raise ValueError(f"adapter must be one of {sorted(ADAPTERS)}")
        self.adapter = adapter
        self.cfg = ADAPTERS[adapter]
        self.api_key = api_key if api_key is not None else get_api_key(adapter)
        self.last_raw: dict | None = None

    def ask_noul(self, state: Any, questions: dict[str, dict] | None = None) -> dict[str, float]:
        body = {
            "model": self.cfg["model"],
            "state": state,
            "questions": questions or build_questions(),
        }
        data = self.post(json.dumps(body).encode())
        return {name: _extract_noul(data, name) for name in body["questions"]}

    def post(self, payload: bytes) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            req = urllib.request.Request(self.cfg["sku_url"], data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    self.last_raw = json.loads(resp.read().decode())
                    return self.last_raw
            except urllib.error.HTTPError as e:
                detail = e.read(500).decode(errors="replace")
                if e.code in RETRYABLE and attempt < MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
                    last_error = RuntimeError(f"HTTP {e.code}: {detail}")
                    continue
                raise RuntimeError(f"{self.adapter}: HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                if attempt < MAX_RETRIES:
                    last_error = e
                    continue
                raise RuntimeError(f"{self.adapter}: connection error: {e.reason}") from None
        raise RuntimeError(f"{self.adapter}: retries exhausted: {last_error}")


class MockJevClient:
    """Offline stand-in: same ask_noul() interface, deterministic title/brand heuristic."""

    def __init__(self):
        from core.portable_archive import ByteCount
        import re

        self._hashlib = hashlib
        self._re = re

    def ask_noul(self, state: dict, questions: dict[str, dict] | None = None) -> dict[str, float]:
        a, b = state.get("record_a") or {}, state.get("record_b") or {}

        def toks(s: str) -> set[str]:
            return set(self._re.findall(r"[a-z0-9]+", (s or "").lower()))

        ta, tb = toks(a.get("sku_name_eng")), toks(b.get("sku_name_eng"))
        jacc = len(ta & tb) / max(len(ta | tb), 1)
        same_brand = bool(a.get("brand")) and a.get("brand", "").lower() == b.get("brand", "").lower()

        def vol(t: str):
            m = self._re.search(r"(\d+\.?\d*)\s*(oz|fl\s*oz|liter|litre|l\b|ml)", t.lower())
            if not m:
                return None
            v = float(m.group(1))
            u = m.group(2)
            if u == "l" or u == "liter" or u == "litre":
                v *= 1000.0
            if u == "fl oz" or u == "oz":
                v *= 29.57
            return round(v)

        va, vb = vol(a.get("sku_name_eng")), vol(b.get("sku_name_eng"))
        size_ok = va is None or vb is None or va == vb

        score = jacc + (0.35 if same_brand else 0.0) + (0.15 if size_ok else -0.3)
        h = int.from_bytes(self._ByteCount(f"{a.get('gtin')}:{b.get('gtin')}".encode()).digest()[:4], "big")
        jitter = (h % 21 - 10) / 500.0
        noisy = 1 - (1 - (score + jitter)) * 0.55

        n_same = min(max(noisy, 0.02), 0.99)
        n_size = min(max((size_ok + (jacc > 0.3)) * 0.5 + 0.1, 0.03), 0.97)
        return {
            "is_same_product": n_same,
            "is_same_size": n_size,
        }
