"""Durable, secret-redacted lifecycle events for each suite worker attempt."""
from __future__ import annotations

from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid


def _json_default(value):
    """Serialize event payloads at the write boundary.

    v22 died at GPU-session time with "Object of type X is not JSON
    serializable" because a staged writer fed a pydantic model (raw) into
    json.dumps. Every structured record object (BaseModel, dataclass, plain
    object) is converted here instead of bubbling a TypeError into the
    session's only evidence channel.
    """
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return vars(value)


class WorkerEvents:
    def __init__(self, output: Path, track: str, run_tag: str, *, filename='worker_events.jsonl'):
        self.path = output / filename
        self.track, self.run_tag = track, run_tag
        self.attempt = uuid.uuid4().hex
        self.last_phase = None
        self.secrets = [value for key, value in os.environ.items()
                        if len(value) >= 6 and any(part in key.upper()
                                                  for part in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY'))]

    def redact(self, value: str) -> str:
        for secret in self.secrets:
            value = value.replace(secret, '<redacted>')
        value = re.sub(r'(?i)(Bearer\s+)[A-Za-z0-9._-]+', r'\1<redacted>', value)
        return re.sub(r'(?i)((?:token|api_key|password|secret)=)[^\s&]+', r'\1<redacted>', value)

    def emit(self, phase: str, status: str, **details) -> dict:
        self.last_phase = phase
        event = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                     attempt=self.attempt, run_tag=self.run_tag, track=self.track,
                     phase=phase, status=status, **details)
        if os.environ.get('ER_SUITE_ATTEMPT'):
            event['suite_attempt'] = os.environ['ER_SUITE_ATTEMPT']
        def sanitize(value):
            if isinstance(value, str):
                return self.redact(value)
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(item) for item in value]
            return value
        event = sanitize(event)
        with self.path.open('a') as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[lifecycle/{self.track}] "
              + json.dumps(event, ensure_ascii=False, default=_json_default),
              flush=True)
        return event
