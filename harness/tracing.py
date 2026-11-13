"""Lightweight JSONL tracing.

We log metadata (task, model, token usage, latency, error) but not full
prompts by default.  This keeps logs small and avoids leaking task content or
API keys into the repository.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


class Tracer:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.path = None

    def log(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = {"ts": time.time(), **event}
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                    handle.flush()
        except OSError:
            pass

    @staticmethod
    def scrub(text: str, limit: int = 400) -> str:
        if text is None:
            return ""
        value = str(text).replace("\n", "\\n")
        return value[:limit]
