from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class JsonlTelemetryLogger:
    path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def log(self, payload: dict[str, Any]) -> None:
        payload = {"ts_epoch_s": time.time(), **payload}
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlTelemetryLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
