from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class BackendInfo:
    name: str
    model_path: str
    extra: dict


class BackendSession(Protocol):
    def load(self, model_path: str) -> "BackendSession": ...

    def warmup(self, x_nchw: np.ndarray, iters: int = 20) -> None: ...

    def get_input_name(self) -> str: ...

    def infer(self, x_nchw: np.ndarray) -> list[np.ndarray]: ...

    def info(self) -> BackendInfo: ...
