from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path


@cache
def load_library() -> ctypes.CDLL:
    path = Path(__file__).with_name("_core.so")
    if not path.exists():
        raise FileNotFoundError(f"could not find {path}; build it with `pip install -e .`")
    return ctypes.CDLL(str(path))
