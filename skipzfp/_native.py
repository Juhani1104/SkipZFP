from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path

# Result codes of every C function, the SzResult enum in csrc/util.h.
ERROR_NAMES = {
    0: "SZ_OK",
    1: "SZ_ERR_NULL",
    2: "SZ_ERR_STREAM",
    3: "SZ_ERR_ZFP",
    4: "SZ_ERR_FIELD",
    5: "SZ_ERR_DECOMPRESS",
    6: "SZ_ERR_DIMS",
    7: "SZ_ERR_ARG",
    8: "SZ_ERR_SIZE",
    9: "SZ_ERR_COMPRESS",
    10: "SZ_ERR_MALLOC",
}


def check(code: int, name: str) -> None:
    """Raise RuntimeError naming the C function and its result code unless code is 0."""
    if code != 0:
        msg = ERROR_NAMES.get(code, f"unknown error {code}")
        raise RuntimeError(f"{name} failed: {msg}")


@cache
def load_library() -> ctypes.CDLL:
    """Load the compiled C library _core.so, once per process."""
    path = Path(__file__).with_name("_core.so")
    if not path.exists():
        raise FileNotFoundError(
            f"could not find {path}; build it with `pip install -e .`"
        )
    return ctypes.CDLL(str(path))
