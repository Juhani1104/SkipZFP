from importlib.metadata import entry_points

import pytest

import skipzfp
from skipzfp import _native


def test_public_api():
    for name in skipzfp.__all__:
        assert hasattr(skipzfp, name)


def test_codec_entry_point():
    (ep,) = entry_points(group="zarr.codecs", name="skipzfp")
    assert ep.load() is skipzfp.SkipZFPCodec


def test_native_library_exports():
    lib = _native.load_library()
    for sym in ("szfp_layout", "szfp_plan_gt", "szfp_decode_block"):
        assert hasattr(lib, sym)


def test_missing_native_library(monkeypatch, tmp_path):
    monkeypatch.setattr(_native, "__file__", str(tmp_path / "_native.py"))
    with pytest.raises(FileNotFoundError, match="pip install -e"):
        _native.load_library.__wrapped__()
