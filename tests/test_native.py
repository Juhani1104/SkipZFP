import ctypes

import numpy as np
import pytest

from skipzfp import _native, codec

from helpers import SHAPE, smooth_field, true_blocks

P = ctypes.c_void_p
SZ = ctypes.c_size_t
DBL = ctypes.c_double
INT = ctypes.c_int
PSZ = ctypes.POINTER(ctypes.c_size_t)

SIGNATURES = {
    "szfp_layout": [SZ, SZ, SZ, DBL, INT, PSZ, PSZ],
    "szfp_encode": [P, SZ, SZ, SZ, DBL, INT, P, SZ, PSZ],
    "szfp_decode": [P, SZ, SZ, SZ, SZ, DBL, INT, P, SZ],
    "szfp_meta": [P, SZ, SZ, SZ, DBL, INT, P, SZ],
    "szfp_decode_block": [P, SZ, DBL, INT, INT, P],
    "szfp_plan_gt": [P, SZ, SZ, SZ, DBL, INT, P, P, SZ, PSZ, PSZ, PSZ],
    "szfp_count_gt_blocks": [P, SZ, SZ, DBL, INT, DBL, INT, PSZ],
    "szfp_decode_blocks": [P, SZ, SZ, DBL, INT, INT, P],
    "szfp_merge_ranges": [P, P, SZ, SZ, P, P, P, P, PSZ],
    "szfp_count_offsets": [P, SZ, P, SZ, SZ, DBL, INT, DBL, PSZ],
}
OK, ERR_NULL, ERR_ARG, ERR_SIZE = 0, 1, 7, 8
RATE = 8.0
NB = int(64 * RATE / 8)


@pytest.fixture(scope="module")
def lib():
    # a private handle, so these loose signatures do not leak into the package
    lib = ctypes.CDLL(str(_native.load_library()._name))
    for name, args in SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int
    return lib


@pytest.fixture(scope="module")
def encoded():
    a = smooth_field()
    buf = codec.native().encode(a, RATE, 4)
    full = true_blocks(codec.native().decode(buf, SHAPE, RATE, 4))
    return a, buf, full


def ptr(a):
    return a.ctypes.data


@pytest.mark.parametrize("threads", (1, 4))
def test_decode_blocks_matches_full_decode(lib, encoded, threads):
    _, buf, full = encoded
    ids = np.array([0, 3, 7, 100, 250, 511])
    blocks = np.ascontiguousarray(buf.reshape(-1, NB)[ids])
    out = np.empty((len(ids), 64), np.float32)
    rc = lib.szfp_decode_blocks(ptr(blocks), len(ids), NB, RATE, 4, threads, ptr(out))
    assert rc == OK
    assert np.array_equal(out, full[ids])


def test_decode_blocks_empty_is_ok(lib):
    assert lib.szfp_decode_blocks(None, 0, NB, RATE, 4, 1, None) == OK


def test_count_offsets_matches_numpy(lib, encoded):
    _, buf, full = encoded
    offs = np.arange(0, len(full), 5, dtype=np.uint64) * np.uint64(NB)
    th = float(np.median(full))
    out = ctypes.c_size_t()
    rc = lib.szfp_count_offsets(
        ptr(buf), buf.size, ptr(offs), len(offs), NB, RATE, 4, th, ctypes.byref(out)
    )
    assert rc == OK
    assert out.value == int((full[::5].astype(np.float64) > th).sum())


def _out():
    return ctypes.byref(ctypes.c_size_t())


KEEP = []  # arrays whose addresses are handed to C must outlive the call


def error_cases(buf, a):
    small = np.empty(8, np.uint8)
    f = np.empty(a.size, np.float32)
    ids = np.zeros(4, np.uint32)
    u64 = np.zeros(4, np.uint64)
    past_end = u64 + np.uint64(buf.size)
    KEEP[:] = [small, f, ids, u64, past_end]
    x, y, z = SHAPE
    return {
        "layout rate 0": ("szfp_layout", (x, y, z, 0.0, 4, _out(), _out()), ERR_ARG),
        "layout bad block_dim": ("szfp_layout", (x, y, z, RATE, 3, _out(), _out())),
        "layout ragged": ("szfp_layout", (x, y, 30, RATE, 4, _out(), _out())),
        "layout null out": ("szfp_layout", (x, y, z, RATE, 4, None, None), ERR_NULL),
        "encode null": ("szfp_encode", (None, x, y, z, RATE, 4, None, 0, _out())),
        "encode small out": (
            "szfp_encode",
            (ptr(a), x, y, z, RATE, 4, ptr(small), small.size, _out()),
        ),
        "decode null": ("szfp_decode", (None, 0, x, y, z, RATE, 4, None, 0)),
        "decode short src": (
            "szfp_decode",
            (ptr(buf), 8, x, y, z, RATE, 4, ptr(f), f.size),
        ),
        "decode small out": (
            "szfp_decode",
            (ptr(buf), buf.size, x, y, z, RATE, 4, ptr(f), 8),
        ),
        "meta null": ("szfp_meta", (None, x, y, z, RATE, 4, None, 0)),
        "meta small": ("szfp_meta", (ptr(a), x, y, z, RATE, 4, ptr(small), 8)),
        "decode_block null": ("szfp_decode_block", (None, NB, RATE, 3, 3, None)),
        "decode_block wrong size": (
            "szfp_decode_block",
            (ptr(buf), NB - 1, RATE, 3, 3, ptr(f)),
            ERR_SIZE,
        ),
        "plan null": (
            "szfp_plan_gt",
            (None, 1, 0, 1, 0.0, 1, None, None, 0, _out(), _out(), _out()),
        ),
        "count_gt null": ("szfp_count_gt_blocks", (None, 1, NB, RATE, 4, 0.0, 1, None)),
        "decode_blocks null": ("szfp_decode_blocks", (None, 1, NB, RATE, 4, 1, None)),
        "merge null": (
            "szfp_merge_ranges",
            (None, None, 1, 0, None, None, None, None, None),
        ),
        "merge null out": (
            "szfp_merge_ranges",
            (ptr(ids), ptr(ids), 4, 0, None, ptr(u64), ptr(u64), ptr(u64), _out()),
        ),
        "count_offsets null": (
            "szfp_count_offsets",
            (None, 0, None, 1, NB, RATE, 4, 0.0, _out()),
        ),
        "count_offsets past end": (
            "szfp_count_offsets",
            (ptr(buf), buf.size, ptr(past_end), 1, NB, RATE, 4, 0.0, _out()),
        ),
    }


CASES = list(error_cases(np.zeros(1, np.uint8), np.zeros(SHAPE, np.float32)))


@pytest.mark.parametrize("case", CASES)
def test_native_rejects_bad_input(lib, encoded, case):
    a, buf, _ = encoded
    name, args, *want = error_cases(buf, a)[case]
    rc = getattr(lib, name)(*args)
    assert rc != OK
    if want:
        assert rc == want[0]
