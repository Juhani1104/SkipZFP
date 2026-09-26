import ctypes

import numpy as np
import pytest

from skipzfp import _native, codec

from helpers import smooth_field

P = ctypes.c_void_p
SZ = ctypes.c_size_t
DBL = ctypes.c_double
INT = ctypes.c_int
PSZ = ctypes.POINTER(ctypes.c_size_t)
SHAPE = (16, 16, 32)
RATES = (0.25, 1.0, 2.0, 3.5, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
LIBZFP, FAST, FAST_SCALAR = 0, 1, 2


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(_native.load_library()._name))
    lib.szfp_decode_blocks.argtypes = [P, SZ, SZ, DBL, INT, INT, P]
    lib.szfp_count_offsets.argtypes = [P, SZ, P, SZ, SZ, DBL, INT, DBL, PSZ]
    lib.szfp_fast_ok.argtypes = [INT, DBL, SZ]
    lib.szfp_set_fast_decode.argtypes = [INT]
    yield lib
    lib.szfp_set_fast_decode(FAST)


def fields():
    rng = np.random.default_rng(7)
    smooth = smooth_field()[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    wild = rng.standard_normal(SHAPE) * 10.0 ** rng.integers(-35, 35, SHAPE)
    wild[:4] = 0.0
    wild[4:8, :4] = 3.0e38
    wild[4:8, 4:8] = -1.0e-38
    ints = rng.integers(-8, 8, SHAPE)
    return {
        "smooth": np.ascontiguousarray(smooth, dtype=np.float32),
        "wild": wild.astype(np.float32),
        "ints": ints.astype(np.float32),
    }


def decode(lib, buf, nb, rate, mode):
    lib.szfp_set_fast_decode(mode)
    out = np.empty(buf.size // nb * 64, dtype=np.float32)
    code = lib.szfp_decode_blocks(
        buf.ctypes.data, buf.size // nb, nb, rate, 4, 1, out.ctypes.data
    )
    assert code == 0
    return out


@pytest.mark.parametrize("name", ("smooth", "wild", "ints"))
@pytest.mark.parametrize("rate", RATES)
def test_fast_decode_is_bit_exact(lib, name, rate):
    buf = codec.native().encode(fields()[name], rate, 4)
    nb = int(64 * rate / 8)
    ref = decode(lib, buf, nb, rate, LIBZFP).view(np.uint32)
    assert np.array_equal(decode(lib, buf, nb, rate, FAST).view(np.uint32), ref)
    assert np.array_equal(decode(lib, buf, nb, rate, FAST_SCALAR).view(np.uint32), ref)


@pytest.mark.parametrize("mode", (LIBZFP, FAST, FAST_SCALAR))
def test_count_offsets_same_with_and_without_fast_path(lib, mode):
    buf = codec.native().encode(fields()["smooth"], 8.0, 4)
    offsets = np.arange(0, buf.size, 64 * 3, dtype=np.uint64)
    threshold = float(np.median(fields()["smooth"]))
    lib.szfp_set_fast_decode(LIBZFP)
    ref = ctypes.c_size_t()
    args = (buf.ctypes.data, buf.size, offsets.ctypes.data, offsets.size, 64, 8.0, 4)
    assert lib.szfp_count_offsets(*args, threshold, ctypes.byref(ref)) == 0
    lib.szfp_set_fast_decode(mode)
    got = ctypes.c_size_t()
    assert lib.szfp_count_offsets(*args, threshold, ctypes.byref(got)) == 0
    assert got.value == ref.value > 0


def test_fast_ok_limits(lib):
    lib.szfp_set_fast_decode(FAST)
    assert lib.szfp_fast_ok(4, 8.0, 64)
    assert lib.szfp_fast_ok(4, 0.25, 2)
    assert lib.szfp_fast_ok(4, 32.0, 256)
    assert not lib.szfp_fast_ok(4, 0.125, 1)
    assert not lib.szfp_fast_ok(4, 33.0, 264)
    assert not lib.szfp_fast_ok(4, 8.0, 63)
    assert not lib.szfp_fast_ok(2, 8.0, 8)
    lib.szfp_set_fast_decode(LIBZFP)
    assert not lib.szfp_fast_ok(4, 8.0, 64)
