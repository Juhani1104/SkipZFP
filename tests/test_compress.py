import numpy as np
import pytest
import zfpy

from skipzfp import codec, query

from helpers import RATES, SHAPE, decode_one_block, smooth_field, split, true_blocks


@pytest.mark.parametrize("rate", RATES)
def test_payload_matches_official_zfpy(rate):
    a = smooth_field()
    payload, *_ = split(a, rate)
    ref = np.frombuffer(zfpy.compress_numpy(a, rate=rate, write_header=False), np.uint8)
    assert np.array_equal(payload, ref[: payload.size])


def test_block_zero_is_contiguous_4x4x4():
    i, j, k = np.indices(SHAPE)
    coded = (i * 10000 + j * 100 + k).astype(np.float32)
    buf = codec.native().encode(coded, 32.0, 4)
    v = np.rint(decode_one_block(buf, 0, 32.0)).astype(int)
    covered = {(x // 10000, (x // 100) % 100, x % 100) for x in v}
    assert covered == {(a, b, c) for a in range(4) for b in range(4) for c in range(4)}


@pytest.mark.parametrize("rate", RATES)
def test_single_block_decode_is_bit_exact(rate):
    a = smooth_field()
    buf = codec.native().encode(a, rate, 4)
    full = true_blocks(codec.native().decode(buf, SHAPE, rate, 4))
    for bid in (0, 1, 7, 100, 511):
        assert np.array_equal(decode_one_block(buf, bid, rate), full[bid])


def test_constant_chunk():
    a = np.full(SHAPE, 3.5, np.float32)
    buf, cmin, cmax, eps, offs = split(a, 8.0)
    assert cmin == cmax == 3.5 and not offs.any()
    assert np.array_equal(codec.native().decode(buf, SHAPE, 8.0, 4), a)


@pytest.mark.parametrize("native", (codec.native, query.native))
def test_native_errors_become_exceptions(native):
    with pytest.raises(RuntimeError, match="szfp_layout failed"):
        native().layout(SHAPE, 0.0, 4)


def test_encode_rejects_non_3d():
    with pytest.raises(ValueError, match="3D"):
        codec.native().encode(np.zeros((4, 4), np.float32), 8.0, 4)
