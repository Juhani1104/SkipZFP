"""4-bit per-block error bounds: format, rounding and the queries that use them."""

import math

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync

from skipzfp import QueryOptions, aggregate_gt, codec, query

from helpers import SHAPE, block_eps, make_array, smooth_field, true_blocks


def test_config_round_trip_and_checks():
    c = codec.SkipZFPCodec(rate=8.0, block_error_bits=4)
    assert codec.SkipZFPCodec.from_dict(c.to_dict()) == c
    plain = codec.SkipZFPCodec(rate=8.0)
    assert "block_error_bits" not in plain.to_dict()["configuration"]
    assert codec.SkipZFPCodec.from_dict(plain.to_dict()).block_error_bits == 0
    with pytest.raises(ValueError, match="0 or 4"):
        codec.SkipZFPCodec(block_error_bits=2)
    with pytest.raises(ValueError, match="single layer"):
        codec.SkipZFPCodec(rate=8.0, layers=(4.0, 8.0), block_error_bits=4)


def test_record_size():
    """A 64 x 16 x 32 sub-chunk has 512 blocks, so 1036 + 256 = 1292 bytes."""
    c = codec.SkipZFPCodec(rate=8.0, sub_chunk=SHAPE, block_error_bits=4)
    assert codec.meta_size(c, SHAPE) == 1292
    assert codec.meta_size(codec.SkipZFPCodec(rate=8.0), SHAPE) == 1036
    assert codec.meta_size(c, (4, 4, 12)) == 12 + 2 * 3 + 2  # odd count rounds up


def test_codes_pack_round_trip():
    q = np.random.default_rng(0).integers(0, 16, 37).astype(np.uint8)
    assert np.array_equal(codec.unpack_codes(codec.pack_codes(q), q.size), q)


def test_codes_round_up_past_a_short_product():
    """When eps * q / 15 lands just under the error, the code is raised by one.

    For this eps and error, ceil(15 * err / eps) is 13, yet eps * 13 / 15 in
    double precision falls one ulp short of the error, so the code must be 14.
    """
    eps = 0.28531554341316223  # a float32
    err = math.nextafter(eps * 13 / 15.0, math.inf)
    assert math.ceil(15.0 * err / eps) == 13 and eps * 13 / 15.0 < err
    x = np.zeros((4, 4, 4), np.float32)
    rec = np.zeros((4, 4, 4), np.float64)
    rec[0, 0, 0] = err
    q = codec.block_error_codes(x, rec, eps)
    assert q.tolist() == [14]
    with pytest.raises(ValueError):
        codec.block_error_codes(x, rec, err / 2)


def test_planner_uses_each_blocks_own_code():
    """The planner classifies with eps * q / 15, reading the low nibble first.

    One unit of two blocks, both with block bounds [100, 200] (cmin 0, cmax 255,
    so offsets map to whole numbers), eps 15 and codes 3 and 12, so their error
    bounds are 3 and 12. A threshold just inside or outside 100 - bound tells
    whether each block got its own code, with the right formula.
    """
    rec = np.concatenate(
        [
            np.array([0.0, 255.0, 15.0], np.float32).view(np.uint8),
            np.array([100, 200, 100, 200], np.uint8),
            codec.pack_codes(np.array([3, 12], np.uint8)),
        ]
    )

    def states(th):
        chunks, blocks, st, n_in, n_out = query.native().plan_read(
            rec, rec.size, (1, 1, 1), (1, 1, 1), 2, th, np.inf, 1
        )
        out = dict(zip(blocks.tolist(), st.tolist()))
        return [out.get(b, 0) for b in (0, 1)]

    assert states(96.9) == [1, 2]  # 97 > 96.9 for block 0, 88 is not
    assert states(97.1) == [2, 2]  # within 3 of 100: block 0 may not match
    assert states(87.9) == [1, 1]
    assert states(212.1) == [0, 0]
    assert states(203.5) == [0, 2]  # 200 + 3 <= 203.5 but 200 + 12 is not


@pytest.fixture(scope="module", params=(2.0, 8.0))
def pair(request, tmp_path_factory):
    """The same data with and without per-block codes, one unit per chunk."""
    a = smooth_field(shape=(128, 32, 64), seed=6)
    base = tmp_path_factory.mktemp("bits")
    z0 = make_array(base / "plain", a, rate=request.param)
    z4 = make_array(base / "bits", a, rate=request.param, block_error_bits=4)
    return a, z0[:], z0, z4


def test_codes_bound_every_block(pair):
    a, rec, _, z4 = pair
    lt = query.get_layout(z4)
    assert lt.meta_size == 12 + 2 * lt.blocks_per_unit + lt.blocks_per_unit // 2
    meta, _, _ = sync(query.read_meta(z4, lt))
    recs = meta.reshape(-1, lt.meta_size)
    for c in range(int(np.prod(lt.grid_shape))):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(i * s, (i + 1) * s) for i, s in zip(coord, SHAPE))
        err = np.abs(true_blocks(a[sl]).astype(np.float64) - true_blocks(rec[sl])).max(
            1
        )
        eps = float(np.frombuffer(recs[c, 8:12].tobytes(), np.float32)[0])
        q = codec.unpack_codes(recs[c, 12 + 2 * lt.blocks_per_unit :], err.size)
        assert (eps * q.astype(np.float64) / 15.0 >= err).all()
        assert q.max() == 15  # the worst block sets the unit's bound


@pytest.mark.parametrize("q", (0.001, 0.1, 0.5, 0.9))
def test_counts_stay_exact_and_skip_more(pair, q):
    a, rec, z0, z4 = pair
    th = float(np.quantile(a, q))
    r0, r4 = query.query_gt(z0, th), query.query_gt(z4, th)
    assert r4.count == r0.count == int((rec > th).sum())
    assert r4.maybe_blocks <= r0.maybe_blocks
    assert r4.in_blocks >= r0.in_blocks and r4.out_blocks >= r0.out_blocks


@pytest.mark.parametrize("q", (0.1, 0.5, 0.9))
def test_in_and_out_still_hold(pair, q):
    """With per-block bounds, IN and OUT blocks are still right for both values."""
    a, rec, _, z4 = pair
    th = float(np.quantile(a, q))
    r = aggregate_gt(z4, th, return_blocks=True)
    lt = query.get_layout(z4)
    state = {}
    for c, b in r.read_blocks:
        state[(int(c), int(b))] = True
    planned = query.native().plan_read(
        sync(query.read_meta(z4, lt))[0],
        lt.meta_size,
        lt.unit_grid,
        lt.unit_grid_per_chunk,
        lt.blocks_per_unit,
        th,
        np.inf,
        1,
    )
    in_set = {
        (int(c), int(b))
        for c, b, s in zip(planned[0], planned[1], planned[2])
        if s == 1
    }
    for c in range(int(np.prod(lt.grid_shape))):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(i * s, (i + 1) * s) for i, s in zip(coord, SHAPE))
        ob, rb = true_blocks(a[sl]), true_blocks(rec[sl])
        for b in range(lt.blocks_per_chunk):
            if (c, b) in in_set:
                assert (ob[b] > th).all() and (rb[b] > th).all()
            elif (c, b) not in state:
                assert (ob[b] <= th).all() and (rb[b] <= th).all()


@pytest.mark.parametrize("q", (0.01, 0.5, 0.99))
def test_aggregate_bounds_hold_and_tighten(pair, q):
    a, rec, z0, z4 = pair
    th = float(np.quantile(a, q))
    r0, r4 = aggregate_gt(z0, th), aggregate_gt(z4, th)
    o = a > th
    assert r4.count == r0.count and r4.sum == r0.sum
    assert r4.original_count[0] <= o.sum() <= r4.original_count[1]
    mean = a[o].astype(np.float64).mean()
    assert r4.original_mean[0] <= mean <= r4.original_mean[1]
    w0 = r0.original_mean[1] - r0.original_mean[0]
    w4 = r4.original_mean[1] - r4.original_mean[0]
    assert w4 <= w0


@pytest.mark.parametrize("order", ((0, 1, 2), (1, 2, 0)))
def test_sub_chunks_with_codes(tmp_path, order):
    """Codes follow the block order, and each block's bound covers its error."""
    a = smooth_field(shape=(128, 32, 64), seed=7)
    g = zarr.open_group(str(tmp_path), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=a.shape,
        dtype="float32",
        serializer=codec.SkipZFPCodec(
            rate=4.0, sub_chunk=SHAPE, block_order=order, block_error_bits=4
        ),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)
    rec = z[:]
    lt = query.get_layout(z)
    meta, _, _ = sync(query.read_meta(z, lt))
    chunks, blocks, _, _, _, got = query.native().plan_read(
        meta,
        lt.meta_size,
        lt.unit_grid,
        lt.unit_grid_per_chunk,
        lt.blocks_per_unit,
        -np.inf,
        np.inf,
        1,
        eps=True,
    )
    assert np.array_equal(
        got,
        block_eps(
            meta,
            lt.meta_size,
            lt.unit_grid,
            lt.unit_grid_per_chunk,
            lt.grid_shape,
            lt.blocks_per_unit,
            chunks,
            blocks,
        ),
    )
    # the error of the block at each storage position, from the C-order blocks
    rank_of = codec.block_rank(SHAPE, order)
    units = list(np.ndindex(2, 2, 2))
    for k, b in enumerate(blocks):
        u, pos = divmod(int(b), lt.blocks_per_unit)
        t, y, x = units[u]
        sl = np.s_[t * 64 : (t + 1) * 64, y * 16 : (y + 1) * 16, x * 32 : (x + 1) * 32]
        i = int(np.flatnonzero(rank_of == pos)[0])
        err = np.abs(true_blocks(a[sl])[i].astype(np.float64) - true_blocks(rec[sl])[i])
        assert got[k] >= err.max()
    for q in (0.05, 0.5, 0.95):
        th = float(np.quantile(a, q))
        r = aggregate_gt(z, th, options=QueryOptions(merge_gap_blocks=0))
        assert r.count == int((rec > th).sum())
        assert r.original_count[0] <= int((a > th).sum()) <= r.original_count[1]
