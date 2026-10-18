"""Filtered aggregates are exact on the decoded values and bound the original ones."""

import numpy as np
import pytest
import zarr
from zarr.core.sync import sync

from skipzfp import (
    QueryOptions,
    aggregate_gt,
    aggregate_lt,
    aggregate_range,
    codec,
    query,
)

from helpers import (
    LAYERS,
    RATES,
    SHAPE,
    block_eps,
    decode_one_block,
    make_array,
    smooth_field,
    true_blocks,
)


def brute(a, rec, lo, hi):
    """Exact aggregates of lo < x <= hi on the decoded and the original values."""
    m = (rec > lo) & (rec <= hi)
    o = (a > lo) & (a <= hi)
    x = rec[m].astype(np.float64)
    y = a[o].astype(np.float64)
    return m.sum(), x.sum(), y.size, y.sum(), (y.mean() if y.size else None)


def check(res, a, rec, lo, hi):
    n, s, on, osum, omean = brute(a, rec, lo, hi)
    assert res.count == n
    assert res.sum == pytest.approx(s, rel=1e-12, abs=1e-9)
    assert res.mean == (pytest.approx(s / n, rel=1e-12) if n else None)
    assert res.original_count[0] <= on <= res.original_count[1]
    assert res.original_sum[0] <= osum <= res.original_sum[1]
    if res.original_mean is None:
        assert on == 0
    elif on:
        assert res.original_mean[0] <= omean <= res.original_mean[1]
    assert res.in_blocks + res.out_blocks + res.maybe_blocks == res.total_blocks


@pytest.fixture(scope="module", params=RATES)
def stored(request, tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=1)
    z = make_array(tmp_path_factory.mktemp("agg"), a, rate=request.param)
    return a, z[:], z


@pytest.mark.parametrize("q", (0.0, 0.001, 0.1, 0.5, 0.9, 0.999))
def test_aggregate_gt_matches_brute_force(stored, q):
    a, rec, z = stored
    th = float(np.quantile(a, q))
    check(aggregate_gt(z, th), a, rec, th, np.inf)


@pytest.mark.parametrize("q", (0.001, 0.5, 1.0))
def test_aggregate_lt_matches_brute_force(stored, q):
    a, rec, z = stored
    th = float(np.quantile(a, q))
    # x < th is x <= the largest float32 below th
    below = float(np.nextafter(np.float32(th), np.float32(-np.inf)))
    check(aggregate_lt(z, th), a, rec, -np.inf, below)


@pytest.mark.parametrize("qs", ((0.1, 0.2), (0.4, 0.6), (0.0, 1.0), (0.5, 0.5)))
def test_aggregate_range_matches_brute_force(stored, qs):
    a, rec, z = stored
    lo, hi = (float(np.quantile(a, q)) for q in qs)
    check(aggregate_range(z, lo, hi), a, rec, lo, hi)


def test_aggregate_counts_like_query(stored):
    a, _, z = stored
    th = float(np.quantile(a, 0.3))
    r, q = aggregate_gt(z, th), query.query_gt(z, th)
    assert r.count == q.count
    assert (r.in_blocks, r.out_blocks) == (q.in_blocks, q.out_blocks)
    assert r.maybe_blocks == q.maybe_blocks


def test_aggregate_reads_in_and_maybe_blocks_only(stored):
    a, _, z = stored
    th = float(np.quantile(a, 0.5))
    lt = query.get_layout(z)
    r = aggregate_gt(z, th)
    assert r.in_blocks and r.out_blocks and r.maybe_blocks
    assert r.useful_payload_bytes == (r.in_blocks + r.maybe_blocks) * lt.block_size
    assert r.payload_bytes_read == r.useful_payload_bytes  # no merging by default
    assert r.payload_overread_bytes == 0
    assert r.metadata_bytes_read > 0 and r.payload_requests > 0


def test_aggregate_merged_reads_give_the_same_answer(stored):
    a, rec, z = stored
    th = float(np.quantile(a, 0.5))
    plain = aggregate_gt(z, th)
    merged = aggregate_gt(z, th, options=QueryOptions(merge_gap_blocks=10**6))
    assert merged.count == plain.count and merged.sum == pytest.approx(plain.sum)
    assert merged.original_count == plain.original_count
    assert merged.payload_requests < plain.payload_requests
    assert merged.payload_overread_bytes > 0
    check(merged, a, rec, th, np.inf)


def test_empty_selection(stored):
    a, rec, z = stored
    r = aggregate_gt(z, float(a.max()) + 100.0)
    assert (r.count, r.sum, r.mean) == (0, 0.0, None)
    assert r.original_count == (0, 0) and r.original_mean is None
    assert r.in_blocks == 0 and r.maybe_blocks == 0
    assert r.payload_bytes_read == 0 and r.payload_requests == 0


def test_threshold_on_a_decoded_value(stored):
    """x > v leaves out the values equal to v, and x >= v keeps them."""
    a, rec, z = stored
    v = float(np.sort(rec.ravel())[rec.size // 3])
    check(aggregate_gt(z, v), a, rec, v, np.inf)
    below = float(np.nextafter(np.float32(v), np.float32(-np.inf)))
    r = aggregate_gt(z, v, inclusive=True)
    check(r, a, rec, below, np.inf)
    assert r.count > aggregate_gt(z, v).count


@pytest.mark.parametrize("q", (0.1, 0.5, 0.9))
def test_values_of_both_signs(tmp_path, q):
    a = smooth_field(shape=(128, 32, 64), seed=2) - np.float32(285.0)
    assert a.min() < 0 < a.max()
    z = make_array(tmp_path, a, rate=8.0)
    rec = z[:]
    th = float(np.quantile(a, q))
    check(aggregate_gt(z, th), a, rec, th, np.inf)
    check(aggregate_range(z, -1.0, 1.0), a, rec, -1.0, 1.0)


def test_constant_field(tmp_path):
    a = np.full((64, 16, 32), 3.5, np.float32)
    z = make_array(tmp_path, a, rate=8.0)
    rec = z[:]
    for th, inclusive in ((3.5, False), (3.5, True), (3.0, False), (4.0, False)):
        r = aggregate_gt(z, th, inclusive=inclusive)
        lo = (
            float(np.nextafter(np.float32(th), np.float32(-np.inf)))
            if inclusive
            else th
        )
        check(r, a, rec, lo, np.inf)


@pytest.fixture(scope="module")
def subbed(tmp_path_factory):
    """Arrays whose chunks hold several sub-chunks, each with its own error bound."""
    a = smooth_field(shape=(128, 32, 64), seed=3)
    base = tmp_path_factory.mktemp("sub")
    out = {}
    for name, kw in (
        ("plain", {}),
        ("order", dict(block_order=(1, 2, 0))),
        ("layers", dict(layers=LAYERS)),
    ):
        g = zarr.open_group(str(base / name), mode="w")
        z = g.create_array(
            "data",
            shape=a.shape,
            chunks=(128, 32, 64),
            dtype="float32",
            serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=SHAPE, **kw),
            compressors=None,
            filters=None,
        )
        z[:] = a
        codec.write_meta(z, a)
        out[name] = z
    return a, out


@pytest.mark.parametrize("name", ("plain", "order"))
@pytest.mark.parametrize("q", (0.01, 0.5, 0.99))
def test_sub_chunks_use_their_own_error_bounds(subbed, name, q):
    a, arrs = subbed
    z = arrs[name]
    th = float(np.quantile(a, q))
    check(aggregate_gt(z, th), a, z[:], th, np.inf)


def test_layered_array_needs_layer_0(subbed):
    a, arrs = subbed
    z = arrs["layers"]
    th = float(np.quantile(a, 0.5))
    with pytest.raises(ValueError, match="layer 0"):
        aggregate_gt(z, th)
    with pytest.raises(ValueError, match="layer 0"):
        aggregate_gt(z, th, options=QueryOptions(layer=1))
    r = aggregate_gt(z, th, options=QueryOptions(layer=0))
    q = query.query_gt(z, th, options=QueryOptions(layer=0))
    assert r.count == q.count
    assert r.original_count[0] <= int((a > th).sum()) <= r.original_count[1]


def test_bad_bounds_are_rejected(stored):
    _, _, z = stored
    with pytest.raises(ValueError):
        aggregate_gt(z, float("nan"))
    with pytest.raises(ValueError):
        aggregate_range(z, 2.0, 1.0)
    with pytest.raises(TypeError):
        aggregate_range(z, None, 1.0)


def test_native_aggregate_checks_its_input():
    nat = query.native()
    empty = nat.aggregate_offsets(
        np.zeros(0, np.uint8),
        np.zeros(0, np.uint64),
        np.zeros(0, np.float32),
        np.zeros(0, np.uint8),
        64,
        8.0,
        4,
        0.0,
        np.inf,
    )
    assert empty.rec_count == 0 and empty.maybe_min == np.inf
    buf = np.zeros(64, np.uint8)
    for eps in (-1.0, np.nan, np.inf):
        with pytest.raises(RuntimeError, match="SZ_ERR_ARG"):
            nat.aggregate_offsets(
                buf,
                np.zeros(1, np.uint64),
                np.array([eps], np.float32),
                np.zeros(1, np.uint8),
                64,
                8.0,
                4,
                0.0,
                np.inf,
            )
    with pytest.raises(RuntimeError, match="SZ_ERR_SIZE"):  # block past the end
        nat.aggregate_offsets(
            buf,
            np.array([8], np.uint64),
            np.zeros(1, np.float32),
            np.zeros(1, np.uint8),
            64,
            8.0,
            4,
            0.0,
            np.inf,
        )


@pytest.mark.parametrize("order", ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
def test_each_block_gets_its_own_sub_chunks_error_bound(tmp_path, order):
    """block_eps gives every block the error bound of the sub-chunk it lies in.

    Two chunks of 2 x 2 x 4 sub-chunks each, so that every axis of the chunk
    and of the sub-chunk grid matters. Sub-chunk i holds values near 1000 * i,
    with noise that grows with i so the error bounds differ, and decoding a
    stored block tells which sub-chunk it came from, whatever the block order.
    """
    rng = np.random.default_rng(5)
    a = smooth_field(shape=(256, 32, 128), seed=4)
    units = list(np.ndindex(4, 2, 4))
    want = {}
    for i, (t, y, x) in enumerate(units):
        sl = np.s_[t * 64 : (t + 1) * 64, y * 16 : (y + 1) * 16, x * 32 : (x + 1) * 32]
        noise = 0.01 * 1.3**i * rng.standard_normal((64, 16, 32))
        a[sl] += np.float32(1000 * i) + noise.astype(np.float32)
        want[i] = float(
            np.frombuffer(codec.native().meta(a[sl], 8.0, 4)[8:12], "<f4")[0]
        )
    assert len(set(want.values())) >= len(units) - 2  # nearly all differ

    g = zarr.open_group(str(tmp_path), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=(128, 32, 128),
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=SHAPE, block_order=order),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)

    lt = query.get_layout(z)
    assert lt.unit_grid_per_chunk == (2, 2, 4) and lt.grid_shape == (2, 1, 1)
    meta, _, _ = sync(query.read_meta(z, lt))
    chunk_ids, block_ids, _, n_in, _, got = query.native().plan_read(
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
    assert n_in == len(block_ids) == 2 * lt.blocks_per_chunk
    assert np.array_equal(
        got,
        block_eps(
            meta,
            lt.meta_size,
            lt.unit_grid,
            lt.unit_grid_per_chunk,
            lt.grid_shape,
            lt.blocks_per_unit,
            chunk_ids,
            block_ids,
        ),
    )
    payloads = [
        np.fromfile(tmp_path / "data" / "c" / str(c) / "0" / "0", np.uint8)
        for c in range(2)
    ]
    for c, b, e in zip(chunk_ids, block_ids, got):
        vals = decode_one_block(payloads[int(c)], int(b), 8.0)
        unit = int(round(float(vals.mean()) / 1000))
        assert e == np.float32(want[unit])


@pytest.mark.parametrize("q", (0.01, 0.5, 0.99))
def test_read_blocks_are_exactly_the_blocks_that_can_match(stored, q):
    """Every block left unread has no matching value, decoded or original.

    The listed blocks are the IN and MAYBE ones, so no block that holds a
    matching value is skipped, and the IN blocks are read rather than counted.
    """
    a, rec, z = stored
    th = float(np.quantile(a, q))
    r = aggregate_gt(z, th, return_blocks=True)
    assert aggregate_gt(z, th).read_blocks is None
    lt = query.get_layout(z)
    listed = {tuple(map(int, x)) for x in r.read_blocks}
    assert len(listed) == len(r.read_blocks) == r.in_blocks + r.maybe_blocks
    for c in range(int(np.prod(lt.grid_shape))):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(i * s, (i + 1) * s) for i, s in zip(coord, SHAPE))
        ob, rb = true_blocks(a[sl]), true_blocks(rec[sl])
        hit = (ob > th).any(axis=1) | (rb > th).any(axis=1)
        for b in range(lt.blocks_per_chunk):
            assert ((c, b) in listed) >= bool(hit[b])
            if (c, b) not in listed:
                assert (ob[b] <= th).all() and (rb[b] <= th).all()
