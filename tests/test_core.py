import ctypes

import numpy as np
import pytest
import zarr
import zfpy
from zarr.registry import register_codec

from skipzfp import codec, query

register_codec("skipzfp", codec.SkipZFPCodec)

SHAPE = (64, 16, 32)  # (time, lat, lon) chunk used in the paper
RATES = (2.0, 4.0, 8.0, 12.0)


def smooth_field(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    i, j, k = np.indices(shape)
    a = 280 + 10 * np.sin(k / 5) + 8 * np.cos(j / 3) + 3 * np.sin(i / 7)
    return (a + 0.1 * rng.standard_normal(shape)).astype(np.float32)


def true_blocks(a):
    """Group a C-order 3D array into 4x4x4 blocks, numbered in C order."""
    x, y, z = a.shape
    b = a.reshape(x // 4, 4, y // 4, 4, z // 4, 4).transpose(0, 2, 4, 1, 3, 5)
    return b.reshape(-1, 64)


def split(a, rate):
    buf = codec.native().encode(a, rate, 4)
    meta = codec.native().meta(a, rate, 4)
    cmin, cmax, eps = np.frombuffer(meta[:12].tobytes(), np.float32)
    offs = meta[12:].reshape(-1, 2).astype(np.float64)
    return buf, float(cmin), float(cmax), float(eps), offs


def decode_one_block(payload, block_id, rate):
    nbytes = int(64 * rate / 8)
    out = np.empty(64, np.float32)
    lib = codec.native().lib
    lib.szfp_decode_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    src = np.ascontiguousarray(payload[block_id * nbytes : (block_id + 1) * nbytes])
    rc = lib.szfp_decode_block(src.ctypes.data, nbytes, rate, 3, 3, out.ctypes.data)
    assert rc == 0
    return out


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


@pytest.mark.parametrize("rate", RATES)
def test_metadata_bounds_original_and_reconstructed(rate):
    a = smooth_field()
    buf, cmin, cmax, eps, offs = split(a, rate)
    rec = codec.native().decode(buf, SHAPE, rate, 4)

    assert cmin == a.min() and cmax == a.max()
    assert eps >= np.abs(rec.astype(np.float64) - a).max()

    span = cmax - cmin
    lo = cmin + offs[:, 0] / 255 * span
    hi = cmin + offs[:, 1] / 255 * span
    orig_b, rec_b = true_blocks(a), true_blocks(rec)
    assert np.all(lo <= orig_b.min(1)) and np.all(hi >= orig_b.max(1))
    assert np.all(lo - eps <= rec_b.min(1)) and np.all(hi + eps >= rec_b.max(1))


def test_constant_chunk():
    a = np.full(SHAPE, 3.5, np.float32)
    buf, cmin, cmax, eps, offs = split(a, 8.0)
    assert cmin == cmax == 3.5 and not offs.any()
    assert np.array_equal(codec.native().decode(buf, SHAPE, 8.0, 4), a)


@pytest.fixture(params=RATES)
def stored(request, tmp_path):
    rate = request.param
    a = smooth_field(shape=(128, 32, 64), seed=1)
    g = zarr.open_group(str(tmp_path / "a.zarr"), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(rate=rate),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)
    return a, z, rate


@pytest.mark.parametrize("q", (0.01, 0.1, 0.5, 0.9, 0.99))
def test_query_gt_count_is_exact_on_reconstruction(stored, q):
    a, z, _ = stored
    rec = z[:]
    th = float(np.quantile(a, q))
    res = query.query_gt(z, th)
    assert res.count == int((rec > th).sum())
    assert res.in_blocks + res.out_blocks + res.maybe_blocks == res.total_blocks


@pytest.mark.parametrize("q", (0.1, 0.5, 0.9))
def test_in_out_guarantees_hold_for_original_and_reconstruction(stored, q):
    a, z, rate = stored
    rec = z[:]
    th = np.float32(np.quantile(a, q))
    lt = query.get_layout(z)
    n_chunk = int(np.prod(lt.grid_shape))

    for c in range(n_chunk):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(ci * s, (ci + 1) * s) for ci, s in zip(coord, SHAPE))
        meta = codec.native().meta(a[sl], rate, 4)
        _, _, n_in, n_out = query.native().plan(meta, 1, lt.meta_size, lt.blocks_per_chunk, th, 1)
        states = classify(meta, lt, th)
        assert (states == 1).sum() == n_in and (states == 0).sum() == n_out
        ob, rb = true_blocks(a[sl]), true_blocks(rec[sl])
        assert np.all(ob[states == 1] > th) and np.all(rb[states == 1] > th)
        assert np.all(ob[states == 0] <= th) and np.all(rb[states == 0] <= th)


def classify(meta, lt, th):
    """Per-block states from the native planner: 1=IN, 0=OUT, 2=MAYBE."""
    chunk_ids, block_ids, _, _ = query.native().plan(
        meta, 1, lt.meta_size, lt.blocks_per_chunk, th, 1
    )
    states = np.full(lt.blocks_per_chunk, -1)
    states[block_ids] = 2
    # IN/OUT are not returned individually, so recover them from the metadata
    # using the same rule and check the totals match the planner.
    cmin, cmax, eps = np.frombuffer(meta[:12].tobytes(), np.float32).astype(np.float64)
    offs = meta[12 : lt.meta_size].reshape(-1, 2).astype(np.float64)
    span = cmax - cmin
    lo = cmin + offs[:, 0] / 255 * span - eps
    hi = cmin + offs[:, 1] / 255 * span + eps
    rest = states != 2
    states[rest & (lo > th)] = 1
    states[rest & (hi <= th)] = 0
    assert not (states == -1).any()
    return states


def test_threshold_compared_in_double(stored):
    a, z, _ = stored
    rec = z[:]
    v = np.float32(np.median(rec))
    # just below v in double, but rounds to v in float32
    th = float(v) - float(np.spacing(v)) / 4
    assert np.float32(th) == v
    res = query.query_gt(z, th)
    assert res.count == int((rec.astype(np.float64) > th).sum())


LAYERS = (2.0, 4.0, 8.0)
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def make_array(path, a, **kw):
    g = zarr.open_group(str(path), mode="w")
    z = g.create_array("data", shape=a.shape, chunks=SHAPE, dtype="float32",
                       serializer=codec.SkipZFPCodec(**kw), compressors=None, filters=None)
    z[:] = a
    codec.write_meta(z, a)
    return z


@pytest.fixture(scope="module")
def layered(tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=2)
    base = tmp_path_factory.mktemp("layered")
    ref = {r: make_array(base / f"plain{r}", a, rate=r)[:] for r in LAYERS}
    arrs = {o: make_array(base / f"lay{o}", a, rate=8.0, layers=LAYERS, block_order=o) for o in ORDERS}
    return a, ref, arrs


@pytest.mark.parametrize("order", ORDERS)
def test_layered_full_read_matches_plain(layered, order):
    a, ref, arrs = layered
    assert np.array_equal(arrs[order][:], ref[8.0])


@pytest.mark.parametrize("order", ORDERS)
def test_layered_meta_eps_covers_each_layer(layered, order):
    a, ref, arrs = layered
    z = arrs[order]
    meta = zarr.open_array(store=z.store_path.store, path="data_meta", mode="r")[:]
    eps = meta[..., 8:20].copy().view(np.float32)
    for k, r in enumerate(LAYERS):
        err = np.abs(ref[r].astype(np.float64) - a).reshape(2, 64, 2, 16, 2, 32).max(axis=(1, 3, 5))
        assert np.all(eps[..., k] >= err)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("k", range(len(LAYERS)))
@pytest.mark.parametrize("q", (0.001, 0.1, 0.5))
@pytest.mark.parametrize("max_req", (1, 10**9))
def test_layered_query_matches_plain_rate(layered, order, k, q, max_req):
    a, ref, arrs = layered
    th = float(np.quantile(a, 1 - q))
    res = query.query_gt(arrs[order], th, layer=k, merge_gap_blocks=4, max_chunk_requests=max_req)
    assert res.count == int((ref[LAYERS[k]].astype(np.float64) > th).sum())


SUB_CASES = [((64, 32, 64), {}), ((128, 32, 64), {}), ((128, 32, 64), dict(layers=LAYERS)),
             ((128, 32, 64), dict(block_order=(1, 2, 0))), ((128, 32, 64), dict(layers=LAYERS, block_order=(2, 0, 1)))]


@pytest.fixture(scope="module")
def subbed(tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=3)
    base = tmp_path_factory.mktemp("sub")
    ref = {r: make_array(base / f"plain{r}", a, rate=r)[:] for r in LAYERS}
    arrs = {}
    for i, (chunk, kw) in enumerate(SUB_CASES):
        g = zarr.open_group(str(base / f"s{i}"), mode="w")
        z = g.create_array("data", shape=a.shape, chunks=chunk, dtype="float32",
                           serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=SHAPE, **kw),
                           compressors=None, filters=None)
        z[:] = a
        codec.write_meta(z, a)
        arrs[i] = z
    return a, ref, arrs


@pytest.mark.parametrize("case", range(len(SUB_CASES)))
def test_sub_chunk_full_read_matches_small_chunks(subbed, case):
    a, ref, arrs = subbed
    assert np.array_equal(arrs[case][:], ref[8.0])


@pytest.mark.parametrize("case", range(len(SUB_CASES)))
@pytest.mark.parametrize("q", (0.001, 0.1, 0.5))
@pytest.mark.parametrize("gap", (0, 64, 10**6))
@pytest.mark.parametrize("max_req", (1, 10**9))
def test_sub_chunk_query_matches_small_chunks(subbed, case, q, gap, max_req):
    a, ref, arrs = subbed
    th = float(np.quantile(a, 1 - q))
    n_layer = len(SUB_CASES[case][1].get("layers", (8.0,)))
    for k in range(n_layer):
        r = LAYERS[k] if n_layer > 1 else 8.0
        res = query.query_gt(arrs[case], th, layer=k, merge_gap_blocks=gap, max_chunk_requests=max_req)
        assert res.count == int((ref[r].astype(np.float64) > th).sum())
