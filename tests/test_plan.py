import asyncio

import numpy as np
import pytest
import zarr

from skipzfp import codec, query

from helpers import LAYERS, SHAPE, make_array, smooth_field

FULL = (128, 32, 64)
CASES = [
    (SHAPE, {}),
    (SHAPE, dict(block_order=(1, 2, 0))),
    ((64, 32, 64), dict(sub_chunk=SHAPE)),
    ((64, 16, 64), dict(sub_chunk=SHAPE, block_order=(2, 0, 1))),
    ((128, 32, 64), dict(sub_chunk=SHAPE, layers=LAYERS)),
    ((128, 32, 64), dict(sub_chunk=(32, 16, 16), block_order=(1, 2, 0))),
]
QUANTILES = (0.0, 0.001, 0.1, 0.5, 0.9, 1.0)


@pytest.fixture(scope="module")
def arrays(tmp_path_factory):
    a = smooth_field(shape=FULL, seed=5)
    base = tmp_path_factory.mktemp("plan")
    out = []
    for i, (chunk, kw) in enumerate(CASES):
        if chunk == SHAPE:
            z = make_array(base / f"c{i}", a, rate=8.0, **kw)
        else:
            g = zarr.open_group(str(base / f"c{i}"), mode="w")
            z = g.create_array(
                "data",
                shape=a.shape,
                chunks=chunk,
                dtype="float32",
                serializer=codec.SkipZFPCodec(rate=8.0, **kw),
                compressors=None,
                filters=None,
            )
            z[:] = a
            codec.write_meta(z, a)
        out.append(z)
    return a, out


def old_plan(meta, lt, th):
    unit_ids, ranks, n_in, n_out = query.native().plan(
        meta, int(np.prod(lt.unit_grid)), lt.meta_size, lt.blocks_per_unit, th, 0
    )
    chunks, blocks = query.unit_to_chunk(lt, unit_ids, ranks)
    return chunks, blocks, n_in, n_out


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("q", QUANTILES)
@pytest.mark.parametrize("threads", (1, 0))
def test_chunk_order_plan_matches_unit_plan(arrays, case, q, threads):
    a, arrs = arrays
    z = arrs[case]
    lt = query.get_layout(z)
    meta, _, _ = asyncio.run(query.read_meta(z, lt))
    meta = query.plan_meta(meta, lt, len(lt.layers) - 1)
    size = lt.meta_size - 4 * (len(lt.layers) - 1)
    th = float(np.quantile(a, q)) + (1.0 if q == 1.0 else 0.0)
    subs = tuple(c // u for c, u in zip(lt.chunk_shape, lt.unit_shape))
    new = query.native().plan_chunks(
        meta, size, lt.unit_grid, subs, lt.blocks_per_unit, th, threads
    )
    lt_old = query._Layout(**{**lt.__dict__, "meta_size": size})
    old = old_plan(meta, lt_old, th)
    assert np.array_equal(new[0], old[0])
    assert np.array_equal(new[1], old[1])
    assert new[2:] == old[2:]
    assert new[2] + new[3] + len(new[1]) == lt.blocks_per_chunk * int(
        np.prod(lt.grid_shape)
    )


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("q", (0.001, 0.5))
def test_query_same_with_either_planner(arrays, case, q, monkeypatch):
    a, arrs = arrays
    th = float(np.quantile(a, q))
    results = []
    for fused in (False, True):
        monkeypatch.setattr(query, "FUSED_PLAN", fused)
        r = query.query_gt(arrs[case], th, merge_gap_blocks=16)
        results.append(
            (r.count, r.maybe_blocks, r.payload_bytes_read, r.payload_requests)
        )
    assert results[0] == results[1]


def test_plan_chunks_rejects_bad_geometry():
    meta = np.zeros(12 + 2 * 8, dtype=np.uint8)
    with pytest.raises(RuntimeError):
        query.native().plan_chunks(meta, 12 + 2 * 8, (2, 1, 1), (3, 1, 1), 8, 0.0, 1)
    with pytest.raises(RuntimeError):
        query.native().plan_chunks(meta, 12 + 2 * 7, (1, 1, 1), (1, 1, 1), 8, 0.0, 1)


@pytest.mark.parametrize("gap", (0, 3, 1000))
def test_merge_ranges_sorts_unsorted_input(gap):
    rng = np.random.default_rng(1)
    chunks = np.repeat(np.arange(5, dtype=np.uint32), 40)
    blocks = np.tile(np.arange(0, 400, 10, dtype=np.uint32), 5)
    keys = [f"k{i}" for i in range(5)]
    ref, ref_blocks = query.merge_ranges(keys, chunks, blocks, 64, gap)
    perm = rng.permutation(len(blocks))
    got, got_blocks = query.merge_ranges(keys, chunks[perm], blocks[perm], 64, gap)
    assert got == ref
    assert np.array_equal(got_blocks, ref_blocks)


def test_plan_meta_single_layer_keeps_the_bytes(arrays):
    _, arrs = arrays
    z = arrs[0]
    lt = query.get_layout(z)
    meta, _, _ = asyncio.run(query.read_meta(z, lt))
    out = query.plan_meta(meta, lt, 0)
    assert out.shape == (meta.size // lt.meta_size, lt.meta_size)
    assert np.array_equal(out.reshape(-1), meta)
