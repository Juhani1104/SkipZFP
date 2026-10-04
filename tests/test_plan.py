"""The planner classifies every block correctly, for every storage layout."""

import asyncio

import numpy as np
import pytest
import zarr

from skipzfp import codec, query

from helpers import LAYERS, SHAPE, make_array, smooth_field, true_blocks

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


def stored_blocks(z, lt):
    """Decoded values of every block, indexed by (chunk id, position in storage)."""
    rec = z[:]
    rank = codec.block_rank(lt.unit_shape, codec.find_codec(z).block_order)
    out = []
    for coord in query.chunk_coords(lt.grid_shape):
        sl = tuple(slice(c * s, (c + 1) * s) for c, s in zip(coord, lt.chunk_shape))
        chunk = rec[sl]
        for usl in codec.unit_slices(lt.chunk_shape, lt.unit_shape):
            blocks = true_blocks(chunk[usl])  # C order within the unit
            ranked = np.empty_like(blocks)
            ranked[rank] = blocks
            out.append(ranked)
    return np.concatenate(out).reshape(-1, lt.blocks_per_chunk, 64)


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("q", QUANTILES)
@pytest.mark.parametrize("threads", (1, 0))
def test_plan_classifies_every_block_correctly(arrays, case, q, threads):
    """Check the planner's output against the decoded values of every block.

    MAYBE blocks must come out sorted by (chunk, position), and every other block
    must lie wholly above or wholly at or below the threshold, in numbers that
    match the IN and OUT counts. This holds for every layout below, including
    sub_chunk and a reordered block_order.
    """
    a, arrs = arrays
    z = arrs[case]
    lt = query.get_layout(z)
    meta, _, _ = asyncio.run(query.read_meta(z, lt))
    meta = query.plan_meta(meta, lt, len(lt.layers) - 1)
    size = lt.meta_size - 4 * (len(lt.layers) - 1)
    th = float(np.quantile(a, q)) + (1.0 if q == 1.0 else 0.0)
    subs = tuple(c // u for c, u in zip(lt.chunk_shape, lt.unit_shape))
    chunks, blocks, n_in, n_out = query.native().plan_chunks(
        meta, size, lt.unit_grid, subs, lt.blocks_per_unit, th, threads
    )

    key = chunks.astype(np.int64) * lt.blocks_per_chunk + blocks
    assert np.all(np.diff(key) > 0)
    vals = stored_blocks(z, lt).reshape(-1, 64)
    rest = np.ones(len(vals), bool)
    rest[key] = False
    assert (vals[rest] > th).all(axis=1).sum() == n_in
    assert (vals[rest] <= th).all(axis=1).sum() == n_out
    assert n_in + n_out + len(key) == len(vals)


def test_plan_chunks_rejects_bad_geometry():
    meta = np.zeros(12 + 2 * 8, dtype=np.uint8)
    with pytest.raises(RuntimeError):
        query.native().plan_chunks(meta, 12 + 2 * 8, (2, 1, 1), (3, 1, 1), 8, 0.0, 1)
    with pytest.raises(RuntimeError):
        query.native().plan_chunks(meta, 12 + 2 * 7, (1, 1, 1), (1, 1, 1), 8, 0.0, 1)
    with pytest.raises(RuntimeError):
        query.native().plan_chunks(
            meta, 12 + 2 * 8, (2, 1, 1), (3, 1, 1), 8, 0.0, 1, 1.0
        )
