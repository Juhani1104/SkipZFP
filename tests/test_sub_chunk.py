import numpy as np
import pytest
import zarr

from skipzfp import codec, query

from helpers import LAYERS, SHAPE, make_array, smooth_field

SUB_CASES = [
    ((64, 32, 64), {}),
    ((128, 32, 64), {}),
    ((128, 32, 64), dict(layers=LAYERS)),
    ((128, 32, 64), dict(block_order=(1, 2, 0))),
    ((128, 32, 64), dict(layers=LAYERS, block_order=(2, 0, 1))),
]


@pytest.fixture(scope="module")
def subbed(tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=3)
    base = tmp_path_factory.mktemp("sub")
    ref = {r: make_array(base / f"plain{r}", a, rate=r)[:] for r in LAYERS}
    arrs = {}
    for i, (chunk, kw) in enumerate(SUB_CASES):
        g = zarr.open_group(str(base / f"s{i}"), mode="w")
        z = g.create_array(
            "data",
            shape=a.shape,
            chunks=chunk,
            dtype="float32",
            serializer=codec.SkipZFPCodec(rate=8.0, sub_chunk=SHAPE, **kw),
            compressors=None,
            filters=None,
        )
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
        res = query.query_gt(
            arrs[case], th, layer=k, merge_gap_blocks=gap, max_chunk_requests=max_req
        )
        assert res.count == int((ref[r].astype(np.float64) > th).sum())
