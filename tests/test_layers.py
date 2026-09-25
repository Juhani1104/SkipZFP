import numpy as np
import pytest
import zarr

from skipzfp import query

from helpers import LAYERS, ORDERS, make_array, smooth_field


@pytest.fixture(scope="module")
def layered(tmp_path_factory):
    a = smooth_field(shape=(128, 32, 64), seed=2)
    base = tmp_path_factory.mktemp("layered")
    ref = {r: make_array(base / f"plain{r}", a, rate=r)[:] for r in LAYERS}
    arrs = {
        o: make_array(base / f"lay{o}", a, rate=8.0, layers=LAYERS, block_order=o)
        for o in ORDERS
    }
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
        err = (
            np.abs(ref[r].astype(np.float64) - a)
            .reshape(2, 64, 2, 16, 2, 32)
            .max(axis=(1, 3, 5))
        )
        assert np.all(eps[..., k] >= err)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("k", range(len(LAYERS)))
@pytest.mark.parametrize("q", (0.001, 0.1, 0.5))
@pytest.mark.parametrize("max_req", (1, 10**9))
def test_layered_query_matches_plain_rate(layered, order, k, q, max_req):
    a, ref, arrs = layered
    th = float(np.quantile(a, 1 - q))
    res = query.query_gt(
        arrs[order], th, layer=k, merge_gap_blocks=4, max_chunk_requests=max_req
    )
    assert res.count == int((ref[LAYERS[k]].astype(np.float64) > th).sum())
