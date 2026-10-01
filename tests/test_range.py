import math

import numpy as np
import pytest

from skipzfp import query, range_query

from helpers import LAYERS, make_array, smooth_field

FULL = (128, 32, 64)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    a = smooth_field(shape=FULL, seed=11)
    base = tmp_path_factory.mktemp("range")
    arrs = {
        "plain": make_array(base / "plain", a, rate=8.0),
        "sub_t": make_array(base / "sub", a, rate=8.0, block_order=(1, 2, 0)),
        "layers": make_array(base / "lay", a, rate=LAYERS[-1], layers=LAYERS),
    }
    dec = {k: z[:].astype(np.float64) for k, z in arrs.items()}
    plain = {r: make_array(base / f"p{r}", a, rate=r)[:] for r in LAYERS}
    return a, arrs, dec, plain


def brute(v, lo, hi, lo_inc=False, hi_inc=True):
    v = v.astype(np.float64)
    m = np.ones(v.shape, bool)
    if lo is not None:
        m &= v >= lo if lo_inc else v > lo
    if hi is not None:
        m &= v <= hi if hi_inc else v < hi
    return int(m.sum())


def bounds(a):
    q = lambda p: float(np.quantile(a, p))  # noqa: E731
    return [
        (q(0.3), None),
        (None, q(0.3)),
        (q(0.2), q(0.7)),
        (q(0.5), q(0.5001)),
        (q(0.999), q(0.9999)),
        (float(a.min()) - 1, float(a.max()) + 1),
        (q(0.4), q(0.4)),
        (None, None),
    ]


@pytest.mark.parametrize("name", ("plain", "sub_t", "layers"))
@pytest.mark.parametrize("k", range(8))
@pytest.mark.parametrize("gap", (0, 64))
def test_range_matches_brute_force(data, name, k, gap):
    a, arrs, dec, _ = data
    lo, hi = bounds(a)[k]
    r = range_query.query_range(arrs[name], lo, hi, merge_gap_blocks=gap)
    assert r.count == brute(dec[name], lo, hi)
    assert r.in_blocks + r.out_blocks + r.maybe_blocks == r.total_blocks


@pytest.mark.parametrize("lo_inc", (False, True))
@pytest.mark.parametrize("hi_inc", (False, True))
def test_inclusive_ends_on_values_that_occur(data, lo_inc, hi_inc):
    _, arrs, dec, _ = data
    v = np.unique(dec["plain"])
    lo, hi = float(v[len(v) // 3]), float(v[len(v) // 3 + 50])
    r = range_query.query_range(
        arrs["plain"], lo, hi, lo_inclusive=lo_inc, hi_inclusive=hi_inc
    )
    assert r.count == brute(dec["plain"], lo, hi, lo_inc, hi_inc)


def test_single_value_and_empty_ranges(data):
    _, arrs, dec, _ = data
    x = float(np.unique(dec["plain"])[1000])
    point = range_query.query_range(arrs["plain"], x, x, lo_inclusive=True)
    assert point.count == int((dec["plain"] == x).sum()) > 0
    assert range_query.query_range(arrs["plain"], x, x).count == 0


@pytest.mark.parametrize("k", range(len(LAYERS)))
@pytest.mark.parametrize("max_req", (1, 10**9))
def test_range_on_each_layer(data, k, max_req):
    a, arrs, _, plain = data
    lo, hi = float(np.quantile(a, 0.25)), float(np.quantile(a, 0.6))
    r = range_query.query_range(
        arrs["layers"], lo, hi, layer=k, merge_gap_blocks=4, max_chunk_requests=max_req
    )
    assert r.count == brute(plain[LAYERS[k]], lo, hi)


def test_gt_is_the_open_upper_range(data):
    a, arrs, _, _ = data
    t = float(np.quantile(a, 0.8))
    gt = query.query_gt(arrs["sub_t"], t)
    rng = range_query.query_range(arrs["sub_t"], lo=t)
    assert (gt.count, gt.maybe_blocks, gt.in_blocks) == (
        rng.count,
        rng.maybe_blocks,
        rng.in_blocks,
    )


def test_below_and_at_or_above_add_up(data):
    a, arrs, dec, _ = data
    t = float(np.unique(dec["plain"])[5000])
    below = range_query.query_range(arrs["plain"], hi=t, hi_inclusive=False).count
    at_or_above = range_query.query_range(arrs["plain"], lo=t, lo_inclusive=True).count
    assert below + at_or_above == a.size


@pytest.mark.parametrize(
    "lo, hi", ((2.0, 1.0), (math.nan, 1.0), (0.0, math.nan)), ids=("order", "lo", "hi")
)
def test_bad_bounds(data, lo, hi):
    _, arrs, _, _ = data
    with pytest.raises(ValueError):
        range_query.query_range(arrs["plain"], lo, hi)


@pytest.mark.parametrize("x", (1.1, 0.5, -3.25, 1e-40))
def test_range_bounds_use_float32_neighbours(x):
    # smallest float32 >= x, and the float32 just below it
    f = np.float32(x)
    if float(f) < x:
        f = np.nextafter(f, np.float32(np.inf))
    below = float(np.nextafter(f, np.float32(-np.inf)))
    a, b = range_query.range_bounds(x, x, lo_inclusive=True, hi_inclusive=False)
    assert a == below  # x >= value  <=>  value > below
    assert b == below  # x <  value  <=>  value <= below
    assert range_query.range_bounds(x, x) == (x, x)
    assert range_query.range_bounds(None, None) == (-math.inf, math.inf)
