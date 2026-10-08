"""query_gt, query_lt and query_range match a brute-force count for any bounds."""

import math

import numpy as np
import pytest
import zarr

from skipzfp import query

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


def run(z, lo, hi, **kw):
    """Count lo < x <= hi with whichever public function fits the bounds."""
    if hi is None and lo is None:
        return query.query_range(z, -math.inf, math.inf, **kw)
    if hi is None:
        return query.query_gt(z, lo, **kw)
    if lo is None:
        return query.query_lt(z, hi, inclusive=True, **kw)
    return query.query_range(z, lo, hi, **kw)


@pytest.mark.parametrize("name", ("plain", "sub_t", "layers"))
@pytest.mark.parametrize("k", range(8))
@pytest.mark.parametrize("gap", (0, 64))
def test_range_matches_brute_force(data, name, k, gap):
    a, arrs, dec, _ = data
    lo, hi = bounds(a)[k]
    r = run(arrs[name], lo, hi, options=query.QueryOptions(merge_gap_blocks=gap))
    assert r.count == brute(dec[name], lo, hi)
    assert r.in_blocks + r.out_blocks + r.maybe_blocks == r.total_blocks


@pytest.mark.parametrize("lo_inc", (False, True))
@pytest.mark.parametrize("hi_inc", (False, True))
def test_inclusive_ends_on_values_that_occur(data, lo_inc, hi_inc):
    _, arrs, dec, _ = data
    v = np.unique(dec["plain"])
    lo, hi = float(v[len(v) // 3]), float(v[len(v) // 3 + 50])
    r = query.query_range(
        arrs["plain"], lo, hi, lo_inclusive=lo_inc, hi_inclusive=hi_inc
    )
    assert r.count == brute(dec["plain"], lo, hi, lo_inc, hi_inc)


def test_single_value_and_empty_ranges(data):
    _, arrs, dec, _ = data
    x = float(np.unique(dec["plain"])[1000])
    point = query.query_range(arrs["plain"], x, x, lo_inclusive=True)
    assert point.count == int((dec["plain"] == x).sum()) > 0
    assert query.query_range(arrs["plain"], x, x).count == 0


@pytest.mark.parametrize("k", range(len(LAYERS)))
@pytest.mark.parametrize("max_req", (1, 10**9))
def test_range_on_each_layer(data, k, max_req):
    a, arrs, _, plain = data
    lo, hi = float(np.quantile(a, 0.25)), float(np.quantile(a, 0.6))
    r = query.query_range(
        arrs["layers"],
        lo,
        hi,
        options=query.QueryOptions(
            layer=k, merge_gap_blocks=4, max_chunk_requests=max_req
        ),
    )
    assert r.count == brute(plain[LAYERS[k]], lo, hi)


def test_gt_is_the_open_upper_range(data):
    a, arrs, _, _ = data
    t = float(np.quantile(a, 0.8))
    gt = query.query_gt(arrs["sub_t"], t)
    rng = query.query_range(arrs["sub_t"], t, math.inf)
    assert (gt.count, gt.maybe_blocks, gt.in_blocks) == (
        rng.count,
        rng.maybe_blocks,
        rng.in_blocks,
    )


def test_below_and_at_or_above_add_up(data):
    a, arrs, dec, _ = data
    t = float(np.unique(dec["plain"])[5000])
    below = query.query_lt(arrs["plain"], t).count
    at_or_above = query.query_gt(arrs["plain"], t, inclusive=True).count
    assert below + at_or_above == a.size
    at_or_below = query.query_lt(arrs["plain"], t, inclusive=True).count
    above = query.query_gt(arrs["plain"], t).count
    assert at_or_below + above == a.size
    assert at_or_below - below == int((dec["plain"] == t).sum()) > 0


@pytest.mark.parametrize(
    "lo, hi", ((2.0, 1.0), (math.nan, 1.0), (0.0, math.nan)), ids=("order", "lo", "hi")
)
def test_rejects_nan_or_reversed_bounds(data, lo, hi):
    _, arrs, _, _ = data
    with pytest.raises(ValueError):
        query.query_range(arrs["plain"], lo, hi)


@pytest.mark.parametrize("x", (1.1, 0.5, -3.25, 1e-40))
def test_range_bounds_use_float32_neighbours(x):
    """Inclusive and exclusive bounds map to the float32 just below the bound.

    Decoded values are float32, so x >= v is the same as x > below and x < v the
    same as x <= below, where below is the largest float32 under v.
    """
    # smallest float32 >= x, and the float32 just below it
    f = np.float32(x)
    if float(f) < x:
        f = np.nextafter(f, np.float32(np.inf))
    below = float(np.nextafter(f, np.float32(-np.inf)))
    a, b = query.range_bounds(x, x, lo_inclusive=True, hi_inclusive=False)
    assert a == below  # x >= value  <=>  value > below
    assert b == below  # x <  value  <=>  value <= below
    assert query.range_bounds(x, x) == (x, x)
    assert query.range_bounds(None, None) == (-math.inf, math.inf)


@pytest.mark.parametrize("lo, hi", ((None, 1.0), (1.0, None), (None, None)))
def test_range_needs_both_bounds(data, lo, hi):
    """One-sided counts go through query_gt and query_lt instead."""
    _, arrs, _, _ = data
    with pytest.raises(TypeError, match="both bounds"):
        query.query_range(arrs["plain"], lo, hi)
    with pytest.raises(TypeError, match="both bounds"):
        zarr.core.sync.sync(query.query_range_async(arrs["plain"], lo, hi))


@pytest.mark.parametrize("fn", ("query_gt", "query_lt"))
def test_one_sided_rejects_nan(data, fn):
    _, arrs, _, _ = data
    with pytest.raises(ValueError, match="NaN"):
        getattr(query, fn)(arrs["plain"], math.nan)
