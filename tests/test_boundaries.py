"""Counts stay exact, and IN and OUT stay right, on awkward data and thresholds.

For every case, the counts of query_gt, query_lt and query_range must equal a
brute-force count on the decoded array, and every block the planner calls IN
(OUT) must hold only matching (no matching) values, both original and decoded.
"""

import math

import numpy as np
import pytest

from skipzfp import QueryOptions, query

from helpers import SHAPE, make_array, true_blocks

F32 = np.float32


def neighbours(x):
    """x and the float32 values just below and above it."""
    f = F32(x)
    return [
        float(np.nextafter(f, F32(-np.inf))),
        float(f),
        float(np.nextafter(f, F32(np.inf))),
    ]


def fields():
    rng = np.random.default_rng(0)
    i, j, k = np.indices(SHAPE)
    smooth = (np.sin(k / 5) + np.cos(j / 3) + 0.1 * rng.standard_normal(SHAPE)).astype(
        F32
    )
    mixed = smooth.copy()
    mixed[:16, :8, :16] = F32(0.5)  # some constant blocks inside varying data
    mixed[32:48] = F32(-2.0)
    return {
        "constant": np.full(SHAPE, 3.5, F32),
        "mixed_constant": mixed,
        "both_signs": (smooth * F32(100)).astype(F32),
        "tiny": (smooth * F32(1e-30)).astype(F32),
        "huge": (smooth * F32(1e36)).astype(F32),
        "narrow": (
            F32(1000) + np.arange(np.prod(SHAPE)).reshape(SHAPE) % 7 * F32(1e-4)
        ).astype(F32),
    }


def thresholds(a, rec, meta, lt):
    """Thresholds at every edge the planner compares against, and their neighbours."""
    cmin, cmax, eps = np.frombuffer(meta[:12].tobytes(), F32).astype(np.float64)
    offs = meta[12 : 12 + 2 * lt.blocks_per_unit].reshape(-1, 2).astype(np.float64)
    span = cmax - cmin
    bmin = cmin + offs[:, 0] / 255 * span
    bmax = cmin + offs[:, 1] / 255 * span
    picks = [cmin, cmax, cmin - eps, cmax + eps, 0.0]
    picks += [bmin[3], bmax[3], bmin[3] - eps, bmax[-1] + eps]
    picks += [float(np.median(a)), float(rec.ravel()[17]), float(a.ravel()[123])]
    out = []
    for p in picks:
        if math.isfinite(p) and abs(p) < 3.4e38:
            out += neighbours(p)
    return sorted(set(out))


def check_states(z, a, rec, lo, hi):
    lt = query.get_layout(z)
    meta, _, _ = query.sync(query.read_meta(z, lt))
    chunks, blocks, states, _, _ = query.native().plan_read(
        meta,
        lt.meta_size,
        lt.unit_grid,
        lt.unit_grid_per_chunk,
        lt.blocks_per_unit,
        lo,
        hi,
        1,
    )
    listed = {(int(c), int(b)): int(s) for c, b, s in zip(chunks, blocks, states)}
    for c in range(int(np.prod(lt.grid_shape))):
        coord = np.unravel_index(c, lt.grid_shape)
        sl = tuple(slice(i * s, (i + 1) * s) for i, s in zip(coord, SHAPE))
        for which in (true_blocks(a[sl]), true_blocks(rec[sl])):
            inside = (which > lo) & (which <= hi)
            for b in range(lt.blocks_per_chunk):
                state = listed.get((c, b), 0)
                if state == 1:
                    assert inside[b].all(), ("IN block has a value outside", c, b)
                elif state == 0:
                    assert not inside[b].any(), ("OUT block has a value inside", c, b)


@pytest.mark.parametrize("name", list(fields()))
@pytest.mark.parametrize("rate", (2.0, 8.0, 12.0))
def test_counts_and_states_at_every_edge(tmp_path, name, rate):
    a = np.concatenate([fields()[name]] * 2)  # two chunks
    z = make_array(tmp_path, a, rate=rate)
    rec = z[:]
    lt = query.get_layout(z)
    meta, _, _ = query.sync(query.read_meta(z, lt))
    one = QueryOptions(threads=1)
    for th in thresholds(a, rec, meta[: lt.meta_size], lt):
        assert query.query_gt(z, th).count == int((rec > th).sum()), th
        r = query.query_gt(z, th, inclusive=True, options=one)
        assert r.count == int(
            (rec >= F32(th)).sum() if F32(th) == th else (rec > th).sum()
        )
        assert query.query_lt(z, th).count == int(
            (rec < F32(th)).sum() if F32(th) == th else (rec <= th).sum()
        ), th
        check_states(z, a, rec, th, math.inf)
        check_states(z, a, rec, -math.inf, th)


@pytest.mark.parametrize("name", list(fields()))
def test_ranges_with_equal_and_crossing_bounds(tmp_path, name):
    a = fields()[name]
    z = make_array(tmp_path, a, rate=8.0)
    rec = z[:]
    v = float(rec.ravel()[200])
    w = float(np.quantile(rec, 0.7))
    lo, hi = min(v, w), max(v, w)
    for lo_inc in (False, True):
        for hi_inc in (False, True):
            r = query.query_range(z, lo, hi, lo_inclusive=lo_inc, hi_inclusive=hi_inc)
            m = (rec >= lo if lo_inc else rec > lo) & (
                rec <= hi if hi_inc else rec < hi
            )
            assert r.count == int(m.sum()), (lo_inc, hi_inc)
    # a single value: lo <= x <= lo matches exactly the values equal to it
    r = query.query_range(z, v, v, lo_inclusive=True, hi_inclusive=True)
    assert r.count == int((rec == F32(v)).sum())
    assert query.query_range(z, v, v).count == 0  # v < x <= v is empty
    a_lo, a_hi = query.range_bounds(lo, hi)
    check_states(z, a, rec, a_lo, a_hi)


def test_planner_never_rounds_a_bound_past_the_threshold():
    """A dequantized bound computed in double never crosses an exact threshold.

    Records with eps = 0 leave nothing but the planner's slack between a block
    bound and the threshold. With the threshold at the double nearest the exact
    bound cmin + off / 255 * span, computed with fractions, a block may only be
    IN when the exact bound really exceeds the threshold, and OUT when the exact
    upper bound really stays at or below it.
    """
    from fractions import Fraction

    rng = np.random.default_rng(3)
    wrong = []
    for _ in range(4000):
        cmin, cmax = sorted(
            F32(x) for x in rng.uniform(-1e3, 1e3, 2) * 10.0 ** rng.integers(-3, 4)
        )
        if cmin == cmax:
            continue
        lo_off, hi_off = sorted(int(x) for x in rng.integers(0, 256, 2))
        rec = np.concatenate(
            [
                np.array([cmin, cmax, 0.0], F32).view(np.uint8),
                np.array([lo_off, hi_off], np.uint8),
            ]
        )
        span = Fraction(float(cmax)) - Fraction(float(cmin))
        bmin = Fraction(float(cmin)) + Fraction(lo_off, 255) * span
        bmax = Fraction(float(cmin)) + Fraction(hi_off, 255) * span
        for th in (float(bmin), float(bmax)):
            _, _, st, n_in, n_out = query.native().plan_read(
                rec, rec.size, (1, 1, 1), (1, 1, 1), 1, th, math.inf, 1
            )
            if n_in and not bmin > Fraction(th):
                wrong.append(("IN", float(cmin), float(cmax), lo_off, th))
            if n_out and not bmax <= Fraction(th):
                wrong.append(("OUT", float(cmin), float(cmax), hi_off, th))
    assert not wrong, wrong[:5]
