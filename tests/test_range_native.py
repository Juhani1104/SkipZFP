"""The range C functions in csrc/query.c, called directly through ctypes."""

import ctypes
import math

import numpy as np
import pytest

from skipzfp import _native, codec

from helpers import SHAPE, smooth_field, true_blocks

P = ctypes.c_void_p
SZ = ctypes.c_size_t
DBL = ctypes.c_double
INT = ctypes.c_int
PSZ = ctypes.POINTER(ctypes.c_size_t)

PLAN = [P, SZ, SZ, SZ, SZ, SZ, SZ, SZ, SZ]  # meta .. blocks_per_unit
OUTS = [P, P, SZ, PSZ, PSZ, PSZ]  # maybe_chunks .. out_out
SIGNATURES = {
    "szfp_plan_range_chunks": [*PLAN, DBL, DBL, INT, *OUTS],
    "szfp_plan_gt_chunks": [*PLAN, DBL, INT, *OUTS],
    "szfp_count_range_blocks": [P, SZ, SZ, DBL, INT, DBL, DBL, INT, PSZ],
    "szfp_count_offsets_range": [P, SZ, P, SZ, SZ, DBL, INT, DBL, DBL, PSZ],
    "szfp_set_fast_decode": [INT],
}
# codes from csrc/query.c, which differ from the SZ_* codes in core.h
OK, ERR_NULL, ERR_ARG, ERR_SIZE = 0, 1, 2, 3
LIBZFP, FAST, FAST_SCALAR = 0, 1, 2
RATE = 8.0
NB = int(64 * RATE / 8)
NBLK = (SHAPE[0] // 4) * (SHAPE[1] // 4) * (SHAPE[2] // 4)
META = 12 + 2 * NBLK
INF = math.inf


@pytest.fixture(scope="module")
def lib():
    # a private handle, so these signatures do not leak into the package
    lib = ctypes.CDLL(str(_native.load_library()._name))
    for name, args in SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int if name != "szfp_set_fast_decode" else None
    yield lib
    lib.szfp_set_fast_decode(FAST)


@pytest.fixture(scope="module")
def encoded():
    a = smooth_field()
    buf = codec.native().encode(a, RATE, 4)
    full = true_blocks(codec.native().decode(buf, SHAPE, RATE, 4))
    return a, buf, full


def ptr(a):
    return a.ctypes.data


def _out():
    return ctypes.byref(ctypes.c_size_t())


def brute(v, lo, hi):
    v = v.astype(np.float64)
    return int(((v > lo) & (v <= hi)).sum())


def ranges(full):
    q = lambda p: float(np.quantile(full, p))  # noqa: E731
    return [
        (q(0.3), INF),
        (-INF, q(0.3)),
        (q(0.2), q(0.7)),
        (-INF, INF),
        (q(0.4), q(0.4)),
    ]


def plan(lib, meta, units, subs, lo, hi, gt=False):
    n = int(np.prod(units)) * NBLK
    chunks = np.empty(n, np.uint32)
    blocks = np.empty(n, np.uint32)
    counts = [ctypes.c_size_t() for _ in range(3)]
    bounds = (lo,) if gt else (lo, hi)
    fn = lib.szfp_plan_gt_chunks if gt else lib.szfp_plan_range_chunks
    head = (ptr(meta), META, *units, *subs, NBLK)
    outs = (ptr(chunks), ptr(blocks), n, *(ctypes.byref(c) for c in counts))
    assert fn(*head, *bounds, 1, *outs) == OK
    m, n_in, n_out = (c.value for c in counts)
    return chunks[:m], blocks[:m], n_in, n_out


@pytest.mark.parametrize("mode", (LIBZFP, FAST, FAST_SCALAR))
@pytest.mark.parametrize("threads", (1, 0))
def test_count_range_blocks_matches_numpy(lib, encoded, mode, threads):
    """Every decode path counts the same as numpy on the decoded blocks."""
    _, buf, full = encoded
    lib.szfp_set_fast_decode(mode)
    for lo, hi in ranges(full):
        cnt = ctypes.c_size_t()
        rc = lib.szfp_count_range_blocks(
            ptr(buf), len(full), NB, RATE, 4, lo, hi, threads, ctypes.byref(cnt)
        )
        assert rc == OK and cnt.value == brute(full, lo, hi)


@pytest.mark.parametrize("mode", (LIBZFP, FAST, FAST_SCALAR))
def test_count_offsets_range_matches_numpy(lib, encoded, mode):
    """Every decode path counts the same as numpy on the blocks at the offsets."""
    _, buf, full = encoded
    lib.szfp_set_fast_decode(mode)
    offs = np.arange(0, len(full), 5, dtype=np.uint64) * np.uint64(NB)
    args = (ptr(buf), buf.size, ptr(offs), len(offs), NB, RATE, 4)
    for lo, hi in ranges(full):
        cnt = ctypes.c_size_t()
        rc = lib.szfp_count_offsets_range(*args, lo, hi, ctypes.byref(cnt))
        assert rc == OK and cnt.value == brute(full[::5], lo, hi)


@pytest.mark.parametrize(
    "units, subs", [((2, 1, 1), (1, 1, 1)), ((2, 1, 1), (2, 1, 1))]
)
def test_open_upper_range_plans_like_gt(lib, encoded, units, subs):
    """With hi = inf, the range planner returns exactly what the gt planner does.

    Two units are planned both as two chunks and as one chunk of two units, so
    the grouping of units into chunks is covered as well.
    """
    a, _, full = encoded
    rec = codec.native().meta(a, RATE, 4)
    meta = np.concatenate([rec, codec.native().meta(a[::-1].copy(), RATE, 4)])
    for th in (float(np.quantile(full, p)) for p in (0.01, 0.5, 0.99)):
        new = plan(lib, meta, units, subs, th, INF)
        old = plan(lib, meta, units, subs, th, None, gt=True)
        assert all(np.array_equal(x, y) for x, y in zip(new[:2], old[:2]))
        assert new[2:] == old[2:]


def test_in_and_out_blocks_hold_for_a_closed_range(lib, encoded):
    """IN blocks lie wholly inside lo < x <= hi and OUT blocks wholly outside.

    This must hold for the original and the decoded values alike, since queries
    count IN blocks and skip OUT blocks without decoding them.
    """
    a, _, full = encoded
    meta = codec.native().meta(a, RATE, 4)
    orig = true_blocks(a)
    for lo, hi in ranges(full):
        _, maybe, n_in, n_out = plan(lib, meta, (1, 1, 1), (1, 1, 1), lo, hi)
        rest = np.ones(NBLK, bool)
        rest[maybe] = False
        for v in (orig, full):
            hit = (v > lo) & (v <= hi)
            # the planner only reports how many blocks are IN and OUT, so every
            # non-MAYBE block must be wholly inside or wholly outside, and the
            # two groups must match those counts
            assert hit[rest].all(axis=1).sum() == n_in
            assert (~hit[rest]).all(axis=1).sum() == n_out
        assert n_in + n_out + len(maybe) == NBLK


def test_constant_unit_is_all_in_or_all_out(lib):
    """A unit with a single value is wholly IN or wholly OUT, never MAYBE.

    Its min equals its max, which the planner handles separately because the
    block offsets are scaled by max - min.
    """
    a = np.full(SHAPE, 3.5, np.float32)
    meta = codec.native().meta(a, RATE, 4)
    one = ((1, 1, 1), (1, 1, 1))
    _, maybe, n_in, n_out = plan(lib, meta, *one, 3.0, 4.0)
    assert len(maybe) == 0 and n_in == NBLK and n_out == 0
    _, maybe, n_in, n_out = plan(lib, meta, *one, 4.0, 5.0)
    assert len(maybe) == 0 and n_in == 0 and n_out == NBLK


def error_cases(buf):
    meta = np.zeros(META, np.uint8)
    ids = np.zeros(NBLK, np.uint32)
    offs = np.zeros(1, np.uint64)
    past_end = offs + np.uint64(buf.size)
    KEEP[:] = [meta, ids, offs, past_end]

    def plan_args(units=(1, 1, 1), subs=(1, 1, 1), size=META, nblk=NBLK, cap=NBLK):
        head = (ptr(meta), size, *units, *subs, nblk, 0.0, INF, 1)
        return (*head, ptr(ids), ptr(ids), cap, _out(), _out(), _out())

    def plan_null():
        head = (None, META, 1, 1, 1, 1, 1, 1, NBLK, 0.0, INF, 1)
        return (*head, None, None, 0, _out(), _out(), _out())

    def count_args(nb=NB, rate=RATE, out=True, blocks=True):
        src = ptr(buf) if blocks else None
        return (src, 1, nb, rate, 4, 0.0, INF, 1, _out() if out else None)

    def offs_args(nb=NB, rate=RATE, dim=4, at=offs, out=True, src=True):
        head = (ptr(buf) if src else None, buf.size, ptr(at), 1, nb, rate, dim)
        return (*head, 0.0, INF, _out() if out else None)

    return {
        "plan null": ("szfp_plan_range_chunks", plan_null(), ERR_NULL),
        "plan zero sub": ("szfp_plan_range_chunks", plan_args(subs=(0, 1, 1)), ERR_ARG),
        "plan ragged": (
            "szfp_plan_range_chunks",
            plan_args(units=(3, 1, 1), subs=(2, 1, 1)),
            ERR_ARG,
        ),
        "plan zero units": (
            "szfp_plan_range_chunks",
            plan_args(units=(0, 1, 1)),
            ERR_ARG,
        ),
        "plan zero blocks": ("szfp_plan_range_chunks", plan_args(nblk=0), ERR_ARG),
        "plan wrong meta size": (
            "szfp_plan_range_chunks",
            plan_args(size=META - 2),
            ERR_ARG,
        ),
        "plan small cap": ("szfp_plan_range_chunks", plan_args(cap=1), ERR_ARG),
        "count null out": ("szfp_count_range_blocks", count_args(out=False), ERR_NULL),
        "count null blocks": (
            "szfp_count_range_blocks",
            count_args(blocks=False),
            ERR_NULL,
        ),
        "count bad rate": ("szfp_count_range_blocks", count_args(rate=0.0), ERR_ARG),
        "count wrong size": (
            "szfp_count_range_blocks",
            count_args(nb=NB - 1),
            ERR_SIZE,
        ),
        "offsets null out": (
            "szfp_count_offsets_range",
            offs_args(out=False),
            ERR_NULL,
        ),
        "offsets null buf": (
            "szfp_count_offsets_range",
            offs_args(src=False),
            ERR_NULL,
        ),
        "offsets bad rate": ("szfp_count_offsets_range", offs_args(rate=0.0), ERR_ARG),
        "offsets tiny rate": ("szfp_count_offsets_range", offs_args(rate=0.1), ERR_ARG),
        "offsets huge block_dim": (
            "szfp_count_offsets_range",
            offs_args(dim=1 << 22),
            ERR_ARG,
        ),
        "offsets block_dim 2": (
            "szfp_count_offsets_range",
            offs_args(nb=8, dim=2),
            ERR_ARG,
        ),
        "offsets wrong size": (
            "szfp_count_offsets_range",
            offs_args(nb=NB - 1),
            ERR_SIZE,
        ),
        "offsets past end": (
            "szfp_count_offsets_range",
            offs_args(at=past_end),
            ERR_SIZE,
        ),
    }


KEEP = []  # arrays whose addresses are handed to C must outlive the call
CASES = list(error_cases(np.zeros(NB, np.uint8)))


@pytest.mark.parametrize("case", CASES)
def test_range_functions_reject_bad_input(lib, encoded, case):
    _, buf, _ = encoded
    name, args, want = error_cases(buf)[case]
    assert getattr(lib, name)(*args) == want


@pytest.mark.parametrize(
    "name, args",
    [
        ("szfp_count_range_blocks", (None, 0, NB, RATE, 4, 0.0, INF, 1, _out())),
        ("szfp_count_offsets_range", (None, 0, None, 0, NB, RATE, 4, 0.0, INF, _out())),
    ],
)
def test_range_functions_accept_empty_input(lib, name, args):
    assert getattr(lib, name)(*args) == OK
