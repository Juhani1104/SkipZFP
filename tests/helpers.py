"""Shared data and helpers for the tests."""

import ctypes

import numpy as np
import zarr
from zarr.registry import register_codec

from skipzfp import codec, query

register_codec("skipzfp", codec.SkipZFPCodec)

SHAPE = (64, 16, 32)  # (time, lat, lon) chunk used in the paper
RATES = (2.0, 4.0, 8.0, 12.0)
LAYERS = (2.0, 4.0, 8.0)
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def smooth_field(shape=SHAPE, seed=0):
    """Return a smooth float32 field with a little noise, like temperature data."""
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
    """Encode a at rate and return the payload with its zone map record unpacked.

    The record comes back as the min, max, error bound and per-block offsets.
    """
    buf = codec.native().encode(a, rate, 4)
    meta = codec.native().meta(a, rate, 4)
    cmin, cmax, eps = np.frombuffer(meta[:12].tobytes(), np.float32)
    offs = meta[12:].reshape(-1, 2).astype(np.float64)
    return buf, float(cmin), float(cmax), float(eps), offs


def decode_one_block(payload, block_id, rate):
    """Decode one block of a payload through szfp_decode_block."""
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


def classify(meta, lt, th):
    """Per-block states from the native planner: 1=IN, 0=OUT, 2=MAYBE."""
    chunk_ids, block_ids, _, _ = query.native().plan_chunks(
        meta, lt.meta_size, (1, 1, 1), (1, 1, 1), lt.blocks_per_chunk, th, 1
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


def make_array(path, a, **kw):
    """Write a to a new SkipZFP array at path and build its zone map."""
    g = zarr.open_group(str(path), mode="w")
    z = g.create_array(
        "data",
        shape=a.shape,
        chunks=SHAPE,
        dtype="float32",
        serializer=codec.SkipZFPCodec(**kw),
        compressors=None,
        filters=None,
    )
    z[:] = a
    codec.write_meta(z, a)
    return z


def block_eps(
    meta, meta_size, unit_grid, subs, grid_shape, blocks_per_unit, chunk_ids, block_ids
):
    """Return the error bound of each (chunk, block), as aggregates once computed it.

    The planner now returns these bounds itself (plan_read with eps=True); this
    is the earlier Python version, kept to check the planner against.

    meta holds single-layer records, as plan_meta returns them, so the unit's
    error bound is the float at bytes 8 to 12 of each record. When the
    records end with 4-bit codes, a block's bound is eps * q / 15, computed as
    the C planner does. Blocks are numbered unit by unit inside a chunk, as
    in the C planner.
    """
    rec = np.ascontiguousarray(meta, dtype=np.uint8).reshape(-1, meta_size)
    eps = np.ascontiguousarray(rec[:, 8:12]).view(np.float32).reshape(-1)
    n_codes = meta_size - 12 - 2 * blocks_per_unit
    if n_codes not in (0, (blocks_per_unit + 1) // 2):
        raise ValueError("zone map records have an unexpected size")

    _, gy, gz = (int(g) for g in grid_shape)
    sx, sy, sz = (int(s) for s in subs)
    _, uy, uz = (int(u) for u in unit_grid)
    c = chunk_ids.astype(np.int64)
    sub = block_ids.astype(np.int64) // blocks_per_unit
    ix = (c // (gy * gz)) * sx + sub // (sy * sz)
    iy = ((c // gz) % gy) * sy + (sub // sz) % sy
    iz = (c % gz) * sz + sub % sz
    uid = (ix * uy + iy) * uz + iz
    if not n_codes:
        return eps[uid]
    pos = block_ids.astype(np.int64) % blocks_per_unit
    codes = rec[uid, 12 + 2 * blocks_per_unit + pos // 2]
    q = (codes >> (4 * (pos % 2))) & 15
    bound = eps[uid].astype(np.float64) * q.astype(np.float64) / 15.0
    # the C aggregate takes a float32 bound, so round it up to stay safe
    out = bound.astype(np.float32)
    low = out.astype(np.float64) < bound
    out[low] = np.nextafter(out[low], np.float32(np.inf))
    return out
