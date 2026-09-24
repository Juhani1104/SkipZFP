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


def make_array(path, a, **kw):
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
