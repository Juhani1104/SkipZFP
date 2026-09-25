import os
import subprocess
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import WrapperStore

from skipzfp import codec, query  # noqa: F401

SHARD = (64, 128, 256)
SUB_CHUNK = (64, 16, 32)
GAP_BYTES = 850_000
CONCURRENCY = 128
EMPTY = np.iinfo(np.uint64).max
VARS = ("t2m", "ws")


def shape_of(tag):
    return tuple(int(x) for x in tag.split("x"))


def tag_of(shape):
    return "x".join(map(str, shape))


def lossless():
    return BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", typesize=4)


def skipzfp_codec(layout):
    """k8: sub-chunk 64x16x32 in C order; k8t: time-fastest block order."""
    order = (1, 2, 0) if layout.endswith("t") else (0, 1, 2)
    return codec.SkipZFPCodec(rate=8.0, sub_chunk=SUB_CHUNK, block_order=order)


def format_kwargs(fmt, tol=None, chunk=SUB_CHUNK):
    if fmt == "zstd":
        return dict(chunks=chunk, compressors=lossless())
    if fmt == "zfpacc":
        import zfpy
        from zarr.codecs.numcodecs import ZFPY

        return dict(
            chunks=chunk,
            serializer=ZFPY(mode=zfpy.mode_fixed_accuracy, tolerance=tol),
            compressors=None,
            filters=None,
        )
    raise ValueError(fmt)


def load_era5(path, var, nt=None):
    """Load one variable, trimmed to whole 64-step objects; ws is derived if needed."""
    g = zarr.open_group(str(path), mode="r")
    n = g["t2m"].shape[0]
    n = (min(n, nt) if nt else n) // 64 * 64
    if var in g:
        return np.asarray(g[var][:n])
    u = np.asarray(g["u10"][:n]).astype(np.float64)
    v = np.asarray(g["v10"][:n]).astype(np.float64)
    return np.sqrt(u * u + v * v).astype(np.float32)


def chunk_stats(a, chunk):
    g = [s // c for s, c in zip(a.shape, chunk)]
    b = a.reshape(g[0], chunk[0], g[1], chunk[1], g[2], chunk[2])
    return np.stack([b.min(axis=(1, 3, 5)), b.max(axis=(1, 3, 5))], -1).astype(
        np.float32
    )


def inner_stats(a, shard, inner):
    g = [s // c for s, c in zip(a.shape, shard)]
    k = [c // i for c, i in zip(shard, inner)]
    b = a.reshape(g[0], k[0], inner[0], g[1], k[1], inner[1], g[2], k[2], inner[2])
    b = b.transpose(0, 3, 6, 1, 4, 7, 2, 5, 8).reshape(*g, int(np.prod(k)), -1)
    return np.stack([b.min(-1), b.max(-1)], -1).astype(np.float32)


def shard_index(path, grid, n_inner):
    out = np.full((*grid, n_inner, 2), EMPTY, dtype=np.uint64)
    tail = n_inner * 16 + 4
    for idx in np.ndindex(*grid):
        f = Path(path, "c", *map(str, idx))
        if f.exists():
            b = f.read_bytes()
            out[idx] = np.frombuffer(b[-tail:-4], "<u8").reshape(n_inner, 2)
    return out


def put(g, name, data):
    z = g.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        compressors=None,
        filters=None,
        chunks=(min(16, data.shape[0]), *data.shape[1:]),
    )
    z[...] = data


def upload(path, dest):
    if dest.startswith("gs://"):
        cmd = ["gcloud", "storage", "cp", "-r", "-q", str(path), dest + "/"]
        subprocess.run(cmd, check=True)
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-r", str(path), dest + "/"], check=True)


class CountingStore(WrapperStore):
    """Counts requests (one per get) and bytes actually returned."""

    def __init__(self, store):
        super().__init__(store)
        self.requests = 0
        self.bytes = 0

    async def get(self, key, prototype, byte_range=None):
        buf = await self._store.get(key, prototype, byte_range)
        self.requests += 1
        if buf is not None:
            self.bytes += len(buf)
        return buf

    async def get_partial_values(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        out = await self._store.get_partial_values(prototype, key_ranges)
        self.requests += len(key_ranges)
        self.bytes += sum(len(b) for b in out if b is not None)
        return out


def open_counted(url):
    from obstore.store import from_url
    from zarr.storage import ObjectStore

    full = url if "://" in url else "file://" + os.path.abspath(url)
    store = CountingStore(ObjectStore(from_url(full), read_only=True))
    arr = zarr.open_array(store=store, path="data", mode="r")
    return arr, store


def store_url(root, var, method, size):
    if method in ("skipzfp", "fullscan_skipzfp"):
        return f"{root}/stores_sub/{var}/{size}.zarr"
    if method in ("zstd_zm", "zfpacc_zm", "zstd", "zfpacc"):
        fmt = method.removesuffix("_zm")
        return f"{root}/stores_chunks/{var}/{size}/{fmt}.zarr"
    if method.startswith("fullscan_"):
        return (
            f"{root}/stores_chunks/{var}/{size}/{method.removeprefix('fullscan_')}.zarr"
        )
    return f"{root}/stores_zshard/{var}/{size}.zarr"
