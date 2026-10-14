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
# SkipZFP threshold queries: 256 KB to 850 KB run equally fast on GCS, and the
# smaller gap reads fewer bytes; 128 KB and below get slower (more requests)
SKIPZFP_GAP_BYTES = 256_000
CONCURRENCY = 128
EMPTY = np.iinfo(np.uint64).max
VARS = ("t2m", "ws")
# time steps handled at once by code that must not hold a whole variable in memory;
# a multiple of every chunk and shard length along time
SLAB = 64 * 16


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


def slabs(n, step=SLAB):
    """Yield (start, stop) of consecutive time slabs covering n steps."""
    for s in range(0, n, step):
        yield s, min(s + step, n)


def _era5_slab(g, var, s, e):
    if var in g:
        return np.asarray(g[var][s:e])
    u = np.asarray(g["u10"][s:e]).astype(np.float64)
    v = np.asarray(g["v10"][s:e]).astype(np.float64)
    return np.sqrt(u * u + v * v).astype(np.float32)


def load_era5(path, var, nt=None, cache=None):
    """Load one variable, trimmed to whole 64-step objects; ws is derived if needed.

    With cache, the variable is copied slab by slab to <cache>/<var>.npy and returned
    as a read-only memory map, so it never has to fit in memory. An existing copy is
    reused.
    """
    g = zarr.open_group(str(path), mode="r")
    n = g["t2m"].shape[0]
    n = (min(n, nt) if nt else n) // 64 * 64
    if cache is None:
        return _era5_slab(g, var, 0, n)
    out = Path(cache, f"{var}.npy")
    if not out.exists():
        part = out.with_suffix(".part.npy")
        shape = (n, *g["t2m"].shape[1:])
        mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32, shape=shape)
        for s, e in slabs(n):
            mm[s:e] = _era5_slab(g, var, s, e)
        mm.flush()
        del mm
        part.rename(out)
    a = np.load(out, mmap_mode="r")
    if a.shape[0] != n:
        raise ValueError(f"{out} holds {a.shape[0]} steps, not {n}; delete it")
    return a


def write_slabs(z, a):
    """z[:] = a, one time slab at a time."""
    for s, e in slabs(a.shape[0]):
        z[s:e] = a[s:e]


def chunk_stats(a, chunk):
    def one(x):
        g = [s // c for s, c in zip(x.shape, chunk)]
        b = x.reshape(g[0], chunk[0], g[1], chunk[1], g[2], chunk[2])
        return np.stack([b.min(axis=(1, 3, 5)), b.max(axis=(1, 3, 5))], -1)

    parts = [one(np.asarray(a[s:e])) for s, e in slabs(a.shape[0])]
    return np.concatenate(parts).astype(np.float32)


def inner_stats(a, shard, inner):
    def one(x):
        g = [s // c for s, c in zip(x.shape, shard)]
        k = [c // i for c, i in zip(shard, inner)]
        b = x.reshape(g[0], k[0], inner[0], g[1], k[1], inner[1], g[2], k[2], inner[2])
        b = b.transpose(0, 3, 6, 1, 4, 7, 2, 5, 8).reshape(*g, int(np.prod(k)), -1)
        return np.stack([b.min(-1), b.max(-1)], -1)

    parts = [one(np.asarray(a[s:e])) for s, e in slabs(a.shape[0])]
    return np.concatenate(parts).astype(np.float32)


def _sort_keys(x):
    """Map float32 values to uint32 keys that sort in the same order."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return np.where(u >> 31, ~u, u | np.uint32(1 << 31))


def _key_value(k):
    k = np.uint32(k)
    u = k & np.uint32(0x7FFFFFFF) if k >> 31 else ~k
    return np.array([u], np.uint32).view(np.float32)[0]


def quantiles(a, qs):
    """Return np.quantile(a, qs) for a float32 array, reading it one time slab at a time.

    It finds the needed order statistics exactly in three passes, each narrowing every
    one down by a byte (16, then 8, then 8 bits of a sortable key), so a can be a
    memory map larger than memory. The interpolation is np.quantile's own.
    """
    n = a.size
    pos = [q * (n - 1) for q in qs]
    ranks = sorted(
        {int(np.floor(p)) for p in pos}
        | {min(int(np.floor(p)) + 1, n - 1) for p in pos}
    )
    prefix = {r: 0 for r in ranks}  # key bits found so far
    below = {r: 0 for r in ranks}  # values with a smaller key prefix
    for shift, bits in ((16, 16), (8, 8), (0, 8)):
        known = 32 - shift - bits  # bits already fixed in prefix
        groups = sorted({prefix[r] for r in ranks})
        hist = {p: np.zeros(1 << bits, np.int64) for p in groups}
        for s, e in slabs(a.shape[0]):
            x = np.asarray(a[s:e])
            if np.isnan(x).any():
                raise ValueError("quantiles needs data without NaN")
            k = _sort_keys(x).ravel()
            top = k >> np.uint32(shift + bits) if known else None
            for p in groups:
                sel = k if top is None else k[top == p]
                hist[p] += np.bincount(
                    (sel >> np.uint32(shift)) & np.uint32((1 << bits) - 1),
                    minlength=1 << bits,
                )
        for r in ranks:
            c = np.cumsum(hist[prefix[r]])
            b = int(np.searchsorted(c, r - below[r], side="right"))
            below[r] += int(c[b - 1]) if b else 0
            prefix[r] = (prefix[r] << bits) | b
    value = {r: _key_value(prefix[r]) for r in ranks}
    out = []
    for p in pos:
        lo = int(np.floor(p))
        hi = min(lo + 1, n - 1)
        pair = np.array([value[lo], value[hi]], np.float32)
        # an array of q, as np.quantile gets from a list, so it interpolates in float64
        out.append(np.quantile(pair, np.array([p - lo]))[0])
    return out


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
    elif dest.startswith("s3://"):
        # unlike gcloud, aws copies a folder's contents, so name the folder
        cmd = ["aws", "s3", "cp", "--only-show-errors"]
        if Path(path).is_dir():
            cmd += ["--recursive", str(path), f"{dest}/{Path(path).name}"]
        else:
            cmd += [str(path), dest + "/"]
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
    # S3 Express One Zone (directory bucket) names end in --x-s3
    opts = {"s3_express": True} if "--x-s3" in full else {}
    store = CountingStore(ObjectStore(from_url(full, **opts), read_only=True))
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
