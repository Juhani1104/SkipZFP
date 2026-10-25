import ctypes
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import WrapperStore

from skipzfp import codec, query  # noqa: F401

# the storage object: a SkipZFP chunk and a Zarr shard. SKIPZFP_OBJECT, such as
# 64x320x640, builds and reads a suite with another object size instead
SHARD = tuple(int(x) for x in os.environ.get("SKIPZFP_OBJECT", "64x128x256").split("x"))
SUB_CHUNK = (64, 16, 32)
GAP_BYTES = 850_000
# SkipZFP threshold queries: 256 KB to 850 KB run equally fast on GCS, and the
# smaller gap reads fewer bytes; 128 KB and below get slower (more requests)
SKIPZFP_GAP_BYTES = 256_000
CONCURRENCY = 128
EMPTY = np.iinfo(np.uint64).max
VARS = ("t2m", "ws")
# time steps handled at once by code that must not hold a whole variable in memory;
# a multiple of every chunk and shard length along time. 4096 steps of 128 x 256
# float32 is 537 MB, enough chunks to keep every core busy within one slab.
SLAB = 64 * 64
# and at most this many bytes, so a slab of a wider grid, such as the global ERA5's
# 640 x 1280 (13 GB for 4096 steps), still fits in memory with its copies
SLAB_BYTES = 1 << 30


def slab_step(shape, itemsize=4, step=SLAB):
    """step, cut to a multiple of 64 steps of at most SLAB_BYTES for this grid."""
    per_step = int(np.prod(shape[1:])) * itemsize
    return min(step, max(64, SLAB_BYTES // per_step // 64 * 64))


def shape_of(tag):
    return tuple(int(x) for x in tag.split("x"))


def tag_of(shape):
    return "x".join(map(str, shape))


def lossless():
    return BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", typesize=4)


def skipzfp_codec(layout):
    """k8: sub-chunk 64x16x32 in C order; k8t: time-fastest block order.

    A "-b4" suffix, as in k8t-b4, also stores a 4-bit error bound per block. A
    "-z" suffix, as in k8t-z, only compresses the zone map, so the codec is the
    same as without it.
    """
    base, _, bits = layout.removesuffix("-z").partition("-b")
    order = (1, 2, 0) if base.endswith("t") else (0, 1, 2)
    return codec.SkipZFPCodec(
        rate=8.0,
        sub_chunk=SUB_CHUNK,
        block_order=order,
        block_error_bits=int(bits or 0),
    )


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


def _copy(a, s, e, parts=16):
    """np.array(a[s:e]); a memory map is copied in parts on several threads at once.

    Copying a memory map faults its pages in one read at a time, which a network disk
    serves at a few tens of MB/s; reads issued together run many times faster.
    """
    if not isinstance(a, np.memmap) or e - s < parts:
        return np.array(a[s:e])
    out = np.empty((e - s, *a.shape[1:]), a.dtype)
    cuts = np.linspace(s, e, parts + 1).astype(int)

    def one(i):
        out[cuts[i] - s : cuts[i + 1] - s] = a[cuts[i] : cuts[i + 1]]

    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(one, range(parts)))
    return out


def read_slabs(a, step=SLAB, read=None):
    """Yield (start, stop, values) for each time slab of a, in memory.

    The next slab is read on another thread while the caller works on this one, so
    the disk and the CPU stay busy together. read(start, stop) defaults to copying
    a[start:stop], which is what pulls a memory map from disk; with a custom read, a
    can be just the number of steps.
    """
    read = read or (lambda s, e: _copy(a, s, e))
    if hasattr(a, "shape"):
        step = slab_step(a.shape, a.dtype.itemsize, step)
    spans = list(slabs(a.shape[0] if hasattr(a, "shape") else a, step))
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(read, *spans[0]) if spans else None
        for i, (s, e) in enumerate(spans):
            cur = nxt.result()
            if i + 1 < len(spans):
                nxt = ex.submit(read, *spans[i + 1])
            yield s, e, cur


def _era5_slab(g, var, s, e):
    if var in g:
        return np.asarray(g[var][s:e])
    u = np.asarray(g["u10"][s:e]).astype(np.float64)
    v = np.asarray(g["v10"][s:e]).astype(np.float64)
    return np.sqrt(u * u + v * v).astype(np.float32)


def load_era5(path, var, nt=None, cache=None):
    """Load one variable, trimmed to whole 64-step objects; ws is derived if needed.

    Works on any raw group whose variables are (time, lat, lon) arrays, such as the
    CMIP6 precipitation of fetch_cmip6.py. With cache, the variable is copied slab by
    slab to <cache>/<var>.npy and returned as a read-only memory map, so it never has
    to fit in memory. An existing copy is reused.
    """
    g = zarr.open_group(str(path), mode="r")
    ref = g[var] if var in g else g["u10"]
    n = ref.shape[0]
    n = (min(n, nt) if nt else n) // 64 * 64
    if cache is None:
        return _era5_slab(g, var, 0, n)
    out = Path(cache, f"{var}.npy")
    if not out.exists():
        part = out.with_suffix(".part.npy")
        shape = (n, *ref.shape[1:])
        mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32, shape=shape)
        step = slab_step(shape)
        for s, e, x in read_slabs(n, step, read=lambda s, e: _era5_slab(g, var, s, e)):
            mm[s:e] = x
        mm.flush()
        del mm
        part.rename(out)
    a = np.load(out, mmap_mode="r")
    if a.shape[0] != n:
        raise ValueError(f"{out} holds {a.shape[0]} steps, not {n}; delete it")
    return a


# The point suite's k64t layout stores 512-step objects, so every point-suite
# store, and every threshold and check on them, uses the same whole number of
# 512-step objects: one year's 8768 steps become 8704.
POINT_STEP_MULTIPLE = 512


def load_point(raw, var, cache=None):
    """load_era5 trimmed to whole 512-step objects, the range every point store holds."""
    a = load_era5(raw, var, cache=cache)
    return a[: a.shape[0] // POINT_STEP_MULTIPLE * POINT_STEP_MULTIPLE]


def write_slabs(z, a):
    """z[:] = a, one time slab at a time."""
    for s, e, x in read_slabs(a):
        z[s:e] = x


def chunk_stats(a, chunk):
    def one(x):
        g = [s // c for s, c in zip(x.shape, chunk)]
        b = x.reshape(g[0], chunk[0], g[1], chunk[1], g[2], chunk[2])
        return np.stack([b.min(axis=(1, 3, 5)), b.max(axis=(1, 3, 5))], -1)

    parts = [one(x) for _, _, x in read_slabs(a)]
    return np.concatenate(parts).astype(np.float32)


def inner_stats(a, shard, inner):
    def one(x):
        g = [s // c for s, c in zip(x.shape, shard)]
        k = [c // i for c, i in zip(shard, inner)]
        b = x.reshape(g[0], k[0], inner[0], g[1], k[1], inner[1], g[2], k[2], inner[2])
        b = b.transpose(0, 3, 6, 1, 4, 7, 2, 5, 8).reshape(*g, int(np.prod(k)), -1)
        return np.stack([b.min(-1), b.max(-1)], -1)

    parts = [one(x) for _, _, x in read_slabs(a)]
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
        for _, _, x in read_slabs(a):
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
    # every store here is Zarr v3, so only zarr.json is requested, for every method
    arr = zarr.open_array(store=store, path="data", mode="r", zarr_format=3)
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


# What the timing fields mean, saved with every run so a reader can tell the
# critical path from work summed over threads.
FIELDS = {
    "seconds": "wall time of the query, from the first metadata read to the answer",
    "open_seconds": "wall time to open the array (zarr.json), before the query",
    "open_requests": "requests made while opening the array",
    "metadata_seconds": "wall time until the zone map or indexes were read",
    "planning_seconds": "planning time; with the pipeline, summed over objects",
    "payload_seconds": "wall time from the first payload read to the last decode",
    "decode_seconds": "decode time summed over every read, across threads, so "
    "it can exceed the wall time and must not be stacked with the others",
    "requests": "requests made by the query, counted by a wrapper store",
    "bytes": "bytes returned to the query, counted by a wrapper store",
    "peak_rss_kb": "peak resident memory of the measuring process (VmHWM), page "
    "cache not included",
    "cgroup_peak_bytes": "memory.peak of the job's cgroup, page cache included",
    "peak_inflight_bytes": "most payload bytes reserved at once, from a read's start "
    "until its blocks are decoded: an upper bound on payload buffers, not memory "
    "used, since reads still downloading count too; see peak_rss_kb for memory",
}


def source_version():
    """The code version: git commit and uncommitted diff, or SOURCE.json if no git.

    run_cloud ships code without .git, so the shipping step writes SOURCE.json
    next to this file with the same fields.
    """
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(here), "diff", "HEAD"],
            capture_output=True,
            check=True,
        ).stdout
        return dict(
            commit=commit,
            dirty=bool(diff),
            diff_sha256=hashlib.sha256(diff).hexdigest() if diff else None,
        )
    except (OSError, subprocess.CalledProcessError):
        f = here / "SOURCE.json"
        return json.loads(f.read_text()) if f.exists() else dict(commit=None)


def _gce(path):
    """A value from the GCE metadata server, or None off Google Cloud."""
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/" + path,
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        return urllib.request.urlopen(req, timeout=0.5).read().decode().split("/")[-1]
    except OSError:
        return None


def machine():
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in open("/proc/cpuinfo")
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    mem = next(
        int(line.split()[1])
        for line in open("/proc/meminfo")
        if line.startswith("MemTotal")
    )
    return dict(
        host=platform.node(),
        gce_machine_type=_gce("machine-type"),
        gce_zone=_gce("zone"),
        cpu=cpu,
        cpus=os.cpu_count(),
        mem_kb=mem,
        kernel=platform.release(),
    )


def calibration():
    """Seconds for fixed work on this machine, best of three, so a run on a slower
    host or boot shows up as such: encoding and decoding one SkipZFP chunk in
    memory with one thread, and summing 512 MB."""
    from zarr.storage import MemoryStore

    rng = np.random.default_rng(0)
    a = (280 + np.cumsum(rng.standard_normal((64, 128, 256)), axis=0)).astype(
        np.float32
    )
    big = np.ones(2**27, np.float32)
    enc, dec, mem = [], [], []
    with zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1}):
        for _ in range(3):
            z = zarr.create_array(
                MemoryStore(),
                shape=a.shape,
                chunks=a.shape,
                dtype="float32",
                serializer=skipzfp_codec("k8t"),
                compressors=None,
                filters=None,
            )
            t = time.perf_counter()
            z[:] = a
            enc.append(time.perf_counter() - t)
            t = time.perf_counter()
            z[:]
            dec.append(time.perf_counter() - t)
            t = time.perf_counter()
            big.sum()
            mem.append(time.perf_counter() - t)
    return dict(encode_chunk_s=min(enc), decode_chunk_s=min(dec), sum_512mb_s=min(mem))


def packages():
    import importlib.metadata as md

    out = dict(python=platform.python_version())
    for name in ("numpy", "zarr", "obstore", "zfpy", "numcodecs", "skipzfp"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    # a const char* variable in libzfp, which the C library links against
    lib = codec.native().lib
    out["libzfp"] = ctypes.c_char_p.in_dll(lib, "zfp_version_string").value.decode()
    return out


def data_version(raw):
    """Where the raw data came from, its shape, time range and a cheap fingerprint.

    The fingerprint hashes the time axis, the grid and the first and last time
    step of every variable, so two runs on different data cannot share it.
    """
    if raw is None:
        return None
    g = zarr.open_group(str(raw), mode="r")
    h = hashlib.sha256()
    out = dict(path=str(raw), attrs=dict(g.attrs), arrays={})
    for name, a in sorted(g.arrays()):
        out["arrays"][name] = dict(shape=list(a.shape), dtype=str(a.dtype))
        if a.ndim == 1:
            h.update(np.asarray(a[:]).tobytes())
        else:
            h.update(np.asarray(a[0]).tobytes() + np.asarray(a[-1]).tobytes())
    t = np.asarray(g["time"][[0, -1]])
    if g.attrs.get("time_unit") == "hours since 1970-01-01":  # ERA5
        t = t.astype("datetime64[h]")
        out["time_range"] = [str(t[0]), str(t[1])]
    else:  # a model calendar or simulation steps, kept in their own units
        out["time_range"] = [float(t[0]), float(t[1])]
    out["fingerprint_sha256"] = h.hexdigest()
    return out


def start_run(out_path, raw=None, **extra):
    """Record what produced the results in out_path and return this run's id.

    Each run adds an entry to <out_path>.runs.json: code, machine, packages,
    data and any extra settings. Every result row should carry the returned
    run_id, since one results file can gather rows from several runs.
    """
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    side = Path(str(out_path) + ".runs.json")
    runs = json.loads(side.read_text()) if side.exists() else {}
    runs[run_id] = dict(
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source=source_version(),
        machine=machine(),
        calibration=calibration(),
        packages=packages(),
        data=data_version(raw),
        fields=FIELDS,
        **extra,
    )
    side.write_text(json.dumps(runs, indent=1))
    return run_id
