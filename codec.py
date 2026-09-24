from __future__ import annotations

import asyncio
import ctypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import zarr
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON

_ERR = {
    0: "SZ_OK",
    1: "SZ_ERR_NULL",
    2: "SZ_ERR_STREAM",
    3: "SZ_ERR_ZFP",
    4: "SZ_ERR_FIELD",
    5: "SZ_ERR_DECOMPRESS",
    6: "SZ_ERR_DIMS",
    7: "SZ_ERR_ARG",
    8: "SZ_ERR_SIZE",
    9: "SZ_ERR_COMPRESS",
    10: "SZ_ERR_MALLOC",
}


class _Native:
    def __init__(self) -> None:
        path = Path(__file__).with_name("core.so")
        if not path.exists():
            raise FileNotFoundError(f"could not find {path}; compile core.so first")

        self.lib = ctypes.CDLL(str(path))

        self.lib.szfp_layout.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_layout.restype = ctypes.c_int

        self.lib.szfp_encode.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.szfp_encode.restype = ctypes.c_int

        self.lib.szfp_decode.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.szfp_decode.restype = ctypes.c_int

        self.lib.szfp_meta.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
        ]
        self.lib.szfp_meta.restype = ctypes.c_int

    @staticmethod
    def check(code: int, name: str) -> None:
        if code != 0:
            msg = _ERR.get(code, f"unknown error {code}")
            raise RuntimeError(f"{name} failed: {msg}")

    def layout(
        self,
        shape: tuple[int, int, int],
        rate: float,
        block_dim: int,
    ) -> tuple[int, int]:
        data_size = ctypes.c_size_t()
        meta_size = ctypes.c_size_t()

        code = self.lib.szfp_layout(
            shape[0],
            shape[1],
            shape[2],
            rate,
            block_dim,
            ctypes.byref(data_size),
            ctypes.byref(meta_size),
        )
        self.check(code, "szfp_layout")
        return int(data_size.value), int(meta_size.value)

    def encode(
        self,
        arr: np.ndarray,
        rate: float,
        block_dim: int,
    ) -> np.ndarray:
        src = np.ascontiguousarray(arr, dtype=np.float32)
        shape = tuple(int(x) for x in src.shape)

        if len(shape) != 3:
            raise ValueError("SkipZFP currently only supports 3D chunks")

        cap, _ = self.layout(shape, rate, block_dim)
        out = np.empty(cap, dtype=np.uint8)
        n = ctypes.c_size_t()

        code = self.lib.szfp_encode(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            shape[0],
            shape[1],
            shape[2],
            rate,
            block_dim,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            out.size,
            ctypes.byref(n),
        )
        self.check(code, "szfp_encode")

        if int(n.value) != cap:
            raise RuntimeError(
                "szfp_encode returned unexpected size: "
                f"actual={int(n.value)}, expected={cap}"
            )

        return out

    def decode(
        self,
        buf: np.ndarray,
        shape: tuple[int, int, int],
        rate: float,
        block_dim: int,
    ) -> np.ndarray:
        src = np.ascontiguousarray(buf, dtype=np.uint8).reshape(-1)
        out = np.empty(shape, dtype=np.float32)

        code = self.lib.szfp_decode(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            src.size,
            shape[0],
            shape[1],
            shape[2],
            rate,
            block_dim,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.size,
        )
        self.check(code, "szfp_decode")

        return out

    def meta(
        self,
        arr: np.ndarray,
        rate: float,
        block_dim: int,
    ) -> np.ndarray:
        src = np.ascontiguousarray(arr, dtype=np.float32)
        shape = tuple(int(x) for x in src.shape)
        _, meta_size = self.layout(shape, rate, block_dim)
        out = np.empty(meta_size, dtype=np.uint8)

        code = self.lib.szfp_meta(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            shape[0],
            shape[1],
            shape[2],
            rate,
            block_dim,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            out.size,
        )
        self.check(code, "szfp_meta")

        return out


_NATIVE: _Native | None = None


def native() -> _Native:
    global _NATIVE

    if _NATIVE is None:
        _NATIVE = _Native()

    return _NATIVE


@dataclass(frozen=True)
class SkipZFPCodec(ArrayBytesCodec):
    codec_name: ClassVar[str] = "skipzfp"
    is_fixed_size: ClassVar[bool] = True

    rate: float = 8.0
    block_dim: int = 4
    layers: tuple[float, ...] = ()
    block_order: tuple[int, ...] = (0, 1, 2)
    sub_chunk: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be greater than 0")
        if self.block_dim != 4:
            raise ValueError("SkipZFP currently requires block_dim=4")

        layers = tuple(float(r) for r in self.layers) or (float(self.rate),)
        if list(layers) != sorted(set(layers)) or layers[-1] != self.rate:
            raise ValueError("layers must be increasing and end at rate")
        if any(r <= 0 or (8 * r) % 1 for r in layers):
            raise ValueError("each layer rate must be a positive multiple of 0.125")
        object.__setattr__(self, "layers", layers)

        order = tuple(int(a) for a in self.block_order)
        if sorted(order) != [0, 1, 2]:
            raise ValueError("block_order must be a permutation of (0, 1, 2)")
        object.__setattr__(self, "block_order", order)

        sub_chunk = tuple(int(x) for x in self.sub_chunk)
        if sub_chunk and (len(sub_chunk) != 3 or any(x <= 0 or x % self.block_dim for x in sub_chunk)):
            raise ValueError("sub_chunk must be three positive multiples of block_dim")
        object.__setattr__(self, "sub_chunk", sub_chunk)

    def unit(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return self.sub_chunk or tuple(shape)

    @property
    def layer_bytes(self) -> tuple[int, ...]:
        prev = (0.0, *self.layers[:-1])
        return tuple(int(8 * (r - p)) for r, p in zip(self.layers, prev))

    def plain(self, shape: tuple[int, ...]) -> bool:
        return (
            len(self.layers) == 1
            and self.block_order == (0, 1, 2)
            and self.unit(shape) == tuple(shape)
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        if data.get("name") != cls.codec_name:
            raise ValueError(f"wrong codec name: {data.get('name')}")

        cfg = data.get("configuration") or {}
        if not isinstance(cfg, dict):
            raise TypeError("configuration must be a JSON object")

        return cls(
            rate=float(cfg.get("rate", 8.0)),
            block_dim=int(cfg.get("block_dim", 4)),
            layers=tuple(cfg.get("layers") or ()),
            block_order=tuple(cfg.get("block_order") or (0, 1, 2)),
            sub_chunk=tuple(cfg.get("sub_chunk") or ()),
        )

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": self.codec_name,
            "configuration": {
                "rate": self.rate,
                "block_dim": self.block_dim,
                "layers": list(self.layers),
                "block_order": list(self.block_order),
                "sub_chunk": list(self.sub_chunk),
            },
        }

    def validate(self, *, shape: tuple[int, ...], dtype: Any, chunk_grid: Any) -> None:
        del chunk_grid

        dt = np.dtype(dtype.to_native_dtype())
        if dt != np.dtype("float32"):
            raise TypeError("SkipZFP currently only supports float32")

        if len(shape) != 3:
            raise ValueError("SkipZFP currently only supports 3D arrays")

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        self.chunk_shape(array_spec)
        return self

    def chunk_shape(self, spec: ArraySpec) -> tuple[int, int, int]:
        shape = tuple(int(x) for x in spec.shape)

        if len(shape) != 3:
            raise ValueError(f"SkipZFP requires 3D chunks, received {shape}")

        if any(x <= 0 or x % self.block_dim != 0 for x in shape):
            raise ValueError(
                f"each chunk dimension must be divisible by block_dim={self.block_dim}"
            )

        if any(c % u for c, u in zip(shape, self.unit(shape))):
            raise ValueError(f"chunk {shape} is not divisible by sub_chunk {self.sub_chunk}")

        dt = np.dtype(spec.dtype.to_native_dtype())
        if dt != np.dtype("float32"):
            raise TypeError("SkipZFP currently only supports float32")

        return shape

    def compute_encoded_size(
        self,
        input_byte_length: int,
        chunk_spec: ArraySpec,
    ) -> int:
        del input_byte_length

        shape = self.chunk_shape(chunk_spec)
        return native().layout(shape, self.rate, self.block_dim)[0]

    async def _encode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        shape = self.chunk_shape(chunk_spec)
        arr = np.asarray(
            chunk_data.as_ndarray_like(),
            dtype=np.float32,
        ).reshape(shape)

        out = await asyncio.to_thread(self.pack, arr)
        return chunk_spec.prototype.buffer.from_array_like(out)

    async def _decode_single(
        self,
        chunk_data: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        shape = self.chunk_shape(chunk_spec)
        buf = np.asarray(
            chunk_data.as_array_like(),
            dtype=np.uint8,
        ).reshape(-1)

        out = await asyncio.to_thread(self.unpack, buf, shape)
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(out)

    def pack(self, arr: np.ndarray) -> np.ndarray:
        shape = tuple(arr.shape)
        if self.plain(shape):
            return native().encode(arr, self.rate, self.block_dim)

        unit = self.unit(shape)
        rank = block_rank(unit, self.block_order, self.block_dim)
        rows = []
        for sl in unit_slices(shape, unit):
            r = native().encode(arr[sl], self.rate, self.block_dim).reshape(len(rank), -1)
            ranked = np.empty_like(r)
            ranked[rank] = r
            rows.append(ranked)
        rows = np.concatenate(rows)
        cuts = np.cumsum((0, *self.layer_bytes))
        return np.concatenate([rows[:, a:b].reshape(-1) for a, b in zip(cuts[:-1], cuts[1:])])

    def unpack(self, buf: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        if self.plain(shape):
            return native().decode(buf, shape, self.rate, self.block_dim)

        unit = self.unit(shape)
        rank = block_rank(unit, self.block_order, self.block_dim)
        n = int(np.prod([s // self.block_dim for s in shape]))
        cuts = np.cumsum((0, *(n * b for b in self.layer_bytes)))
        rows = np.concatenate([buf[a:b].reshape(n, -1) for a, b in zip(cuts[:-1], cuts[1:])], axis=1)
        out = np.empty(shape, dtype=np.float32)
        for i, sl in enumerate(unit_slices(shape, unit)):
            ranked = rows[i * len(rank) : (i + 1) * len(rank)]
            out[sl] = native().decode(np.ascontiguousarray(ranked[rank]).reshape(-1), unit, self.rate, self.block_dim)
        return out


def unit_slices(shape: tuple[int, ...], unit: tuple[int, ...]) -> list[tuple[slice, ...]]:
    grid = tuple(s // u for s, u in zip(shape, unit))
    return [tuple(slice(i * u, (i + 1) * u) for i, u in zip(idx, unit)) for idx in np.ndindex(*grid)]


def block_rank(shape: tuple[int, ...], order: tuple[int, ...], block_dim: int = 4) -> np.ndarray:
    blocks = tuple(s // block_dim for s in shape)
    coords = np.indices(blocks).reshape(len(blocks), -1)
    return np.ravel_multi_index(tuple(coords[a] for a in order), tuple(blocks[a] for a in order))


def meta_size(codec: SkipZFPCodec, chunk: tuple[int, ...]) -> int:
    n = int(np.prod([s // codec.block_dim for s in chunk]))
    return 8 + 4 * len(codec.layers) + 2 * n


META_ATTR = "skipzfp_meta"


def find_codec(arr: zarr.Array) -> SkipZFPCodec:
    for c in arr.metadata.codecs:
        if isinstance(c, SkipZFPCodec):
            return c

    raise ValueError("array does not use the skipzfp codec")


def write_meta(
    arr: zarr.Array,
    data: np.ndarray,
    *,
    path: str | None = None,
    t_chunk: int = 16,
    threads: int = 32,
) -> zarr.Array:
    codec = find_codec(arr)
    chunk = tuple(int(x) for x in arr.metadata.chunk_grid.chunk_shape)
    grid = tuple(s // c for s, c in zip(arr.shape, chunk))

    if data.shape != arr.shape or any(s % c for s, c in zip(arr.shape, chunk)):
        raise ValueError("data must match the array shape and divide into whole chunks")

    if not arr.path:
        raise ValueError("the array must live inside a group so metadata can sit next to it")

    unit = codec.unit(chunk)
    grid = tuple(s // u for s, u in zip(arr.shape, unit))
    size = meta_size(codec, unit)
    rank = block_rank(unit, codec.block_order, codec.block_dim)
    path = path or f"{arr.path}_meta"

    meta = zarr.create_array(
        store=arr.store_path.store,
        name=path,
        shape=(*grid, size),
        chunks=(min(t_chunk, grid[0]), *grid[1:], size),
        dtype="uint8",
        compressors=None,
        filters=None,
        overwrite=True,
    )

    def one(idx: tuple[int, ...]) -> np.ndarray:
        sl = tuple(slice(i * u, (i + 1) * u) for i, u in zip(idx, unit))
        per = [native().meta(data[sl], r, codec.block_dim) for r in codec.layers]
        offs = per[-1][12:].reshape(-1, 2)
        ranked = np.empty_like(offs)
        ranked[rank] = offs
        return np.concatenate([per[-1][:8], *(m[8:12] for m in per), ranked.reshape(-1)])

    idxs = list(np.ndindex(*grid))
    with ThreadPoolExecutor(threads) as ex:
        rows = list(ex.map(one, idxs))

    meta[...] = np.stack(rows).reshape(*grid, size)
    arr.update_attributes({META_ATTR: path})
    return meta
