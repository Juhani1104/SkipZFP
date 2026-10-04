"""SkipZFP: ZFP compression for Zarr with a zone map that lets queries skip blocks."""

from .codec import SkipZFPCodec, write_meta
from .query import QueryResult, open_skipzfp, query_gt, query_range

__all__ = [
    "SkipZFPCodec",
    "QueryResult",
    "open_skipzfp",
    "query_gt",
    "query_range",
    "write_meta",
]
