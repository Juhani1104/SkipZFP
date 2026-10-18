"""SkipZFP: ZFP compression for Zarr with a zone map that lets queries skip blocks."""

from .aggregate import AggregateResult, aggregate_gt, aggregate_lt, aggregate_range
from .codec import SkipZFPCodec, check_meta, write_meta
from .query import (
    QueryOptions,
    QueryResult,
    open_skipzfp,
    query_gt,
    query_lt,
    query_range,
)

__all__ = [
    "AggregateResult",
    "aggregate_gt",
    "aggregate_lt",
    "aggregate_range",
    "SkipZFPCodec",
    "check_meta",
    "QueryOptions",
    "QueryResult",
    "open_skipzfp",
    "query_gt",
    "query_lt",
    "query_range",
    "write_meta",
]
