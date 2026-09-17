# SkipZFP

SkipZFP is a prototype layout for metadata-guided block-level query processing over compressed cloud-native scientific arrays.

It combines fixed-rate ZFP compression, implicit block-level addressing, compact per-block metadata, and byte-range reads. The goal is to reduce unnecessary payload transfer and decompression for sparse and selective scientific array queries.

## Overview

Scientific array datasets are often stored in chunked cloud-native formats such as Zarr. However, when each chunk is compressed as a single byte stream, the compressed chunk becomes the minimum unit of retrieval and decompression.

SkipZFP reduces this bottleneck by reorganizing each chunk as:

[ fixed-rate ZFP payload ][ metadata trailer ]

The fixed-rate ZFP payload allows compressed block offsets to be computed directly from block indices. The metadata trailer stores lightweight summaries that allow the query engine to skip irrelevant blocks before reading payload bytes.

## Main Ideas

### Fixed-rate block addressing

ZFP stores multidimensional floating-point data in small fixed-size blocks. In the current 3D setting, each ZFP block contains:

4 x 4 x 4 = 64 values

At a fixed bit rate, each compressed block has a predictable byte size:

block_size = values_per_block * rate / 8

This makes it possible to locate selected compressed blocks directly without storing an explicit per-block index.

### Metadata-guided pruning

Each encoded chunk stores:

- chunk minimum
- chunk maximum
- chunk reconstruction-error margin
- per-block quantized minimum and maximum offsets

For a threshold predicate such as:

x > T

SkipZFP classifies each block as one of three states:

IN      the block satisfies the predicate under the metadata rule
OUT     the block cannot satisfy the predicate under the metadata rule
MAYBE   the block must be read and decoded

OUT blocks are skipped before payload reads. IN blocks can contribute directly to count queries. Only MAYBE blocks are retrieved and decoded for final predicate evaluation.

### Range coalescing

SkipZFP reasons at block granularity, but issuing one request per block can be inefficient on object storage systems.

To reduce request overhead, neighboring selected block ranges are merged into larger byte-range requests. This keeps query planning fine-grained while making physical I/O more practical.

## Prototype Scope

The current prototype focuses on:

- 3D float32 arrays
- fixed-rate ZFP compression
- Zarr-compatible codec interface
- metadata trailer per encoded chunk
- threshold predicate queries
- byte-range reads
- range coalescing

The main target layout is time-latitude-longitude scientific data, such as ERA5 temperature and wind-speed fields.

## Core Files

codec.py        Python Zarr codec wrapper
codec.c         encoded size and full chunk decode
core.c          ZFP chunk packing and metadata generation
query.py        Python query pipeline
query.c         native predicate planning and block decoding
core.h          native codec/core API declarations
query.h         native query API declarations
util.c          C helper functions
util.h          C helper declarations
pyproject.toml  Python package and Zarr codec registration

## Build

Compile the native library:

gcc -O3 -fPIC -shared -fopenmp codec.c core.c query.c util.c -lzfp -lm -o core.so

Install the Python package in editable mode:

python -m pip install -e .

## Basic Usage

import query

result = query.query_gt(
    "path/to/skipzfp.zarr",
    threshold=290.7,
    threads=0,
    request_concurrency=32,
    merge_gap_blocks=0,
)

print(result.count)
print(result.bytes_read)
print(result.range_requests)
print(result.total_seconds)

## Query Result Fields

count                   threshold count result
in_blocks               number of blocks classified as IN
out_blocks              number of blocks classified as OUT
maybe_blocks            number of blocks that required decoding
total_blocks            total number of blocks
metadata_bytes_read     bytes read for metadata
payload_bytes_read      bytes read for compressed payload
metadata_requests       number of metadata range requests
payload_requests        number of payload range requests
useful_payload_bytes    requested MAYBE block payload bytes
payload_overread_bytes  extra bytes read due to range coalescing
metadata_read_seconds   metadata read time
planning_seconds        metadata planning time
payload_read_seconds    payload read time
decode_seconds          MAYBE block decode/count time
total_seconds           total query time

## Evaluation Summary

The paper evaluates SkipZFP on ERA5 datasets and synthetic fields.

At 8 bits/value, SkipZFP reduces transferred bytes for 10,000 full-year grid-point time-series queries by 4.02x relative to Zarr+zstd and by 16.0x relative to a 1 KB lossless inner-chunk baseline.

For selective threshold predicates on ERA5 temperature data, metadata-guided pruning accelerates end-to-end query execution by up to 8.32x.

In the main 3D configuration, metadata overhead is 3.16% at 8 bits/value.

## Limitations

The current prototype is intentionally narrow. It focuses on 3D float32 scientific arrays and threshold predicate queries.

Point-query experiments are reported as data-transfer analysis rather than end-to-end latency benchmarks, because issuing one object-store request per touched block would be dominated by request overhead unless ranges are coalesced.

Future work includes alternative metadata groupings for lower-dimensional arrays, four-dimensional layouts, and further optimization of range coalescing policies.