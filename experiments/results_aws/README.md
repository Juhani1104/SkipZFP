# AWS results

A rerun of the experiments on AWS, reported in the main README under "Also on AWS".
The results in `../results/` (Google Cloud) are unchanged.

Setup: an m5n.4xlarge VM (16 vCPU, Intel Cascade Lake, up to 25 Gbps) in us-east-1
(zone `use1-az4`), Ubuntu 24.04, reading from

- S3: a general purpose bucket in us-east-1, and
- S3 Express One Zone: a directory bucket in `use1-az4`, the VM's own zone.

The stores were built once with `prepare_stores.py` and copied between buckets, so
every run reads the same bytes. Queries use 128 concurrent requests and a 256 KB merge
gap, as in the GCP run.

| Path | Query code | Store | How it was run |
|---|---|---|---|
| `main/` | as in the GCP run | S3 | `run_cloud.sh exp1 exp2 sensitivity local` |
| `mdr/` | + zone map read by key | S3 | `exp2_threshold_query.py run` |
| `pipe/` | + pipelined zone map reads | S3 | `exp2_threshold_query.py run` |
| `express_pipe/` | + pipelined zone map reads | S3 Express | `exp2_threshold_query.py run` |
| `versions_s3.jsonl` | all three | S3 | `compare_versions.py`, SkipZFP k8t only |
| `versions_express.jsonl` | all three | S3 Express | `compare_versions.py`, SkipZFP k8t only |
| `merge_gap/gap_<bytes>.jsonl` | pipelined | S3 | `compare_versions.py --gap <bytes>` |
| `shard_size.jsonl` | pipelined | S3 | `prep_shards.py`, then `compare_versions.py --size T64,...` |

The `versions_*`, `merge_gap` and `shard_size` files interleave the code versions (or
sizes) in a shuffled order within each repetition, so changes in the network over time
hit each one alike. In `versions_s3.jsonl` and `merge_gap/` the rows have no `var`
field: the first half of each file is t2m and the second half is ws.

`exp3`–`exp5` in `main/` match `../results/` exactly.

## Attribution and the numbers in the main README

`attribution/` holds the runs behind the "Also on AWS" table in the main README. Each
repetition runs every version of the query and the best baseline for that selectivity
(`cases.txt`) in a shuffled order, each query in a fresh process, as
`exp2_threshold_query.py one` does.

| File | Versions | Store | Script |
|---|---|---|---|
| `s3.jsonl` | every step below, plus the baseline reading with and without a copy | S3 | `run_attribution.sh`, 10 repetitions |
| `express.jsonl` | `v0_before`, `v4_zerocopy`, baseline without a copy | S3 Express | `run_attribution_express.sh`, 8 repetitions |
| `pipeline_s3.jsonl` | `v4_zerocopy` with and without pipelining, t2m only | S3 | `run_pipeline.sh`, 8 repetitions |

The versions are cumulative commits of `skipzfp/`, each with the same compiled library
(the C code did not change):

| Version | Commit | Adds |
|---|---|---|
| `v0_before` | f4cf06a | the code of the GCP run |
| `v1_direct` | 158d471 | read the zone map objects by key |
| `v2_copyonce` | 5c8a506 | copy each zone map object once |
| `v3_pipeline` | 3ef7636 | plan and read each zone map object as it arrives |
| `v4_zerocopy` | b0a4a38 | decode each read in place instead of copying it |

`v1_direct` alone is slower than `v0_before`, because it copies each zone map object
twice; `v2_copyonce` removes those copies, so the two are best read together.

The scripts refer to paths on the VM: `~/attr/tree_<commit>` is a copy of the repository
with `skipzfp/` taken from that commit, and `~/exp-new` is `experiments/` with the
zero-copy `baselines.py`.

## Caveats on the earlier files

- `main/`, `mdr/`, `pipe/` and `express_pipe/` were each run on their own, hours apart,
  and their baselines copied every read into a Python `bytes` object, as the GCP run
  did. Compare versions with `attribution/` instead.
- In `versions_s3.jsonl` and `versions_express.jsonl`, `compare_versions.py` did not set
  Zarr's `async.concurrency`, so the `main` version read its zone map 10 objects at a
  time instead of 128. Its times there are too slow; the other versions are not
  affected. The script now sets it.
