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
