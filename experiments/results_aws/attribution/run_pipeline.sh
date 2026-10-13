#!/usr/bin/env bash
# zero-copy with and without pipelining: untraced timing (like exp2) and traced runs, interleaved
export AWS_REGION=us-east-1
P=~/skipZFP-mdr/.venv/bin/python
U=s3://skipzfp/era5/era10y/stores_sub/t2m/k8t.zarr
mkdir -p ~/ptrace; rm -f ~/ptrace/*; OUT=~/pipe_cmp.jsonl; : > $OUT
for rep in $(seq 8); do
  for sel in "0.001 307.3660888671875" "0.01 301.4148864746094" "0.1 292.49176025390625" "0.5 280.8170471191406"; do
    set -- $sel
    for job in $(printf "pipe:time\nnopipe:time\npipe:trace\nnopipe:trace\n" | shuf); do
      v=${job%%:*}; k=${job##*:}
      [ $v = pipe ] && PP=~/attr/tree_b0a4a38 || PP=~/attr/tree_zc_nopipe
      if [ $k = time ]; then
        r=$(cd ~/skipZFP/experiments && PYTHONPATH=$PP $P exp2_threshold_query.py one $U skipzfp $2 2>/dev/null | tail -1)
        echo "{\"v\":\"$v\",\"sel\":$1,\"rep\":$rep,\"r\":${r:-null}}" >> $OUT
      else
        PYTHONPATH=$PP $P ~/trace_query.py $U $2 ~/ptrace/${v}_$1_r$rep.json > /dev/null 2>&1
      fi
    done
  done
done
echo PIPE_DONE >> ~/pipe_cmp.log
