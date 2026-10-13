#!/usr/bin/env bash
# Interleave each cumulative optimization and the best baseline (old and zero-copy reader).
export AWS_REGION=us-east-1
P=~/skipZFP-mdr/.venv/bin/python
R=s3://skipzfp/era5/era10y
OUT=~/attr.jsonl; : > $OUT
for rep in $(seq 10); do
  while read var sel th m path; do
    for who in $(printf "v0_before\nv1_direct\nv2_copyonce\nv3_pipeline\nv4_zerocopy\nbase_copy\nbase_zerocopy\n" | shuf); do
      case $who in
        v0_before)   PP=~/attr/tree_f4cf06a; D=~/skipZFP/experiments ;;
        v1_direct)   PP=~/attr/tree_158d471; D=~/skipZFP/experiments ;;
        v2_copyonce) PP=~/attr/tree_5c8a506; D=~/skipZFP/experiments ;;
        v3_pipeline) PP=~/attr/tree_3ef7636; D=~/skipZFP/experiments ;;
        v4_zerocopy) PP=~/attr/tree_b0a4a38; D=~/skipZFP/experiments ;;
        base_copy)     PP=~/attr/tree_3ef7636; D=~/skipZFP/experiments ;;
        base_zerocopy) PP=~/attr/tree_3ef7636; D=~/exp-new ;;
      esac
      if [[ $who == base_* ]]; then U=$R/$path; M=$m; else U=$R/stores_sub/$var/k8t.zarr; M=skipzfp; fi
      r=$(cd $D && PYTHONPATH=$PP $P exp2_threshold_query.py one $U $M $th 2>/dev/null | tail -1)
      echo "{\"who\":\"$who\",\"var\":\"$var\",\"sel\":$sel,\"rep\":$rep,\"r\":${r:-null}}" >> $OUT
    done
  done < ~/cases.txt
done
echo ATTR_DONE >> ~/attr.log
