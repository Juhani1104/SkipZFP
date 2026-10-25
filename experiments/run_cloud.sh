#!/usr/bin/env bash
# Reproduce every experiment on a fresh Google Cloud or AWS VM (Debian or Ubuntu).
#
#   BUCKET=gs://your-bucket/skipzfp bash experiments/run_cloud.sh [step ...]
#   BUCKET=s3://your-bucket/skipzfp bash experiments/run_cloud.sh [step ...]
#
# Steps run in order: setup fetch prepare prepare_bits exp1 exp2 ablation memory build
# sensitivity local fetch_pr prepare_pr exp2_pr fetch_bas prepare_bas exp2_bas summary.
# The *_pr and *_bas steps repeat exp2 on two other datasets, CMIP6 precipitation and
# SDRBench Basilisk turbulence; on Basilisk, SkipZFP also runs with a compressed zone
# map (layouts k8-z and k8t-z). prepare_z, not run by default, builds the same two
# layouts for ERA5, and fetch_glob, prepare_glob and exp2_glob, also not run by
# default, repeat exp2 on ERA5 over most of the globe, and memory_glob, after
# exp2_glob, the memory limits on it; prepare_glob_o and exp2_glob_o repeat
# exp2_glob with larger objects. Pass step names to run only some of them.
# Every step can be rerun; downloads and measurements resume where they stopped.
# EXP2_REPS (default 30), ABL_REPS (20), TAIL_REPS (30), MEM_REPS (3) and BUILD_REPS
# (2) set the repetitions, and LIMITS (1G,2G,4G,none) the memory limits. NT, if set,
# cuts the ten-year data to its first NT steps, for a quick trial run on a small store
# (BUCKET may then be a local folder).
# Needs Python 3.11 or newer (Debian 12 has it).
# Every results file gets a <file>.runs.json with the code version, machine,
# packages and data of each run; run from a git checkout, or ship SOURCE.json. The VM's service account (GCP) or instance
# role (AWS) needs read/write access to BUCKET, and the VM should be in the same region
# as the bucket. On AWS, set AWS_REGION to the bucket's region (default us-east-1).
set -euo pipefail

: "${BUCKET:?set BUCKET to a gs:// or s3:// prefix you can write to}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$REPO/experiments"
DATA="${DATA:-$HOME/skipzfp_data}"
OUT="${OUT:-$EXP/results_rerun}"
PY="$REPO/.venv/bin/python"
TEN="$DATA/era5_2000_2009.zarr"
PR="$DATA/cmip6_pr_2000_2009.zarr"
BAS="$DATA/basilisk_u.zarr"
GLOB="$DATA/era5_global.zarr"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(setup fetch prepare prepare_bits exp1 exp2 ablation
    memory build sensitivity local fetch_pr prepare_pr exp2_pr fetch_bas prepare_bas
    exp2_bas summary)
mkdir -p "$DATA" "$OUT"

step_setup() {
    sudo apt-get update -q
    sudo apt-get install -y -q build-essential cmake git python3-venv
    if [ ! -f "$HOME/.local/lib/libzfp.so" ]; then
        git clone -q --depth 1 --branch 1.0.1 https://github.com/LLNL/zfp.git /tmp/zfp
        cmake -S /tmp/zfp -B /tmp/zfp/build -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
            -DBUILD_TESTING=OFF -DBUILD_UTILITIES=OFF
        cmake --build /tmp/zfp/build -j "$(nproc)" --target install
        rm -rf /tmp/zfp
    fi
    if [[ "$BUCKET" == s3://* ]] && ! command -v aws >/dev/null; then
        sudo snap install aws-cli --classic
    fi
    python3 -m venv "$REPO/.venv"
    "$PY" -m pip install -q -r "$REPO/requirements.txt"
    ZFP_DIR="$HOME/.local" "$PY" -m pip install -q -e "$REPO"
}

step_fetch() {
    "$PY" "$EXP/fetch_era5.py" "$DATA/era5_2000.zarr" 2000
    "$PY" "$EXP/fetch_era5.py" "$DATA/era5_2000_2009.zarr" 2000 2009
}

step_prepare() {
    cd "$EXP"
    "$PY" prepare_stores.py "$DATA/era5_2000.zarr" "$BUCKET" --suite point --tmp "$DATA/tmp"
    "$PY" prepare_stores.py "$DATA/era5_2000_2009.zarr" "$BUCKET/era10y" \
        --suite threshold --tmp "$DATA/tmp" ${NT:+--nt "$NT"}
}

step_prepare_bits() {
    cd "$EXP"
    "$PY" prepare_stores.py "$TEN" "$BUCKET/era10y" --suite bits --tmp "$DATA/tmp" \
        ${NT:+--nt "$NT"}
}

step_exp1() {
    cd "$EXP"
    "$PY" exp1_point_query.py check "$DATA/era5_2000.zarr" "$BUCKET"
    "$PY" exp1_point_query.py run "$BUCKET" "$OUT/exp1_point_query.jsonl"
}

step_exp2() {
    cd "$EXP"
    "$PY" exp2_threshold_query.py run "$TEN" "$BUCKET/era10y" \
        "$OUT/exp2_threshold_query.jsonl" --reps "${EXP2_REPS:-30}" --cache "$DATA/tmp" \
        ${EXP2_LAYOUTS:+--layouts "$EXP2_LAYOUTS"} ${NT:+--nt "$NT"}
}

# k8-z and k8t-z next to exp2's stores; with EXP2_LAYOUTS=k8t,k8,k8t-z,k8-z, exp2 then
# also measures the compressed zone map
step_prepare_z() {
    cd "$EXP"
    "$PY" prepare_stores.py "$TEN" "$BUCKET/era10y" --suite zmeta --tmp "$DATA/tmp" \
        ${NT:+--nt "$NT"}
}

step_fetch_pr() {
    "$PY" "$EXP/fetch_cmip6.py" "$PR" ${NT:+"$NT"}
}

step_prepare_pr() {
    cd "$EXP"
    for suite in threshold tight; do
        "$PY" prepare_stores.py "$PR" "$BUCKET/cmip6pr" --suite "$suite" --vars pr \
            --tmp "$DATA/tmp" ${NT:+--nt "$NT"}
    done
}

step_exp2_pr() {
    cd "$EXP"
    "$PY" exp2_threshold_query.py run "$PR" "$BUCKET/cmip6pr" "$OUT/exp2_cmip6_pr.jsonl" \
        --reps "${EXP2_REPS:-30}" --cache "$DATA/tmp" --vars pr --tight ${NT:+--nt "$NT"}
}

step_fetch_bas() {
    "$PY" "$EXP/fetch_basilisk.py" "$BAS" ${NT:+"$NT"}
}

step_prepare_bas() {
    cd "$EXP"
    for suite in threshold zmeta; do
        "$PY" prepare_stores.py "$BAS" "$BUCKET/basilisk" --suite "$suite" --vars u \
            --tmp "$DATA/tmp" ${NT:+--nt "$NT"}
    done
}

step_exp2_bas() {
    cd "$EXP"
    "$PY" exp2_threshold_query.py run "$BAS" "$BUCKET/basilisk" \
        "$OUT/exp2_basilisk_u.jsonl" --reps "${EXP2_REPS:-30}" --cache "$DATA/tmp" \
        --vars u --layouts k8t,k8,k8t-z,k8-z ${NT:+--nt "$NT"}
}

# exp2 on ERA5 over most of the globe, GLOB_FIRST to GLOB_LAST (default 2000 only,
# 2.5 times the ten European years); not run by default
step_fetch_glob() {
    "$PY" "$EXP/fetch_era5.py" "$GLOB" "${GLOB_FIRST:-2000}" "${GLOB_LAST:-2000}" global
}

step_prepare_glob() {
    cd "$EXP"
    "$PY" prepare_stores.py "$GLOB" "$BUCKET/global" --suite threshold --tmp "$DATA/tmp" \
        ${NT:+--nt "$NT"}
}

step_exp2_glob() {
    cd "$EXP"
    "$PY" exp2_threshold_query.py run "$GLOB" "$BUCKET/global" "$OUT/exp2_global.jsonl" \
        --reps "${EXP2_REPS:-30}" --cache "$DATA/tmp" ${NT:+--nt "$NT"}
}

# exp2 on the global ERA5 with larger objects (GLOB_OBJECT, default 64x320x640) for
# SkipZFP and Zarr sharding; the chunked baselines are read from the global suite
step_prepare_glob_o() {
    cd "$EXP"
    SKIPZFP_OBJECT="${GLOB_OBJECT:-64x320x640}" "$PY" prepare_stores.py "$GLOB" \
        "$BUCKET/global_o" --suite objects --tmp "$DATA/tmp" ${NT:+--nt "$NT"}
}

step_exp2_glob_o() {
    cd "$EXP"
    SKIPZFP_OBJECT="${GLOB_OBJECT:-64x320x640}" "$PY" exp2_threshold_query.py run \
        "$GLOB" "$BUCKET/global_o" "$OUT/exp2_global_o.jsonl" \
        --reps "${EXP2_REPS:-30}" --cache "$DATA/tmp" --chunk-root "$BUCKET/global" \
        ${NT:+--nt "$NT"}
}

# the memory limits on the global ERA5, with exp2_glob's thresholds
step_memory_glob() {
    cd "$EXP"
    "$PY" memory_limit.py run "$GLOB" "$BUCKET/global" "$OUT/memory_limit_global.jsonl" \
        --limits "${LIMITS:-1G,2G,4G,none}" --reps "${MEM_REPS:-3}" \
        --thresholds-from "$OUT/exp2_global.jsonl"
}

step_ablation() {
    cd "$EXP"
    "$PY" ablation.py run "$TEN" "$BUCKET/era10y" "$OUT/ablation.jsonl" \
        --reps "${ABL_REPS:-20}" --tail-reps "${TAIL_REPS:-30}" --cache "$DATA/tmp" \
        ${NT:+--nt "$NT"}
}

step_memory() {
    cd "$EXP"
    "$PY" memory_limit.py run "$TEN" "$BUCKET/era10y" "$OUT/memory_limit.jsonl" \
        --limits "${LIMITS:-1G,2G,4G,none}" --reps "${MEM_REPS:-3}" --cache "$DATA/tmp" \
        ${NT:+--nt "$NT"}
}

step_build() {
    cd "$EXP"
    "$PY" build_cost.py run "$TEN" "$OUT/build_cost.jsonl" --tmp "$DATA/build_tmp" \
        --reps "${BUILD_REPS:-2}" ${NT:+--nt "$NT"}
}

step_summary() {
    cd "$EXP"
    "$PY" summarize.py "$OUT" --out "$OUT/summary.md"
    "$PY" figures.py "$OUT" --out "$OUT/figures"
}

step_sensitivity() {
    cd "$EXP"
    "$PY" sensitivity_object_size.py run "$DATA/era5_2000.zarr" "$BUCKET" \
        "$OUT/sensitivity_object_size.jsonl"
}

step_local() {
    cd "$EXP"
    "$PY" exp3_aggregate.py "$DATA/era5_2000.zarr" "$OUT/exp3_aggregate.json"
    # exp4 builds its own stores to read from; keep them, and the blocks each query
    # read, with the data rather than among the results
    "$PY" exp4_filtered_aggregate.py "$DATA/era5_2000.zarr" "$OUT/exp4_filtered_aggregate.json" \
        --stores "$DATA/exp4_stores"
    "$PY" exp5_overhead.py "$OUT/exp5_overhead.json"
}

for s in "${STEPS[@]}"; do
    echo "== $s"
    "step_$s"
done
