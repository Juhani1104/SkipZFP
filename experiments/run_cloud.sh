#!/usr/bin/env bash
# Reproduce every experiment on a fresh Google Cloud VM (Debian or Ubuntu).
#
#   BUCKET=gs://your-bucket/skipzfp bash experiments/run_cloud.sh [step ...]
#
# Steps run in order: setup fetch prepare exp1 exp2 sensitivity local.
# Pass step names to run only some of them. Every step can be rerun; downloads and
# measurements resume where they stopped. The VM's service account needs read/write
# access to BUCKET, and the VM should be in the same region as the bucket.
set -euo pipefail

: "${BUCKET:?set BUCKET to a gs:// prefix you can write to}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$REPO/experiments"
DATA="${DATA:-$HOME/skipzfp_data}"
OUT="${OUT:-$EXP/results_rerun}"
PY="$REPO/.venv/bin/python"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(setup fetch prepare exp1 exp2 sensitivity local)
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
        --suite threshold --tmp "$DATA/tmp"
}

step_exp1() {
    cd "$EXP"
    "$PY" exp1_point_query.py check "$DATA/era5_2000.zarr" "$BUCKET"
    "$PY" exp1_point_query.py run "$BUCKET" "$OUT/exp1_point_query.jsonl"
}

step_exp2() {
    cd "$EXP"
    "$PY" exp2_threshold_query.py run "$DATA/era5_2000_2009.zarr" "$BUCKET/era10y" \
        "$OUT/exp2_threshold_query.jsonl"
}

step_sensitivity() {
    cd "$EXP"
    "$PY" sensitivity_object_size.py run "$DATA/era5_2000.zarr" "$BUCKET" \
        "$OUT/sensitivity_object_size.jsonl"
}

step_local() {
    cd "$EXP"
    "$PY" exp3_aggregate.py "$DATA/era5_2000.zarr" "$OUT/exp3_aggregate.json"
    "$PY" exp4_filtered_aggregate.py "$DATA/era5_2000.zarr" "$OUT/exp4_filtered_aggregate.json"
    "$PY" exp5_overhead.py "$OUT/exp5_overhead.json"
}

for s in "${STEPS[@]}"; do
    echo "== $s"
    "step_$s"
done
