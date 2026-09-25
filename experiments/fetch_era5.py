"""Download ERA5 t2m and 10 m wind speed over Europe from the public ARCO-ERA5 store.

usage: python fetch_era5.py <out.zarr> [first_year] [last_year]

Region 71.75N-40N, 10W-53.75E (128 x 256 at 0.25 deg), hourly. Wind speed is
sqrt(u10^2 + v10^2) computed in float64 and stored as float32. Reads are anonymous and
the download resumes where it stopped. One year is used by experiments 1, 3 and 4 and
the sensitivity study; experiment 2 uses 2000-2009.
"""

import json
import os
import sys
import time

import dask
import numpy as np
import xarray as xr
import zarr

SRC = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARS = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}
LAT = slice(73, 201)
LON = np.r_[1400:1440, 0:216]
STEP = 96


def main(out, first, last):
    done_path = out.rstrip("/") + ".done.json"
    ds = xr.open_zarr(SRC, storage_options={"token": "anon"}, chunks={"time": 24})
    start = first if "-" in first else f"{first}-01-01T00"
    end = last if "-" in last else f"{last}-12-31T23"
    ds = ds[list(VARS.values())].sel(time=slice(start, end)).isel(latitude=LAT)
    nt = ds.sizes["time"]

    root = zarr.open_group(out, mode="a")
    for name in ("t2m", "ws"):
        if name not in root:
            root.create_array(
                name,
                shape=(nt, 128, 256),
                chunks=(STEP, 128, 256),
                dtype="float32",
                compressors=None,
                fill_value=np.nan,
            )
    if "time" not in root:
        hours = ds.time.values.astype("datetime64[h]").astype("int64")
        root.create_array("time", data=hours)
        root.attrs.update(
            source=SRC, years=f"{first}-{last}", time_unit="hours since 1970-01-01"
        )

    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    t0 = time.time()
    total = -(-nt // STEP)
    with dask.config.set(scheduler="threads", num_workers=48):
        for s in range(0, nt, STEP):
            if s in done:
                continue
            sub = ds.isel(time=slice(s, s + STEP))
            arrays = dask.compute(*[sub[n].data for n in VARS.values()])
            t, u, v = (np.asarray(x)[:, :, LON] for x in arrays)
            assert (
                np.isfinite(t).all() and np.isfinite(u).all() and np.isfinite(v).all()
            )
            u = u.astype(np.float64)
            v = v.astype(np.float64)
            root["t2m"][s : s + len(t)] = t.astype(np.float32)
            root["ws"][s : s + len(t)] = np.sqrt(u * u + v * v).astype(np.float32)
            done.add(s)
            json.dump(sorted(done), open(done_path, "w"))
            if len(done) % 20 == 0 or len(done) == total:
                minutes = (time.time() - t0) / 60
                print(f"{len(done)}/{total} blocks, {minutes:.1f} min", flush=True)
    print(f"finished: {nt} hours", flush=True)


if __name__ == "__main__":
    first = sys.argv[2] if len(sys.argv) > 2 else "2000"
    last = sys.argv[3] if len(sys.argv) > 3 else first
    main(sys.argv[1], first, last)
