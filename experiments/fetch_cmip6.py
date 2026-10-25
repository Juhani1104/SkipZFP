"""Download 3-hourly CMIP6 precipitation, a dataset beside ERA5, from the Pangeo store.

usage: python fetch_cmip6.py <out.zarr> [steps]

HadGEM3-GC31-HM highresSST-present (Met Office, about 25 km), variable pr in
kg m-2 s-1, region 0-60N, 0-180E (256 x 512 grid points). The first 28800 steps, the
default, are 2000-2009 in the model's 360-day calendar. Unlike the ERA5 fields,
precipitation is zero or near zero at most places and heavy in small patches, so it
tests pruning on a different distribution. Reads are anonymous and the download
resumes where it stopped.
"""

import json
import os
import sys
import time

import dask
import numpy as np
import xarray as xr
import zarr

SRC = (
    "gs://cmip6/CMIP6/HighResMIP/MOHC/HadGEM3-GC31-HM/highresSST-present/"
    "r1i1p1f1/3hr/pr/gn/v20170831/"
)
LAT = slice(384, 640)
LON = slice(0, 512)
STEP = 41 * 24  # 24 of the source's 41-step chunks


def main(out, nt):
    done_path = out.rstrip("/") + ".done.json"
    ds = xr.open_zarr(SRC, storage_options={"token": "anon"}, decode_times=False)
    pr = ds["pr"].isel(time=slice(0, nt), lat=LAT, lon=LON)
    nt = pr.sizes["time"]
    shape = (nt, LAT.stop - LAT.start, LON.stop - LON.start)

    root = zarr.open_group(out, mode="a")
    if "time" not in root:
        root.create_array(
            "pr",
            shape=shape,
            chunks=(24, *shape[1:]),
            dtype="float32",
            compressors=None,
            fill_value=np.nan,
        )
        root.create_array("time", data=ds.time.values[:nt].astype(np.float64))
        lat, lon = ds.lat.values[LAT], ds.lon.values[LON]
        root.attrs.update(
            source=SRC,
            steps=nt,
            time_unit=ds.time.attrs["units"],
            calendar=ds.time.attrs["calendar"],
            lat=[float(lat[0]), float(lat[-1])],
            lon=[float(lon[0]), float(lon[-1])],
        )

    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    t0 = time.time()
    total = -(-nt // STEP)
    with dask.config.set(scheduler="threads", num_workers=48):
        for s in range(0, nt, STEP):
            if s in done:
                continue
            x = np.asarray(pr.isel(time=slice(s, s + STEP)).data.compute())
            assert np.isfinite(x).all()
            root["pr"][s : s + len(x)] = x.astype(np.float32)
            done.add(s)
            json.dump(sorted(done), open(done_path, "w"))
            if len(done) % 5 == 0 or len(done) == total:
                minutes = (time.time() - t0) / 60
                print(f"{len(done)}/{total} blocks, {minutes:.1f} min", flush=True)
    print(f"finished: {nt} steps", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 28800)
