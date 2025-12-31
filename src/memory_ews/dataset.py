from __future__ import annotations
import numpy as np
import xarray as xr

def _guess_coord_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.variables:
            return c
    return None

def guess_lat_lon_time(ds: xr.Dataset) -> tuple[str, str, str]:
    lat = _guess_coord_name(ds, ["lat", "latitude", "y"])
    lon = _guess_coord_name(ds, ["lon", "longitude", "x"])
    time = _guess_coord_name(ds, ["time", "date", "t"])
    if lat is None or lon is None:
        raise ValueError(f"Could not infer lat/lon. Coords={list(ds.coords)} Vars={list(ds.variables)}")
    if time is None:
        raise ValueError(f"Could not infer time. Coords={list(ds.coords)} Vars={list(ds.variables)}")
    return lat, lon, time

def ensure_lon_180(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """
    Normalize longitudes to [-180, 180) if the dataset uses [0, 360).
    """
    lon = ds[lon_name]
    if float(lon.max()) > 180.0:
        lon_new = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords({lon_name: lon_new}).sortby(lon_name)
    return ds

def spatial_mean_box(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    bounds: tuple[float, float, float, float],
    area_weighted: bool = True,
) -> xr.DataArray:
    """
    Extract a lat-lon box and compute a spatial mean.
    Optionally uses cos(lat) area weights.
    """
    lat_min, lat_max, lon_min, lon_max = bounds

    lat_slice = slice(lat_min, lat_max) if lat_min < lat_max else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if lon_min < lon_max else slice(lon_max, lon_min)

    sub = da.sel({lat_name: lat_slice, lon_name: lon_slice})

    if not area_weighted:
        return sub.mean(dim=(lat_name, lon_name), skipna=True)

    w = np.cos(np.deg2rad(sub[lat_name]))
    w = xr.DataArray(w, coords={lat_name: sub[lat_name]}, dims=[lat_name])
    return sub.weighted(w).mean(dim=(lat_name, lon_name), skipna=True)
