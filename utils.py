"""
Utility functions for sampling climate model precipitation change (PR)
at spatial grid points in NZTM2000 coordinates.
"""
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree


def prepare_pr_kdtree_nztm(
    ds: xr.Dataset,
    var: str = "PR",
    grid_crs: str = "EPSG:4326",
    point_crs: str = "EPSG:2193",
):
    """
    Build a KDTree from the valid (non-NaN) cells of a PR grid,
    with coordinates projected into NZTM metres for distance queries.

    Returns
    -------
    tree : cKDTree
    v_valid : ndarray of shape (n_valid,)
        PR values at the valid grid cells.
    """
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    da = da.sortby("latitude").sortby("longitude")

    arr = da.to_numpy()
    lat = da["latitude"].to_numpy()
    lon = da["longitude"].to_numpy()
    LON, LAT = np.meshgrid(lon, lat)

    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        raise ValueError("No valid (non-NaN) cells found in PR grid.")

    to_nztm = Transformer.from_crs(grid_crs, point_crs, always_xy=True)
    valid_e, valid_n = to_nztm.transform(LON[valid].ravel(), LAT[valid].ravel())

    tree = cKDTree(np.column_stack([valid_e, valid_n]))
    v_valid = arr[valid].ravel().astype(float)
    return tree, v_valid


def add_pr_change_to_points(
    ds: xr.Dataset,
    points: pd.DataFrame,
    east_col: str = "easting",
    north_col: str = "northing",
    var: str = "PR",
    out_col: str = "PR_change_pct",
    fill_missing: bool = True,
    max_dist_m: float = 15_000,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """
    Sample PR (% precipitation change) from a climate NetCDF at NZTM grid points.

    Nearest-cell sampling is applied first. Points that fall outside the grid
    extent (coastal/edge NaNs) are optionally filled by snapping to the nearest
    valid cell within `max_dist_m` metres.

    Parameters
    ----------
    ds : xr.Dataset
        NetCDF dataset with a PR variable on a longitude/latitude grid.
    points : pd.DataFrame
        Must contain easting/northing columns in NZTM2000 (EPSG:2193).
    east_col, north_col : str
        Column names for easting and northing in `points`.
    var : str
        Name of the precipitation-change variable in `ds`.
    out_col : str
        Name of the output column added to the returned DataFrame.
    fill_missing : bool
        If True, fill edge NaNs by snapping to nearest valid grid cell.
    max_dist_m : float
        Maximum snap distance in metres when filling edge NaNs.
    chunk_size : int
        Rows processed per batch (reduces peak memory for large grids).

    Returns
    -------
    pd.DataFrame
        Copy of `points` with `out_col` column appended.
    """
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    da = da.sortby("latitude").sortby("longitude")

    to_lonlat = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)

    out = points.copy()
    out[out_col] = np.nan

    if fill_missing:
        tree, v_valid = prepare_pr_kdtree_nztm(ds, var=var)

    n = len(out)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        sl = out.iloc[i0:i1]

        e = sl[east_col].to_numpy(float)
        n_ = sl[north_col].to_numpy(float)
        lon, lat = to_lonlat.transform(e, n_)

        lon_da = xr.DataArray(lon, dims="points")
        lat_da = xr.DataArray(lat, dims="points")
        vals = da.sel(longitude=lon_da, latitude=lat_da, method="nearest").to_numpy()
        out.iloc[i0:i1, out.columns.get_loc(out_col)] = vals

        if fill_missing:
            miss = np.isnan(vals)
            if miss.any():
                xy_pts = np.column_stack([e[miss], n_[miss]])
                dist_m, idx = tree.query(xy_pts, k=1)
                fill_vals = v_valid[idx]
                ok = dist_m <= max_dist_m

                block = out.iloc[i0:i1, out.columns.get_loc(out_col)].to_numpy()
                block_idx = np.where(miss)[0]
                block[block_idx[ok]] = fill_vals[ok]
                out.iloc[i0:i1, out.columns.get_loc(out_col)] = block

    return out
