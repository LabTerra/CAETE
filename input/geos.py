# -*-coding:utf-8-*-
# "CAETÊ"
# Author: João Paulo Darela Filho
#
# Copyright 2017- LabTerra
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

"""
Self-contained geospatial helpers for the CAETÊ pre-processing pipeline.

Coordinate convention
---------------------
* Global 0.5° regular lat/lon grid of shape ``(360, 720)``.
* ``y = 0`` is the northernmost row (90°N), ``x = 0`` is the westernmost
  column (180°W). Latitude decreases with increasing ``y``; longitude
  increases with increasing ``x``.

Public API
----------
* ``YRES``, ``XRES`` — module-level resolution constants (degrees).
* ``find_indices_xy(N, W)`` — global ``(y, x)`` indices for a given lat/lon.
* ``find_coordinates_xy(y, x)`` — cell-center lat/lon for given indices.
* ``define_region(north, south, west, east)`` — bounding-box dict with
  ``ymin, ymax, xmin, xmax`` global indices.
* ``pan_amazon_region`` — Pan-Amazon bounding box on the 0.5° grid.
* ``global_region`` — full-globe bounding box (sanity reference).
"""

from typing import Tuple, Dict
import numpy as np

__all__ = [
    "YRES",
    "XRES",
    "find_indices_xy",
    "find_coordinates_xy",
    "define_region",
    "pan_amazon_region",
    "global_region",
]


# Hard-coded grid resolution. CAETÊ runs on a 0.5° global grid.
# TODO: sent this to a config file
YRES: float = 0.5
XRES: float = 0.5


def _calc_min_rounding_log(res: float) -> int:
    """Return a safe number of decimals for coordinate rounding at ``res``."""
    if res >= 1.0:
        return 2
    decimal_places = max(1, int(-np.log10(res)) + 1)
    if res < 0.01:
        safety_margin = 6
    elif res < 0.1:
        safety_margin = 5
    else:
        safety_margin = 4
    return max(2, decimal_places + safety_margin)


def find_indices_xy(
    N: float,
    W: float,
    res_y: float = YRES,
    res_x: float = XRES,
    rounding: int = 2,
) -> Tuple[int, int]:
    """Find global ``(y, x)`` indices for a latitude/longitude pair.

    Parameters
    ----------
    N : float
        Latitude in decimal degrees north.
    W : float
        Longitude in decimal degrees east.
    res_y, res_x : float, optional
        Grid resolution. Defaults to the module-level 0.5°.
    rounding : int, optional
        Minimum decimal precision for coordinate rounding.

    Returns
    -------
    (Yind, Xind) : tuple[int, int]
        Indices on the 360 × 720 grid. ``(0, 0)`` is the upper-left corner.
        Returns ``(-1, -1)`` for invalid resolutions or out-of-range
        coordinates.
    """
    if res_y <= 0 or res_x <= 0:
        return -1, -1

    eff_y = max(rounding, _calc_min_rounding_log(res_y))
    eff_x = max(rounding, _calc_min_rounding_log(res_x))

    Yc = round(N, eff_y)
    Xc = round(W, eff_x)

    half_y = res_y / 2
    half_x = res_x / 2

    lat = np.arange(-90 + half_y, 90, res_y)
    lon = np.arange(-180 + half_x, 180, res_x)

    Yind = int(np.searchsorted(lat, -Yc - half_y, side="left"))
    Xind = int(np.searchsorted(lon, Xc - half_x, side="left"))

    if Yc > 90 or Yc < -90:
        Yind = -1
    if Xc < -180 or Xc > 180:
        Xind = -1

    return Yind, Xind


def find_coordinates_xy(
    Yind: int,
    Xind: int,
    res_y: float = YRES,
    res_x: float = XRES,
    rounding: int = 2,
) -> Tuple[float, float]:
    """Return ``(N, W)`` cell-center coordinates for grid indices."""
    half_y = res_y / 2
    half_x = res_x / 2

    eff_y = max(rounding, _calc_min_rounding_log(res_y))
    eff_x = max(rounding, _calc_min_rounding_log(res_x))

    lat = np.arange(-90 + half_y, 90, res_y)[::-1]
    lon = np.arange(-180 + half_x, 180, res_x)

    return round(float(lat[Yind]), eff_y), round(float(lon[Xind]), eff_x)


def define_region(
    north: float,
    south: float,
    west: float,
    east: float,
    res_y: float = YRES,
    res_x: float = XRES,
    rounding: int = 2,
) -> Dict[str, int]:
    """Define a bounding box (in global grid indices) for a region.

    Parameters
    ----------
    north, south : float
        Northernmost / southernmost latitudes (degrees north).
    west, east : float
        Westernmost / easternmost longitudes (degrees east).

    Returns
    -------
    dict
        ``{"ymin", "ymax", "xmin", "xmax"}`` integer indices on the global
        grid. ``ymin/xmin`` correspond to the ``(north, west)`` corner;
        ``ymax/xmax`` to the ``(south, east)`` corner.
    """
    ymin, xmin = find_indices_xy(north, west, res_y, res_x, rounding)
    ymax, xmax = find_indices_xy(south, east, res_y, res_x, rounding)
    return {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}


# Pan-Amazon bounding box on the 0.5° global grid.
pan_amazon_region: Dict[str, int] = define_region(
    north=10.5, south=-21.5, west=-80.0, east=-43.0
)

# Full-globe reference region.
global_region: Dict[str, int] = define_region(
    north=90.0, south=-90.0, west=-180.0, east=180.0
)


if __name__ == "__main__":  # pragma: no cover - quick sanity check
    print("YRES, XRES:", YRES, XRES)
    print("pan_amazon_region:", pan_amazon_region)
    print("global_region:", global_region)
