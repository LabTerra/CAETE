"""Pre-processing of input data to feed CAETÊ.

Reads ISIMIP-style NetCDF climate data and soil nutrient ``.npy`` arrays and
writes one bzip2-pickle file per land gridcell, plus a single metadata file,
in the format consumed by :mod:`caete` (see ``input/creating_caete_input_files.md``
for the full format specification).

Layout expected for the raw climate data::

    {climate_data}/{dataset}/{mode}_raw/*_{var}_*.nc[4]

with ``{var}`` ∈ ``{hurs, tas, pr, ps, rsds}``. Output files are written to
``./{dataset}/{mode}/`` next to this script. The metadata file is named
``ISIMIP_HISTORICAL_METADATA.pbz2`` and per-gridcell files are named
``input_data_{Y}-{X}.pbz2`` with **global** ``(Y, X)`` indices on the
360 × 720 grid (see ``geos.py``).

Configuration lives in ``pre_processing.toml``; CLI flags override defaults.

Originally authored by jpdarela (Mon Dec 28 18:08:27 -03 2020). Rewritten to
use a vectorized I/O pipeline while preserving the on-disk layout
(filenames, dict keys, metadata file name) of the legacy producer.
"""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import os
import sys
import tomllib
from copy import deepcopy
from pathlib import Path

import _pickle as pkl
import numpy as np
from netCDF4 import Dataset, MFDataset, MFTime  # type: ignore

from geos import pan_amazon_region


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# These are the variables CAETÊ expects in every per-gridcell .pbz2 file.
CLIMATE_VARS: tuple[str, ...] = ("hurs", "tas", "pr", "ps", "rsds")
SOIL_VARS: tuple[str, ...] = ("tn", "tp", "ap", "ip", "op")

# Filename used by the legacy producer; kept identical for backward compat.
METADATA_FILENAME = "ISIMIP_HISTORICAL_METADATA.pbz2"

GRID_SHAPE = (360, 720)

# ANSI colors for friendlier console output.
_C_BLUE = "\033[94m"
_C_RED = "\033[91m"
_C_CYAN = "\033[96m"
_C_GREEN = "\033[92m"
_C_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# CLI & configuration
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset folder name (overrides toml).")
    p.add_argument("--mode", type=str, default=None,
                   help="Mode subfolder (e.g. obsclim, spinclim).")
    p.add_argument("--mask-file", type=str, default=None,
                   help="Override mask file path from toml.")
    p.add_argument("--test", action="store_true",
                   help="Validate previously written .pbz2 files against raw NetCDF.")
    return p


def _load_config(toml_path: Path) -> dict:
    if not toml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {toml_path}")
    with open(toml_path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

def open_clim_dataset(raw_data: Path, var: str) -> Dataset | MFDataset:
    """Open ``var``'s NetCDF file(s) under ``raw_data`` as a single dataset."""
    files = sorted(raw_data.glob(f"*_{var}_*"))
    if not files:
        raise FileNotFoundError(
            f"No NetCDF file for variable {var!r} in {raw_data}"
        )
    if len(files) == 1:
        return Dataset(str(files[0]))
    return MFDataset([str(f) for f in files])


def get_time_var(ds: Dataset | MFDataset):
    """Return the time variable, wrapped in ``MFTime`` for ``MFDataset``."""
    if isinstance(ds, MFDataset):
        return MFTime(ds.variables["time"])
    return ds.variables["time"]


# ---------------------------------------------------------------------------
# Vectorized climate variable extraction
# ---------------------------------------------------------------------------

def process_climate_variable(var: str, raw_data: Path, region: dict,
                             local_mask: np.ndarray) -> np.ndarray:
    """Read ``var`` for the regional bbox and return ``(n_stations, time)``.

    The returned array has one row per *unmasked* gridcell (in row-major
    ``(Y, X)`` order over ``local_mask``) and one column per time step.
    Any masked entries are filled with the variable's regional mean.
    """
    print(f"  {_C_BLUE}reading{_C_RESET} {var} ...", flush=True)
    ds = open_clim_dataset(raw_data, var)
    try:
        cube = ds.variables[var][
            :,
            region["ymin"]: region["ymax"],
            region["xmin"]: region["xmax"],
        ]  # (time, ny, nx) masked array
    finally:
        ds.close()

    valid = ~local_mask
    # Vectorized fancy indexing → (time, n_stations) → transpose.
    station = cube[:, valid].T  # ndarray-or-masked-array

    # The model expects no masked / NaN values. Fill any masked entries
    # with the regional mean of the variable; print a warning if any exist.
    if np.ma.isMaskedArray(station):
        n_masked = int(np.ma.count_masked(station))
        if n_masked:
            print(
                f"  {_C_RED}warning:{_C_RESET} {n_masked} masked values "
                f"in {var}; filling with regional mean.",
                flush=True,
            )
        station = station.filled(float(station.mean()))

    print(
        f"  extracted {station.shape[0]} stations × {station.shape[1]} timesteps for {var}",
        flush=True,
    )
    return np.ascontiguousarray(station)


def read_soil(soil_data: Path, soil_files: dict, var: str) -> np.ndarray:
    fname = soil_files[var]
    arr = np.load(soil_data / fname)
    if arr.shape != GRID_SHAPE:
        raise ValueError(
            f"Soil array {fname} has shape {arr.shape}; expected {GRID_SHAPE}"
        )
    return arr


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class DSMetadata:
    """Collects coordinate metadata and writes the metadata pbz2 file.

    Mirrors the legacy ``ds_metadata`` class so the on-disk format is
    unchanged: a 3-tuple ``(time, lat, lon)`` of plain dicts.
    """

    def __init__(self) -> None:
        self.time: dict = {
            "standard_name": None, "units": None,
            "calendar": None, "time_index": None,
        }
        self.lat: dict = {
            "standard_name": None, "units": None,
            "axis": None, "lat_index": None,
        }
        self.lon: dict = {
            "standard_name": None, "units": None,
            "axis": None, "lon_index": None,
        }
        self._ok = False

    def fill(self, ds: Dataset | MFDataset, time_var) -> None:
        self.time["standard_name"] = time_var.standard_name
        self.time["units"] = time_var.units
        self.time["calendar"] = time_var.calendar
        self.time["time_index"] = time_var[:]

        latv = ds.variables["lat"]
        lonv = ds.variables["lon"]
        self.lat["standard_name"] = latv.standard_name
        self.lat["units"] = latv.units
        self.lat["axis"] = latv.axis
        self.lat["lat_index"] = latv[:]
        self.lon["standard_name"] = lonv.standard_name
        self.lon["units"] = lonv.units
        self.lon["axis"] = lonv.axis
        self.lon["lon_index"] = lonv[:]
        self._ok = True

    def write(self, fpath: Path) -> None:
        if not self._ok:
            raise RuntimeError("Metadata not filled; call fill() first.")
        with bz2.BZ2File(fpath, mode="w") as fh:
            pkl.dump((self.time, self.lat, self.lon), fh)


class GridcellWriter:
    """Per-gridcell .pbz2 manager.

    ``y`` and ``x`` are stored as **global** indices on the 360×720 grid,
    matching the legacy filename convention.
    """

    __slots__ = ("y", "x", "fpath", "_data")

    def __init__(self, global_y: int, global_x: int, dpath: Path) -> None:
        self.y = global_y
        self.x = global_x
        self.fpath = dpath / f"input_data_{self.y}-{self.x}.pbz2"
        # Preserve dict key order: 5 climate then 5 soil.
        self._data: dict = {k: None for k in CLIMATE_VARS + SOIL_VARS}

    def set(self, var: str, value) -> None:
        if var not in self._data:
            raise KeyError(f"Unknown variable {var!r}")
        # deepcopy avoids cross-cell aliasing for soil scalars (cheap)
        # and is a no-op for the per-cell climate slices.
        self._data[var] = deepcopy(value)

    def load(self) -> None:
        with bz2.BZ2File(self.fpath, mode="r") as fh:
            self._data = pkl.load(fh)

    def write(self) -> None:
        with bz2.BZ2File(self.fpath, mode="w") as fh:
            pkl.dump(self._data, fh)


def _write_climate_var_to_cell(cell: GridcellWriter, var: str, ts: np.ndarray) -> None:
    cell.load()
    cell.set(var, ts)
    cell.write()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: dict, dataset: str, mode: str, mask: np.ndarray,
        out_dir: Path) -> None:
    region = pan_amazon_region
    if not (0 <= region["ymin"] < region["ymax"] < GRID_SHAPE[0]):
        raise ValueError(f"Invalid y bounds: {region}")
    if not (0 <= region["xmin"] < region["xmax"] < GRID_SHAPE[1]):
        raise ValueError(f"Invalid x bounds: {region}")

    raw_data = Path(config["climate_data"]) / dataset / f"{mode}_raw"
    if not raw_data.exists():
        raise FileNotFoundError(f"Raw climate folder not found: {raw_data}")

    soil_data = Path(config["soil_data"]).resolve()
    if not soil_data.exists():
        raise FileNotFoundError(f"Soil data folder not found: {soil_data}")
    soil_files: dict = config["soil_files"]

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous outputs (legacy behavior).
    for old in out_dir.glob("*.pbz2"):
        old.unlink()

    print(f"{_C_BLUE}Raw climate folder:{_C_RESET} {raw_data}")
    print(f"{_C_BLUE}Output folder:    {_C_RESET} {out_dir}")

    # ---- Metadata ------------------------------------------------------
    print(f"\n{_C_BLUE}1/4 Building metadata{_C_RESET}")
    datasets = {v: open_clim_dataset(raw_data, v) for v in CLIMATE_VARS}
    try:
        time_vars = {v: get_time_var(datasets[v]) for v in CLIMATE_VARS}
        ref = time_vars[CLIMATE_VARS[0]]
        ref_arr = ref[:]
        for v, tv in time_vars.items():
            if tv.units != ref.units or tv.calendar != ref.calendar:
                raise ValueError(
                    f"time units/calendar mismatch for {v}: "
                    f"({tv.units}, {tv.calendar}) vs ({ref.units}, {ref.calendar})"
                )
            if not np.array_equal(tv[:], ref_arr):
                raise ValueError(f"time index mismatch for {v}")

        meta = DSMetadata()
        meta.fill(datasets[CLIMATE_VARS[0]], ref)
        meta.write(out_dir / METADATA_FILENAME)
        print(f"  wrote {METADATA_FILENAME} ({ref_arr.size} timesteps)")
    finally:
        for ds in datasets.values():
            ds.close()

    # ---- Build gridcell writers ---------------------------------------
    print(f"\n{_C_BLUE}2/4 Selecting valid gridcells{_C_RESET}")
    local_mask = mask[
        region["ymin"]: region["ymax"],
        region["xmin"]: region["xmax"],
    ]
    ny, nx = local_mask.shape
    cells: list[GridcellWriter] = []
    for ly in range(ny):
        for lx in range(nx):
            if not local_mask[ly, lx]:
                cells.append(GridcellWriter(
                    region["ymin"] + ly,
                    region["xmin"] + lx,
                    out_dir,
                ))
    print(f"  {len(cells)} land gridcells in region "
          f"(box {ny}×{nx} = {ny*nx} cells, "
          f"{int(local_mask.sum())} masked out)")

    # ---- Soil data: write soil + initialize each cell file ------------
    print(f"\n{_C_BLUE}3/4 Writing soil data{_C_RESET}")
    soil_arrs = {v: read_soil(soil_data, soil_files, v) for v in SOIL_VARS}
    for cell in cells:
        for v in SOIL_VARS:
            value = float(soil_arrs[v][cell.y, cell.x])
            if value < 0:
                raise ValueError(
                    f"Negative soil {v}={value} at (y={cell.y}, x={cell.x})"
                )
            cell.set(v, value)
        cell.write()
    print(f"  wrote soil data for {len(cells)} cells")

    # ---- Climate data: vectorized read, parallel write ----------------
    print(f"\n{_C_BLUE}4/4 Writing climate data{_C_RESET}")
    for var in CLIMATE_VARS:
        station = process_climate_variable(var, raw_data, region, local_mask)
        if station.shape[0] != len(cells):
            raise RuntimeError(
                f"Station count mismatch for {var}: "
                f"{station.shape[0]} vs {len(cells)} cells"
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futs = [
                pool.submit(_write_climate_var_to_cell, cell, var, station[i])
                for i, cell in enumerate(cells)
            ]
            concurrent.futures.wait(futs)
            for f in futs:
                exc = f.exception()
                if exc is not None:
                    raise exc

    print(f"\n{_C_GREEN}Done.{_C_RESET} Outputs at: {out_dir}")


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _test_one(out_dir: Path, raw_data: Path, var: str,
              y: int, x: int, sample: int = 500) -> bool:
    fpath = out_dir / f"input_data_{y}-{x}.pbz2"
    if not fpath.exists():
        print(f"{_C_RED}MISSING{_C_RESET} {fpath}")
        return False
    with bz2.BZ2File(fpath, "r") as fh:
        cell = pkl.load(fh)
    ds = open_clim_dataset(raw_data, var)
    try:
        raw = ds.variables[var][:sample, y, x]
    finally:
        ds.close()
    saved = cell[var][:sample]
    ok = np.allclose(raw, saved)
    err = float(np.mean(np.abs(np.asarray(raw) - np.asarray(saved))))
    tag = f"{_C_CYAN}PASS{_C_RESET}" if ok else f"{_C_RED}FAIL{_C_RESET}"
    print(f"  {tag} {var} cell ({y},{x}) mean|err|={err:.3e}")
    return ok


def run_tests(out_dir: Path, raw_data: Path, n_cells: int = 5,
              sample: int = 500) -> None:
    files = sorted(out_dir.glob("input_data_*-*.pbz2"))
    if not files:
        raise FileNotFoundError(f"No .pbz2 files in {out_dir}")
    rng = np.random.default_rng()
    pick = rng.choice(len(files), size=min(n_cells, len(files)), replace=False)
    coords: list[tuple[int, int]] = []
    for i in pick:
        stem = files[i].stem.split("_")[-1]
        y, x = stem.split("-")
        coords.append((int(y), int(x)))
    print(f"Testing cells: {coords}")
    all_ok = True
    for var in CLIMATE_VARS:
        for y, x in coords:
            if not _test_one(out_dir, raw_data, var, y, x, sample):
                all_ok = False
    print(f"\n{_C_GREEN if all_ok else _C_RED}"
          f"{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}{_C_RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    config = _load_config(Path("./pre_processing.toml"))
    dataset = args.dataset or config.get("dataset")
    mode = args.mode or config.get("mode")
    if not dataset or not mode:
        print(f"{_C_RED}--dataset and --mode required (or set in toml).{_C_RESET}")
        return 2

    mask_path = Path(args.mask_file) if args.mask_file else Path(config["mask_file"])
    if not mask_path.exists():
        print(f"{_C_RED}Mask file not found: {mask_path}{_C_RESET}")
        return 2
    mask = np.load(mask_path)
    if mask.shape != GRID_SHAPE:
        print(f"{_C_RED}Mask shape {mask.shape} != {GRID_SHAPE}{_C_RESET}")
        return 2

    out_dir = Path(f"./{dataset}/{mode}").resolve()

    print(f"{_C_BLUE}CAETÊ pre-processing{_C_RESET}  dataset={dataset!r}  mode={mode!r}")

    if args.test:
        raw_data = Path(config["climate_data"]) / dataset / f"{mode}_raw"
        run_tests(out_dir, raw_data)
        return 0

    run(config, dataset, mode, mask, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
