# Copyright 2017- LabTerra
#
# Licensed under the GNU GPL v3 or later. See model_driver.py for the full
# notice. This file is a stripped-down local-run driver derived from
# model_driver.py.
"""Minimal local driver for CAETÊ-DVM.

Compared to :mod:`model_driver`, this script is intentionally narrowed
down to a single configuration path so it can be used as a clean entry
point for development, smoke tests and small regional runs on a
workstation. Specifically, it drops:

- the interactive prompts (``check_start``, zone selector, run name,
  climatology selector);
- the ``sombrero`` HPC branch (whole-mask iteration);
- the ESM ensemble branch (``GFDL-ESM2M``/``HadGEM2-ES``/...);
- the hardcoded zone bounding boxes (``central``/``south``/...);
- the per-zone ``y0/y1, x0/x1`` indexing;
- selecting one of the three ``rbrk`` lists at runtime — index ``0``
  (historical/observed) is always used.

Instead, the driver discovers whatever ``input_data_{Y}-{X}.pbz2`` cells
are present in :data:`INPUT_PATH` (alongside
``ISIMIP_HISTORICAL_METADATA.pbz2``) and runs the same spinup →
transient → HDF5 → netCDF pipeline as ``model_driver.py``. All
configuration lives at the top of this file as module-level constants.

Multiprocessing uses the ``spawn`` start method (Python's default on
Windows; required to avoid the ``fork()`` deadlock warning on Linux when
the parent is multi-threaded). All heavy initialization lives inside
``main()`` so it runs only in the parent process.

Usage::

    cd src
    python model_driver_local.py            # uses defaults below

Pipeline
--------
1. ``initialize_spinup`` — water + soil-organic pre-spinup
   (``bdg_spinup`` + ``sdc_spinup``).
2. ``run_spinup_phase`` — ``NLOOPS_SPINUP_1`` repetitions of
   :data:`SPINUP_START`-:data:`SPINUP_END` with fixed CO₂ and
   ``nutri_cycle=False``.
3. ``execute_co2_fixed_spinup`` — ``NLOOPS_SPINUP_2`` repetitions with
   fixed CO₂ and the full nutrient cycle on.
4. Transient — iterates ``rbrk[0]`` (built from
   ``build_run_breaks`` in ``caete.py``); one ``starmap`` call per chunk.
5. ``write_h5`` — collapse the per-cell pickles into ``CAETE.h5``.
6. ``h52nc`` — write per-interval netCDFs into :data:`NC_OUTPUTS`.

Notes
-----
- The transient interval list comes from ``rbrk[rbrk_index]`` in
  ``caete.py``. The chunk size is set there (the ``chunk_years``
  argument to ``build_run_breaks``); this script only selects which of
  the three pre-built lists to use via :data:`rbrk_index` (always 0
  here). See ``run_caete`` in ``caete.py`` for the per-chunk semantics.
- ``stime.txt`` is written with four lines (``time_units``, ``calendar``,
  experiment label, ``rbrk_index``) so :func:`h52nc.h52nc` can recover
  the interval list used at runtime.

Status of the original TODOs
----------------------------
e1. [done] Extended CO₂ file to 1765-2024
    (``historical_CO2_annual_1765_2024.txt``). Future work: switch to
    ISIMIP3a CO₂ directly and replace the linear scan in ``find_co2``
    with a dict lookup.
e2. [done] ``h52nc`` now derives netCDF intervals from the HDF5 dates
    via ``build_run_breaks`` and accepts ``time_units``/``calendar``/
    ``experiment`` overrides; it also prefers the runtime ``run_breaks``
    recovered from ``stime.txt`` so daily and snapshot netCDFs stay
    consistent with the runtime chunking.
e3. [done] ``build_run_breaks(start, end, chunk_years=...)`` lives in
    ``caete.py`` and the three ``run_breaks_*`` lists are derived from
    it.
"""

from __future__ import annotations

import bz2
import copy
import multiprocessing as mp
import re
import time
from pathlib import Path

import _pickle as pkl
import joblib
import numpy as np

import plsgen as pls
from caete import grd, npls, print_progress, rbrk


# ---------------------------------------------------------------------------
# Local-run configuration (edit these to point at a different test bundle)
# ---------------------------------------------------------------------------

# Folder containing ISIMIP_HISTORICAL_METADATA.pbz2 and input_data_Y-X.pbz2.
INPUT_PATH = Path("../input/test_new_input").resolve()
# Only the name of the metadata file. The driver looks for it inside :data:`INPUT_PATH`.
DATASET_METADATA_FILE = Path("ISIMIP_HISTORICAL_METADATA.pbz2")

# Name used to namespace the run outputs (../outputs/{RUN_NAME}).
RUN_NAME = "test_new_input"

# CO2 forcing file (annual, columnar text).
CO2_FILE = Path("../input/co2/historical_CO2_annual_1765_2024.txt").resolve()

# Soil/hydraulic maps (as in model_driver.py).
SOIL_DIR = Path("../input/soil").resolve()
HYDRA_DIR = Path("../input/hydra").resolve()

# Output paths.
OUTPUT_PATH = Path("../outputs").resolve()
DUMP_FOLDER = (OUTPUT_PATH / RUN_NAME).resolve()
NC_OUTPUTS = (DUMP_FOLDER / "nc_outputs").resolve()


# Simulation time bounds. The spinup and transient phases
# TODO: these are haardcoded in the ``rbrk`` lists in ``caete.py``; they must be consistent with the
# dates in the driver dataset you are using. Edit the ``build_run_breaks`` function calls in caete.py to change the break
# We need to imporve here.
SIMULATION_START = "19010101"
SIMULATION_END = "20241231"

# Spinup time bounds (yyyymmdd strings).
SPINUP_START = "19010101"
SPINUP_END = "19301231"
FIXED_CO2_SPINUP_YEAR = "1901"

NLOOPS_SPINUP_1 = 3  # N of loops in the range [SPINUP_START, SPINUP_END] (no nutrient cycle)
NLOOPS_SPINUP_2 = 3  # N of loops in the range [SPINUP_START, SPINUP_END] (full cycle)

# The final number of spinup years is
# Initit_years + Phase1 + Phase2
# --- where:
# Init_years = SPINUP_END - SPINUP_START + 1
# Phase1 = NLOOPS_SPINUP_1 * Init_years
# Phase2 = NLOOPS_SPINUP_2 * Init_years
# ---

# Multiprocessing start method. "spawn" is safe on Linux + multi-threaded
# parents; "forkserver" is also fine. Avoid "fork".
MP_START_METHOD = "spawn"


# ---------------------------------------------------------------------------
# Worker functions (must be at module scope so spawn can pickle them).
# They take a `grd` instance and return it; they do NOT use module globals
# beyond the SPINUP_* / FIXED_CO2_SPINUP_YEAR / NLOOPS_SPINUP_* constants,
# which the spawned children re-import as part of this module.
# ---------------------------------------------------------------------------

def initialize_spinup(gridcell: grd) -> grd:
    """Pre-spinup pass: build initial water and soil-organic pools.

    Calls ``grd.bdg_spinup(SPINUP_START, SPINUP_END)`` to estimate water
    pools and approximate C/N/P fluxes from vegetation to soil, then
    feeds those fluxes into ``grd.sdc_spinup`` to populate the initial
    soil C/N/P pools. No daily output is written.

    Returns the mutated ``gridcell`` so it can be moved between
    ``multiprocessing.Pool`` calls.
    """
    w, ll, cwd, rl, lnc = gridcell.bdg_spinup(
        start_date=SPINUP_START, end_date=SPINUP_END)
    gridcell.sdc_spinup(w, ll, cwd, rl, lnc)
    return gridcell


def run_spinup_phase(gridcell: grd) -> grd:
    """Phase-1 main spinup: vegetation only.

    Repeats the ``[SPINUP_START, SPINUP_END]`` window
    ``NLOOPS_SPINUP_1`` times with CO₂ fixed at
    ``FIXED_CO2_SPINUP_YEAR`` and ``nutri_cycle=False`` (soil mineral
    pools held fixed). ``save=False`` so no pickle is emitted; the only
    effect is to evolve the gridcell state in place.
    """
    gridcell.run_caete(SPINUP_START, SPINUP_END, spinup=NLOOPS_SPINUP_1,
                       fix_co2=FIXED_CO2_SPINUP_YEAR, save=False, nutri_cycle=False)
    return gridcell


def execute_co2_fixed_spinup(gridcell: grd) -> grd:
    """Phase-2 main spinup: full biogeochemistry, fixed CO₂.

    Same window as :func:`run_spinup_phase` but with the nutrient cycle
    switched on, repeated ``NLOOPS_SPINUP_2`` times. ``save=False``.
    Produces the post-spinup gridcell state that the transient phase
    starts from.
    """
    gridcell.run_caete(SPINUP_START, SPINUP_END, spinup=NLOOPS_SPINUP_2,
                       fix_co2=FIXED_CO2_SPINUP_YEAR, save=False)
    return gridcell


def execute_simulation(gridcell: grd, brk: list) -> grd:
    """Run one transient chunk on a gridcell.

    Parameters
    ----------
    gridcell : grd
        Gridcell already brought through the three spinup phases.
    brk : tuple[str, str]
        ``('YYYYMMDD', 'YYYYMMDD')`` chunk bounds, taken from
        ``rbrk[rbrk_index]``. CO₂ follows the file (transient), the
        nutrient cycle is on, and ``save=True`` so each chunk emits a
        ``spinNN.pkz`` pickle in ``grid.out_dir``.
    """
    gridcell.run_caete(brk[0], brk[1])
    return gridcell


def zip_gridtime(grd_pool, interval):
    """Pair every gridcell with the (single) interval to be run next.

    ``Pool.starmap`` expects an iterable of ``(grid, brk)`` tuples; this
    helper recycles ``interval`` for every grid in ``grd_pool``. The
    ``i % len(interval)`` form is kept for symmetry with
    ``model_driver.py``, even though the transient loop here always
    passes ``interval`` as a 1-tuple.
    """
    return [(g, interval[i % len(interval)]) for i, g in enumerate(grd_pool)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CELL_RE = re.compile(r"input_data_(\d+)-(\d+)\.pbz2$")


def _discover_cells(folder: Path) -> list[tuple[int, int]]:
    """List every ``input_data_Y-X.pbz2`` file in ``folder``.

    Returns a sorted list of ``(y, x)`` 0.5°-grid indices to instantiate
    :class:`grd` objects for. Raises :class:`FileNotFoundError` when no
    matching files are present so the driver fails fast instead of
    silently running on an empty pool.
    """
    cells: list[tuple[int, int]] = []
    for f in sorted(folder.glob("input_data_*-*.pbz2")):
        m = _CELL_RE.match(f.name)
        if m:
            y, x = int(m.group(1)), int(m.group(2))
            cells.append((y, x))
    if not cells:
        raise FileNotFoundError(f"No input_data_*-*.pbz2 files in {folder}")
    return cells


def _load_static_inputs():
    """Read every file the parent process needs to instantiate gridcells.

    Returns
    -------
    tuple
        ``(tsoil, ssoil, hsoil, stime, co2_data)`` where:

        - ``tsoil`` / ``ssoil`` are 3-tuples of top-/sub-soil
          ``(ws, fc, wp)`` arrays.
        - ``hsoil`` is a 3-tuple of hydraulic arrays
          ``(theta_sat, psi_sat, soil_text)``.
        - ``stime`` is the deserialised time-axis dict from
          ``ISIMIP_HISTORICAL_METADATA.pbz2`` (units, calendar,
          ``time_index``).
        - ``co2_data`` is the raw line list from :data:`CO2_FILE` (one
          ``year<sep>ppm`` row per line) handed verbatim to
          ``grd.init_caete_dyn``.

    Raises :class:`FileNotFoundError` early if any of the expected files
    are missing so workers do not crash later.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_PATH}")
    clim_metadata_file = INPUT_PATH / DATASET_METADATA_FILE
    if not clim_metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {clim_metadata_file}")
    if not CO2_FILE.exists():
        raise FileNotFoundError(f"CO2 file not found: {CO2_FILE}")

    # Topsoil
    map_ws = np.load(SOIL_DIR / "ws.npy")
    map_fc = np.load(SOIL_DIR / "fc.npy")
    map_wp = np.load(SOIL_DIR / "wp.npy")
    # Subsoil
    map_subws = np.load(SOIL_DIR / "sws.npy")
    map_subfc = np.load(SOIL_DIR / "sfc.npy")
    map_subwp = np.load(SOIL_DIR / "swp.npy")

    tsoil = (map_ws, map_fc, map_wp)
    ssoil = (map_subws, map_subfc, map_subwp)

    hsoil = (
        np.load(HYDRA_DIR / "theta_sat.npy"),
        np.load(HYDRA_DIR / "psi_sat.npy"),
        np.load(HYDRA_DIR / "soil_text.npy"),
    )

    with bz2.BZ2File(clim_metadata_file, mode="r") as fh:
        clim_metadata = pkl.load(fh)
    stime = copy.deepcopy(clim_metadata[0])
    del clim_metadata

    with open(CO2_FILE) as fh:
        co2_data = fh.readlines()

    return tsoil, ssoil, hsoil, stime, co2_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full local pipeline: spinup → transient → HDF5 → netCDF.

    Side effects
    ------------
    - Writes ``stime.txt`` (4 lines: ``units``, ``calendar``, experiment
      label, ``rbrk_index``) consumed by :func:`h52nc.h52nc`.
    - Saves ``pls_attrs-{npls}.csv`` and the per-cell ``spinNN.pkz``
      pickles under :data:`DUMP_FOLDER`.
    - Saves the post-spinup and post-transient gridcell-list snapshots
      as ``CAETE_STATE_START_*.pkz`` / ``CAETE_STATE_END_*.pkz`` for
      reproducibility.
    - Builds ``CAETE.h5`` from the per-cell pickles via
      :func:`post_processing.write_h5`.
    - Emits one netCDF per variable per interval into
      :data:`NC_OUTPUTS` via :func:`h52nc.h52nc`.

    All multiprocessing pools use ``mp.cpu_count() - 1`` workers and the
    ``spawn`` start method (see :data:`MP_START_METHOD`).
    """
    from post_processing import write_h5
    from h52nc import h52nc 

    OUTPUT_PATH.mkdir(exist_ok=True)
    DUMP_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"Input folder : {INPUT_PATH}")
    print(f"Output folder: {DUMP_FOLDER}")

    tsoil, ssoil, hsoil, stime, co2_data = _load_static_inputs()

    # In caete.py, the ``rbrk`` lists are defined at the module level and must match
    # the dates in the driver dataset that you are using. 
    # Edit the ``build_run_breaks`` function calls in caete.py to change the break
    # intervals to match the time span of your driver dataset.

    rbrk_index = 0 # Index of the current break interval, used for naming outputs and logging.
    run_breaks = rbrk[rbrk_index] # Select historical breaks from caete.py
    # TODO: Make runbreaks local to the driver. We should remove a lot of things
    # from caete.py and move them here, including the run_breaks construction and the h52nc time axis handling.
    # The driver should be self-contained and not rely on global variables in caete.py.

    # TODO: remove this gambiarra. The file stime.txt is only needed for h52nc to recover the time axis 
    # and run breaks used at runtime. We should pass that info donwsteam via the .h5 file.
    with open("stime.txt", "w") as fh:
        fh.writelines([
            f"{stime['units']}\n",
            f"{stime['calendar']}\n",
            f"{RUN_NAME}-LOCAL\n",
            f"{rbrk_index}\n",
        ])

    pls_table = pls.table_gen(npls, DUMP_FOLDER)

    coords = _discover_cells(INPUT_PATH)
    print(f"Found {len(coords)} gridcell(s): {coords}")

    grid_mn: list[grd] = [grd(x, y, RUN_NAME) for (y, x) in coords]

    # TODO: the creation of gridcells can be done in parallel and dynamically 
    # initialized at runtime. We could process batches of gridcells and pickle
    # them to disk to decrease memory foorprint at runtime.    
    print("Starting gridcells")
    print_progress(0, len(grid_mn), prefix="Progress:", suffix="Complete")
    for i, g in enumerate(grid_mn):
        g.init_caete_dyn(INPUT_PATH, stime, co2_data,
                         pls_table, tsoil, ssoil, hsoil)
        print_progress(i + 1, len(grid_mn),
                       prefix="Progress:", suffix="Complete")

    # Free big arrays before forking pools.
    del pls_table
    del co2_data
    del stime
    del tsoil, ssoil, hsoil

    ctx = mp.get_context(MP_START_METHOD)
    n_proc = max(1, mp.cpu_count() - 1)

    log = open("logfile.log", mode="w")
    print("START:", time.ctime())
    log.write(time.ctime() + "\n\n")
    log.write("SOIL SPINUP...\n")
    start = time.time()
    print("SOIL SPINUP...")

    with ctx.Pool(processes=n_proc) as p:
        spun = p.map(initialize_spinup, grid_mn)
    log.write(f"END_OF_SPINUP after (s){time.time() - start}\n")
    del grid_mn

    print("MAIN SPINUP - phase 1 (no nutrient cycle)")
    with ctx.Pool(processes=n_proc) as p:
        result = p.map(run_spinup_phase, spun)
    del spun

    print("MAIN SPINUP - phase 2 (full cycle)")
    with ctx.Pool(processes=n_proc) as p:
        result = p.map(execute_co2_fixed_spinup, result)

    g0_path = (DUMP_FOLDER / f"CAETE_STATE_START_{RUN_NAME}_.pkz").resolve()
    with open(g0_path, "wb") as fh2:
        print(f"Saving post-spinup state: {g0_path}")
        joblib.dump(result, fh2, compress=("zlib", 1), protocol=4)

    for i, brk in enumerate(run_breaks):
        print(f"Transient interval {brk[0]}-{brk[1]}")
        zipped = zip_gridtime(result, (brk,))
        with ctx.Pool(processes=n_proc) as p:
            result = p.starmap(execute_simulation, zipped)

    g1_path = (DUMP_FOLDER / f"CAETE_STATE_END_{RUN_NAME}_.pkz").resolve()
    with open(g1_path, "wb") as fh2:
        print(f"Saving final state: {g1_path}")
        joblib.dump(result, fh2, compress=("zlib", 1), protocol=4)

    log.close()

    print("\nEND OF MODEL EXECUTION", time.ctime())
    print("Writing HDF5 ...")
    write_h5(DUMP_FOLDER)
    print("Writing netCDF4 files ...")
    NC_OUTPUTS.mkdir(parents=True, exist_ok=True)
    h5path = (DUMP_FOLDER / "CAETE.h5").resolve()
    h52nc(h5path, NC_OUTPUTS)
    print(time.ctime())


if __name__ == "__main__":
    try:
        mp.set_start_method(MP_START_METHOD, force=True)
    except RuntimeError:
        raise RuntimeError(f"Failed to set multiprocessing start method to {MP_START_METHOD}")
    main()
