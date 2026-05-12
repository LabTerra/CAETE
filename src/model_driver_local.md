# `model_driver_local.py` — design notes

[src/model_driver_local.py](src/model_driver_local.py) is a stripped-down
sibling of [src/model_driver.py](src/model_driver.py). It exists so a
developer can run CAETÊ-DVM end-to-end on a workstation with one command
and a small input bundle, without touching the HPC- and ESM-aware
machinery in the original driver. This document records what the local
driver does, what it removed compared to the original, and the status of
the three rebuild TODOs that motivated it.

The post-processing review at the end of the document is unrelated to
the local driver itself and is treated separately.

---

## End-to-end picture: from new inputs to new outputs

The work on this branch (`jd_new_inputs`) is best read as one change
that propagates through the full CAETÊ pipeline rather than a set of
independent fixes in the Input/Output and update of driver datasets.
The trigger is the new climate forcing: 
ISIMIP3a `20CRv3-ERA5_ISIMIP3a / obsclim` daily 1901–2024,
together with a refreshed soil dataset. Because every downstream stage embeds
assumptions about *which years are available* and *how they are
chunked*, replacing the inputs forces a coordinated change in the
driver, in `caete.py`, and in `h52nc.py`. The chain is:

1. **Inputs** ([input/pre_processing.py](input/pre_processing.py),
   [input/pre_processing.toml](input/pre_processing.toml),
   [input/geos.py](input/geos.py),
   [input/creating_caete_input_files.md](input/creating_caete_input_files.md))
   produce one bz2-pickle per gridcell + a single metadata file. The
   on-disk layout is unchanged from the legacy producer, but the
   pipeline is now vectorized, TOML-configured, and validated by a
   small `--test` where the newly created files are checked for correctness.
2. **CO₂ forcing**
   ([input/co2/historical_CO2_annual_1765_2024.txt](input/co2/historical_CO2_annual_1765_2024.txt))
   was extended from 2018 → 2024 to cover the new climate span, and
   `find_co2` in [src/caete.py](src/caete.py) was hardened to tolerate
   the whitespace-separated last row.
3. **Transient chunking** in [src/caete.py](src/caete.py) is now built
   programmatically by `build_run_breaks(start, end, chunk_years=...)`,
   so extending the historical schedule to 1901–2024 is a one-line
   change rather than a hand-edited tuple list.
4. **Post-processing** in [src/h52nc.py](src/h52nc.py) consumes that
   same runtime `run_breaks` instead of stale module-level globals,
   so the daily netCDFs use exactly the cadence the simulation ran
   with — even when the inputs span more years than the legacy
   1979–2016 default.
5. **Driver** ([src/model_driver_local.py](src/model_driver_local.py))
   ties it all together with `mp.get_context("spawn")`. The change in the 
   multiprocessing start method is the single biggest difference between the
   two drivers. The "fork" method, previously used by `model_driver.py` and the
   default in linux is now deprecated and emits warnings on Python 3.12+. 
   The spawn method is the default in macOS and the only method available in windows.


In other words, the input rebuild and the post-processing rebuild are
two halves of the same change: the new forcing dictates the time
span, and `build_run_breaks` + the keyword-driven `h52nc` make every
downstream stage track that span automatically. The local driver is
the smallest end-to-end harness that exercises the whole chain on a
workstation. Below I describe the differences between the two drivers.

---

## How it differs from `model_driver.py`

Both scripts share the same conceptual pipeline (pre-spinup →
phase-1/phase-2 spinup → transient chunks → `write_h5` → `h52nc`) and
the same `grd` / `run_caete` API. The differences are deliberate
simplifications, not behavioural changes inside a simulation.

### Removed / simplified

| Area | `model_driver.py` | `model_driver_local.py` |
|------|-------------------|--------------------------|
| Interactive prompts | `check_start()`, run-name prompt, zone prompt, climatology menu (5 ESMs). | None. Configuration is module-level constants. |
| HPC branch | `sombrero` flag selects whole-mask iteration over `(360, 720)` and the `/home/amazonfaceme/shared_data` tree. | No `sombrero`. Always reads from a single local folder. |
| ESM ensemble | `HISTORICAL-RUN`, `GFDL-ESM2M`, `HadGEM2-ES`, `IPSL-CM5A-LR`, `MIROC5` branches with per-model `<outf>-historical_METADATA.pbz2` / `co2-<outf>-historical.txt`. | One forcing path: ISIMIP3a-style folder with `ISIMIP_HISTORICAL_METADATA.pbz2` and `historical_CO2_annual_1765_2024.txt`. |
| Region selection | Four hand-coded zones (`central`, `south`, `east`, `north_west`) each with `y0/y1, x0/x1` slabs and a matching folder. | No zones. The driver `glob`s every `input_data_Y-X.pbz2` in `INPUT_PATH`. |
| `rbrk` index choice | Branches set `rbrk_index = 0` or `1` depending on climatology vs ESM-historical. | Hard-wired `rbrk_index = 0` (the historical/observed list). |
| Transient PLS table | Reads `pls_attrs.csv` from a "base run" folder for ESM-historical experiments. | Always generates a fresh PLS table via `pls.table_gen` for the local run. |
| Top-level state at import | Module-scope side effects (file loads, prompts, pool setup). | Side-effect-free at import; everything happens inside `main()`. |
| Multiprocessing | `mp.Pool(...)` (default fork on Linux). | Explicit `mp.get_context("spawn")` to (a) match Windows, (b) avoid the fork-after-threads warning on Linux. |

### Kept (semantics unchanged)

- The three spinup phases call exactly the same `bdg_spinup` /
  `sdc_spinup` / `run_caete` methods of `grd`. Spinup repetition counts
  are exposed as constants (`NLOOPS_SPINUP_1`, `NLOOPS_SPINUP_2`)
  instead of being buried in the function calls.
- Transient chunks come from `rbrk[0]` in
  [src/caete.py](src/caete.py) — the same source `model_driver.py` uses.
  Chunk size is controlled there via `build_run_breaks(..., chunk_years=...)`,
  not in this driver.
- `stime.txt` is written with the same four-line layout
  (`units / calendar / experiment / rbrk_index`) so `h52nc.catch_stime`
  works without modification.
- Output layout under `../outputs/<RUN_NAME>/` (per-cell `gridcell*/`
  pickle directories, `CAETE.h5`, `nc_outputs/`) is identical.

### Module-level configuration

All run-time choices are constants near the top of the script:

- `INPUT_PATH`, `RUN_NAME`, `CO2_FILE`, `SOIL_DIR`, `HYDRA_DIR`,
  `OUTPUT_PATH`, `DUMP_FOLDER`, `NC_OUTPUTS`.
- `SPINUP_START`, `SPINUP_END`, `FIXED_CO2_SPINUP_YEAR`.
- `NLOOPS_SPINUP_1`, `NLOOPS_SPINUP_2`.
- `MP_START_METHOD = "spawn"`.

The total number of spinup years is therefore
`(SPINUP_END − SPINUP_START + 1) · (1 + NLOOPS_SPINUP_1 + NLOOPS_SPINUP_2)`,
matching the breakdown given in the original `model_driver.py`'s
`README`.

### Worker layout

Worker functions (`initialize_spinup`, `run_spinup_phase`,
`execute_co2_fixed_spinup`, `execute_simulation`) are at module scope so
`spawn` can pickle them. They take a `grd` and return a `grd`; they
read only the `SPINUP_*` / `NLOOPS_*` / `FIXED_CO2_SPINUP_YEAR` module
constants, which the spawned children re-import as part of this module.

---

## Multiprocessing start method: `fork` → `spawn`

The single most consequential behavioural difference between
[src/model_driver.py](src/model_driver.py) and
[src/model_driver_local.py](src/model_driver_local.py) is the
multiprocessing start method. The original driver relies on the platform
default (which on Linux has historically been `fork`); the local driver
hard-codes `spawn` via:

```python
MP_START_METHOD = "spawn"
...
ctx = mp.get_context(MP_START_METHOD)
with ctx.Pool(processes=NPROCS) as pool:
    ...
if __name__ == "__main__":
    mp.set_start_method(MP_START_METHOD, force=True)
    ...
```

This subsection records *why* the change is necessary, not just
convenient.

### Why `fork` is being retired

From the CPython
[`multiprocessing` — Contexts and start methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
documentation:

- **Python 3.12** — "If Python is able to detect that your process has
  multiple threads, the `os.fork()` function that this start method
  calls internally will raise a `DeprecationWarning`. Use a different
  start method." (The note explicitly tells users to migrate.)
- **Python 3.14** — "On POSIX platforms the default start method was
  changed from `fork` to `forkserver` to retain the performance but
  avoid common multithreaded process incompatibilities. See
  [gh-84559](https://github.com/python/cpython/issues/84559)." `fork`
  is "no longer the default start method on any platform. Code that
  requires `fork` must explicitly specify that via `get_context()` or
  `set_start_method()`."
- **macOS since 3.8** — `spawn` is already the default; `fork` is
  documented as "unsafe as it can lead to crashes of the subprocess as
  macOS system libraries may start threads"
  ([bpo-33725](https://bugs.python.org/issue?@action=redirect&bpo=33725)).
- **Windows** — only `spawn` is available. Code that worked under fork
  on Linux but assumed inherited globals breaks the moment a Windows
  user runs it.

The root problem is that `fork()` duplicates the parent's address space
*including all of its threads' state*, but only the calling thread
survives in the child. Any mutex, condition variable, malloc arena or
internal queue held by another thread at the moment of `fork()` is
inherited in an indeterminate state and can deadlock or corrupt memory
the first time the child touches it. CAETÊ-DVM is exactly the kind of
program that triggers this: it links a Fortran extension
([src/budget.f90](src/budget.f90), [src/productivity.f90](src/productivity.f90),
etc.) compiled via f2py, plus NumPy/HDF5/netCDF4 stacks that may spawn
worker threads (OpenMP, BLAS, HDF5 I/O threads). Forking after any of
those have started is unsafe by construction.

### Why `spawn` for the local driver

Three reasons, in order of priority:

1. **Correctness with native extensions.** `spawn` starts a fresh
   Python interpreter with no inherited mutexes, no inherited Fortran
   COMMON blocks, and no inherited HDF5 file handles. Each worker
   re-imports `model_driver_local` and re-initialises its Fortran
   state cleanly.
2. **Cross-platform reproducibility.** `spawn` is the only method
   available on Windows and the default on macOS. Writing the local
   driver against `spawn` means a developer on any of the three major
   OSes runs the *same* code path. Fork-only assumptions (e.g.
   "children see the parent's already-loaded `global1.pkl`") never get
   baked in.
3. **Forward compatibility.** From Python 3.14 onward `fork` is no
   longer the default even on Linux. Any code that silently relied on
   it will start emitting `DeprecationWarning` (3.12) and then
   functionally change behaviour (3.14+). Pinning `spawn` now insulates
   the driver from that transition.

`forkserver` would also be a defensible choice on Linux (it is the new
3.14 default), but it is POSIX-only, so it would re-introduce the
Linux/Windows divergence that this driver tries to remove.

### Implications for code that runs under `spawn`

Per the multiprocessing docs' "The spawn and forkserver start methods"
section, switching from `fork` is not free. The local driver pays for
it deliberately:

- **Picklable arguments and targets.** All worker functions
  (`initialize_spinup`, `run_spinup_phase`, `execute_co2_fixed_spinup`,
  `execute_simulation`, `zip_gridtime`) live at module scope so they
  can be located by name when the child re-imports the module. The
  `grd` instances passed to them must be picklable as well — this is
  why the driver avoids stuffing open file handles or live HDF5
  contexts into `grd` before dispatching to the pool.
- **Safe `if __name__ == "__main__":` guard.** Every spawned child
  re-imports `model_driver_local` from the top, so the module must be
  importable without side effects. The driver enforces this: no
  prompts, no file loads, no pool creation at import time. All of that
  lives inside `main()` behind the `__main__` guard.
- **Module-level constants are re-evaluated, not inherited.** A child
  process does not see runtime mutations of globals in the parent. The
  driver therefore exposes the data workers need (`SPINUP_START`,
  `SPINUP_END`, `FIXED_CO2_SPINUP_YEAR`, `NLOOPS_SPINUP_*`,
  `CO2_FILE`, etc.) as plain module-level constants that are identical
  in parent and children by construction.
- **Slower worker startup.** Each child re-imports the world (NumPy,
  netCDF4, the f2py extension). For CAETÊ this overhead is negligible
  compared to a multi-decade transient run, but it is the reason
  workers should be reused (`with ctx.Pool(...) as pool: pool.starmap(...)`)
  rather than re-spawned per chunk.
- **No accidental sharing of file descriptors.** Open HDF5 files,
  netCDF datasets, and pickle handles in the parent are *not*
  inherited. Anything the worker needs on disk it must open itself.

### Recommended follow-ups

- Once the local driver is validated end-to-end on Linux + Windows,
  apply the same `mp.get_context("spawn")` change to
  [src/model_driver.py](src/model_driver.py) so the production driver
  no longer depends on the deprecated default.
- Audit any remaining call sites that use the bare `multiprocessing`
  API (e.g. `mp.Pool(...)` without an explicit context) — they
  implicitly set the global start method on first use and will start
  emitting `DeprecationWarning` on Python ≥ 3.12 if the platform
  default is still `fork`.

---

## Input pipeline rebuild

The `input/` folder was reorganised so that producing a CAETÊ input
bundle is a vectorized, configuration-driven step rather than a
hand-edited script. The on-disk format consumed by
[src/caete.py](src/caete.py) is unchanged — same filenames, same dict
keys, same metadata layout — so existing readers keep working without
modification. What changed is everything *upstream* of those files.

### New `pre_processing.py` (vectorized, TOML-driven)

[input/pre_processing.py](input/pre_processing.py) is a full rewrite of
the legacy producer. Highlights:

- **Single-pass, vectorized I/O.** Climate variables are read as a
  `(time, ny, nx)` cube with `netCDF4.MFDataset` + `MFTime`, then
  reduced to per-station rows via fancy indexing
  (`cube[:, valid].T`). Multi-file ISIMIP NetCDFs are stitched
  transparently. Masked entries are filled with the variable's
  regional mean and reported.
- **Two small classes** mirror the legacy producer's on-disk format:
  `DSMetadata` writes the 3-tuple `(time_dict, lat_dict, lon_dict)`
  metadata file; `GridcellWriter` writes one bz2-pickle per cell with
  the canonical 10-key payload (`hurs`, `tas`, `pr`, `ps`, `rsds`,
  `tn`, `tp`, `ap`, `ip`, `op`).
- **CLI flags** override the TOML defaults for ad-hoc runs:
  `--dataset`, `--mode`, `--mask-file`, and `--test`. The `--test`
  mode re-opens a random sample of previously written `.pbz2` files
  and validates them against the raw NetCDF — a small but real
  smoke-test for the writer.
- **Friendlier console output**: ANSI-coloured progress and warnings
  so a long preprocessing run is legible without a logging framework.
- **Filename / format compatibility** is explicit: per-cell files are
  named `input_data_{Y}-{X}.pbz2` with **global** indices on the
  360 × 720 grid; the metadata file is `ISIMIP_HISTORICAL_METADATA.pbz2`.
  Both names match the legacy producer so model-side readers are
  untouched.

### Configuration via `pre_processing.toml`

[input/pre_processing.toml](input/pre_processing.toml) replaces the
hard-coded paths and constants of the legacy script with a single
config file:

- `climate_data` — root folder containing the dataset trees.
- `dataset` — e.g. `"20CRv3-ERA5_ISIMIP3a"`; selects the subfolder.
- `mode` — e.g. `"obsclim"`; selects the experiment.
- `yres`, `xres` — grid resolution (0.5°), so the same script can
  later target other resolutions without code edits.
- `mask_file` — path to the land mask (`.npy`,
  `(360, 720)` boolean).
- `soil_data` — folder of soil `.npy` arrays.
- `[soil_files]` — table mapping the five soil keys to filenames
  (`tn → total_n.npy`, `tp → total_p.npy`, `ap → avail_p.npy`,
  `ip → inorg_p.npy`, `op → org_p.npy`).

This is the only file a developer needs to edit to retarget the
producer at a different dataset, mask, or soil database.

### Geospatial helpers extracted to `geos.py`

[input/geos.py](input/geos.py) collects the grid math that used to be
scattered across the legacy producer:

- `YRES`, `XRES` — 0.5° grid constants; `(0, 0)` is the **northwest**
  corner.
- `find_indices_xy(N, W)` and `find_coordinates_xy(y, x)` — the
  inverse pair, used both during preprocessing and inside
  `caete.grd` for round-tripping coordinates.
- `define_region(north, south, west, east)` — returns the
  `{ymin, ymax, xmin, xmax}` slab that crops the global cube to a
  bounding box.
- `pan_amazon_region` (north=10.5, south=-21.5, west=-80.0,
  east=-43.0) and `global_region` — the two regions currently
  supported.

Keeping these in a single, dependency-light module means the model
and the preprocessor agree on indexing by construction.

### Format spec `creating_caete_input_files.md`

[input/creating_caete_input_files.md](input/creating_caete_input_files.md)
is the contract every producer (legacy or new) must honour. It
documents:

- The bz2 + pickle container layout for the metadata file and the
  per-cell files.
- The metadata 3-tuple `(time_dict, lat_dict, lon_dict)` and the
  required keys inside each dict.
- The 10-key per-cell payload and a units table
  (K → °C, kg m⁻² s⁻¹ → mm day⁻¹, Pa → hPa, W m⁻² → mol photons
  m⁻² day⁻¹, % → fraction; soil pools in g m⁻²).
- Consistency requirements (matching `time_index` lengths across
  variables, matching lat/lon indices across cells, soil scalars per
  cell).
- A minimal reference writer so the format can be reproduced from
  another pipeline.

### Refreshed soil dataset

[input/soil/](input/soil/) now ships five `(360, 720)` `.npy` arrays in
g m⁻²:

- `total_n.npy` — total nitrogen, regridded from **SoilGrids 2.0**
  (Poggio et al., 2021, *SOIL* 7:217–240,
  [doi:10.5194/soil-7-217-2021](https://doi.org/10.5194/soil-7-217-2021)).
- `total_p.npy`, `avail_p.npy`, `inorg_p.npy`, `org_p.npy` — total,
  available, inorganic and organic phosphorus area densities, taken
  from the **Pan-Amazon reference maps of soil P** of Darela-Filho
  et al. (2024, *ESSD* 16:715–729,
  [doi:10.5194/essd-16-715-2024](https://doi.org/10.5194/essd-16-715-2024)).

The legacy water-balance fields (`fc.npy`, `wp.npy`, `ws.npy`,
`sfc.npy`, `swp.npy`, `sws.npy`) were derived from the **Regridded
Harmonized World Soil Database v1.2** (Wieder et al., 2014, ORNL DAAC,
[doi:10.3334/ORNLDAAC/1247](http://dx.doi.org/10.3334/ORNLDAAC/1247))
and are kept on disk for compatibility with the existing hydrology
code; this branch did not regenerate them.

Each `.npy` is paired with the source NetCDF it was derived from
(`*_area_density_g-per-m2.nc` for the P pools,
`total_n_SoilGrids_g_per_m2.nc` for nitrogen) so the provenance is
checked into the repository alongside the model-ready arrays. The
`[soil_files]` table in `pre_processing.toml` is the single place that
maps these files to the model's `tn / tp / ap / ip / op` keys, and
[input/README.md](input/README.md) lists the full citations.

For completeness, the climate forcing that pairs with these soil
inputs is the **ISIMIP3a atmospheric climate input data v1.3**
(Lange et al., 2025, ISIMIP Repository,
[doi:10.48364/ISIMIP.982724.3](https://doi.org/10.48364/ISIMIP.982724.3)),
downloaded from the [ISIMIP repository](https://data.isimip.org/) and
ingested through `pre_processing.py` under the `dataset =
"20CRv3-ERA5_ISIMIP3a"`, `mode = "obsclim"` configuration.

### Test bundle `input/test_new_input/`

[input/test_new_input/](input/test_new_input/) contains 15
non-contiguous Pan-Amazon cells (e.g. `168-245`, `169-230`, `175-220`,
`185-236`, `191-265`, `195-229`, …) plus a matching
`ISIMIP_HISTORICAL_METADATA.pbz2`. It exists so the local driver can
be exercised end-to-end on a tiny dataset — spinup, transient,
`write_h5`, `h52nc` — without requiring the full ~10⁴-cell forcing.
The `INPUT_PATH` constant in
[src/model_driver_local.py](src/model_driver_local.py) points at this
folder by default.

### Suggested follow-ups

1. **Region selection from config.** `pre_processing.py` currently
   hard-codes `pan_amazon_region`; lifting it into the TOML
   (`region = "pan_amazon" | "global" | {north, south, west, east}`)
   would let the same script produce input bundles for arbitrary
   bounding boxes.
2. **Dataset registry.** Expand `dataset = "<name>"` so it resolves
   via a small registry to a known set of NetCDF filename patterns
   and units, instead of relying on the `*_{var}_*.nc` glob.
3. **Stronger validation.** Extend `--test` from a random sample to
   a deterministic sweep (every cell).
4. **Single source of truth for paths.** Improve the path management
    so in the end we have a more user friendly configuration of the 
    model. 

---

## Changes from `main`

The three subsections below describe the changes this branch
(`jd_new_inputs`) introduces relative to `main`, in the order they were
implemented. Each subsection ends with **Suggested
follow-ups** listing the work that is intentionally still open.

## Extended CO₂ forcing (1765–2024) and a whitespace-tolerant parser

Previously on `main`, the historical CO₂ forcing distributed with the
repository was
[input/co2/historical_CO2_annual_1765_2018.txt](input/co2/historical_CO2_annual_1765_2018.txt)
(254 lines, annual, tab-separated `YYYY<TAB>ppm`, last row
`2018	407.38`), and its path was hard-coded in three places
([src/model_driver.py](src/model_driver.py#L196),
[src/cax_experiment.py](src/cax_experiment.py#L41), and the local
driver). The new ISIMIP3a `obsclim` inputs cover **1901–2024**, so 
it becomes necessary to extend the CO₂ forcing to match that span. The new
file is [input/co2/historical_CO2_annual_1765_2024.txt](input/co2/historical_CO2_annual_1765_2024.txt), which is the same format as the old one but with additional rows (2019–2025) and a trailing newline that makes the last row whitespace-separated rather than tab-separated. There is a .csv file with the the annual CO₂ from the ISIMIP3a dataset.

### What changed on this branch

- **New file**:
  [input/co2/historical_CO2_annual_1765_2024.txt](input/co2/historical_CO2_annual_1765_2024.txt)
  (260 annual rows, tab-separated, plus an appended `2025` row).
- **Local driver wired to it** via the module-level constant
  `CO2_FILE = Path("../input/co2/historical_CO2_annual_1765_2024.txt")`
  in [src/model_driver_local.py](src/model_driver_local.py). The
  hard-coded path used by `model_driver.py` and `cax_experiment.py` is
  unchanged on this branch.
- **`find_co2` hardened** in [src/caete.py](src/caete.py) (both call
  sites near lines 782 and 1225): the parser now uses `str.split()`
  with no argument, which tolerates either tabs or arbitrary runs of
  spaces, and skips non-numeric rows.

### Format constraints that still hold

- One row per year, no header, ASCII; first column a contiguous
  4-digit year, second column parsable by `float()`.
- `find_co2` still does a linear scan and returns the first matching
  year, so years must remain monotonic and gap-free.
- The `,`-separator path used by `self.plot is True` is unchanged.

### Suggested follow-ups

1. **Centralise the path.** Replace the three hard-coded literals with
   a single `CO2_FILE` constant in `caete.py` (or a small
   `forcings.py`) so a future re-extension only edits one line.
2. **Replace the linear scan with a dict lookup.** Build
   `dict[int, float]` once at `init_caete_dyn` time and raise an
   explicit `KeyError` past the last known year instead of returning
   `None`. Optionally clamp-and-warn for years past the end.

---

## Data-driven netCDF intervals in `h52nc`

Previously on `main`, [src/h52nc.py](src/h52nc.py) populated the
module-level globals `TIME_UNITS`, `CALENDAR`, `EXPERIMENT` and
`run_breaks` via `catch_stime("stime.txt")` **at import time**, and
`h52nc(input_file, dump_nc_folder)` looped over those globals:

```python
for interval in run_breaks:
    create_ncG1(t1d, interval, dump_nc_folder)
    create_ncG2(t2d, interval, dump_nc_folder)
    create_ncG3(t3d, interval, dump_nc_folder)
```

`run_breaks` was always one of the three hard-coded lists in
[src/caete.py](src/caete.py) (`run_breaks_hist`,
`run_breaks_CMIP5_hist`, `run_breaks_CMIP5_proj`), selected by the
`rbrk_index` written on line 4 of `stime.txt`. Any mismatch between the
runtime chunking and those literals silently produced netCDFs with the
wrong cadence (e.g. 10-year transient chunks getting sliced into the
default 2-year `rbrk` buckets).

### What changed on this branch

- **`h52nc.h52nc` is now keyword-driven**:

  ```python
  def h52nc(input_file, dump_nc_folder, *,
            chunk_years=2,
            time_units=None,
            calendar=None,
            experiment=None,
            stime_file="stime.txt",
            intervals=None):
  ```

  When `time_units` / `calendar` / `experiment` / `intervals` are not
  supplied, the function falls back to `catch_stime(stime_file)`
  (which is now called *inside* `h52nc()`, not as an import-time side
  effect).
- **Interval source priority** — when `intervals` is not passed
  explicitly, `h52nc` now prefers the runtime `run_breaks` recovered
  from `stime.txt` over data-derived intervals. This guarantees that
  the daily netCDFs use the *same* chunking the simulation actually
  ran with. As a last resort (no `stime.txt`, no `intervals`), it
  derives them from the HDF5 date column with
  `build_run_breaks(d0, d1, chunk_years=chunk_years)`.
- **Snapshot writers** (`lim_data`, `ustrat_data`, `ccc`,
  `create_nc_area`) still use `run_breaks`, but that `run_breaks` is
  now the runtime one read from `stime.txt`, so the snapshot and
  daily-flux netCDFs agree by construction.

### What is *not* yet changed

- `TIME_UNITS`, `CALENDAR`, `EXPERIMENT` are still readable from
  `stime.txt` as a backwards-compatible default. They are no longer
  populated as a side effect of importing `h52nc`, but `stime.txt`
  itself is still written by every driver.
- The hard-coded lat/lon crop slices `[160:221, 201:272]` (see
  around [src/h52nc.py#L88](src/h52nc.py#L88)) are untouched.
- `custom_rbrk(tp)` (the manual escape hatch around
  [src/h52nc.py#L66-L68](src/h52nc.py#L66)) is preserved.

### Suggested follow-ups

1. **Store `time_units`, `calendar`, `experiment` as HDF5 root
   attributes** during `post_processing.write_h5`, so `h52nc` can be
   fully self-contained and `stime.txt` can be deprecated.
2. **Drop `stime.txt` entirely** once (1) is in place and
   `rbrk_index` is no longer needed (it stops being needed once the
   driver writes the intervals it actually used into the HDF5 as well,
   instead of an integer pointer into a hard-coded list).
3. **Move the lat/lon crop into the metadata** that `write_h5`
   already records, instead of two magic slices in `h52nc.py`.

---

## Programmatic `build_run_breaks(start, end, chunk_years=...)`

Previously on `main`, the three transient-chunk schedules in
[src/caete.py](src/caete.py) were hand-written 2-year tuples:

- `run_breaks_hist` — 1979-01-01 → 2016-12-31, 19 × 2-year chunks.
- `run_breaks_CMIP5_hist` — 1979-01-01 → 2005-12-31 (last chunk 1 year).
- `run_breaks_CMIP5_proj` — 2006-01-01 → 2099-12-31, 47 × 2-year chunks.

Changing the cadence (e.g. switching from 2-year to 10-year netCDF
files) meant re-typing every tuple by hand, and changing the start/end
year of any list meant re-deriving every break manually.

### What changed on this branch

- **New helper** `build_run_breaks(start, end, *, chunk_years=2,
  align="calendar")` lives in [src/caete.py](src/caete.py) (around
  line 73). It accepts `'yyyymmdd'` strings or `datetime.date` /
  `cftime.real_datetime` objects, returns
  `list[tuple[str, str]]` in the same `'%Y%m%d'` format the rest of
  the pipeline already uses, and supports a "from_start" alignment for
  experiments that don't begin on Jan 1.
- **The three `run_breaks_*` lists are now built from it** (line 132
  area):
  ```python
  run_breaks_hist        = build_run_breaks('19010101', '20241231')
  run_breaks_CMIP5_hist  = build_run_breaks('19790101', '20051231')
  run_breaks_CMIP5_proj  = build_run_breaks('20060101', '20991231')
  rbrk = [run_breaks_hist, run_breaks_CMIP5_hist, run_breaks_CMIP5_proj]
  ```

- **Backwards-compatible**: every existing consumer that did
  `rbrk[0]`, `for interval in run_breaks`, or
  `enumerate(run_breaks_hist)` keeps working unchanged — only the
  *source* of those lists changed.
- **`rbrk_index`** (the 4th line of `stime.txt`) still picks which
  list `h52nc.catch_stime` rebuilds, exactly as before.

### Edge cases preserved

- Final chunks shorter than `chunk_years` are still allowed (e.g.
  CMIP5_hist's `('20050101', '20051231')`).
- Chunk strings are emitted as `'%Y%m%d'` (no separators), matching
  `cf_date2str` and the HDF5 row keys, so `time_queries` works
  unchanged.

### Suggested follow-ups

1. **Validation tests.** Add round-trip tests asserting that the
   helper reproduces the legacy literals byte-for-byte when called
   with the original year ranges, and behaves correctly for
   `chunk_years=1`, leap years, odd-length spans, and mid-year start
   dates.
2. **Move date helpers into a `time_axis.py` module** alongside
   `cf_date2str` / `str2cf_date` (currently in
   [src/post_processing.py](src/post_processing.py#L42-L54)).
   Cleaner long-term, but a larger refactor than this branch covers.

---

## Cross-cutting observations

- The three changes above converge on the principle **"derive time
  information from the metadata/HDF5, not from hard-coded constants"**.
  Together they make the chunking cadence editable in exactly one
  place per schedule (the `build_run_breaks` call) and ensure the
  netCDF stage uses the same intervals the simulation ran with.
- `stime.txt` is still written by every driver as a backwards-
  compatible bridge, but it can be retired once `time_units` /
  `calendar` / `experiment` / `intervals` are stored as HDF5 root
  attributes by `write_h5`.
- `find_co2` would benefit from a one-line refactor (dict lookup) the
  next time `caete.py` is touched.
- Input paths in [src/model_driver_local.py](src/model_driver_local.py)
  are still hard-coded constants. Lifting them into a tiny config
  block (or CLI/env overrides) is the natural next step for making
  the driver retargettable to other regions without editing the
  source.


---
