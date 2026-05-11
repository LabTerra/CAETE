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

## Post-processing review — performance, memory & parallelization roadmap

This section captures a focused review of the post-processing pipeline
([src/post_processing.py](src/post_processing.py) and
[src/h52nc.py](src/h52nc.py)) carried out after the three TODOs above
were implemented. The intent is to set up a refactor that **unlocks
multiprocessing/multithreading** for post-processing while keeping each
step small, bisectable, and bit-for-bit verifiable against the current
output. No code changes have been made yet.

### Pipeline overview

The post-processing pipeline has two stages.

1. **`post_processing.write_h5(out_dir, RUN, reclen)`** gathers per-cell
   `.pkz` (joblib) outputs scattered across
   `outputs/<run>/gridcell*/` and writes them into a single `CAETE.h5`
   PyTables file with five tables:
   - `RUN0/Outputs_G1`, `Outputs_G2`, `Outputs_G3` — daily fluxes/state.
   - `RUN0/spin_snapshot` — per-spin/per-interval summaries.
   - `RUN0/PLS` — PLS table.

   It currently performs **four full passes over disk** (one per table),
   reloading every `.pkz` four times, with one row-by-row pure-Python
   PyTables write per day per cell.

2. **`h52nc.h52nc(input_file, dump_nc_folder, ...)`** opens `CAETE.h5`,
   builds sorted copies of each daily table by `date`
   (`indexedT{1,2,3}date`), then for each interval and each variable
   issues `read_where("(date == b'YYYYMMDD')")` once per day, scattering
   rows into dense `(ndays, 61, 71)` arrays via the Python loop in
   `assemble_layer`, and finally writes one netCDF per variable with
   daily `zlib`+`fletcher32` compression. Snapshot writers (`lim_data`,
   `ustrat_data`, `ccc`, `create_nc_area`) iterate the module-global
   `run_breaks` rather than the data-derived `intervals`.

Both stages work end-to-end but leave significant speed, memory, and
parallelism on the table. The per-PLS Python loops in `process_lim`
/ `process_ustrat` scale linearly with `gp.npls`, which is a
compile-time constant in [src/global.f90](src/global.f90) that users
routinely raise (e.g. from 1000 to 2000+) to explore wider trait
spaces — so any growth in `npls` translates directly into snapshot-
stage runtime.

### Tiered refactor plan

The ordering below is by ROI × structural disruption: trivial wins
first, then the structural rewrite that *enables* parallelism, then the
parallel layer itself, then file-size / IO polish, then code-health
follow-ups. Tiers 0–2 are independently shippable and verifiable on the
[test_new_input/](../input/test_new_input/) bundle.

### Tier 0 — Trivial wins (one commit, no structural change)

These are one-line / one-block fixes. None of them changes data layout
or semantics, so they ship before anything else.

- **Bug: `spin_table.flush` → `spin_table.flush()`** in
  [src/post_processing.py](post_processing.py) (around line 581).
  Currently a bound-method reference, not a call. Harmless today only
  because the surrounding `with` block closes the file, but it must be
  fixed before any refactor relies on partial flushes.
- **Drop the hardcoded `expected_rows`**. `exp_rows = 41965625` and
  `exp_rows_snap = 57033` in `write_h5` are sized for a specific 60-cell
  run and silently degrade chunking on every other run. Compute from
  the actual cell list before `create_table`. Note that `cells` already
  enumerates *one entry per `.pkz` file* (i.e. one entry per cell **per
  transient interval**), so the correct expressions are:
  - `expected_rows = sum(ndays_per_pkz for _ in cells)` — sized from a
    cheap pre-pass over `cells` reading just `dt['emaxm'].size`, or
    estimated as `len(cells) * mean_interval_days`.
  - `expected_rows_snap = len(cells)` — exactly one snapshot row per
    `.pkz`, not `len(cells) * n_intervals` (that double-counts).
- **Hoist `get_var_metadata`'s ~200-line dict literal** in
  [src/h52nc.py](src/h52nc.py) to a module-level constant so it stops
  being rebuilt on every call.

**Expected gain.** Negligible wall time on its own; the value is
hygiene and removing a latent bug before structural work begins.

### Tier 1 — `write_h5` consolidation (prerequisite for parallelism)

A single per-cell pass over the `.pkz` files that builds structured
`numpy.recarray`s per table and appends them in one C call. This is the
biggest single-process win **and** the change that makes per-cell
parallelism trivial in Tier 3.

- **One pass per cell instead of four.** Today `write_h5` loops the cell
  list four times (G1, G2, G3, spin_snapshot), each loop calls
  `open_fh(fp)` and `se_dates(dt)` again. Replace with a single
  `for fp, X, Y in cells:` that loads the `.pkz` once and emits all
  four tables.
- **Recarray append instead of row-by-row writes.** Replace the per-day
  `row['k'] = v; row.append()` loop with
  `rec = np.empty(ndays, dtype=table.description._v_dtype)` filled by
  array assignments, then a single `table.append(rec)` per cell per
  table. PyTables writes the buffer in one C call.
- **Compute dates once per cell.** `cf_date2str(date_range[day].date())`
  inside the day loop becomes
  `dates_str = np.array([cf_date2str(d.date()) for d in date_range],
  dtype='S8')` once, used as a column.
- **Audit field coverage during the merge.** While restructuring, assert
  every column listed in [src/template_tables.py](src/template_tables.py)
  for G2/G3 is populated before append — protects against latent
  drift that the current four-separate-loops layout makes easy to miss.

**Expected gain.** 5–20× on the table-write step of `write_h5`, plus
~4× less I/O from loading each `.pkz` once. Peak RAM drops because
no `dt` is held across passes.

**Verification.** Run [src/model_driver_local.py](src/model_driver_local.py)
end-to-end on `test_new_input/` before and after; assert
`tb.open_file(...).root.RUN0.Outputs_G{1,2,3}` are bit-identical (or
modulo row order, since the new path can sort rows by date — see
Tier 4).

### Tier 2 — Vectorize the per-PLS snapshot reductions

`process_lim` and `process_ustrat` in
[src/post_processing.py](src/post_processing.py) (around line 70) use
explicit `for pls in range(npls)` Python loops with 6–15 `np.sum` /
`np.count_nonzero` calls *per PLS per interval*. Because `npls` is a
compile-time constant set in [src/global.f90](src/global.f90) and
users commonly raise it to explore wider trait spaces, this loop
dominates the snapshot stage at any non-trivial `npls` and gets
strictly worse as users push the value upward.

- **`process_lim`**: replace the inner loop with a single broadcast
  reduction:
  ```python
  codes = np.array([0, 1, 2, 4, 5, 6])           # 6 categories
  counts = (pool_lim[:, :, None] == codes).sum(axis=1)  # (npls, 6)
  fracs  = counts.astype(np.float64) / ndays            # (npls, 6)
  weighted = (fracs * area[:, None]).sum(axis=0)        # (6,)
  ```
  No Python loop, all NumPy.
- **`process_ustrat`**: same shape, two output tuples (n_strat, p_strat)
  of widths 6 and 9 — apply the same `(values[:, :, None] == codes).sum`
  pattern to each.
- This is independent of Tier 1 and can ship in parallel.

**Expected gain.** Snapshot stage runtime drops from `O(npls · ncat ·
ndays)` Python to one NumPy call per cell. The speedup scales roughly
linearly with `npls`, so the win is proportionally larger for runs
that raise the compile-time `npls`.

### Tier 3 — `h52nc` core speedup (prerequisite for h52nc parallelism)

Replace the per-day point queries with one range query per interval and
vectorize the layer assembly. Three issues fixed together; doing any
one in isolation leaves the others as the bottleneck.

- **Single range query per interval.** Today
  `create_ncG{1,2,3}` issue `~730` separate
  `read_where("(date == b'YYYYMMDD')")` calls per interval per group.
  After `_get_or_build_sorted`, replace with one
  `table.read_where(f"(date >= b'{s}') & (date <= b'{e}')")` per
  interval, then split on the `date` column with a single
  `np.unique(..., return_index=True)` or
  `np.searchsorted` boundary pass.
- **Vectorize `assemble_layer`.** Replace the
  `for i, val in enumerate(var): out[ny[i], nx[i]] = val` loop with
  `out[ny, nx] = var` (rows are unique per (y, x, day) by
  construction — no duplicate-index hazard).
- **Pre-allocate the per-interval buffer.** Replace the 14+ separate
  `np.zeros((dm1, 61, 71)) - 9999.0` allocations with a single
  `np.full((nvars, dm1, *region_shape), -9999.0, np.float32)` indexed
  by var name. `time_queries(interval)` becomes dead code.

**Expected gain.** 5–30× on `create_ncG{1,2,3}` overall (the read side
collapses from ~730 lookups to 1; the assembly loop collapses to a
single fancy-index assignment).

### Tier 4 — Parallelization layer

Tiers 1–3 produce two natural unit-of-work artifacts that are cheap to
move between processes (NumPy arrays). With those in place,
parallelization is a small wrapper, not a rewrite.

- **`write_h5`: per-cell worker pool.** Each worker takes
  `(fp, X, Y)`, loads the `.pkz`, runs the Tier 1 + Tier 2 work, and
  returns a tuple `(rec_g1, rec_g2, rec_g3, snap_rows)` of recarrays.
  The **parent process** owns the open `tb.File` and does
  `table.append(rec)` sequentially as results come in. PyTables is not
  safe for concurrent writers, but a single-writer / many-readers
  pattern matches it natively.
  - Use `mp.get_context("spawn").Pool(...)` — the **same spawn
    context** as [src/model_driver_local.py](src/model_driver_local.py),
    so f2py extensions and HDF5 stay safe on Windows and on
    Python ≥ 3.14 Linux (see the *Recommended follow-ups* near
    [src/model_driver_local.md L246](src/model_driver_local.md#L246)).
  - `imap_unordered` with a small `chunksize` keeps the parent's append
    queue flowing without buffering all results in RAM.
- **`h52nc`: per-`(interval, table_kind)` worker pool.** The pairs
  `(interval_i, G1|G2|G3)` write to *different* netCDF files and share
  no state, so a `concurrent.futures.ProcessPoolExecutor` over the
  product is contention-free. Each worker opens the HDF5 read-only,
  *without* `H5FD_CORE` (otherwise each child would load the full file
  into its own RAM). Snapshot writers (`lim_data`, `ustrat_data`,
  `ccc`, `create_nc_area`) stay single-threaded — they're cheap and
  share the snapshot table.

**Parallelization caveats to keep in mind.**

- PyTables: **single writer**. Workers must return data; only the parent
  appends. Do not pass an open `tb.File` across the fork/spawn boundary.
- `H5FD_CORE` directly conflicts with multi-process readers — each
  worker would replicate the whole file in RAM. Expose the driver as a
  kwarg of `h52nc()` (Tier 5) and default to `None` (sec2) when
  `NPROCS > 1`.
- Blosc compression releases the GIL during encode/decode, but the
  HDF5 library serializes file-level writes inside one process. So a
  thread pool can overlap *blosc encode + recarray fill* across
  workers, but the actual `table.append` calls still execute one at a
  time. Process pool is therefore preferred — both because
  `joblib.load` on the `.pkz` does not release the GIL, and because
  workers in different processes overlap the decompression / recarray
  build with the parent's HDF5 write.
- Read-side concurrency on `CAETE.h5`: multiple worker processes can
  open the same file with `mode="r"` simultaneously **only if** the
  HDF5 file is not currently open RW elsewhere and either the build
  supports SWMR or `HDF5_USE_FILE_LOCKING=FALSE` is set in the worker
  environment. Easiest invariant: close the writer before launching
  the `h52nc` pool.
- Number of workers: cap at `min(os.cpu_count(), len(cells))` for
  `write_h5`; cap at `min(os.cpu_count(), len(intervals) * 3)` for
  `h52nc`. Memory per worker is dominated by one interval's
  `(nvars, dm1, 61, 71)` buffer (~ `nvars · dm1 · 17 KiB`).

**Expected gain.** Near-linear up to ~8 cores on `write_h5` once Tier 1
lands (the parent's append is fast compared to per-cell encode + scan).
On `h52nc`, 3–4× from running G1/G2/G3 of the same interval in parallel,
more if there are several intervals.

### Tier 5 — File-size, IO & driver options

These are independent of the parallel layer and can land before or
after Tier 4. They mostly affect on-disk size and peak memory.

- **netCDF chunking + compression.** `create_ncG*` currently creates
  variables with `zlib=True, fletcher32=True` and no `chunksizes`,
  which produces 1-row chunks against the unlimited `time` dimension.
  Add `chunksizes=(min(dm1, 365), 61, 71)` and either keep
  `complevel=2` or switch to `compression='zstd'` (netCDF4 ≥ 1.6.5).
  Combined with Tier 3 this lets `write_daily_output` *stream* one
  day at a time and drop the per-variable dense `(dm1, 61, 71)`
  buffer — cuts peak `h52nc` memory by roughly `dm1`×.
- **HDF5 driver kwarg.** Expose `driver=None` (sec2) as a kwarg of
  `h52nc()`; the current `H5FD_CORE` becomes opt-in. Required for
  Tier 4's `h52nc` worker pool, and required to scale past 60 cells in
  general.
- **Snapshot writers consume `intervals`.** `lim_data`, `ustrat_data`,
  `ccc`, `create_nc_area` still iterate the module-global `run_breaks`.
  Pass the runtime `intervals` (or just `[i[0] for i in intervals]`)
  into them and delete the module global. Closes the TODO 2 cleanup
  end-to-end.
- **Sorted-on-write.** Tier 1 can emit rows pre-sorted by `date` per
  cell. If the merged rows are globally sortable cheaply (group cells
  by year and append in date order), `_get_or_build_sorted` becomes a
  no-op for new files — saves the up-front CSI index + full-table
  copy on the read side.
- **`write_h5` compression.** `tb.Filters(complevel=1,
  complib="blosc:blosclz")` is fine; revisiting `blosc2:zstd` should
  wait until the build environment guarantees PyTables ≥ 3.8 with
  blosc2 across Windows / WSL / HPC.

### Tier 6 — Code-health follow-ups (no perf impact)

Track but do not block the perf work on these.

- Lift the hardcoded crop `[160:221, 201:272]` out of
  `assemble_layer` / `assemble_cwm` / `create_lband` / per-variable
  allocations into module-level constants
  `(Y_SLICE, X_SLICE, REGION_SHAPE)` derived from the metadata
  bounding box. Lets `h52nc` work on regions other than Pan-Amazon
  without source edits.
- `dates = np.unique(t1d.cols.date[:])` (line ~1316) reads the entire
  date column to compute its min/max. Replace with
  `t1d.cols.date[0]` / `t1d.cols.date[-1]` on a sorted table. Only
  triggered when both `intervals=None` and `run_breaks` is falsy —
  rare path, low priority.
- `snap = snap_table.read()` once and NumPy-filter by `start_date` in
  the snapshot helpers, instead of one `read_where(build_strds(...))`
  per interval. Snapshot table is small; this is hygiene.
- `np.where(mask, NO_DATA, a + b + c + d)` instead of the polluted-sum +
  `np.place` pattern in `create_ncG2` — correctness clarity, not perf.
- Dropping the unsorted source tables after `_get_or_build_sorted`
  builds the indexed copy was on the original list as a way to halve
  on-disk size — but `remove_node` without an `h5repack` pass does not
  shrink the file. **Removed from this plan**; the proper fix is the
  *sorted-on-write* item in Tier 5.

### Cross-cutting follow-ups already on the list

- Hardening `find_co2(year)` to a dict lookup (mentioned in TODO 1's
  "suggested path", not yet implemented).
- Removing the residual module-global `run_breaks` from
  [src/h52nc.py](src/h52nc.py) once every consumer takes `intervals`
  as an argument (finished by Tier 5).
- Reading `time_units` / `calendar` / `experiment` from HDF5 root
  attributes written during `write_h5`, eliminating `stime.txt`
  entirely.

### Alternative storage-layout strategies (longer-horizon refactors)

Tiers 0–6 keep the wide `Outputs_G1 / G2 / G3` PyTables schema in
[src/template_tables.py](src/template_tables.py) and just write it
faster. A second category of refactor is to **change the on-disk
shape** so that parallelism and selective reads come for free. These
are larger changes — most of them break the existing `h52nc` reader
or the HDF5 contract — so they are deliberately separated from the
incremental plan above. They are listed roughly in order of
disruption.

#### Strategy A — Column-store layout (per-variable arrays, shared index)

Replace each wide table with one **shared coordinate group** plus one
1-D `EArray` per variable, all the same length:

```
RUN0/coords/date     (Sx, nrows)        # 'YYYYMMDD'
RUN0/coords/grid_y   (int16, nrows)
RUN0/coords/grid_x   (int16, nrows)
RUN0/G1/photo        (float32, nrows)
RUN0/G1/npp          (float32, nrows)
RUN0/G1/lai          (float32, nrows)
... one EArray per variable ...
```

- **Selective reads.** `h52nc` (or any downstream consumer) reads only
  the columns it actually needs, e.g. `f.root.G1.npp[:]` instead of
  pulling the full 14-column record. With blosc compression, each
  EArray decompresses independently — non-trivial savings on memory
  bandwidth for tools that only want one or two fields.
- **Better compression ratios.** Per-column homogeneous data compresses
  significantly better than mixed records; `blosc:zstd` typically
  shows 1.5–2× smaller files for this kind of geophysical column data
  vs. wide-row layout.
- **Write parallelism (limited).** Different EArrays in the same HDF5
  file cannot be written concurrently — neither by separate processes
  (file-level lock without parallel-HDF5) nor by threads within one
  process (PyTables / h5py serialize writes through the HDF5 library
  mutex). What *does* parallelize with threads is the **blosc encode**
  and the **recarray fill** that happen before each append. So a small
  thread pool gains only the per-variable encode cost, not the write
  itself. Real per-variable parallelism still needs Strategy C / E
  (separate files / Zarr chunks).
- **Cost.** Breaks `h52nc`'s current `read_where("(date == ...)")`
  pattern, because the date column lives in a sibling array. The
  replacement is straightforward — `np.searchsorted(coords.date,
  interval)` once, then slice every variable's EArray with the same
  start/stop — and is actually *simpler* than the current code.
- **Migration.** Doable in one commit alongside Tier 1: the per-cell
  recarray is split into per-variable slices before append, instead of
  appended whole. The PyTables description in
  [src/template_tables.py](src/template_tables.py) is replaced by a
  variable-name → dtype dict.

This is the recommended longer-horizon target: it cleanly subsumes
Tier 1, makes selective reads trivial, and removes the
"row-vs-column" awkwardness that motivates the snapshot column-list
audit (Tier 1's last bullet).

#### Strategy B — Dense `(time, y, x)` cubes per variable (netCDF-shaped HDF5)

The end state of post-processing is already a dense
`(ndays, 61, 71)` cube per variable per interval (the netCDF).
`write_h5` could **write that shape directly** and skip the
long-form table entirely:

```
RUN0/G1/photo        (float32, ndays, ny, nx)   chunked, compressed
RUN0/G1/npp          (float32, ndays, ny, nx)
RUN0/coords/time     (Sx, ndays)
RUN0/coords/lat      (float32, ny)
RUN0/coords/lon      (float32, nx)
```

- **`h52nc` becomes a transcoder, not a reshuffler.** Today `h52nc`
  spends most of its time scattering long-form rows into a dense
  cube; if the cube already exists on disk, `h52nc` collapses to a
  per-variable `dst[:] = src[:]` copy with attribute decoration.
  Tier 3 of the incremental plan is mooted by this strategy.
- **Selective writes per worker.** With chunked, fill-value-initialized
  EArrays, a worker that owns a cell can write
  `arr[:, cell_y, cell_x] = values` to its slot **without touching
  any other slot**. HDF5's chunk-level write still requires the
  file-level lock, but if each worker writes a *different chunk* (or
  the file is opened by one writer who batches per-chunk writes),
  contention is minimal. With chunk size `(ndays_chunk, ny, nx)`
  spanning the whole region, sequential per-cell writes from a
  thread pool are GIL-released during compression.
- **Storage cost.** Pan-Amazon crop is 61 × 71 = 4 331 cells; the
  land mask covers a few thousand. A dense cube with `-9999` fill at
  blosc default compression costs ~the same as the long-form table
  because the missing cells compress to nearly nothing. For the
  global grid this is *not* true and a CSR-style sparse layout would
  be needed.
- **Cost.** This is essentially "write the netCDF directly into HDF5".
  Worth it only if downstream tooling is happy reading HDF5 in that
  layout (it is — Xarray's `h5netcdf` backend treats this layout as a
  netCDF4 file with no conversion).
- **Why not do it now.** It is the largest single change of any
  strategy here: `write_h5`, `h52nc`, and every snapshot consumer
  would all need to change at once. Worth scheduling, not worth
  rushing.

#### Strategy C — One HDF5 file per variable (or per group)

Sidestep the HDF5 single-writer limit by writing **N independent
HDF5 files** in parallel — one per variable, or one per `G{1,2,3}`
group — and either leaving them split or merging them at the end.

- **Parallelism.** Becomes trivial: each worker owns its own file
  descriptor, no cross-process locking. Scales linearly with cores up
  to the number of files.
- **Consumer impact.** `h52nc` already produces one netCDF per
  variable, so a "one HDF5 per variable" intermediate is not a
  regression. If `h52nc` is reworked to consume the per-variable HDF5
  directly (Strategy B style), the intermediate even simplifies
  things.
- **Cost.** Bookkeeping (one file per variable means dozens of
  artifacts), and the `RUN0/PLS` table no longer has a natural home.
  Reasonable compromise: one HDF5 per `G1/G2/G3/snap/PLS` group,
  giving 5 files and 5-way write parallelism without proliferating
  artifacts.

#### Strategy D — Skip the merge: feed `h52nc` from `.pkz` directly

The strongest claim is that `write_h5` is **load-bearing in form
only**: nothing downstream consumes its long-form schema except
`h52nc`, which immediately reshapes it back into the
`(ndays, y, x)` cube. A small `h52nc` rewrite could read each
`.pkz` directly, scatter into the cube in memory, and write the
netCDF — eliminating the intermediate HDF5 entirely.

- **Parallelism.** Per-cell reads are embarrassingly parallel — one
  worker per cell loads its `.pkz` and contributes its slice to a
  shared-memory buffer (`multiprocessing.shared_memory`) or to a
  reducer in the parent. No HDF5, no file-level lock, no PyTables.
- **Write side.** Only `h52nc` writes the final netCDF — single
  writer, no contention. Chunked + zstd-compressed netCDF gives the
  same on-disk size as the current `CAETE.h5` → netCDF pipeline.
- **What breaks.** Any consumer that today does `read_where` against
  `CAETE.h5` (currently: only `h52nc` itself) needs an alternative
  query path. The PLS table and snapshots stay as small standalone
  files or get inlined as netCDF auxiliary variables.
- **Cost.** Loses `CAETE.h5` as a portable intermediate (some
  downstream notebooks may depend on it; needs a survey before
  committing). Conceptually the cleanest endpoint of the
  post-processing refactor.

#### Strategy E — Zarr / Parquet instead of HDF5

If we are willing to depend on a new storage backend, **Zarr** is the
natural fit for this workload:

- Chunks are individual files (or object-store keys), so independent
  workers writing to *non-overlapping* chunks have **zero
  contention** — true parallel writes without parallel-HDF5.
- Same `(time, y, x)` chunked layout as Strategy B, but the
  parallel-write story is dramatically simpler.
- Reads via `xarray.open_zarr(...)` integrate with the rest of the
  scientific Python stack without a transcode step.
- Cost: new dependency (`zarr`, `numcodecs`), and `caete.py`
  consumers / archival pipelines built around the existing `.h5`
  artifact would need to migrate or keep a parallel HDF5 writer.

Parquet is also viable for the long-form table layout (Strategy A on
disk) — `pyarrow` writes one Parquet file per partition in parallel,
and DuckDB / Polars can query the result without loading it — but it
fits the long-form schema less naturally than Zarr fits the dense
cube. Mentioned for completeness.

#### Strategy F — Parallel HDF5 (MPI-IO)

Build h5py / PyTables against parallel HDF5 and use MPI-IO to
coordinate concurrent writes to a single file from multiple processes.
True parallel writes, single output artifact.

- **Why not.** Requires a parallel HDF5 build on every machine that
  runs the post-processor (Windows builds in particular are painful),
  and `tables` (PyTables) does not expose the parallel API — only
  raw `h5py` does, so the `IsDescription`-based code path would have
  to be reimplemented. Disproportionate dependency burden for the
  performance gained over Strategy C or E.

#### Recommendation between the strategies

If only one of these is pursued past the incremental plan, the most
favourable trade-off is **Strategy A** (column-store HDF5):

1. Same file, same readers, same archival story — only the internal
   layout changes.
2. Selective reads (one variable at a time) drop from "read 14 cols
   then discard 13" to "read 1 col" — directly speeds up `h52nc` and
   any analysis notebook.
3. Threaded encode parallelism becomes usable inside one process,
   without changing the multiprocessing model.
4. It is a natural superset of Tier 1's recarray work: the recarray
   is just split column-wise before append.

**Strategy B** is the right *eventual* target if `h52nc` is ever
rewritten end-to-end, and **Strategy D** is the right target if the
intermediate HDF5 turns out not to have any real downstream consumer.
**Strategies C / E / F** are escape hatches if HDF5's single-writer
limit becomes the binding constraint after Tiers 1–4 land — they
should not be adopted preemptively.

### Reviewer findings (cross-check against the current code)

A pass through [src/post_processing.py](src/post_processing.py) and
[src/h52nc.py](src/h52nc.py) confirms the plan above is accurate apart
from the three items already corrected inline (`expected_rows_snap`
sizing, thread-vs-process write parallelism, and the column-store
threaded-write claim in Strategy A). Two more items deserve calling out
explicitly:

- **`process_lim` codes are `[0, 1, 2, 4, 5, 6]`** (skipping `3`), but
  **`process_ustrat` N-codes are `[0, 1, 2, 3, 4, 6]`** (skipping `5`)
  and **P-codes are `[0..8]`**. Tier 2's vectorization recipe must use
  the right code list per channel — the snippet in Tier 2 happens to
  be correct for `process_lim` only.
- **`process_ustrat` reduction is asymmetric**: 6 outputs on the N
  axis, 9 on the P axis, derived from the same `u_strat` array. The
  vectorized rewrite should produce one `(npls, 6)` and one
  `(npls, 9)` count matrix, then `(counts / ndays * area[:, None]).sum(axis=0)`
  per channel — not a single shared reduction.

### Additional improvements not yet captured in the plan

The review surfaced several wins outside the Tier 0–6 framing and
outside the storage-layout strategies. They are smaller and largely
orthogonal, so they ship whenever convenient.

#### Producer-side wins (model worker, before post-processing runs)

- **Drop bz2 from the `.pkz` artifacts.** `joblib.dump(..., compress=3)`
  with bzip2 is the dominant cost in `open_fh` because bz2 decompression
  is both slow and GIL-bound. Switching to `compress=('lz4', 1)` or
  `compress=('zstd', 3)` gives roughly 3–10× faster reads at similar
  on-disk size, and would speed up `write_h5` end-to-end before any
  other change. Backwards compatibility: keep `open_fh` reading both
  formats (joblib auto-detects).
- **Stream straight to HDF5 from the model workers.** The strongest
  long-term simplification: each transient-chunk worker, instead of
  writing a `.pkz`, opens its own per-cell HDF5 (Strategy C style) and
  writes the recarray directly. Removes the entire join-up step in
  `write_h5` and the `.pkz` artifact tree.

#### `write_h5` (post-processing) wins beyond Tier 1

- **`open_fh` masks errors.** The bare `except: raise FileNotFoundError`
  catches *every* exception (unpickle errors, KeyboardInterrupt,
  corrupted files) and re-raises as `FileNotFoundError`, hiding the
  real cause. Replace with a narrow `except OSError` and let other
  exceptions propagate. Pure correctness fix, no perf impact.
- **`PLS_table` is still row-by-row.** The `for n in range(gp.npls):
  for key in tt.PLS_head: PLS_row[key] = pls_df[key][n]` block writes
  one row at a time in Python. Cost scales linearly with the
  compile-time `gp.npls`, so this is already measurable at default
  settings and grows with any user-side increase. Same fix as Tier
  1's recarray pattern:
  `rec = pls_df[list(tt.PLS_head)].to_records(index=False)` →
  `PLS_table.append(rec)`.
- **Directory scan uses `os.listdir` + `os.path.is_dir`.** On large
  output trees (Pan-Amazon ≈ 10⁴ gridcells × N intervals) this does
  one stat per entry. `os.scandir(out_dir)` returns `DirEntry` objects
  with cached `is_dir()`; switch to it. The `sorted(os.listdir(grd))`
  inside the inner loop is also redundant for correctness — only
  needed if downstream depends on order, which it does not (rows get
  a monotonic `row_id` either way).
- **Avoid the snapshot table's intermediate Python lists.** Even
  *before* the Tier 2 vectorization lands, `process_lim` /
  `process_ustrat` build 6–15 Python lists per call only to
  `np.sum` them. A pre-allocated NumPy `out[6]` filled by index drops
  the per-call allocator pressure and is a 30-line diff.
- **Sentinel `row_id` is global across cells per table.** The current
  `rec = 0` counter and the four separate loops imply `row_id` is
  contiguous *per table*. After Tier 4 (parallel write), preserving
  that property requires the parent to assign `row_id` at append
  time, not the worker — easy to forget. Either document that
  `row_id` may be reordered (and is therefore advisory), or rebuild
  it post-hoc with `table.modify_column(0, len(table), 'row_id',
  np.arange(len(table)))`.

#### `h52nc` wins beyond Tier 3

- **One netCDF per *interval* instead of one per *variable*.**
  Currently `write_daily_output` (line 227) creates a separate
  `.nc4` per variable (`photo.nc4`, `npp.nc4`, `lai.nc4`, …), each
  with its own duplicated `time`, `lat`, `lon` coordinate variables.
  Packing all variables of one group into one `G1_<interval>.nc4`
  (CF-compliant, opened in one shot by `xarray.open_dataset`) is
  more user-friendly, halves file handles, and avoids re-emitting
  the coordinate arrays 14× per interval. The current per-variable
  split fits the legacy reader pattern, but no documented consumer
  relies on it.
- **`time_queries(interval)` is dead code post-Tier-3.** Once the
  per-day `read_where` is replaced by a single range query, the
  helper that builds the per-day query string list has no callers.
  Delete it rather than letting it bit-rot.
- **`indexedT{1,2,3}date` rebuilt every run.** `_get_or_build_sorted`
  reuses the sorted copy if present, but `write_h5` writes a fresh
  `CAETE.h5` (`mode="w"`), so the cached sorted copy is *always*
  absent on the first `h52nc` call. Either (a) make `write_h5` emit
  rows already sorted by `date` (Tier 5 "sorted-on-write"), or
  (b) build the index immediately at end of `write_h5` while the
  data is still hot in the page cache — saves one full read pass at
  `h52nc` startup.
- **Snapshot writers read `snap_table` four times.** `lim_data`,
  `ustrat_data`, `ccc`, `create_nc_area` each issue their own
  `read_where(build_strds(...))` per interval; for `n_intervals`
  intervals and 4 writers that is `4 · n_intervals` table scans. One
  `snap = snap_table.read()` at the top of `h52nc()` plus NumPy
  filtering by `start_date` collapses it to one scan total. Listed in
  Tier 6 as hygiene but is actually a measurable win once the rest
  of the pipeline is fast.

#### Observability / verification

- **Add a fast `--validate` mode.** After Tier 1 and Tier 3 land, the
  refactored writers should produce bit-identical output to `main`
  on the [test_new_input/](../input/test_new_input/) bundle, modulo
  documented row ordering. A small script that opens both
  `CAETE.h5` files with PyTables, sorts by `(grid_y, grid_x, date)`,
  and asserts column-wise equality (with `np.allclose` for float32)
  would catch regressions cheaply. Add to the repo once Tier 1 lands.
- **`%timeit`-style per-stage timing block.** `write_h5` and `h52nc`
  have no internal timing; the only feedback is the
  `print_progress` bar. A `contextlib.contextmanager` named
  `stage("g1 loop")` that logs wall time to stderr would make Tier
  comparisons quantitative without a profiler.

#### Out-of-scope but worth tracking

- **Pre-aggregation to monthly / annual.** Many CAETÊ consumers reduce
  the daily netCDF to monthly means before plotting. Doing the
  reduction during `h52nc` (when the dense cube already lives in RAM)
  is essentially free, vs. doing it later via `xarray.resample` which
  reads the chunked netCDF back. One extra `var.mean(axis=0)` and a
  parallel `_monthly.nc4` output covers most use cases.
- **`HDF5_USE_FILE_LOCKING=FALSE` recipe.** On networked filesystems
  (HPC `$SCRATCH`, NFS) HDF5 file locking can refuse to open the
  file at all. The local driver should set this env var before
  importing PyTables/h5py for those environments. Document, do not
  bake into the code path.

