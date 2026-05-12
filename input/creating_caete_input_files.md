# CAETÊ input file format (`.pbz2`)

This document specifies the binary format of the per-gridcell input files
consumed by the CAETÊ Dynamic Vegetation Model. Use it as a reference if you
need to generate inputs from a data source other than the included
pre-processing script (`input/pre_processing.py`).

There are **two** file types per simulation:

1. A single **metadata** file describing time/lat/lon coordinates of the
   climate forcing.
2. **One per-gridcell** climate + soil file, named by global grid indices.

Both files are bzip2-compressed Python pickles. The model reads them with
`bz2.BZ2File(..., mode='r')` followed by `pickle.load(fh)`.

---

## 1. Container format

| Property            | Value                                                                  |
| ------------------- | ---------------------------------------------------------------------- |
| Compression         | bzip2 (`bz2.BZ2File`)                                                  |
| Serialization       | `pickle` (`_pickle as pkl`); default protocol used by the writing script |
| Writer in repo      | `bz2.BZ2File(fpath, mode='w'); pkl.dump(obj, fh)`                      |
| Reader in repo      | `bz2.BZ2File(fpath, mode='r'); obj = pkl.load(fh)`                     |
| File extension      | `.pbz2`                                                                |
| Endianness/encoding | Whatever NumPy/pickle defaults to on the producing host (cross-host OK)|

> Pickle is not a safe format for untrusted data. Only load `.pbz2` files
> from sources you trust.

---

## 2. Grid convention

CAETÊ uses a global 0.5° regular lat/lon grid:

- Shape: `(360, 720)` → `(y, x)` with `y` ∈ `[0, 360)`, `x` ∈ `[0, 720)`.
- Origin: `y = 0` is the **northernmost** row, `x = 0` is the
  **westernmost** column (180°W). Latitude decreases with increasing `y`;
  longitude increases with increasing `x`.
- Cell size: 0.5° × 0.5° (`yres = xres = 0.5`).
- The file `input/mask/mask_raisg-360-720.npy` is a boolean array with the
  same shape; `True` means *masked out* (no data / not processed),
  `False` means *valid land cell to process*.

---

## 3. Metadata file

### 3.1 Filename

By convention:

```
ISIMIP_HISTORICAL_METADATA.pbz2
```

stored alongside the per-gridcell `input_data_*.pbz2` files in the same
folder. Other scripts in this repository use prefixed variants
(e.g. `<model>-<rcp>_METADATA.pbz2`); the **payload structure is identical**.

### 3.2 Payload

A 3-tuple of plain Python `dict`s in this order:

```python
(time_dict, lat_dict, lon_dict)
```

Each dict mirrors the corresponding NetCDF coordinate variable's metadata.

#### `time_dict`

| Key             | Type                  | Notes                                                                 |
| --------------- | --------------------- | --------------------------------------------------------------------- |
| `standard_name` | `str`                 | Typically `"time"`.                                                   |
| `units`         | `str`                 | CF-compliant, e.g. `"days since 1901-01-01 00:00:00"`. Required.      |
| `calendar`      | `str`                 | CF calendar, e.g. `"proleptic_gregorian"`, `"standard"`. Required.    |
| `time_index`    | `np.ndarray` (1-D)    | Numeric time stamps. Used by `cftime.num2date` together with `units`/`calendar`. Length **must equal** the time length of every climate array in every per-gridcell file produced from the same forcing. |

#### `lat_dict`

| Key             | Type               | Notes                                                |
| --------------- | ------------------ | ---------------------------------------------------- |
| `standard_name` | `str`              | Usually `"latitude"`.                                |
| `units`         | `str`              | Usually `"degrees_north"`.                           |
| `axis`          | `str`              | Usually `"Y"`.                                       |
| `lat_index`     | `np.ndarray` (1-D) | Latitude values for the global 360-row grid.         |

#### `lon_dict`

| Key             | Type               | Notes                                                |
| --------------- | ------------------ | ---------------------------------------------------- |
| `standard_name` | `str`              | Usually `"longitude"`.                               |
| `units`         | `str`              | Usually `"degrees_east"`.                            |
| `axis`          | `str`              | Usually `"X"`.                                       |
| `lon_index`     | `np.ndarray` (1-D) | Longitude values for the global 720-column grid.     |

### 3.3 Reading example

```python
import bz2, pickle as pkl, copy
with bz2.BZ2File("ISIMIP_HISTORICAL_METADATA.pbz2", "r") as fh:
    time_d, lat_d, lon_d = pkl.load(fh)
stime = copy.deepcopy(time_d)
units, calendar = stime["units"], stime["calendar"]
time_index = stime["time_index"]
```

---

## 4. Per-gridcell file

### 4.1 Filename

```
input_data_{Y}-{X}.pbz2
```

where `Y` and `X` are the **global** integer grid indices defined in §2.
For example, `input_data_175-235.pbz2` is the cell at `(y=175, x=235)`.

The model derives the filename in [`src/caete.py`](../src/caete.py) as
`f"input_data_{self.xyname}.pbz2"` with `self.xyname = f"{y}-{x}"`. Names
are case-sensitive and must match exactly.

### 4.2 Payload

A single Python `dict` with **exactly 10 keys**. Order is not significant
to the model loader, but the script in this repository writes them in this order:

```python
{
    "hurs": <np.ndarray>,   # climate, 1-D over time
    "tas":  <np.ndarray>,   # climate, 1-D over time
    "ps":   <np.ndarray>,   # climate, 1-D over time
    "pr":   <np.ndarray>,   # climate, 1-D over time
    "rsds": <np.ndarray>,   # climate, 1-D over time
    "tn":   <scalar>,       # soil, single value
    "tp":   <scalar>,       # soil, single value
    "ap":   <scalar>,       # soil, single value
    "ip":   <scalar>,       # soil, single value
    "op":   <scalar>,       # soil, single value
}
```

No additional keys are read by the model; extra keys are ignored but waste
space. Missing keys raise `KeyError` at run time.

### 4.3 Climate arrays

All five climate arrays must satisfy:

- 1-D NumPy array.
- Identical length, equal to `len(time_dict["time_index"])` of the metadata
  file. The model uses one shared time axis for all variables and all
  cells.
- `dtype`: floating-point. `float32` is what the script writes; `float64`
  also works. No masked arrays — fill any missing values before saving.
- Memory layout: contiguous; the script uses `.copy(order="F")`, but
  C-contiguous arrays work equally well.
- No `NaN` / fill values: the model does no masking and will propagate them
  through the integration.

### 4.4 Climate variable definitions and units (REQUIRED)

TODO: Review the conversion of rsds (W m⁻²) to mol(photons)m⁻² day⁻¹ ; the current factor was never checked since CPTEC-PVM2 times(2016). The model applies it as `rsds * 0.5 / 2.18e5` to get mol(photons) m⁻² s⁻¹. This conversion factor misses the 86400 multiplier, so the model is currently underestimating the PAR input by a factor of 86400. The correct conversion should be `rsds * 0.198` to get mol(photons) m⁻² day⁻¹ directly. This is based on the following calculations: 

```
# Convert W m⁻² to J m⁻² day⁻¹:
# 1 W = 1 J/s
# 1 day = 86400 s
# So, 1 W m⁻² = 86400 J m⁻² day⁻¹

# Joules per mol photons:
# The value 218,000 J/mol corresponds to photons at approximately
# 550 nm, which is near the middle of the PAR spectrum.
# The energy of 1 mol photons in PAR is about 218,000 J (2.18e5 J).

# Fraction of PAR:
# Multiply by 0.5 (if we assume 50% of total solar is PAR).
# We should test ranges (0.40 to 0.50) 50% is too much

# In [1]: 1 * 86400 * 0.5 / 218000
# Out[1]: 0.1981651376146789

# In [2]: 1 * 86400 * 0.45 / 218000
# Out[2]: 0.178348623853211

# In [3]:  0.178348623
# In [3]: 1 * 86400 * 0.40 / 218000
# Out[3]: 0.15853211009174312

#Using 0.5 as conversion factor.
# (photons) m⁻² day⁻¹ = (W m⁻²) * 86400 * 0.5 / 218000
# = (W m⁻²) * 0.198

```

NOTE: The change in the conversion factor is applied in this commit. The table below reflects the main branch of the repository.


These are the units the model expects. The conversion factors hard-coded
in [`src/caete.py`](../src/caete.py) are listed for reference; if your
source uses different units you **must** convert before writing the
`.pbz2`.

| Key    | Long name                | Required input unit          | Used in model as                                         | Source unit (ISIMIP) |
| ------ | ------------------------ | ---------------------------- | -------------------------------------------------------- | ----------------------------- |
| `tas`  | Near-surface air temp.   | **K** (kelvin)               | °C (`tas - 273.15`)                                      | K                             |
| `pr`   | Precipitation flux       | **kg m⁻² s⁻¹**               | mm day⁻¹ (`pr * 86400`)                                  | kg m⁻² s⁻¹                    |
| `ps`   | Surface air pressure     | **Pa**                       | hPa (`ps * 0.01`)                                        | Pa                            |
| `rsds` | Surface SW down. flux    | **W m⁻²**                    | mol(photons) m⁻² day⁻¹ (`rsds * 0.5 / 2.18e5`, ×86400)   | W m⁻²                         |
| `hurs` | Near-surface relative humidity | **%** (0–100)          | fraction (`hurs / 100`)                                  | %                             |

Frequency: **daily** time steps. The full multi-year time series is stored
as a single array per variable.

TODO: Inlcude the units of input data as metadata in the future, so that the model can perform the conversions internally and be more flexible with input sources. We can use pint or a custom unit handling system for this purpose. For now, the strict unit requirements are necessary to ensure consistency and correctness in the model's calculations.

### 4.5 Soil variables

All five soil entries are **scalars** representing a single value per
gridcell (the script extracts a single `(y, x)` element from a global
`(360, 720)` `.npy` array).

| Key  | Long name                         | Required unit | Notes                            |
| ---- | --------------------------------- | ------------- | -------------------------------- |
| `tn` | Total soil nitrogen               | g m⁻²         | Must be ≥ 0                      |
| `tp` | Total soil phosphorus             | g m⁻²         | Must be ≥ 0                      |
| `ap` | Available (labile) phosphorus     | g m⁻²         | Must be ≥ 0; `ap ≤ tp`           |
| `ip` | Inorganic (mineral) phosphorus    | g m⁻²         | Must be ≥ 0                      |
| `op` | Organic phosphorus                | g m⁻²         | Must be ≥ 0; `ip + op + ap ≈ tp` (subject to source-data assumptions) |

Type: any Python numeric or 0-D NumPy scalar. The model treats them as
floats.

### 4.6 Reading example

```python
import bz2, _pickle as pkl
with bz2.BZ2File("input_data_175-235.pbz2", "r") as fh:
    cell = pkl.load(fh)

assert set(cell) >= {"hurs","tas","ps","pr","rsds","tn","tp","ap","ip","op"}
assert cell["tas"].ndim == 1
assert len(cell["tas"]) == len(cell["pr"]) == len(cell["hurs"])
```

---

## 5. Consistency requirements across the file set

When you generate a set of `.pbz2` files for one simulation:

1. The metadata file and **every** per-gridcell file must use the same
   time axis (same length, same `units`, same `calendar`, same
   `time_index`). The model does no resampling.
2. All gridcells must be on the same global 360 × 720 grid. Filenames
   carry the only positional information the model uses to locate a cell.
3. Land-cell selection: the model iterates over cells listed in (or
   compatible with) `mask/mask_raisg-360-720.npy`. Files for masked-out
   cells are simply not produced.
4. Cells produced for a different region need a compatible mask passed at
   model run time.

---

## 6. Minimal writer (reference)

The block below is the smallest possible writer. It is for documentation
only — production users should prefer `input/pre_processing.py`, which
handles the full grid, metadata, and validation.

```python
import bz2
import _pickle as pkl
import numpy as np

# --- metadata (one per simulation) ---
T = 365 * 10  # 10 years of daily data, for example
time_dict = {
    "standard_name": "time",
    "units": "days since 1901-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "time_index": np.arange(T, dtype=np.float64),
}
lat_dict = {
    "standard_name": "latitude",
    "units": "degrees_north",
    "axis": "Y",
    "lat_index": np.linspace(89.75, -89.75, 360),
}
lon_dict = {
    "standard_name": "longitude",
    "units": "degrees_east",
    "axis": "X",
    "lon_index": np.linspace(-179.75, 179.75, 720),
}
with bz2.BZ2File("ISIMIP_HISTORICAL_METADATA.pbz2", "w") as fh:
    pkl.dump((time_dict, lat_dict, lon_dict), fh)

# --- one gridcell ---
y, x = 175, 235
cell = {
    "hurs": np.full(T, 80.0, dtype=np.float32),       # %
    "tas":  np.full(T, 298.15, dtype=np.float32),     # K
    "ps":   np.full(T, 101325.0, dtype=np.float32),   # Pa
    "pr":   np.full(T, 5.787e-5, dtype=np.float32),   # kg m-2 s-1 (~5 mm/day)
    "rsds": np.full(T, 220.0, dtype=np.float32),      # W m-2
    "tn":   np.float32(450.0),   # g m-2
    "tp":   np.float32(40.0),    # g m-2
    "ap":   np.float32(2.0),     # g m-2
    "ip":   np.float32(15.0),    # g m-2
    "op":   np.float32(23.0),    # g m-2
}
with bz2.BZ2File(f"input_data_{y}-{x}.pbz2", "w") as fh:
    pkl.dump(cell, fh)
```

---

## 7. Validation checklist

Before using a generated file set in simulations, verify:

- [ ] One `ISIMIP_HISTORICAL_METADATA.pbz2` exists alongside the cell files.
- [ ] Loading it yields a 3-tuple of dicts with the keys listed in §3.2.
- [ ] `time_index` length `T` is consistent and matches every climate array.
- [ ] Every cell file is named `input_data_{Y}-{X}.pbz2` with integer
      `Y ∈ [0, 360)`, `X ∈ [0, 720)`.
- [ ] Loading each cell yields a `dict` with exactly the 10 keys in §4.2.
- [ ] Climate arrays are 1-D, length `T`, no NaN/fill values.
- [ ] Units match §4.4.
- [ ] Soil values are scalars and ≥ 0.
- [ ] Files are loadable on a CAETÊ-compatible Python environment
  (NumPy ≥ the writing environment's; pickle protocol ≤ that supported by
  the target interpreter).

---

## 8. Running `pre_processing.py`

The included pre-processing script [`input/pre_processing.py`](pre_processing.py) reads
ISIMIP-style NetCDF climate files plus the soil `.npy` arrays and writes
the metadata file and one `.pbz2` per land cell of the Pan-Amazon region.
This script can serve as a reference implementation for generating CAETÊ
inputs from other sources, or as a starting point for your own workflow 
if your source data is ISIMIP or similar.

**Tips for the script**

The pre-processing script [`input/pre_processing.py`](pre_processing.py) 
does not perform any check on the raw input data, so ensure they are in the expected
format and units before running. Pay close attention to variables and files 
naming in the raw input folder, as the script relies on a specific structure
to locate the raw data and to open multi-file datasets in the correct temporal
order. Don't forget to check the time axes and time metadata of your 
raw NetCDF files, as the script assumes they are consistent across all
variables and files.

### 8.1 Prerequisites for the script

- Python ≥ 3.11 (uses the stdlib `tomllib`).
- Packages: `numpy`, `netCDF4`.
- Raw climate data on disk, organised as:

  ```
  {climate_data}/{dataset}/{mode}_raw/*_{var}_*.nc[4]
  ```

  Naming convention used by the script:

  - `climate_data`: top-level root directory where all datasets live.
  - `dataset`: dataset family folder (for example,
    `20CRv3-ERA5_ISIMIP3a`).
  - `mode`: run/scenario identifier (for example, `obsclim`, `spinclim`).
  - `_raw`: a **literal suffix** expected by the script. The code always
    looks for climate NetCDF files in a folder named exactly
    `{mode}_raw` (for example, `obsclim_raw`). This suffix distinguishes
    source NetCDF inputs from processed outputs, which are written under
    `input/{dataset}/{mode}/` (without `_raw`).

  with one NetCDF (or one set of time-sliced NetCDFs) per variable
  `var ∈ {hurs, tas, pr, ps, rsds}`. Files are opened with
  `netCDF4.Dataset` when a single match is found, or `MFDataset` when
  multiple files match (they are concatenated along `time`).
- A boolean global mask of shape `(360, 720)` saved as `.npy`
  (`True` = masked out).
- Soil nutrient arrays (`.npy`, shape `(360, 720)`, units g m⁻²) for
  `tn`, `tp`, `ap`, `ip`, `op`.

### 8.2 Configuring `pre_processing.toml`

The script reads `./pre_processing.toml` from the current working
directory. All paths are relative to that working directory unless
absolute. Example:

```toml
# Root folder containing one subdirectory per dataset.
climate_data = "C:\\Users\\darel\\Desktop"

# Default dataset and mode (override on the CLI with --dataset / --mode).
dataset = "20CRv3-ERA5_ISIMIP3a"
mode    = "obsclim"

# Resolution settings (degrees). Must be 0.5; geos.py assumes this.
yres = 0.5
xres = 0.5

# Land/region mask. Boolean (360, 720); True = masked out.
mask_file = "./mask/mask_raisg-360-720.npy"

# Folder containing the soil .npy files referenced under [soil_files].
soil_data = "./soil"

# Filenames of the soil arrays (g/m²). Resolved against soil_data.
[soil_files]
tn = "total_n.npy"
tp = "total_p.npy"
ap = "avail_p.npy"
ip = "inorg_p.npy"
op = "org_p.npy"
```

Field reference:

| Key                | Required | Meaning                                                                 |
| ------------------ | -------- | ----------------------------------------------------------------------- |
| `climate_data`     | yes      | Root folder; the script looks for `{climate_data}/{dataset}/{mode}_raw`.|
| `dataset`          | yes\*    | Dataset folder name. \*May be omitted if passed via `--dataset`.        |
| `mode`             | yes\*    | Mode/scenario subfolder (e.g. `obsclim`, `spinclim`). \*Or `--mode`.    |
| `yres`, `xres`     | yes      | Grid resolution in degrees. Must be `0.5`.                              |
| `mask_file`        | yes      | Path to boolean `(360, 720)` `.npy` mask (`True` = exclude).            |
| `soil_data`        | yes      | Folder containing the soil `.npy` files.                                |
| `[soil_files]` × 5 | yes      | One filename per soil variable (`tn`, `tp`, `ap`, `ip`, `op`).          |

Important naming detail: `mode` in the TOML must be the base name only
(for example, `obsclim`), **not** `obsclim_raw`. The script appends
`_raw` internally when building the climate-input path.

The processed region is hard-coded to the Pan-Amazon bounding box
(`pan_amazon_region` in [`geos.py`](geos.py)): north 10.5°, south −21.5°,
west −80.0°, east −43.0°. Only land cells inside this box that are
*not* masked out are written.

### 8.3 Command-line usage

Always run from the `input/` directory (the script resolves the TOML and
mask paths relative to the CWD):

```powershell
cd C:\Users\darel\Desktop\CAETE\input

# 1) Generate metadata + per-cell .pbz2 files using the toml defaults.
python pre_processing.py

# 2) Validate previously written files against the raw NetCDF.
python pre_processing.py --test
```

CLI flags (all optional; override the TOML when given):

| Flag           | Description                                               |
| -------------- | --------------------------------------------------------- |
| `--dataset`    | Dataset folder name (overrides `dataset` in the TOML).    |
| `--mode`       | Mode subfolder (overrides `mode`).                        |
| `--mask-file`  | Path to a different mask `.npy` (overrides `mask_file`).  |
| `--test`       | Validation mode: re-read random cells and compare with raw NetCDF; **does not** regenerate files. |
| `-h`, `--help` | Show argparse help and exit.                              |

Example with overrides:

```powershell
python pre_processing.py --dataset 20CRv3-ERA5_ISIMIP3a --mode spinclim --mask-file .\mask\mask_raisg-360-720.npy
```

### 8.4 Outputs

For a given run the script creates:

```
input/{dataset}/{mode}/
    ISIMIP_HISTORICAL_METADATA.pbz2
    input_data_{Y}-{X}.pbz2          # one per unmasked land cell
```

The folder is created if missing; existing files with the same names are
overwritten. File contents follow the format specified in §3 and §4 above.

### 8.5 Validation (`--test`)

`--test` picks 5 random cell files from the output folder and, for each of
the 5 climate variables, compares the first 500 daily values against the
raw NetCDF re-opened from `{climate_data}/{dataset}/{mode}_raw`. A run is
clean when every line prints `PASS` and the final summary reads
`ALL TESTS PASSED`. A successful round-trip should report
`mean|err|=0.000e+00`.

### 8.6 Time handling and NetCDF time-variable requirements

The script **does not standardize time**. It does not convert calendars,
re-grid the cadence, deduplicate stamps, infer a daily axis, or rewrite the
`units` string. It only:

1. Wraps the `time` variable of each multi-file dataset in
   [`netCDF4.MFTime`](https://unidata.github.io/netcdf4-python/#MFTime) so
   that, across the segment files of a single variable, all numeric time
   values are reported against one common reference epoch (the first
   file's `units`). Single-file datasets are read as-is, with no wrapping.
2. Picks the first variable in `CLIMATE_VARS` (`hurs`) as the reference
   and asserts that **every other** climate variable's time variable has
   the *same* `units` string, the *same* `calendar`, and a numerically
   identical `time_index` array. Any mismatch aborts the run with
   `ValueError`.
3. Copies the reference variable's `standard_name`, `units`, `calendar`,
   and `time_var[:]` verbatim into the metadata file (`time_dict` of §3.2).

What this means for the input NetCDFs:

| Aspect                | Requirement                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| Variable name         | The unlimited/time variable **must** be named `time`.                       |
| Attributes on `time`  | `standard_name`, `units` (CF, e.g. `"days since 1901-01-01 00:00:00"`), `calendar` (CF, e.g. `"proleptic_gregorian"`, `"standard"`, `"noleap"`) — all read directly. |
| Cadence               | Must already be daily, contiguous, in order. The script does not check this; the model assumes daily. |
| Cross-variable axis   | All five variables (`hurs`, `tas`, `pr`, `ps`, `rsds`) must share *byte-identical* `units`, `calendar`, and time stamps after `MFTime` normalization. |
| Within-variable files | When multiple files exist for one variable (`MFDataset` path), they may carry per-file `units` strings; `MFTime` re-aligns the numeric values to the first file's epoch. They **must** still share the same `calendar`. |
| Calendar conversion   | Not performed. Whatever `calendar` the source declares propagates to the metadata file and is what the model will see. |
| Duplicates / gaps     | Not detected. The script trusts the source.                                 |
| Forbidden values      | The model has no NaN/fill handling for the time axis; ensure stamps are real numeric values, not masked. |

If your sources disagree on calendar or units, harmonise them upstream
(e.g. with CDO, NCO, or `xarray.convert_calendar`) **before** running
`pre_processing.py`.

### 8.7 Common pitfalls

- **Wrong CWD.** Run from `input/`; otherwise `./pre_processing.toml`,
  `./mask/...`, and `./soil/...` will not resolve.
- **Soil filename mismatch.** The keys under `[soil_files]` must point to
  files that actually exist in `soil_data`. The script aborts before
  writing any cell files if one is missing.
- **Climate time axes disagree.** The five climate datasets must share
  identical `units`, `calendar`, and time stamps. The script asserts this
  and aborts with a clear error otherwise.
- **Mask shape.** Must be exactly `(360, 720)`; the script refuses to
  proceed with any other shape.
- **Units.** The script copies the source values verbatim into the
  `.pbz2`. They must already be in the units listed in §4.4 — the model
  performs the unit conversions, but assumes the input is in SI as
  documented.

---

## 9. References inside this repository

- Writing script: [`input/pre_processing.py`](pre_processing.py)
- Reference cells: [`input/central/`](central/), `cax/input_data_183-257.pbz2`,
  `k34/input_data_185-240.pbz2`
- Model loader (full grid): `grd.init_caete_dyn` in [`src/caete.py`](../src/caete.py)
- Model loader (single plot from in-memory dict): `plot.init_plot` in
  [`src/caete.py`](../src/caete.py)
- Metadata reader: [`src/model_driver.py`](../src/model_driver.py)
- Unit conversions applied to the loaded arrays: see `daily_budget` /
  `bdg_spinup` in [`src/caete.py`](../src/caete.py).
