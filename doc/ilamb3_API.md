# ilamb3 API Documentation

This document provides a comprehensive reference for the ilamb3 Python package, a benchmarking framework for Earth system models.

## Package Structure

```
ilamb3/
├── __init__.py
├── cache.py              # Caching utilities
├── compare.py            # Dataset comparison utilities
├── config.py             # Configuration management
├── dataset.py            # Dataset loading and preprocessing
├── exceptions.py         # Custom exceptions
├── meta.py               # Metadata handling
├── parallel.py           # Parallel processing utilities
├── plot.py               # Visualization utilities
├── regions.py            # Geographic region handling
├── run.py                # Benchmark execution orchestration
├── analysis/             # Statistical analysis methods
│   ├── __init__.py
│   ├── accumulate.py
│   ├── area.py
│   ├── base.py
│   ├── bias.py
│   ├── cycle.py
│   ├── hydro.py
│   ├── nbp.py
│   ├── quantiles.py
│   ├── relationship.py
│   ├── rmse.py
│   ├── runoff_sensitivity.py
│   ├── spatial_distribution.py
│   └── timeseries.py
├── cli/                  # Command-line interface
│   └── __init__.py
├── configure/            # Configuration files
│   ├── ilamb.yaml
│   └── iomb.yaml
├── registry/             # Pre-defined benchmark configurations
│   ├── __init__.py
│   ├── ilamb.txt
│   ├── ilamb3.txt
│   ├── iomb.txt
│   └── test.txt
├── templates/            # HTML report templates
│   ├── __init__.py
│   ├── dataset_page.html
│   ├── directory.html
│   └── unified_dashboard.html
├── tests/                # Unit tests
│   └── ...
└── transform/            # Data transformation utilities
    ├── __init__.py
    ├── amoc.py
    ├── base.py
    ├── expression.py
    ├── gradient.py
    ├── integrate.py
    ├── ohc.py
    ├── permafrost.py
    ├── runoff_sensitivity.py
    ├── select.py
    ├── soilmoisture.py
    └── stratification_index.py
```

---

## Core Modules

### `compare.py` - Dataset Comparison Utilities

This module provides functions to align and compare datasets from different sources.

#### `adjust_lon(ds: xr.Dataset) -> xr.Dataset`

Adjust longitude coordinates from [0, 360] to [-180, 180] range if needed.

**Parameters:**
- `ds`: xarray Dataset with longitude coordinate

**Returns:**
- Dataset with adjusted longitude coordinates

**Example:**
```python
from ilamb3.compare import adjust_lon

ds = xr.open_dataset("model_output.nc")
ds_adjusted = adjust_lon(ds)
```

#### `make_comparable(ref: xr.DataArray, com: xr.DataArray, varname: str | None = None) -> tuple[xr.DataArray, xr.DataArray]`

Make two DataArrays comparable by:
1. Trimming to overlapping time period
2. Regridding to the coarser resolution
3. Converting units if needed

**Parameters:**
- `ref`: Reference DataArray (e.g., observations)
- `com`: Comparison DataArray (e.g., model output)
- `varname`: Optional variable name for unit conversion lookup

**Returns:**
- Tuple of (ref_aligned, com_aligned) DataArrays

**Example:**
```python
from ilamb3.compare import make_comparable

ref_da = ref_ds["gpp"]
com_da = model_ds["gpp"]
ref_aligned, com_aligned = make_comparable(ref_da, com_da, varname="gpp")
```

---

### `regions.py` - Geographic Region Handling

This module provides the `Regions` class for defining and managing geographic regions.

#### `class Regions`

Manages geographic regions for subsetting data.

##### `__init__(self)`

Initialize with empty region dictionary.

##### `add_latlon_bounds(self, name: str, lat_bnds: tuple[float, float], lon_bnds: tuple[float, float]) -> None`

Add a region defined by latitude/longitude bounds.

**Parameters:**
- `name`: Region identifier
- `lat_bnds`: Tuple of (min_lat, max_lat)
- `lon_bnds`: Tuple of (min_lon, max_lon)

**Example:**
```python
from ilamb3.regions import Regions

regions = Regions()
regions.add_latlon_bounds("pan_amazon", lat_bnds=(-20.5, 10.5), lon_bnds=(-80, -43))
regions.add_latlon_bounds("tropics", lat_bnds=(-23.5, 23.5), lon_bnds=(-180, 180))
```

##### `restrict_to_region(self, ds: xr.Dataset | xr.DataArray, region: str) -> xr.Dataset | xr.DataArray`

Subset dataset to specified region.

**Parameters:**
- `ds`: Dataset or DataArray to subset
- `region`: Region name (must be previously added)

**Returns:**
- Subsetted Dataset or DataArray

**Example:**
```python
amazon_data = regions.restrict_to_region(global_data, "pan_amazon")
```

##### `get_mask(self, ds: xr.Dataset | xr.DataArray, region: str) -> xr.DataArray`

Get boolean mask for region.

**Parameters:**
- `ds`: Dataset or DataArray to create mask for
- `region`: Region name

**Returns:**
- Boolean DataArray mask (True inside region)

**Example:**
```python
mask = regions.get_mask(ds, "pan_amazon")
masked_data = ds.where(mask)
```

---

### `dataset.py` - Dataset Loading and Preprocessing

#### `open_dataset(path: str | Path, variable: str | None = None, **kwargs) -> xr.Dataset`

Open a dataset with automatic preprocessing. Handles time decoding issues, unit conversions, etc.

**Parameters:**
- `path`: Path to NetCDF file
- `variable`: Optional variable name to extract
- `**kwargs`: Additional arguments passed to `xr.open_dataset`

**Returns:**
- Preprocessed xarray Dataset

**Example:**
```python
from ilamb3.dataset import open_dataset

ds = open_dataset("observations/gpp_FLUXCOM.nc", variable="gpp")
```

#### `get_variable(ds: xr.Dataset, varname: str, synonyms: list[str] | None = None) -> xr.DataArray`

Get variable by name, checking synonyms if not found.

**Parameters:**
- `ds`: Dataset to search
- `varname`: Primary variable name
- `synonyms`: List of alternative names to try

**Returns:**
- DataArray for the variable

**Example:**
```python
from ilamb3.dataset import get_variable

# Will try "gpp", then "GPP", then "gross_primary_productivity"
gpp = get_variable(ds, "gpp", synonyms=["GPP", "gross_primary_productivity"])
```

---

## Analysis Modules

All analysis classes inherit from `AnalysisBase` and follow a consistent interface.

### `analysis/base.py` - Base Analysis Class

#### `class AnalysisBase`

Base class for all analysis methods.

**Attributes:**
- `name`: str - Analysis identifier

**Methods:**

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, **kwargs) -> xr.Dataset`

Perform the analysis.

**Parameters:**
- `ref`: Reference DataArray (observations)
- `com`: Comparison DataArray (model)
- `**kwargs`: Analysis-specific parameters

**Returns:**
- Dataset containing analysis results

---

### `analysis/bias.py` - Bias Analysis

#### `class Bias(AnalysisBase)`

Compute bias between reference and comparison data.

**Attributes:**
- `name = "bias"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, region: str | None = None, skip_time_integral: bool = False) -> xr.Dataset`

Compute bias metrics.

**Parameters:**
- `ref`: Reference DataArray
- `com`: Comparison DataArray
- `region`: Optional region name for masking
- `skip_time_integral`: If True, skip time averaging

**Returns:**
- Dataset with variables:
  - `bias_map`: Spatial map of mean bias (com - ref)
  - `bias_score`: Scalar score [0, 1] (1 = perfect)
  - `ref_mean`: Reference time mean
  - `com_mean`: Comparison time mean

**Example:**
```python
from ilamb3.analysis.bias import Bias

bias_analysis = Bias()
results = bias_analysis(ref_da, com_da, region="pan_amazon")

print(f"Bias Score: {float(results['bias_score']):.3f}")
bias_map = results["bias_map"]
```

#### `compute_bias(ref: xr.DataArray, com: xr.DataArray, **kwargs) -> xr.Dataset`

Functional interface to Bias analysis.

```python
from ilamb3.analysis.bias import compute_bias

results = compute_bias(ref_da, com_da, region="tropics")
```

---

### `analysis/rmse.py` - Root Mean Square Error

#### `class RMSE(AnalysisBase)`

Compute Root Mean Square Error.

**Attributes:**
- `name = "rmse"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, region: str | None = None) -> xr.Dataset`

Compute RMSE metrics.

**Parameters:**
- `ref`: Reference DataArray
- `com`: Comparison DataArray
- `region`: Optional region name

**Returns:**
- Dataset with variables:
  - `rmse_map`: Spatial map of RMSE
  - `rmse_score`: Scalar score [0, 1]
  - `crmse_map`: Centered RMSE (bias removed)

**Example:**
```python
from ilamb3.analysis.rmse import RMSE

rmse_analysis = RMSE()
results = rmse_analysis(ref_da, com_da)

print(f"RMSE Score: {float(results['rmse_score']):.3f}")
```

#### `compute_rmse(ref: xr.DataArray, com: xr.DataArray, **kwargs) -> xr.Dataset`

Functional interface to RMSE analysis.

---

### `analysis/cycle.py` - Seasonal Cycle Analysis

#### `class SeasonalCycle(AnalysisBase)`

Analyze seasonal/annual cycle.

**Attributes:**
- `name = "cycle"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, region: str | None = None) -> xr.Dataset`

Compute seasonal cycle metrics.

**Parameters:**
- `ref`: Reference DataArray
- `com`: Comparison DataArray
- `region`: Optional region name

**Returns:**
- Dataset with variables:
  - `ref_cycle`: Reference monthly climatology (12 months)
  - `com_cycle`: Comparison monthly climatology
  - `cycle_score`: Phase and amplitude score [0, 1]
  - `phase_shift`: Difference in peak month
  - `amplitude_ratio`: Ratio of amplitudes

**Example:**
```python
from ilamb3.analysis.cycle import SeasonalCycle

cycle_analysis = SeasonalCycle()
results = cycle_analysis(ref_da, com_da, region="pan_amazon")

print(f"Cycle Score: {float(results['cycle_score']):.3f}")
print(f"Phase Shift: {float(results['phase_shift']):.1f} months")
```

#### `compute_cycle(ref: xr.DataArray, com: xr.DataArray, **kwargs) -> xr.Dataset`

Functional interface to SeasonalCycle analysis.

---

### `analysis/spatial_distribution.py` - Spatial Pattern Analysis

#### `class SpatialDistribution(AnalysisBase)`

Analyze spatial patterns.

**Attributes:**
- `name = "spatial_distribution"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, region: str | None = None) -> xr.Dataset`

Compute spatial distribution metrics.

**Parameters:**
- `ref`: Reference DataArray
- `com`: Comparison DataArray
- `region`: Optional region name

**Returns:**
- Dataset with variables:
  - `spatial_correlation`: Pattern correlation coefficient
  - `spatial_std_ratio`: Standard deviation ratio (com/ref)
  - `spatial_score`: Combined spatial score [0, 1]

**Example:**
```python
from ilamb3.analysis.spatial_distribution import SpatialDistribution

spatial_analysis = SpatialDistribution()
results = spatial_analysis(ref_da, com_da)

print(f"Spatial Correlation: {float(results['spatial_correlation']):.3f}")
print(f"Spatial Score: {float(results['spatial_score']):.3f}")
```

---

### `analysis/timeseries.py` - Time Series Analysis

#### `class Timeseries(AnalysisBase)`

Analyze time series behavior.

**Attributes:**
- `name = "timeseries"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, region: str | None = None) -> xr.Dataset`

Compute time series metrics.

**Parameters:**
- `ref`: Reference DataArray
- `com`: Comparison DataArray
- `region`: Optional region name

**Returns:**
- Dataset with variables:
  - `ref_timeseries`: Reference area-weighted mean time series
  - `com_timeseries`: Comparison time series
  - `temporal_correlation`: Correlation coefficient
  - `temporal_std_ratio`: Standard deviation ratio
  - `interannual_variability_score`: IAV score [0, 1]

**Example:**
```python
from ilamb3.analysis.timeseries import Timeseries

ts_analysis = Timeseries()
results = ts_analysis(ref_da, com_da, region="pan_amazon")

print(f"Temporal Correlation: {float(results['temporal_correlation']):.3f}")
```

---

### `analysis/quantiles.py` - Quantile Analysis

#### `class Quantiles(AnalysisBase)`

Quantile-based comparison.

**Attributes:**
- `name = "quantiles"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> xr.Dataset`

Compute quantile metrics.

**Parameters:**
- `ref`: Reference DataArray
- `com`: Comparison DataArray
- `quantiles`: List of quantile values to compute

**Returns:**
- Dataset with variables:
  - `ref_quantiles`: Reference quantile values
  - `com_quantiles`: Comparison quantile values
  - `quantile_score`: Score based on quantile matching

**Example:**
```python
from ilamb3.analysis.quantiles import Quantiles

quant_analysis = Quantiles()
results = quant_analysis(ref_da, com_da, quantiles=[0.1, 0.5, 0.9])
```

---

### `analysis/relationship.py` - Variable Relationship Analysis

#### `class Relationship(AnalysisBase)`

Analyze relationship between two variables.

**Attributes:**
- `name = "relationship"`

##### `__call__(self, ref_x: xr.DataArray, ref_y: xr.DataArray, com_x: xr.DataArray, com_y: xr.DataArray) -> xr.Dataset`

Compare relationships between variable pairs.

**Parameters:**
- `ref_x`: Reference independent variable
- `ref_y`: Reference dependent variable
- `com_x`: Comparison independent variable
- `com_y`: Comparison dependent variable

**Returns:**
- Dataset with variables:
  - `ref_slope`: Reference regression slope
  - `com_slope`: Comparison regression slope
  - `relationship_score`: Score based on slope similarity

**Example:**
```python
from ilamb3.analysis.relationship import Relationship

rel_analysis = Relationship()
results = rel_analysis(
    ref_precip, ref_gpp,
    com_precip, com_gpp
)
print(f"Relationship Score: {float(results['relationship_score']):.3f}")
```

---

### `analysis/hydro.py` - Hydrology Analysis

#### `class HydrologicalCycle(AnalysisBase)`

Analyze hydrological variables (runoff, ET, etc.).

**Attributes:**
- `name = "hydro"`

##### `__call__(self, ref: xr.DataArray, com: xr.DataArray, precip_ref: xr.DataArray | None = None, precip_com: xr.DataArray | None = None) -> xr.Dataset`

Compute hydrological metrics.

**Parameters:**
- `ref`: Reference hydrological variable (runoff, ET)
- `com`: Comparison hydrological variable
- `precip_ref`: Optional reference precipitation
- `precip_com`: Optional comparison precipitation

**Returns:**
- Dataset with:
  - Water balance metrics
  - Runoff ratio if precipitation provided

**Example:**
```python
from ilamb3.analysis.hydro import HydrologicalCycle

hydro_analysis = HydrologicalCycle()
results = hydro_analysis(
    ref_runoff, com_runoff,
    precip_ref=ref_precip,
    precip_com=com_precip
)
```

---

### `analysis/nbp.py` - Net Biome Production Analysis

#### `class NBP(AnalysisBase)`

Analyze Net Biome Production.

**Attributes:**
- `name = "nbp"`

Specialized analysis for carbon flux variables with cumulative and trend analysis.

---

## Transform Modules

Transform modules provide data manipulation utilities.

### `transform/base.py` - Base Transform Class

#### `class TransformBase`

Base class for data transformations.

**Attributes:**
- `name`: str - Transform identifier

**Methods:**

##### `__call__(self, ds: xr.Dataset, varname: str, **kwargs) -> xr.Dataset`

Apply transformation to dataset.

---

### `transform/integrate.py` - Integration Transforms

#### `class IntegrateTime(TransformBase)`

Integrate over time dimension.

**Attributes:**
- `name = "integrate_time"`

##### `__call__(self, ds: xr.Dataset, varname: str, method: str = "trapz") -> xr.Dataset`

Integrate variable over time.

**Parameters:**
- `ds`: Input Dataset
- `varname`: Variable to integrate
- `method`: Integration method ("trapz" or "sum")

**Returns:**
- Dataset with time-integrated variable

**Example:**
```python
from ilamb3.transform.integrate import IntegrateTime

integrator = IntegrateTime()
cumulative_gpp = integrator(ds, "gpp", method="trapz")
```

#### `class IntegrateSpace(TransformBase)`

Integrate over spatial dimensions.

**Attributes:**
- `name = "integrate_space"`

##### `__call__(self, ds: xr.Dataset, varname: str, region: str | None = None) -> xr.Dataset`

Integrate variable over lat/lon (area-weighted).

**Parameters:**
- `ds`: Input Dataset
- `varname`: Variable to integrate
- `region`: Optional region to integrate over

**Returns:**
- Dataset with spatially-integrated variable (time series)

**Example:**
```python
from ilamb3.transform.integrate import IntegrateSpace

integrator = IntegrateSpace()
total_gpp = integrator(ds, "gpp", region="pan_amazon")
```

---

### `transform/select.py` - Selection Transforms

#### `class SelectTime(TransformBase)`

Select time subset.

##### `__call__(self, ds: xr.Dataset, varname: str, start: str | None = None, end: str | None = None, months: list[int] | None = None) -> xr.Dataset`

Select time range or specific months.

**Parameters:**
- `ds`: Input Dataset
- `varname`: Variable name
- `start`: Start date string (e.g., "2000-01-01")
- `end`: End date string
- `months`: List of months to select (e.g., [6, 7, 8] for JJA)

**Returns:**
- Dataset with selected time range

**Example:**
```python
from ilamb3.transform.select import SelectTime

selector = SelectTime()
# Select 2000-2010
subset = selector(ds, "gpp", start="2000-01-01", end="2010-12-31")
# Select only summer months
summer = selector(ds, "gpp", months=[6, 7, 8])
```

#### `class SelectRegion(TransformBase)`

Select spatial subset.

##### `__call__(self, ds: xr.Dataset, varname: str, region: str, regions: Regions | None = None) -> xr.Dataset`

Select spatial region.

**Parameters:**
- `ds`: Input Dataset
- `varname`: Variable name
- `region`: Region name
- `regions`: Regions object (uses global if None)

**Returns:**
- Dataset subset to region

---

## Plotting Module

### `plot.py` - Visualization Utilities

#### `plot_bias_map(bias: xr.DataArray, ax: plt.Axes | None = None, cmap: str = "RdBu_r", **kwargs) -> plt.Axes`

Plot spatial bias map.

**Parameters:**
- `bias`: Bias DataArray to plot
- `ax`: Optional matplotlib Axes
- `cmap`: Colormap name
- `**kwargs`: Additional plot arguments

**Returns:**
- Matplotlib Axes object

**Example:**
```python
from ilamb3.plot import plot_bias_map
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
plot_bias_map(results["bias_map"], ax=ax, cmap="RdBu_r")
plt.savefig("bias_map.png")
```

#### `plot_cycle(ref_cycle: xr.DataArray, com_cycle: xr.DataArray, ax: plt.Axes | None = None, **kwargs) -> plt.Axes`

Plot seasonal cycles.

**Parameters:**
- `ref_cycle`: Reference monthly climatology
- `com_cycle`: Comparison monthly climatology
- `ax`: Optional matplotlib Axes
- `**kwargs`: Additional plot arguments

**Returns:**
- Matplotlib Axes object

**Example:**
```python
from ilamb3.plot import plot_cycle

fig, ax = plt.subplots()
plot_cycle(results["ref_cycle"], results["com_cycle"], ax=ax)
plt.savefig("seasonal_cycle.png")
```

#### `plot_timeseries(ref_ts: xr.DataArray, com_ts: xr.DataArray, ax: plt.Axes | None = None, **kwargs) -> plt.Axes`

Plot time series comparison.

**Parameters:**
- `ref_ts`: Reference time series
- `com_ts`: Comparison time series
- `ax`: Optional matplotlib Axes

**Returns:**
- Matplotlib Axes object

#### `plot_spatial_correlation(ref: xr.DataArray, com: xr.DataArray, ax: plt.Axes | None = None, **kwargs) -> plt.Axes`

Plot Taylor diagram or spatial correlation scatter.

---

## Run Module

### `run.py` - Benchmark Orchestration

#### `class Benchmark`

Run a complete benchmark comparison.

##### `__init__(self, ref_path: str | Path, com_path: str | Path, variable: str, analyses: list[str] = ["bias", "rmse", "cycle", "spatial_distribution"], region: str | None = None)`

Initialize benchmark configuration.

**Parameters:**
- `ref_path`: Path to reference dataset
- `com_path`: Path to comparison dataset
- `variable`: Variable name to analyze
- `analyses`: List of analysis names to run
- `region`: Optional region name

##### `run(self) -> xr.Dataset`

Execute all analyses.

**Returns:**
- Dataset containing all analysis results and scores

##### `to_html(self, output_dir: str | Path) -> None`

Generate HTML report.

**Example:**
```python
from ilamb3.run import Benchmark

benchmark = Benchmark(
    ref_path="observations/gpp_FLUXCOM.nc",
    com_path="model/gpp_output.nc",
    variable="gpp",
    analyses=["bias", "rmse", "cycle", "spatial_distribution"],
    region="pan_amazon"
)

results = benchmark.run()
print(f"Overall Score: {float(results['overall_score']):.3f}")

benchmark.to_html("benchmark_results/")
```

---

## Complete Example: CAETE Model Benchmarking

```python
"""Complete example of benchmarking CAETE model output against observations."""

import xarray as xr
from pathlib import Path

# ilamb3 imports
from ilamb3.compare import adjust_lon, make_comparable
from ilamb3.regions import Regions
from ilamb3.analysis.bias import Bias
from ilamb3.analysis.rmse import RMSE
from ilamb3.analysis.cycle import SeasonalCycle
from ilamb3.analysis.spatial_distribution import SpatialDistribution
from ilamb3.analysis.timeseries import Timeseries
from ilamb3.plot import plot_bias_map, plot_cycle, plot_timeseries

# Define regions
regions = Regions()
regions.add_latlon_bounds("pan_amazon", lat_bnds=(-20.5, 10.5), lon_bnds=(-80, -43))


def load_and_prepare_data(model_path: str, ref_path: str, variable: str):
    """Load and prepare model and reference datasets."""

    # Load datasets
    model_ds = xr.open_dataset(model_path)
    ref_ds = xr.open_dataset(ref_path)

    # Adjust longitudes if needed
    model_ds = adjust_lon(model_ds)
    ref_ds = adjust_lon(ref_ds)

    # Get DataArrays
    model_da = model_ds[variable]
    ref_da = ref_ds[variable]

    # Make comparable (align time, regrid)
    ref_aligned, model_aligned = make_comparable(ref_da, model_da, varname=variable)

    return ref_aligned, model_aligned


def run_full_benchmark(ref_da: xr.DataArray, com_da: xr.DataArray,
                       region: str = "pan_amazon") -> dict:
    """Run all benchmark analyses."""

    results = {}

    # Bias analysis
    bias_analysis = Bias()
    results["bias"] = bias_analysis(ref_da, com_da, region=region)

    # RMSE analysis
    rmse_analysis = RMSE()
    results["rmse"] = rmse_analysis(ref_da, com_da, region=region)

    # Seasonal cycle
    cycle_analysis = SeasonalCycle()
    results["cycle"] = cycle_analysis(ref_da, com_da, region=region)

    # Spatial distribution
    spatial_analysis = SpatialDistribution()
    results["spatial"] = spatial_analysis(ref_da, com_da, region=region)

    # Time series
    ts_analysis = Timeseries()
    results["timeseries"] = ts_analysis(ref_da, com_da, region=region)

    # Compute overall score
    scores = [
        float(results["bias"]["bias_score"]),
        float(results["rmse"]["rmse_score"]),
        float(results["cycle"]["cycle_score"]),
        float(results["spatial"]["spatial_score"]),
    ]
    results["overall_score"] = sum(scores) / len(scores)

    return results


def print_benchmark_summary(results: dict):
    """Print summary of benchmark results."""

    print("=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    print(f"Bias Score:    {float(results['bias']['bias_score']):.3f}")
    print(f"RMSE Score:    {float(results['rmse']['rmse_score']):.3f}")
    print(f"Cycle Score:   {float(results['cycle']['cycle_score']):.3f}")
    print(f"Spatial Score: {float(results['spatial']['spatial_score']):.3f}")
    print("-" * 50)
    print(f"OVERALL SCORE: {results['overall_score']:.3f}")
    print("=" * 50)


# Main execution
if __name__ == "__main__":
    # Paths
    model_path = "output/caete_gpp.nc"
    ref_path = "benchmark_data/custom_data/gpp/FLUXCOM/gpp_FLUXCOM.nc"

    # Load and prepare data
    ref_da, model_da = load_and_prepare_data(model_path, ref_path, "gpp")

    # Run benchmark
    results = run_full_benchmark(ref_da, model_da, region="pan_amazon")

    # Print summary
    print_benchmark_summary(results)

    # Generate plots
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_bias_map(results["bias"]["bias_map"], ax=axes[0, 0])
    axes[0, 0].set_title("Bias Map")

    plot_cycle(results["cycle"]["ref_cycle"], results["cycle"]["com_cycle"], ax=axes[0, 1])
    axes[0, 1].set_title("Seasonal Cycle")

    plot_timeseries(results["timeseries"]["ref_timeseries"],
                    results["timeseries"]["com_timeseries"], ax=axes[1, 0])
    axes[1, 0].set_title("Time Series")

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
```

---

## Score Interpretation

All scores in ilamb3 are normalized to [0, 1] range:

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0   | Excellent agreement |
| 0.7 - 0.9   | Good agreement |
| 0.5 - 0.7   | Moderate agreement |
| 0.3 - 0.5   | Poor agreement |
| 0.0 - 0.3   | Very poor agreement |

---

## References

- ILAMB: International Land Model Benchmarking
- ilamb3 GitHub repository
- Collier, N., et al. (2018). The International Land Model Benchmarking (ILAMB) System: Design, Theory, and Implementation. JAMES.
