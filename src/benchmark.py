"""Benchmark script for CAETE model evaluation.

This module provides functions to load, preprocess, and compare CAETE model outputs
with reference datasets for benchmarking purposes.

Requires CDO to be installed and accessible in the system path.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import numpy.ma as ma
import pandas as pd
import ilamb3
import ilamb3.compare as cmp
import ilamb3.dataset as dset
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ilamb3.analysis.bias import bias_analysis
from ilamb3.analysis.cycle import cycle_analysis
from ilamb3.analysis.spatial_distribution import spatial_distribution_analysis

from benchmark_pp import get_dataset, get_caete_dataset, monsum
from benchmark_utils import available_variables as caete_data
from benchmark_utils import ref_datasets, get_caete_varname
from benchmark_utils import BENCHMARCK_CACHE_DIR, ensure_cache_dir
import _geos as geos

ensure_cache_dir()

# RAISG mask file for Pan Amazon region
RAISG_MASK_FILE = Path("../input/mask/mask.nc")

# Define benchmark region(s)
Regions = ilamb3.regions.Regions()

# Pan Amazon region bounds from _geos module
_pana_bbox = geos.pan_amazon_bbox
lons = [_pana_bbox["west"], _pana_bbox["east"]]
lats = [_pana_bbox["south"], _pana_bbox["north"]]

# Add Pan Amazon region to Regions
Regions.add_latlon_bounds(
    label="pana",
    name="Pan Amazon",
    lats=lats,
    lons=lons,
    source="Pan Amazon (Northern South America)"
)


# Variable-specific preprocessing configurations
# Format: variable -> (conversion_factor, units)
VARIABLE_PREPROCESSING = {
    "gpp": (monsum, 0.00273791, "kg*m**-2*month**-1"),   # daily kgC/m2/year -> monthly kgC/m2/month
    "rnpp": (monsum, 0.001, "kg*m**-2*month**-1"),       # daily gC/m2/day -> monthly kgC/m2/month
    "et": (monsum, 1, "mm*month**-1"),                   # daily mm/day -> monthly mm/month
}


def get_model_data(variable: str, experiment: str | None = None, use_cache: bool = True) -> Tuple[xr.Dataset, str, xr.Dataset]:
    """Load and preprocess CAETE model output with caching support.

    Retrieves CAETE model output for the specified variable, applies necessary
    unit conversions, and extracts the model mask from the data.

    Args:
        variable: Variable name to read from CAETE output (e.g., "gpp", "rnpp", "et").
        experiment: Experiment name. If None, uses the first available experiment.
        use_cache: If True, use cached preprocessed file if available. Default is True.

    Returns:
        A tuple containing:
            - model_dataset (xr.Dataset): Preprocessed CAETE dataset.
            - experiment (str): Experiment identifier extracted from filename.
            - model_mask (xr.Dataset): Mask dataset indicating valid data cells.

    Raises:
        ValueError: If the variable is not supported for preprocessing.
        RuntimeError: If the dataset fails to open.
    """
    # Get CAETE variable name mapping and file path
    caete_varname = get_caete_varname(variable)
    model_filepath, experiment = get_caete_dataset(caete_data, caete_varname, experiment)

    # Define cached file path
    cached_filepath = BENCHMARCK_CACHE_DIR / Path(f"{variable}_{experiment}.nc")

    # Preprocess CAETE output if needed
    if not cached_filepath.exists() or not use_cache:
        if variable not in VARIABLE_PREPROCESSING:
            raise ValueError(f"Variable '{variable}' not supported for preprocessing. "
                           f"Supported variables: {list(VARIABLE_PREPROCESSING.keys())}")
        preprocess_func, conv_factor, units = VARIABLE_PREPROCESSING[variable]
        preprocess_func(model_filepath, cached_filepath, variable, conv_factor, units)
    else:
        print(f"Using cached file {cached_filepath}")

    # Open model dataset
    try:
        model_ds = xr.open_dataset(cached_filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to open cached CAETE dataset: {e}")

    # Extract model mask from first timestep
    model_mask = _extract_model_mask(model_ds, variable)

    return model_ds, experiment, model_mask


def _extract_model_mask(model_ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Extract a mask dataset from model data indicating valid cells.

    Args:
        model_ds: Model dataset to extract mask from.
        variable: Variable name to use for mask extraction.

    Returns:
        A Dataset with a 'mask' variable indicating invalid (True) or valid (False) cells.
    """
    data_values = model_ds[variable].isel(time=0).values

    if ma.is_masked(data_values):
        mask_array = ma.getmask(data_values)
    else:
        mask_array = np.isnan(data_values)

    model_mask = xr.DataArray(
        mask_array,
        coords={'lat': model_ds.lat, 'lon': model_ds.lon},
        dims=['lat', 'lon'],
        name='mask'
    ).to_dataset()

    return model_mask


def get_reference_data(dataset_name: str, filename: str) -> xr.Dataset:
    """Load a reference dataset for benchmarking.

    Args:
        dataset_name: Name of the reference dataset (e.g., "MADANI", "GLEAMv3.3a").
        filename: Filename within the dataset directory (e.g., "gpp_masked.nc").

    Returns:
        The reference dataset as an xarray Dataset.

    Raises:
        ValueError: If the dataset or file is not found.
        RuntimeError: If the dataset fails to open.
    """
    ref_filepath = get_dataset(ref_datasets, dataset_name, filename)

    try:
        ref_ds = xr.open_dataset(ref_filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to open reference dataset: {e}")

    return ref_ds


def get_region_mask(region: str = "pana") -> xr.Dataset:
    """Load the RAISG mask file restricted to a specified region.

    Args:
        region: Region label to restrict the mask to. Default is "pana" (Pan Amazon).

    Returns:
        The mask dataset restricted to the specified region.

    Raises:
        RuntimeError: If the mask file fails to open.
    """
    try:
        mask_ds = xr.open_dataset(RAISG_MASK_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to open RAISG mask file: {e}")

    mask_ds = Regions.restrict_to_region(mask_ds, region)

    return mask_ds


def conform_datasets(
    model_ds: xr.Dataset,
    ref_ds: xr.Dataset,
    variable: str,
    region: str = "pana"
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Align and make model and reference datasets comparable.

    Performs longitude adjustment, temporal/spatial alignment via ilamb3,
    and restricts both datasets to the specified region.

    Args:
        model_ds: Model dataset to conform.
        ref_ds: Reference dataset to conform.
        variable: Variable name for comparison.
        region: Region label to restrict datasets to. Default is "pana".

    Returns:
        A tuple containing:
            - conformed_model_ds (xr.Dataset): Aligned model dataset.
            - conformed_ref_ds (xr.Dataset): Aligned reference dataset.
    """
    # Adjust longitude alignment
    ref_ds, model_ds = ilamb3.compare.adjust_lon(ref_ds, model_ds)

    # Make datasets comparable (temporal alignment, spatial regridding)
    ref_ds, model_ds = ilamb3.compare.make_comparable(ref_ds, model_ds, variable)

    # Restrict both datasets to the specified region
    ref_ds = Regions.restrict_to_region(ref_ds, region)
    model_ds = Regions.restrict_to_region(model_ds, region)

    return model_ds, ref_ds


def get_model_and_ref(
    variable: str = "gpp",
    dataset_name: str = "GOSIF",
    filename: str = "gpp_GOSIF_2000-2024.nc",
    experiment: str | None = None
) -> Tuple[xr.Dataset, xr.Dataset, str, xr.Dataset, xr.Dataset]:
    """Orchestrator function to load and prepare model and reference datasets for comparison.

    This function combines the functionality of get_model_data, get_reference_data,
    conform_datasets, and get_region_mask to provide a complete workflow for
    benchmarking CAETE model outputs against reference datasets.

    Args:
        variable: Variable name to read from CAETE output. Default is "gpp".
        dataset_name: Name of the reference dataset. Default is "MADANI".
        filename: Filename within the reference datasets. Default is "gpp_masked.nc".
        experiment: Experiment name. If None, uses the first available experiment.

    Returns:
        A tuple containing:
            - model_ds (xr.Dataset): Preprocessed and conformed CAETE dataset.
            - ref_ds (xr.Dataset): Preprocessed and conformed reference dataset.
            - experiment (str): Experiment identifier from CAETE output.
            - region_mask (xr.Dataset): RAISG mask for the Pan Amazon region.
            - model_mask (xr.Dataset): Mask indicating valid model data cells.
    """
    # Step 1: Get model data with mask
    model_ds, experiment, model_mask = get_model_data(variable, experiment)

    # Step 2: Get reference data
    ref_ds = get_reference_data(dataset_name, filename)

    # Step 3: Conform datasets (align, regrid, restrict to region)
    model_ds, ref_ds = conform_datasets(model_ds, ref_ds, variable)

    # Step 4: Get region mask (RAISG)
    region_mask = get_region_mask()

    return model_ds, ref_ds, experiment, region_mask, model_mask


def temporal_analysis(
    model_ds: xr.Dataset,
    ref_ds: xr.Dataset,
    variable: str,
    region: str = "pana",
    output_dir: Path | None = None
) -> Dict[str, Any]:
    """Perform comprehensive temporal analysis between model and reference datasets.

    This function applies ILAMB3 analysis methods to compare temporal characteristics
    of model output against reference data, including bias, seasonal cycle, and
    time series analysis.

    Args:
        model_ds: Model dataset (CAETE output).
        ref_ds: Reference dataset (observations).
        variable: Variable name to analyze (e.g., "gpp", "et").
        region: Region label for analysis. Default is "pana" (Pan Amazon).
        output_dir: Optional directory to save plots. If None, plots are displayed.

    Returns:
        dict: Dictionary containing:
            - 'scalars': pd.DataFrame with scalar metrics and scores
            - 'ref_gridded': xr.Dataset with reference gridded outputs
            - 'com_gridded': xr.Dataset with comparison gridded outputs
            - 'figures': dict of matplotlib Figure objects
    """
    results = {
        'scalars': pd.DataFrame(),
        'ref_gridded': xr.Dataset(),
        'com_gridded': xr.Dataset(),
        'figures': {}
    }

    # Ensure units attribute exists
    if 'units' not in ref_ds[variable].attrs:
        ref_ds[variable].attrs['units'] = '1'
    if 'units' not in model_ds[variable].attrs:
        model_ds[variable].attrs['units'] = ref_ds[variable].attrs['units']

    # =========================================================================
    # 1. BIAS ANALYSIS
    # =========================================================================
    print("Running bias analysis...")
    try:
        bias_analyzer = bias_analysis(
            required_variable=variable,
            regions=[region],
            use_uncertainty=False,
            mass_weighting=False
        )
        bias_df, bias_ref, bias_com = bias_analyzer(ref_ds, model_ds)
        results['scalars'] = pd.concat([results['scalars'], bias_df], ignore_index=True)

        # Store gridded outputs
        for var in bias_ref.data_vars:
            results['ref_gridded'][f'bias_{var}'] = bias_ref[var]
        for var in bias_com.data_vars:
            results['com_gridded'][f'bias_{var}'] = bias_com[var]

        # Create bias map figure
        if 'bias' in bias_com:
            fig_bias = _plot_bias_map(
                bias_com['bias'],
                title=f'{variable.upper()} Bias (Model - Reference)',
                region=region
            )
            results['figures']['bias_map'] = fig_bias

    except Exception as e:
        print(f"Bias analysis failed: {e}")

    # =========================================================================
    # 2. SEASONAL CYCLE ANALYSIS
    # =========================================================================
    print("Running seasonal cycle analysis...")
    try:
        cycle_analyzer = cycle_analysis(
            required_variable=variable,
            regions=[region]
        )
        cycle_df, cycle_ref, cycle_com = cycle_analyzer(ref_ds, model_ds)
        results['scalars'] = pd.concat([results['scalars'], cycle_df], ignore_index=True)

        # Store gridded outputs
        for var in cycle_ref.data_vars:
            results['ref_gridded'][f'cycle_{var}'] = cycle_ref[var]
        for var in cycle_com.data_vars:
            results['com_gridded'][f'cycle_{var}'] = cycle_com[var]

        # Create seasonal cycle figure
        ref_cycle_var = f'cycle_{region}'
        if ref_cycle_var in cycle_ref and ref_cycle_var in cycle_com:
            fig_cycle = _plot_seasonal_cycle(
                cycle_ref[ref_cycle_var],
                cycle_com[ref_cycle_var],
                variable=variable,
                region=region
            )
            results['figures']['seasonal_cycle'] = fig_cycle

        # Create phase shift map
        if 'shift' in cycle_com:
            fig_phase = _plot_phase_shift_map(
                cycle_com['shift'],
                title=f'{variable.upper()} Phase Shift',
                region=region
            )
            results['figures']['phase_shift'] = fig_phase

    except Exception as e:
        print(f"Seasonal cycle analysis failed: {e}")

    # =========================================================================
    # 3. TIME SERIES ANALYSIS
    # =========================================================================
    print("Running time series analysis...")
    try:
        # Compute area-weighted time series for the region
        ref_ts, model_ts = _compute_regional_timeseries(ref_ds, model_ds, variable, region)

        # Compute time series metrics
        ts_metrics = _compute_timeseries_metrics(ref_ts, model_ts)

        # Add to scalars
        ts_df = pd.DataFrame([
            {
                'source': 'Reference',
                'region': region,
                'analysis': 'Time Series',
                'name': 'Period Mean',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': ts_metrics['ref_mean']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Time Series',
                'name': 'Period Mean',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': ts_metrics['com_mean']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Time Series',
                'name': 'Bias',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': ts_metrics['bias']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Time Series',
                'name': 'RMSE',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': ts_metrics['rmse']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Time Series',
                'name': 'Correlation',
                'type': 'scalar',
                'units': '1',
                'value': ts_metrics['correlation']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Time Series',
                'name': 'Normalized Std Dev',
                'type': 'scalar',
                'units': '1',
                'value': ts_metrics['norm_std']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Time Series',
                'name': 'Taylor Score',
                'type': 'score',
                'units': '1',
                'value': ts_metrics['taylor_score']
            },
        ])
        results['scalars'] = pd.concat([results['scalars'], ts_df], ignore_index=True)

        # Store time series
        results['ref_gridded']['timeseries'] = ref_ts
        results['com_gridded']['timeseries'] = model_ts

        # Create time series figure
        fig_ts = _plot_timeseries(
            ref_ts, model_ts,
            variable=variable,
            region=region,
            metrics=ts_metrics
        )
        results['figures']['timeseries'] = fig_ts

    except Exception as e:
        print(f"Time series analysis failed: {e}")

    # =========================================================================
    # 4. INTERANNUAL VARIABILITY ANALYSIS
    # =========================================================================
    print("Running interannual variability analysis...")
    try:
        iav_results = _compute_interannual_variability(ref_ds, model_ds, variable, region)

        iav_df = pd.DataFrame([
            {
                'source': 'Reference',
                'region': region,
                'analysis': 'Interannual Variability',
                'name': 'IAV Std Dev',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': iav_results['ref_iav_std']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Interannual Variability',
                'name': 'IAV Std Dev',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': iav_results['com_iav_std']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Interannual Variability',
                'name': 'IAV Score',
                'type': 'score',
                'units': '1',
                'value': iav_results['iav_score']
            },
        ])
        results['scalars'] = pd.concat([results['scalars'], iav_df], ignore_index=True)

        # Create IAV figure
        fig_iav = _plot_interannual_variability(
            iav_results['ref_annual'],
            iav_results['com_annual'],
            variable=variable,
            region=region
        )
        results['figures']['interannual_variability'] = fig_iav

    except Exception as e:
        print(f"Interannual variability analysis failed: {e}")

    # =========================================================================
    # SAVE PLOTS IF OUTPUT DIRECTORY PROVIDED
    # =========================================================================
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in results['figures'].items():
            fig.savefig(output_dir / f'{variable}_{name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {output_dir / f'{variable}_{name}.png'}")
        # Clear figures from results after saving to free memory
        results['figures'] = {}
    else:
        # Close all figures to prevent memory leak when not saving
        for fig in results['figures'].values():
            plt.close(fig)
        results['figures'] = {}

    # Print summary
    _print_temporal_analysis_summary(results['scalars'], variable, region)

    return results


# =============================================================================
# HELPER FUNCTIONS FOR TEMPORAL ANALYSIS
# =============================================================================

def _compute_regional_timeseries(
    ref_ds: xr.Dataset,
    model_ds: xr.Dataset,
    variable: str,
    region: str
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute area-weighted regional mean time series."""
    # Restrict to region
    ref_regional = Regions.restrict_to_region(ref_ds, region)
    model_regional = Regions.restrict_to_region(model_ds, region)

    # Compute cell areas for weighting
    ref_weights = dset.compute_cell_measures(ref_regional)
    model_weights = dset.compute_cell_measures(model_regional)

    # Compute weighted mean over space
    ref_ts = ref_regional[variable].weighted(ref_weights.fillna(0)).mean(dim=['lat', 'lon'])
    model_ts = model_regional[variable].weighted(model_weights.fillna(0)).mean(dim=['lat', 'lon'])

    return ref_ts, model_ts


def _compute_timeseries_metrics(ref_ts: xr.DataArray, model_ts: xr.DataArray) -> Dict[str, float]:
    """Compute time series comparison metrics."""
    # Basic statistics
    ref_mean = float(ref_ts.mean())
    com_mean = float(model_ts.mean())
    bias = com_mean - ref_mean

    # RMSE
    rmse = float(np.sqrt(((model_ts - ref_ts) ** 2).mean()))

    # Correlation
    ref_vals = ref_ts.values.flatten()
    com_vals = model_ts.values.flatten()
    valid = ~(np.isnan(ref_vals) | np.isnan(com_vals))
    if valid.sum() > 2:
        correlation = float(np.corrcoef(ref_vals[valid], com_vals[valid])[0, 1])
    else:
        correlation = np.nan

    # Normalized standard deviation
    ref_std = float(ref_ts.std())
    com_std = float(model_ts.std())
    norm_std = com_std / ref_std if ref_std > 0 else np.nan

    # Taylor score: S = 4(1+R) / ((σ̂ + 1/σ̂)² * 2)
    # where σ̂ = σ_model / σ_ref
    if not np.isnan(correlation) and not np.isnan(norm_std) and norm_std > 0:
        taylor_score = 4 * (1 + correlation) / ((norm_std + 1/norm_std) ** 2 * 2)
    else:
        taylor_score = np.nan

    return {
        'ref_mean': ref_mean,
        'com_mean': com_mean,
        'bias': bias,
        'rmse': rmse,
        'correlation': correlation,
        'ref_std': ref_std,
        'com_std': com_std,
        'norm_std': norm_std,
        'taylor_score': taylor_score
    }


def _compute_interannual_variability(
    ref_ds: xr.Dataset,
    model_ds: xr.Dataset,
    variable: str,
    region: str
) -> Dict[str, Any]:
    """Compute interannual variability metrics."""
    # Get regional time series
    ref_ts, model_ts = _compute_regional_timeseries(ref_ds, model_ds, variable, region)

    # Compute annual means
    ref_annual = ref_ts.groupby('time.year').mean()
    com_annual = model_ts.groupby('time.year').mean()

    # IAV standard deviation
    ref_iav_std = float(ref_annual.std())
    com_iav_std = float(com_annual.std())

    # IAV score (ratio of standard deviations, penalized for deviation from 1)
    if ref_iav_std > 0:
        ratio = com_iav_std / ref_iav_std
        iav_score = np.exp(-np.abs(np.log(ratio)))  # Score close to 1 when ratio ~ 1
    else:
        iav_score = np.nan

    return {
        'ref_annual': ref_annual,
        'com_annual': com_annual,
        'ref_iav_std': ref_iav_std,
        'com_iav_std': com_iav_std,
        'iav_score': iav_score
    }


def _print_temporal_analysis_summary(df: pd.DataFrame, variable: str, region: str):
    """Print a summary of temporal analysis results."""
    print("\n" + "=" * 60)
    print(f"TEMPORAL ANALYSIS SUMMARY: {variable.upper()} ({region})")
    print("=" * 60)

    # Filter for scores
    scores = df[df['type'] == 'score']
    if not scores.empty:
        print("\nSCORES (0-1, higher is better):")
        print("-" * 40)
        for _, row in scores.iterrows():
            print(f"  {row['analysis']:25s} | {row['name']:20s}: {row['value']:.3f}")

    # Key scalars
    print("\nKEY METRICS:")
    print("-" * 40)
    scalars = df[df['type'] == 'scalar']
    for analysis in scalars['analysis'].unique():
        analysis_df = scalars[scalars['analysis'] == analysis]
        print(f"\n  {analysis}:")
        for _, row in analysis_df.iterrows():
            print(f"    {row['source']:12s} {row['name']:20s}: {row['value']:10.4f} {row['units']}")

    print("\n" + "=" * 60)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _plot_bias_map(
    bias_data: xr.DataArray,
    title: str = "Bias Map",
    region: str | None = None
) -> plt.Figure:
    """Plot spatial bias map with cartopy."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Restrict to region if specified
    if region is not None:
        bias_data = Regions.restrict_to_region(bias_data, region)

    # Determine symmetric colorbar limits
    vmax = float(np.abs(bias_data).quantile(0.95))
    vmin = -vmax

    # Plot
    im = bias_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': bias_data.attrs.get('units', ''), 'shrink': 0.8}
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    return fig


def _plot_phase_shift_map(
    shift_data: xr.DataArray,
    title: str = "Phase Shift",
    region: str | None = None
) -> plt.Figure:
    """Plot phase shift map."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if region is not None:
        shift_data = Regions.restrict_to_region(shift_data, region)

    im = shift_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='PRGn',
        vmin=-6,
        vmax=6,
        add_colorbar=True,
        cbar_kwargs={'label': 'Months', 'shrink': 0.8}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_seasonal_cycle(
    ref_cycle: xr.DataArray,
    com_cycle: xr.DataArray,
    variable: str,
    region: str
) -> plt.Figure:
    """Plot seasonal cycle comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    months = np.arange(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot reference
    ax.plot(months, ref_cycle.values, 'o-', color='black', linewidth=2,
            markersize=8, label='Reference', zorder=3)

    # Plot model
    ax.plot(months, com_cycle.values, 's--', color='#1f77b4', linewidth=2,
            markersize=8, label='CAETE Model', zorder=2)

    # Formatting
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel(f'{variable.upper()} ({ref_cycle.attrs.get("units", "")})', fontsize=12)
    ax.set_title(f'Seasonal Cycle: {variable.upper()} ({region})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add fill between
    ax.fill_between(months, ref_cycle.values, com_cycle.values, alpha=0.2, color='gray')

    plt.tight_layout()
    return fig


def _plot_timeseries(
    ref_ts: xr.DataArray,
    model_ts: xr.DataArray,
    variable: str,
    region: str,
    metrics: Dict[str, float]
) -> plt.Figure:
    """Plot time series comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot time series
    ref_ts.plot(ax=ax, color='black', linewidth=1.5, label='Reference', alpha=0.8)
    model_ts.plot(ax=ax, color='#1f77b4', linewidth=1.5, label='CAETE Model', alpha=0.8)

    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(f'{variable.upper()} ({ref_ts.attrs.get("units", "")})', fontsize=12)
    ax.set_title(f'Time Series: {variable.upper()} ({region})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    textstr = '\n'.join([
        f"Correlation: {metrics['correlation']:.3f}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"Bias: {metrics['bias']:.4f}",
        f"Taylor Score: {metrics['taylor_score']:.3f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def _plot_interannual_variability(
    ref_annual: xr.DataArray,
    com_annual: xr.DataArray,
    variable: str,
    region: str
) -> plt.Figure:
    """Plot interannual variability comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years = ref_annual['year'].values

    ax.bar(years - 0.2, ref_annual.values, width=0.4, color='black',
           alpha=0.7, label='Reference')
    ax.bar(years + 0.2, com_annual.values, width=0.4, color='#1f77b4',
           alpha=0.7, label='CAETE Model')

    # Add trend lines
    z_ref = np.polyfit(years, ref_annual.values, 1)
    z_com = np.polyfit(years, com_annual.values, 1)
    ax.plot(years, np.polyval(z_ref, years), 'k--', linewidth=2, alpha=0.5)
    ax.plot(years, np.polyval(z_com, years), 'b--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'{variable.upper()} ({ref_annual.attrs.get("units", "")})', fontsize=12)
    ax.set_title(f'Interannual Variability: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def spatial_analysis(
    model_ds: xr.Dataset,
    ref_ds: xr.Dataset,
    variable: str,
    region: str = "pana",
    output_dir: Path | None = None
) -> Dict[str, Any]:
    """Perform comprehensive spatial analysis between model and reference datasets.

    This function applies ILAMB3 analysis methods and custom metrics to compare
    spatial characteristics of model output against reference data, including
    spatial distribution, pattern correlation, RMSE maps, and spatial statistics.

    Args:
        model_ds: Model dataset (CAETE output).
        ref_ds: Reference dataset (observations).
        variable: Variable name to analyze (e.g., "gpp", "et").
        region: Region label for analysis. Default is "pana" (Pan Amazon).
        output_dir: Optional directory to save plots. If None, plots are displayed.

    Returns:
        dict: Dictionary containing:
            - 'scalars': pd.DataFrame with scalar metrics and scores
            - 'ref_gridded': xr.Dataset with reference gridded outputs
            - 'com_gridded': xr.Dataset with comparison gridded outputs
            - 'figures': dict of matplotlib Figure objects
    """
    results = {
        'scalars': pd.DataFrame(),
        'ref_gridded': xr.Dataset(),
        'com_gridded': xr.Dataset(),
        'figures': {}
    }

    # Ensure units attribute exists
    if 'units' not in ref_ds[variable].attrs:
        ref_ds[variable].attrs['units'] = '1'
    if 'units' not in model_ds[variable].attrs:
        model_ds[variable].attrs['units'] = ref_ds[variable].attrs['units']

    # =========================================================================
    # 1. COMPUTE TEMPORAL MEANS FOR SPATIAL ANALYSIS
    # =========================================================================
    print("Computing temporal means...")
    try:
        # Compute time-mean fields
        ref_mean = ref_ds[variable].mean(dim='time')
        model_mean = model_ds[variable].mean(dim='time')

        # Store in results
        results['ref_gridded']['mean'] = ref_mean
        results['com_gridded']['mean'] = model_mean

        # Create mean field comparison figure
        fig_mean = _plot_mean_field_comparison(
            ref_mean, model_mean,
            variable=variable,
            region=region
        )
        results['figures']['mean_field_comparison'] = fig_mean

    except Exception as e:
        print(f"Mean field computation failed: {e}")

    # =========================================================================
    # 2. SPATIAL DISTRIBUTION ANALYSIS (using ilamb3)
    # =========================================================================
    print("Running spatial distribution analysis...")
    try:
        spatial_analyzer = spatial_distribution_analysis(
            required_variable=variable,
            regions=[region]
        )
        spatial_df, spatial_ref, spatial_com = spatial_analyzer(ref_ds, model_ds)
        results['scalars'] = pd.concat([results['scalars'], spatial_df], ignore_index=True)

        # Create Taylor diagram for spatial distribution
        fig_taylor_spatial = _plot_spatial_taylor_diagram(
            spatial_df[spatial_df['region'] == region],
            variable=variable,
            region=region
        )
        results['figures']['spatial_taylor_diagram'] = fig_taylor_spatial

    except Exception as e:
        print(f"Spatial distribution analysis failed: {e}")

    # =========================================================================
    # 3. SPATIAL BIAS ANALYSIS
    # =========================================================================
    print("Running spatial bias analysis...")
    try:
        # Compute spatial bias (model - reference)
        ref_mean = ref_ds[variable].mean(dim='time')
        model_mean = model_ds[variable].mean(dim='time')

        # Nest grids for comparison
        ref_nested, model_nested = cmp.nest_spatial_grids(ref_mean, model_mean)
        spatial_bias = model_nested - ref_nested

        # Store gridded outputs
        results['com_gridded']['spatial_bias'] = spatial_bias

        # Compute spatial bias statistics
        spatial_bias_metrics = _compute_spatial_bias_metrics(
            ref_nested, model_nested, spatial_bias, region
        )

        # Add to scalars
        bias_df = pd.DataFrame([
            {
                'source': 'Reference',
                'region': region,
                'analysis': 'Spatial Bias',
                'name': 'Spatial Mean',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': spatial_bias_metrics['ref_spatial_mean']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Spatial Bias',
                'name': 'Spatial Mean',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': spatial_bias_metrics['com_spatial_mean']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Spatial Bias',
                'name': 'Mean Bias',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': spatial_bias_metrics['mean_bias']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Spatial Bias',
                'name': 'Bias RMSE',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': spatial_bias_metrics['bias_rmse']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Spatial Bias',
                'name': 'Relative Bias (%)',
                'type': 'scalar',
                'units': '%',
                'value': spatial_bias_metrics['relative_bias_pct']
            },
        ])
        results['scalars'] = pd.concat([results['scalars'], bias_df], ignore_index=True)

        # Create spatial bias map
        fig_spatial_bias = _plot_spatial_bias_map(
            spatial_bias,
            title=f'{variable.upper()} Spatial Bias (Model - Reference)',
            region=region,
            units=ref_ds[variable].attrs.get('units', '')
        )
        results['figures']['spatial_bias_map'] = fig_spatial_bias

    except Exception as e:
        print(f"Spatial bias analysis failed: {e}")

    # =========================================================================
    # 4. PATTERN CORRELATION ANALYSIS
    # =========================================================================
    print("Running pattern correlation analysis...")
    try:
        pattern_metrics = _compute_pattern_correlation(ref_ds, model_ds, variable, region)

        pattern_df = pd.DataFrame([
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Pattern Correlation',
                'name': 'Spatial Correlation',
                'type': 'scalar',
                'units': '1',
                'value': pattern_metrics['spatial_correlation']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Pattern Correlation',
                'name': 'Centered RMSE',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': pattern_metrics['centered_rmse']
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Pattern Correlation',
                'name': 'Pattern Score',
                'type': 'score',
                'units': '1',
                'value': pattern_metrics['pattern_score']
            },
        ])
        results['scalars'] = pd.concat([results['scalars'], pattern_df], ignore_index=True)

        # Create scatter plot
        fig_scatter = _plot_spatial_scatter(
            pattern_metrics['ref_values'],
            pattern_metrics['com_values'],
            variable=variable,
            region=region,
            metrics=pattern_metrics
        )
        results['figures']['spatial_scatter'] = fig_scatter

    except Exception as e:
        print(f"Pattern correlation analysis failed: {e}")

    # =========================================================================
    # 5. SPATIAL RMSE ANALYSIS
    # =========================================================================
    print("Running spatial RMSE analysis...")
    try:
        # Compute RMSE at each grid point (temporal RMSE)
        ref_trimmed, model_trimmed = cmp.trim_time(ref_ds, model_ds)
        ref_var = ref_trimmed[variable]
        model_var = model_trimmed[variable]

        # Nest grids
        ref_nested, model_nested = cmp.nest_spatial_grids(ref_var, model_var)

        # Compute temporal RMSE at each point
        rmse_map = np.sqrt(((model_nested - ref_nested) ** 2).mean(dim='time'))
        results['com_gridded']['rmse_map'] = rmse_map

        # Compute RMSE statistics
        rmse_regional = Regions.restrict_to_region(rmse_map, region)
        mean_rmse = float(rmse_regional.mean())
        max_rmse = float(rmse_regional.max())

        rmse_df = pd.DataFrame([
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Spatial RMSE',
                'name': 'Mean RMSE',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': mean_rmse
            },
            {
                'source': 'Comparison',
                'region': region,
                'analysis': 'Spatial RMSE',
                'name': 'Max RMSE',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': max_rmse
            },
        ])
        results['scalars'] = pd.concat([results['scalars'], rmse_df], ignore_index=True)

        # Create RMSE map
        fig_rmse = _plot_rmse_map(
            rmse_map,
            title=f'{variable.upper()} RMSE Map',
            region=region,
            units=ref_ds[variable].attrs.get('units', '')
        )
        results['figures']['rmse_map'] = fig_rmse

    except Exception as e:
        print(f"Spatial RMSE analysis failed: {e}")

    # =========================================================================
    # 6. ZONAL MEAN ANALYSIS
    # =========================================================================
    print("Running zonal mean analysis...")
    try:
        zonal_metrics = _compute_zonal_means(ref_ds, model_ds, variable, region)

        # Store zonal means
        results['ref_gridded']['zonal_mean'] = zonal_metrics['ref_zonal']
        results['com_gridded']['zonal_mean'] = zonal_metrics['com_zonal']

        # Create zonal mean figure
        fig_zonal = _plot_zonal_means(
            zonal_metrics['ref_zonal'],
            zonal_metrics['com_zonal'],
            variable=variable,
            region=region
        )
        results['figures']['zonal_means'] = fig_zonal

    except Exception as e:
        print(f"Zonal mean analysis failed: {e}")

    # =========================================================================
    # 7. SPATIAL QUANTILE ANALYSIS
    # =========================================================================
    print("Running spatial quantile analysis...")
    try:
        quantile_metrics = _compute_spatial_quantiles(ref_ds, model_ds, variable, region)

        quantile_df = pd.DataFrame([
            {
                'source': src,
                'region': region,
                'analysis': 'Spatial Quantiles',
                'name': f'Q{int(q*100)}',
                'type': 'scalar',
                'units': ref_ds[variable].attrs.get('units', '1'),
                'value': val
            }
            for src, qvals in [('Reference', quantile_metrics['ref_quantiles']),
                               ('Comparison', quantile_metrics['com_quantiles'])]
            for q, val in qvals.items()
        ])
        results['scalars'] = pd.concat([results['scalars'], quantile_df], ignore_index=True)

        # Create quantile-quantile plot
        fig_qq = _plot_qq_diagram(
            quantile_metrics['ref_quantiles'],
            quantile_metrics['com_quantiles'],
            variable=variable,
            region=region
        )
        results['figures']['qq_plot'] = fig_qq

    except Exception as e:
        print(f"Spatial quantile analysis failed: {e}")

    # =========================================================================
    # SAVE PLOTS IF OUTPUT DIRECTORY PROVIDED
    # =========================================================================
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in results['figures'].items():
            filepath = output_dir / f'{variable}_spatial_{name}.png'
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {filepath}")
        # Clear figures from results after saving to free memory
        results['figures'] = {}
    else:
        # Close all figures to prevent memory leak when not saving
        for fig in results['figures'].values():
            plt.close(fig)
        results['figures'] = {}

    # Print summary
    _print_spatial_analysis_summary(results['scalars'], variable, region)

    return results


# =============================================================================
# HELPER FUNCTIONS FOR SPATIAL ANALYSIS
# =============================================================================

def _compute_spatial_bias_metrics(
    ref_data: xr.DataArray,
    model_data: xr.DataArray,
    bias_data: xr.DataArray,
    region: str
) -> Dict[str, float]:
    """Compute spatial bias metrics."""
    # Restrict to region
    ref_regional = Regions.restrict_to_region(ref_data, region)
    model_regional = Regions.restrict_to_region(model_data, region)
    bias_regional = Regions.restrict_to_region(bias_data, region)

    # Compute weights for area-weighted statistics
    weights = dset.compute_cell_measures(ref_regional)

    # Weighted spatial means
    ref_spatial_mean = float(ref_regional.weighted(weights.fillna(0)).mean())
    com_spatial_mean = float(model_regional.weighted(weights.fillna(0)).mean())
    mean_bias = float(bias_regional.weighted(weights.fillna(0)).mean())

    # Bias RMSE (spatial)
    bias_rmse = float(np.sqrt((bias_regional ** 2).weighted(weights.fillna(0)).mean()))

    # Relative bias
    if ref_spatial_mean != 0:
        relative_bias_pct = (mean_bias / ref_spatial_mean) * 100
    else:
        relative_bias_pct = np.nan

    return {
        'ref_spatial_mean': ref_spatial_mean,
        'com_spatial_mean': com_spatial_mean,
        'mean_bias': mean_bias,
        'bias_rmse': bias_rmse,
        'relative_bias_pct': relative_bias_pct
    }


def _compute_pattern_correlation(
    ref_ds: xr.Dataset,
    model_ds: xr.Dataset,
    variable: str,
    region: str
) -> Dict[str, Any]:
    """Compute spatial pattern correlation metrics."""
    # Get temporal means
    ref_mean = ref_ds[variable].mean(dim='time')
    model_mean = model_ds[variable].mean(dim='time')

    # Nest grids and restrict to region
    ref_nested, model_nested = cmp.nest_spatial_grids(ref_mean, model_mean)
    ref_regional = Regions.restrict_to_region(ref_nested, region)
    model_regional = Regions.restrict_to_region(model_nested, region)

    # Flatten and remove NaNs
    ref_vals = ref_regional.values.flatten()
    com_vals = model_regional.values.flatten()
    valid = ~(np.isnan(ref_vals) | np.isnan(com_vals))
    ref_valid = ref_vals[valid]
    com_valid = com_vals[valid]

    # Spatial correlation
    if len(ref_valid) > 2:
        spatial_correlation = float(np.corrcoef(ref_valid, com_valid)[0, 1])
    else:
        spatial_correlation = np.nan

    # Centered RMSE (pattern RMSE after removing mean bias)
    ref_anomaly = ref_valid - ref_valid.mean()
    com_anomaly = com_valid - com_valid.mean()
    centered_rmse = float(np.sqrt(np.mean((com_anomaly - ref_anomaly) ** 2)))

    # Pattern score (based on Taylor skill score)
    ref_std = float(np.std(ref_valid))
    com_std = float(np.std(com_valid))
    if ref_std > 0:
        norm_std = com_std / ref_std
        if not np.isnan(spatial_correlation) and norm_std > 0:
            pattern_score = 4 * (1 + spatial_correlation) / ((norm_std + 1/norm_std) ** 2 * 2)
        else:
            pattern_score = np.nan
    else:
        norm_std = np.nan
        pattern_score = np.nan

    return {
        'spatial_correlation': spatial_correlation,
        'centered_rmse': centered_rmse,
        'ref_std': ref_std,
        'com_std': com_std,
        'norm_std': norm_std,
        'pattern_score': pattern_score,
        'ref_values': ref_valid,
        'com_values': com_valid
    }


def _compute_zonal_means(
    ref_ds: xr.Dataset,
    model_ds: xr.Dataset,
    variable: str,
    region: str
) -> Dict[str, xr.DataArray]:
    """Compute zonal mean profiles."""
    # Get temporal means
    ref_mean = ref_ds[variable].mean(dim='time')
    model_mean = model_ds[variable].mean(dim='time')

    # Nest grids to common resolution before restricting to region
    ref_nested, model_nested = cmp.nest_spatial_grids(ref_mean, model_mean)

    # Restrict to region
    ref_regional = Regions.restrict_to_region(ref_nested, region)
    model_regional = Regions.restrict_to_region(model_nested, region)

    # Zonal mean (average over longitude)
    ref_zonal = ref_regional.mean(dim='lon')
    com_zonal = model_regional.mean(dim='lon')

    return {
        'ref_zonal': ref_zonal,
        'com_zonal': com_zonal
    }


def _compute_spatial_quantiles(
    ref_ds: xr.Dataset,
    model_ds: xr.Dataset,
    variable: str,
    region: str,
    quantiles: list = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
) -> Dict[str, Dict[float, float]]:
    """Compute spatial quantiles for reference and model."""
    # Get temporal means
    ref_mean = ref_ds[variable].mean(dim='time')
    model_mean = model_ds[variable].mean(dim='time')

    # Restrict to region
    ref_regional = Regions.restrict_to_region(ref_mean, region)
    model_regional = Regions.restrict_to_region(model_mean, region)

    # Compute quantiles
    ref_quantiles = {q: float(ref_regional.quantile(q)) for q in quantiles}
    com_quantiles = {q: float(model_regional.quantile(q)) for q in quantiles}

    return {
        'ref_quantiles': ref_quantiles,
        'com_quantiles': com_quantiles
    }


def _print_spatial_analysis_summary(df: pd.DataFrame, variable: str, region: str):
    """Print a summary of spatial analysis results."""
    print("\n" + "=" * 60)
    print(f"SPATIAL ANALYSIS SUMMARY: {variable.upper()} ({region})")
    print("=" * 60)

    # Filter for scores
    scores = df[df['type'] == 'score']
    if not scores.empty:
        print("\nSCORES (0-1, higher is better):")
        print("-" * 40)
        for _, row in scores.iterrows():
            print(f"  {row['analysis']:25s} | {row['name']:20s}: {row['value']:.3f}")

    # Key scalars by analysis
    print("\nKEY METRICS:")
    print("-" * 40)
    scalars = df[df['type'] == 'scalar']
    for analysis in scalars['analysis'].unique():
        analysis_df = scalars[scalars['analysis'] == analysis]
        print(f"\n  {analysis}:")
        for _, row in analysis_df.iterrows():
            print(f"    {row['source']:12s} {row['name']:20s}: {row['value']:10.4f} {row['units']}")

    print("\n" + "=" * 60)


# =============================================================================
# SPATIAL PLOTTING FUNCTIONS
# =============================================================================

def _plot_mean_field_comparison(
    ref_mean: xr.DataArray,
    model_mean: xr.DataArray,
    variable: str,
    region: str
) -> plt.Figure:
    """Plot side-by-side comparison of mean fields."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              subplot_kw={'projection': ccrs.PlateCarree()})

    # Restrict to region
    ref_regional = Regions.restrict_to_region(ref_mean, region)
    model_regional = Regions.restrict_to_region(model_mean, region)

    # Common colorbar limits
    vmin = min(float(ref_regional.min()), float(model_regional.min()))
    vmax = max(float(ref_regional.max()), float(model_regional.max()))

    # Plot reference
    ref_regional.plot(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': ref_mean.attrs.get('units', ''), 'shrink': 0.8}
    )
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.5)
    axes[0].add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    axes[0].set_title(f'Reference Mean {variable.upper()}', fontsize=12, fontweight='bold')

    # Plot model
    model_regional.plot(
        ax=axes[1],
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': model_mean.attrs.get('units', ''), 'shrink': 0.8}
    )
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.5)
    axes[1].add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    axes[1].set_title(f'CAETE Model Mean {variable.upper()}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def _plot_spatial_bias_map(
    bias_data: xr.DataArray,
    title: str,
    region: str,
    units: str = ''
) -> plt.Figure:
    """Plot spatial bias map."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Restrict to region
    bias_regional = Regions.restrict_to_region(bias_data, region)

    # Symmetric colorbar
    vmax = float(np.abs(bias_regional).quantile(0.95))
    vmin = -vmax

    im = bias_regional.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': units, 'shrink': 0.8}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_rmse_map(
    rmse_data: xr.DataArray,
    title: str,
    region: str,
    units: str = ''
) -> plt.Figure:
    """Plot RMSE map."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Restrict to region
    rmse_regional = Regions.restrict_to_region(rmse_data, region)

    im = rmse_regional.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='YlOrRd',
        add_colorbar=True,
        cbar_kwargs={'label': units, 'shrink': 0.8}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_spatial_scatter(
    ref_values: np.ndarray,
    com_values: np.ndarray,
    variable: str,
    region: str,
    metrics: Dict[str, float]
) -> plt.Figure:
    """Plot spatial scatter (model vs reference)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(ref_values, com_values, alpha=0.3, s=10, c='#1f77b4', edgecolors='none')

    # 1:1 line
    lims = [
        min(ref_values.min(), com_values.min()),
        max(ref_values.max(), com_values.max())
    ]
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line')

    # Regression line
    if len(ref_values) > 2:
        z = np.polyfit(ref_values, com_values, 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), 'r-', linewidth=2, alpha=0.7,
                label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlabel(f'Reference {variable.upper()}', fontsize=12)
    ax.set_ylabel(f'Model {variable.upper()}', fontsize=12)
    ax.set_title(f'Spatial Scatter: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    textstr = '\n'.join([
        f"R = {metrics['spatial_correlation']:.3f}",
        f"Pattern Score = {metrics['pattern_score']:.3f}",
        f"Centered RMSE = {metrics['centered_rmse']:.4f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig


def _plot_zonal_means(
    ref_zonal: xr.DataArray,
    com_zonal: xr.DataArray,
    variable: str,
    region: str
) -> plt.Figure:
    """Plot zonal mean profiles."""
    fig, ax = plt.subplots(figsize=(8, 10))

    # Get latitude values (should now match after nesting)
    ref_lats = ref_zonal['lat'].values
    com_lats = com_zonal['lat'].values
    ref_vals = ref_zonal.values
    com_vals = com_zonal.values

    # Plot zonal means
    ax.plot(ref_vals, ref_lats, 'k-', linewidth=2, label='Reference')
    ax.plot(com_vals, com_lats, 'b--', linewidth=2, label='CAETE Model')

    ax.set_xlabel(f'{variable.upper()} ({ref_zonal.attrs.get("units", "")})', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Zonal Mean: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Fill between (only if latitudes match)
    if np.array_equal(ref_lats, com_lats):
        ax.fill_betweenx(ref_lats, ref_vals, com_vals, alpha=0.2, color='gray')

    plt.tight_layout()
    return fig


def _plot_qq_diagram(
    ref_quantiles: Dict[float, float],
    com_quantiles: Dict[float, float],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot quantile-quantile diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))

    quantiles = list(ref_quantiles.keys())
    ref_vals = [ref_quantiles[q] for q in quantiles]
    com_vals = [com_quantiles[q] for q in quantiles]

    # Q-Q plot
    ax.scatter(ref_vals, com_vals, s=100, c='#1f77b4', edgecolors='black', zorder=3)

    # Label points
    for q, rx, cx in zip(quantiles, ref_vals, com_vals):
        ax.annotate(f'Q{int(q*100)}', (rx, cx), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    # 1:1 line
    lims = [min(min(ref_vals), min(com_vals)), max(max(ref_vals), max(com_vals))]
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line')

    ax.set_xlabel(f'Reference Quantiles ({variable.upper()})', fontsize=12)
    ax.set_ylabel(f'Model Quantiles ({variable.upper()})', fontsize=12)
    ax.set_title(f'Q-Q Plot: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def _plot_spatial_taylor_diagram(
    df: pd.DataFrame,
    variable: str,
    region: str
) -> plt.Figure:
    """Plot Taylor diagram for spatial distribution."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Reference point
    ax.plot(0, 1, 'ko', markersize=12, label='Reference')

    # Get metrics from dataframe
    try:
        norm_std = float(df[df['name'] == 'Normalized Standard Deviation']['value'].iloc[0])
        corr = float(df[df['name'] == 'Correlation']['value'].iloc[0])

        if not np.isnan(corr) and not np.isnan(norm_std):
            theta = np.arccos(corr)
            ax.plot(theta, norm_std, 's', color='#1f77b4', markersize=12, label='CAETE Model')
    except (IndexError, KeyError):
        pass

    # Configure axes
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('E')
    ax.set_rlabel_position(0)
    ax.set_rticks([0.5, 1.0, 1.5, 2.0])

    ax.set_title(f'Spatial Taylor Diagram: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    return fig


def run_benchmark(
    variable: str = "gpp",
    dataset_name: str = "GOSIF",
    filename: str = "gpp_GOSIF_2000-2024.nc",
    experiment: str | None = None,
    output_dir: str | None = None
) -> Dict[str, Any]:
    """Run complete benchmark for a variable.

    Args:
        variable: Variable to benchmark (e.g., "gpp", "et").
        dataset_name: Reference dataset name.
        filename: Reference dataset filename.
        experiment: Experiment name. If None, uses the first available experiment.
        output_dir: Optional directory to save outputs.

    Returns:
        Dictionary with benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING BENCHMARK: {variable.upper()}")
    print(f"Reference: {dataset_name}/{filename}")
    print(f"{'='*60}\n")

    # Load and prepare data
    model_ds, ref_ds, experiment, region_mask, model_mask = get_model_and_ref(
        variable=variable,
        dataset_name=dataset_name,
        filename=filename,
        experiment=experiment
    )
    print(f"Experiment: {experiment}")

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir) / variable
    else:
        output_path = None

    # Run temporal analysis
    print("\n" + "-"*60)
    print("TEMPORAL ANALYSIS")
    print("-"*60)
    temporal_results = temporal_analysis(
        model_ds=model_ds,
        ref_ds=ref_ds,
        variable=variable,
        region="pana",
        output_dir=output_path
    )

    # Run spatial analysis
    print("\n" + "-"*60)
    print("SPATIAL ANALYSIS")
    print("-"*60)
    spatial_results = spatial_analysis(
        model_ds=model_ds,
        ref_ds=ref_ds,
        variable=variable,
        region="pana",
        output_dir=output_path
    )

    # Combine all scalars
    all_scalars = pd.concat([
        temporal_results['scalars'],
        spatial_results['scalars']
    ], ignore_index=True)

    # Print overall summary
    _print_overall_benchmark_summary(all_scalars, variable, experiment)

    return {
        'experiment': experiment,
        'variable': variable,
        'temporal_results': temporal_results,
        'spatial_results': spatial_results,
        'all_scalars': all_scalars,
        'model_ds': model_ds,
        'ref_ds': ref_ds
    }


def _print_overall_benchmark_summary(df: pd.DataFrame, variable: str, experiment: str):
    """Print overall benchmark summary with all scores."""
    print("\n" + "=" * 70)
    print(f"OVERALL BENCHMARK SUMMARY: {variable.upper()}")
    print(f"Experiment: {experiment}")
    print("=" * 70)

    # All scores
    scores = df[df['type'] == 'score']
    if not scores.empty:
        print("\nALL SCORES (0-1, higher is better):")
        print("-" * 50)
        for _, row in scores.iterrows():
            print(f"  {row['analysis']:30s} | {row['name']:25s}: {row['value']:.3f}")

        # Overall score (mean of all scores)
        mean_score = scores['value'].mean()
        print("-" * 50)
        print(f"  {'OVERALL MEAN SCORE':30s} | {'':25s}: {mean_score:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run benchmark for ET variable
    et_results = run_benchmark(
        variable="et",
        dataset_name="GLEAMv4.2b",
        filename="et_GLEAM_2003-2024.nc",
        output_dir="../outputs/benchmark_results"
    )

    # Run benchmark for GPP variable
    gpp_results = run_benchmark(
        variable="gpp",
        dataset_name="GOSIF",
        filename="gpp_GOSIF_2000-2024.nc",
        output_dir="../outputs/benchmark_results"
    )

