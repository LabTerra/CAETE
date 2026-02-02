"""Benchmark script for CAETE model evaluation.

This module provides functions to load, preprocess, and compare CAETE model outputs
with reference datasets for benchmarking purposes. Supports comparing multiple
experiments against a single reference dataset in the same benchmark.

Requires CDO to be installed and accessible in the system path.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Collection

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

# Color palette for distinguishing multiple experiments in plots
EXPERIMENT_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

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


def _normalize_experiments(experiments: List[str] | str | Collection[str] | None) -> List[str] | None:
    """Normalize experiments input to a list of strings or None.

    Accepts a single experiment name, a list/collection of names, or None.
    None means "use all available experiments".
    """
    if experiments is None:
        return None
    if isinstance(experiments, str):
        return [experiments]
    if isinstance(experiments, Collection):
        return list(experiments)
    raise TypeError("experiments must be a string, a collection of strings, or None")


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


def get_multiple_model_data(
    variable: str,
    experiments: List[str] | None = None,
    use_cache: bool = True
) -> Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    """Load and preprocess CAETE model output for multiple experiments.

    Args:
        variable: Variable name to read from CAETE output.
        experiments: List of experiment names. If None, loads all available experiments.
        use_cache: If True, use cached preprocessed files if available.

    Returns:
        A tuple containing:
            - model_datasets: Dict mapping experiment name to preprocessed Dataset.
            - model_masks: Dict mapping experiment name to mask Dataset.
    """
    caete_varname = get_caete_varname(variable)

    experiments = _normalize_experiments(experiments)

    # Get available experiments for this variable
    if experiments is None:
        experiments = list(caete_data.keys())
        # Filter to experiments that have this variable
        experiments = [exp for exp in experiments if caete_varname in caete_data.get(exp, {})]

    if not experiments:
        raise ValueError(f"No experiments found with variable '{variable}'")

    print(f"Loading {len(experiments)} experiment(s): {experiments}")

    model_datasets = {}
    model_masks = {}

    for exp in experiments:
        try:
            model_ds, _, model_mask = get_model_data(variable, exp, use_cache)
            model_datasets[exp] = model_ds
            model_masks[exp] = model_mask
        except Exception as e:
            print(f"Warning: Failed to load experiment '{exp}': {e}")
            continue

    if not model_datasets:
        raise RuntimeError("Failed to load any experiments")

    return model_datasets, model_masks


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


def get_models_and_ref(
    variable: str = "gpp",
    dataset_name: str = "GOSIF",
    filename: str = "gpp_GOSIF_2000-2024.nc",
    experiments: List[str] | None = None
) -> Tuple[Dict[str, xr.Dataset], xr.Dataset, xr.Dataset, Dict[str, xr.Dataset]]:
    """Load and prepare multiple model experiments and reference dataset for comparison.

    This function supports multi-experiment benchmarking by loading multiple CAETE
    experiments and comparing them against a single reference dataset.

    Args:
        variable: Variable name to read from CAETE output. Default is "gpp".
        dataset_name: Name of the reference dataset. Default is "GOSIF".
        filename: Filename within the reference datasets.
        experiments: List of experiment names. If None, loads all available experiments.

    Returns:
        A tuple containing:
            - model_datasets: Dict[str, xr.Dataset] mapping experiment names to datasets.
            - ref_ds: Reference dataset (single).
            - region_mask: RAISG mask for the Pan Amazon region.
            - model_masks: Dict[str, xr.Dataset] mapping experiment names to masks.
    """
    # Normalize experiments input (str -> [str], other iterables -> list)
    experiments = _normalize_experiments(experiments)

    # Step 1: Get multiple model datasets
    model_datasets, model_masks = get_multiple_model_data(variable, experiments)

    # Step 2: Get reference data
    ref_ds = get_reference_data(dataset_name, filename)

    # Step 3: Conform each model dataset with reference
    conformed_models = {}
    for exp_name, model_ds in model_datasets.items():
        model_conformed, ref_conformed = conform_datasets(model_ds.copy(), ref_ds.copy(), variable)
        conformed_models[exp_name] = model_conformed
        # Keep the last conformed ref (they should all be the same)
        ref_ds_conformed = ref_conformed

    # Step 4: Get region mask
    region_mask = get_region_mask()

    return conformed_models, ref_ds_conformed, region_mask, model_masks


def temporal_analysis(
    model_datasets: Dict[str, xr.Dataset],
    ref_ds: xr.Dataset,
    variable: str,
    region: str = "pana",
    output_dir: Path | None = None
) -> Dict[str, Any]:
    """Perform comprehensive temporal analysis between multiple model experiments and reference.

    This function applies ILAMB3 analysis methods to compare temporal characteristics
    of multiple model outputs against a single reference dataset, including bias,
    seasonal cycle, and time series analysis.

    Args:
        model_datasets: Dict mapping experiment names to model datasets.
        ref_ds: Reference dataset (observations).
        variable: Variable name to analyze (e.g., "gpp", "et").
        region: Region label for analysis. Default is "pana" (Pan Amazon).
        output_dir: Optional directory to save plots. If None, plots are displayed.

    Returns:
        dict: Dictionary containing:
            - 'scalars': pd.DataFrame with scalar metrics and scores (includes 'experiment' column)
            - 'ref_gridded': xr.Dataset with reference gridded outputs
            - 'com_gridded': Dict[str, xr.Dataset] with comparison gridded outputs per experiment
            - 'figures': dict of matplotlib Figure objects
    """
    experiment_names = list(model_datasets.keys())
    n_experiments = len(experiment_names)

    results = {
        'scalars': pd.DataFrame(),
        'ref_gridded': xr.Dataset(),
        'com_gridded': {exp: xr.Dataset() for exp in experiment_names},
        'figures': {}
    }

    # Ensure units attribute exists
    if 'units' not in ref_ds[variable].attrs:
        ref_ds[variable].attrs['units'] = '1'
    for exp_name, model_ds in model_datasets.items():
        if 'units' not in model_ds[variable].attrs:
            model_ds[variable].attrs['units'] = ref_ds[variable].attrs['units']

    # =========================================================================
    # 1. BIAS ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running bias analysis for {n_experiments} experiment(s)...")
    bias_data_per_exp = {}
    try:
        bias_analyzer = bias_analysis(
            required_variable=variable,
            regions=[region],
            use_uncertainty=False,
            mass_weighting=False
        )

        for exp_name, model_ds in model_datasets.items():
            bias_df, bias_ref, bias_com = bias_analyzer(ref_ds, model_ds)
            # Add experiment column
            bias_df['experiment'] = exp_name
            results['scalars'] = pd.concat([results['scalars'], bias_df], ignore_index=True)

            # Store gridded outputs (only ref once)
            if not results['ref_gridded'].data_vars:
                for var in bias_ref.data_vars:
                    results['ref_gridded'][f'bias_{var}'] = bias_ref[var]

            for var in bias_com.data_vars:
                results['com_gridded'][exp_name][f'bias_{var}'] = bias_com[var]

            if 'bias' in bias_com:
                bias_data_per_exp[exp_name] = bias_com['bias']

        # Create multi-experiment bias map figure
        if bias_data_per_exp:
            fig_bias = _plot_multi_experiment_bias_maps(
                bias_data_per_exp,
                title=f'{variable.upper()} Bias (Model - Reference)',
                region=region
            )
            results['figures']['bias_map'] = fig_bias

    except Exception as e:
        print(f"Bias analysis failed: {e}")

    # =========================================================================
    # 2. SEASONAL CYCLE ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running seasonal cycle analysis for {n_experiments} experiment(s)...")
    cycle_data_per_exp = {}
    ref_cycle_data = None
    try:
        cycle_analyzer = cycle_analysis(
            required_variable=variable,
            regions=[region]
        )

        for exp_name, model_ds in model_datasets.items():
            cycle_df, cycle_ref, cycle_com = cycle_analyzer(ref_ds, model_ds)
            cycle_df['experiment'] = exp_name
            results['scalars'] = pd.concat([results['scalars'], cycle_df], ignore_index=True)

            # Store gridded outputs
            if not any('cycle_' in str(v) for v in results['ref_gridded'].data_vars):
                for var in cycle_ref.data_vars:
                    results['ref_gridded'][f'cycle_{var}'] = cycle_ref[var]

            for var in cycle_com.data_vars:
                results['com_gridded'][exp_name][f'cycle_{var}'] = cycle_com[var]

            ref_cycle_var = f'cycle_{region}'
            if ref_cycle_var in cycle_com:
                cycle_data_per_exp[exp_name] = cycle_com[ref_cycle_var]
            if ref_cycle_var in cycle_ref and ref_cycle_data is None:
                ref_cycle_data = cycle_ref[ref_cycle_var]

        # Create multi-experiment seasonal cycle figure
        if cycle_data_per_exp and ref_cycle_data is not None:
            fig_cycle = _plot_multi_experiment_seasonal_cycle(
                ref_cycle_data,
                cycle_data_per_exp,
                variable=variable,
                region=region
            )
            results['figures']['seasonal_cycle'] = fig_cycle

    except Exception as e:
        print(f"Seasonal cycle analysis failed: {e}")

    # =========================================================================
    # 3. TIME SERIES ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running time series analysis for {n_experiments} experiment(s)...")
    ts_data_per_exp = {}
    ref_ts_data = None
    ts_metrics_per_exp = {}
    try:
        for exp_name, model_ds in model_datasets.items():
            ref_ts, model_ts = _compute_regional_timeseries(ref_ds, model_ds, variable, region)
            ts_metrics = _compute_timeseries_metrics(ref_ts, model_ts)

            if ref_ts_data is None:
                ref_ts_data = ref_ts

            ts_data_per_exp[exp_name] = model_ts
            ts_metrics_per_exp[exp_name] = ts_metrics

            # Add to scalars with experiment column
            ts_df = pd.DataFrame([
                {
                    'experiment': exp_name,
                    'source': 'Reference',
                    'region': region,
                    'analysis': 'Time Series',
                    'name': 'Period Mean',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': ts_metrics['ref_mean']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Time Series',
                    'name': 'Period Mean',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': ts_metrics['com_mean']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Time Series',
                    'name': 'Bias',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': ts_metrics['bias']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Time Series',
                    'name': 'RMSE',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': ts_metrics['rmse']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Time Series',
                    'name': 'Correlation',
                    'type': 'scalar',
                    'units': '1',
                    'value': ts_metrics['correlation']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Time Series',
                    'name': 'Normalized Std Dev',
                    'type': 'scalar',
                    'units': '1',
                    'value': ts_metrics['norm_std']
                },
                {
                    'experiment': exp_name,
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
            results['com_gridded'][exp_name]['timeseries'] = model_ts

        results['ref_gridded']['timeseries'] = ref_ts_data

        # Create multi-experiment time series figure
        if ts_data_per_exp and ref_ts_data is not None:
            fig_ts = _plot_multi_experiment_timeseries(
                ref_ts_data,
                ts_data_per_exp,
                variable=variable,
                region=region,
                metrics=ts_metrics_per_exp
            )
            results['figures']['timeseries'] = fig_ts

    except Exception as e:
        print(f"Time series analysis failed: {e}")

    # =========================================================================
    # 4. INTERANNUAL VARIABILITY ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running interannual variability analysis for {n_experiments} experiment(s)...")
    iav_data_per_exp = {}
    ref_annual_data = None
    try:
        for exp_name, model_ds in model_datasets.items():
            iav_results = _compute_interannual_variability(ref_ds, model_ds, variable, region)

            if ref_annual_data is None:
                ref_annual_data = iav_results['ref_annual']

            iav_data_per_exp[exp_name] = {
                'com_annual': iav_results['com_annual'],
                'com_iav_std': iav_results['com_iav_std'],
                'iav_score': iav_results['iav_score']
            }

            iav_df = pd.DataFrame([
                {
                    'experiment': exp_name,
                    'source': 'Reference',
                    'region': region,
                    'analysis': 'Interannual Variability',
                    'name': 'IAV Std Dev',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': iav_results['ref_iav_std']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Interannual Variability',
                    'name': 'IAV Std Dev',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': iav_results['com_iav_std']
                },
                {
                    'experiment': exp_name,
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

        # Create multi-experiment IAV figure
        if iav_data_per_exp and ref_annual_data is not None:
            fig_iav = _plot_multi_experiment_iav(
                ref_annual_data,
                {exp: data['com_annual'] for exp, data in iav_data_per_exp.items()},
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
    """Print a summary of temporal analysis results for multiple experiments."""
    print("\n" + "=" * 70)
    print(f"TEMPORAL ANALYSIS SUMMARY: {variable.upper()} ({region})")
    print("=" * 70)

    # Get experiment list
    experiments = df['experiment'].unique() if 'experiment' in df.columns else ['default']

    for exp in experiments:
        exp_df = df[df['experiment'] == exp] if 'experiment' in df.columns else df
        print(f"\n>>> Experiment: {exp}")
        print("-" * 60)

        # Filter for scores
        scores = exp_df[exp_df['type'] == 'score']
        if not scores.empty:
            print("\n  SCORES (0-1, higher is better):")
            for _, row in scores.iterrows():
                print(f"    {row['analysis']:25s} | {row['name']:20s}: {row['value']:.3f}")

        # Key scalars
        print("\n  KEY METRICS:")
        scalars = exp_df[exp_df['type'] == 'scalar']
        for analysis in scalars['analysis'].unique():
            analysis_df = scalars[scalars['analysis'] == analysis]
            print(f"\n    {analysis}:")
            for _, row in analysis_df.iterrows():
                print(f"      {row['source']:12s} {row['name']:20s}: {row['value']:10.4f} {row['units']}")

    print("\n" + "=" * 70)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _plot_multi_experiment_bias_maps(
    bias_data_per_exp: Dict[str, xr.DataArray],
    title: str,
    region: str
) -> plt.Figure:
    """Plot bias maps for multiple experiments in a grid layout."""
    n_exp = len(bias_data_per_exp)
    ncols = min(3, n_exp)
    nrows = (n_exp + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()},
        squeeze=False
    )

    # Find common colorbar limits across all experiments
    all_bias = [Regions.restrict_to_region(b, region) for b in bias_data_per_exp.values()]
    vmax = max(float(np.abs(b).quantile(0.95)) for b in all_bias)
    vmin = -vmax

    for idx, (exp_name, bias_data) in enumerate(bias_data_per_exp.items()):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        bias_regional = Regions.restrict_to_region(bias_data, region)
        im = bias_regional.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={'label': bias_data.attrs.get('units', ''), 'shrink': 0.8}
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        ax.set_title(f'{exp_name}', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_exp, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def _plot_multi_experiment_seasonal_cycle(
    ref_cycle: xr.DataArray,
    cycle_per_exp: Dict[str, xr.DataArray],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot seasonal cycle comparison for multiple experiments."""
    fig, ax = plt.subplots(figsize=(12, 7))

    months = np.arange(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot reference
    ax.plot(months, ref_cycle.values, 'o-', color='black', linewidth=3,
            markersize=10, label='Reference', zorder=10)

    # Plot each experiment
    for idx, (exp_name, com_cycle) in enumerate(cycle_per_exp.items()):
        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
        ax.plot(months, com_cycle.values, 's--', color=color, linewidth=2,
                markersize=7, label=exp_name, alpha=0.8)

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel(f'{variable.upper()} ({ref_cycle.attrs.get("units", "")})', fontsize=12)
    ax.set_title(f'Seasonal Cycle: {variable.upper()} ({region})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_multi_experiment_timeseries(
    ref_ts: xr.DataArray,
    ts_per_exp: Dict[str, xr.DataArray],
    variable: str,
    region: str,
    metrics: Dict[str, Dict[str, float]]
) -> plt.Figure:
    """Plot time series comparison for multiple experiments."""
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot reference
    ref_ts.plot(ax=ax, color='black', linewidth=2, label='Reference', alpha=0.9)

    # Plot each experiment
    for idx, (exp_name, model_ts) in enumerate(ts_per_exp.items()):
        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
        model_ts.plot(ax=ax, color=color, linewidth=1.5, label=exp_name, alpha=0.7)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(f'{variable.upper()} ({ref_ts.attrs.get("units", "")})', fontsize=12)
    ax.set_title(f'Time Series: {variable.upper()} ({region})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    # Add metrics summary text box
    lines = []
    for exp_name, exp_metrics in metrics.items():
        lines.append(f"{exp_name}: R={exp_metrics['correlation']:.2f}, "
                    f"RMSE={exp_metrics['rmse']:.3f}, Score={exp_metrics['taylor_score']:.2f}")
    textstr = '\n'.join(lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def _plot_multi_experiment_iav(
    ref_annual: xr.DataArray,
    annual_per_exp: Dict[str, xr.DataArray],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot interannual variability comparison for multiple experiments."""
    n_exp = len(annual_per_exp)
    fig, ax = plt.subplots(figsize=(14, 7))

    years = ref_annual['year'].values
    n_years = len(years)

    # Width of bars
    total_width = 0.8
    bar_width = total_width / (n_exp + 1)

    # Plot reference
    positions = years - total_width/2 + bar_width/2
    ax.bar(positions, ref_annual.values, width=bar_width, color='black',
           alpha=0.7, label='Reference')

    # Plot each experiment
    for idx, (exp_name, com_annual) in enumerate(annual_per_exp.items()):
        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
        positions = years - total_width/2 + bar_width * (idx + 1.5)
        ax.bar(positions, com_annual.values, width=bar_width, color=color,
               alpha=0.7, label=exp_name)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'{variable.upper()} ({ref_annual.attrs.get("units", "")})', fontsize=12)
    ax.set_title(f'Interannual Variability: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def spatial_analysis(
    model_datasets: Dict[str, xr.Dataset],
    ref_ds: xr.Dataset,
    variable: str,
    region: str = "pana",
    output_dir: Path | None = None,
    region_mask: xr.Dataset | None = None
) -> Dict[str, Any]:
    """Perform comprehensive spatial analysis between multiple model experiments and reference.

    This function applies ILAMB3 analysis methods and custom metrics to compare
    spatial characteristics of multiple model outputs against a single reference dataset.

    Args:
        model_datasets: Dict mapping experiment names to model datasets.
        ref_ds: Reference dataset (observations).
        variable: Variable name to analyze (e.g., "gpp", "et").
        region: Region label for analysis. Default is "pana" (Pan Amazon).
        output_dir: Optional directory to save plots. If None, plots are displayed.
        region_mask: Optional RAISG mask dataset to apply to reference mean field plots.

    Returns:
        dict: Dictionary containing:
            - 'scalars': pd.DataFrame with scalar metrics and scores (includes 'experiment' column)
            - 'ref_gridded': xr.Dataset with reference gridded outputs
            - 'com_gridded': Dict[str, xr.Dataset] with comparison gridded outputs per experiment
            - 'figures': dict of matplotlib Figure objects
    """
    experiment_names = list(model_datasets.keys())
    n_experiments = len(experiment_names)

    results = {
        'scalars': pd.DataFrame(),
        'ref_gridded': xr.Dataset(),
        'com_gridded': {exp: xr.Dataset() for exp in experiment_names},
        'figures': {}
    }

    # Ensure units attribute exists
    if 'units' not in ref_ds[variable].attrs:
        ref_ds[variable].attrs['units'] = '1'
    for exp_name, model_ds in model_datasets.items():
        if 'units' not in model_ds[variable].attrs:
            model_ds[variable].attrs['units'] = ref_ds[variable].attrs['units']

    # =========================================================================
    # 1. COMPUTE TEMPORAL MEANS FOR SPATIAL ANALYSIS
    # =========================================================================
    print(f"Computing temporal means for {n_experiments} experiment(s)...")
    mean_fields_per_exp = {}
    ref_mean = None
    try:
        ref_mean = ref_ds[variable].mean(dim='time')
        results['ref_gridded']['mean'] = ref_mean

        for exp_name, model_ds in model_datasets.items():
            model_mean = model_ds[variable].mean(dim='time')
            results['com_gridded'][exp_name]['mean'] = model_mean
            mean_fields_per_exp[exp_name] = model_mean

        # Create multi-experiment mean field comparison figure
        if mean_fields_per_exp and ref_mean is not None:
            fig_mean = _plot_multi_experiment_mean_fields(
                ref_mean,
                mean_fields_per_exp,
                variable=variable,
                region=region,
                region_mask=region_mask
            )
            results['figures']['mean_field_comparison'] = fig_mean

    except Exception as e:
        print(f"Mean field computation failed: {e}")

    # =========================================================================
    # 2. SPATIAL DISTRIBUTION ANALYSIS (per experiment) + TAYLOR DIAGRAM
    # =========================================================================
    print(f"Running spatial distribution analysis for {n_experiments} experiment(s)...")
    taylor_metrics_per_exp = {}
    try:
        spatial_analyzer = spatial_distribution_analysis(
            required_variable=variable,
            regions=[region]
        )

        for exp_name, model_ds in model_datasets.items():
            spatial_df, spatial_ref, spatial_com = spatial_analyzer(ref_ds, model_ds)
            spatial_df['experiment'] = exp_name
            results['scalars'] = pd.concat([results['scalars'], spatial_df], ignore_index=True)

            # Extract Taylor diagram metrics for this experiment
            exp_region_df = spatial_df[spatial_df['region'] == region]
            try:
                norm_std = float(exp_region_df[exp_region_df['name'] == 'Normalized Standard Deviation']['value'].iloc[0])
                corr = float(exp_region_df[exp_region_df['name'] == 'Correlation']['value'].iloc[0])
                taylor_metrics_per_exp[exp_name] = {'norm_std': norm_std, 'correlation': corr}
            except (IndexError, KeyError):
                pass

        # Create multi-experiment Taylor diagram
        if taylor_metrics_per_exp:
            fig_taylor = _plot_multi_experiment_taylor_diagram(
                taylor_metrics_per_exp,
                variable=variable,
                region=region
            )
            results['figures']['spatial_taylor_diagram'] = fig_taylor

    except Exception as e:
        print(f"Spatial distribution analysis failed: {e}")

    # =========================================================================
    # 3. SPATIAL BIAS ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running spatial bias analysis for {n_experiments} experiment(s)...")
    spatial_bias_per_exp = {}
    try:
        ref_mean = ref_ds[variable].mean(dim='time')

        for exp_name, model_ds in model_datasets.items():
            model_mean = model_ds[variable].mean(dim='time')

            ref_nested, model_nested = cmp.nest_spatial_grids(ref_mean, model_mean)
            spatial_bias = model_nested - ref_nested

            results['com_gridded'][exp_name]['spatial_bias'] = spatial_bias
            spatial_bias_per_exp[exp_name] = spatial_bias

            # Compute spatial bias statistics
            spatial_bias_metrics = _compute_spatial_bias_metrics(
                ref_nested, model_nested, spatial_bias, region
            )

            bias_df = pd.DataFrame([
                {
                    'experiment': exp_name,
                    'source': 'Reference',
                    'region': region,
                    'analysis': 'Spatial Bias',
                    'name': 'Spatial Mean',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': spatial_bias_metrics['ref_spatial_mean']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Spatial Bias',
                    'name': 'Spatial Mean',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': spatial_bias_metrics['com_spatial_mean']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Spatial Bias',
                    'name': 'Mean Bias',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': spatial_bias_metrics['mean_bias']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Spatial Bias',
                    'name': 'Bias RMSE',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': spatial_bias_metrics['bias_rmse']
                },
                {
                    'experiment': exp_name,
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

        # Create multi-experiment spatial bias map
        if spatial_bias_per_exp:
            fig_spatial_bias = _plot_multi_experiment_bias_maps(
                spatial_bias_per_exp,
                title=f'{variable.upper()} Spatial Bias (Model - Reference)',
                region=region
            )
            results['figures']['spatial_bias_map'] = fig_spatial_bias

    except Exception as e:
        print(f"Spatial bias analysis failed: {e}")

    # =========================================================================
    # 4. PATTERN CORRELATION ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running pattern correlation analysis for {n_experiments} experiment(s)...")
    pattern_per_exp = {}
    try:
        for exp_name, model_ds in model_datasets.items():
            pattern_metrics = _compute_pattern_correlation(ref_ds, model_ds, variable, region)
            pattern_per_exp[exp_name] = pattern_metrics

            pattern_df = pd.DataFrame([
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Pattern Correlation',
                    'name': 'Spatial Correlation',
                    'type': 'scalar',
                    'units': '1',
                    'value': pattern_metrics['spatial_correlation']
                },
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Pattern Correlation',
                    'name': 'Centered RMSE',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': pattern_metrics['centered_rmse']
                },
                {
                    'experiment': exp_name,
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

        # Create multi-experiment scatter plot
        if pattern_per_exp:
            fig_scatter = _plot_multi_experiment_scatter(
                pattern_per_exp,
                variable=variable,
                region=region
            )
            results['figures']['spatial_scatter'] = fig_scatter

    except Exception as e:
        print(f"Pattern correlation analysis failed: {e}")

    # =========================================================================
    # 5. SPATIAL RMSE ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running spatial RMSE analysis for {n_experiments} experiment(s)...")
    try:
        for exp_name, model_ds in model_datasets.items():
            ref_trimmed, model_trimmed = cmp.trim_time(ref_ds, model_ds)
            ref_var = ref_trimmed[variable]
            model_var = model_trimmed[variable]

            ref_nested, model_nested = cmp.nest_spatial_grids(ref_var, model_var)

            rmse_map = np.sqrt(((model_nested - ref_nested) ** 2).mean(dim='time'))
            results['com_gridded'][exp_name]['rmse_map'] = rmse_map

            rmse_regional = Regions.restrict_to_region(rmse_map, region)
            mean_rmse = float(rmse_regional.mean())
            max_rmse = float(rmse_regional.max())

            rmse_df = pd.DataFrame([
                {
                    'experiment': exp_name,
                    'source': 'Comparison',
                    'region': region,
                    'analysis': 'Spatial RMSE',
                    'name': 'Mean RMSE',
                    'type': 'scalar',
                    'units': ref_ds[variable].attrs.get('units', '1'),
                    'value': mean_rmse
                },
                {
                    'experiment': exp_name,
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

    except Exception as e:
        print(f"Spatial RMSE analysis failed: {e}")

    # =========================================================================
    # 6. ZONAL MEAN ANALYSIS (per experiment)
    # =========================================================================
    print(f"Running zonal mean analysis for {n_experiments} experiment(s)...")
    zonal_per_exp = {}
    ref_zonal = None
    try:
        for exp_name, model_ds in model_datasets.items():
            zonal_metrics = _compute_zonal_means(ref_ds, model_ds, variable, region)

            if ref_zonal is None:
                ref_zonal = zonal_metrics['ref_zonal']
                results['ref_gridded']['zonal_mean'] = ref_zonal

            results['com_gridded'][exp_name]['zonal_mean'] = zonal_metrics['com_zonal']
            zonal_per_exp[exp_name] = zonal_metrics['com_zonal']

        # Create multi-experiment zonal mean figure
        if zonal_per_exp and ref_zonal is not None:
            fig_zonal = _plot_multi_experiment_zonal_means(
                ref_zonal,
                zonal_per_exp,
                variable=variable,
                region=region
            )
            results['figures']['zonal_means'] = fig_zonal

    except Exception as e:
        print(f"Zonal mean analysis failed: {e}")

    # =========================================================================
    # 7. SPATIAL QUANTILE ANALYSIS (per experiment) + Q-Q PLOT
    # =========================================================================
    print(f"Running spatial quantile analysis for {n_experiments} experiment(s)...")
    quantiles_per_exp = {}
    ref_quantiles_data = None
    try:
        for exp_name, model_ds in model_datasets.items():
            quantile_metrics = _compute_spatial_quantiles(ref_ds, model_ds, variable, region)

            if ref_quantiles_data is None:
                ref_quantiles_data = quantile_metrics['ref_quantiles']

            quantiles_per_exp[exp_name] = quantile_metrics['com_quantiles']

            quantile_df = pd.DataFrame([
                {
                    'experiment': exp_name,
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

        # Create multi-experiment Q-Q plot
        if quantiles_per_exp and ref_quantiles_data is not None:
            fig_qq = _plot_multi_experiment_qq(
                ref_quantiles_data,
                quantiles_per_exp,
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


def _get_mask_dataarray(mask_ds: xr.Dataset) -> xr.DataArray:
    """Return a mask DataArray from a mask Dataset."""
    if 'mask' in mask_ds.data_vars:
        mask_da = mask_ds['mask']
    else:
        mask_da = next(iter(mask_ds.data_vars.values()))

    if 'time' in mask_da.dims:
        mask_da = mask_da.isel(time=0)

    return mask_da


def _apply_reference_mask(ref_mean: xr.DataArray, region_mask: xr.Dataset | None) -> xr.DataArray:
    """Apply a region mask to the reference mean field for plotting."""
    if region_mask is None:
        return ref_mean

    mask_da = _get_mask_dataarray(region_mask)

    try:
        ref_nested, mask_nested = cmp.nest_spatial_grids(ref_mean, mask_da)
    except Exception:
        ref_nested, mask_nested = ref_mean, mask_da

    mask_clean = xr.where(mask_nested.notnull(), mask_nested, 0)
    mask_bool = mask_clean.astype(bool)

    return ref_nested.where(mask_bool)


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
    """Print a summary of spatial analysis results for multiple experiments."""
    print("\n" + "=" * 70)
    print(f"SPATIAL ANALYSIS SUMMARY: {variable.upper()} ({region})")
    print("=" * 70)

    # Get experiment list
    experiments = df['experiment'].unique() if 'experiment' in df.columns else ['default']

    for exp in experiments:
        exp_df = df[df['experiment'] == exp] if 'experiment' in df.columns else df
        print(f"\n>>> Experiment: {exp}")
        print("-" * 60)

        # Filter for scores
        scores = exp_df[exp_df['type'] == 'score']
        if not scores.empty:
            print("\n  SCORES (0-1, higher is better):")
            for _, row in scores.iterrows():
                print(f"    {row['analysis']:25s} | {row['name']:20s}: {row['value']:.3f}")

        # Key scalars by analysis
        print("\n  KEY METRICS:")
        scalars = exp_df[exp_df['type'] == 'scalar']
        for analysis in scalars['analysis'].unique():
            analysis_df = scalars[scalars['analysis'] == analysis]
            print(f"\n    {analysis}:")
            for _, row in analysis_df.iterrows():
                print(f"      {row['source']:12s} {row['name']:20s}: {row['value']:10.4f} {row['units']}")

    print("\n" + "=" * 70)


# =============================================================================
# SPATIAL PLOTTING FUNCTIONS
# =============================================================================

def _plot_multi_experiment_taylor_diagram(
    taylor_metrics_per_exp: Dict[str, Dict[str, float]],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot Taylor diagram for multiple experiments.

    The Taylor diagram shows how well model patterns match the reference
    in terms of correlation (angle) and normalized standard deviation (radius).
    The reference is at (0, 1) - perfect correlation and matching variability.

    Args:
        taylor_metrics_per_exp: Dict mapping experiment names to metrics dict
            containing 'norm_std' and 'correlation' keys.
        variable: Variable name for title.
        region: Region name for title.

    Returns:
        matplotlib Figure with Taylor diagram.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Reference point (correlation=1, normalized std=1)
    ax.plot(0, 1, 'ko', markersize=15, label='Reference', zorder=10)

    # Plot each experiment
    for idx, (exp_name, metrics) in enumerate(taylor_metrics_per_exp.items()):
        norm_std = metrics['norm_std']
        corr = metrics['correlation']

        if not np.isnan(corr) and not np.isnan(norm_std):
            # Convert correlation to angle (theta = arccos(correlation))
            theta = np.arccos(np.clip(corr, -1, 1))
            color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
            ax.plot(theta, norm_std, 's', color=color, markersize=12,
                    label=f'{exp_name} (R={corr:.2f}, σ={norm_std:.2f})', zorder=5)

    # Add reference circles for RMSE (centered on reference point)
    # RMSE contours: E' = sqrt(1 + σ² - 2σR) where σ is norm_std and R is correlation
    angles = np.linspace(0, np.pi/2, 100)
    for rmse in [0.25, 0.5, 0.75, 1.0, 1.25]:
        # For each angle (correlation), compute the radius (norm_std) that gives this RMSE
        # RMSE² = 1 + σ² - 2σcos(θ)
        # This is a circle centered at (0, 1) with radius = RMSE
        rs = []
        for angle in angles:
            corr_val = np.cos(angle)
            # Solve: rmse² = 1 + σ² - 2σ*corr for σ
            # σ² - 2*corr*σ + (1 - rmse²) = 0
            # σ = corr ± sqrt(corr² - 1 + rmse²)
            discriminant = corr_val**2 - 1 + rmse**2
            if discriminant >= 0:
                sigma = corr_val + np.sqrt(discriminant)
                if sigma > 0:
                    rs.append(sigma)
                else:
                    rs.append(np.nan)
            else:
                rs.append(np.nan)
        ax.plot(angles, rs, 'k:', linewidth=0.5, alpha=0.5)

    # Add correlation arc labels
    corr_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for corr_val in corr_labels:
        angle = np.arccos(corr_val)
        ax.annotate(f'{corr_val}', xy=(angle, ax.get_ylim()[1]),
                   fontsize=8, ha='center', va='bottom')

    # Configure axes
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('E')
    ax.set_rlabel_position(0)

    # Set reasonable radial limits
    max_std = max((m['norm_std'] for m in taylor_metrics_per_exp.values() if not np.isnan(m['norm_std'])), default=2.0)
    ax.set_ylim(0, max(2.0, max_std * 1.2))
    ax.set_rticks([0.5, 1.0, 1.5, 2.0])

    # Labels
    ax.set_xlabel('Normalized Standard Deviation', fontsize=12, labelpad=20)
    ax.text(np.pi/4, ax.get_ylim()[1] * 1.15, 'Correlation', fontsize=12,
            ha='center', va='bottom', rotation=-45)

    ax.set_title(f'Taylor Diagram: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=10)

    plt.tight_layout()
    return fig


def _plot_multi_experiment_mean_fields(
    ref_mean: xr.DataArray,
    mean_fields_per_exp: Dict[str, xr.DataArray],
    variable: str,
    region: str,
    region_mask: xr.Dataset | None = None
) -> plt.Figure:
    """Plot mean field comparison for multiple experiments."""
    n_exp = len(mean_fields_per_exp)
    ncols = min(3, n_exp + 1)  # +1 for reference
    nrows = (n_exp + 1 + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()},
        squeeze=False
    )

    # Apply region mask to reference (for plotting) and find common colorbar limits
    masked_ref_mean = _apply_reference_mask(ref_mean, region_mask)
    ref_regional = Regions.restrict_to_region(masked_ref_mean, region)
    all_means = [ref_regional] + [Regions.restrict_to_region(m, region) for m in mean_fields_per_exp.values()]
    vmin = min(float(m.min()) for m in all_means)
    vmax = max(float(m.max()) for m in all_means)

    # Plot reference
    ax = axes[0, 0]
    ref_regional.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': ref_mean.attrs.get('units', ''), 'shrink': 0.8}
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.set_title('Reference', fontsize=12, fontweight='bold')

    # Plot each experiment
    for idx, (exp_name, model_mean) in enumerate(mean_fields_per_exp.items()):
        plot_idx = idx + 1
        row, col = plot_idx // ncols, plot_idx % ncols
        ax = axes[row, col]

        model_regional = Regions.restrict_to_region(model_mean, region)
        model_regional.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={'label': model_mean.attrs.get('units', ''), 'shrink': 0.8}
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        ax.set_title(f'{exp_name}', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_exp + 1, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Mean {variable.upper()} Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def _plot_multi_experiment_scatter(
    pattern_per_exp: Dict[str, Dict[str, Any]],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot spatial scatter for multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Find global limits
    all_ref = []
    all_com = []
    for metrics in pattern_per_exp.values():
        all_ref.extend(metrics['ref_values'])
        all_com.extend(metrics['com_values'])

    lims = [min(min(all_ref), min(all_com)), max(max(all_ref), max(all_com))]

    # Plot 1:1 line
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line', zorder=1)

    # Plot each experiment
    for idx, (exp_name, metrics) in enumerate(pattern_per_exp.items()):
        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
        ax.scatter(
            metrics['ref_values'],
            metrics['com_values'],
            alpha=0.3,
            s=10,
            c=color,
            edgecolors='none',
            label=f"{exp_name} (R={metrics['spatial_correlation']:.2f})"
        )

    ax.set_xlabel(f'Reference {variable.upper()}', fontsize=12)
    ax.set_ylabel(f'Model {variable.upper()}', fontsize=12)
    ax.set_title(f'Spatial Scatter: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def _plot_multi_experiment_zonal_means(
    ref_zonal: xr.DataArray,
    zonal_per_exp: Dict[str, xr.DataArray],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot zonal mean profiles for multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 12))

    ref_lats = ref_zonal['lat'].values
    ref_vals = ref_zonal.values

    # Plot reference
    ax.plot(ref_vals, ref_lats, 'k-', linewidth=3, label='Reference', zorder=10)

    # Plot each experiment
    for idx, (exp_name, com_zonal) in enumerate(zonal_per_exp.items()):
        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
        com_lats = com_zonal['lat'].values
        com_vals = com_zonal.values
        ax.plot(com_vals, com_lats, '--', color=color, linewidth=2, label=exp_name, alpha=0.8)

    ax.set_xlabel(f'{variable.upper()} ({ref_zonal.attrs.get("units", "")})', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Zonal Mean: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_multi_experiment_qq(
    ref_quantiles: Dict[float, float],
    quantiles_per_exp: Dict[str, Dict[float, float]],
    variable: str,
    region: str
) -> plt.Figure:
    """Plot Q-Q diagram for multiple experiments.

    Compares quantile distributions of each experiment against the reference.
    Points on the 1:1 line indicate perfect agreement.

    Args:
        ref_quantiles: Dict mapping quantile values (0-1) to reference values.
        quantiles_per_exp: Dict mapping experiment names to their quantile dicts.
        variable: Variable name for title.
        region: Region name for title.

    Returns:
        matplotlib Figure with multi-experiment Q-Q plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    quantile_keys = list(ref_quantiles.keys())
    ref_vals = [ref_quantiles[q] for q in quantile_keys]

    # Find global limits
    all_vals = list(ref_vals)
    for exp_quantiles in quantiles_per_exp.values():
        all_vals.extend([exp_quantiles[q] for q in quantile_keys])
    lims = [min(all_vals), max(all_vals)]

    # 1:1 line
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line', zorder=1)

    # Plot each experiment
    for idx, (exp_name, exp_quantiles) in enumerate(quantiles_per_exp.items()):
        color = EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]
        exp_vals = [exp_quantiles[q] for q in quantile_keys]

        ax.scatter(ref_vals, exp_vals, s=100, c=color, edgecolors='black',
                   label=exp_name, zorder=3, alpha=0.8)

        # Connect points with lines for this experiment
        ax.plot(ref_vals, exp_vals, '-', color=color, linewidth=1.5, alpha=0.5, zorder=2)

    # Label quantile points (only once, using first experiment position)
    first_exp = list(quantiles_per_exp.keys())[0]
    first_exp_vals = [quantiles_per_exp[first_exp][q] for q in quantile_keys]
    for q, rx, cx in zip(quantile_keys, ref_vals, first_exp_vals):
        ax.annotate(f'Q{int(q*100)}', (rx, cx), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, alpha=0.7)

    ax.set_xlabel(f'Reference Quantiles ({variable.upper()})', fontsize=12)
    ax.set_ylabel(f'Model Quantiles ({variable.upper()})', fontsize=12)
    ax.set_title(f'Q-Q Plot: {variable.upper()} ({region})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def run_benchmark(
    variable: str = "gpp",
    dataset_name: str = "GOSIF",
    filename: str = "gpp_GOSIF_2000-2024.nc",
    experiments: List[str] | None = None,
    output_dir: str | None = None
) -> Dict[str, Any]:
    """Run complete benchmark for a variable comparing multiple experiments against one reference.

    This is the main entry point for benchmarking CAETE model outputs against
    reference datasets. It supports comparing multiple experiments in the same
    benchmark run.

    Args:
        variable: Variable to benchmark (e.g., "gpp", "et").
        dataset_name: Reference dataset name.
        filename: Reference dataset filename.
        experiments: List of experiment names. If None, uses all available experiments.
        output_dir: Optional directory to save outputs.

    Returns:
        Dictionary with benchmark results including:
            - 'experiments': list of experiment names
            - 'variable': variable name
            - 'temporal_results': temporal analysis results
            - 'spatial_results': spatial analysis results
            - 'all_scalars': combined DataFrame with all metrics
            - 'model_datasets': Dict of model datasets
            - 'ref_ds': reference dataset
    """
    print(f"\n{'='*70}")
    print(f"RUNNING BENCHMARK: {variable.upper()}")
    print(f"Reference: {dataset_name}/{filename}")
    print(f"{'='*70}\n")

    # Normalize experiments input (str -> [str], other iterables -> list)
    experiments = _normalize_experiments(experiments)

    # Load and prepare data for multiple experiments
    model_datasets, ref_ds, region_mask, model_masks = get_models_and_ref(
        variable=variable,
        dataset_name=dataset_name,
        filename=filename,
        experiments=experiments
    )

    experiment_names = list(model_datasets.keys())
    print(f"Experiments: {experiment_names}")

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir) / variable
    else:
        output_path = None

    # Run temporal analysis (multi-experiment)
    print("\n" + "-"*70)
    print("TEMPORAL ANALYSIS")
    print("-"*70)
    temporal_results = temporal_analysis(
        model_datasets=model_datasets,
        ref_ds=ref_ds,
        variable=variable,
        region="pana",
        output_dir=output_path
    )

    # Run spatial analysis (multi-experiment)
    print("\n" + "-"*70)
    print("SPATIAL ANALYSIS")
    print("-"*70)
    spatial_results = spatial_analysis(
        model_datasets=model_datasets,
        ref_ds=ref_ds,
        variable=variable,
        region="pana",
        output_dir=output_path,
        region_mask=region_mask
    )

    # Combine all scalars
    all_scalars = pd.concat([
        temporal_results['scalars'],
        spatial_results['scalars']
    ], ignore_index=True)

    # Print overall summary
    _print_overall_benchmark_summary(all_scalars, variable, experiment_names)

    return {
        'experiments': experiment_names,
        'variable': variable,
        'temporal_results': temporal_results,
        'spatial_results': spatial_results,
        'all_scalars': all_scalars,
        'model_datasets': model_datasets,
        'ref_ds': ref_ds
    }


def _print_overall_benchmark_summary(df: pd.DataFrame, variable: str, experiments: List[str]):
    """Print overall benchmark summary with all scores for multiple experiments."""
    print("\n" + "=" * 80)
    print(f"OVERALL BENCHMARK SUMMARY: {variable.upper()}")
    print(f"Experiments: {experiments}")
    print("=" * 80)

    for exp in experiments:
        exp_df = df[df['experiment'] == exp] if 'experiment' in df.columns else df
        print(f"\n>>> Experiment: {exp}")
        print("-" * 70)

        # All scores for this experiment
        scores = exp_df[exp_df['type'] == 'score']
        if not scores.empty:
            print("\n  SCORES (0-1, higher is better):")
            for _, row in scores.iterrows():
                print(f"    {row['analysis']:30s} | {row['name']:25s}: {row['value']:.3f}")

            # Overall score (mean of all scores)
            mean_score = scores['value'].mean()
            print("-" * 60)
            print(f"    {'OVERALL MEAN SCORE':30s} | {'':25s}: {mean_score:.3f}")

    # Compare experiments if multiple
    if len(experiments) > 1:
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPARISON")
        print("=" * 80)
        print("\n  Mean Scores by Experiment:")
        for exp in experiments:
            exp_df = df[df['experiment'] == exp] if 'experiment' in df.columns else df
            scores = exp_df[exp_df['type'] == 'score']
            if not scores.empty:
                mean_score = scores['value'].mean()
                print(f"    {exp:40s}: {mean_score:.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run multi-experiment benchmark for ET variable
    et_results = run_benchmark(
        variable="et",
        dataset_name="GLEAMv4.2b",
        filename="et_GLEAM_2003-2024.nc",
        experiments=None,  # Use all available experiments
        output_dir="../outputs/benchmark_results"
    )

    # Run multi-experiment benchmark for GPP variable
    gpp_results = run_benchmark(
        variable="gpp",
        dataset_name="GOSIF",
        filename="gpp_GOSIF_2000-2024.nc",
        experiments=None,  # Use all available experiments
        output_dir="../outputs/benchmark_results"
    )

