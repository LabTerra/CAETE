"""Refactored benchmark module for CAETE model evaluation.

This module provides functions to load, preprocess, and compare CAETE model outputs
with reference datasets for benchmarking purposes.

Requires CDO to be installed and accessible in the system path.
"""

import sys

# This program requires CDO to be installed and accessible in the system path.
# if sys.platform != "linux":
#     raise EnvironmentError("This script only works in linux/WSL environments.")


from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.ma as ma
import ilamb3
import xarray as xr

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


def get_model_data(variable: str, use_cache: bool = True) -> Tuple[xr.Dataset, str, xr.Dataset]:
    """Load and preprocess CAETE model output with caching support.

    Retrieves CAETE model output for the specified variable, applies necessary
    unit conversions, and extracts the model mask from the data.

    Args:
        variable: Variable name to read from CAETE output (e.g., "gpp", "rnpp", "et").
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
    model_filepath, experiment = get_caete_dataset(caete_data, caete_varname)

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
    dataset_name: str = "MADANI",
    filename: str = "gpp_masked.nc"
) -> Tuple[xr.Dataset, xr.Dataset, str, xr.Dataset, xr.Dataset]:
    """Orchestrator function to load and prepare model and reference datasets for comparison.

    This function combines the functionality of get_model_data, get_reference_data,
    conform_datasets, and get_region_mask to provide a complete workflow for
    benchmarking CAETE model outputs against reference datasets.

    Args:
        variable: Variable name to read from CAETE output. Default is "gpp".
        dataset_name: Name of the reference dataset. Default is "MADANI".
        filename: Filename within the reference datasets. Default is "gpp_masked.nc".

    Returns:
        A tuple containing:
            - model_ds (xr.Dataset): Preprocessed and conformed CAETE dataset.
            - ref_ds (xr.Dataset): Preprocessed and conformed reference dataset.
            - experiment (str): Experiment identifier from CAETE output.
            - region_mask (xr.Dataset): RAISG mask for the Pan Amazon region.
            - model_mask (xr.Dataset): Mask indicating valid model data cells.
    """
    # Step 1: Get model data with mask
    model_ds, experiment, model_mask = get_model_data(variable)

    # Step 2: Get reference data
    ref_ds = get_reference_data(dataset_name, filename)

    # Step 3: Conform datasets (align, regrid, restrict to region)
    model_ds, ref_ds = conform_datasets(model_ds, ref_ds, variable)

    # Step 4: Get region mask (RAISG)
    region_mask = get_region_mask()

    return model_ds, ref_ds, experiment, region_mask, model_mask


def temporal_analysis(model_ds: xr.Dataset, ref_ds: xr.Dataset, variable: str = "gpp"):
    """Perform temporal analysis between model and reference datasets.

    Args:
        model_ds: Model dataset.
        ref_ds: Reference dataset.
        variable: Variable name to analyze. Default is "gpp".

    Returns:
        dict: Dictionary containing temporal analysis results.
    """
    pass


def spatial_analysis(model_ds: xr.Dataset, ref_ds: xr.Dataset, variable: str = "gpp"):
    """Perform spatial analysis between model and reference datasets.

    Args:
        model_ds: Model dataset.
        ref_ds: Reference dataset.
        variable: Variable name to analyze. Default is "gpp".

    Returns:
        dict: Dictionary containing spatial analysis results.
    """
    pass


def main():
    """Main function demonstrating usage of the benchmark module."""
    model_gpp, ref_gpp, experiment, region_mask, model_mask = get_model_and_ref(
        variable="gpp",
        dataset_name="MADANI",
        filename="gpp_masked.nc"
    )
    print(f"Loaded experiment: {experiment}")
    print(f"Model dataset: {model_gpp}")
    print(f"Reference dataset: {ref_gpp}")


if __name__ == "__main__":
    # Example usage with ET variable
    m, r, e, mask, mask_model = get_model_and_ref(
        variable="et",
        dataset_name="GLEAMv4.2a",
        filename="et_GLEAM_PANA_1980_2024.nc"
    )
