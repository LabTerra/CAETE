import sys

# This program requires CDO to be installed and accessible in the system path.
# if sys.platform != "linux":
#     raise EnvironmentError("This script only works in linux/WSL environments.")



from pathlib import Path
import numpy as np
import ilamb3
import xarray as xr

from benchmark_pp import get_dataset, get_caete_dataset, monsum
from benchmark_utils import available_variables as caete_data
from benchmark_utils import ref_datasets, get_caete_varname
from benchmark_utils import BENCHMARCK_CACHE_DIR, ensure_cache_dir

ensure_cache_dir()

# RAISG mask file for Pan Amazon region
RAISG_MASK_FILE = Path("../input/mask/mask.nc")

# Define benchmark region(s)
Regions = ilamb3.regions.Regions()

# Pan Amazon region
lons = [-80.25, -43.25]
lats = [-21.75, 10.75]

# Add Pan Amazon region to Regions
Regions.add_latlon_bounds(
    label="pana",
    name="Pan Amazon",
    lats=lats,
    lons=lons,
    source="Pan Amazon (Northern South America)"
)

def get_model_and_ref(variable="gpp", dataset_name="MADANI", filename="gpp_masked.nc"):

    """Reads data from CAETE output and  a reference dataset and preprocesses it for comparison.
    Args:
        variable (str): Variable name to read from CAETE output. Default is "gpp".
        dataset_name (str): Name of the reference dataset. Default is "MADANI".
        filename (str): Filename within the reference datasets. Default is "gpp_masked.nc".
    Returns:
        model_gpp_ds (xarray.Dataset): Preprocessed CAETE GPP dataset.
        ref_gpp_ds (xarray.Dataset): Preprocessed reference GPP dataset.
        experiment (str): Experiment identifier from CAETE output.
        mask (xarray.Dataset): Mask dataset for the Pan Amazon region.
    """

    # Get reference GPP data filepath
    ref_data = get_dataset(ref_datasets, dataset_name, filename)

    # Caete outputs are daily GPP in kgC/m2/year. Get the file path and the experiment name (from filename).
    model_variable, experiment = get_caete_dataset(caete_data, get_caete_varname(variable)) # Path to CAETE GPP output file (daily)

    ## Preprocess CAETE output to monthly sums in kgC/m2/month
    # Cache model output
    got_model = False
    got_reference = False

    # Define cached file path
    var_CAETE = BENCHMARCK_CACHE_DIR / Path(f"{variable}_{experiment}.nc")

    # Preprocess CAETE output if needed and not cached
    if not var_CAETE.exists():
        # Preprocess CAETE output based on variable
        match variable:
            case "gpp": # Model GPP need conversion from daily kgC/m2/year to monthly kgC/m2/month
                monsum(model_variable, var_CAETE, variable, 0.00273791, "kg*m**-2*month**-1")
            case "rnpp": # Model RNPP needs conversion from daily gC/m2/day to monthly kgC/m2/month
                monsum(model_variable, var_CAETE, variable, 0.001, "kg*m**-2*month**-1")
            case "et": # Model ET needs conversion from daily mm/day to monthly mm/month
                monsum(model_variable, var_CAETE, variable, 1, "mm*month**-1")
            case _:
                raise ValueError(f"Variable '{variable}' not supported for preprocessing.")
    else:
        print(f"Using cached file {var_CAETE}")

    # Open model dataset
    try:
        model_variable_ds = xr.open_dataset(var_CAETE)
        got_model = True
    except Exception as e:
        print(f"Error opening cached CAETE GPP dataset: {e}")

    # Open reference dataset
    try:
        ref_variable_ds = xr.open_dataset(ref_data)
        got_reference = True
    except Exception as e:
        print(f"Error opening reference GPP dataset: {e}")

    if not (got_model and got_reference):
        raise RuntimeError("Failed to load model or reference datasets.")

    # Adjust datasets to be comparable
    ref_variable_ds, model_variable_ds = ilamb3.compare.adjust_lon(ref_variable_ds, model_variable_ds)
    ref_variable_ds, model_variable_ds = ilamb3.compare.make_comparable(ref_variable_ds, model_variable_ds, variable)
    ref_variable_ds = Regions.restrict_to_region(ref_variable_ds, "pana")

    # Get pan amazon mask
    mask = Regions.restrict_to_region(xr.open_dataset(RAISG_MASK_FILE), "pana")

    # Get model mask as xarray Dataset with proper coordinates
    # Extract the mask from the underlying masked array
    import numpy.ma as ma
    data_values = model_variable_ds[variable].isel(time=0).values
    if ma.is_masked(data_values):
        mask_array = ma.getmask(data_values)
    else:
        mask_array = np.isnan(data_values)

    model_mask_data = xr.DataArray(
        mask_array,
        coords={'lat': model_variable_ds.lat, 'lon': model_variable_ds.lon},
        dims=['lat', 'lon'],
        name='mask'
    ).to_dataset()

    return model_variable_ds, ref_variable_ds, experiment, mask, model_mask_data



def temporal_analysis(model_ds, ref_ds, variable="gpp"):
    """Perform temporal analysis between model and reference datasets.

    Args:
        model_ds (xarray.Dataset): Model dataset.
        ref_ds (xarray.Dataset): Reference dataset.
        variable (str): Variable name to analyze. Default is "gpp".

    Returns:
        dict: Dictionary containing temporal analysis results.
    """
    pass


def spatial_analysis(model_ds, ref_ds, variable="gpp"):
    """Perform spatial analysis between model and reference datasets.

    Args:
        model_ds (xarray.Dataset): Model dataset.
        ref_ds (xarray.Dataset): Reference dataset.
        variable (str): Variable name to analyze. Default is "gpp".

    Returns:
        dict: Dictionary containing spatial analysis results.
    """
    pass


def main():
    (mgpp,
     rgpp,
     egpp,
     emask,
     em2) = get_model_and_ref(variable="gpp",
                          dataset_name="MADANI",
                          filename="gpp_masked.nc")



if __name__ == "__main__":
    # clean_cache()
    # m,r,e,mask, mask_model = make_a_comparisson(variable="gpp", dataset_name="MADANI", filename="gpp_masked.nc")
    m,r,e,mask, mask_model = get_model_and_ref(variable="et", dataset_name="GLEAMv3.3a", filename="et_masked.nc")
