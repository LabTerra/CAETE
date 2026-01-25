# Post - Processing CAETE utilities for netcdf files and tables from CAETE model runs.
## Works in linux and WSL. Requires CDO installed.

from pathlib import Path
try:
    from pycdo import cdo # type: ignore
except ImportError:
    raise ImportError("pycdo (requires cdo) module not found. Please install pycdo and cdo to use this script.")


def get_caete_dataset(available_variables, variable=None, experiment=None):
    """Get the path to a CAETE output dataset file.

    Args:
        available_variables (dict): Nested dictionary {experiment: {variable: path}}.
        variable (str): Variable name prefix of the output file.
        experiment (str, optional): Experiment name. If None, uses the first available experiment.

    Returns:
        tuple: (Path to the dataset file, experiment name)

    Raises:
        ValueError: If the variable or experiment is not found.

    """
    assert isinstance(available_variables, dict)

    if variable is None:
        raise ValueError("Variable name must be provided.")

    # If no experiment specified, use the first available experiment
    if experiment is None:
        if not available_variables:
            raise ValueError("No experiments found in available variables.")
        experiment = next(iter(available_variables))

    if experiment not in available_variables:
        raise ValueError(f"Experiment '{experiment}' not found. Available: {list(available_variables.keys())}")

    if variable not in available_variables[experiment]:
        raise ValueError(f"Variable '{variable}' not found in experiment '{experiment}'. "
                        f"Available: {list(available_variables[experiment].keys())}")

    filepath = available_variables[experiment][variable]
    return filepath, experiment


def get_dataset(ref_datasets, dataset, filename):
    """Get the path to a reference dataset file.

    Args:
        ref_datasets (dict): Dictionary of reference datasets, build in benchmark_utils module.
        dataset (str): Name of the dataset.
        filename (str): Filename within the dataset directory.

    Returns:
        a Path to the dataset file.

    """
    if dataset not in ref_datasets:
        raise ValueError(f"Dataset '{dataset}' not found in reference datasets.")
    if filename not in ref_datasets[dataset]:
        raise ValueError(f"File '{filename}' not found in dataset '{dataset}'.")
    return ref_datasets[dataset][filename]


def monsum(input_file:Path, output_file:Path, variable:str, conv_fac:int|float, units:str) -> Path:
    """
    Accumulate (monthly) a netCDF file with daily data, adjusting units accordingly.
    Apply a conversion factor before summation. This is useful for converting daily data
    to monthly totals with unit adjustments.
    Args:
        input_file (Path): Path to the input netCDF file.
        output_file (Path): Path to the output netCDF file.
        variable (str): Variable name to be set in the output file.
        conv_fac (int|float): Conversion factor to adjust units. It is applied before summation (should convert whatever time units to daily).
        units (str): Units string to set for the output variable.
    """
    # All args must be provided
    if not all([input_file, output_file, variable, conv_fac, units]):
        raise ValueError("All arguments must be provided to monsum function.")
    assert isinstance(input_file,  (Path, str)), f"input_file must be a Path or str, got {type(input_file)}"
    assert isinstance(output_file, (Path, str)), f"output_file must be a Path or str, got {type(output_file)}"
    assert input_file.exists(), f"Input file {input_file} does not exist."

    match conv_fac:
        case 1:
            cdo(input_file).monsum().setunit(units).setname(variable).execute(output_file)
        case _:
            cdo(input_file).mulc(conv_fac).monsum().setunit(units).setname(variable).execute(output_file)



if __name__ == "__main__":
    pass
