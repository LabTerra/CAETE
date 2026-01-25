import os

from pathlib import Path
from config import Config, fetch_config

cfg:Config = fetch_config()

# Variables coming out of CAETE have non-standard names. Map them to more common names.
# Add more variable name mappings as needed.
caete_names = {
    "gpp": "photo",
    "et": "evapm"
    }

get_caete_varname = lambda var: caete_names.get(var, var) # Return mapped name or original if not found

# ----------------------------------------------------------------


# Construct a dictionary of available CAETE output variables (gets all netCDF files in the CAETE output directory)
# Important change here: The dataframes.py module now saves output files with the format:
# TODO: ensure this is consistent everywhere specially in the dataframes module
# <variable>-<experiment>-<timestamp>.nc
#
# OLD VERSION (flat structure: {variable: path}):
# available_variables = dict([(p.name.split("_")[0].split("-")[0], p) for p in Path(cfg.output.output_dir).glob("*.nc")])
#
# NEW VERSION (nested structure: {experiment: {variable: path}}):
# Parse filenames with format: <variable>-<experiment>-<timestamp>.nc
def _build_available_variables():
    """Build nested dictionary of available CAETE outputs by experiment and variable."""
    result = {}
    for p in Path(cfg.output.output_dir).glob("*.nc"):
        parts = p.name.split("-")
        if len(parts) >= 2:
            variable = parts[0]  # e.g., "photo", "evapm", "lai"
            experiment = parts[1]  # e.g., "pan_amazon_hist_da"
            if experiment not in result:
                result[experiment] = {}
            result[experiment][variable] = p
    return result

available_variables = _build_available_variables()
# ----------------------------------------------------------------

# Some CAETE output variables may need unit conversions / naming changes for benchmarking.
# Most variables are output in daily resolution and different units and generally need to be converted to monthly totals.
# We preprocess CAETE outputs and store the intermediate datasets in the
BENCHMARCK_CACHE_DIR = Path("./benchmark_cache/")
ensure_cache_dir = lambda : os.makedirs(BENCHMARCK_CACHE_DIR, exist_ok=True)
clean_cache = lambda : [f.unlink() for f in BENCHMARCK_CACHE_DIR.glob("*") if f.is_file()]

ensure_cache_dir()
# ----------------------------------------------------------------


# Construct a dictionary of reference datasets. BENCHMARK_DIR should contain directories
# per variable type, e.g., gpp/GBAF, gpp/FLUXCOM, etc.
# Expected directory structure:
# BENCHMARK_DIR/
# ├── gpp/
# │   ├── GBAF/
# │   │   └── data.nc
# │   └── FLUXCOM/
# │       └── data.nc
# ├── et/
# │   └── MODIS/
# │       └── data.nc
# └── lai/
#     └── AVHRR/
#         └── data.nc
# BENCHMARK_DIR = Path("../../../../OneDrive/Desktop/GUESS_data/LPJG-home/benchmark/") # EDIT HERE TO POINT TO BENCHMARK DATA DIRECTORY
BENCHMARK_DIR = Path("../../../../OneDrive/Desktop/CAETE/benchmark_data/custom_data")

ref_datasets = dict()

try:
    for dirpath in Path(BENCHMARK_DIR).rglob("*"):
        if dirpath.is_file():
            dataset_name = dirpath.parent.name
            if dataset_name not in ref_datasets:
                ref_datasets[dataset_name] = {}

            if dirpath.name in ref_datasets[dataset_name]:
                print(f"Warning: Duplicate file name '{dirpath.name}' found in {dirpath.parent}")

            ref_datasets[dataset_name][dirpath.name] = dirpath
except Exception as e:
    print(f"Error while constructing reference datasets tree: {e}")

# # Write out a benchmark_datasets.md file listing available datasets and files in the BENCHMARK_DIR
# try:
#     with open("benchmark_datasets.md", "w") as f:
#         f.write("# Benchmark Datasets\n\n")
#         f.write("| Variable | Dataset | Filename |\n")
#         f.write("|----------|---------|----------|\n")
#         for dataset, files in ref_datasets.items():
#             for filename, filepath in files.items():
#                 # Variable is the grandparent directory (e.g., gpp/, et/)
#                 variable = filepath.parent.parent.name
#                 f.write(f"| {variable} | {dataset} | {filename} |\n")
# except Exception as e:
#     print(f"Error while writing benchmark_datasets.md: {e}")
# # ----------------------------------------------------------------

# if __name__ == "__main__":
#     pass
#     # Unit tests for benchmark utilities
