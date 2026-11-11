#!/usr/bin/env python3
"""Export daily CAETÊ output and PLS area from historical spin*.pkz files for each gridcell folder.

This script writes ONLY two CSV files:
    gridcell...daily.csv  One row per day. Includes all 1D daily variables and small
                multi-component daily arrays (e.g., nupt_1, nupt_2).
    gridcell...area.csv    One row per day and one column per PLS (pls_0000, ...).

The original PKZ files are read only; no temporal averaging is performed.

Requirements:
    Python 3 with numpy and joblib installed.

Suggested project structure (your analysis folder can have ANY name):
    CAETE/
      outputs/<run_name>/<gridcell>/spin36.pkz ... spin73.pkz
      outputs/<run_name>/tabelas/  (created automatically)

HOW TO USE:
    1. Run script and it will ask RUN_NAME .
    2. If necessary, change FIRST_SPIN, START_YEAR and END_YEAR.
    3. From the CAETE directory (e.g. open a terminal in vscode), run:
       python src/analyze_outputs/01_extract_daily_and_area_manual.py

Existing files 'gridcell...daily.csv' and 'gridcell...area.csv' are overwritten when the script runs.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from joblib import load

# BASE is the CAETE project directory, regardless of the analysis
BASE = Path(__file__).resolve().parents[2]

# Set the run name (For example, in the case 'CAETE/outputs/run_for_caete_ws',
# ...your run name is run_for_caete_ws.
RUN_NAME = input(
    "Run name (For example, if you have the folder "
    "'CAETE/outputs/run_for_caete_ws', "
    "add here just run_for_caete_ws: "
)

# Set output folder to be processed
OUTPUT_DIR_TO_PROCESS = BASE / "outputs" / RUN_NAME

# Output: Folder where all CSV files will be generated.
OUTPUT_DIR_GRIDS_PKZS_TO_CSVS = OUTPUT_DIR_TO_PROCESS / "grids_pkzs_to_csvs"

# Change these values if your run has a different spinup or time range.
FIRST_SPIN = 36
START_YEAR = 1979
END_YEAR = 2016

TIME_ORIGIN = date(1860, 1, 1)  # CAETÊ date reference for 'sind'
SPIN_FILENAME = re.compile(r"spin(\d+)\.pkz$")


def get_spin_files(folder: Path, first_spin: int):
    """Function to get all 'spin....pkz' filepaths."""
    files = []
    for path in folder.iterdir():
        match = SPIN_FILENAME.fullmatch(path.name)
        if match and int(match.group(1)) >= first_spin:
            files.append((int(match.group(1)), path))
    return [path for _, path in sorted(files)]


def get_daily_columns(data: dict, n_days: int, day_indices: list[int]):
    """Return daily cell-level variables; skip the PLS-scale arrays."""
    result = {}
    for name, value in data.items():
        if name == "area":
            continue
        array = np.asarray(value)
        if array.ndim == 1 and array.shape[0] == n_days:
            result[name] = array[day_indices]
        elif array.ndim == 2 and array.shape[-1] == n_days and array.shape[0] <= 20:
            for component in range(array.shape[0]):
                result[f"{name}_{component + 1}"] = array[component, day_indices]
    return result


def process_csvs_for_one_gridcell(gridcell_folder_name):
    """Function to process one gridcell folder."""

    # Set gridcell folder to process pkz files
    GRIDCELL_DIR = OUTPUT_DIR_TO_PROCESS / gridcell_folder_name

    if START_YEAR > END_YEAR:
        raise ValueError("START_YEAR must not exceed END_YEAR")
    if not GRIDCELL_DIR.is_dir():
        raise SystemExit(f"Input directory does not exist: {GRIDCELL_DIR}")
    files = get_spin_files(GRIDCELL_DIR, FIRST_SPIN)
    if not files:
        raise SystemExit(f"No spin files found from spin{FIRST_SPIN:02d}.pkz onward: {GRIDCELL_DIR}")

    OUTPUT_DIR_GRIDS_PKZS_TO_CSVS.mkdir(parents=True, exist_ok=True)
    seen_dates = set()
    daily_header = None
    area_header = None
    processed = 0

    # Set processed "daily.csv" and "area.csv" filepaths
    daily_csv_filepath = OUTPUT_DIR_GRIDS_PKZS_TO_CSVS / f"{gridcell_folder_name}_daily.csv"
    area_csv_filepath = OUTPUT_DIR_GRIDS_PKZS_TO_CSVS / f"{gridcell_folder_name}_area.csv"

    # The outputs are generated in the specified folder, overwriting existing files.
    with daily_csv_filepath.open("w", newline="", encoding="utf-8") as daily_file, \
         area_csv_filepath.open("w", newline="", encoding="utf-8") as area_file:
        daily_writer = csv.writer(daily_file)
        area_writer = csv.writer(area_file)

        for path in files:
            data = load(path)
            if "sind" not in data or "area" not in data:
                raise ValueError(f"Missing 'sind' or 'area' in {path.name}")

            area = np.asarray(data["area"])
            if area.ndim != 2:
                raise ValueError(f"Expected area to be a 2D array (PLS, days): {path.name}, {area.shape}")
            n_pls, n_days = area.shape
            start_index = int(data["sind"])
            dates = [TIME_ORIGIN + timedelta(days=start_index + i) for i in range(n_days)]
            selected = [i for i, current_date in enumerate(dates)
                        if START_YEAR <= current_date.year <= END_YEAR]
            if not selected:
                continue

            selected_dates = [dates[i] for i in selected]
            overlap = seen_dates.intersection(selected_dates)
            if overlap:
                raise ValueError(f"Duplicate date {min(overlap)} in {path.name}; check FIRST_SPIN")
            seen_dates.update(selected_dates)

            columns = get_daily_columns(data, n_days, selected)
            current_daily_header = ["date", "year", "spin"] + list(columns)
            current_area_header = ["date"] + [f"pls_{i:04d}" for i in range(n_pls)]

            if daily_header is None:
                daily_header = current_daily_header
                area_header = current_area_header
                daily_writer.writerow(daily_header)
                area_writer.writerow(area_header)
            elif current_daily_header != daily_header or current_area_header != area_header:
                raise ValueError(f"Variable list or number of PLSs changed in {path.name}")

            for row, index in enumerate(selected):
                current_date = selected_dates[row]
                daily_writer.writerow([current_date.isoformat(), current_date.year, path.name]
                                      + [values[row] for values in columns.values()])
                area_writer.writerow([current_date.isoformat()] + area[:, index].tolist())

            processed += 1
            print(f"Read: {path.name} | {selected_dates[0]} to {selected_dates[-1]}", flush=True)

    if processed == 0:
        # Avoid keeping empty files when the selected date range contains no data.
        daily_csv_filepath.unlink(missing_ok=True)
        area_csv_filepath.unlink(missing_ok=True)
        raise SystemExit("No data found within the requested years")

    print(f"\nDone: {processed} spin files. CSV files saved in: {OUTPUT_DIR_GRIDS_PKZS_TO_CSVS.resolve()}")
    print(f"Files '{daily_csv_filepath}' and '{area_csv_filepath}' were created!")


if __name__ == "__main__":

    # List all folder starting with "gridcell..."
    gridcell_folder_names = [
        p for p in os.listdir(OUTPUT_DIR_TO_PROCESS)
        if os.path.isdir(os.path.join(OUTPUT_DIR_TO_PROCESS, p)) and p.startswith("gridcell")
    ]

    print("\n")
    print("="*50)
    print(f"Generating csvs for all gridfolders {gridcell_folder_names}...")
    print("="*50)
    print(f"Output csv files will be saved into '{OUTPUT_DIR_GRIDS_PKZS_TO_CSVS}' folder...")
    print("="*50)
    print("\n")

    # Creates a daily.csv and area.csv for each gridcell folder
    for gridcell_foldername in gridcell_folder_names:
        print(f"Processing pkz files for gridcell {gridcell_foldername}...\n")
        process_csvs_for_one_gridcell(gridcell_foldername)
        print("-" * 50 + "\n")
