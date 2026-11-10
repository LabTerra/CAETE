#!/usr/bin/env python3
"""Export daily CAETÊ output and PLS area from historical spin*.pkz files.

This script writes ONLY two CSV files:
    diario.csv  One row per day. Includes all 1D daily variables and small
                multi-component daily arrays (e.g., nupt_1, nupt_2).
    area.csv    One row per day and one column per PLS (pls_0000, ...).

The original PKZ files are read only; no temporal averaging is performed.

Requirements:
    Python 3 with numpy and joblib installed.

Suggested project structure (your analysis folder can have ANY name):
    CAETE/
      outputs/<run_name>/<gridcell>/spin36.pkz ... spin73.pkz
      docs/<your_analysis_folder>/scripts/01_extract_daily_and_area_manual.py
      docs/<your_analysis_folder>/tabelas/  (created automatically)

HOW TO USE:
    1. Save this script in CAETE/docs/<your_analysis_folder>/scripts/.
    2. Change RUN_NAME and GRIDCELL below to match your simulation.
    3. If necessary, change FIRST_SPIN, START_YEAR and END_YEAR.
    4. From the CAETE directory (e.g. open a terminal in vscode), run:
       python docs/<your_analysis_folder>/scripts/01_extract_daily_and_area_manual.py

       If necessary, specify the full path to your Python interpreter.
       Example (Anaconda on Linux):
       /home/<username>/anaconda3/bin/python docs/<your_analysis_folder>/scripts/01_extract_daily_and_area_manual.py
       Replace <username> and <your_analysis_folder> with your own paths.


The script assumes the above folder structure. If your data are elsewhere,
set INPUT_DIR and OUTPUT_DIR to explicit paths instead.

Existing diario.csv and area.csv are overwritten when the script runs.
"""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from joblib import load

# ======================= USER SETTINGS =======================
# Place this script in CAETE/docs/<your_analysis_folder>/scripts/.
# BASE is then the CAETE project directory, regardless of the analysis
# folder's name (e.g. caete_ws, allom_nutri_gridcell186-239, eixo_carbono_agua_gridcell186-239, etc.).
BASE = Path(__file__).resolve().parents[3]

# Change these two values to match your model output folder:
RUN_NAME = "allomnutri_hydraulics_3k_PKLS"
GRIDCELL = "gridcell186-239"

# Input: existing CAETE simulation results (read only).
INPUT_DIR = BASE / "outputs" / RUN_NAME / GRIDCELL

# Output: a 'tabelas' folder alongside the 'scripts' folder.
# Example: CAETE/docs/student_analysis/tabelas/
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tabelass"

# Change these values if your run has a different spinup or time range.
FIRST_SPIN = 36
START_YEAR = 1979
END_YEAR = 2016

TIME_ORIGIN = date(1860, 1, 1)  # CAETÊ date reference for 'sind'
SPIN_FILENAME = re.compile(r"spin(\d+)\.pkz$")


def get_spin_files(folder: Path, first_spin: int):
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


def main():
    if START_YEAR > END_YEAR:
        raise ValueError("START_YEAR must not exceed END_YEAR")
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Input directory does not exist: {INPUT_DIR}")
    files = get_spin_files(INPUT_DIR, FIRST_SPIN)
    if not files:
        raise SystemExit(f"No spin files found from spin{FIRST_SPIN:02d}.pkz onward: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_dates = set()
    daily_header = None
    area_header = None
    processed = 0

    # The outputs are generated in the specified folder, overwriting existing files.
    with (OUTPUT_DIR / "diario.csv").open("w", newline="", encoding="utf-8") as daily_file, \
         (OUTPUT_DIR / "area.csv").open("w", newline="", encoding="utf-8") as area_file:
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
        (OUTPUT_DIR / "diario.csv").unlink(missing_ok=True)
        (OUTPUT_DIR / "area.csv").unlink(missing_ok=True)
        raise SystemExit("No data found within the requested years")

    print(f"\nDone: {processed} spin files. CSV files saved in: {OUTPUT_DIR.resolve()}")
    print("Created: diario.csv and area.csv")


if __name__ == "__main__":
    main()
