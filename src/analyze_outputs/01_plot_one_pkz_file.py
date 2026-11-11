
from __future__ import annotations

import csv
import os
import re
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
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

# Function to read the pickled file
def read_pkz(filename):
    """Function to load a pkz file."""
    with open(filename, 'rb') as fh:
        dt = load(fh)
    # print(dt.keys())  # List the available keys for the outputs
    return dt

def generate_plot_for_one_gridcell(gridcell_folder_name):
    """Function to generate plot for one gridcell folder."""

    # Set gridcell folder to process pkz files
    GRIDCELL_DIR = OUTPUT_DIR_TO_PROCESS / gridcell_folder_name
    print(GRIDCELL_DIR)

    # Change dir to gridcell folder
    os.chdir(GRIDCELL_DIR)

    # Filter files in the current path that start with "spin" to get all spins
    spins = [item.replace("spin", "").replace(".pkz", "") for item in os.listdir() if item.startswith("spin")]
    spins.sort()
    print(f"All spins = {spins}")

    spin = input("Which spin? ")
    if spin not in spins:
        print(f"Error: spin '{spin}' not found!")
        exit(1)

    # Read spin file
    spin_filename = "spin{}.pkz".format(spin)
    print(f"Generating plot for file '{spin_filename}'...")
    dt = read_pkz(spin_filename)

    # Plot the specified variables in a 4x4 grid
    ncols = 6
    nrows = 9
    figsize = (40, 30)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Specify the variables to be plotted
    variables_to_plot = dt.keys()
    # variables_to_plot = ['emaxm', 'tsoil', 'photo', 'ar', 'npp', 'lai', 'rcm', 'f5', 'runom', 'evapm', 'wsoil', 'rm', 'rg', 'cleaf', 'csap', 'cheart', 'csto', 'wue', 'area', 'ls']

    for i, key in enumerate(variables_to_plot):
        # print(f"Plotting key {key}...")
        if key in ["lim_status", "u_strat"]:
            continue

        values = dt[key]
        row = i // ncols
        col = i % ncols
        axs[row, col].plot(values)
        axs[row, col].set_title(key)
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel(key)
        axs[row, col].grid(True)

    # Adjust the layout to prevent title overlap
    plt.tight_layout()

    # Display the subplots
    plt.show()



if __name__ == "__main__":

    # List all folder starting with "gridcell..."
    gridcell_folder_names = [
        p for p in os.listdir(OUTPUT_DIR_TO_PROCESS)
        if os.path.isdir(os.path.join(OUTPUT_DIR_TO_PROCESS, p)) and p.startswith("gridcell")
    ]

    print("\n")
    print("="*50)
    print(f"Ploting one pkz file...")
    print("="*50)
    print("\n")

    # Creates a daily.csv and area.csv for each gridcell folder
    for gridcell_foldername in gridcell_folder_names:
        print(f"Processing pkz files for gridcell {gridcell_foldername}...\n")
        generate_plot_for_one_gridcell(gridcell_foldername)
        print("-" * 50 + "\n")
