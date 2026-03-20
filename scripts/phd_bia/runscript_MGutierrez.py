"""

Code documentation:

The clean_run function performs the following actions:

    1. Save Current Conditions:
Before starting a new round or experiment, the function saves the current conditions associated with a save_id identifier.
    2. Check Output Directory Existence:
Checks if the output directory associated with the new experiment already exists. If the directory already exists, the code takes measures to avoid overwriting.
    3. Revert Changes in Case of Failure:
If the attempt to create the directory fails due to its preexistence, the code reverts the changes made earlier (restoring self.out_dir to its original value) and raises an exception.
    4. Record Current Conditions:
Adds a pair (save_id, self.outputs.copy()) to the realized_runs list, thus recording the current state of the simulation.
    5. Clear Attributes for the New Round:
Resets relevant attributes for the simulation, such as self.outputs and self.run_counter, preparing for a new simulation round.

This function appears prepare for a new round of simulation, ensuring the integrity of saved data and avoiding accidental overwriting of previous results.


"""
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import joblib
import numpy as np
import csv
from parameters import base_run, ATTR_FILENAME, run_path, pls_path


### Marcela:
# Put here the logics where you ask to the user to select uniform or normal selection of traits
####

# Verifies if everything is ok with the paths
assert run_path.exists(), "Wrong path to initial conditions"
assert pls_path.exists(), "Wrong path to Attributes Table"

# Uses initial conditions file
# Open the binary file at 'run_path' for reading ('rb')
with open(run_path, 'rb') as fh:
    # Load data from the file using the joblib library
    init_conditions = joblib.load(fh)  #init_conditions contais all the attributes and methods of caete.grd

all_attributes_and_methods = dir(init_conditions)

# new outputs folder
run_name = input(f"Give a name to this output: ")
dump_folder = Path(f"{run_name}")


for gridcell in init_conditions:
    gridcell.clean_run(dump_folder, "init_cond")
    

def zip_gridtime(grd_pool, interval):
    res = []
    for i, j in enumerate(grd_pool):
        res.append((j, interval[i % len(interval)]))
    return res
   
# Loop para executar para cada ano de '19790101' a '20161231'
for year in range(1979, 2017):
    start_date = f"{year}0101"
    end_date = f"{year}1231"

    gridcell.run_caete_allom(start_date, end_date)
        
