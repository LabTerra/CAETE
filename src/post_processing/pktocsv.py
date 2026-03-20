import os
import joblib
from pathlib import Path
from caete_module import global_par as gp
import caete_module as mod
import _pickle as pkl
import bz2
from numpy import asfortranarray
from pandas import read_csv

import cfunits
import numpy as np
import pandas as pd
import cftime as cf

import plsgen as pls
from aux_plot import get_var

# Get the number of PLSs from user input
npls = input('How many PLSs? ')

# Get PLS table from CSV file
pls_table = pd.read_csv("../outputs/pls_attrs-{}.csv".format(npls))

# Get the run name, gridcell, and spin from user input
run_name = input('Run name: ')
grd = input('Which gridcell (lat-long): ')
path = f"../outputs/{run_name}/gridcell{grd}"
grd_name = f"gridcell{grd}"

# Navigate to the folder
os.chdir(path)

spin = input('Which spin? ')

# Function to read the pickled file
def read_pkz(spin):
    with open("spin{}.pkz".format(spin), 'rb') as fh:
        dt = joblib.load(fh)
    print(dt.keys())  # List the available keys for the outputs
    return dt

file = read_pkz(spin)

# Function to convert pickled file to CSV
def pkz2csv(file, path, grd_name) -> pd.DataFrame:
    print(grd_name)

    # Read code table from CSV
    CT1 = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/code_table_allom.csv")

    MICV = ['year', 'pid', 'ocp']

    area = file['area']
    area_dim = area.shape

    # Find indices where the first column of area is greater than 0.0
    idx1 = np.where(area[:, 0] > 0.0)[0]

    # Get VariableCode column from code table
    cols = CT1.VariableCode.__array__()

    # Loop over living strategies in the simulation
    idxT1 = pd.date_range("2015-01-01", "2016-12-31", freq='D')
    print('Shape idxT1', idxT1.shape)

    # Create a folder for the output files
    fname = f"{run_name}_spin{spin}"
    folder_path = f"./{fname}"
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)

    for lev in idx1:
        area_TS = area[lev, :]
        area_TS = pd.Series(area_TS, index=idxT1)

        # Create an annual date range
        idxT2 = pd.date_range("2015-12-31", "2016-12-31", freq='Y')
        YEAR = []
        PID = []
        OCP = []

        for i in idxT2:
            # Append the year to the YEAR list
            YEAR.append(i.year)

            # Append the index of the living PLS
            PID.append(int(lev))

            # Append the occupancy value for the corresponding date
            OCP.append(float(area_TS.loc[[i.date()]].iloc[0]))

        # Create pandas Series for each variable
        ocp_ts = pd.Series(OCP, index=idxT2)
        pid_ts = pd.Series(PID, index=idxT2)
        y_ts = pd.Series(YEAR, index=idxT2)

        # Create a DataFrame with the specified columns
        series = []
        for i, var in enumerate(MICV):
            if var == 'year':
                series.append(y_ts)
            elif var == 'pid':
                series.append(pid_ts)
            elif var == 'ocp':
                series.append(ocp_ts)
            else:
                pass
        dt1 = pd.DataFrame(dict(list(zip(cols, series))))

        # Save the DataFrame to a CSV file
        dt1.to_csv(f"./{fname}/AmzFACE_Y_CAETE_spin{spin}_EV_{int(lev)}.csv", index=False)

# Call the function to convert the pickled file to CSV
pkz2csv(file, path, grd_name)
