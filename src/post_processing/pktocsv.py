import os
import joblib
import numpy as np
import pandas as pd

# Function to read the pickled file
def read_pkz(pkz_filepath):
    with open(pkz_filepath, 'rb') as fh:
        dt = joblib.load(fh)
    print(dt.keys())  # List the available keys for the outputs
    return dt

# Function to convert pickled file to CSV
def pkz2csv(file, spin, grd_folder_path, table_allom_csv_path) -> pd.DataFrame:
    # Read code table from CSV
    CT1 = pd.read_csv(table_allom_csv_path)
    # Get VariableCode column from code table
    cols = CT1.VariableCode.__array__()
    
    MICV = ['year', 'pid', 'ocp']

    area = file['area']
    # area_dim = area.shape

    # Find indices where the first column of area is greater than 0.0
    idx1 = np.where(area[:, 0] > 0.0)[0]

    # Loop over living strategies in the simulation
    idxT1 = pd.date_range("2015-01-01", "2016-12-31", freq='D')
    print('Shape idxT1', idxT1.shape)

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
        dt1.to_csv(f"{grd_folder_path}/AmzFACE_Y_CAETE_spin{spin}_EV_{int(lev)}.csv", index=False)


if __name__ == '__main__':
    outputs_folder = "/home/luana/Documents/outputs_JUSTRUNNED_CAETE_just_bia_and_precision"

    # ------
    # INPUTS
    # ------
    # Get the number of PLSs from user input
    # npls = input('How many PLSs? ')
    npls = 6000
    # Get the run name, gridcell, and spin from user input
    # run_name = input('Run name: ')
    run_name = "lu"
    # grd = input('Which gridcell (lat-long): ')
    grd = "186-239"
    spin = input('Which spin? ')

    table_allom_csv_path = "code_table_allom.csv"


    # Get PLS table from CSV file
    pls_table = pd.read_csv(f"{outputs_folder}/{run_name}/pls_attrs-{npls}.csv")

    grd_folder_path = f"{outputs_folder}/{run_name}/gridcell{grd}"


    # Create csv folder path in the same folder where spins pkz are located
    csv_folder_path = f"{grd_folder_path}/csv"
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # Read pkz
    pkz_filepath = f"{grd_folder_path}/spin{spin}.pkz"
    csv_filepath = pkz_filepath.replace(".pkz", ".csv")

    file = read_pkz(pkz_filepath)

    # Call the function to convert the pickled file to CSV
    pkz2csv(file, spin, grd_folder_path, table_allom_csv_path)

