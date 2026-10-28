
import pktocsv_allspins as p 
import time_series as t
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_date = '19790101'
end_date = '19891231'

main_path = f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/'

while True:
    grd_acro = input('Gridcell acronym [AFL, ALP, FEC, MAN, CAX]: ')

    if grd_acro == 'ALP':
        grd = '188-213'
        break
    elif grd_acro == 'FEC':
        grd = '200-225'
        break
    elif grd_acro == 'MAN':
        grd = '186-239'
        break
    elif grd_acro == 'CAX':
        grd = '183-257'
        break
    elif grd_acro == 'AFL':
        grd = '199-248'
        break
    else:
        print('This acronym does not correspond')
        break

num_spins = input('how many spins? ')

# Lista de run_name gerados automaticamente
run_names = [f'{grd_acro}_save_spin_{i}' for i in range(1, int(num_spins) + 1)]

for run_name in run_names:

    path = f"../outputs/{run_name}/gridcell{grd}"
    grd_name = f"gridcell{grd}"


    print(f"Processing run_name: {run_name}")


    with open(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/spin35.pkz", 'rb') as fh:
        file = joblib.load(fh)


    CT1 = pd.read_csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/src/code_table_allom.csv")

    MICV = ['year', 'pid', 'ocp']

    area = file['area']
    area_dim = area.shape

    idx1 = np.where(area[:, 0] > 0.0)[0]
    cols = CT1.VariableCode.__array__()


    fname = f"{run_name}_spin35"

    folder_path = f"./csv/{fname}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path2 = f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/csv"
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)

    for lev in idx1:
        area_TS = area[lev, :]
        print(f"lev: {lev}, area_TS length: {len(area_TS)}")

        idxT1 = pd.date_range(start=start_date, end=end_date, freq='D')    
        print('')
        print('')
        print(f"idxt1: {idxT1}")
        print('')
        print('')
        print(f'len area_TS {len(area_TS)} len idxT1 {len(idxT1)}')

        assert len(area_TS) == len(idxT1), "Length mismatch between area_TS and idxT1"

        area_TS = pd.Series(area_TS, index=idxT1)

                # Use the date range specified for the current spin
        idxT2 = pd.date_range(start=start_date, end=end_date, freq='D')
        YEAR = []
        PID = []
        OCP = []

        for i in idxT2:
            YEAR.append(i.year)
            PID.append(int(lev))
            OCP.append(float(area_TS.loc[[i.date()]].iloc[0]))
            # print(f"lev: {lev}, PID: {int(lev)}")

        ocp_ts = pd.Series(OCP, index=idxT2)
        pid_ts = pd.Series(PID, index=idxT2)
        y_ts = pd.Series(YEAR, index=idxT2)

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

                # Save the CSV file with spin information in the name
        csv_filename = f"{run_name}_{grd_name}_spin35_EV_{int(lev)}.csv"
        dt1.to_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/csv/{csv_filename}", index=False)

    variables_to_plot = ['photo', 'ar', 'npp',  'lai', 'f5', 'evapm', 'cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto', 'wue', 'ls']

    all_dates = []
    all_spins = []
    all_data = {variable: [] for variable in variables_to_plot}

    with open(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/spin35.pkz", 'rb') as fh:
        dt = joblib.load(fh)

    # Iterate over all variables and extract time series
    for variable in variables_to_plot:
        variable_series = dt.get(variable, [])
        all_data[variable].extend(variable_series)

            # Create date index and update dates and spins lists
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dates.extend(date_index)
    all_spins.extend(['35'] * len(date_index))

        # Create a DataFrame with time series, dates, and spin numbers for all variables
    all_data['Date'] = all_dates
    all_data['Spin'] = all_spins
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    df.to_csv(f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/csv/concatenated_series_{run_name}.csv', index=False)

    # # Create a figure and an array of subplots based on the number of variables
    num_variables = len(variables_to_plot)
    num_rows = (num_variables + 1) // 4  # Ensure at least 1 row
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(15, 5 * num_rows), sharex=True)

    # # Flatten the axs array to handle 1D indexing
    axs = axs.flatten()

    # # Iterate over variables and plot each one
    for i, variable in enumerate(variables_to_plot):
        axs[i].plot(df['Date'], df[variable])
        axs[i].set_ylabel(variable)
        axs[i].set_title(f'Time Series of {variable}')
        # 

    # Adjust the layout to prevent title overlap
    plt.tight_layout()
        # 

        # Save the plot as an image
    plt.savefig(os.path.join(f'{main_path}{run_name}/{grd_name}/', f'timeseries_{run_name}_all_variables.png'))
        # 
        # Display the subplots
    plt.show()

    print(f"Processing completed for run_name: {run_name}")
    print("====================================")












