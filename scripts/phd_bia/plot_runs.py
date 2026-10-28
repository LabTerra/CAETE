# Import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

grd_acro = input('Gridcell acronym [AFL, ALP, FEC, MAN, CAX]: ')

# Prompt the user to determine if the script is running on a server or locally
while True:
    server = 'y'#input('Are you running in the server? y/n ')

    # Set the main_path accordingly based on the user's response
    if server == 'y':
        main_path = f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{grd_acro}/state_start/'
        break
    elif server == 'n':
        # Set the main_path accordingly for local machine
        main_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/outputs/'
        break

# Input the main run name
main_run_name = input('Main run name: ')

# Input the number of runs
run_number = int(input('Number of runs: '))

# Input the latitude and longitude from grd
grd_name = input('lat and long from grd: ')

# Create a dictionary to store DataFrames
dfs = {}

# Loop through the specified number of runs
for i in range(1, run_number + 1):
    run_name = f'{main_run_name}_{i}'

    # Navigate to the specified folder to access spins
    os.chdir(f'{main_path}{run_name}/gridcell{grd_name}')

    # Read the CSV file corresponding to the concatenated series
    df = pd.read_csv(f'{main_path}{run_name}/gridcell{grd_name}/csv/concatenated_series_{run_name}.csv')

    # Convert the 'Date' column to the datetime data type
    df['Date'] = pd.to_datetime(df['Date'])

    # Store the DataFrame in the dictionary
    dfs[run_name] = df
variables_to_plot = ['photo', 'npp', 'evapm', 'cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto','ls']
# Create a figure and an array of subplots based on the number of variables
num_variables = len(variables_to_plot)
num_cols = 3
num_rows = (num_variables + num_cols - 1) // num_cols  # Ensure at least 1 row
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharex=True)

# Flatten the axs array to handle 1D indexing
axs = axs.flatten()

# Iterar sobre as variáveis e plotar todas as rodadas para cada variável
for i, variable in enumerate(variables_to_plot):
    # Iterate over rodadas and plot each one
    for run_name, df in dfs.items():
        axs[i].plot(df['Date'], df[variable], alpha=0.5, linewidth=0.6, label=run_name)

    axs[i].set_ylabel(variable)
    axs[i].set_title(f'Time Series of {variable}')
    axs[i].legend()

# Adjust the layout to prevent title overlap
plt.tight_layout()

# Salvar os gráficos como um único arquivo PNG
fig.savefig(f'{main_path}/timeseries_all_runs.png')

# Exibir os gráficos
plt.show()