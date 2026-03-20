# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.)

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# AUTHOR: Bianca Fazio Rius


#This script is designed to process and concatenate Net Primary Productivity (NPP) 
#time series data from different spins in a run. It collects data from multiple spins, 
#organizes it into a pandas DataFrame, and saves the concatenated time series 
#into a CSV file. Additionally, the script generates a plot of the NPP time series 
#against dates and saves it as a PNG image. 

import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

#Joining together all time series, dates, and spins
def join_plot(start_date, end_date, run_breaks_hist, main_path, run_name, grd_name): # Initialize lists to store all time series, dates, and spins
    
    variables_to_plot = ['photo', 'ar', 'npp',  'lai', 'f5', 'evapm', 'cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto', 'wue', 'ls']

    all_dates = []
    all_spins = []
    all_data = {variable: [] for variable in variables_to_plot}

    # Iterate over all available spins
    for spin, (start_date, end_date) in enumerate(run_breaks_hist, start=1):
        # Load data for the current spin
        with open(f"spin{spin:02d}.pkz", 'rb') as fh:
            dt = joblib.load(fh)

        #calculating total carbon
        # dt['totalcarbon'] = dt['cleaf'] + dt['cwood'] + dt['croot'] + dt['csap'] + dt['cheart'] + dt['csto']

        # Iterate over all variables and extract time series
        for variable in variables_to_plot:
            variable_series = dt.get(variable, [])
            all_data[variable].extend(variable_series)

        # Create date index and update dates and spins lists
        date_index = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates.extend(date_index)
        all_spins.extend([spin] * len(date_index))

    # Create a DataFrame with time series, dates, and spin numbers for all variables
    all_data['Date'] = all_dates
    all_data['Spin'] = all_spins
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    df.to_csv(f'concatenated_series_{run_name}.csv', index=False)

    # Convert the 'Date' column to the datetime data type if it's not already
    df['Date'] = pd.to_datetime(df['Date'])

# # Create a figure and an array of subplots based on the number of variables
    num_variables = len(variables_to_plot) 
    num_rows = (num_variables + 1) // 4  # Ensure at least 1 row
    fig, axs = plt.subplots(nrows = 5, ncols = 3, figsize=(15, 5 * num_rows), sharex=True)

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




# # Prompt the user to determine if the script is running on a server or not
# while True:
#     server = input('Are you running in the server? y/n ')

#     if server == 'y':
#         # Set the main_path accordingly for server
#         main_path = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/'
#         break
#     elif server == 'n':
#         # Set the main_path accordingly for local machine
#         main_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/'
#         break

# # Input the run name
# run_name = input('run name: ')
# grid_xy = input('which gridcell? (lat-long)')

# # Navigate to the specified folder
# os.chdir(f'{main_path}outputs/{run_name}/gridcell{grid_xy}/')

# # Initialize lists to store all time series, dates, and spins
# all_series = []
# all_series_ls = []
# all_dates = []
# all_spins = []


# start_year = 1979
# end_year   = 2017

# run_breaks_hist = []

# for year in range(start_year, end_year, 1):
#     #Crie as datas de início e fim no formato 'YYYYMMDD'
#     start_date = f"{year}0101"
#     end_date = f"{year}1231"
 

#     # Adicione a tupla à lista run_breaks_hist
#     run_breaks_hist.append((start_date, end_date))

# print(run_breaks_hist)

# # variables_to_plot = ['emaxm', 'tsoil', 'photo', 'ar', 'npp', 'lai', 'rcm', 'f5', 'runom', 'evapm', 'wsoil', 'rm', 'rg', 'cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto', 'wue', 'area', 'ls']

# variables_to_plot = ['photo', 'ar', 'npp',  'lai', 'f5', 'evapm', 'cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto', 'wue', 'ls']
# all_data = {variable: [] for variable in variables_to_plot}
# all_dates = []
# all_spins = []

# # Initialize lists to store all time series, dates, and spins for all variables
# all_data = {variable: [] for variable in variables_to_plot}
# all_dates = []
# all_spins = []

# # Iterate over all available spins
# for spin, (start_date, end_date) in enumerate(run_breaks_hist, start=1):
#     # Load data for the current spin
#     with open(f"spin{spin:02d}.pkz", 'rb') as fh:
#         dt = joblib.load(fh)

#     # Iterate over all variables and extract time series
#     for variable in variables_to_plot:
#         variable_series = dt.get(variable, [])
#         all_data[variable].extend(variable_series)

#     # Create date index and update dates and spins lists
#     date_index = pd.date_range(start=start_date, end=end_date, freq='D')
#     all_dates.extend(date_index)
#     all_spins.extend([spin] * len(date_index))

# # Create a DataFrame with time series, dates, and spin numbers for all variables
# all_data['Date'] = all_dates
# all_data['Spin'] = all_spins
# df = pd.DataFrame(all_data)

# # Save the DataFrame to a CSV file
# df.to_csv('concatenated_series_all_spins.csv', index=False)

# # Convert the 'Date' column to the datetime data type if it's not already
# df['Date'] = pd.to_datetime(df['Date'])


# # Create a figure and an array of subplots based on the number of variables
# num_variables = len(variables_to_plot)
# num_rows = (num_variables + 1) // 4  # Ensure at least 1 row
# fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(15, 5 * num_rows), sharex=True)

# # Flatten the axs array to handle 1D indexing
# axs = axs.flatten()

# # Iterate over variables and plot each one
# for i, variable in enumerate(variables_to_plot):
#     axs[i].plot(df['Date'], df[variable])
#     axs[i].set_ylabel(variable)
#     axs[i].set_title(f'Time Series of {variable}')

# # Adjust the layout to prevent title overlap
# plt.tight_layout()

# # Save the plot as an image
# plt.savefig(os.path.join(f'{main_path}/outputs/{run_name}/gridcell{grid_xy}/', f'timeseries_{run_name}_all_variables.png'))

# # Display the subplots
# plt.show()

