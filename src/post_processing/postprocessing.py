import pktocsv_allspins as p 
import src.post_processing.time_series as t
import os
import joblib
import pandas as pd
import numpy as np


while True:
    grd_acro = input('Gridcell acronym [AFL, ALP, FEC, MAN, CAX, NVX]: ')

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
    elif grd_acro == 'NVX':
        grd = '210-249'
        break
    elif grd_acro == 'AFL':
        grd = '199-248'
        break
    else:
        print('This acronym does not correspond')
        break


while True:
    server = input('Are you running in the server? y/n ')

    if server == 'y':
        # Set the main_path accordingly for server
        main_path = f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/'
        break
    elif server == 'n':
        # Set the main_path accordingly for local machine
        main_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/'
        break
# run_names = ['ALP_30prec_1y']
# run_names = [
    
#     'AFL_30prec_1y',
#     'AFL_30prec_3y',
#     'AFL_30prec_5y',
#     'AFL_30prec_7y'
# ]
    
# run_names = [
#     # 
#     'ALP_30prec_1y',
#     'ALP_30prec_3y',
#     'ALP_30prec_5y',
#     'ALP_30prec_7y'
# ]
    
# run_names = [
#     'FEC_30prec_1y',
#     'FEC_30prec_3y',
#     'FEC_30prec_5y',
#     'FEC_30prec_7y']
    
# run_names = [
#     'CAX_30prec_1y',
#     'CAX_30prec_3y',
#     'CAX_30prec_5y',
#     'CAX_30prec_7y']

run_names = [
   'MAN_30prec_7y']
    


# run_name = input('Run name: ')
for run_name in run_names:
    print('GRDDDDD == ', grd)
    path = f"../outputs/{run_name}/gridcell{grd}"
    grd_name = f"gridcell{grd}"


    start_year = 1979
    end_year   = 2017

    run_breaks_hist1 = []
    run_breaks_hist2 = []


    for year in range(start_year, end_year, 1):
        #Crie as datas de início e fim no formato 'YYYYMMDD'
        start_date = f"{year}0101"
        end_date = f"{year}1231"
    
        # Obtenha o número do spin 
        spin_id = str((year - start_year) // 1 + 1).zfill(2)

        # Adicione a tupla à lista run_breaks_hist
        run_breaks_hist1.append((start_date, end_date, spin_id))


    for year in range(start_year, end_year, 1):
        #Crie as datas de início e fim no formato 'YYYYMMDD'
        start_date = f"{year}0101"
        end_date = f"{year}1231"
    

        # Adicione a tupla à lista run_breaks_hist
        run_breaks_hist2.append((start_date, end_date))


    # # Process spins 1 to ..
    for date_range in run_breaks_hist1:

        start_date, end_date, spin_id = date_range
        file = p.read_pkz(int(spin_id), run_name, grd_name, grd_acro)
        p.pkz2csv(file, path, grd_name, run_name, int(spin_id), date_range, grd_acro)

    # # Navigate to the specified folder to access spins
    os.chdir(f'{main_path}{run_name}/{grd_name}/')

    for date_range in run_breaks_hist2:
        print('Joining together all time series, dates, and spins =====',date_range)

    print('Plotting')
    t.join_plot(start_date, end_date, run_breaks_hist2, main_path, run_name, grd_name)



