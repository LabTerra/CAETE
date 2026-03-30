import pktocsv_allspins2 as p2
import pktocsv_allspins as p
import time_series as t
import os

#Choose the gridcell acronym to access the data
#this step is to facilitate accessing the folder where the results were saved
while True:
    # grd_acro = input('Gridcell acronym [AFL, ALP, FEC, MAN, CAX, NVX]: ')
    grd_acro = "MAN"

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
    # server = input('Are you running in the server? y/n ')
    server = "n"

    if server == 'y':
        # Set the main_path accordingly for server
        outputs_path = f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/'
        break
    elif server == 'n':
        # Set the main_path accordingly for local machine
        # outputs_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/'
        outputs_path = '/home/luana/Documents/outputs_JUSTRUNNED_CAETE_just_bia_and_precision'
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
#    'test_marcela']
   'lu']

start_year = 1979
end_year   = 1989
# start_date = '19790101'
# end_date = '19891231'

# run_name = input('Run name: ')
for run_name in run_names:
    print('GRDDDDD == ', grd)
    grd_path = f"{outputs_path}/{run_name}/gridcell{grd}"
    grd_name = f"gridcell{grd}"

    run_breaks_hist1 = []
    run_breaks_hist2 = []

    for year in range(start_year, end_year, 1):
        # Obtenha o número do spin 
        spin_id = str((year - start_year) + 1).zfill(2)

        # Adicione a tupla à lista run_breaks_hist
        # Crie as datas de início e fim no formato 'YYYYMMDD'
        # run_breaks_hist1.append((f"{year}0101", f"{year}1231", spin_id))
        run_breaks_hist1.append((f"{start_year}0101", f"{end_year}1231", spin_id))

    # # Convert spins to csvs
    # for date_range in run_breaks_hist1:
    #     start_date, end_date, spin_id = date_range
    #     print(f"Spin: {spin_id}...")
    #     file = p2.read_pkz(spin_id, grd_path)
    #     # print(file)
    #     p.pkz2csv(file, grd_path, grd_name, run_name, int(spin_id), date_range, grd_acro)

    # Navigate to the specified folder to access spins
    os.chdir(grd_path)

    for date_range in run_breaks_hist1:
        print('Joining together all time series, dates, and spins =====', date_range)

    print('Plotting')
    start_date = f"{start_year}0101"
    end_date = f"{start_year}1231"
    t.join_plot(start_date, end_date, run_breaks_hist1, grd_path, run_name)
