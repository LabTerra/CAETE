import os
import joblib
import pandas as pd
import numpy as np

def read_pkz(spin, run_name, grd_name, grd_acro):
    with open(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/spin{spin:02d}.pkz", 'rb') as fh:
        dt = joblib.load(fh)
    # print(f"Loaded data for spin {spin:02d}")
    return dt

def pkz2csv(file, path, grd_name, run_name, spin_id, date_range, grd_acro) -> pd.DataFrame:
    assert date_range[2] == f"{spin_id:02d}", f"ID do spin {spin_id} não corresponde ao ID no date_range: {date_range[2]}"


    CT1 = pd.read_csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/src/code_table_allom.csv")

    MICV = ['year', 'pid', 'ocp']

    area = file['area']
    area_dim = area.shape
    
    idx1 = np.where(area[:, 0] > 0.0)[0]
    cols = CT1.VariableCode.__array__()


    fname = f"{run_name}_spin{spin_id:02d}"
    folder_path = f"./csv/{fname}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path2 = f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/csv"
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)

    for lev in idx1:
        area_TS = area[lev, :]
        # print(f"lev: {lev}, area_TS length: {len(area_TS)}")
        # Crie idxT1 com freq='Y'
        # Crie idxT1 manualmente representando anos inteiros
        start_date = date_range[0]
        print(date_range)
        end_date = date_range[1]
        print(end_date)
        idxT1 = pd.date_range(start=start_date, end=end_date, freq='D')    
        print('')
        print('')
        # print(f"idxt1: {idxT1}")
        print('')
        print('')
        # print(f'len area_TS {len(area_TS)} len idxT1 {len(idxT1)}')

    # Verifique se o comprimento de area_TS é consistente com o índice idxT1
        assert len(area_TS) == len(idxT1), "Length mismatch between area_TS and idxT1"
    
        area_TS = pd.Series(area_TS, index=idxT1)


        # Use the date range specified for the current spin
        idxT2 = pd.date_range(date_range[0], date_range[1], freq='Y')
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
        csv_filename = f"{run_name}_{grd_name}_spin{spin_id:02d}_EV_{int(lev)}.csv"
        dt1.to_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name}/{grd_name}/csv/{csv_filename}", index=False)

# # # User inputs
# run_name = input('Run name: ')
# grd = input('Which gridcell (lat-long): ')
# path = f"../outputs/{run_name}/gridcell{grd}"
# grd_name = f"gridcell{grd}"
# # 
# start_year = 1979
# end_year   = 2017
# # 
# run_breaks_hist = []
# # 
# for year in range(start_year, end_year, 1):
#     # Crie as datas de início e fim no formato 'YYYYMMDD'
#     start_date = f"{year}0101"
#     end_date = f"{year}1231"
# # 
#     # Obtenha o número do spin 
#     spin_id = str((year - start_year) // 1 + 1).zfill(2)
# # 
#     # Adicione a tupla à lista run_breaks_hist
#     run_breaks_hist.append((start_date, end_date, spin_id))
# # 
# # Exiba a lista resultante
# print(run_breaks_hist)
# # 
# # Process spins 1 to ..
# for date_range in run_breaks_hist:
#     start_date, end_date, spin_id = date_range
#     file = read_pkz(int(spin_id))
#     pkz2csv(file, path, grd_name, int(spin_id), date_range)
# # 
# # 