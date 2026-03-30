import os
import joblib
import pandas as pd
import numpy as np

def read_pkz(spin_id_str, grd_path):
    with open(f"{grd_path}/spin{spin_id_str}.pkz", 'rb') as fh:
        dt = joblib.load(fh)
    return dt

def pkz2csv(grd_path, start_date, end_date, spin_id_str, table_allom_csv_path) -> pd.DataFrame:
    # print(start_date)
    # print(end_date)
    # print(spin_id_str)

    CT1 = pd.read_csv(table_allom_csv_path)
    cols = CT1.VariableCode.__array__()

    # Read pkz
    file = read_pkz(spin_id_str, grd_path)

    # Get file area
    area = file['area']
    # print(area)
    # print(area.shape)
    # print(len(area))

    idx1 = np.where(area[:, 0] > 0.0)[0]
    # print(idx1)
    # print(idx1.shape)
    # print(len(idx1))

    for lev in idx1:
        area_TS = area[lev, :]
        print(len(area_TS))
        # print(f"lev: {lev}, area_TS length: {len(area_TS)}")
        # Crie idxT1 com freq='Y'
        # Crie idxT1 manualmente representando anos inteiros
        idxT1 = pd.date_range(start=start_date, end=end_date, freq='D')
        # print(len(idxT1))
        # print('')
        # print('')
        # print(f"idxt1: {idxT1}")
        # print('')
        # print('')
        # print(f'len area_TS {len(area_TS)} len idxT1 {len(idxT1)}')

        # Verifique se o comprimento de area_TS é consistente com o índice idxT1
        assert len(area_TS) == len(idxT1), "Length mismatch between area_TS and idxT1"

        # Cast to pandas series
        area_TS = pd.Series(area_TS, index=idxT1)

        # Use the date range specified for the current spin
        idxT2 = pd.date_range(start_date, end_date, freq='Y')
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
        MICV = ['year', 'pid', 'ocp']
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
        csv_filename = f"spin{spin_id_str}_EV_{int(lev)}.csv"
        dt1.to_csv(f"{grd_path}/csv/{csv_filename}", index=False)


# # User inputs
# # run_name = input('Run name: ')
# run_name = "lu"
# # grd = input('Which gridcell (lat-long): ')
# grd = "186-239"

# start_year = 1979
# # end_year   = 2017
# end_year   = 1989

# table_allom_csv_path = "code_table_allom.csv"

# outputs_folder = "/home/luana/Documents/outputs_JUSTRUNNED_CAETE_just_bia_and_precision/"

# grd_path = f"{outputs_folder}/{run_name}/gridcell{grd}"

# # Create csv folder path in the same folder where spins pkz are located
# csv_folder_path = f"{grd_path}/csv"
# if not os.path.exists(csv_folder_path):
#     os.makedirs(csv_folder_path)

# run_breaks_hist = []
# for year in range(start_year, end_year, 1):
#     # Crie as datas de início e fim no formato 'YYYYMMDD'
#     start_date = f"{year}0101"
#     end_date = f"{year}1231"

#     # Obtenha o número do spin
#     spin_id = str(year - start_year + 1).zfill(2)

#     # Adicione a tupla à lista run_breaks_hist
#     run_breaks_hist.append((start_date, end_date, spin_id))

# # # Exiba a lista resultante
# # print(run_breaks_hist)

# # Process spins 1 to ..
# for date_range in run_breaks_hist:
#     start_date, end_date, spin_id = date_range
#     pkz2csv(grd_path, start_date, end_date, spin_id, table_allom_csv_path)
