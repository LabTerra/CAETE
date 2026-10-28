import _pickle as pkl
import bz2
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
import caete as mod
import os
from parameters import pls_path, ATTR_FILENAME

print("Set the folder to store outputs:")
outf = input(
    "Give a name to your run (ASCII letters and numbers only. No spaces): ")
dump_folder = Path(f'../outputs/grd_experiment/{outf}').resolve()

#Use the PLS table that already exists
if pls_path.exists():
    from caete_utils import read_pls_table
    print("Using PLS TABLE from BASE_RUN")
    os.makedirs(dump_folder, exist_ok=True)
    pls_table = read_pls_table(out=Path(os.path.join(dump_folder, ATTR_FILENAME)))

#creates a new PLS table
else:
    print(f"WARNING: Creating a new PLS table for a historical simulated ({outf}) run ")
    pls_table = pls.table_gen(npls, dump_folder)

# Soil Parameters
# Water saturation, field capacity & wilting point
# Topsoil
map_ws = np.load("../input/soil/ws.npy")
map_fc = np.load('../input/soil/fc.npy')
map_wp = np.load('../input/soil/wp.npy')

# Subsoil
map_subws = np.load("../input/soil/sws.npy")
map_subfc = np.load("../input/soil/sfc.npy")
map_subwp = np.load("../input/soil/swp.npy")

tsoil = (map_ws, map_fc, map_wp)
ssoil = (map_subws, map_subfc, map_subwp)

# Hydraulics
theta_sat = np.load("../input/hydra/theta_sat.npy")
psi_sat = np.load("../input/hydra/psi_sat.npy")
soil_texture = np.load("../input/hydra/soil_text.npy")

hsoil = (theta_sat, psi_sat, soil_texture)

# Read time metadata
with bz2.BZ2File("../input/task5/hist_obs/ISIMIP_HISTORICAL_METADATA.pbz2", mode='rb') as fh:
    mdt = pkl.load(fh)
stime = copy.deepcopy(mdt[0])

# Read CO2 data
co2 = "../input/co2/historical_CO2_annual_1765_2018.txt"
with open(co2) as fh:
    co2_data = fh.readlines()

# DEFINE HARVERSTERS - funcs that will apply grd methods(run the CAETÊ model) over the instanvces
def apply_spin(grid):
    """pre-spinup use some outputs of daily budget (water, litter C, N and P) to start soil organic pools"""
    w, ll, cwd, rl, lnc = grid.bdg_spinup(
        start_date = "19790101", end_date = "19830101")
    grid.sdc_spinup(w, ll, cwd, rl, lnc)
    return grid

def run_experiment(pls_table):
        # Open a dataset with the Standard time variable
    tm = nc.Dataset("./time_ISIMIP_hist_obs.nc4", 'r')
    tm1 = tm.variables["time"]
# 
    t1 = datetime.datetime(year=2006,month=1,day=1,hour=0,minute=0,second=0)
    t2 = datetime.datetime(year=2006,month=12,day=31,hour=0,minute=0,second=0)
# 
    # Find the index of the input data array for required dates
    # Will use this to manipulate the input data in sensitivity experiments  
    idx0 = int(nc.date2index(t1, tm1, calendar="proleptic_gregorian", select='nearest'))
    idx1 = int(nc.date2index(t2, tm1, calendar="proleptic_gregorian", select='nearest'))

#     print(idx0, idx1)
#     # # Create the plot object
    sdata = Path("../grd").resolve()
    grd = mod.grd(235, 175, 'GRD-ISIMIP') #sequence to identify the grid: X-Y (lon-lat)

#     # Fill the plot object with input data
    grd.init_caete_dyn(sdata, stime_i=stime, co2=co2_data,
                       pls_table=pls_table, tsoil=tsoil,
                       ssoil=ssoil, hsoil=hsoil)

    # # Apply a numerical spinup in the soil pools of resources
    grd = apply_spin(grd)

    grd.run_caete_allom('19790101','19891231', spinup=5, 
                   fix_co2='1999', save=False, nutri_cycle=False)

    grd.run_caete_allom('19790101', '19991231', spinup=35,
                   fix_co2='1999', save=False, nutri_cycle=False)

#         # Run the experiment!
    grd.run_caete_allom('20000101', '20151231', spinup=1, save=True)
    tm.close()
    return grd

def get_spin(grd: mod.grd, spin) -> dict:
    import joblib
    if spin < 10:
        name = f'spin0{spin}.pkz'
    else:
        name = f'spin{spin}.pkz'
    with open(grd.outputs[name], 'rb') as fh:
        spin_dt = joblib.load(fh)
    return spin_dt

if __name__ == "__main__":
    pass
    if pls_path.exists():
        from caete_utils import read_pls_table
        print("Using PLS TABLE from BASE_RUN")
        os.makedirs(dump_folder, exist_ok=True)
        pls_table = read_pls_table(out=Path(os.path.join(dump_folder, ATTR_FILENAME)))
        cax = run_experiment(pls_table)
    else:
        print("you need a PLS table")
