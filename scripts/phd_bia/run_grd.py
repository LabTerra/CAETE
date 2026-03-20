import os
import _pickle as pkl
import bz2
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
import caete
from caete import grd, mask, npls, print_progress, rbrk
import plsgen as pls
from caete_module import budget as model

outf = input(
    "Give a name to your run (ASCII letters and numbers only. No spaces): ")
dump_folder = Path(f'../outputs/{outf}').resolve()

# Water saturation, field capacity & wilting point (maps of 0.5° res)
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

zone = ""
y0, y1 = 0, 0
x0, x1 = 0, 0
folder = "central"

# Water saturation, field capacity & wilting point (maps of 0.5° res)
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

y0, y1 = 175, 176
x0, x1 = 235, 236
folder = "central"

s_data = Path("../input").resolve()
clim_and_soil_data = Path(folder)


input_path = Path(os.path.join(s_data, clim_and_soil_data))
clim_metadata = Path(os.path.join(s_data, clim_and_soil_data,
                                  "ISIMIP_HISTORICAL_METADATA.pbz2"))
with bz2.BZ2File(clim_metadata, mode='r') as fh:
    clim_metadata = pkl.load(fh)
stime = copy.deepcopy(clim_metadata[0])
del clim_metadata


# # open co2 data
with open(os.path.join(s_data, "co2/historical_CO2_annual_1765_2018.txt")) as fh:
    co2_data = fh.readlines()
run_breaks = rbrk[0]
rbrk_index = 0


with open("stime.txt", 'w') as fh:
    fh.writelines([f"{stime['units']}\n",
                   f"{stime['calendar']}\n",
                   f"historical-ISIMIP2b-TEST-{folder}\n",
                   f"{rbrk_index}\n"])
# FUNCTIONAL TRAITS DATA
pls_table = pls.table_gen(npls, dump_folder)

grid = caete.grd(236, 175, outf)

grid.init_caete_dyn(input_path, stime, co2_data,
                    pls_table, tsoil, ssoil, hsoil)

#grid.run_caete('19790101', '19891231', nutri_cycle = True)
gr1 = grid.run_caete_allom('19790101', '19891231', nutri_cycle = True)

    # print('cleaf:',grid.cleaf_allom,'cwood: ', grid.cwood_allom, 
    #   'cheart',grid.cheart_allom, 'csap', grid.csap_allom)
print(grid.cleaf_allom,grid.cwood_allom, grid.cheart_allom, grid.csap_allom)