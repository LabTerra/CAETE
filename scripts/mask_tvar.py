import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr


ROOT = Path(__file__).parent
mask_file = ROOT / "../input/mask/mask.nc"


filename = sys.argv[1]
vname166 = sys.argv[2]

os.system(f"cp {filename} infile.nc")
# os.system(f"ncks -A -v mask mask.nc infile.nc")


def make_mask(dm1, mask):
    mtype = mask.dtype
    Y = mask.shape[0]
    X = mask.shape[1]
    out = np.zeros(shape=(dm1, Y, X), dtype=mtype)

    for i in range(dm1):
        out[i, :, :] = mask

    return out


ds = xr.open_dataset("infile.nc")
mask_ra = xr.open_dataset(mask_file)
print(mask_ra.mask.shape)

comm = f"ds.{vname166}.shape[0]"
mask_shape = compile(comm, "<string>", "eval")
dm1 = eval(mask_shape)

mask1 = make_mask(dm1, mask_ra.mask)

comm = f"ds.{vname166}.where(mask1 == 1.0)"

mask_com = compile(comm, "<string>", "eval")
masked = eval(mask_com)

masked_ds = f"{vname166}_masked0.nc"
masked.to_netcdf(masked_ds, format="NETCDF4", encoding={"zlib": True, "complevel": 9})

out_file = f"{vname166}_masked.nc"
os.system(f"cdo settbounds,month {masked_ds} {out_file}")

os.system(f"rm -rf infile.nc {masked_ds}")
