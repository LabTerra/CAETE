#!/usr/bin/python3

import os
import sys
import xarray as xr

filename = sys.argv[1]
vname166 = sys.argv[2]

os.system(f"cp {filename} infile.nc")
os.system(f"ncks -A -v mask mask.nc infile.nc")


ds = xr.open_dataset("infile.nc")

comm = f"ds.{vname166}.where(ds.mask == 1)"


mask_com = compile(comm, "<string>", "eval")
masked = eval(mask_com)

masked_ds = f"{vname166}_masked0.nc"

masked.to_netcdf(masked_ds)

out_file = f"{vname166}_masked.nc"
os.system(f"set_t {masked_ds} {out_file}")

os.system(f"rm -rf infile.nc {masked_ds}")
