from netCDF4 import Dataset
import numpy as np


mask = np.load("./mask_BIOMA.npy")

# Create the lat/lon dimensions
res = 0.5
lon = np.arange(-179.75, 180, res, dtype=np.float64)
lat = np.arange(89.75, -90, -res, dtype=np.float64)[::-1]
half = res / 2.0
latbnd = np.array([[l - half, l + half] for l in lat])
lonbnd = np.array([[l - half, l + half] for l in lon])

# Create the number array, initialize to a missing value
miss = 0.0
ids = np.ma.masked_all((360, 720), dtype=np.float64)
ids.fill(miss)
ids.fill_value = miss
ids[np.where(mask == False)] = 1.0
ids[np.where(mask == True)] = miss

print(ids.shape)
# Create netCDF dimensions
dset = Dataset("mask_bioma.nc", mode="w")
dset.createDimension("lat", size=lat.size)
dset.createDimension("lon", size=lon.size)

# Create netCDF variables
Y = dset.createVariable("lat", lat.dtype, ("lat"))
X = dset.createVariable("lon", lon.dtype, ("lon"))
I = dset.createVariable("mask", ids.dtype, ("lat", "lon"))

# Load data and encode attributes
X[...] = lon
X.units = "degrees_north"

Y[...] = lat
Y.units = "degrees_east"

I[:, :] = np.flipud(ids)

dset.close()
