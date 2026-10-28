import numpy as np

def find_coord(N, W):
    """ Given a pair of geographic (WGS84) coordinates (decimal degrees)
        returns the Y and X indices in the array (360,720//0.5° lon-lat)
        (C_contiguous) Tested only in south america"""
    Yc = round(N, 2)
    Xc = round(W, 2)

    if abs(Yc) > 89.75:
        if Yc < 0:
            Yc = -89.75
        else:
            Yc = 89.75

    if abs(Xc) > 179.75:
        if Xc < 0:
            Xc = -179.75
        else:
            Xc = 179.75

    Yind = 0
    Xind = 0

    lon = np.arange(-179.75, 180, 0.5)
    lat = np.arange(89.75, -90, -0.5)

    if True:
        while Yc < lat[Yind]:
            Yind += 1
    # else:
    #     Yind += lat.size // 2
    #     while Yc > lat[Yind]:
    #         Yind += 1
    if Xc <= 0:
        while Xc > lon[Xind]:
            Xind += 1
    else:
        Xind += lon.size // 2
        while Xc > lon[Xind]:
            Xind += 1
    
    
    return Yind, Xind

"ALP - Allpahuayo"
coord = find_coord(-3.9333, -73.4167)
print('ALP',coord)

"FEC - Fazenda Experimental de Catuaba"
coord = find_coord(-10.0667, -67.6167)
print('FEC',coord)

"MAN - Manaus"
coord = find_coord(-3.0167, -60.35)
print('MAN',coord)

"CAX - Caxiuanã"
coord = find_coord(-1.7167, -51.45)
print('CAX',coord)

"AFL - Alta Floresta"
coord = find_coord(-9.52, -55.93)
print('AFL',coord)
import numpy as np

def reverse_coord(Yind, Xind):
    """ Given Y and X indices in the array (360,720//0.5° lon-lat)
        returns the corresponding geographic (WGS84) coordinates (decimal degrees)"""
    lon = np.arange(-179.75, 180, 0.5)
    lat = np.arange(89.75, -90, -0.5)

    Yc = lat[Yind]
    Xc = lon[Xind]

    return Yc, Xc

# Exemplos de uso
Yind, Xind = 200, 225
coordinates = reverse_coord(Yind, Xind)
print(f'Coordinates for Yind={Yind}, Xind={Xind}: {coordinates}')
