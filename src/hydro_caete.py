# ! Copyright 2017- LabTerra

# !     This program is free software: you can redistribute it and/or modify
# !     it under the terms of the GNU General Public License as published by
# !     the Free Software Foundation, either version 3 of the License, or
# !     (at your option) any later version.)

# !     This program is distributed in the hope that it will be useful,
# !     but WITHOUT ANY WARRANTY; without even the implied warranty of
# !     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# !     GNU General Public License for more details.

# !     You should have received a copy of the GNU General Public License
# !     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ! AUTHORS: Gabriel Marandola & JP Darela

import warnings
import numpy as np
from math import log as ln
import caete_module as caete_mod




def rwarn(txt='RuntimeWarning'):
    warnings.warn(f"{txt}", RuntimeWarning)


def B_func(Th33, Th1500):
    """calculates the coefficient of moisture-tension"""

    D = ln(Th33) - ln(Th1500)
    B = (ln(1500) - ln(33)) / D

    def lbd_func(C):
        """return the slope of logarithmic tension-moisture curve"""
        if C == 0:
            return 0.0
        lbd = 1 / C
        return lbd

    return lbd_func(B)


def ksat_func(ThS, Th33, lbd):
    """soil conductivity in saturated condition. Output in mm/h"""

    # assert ThS > Th33, "sat <= fc IN ksat_func"
    ksat = 1930 * (ThS - Th33) ** (3 - lbd)
    return ksat


def kth_func(Th, ThS, lbd, ksat):
    """soil conductivity in unsaturated condition. Output in mm/h"""
    if Th < 0.0:
        # rwarn("water content < 0 IN kth_func")
        Th = 0.0
    kth = ksat * (Th / ThS) ** (3 + (2 / lbd))

    return kth


class soil_water:

    def __init__(self, ws1, ws2, fc1, fc2, wp1, wp2):
        """
        ws: float: water saturation (Fraction  of soil volume - all)
        fc: float: field capacity
        wp: float: wilting point
        """
        assert ws1 > fc1 and fc1 > wp1
        assert ws1 > 0.0 and ws1 <= 1.0
        assert fc1 > 0.0 and fc1 <= 1.0
        assert wp1 > 0.0 and wp1 <= 1.0

        assert ws2 > fc2 and fc2 > wp2
        assert ws2 > 0.0 and ws2 <= 1.0
        assert fc2 > 0.0 and fc2 <= 1.0
        assert wp2 > 0.0 and wp2 <= 1.0

        self.ws1 = np.float64(ws1)
        self.fc1 = np.float64(fc1)
        self.wp1 = np.float64(wp1)

        self.ws2 = np.float64(ws2)
        self.fc2 = np.float64(fc2)
        self.wp2 = np.float64(wp2)

        # Soil Water POols mm (Kg m-2)
        self.w1 = np.float64(20.0)
        self.w2 = np.float64(60.0)

        # Set pool sizes
        # Kilograms of water that would fit the  soil layers in the ausence of soil
        self.p1_vol = np.float64(300.0)  # Kg(H2O) - Layer 1
        self.p2_vol = np.float64(700.0)  # Kg(H2O) - Layer 2

        # MAX soil water depth kg m-2
        self.w1_max = self.p1_vol * self.ws1
        self.w2_max = self.p2_vol * self.ws2

        self.wmax = np.float64(self.w1_max + self.w2_max)

        self.lbd_1 = B_func(self.fc1, self.wp1)

        self.ksat_1 = ksat_func(self.ws1, self.fc1, self.lbd_1)

        self.lbd_2 = B_func(self.fc2, self.wp2)

        self.ksat_2 = ksat_func(self.ws2, self.fc2, self.lbd_2)

    def _update_pool(self, prain, evapo):
        """Calculates upper and lower soil water pools for the grid cell,
        as well as the grid runoff and the water fluxes between layers"""

        # evapo adaptation to use in both layers
        ev1 = evapo * 0.3  # Kg m-2
        ev2 = evapo - ev1  # Kg m-2

        # runoff initial values
        runoff1 = 0.0  # Kg m-2
        runoff2 = 0.0  # Kg m-2
        ret1 = 0.0  # Kg m-2

        # POOL 1 - 0-30 cm
        # rainfall addition to upper water content
        self.w1 += prain  # kg m-2

        if self.w1 > self.w1_max:
            # saturated condition (runoff and flux)
            runoff1 += (self.w1 - self.w1_max)
            self.w1 = self.w1_max
            flux1_mm = self.ksat_1 * 24.0  # Kg m-2 day-1
        else:
            # unsaturated condition (no runoff)
            w1_vol = self.w1 / self.p1_vol
            flux1_mm = kth_func(w1_vol, self.ws1, self.lbd_1,
                                self.ksat_1) * 24.0  # Kg m-2 day-1

        # Update pool 1

        self.w1 -= (ev1 + flux1_mm)

        # POOL 2 30-100 cm
        # Flux comming from w1
        self.w2 += flux1_mm  # Kg m-2 day-1
        if self.w2 < 0.0:
            self.w2 = 0.0

        if self.w2 > self.w2_max:
            # saturated condition
            ret1 = self.w2 - self.w2_max
            # The surplus remains in w1
            self.w1 += ret1
            # Check w1 runoff by return
            if self.w1 > self.w1_max:
                runoff1 += (self.w1 - self.w1_max)
                self.w1 = self.w1_max
            self.w2 = self.w2_max
            runoff2 += self.ksat_2 * 24.0
        else:
            w2_vol = self.w2 / self.p2_vol
            # unsaturated condition (runoff)
            runoff2 += kth_func(w2_vol, self.ws2,
                                self.lbd_2, self.ksat_2) * 24.0

        self.w2 -= (runoff2 + ev2)
        if runoff1 < 1e-17:
            runoff1 = 0.0
        if runoff2 < 1e-17:
            runoff2 = 0.0

        return runoff1 + runoff2

    def _update_pool_allom(self, prain, evapo):
        """Same as _update_pool, used by the allometric version (run_caete_allom),
        with the water extraction limited by the soil water (note [N2] at the end):

        - evapotranspiration only removes water above the wilting point of each
          layer (the roots cannot extract water held at tensions beyond it);
        - the demand not met by one layer is taken from the other one, if it
          still has available water (the 30%/70% split is kept otherwise);
        - drainage cannot leave a layer with negative water.

        Returns (runoff, actual evapotranspiration), both in kg m-2 day-1 (mm/day).
        The actual evapotranspiration is <= evapo (the atmospheric/canopy demand).
        """

        # water held below the wilting point (not available to the roots)
        wp1_mm = self.p1_vol * self.wp1  # Kg m-2
        wp2_mm = self.p2_vol * self.wp2  # Kg m-2

        # demand in each layer (same split used in _update_pool)
        ev1_dem = evapo * 0.3  # Kg m-2
        ev2_dem = evapo - ev1_dem  # Kg m-2

        runoff1 = 0.0  # Kg m-2
        runoff2 = 0.0  # Kg m-2

        # POOL 1 - 0-30 cm
        # rainfall addition to upper water content
        self.w1 += prain  # kg m-2

        if self.w1 > self.w1_max:
            # saturated condition (runoff and flux)
            runoff1 += (self.w1 - self.w1_max)
            self.w1 = self.w1_max
            flux1_mm = self.ksat_1 * 24.0  # Kg m-2 day-1
        else:
            # unsaturated condition (no runoff)
            w1_vol = self.w1 / self.p1_vol
            flux1_mm = kth_func(w1_vol, self.ws1, self.lbd_1,
                                self.ksat_1) * 24.0  # Kg m-2 day-1

        # Root water uptake limited to the water above the wilting point
        # (plant available water; Allen et al. 1998; Pavlick et al. 2013, Eq. B18)
        av1 = max(0.0, self.w1 - wp1_mm)
        av2 = max(0.0, self.w2 - wp2_mm)
        ev1 = min(ev1_dem, av1)
        ev2 = min(ev2_dem, av2)
        # unmet demand of one layer taken from the other (compensated root water
        # uptake; Jarvis 1989; Simunek & Hopmans 2009)
        extra2 = min(ev1_dem - ev1, av2 - ev2)
        extra1 = min(ev2_dem - ev2, av1 - ev1)
        ev1 += extra1
        ev2 += extra2

        # Update pool 1 (drainage limited to the water left in the layer)
        self.w1 -= ev1
        flux1_mm = min(flux1_mm, max(0.0, self.w1))
        self.w1 -= flux1_mm

        # POOL 2 30-100 cm
        # Flux comming from w1
        self.w2 += flux1_mm  # Kg m-2 day-1

        if self.w2 > self.w2_max:
            # saturated condition
            ret1 = self.w2 - self.w2_max
            # The surplus remains in w1
            self.w1 += ret1
            # Check w1 runoff by return
            if self.w1 > self.w1_max:
                runoff1 += (self.w1 - self.w1_max)
                self.w1 = self.w1_max
            self.w2 = self.w2_max
            runoff2 += self.ksat_2 * 24.0
        else:
            w2_vol = self.w2 / self.p2_vol
            # unsaturated condition (runoff)
            runoff2 += kth_func(w2_vol, self.ws2,
                                self.lbd_2, self.ksat_2) * 24.0

        # Update pool 2 (drainage limited to the water left in the layer)
        self.w2 -= ev2
        runoff2 = min(runoff2, max(0.0, self.w2))
        self.w2 -= runoff2

        if runoff1 < 1e-17:
            runoff1 = 0.0
        if runoff2 < 1e-17:
            runoff2 = 0.0

        return runoff1 + runoff2, ev1 + ev2

def net_radiation(rsds, temp, rh, lat, doy, albedo=0.13):
    """Daily mean net radiation at the surface (W m-2), FAO-56 method
    (Allen et al. 1998, Eqs. 21-39). Used by the allometric version as the
    available energy in the Penman equations (note [N3] at the end).

    rsds:   downward shortwave radiation (W m-2)
    temp:   air temperature (oC)
    rh:     relative humidity (0-1)
    lat:    latitude (degrees)
    doy:    day of year (1-366)
    albedo: surface albedo (0.13 = Amazonian forest; Culf et al. 1995)

    Rn = (1 - albedo) Rs - Rnl, with the net longwave loss Rnl estimated from
    temperature, vapour pressure and cloudiness (Rs / clear-sky radiation).
    All arguments can be numpy arrays.
    """
    phi = np.deg2rad(lat)
    # extraterrestrial radiation (MJ m-2 day-1)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    rso = 0.75 * ra  # clear-sky radiation
    rs = rsds * 0.0864  # W m-2 -> MJ m-2 day-1
    # relative shortwave radiation (cloudiness); 1.0 when there is no sun (polar night)
    rs_rso = np.where(rso > 0.0, np.minimum(rs / np.where(rso > 0.0, rso, 1.0), 1.0), 1.0)
    ea = rh * 0.6108 * np.exp(17.27 * temp / (temp + 237.3))  # actual vapour pressure (kPa)
    rnl = 4.903e-9 * (temp + 273.16) ** 4 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * rs_rso - 0.35)
    rn = (1.0 - albedo) * rs - rnl  # MJ m-2 day-1
    return rn / 0.0864  # W m-2


def CPTEC_PVM2_bdget(temp, p0, rh):
   return  caete_mod.water.evpot2(p0, temp, rh, caete_mod.water.available_energy(temp))



if __name__ == "__main__":

    ws1, ws2 = 0.45, 0.48
    fc1, fc2 = 0.39, 0.43
    wp1, wp2 = 0.24, 0.27

    wp = soil_water(ws1, ws2, fc1, fc2, wp1, wp2)

    for x in range(5000):
        evapo = 5 if np.random.normal() > 0 else 0
        roff = wp._update_pool(evapo, evapo)
        wp.w1 = np.float(0.0) if wp.w1 < 0.0 else wp.w1
        wp.w2 = np.float(0.0) if wp.w2 < 0.0 else wp.w2
        print(wp.w1, ':w1')
        print(wp.w2, ':w2')
        print(roff, ' :runoff')


# ==============================================================================
# NOTES
# ==============================================================================
#
# [N2] Water extraction in soil_water._update_pool_allom
#
#   Plant available water is the water between field capacity and the wilting
#   point; water below the wilting point is held at tensions the roots cannot
#   overcome (Allen et al. 1998, Ch. 8). JeDi-DGVM, the source of the root water
#   supply used in CAETE (water_stress_modifier, funcs.f90), limits transpiration
#   in the same way: Etrans = min(alpha_H2O * demand, W), with W the plant
#   available soil water (Pavlick et al. 2013, Eq. B18).
#
#   Here, per layer i (1 = 0-30 cm, 2 = 30-100 cm):
#       available_i = max(0, w_i - wp_i * volume_i)
#       ev_i        = min(demand_i, available_i),  demand_1 = 0.3 E, demand_2 = 0.7 E
#   The demand not met in one layer is taken from the other one while it still
#   has available water. This is the full-compensation limit of compensated root
#   water uptake (Jarvis 1989; Simunek & Hopmans 2009): the root system is
#   connected and takes more water from the wetter layer (review in Javaux et al.
#   2013).
#
#   Drainage is limited to the water left in the layer (mass conservation). In
#   _update_pool a layer can become negative and the reset to zero in caete.py
#   adds water to the soil.
#
#   Limitation: the reduction of the extraction is abrupt at the wilting point;
#   the gradual reduction as the soil dries (soil/xylem hydraulics) is not
#   represented here.
#
# [N3] Net radiation (net_radiation)
#
#   Penman's equation uses the net radiation Rn as the available energy. FAO-56
#   (Allen et al. 1998):
#       Rn  = (1 - albedo) Rs - Rnl                                        (Eq. 38-40)
#       Rnl = sigma Tk^4 (0.34 - 0.14 sqrt(ea)) (1.35 Rs/Rso - 0.35)      (Eq. 39)
#       Rso = 0.75 Ra (clear-sky radiation), Ra from latitude and day of year (Eq. 21-25, 37)
#   Rs/Rso (<= 1) works as a cloudiness index. The elevation term of Rso and the
#   soil heat flux (negligible at the daily scale; Eq. 42) are ignored. The forcing
#   has no Tmax/Tmin, so the daily mean temperature is used in Rnl instead of
#   (Tmax^4 + Tmin^4)/2.
#
#   Albedo: mean forest albedo of 0.134 in three Amazonian sites (Culf et al. 1995);
#   0.12 (Shuttleworth 1984) and 0.13 (Bastable et al. 1993) in other measurements.
#   A single value is used for all vegetation.
#
#   Uncertainty: at the Manaus forest, Shuttleworth (1984) measured a net longwave
#   loss of about 30 W m-2 and Rn = 0.858 S - 35 W m-2. For the mean shortwave of
#   gridcell 186-239 (190 W m-2) this gives Rn = 128 W m-2, against 142 W m-2 from
#   FAO-56 (Rnl = 22 W m-2): FAO-56 may overestimate Rn by ~10% in the Amazon.
#   The previous estimate, available_energy(temp) = 2.895 T + 52.3 (funcs.f90),
#   gives 132 W m-2 with no seasonal cycle (it does not depend on radiation).
#
# References
#   Allen, R.G., Pereira, L.S., Raes, D. & Smith, M. (1998). Crop evapotranspiration:
#     guidelines for computing crop water requirements. FAO Irrigation and
#     Drainage Paper 56. FAO, Rome.
#   Bastable, H.G. et al. (1993). Observations of climate, albedo, and surface
#     radiation over cleared and undisturbed Amazonian forest. International
#     Journal of Climatology, 13, 783-796.
#   Culf, A.D., Fisch, G. & Hodnett, M.G. (1995). The albedo of Amazonian forest
#     and ranch land. Journal of Climate, 8, 1544-1554.
#   Jarvis, N.J. (1989). A simple empirical model of root water uptake. Journal of
#     Hydrology, 107, 57-72.
#   Javaux, M., Couvreur, V., Vanderborght, J. & Vereecken, H. (2013). Root water
#     uptake: from three-dimensional biophysical processes to macroscopic modeling
#     approaches. Vadose Zone Journal, 12(4).
#   Pavlick, R. et al. (2013). The Jena Diversity-Dynamic Global Vegetation Model
#     (JeDi-DGVM). Biogeosciences, 10, 4137-4177.
#   Shuttleworth, W.J. (1984). Observations of radiation exchange above and below
#     Amazonian forest. Quarterly Journal of the Royal Meteorological Society, 110,
#     1163-1169.
#   Simunek, J. & Hopmans, J.W. (2009). Modeling compensated root water and nutrient
#     uptake. Ecological Modelling, 220, 505-521.
# ==============================================================================
