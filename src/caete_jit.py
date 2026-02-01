# -*-coding:utf-8-*-
# "CAETÊ"
# Author:  João Paulo Darela Filho

# _ = """ CAETE-DVM-CNP - Carbon and Ecosystem Trait-based Evaluation Model"""

# """
# Copyright 2017- LabTerra

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
# """


"""
Jit compiled functions
These functions are JIT compiled and cached by numba.
If you change any of the cached functions, you should delete the cache
folder in the src folder, generally named __pycache__. This will force numba
to recompile the functions and cache them again."""

from typing import List, Union, Tuple
import numpy as np
import numba
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def process_tuple(t: Tuple[NDArray, NDArray]) -> Tuple[int, float]:
    """Process a single tuple to find the strategy ID and the percentage of days of the most used strategy."""
    strategies, days = t
    total_days = days.sum()
    max_index = days.argmax()
    max_days = days[max_index]
    strategy_id = strategies[max_index]
    percentage = (max_days / total_days) * 100
    return strategy_id, percentage

def process_tuples(data: Tuple[Tuple[NDArray, NDArray], ...]) -> Tuple[Tuple[int, float], ...]:
    """Process a tuple of tuples to find the strategy ID and the percentage of days of the most used strategy for each pair."""
    results = []
    for t in data:
        results.append(process_tuple(t))
    return tuple(results)

@numba.jit(numba.float32[:](numba.float32[:], numba.float32[:], numba.float32[:]), nopython=True, cache=True)
def pft_area_frac(cleaf1:NDArray[np.float32],
                  cfroot1:NDArray[np.float32],
                  cawood1:NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate the area fraction of each PFT based on the leaf, root and wood biomass."""
    # Initialize variables
    npft = cleaf1.size
    ocp_coeffs = np.zeros(npft, dtype=np.float32)
    total_biomass_pft = np.zeros(npft, dtype=np.float32)
    # Compute total biomass for each PFT
    total_biomass_pft = cleaf1 + cfroot1 + cawood1
    # Compute total biomass for all PFTs
    total_biomass = np.sum(total_biomass_pft)
    # Calculate occupation coefficients
    if total_biomass > 0.0:
        ocp_coeffs = total_biomass_pft / total_biomass
        ocp_coeffs[ocp_coeffs < 0.0] = 0.0
    return ocp_coeffs

@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:]), nopython=True, cache=True)
def pft_area_frac64(cleaf1:NDArray[np.float64],
                  cfroot1:NDArray[np.float64],
                  cawood1:NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the area fraction of each PFT based on the leaf, root and wood biomass."""
    # Initialize variables
    npft = cleaf1.size
    ocp_coeffs = np.zeros(npft, dtype=np.float64)
    total_biomass_pft = np.zeros(npft, dtype=np.float64)
    # Compute total biomass for each PFT
    total_biomass_pft = cleaf1 + cfroot1 + cawood1
    # Compute total biomass for all PFTs
    total_biomass = np.sum(total_biomass_pft)
    # Calculate occupation coefficients
    if total_biomass > 0.0:
        ocp_coeffs = total_biomass_pft / total_biomass
        ocp_coeffs[ocp_coeffs < 0.0] = 0.0
    return ocp_coeffs


@numba.jit(nopython=True, cache=True)
def neighbours_index(pos: Union[List, NDArray], matrix: NDArray) -> List:
    """Get the indices of the neighbours of a given position in a 2D matrix.
    Args:
        pos (Union[List, NDArray]): The position of the cell in the matrix.
        matrix (NDArray): The matrix to get the neighbours from.
    Returns: List of tuples containing the indices of the neighbours."""
    neighbours = []
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    for i in range(max(0, pos[0] - 1), min(rows, pos[0] + 2)):
        for j in range(max(0, pos[1] - 1), min(cols, pos[1] + 2)):
            if not (i == pos[0] and j == pos[1]):
                neighbours.append((i, j))
    return neighbours

@numba.njit(cache=True)
def inflate_array(nsize: int, partial:NDArray[Union[np.float32, np.float64]], id_living:NDArray[np.intp]):
    """_summary_

    Args:
        nsize (int): _description_
        partial (NDArray[np.float32]): _description_
        id_living (NDArray[np.intp]): _description_

    Returns:
        _type_: _description_
    """
    c = 0
    complete = np.zeros(nsize, dtype=np.float32)
    for n in id_living:
        complete[n] = partial[c]
        c += 1
    return complete

@numba.jit(nopython=True, cache=True)
def linear_func(temp: float,
                vpd: float,
                T_max: float = 60.0,
                VPD_max : float = 8.0) -> float:
    """Linear function to calculate the coupling between the atmosphere and the surface"""
    linear_func = (temp / T_max + vpd / VPD_max) / 2.0

    # Ensure the output is between 0 and 1
    linear_func = 0.0 if linear_func < 0.0 else linear_func
    linear_func = 1.0 if linear_func > 1.0 else linear_func

    return linear_func

@numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32), nopython=True, cache=True)
def atm_canopy_coupling(emaxm: float, evapm: float, air_temp: float, vpd: float) -> float:
    """Calculate the coupling between the atmosphere and the canopy based on a simple linear function
    of the air temperature and the vapor pressure deficit.
    Args:
        emaxm: float -> maximum evaporation rate mm/day
        evapm: float -> evaporation rate mm/day
        air_temp: float -> air temperature in Celsius
        vpd: float -> vapor pressure deficit in kPa
    Returns:
        float: Evapotranspiration rate mm/day
        """

    omega = linear_func(air_temp, vpd)
    return emaxm * omega + evapm * (1 - omega)

@numba.jit(numba.float32(numba.int8[:], numba.float32[:]), nopython=True, cache=True)
def masked_mean(mask: NDArray[np.int8], values: NDArray[np.float32]) -> float:
    """Calculate the mean of the values array ignoring the masked values"""
    mean = 0.0
    count = np.logical_not(mask).sum()
    if count == 0:
        return np.nan

    for i in range(mask.size):
        if mask[i] == 0:
            mean += values[i] / count
    return mean

@numba.jit(numba.float32[:](numba.int8[:], numba.float32[:,:]), nopython=True, cache=True)
def masked_mean_2D(mask: NDArray[np.int8], values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate the mean of the values array ignoring the masked values"""
    integrate_dim = values.shape[0]
    dim_sum = np.zeros(integrate_dim, dtype=np.float32)
    count = np.zeros(integrate_dim, dtype=np.int32)
    for i in range(mask.size):
        if mask[i] == 0:
            for j in range(integrate_dim):
                dim_sum[j] += values[j, i]
                count[j] += 1
    return dim_sum / count

@numba.jit(numba.float32(numba.float64[:], numba.float32[:]), nopython=True, cache=True)
def cw_mean(ocp: NDArray[np.float64], values: NDArray[np.float32]) -> np.float32:
    """
    Calculate the Community weighted mean for values using an
    array of area occupation (0 (empty) 1 (Total dominance))"""

    return np.sum(ocp * values, dtype = np.float32)

@numba.jit(numba.float32(numba.float64[:], numba.float32[:], numba.float32), nopython=True, cache=True)
def cw_variance(ocp: NDArray[np.float64], values: NDArray[np.float32], mean: float) -> float:
    """Calculate the Community weighted variance for values using an
    array of area occupation (0 (empty) -1 (Total dominance))"""

    variance = 0.0
    for i in range(ocp.size):
        variance += ocp[i] * ((values[i] - mean) ** 2)
    return variance

# Some functions to calculate diversity and evenness indices coded by copilot
# TODO: Check the implementation of these functions
@numba.jit(nopython=True, cache=True)
def shannon_entropy(ocp: NDArray[np.float64]) -> float:
    """Calculate the Shannon entropy for a community"""
    if np.sum(ocp) == 0:
        return -9999.0
    entropy = 0.0
    for i in range(ocp.size):
        if ocp[i] > 0:
            entropy -= ocp[i] * np.log(ocp[i])
    return entropy

@numba.jit(nopython=True, cache=True)
def shannon_evenness(ocp: NDArray[np.float64]) -> float:
    """Calculate the Shannon evenness for a community"""
    max_entropy = np.log(ocp.size)
    if max_entropy == 0:
        return -9999.0
    return shannon_entropy(ocp) / max_entropy

@numba.jit(nopython=True, cache=True)
def shannon_diversity(ocp: NDArray[np.float64]) -> float:
    """Calculate the Shannon diversity for a community"""
    if np.sum(ocp) == 0:
        return -9999.0
    return np.exp(shannon_entropy(ocp))

if __name__ == "__main__":
    # Test functions
    import unittest
    from pathlib import Path
    import sys

    # In the first pass all functions are compiled and cached if necessary.
    # If this runs, then the functions are correctly compiled and cached.
    print("Testing CAETE JIT compiled functions...")

    # Remove __pycache__ folder
    # In any case, to force recompilation of the functions, delete the __pycache__ folder.
    pycache_path = Path(__file__).parent / "__pycache__"
    if pycache_path.exists() and pycache_path.is_dir():
        import shutil
        try:
            shutil.rmtree(pycache_path)
        except OSError:
            print("Could not remove __pycache__ - continuing (files might be locked)", file=sys.stdout)
    print("0 - OK ... ", file=sys.stdout)

    class TestCaeteJit(unittest.TestCase):

        def test_process_tuple(self):
            strategies = np.array([1, 2, 3], dtype=np.int64)
            days = np.array([10, 50, 40], dtype=np.float64)
            t = (strategies, days)
            sid, pct = process_tuple(t)
            self.assertEqual(sid, 2)
            self.assertEqual(pct, 50.0)

        def test_process_tuples(self):
            t1 = (np.array([1, 2], dtype=np.int64), np.array([20, 80], dtype=np.float64))
            t2 = (np.array([3, 4], dtype=np.int64), np.array([10, 90], dtype=np.float64))
            data = (t1, t2)
            results = process_tuples(data)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], (2, 80.0))
            self.assertEqual(results[1], (4, 90.0))

        def test_pft_area_frac(self):
            cleaf = np.array([10, 20], dtype=np.float32)
            cfroot = np.array([10, 20], dtype=np.float32)
            cawood = np.array([10, 20], dtype=np.float32)
            # Total PFT1=30, PFT2=60. Total=90.
            # Fractions: 1/3, 2/3
            res = pft_area_frac(cleaf, cfroot, cawood)
            expected = np.array([1/3, 2/3], dtype=np.float32)
            np.testing.assert_allclose(res, expected, rtol=1e-5)

        def test_pft_area_frac64(self):
            cleaf = np.array([10, 20], dtype=np.float64)
            cfroot = np.array([10, 20], dtype=np.float64)
            cawood = np.array([10, 20], dtype=np.float64)
            res = pft_area_frac64(cleaf, cfroot, cawood)
            expected = np.array([1/3, 2/3], dtype=np.float64)
            np.testing.assert_allclose(res, expected, rtol=1e-5)

        def test_neighbours_index(self):
            matrix = np.zeros((3, 3))
            pos = np.array([1, 1])
            nbs = neighbours_index(pos, matrix)
            self.assertEqual(len(nbs), 8)
            self.assertIn((0, 0), nbs)
            self.assertNotIn((1, 1), nbs)

            pos_corner = np.array([0, 0])
            nbs_corner = neighbours_index(pos_corner, matrix)
            self.assertEqual(len(nbs_corner), 3)

        def test_inflate_array(self):
            nsize = 5
            partial = np.array([10.0, 20.0], dtype=np.float32)
            id_living = np.array([0, 4], dtype=np.intp)
            res = inflate_array(nsize, partial, id_living)
            expected = np.array([10.0, 0.0, 0.0, 0.0, 20.0], dtype=np.float32)
            np.testing.assert_array_equal(res, expected)

        def test_linear_func(self):
            # temp=30, T_max=60 -> 0.5. vpd=4, VPD_max=8 -> 0.5. AVG -> 0.5
            res = linear_func(30.0, 4.0, 60.0, 8.0)
            self.assertAlmostEqual(res, 0.5)

            res_low = linear_func(-10.0, 0.0)
            self.assertEqual(res_low, 0.0)

            res_high = linear_func(100.0, 100.0)
            self.assertEqual(res_high, 1.0)

        def test_atm_canopy_coupling(self):
            # linear_func(30, 4) -> 0.5
            # emaxm*0.5 + evapm*0.5
            val = atm_canopy_coupling(10.0, 2.0, 30.0, 4.0)
            self.assertAlmostEqual(val, 6.0)

        def test_masked_mean(self):
            mask = np.array([0, 1, 0, 0], dtype=np.int8)  # 0 is valid
            values = np.array([10.0, 100.0, 20.0, 30.0], dtype=np.float32)
            res = masked_mean(mask, values)
            # valid: 10, 20, 30 -> mean 20
            self.assertEqual(res, 20.0)

            mask_all = np.array([1, 1], dtype=np.int8)
            res_nan = masked_mean(mask_all, values[:2])
            self.assertTrue(np.isnan(res_nan))

        def test_masked_mean_2D(self):
            mask = np.array([0, 1, 0], dtype=np.int8) # Indices 0 and 2 are valid
            values = np.array([
                [10, 20, 30],
                [1,  2,  3]
            ], dtype=np.float32)
            # Row 0 valid: 10, 30 -> mean 20
            # Row 1 valid: 1, 3   -> mean 2
            res = masked_mean_2D(mask, values)
            expected = np.array([20.0, 2.0], dtype=np.float32)
            np.testing.assert_array_equal(res, expected)

        def test_cw_mean(self):
            ocp = np.array([0.2, 0.8], dtype=np.float64)
            values = np.array([10.0, 20.0], dtype=np.float32)
            res = cw_mean(ocp, values)
            # 0.2*10 + 0.8*20 = 2 + 16 = 18
            self.assertAlmostEqual(res, 18.0)

        def test_cw_variance(self):
            ocp = np.array([0.5, 0.5], dtype=np.float64)
            values = np.array([10.0, 20.0], dtype=np.float32)
            mean = 15.0
            res = cw_variance(ocp, values, mean)
            # 0.5*(25) + 0.5*(25) = 25
            self.assertAlmostEqual(res, 25.0)

        def test_shannon_entropy(self):
            ocp = np.array([0.5, 0.5], dtype=np.float64)
            res = shannon_entropy(ocp)
            self.assertAlmostEqual(res, np.log(2))

            ocp_zero = np.zeros(2, dtype=np.float64)
            res_zero = shannon_entropy(ocp_zero)
            self.assertEqual(res_zero, -9999.0)

        def test_shannon_evenness(self):
            ocp = np.array([0.5, 0.5], dtype=np.float64)
            res = shannon_evenness(ocp)
            self.assertAlmostEqual(res, 1.0)

            ocp_single = np.array([1.0], dtype=np.float64)
            # max_entropy = log(1) = 0 -> -9999.0
            res_single = shannon_evenness(ocp_single)
            self.assertEqual(res_single, -9999.0)

        def test_shannon_diversity(self):
            ocp = np.array([0.5, 0.5], dtype=np.float64)
            res = shannon_diversity(ocp)
            self.assertAlmostEqual(res, 2.0)

            ocp_zero = np.zeros(2, dtype=np.float64)
            res_zero = shannon_diversity(ocp_zero)
            self.assertEqual(res_zero, -9999.0)

    unittest.main()
