#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Available Energy Model Fitting for CAETE
=========================================

This module fits linear and seasonal models for available energy (AE) as a function
of temperature using NCEP-NCAR Reanalysis 1 long-term mean daily data.

Available Energy is calculated as:
    AE = Rn - G = SW_absorbed - LW_lost - G_into_ground
    AE = (-nswrs) - nlwrs - gflux   (using NCEP sign conventions)

NCEP-NCAR Sign Conventions:
    - nswrs: NEGATIVE when absorbed by surface (positive upward)
    - nlwrs: POSITIVE when leaving surface (net upward longwave loss)
    - gflux: POSITIVE when heat flows into ground

Univariate Models (Temperature only):
    1. Global Linear Model: AE = a * T + b
    2. Seasonal Model (Monthly): AE = a_m * T + b_m (12 sets of coefficients)
    3. Seasonal Model (Daily): AE = a_doy * T + b_doy (365 sets of coefficients)

Multivariate Models (if RH/P data available):
    4. AE = f(T, RH) - Temperature + Relative Humidity
    5. AE = f(T, VPD) - Temperature + Vapor Pressure Deficit
    6. AE = f(VPD) - VPD only
    7. AE = f(T, RH, P) - Full model with pressure
    8. AE = f(T, VPD, P) - Full model with VPD and pressure

Required Data Files:
    - air.2m.gauss.day.ltm.1991-2020.nc (2m air temperature, K)
    - nswrs.sfc.gauss.day.ltm.1991-2020.nc (Net shortwave radiation, W/m²)
    - nlwrs.sfc.gauss.day.ltm.1991-2020.nc (Net longwave radiation, W/m²)
    - gflux.sfc.gauss.day.ltm.1991-2020.nc (Ground heat flux, W/m²)

Optional Data Files (for multivariate models):
    - rhum.sig995.day.ltm.1991-2020.nc (Relative humidity, %)
    - pres.sfc.day.ltm.1991-2020.nc (Surface pressure, Pa)

Notes:
    - NCEP-NCAR data uses longitude in 0-360° format
    - Pan Amazon coordinates use -180/180° format (degrees_east)
    - All input files are daily long-term means (365 time steps)

Author: João Paulo Darela Filho
Date: February 2026
License: GNU GPLv3
"""

from pathlib import Path
from typing import Dict, Any
import json
import warnings

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data directory (same as script location)
DATA_DIR = Path(__file__).parent

# Input files (NCEP-NCAR Reanalysis 1 long-term means)
INPUT_FILES = {
    'air': DATA_DIR / "air.2m.gauss.day.ltm.1991-2020.nc",
    'nswrs': DATA_DIR / "nswrs.sfc.gauss.day.ltm.1991-2020.nc",
    'nlwrs': DATA_DIR / "nlwrs.sfc.gauss.day.ltm.1991-2020.nc",
    'gflux': DATA_DIR / "gflux.sfc.gauss.day.ltm.1991-2020.nc",
    'rhum': DATA_DIR / "rhum.sig995.day.ltm.1991-2020.nc",
    'pres': DATA_DIR / "pres.sfc.day.ltm.1991-2020.nc",
}

# Optional input files (for multivariate models)
OPTIONAL_FILES = {'rhum', 'pres'}

# Output directory and files
OUTPUT_DIR = DATA_DIR / "model_output"
COEFFICIENTS_FILE = OUTPUT_DIR / "ae_model_coefficients.json"
FORTRAN_FILE = OUTPUT_DIR / "ae_seasonal_model.f90"

# Pan Amazon region bounds (standard lat/lon, degrees_east and degrees_north)
# These will be converted to 0-360 longitude for NCEP data
PAN_AMAZON_BBOX = {
    "north": 10.5,    # degrees_north
    "south": -21.5,   # degrees_north
    "west": -80.0,    # degrees_east (will become 280.0 in NCEP)
    "east": -43.0,    # degrees_east (will become 317.0 in NCEP)
}

# Original CAETE model coefficients (for comparison)
ORIGINAL_MODEL = {
    'slope': 2.895,
    'intercept': 52.326,
    'source': 'NCEP-NCAR Reanalysis (original CAETE implementation)'
}


# =============================================================================
# COORDINATE CONVERSION UTILITIES
# =============================================================================

def lon_to_0_360(lon: float) -> float:
    """
    Convert longitude from -180/180 format to 0/360 format.

    NCEP-NCAR Reanalysis uses 0-360° longitude convention.

    Parameters
    ----------
    lon : float
        Longitude in degrees (-180 to 180)

    Returns
    -------
    float
        Longitude in degrees (0 to 360)

    Examples
    --------
    >>> lon_to_0_360(-80.0)
    280.0
    >>> lon_to_0_360(45.0)
    45.0
    """
    if lon < 0:
        return lon + 360.0
    return lon


def lon_to_180(lon: float) -> float:
    """
    Convert longitude from 0/360 format to -180/180 format.

    Parameters
    ----------
    lon : float
        Longitude in degrees (0 to 360)

    Returns
    -------
    float
        Longitude in degrees (-180 to 180)
    """
    if lon > 180:
        return lon - 360.0
    return lon


def doy_to_month(doy: np.ndarray) -> np.ndarray:
    """
    Convert day of year (1-365) to month (1-12).

    Uses standard non-leap year calendar.

    Parameters
    ----------
    doy : np.ndarray
        Day of year values (1-365)

    Returns
    -------
    np.ndarray
        Month values (1-12)
    """
    # Cumulative days at start of each month (non-leap year)
    month_starts = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

    months = np.zeros_like(doy, dtype=np.int32)
    for m in range(12):
        mask = (doy > month_starts[m]) & (doy <= month_starts[m + 1])
        months[mask] = m + 1

    # Handle edge cases
    months[months == 0] = 1

    return months


def saturation_vapor_pressure(temp_c: np.ndarray) -> np.ndarray:
    """
    Calculate saturation vapor pressure using Buck equation.

    This is the same formula used in CAETE's wtt() function.

    Parameters
    ----------
    temp_c : np.ndarray
        Temperature in degrees Celsius

    Returns
    -------
    np.ndarray
        Saturation vapor pressure in hPa (mbar)

    References
    ----------
    Buck AL (1981) New Equations for Computing Vapor Pressure and Enhancement Factor.
    J. Appl. Meteorol. 20:1527-1532.
    """
    es = np.where(
        temp_c >= 0,
        6.1121 * np.exp((18.729 - temp_c / 227.5) * temp_c / (257.87 + temp_c)),
        6.1115 * np.exp((23.036 - temp_c / 333.7) * temp_c / (279.82 + temp_c))
    )
    return es


def calculate_vpd(temp_c: xr.DataArray, rhum: xr.DataArray) -> xr.DataArray:
    """
    Calculate Vapor Pressure Deficit (VPD).

    VPD = es(T) × (1 - RH)

    Where:
        es(T) = saturation vapor pressure at temperature T
        RH = relative humidity (0-1 or 0-100%)

    Parameters
    ----------
    temp_c : xr.DataArray
        Temperature in degrees Celsius
    rhum : xr.DataArray
        Relative humidity (%, 0-100)

    Returns
    -------
    xr.DataArray
        Vapor pressure deficit in hPa (mbar)
    """
    # Convert RH from % to fraction if needed
    rh_frac = rhum / 100.0 if rhum.max() > 1.0 else rhum

    # Calculate saturation vapor pressure
    es = saturation_vapor_pressure(temp_c.values)

    # Calculate VPD
    vpd = es * (1.0 - rh_frac.values)

    # Create DataArray with same coordinates
    vpd_da = xr.DataArray(
        vpd,
        dims=temp_c.dims,
        coords=temp_c.coords,
        name='vpd',
        attrs={
            'units': 'hPa',
            'long_name': 'Vapor Pressure Deficit',
            'formula': 'VPD = es(T) × (1 - RH)'
        }
    )

    return vpd_da


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def check_input_files() -> tuple:
    """
    Verify input files exist.

    Returns
    -------
    tuple
        (required_ok: bool, optional_available: dict)
        - required_ok: True if all required files exist
        - optional_available: dict mapping optional file names to bool
    """
    required_ok = True
    optional_available = {}

    for name, filepath in INPUT_FILES.items():
        is_optional = name in OPTIONAL_FILES
        exists = filepath.exists()

        if is_optional:
            optional_available[name] = exists
            status = "Found" if exists else "Not found (optional)"
        else:
            if not exists:
                required_ok = False
            status = "Found" if exists else "MISSING (required)"

        print(f"  {status}: {filepath.name}")

    return required_ok, optional_available


def load_ncep_data() -> Dict[str, xr.DataArray]:
    """
    Load all NCEP-NCAR Reanalysis data files.

    Returns
    -------
    dict
        Dictionary with keys 'air', 'nswrs', 'nlwrs', 'gflux' containing DataArrays.
        Optionally includes 'rhum' and 'pres' if available.

    Raises
    ------
    FileNotFoundError
        If any required input file is missing
    """
    print("\n" + "=" * 60)
    print("LOADING NCEP-NCAR REANALYSIS DATA")
    print("=" * 60)

    required_ok, optional_available = check_input_files()

    if not required_ok:
        raise FileNotFoundError("One or more required input files are missing. "
                               "Download from: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html")

    data = {}

    # Use CFDatetimeCoder for proper cftime handling (xarray 2025+ API)
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        decode_times_arg = {'decode_times': time_coder}
    except AttributeError:
        # Fallback for older xarray versions
        decode_times_arg = {'use_cftime': True}

    # Load air temperature (K)
    ds = xr.open_dataset(INPUT_FILES['air'], **decode_times_arg)
    data['air'] = ds['air']
    print(f"\n  Air temperature: {data['air'].shape}")
    print(f"    Units: {data['air'].attrs.get('units', 'unknown')}")
    print(f"    Range: [{float(data['air'].min()):.2f}, {float(data['air'].max()):.2f}]")

    # Load net shortwave radiation (W/m²)
    ds = xr.open_dataset(INPUT_FILES['nswrs'], **decode_times_arg)
    data['nswrs'] = ds['nswrs']
    print(f"\n  Net shortwave radiation: {data['nswrs'].shape}")
    print(f"    Units: {data['nswrs'].attrs.get('units', 'unknown')}")
    print(f"    Range: [{float(data['nswrs'].min()):.2f}, {float(data['nswrs'].max()):.2f}]")

    # Load net longwave radiation (W/m²)
    ds = xr.open_dataset(INPUT_FILES['nlwrs'], **decode_times_arg)
    data['nlwrs'] = ds['nlwrs']
    print(f"\n  Net longwave radiation: {data['nlwrs'].shape}")
    print(f"    Units: {data['nlwrs'].attrs.get('units', 'unknown')}")

    # Load ground heat flux (W/m²)
    ds = xr.open_dataset(INPUT_FILES['gflux'], **decode_times_arg)
    data['gflux'] = ds['gflux']
    print(f"\n  Ground heat flux: {data['gflux'].shape}")
    print(f"    Units: {data['gflux'].attrs.get('units', 'unknown')}")

    # Load relative humidity (optional)
    if optional_available.get('rhum', False):
        ds = xr.open_dataset(INPUT_FILES['rhum'], **decode_times_arg)
        data['rhum'] = ds['rhum']
        print(f"\n  Relative humidity: {data['rhum'].shape}")
        print(f"    Units: {data['rhum'].attrs.get('units', 'unknown')}")
        print(f"    Range: [{float(data['rhum'].min()):.2f}, {float(data['rhum'].max()):.2f}]")
        print(f"    Note: 1991-2020 climatology period")

    # Load surface pressure (optional)
    if optional_available.get('pres', False):
        ds = xr.open_dataset(INPUT_FILES['pres'], **decode_times_arg)
        data['pres'] = ds['pres']
        print(f"\n  Surface pressure: {data['pres'].shape}")
        print(f"    Units: {data['pres'].attrs.get('units', 'unknown')}")
        print(f"    Range: [{float(data['pres'].min()):.2f}, {float(data['pres'].max()):.2f}]")

    return data


def align_to_daily(data: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """
    Verify all datasets have consistent daily time resolution.

    All daily LTM files should have 365 time steps. This function
    validates alignment and handles any mismatched time dimensions.

    Parameters
    ----------
    data : dict
        Dictionary with 'air', 'nswrs', 'nlwrs', 'gflux' DataArrays.
        May also include 'rhum' and 'pres'.

    Returns
    -------
    dict
        Dictionary with all DataArrays aligned to daily resolution (365 time steps)
    """
    print("\n" + "-" * 40)
    print("Verifying daily resolution alignment...")

    aligned = {}

    # Required variables - must be daily (365 time steps)
    required_vars = ['air', 'nswrs', 'nlwrs', 'gflux']
    for var in required_vars:
        aligned[var] = data[var]
        print(f"  {var}: {len(data[var].time)} time steps")

    # Optional variables
    for var in ['rhum', 'pres']:
        if var in data:
            aligned[var] = data[var]
            print(f"  {var}: {len(data[var].time)} time steps (optional)")

    # Verify required vars have same time dimension
    shapes = {k: len(v.time) for k, v in aligned.items() if k in required_vars}
    print(f"  Required time dimensions: {shapes}")

    if len(set(shapes.values())) > 1:
        raise ValueError(f"Time dimensions do not match after alignment: {shapes}")

    return aligned


def convert_temperature_to_celsius(temp_k: xr.DataArray) -> xr.DataArray:
    """
    Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    temp_k : xr.DataArray
        Temperature in Kelvin

    Returns
    -------
    xr.DataArray
        Temperature in Celsius
    """
    temp_c = temp_k - 273.15
    temp_c.name = 'air_temperature'
    temp_c.attrs = temp_k.attrs.copy()
    temp_c.attrs['units'] = 'degC'
    temp_c.attrs['long_name'] = 'Air temperature at 2m'
    return temp_c


def calculate_available_energy(
    nswrs: xr.DataArray,
    nlwrs: xr.DataArray,
    gflux: xr.DataArray
) -> xr.DataArray:
    """
    Calculate available energy for evaporation.

    Available energy (AE) is the net radiation minus ground heat flux:
        AE = Rn - G = SW_absorbed - LW_lost - G_into_ground

    NCEP-NCAR Sign Conventions:
        - nswrs: NEGATIVE when absorbed by surface (positive upward convention)
        - nlwrs: POSITIVE when leaving surface (net upward longwave loss)
        - gflux: POSITIVE when heat flows into ground

    Therefore:
        AE = (-nswrs) - nlwrs - gflux

    Parameters
    ----------
    nswrs : xr.DataArray
        Net shortwave radiation at surface (W/m², NCEP: negative = absorbed)
    nlwrs : xr.DataArray
        Net longwave radiation at surface (W/m², NCEP: positive = upward/lost)
    gflux : xr.DataArray
        Ground heat flux (W/m², positive = into ground)

    Returns
    -------
    xr.DataArray
        Available energy (W/m², positive = energy available for evaporation)

    Notes
    -----
    Available energy represents the energy available for latent and sensible heat fluxes.
    For tropical regions, AE should be positive (typically 50-200 W/m²).
    """
    print("\n" + "-" * 40)
    print("Calculating available energy...")
    print("  NCEP sign conventions:")
    print(f"    nswrs range: [{float(nswrs.min()):.1f}, {float(nswrs.max()):.1f}] (negative = absorbed)")
    print(f"    nlwrs range: [{float(nlwrs.min()):.1f}, {float(nlwrs.max()):.1f}] (positive = lost)")
    print(f"    gflux range: [{float(gflux.min()):.1f}, {float(gflux.max()):.1f}] (positive = into ground)")
    print("  Formula: AE = (-nswrs) - nlwrs - gflux")

    ae = (-nswrs) - nlwrs - gflux
    ae.name = 'available_energy'
    ae.attrs['units'] = 'W m-2'
    ae.attrs['long_name'] = 'Available Energy (Net Radiation - Ground Heat Flux)'
    ae.attrs['formula'] = 'AE = (-nswrs) - nlwrs - gflux (NCEP convention)'

    print(f"  Result shape: {ae.shape}")
    print(f"  Range: [{float(ae.min()):.2f}, {float(ae.max()):.2f}] W/m²")

    return ae


def extract_pan_amazon(
    data: xr.DataArray,
    bbox: Dict[str, float] = PAN_AMAZON_BBOX
) -> xr.DataArray:
    """
    Extract Pan Amazon region from global data.

    Handles longitude conversion from -180/180 to 0/360 format used by NCEP.

    Parameters
    ----------
    data : xr.DataArray
        Global data with 'lat' and 'lon' dimensions
    bbox : dict
        Bounding box with 'north', 'south', 'west', 'east' keys

    Returns
    -------
    xr.DataArray
        Data subset to Pan Amazon region
    """
    # Convert longitude bounds to 0-360 format
    west_360 = lon_to_0_360(bbox['west'])
    east_360 = lon_to_0_360(bbox['east'])

    print("\n" + "-" * 40)
    print("Extracting Pan Amazon region...")
    print(f"  Latitude:  {bbox['south']:.1f}°N to {bbox['north']:.1f}°N")
    print(f"  Longitude: {bbox['west']:.1f}°E ({west_360:.1f}° in 0-360) to "
          f"{bbox['east']:.1f}°E ({east_360:.1f}° in 0-360)")

    # Check latitude order in data (NCEP often has descending latitudes)
    lat_ascending = data.lat.values[0] < data.lat.values[-1]

    if lat_ascending:
        lat_slice = slice(bbox['south'], bbox['north'])
    else:
        lat_slice = slice(bbox['north'], bbox['south'])

    # Select region
    regional = data.sel(
        lat=lat_slice,
        lon=slice(west_360, east_360)
    )

    if regional.lat.size == 0 or regional.lon.size == 0:
        raise ValueError(f"Region selection returned empty data. "
                        f"Check coordinate bounds. "
                        f"Data lat range: [{data.lat.min().values}, {data.lat.max().values}], "
                        f"Data lon range: [{data.lon.min().values}, {data.lon.max().values}]")

    print(f"  Selected shape: {regional.shape}")
    print(f"  Lat range: [{float(regional.lat.min()):.2f}, {float(regional.lat.max()):.2f}]")
    print(f"  Lon range: [{float(regional.lon.min()):.2f}, {float(regional.lon.max()):.2f}]")

    return regional


def get_day_of_year(data: xr.DataArray) -> np.ndarray:
    """
    Extract day of year (1-365) from time coordinate.

    For long-term mean (LTM) data, the time coordinate represents climatological
    days, not actual dates. We assume the time index corresponds to day of year.

    Parameters
    ----------
    data : xr.DataArray
        Data with time dimension

    Returns
    -------
    np.ndarray
        Array of day-of-year values (1-365)
    """
    n_times = len(data.time)

    if n_times == 365:
        # LTM data: assume index = day of year
        return np.arange(1, 366)
    elif n_times == 366:
        # Leap year LTM
        return np.arange(1, 367)
    elif hasattr(data.time, 'dt'):
        # Has datetime accessor
        return data.time.dt.dayofyear.values
    else:
        warnings.warn(f"Cannot determine day of year from {n_times} time steps. "
                     "Assuming sequential days starting from 1.")
        return np.arange(1, n_times + 1)


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_linear_model(
    temperature: np.ndarray,
    available_energy: np.ndarray,
    model_name: str = "Linear"
) -> Dict[str, float]:
    """
    Fit a simple linear regression model: AE = slope × T + intercept

    Parameters
    ----------
    temperature : np.ndarray
        Temperature values (°C)
    available_energy : np.ndarray
        Available energy values (W/m²)
    model_name : str
        Name for logging purposes

    Returns
    -------
    dict
        Model coefficients and statistics:
        - slope: Regression slope (W/m²/°C)
        - intercept: Regression intercept (W/m²)
        - r_squared: Coefficient of determination
        - std_err: Standard error of the slope
        - p_value: P-value for the slope
        - n_samples: Number of valid samples used
        - rmse: Root mean square error
    """
    # Remove NaN and infinite values
    mask = np.isfinite(temperature) & np.isfinite(available_energy)
    temp_clean = temperature[mask]
    ae_clean = available_energy[mask]

    n_samples = len(temp_clean)

    if n_samples < 10:
        warnings.warn(f"{model_name}: Insufficient data ({n_samples} samples)")
        return {
            'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan,
            'std_err': np.nan, 'p_value': np.nan, 'n_samples': n_samples, 'rmse': np.nan
        }

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(temp_clean, ae_clean)

    # Calculate RMSE
    predicted = slope * temp_clean + intercept
    rmse = np.sqrt(np.mean((ae_clean - predicted) ** 2))

    r_squared = r_value ** 2

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'std_err': float(std_err),
        'p_value': float(p_value),
        'n_samples': int(n_samples),
        'rmse': float(rmse)
    }


def fit_multivariate_model(
    predictors: Dict[str, np.ndarray],
    target: np.ndarray,
    model_name: str = "Multivariate"
) -> Dict[str, Any]:
    """
    Fit a multivariate linear regression model: AE = Σ(β_i × X_i) + β_0

    Parameters
    ----------
    predictors : dict
        Dictionary mapping predictor names to 1D arrays
        e.g., {'T': temp_array, 'RH': rh_array, 'VPD': vpd_array}
    target : np.ndarray
        Target variable (Available Energy, W/m²)
    model_name : str
        Name for logging purposes

    Returns
    -------
    dict
        Model coefficients and statistics:
        - coefficients: dict mapping predictor names to coefficients
        - intercept: Regression intercept (W/m²)
        - r_squared: Coefficient of determination
        - adj_r_squared: Adjusted R²
        - rmse: Root mean square error
        - n_samples: Number of valid samples used
        - feature_names: List of predictor names in order
    """
    # Stack predictors into matrix
    feature_names = list(predictors.keys())
    X_arrays = [predictors[name] for name in feature_names]
    X = np.column_stack(X_arrays)
    y = target

    # Create combined mask for all finite values
    mask = np.isfinite(y)
    for arr in X_arrays:
        mask &= np.isfinite(arr)

    X_clean = X[mask]
    y_clean = y[mask]

    n_samples = len(y_clean)
    n_features = len(feature_names)

    if n_samples < n_features + 10:
        warnings.warn(f"{model_name}: Insufficient data ({n_samples} samples for {n_features} features)")
        return {
            'coefficients': {name: np.nan for name in feature_names},
            'intercept': np.nan,
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'rmse': np.nan,
            'n_samples': n_samples,
            'feature_names': feature_names
        }

    # Fit model using sklearn
    model = LinearRegression()
    model.fit(X_clean, y_clean)

    # Predictions and metrics
    y_pred = model.predict(X_clean)
    r_squared = r2_score(y_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))

    # Adjusted R²
    adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)

    # Build coefficients dict
    coefficients = {name: float(coef) for name, coef in zip(feature_names, model.coef_)}

    return {
        'coefficients': coefficients,
        'intercept': float(model.intercept_),
        'r_squared': float(r_squared),
        'adj_r_squared': float(adj_r_squared),
        'rmse': float(rmse),
        'n_samples': int(n_samples),
        'feature_names': feature_names
    }


def fit_all_multivariate_models(
    temp: xr.DataArray,
    ae: xr.DataArray,
    rhum: xr.DataArray = None,
    pres: xr.DataArray = None
) -> Dict[str, Dict]:
    """
    Fit multiple multivariate model variants and compare.

    Models fitted:
    1. AE = f(T) - univariate baseline
    2. AE = f(T, RH) - temperature + relative humidity
    3. AE = f(T, VPD) - temperature + vapor pressure deficit
    4. AE = f(T, RH, P) - all predictors

    Parameters
    ----------
    temp : xr.DataArray
        Temperature (°C)
    ae : xr.DataArray
        Available Energy (W/m²)
    rhum : xr.DataArray, optional
        Relative humidity (%)
    pres : xr.DataArray, optional
        Surface pressure (Pa or hPa)

    Returns
    -------
    dict
        Dictionary of model results keyed by model name
    """
    print("\n" + "=" * 70)
    print("FITTING MULTIVARIATE MODELS")
    print("=" * 70)

    # Flatten arrays for temperature and AE (reference grid)
    T = temp.values.flatten()
    AE = ae.values.flatten()

    models = {}

    # Model 1: Univariate T (baseline)
    print("\n1. Univariate: AE = f(T)")
    models['T_only'] = fit_multivariate_model({'T': T}, AE, "T only")
    print(f"   AE = {models['T_only']['coefficients']['T']:.4f}×T + {models['T_only']['intercept']:.4f}")
    print(f"   R² = {models['T_only']['r_squared']:.4f}, RMSE = {models['T_only']['rmse']:.2f}")

    if rhum is not None:
        # Check if grids match, if not interpolate RH to temperature grid
        if rhum.shape != temp.shape:
            print(f"\n   Note: Interpolating RH from {rhum.shape} to {temp.shape}")
            rhum_interp = rhum.interp(lat=temp.lat, lon=temp.lon, method='linear')
        else:
            rhum_interp = rhum

        RH = rhum_interp.values.flatten()

        # Model 2: T + RH
        print("\n2. Bivariate: AE = f(T, RH)")
        models['T_RH'] = fit_multivariate_model({'T': T, 'RH': RH}, AE, "T + RH")
        coef = models['T_RH']['coefficients']
        print(f"   AE = {coef['T']:.4f}×T + {coef['RH']:.4f}×RH + {models['T_RH']['intercept']:.4f}")
        print(f"   R² = {models['T_RH']['r_squared']:.4f}, RMSE = {models['T_RH']['rmse']:.2f}")

        # Model 3: T + VPD
        print("\n3. Bivariate: AE = f(T, VPD)")
        vpd = calculate_vpd(temp, rhum_interp)
        VPD = vpd.values.flatten()
        models['T_VPD'] = fit_multivariate_model({'T': T, 'VPD': VPD}, AE, "T + VPD")
        coef = models['T_VPD']['coefficients']
        print(f"   AE = {coef['T']:.4f}×T + {coef['VPD']:.4f}×VPD + {models['T_VPD']['intercept']:.4f}")
        print(f"   R² = {models['T_VPD']['r_squared']:.4f}, RMSE = {models['T_VPD']['rmse']:.2f}")

        # Model 4: VPD only
        print("\n4. Univariate: AE = f(VPD)")
        models['VPD_only'] = fit_multivariate_model({'VPD': VPD}, AE, "VPD only")
        print(f"   AE = {models['VPD_only']['coefficients']['VPD']:.4f}×VPD + {models['VPD_only']['intercept']:.4f}")
        print(f"   R² = {models['VPD_only']['r_squared']:.4f}, RMSE = {models['VPD_only']['rmse']:.2f}")

        if pres is not None:
            # Check if grids match, if not interpolate pressure to temperature grid
            if pres.shape != temp.shape:
                print(f"\n   Note: Interpolating P from {pres.shape} to {temp.shape}")
                pres_interp = pres.interp(lat=temp.lat, lon=temp.lon, method='linear')
            else:
                pres_interp = pres

            P = pres_interp.values.flatten()
            # Convert Pa to hPa if needed
            if np.nanmean(P) > 10000:
                P = P / 100.0

            # Model 5: T + RH + P
            print("\n5. Full: AE = f(T, RH, P)")
            models['T_RH_P'] = fit_multivariate_model({'T': T, 'RH': RH, 'P': P}, AE, "T + RH + P")
            coef = models['T_RH_P']['coefficients']
            print(f"   AE = {coef['T']:.4f}×T + {coef['RH']:.4f}×RH + {coef['P']:.4f}×P + {models['T_RH_P']['intercept']:.4f}")
            print(f"   R² = {models['T_RH_P']['r_squared']:.4f}, RMSE = {models['T_RH_P']['rmse']:.2f}")

            # Model 6: T + VPD + P
            print("\n6. Full: AE = f(T, VPD, P)")
            models['T_VPD_P'] = fit_multivariate_model({'T': T, 'VPD': VPD, 'P': P}, AE, "T + VPD + P")
            coef = models['T_VPD_P']['coefficients']
            print(f"   AE = {coef['T']:.4f}×T + {coef['VPD']:.4f}×VPD + {coef['P']:.4f}×P + {models['T_VPD_P']['intercept']:.4f}")
            print(f"   R² = {models['T_VPD_P']['r_squared']:.4f}, RMSE = {models['T_VPD_P']['rmse']:.2f}")

    # Print comparison summary
    print("\n" + "-" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("-" * 70)
    print(f"{'Model':<20} {'R²':>10} {'Adj R²':>10} {'RMSE':>10} {'ΔR² vs T':>12}")
    print("-" * 70)

    baseline_r2 = models['T_only']['r_squared']
    for name, m in sorted(models.items(), key=lambda x: -x[1]['r_squared']):
        delta_r2 = m['r_squared'] - baseline_r2
        delta_str = f"+{delta_r2:.4f}" if delta_r2 > 0 else f"{delta_r2:.4f}"
        print(f"{name:<20} {m['r_squared']:>10.4f} {m['adj_r_squared']:>10.4f} {m['rmse']:>10.2f} {delta_str:>12}")

    return models


def fit_global_model(
    temp_regional: xr.DataArray,
    ae_regional: xr.DataArray
) -> Dict[str, float]:
    """
    Fit a global (time-invariant) linear model using all data.

    Parameters
    ----------
    temp_regional : xr.DataArray
        Temperature data for region (°C)
    ae_regional : xr.DataArray
        Available energy data for region (W/m²)

    Returns
    -------
    dict
        Global model coefficients and statistics
    """
    print("\n" + "=" * 60)
    print("FITTING GLOBAL LINEAR MODEL")
    print("=" * 60)

    # Flatten all dimensions
    temp_flat = temp_regional.values.flatten()
    ae_flat = ae_regional.values.flatten()

    model = fit_linear_model(temp_flat, ae_flat, "Global")

    print(f"\n  Equation: AE = {model['slope']:.4f} × T + {model['intercept']:.4f}")
    print(f"  R² = {model['r_squared']:.4f}")
    print(f"  RMSE = {model['rmse']:.2f} W/m²")
    print(f"  N = {model['n_samples']:,} samples")

    return model


def fit_monthly_model(
    temp_regional: xr.DataArray,
    ae_regional: xr.DataArray
) -> Dict[int, Dict[str, float]]:
    """
    Fit monthly linear models (12 sets of coefficients).

    Parameters
    ----------
    temp_regional : xr.DataArray
        Temperature data for region (°C)
    ae_regional : xr.DataArray
        Available energy data for region (W/m²)

    Returns
    -------
    dict
        Dictionary mapping month (1-12) to model coefficients
    """
    print("\n" + "=" * 60)
    print("FITTING MONTHLY SEASONAL MODEL")
    print("=" * 60)

    # Get day of year and convert to months
    doy = get_day_of_year(temp_regional)
    months = doy_to_month(doy)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"\n{'Month':<6} {'Slope':>10} {'Intercept':>12} {'R²':>8} {'RMSE':>8} {'N':>10}")
    print("-" * 60)

    monthly_models = {}

    for month in range(1, 13):
        # Select time indices for this month
        month_mask = months == month

        if not np.any(month_mask):
            warnings.warn(f"No data for month {month}")
            monthly_models[month] = {
                'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan,
                'std_err': np.nan, 'p_value': np.nan, 'n_samples': 0, 'rmse': np.nan
            }
            continue

        # Extract data for this month
        temp_month = temp_regional.isel(time=month_mask)
        ae_month = ae_regional.isel(time=month_mask)

        # Flatten and fit
        temp_flat = temp_month.values.flatten()
        ae_flat = ae_month.values.flatten()

        model = fit_linear_model(temp_flat, ae_flat, f"Month {month}")
        monthly_models[month] = model

        print(f"{month_names[month-1]:<6} {model['slope']:>10.4f} "
              f"{model['intercept']:>12.4f} {model['r_squared']:>8.4f} "
              f"{model['rmse']:>8.2f} {model['n_samples']:>10,}")

    return monthly_models


def fit_daily_model(
    temp_regional: xr.DataArray,
    ae_regional: xr.DataArray
) -> Dict[int, Dict[str, float]]:
    """
    Fit daily linear models (365 sets of coefficients).

    This provides the finest temporal resolution for seasonal variation.

    Parameters
    ----------
    temp_regional : xr.DataArray
        Temperature data for region (°C)
    ae_regional : xr.DataArray
        Available energy data for region (W/m²)

    Returns
    -------
    dict
        Dictionary mapping day of year (1-365) to model coefficients
    """
    print("\n" + "=" * 60)
    print("FITTING DAILY SEASONAL MODEL (365 coefficients)")
    print("=" * 60)

    doy = get_day_of_year(temp_regional)
    daily_models = {}

    valid_count = 0
    slopes = []
    intercepts = []
    r2_values = []

    for day in range(1, 366):
        day_mask = doy == day

        if not np.any(day_mask):
            daily_models[day] = {
                'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan,
                'std_err': np.nan, 'p_value': np.nan, 'n_samples': 0, 'rmse': np.nan
            }
            continue

        # Extract and flatten
        temp_day = temp_regional.isel(time=day_mask)
        ae_day = ae_regional.isel(time=day_mask)

        temp_flat = temp_day.values.flatten()
        ae_flat = ae_day.values.flatten()

        model = fit_linear_model(temp_flat, ae_flat, f"Day {day}")
        daily_models[day] = model

        if not np.isnan(model['slope']):
            valid_count += 1
            slopes.append(model['slope'])
            intercepts.append(model['intercept'])
            r2_values.append(model['r_squared'])

    # Print summary
    print(f"\n  Valid days: {valid_count}/365")
    if slopes:
        print(f"  Slope range: [{min(slopes):.4f}, {max(slopes):.4f}], mean: {np.mean(slopes):.4f}")
        print(f"  Intercept range: [{min(intercepts):.2f}, {max(intercepts):.2f}], mean: {np.mean(intercepts):.2f}")
        print(f"  R² range: [{min(r2_values):.4f}, {max(r2_values):.4f}], mean: {np.mean(r2_values):.4f}")

    return daily_models


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def interpolate_missing_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values in coefficient array.

    Uses linear interpolation with wrap-around for cyclic data (day 365 -> day 1).

    Parameters
    ----------
    coeffs : np.ndarray
        Array of coefficients (length 365), may contain NaN

    Returns
    -------
    np.ndarray
        Array with NaN values interpolated
    """
    arr = np.array(coeffs, dtype=np.float64)
    nans = np.isnan(arr)

    if nans.all():
        warnings.warn("All coefficients are NaN, cannot interpolate")
        return arr

    if not nans.any():
        return arr

    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])

    return arr


def save_coefficients_json(
    results: Dict[str, Any],
    filepath: Path
) -> None:
    """
    Save all model coefficients to JSON file.

    Parameters
    ----------
    results : dict
        Dictionary containing all model results
    filepath : Path
        Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCoefficients saved to: {filepath}")


def add_multivariate_fortran(
    fortran_source: str,
    multivariate_models: Dict[str, Dict[str, Any]]
) -> str:
    """
    Add multivariate model functions to Fortran source.

    Parameters
    ----------
    fortran_source : str
        Existing Fortran source code
    multivariate_models : dict
        Dictionary of multivariate model results

    Returns
    -------
    str
        Updated Fortran source with multivariate functions
    """
    # Find best model (excluding T_only)
    best_name = None
    best_r2 = 0
    for name, model in multivariate_models.items():
        if name != 'T_only' and model['r_squared'] > best_r2:
            best_name = name
            best_r2 = model['r_squared']

    if best_name is None:
        return fortran_source

    best_model = multivariate_models[best_name]
    feature_names = best_model.get('feature_names', [])
    coefficients = best_model.get('coefficients', {})  # Dict: {feature_name: coef_value}
    intercept = float(best_model.get('intercept', 0))

    # Build coefficient declarations
    coef_declarations = []
    coef_names = []
    for feat in feature_names:
        coef = coefficients.get(feat, 0.0)
        coef_name = f"MV_{feat.upper()}_COEF"
        coef_names.append((feat, coef_name))
        coef_declarations.append(f"    real(r_8), parameter :: {coef_name} = {float(coef):.8f}_r_8")

    coef_declarations.append(f"    real(r_8), parameter :: MV_INTERCEPT = {intercept:.8f}_r_8")

    # Determine function parameters based on features
    has_T = 'T' in feature_names
    has_RH = 'RH' in feature_names
    has_VPD = 'VPD' in feature_names
    has_P = 'P' in feature_names

    # Build function signature and body
    params = []
    param_docs = []
    if has_T:
        params.append("temp")
        param_docs.append("    !> @param[in] temp  Air temperature (degrees Celsius)")
    if has_RH:
        params.append("rh")
        param_docs.append("    !> @param[in] rh    Relative humidity (%)")
    if has_VPD:
        params.append("vpd")
        param_docs.append("    !> @param[in] vpd   Vapor pressure deficit (kPa)")
    if has_P:
        params.append("pres")
        param_docs.append("    !> @param[in] pres  Surface pressure (Pa)")

    param_list = ", ".join(params)

    # Build equation components
    eq_parts = []
    for feat, coef_name in coef_names:
        if feat == 'T':
            eq_parts.append(f"{coef_name} * temp")
        elif feat == 'RH':
            eq_parts.append(f"{coef_name} * rh")
        elif feat == 'VPD':
            eq_parts.append(f"{coef_name} * vpd")
        elif feat == 'P':
            eq_parts.append(f"{coef_name} * pres")

    equation = " + ".join(eq_parts) + " + MV_INTERCEPT"

    # Build function name
    func_features = "_".join(f.lower() for f in feature_names)
    func_name = f"available_energy_{func_features}"

    # Create the multivariate function
    multivariate_func = f'''

    !> Calculate available energy using multivariate model ({best_name})
    !>
    !> Best model: R² = {best_r2:.4f}
    !>
{chr(10).join(param_docs)}
    !> @return    ae    Available energy (W/m²)
    pure function {func_name}({param_list}) result(ae)
        real(r_8), intent(in) :: {param_list}
        real(r_8) :: ae

        ae = {equation}

    end function {func_name}
'''

    # Add declarations before 'contains'
    coef_block = '\n'.join(coef_declarations) + '\n'

    # Find position to insert declarations (before 'contains')
    contains_pos = fortran_source.find('contains')
    if contains_pos == -1:
        return fortran_source

    # Find the line with 'contains'
    lines = fortran_source.split('\n')
    new_lines = []
    contains_found = False
    for line in lines:
        if 'contains' in line and not contains_found:
            # Add multivariate coefficients before 'contains'
            new_lines.append('')
            new_lines.append('    ! Multivariate model coefficients (best model: ' + best_name + ')')
            for decl in coef_declarations:
                new_lines.append(decl)
            new_lines.append('')
            contains_found = True
        new_lines.append(line)

    fortran_source = '\n'.join(new_lines)

    # Add function before 'end module'
    end_module_pos = fortran_source.rfind('end module')
    if end_module_pos == -1:
        return fortran_source

    fortran_source = fortran_source[:end_module_pos] + multivariate_func + '\n' + fortran_source[end_module_pos:]

    # Update public interface to include new function
    public_line = f"    public :: {func_name}"
    fortran_source = fortran_source.replace(
        "    public :: available_energy_daily",
        f"    public :: available_energy_daily\n{public_line}"
    )

    return fortran_source


def generate_fortran_module(
    global_model: Dict[str, float],
    monthly_model: Dict[int, Dict[str, float]],
    daily_model: Dict[int, Dict[str, float]],
    filepath: Path,
    multivariate_models: Dict[str, Dict[str, Any]] = None
) -> None:
    """
    Generate Fortran 90 module with available energy model functions.

    Creates a module with multiple functions:
    - available_energy_global(temp): Single linear model
    - available_energy_monthly(temp, month): 12 monthly models
    - available_energy_daily(temp, doy): 365 daily models
    - available_energy_vpd(temp, vpd): Best multivariate model with VPD (if available)

    Parameters
    ----------
    global_model : dict
        Global linear model coefficients
    monthly_model : dict
        Monthly model coefficients (1-12)
    daily_model : dict
        Daily model coefficients (1-365)
    filepath : Path
        Output file path
    multivariate_models : dict, optional
        Dictionary of multivariate model results
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Extract and prepare coefficients
    monthly_slopes = [monthly_model[m]['slope'] for m in range(1, 13)]
    monthly_intercepts = [monthly_model[m]['intercept'] for m in range(1, 13)]

    daily_slopes_raw = [daily_model.get(d, {'slope': np.nan})['slope'] for d in range(1, 366)]
    daily_intercepts_raw = [daily_model.get(d, {'intercept': np.nan})['intercept'] for d in range(1, 366)]

    # Interpolate missing values
    daily_slopes = interpolate_missing_coefficients(daily_slopes_raw)
    daily_intercepts = interpolate_missing_coefficients(daily_intercepts_raw)

    # Format coefficients for Fortran
    def format_array_fortran(values, per_line=6):
        """Format array values for Fortran with proper continuation."""
        lines = []
        for i in range(0, len(values), per_line):
            chunk = values[i:i+per_line]
            line = '        ' + ', '.join(f'{v:.6f}_r_8' for v in chunk)
            if i + per_line < len(values):
                line += ', &'
            lines.append(line)
        return '\n'.join(lines)

    # Build Fortran source
    fortran_source = f'''! =============================================================================
! Available Energy Seasonal Model for CAETE
! =============================================================================
!
! Generated from NCEP-NCAR Reanalysis 1 data (1991-2020 Long-Term Mean)
! Region: Pan Amazon ({PAN_AMAZON_BBOX['south']}°N to {PAN_AMAZON_BBOX['north']}°N,
!                     {PAN_AMAZON_BBOX['west']}°E to {PAN_AMAZON_BBOX['east']}°E)
!
! Models:
!   Global:  AE = {global_model['slope']:.6f} × T + {global_model['intercept']:.6f}  (R² = {global_model['r_squared']:.4f})
!   Monthly: 12 sets of slope/intercept pairs
!   Daily:   365 sets of slope/intercept pairs
!
! Usage:
!   use ae_seasonal_model
!   real(r_8) :: temp_celsius, ae
!   integer :: month, doy
!
!   ae = available_energy_global(temp_celsius)
!   ae = available_energy_monthly(temp_celsius, month)
!   ae = available_energy_daily(temp_celsius, doy)
!
! Author: Generated by available_energy_model.py
! Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
! =============================================================================

module ae_seasonal_model

    use types, only: r_8
    implicit none
    private

    ! Public interface
    public :: available_energy_global
    public :: available_energy_monthly
    public :: available_energy_daily

    ! Global model coefficients
    real(r_8), parameter :: GLOBAL_SLOPE = {global_model['slope']:.8f}_r_8
    real(r_8), parameter :: GLOBAL_INTERCEPT = {global_model['intercept']:.8f}_r_8

    ! Monthly model coefficients (12 months)
    real(r_8), parameter :: MONTHLY_SLOPES(12) = (/ &
{format_array_fortran(monthly_slopes, per_line=4)} /)

    real(r_8), parameter :: MONTHLY_INTERCEPTS(12) = (/ &
{format_array_fortran(monthly_intercepts, per_line=4)} /)

    ! Daily model coefficients (365 days)
    real(r_8), parameter :: DAILY_SLOPES(365) = (/ &
{format_array_fortran(daily_slopes, per_line=7)} /)

    real(r_8), parameter :: DAILY_INTERCEPTS(365) = (/ &
{format_array_fortran(daily_intercepts, per_line=7)} /)

contains

    !> Calculate available energy using global linear model
    !>
    !> @param[in] temp  Air temperature (degrees Celsius)
    !> @return    ae    Available energy (W/m²)
    pure function available_energy_global(temp) result(ae)
        real(r_8), intent(in) :: temp
        real(r_8) :: ae

        ae = GLOBAL_SLOPE * temp + GLOBAL_INTERCEPT

    end function available_energy_global


    !> Calculate available energy using monthly seasonal model
    !>
    !> @param[in] temp   Air temperature (degrees Celsius)
    !> @param[in] month  Month of year (1-12)
    !> @return    ae     Available energy (W/m²)
    pure function available_energy_monthly(temp, month) result(ae)
        real(r_8), intent(in) :: temp
        integer, intent(in) :: month
        real(r_8) :: ae
        integer :: m

        ! Clamp month to valid range [1, 12]
        m = max(1, min(12, month))

        ae = MONTHLY_SLOPES(m) * temp + MONTHLY_INTERCEPTS(m)

    end function available_energy_monthly


    !> Calculate available energy using daily seasonal model
    !>
    !> @param[in] temp  Air temperature (degrees Celsius)
    !> @param[in] doy   Day of year (1-365; leap day 366 maps to 365)
    !> @return    ae    Available energy (W/m²)
    pure function available_energy_daily(temp, doy) result(ae)
        real(r_8), intent(in) :: temp
        integer, intent(in) :: doy
        real(r_8) :: ae
        integer :: d

        ! Clamp day to valid range [1, 365]
        ! Note: Leap year day 366 is treated as day 365
        d = max(1, min(365, doy))

        ae = DAILY_SLOPES(d) * temp + DAILY_INTERCEPTS(d)

    end function available_energy_daily

end module ae_seasonal_model
'''

    # Add multivariate model functions if available
    if multivariate_models:
        fortran_source = add_multivariate_fortran(fortran_source, multivariate_models)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fortran_source)

    print(f"Fortran module saved to: {filepath}")


def print_model_summary(
    global_model: Dict[str, float],
    monthly_model: Dict[int, Dict[str, float]]
) -> None:
    """
    Print a formatted summary of all model equations.
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    # Global model
    print("\n1. GLOBAL LINEAR MODEL")
    print("-" * 50)
    print(f"   Equation: AE = {global_model['slope']:.4f} × T + {global_model['intercept']:.4f}")
    print(f"   R² = {global_model['r_squared']:.4f}, RMSE = {global_model['rmse']:.2f} W/m²")
    print(f"   N = {global_model['n_samples']:,} samples")

    # Comparison with original
    print("\n2. COMPARISON WITH ORIGINAL CAETE MODEL")
    print("-" * 50)
    print(f"   Original:  AE = {ORIGINAL_MODEL['slope']:.3f} × T + {ORIGINAL_MODEL['intercept']:.3f}")
    print(f"   New:       AE = {global_model['slope']:.3f} × T + {global_model['intercept']:.3f}")

    # Difference at typical temperatures
    print("\n   At typical Pan Amazon temperatures:")
    print(f"   {'T (°C)':<8} {'Original':>12} {'New':>12} {'Diff (%)':>12}")
    for t in [20, 25, 30]:
        orig = ORIGINAL_MODEL['slope'] * t + ORIGINAL_MODEL['intercept']
        new = global_model['slope'] * t + global_model['intercept']
        diff = (new - orig) / orig * 100
        print(f"   {t:<8} {orig:>12.1f} {new:>12.1f} {diff:>+12.1f}")

    # Monthly model
    print("\n3. MONTHLY SEASONAL MODEL")
    print("-" * 50)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"   {'Month':<6} {'Equation':<32} {'R²':>8}")
    for m in range(1, 13):
        coef = monthly_model[m]
        eq = f"AE = {coef['slope']:.3f}×T + {coef['intercept']:.1f}"
        print(f"   {month_names[m-1]:<6} {eq:<32} {coef['r_squared']:>8.4f}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_multivariate_comparison_plot(
    multivariate_models: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Create a comparison plot for multivariate models.

    Shows:
    1. Bar chart of R² values for each model
    2. Bar chart of RMSE values
    3. Feature importance for best model
    4. Model equations table
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sort models by R² for consistent ordering
    model_names = list(multivariate_models.keys())
    r2_values = [multivariate_models[name]['r_squared'] for name in model_names]
    rmse_values = [multivariate_models[name]['rmse'] for name in model_names]

    # Sort by R²
    sorted_idx = np.argsort(r2_values)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    r2_values = [r2_values[i] for i in sorted_idx]
    rmse_values = [rmse_values[i] for i in sorted_idx]

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))

    # --- Panel 1: R² comparison ---
    ax1 = axes[0, 0]
    bars1 = ax1.barh(range(len(model_names)), r2_values, color=colors, edgecolor='black', linewidth=0.5)

    # Highlight T_only as baseline
    t_only_idx = model_names.index('T_only') if 'T_only' in model_names else None
    if t_only_idx is not None:
        bars1[t_only_idx].set_edgecolor('red')
        bars1[t_only_idx].set_linewidth(2)

    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels([name.replace('_', '+') for name in model_names])
    ax1.set_xlabel('R²', fontsize=11)
    ax1.set_title('Model Performance (R²)', fontsize=12, fontweight='bold')
    ax1.set_xlim(min(r2_values) - 0.02, max(r2_values) + 0.02)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, r2_values)):
        ax1.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

    ax1.axvline(r2_values[model_names.index('T_only')] if 'T_only' in model_names else 0,
                color='red', linestyle='--', alpha=0.5, label='T-only baseline')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')

    # --- Panel 2: RMSE comparison ---
    ax2 = axes[0, 1]
    bars2 = ax2.barh(range(len(model_names)), rmse_values, color=colors, edgecolor='black', linewidth=0.5)

    if t_only_idx is not None:
        bars2[t_only_idx].set_edgecolor('red')
        bars2[t_only_idx].set_linewidth(2)

    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels([name.replace('_', '+') for name in model_names])
    ax2.set_xlabel('RMSE (W/m²)', fontsize=11)
    ax2.set_title('Model Error (RMSE)', fontsize=12, fontweight='bold')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
        ax2.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=9)

    ax2.grid(True, alpha=0.3, axis='x')

    # --- Panel 3: R² improvement over T-only ---
    ax3 = axes[1, 0]

    if 'T_only' in model_names:
        t_only_r2 = multivariate_models['T_only']['r_squared']
        improvements = [(r2 - t_only_r2) * 100 for r2 in r2_values]  # % improvement

        bar_colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax3.barh(range(len(model_names)), improvements, color=bar_colors, alpha=0.7,
                         edgecolor='black', linewidth=0.5)

        ax3.axvline(0, color='black', linewidth=1)
        ax3.set_yticks(range(len(model_names)))
        ax3.set_yticklabels([name.replace('_', '+') for name in model_names])
        ax3.set_xlabel('R² Improvement (%)', fontsize=11)
        ax3.set_title('R² Improvement vs T-only Model', fontsize=12, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, improvements)):
            offset = 0.2 if val >= 0 else -0.2
            ha = 'left' if val >= 0 else 'right'
            ax3.text(val + offset, i, f'{val:+.2f}%', va='center', ha=ha, fontsize=9)

        ax3.grid(True, alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'T-only model not available', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title('R² Improvement vs T-only Model', fontsize=12, fontweight='bold')

    # --- Panel 4: Best model coefficients ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Find best model
    best_name = model_names[0]
    best_model = multivariate_models[best_name]

    # Create table text
    table_lines = [
        "BEST MODEL SUMMARY",
        "=" * 40,
        f"Model: {best_name.replace('_', ' + ')}",
        f"R² = {best_model['r_squared']:.4f}",
        f"RMSE = {best_model['rmse']:.2f} W/m²",
        "",
        "Coefficients:",
        "-" * 40
    ]

    feature_names = best_model.get('feature_names', ['T'])
    coefficients = best_model.get('coefficients', {})  # Dict: {feature_name: coef_value}
    intercept = float(best_model.get('intercept', 0))

    for feat in feature_names:
        coef = coefficients.get(feat, 0.0)
        table_lines.append(f"  {feat:12s}: {float(coef):+.4f}")

    table_lines.append(f"  {'Intercept':12s}: {intercept:+.2f}")
    table_lines.append("")
    table_lines.append("Equation:")
    table_lines.append("-" * 40)

    # Build equation
    eq_parts = []
    for feat in feature_names:
        coef = float(coefficients.get(feat, 0.0))
        if feat == 'T':
            eq_parts.append(f"{coef:.3f}×T")
        elif feat == 'RH':
            eq_parts.append(f"{coef:+.3f}×RH")
        elif feat == 'VPD':
            eq_parts.append(f"{coef:+.3f}×VPD")
        elif feat == 'P':
            eq_parts.append(f"{coef:+.4f}×P")

    equation = f"AE = {' '.join(eq_parts)} {intercept:+.1f}"
    table_lines.append(f"  {equation}")

    table_text = '\n'.join(table_lines)
    ax4.text(0.05, 0.95, table_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('Multivariate Model Comparison\n'
                 'Available Energy = f(Temperature, Humidity, Pressure)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'ae_multivariate_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_dir / 'ae_multivariate_comparison.png'}")


def create_diagnostic_plots(
    temp_regional: xr.DataArray,
    ae_regional: xr.DataArray,
    global_model: Dict[str, float],
    monthly_model: Dict[int, Dict[str, float]],
    output_dir: Path,
    multivariate_models: Dict[str, Dict[str, Any]] = None
) -> None:
    """
    Create diagnostic plots for model validation.

    Creates:
    1. Scatter plot with regression lines
    2. Residuals distribution
    3. Monthly coefficient variation
    4. Seasonal cycle comparison
    5. Multivariate model comparison (if provided)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 60)

    # Flatten data
    temp_flat = temp_regional.values.flatten()
    ae_flat = ae_regional.values.flatten()

    mask = np.isfinite(temp_flat) & np.isfinite(ae_flat)
    temp_clean = temp_flat[mask]
    ae_clean = ae_flat[mask]

    # --- Figure 1: Main diagnostic plot ---
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1a. Scatter plot with regression
    ax1 = fig.add_subplot(gs[0, 0])

    # Subsample for efficiency
    n_plot = min(30000, len(temp_clean))
    idx = np.random.choice(len(temp_clean), n_plot, replace=False)

    ax1.scatter(temp_clean[idx], ae_clean[idx], alpha=0.15, s=2, c='steelblue',
                edgecolors='none', rasterized=True)

    x_line = np.linspace(temp_clean.min(), temp_clean.max(), 100)

    # New model line
    y_new = global_model['slope'] * x_line + global_model['intercept']
    ax1.plot(x_line, y_new, 'r-', linewidth=2.5,
             label=f"New: {global_model['slope']:.3f}×T + {global_model['intercept']:.1f}")

    # Original model line
    y_orig = ORIGINAL_MODEL['slope'] * x_line + ORIGINAL_MODEL['intercept']
    ax1.plot(x_line, y_orig, 'g--', linewidth=2,
             label=f"Original: {ORIGINAL_MODEL['slope']:.3f}×T + {ORIGINAL_MODEL['intercept']:.1f}")

    ax1.set_xlabel('Temperature (°C)', fontsize=11)
    ax1.set_ylabel('Available Energy (W/m²)', fontsize=11)
    ax1.set_title('Temperature vs Available Energy', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 1b. Residuals histogram
    ax2 = fig.add_subplot(gs[0, 1])

    residuals = ae_clean - (global_model['slope'] * temp_clean + global_model['intercept'])

    ax2.hist(residuals, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(residuals.mean(), color='orange', linestyle='-', linewidth=1.5,
                label=f'Mean: {residuals.mean():.1f}')

    ax2.set_xlabel('Residuals (W/m²)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Residuals (σ = {residuals.std():.1f} W/m²)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 1c. Monthly coefficients
    ax3 = fig.add_subplot(gs[0, 2])

    months = np.arange(1, 13)
    slopes = [monthly_model[m]['slope'] for m in months]
    intercepts = [monthly_model[m]['intercept'] for m in months]

    color_slope = 'tab:blue'
    ax3.bar(months - 0.2, slopes, width=0.35, color=color_slope, alpha=0.7, label='Slope')
    ax3.axhline(global_model['slope'], color=color_slope, linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_ylabel('Slope (W/m²/°C)', color=color_slope, fontsize=11)
    ax3.tick_params(axis='y', labelcolor=color_slope)

    ax3b = ax3.twinx()
    color_int = 'tab:orange'
    ax3b.bar(months + 0.2, intercepts, width=0.35, color=color_int, alpha=0.7, label='Intercept')
    ax3b.axhline(global_model['intercept'], color=color_int, linestyle='--', alpha=0.5, linewidth=1)
    ax3b.set_ylabel('Intercept (W/m²)', color=color_int, fontsize=11)
    ax3b.tick_params(axis='y', labelcolor=color_int)

    ax3.set_xlabel('Month', fontsize=11)
    ax3.set_xticks(months)
    ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax3.set_title('Monthly Coefficients', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 1d. R² by month
    ax4 = fig.add_subplot(gs[1, 0])

    r2_values = [monthly_model[m]['r_squared'] for m in months]
    colors = plt.cm.RdYlGn(np.array(r2_values))

    bars = ax4.bar(months, r2_values, color=colors, edgecolor='black', linewidth=0.5)
    ax4.axhline(global_model['r_squared'], color='red', linestyle='--', linewidth=1.5,
                label=f"Global R² = {global_model['r_squared']:.3f}")

    ax4.set_xlabel('Month', fontsize=11)
    ax4.set_ylabel('R²', fontsize=11)
    ax4.set_xticks(months)
    ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax4.set_ylim(0, 1)
    ax4.set_title('Monthly Model R²', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 1e. Seasonal cycle
    ax5 = fig.add_subplot(gs[1, 1:])

    doy = get_day_of_year(temp_regional)
    month_arr = doy_to_month(doy)

    # Compute monthly means
    obs_ae = []
    obs_temp = []
    obs_std = []

    for m in range(1, 13):
        m_mask = month_arr == m
        ae_m = ae_regional.isel(time=m_mask).values
        temp_m = temp_regional.isel(time=m_mask).values
        obs_ae.append(np.nanmean(ae_m))
        obs_temp.append(np.nanmean(temp_m))
        obs_std.append(np.nanstd(ae_m))

    # Plot observed
    ax5.errorbar(months, obs_ae, yerr=obs_std, fmt='ko-', linewidth=2, markersize=8,
                 capsize=4, label='Observed (± 1σ)')

    # Plot monthly model prediction
    pred_monthly = [monthly_model[m]['slope'] * obs_temp[m-1] + monthly_model[m]['intercept']
                    for m in range(1, 13)]
    ax5.plot(months, pred_monthly, 'b^--', linewidth=2, markersize=8, label='Monthly Model')

    # Plot global model prediction
    pred_global = [global_model['slope'] * t + global_model['intercept'] for t in obs_temp]
    ax5.plot(months, pred_global, 'rs:', linewidth=2, markersize=8, label='Global Model')

    ax5.set_xlabel('Month', fontsize=11)
    ax5.set_ylabel('Available Energy (W/m²)', fontsize=11)
    ax5.set_xticks(months)
    ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax5.set_title('Seasonal Cycle (Pan Amazon)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Available Energy Model Diagnostics\n'
                 f'NCEP-NCAR Reanalysis 1 (1991-2020 LTM) | Pan Amazon Region',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_dir / 'ae_model_diagnostics.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Saved: {output_dir / 'ae_model_diagnostics.png'}")

    # --- Figure 2: Multivariate Model Comparison ---
    if multivariate_models and len(multivariate_models) > 1:
        create_multivariate_comparison_plot(multivariate_models, output_dir)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> Dict[str, Any]:
    """
    Main workflow for fitting available energy models.

    Steps:
    1. Load NCEP-NCAR Reanalysis data
    2. Align datasets to daily resolution
    3. Calculate available energy
    4. Extract Pan Amazon region
    5. Fit global, monthly, and daily models
    6. Generate outputs (JSON, Fortran module, plots)

    Returns
    -------
    dict
        All model results and metadata
    """
    print("\n" + "=" * 70)
    print("AVAILABLE ENERGY MODEL FITTING FOR CAETE")
    print("=" * 70)
    print(f"Data source: NCEP-NCAR Reanalysis 1 (1991-2020 Long-Term Mean)")
    print(f"Region: Pan Amazon ({PAN_AMAZON_BBOX['south']}°N to {PAN_AMAZON_BBOX['north']}°N, "
          f"{PAN_AMAZON_BBOX['west']}°E to {PAN_AMAZON_BBOX['east']}°E)")

    # Step 1: Load data
    data = load_ncep_data()

    # Step 2: Align to daily resolution
    data = align_to_daily(data)

    # Step 3: Convert temperature to Celsius
    temp_c = convert_temperature_to_celsius(data['air'])

    # Step 4: Calculate available energy
    ae = calculate_available_energy(data['nswrs'], data['nlwrs'], data['gflux'])

    # Step 5: Extract Pan Amazon region
    temp_pana = extract_pan_amazon(temp_c)
    ae_pana = extract_pan_amazon(ae)

    # Extract optional variables for Pan Amazon
    rhum_pana = None
    pres_pana = None

    if 'rhum' in data:
        rhum_pana = extract_pan_amazon(data['rhum'])
        print(f"  RH range in Pan Amazon: [{float(rhum_pana.min()):.1f}, {float(rhum_pana.max()):.1f}]%")

    if 'pres' in data:
        pres_pana = extract_pan_amazon(data['pres'])
        print(f"  P range in Pan Amazon: [{float(pres_pana.min()):.1f}, {float(pres_pana.max()):.1f}]")

    # Step 6: Fit univariate models (T only)
    global_model = fit_global_model(temp_pana, ae_pana)
    monthly_model = fit_monthly_model(temp_pana, ae_pana)
    daily_model = fit_daily_model(temp_pana, ae_pana)

    # Step 7: Fit multivariate models
    multivariate_models = fit_all_multivariate_models(
        temp_pana, ae_pana, rhum_pana, pres_pana
    )

    # Step 8: Print summary
    print_model_summary(global_model, monthly_model)

    # Step 9: Compile results
    results = {
        'metadata': {
            'data_source': 'NCEP-NCAR Reanalysis 1',
            'data_period': '1991-2020 Long-Term Mean',
            'region': 'Pan Amazon',
            'region_bounds': PAN_AMAZON_BBOX,
            'generated_date': pd.Timestamp.now().isoformat(),
            'generator': 'available_energy_model.py'
        },
        'global_model': global_model,
        'monthly_model': {str(k): v for k, v in monthly_model.items()},
        'daily_model': {str(k): v for k, v in daily_model.items()},
        'multivariate_models': multivariate_models,
        'original_caete_model': ORIGINAL_MODEL
    }

    # Step 10: Save outputs
    save_coefficients_json(results, COEFFICIENTS_FILE)
    generate_fortran_module(global_model, monthly_model, daily_model, FORTRAN_FILE, multivariate_models)

    # Step 11: Generate plots
    create_diagnostic_plots(temp_pana, ae_pana, global_model, monthly_model, OUTPUT_DIR, multivariate_models)

    # Final summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nOutput files generated:")
    print(f"  1. Coefficients (JSON): {COEFFICIENTS_FILE}")
    print(f"  2. Fortran module:      {FORTRAN_FILE}")
    print(f"  3. Diagnostic plots:    {OUTPUT_DIR / 'ae_model_diagnostics.png'}")
    if multivariate_models and len(multivariate_models) > 1:
        print(f"  4. Model comparison:    {OUTPUT_DIR / 'ae_multivariate_comparison.png'}")

    # Recommend best multivariate model
    if multivariate_models:
        best_model = max(multivariate_models.items(), key=lambda x: x[1]['r_squared'])
        print(f"\nBest multivariate model: {best_model[0]}")
        print(f"  R² = {best_model[1]['r_squared']:.4f} (vs T-only: {multivariate_models['T_only']['r_squared']:.4f})")
        if best_model[0] != 'T_only':
            improvement = (best_model[1]['r_squared'] - multivariate_models['T_only']['r_squared']) * 100
            print(f"  Improvement: {improvement:+.2f}% in explained variance")

    print("\nTo use in CAETE:")
    print("  1. Copy ae_seasonal_model.f90 to src/")
    print("  2. Add to Makefile sources")
    print("  3. Replace available_energy() calls with ae_seasonal_model functions")

    return results


if __name__ == "__main__":
    results = main()
