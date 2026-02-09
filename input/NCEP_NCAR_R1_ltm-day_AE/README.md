# Available Energy Model for CAETE

This directory contains tools and data for re-estimating the empirical relationship between temperature and available energy used in CAETE's evapotranspiration calculations.

## Background

In CAETE, available energy (AE) for evapotranspiration is approximated using a simple linear model based on air temperature:

$$AE = a \times T + b$$

Where:
- $AE$ is available energy (W/m²)
- $T$ is air temperature (°C)
- $a$ and $b$ are empirically derived coefficients

The **original CAETE model** (in `evap.f90`) uses:
- Slope $a = 2.895$ W/m²/°C
- Intercept $b = 52.326$ W/m²

This script re-estimates these parameters using modern reanalysis data and provides:
- **Seasonal variants** (monthly and daily)
- **Multivariate models** that include relative humidity, vapor pressure deficit (VPD), and surface pressure

## Data Source

**NCEP-NCAR Reanalysis 1** Long-Term Mean (LTM) Daily Data

[https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)

### Required Files

Download the following files from NOAA PSL:

| Variable | File | Description | Period | Resolution |
|----------|------|-------------|--------|------------|
| Air Temperature | `air.2m.gauss.day.ltm.1991-2020.nc` | 2m air temperature | 1991-2020 | Daily, ~1.9° |
| Net Shortwave | `nswrs.sfc.gauss.day.ltm.1991-2020.nc` | Net shortwave radiation | 1991-2020 | Daily, ~1.9° |
| Net Longwave | `nlwrs.sfc.gauss.day.ltm.1991-2020.nc` | Net longwave radiation | 1991-2020 | Daily, ~1.9° |
| Ground Heat Flux | `gflux.sfc.gauss.day.ltm.1991-2020.nc` | Ground heat flux | 1991-2020 | Daily, ~1.9° |
| Relative Humidity | `rhum.sig995.day.ltm.1991-2020.nc` | Near-surface humidity | 1991-2020 | Daily, ~2.5° |
| Surface Pressure | `pres.sfc.day.ltm.1991-2020.nc` | Surface pressure | 1991-2020 | Daily, ~2.5° |

**Note:** NCEP-NCAR data uses longitude in 0-360° format (e.g., -80°E = 280°).

## Available Energy Calculation

Available energy is the net radiation minus ground heat flux:

$$AE = R_n - G = SW_{absorbed} - LW_{lost} - G_{into\_ground}$$

### NCEP-NCAR Sign Conventions

**Important:** NCEP-NCAR uses specific sign conventions that must be accounted for:

| Variable | Sign Convention | Typical Range (Pan Amazon) |
|----------|----------------|---------------------------|
| `nswrs` | **Negative** = absorbed by surface | -337 to +1 W/m² |
| `nlwrs` | **Positive** = leaving surface (upward) | +1 to +168 W/m² |
| `gflux` | **Positive** = into ground | -43 to +173 W/m² |

Therefore, the formula using NCEP data is:

$$AE = (-nswrs) - nlwrs - gflux$$

## Models Fitted

The script fits two types of models: **univariate (temperature-only)** and **multivariate** (temperature plus humidity variables).

### Univariate Models (Temperature Only)

#### 1. Global Linear Model
Single set of coefficients for the entire region and year:
```
AE = slope × T + intercept
```

#### 2. Monthly Seasonal Model
12 sets of coefficients, one per month:
```
AE_m = slope_m × T + intercept_m    (m = 1, ..., 12)
```

#### 3. Daily Seasonal Model
365 sets of coefficients, one per day of year:
```
AE_doy = slope_doy × T + intercept_doy    (doy = 1, ..., 365)
```

### Multivariate Models

The script also fits multivariate models that include additional predictors to improve the explained variance. These models use:

| Variable | Symbol | Description | Units |
|----------|--------|-------------|-------|
| Temperature | T | 2m air temperature | °C |
| Relative Humidity | RH | Near-surface relative humidity | % |
| Vapor Pressure Deficit | VPD | Saturation deficit | kPa |
| Surface Pressure | P | Atmospheric pressure at surface | Pa |

**VPD Calculation:**

$$VPD = e_s(T) \times (1 - RH/100)$$

Where saturation vapor pressure $e_s(T)$ is calculated using the Buck equation:

$$e_s(T) = 0.61121 \times \exp\left(\frac{17.502 \times T}{240.97 + T}\right) \quad \text{[kPa]}$$

#### Models Compared

| Model Name | Equation |
|------------|----------|
| T_only | $AE = a \times T + b$ |
| T_RH | $AE = a \times T + c \times RH + b$ |
| T_VPD | $AE = a \times T + d \times VPD + b$ |
| VPD_only | $AE = d \times VPD + b$ |
| T_RH_P | $AE = a \times T + c \times RH + e \times P + b$ |
| T_VPD_P | $AE = a \times T + d \times VPD + e \times P + b$ |

The script automatically selects the best performing model based on R² and generates the corresponding Fortran function.

## Usage

### Prerequisites

```bash
pip install numpy xarray scipy matplotlib pandas netcdf4 scikit-learn
```

### Running the Script

```bash
cd CAETE/input/NCEP_NCAR_R1_ltm-day_AE
python available_energy_model.py
```

### Outputs

The script generates:

1. **JSON Coefficients** (`model_output/ae_model_coefficients.json`)
   - All model parameters and statistics
   - Univariate and multivariate model coefficients
   - Metadata about data source and region

2. **Fortran Module** (`model_output/ae_seasonal_model.f90`)
   - Ready-to-use Fortran 90 module for CAETE
   - Functions for univariate models: `available_energy_global()`, `available_energy_monthly()`, `available_energy_daily()`
   - Function for best multivariate model (e.g., `available_energy_t_vpd()`)

3. **Diagnostic Plots** (`model_output/ae_model_diagnostics.png`)
   - Scatter plot with regression lines
   - Residuals distribution
   - Monthly coefficient variation
   - Seasonal cycle comparison

4. **Model Comparison Plot** (`model_output/ae_multivariate_comparison.png`)
   - R² comparison across all models
   - RMSE comparison
   - R² improvement over T-only baseline
   - Best model coefficients summary

## Pan Amazon Region

The model is fitted specifically for the Pan Amazon region:

| Bound | Value |
|-------|-------|
| North | 10.5°N |
| South | -21.5°N |
| West | -80.0°E |
| East | -43.0°E |

This corresponds to the CAETE study domain defined by the forest mask from MapBiomass.

## Integration with CAETE

### Option 1: Use univariate model (temperature only)

In `src/evap.f90`, replace:

```fortran
function available_energy(temp) result(ae)
    use types, only: r_8
    real(r_8),intent(in) :: temp
    real(r_8) :: ae
    ae = 2.895 * temp + 52.326
end function available_energy
```

With a call to the new module:

```fortran
use ae_seasonal_model, only: available_energy_daily
! ... in your code:
ae = available_energy_daily(temp, day_of_year)
```

### Option 2: Use multivariate model (temperature + VPD)

If you have VPD available (calculated from temperature and relative humidity):

```fortran
use ae_seasonal_model, only: available_energy_t_vpd
! ... in your code:
ae = available_energy_t_vpd(temp, vpd)
```

### Option 3: Update coefficients only

Simply update the coefficients in `evap.f90` with the new global model values.

## File Structure

```
NCEP_NCAR_R1_ltm-day_AE/
├── README.md                                      # This file
├── available_energy_model.py                      # Main script
├── air.2m.gauss.day.ltm.1991-2020.nc             # Input: temperature
├── nswrs.sfc.gauss.day.ltm.1991-2020.nc          # Input: SW radiation
├── nlwrs.sfc.gauss.day.ltm.1991-2020.nc          # Input: LW radiation
├── gflux.sfc.gauss.day.ltm.1991-2020.nc          # Input: ground heat flux
├── rhum.sig995.day.ltm.1981-2010.nc              # Input: relative humidity (optional)
├── pres.sfc.day.ltm.1981-2010.nc                 # Input: surface pressure (optional)
└── model_output/
    ├── ae_model_coefficients.json                # Fitted coefficients (JSON)
    ├── ae_seasonal_model.f90                     # Fortran module
    ├── ae_model_diagnostics.png                  # Univariate diagnostic plots
    └── ae_multivariate_comparison.png            # Multivariate model comparison
```

## References

1. Kalnay, E., et al. (1996). The NCEP/NCAR 40-Year Reanalysis Project. *Bulletin of the American Meteorological Society*, 77(3), 437-472.

2. NOAA Physical Sciences Laboratory. NCEP-NCAR Reanalysis 1. https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

## License

This script is part of the CAETE model and is distributed under the GNU General Public License v3.0.

## Author

João Paulo Darela Filho - LabTerra, UNICAMP
February 2026

