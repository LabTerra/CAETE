Code Structure (TODO)
=====================

TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR
./TEMP_caete_architecture.md
TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR



- The code is divided in two main parts:
    - Python interface for data handling and configuration
    - Fortran backend for numerical computations
- Requires compilation of Python extension modules
- Cross-platform support (Windows, Linux, macOS)


Python source code
------------------

- ``region.py``: handles the execution over gridcells
- ``caete.py:`` Definition of the gridcell (grd_mt class)
- ``metacommunity.py:`` metacommunity class, contains multiple communities
- ``community.py:`` community class, plant biomass is handled here
- ``other scripts:`` utility functions, input handling, preprocessing, jit compiled routines ...


Fortran source code
-------------------

- Core Library Files (src_lib)
    - ``types.f90``: numerical type definitions
    - ``global.f90:`` Global parameters and constants
    - ``photo_par.f90:`` Photosynthesis parameters
    - ``funcs.f90:`` GPP related functions and utilities
    - ``evap.f90:`` Evapotranspiration calculations
    - ``soil_dec.f90:`` Soil decomposition processes
    - ``cc.f90:`` Carbon costs calculations
    - ``allocation.f90:`` Resource allocation algorithms
    - ``productivity.f90:`` Primary Productivity calculations

- Budget Module Options: Daily Time Step Budget Calculation for all PLSs
    - ``budget_fixed.F90:`` Stack allocation (faster, default)
    - ``budget.F90:`` Heap allocation (slower, larger PLS support)

- Makefiles:
    - ``Makefile:`` For Linux and macOS builds
    - ``Makefile_win:`` For Windows builds (MSVC + OneAPI)