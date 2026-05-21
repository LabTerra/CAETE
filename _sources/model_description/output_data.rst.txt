Output Data
===========

CAETÊ produces daily outputs for all model variables, including those derived from functional traits. Because of the large data volume, results are written in two primary formats, each covering two‑year blocks:

- **Grid‑cell time series**

  *Format:* ``.pkz``

  Each file contains the full daily series for a single grid cell, allowing for point‑wise analyses.

- **Spatial maps**

  *Format:* NetCDF (``.nc``)

  For each variable, one file aggregates two consecutive years of daily data across the entire grid, enabling efficient spatial processing.

This two‑year bundling reduces the overall number of files, optimizes storage requirements, and streamlines access for both temporal and spatial analyses.