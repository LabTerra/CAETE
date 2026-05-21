Input Data
==========

Climatic Data
-------------

The following climatic data were used as inputs for CAETÊ:

1. Daily data of temperature at surface, precipitation, incoming shortwave radiation, atmospheric pressure, relative humidity and atmospheric CO\ :sub:`2`\  concentration, for the period 1979 to 2016, obtained from the Inter-Sectoral Impact Model Intercomparison Project 2 (ISIMIP2).

To initialize inorganic nitrogen pools, data from the International Soil Reference and Information Centre (ISRIC) and Darela-Filho et al. (2024) were employed. Summary of these climatic variables are given in (:ref:`tab:climate_inputs`). All input data are provided at a 0.5°× 0.5° spatial resolution (WGS‑84), matching the grid‑cell resolution used in CAETÊ.

.. list-table:: Climate inputs used by the model: symbols, descriptions, and units.
   :header-rows: 1
   :widths: 20 60 20
   :name: tab:climate_inputs
   :class: wrap-table

   * - **Symbol**
     - **Description**
     - **Unit**
   * - :math:`T`
     - Surface air temperature
     - °C
   * - :math:`Prec`
     - Precipitation
     - mm day\ :sup:`-1`
   * - :math:`Pa`
     - Atmospheric pressure
     - hPa
   * - :math:`RS_{\text{in}}`
     - Incoming shortwave radiation
     - W m\ :sup:`-2`
   * - :math:`RH`
     - Relative humidity
     - %
   * - :math:`CO_{2}`
     - Atmospheric CO\ :sub:`2` concentration
     - ppm


Soil data
---------

- Source: HWSD and IGBP.


CAETÊ ``input/`` folder
-----------------------

The ``input/`` folder contains all the input data used to run the CAETE model. These data are divided into folders:

``mask/``, ``co2``, ``hydra/`` and ``soil/`` folders: Contains some boolean masks used to preprocess input data and configure model execution. There are also files with soil hydraulic parameters and nutrient content (N & P). The co2 folder has timeseries of annual ATM CO2 concentration. Observed and projected.

``central/``, ``east/``, ``north_west`` and ``south/`` folders: Contains ``.pbz2`` files, each one containing data for one grid.


Pre-processing
--------------

The ``pre_processing.py`` file is used to prepare files that are used as input for CAETÊ model. The raw climatic and edaphic data in these files are publicly available from other sources.


References
----------

ISIMIP climate input
~~~~~~~~~~~~~~~~~~~~

- Weedon, G. P., Balsamo, G., Bellouin, N., Gomes, S., Best, M. J., & Viterbo, P. (2014). The WFDEI meteorological forcing data set: WATCH Forcing Data methodology applied to ERA-Interim reanalysis data. Water Resources Research, 50(9), 7505–7514. https://doi.org/10.1002/2014WR015638

- Lange, Stefan (2019): EartH2Observe, WFDEI and ERA-Interim data Merged and Bias-corrected for ISIMIP (EWEMBI). V. 1.1. GFZ Data Services. https://doi.org/10.5880/pik.2019.004

- Lange, S. & Büchner, M. (2020). ISIMIP2a atmospheric climate input data. ISIMIP Repository. https://doi.org/10.48364/ISIMIP.886955

The raw input climatic data was downloaded from the ISIMIP REPOSITORY.


Soil data
~~~~~~~~~

- Wieder, W.R., J. Boehnert, G.B. Bonan, and M. Langseth. 2014. Regridded Harmonized World Soil Database v1.2. Data set. Available on-line [http://daac.ornl.gov] from Oak Ridge National Laboratory Distributed Active Archive Center, Oak Ridge, Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1247

- Poggio, L., L. M. de Sousa, N. H. Batjes, G. B. M. Heuvelink, B. Kempen, E. Ribeiro, and D. Rossiter. "Soilgrids 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty." SOIL 7, no. 1 (2021): 217-40. https://doi.org/10.5194/soil-7-217-2021

- Darela-Filho, João Paulo, Anja Rammig, Katrin Fleischer, Tatiana Reichert, Laynara Figueiredo Lugli, Carlos Alberto Quesada, Luis Carlos Colocho Hurtarte, Mateus Dantas de Paula, and David M. Lapola. "Reference Maps of Soil Phosphorus for the Pan-Amazon Region." Earth System Science Data 16, no. 1 (2024): 715-29. https://doi.org/10.5194/essd-16-715-2024.
