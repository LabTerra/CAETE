# Input data for CAETÊ
The pre_processing.py file is used to prepare files that are employed to feed the CAETÊ model. Check the [creating_caete_input_files.md](creating_caete_input_files.md) file for detailed instructions on how to use the script and prepare the input files. Part of these data is provided in this repository in the format specified in the documentation and ready to be used as input for the model. You can find these example input files in folders scattered across the ```/input``` directory, such as ```/input/south```, ```/input/nw```, etc. These files are provided as examples to illustrate the expected format and structure of the input data. They are not intended to be used directly as input for the model, but rather to serve as templates for preparing your own input data.

The raw climatic and edaphic data used in these files are publicly available from other sources.

## References

### ISIMIP climate input
Stefan Lange, Matthias Büchner (2020): ISIMIP2a atmospheric climate input data (v1.0). ISIMIP Repository. https://doi.org/10.48364/ISIMIP.886955

Stefan Lange, Dánnell Quesada-Chacón, Matthias Mengel, Simon Treu, Matthias Büchner (2025): ISIMIP3a atmospheric climate input data (v1.3). ISIMIP Repository. https://doi.org/10.48364/ISIMIP.982724.3

The raw input climatic data was downloaded from the [ISIMIP REPOSITORY](https://data.isimip.org/).

### Soil data

Wieder, W.R., J. Boehnert, G.B. Bonan, and M. Langseth. 2014. Regridded Harmonized World Soil Database v1.2. Data set. Available on-line \[[http://daac.ornl.gov](http://daac.ornl.gov)\] from Oak Ridge National Laboratory Distributed Active Archive Center, Oak Ridge, Tennessee, USA. [http://dx.doi.org/10.3334/ORNLDAAC/1247](http://dx.doi.org/10.3334/ORNLDAAC/1247)

Poggio, L., L. M. de Sousa, N. H. Batjes, G. B. M. Heuvelink, B. Kempen, E. Ribeiro, and D. Rossiter. "Soilgrids 2.0: Producing Soil Information for the Globe with Quantified Spatial Uncertainty." SOIL 7, no. 1 (2021): 217-40. [https://doi.org/10.5194/soil-7-217-2021](https://doi.org/10.5194/soil-7-217-2021)

Darela-Filho, João Paulo, Anja Rammig, Katrin Fleischer, Tatiana Reichert, Laynara Figueiredo Lugli, Carlos Alberto Quesada, Luis Carlos Colocho Hurtarte, Mateus Dantas de Paula, and David M. Lapola. "Reference Maps of Soil Phosphorus for the Pan-Amazon Region." Earth System Science Data 16, no. 1 (2024): 715-29. [https://doi.org/10.5194/essd-16-715-2024](https://doi.org/10.5194/essd-16-715-2024).