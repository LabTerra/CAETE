
# ============================================================
# CAETE - Reading PKZ outputs directly in R
# ============================================================
#
# DESCRIPTION:
# This script demonstrates how to read CAETE .pkz output files
# directly in R using the reticulate package.
#
# CAETE .pkz files are serialized Python objects. Therefore,
# R requires reticulate to access Python's joblib library.
#
# EXAMPLE in this script:
#   - spin35.pkz: Last spinup output file
#   - spin73.pkz: Historical simulation output for 2016
#
# VARIABLES:
#   - GPP (photo): Gross Primary Productivity
#   - NPP (npp): Net Primary Productivity
#   - ET (evapm): Evapotranspiration
#   - LAI (lai): Leaf Area Index
#
# This example reads one PKZ file at a time and plots its daily
# outputs. The same approach can be extended to multiple files
# using a loop, as demonstrated in the 01_extract_daily_and_area_manual.py script.
#
# REQUIREMENTS:
#   R packages: reticulate, ggplot2, tidyr
#   Python package: joblib (and its dependencies)
#
# HOW TO USE:
#   1. Update PYTHON_PATH and OUTPUT_DIR_GRIDCELL below.
#   2. Make sure reticulate uses a Python environment with joblib.
#   3. Run this script in RStudio.
#   4. Change the PKZ filenames to explore other simulation periods.
#
# IMPORTANT:
#   If reticulate has already initialized another Python environment,
#   restart the R session before running this script.
#
# ============================================================


# ====================== LIBRARIES ===========================

library(reticulate)
library(ggplot2)
library(tidyr)

# ====================== USER SETTINGS =======================

# Python interpreter containing the joblib package
# Replace with the path to your own Python environment.
PYTHON_PATH <- "/home/<username>/anaconda3/bin/python"

# Set run name
run_name <- "run_for_caete_ws"

# Set gridcell prefix
gridcell_prefix <- "gridcell186-239"


# ===========================================================

# Folder containing the CSV files (relative to this script)
OUTPUT_DIR_TO_PROCESS = file.path(normalizePath("../../outputs"), run_name)
OUTPUT_DIR_GRIDCELL <- file.path(OUTPUT_DIR_TO_PROCESS, gridcell_prefix)


# Select the files to analyze
SPINUP_FILE <- "spin35.pkz"
HISTORICAL_FILE <- "spin73.pkz"


# ====================== PYTHON CONNECTION ===================

# Connect R to the selected Python environment
use_python(PYTHON_PATH, required = TRUE)

# Import Python's joblib library
joblib <- import("joblib")


# ====================== FUNCTIONS ===========================

# Read a PKZ file and extract selected daily variables
read_caete_pkz <- function(filename) {
  
  file_path <- path.expand(
    file.path(OUTPUT_DIR_GRIDCELL, filename)
  )
  
  # Load the Python object
  data <- joblib$load(file_path)
  
  # Extract selected variables into an R data frame
  df <- data.frame(
    day = seq_along(data[["npp"]]),
    GPP = as.numeric(data[["photo"]]),
    NPP = as.numeric(data[["npp"]]),
    ET  = as.numeric(data[["evapm"]]),
    LAI = as.numeric(data[["lai"]])
  )
  
  return(df)
}


# Plot the daily variables in separate panels
plot_caete <- function(df, plot_title) {
  
  # Convert from wide to long format
  df_long <- pivot_longer(
    df,
    cols = c(GPP, NPP, ET, LAI),
    names_to = "variable",
    values_to = "value"
  )
  
  ggplot(df_long, aes(x = day, y = value)) +
    geom_line(color = "forestgreen", linewidth = 0.5) +
    facet_wrap(~variable, scales = "free_y", ncol = 2) +
    labs(
      title = plot_title,
      x = "Day of simulation",
      y = NULL
    ) +
    theme_minimal()
}


# ====================== SPINUP ==============================

# Read the last spinup output
spinup <- read_caete_pkz(SPINUP_FILE)

# Explore the extracted data
View(spinup)

# Plot the selected variables
plot_caete(
  spinup,
  "CAETE - Last spinup output (spin35)"
)


# ====================== HISTORICAL ==========================

# Read the historical output for 2016
historical <- read_caete_pkz(HISTORICAL_FILE)

# Explore the extracted data
View(historical)

# Plot the selected variables
plot_caete(
  historical,
  "CAETE - Historical simulation (2016)"
)

# ============================================================
# NOTES:
#
# 1. Only four variables are extracted in this example.
#    Additional variables can be included in read_caete_pkz().
#
# 2. Each PKZ contains multiple model outputs. You can inspect
#    their names using:
#
#    names(reticulate::py_to_r(
#      joblib$load(path.expand(file.path(OUTPUT_DIR_GRIDCELL, SPINUP_FILE)))
#    ))
#
# 3. To analyze multiple years, use a loop to read successive
#    spin*.pkz files and combine their daily outputs.
#
# ============================================================