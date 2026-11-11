# ============================================================
# CAETE - Explore daily and area csvs from output 
#         ...'grids_pkzs_to_csvs' folder
# ============================================================


# ====================== USER INPUT ==============================


# Set run name
run_name <- "run_for_caete_ws"

# Set gridcell foldername
gridcell_foldername <- "gridcell186-239"




# ====================== FIXED SETTINGS =======================

# Folder containing the CSV files (relative to this script)
OUTPUT_DIR_TO_PROCESS = file.path(normalizePath("../../outputs"), run_name)
OUTPUT_DIR_GRIDS_PKZS_TO_CSVS <- file.path(OUTPUT_DIR_TO_PROCESS, "grids_pkzs_to_csvs")

# ====================== IMPORT DATA =========================

# PLS functional traits
pls_table_filepath <- list.files(OUTPUT_DIR_TO_PROCESS, pattern = "^pls_attrs.*\\.csv$", full.names = TRUE, recursive = TRUE)[1]
print(paste0("Loading PLS table from ", pls_table_filepath, "..."))
pls_attrs <- read.csv(pls_table_filepath)

# Daily ecosystem-level outputs
daily_csv_filepath <- file.path(OUTPUT_DIR_GRIDS_PKZS_TO_CSVS, paste0(gridcell_foldername, "_daily.csv"))
print(paste0("Loading daily.csv csv file from ", daily_csv_filepath, "..."))
daily <- read.csv(daily_csv_filepath)

# Daily PLS occupation
area_csv_filepath <- file.path(OUTPUT_DIR_GRIDS_PKZS_TO_CSVS, paste0(gridcell_foldername, "_area.csv"))
print(paste0("Loading area.csv csv file from ", area_csv_filepath, "..."))
area <- read.csv(area_csv_filepath)


# ====================== EXPLORE DATA ========================

names(pls_attrs)
names(daily)
names(area)

dim(pls_attrs)
dim(daily)
dim(area)

View(pls_attrs)
View(daily)
View(area)
