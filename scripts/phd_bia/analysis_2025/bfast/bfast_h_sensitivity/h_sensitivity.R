library(bfast)
library(zoo)
library(dplyr)
library(purrr)
library(tidyr)
library(ggplotify)
library(patchwork)
library(scales)  # For rescale()
library(ggplot2)

path_file <- "/Users/biancarius/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/monthly_mean_tables/"
df_regclim <- read.csv(file.path(path_file, "MAN_regularclimate_monthly.csv"))
df_1y      <- read.csv(file.path(path_file, "MAN_30prec_1y_monthly.csv"))
#df_3y      <- read.csv(file.path(path_file, "MAN_30prec_3y_monthly.csv"))
#df_5y      <- read.csv(file.path(path_file, "MAN_30prec_5y_monthly.csv"))
df_7y      <- read.csv(file.path(path_file, "MAN_30prec_7y_monthly.csv"))


# List of all data frames you want to convert into time series
df_list <- list(
  regclim = df_regclim,
  y1      = df_1y,
#  y3      = df_3y,
#  y5      = df_5y,
  y7      = df_7y
)

# Function that converts all columns (except the first one) of a dataframe into time series objects
to_ts_list <- function(df) {
  # Create a time series object for each column except the first (assumed to be time/year)
  lapply(df[, -1], function(col) {
    ts(
      col,
      start = c(1979, 1),  # Beginning of the time series
      frequency = 12       # Monthly data
    )
  })
}

# treating data from the 1y frequency because the system collapse. We have to analyse before it
collapse_date <- "2006-12"
# filter data (INCLUDE the collapse month)
df_1y_precollapse <- df_1y %>%filter(date <= collapse_date)

df_list <- list(
  regclim = df_regclim,
  y1      = df_1y_precollapse,  # pre-collapse version
  #y3      = df_3y,
  #y5      = df_5y,
  y7      = df_7y
)

# Apply the function to every dataframe in the list
ts_lists <- lapply(df_list, to_ts_list)

columns_to_process <- c("npp", "ctotal","evapm","wue")

# Sequence of h values to test (change as you like)
h_values <- seq(0.05, 0.20, by = 0.05)

# Function to run BFAST sensitivity analysis for one data frame (one climate scenario)
run_bfast_sensitivity <- function(df,
                                  scenario_name,
                                  columns_to_process,
                                  h_values,
                                  start_year = 1979,
                                  freq = 12) {
  
  # Loop over each variable of interest
  map_dfr(columns_to_process, function(var_name) {
    
    # Extract the numeric vector for the current variable
    x <- df[[var_name]]
    
    # Loop over all h values
    map_dfr(h_values, function(h) {
      
      # Create time series object from the variable
      ts_x <- ts(
        x,
        start     = c(start_year, 1),  # first year and first month
        frequency = freq               # monthly data
      )
      
      # Run BFAST inside try() to avoid stopping if one run fails
      fit <- try(
        bfast(ts_x, h = h, season = "harmonic", max.iter = 1),
        silent = TRUE
      )
      
      # If BFAST failed, return NA values for this combination
      if (inherits(fit, "try-error")) {
        return(tibble(
          scenario                 = scenario_name,
          variable                 = var_name,
          h                        = h,
          n_breaks_trend           = NA_integer_,
          first_break_trend_time   = NA_real_,
          biggest_magnitude        = NA_real_,
          biggest_magnitude_time   = NA_real_
        ))
      }
      
      # Extract trend breakpoints (Vt = deseasonalized trend component)
      bp_idx <- fit$output[[1]]$bp.Vt$breakpoints
      
      # If no breakpoints are detected
      if (all(is.na(bp_idx))) {
        n_breaks_trend   <- 0L
        first_break_time <- NA_real_
      } else {
        n_breaks_trend   <- length(bp_idx)
        first_break_time <- time(ts_x)[bp_idx[1]]
      }
      
      # Build summary row for this (scenario, variable, h)
      tibble(
        scenario               = scenario_name,
        variable               = var_name,
        h                      = h,
        n_breaks_trend         = n_breaks_trend,
        first_break_trend_time = first_break_time,
        # Overall BFAST summary: magnitude and timing of biggest change in trend
        biggest_magnitude      = fit$Magnitude,
        biggest_magnitude_time = fit$Time
      )
    })
  })
}


# Apply the sensitivity function to every climate scenario
sensitivity_results <- map2_dfr(
  df_list,
  names(df_list),
  ~ run_bfast_sensitivity(
    df                = .x,
    scenario_name     = .y,
    columns_to_process = columns_to_process,
    h_values          = h_values
  )
)

# Take a quick look at the results
sensitivity_results


write.csv(
  sensitivity_results,
  file = "~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/analysis_2025/bfast/bfast_h_sensitivity/sensitivity_results.csv",
  row.names = FALSE
)


# Plotting and saving

# plots_dir <- "/Users/biancarius/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/analysis_2025/bfast_h_sensitivity/bfast_plots/"
# 
# dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
# 
# library(bfast)
# 
# plot_bfast_for_h <- function(df, scenario_name, var_name, h_values, plots_dir, start_year = 1979) {
#   
#   # Extract the numeric series
#   x <- df[[var_name]]
#   
#   # Create time series
#   ts_x <- ts(
#     x,
#     start = c(start_year, 1),
#     frequency = 12
#   )
#   
#   # Loop over each h
#   for (h in h_values) {
#     
#     # Try running BFAST (skip if it fails)
#     fit <- try(
#       bfast(ts_x, h = h, season = "harmonic", max.iter = 1),
#       silent = TRUE
#     )
#     
#     if (inherits(fit, "try-error")) {
#       message("Skipping: ", scenario_name, " - ", var_name, " - h=", h, " (BFAST error)")
#       next
#     }
#     
#     # Build filename (clean and consistent)
#     file_name <- sprintf("%s/%s_%s_h%.2f.png",
#                          plots_dir, scenario_name, var_name, h)
#     
#     # Save plot as PNG
#     png(file_name, width = 1400, height = 900)
#     plot(fit, main = paste0("BFAST - ", scenario_name,
#                             " - ", var_name,
#                             " - h = ", h))
#     dev.off()
#   }
# }
# 
# for (scenario in names(df_list)) {
#   for (var_name in columns_to_process) {
#     
#     message("Processing: ", scenario, " - ", var_name)
#     
#     plot_bfast_for_h(
#       df = df_list[[scenario]],
#       scenario_name = scenario,
#       var_name = var_name,
#       h_values = h_values,
#       plots_dir = plots_dir
#     )
#   }
# }


