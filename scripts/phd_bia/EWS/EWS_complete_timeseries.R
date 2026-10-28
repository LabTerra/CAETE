library(earlywarnings)
library(ggplot2)
library(dplyr)
library(tidyr)

output_path = "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/csv_allvar/"
#----------------------------------------------

#               MANAUS

#----------------------------------------------

#investigating EWS during the whole time series


#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 1 year
#----------------------------------------------

# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")
ews_results <- generic_ews(df_1y$npp,
                           winsize = 15, detrending = 'loess',
                           logtransform = FALSE, interpolate = FALSE,
                           AR_n = FALSE, powerspectrum = FALSE)
# Specify the columns to process
columns_to_process <- names(df_1y)[-1]

# Loop through each column and apply generic early warning signals
for (col_name in columns_to_process) {
  ews_results <- generic_ews(df_1y[[col_name]],
                             winsize = 15, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE,
                             AR_n = FALSE, powerspectrum = FALSE)

  # Create a new data frame with the results
  result_df <- data.frame(ews_results, var = col_name)
  result_df$frequency = "1y"
  
  # Save the result to a separate CSV file for each variable
  write.csv(result_df, file = paste0(output_path, "ews_results_1y_", col_name, ".csv"), row.names = FALSE)

  # Save the plot with the complement of the column name and specified path
  # png(paste0(output_path, "ews_plot_", col_name, ".png"))
  plot(ews_results, main = col_name)
  dev.off()
}


result <- read.csv('/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/csv_allvar/ews_results_1y_ar.csv')



#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 3 year
#----------------------------------------------

# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_3y_monthly.csv")

# Specify the columns to process
columns_to_process <- names(df_3y)[-1]

# Loop through each column and apply generic early warning signals
for (col_name in columns_to_process) {
  ews_results <- generic_ews(df_3y[[col_name]], 
                             winsize = 15, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE, 
                             AR_n = FALSE, powerspectrum = FALSE)
  
  # Create a new data frame with the results
  result_df <- data.frame(ews_results, var = col_name)
  result_df$frequency = "3y"
  # Save the result to a separate CSV file for each variable
  write.csv(result_df, file = paste0(output_path, "ews_results_3y_", col_name, ".csv"), row.names = FALSE)
  
  # Save the plot with the complement of the column name and specified path
  # png(paste0(output_path, "ews_plot_", col_name, ".png"))
  # plot(ews_results, main = col_name)
  dev.off()
}


# 
# write.csv(df_1y_npp_gws, file =
#             "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_3y_timeseries_ews.csv", row.names = FALSE)

#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 5 year
#----------------------------------------------


# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_5y_monthly.csv")

# Specify the columns to process
columns_to_process <- names(df_5y)[-1]

# Loop through each column and apply generic early warning signals
for (col_name in columns_to_process) {
  ews_results <- generic_ews(df_5y[[col_name]], 
                             winsize = 15, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE, 
                             AR_n = FALSE, powerspectrum = FALSE)
  
  # Create a new data frame with the results
  result_df <- data.frame(ews_results, var = col_name)
  result_df$frequency = "5y"
  # Save the result to a separate CSV file for each variable
  write.csv(result_df, file = paste0(output_path, "ews_results_5y_", col_name, ".csv"), row.names = FALSE)
  
  # Save the plot with the complement of the column name and specified path
  # png(paste0(output_path, "ews_plot_", col_name, ".png"))
  # plot(ews_results, main = col_name)
  dev.off()
}

#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 7 year
#----------------------------------------------

# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")

# Specify the columns to process
columns_to_process <- names(df_7y)[-1]

# Loop through each column and apply generic early warning signals
for (col_name in columns_to_process) {
  ews_results <- generic_ews(df_7y[[col_name]], 
                             winsize = 15, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE, 
                             AR_n = FALSE, powerspectrum = FALSE)
  
  # Create a new data frame with the results
  result_df <- data.frame(ews_results, var = col_name)
  result_df$frequency = "7y"
  # Save the result to a separate CSV file for each variable
  write.csv(result_df, file = paste0(output_path, "ews_results_7y_", col_name, ".csv"), row.names = FALSE)
  
  # Save the plot with the complement of the column name and specified path
  # png(paste0(output_path, "ews_plot_", col_name, ".png"))
  # plot(ews_results, main = col_name)
  dev.off()
}



#----------------------------------------------
#   precipitation reduction: regular climate
#   frequency: NA
#----------------------------------------------

# # !!!!! note this is the monthly integrated data frame!!!!!!!

# Read CSV file
df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")

# Specify the columns to process
# columns_to_process <- names(df_regclim)[-1]

ews_results <- generic_ews(df_1y[[col_name]],
                             winsize = 15, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE,
                             AR_n = FALSE, powerspectrum = FALSE)

ews_results <- generic_ews(df_1y$npp,
                           winsize = 15, detrending = 'loess',
                           logtransform = FALSE, interpolate = FALSE,
                           AR_n = FALSE, powerspectrum = FALSE)

ews_results <- generic_ews(df_1y$ctotal,
                           winsize = 15, detrending = 'loess',
                           logtransform = FALSE, interpolate = FALSE,
                           AR_n = FALSE, powerspectrum = FALSE)

ews_results <- generic_ews(df_1y$evapm,
                           winsize = 15, detrending = 'loess',
                           logtransform = FALSE, interpolate = FALSE,
                           AR_n = FALSE, powerspectrum = FALSE)

ews_results <- generic_ews(df_1y$wue,
                           winsize = 15, detrending = 'loess',
                           logtransform = FALSE, interpolate = FALSE,
                           AR_n = FALSE, powerspectrum = FALSE)

# Loop through each column and apply generic early warning signals
for (col_name in columns_to_process) {
  ews_results <- generic_ews(df_regclim[[col_name]], 
                             winsize = 15, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE, 
                             AR_n = FALSE, powerspectrum = FALSE)
  
  # Create a new data frame with the results
  # result_df <- data.frame(ews_results, var = col_name)
  
  # result_df$frequency = "regclim"
  # Save the result to a separate CSV file for each variable
  # write.csv(result_df, file = paste0(output_path, "ews_results_regclim_", col_name, ".csv"), row.names = FALSE)
  
  # Save the plot with the complement of the column name and specified path
  # png(paste0(output_path, "ews_plot_", col_name, ".png"))
  plot(ews_results, main = col_name)
  # dev.off()
}

