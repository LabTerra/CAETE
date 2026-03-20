library(earlywarnings)
library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(zoo)
library(tidyverse)
library(readr)

## TESTING SLIDING WINDOW SIZE ##


# Function to perform common operations
process_data <- function(file_path, output_csv_path, freq) {
  df <- read.csv(file_path)
  df_npp <- df$Monthly_NPP_Mean
  
  winsize <- seq(15, 35, by = 5)
  
  results_list <- lapply(winsize, function(ws) {
    gws <- generic_ews(df_npp, winsize = ws, detrending = 'loess',
                       logtransform = FALSE, interpolate = FALSE, 
                       AR_n = FALSE, powerspectrum = FALSE)
    dev.off()
    gws$ws <- ws
    return(gws)
  })
  
  results_list <- lapply(results_list, function(result) {
    if (result$timeindex[1] != 1) {
      additional_rows <- data.frame(
        timeindex = seq(1, result$timeindex[1] - 1),
        ar1 = rep(NA, result$timeindex[1] - 1),
        ws = rep(result$ws[1], result$timeindex[1] - 1)
      )
      return(bind_rows(additional_rows, result))
    } else {
      return(result)
    }
  })
  
  
  final_df <- bind_rows(results_list)
  
  if (freq == "1"){
    tau_values <- c('15' = 0.434, '20' = 0.413, '25' = 0.291, 
                    '30' = 0.160, '35' = 0.015)
    title <- "Freq: 1 year | Prec. red: 30% | Location: Manaus "
    
  } else if (freq == "3"){
  tau_values <- c('15' = 0.534, '20' = 0.623, '25' = 0.613, 
                  '30' = 0.662, '35' = 0.711)
  title <- "Freq: 3 years | Prec. red: 30% | Location: Manaus "
  
  
  } else if (freq == "5"){
    tau_values <- c('15' = 0.721, '20' = 0.777, '25' = 0.839, 
                    '30' = 0.904, '35' = 0.912)
    title <- "Freq: 5 years | Prec. red: 30% | Location: Manaus "
    
    
  } else if (freq == "7"){
    tau_values <- c('15' = 0.774, '20' = 0.788, '25' = 0.842, 
                    '30' = 0.376, '35' = 0.501)
    title <- "Freq: 7 years | Prec. red: 30% | Location: Manaus "
    
  } else if (freq == "regclim"){
    tau_values <- c('15' = 0.037, '20' = 0.102, '25' = 0.242, 
                    '30' = 0.376, '35' = 0.501)
    title <- "Freq: NA | Prec. red: reg. clim. | Location: Manaus "
    
  }
  
  # Map values and handle NaN by using a default value
  final_df$tau <- sapply(final_df$ws, 
                         function(x) ifelse(is.na(tau_values[as.character(x)]), 
                         NA, tau_values[as.character(x)]))
  
  # Create a new column for legend labels
  final_df$legend_label <- paste("ws =", final_df$ws, ", tau =", final_df$tau)
  
  # Save final_df to a CSV file
  write.csv(final_df, file = output_csv_path, row.names = FALSE)
  
  final_df
}

### Plot function
ploting <- function(output_csv_path, freq) {
  file = read.csv(output_csv_path)
  if (freq == "1"){
    title <- "Freq: 1 year | Prec. red: 30% | Location: Manaus "
    #put vertical lines in the brakpoints
    xintercept = c(114,242)
    
  } else if (freq == "3"){
    title <- "Freq: 3 years | Prec. red: 30% | Location: Manaus "
    xintercept = c(155,325)
    
    
  } else if (freq == "5"){
    title <- "Freq: 5 years | Prec. red: 30% | Location: Manaus "
    xintercept = c(225)
    
    
  } else if (freq == "7"){
    title <- "Freq: 7 years | Prec. red: 30% | Location: Manaus "
    xintercept = c(227,341)
    
  } else if (freq == "regclim"){
    title <- "Freq: NA | Prec. red: reg. clim. | Location: Manaus "
    xintercept = c(234)
    
  }
  
  # Plotting
  print(ggplot(file, aes(x = timeindex, y = ar1, color = file$legend_label)) +
          geom_line() +
          scale_color_manual(values = 
                               viridis::viridis_pal()(length(unique(file$ws)))) +
          labs(title = title,
               x = "Timeindex",
               y = "AR1") +
          theme_minimal() +
          guides(color = guide_legend(override.aes = list(shape = NA))) +
          scale_linetype_manual(values = rep("solid", 
                                             length(unique(file$ws)))) +
          scale_shape_manual(values = rep(16, length(unique(file$ws)))) +
          guides(linetype = guide_legend(override.aes = list(color = "black", 
                                                             shape = NA)),
                 shape = guide_legend(override.aes = list(color = "black", 
                                                          linetype = "solid"))) +
          theme(legend.text = element_text(color = "black"),
                legend.position = "top",
                legend.title = element_blank(),  # Remove legend title
                legend.box.background = element_rect(colour = "black"))+
          geom_vline(xintercept = xintercept, linetype = "dashed", color = "grey"))
}


# applying
file_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_1y_monthly.csv"
output_csv_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/window_size_sensitivity/window_size_sens_1y.csv"
result_df_1y <- process_data(file_path, output_csv_path,"1")
plot_result <- ploting(output_csv_path, "1")

file_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_3y_monthly.csv"
output_csv_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/window_size_sensitivity/window_size_sens_3y.csv"
result_df_3y <- process_data(file_path, output_csv_path, "3")
plot_result <- ploting(output_csv_path, "3")


file_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_5y_monthly.csv"
output_csv_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/window_size_sensitivity/window_size_sens_5y.csv"
result_df_5y <- process_data(file_path, output_csv_path, "5")
plot_result <- ploting(output_csv_path, "5")


file_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_7y_monthly.csv"
output_csv_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/window_size_sensitivity/window_size_sens_7y.csv"
result_df_7y <- process_data(file_path, output_csv_path, "7")
plot_result <- ploting(output_csv_path, "7")


file_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_regularclimate_monthly.csv"
output_csv_path <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/window_size_sensitivity/window_size_sens_regclim.csv"
result_df_regclim <- process_data(file_path, output_csv_path, "regclim")
plot_result <- ploting(output_csv_path, "regclim")



