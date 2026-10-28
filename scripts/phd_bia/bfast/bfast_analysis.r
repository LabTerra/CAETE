library(bfast)
library(zoo)

#output path to save the plots of ecosystem functions
output_path_plots <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/plot_anova_trend/"

# columns_to_process <- c("npp", "photo", "ar","lai","f5","evapm", "cleaf", 
#                         "croot", "cwood", "cheart", "csap", "csto", "ctotal","wue")

columns_to_process <- c("npp", "ctotal","evapm","wue")

# ################################
# #-------------------------------
# #           Manaus
# #-------------------------------
# ################################
# 
# #-------------------------------
# #        regular climate
# #-------------------------------
# 
# # !!!!! note this is the monthly integrated data frame!!!!!!!

# df_regclim <- read.csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/MAN_regularclimate/gridcell186-239/MAN_regularclimate_monthly.csv")
#df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")
df_regclim <- read.csv("/Users/biancarius/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")

# Criando objetos de série temporal para todas as colunas
time_series_list_regclim <- lapply(df_regclim[, -1], function(col) {
  ts(col, start = c(1979, 1), frequency = 12)
})

# Atribuindo nomes às séries temporais
names(time_series_list_regclim) <- names(df_regclim[, -1])

bfast_res_list_regclim <- list()

res_bfast_regclim_npp <- bfast(time_series_list_regclim$npp,
                           h = 0.25, max.iter = 1)
#plot(res_bfast_regclim, type = "trend", ylab = "Trend")
plot(res_bfast_regclim_npp, type = "trend")

res_bfast_regclim_evapm <- bfast(time_series_list_regclim$evapm,
                               h = 0.25, max.iter = 1)
#plot(res_bfast_regclim, type = "trend", ylab = "Trend")
plot(res_bfast_regclim_evapm, type = "trend")


str(res_bfast_regclim_evapm, max.level = 3)
res_bfast_regclim_evapm$output
res_bfast_regclim_evapm$output[[1]]$ci
res_bfast_regclim_evapm$output[[1]]$confint
confint(res_bfast_regclim_evapm)

# # Executar bfast e plotar resultados
# 
for (col_name in columns_to_process) {
  res_bfast_regclim <- bfast(time_series_list_regclim[[col_name]],
                       h = 0.25, max.iter = 1)
  bfast_res_list_regclim[[col_name]] <- res_bfast_regclim

  
  # Save the plot 
  # png(paste0(output_path_plots, "bfast_regclim_anova_", col_name, ".png"))
  # plot(res_bfast_regclim, main = col_name, type = "components", ANOVA = TRUE)
  # dev.off()
}

for (col_name in columns_to_process) {
  res_bfast_regclim <- bfast(time_series_list_regclim[[col_name]],
                             h = 0.25, max.iter = 1)
  bfast_res_list_regclim[[col_name]] <- res_bfast_regclim
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_regclim_trend_", col_name, ".png"))
  plot(res_bfast_regclim, main = col_name, type = "trend", ANOVA = TRUE)
  dev.off()
}


# #-------------------------------
# # testing h values - reg clim
# #-------------------------------
#dir for testing h values:
# dir_path = "h_sensitivity/regclim"
# 
# h_values = seq(0.05, 0.95, by = 0.05)
# 
# # Criar uma lista para armazenar os resultados
# result_list <- list()
# 
# for (h in h_values) {
#   print(h)
#   
#   # Executar bfast e armazenar o resultado na lista
#   res_bfast_regclim <- bfast(time_series_regclim, h = h, max.iter = 1)
#   # Criar o nome do arquivo com base no valor de h
#   filename <- file.path(dir_path, paste("regclim_h_", 
#                                         gsub("\\.", "_", as.character(h)), ".png", sep = ""))
#   
#   # Iniciar a gravação do arquivo PNG
#   png(filename)
#   
#   # Gerar o plot
#   plot(res_bfast_regclim, main = paste("h = ", h))
#   
#   # Encerrar a gravação do arquivo PNG
#   dev.off()
#   
#   # Adicionar o resultado à lista
#   result_list[[as.character(h)]] <- res_bfast_regclim
# }
# result_list


# 
# 
# res_bfast_regclim <- bfast(time_series_regclim, h = 0.09)
# print(res_bfast_regclim)
# 
# # # # Crie o gráfico
# # png("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/res_bfast_regclim.png", width = 800, height = 600, units = "px", pointsize = 12, res = 96)
# plot(res_bfast_regclim, main = "NPP\n Regular climate", ylab = "NPP", xlab = "Time")
# # dev.off()
# 
# 
# 
# 
# #-------------------------------
# #-------------------------------
# #    30% prec reduction
# #-------------------------------
# #-------------------------------
# 
# #-------------------------------
# #           1y freq
# #-------------------------------
# 
# # Manaus - 30% prec reduction - 1 year frequency application
# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_1y <- read.csv("/Users/biancarius/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")

# Criando objetos de série temporal para todas as colunas
time_series_list_1y <- lapply(df_1y[, -1], function(col) {
  ts(col, start = c(1979, 1), frequency = 12)
})

# Atribuindo nomes às séries temporais
names(time_series_list_1y) <- names(df_1y[, -1])

bfast_res_list_1y <- list()

res_bfast_1y_npp <- bfast(time_series_list_1y$npp,
                               h = 0.25, max.iter = 1)
#plot(res_bfast_regclim, type = "trend", ylab = "Trend")
plot(res_bfast_1y_npp, type = "trend")

res_bfast_1y_evapm <- bfast(time_series_list_1y$evapm,
                                 h = 0.25, max.iter = 1)
#plot(res_bfast_regclim, type = "trend", ylab = "Trend")
plot(res_bfast_1y_evapm, type = "trend")

# Executar bfast 
# Executar bfast e plotar resultados

# # Executar bfast e plotar resultados
# 
for (col_name in columns_to_process) {
  res_bfast_1y <- bfast(time_series_list_1y[[col_name]],
                             h = 0.25, max.iter = 1)
  bfast_res_list_1y[[col_name]] <- res_bfast_1y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_1y_anova_", col_name, ".png"))
  plot(res_bfast_1y, main = col_name, type = "components", ANOVA = TRUE)
  dev.off()
}

for (col_name in columns_to_process) {
  res_bfast_1y <- bfast(time_series_list_1y[[col_name]],
                             h = 0.25, max.iter = 1)
  bfast_res_list_1y[[col_name]] <- res_bfast_1y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_1y_trend_", col_name, ".png"))
  plot(res_bfast_1y, main = col_name, type = "trend", ANOVA = TRUE)
  dev.off()
}



# # # # Converta a coluna 'Date' para o tipo de data 'yearmon'
# df_1y$Date <- as.yearmon(df_1y$Date)
# # 
# # # # Crie um objeto de série temporal usando a função ts de uma maneira diferente
# time_series_1y <- ts(df_1y$Monthly_NPP_Mean, start = c(1979, 1), frequency = 12)
# 
# # Executar bfast 
# res_bfast_1y <- bfast(time_series_1y, h = 0.25, max.iter = 1)
# 
# plot(res_bfast_1y)
# 
# res_bfast_1y
# 
# 
# # #-------------------------------
# # # testing h values - 1y
# # #-------------------------------
# #dir for testing h values:
# dir_path = "h_sensitivity/freq_1y"
# 
# 
# h_values = seq(0.05, 0.95, by = 0.05)
# 
# # Criar uma lista para armazenar os resultados
# result_list <- list()
# 
# for (h in h_values) {
#   print(h)
#   
#   # Executar bfast e armazenar o resultado na lista
#   res_bfast_1y <- bfast(time_series_1y, h = h, max.iter = 1)
#   # Criar o nome do arquivo com base no valor de h
#   filename <- file.path(dir_path, paste("1yfreq_h_", 
#                                         gsub("\\.", "_", as.character(h)), ".png", sep = ""))
#   
#   # Iniciar a gravação do arquivo PNG
#   png(filename)
#   
#   # Gerar o plot
#   plot(res_bfast_1y, main = paste("h = ", h))
#   
#   # Encerrar a gravação do arquivo PNG
#   dev.off()
#   
#   # Adicionar o resultado à lista
#   result_list[[as.character(h)]] <- res_bfast_1y
# }
# result_list
# # 
# 
# res_bfast <- bfast(time_series, h =0.2)
# print(res_bfast)
# 
# # # # Crie o gráfico
# # png("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/res_bfast_30perc_1y.png", width = 800, height = 600, units = "px", pointsize = 12, res = 96)
# plot(res_bfast, main = "NPP\n-30% prec  1year freq", ylab = "NPP", xlab = "Time")
# # dev.off()
# 
# #-------------------------------
# #           3y freq
# #-------------------------------
# 
# # Manaus - 30% prec reduction - 3 year frequency application
# # !!!!! note this is the monthly integrated data frame!!!!!!!
# df_3y <- read.csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/30perc_reduction/MAN_30prec_3y/gridcell186-239/MAN_30prec_3y_monthly.csv")
df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_3y_monthly.csv")

# Criando objetos de série temporal para todas as colunas
time_series_list_3y <- lapply(df_3y[, -1], function(col) {
  ts(col, start = c(1979, 1), frequency = 12)
})

# Atribuindo nomes às séries temporais
names(time_series_list_3y) <- names(df_3y[, -1])

bfast_res_list_3y <- list()

res_bfast_3y <- bfast(time_series_list_3y$npp, 
                      h = 0.25, max.iter = 1)
# Executar bfast 
# Executar bfast e plotar resultados

for (col_name in columns_to_process) {
  res_bfast_3y <- bfast(time_series_list_3y[[col_name]],
                        h = 0.25, max.iter = 1)
  bfast_res_list_3y[[col_name]] <- res_bfast_3y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_3y_anova_", col_name, ".png"))
  plot(res_bfast_3y, main = col_name, type = "components", ANOVA = TRUE)
  dev.off()
}

for (col_name in columns_to_process) {
  res_bfast_3y <- bfast(time_series_list_3y[[col_name]],
                        h = 0.25, max.iter = 1)
  bfast_res_list_3y[[col_name]] <- res_bfast_3y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_3y_trend_", col_name, ".png"))
  plot(res_bfast_3y, main = col_name, type = "trend", ANOVA = TRUE)
  dev.off()
}

# # # Converta a coluna 'Date' para o tipo de data 'yearmon'
# df_3y$Date <- as.yearmon(df_3y$Date)
# 
# # # # Crie um objeto de série temporal usando a função ts de uma maneira diferente
# time_series_3y <- ts(df_3y$Monthly_NPP_Mean, start = c(1979, 1), frequency = 12)
# 
# res_bfast_3y <- bfast(time_series_3y, h = 0.25, max.iter = 1)
# 
# plot(res_bfast_3y)
# 
# res_bfast_3y
# 
# #-------------------------------
# # testing h values - 3y
# #-------------------------------
# 
# #dir for testing h values:
# dir_path = "h_sensitivity/freq_3y"
# 
# 
# h_values = seq(0.05, 0.95, by = 0.05)
# 
# # Criar uma lista para armazenar os resultados
# result_list <- list()
# 
# for (h in h_values) {
#   print(h)
#   
#   # Executar bfast e armazenar o resultado na lista
#   res_bfast_3y <- bfast(time_series_3y, h = h, max.iter = 1)
#   # Criar o nome do arquivo com base no valor de h
#   filename <- file.path(dir_path, paste("3yfreq_h_", 
#                                         gsub("\\.", "_", as.character(h)), ".png", sep = ""))
#   
#   # Iniciar a gravação do arquivo PNG
#   png(filename)
#   
#   # Gerar o plot
#   plot(res_bfast_3y, main = paste("h = ", h))
#   
#   # Encerrar a gravação do arquivo PNG
#   dev.off()
#   
#   # Adicionar o resultado à lista
#   result_list[[as.character(h)]] <- res_bfast_3y
# }

# 
# res_bfast_3y <- bfast(time_series)
# print(res_bfast_3y)
# 
# # # # Crie o gráfico
# # png("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/res_bfast_30perc_3y.png", width = 800, height = 600, units = "px", pointsize = 12, res = 96)
# # plot(res_bfast_3y, main = "NPP\n-30% prec  3year freq", ylab = "NPP", xlab = "Time")
# # dev.off()
# 
# #-------------------------------
# #           5y freq
# #-------------------------------
# 
# 
# # Manaus - 30% prec reduction - 3 year frequency application
# # !!!!! note this is the monthly integrated data frame!!!!!!!
# # df_5y <- read.csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/30perc_reduction/MAN_30prec_5y/gridcell186-239/MAN_30prec_5y_monthly.csv")
df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_5y_monthly.csv")

# Criando objetos de série temporal para todas as colunas
time_series_list_5y <- lapply(df_5y[, -1], function(col) {
  ts(col, start = c(1979, 1), frequency = 12)
})

# Atribuindo nomes às séries temporais
names(time_series_list_5y) <- names(df_5y[, -1])

bfast_res_list_5y <- list()


# Executar bfast 
for (col_name in columns_to_process) {
  res_bfast_5y <- bfast(time_series_list_5y[[col_name]],
                        h = 0.25, max.iter = 1)
  bfast_res_list_5y[[col_name]] <- res_bfast_5y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_5y_anova_", col_name, ".png"))
  plot(res_bfast_5y, main = col_name, type = "components", ANOVA = TRUE)
  dev.off()
}

for (col_name in columns_to_process) {
  res_bfast_5y <- bfast(time_series_list_5y[[col_name]],
                        h = 0.25, max.iter = 1)
  bfast_res_list_5y[[col_name]] <- res_bfast_5y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_5y_trend_", col_name, ".png"))
  plot(res_bfast_5y, main = col_name, type = "trend", ANOVA = TRUE)
  dev.off()
}


# # Converta a coluna 'Date' para o tipo de data 'yearmon'
# df_5y$Date <- as.yearmon(df_5y$Date)
# 
# 
# # # # Crie um objeto de série temporal usando a função ts de uma maneira diferente
# time_series_5y <- ts(df_5y$Monthly_NPP_Mean, start = c(1979, 1), frequency = 12)
# 
# res_bfast_5y <- bfast(time_series_5y, h = 0.25, max.iter = 1)
# 
# plot(res_bfast_5y)
# 
# res_bfast_5y
# 
# #-------------------------------
# # testing h values - 5y
# #-------------------------------
# 
# #dir for testing h values:
# dir_path = "h_sensitivity/freq_5y"
# 
# 
# h_values = seq(0.05, 0.95, by = 0.05)
# 
# # Criar uma lista para armazenar os resultados
# result_list <- list()
# 
# for (h in h_values) {
#   print(h)
#   
#   # Executar bfast e armazenar o resultado na lista
#   res_bfast_5y <- bfast(time_series_5y, h = h, max.iter = 1)
#   # Criar o nome do arquivo com base no valor de h
#   # filename <- file.path(dir_path, paste("5yfreq_h_", 
#   #                                       gsub("\\.", "_", as.character(h)), ".png", sep = ""))
#   
  # Iniciar a gravação do arquivo PNG
  # png(filename)
  
  # Gerar o plot
#   plot(res_bfast_5y, main = paste("h = ", h))
#   
#   # Encerrar a gravação do arquivo PNG
#   # dev.off()
#   
#   # Adicionar o resultado à lista
#   result_list[[as.character(h)]] <- res_bfast_5y
# }

# 
# # #------------------------
# # #       regular bfast
# # #------------------------
# res_bfast_5y <- bfast(time_series)
# print(res_bfast_5y)
# 
# # # # Crie o gráfico
# # png("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/res_bfast_30perc_5y.png", width = 800, height = 600, units = "px", pointsize = 12, res = 96)
# # plot(res_bfast_5y, main = "NPP\n-30% prec  5year freq", ylab = "NPP", xlab = "Time")
# # dev.off()
# 
# #-------------------------------
# #           7y freq
# #-------------------------------
# 
# 
# # Manaus - 30% prec reduction - 7 year frequency application
# # !!!!! note this is the monthly integrated data frame!!!!!!!
# # df_7y <- read.csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/30perc_reduction/MAN_30prec_7y/gridcell186-239/MAN_30prec_7y_monthly.csv")
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")

# Criando objetos de série temporal para todas as colunas
time_series_list_7y <- lapply(df_7y[, -1], function(col) {
  ts(col, start = c(1979, 1), frequency = 12)
})

# Atribuindo nomes às séries temporais
names(time_series_list_7y) <- names(df_7y[, -1])

bfast_res_list_7y<- list()
# Executar bfast 
# Executar bfast e plotar resultados


# Executar bfast 
for (col_name in columns_to_process) {
  res_bfast_7y <- bfast(time_series_list_7y[[col_name]],
                        h = 0.25, max.iter = 1)
  bfast_res_list_7y[[col_name]] <- res_bfast_7y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_7y_anova_", col_name, ".png"))
  plot(res_bfast_7y, main = col_name, type = "components", ANOVA = TRUE)
  dev.off()
}

for (col_name in columns_to_process) {
  res_bfast_7y <- bfast(time_series_list_7y[[col_name]],
                        h = 0.25, max.iter = 1)
  bfast_res_list_7y[[col_name]] <- res_bfast_7y
  
  
  # Save the plot 
  png(paste0(output_path_plots, "bfast_7y_trend_", col_name, ".png"))
  plot(res_bfast_7y, main = col_name, type = "trend", ANOVA = TRUE)
  dev.off()
}


#-------------------------------
# testing h values - 7y
#-------------------------------

#dir for testing h values:
# dir_path = "h_sensitivity/freq_7y"
# 
# 
# h_values = seq(0.05, 0.95, by = 0.05)
# 
# # Criar uma lista para armazenar os resultados
# result_list <- list()
# 
# for (h in h_values) {
#   print(h)
#   
#   # Executar bfast e armazenar o resultado na lista
#   res_bfast_7y <- bfast(time_series_7y, h = h, max.iter = 1)
#   # Criar o nome do arquivo com base no valor de h
#   filename <- file.path(dir_path, paste("7yfreq_h_", 
#                                         gsub("\\.", "_", as.character(h)), ".png", sep = ""))
#   
#   # Iniciar a gravação do arquivo PNG
#   png(filename)
#   
#   # Gerar o plot
#   plot(res_bfast_7y, main = paste("h = ", h))
#   
#   # Encerrar a gravação do arquivo PNG
#   dev.off()
#   
#   # Adicionar o resultado à lista
#   result_list[[as.character(h)]] <- res_bfast_7y
# }
# 
# # 
# # #------------------------
# # #       regular bfast
# # #------------------------
# # res_bfast_7y <- bfast(time_series)
# # print(res_bfast_7y)
# # 
# # # # # # Crie o gráfico
# # # png("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/res_bfast_30perc_7y.png", width = 800, height = 600, units = "px", pointsize = 12, res = 96)
# # # plot(res_bfast_7y, main = "NPP\n-30% prec  7year freq", ylab = "NPP", xlab = "Time")
# # # dev.off()
# # 
# # 
# # 
# # 
# # 
# # 
# # 
