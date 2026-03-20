library(earlywarnings)
library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(zoo)
## TESTING SLIDING WINDOW SIZE ##


#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 1 year
#----------------------------------------------


# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_1y_monthly.csv")

# df_1y <- df_1y[df_1y$Date < "2006-12",]

#select the variable of interest
df_1y_npp <- df_1y$Monthly_NPP_Mean

winsize = seq(15, 35, by = 5)

# Create a list to store the results for each sliding window size
results_list <- list()


for (ws in winsize){
  print(ws)
  gws_1y <- generic_ews(df_1y_npp, winsize = ws, detrending = 'loess',
                             logtransform = FALSE, interpolate = FALSE, 
                             AR_n = FALSE, powerspectrum = FALSE)
  # # Adicionar a coluna ws ao gws_1y
  # gws_1y$ws <- ws
  # 
  # print(head(gws_1y))
  # dev.off()
  # Store the results in the list
  results_list[[paste0("ws_", ws)]] <- gws_1y
  
}

# Utilizar lapply para adicionar a coluna "ws"
results_list <- lapply(winsize, function(ws) {
  gws_1y <- generic_ews(df_1y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  dev.off()
  gws_1y$ws <- ws
  return(gws_1y)
})

print(seq(as.Date("1979-01"), length.out = length(gws_1y$timeindex), by = "1 month"))
# Adicionar linhas adicionais se o valor inicial de timeindex não for 1
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

# Verificar novamente se o valor inicial de timeindex é igual a 1 para cada objeto em results_list
print(lapply(results_list, function(result) {
  if (result$timeindex[1] == 1) {
    return("O valor inicial de timeindex é igual a 1.")
  } else {
    return("O valor inicial de timeindex NÃO é igual a 1.")
  }
}))

# Adicione sua nova coluna aqui
results_list <- lapply(results_list, function(result) {
  result$new_column <- as.yearmon(paste(result$new_column, "-01", sep = ""))
  return(result)
})

# Juntar todos os componentes da lista em um único dataframe
final_df <- bind_rows(results_list)

# Converter a coluna ws para um fator para garantir cores diferentes
final_df$ws <- as.factor(final_df$ws)

# Definir uma paleta de cores mais distinta
cores_distintas <- brewer.pal(nlevels(final_df$ws), "Set1")


# Criar um gráfico utilizando ggplot2 com a paleta de cores distinta
ggplot(final_df, aes(x = timeindex, y = ar1, color = ws)) +
  geom_line() +
  scale_color_manual(values = cores_distintas) +
  labs(title = "AR1 ao longo do Timeindex",
       x = "Timeindex",
       y = "AR1") +
  theme_minimal()

########################
#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 3 year
#----------------------------------------------


# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_3y_monthly.csv")

# df_3y <- df_3y[df_3y$Date < "2006-12",]

#select the variable of interest
df_3y_npp <- df_3y$Monthly_NPP_Mean

winsize = seq(15, 35, by = 5)

# Create a list to store the results for each sliding window size
results_list <- list()


for (ws in winsize){
  print(ws)
  gws_3y <- generic_ews(df_3y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  # # Adicionar a coluna ws ao gws_3y
  # gws_3y$ws <- ws
  # 
  # print(head(gws_3y))
  # dev.off()
  # Store the results in the list
  results_list[[paste0("ws_", ws)]] <- gws_3y
  
}

# Utilizar lapply para adicionar a coluna "ws"
results_list <- lapply(winsize, function(ws) {
  gws_3y <- generic_ews(df_3y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  dev.off()
  gws_3y$ws <- ws
  return(gws_3y)
})

# Adicionar linhas adicionais se o valor inicial de timeindex não for 1
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

# Verificar novamente se o valor inicial de timeindex é igual a 1 para cada objeto em results_list
print(lapply(results_list, function(result) {
  if (result$timeindex[1] == 1) {
    return("O valor inicial de timeindex é igual a 1.")
  } else {
    return("O valor inicial de timeindex NÃO é igual a 1.")
  }
}))

# Adicione sua nova coluna aqui
results_list <- lapply(results_list, function(result) {
  result$new_column <- as.yearmon(paste(result$new_column, "-01", sep = ""))
  return(result)
})

# Juntar todos os componentes da lista em um único dataframe
final_df <- bind_rows(results_list)

# Converter a coluna ws para um fator para garantir cores diferentes
final_df$ws <- as.factor(final_df$ws)

# Definir uma paleta de cores mais distinta
cores_distintas <- brewer.pal(nlevels(final_df$ws), "Set1")


# Criar um gráfico utilizando ggplot2 com a paleta de cores distinta
ggplot(final_df, aes(x = timeindex, y = ar1, color = ws)) +
  geom_line() +
  scale_color_manual(values = cores_distintas) +
  labs(title = "AR1 ao longo do Timeindex",
       x = "Timeindex",
       y = "AR1") +
  theme_minimal()

#############################
#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 5 year
#----------------------------------------------


# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_5y_monthly.csv")

# df_5y <- df_5y[df_5y$Date < "2006-12",]

#select the variable of interest
df_5y_npp <- df_5y$Monthly_NPP_Mean

winsize = seq(15, 35, by = 5)

# Create a list to store the results for each sliding window size
results_list <- list()


for (ws in winsize){
  print(ws)
  gws_5y <- generic_ews(df_5y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  # # Adicionar a coluna ws ao gws_5y
  # gws_5y$ws <- ws
  # 
  # print(head(gws_5y))
  # dev.off()
  # Store the results in the list
  results_list[[paste0("ws_", ws)]] <- gws_5y
  
}

# Utilizar lapply para adicionar a coluna "ws"
results_list <- lapply(winsize, function(ws) {
  gws_5y <- generic_ews(df_5y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  dev.off()
  gws_5y$ws <- ws
  return(gws_5y)
})

# Adicionar linhas adicionais se o valor inicial de timeindex não for 1
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

# Verificar novamente se o valor inicial de timeindex é igual a 1 para cada objeto em results_list
print(lapply(results_list, function(result) {
  if (result$timeindex[1] == 1) {
    return("O valor inicial de timeindex é igual a 1.")
  } else {
    return("O valor inicial de timeindex NÃO é igual a 1.")
  }
}))

# Adicione sua nova coluna aqui
results_list <- lapply(results_list, function(result) {
  result$new_column <- as.yearmon(paste(result$new_column, "-01", sep = ""))
  return(result)
})

# Juntar todos os componentes da lista em um único dataframe
final_df <- bind_rows(results_list)

# Converter a coluna ws para um fator para garantir cores diferentes
final_df$ws <- as.factor(final_df$ws)

# Definir uma paleta de cores mais distinta
cores_distintas <- brewer.pal(nlevels(final_df$ws), "Set1")


# Criar um gráfico utilizando ggplot2 com a paleta de cores distinta
ggplot(final_df, aes(x = timeindex, y = ar1, color = ws)) +
  geom_line() +
  scale_color_manual(values = cores_distintas) +
  labs(title = "AR1 ao longo do Timeindex",
       x = "Timeindex",
       y = "AR1") +
  theme_minimal()

#############################
#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 7 year
#----------------------------------------------


# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_7y_monthly.csv")

# df_7y <- df_7y[df_7y$Date < "2006-12",]

#select the variable of interest
df_7y_npp <- df_7y$Monthly_NPP_Mean

winsize = seq(15, 35, by = 5)

# Create a list to store the results for each sliding window size
results_list <- list()


for (ws in winsize){
  print(ws)
  gws_7y <- generic_ews(df_7y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  # # Adicionar a coluna ws ao gws_7y
  # gws_7y$ws <- ws
  # 
  # print(head(gws_7y))
  # dev.off()
  # Store the results in the list
  results_list[[paste0("ws_", ws)]] <- gws_7y
  
}

# Utilizar lapply para adicionar a coluna "ws"
results_list <- lapply(winsize, function(ws) {
  gws_7y <- generic_ews(df_7y_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  dev.off()
  gws_7y$ws <- ws
  return(gws_7y)
})

# Adicionar linhas adicionais se o valor inicial de timeindex não for 1
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

# Verificar novamente se o valor inicial de timeindex é igual a 1 para cada objeto em results_list
print(lapply(results_list, function(result) {
  if (result$timeindex[1] == 1) {
    return("O valor inicial de timeindex é igual a 1.")
  } else {
    return("O valor inicial de timeindex NÃO é igual a 1.")
  }
}))

# Adicione sua nova coluna aqui
results_list <- lapply(results_list, function(result) {
  result$new_column <- as.yearmon(paste(result$new_column, "-01", sep = ""))
  return(result)
})

# Juntar todos os componentes da lista em um único dataframe
final_df <- bind_rows(results_list)

# Converter a coluna ws para um fator para garantir cores diferentes
final_df$ws <- as.factor(final_df$ws)

# Definir uma paleta de cores mais distinta
cores_distintas <- brewer.pal(nlevels(final_df$ws), "Set1")


# Criar um gráfico utilizando ggplot2 com a paleta de cores distinta
ggplot(final_df, aes(x = timeindex, y = ar1, color = ws)) +
  geom_line() +
  scale_color_manual(values = cores_distintas) +
  labs(title = "AR1 ao longo do Timeindex",
       x = "Timeindex",
       y = "AR1") +
  theme_minimal()

df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_regularclimate_monthly.csv")

#############################
#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: regular climate
#----------------------------------------------


# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_regularclimate_monthly.csv")


#select the variable of interest
df_regclim_npp <- df_regclim$Monthly_NPP_Mean

winsize = seq(15, 35, by = 5)

# Create a list to store the results for each sliding window size
results_list <- list()


for (ws in winsize){
  print(ws)
  gws_regclim <- generic_ews(df_regclim_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  # # Adicionar a coluna ws ao gws_regclim
  # gws_regclim$ws <- ws
  # 
  # print(head(gws_regclim))
  # dev.off()
  # Store the results in the list
  results_list[[paste0("ws_", ws)]] <- gws_regclim
  
}

# Utilizar lapply para adicionar a coluna "ws"
results_list <- lapply(winsize, function(ws) {
  gws_regclim <- generic_ews(df_regclim_npp, winsize = ws, detrending = 'loess',
                        logtransform = FALSE, interpolate = FALSE, 
                        AR_n = FALSE, powerspectrum = FALSE)
  dev.off()
  gws_regclim$ws <- ws
  return(gws_regclim)
})

# Adicionar linhas adicionais se o valor inicial de timeindex não for 1
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

# Verificar novamente se o valor inicial de timeindex é igual a 1 para cada objeto em results_list
print(lapply(results_list, function(result) {
  if (result$timeindex[1] == 1) {
    return("O valor inicial de timeindex é igual a 1.")
  } else {
    return("O valor inicial de timeindex NÃO é igual a 1.")
  }
}))

# Adicione sua nova coluna aqui
results_list <- lapply(results_list, function(result) {
  result$new_column <- as.yearmon(paste(result$new_column, "-01", sep = ""))
  return(result)
})

# Juntar todos os componentes da lista em um único dataframe
final_df_regclim <- bind_rows(results_list)

# Converter a coluna ws para um fator para garantir cores diferentes
final_df_regclim$ws <- as.factor(final_df$ws)

# Definir uma paleta de cores mais distinta
cores_distintas <- brewer.pal(nlevels(final_df_regclim$ws), "Set1")


# Criar um gráfico utilizando ggplot2 com a paleta de cores distinta
ggplot(final_df_regclim, aes(x = timeindex, y = ar1, color = ws)) +
  geom_line() +
  scale_color_manual(values = cores_distintas) +
  labs(title = "AR1 ao longo do Timeindex",
       x = "Timeindex",
       y = "AR1") +
  theme_minimal()

write.csv(final_df_regclim, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/window_size_sensitivity/window_size_sensitivity_regclim.csv", row.names = FALSE)






