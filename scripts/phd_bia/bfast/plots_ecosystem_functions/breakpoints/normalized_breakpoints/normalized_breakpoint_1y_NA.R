library(bfast)

library(dplyr)
library(bfast)
library(zoo)
library(ggplot2)

npp = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/bfast_1y_npp_NA.csv")

npp$Year=as.Date(as.yearmon(npp$Year))


# Plotar Vt
plot(npp$Year, npp$value_vt, type = "l", col = "black", ylim = c(0, 1), 
     xlab = "", ylab = "", main = "", axes = FALSE)

# Plotar Tt (na frente de Vt)
lines(npp$Year, npp$value_tt, type = "l", col = "red", lwd = 3.0)

abline(v = as.numeric(as.Date("1987-10-01")), col = "blue", lty = 2, lwd = 1.5)
abline(v = as.numeric(as.Date("1999-02-01")), col = "blue", lty = 2, lwd = 1.5)

max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário

# Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
start_year <- 1975
# Definir os anos para os ticks do eixo x (a cada 5 anos)
years <- seq(start_year, max_year, by = 5)

# Adicionar ticks no eixo x com os anos especificados
axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.)



ctotal = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/bfast_1y_ctotal_NA.csv")
ctotal$Year=as.Date(as.yearmon(ctotal$Year))


# Plotar Vt
plot(ctotal$Year, ctotal$value_vt, type = "l", col = "black", ylim = c(0, 1), 
     xlab = "", ylab = "", main = "", axes = FALSE)

# Plotar Tt (na frente de Vt)
lines(ctotal$Year, ctotal$value_tt, type = "l", col = "red", lwd = 3.0)

abline(v = as.numeric(as.Date("1990-07-01")), col = "blue", lty = 2, lwd = 1.5)
abline(v = as.numeric(as.Date("1999-07-01")), col = "blue", lty = 2, lwd = 1.5)

max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário

# Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
start_year <- 1975
# Definir os anos para os ticks do eixo x (a cada 5 anos)
years <- seq(start_year, max_year, by = 5)

# Adicionar ticks no eixo x com os anos especificados
axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.)



evapm = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/bfast_1y_evapm_NA.csv")
evapm$Year=as.Date(as.yearmon(evapm$Year))


# Plotar Vt
plot(evapm$Year, evapm$value_vt, type = "l", col = "black", ylim = c(0, 1), 
     xlab = "", ylab = "", main = "", axes = FALSE)

# Plotar Tt (na frente de Vt)
lines(evapm$Year, evapm$value_tt, type = "l", col = "red", lwd = 3.0)

abline(v = as.numeric(as.Date("1987-11-01")), col = "blue", lty = 2, lwd = 1.5)
abline(v = as.numeric(as.Date("1998-03-01")), col = "blue", lty = 2, lwd = 1.5)

max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário

# Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
start_year <- 1975
# Definir os anos para os ticks do eixo x (a cada 5 anos)
years <- seq(start_year, max_year, by = 5)

# Adicionar ticks no eixo x com os anos especificados
axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.)

df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")

# wue = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/bfast_1y_wue_NA.csv")
wue$Year=as.Date(as.yearmon(wue$Year))


# Plotar Vt
plot(wue$Year, wue$value_vt, type = "l", col = "black", ylim = c(0, 1), 
     xlab = "", ylab = "", main = "", axes = FALSE)

# Plotar Tt (na frente de Vt)
lines(wue$Year, wue$value_tt, type = "l", col = "red", lwd = 3.0)

abline(v = as.numeric(as.Date("1987-11-01")), col = "blue", lty = 2, lwd = 1.5)
abline(v = as.numeric(as.Date("1998-03-01")), col = "blue", lty = 2, lwd = 1.5)

max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário

# Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
start_year <- 1975
# Definir os anos para os ticks do eixo x (a cada 5 anos)
years <- seq(start_year, max_year, by = 5)

# Adicionar ticks no eixo x com os anos especificados
axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.)

# Usando a função com df_name e variavel de interesse

# df <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly_NA.csv")
# 
# # Normalização dos dados para o intervalo de 0 a 1
# normalize_0_1 <- function(x) {
#   if (max(x) - min(x) == 0) {
#     return(rep(0, length(x)))  # Retorna um vetor de zeros do mesmo comprimento de x
#   } else {
#     return((x - min(x)) / (max(x) - min(x)))
#   }
# }
# 
# # Selecionar as colunas que deseja normalizar
# colunas_para_normalizar <- setdiff(names(df), c("date", "frequency"))
# # Normalizar os dados de 0 a 1
# df_normalizado <- df
# df_normalizado[colunas_para_normalizar] <- lapply(df[colunas_para_normalizar], normalize_0_1)
# # Criando objetos de série temporal para todas as colunas
# time_series_list <- lapply(df_normalizado[, -1], function(col) {
#   ts(col, start = c(1979, 1), frequency = 12)
# })
# # Executar bfast
# res_bfast <- bfast(time_series_list$wue, h = 0.25, max.iter = 1)
# # plot(res_bfast)
# # Converter a série temporal em um data frame
# Vt_df <- data.frame(Year = time(res_bfast$output[[1]]$Vt),
#                     Value = coredata(res_bfast$output[[1]]$Vt))
# # Converter "Year" para um formato de data
# Tt_df <- data.frame(Year = time(res_bfast$output[[1]]$Tt),
#                     Value = coredata(res_bfast$output[[1]]$Tt))
# 
# # Converter "Year" para um formato de data
# Vt_df$Year <- as.Date(as.yearmon(Vt_df$Year))
# Tt_df$Year <- as.Date(as.yearmon(Tt_df$Year))
# 
# 
# 
# # Renomear a coluna
# names(Vt_df)[names(Vt_df) == "Value"] <- "value_vt"
# # Renomear a coluna
# names(Tt_df)[names(Tt_df) == "Value"] <- "value_tt"
# 
# csv = cbind(Vt_df, Tt_df)
# 
# # write.csv(csv,file = "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/bfast_1y_wue_NA.csv") 
# 
# 
# # Ler o arquivo CSV
# npp <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/bfast_1y_npp_NA.csv")
# 
# # Criar as datas
# datas <- seq(as.Date("2007-01-01"), as.Date("2016-12-01"), by = "month")
# 
# # Criar o dataframe
# df <- data.frame(value_vt = numeric(length(datas)), value_tt = numeric(length(datas)),Year = format(datas, "%Y-%m"))
# 
# # Simplificar as datas
# df$Year <- substr(df$Year, 1, 7)
# 
# write.csv(df,"/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/datas_beyond_2007.csv")
