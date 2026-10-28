library(bfast)

library(dplyr)
library(bfast)

# Carregar dados
df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")

# df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly_NA.csv")

# Lista dos dataframes com suas frequências
df_list <- list(df_regclim = df_regclim, df_1y = df_1y, df_7y = df_7y)

# Normalização dos dados para o intervalo de 0 a 1
normalize_0_1 <- function(x) {
  if (max(x) - min(x) == 0) {
    return(rep(0, length(x)))  # Retorna um vetor de zeros do mesmo comprimento de x
  } else {
    return((x - min(x)) / (max(x) - min(x)))
  }
}
# Definindo os breakpoints para cada variável e frequência
breakpoints <- list(npp = list(df_regclim = list(bp1 = "1998-06", bp2 = ""),
                               df_1y = list(bp1 = "1988-06", bp2 = "1999-02"),
                               df_7y = list(bp1 = "1997-11", bp2 = "2007-05")),
                    ctotal = list(df_regclim = list(bp1 = "1992-12", bp2 = "2002-09"),
                                  df_1y = list(bp1 = "1990-08", bp2 = "2006-10"),
                                  df_7y = list(bp1 = "2006-01", bp2 = "")),
                    evapm = list(df_regclim = list(bp1 = "1998-05", bp2 = ""),
                                 df_1y = list(bp1 = "1997-11", bp2 = "2007-05"),
                                 df_7y = list(bp1 = "2007-06", bp2 = "")),
                    wue = list(df_regclim = list(bp1 = "1989-07", bp2 = "2004-10"),
                               df_1y = list(bp1 = "1997-06", bp2 = "2006-12"),
                               df_7y = list(bp1 = "1989-07", bp2 = "2004-10")))


# Função para executar o processo
plot_bfast <- function(df_name, variavel, breakpoints, df_list) {
  # Carregar dados
  df <- df_list[[df_name]]
  
  # Selecionar as colunas que deseja normalizar
  colunas_para_normalizar <- setdiff(names(df), c("date", "frequency"))
  
  # Normalizar os dados de 0 a 1
  df_normalizado <- df
  df_normalizado[colunas_para_normalizar] <- lapply(df[colunas_para_normalizar], normalize_0_1)
  
  # Criando objetos de série temporal para todas as colunas
  time_series_list <- lapply(df_normalizado[, -1], function(col) {
    ts(col, start = c(1979, 1), frequency = 12)
  })
  
  # Definindo os breakpoints para a variável atual e frequência atual
  current_breakpoints <- breakpoints[[variavel]][[df_name]]
  
  # Executar bfast
  res_bfast <- bfast(time_series_list[[variavel]], h = 0.25, max.iter = 1)
  
  # Converter a série temporal em um data frame
  Vt_df <- data.frame(Year = time(res_bfast$output[[1]]$Vt),
                      Value = coredata(res_bfast$output[[1]]$Vt))
  
  Tt_df <- data.frame(Year = time(res_bfast$output[[1]]$Tt),
                      Value = coredata(res_bfast$output[[1]]$Tt))
  
  # Converter "Year" para um formato de data
  Vt_df$Year <- as.Date(as.yearmon(Vt_df$Year))
  Tt_df$Year <- as.Date(as.yearmon(Tt_df$Year))
  

  
  # Encontrar os índices correspondentes aos breakpoints
  idx <- which(format(Vt_df$Year, "%Y-%m") == current_breakpoints$bp1)
  idx2 <- which(format(Vt_df$Year, "%Y-%m") == current_breakpoints$bp2)
  
  directory <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/"
  
  # Plotar Vt
  plot(Vt_df$Year, Vt_df$Value, type = "l", col = "black", ylim = c(0, 1), 
       xlab = "", ylab = "", main = "", axes = TRUE)
  
  # Plotar Tt (na frente de Vt)
  lines(Tt_df$Year, Tt_df$Value, type = "l", col = "red", lwd = 3.0)
  
  # Adicionar linha vertical em bp1
  if (length(idx) > 0) {
    abline(v = Vt_df$Year[idx], col = "blue", lty = 2, lwd = 1.5)
  }
  
  # Adicionar linha vertical em bp2
  if (length(idx2) > 0) {
    abline(v = Vt_df$Year[idx2], col = "blue", lty = 2, lwd = 1.5)
  }
  # Adicionar faixa vermelha entre 1989-01-01 e 1989-12-01
  # Adicionar faixa vermelha entre 1989-01-01 e 1989-12-01
  # # Adicionar faixa vermelha entre 1989-01-01 e 1989-12-01
  # if (df_name[[1]] == "df_1y") {
  #   for (year in 1979:2016) {
  #     if (year %% 2 == 0) {
  #       start_date <- as.Date(paste(year, "-01-01", sep = ""))
  #       end_date <- as.Date(paste(year, "-12-01", sep = ""))
  #       rect(start_date, -0.1, end_date, 1.1, col = rgb(0.5, 0.5, 0.5, alpha = 0.3), border = NA)  
  #     }
  #   }
  # } else if (df_name[[1]] == "df_7y") {
  #   for (year in 1980:2016) {
  #     if ((year - 1980) %% 7 == 0) { # Anos múltiplos de 7 a partir de 1980
  #       start_date <- as.Date(paste(year, "-01-01", sep = ""))
  #       end_date <- as.Date(paste(year, "-12-01", sep = ""))
  #       rect(start_date, -0.1, end_date, 1.1, col = rgb(0.5, 0.5, 0.5, alpha = 0.3), border = NA) 
  #     }
  #   }
  # }
  
  
  
  
  max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário
  
  # Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
  start_year <- 1975
  # Definir os anos para os ticks do eixo x (a cada 5 anos)
  years <- seq(start_year, max_year, by = 5)
  
  # # Adicionar eixos x e y
  # axis(1)
  # axis(2)
  
  # Remover eixo superior e direito
  mtext(side = 1, text = "", line = -1)  # Remover eixo superior
  mtext(side = 2, text = "", line = -1)  # Remover eixo direito
  
  # Adicionar ticks no eixo x com os anos especificados
  axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.)
  
  # Exibir o nome do dataframe e da variável
  cat("DataFrame:", df_name, "- Variável:", variavel, "\n")
}

# Usando a função com df_name e variavel de interesse
plot_bfast(df_name = "df_1y", variavel = "npp", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_1y", variavel = "ctotal", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_1y", variavel = "evapm", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_1y", variavel = "wue", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_7y", variavel = "npp", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_7y", variavel = "ctotal", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_7y", variavel = "evapm", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_7y", variavel = "wue", breakpoints = breakpoints, df_list = df_list)


plot_bfast(df_name = "df_regclim", variavel = "npp", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_regclim", variavel = "ctotal", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_regclim", variavel = "evapm", breakpoints = breakpoints, df_list = df_list)

plot_bfast(df_name = "df_regclim", variavel = "wue", breakpoints = breakpoints, df_list = df_list)

#sem os eixos da direita e o superior

plot_bfast <- function(df_name, variavel, breakpoints, df_list) {
  # Carregar dados
  df <- df_list[[df_name]]
  
  # Selecionar as colunas que deseja normalizar
  colunas_para_normalizar <- setdiff(names(df), c("date", "frequency"))
  
  # Normalizar os dados de 0 a 1
  df_normalizado <- df
  df_normalizado[colunas_para_normalizar] <- lapply(df[colunas_para_normalizar], normalize_0_1)
  
  # Criando objetos de série temporal para todas as colunas
  time_series_list <- lapply(df_normalizado[, -1], function(col) {
    ts(col, start = c(1979, 1), frequency = 12)
  })
  
  # Definindo os breakpoints para a variável atual e frequência atual
  current_breakpoints <- breakpoints[[variavel]][[df_name]]
  
  # Executar bfast
  res_bfast <- bfast(time_series_list[[variavel]], h = 0.25, max.iter = 1)
  
  # Converter a série temporal em um data frame
  Vt_df <- data.frame(Year = time(res_bfast$output[[1]]$Vt),
                      Value = coredata(res_bfast$output[[1]]$Vt))
  
  Tt_df <- data.frame(Year = time(res_bfast$output[[1]]$Tt),
                      Value = coredata(res_bfast$output[[1]]$Tt))
  
  # Converter "Year" para um formato de data
  Vt_df$Year <- as.Date(as.yearmon(Vt_df$Year))
  Tt_df$Year <- as.Date(as.yearmon(Tt_df$Year))
  
  # Encontrar os índices correspondentes aos breakpoints
  idx <- which(format(Vt_df$Year, "%Y-%m") == current_breakpoints$bp1)
  idx2 <- which(format(Vt_df$Year, "%Y-%m") == current_breakpoints$bp2)
  
  directory <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/normalized_breakpoints/"
  
  # Plotar Vt
  plot(Vt_df$Year, Vt_df$Value, type = "l", col = "black", ylim = c(0, 1), 
       xlab = "", ylab = "", main = "", axes = FALSE)
  
  # Plotar Tt (na frente de Vt)
  lines(Tt_df$Year, Tt_df$Value, type = "l", col = "red", lwd = 3.0)
  
  # Adicionar linha vertical em bp1
  if (length(idx) > 0) {
    abline(v = Vt_df$Year[idx], col = "blue", lty = 2, lwd = 1.5)
  }
  
  # Adicionar linha vertical em bp2
  if (length(idx2) > 0) {
    abline(v = Vt_df$Year[idx2], col = "blue", lty = 2, lwd = 1.5)
  }
  
  # # Adicionar eixos x e y
  # axis(1)
  axis(2, cex.axis = 2.) # Ajuste o tamanho da fonte do eixo y
  
  # Remover eixo superior e direito
  mtext(side = 3, text = "", line = -1)  # Remover eixo superior
  mtext(side = 4, text = "", line = -1)  # Remover eixo direito
  
  max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário
  
  # Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
  start_year <- 1975
  # Definir os anos para os ticks do eixo x (a cada 5 anos)
  years <- seq(start_year, max_year, by = 5)
  
  # Adicionar ticks no eixo x com os anos especificados
  axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.6)
  
  # Exibir o nome do dataframe e da variável
  cat("DataFrame:", df_name, "- Variável:", variavel, "\n")
}

# Definindo as variáveis e os dataframes
variaveis <- c("npp", "ctotal", "evapm", "wue")
df_names <- c("df_regclim", "df_7y", "df_1y")

# Iniciar uma nova página gráfica
par(mfrow = c(length(df_names), length(variaveis)), mar = c(4, 4, 2, 1))

# Loop para criar os plots para todas as combinações de variáveis e dataframes
for (df_name in df_names) {
  for (variavel in variaveis) {
    plot_bfast(df_name = df_name, variavel = variavel, breakpoints = breakpoints, df_list = df_list)
  }
}

# Restaurar o layout padrão
par(mfrow = c(1, 1))
