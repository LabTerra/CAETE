library(dplyr)
library(bfast)

# Carregar dados
df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")


# Lista dos dataframes com suas frequências
df_list <- list(df_regclim = df_regclim, df_1y = df_1y, df_7y = df_7y)
df_list <- list(df_regclim = df_regclim)

# Normalização dos dados para o intervalo de 0 a 1
normalize_0_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
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


# Loop sobre cada dataframe e frequência
for (df_name in names(df_list)) {
  for (variavel in names(breakpoints)) {
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
    
    # Abrir o dispositivo PNG
    png(filename = paste(directory, "normalized_bfast_",variavel , "_", df_name, ".png", sep = ""))
    
    # Plotar Vt
    plot(Vt_df$Year, Vt_df$Value, type = "l", col = "black", ylim = c(0, 1), 
         xlab = "", ylab = "", main = "", axes = TRUE)
    
    # Plotar Tt (na frente de Vt)
    lines(Tt_df$Year, Tt_df$Value, type = "l", col = "red", lwd = 3.5)
    
    # Adicionar linha vertical em bp1
    if (length(idx) > 0) {
      abline(v = Vt_df$Year[idx], col = "blue", lty = 2, lwd = 3.5)
    }
    
    # Adicionar linha vertical em bp2
    if (length(idx2) > 0) {
      abline(v = Vt_df$Year[idx2], col = "blue", lty = 2, lwd = 3.5)
    }
    
    # Definir o valor máximo do eixo x
    max_year <- 2025  # Defina o valor máximo do eixo x conforme necessário
    
    # Adicionar ticks no eixo x a cada 5 anos a partir de 1980 até o valor máximo
    start_year <- 1975
    # Definir os anos para os ticks do eixo x (a cada 5 anos)
    years <- seq(start_year, max_year, by = 5)
    
    # Adicionar ticks no eixo x com os anos especificados
    axis(1, at = as.Date(paste0(years, "-01-01")), labels = format(as.Date(paste0(years, "-01-01")), "%Y"), cex.axis = 1.5)
    
    
    
    # Exibir o nome do dataframe e da variável
    cat("DataFrame:", df_name, "- Variável:", variavel, "\n")
    
    # Aguardar um segundo para visualização (opcional)
    Sys.sleep(1)
    dev.off()
  }
}



##############
#WITH THE ANOVA COMPONENT
############
variables = c("npp", "ctotal", "evapm", "wue")

# Loop sobre cada dataframe e frequência
for (df_name in names(df_list)) {
  for (variavel in variables) {
    # Carregar dados
    df <- df_list[[df_name]]
    # Selecionar as colunas que deseja normalizar
    colunas_para_normalizar <- setdiff(names(df), c("date", "frequency"))
    
    #     # Normalizar os dados de 0 a 1
    df_normalizado <- df
    df_normalizado[colunas_para_normalizar] <- lapply(df[colunas_para_normalizar], normalize_0_1)
    
    #     # Criando objetos de série temporal para todas as colunas
    time_series_list <- lapply(df_normalizado[, -1], function(col) {
      ts(col, start = c(1979, 1), frequency = 12)
    })
    
    
    # Executar bfast
    res_bfast <- bfast(time_series_list[[variavel]], h = 0.25, max.iter = 1)
    print(res_bfast$Mags)
    plot(res_bfast, type = "components", ANOVA = TRUE)
    
    # Exibir o nome do dataframe e da variável
    cat("DataFrame:", df_name, "- Variável:", variavel, "\n")
    
    niter = length(res_bfast$output)
    slopes = coef(res_bfast$output[[niter]]$bp.Vt)[,2]
    print(df_name)
    print(variavel)
    print(slopes)
    print(res_bfast)
    
    # Aguardar um segundo para visualização (opcional)
    Sys.sleep(1)
    
    
  }
}

