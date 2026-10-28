library(bfast)
library(zoo)
library(ggplot2)

# Função para plotar Tt e Vt com a linha vertical em junho de 1998
plot_Tt_Vt <- function(table, col_name = var, bp1, bp2, freq) {
  # Criando objetos de série temporal para todas as colunas
  time_series_list <- lapply(table[, -1], function(col) {
    ts(col, start = c(1979, 1), frequency = 12)
  })
  
  # Executar bfast
  res_bfast <- bfast(time_series_list[[col_name]], h = 0.25, max.iter = 1)
  
  
  # Converter a série temporal em um data frame
  Vt_df <- data.frame(Year = time(res_bfast$output[[1]]$Vt),
                      Value = coredata(res_bfast$output[[1]]$Vt))
  
  Tt_df <- data.frame(Year = time(res_bfast$output[[1]]$Tt),
                      Value = coredata(res_bfast$output[[1]]$Tt))
  
  # Converter "Year" para um formato de data
  Vt_df$Year <- as.Date(as.yearmon(Vt_df$Year))
  Tt_df$Year <- as.Date(as.yearmon(Tt_df$Year))
  
  # Encontrar o índice correspondente a #1 breakpoint
  idx <- which(format(Vt_df$Year, "%Y-%m") == bp1)
  
  idx2 <- which(format(Vt_df$Year, "%Y-%m") == bp2)
  
  directory <- "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/breakpoints/"
  
  resolucao <- 500  # Resolução em DPI (dots per inch)
  qualidade <- 0     # Nível de compressão (0-9)
  
  # Abrir o dispositivo PNG
  png(filename = paste(directory, "plot_", col_name, "_", freq, ".png", sep = "",        # Unidades de medida
                       res = 300,             # Resolução em DPI
                       type = "cairo",              # Tipo de renderização
                       compression = "lzw",         # Método de compressão
                       quality = qualidade))
  
  # Plotar Vt
  plot(Vt_df$Year, Vt_df$Value, type = "l", col = "black", ylim = range(c(Tt_df$Value, Vt_df$Value)), 
       xlab = "", ylab = "", main = "")
  
  
  # Plotar Tt (na frente de Vt)
  lines(Tt_df$Year, Tt_df$Value, type = "l", col = "red", lwd = 2.)
  
  # Adicionar linha vertical em junho de 1998
  abline(v = Vt_df$Year[idx], col = "blue", lty = 2, lwd = 2)
  
  # Adicionar linha vertical em junho de 1998
  abline(v = Vt_df$Year[idx2], col = "blue", lty = 2, lwd = 2)
  
  dev.off()
  
  
}

# Carregar dados
df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")

df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")

df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_3y_monthly.csv")

df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_5y_monthly.csv")

df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")

#plot regclim npp
plot_Tt_Vt(df_regclim, "npp", "1998-06", "", "regclim")

#plot regclim npp
plot_Tt_Vt(df_regclim, "ctotal", "1992-12", "2002-09", "regclim")

#plot regclim npp
plot_Tt_Vt(df_regclim, "evapm", "1998-05", "", "regclim")

#plot regclim npp
plot_Tt_Vt(df_regclim, "wue", "1989-07", "2004-10", "regclim")


#plot 1 y frequency


plot_Tt_Vt(df_1y, "npp", "1988-06", "1999-02", "1y")


plot_Tt_Vt(df_1y, "ctotal", "1990-08", "2006-10", "1y")


plot_Tt_Vt(df_1y, "evapm", "1997-11", "2007-05","1y")


plot_Tt_Vt(df_1y, "wue", "1997-06", "2006-12", "1y")



#plot 3 y frequency


plot_Tt_Vt(df_3y, "npp", "1991-11", "2006-01", "3y")


plot_Tt_Vt(df_3y, "ctotal", "1992-12", "2006-04", "3y")


plot_Tt_Vt(df_3y, "evapm", "1992-02", "2007-06", "3y")


plot_Tt_Vt(df_3y, "wue", "2006-11", "", "3y")

#plot 5 y frequency

plot_Tt_Vt(df_5y, "npp", "2002-07", "", "5y")


plot_Tt_Vt(df_5y, "ctotal", "1997-08", "2007-02", "5y")


plot_Tt_Vt(df_5y, "evapm", "2002-07", "", "5y")


plot_Tt_Vt(df_5y, "wue", "1989-07", "2006-05", "5y")



#plot 7 y frequency

plot_Tt_Vt(df_7y, "npp", "1997-11", "2007-05", "7y")


plot_Tt_Vt(df_7y, "ctotal", "2006-01", "", "7y")


plot_Tt_Vt(df_7y, "evapm", "2007-06", "", "7y")


plot_Tt_Vt(df_7y, "wue", "1989-07", "2004-10", "7y")

