library(bfast)

library(dplyr)
library(bfast)
library(zoo)

# Carregar dados
df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")


# Lista dos dataframes com suas frequências
df_list <- list(df_regclim = df_regclim, df_1y = df_1y, df_7y = df_7y)

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

# Função para calcular a média da variável de interesse antes, entre e depois do breakpoint
calculate_means <- function(df, bp1, bp2, variable_of_interest) {
  # Verificar se bp1 e bp2 são NA
  if (is.na(bp1) || is.na(bp2)) {
    # Se algum dos breakpoints for NA, calcular a média para todo o período
    result <- df %>%
      summarise(mean_variable = mean({{variable_of_interest}}, na.rm = TRUE))
  } else {
    # Caso contrário, calcular a média antes, entre e depois dos breakpoints
    result <- df %>%
      mutate(date = as.yearmon(date)) %>%
      group_by(group = case_when(
        date < as.yearmon(bp1) ~ "before",
        date <= as.yearmon(bp2) ~ "between",
        TRUE ~ "after"
      )) %>%
      summarise(mean_variable = mean({{variable_of_interest}}, na.rm = TRUE))
  }
  return(result)
}

# Aplicar a função para df_regclim com a variável npp separadamente
result_df_regclim_npp <- calculate_means(df_regclim, breakpoints$npp$df_regclim$bp1, breakpoints$npp$df_regclim$bp2, npp)

# Exibir o resultado
result_df_regclim_npp

# Aplicar a função para df_regclim com a variável ctotal separadamente
result_df_regclim_ctotal <- calculate_means(df_regclim, 
                                            breakpoints$ctotal$df_regclim$bp1, 
                                            breakpoints$ctotal$df_regclim$bp2, 
                                            ctotal)

# Exibir o resultado
result_df_regclim_ctotal


result_df_regclim_evapm <- calculate_means(df_regclim, 
                                            breakpoints$evapm$df_regclim$bp1, 
                                            breakpoints$evapm$df_regclim$bp2, 
                                            evapm)

# Exibir o resultado
result_df_regclim_evapm


# Aplicar a função para df_regclim com a variável ctotal separadamente
result_df_regclim_wue <- calculate_means(df_regclim, 
                                            breakpoints$wue$df_regclim$bp1, 
                                            breakpoints$wue$df_regclim$bp2, 
                                            wue)

# Exibir o resultado
result_df_regclim_wue


#############
# Aplicar a função para df_1y com a variável npp separadamente
result_df_1y_npp <- calculate_means(df_1y, breakpoints$npp$df_1y$bp1, breakpoints$npp$df_1y$bp2, npp)

# Exibir o resultado
result_df_1y_npp

# Aplicar a função para df_1y com a variável ctotal separadamente
result_df_1y_ctotal <- calculate_means(df_1y, 
                                            breakpoints$ctotal$df_1y$bp1, 
                                            breakpoints$ctotal$df_1y$bp2, 
                                            ctotal)

# Exibir o resultado
result_df_1y_ctotal


result_df_1y_evapm <- calculate_means(df_1y, 
                                           breakpoints$evapm$df_1y$bp1, 
                                           breakpoints$evapm$df_1y$bp2, 
                                           evapm)

# Exibir o resultado
result_df_1y_evapm


# Aplicar a função para df_1y com a variável ctotal separadamente
result_df_1y_wue <- calculate_means(df_1y, 
                                         breakpoints$wue$df_1y$bp1, 
                                         breakpoints$wue$df_1y$bp2, 
                                         wue)

# Exibir o resultado
result_df_1y_wue


########################
# Aplicar a função para df_7y com a variável npp separadamente
result_df_7y_npp <- calculate_means(df_7y, breakpoints$npp$df_7y$bp1, breakpoints$npp$df_7y$bp2, npp)

# Exibir o resultado
result_df_7y_npp

# Aplicar a função para df_7y com a variável ctotal separadamente
result_df_7y_ctotal <- calculate_means(df_7y, 
                                            breakpoints$ctotal$df_7y$bp1, 
                                            breakpoints$ctotal$df_7y$bp2, 
                                            ctotal)

# Exibir o resultado
result_df_7y_ctotal


result_df_7y_evapm <- calculate_means(df_7y, 
                                           breakpoints$evapm$df_7y$bp1, 
                                           breakpoints$evapm$df_7y$bp2, 
                                           evapm)

# Exibir o resultado
result_df_7y_evapm


# Aplicar a função para df_7y com a variável ctotal separadamente
result_df_7y_wue <- calculate_means(df_7y, 
                                         breakpoints$wue$df_7y$bp1, 
                                         breakpoints$wue$df_7y$bp2, 
                                         wue)

# Exibir o resultado
result_df_7y_wue
