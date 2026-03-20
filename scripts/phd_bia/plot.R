library(zoo)
library(anytime)
library(ggplot2)
library(viridis)
library(dplyr)
library(tidyr)

df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv")
df_1y$frequency = "1y"

df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_3y_monthly.csv")
df_3y$frequency = "3y"

df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_5y_monthly.csv")
df_5y$frequency = "5y"

df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_7y_monthly.csv")
df_7y$frequency = "7y"

df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_regularclimate_monthly.csv")
df_regclim$frequency = "regclim"

df_combined <- rbind(df_1y, df_3y, df_5y, df_7y, df_regclim)

df_combined$date = anydate(df_combined$date)

# Converta a coluna "frequency" para o tipo fator
df_combined$frequency <- as.factor(df_combined$frequency)



## Definição das cores para cada categoria
cores <- c("1y" = "lightsalmon2",
           "3y" = "gold1",
           "5y" = "#9370DB",
           "7y" = "seagreen4",
           "regclim" = "steelblue3")

# Lista das variáveis que deseja plotar
variaveis <- c("npp", "ctotal", "evapm", "wue", "ls")

# Plot das séries temporais para cada variável
df_combined %>%
  pivot_longer(cols = all_of(variaveis)) %>%
  ggplot(aes(x = date, y = value, color = frequency)) +
  geom_line() +
  scale_color_manual(values = cores) +  # Aplicando as cores específicas
  labs(title = "",
       x = "",
       y = "") +
  theme_minimal() +
  facet_wrap(~name, nrow = 2, ncol = 3, scales = "free_y", 
             drop = TRUE,   # Remove subplots vazios
             labeller = labeller(name = c("npp" = "NPP",
                                          "ctotal" = "Total carbon",
                                          "evapm" = "Evapotranspiration",
                                          "wue" = "Water use efficiency",
                                          "ls" = "N. of surviving strategies")))

  

#normalizado:
library(ggplot2)
library(tidyr)
library(dplyr)

# Definição das cores para cada categoria
cores <- c("1y" = "firebrick3",
           "3y" = "lightsalmon2",
           "5y" = "#2E8B57",
           "7y" = "#9370DB",
           "regclim" = "steelblue1")

# Lista das variáveis que deseja plotar
variaveis <- c("npp", "ctotal", "evapm", "wue")

# Normalização dos dados usando Z-score
df_combined_normalized <- df_combined %>%
  mutate_at(vars(variaveis), scale)

# Plot das séries temporais normalizadas para cada variável
df_plot <- df_combined_normalized %>%
  pivot_longer(cols = variaveis)

ggplot(data = df_plot, aes(x = date, y = value, color = frequency)) +
  geom_line() +
  scale_color_manual(values = cores) +
  labs(title = "Séries Temporais Normalizadas",
       x = "Data",
       color = "Frequency") +
  theme_minimal() +
  facet_wrap(~name, nrow = 2, ncol = 2, scales = "free_y") +
  lapply(unique(df_plot$name), function(var) {
    list(
      labs(y = var)
    )
  })

