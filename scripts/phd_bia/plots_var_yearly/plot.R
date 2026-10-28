library(zoo)
library(anytime)
library(ggplot2)
library(viridis)
library(dplyr)
library(tidyr)
library(scales)

df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_1y_yearly.csv")
df_1y$frequency = "1y"

# df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_3y_yearly.csv")
# df_3y$frequency = "3y"
# 
# df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_5y_yearly.csv")
# df_5y$frequency = "5y"

df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_7y_yearly.csv")
df_7y$frequency = "7y"

df_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_regularclimate_yearly.csv")
df_regclim$frequency = "regclim"

# df_combined <- rbind(df_1y, df_3y, df_5y, df_7y, df_regclim)
df_combined <- rbind(df_1y, df_7y, df_regclim)


# df_combined$date = anydate(df_combined$date)

# Converta a coluna "frequency" para o tipo fator
df_combined$frequency <- as.factor(df_combined$frequency)



## Definição das cores para cada categoria
# cores <- c("1y" = "lightsalmon2",
#            "3y" = "gold1",
#            "5y" = "#9370DB",
#            "7y" = "seagreen4",
#            "regclim" = "steelblue3")

## Definição das cores para cada categoria
cores <- c("1y" = "lightsalmon2",
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
cores <- c("1y" = "lightsalmon2",
           "7y" = "seagreen4",
           "regclim" = "steelblue3")

# Lista das variáveis que deseja plotar
variaveis <- c("npp", "ctotal", "evapm", "wue", "csto", "cleaf", "croot", "cheart", "csto", "csap", "cwood", "photo")

# Normalização dos dados para o intervalo de 0 a 1
normalize_0_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Aplicar a função de normalização às variáveis desejadas
df_normalized <- df_combined %>%
  mutate(across(all_of(variaveis), normalize_0_1))

# Definição das cores para cada categoria
cores <- c("1y" = "lightsalmon2",
           "7y" = "seagreen4",
           "regclim" = "steelblue3")

# Lista das variáveis que deseja plotar
# variaveis <- c("npp", "ctotal", "evapm", "wue", "csto", "cleaf", "croot")

# Normalização dos dados para o intervalo de 0 a 1
normalize_0_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Aplicar a função de normalização às variáveis desejadas
df_normalized <- df_combined %>%
  mutate(across(all_of(variaveis), normalize_0_1))

# Criar e exibir cada plot separadamente
for (var in variaveis) {
  plot <- ggplot(data = df_normalized, aes(x = date, y = !!sym(var), color = frequency)) +
    geom_line(size=1) +
    scale_color_manual(values = cores) +
    labs(
      x = "Year",
      y = "Normalized value",
      size = 18) +  # Define o tamanho da fonte
    theme_minimal() +
    theme(
      axis.title = element_text(size = 18),  # Tamanho da fonte dos rótulos dos eixos
      axis.text = element_text(size = 16),   # Tamanho da fonte dos números dos eixos
      plot.title = element_text(size = 22),  # Tamanho da fonte do título do gráfico
      legend.position = "none")  # Remover a legenda
  
  print(plot)
}

# Criar e exibir cada plot separadamente
for (var in variaveis) {
  plot <- ggplot(data = df_normalized, aes(x = date, y = !!sym(var), color = frequency)) +
    geom_line(size=1) +
    scale_color_manual(values = cores) +
    labs(
      x = "Year",
      y = "Normalized value",
      size = 18) +  # Define o tamanho da fonte
    theme_minimal() +
    theme(
      axis.title = element_text(size = 18),  # Tamanho da fonte dos rótulos dos eixos
      axis.text = element_text(size = 16),   # Tamanho da fonte dos números dos eixos
      plot.title = element_text(size = 22),  # Tamanho da fonte do título do gráfico
      legend.position = "none") +  # Remover a legenda
    ggtitle(paste(var))  # Adiciona o título com o nome da variável
  
  print(plot)
}


library(gridExtra)

# Criar cada plot separadamente
plot_npp <- ggplot(data = df_normalized, aes(x = date, y = npp, color = frequency)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Year",
    y = "Normalized value",
    size = 16) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 22),
    legend.position = "none")

plot_ctotal <- ggplot(data = df_normalized, aes(x = date, y = ctotal, color = frequency)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Year",
    y = "Normalized value",
    size = 16) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 22),
    legend.position = "none")

plot_evapm <- ggplot(data = df_normalized, aes(x = date, y = evapm, color = frequency)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Year",
    y = "Normalized value",
    size = 14) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 22),
    legend.position = "none")

plot_wue <- ggplot(data = df_normalized, aes(x = date, y = wue, color = frequency)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Year",
    y = "Normalized value",
    size = 16) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 22),
    legend.position = "none")

# Organizar os plots na sequência desejada
grid.arrange(plot_npp, plot_evapm, plot_ctotal, plot_wue, ncol = 2)

library(gridExtra)

# Criar cada plot separadamente
plot_ls <- ggplot(data = df_combined, aes(x = date, y = ls, color = frequency)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Year",
    y = "Surviving strategies",
    size = 14) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 16),
    plot.title = element_text(size = 22),
    legend.position = "none")

# Organizar os plots na sequência desejada
grid.arrange(plot_npp, plot_evapm, plot_ls, plot_ctotal, plot_wue, ncol = 3)


library(tidyverse)

# Supondo que você já tenha lido os dados e definido a coluna frequency para "1y"
# Se necessário, substitua os caminhos do arquivo para ler seus dados

# Ler os dados para a frequência de 1y
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_1y_yearly.csv")
df_1y$frequency <- "1y"

# Normalização das variáveis para o intervalo de 0 a 1
normalize_0_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Selecionar apenas as variáveis desejadas
variaveis <- c("npp", "ctotal", "evapm", "wue", "csto", "cleaf", "croot", "ls")

# Normalizar as variáveis
df_1y_normalized <- df_1y %>%
  mutate(across(all_of(variaveis), normalize_0_1))

# Reorganizar os dados para o formato longo
df_1y_long <- df_1y_normalized %>%
  pivot_longer(cols = all_of(variaveis),
               names_to = "variavel",
               values_to = "valor")

# Definir as cores para cada variável
cores <- c(
  "npp" = "red",
  "ctotal" = "blue",
  "evapm" = "green",
  "wue" = "purple",
  "csto" = "orange",
  "cleaf" = "yellow",
  "croot" = "brown",
  "ls" = "black"
)

# Criar o gráfico
ggplot(data = df_1y_long, aes(x = date, y = valor, color = variavel)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Ano",
    y = "Valor Normalizado",
    color = "Variável"
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 16),
    legend.position = "right"
  )


##plot with ls
library(tidyverse)

# Supondo que você já tenha lido os dados e definido a coluna frequency para "1y"
# Se necessário, substitua os caminhos do arquivo para ler seus dados

# Ler os dados para a frequência de 1y
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_1y_yearly.csv")
df_1y$frequency <- "1y"

# Normalização das variáveis para o intervalo de 0 a 1
normalize_0_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Selecionar apenas as variáveis desejadas
variaveis <- c("npp", "ctotal", "evapm", "wue", "ls")

# Normalizar as variáveis
df_1y_normalized <- df_1y %>%
  mutate(across(all_of(variaveis), normalize_0_1))

# Reorganizar os dados para o formato longo
df_1y_long <- df_1y_normalized %>%
  pivot_longer(cols = all_of(variaveis),
               names_to = "variavel",
               values_to = "valor")

# Definir as cores para cada variável
cores <- c(
  "npp" = "red",
  "ctotal" = "blue",
  "evapm" = "green",
  "wue" = "purple",
  "ls" = "black"
)

# Criar o gráfico
ggplot(data = df_1y_long, aes(x = date, y = valor, color = variavel)) +
  geom_line(size = 1) +
  scale_color_manual(values = cores) +
  labs(
    x = "Ano",
    y = "Valor Normalizado",
    color = "Variável"
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 16),
    legend.position = "right"
  )

corre = cor(df_1y$npp, df_1y$ls)
