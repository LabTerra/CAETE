# install.packages("ggplot2")
# install.packages("viridis")

# Carregue as bibliotecas
library(ggplot2)
library(viridis)

# Caminho para a pasta
folder_path <- '/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/csv_allvar/'

# Lista de arquivos na pasta que terminam com "ews.csv"
file_list <- list.files(folder_path, pattern = "_ews.csv", full.names = TRUE)

# Criar gráficos para cada arquivo
for (file_path in file_list) {
  # Ler o arquivo CSV
  df <- read.csv(file_path)
  df <- df[df$frequency %in% c("regclim", "7y", "1y"), ]
  
  # Extrair o nome para usar como título do gráfico
  plot_name <- gsub("_ews.csv", "", basename(file_path))
  
  # Definir cores para cada categoria
  cores <- c("regclim" = "steelblue3",
             "7y" = "darkgreen",
             "1y" = "goldenrod2")
  
  # Criar o gráfico de linha com linhas mais grossas e eixo y de 0 a 1
  plot <- ggplot(df, aes(x = timeindex, y = ar1, color = as.factor(frequency))) +
    geom_line(size = 1.5) +  # Ajuste a espessura das linhas conforme necessário (por exemplo, 1.5)
    scale_color_manual(values = cores) +  # Definir as cores manualmente
    labs(title = paste("Curvas AR1 -", plot_name), x = "Índice de Tempo", y = "Valor AR1", color = "Frequência") +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white"),
      plot.background = element_rect(fill = "white")
    ) +
    ylim(0, 1)  # Define o intervalo do eixo y de 0 a 1
  
  # Imprimir o gráfico
  print(plot)
}
