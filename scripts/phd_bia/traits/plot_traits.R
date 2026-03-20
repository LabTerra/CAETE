library(dplyr)
library(ggplot2)
library(gridExtra)


# Read the CSV file
table_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/PLS_alive_traits_MAN_30prec_1y.csv")
table_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/PLS_alive_traits_MAN_30prec_7y.csv")
table_regclim <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/PLS_alive_traits_MAN_regularclimate.csv")
# Calculate the weighted mean of SLA for each year for both 1y and 7y
mean_wd_1y <- table_1y %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_wd = sum(wd_random * OC) / sum(OC)) %>%
  mutate(Source = "1y")

mean_wd_7y <- table_7y %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_wd = sum(wd_random * OC) / sum(OC)) %>%
  mutate(Source = "7y")

mean_wd_regclim <- table_regclim %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_wd = sum(wd_random * OC) / sum(OC)) %>%
  mutate(Source = "regclim")

# Merge the two datasets
combined_data_wd <- rbind(mean_wd_1y, mean_wd_7y, mean_wd_regclim)
# Definição das cores para cada categoria
cores <- c("1y" = "lightsalmon2",
           "7y" = "seagreen4",
           "regclim" = "steelblue3")

# Create the line plot with weighted mean of SLA for both 1y and 7y using the specified colors
wd = ggplot(combined_data_wd, aes(x = YEAR, y = weighted_mean_wd, color = Source)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = cores) +  # Definindo as cores manualmente
  labs(x = "Year", y = expression("WD weighted mean (g/cm"^"3"*")"), title = "") +
  labs(
    x = "Year",
    size = 18) +  # Define o tamanho da fonte
  theme_minimal() +
  theme(
    axis.title = element_text(size = 18),  # Tamanho da fonte dos rótulos dos eixos
    axis.text = element_text(size = 16),   # Tamanho da fonte dos números dos eixos
    plot.title = element_text(size = 22),
    legend.position = 'none')  # Tamanho da fonte do título do gráfico
    



#################SLA
# Calculate the weighted mean of SLA for each year
# Calculate the weighted mean of SLA for each year for both 1y and 7y
mean_sla_1y <- table_1y %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_sla = sum(sla_random*1000 * OC) / sum(OC)) %>%
  mutate(Source = "1y")

mean_sla_7y <- table_7y %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_sla = sum(sla_random *1000* OC) / sum(OC)) %>%
  mutate(Source = "7y")

mean_sla_regclim <- table_regclim %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_sla = sum(sla_random * 1000*OC) / sum(OC)) %>%
  mutate(Source = "regclim")

# Merge the two datasets
combined_data_sla <- rbind(mean_sla_1y, mean_sla_7y, mean_sla_regclim)
# Definição das cores para cada categoria
cores <- c("1y" = "lightsalmon2",
           "7y" = "seagreen4",
           "regclim" = "steelblue3")

# Create the line plot with weighted mean of SLA for both 1y and 7y using the specified colors
sla <- ggplot(combined_data_sla, aes(x = YEAR, y = weighted_mean_sla, color = Source)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = cores) +  # Definindo as cores manualmente
  labs(x = "Year", y = expression("SLA weighted mean (m"^"2"*"/kg)"), title = "") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 18),  # Tamanho da fonte dos rótulos dos eixos
    axis.text = element_text(size = 16),   # Tamanho da fonte dos números dos eixos
    plot.title = element_text(size = 22),
    legend.position = 'none')  # Título da legenda

#################g1
# Calculate the weighted mean of g1 for each year
# Calculate the weighted mean of g1 for each year for both 1y and 7y
mean_g1_1y <- table_1y %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_g1 = sum(g1 * OC) / sum(OC)) %>%
  mutate(Source = "1y")

mean_g1_7y <- table_7y %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_g1 = sum(g1 * OC) / sum(OC)) %>%
  mutate(Source = "7y")

mean_g1_regclim <- table_regclim %>%
  group_by(YEAR) %>%
  summarise(weighted_mean_g1 = sum(g1 * OC) / sum(OC)) %>%
  mutate(Source = "regclim")

# Merge the two datasets
combined_data_g1 <- rbind(mean_g1_1y, mean_g1_7y, mean_g1_regclim)

# Definição das cores para cada categoria
cores <- c("1y" = "lightsalmon2",
           "7y" = "seagreen4",
           "regclim" = "steelblue3")

# Create the line plot with weighted mean of g1 for both 1y and 7y using the specified colors
g1 = ggplot(combined_data_g1, aes(x = YEAR, y = weighted_mean_g1, color = Source)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = cores) +  # Definindo as cores manualmente
  labs(x = "Year", y = expression("g1 weighted mean"), title = "") +
  labs(
    x = "Year",
    size = 18) +  # Define o tamanho da fonte
  theme_minimal() +
  theme(
    axis.title = element_text(size = 18),  # Tamanho da fonte dos rótulos dos eixos
    axis.text = element_text(size = 16),   # Tamanho da fonte dos números dos eixos
    plot.title = element_text(size = 22),
    legend.position = 'none')  # Tamanho da fonte do título do gráfico

grid.arrange(wd, sla, g1, ncol = 3, nrow = 1)

# write.csv(combined_data_g1, "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/g1_weighted_mean.csv")
# write.csv(combined_data_wd, "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/wd_weighted_mean.csv")
# write.csv(combined_data_sla, "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/sla_weighted_mean.csv")
# 
# combined_data_regclim = cbind(mean_g1_regclim, mean_sla_regclim, mean_wd_regclim)
# combined_data_7y = cbind(mean_g1_7y,mean_sla_7y,mean_wd_7y)
# combined_data_1y = cbind(mean_g1_1y,mean_sla_1y,mean_wd_1y)
# 
# write.csv(combined_data_regclim, "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/traits_weightedmean_regclim.csv")
# write.csv(combined_data_7y, "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/traits_weightedmean_7y.csv")
# write.csv(combined_data_1y, "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/traits_weightedmean_1y.csv")
