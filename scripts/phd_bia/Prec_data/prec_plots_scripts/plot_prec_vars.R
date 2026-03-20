library(ggplot2)
library(gridExtra)
df_prec_annual = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/Prec_data/prec_values_yearly_grouped.csv")


df_vars = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/Prec_data/merged_prec_vars_regclim.csv")
# Criando a lista de anos de 1980 até 2016 com intervalo de 2 anos
anos_2 <- seq(1980, 2016, by = 2)

# Criando a lista de anos de 1980 até 2016 com intervalo de 8 anos
anos_8 <- seq(1980, 2016, by = 8)


# Criando uma nova coluna 'Precipitation_adjusted' no dataframe original
df_prec_annual$precipitation_1y <- ifelse(df_prec_annual$year %in% anos_2, df_prec_annual$precipitation * 0.7, df_prec_annual$precipitation)

# Criando uma nova coluna 'Precipitation_adjusted_8' no dataframe original
df_prec_annual$precipitation_7y <- ifelse(df_prec_annual$year %in% anos_8, df_prec_annual$precipitation * 0.7, df_prec_annual$precipitation)

# write.csv(df_prec_annual,"/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/Prec_data/prec_values_yearly_regclim_7y_1y.csv", row.names = FALSE)

df_vars = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/yearly_grouped.csv")

colnames(df_vars)[1] <- "year"

df_regclim = df_vars[df_vars$frequency=="regclim",]

df_7y = df_vars[df_vars$frequency=="7y",]

df_1y = df_vars[df_vars$frequency=="1y",]

df_prec_regclim = df_prec_annual[, c("year","precipitation")]

df_prec_7y = df_prec_annual[, c("year","precipitation_7y")]

df_prec_1y = df_prec_annual[, c("year","precipitation_1y")]


merge_regclim = merge(df_prec_regclim, df_regclim, by = "year")

merge_regclim = merge_regclim[,c("year", "precipitation", "npp", "ctotal", "evapm", "wue", "ls")]

merge_7y = merge(df_prec_7y, df_7y, by = "year")


merge_1y = merge(df_prec_1y, df_1y, by = "year")

library(gridExtra)

# Define o gráfico p
p <- ggplot(merge_regclim, aes(x = year)) +
  geom_bar(aes(y = precipitation), stat = "identity", fill = "skyblue") +
  xlab("") +
  ylab("Precipitation (mm/y)") +
  ggtitle("")

# Adiciona a série de linha NPP em um eixo y secundário
p <- p + geom_line(aes(y = npp * max(merge_regclim$precipitation) / max(merge_regclim$npp), colour = "NPP"), size = 1)

# Ajusta a escala para o eixo y secundário para alinhar com os valores de NPP e altera o nome
p <- p + scale_y_continuous(sec.axis = 
                              sec_axis(~ .* max(merge_regclim$npp) / max(merge_regclim$precipitation), 
                                       name = ""))

# Customiza o gráfico
p <- p + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "")


# Define o gráfico p_1
p_1 <- ggplot(merge_1y, aes(x = year)) +
  geom_bar(aes(y = precipitation_1y, fill = ifelse(year %in% anos_2, "darkblue", "skyblue")), stat = "identity") +
  xlab("") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha NPP em um eixo y secundário, com os mesmos ajustes do gráfico p
p_1 <- p_1 + geom_line(aes(y = npp * max(merge_1y$precipitation_1y) / max(merge_1y$npp), colour = "NPP"), size = 1)
p_1 <- p_1 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$npp) / max(merge_regclim$precipitation), 
                                           name = "NPP (kgC/m2/y)"))+
                                  theme(axis.title.y.secondary = element_text(size = 20))

# Customiza o gráfico p_1
p_1 <- p_1 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Define o gráfico p_7
p_7 <- ggplot(merge_7y, aes(x = year)) +
  geom_bar(aes(y = precipitation_7y, fill = ifelse(year %in% anos_8, "darkblue", "skyblue")), stat = "identity") +
  xlab("") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha NPP em um eixo y secundário, com os mesmos ajustes do gráfico p
p_7 <- p_7 + geom_line(aes(y = npp * max(merge_7y$precipitation_7y) / max(merge_7y$npp), colour = "NPP"), size = 1)
p_7 <- p_7 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$npp) / max(merge_regclim$precipitation), 
                                           name = ""))

# Customiza o gráfico p_7
p_7 <- p_7 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Combina os gráficos em uma única grade
combined_plots_npp <- grid.arrange(p, p_7, p_1, ncol = 3)




###################
###plotting total c
###################
# Define o gráfico p
p <- ggplot(merge_regclim, aes(x = year)) +
  geom_bar(aes(y = precipitation), stat = "identity", fill = "skyblue") +
  xlab("") +
  ylab("Precipitation (mm/y)") +
  ggtitle("")

# Adiciona a série de linha ctotal em um eixo y secundário
p <- p + geom_line(aes(y = ctotal * max(merge_regclim$precipitation) / max(merge_regclim$ctotal), colour = "ctotal"), size = 1)

# Ajusta a escala para o eixo y secundário para alinhar com os valores de ctotal e altera o nome
p <- p + scale_y_continuous(sec.axis = 
                              sec_axis(~ .* max(merge_regclim$ctotal) / max(merge_regclim$precipitation), 
                                       name = ""))

# Customiza o gráfico
p <- p + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "")

# Define o gráfico p_1
p_1 <- ggplot(merge_1y, aes(x = year)) +
  geom_bar(aes(y = precipitation_1y, fill = ifelse(year %in% anos_2, "darkblue", "skyblue")), stat = "identity") +
  xlab("") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha ctotal em um eixo y secundário, com os mesmos ajustes do gráfico p
p_1 <- p_1 + geom_line(aes(y = ctotal * max(merge_1y$precipitation_1y) / max(merge_1y$ctotal), colour = "ctotal"), size = 1)
p_1 <- p_1 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$ctotal) / max(merge_regclim$precipitation), 
                                           name = "Total carbon (kgC/m2)"))+
                                  theme(axis.title.y.secondary = element_text(size = 20))

# Customiza o gráfico p_1
p_1 <- p_1 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Define o gráfico p_7
p_7 <- ggplot(merge_7y, aes(x = year)) +
  geom_bar(aes(y = precipitation_7y, fill = ifelse(year %in% anos_8, "darkblue", "skyblue")), stat = "identity") +
  xlab("") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha ctotal em um eixo y secundário, com os mesmos ajustes do gráfico p
p_7 <- p_7 + geom_line(aes(y = ctotal * max(merge_7y$precipitation_7y) / max(merge_7y$ctotal), colour = "ctotal"), size = 1)
p_7 <- p_7 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$ctotal) / max(merge_regclim$precipitation), 
                                           name = ""))

# Customiza o gráfico p_7
p_7 <- p_7 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Combina os gráficos em uma única grade
combined_plots_ctotal <- grid.arrange(p, p_7, p_1, ncol = 3)



###################
###plotting evapm
###################
# Define o gráfico p
p <- ggplot(merge_regclim, aes(x = year)) +
  geom_bar(aes(y = precipitation), stat = "identity", fill = "skyblue") +
  xlab("") +
  ylab("Precipitation (mm/y)") +
  ggtitle("")

# Adiciona a série de linha evapm em um eixo y secundário
p <- p + geom_line(aes(y = evapm * max(merge_regclim$precipitation) / max(merge_regclim$evapm), colour = "evapm"), size = 1)

# Ajusta a escala para o eixo y secundário para alinhar com os valores de evapm e altera o nome
p <- p + scale_y_continuous(sec.axis = 
                              sec_axis(~ .* max(merge_regclim$evapm) / max(merge_regclim$precipitation), 
                                       name = ""))

# Customiza o gráfico
p <- p + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "")

# Define o gráfico p_1
p_1 <- ggplot(merge_1y, aes(x = year)) +
  geom_bar(aes(y = precipitation_1y, fill = ifelse(year %in% anos_2, "darkblue", "skyblue")), stat = "identity") +
  xlab("") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha evapm em um eixo y secundário, com os mesmos ajustes do gráfico p
p_1 <- p_1 + geom_line(aes(y = evapm * max(merge_1y$precipitation_1y) / max(merge_1y$evapm), colour = "evapm"), size = 1)
p_1 <- p_1 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$evapm) / max(merge_regclim$precipitation), 
                                           name = "Evapotranspiration (mm/day)"))+
                                  theme(axis.title.y.secondary = element_text(size = 20))

# Customiza o gráfico p_1
p_1 <- p_1 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Define o gráfico p_7
p_7 <- ggplot(merge_7y, aes(x = year)) +
  geom_bar(aes(y = precipitation_7y, fill = ifelse(year %in% anos_8, "darkblue", "skyblue")), stat = "identity") +
  xlab("") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha evapm em um eixo y secundário, com os mesmos ajustes do gráfico p
p_7 <- p_7 + geom_line(aes(y = evapm * max(merge_7y$precipitation_7y) / max(merge_7y$evapm), colour = "evapm"), size = 1)
p_7 <- p_7 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$evapm) / max(merge_regclim$precipitation), 
                                           name = ""))

# Customiza o gráfico p_7
p_7 <- p_7 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), 
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Combina os gráficos em uma única grade
combined_plots_evapm <- grid.arrange(p, p_7, p_1, ncol = 3)




###################
###plotting wue
###################
# Define o gráfico p
p <- ggplot(merge_regclim, aes(x = year)) +
  geom_bar(aes(y = precipitation), stat = "identity", fill = "skyblue") +
  xlab("Year") +
  ylab("Precipitation (mm/y)") +
  ggtitle("")

# Adiciona a série de linha wue em um eixo y secundário
p <- p + geom_line(aes(y = wue * max(merge_regclim$precipitation) / max(merge_regclim$wue), colour = "wue"), size = 1)

# Ajusta a escala para o eixo y secundário para alinhar com os valores de wue e altera o nome
p <- p + scale_y_continuous(sec.axis = 
                              sec_axis(~ .* max(merge_regclim$wue) / max(merge_regclim$precipitation), 
                                       name = ""))

# Customiza o gráfico
p <- p + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), axis.title.x = element_text(size = 20),
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "")

# Define o gráfico p_1
p_1 <- ggplot(merge_1y, aes(x = year)) +
  geom_bar(aes(y = precipitation_1y, fill = ifelse(year %in% anos_2, "darkblue", "skyblue")), stat = "identity") +
  xlab("Year") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha wue em um eixo y secundário, com os mesmos ajustes do gráfico p
p_1 <- p_1 + geom_line(aes(y = wue * max(merge_1y$precipitation_1y) / max(merge_1y$wue), colour = "wue"), size = 1)
p_1 <- p_1 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$wue) / max(merge_regclim$precipitation), 
                                           name = "WUE"))+
                                  theme(axis.title.y.secondary = element_text(size = 20))

# Customiza o gráfico p_1
p_1 <- p_1 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), axis.title.x = element_text(size = 20),
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Define o gráfico p_7
p_7 <- ggplot(merge_7y, aes(x = year)) +
  geom_bar(aes(y = precipitation_7y, fill = ifelse(year %in% anos_8, "darkblue", "skyblue")), stat = "identity") +
  xlab("Year") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha wue em um eixo y secundário, com os mesmos ajustes do gráfico p
p_7 <- p_7 + geom_line(aes(y = wue * max(merge_7y$precipitation_7y) / max(merge_7y$wue), colour = "wue"), size = 1)
p_7 <- p_7 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$wue) / max(merge_regclim$precipitation), 
                                           name = ""))

# Customiza o gráfico p_7
p_7 <- p_7 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), axis.title.x = element_text(size = 20),
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Combina os gráficos em uma única grade
combined_plots_wue <- grid.arrange(p, p_7, p_1, ncol = 3)

###################
###plotting ls
###################
# Define o gráfico p
p <- ggplot(merge_regclim, aes(x = year)) +
  geom_bar(aes(y = precipitation), stat = "identity", fill = "skyblue") +
  xlab("Year") +
  ylab("Precipitation (mm/y)") +
  ggtitle("")

# Adiciona a série de linha ls em um eixo y secundário
p <- p + geom_line(aes(y = ls * max(merge_regclim$precipitation) / max(merge_regclim$ls), colour = "ls"), size = 1)

# Ajusta a escala para o eixo y secundário para alinhar com os valores de ls e altera o nome
p <- p + scale_y_continuous(sec.axis = 
                              sec_axis(~ .* max(merge_regclim$ls) / max(merge_regclim$precipitation), 
                                       name = ""))

# Customiza o gráfico
p <- p + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), axis.title.x = element_text(size = 20),
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "")

# Define o gráfico p_1
p_1 <- ggplot(merge_1y, aes(x = year)) +
  geom_bar(aes(y = precipitation_1y, fill = ifelse(year %in% anos_2, "darkblue", "skyblue")), stat = "identity") +
  xlab("Year") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha ls em um eixo y secundário, com os mesmos ajustes do gráfico p
p_1 <- p_1 + geom_line(aes(y = ls * max(merge_1y$precipitation_1y) / max(merge_1y$ls), colour = "ls"), size = 1)
p_1 <- p_1 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$ls) / max(merge_regclim$precipitation), 
                                           name = "ls"))+
  theme(axis.title.y.secondary = element_text(size = 20))

# Customiza o gráfico p_1
p_1 <- p_1 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), axis.title.x = element_text(size = 20),
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Define o gráfico p_7
p_7 <- ggplot(merge_7y, aes(x = year)) +
  geom_bar(aes(y = precipitation_7y, fill = ifelse(year %in% anos_8, "darkblue", "skyblue")), stat = "identity") +
  xlab("Year") +
  ylab("") +
  ggtitle("")

# Adiciona a série de linha ls em um eixo y secundário, com os mesmos ajustes do gráfico p
p_7 <- p_7 + geom_line(aes(y = ls * max(merge_7y$precipitation_7y) / max(merge_7y$ls), colour = "ls"), size = 1)
p_7 <- p_7 + scale_y_continuous(sec.axis = 
                                  sec_axis(~ .* max(merge_regclim$ls) / max(merge_regclim$precipitation), 
                                           name = "ls"))

# Customiza o gráfico p_7
p_7 <- p_7 + theme_minimal() +
  theme(legend.position = "none", axis.title.y = element_text(size = 20), axis.title.x = element_text(size = 20),
        axis.text.y = element_text(size = 15), axis.text.x = element_text(size = 15)) +
  scale_colour_manual(values = "salmon", name = "", labels = "") +
  scale_fill_manual(values = c("darkblue" = "darkblue", "skyblue" = "skyblue"), name = "")

# Combina os gráficos em uma única grade
combined_plots_ls <- grid.arrange(p, p_7, p_1, ncol = 3)

grid.arrange(combined_plots_npp, combined_plots_ctotal, combined_plots_evapm, combined_plots_wue,combined_plots_ls, ncol = 1, nrow = 5)

