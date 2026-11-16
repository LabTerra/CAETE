# Folder where we have the CSVs from CAETÊ run
folder <- "~/Documents/CAETE_paper/CAETE/docs/hydraulic_test/tabelass"

# Import tables, including PLS table to check
pls_attrs <- read.csv("~/Documents/CAETE_paper/CAETE/outputs/allomnutri_hydraulics_3k_PKLS/pls_attrs-3000.csv")

diario <- read.csv(file.path(folder, "diario.csv"))

area <- read.csv(file.path(folder, "area.csv"))

View(pls_attrs)
View(diario)
View(area)




################################################

library(ggplot2)
library(dplyr)
library(tidyr)

# Preparar dados
hidro <- diario %>%
  mutate(
    date = as.Date(date),
    agua_solo = wsoil + swsoil
  )

# Calculate the mean seasonal cycle (1979–2016)
psi_ciclo <- hidro %>%
  mutate(month = as.integer(format(date, "%m"))) %>%
  group_by(month) %>%
  summarise(
    psi_mean = mean(psisoil, na.rm = TRUE),
    psi_min = min(psisoil, na.rm = TRUE),
    .groups = "drop"
  )

# Plot psi_solo
ggplot(psi_ciclo, aes(x = month)) +
  geom_line(aes(y = psi_mean), color = "steelblue", linewidth = 1) +
  geom_point(aes(y = psi_mean), color = "steelblue") +
  scale_x_continuous(
    breaks = 1:12,
    labels = month.abb
  ) +
  labs(
    title = "Mean seasonal cycle of soil water potential (1979–2016)",
    x = "Month",
    y = expression(Psi[soil] ~ "(MPa)")
  ) +
  theme_minimal()

##################

# Mean seasonal cycle (1979–2016)
solo_ciclo <- hidro %>%
  mutate(month = as.integer(format(date, "%m"))) %>%
  group_by(month) %>%
  summarise(
    psi_soil = mean(psisoil, na.rm = TRUE),
    water = mean(wsoil + swsoil, na.rm = TRUE),
    .groups = "drop"
  )

# Scale factor for the secondary axis
fator <- diff(range(solo_ciclo$water)) /
  diff(range(solo_ciclo$psi_soil))

offset <- min(solo_ciclo$water) -
  fator * min(solo_ciclo$psi_soil)

# Plot
ggplot(solo_ciclo, aes(x = month)) +
  geom_line(aes(y = water, color = "Soil water"),
            linewidth = 1) +
  geom_point(aes(y = water, color = "Soil water"), size = 2) +
  geom_line(aes(y = psi_soil * fator + offset,
                color = "Soil water potential"),
            linewidth = 1) +
  geom_point(aes(y = psi_soil * fator + offset,
                 color = "Soil water potential"), size = 2) +
  scale_y_continuous(
    name = "Soil water storage (mm)",
    sec.axis = sec_axis(
      ~ (. - offset) / fator,
      name = expression(Psi[soil] ~ "(MPa)")
    )
  ) +
  scale_x_continuous(
    breaks = 1:12,
    labels = month.abb
  ) +
  scale_color_manual(
    values = c(
      "Soil water" = "steelblue",
      "Soil water potential" = "firebrick"
    )
  ) +
  labs(
    title = "Seasonal cycle of soil water and water potential (1979–2016)",
    x = "Month",
    color = NULL
  ) +
  theme_minimal()


#########################

# Check extreme values first
summary(hidro[, c("psisoil", "psi_xylem", "psi_50")])

# Monthly medians
psi_ciclo <- hidro %>%
  mutate(month = as.integer(format(date, "%m"))) %>%
  group_by(month) %>%
  summarise(
    Soil = median(psisoil, na.rm = TRUE),
    Xylem = median(psi_xylem, na.rm = TRUE),
    P50 = median(psi_50, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(Soil, Xylem, P50),
    names_to = "variable",
    values_to = "value"
  )

# Separate panels
ggplot(psi_ciclo, aes(x = month, y = value)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  scale_x_continuous(
    breaks = 1:12,
    labels = month.abb
  ) +
  labs(
    title = "Seasonal cycle of plant water potentials",
    subtitle = "Monthly medians | 1979–2016",
    x = "Month",
    y = "Water potential (MPa)"
  ) +
  theme_minimal()


summary(hidro$psi_xylem)
sum(hidro$psi_xylem < -100, na.rm = TRUE)


##
# Days with extreme xylem water potential
summary(hidro$psi_xylem[hidro$psi_xylem > -100])

quantile(
  hidro$psi_xylem,
  probs = c(0, 0.01, 0.05, 0.5, 0.95, 0.99, 1),
  na.rm = TRUE
)


hidro %>%
  filter(psi_xylem < -5) %>%
  select(date, psisoil, psi_xylem, psi_50,
         krc_max, kl_max, k_norm, f5) %>%
  arrange(psi_xylem)