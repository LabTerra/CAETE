library(zoo)
library(anytime)
library(ggplot2)
library(viridis)
library(dplyr)
library(tidyr)
library(scales)

# base path to access data
base_path <- "~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/yearly_mean_tables"

df_2y <- read.csv(file.path(base_path, "MAN_30prec_2y_yearly.csv"))
df_2y$frequency = "2y"

df_4y <- read.csv(file.path(base_path, "MAN_30prec_4y_yearly.csv"))
df_4y$frequency = "4y"
 
df_6y <- read.csv(file.path(base_path, "MAN_30prec_6y_yearly.csv"))
df_6y$frequency = "6y"

df_8y <- read.csv(file.path(base_path, "MAN_30prec_8y_yearly.csv"))
df_8y$frequency = "8y"

df_regclim <- read.csv(file.path(base_path, "MAN_regularclimate_yearly.csv"))
df_regclim$frequency = "regclim"

df_combined <- rbind(df_2y, df_4y, df_6y, df_8y, df_regclim)

# Convert column "frequency" to factor
df_combined$frequency <- as.factor(df_combined$frequency)

# Define colors for each frequency
colors <- c(
  "regclim" = "#000000",
  "8y"      = "#56B4E9",
  "6y"      = "#009E73",
  "4y"      = "#E69F00",
  "2y"      = "#D55E00"
)

# List of variables to plot
vars <- c("npp", "ctotal", "evapm", "ls")

# Create a simulation-year index
df_combined <- df_combined %>%
  arrange(frequency, date) %>%           # ensure proper order within each run
  group_by(frequency) %>%
  mutate(sim_year = row_number() - 1) %>% # 0, 1, 2, ...
  ungroup()

scale_x_continuous(
  breaks = seq(0, max(df_combined$sim_year, na.rm = TRUE), by = 5)
)

# Plot time series for each variable
df_combined %>%
  pivot_longer(cols = all_of(vars)) %>%
  ggplot(aes(x = sim_year, y = value, color = frequency)) +
  geom_line() +
  scale_color_manual(values = colors) +
  scale_x_continuous(
    breaks = seq(0, max(df_combined$sim_year, na.rm = TRUE), by = 5)
  ) +
  labs(title = "",
       x = "",
       y = "") +
  theme_minimal() +
  facet_wrap(~name, nrow = 2, ncol = 2, scales = "free_y", 
             drop = TRUE,   # Remove subplots vazios
             labeller = labeller(name = c("npp" = "NPP",
                                          "ctotal" = "Total carbon",
                                          "evapm" = "Evapotranspiration",
                                          "ls" = "N. of surviving strategies")))

write.csv(df_combined, "~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/analysis_2025/yearly_ecosystem_state/combined_yearly_vars.csv",row.names = FALSE)
  

# Plot normalized y axis
# Normalizing data from 0 to 1
normalize_0_1 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Apply normalizing function to variables
df_normalized <- df_combined %>%
  mutate(across(all_of(vars), normalize_0_1))

# Plot time series for each variable
df_normalized %>%
  pivot_longer(cols = all_of(vars)) %>%
  ggplot(aes(x = sim_year, y = value, color = frequency)) +
  geom_line() +
  scale_color_manual(values = colors) +
  scale_x_continuous(
    breaks = seq(0, max(df_combined$sim_year, na.rm = TRUE), by = 5)
  ) +
  labs(title = "",
       x = "",
       y = "") +
  theme_minimal() +
  facet_wrap(~name, nrow = 2, ncol = 2, scales = "free_y", 
             drop = TRUE,   # Remove subplots vazios
             labeller = labeller(name = c("npp" = "NPP",
                                          "ctotal" = "Total carbon",
                                          "evapm" = "Evapotranspiration",
                                          "ls" = "N. of surviving strategies")))
