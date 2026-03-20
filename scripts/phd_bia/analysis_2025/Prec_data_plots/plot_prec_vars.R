library(ggplot2)
library(gridExtra)
library(dplyr)
library(tidyr)

# Reading the annual precipitation data
df_prec_annual <- read.csv("~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/Prec_data/prec_values_yearly_grouped.csv")

# Creating the list of drought years from 1980 to 2016 with intervals according to recurrence (2, 4, 6, 8)
yr_2 <- seq(1980, 2016, by = 2)
yr_4 <- seq(1980, 2016, by = 4)
yr_6 <- seq(1980, 2016, by = 6)
yr_8 <- seq(1980, 2016, by = 8)

# Creating new columns with a 30% precipitation reduction for drought years in each scenario
df_prec_annual$precipitation_2y <- ifelse(df_prec_annual$year %in% yr_2, df_prec_annual$precipitation * 0.7, df_prec_annual$precipitation)
df_prec_annual$precipitation_4y <- ifelse(df_prec_annual$year %in% yr_4, df_prec_annual$precipitation * 0.7, df_prec_annual$precipitation)
df_prec_annual$precipitation_6y <- ifelse(df_prec_annual$year %in% yr_6, df_prec_annual$precipitation * 0.7, df_prec_annual$precipitation)
df_prec_annual$precipitation_8y <- ifelse(df_prec_annual$year %in% yr_8, df_prec_annual$precipitation * 0.7, df_prec_annual$precipitation)

# Save the wide table (one column per scenario)
write.csv(
  df_prec_annual,
  "~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/Prec_data/prec_values_yearly_all_recurrence.csv",
  row.names = FALSE
)

# -------------------------------
# Create tags (scenario + phase)
# -------------------------------

# Convert from wide (many columns) to long (one 'scenario' column)
df_prec_long <- df_prec_annual %>%
  pivot_longer(
    cols = c(precipitation, precipitation_2y, precipitation_4y, precipitation_6y, precipitation_8y),
    names_to = "scenario_raw",
    values_to = "precipitation_value"
  ) %>%
  mutate(
    # Create a clean scenario label
    scenario = case_when(
      scenario_raw == "precipitation"    ~ "baseline",
      scenario_raw == "precipitation_2y" ~ "2y",
      scenario_raw == "precipitation_4y" ~ "4y",
      scenario_raw == "precipitation_6y" ~ "6y",
      scenario_raw == "precipitation_8y" ~ "8y",
      TRUE ~ NA_character_
    ),
    # Create a drought/normal tag within each scenario
    phase = case_when(
      scenario == "baseline" ~ "baseline",
      scenario == "2y" & year %in% yr_2 ~ "drought",
      scenario == "4y" & year %in% yr_4 ~ "drought",
      scenario == "6y" & year %in% yr_6 ~ "drought",
      scenario == "8y" & year %in% yr_8 ~ "drought",
      TRUE ~ "normal"
    )
  ) %>%
  select(year, scenario, phase, precipitation_value)

# Save the long (tagged) table
write.csv(
  df_prec_long,
  "~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/Prec_data/prec_values_yearly_all_recurrence_LONG_tagged.csv",
  row.names = FALSE
)

ggplot(df_prec_long, aes(x = year, y = precipitation_value, group = scenario)) +
  geom_line(aes(linetype = scenario)) +
  geom_point(aes(shape = phase)) +
  facet_wrap(~ scenario, ncol = 1, scales = "free_y") +
  theme_bw()

# Bar plot: one panel per scenario, drought years visually highlighted via 'phase'
p_bar <- ggplot(df_prec_long, aes(x = year, y = precipitation_value, fill = phase)) +
  geom_col(width = 0.9) +                       # Bars with a slight gap
  facet_wrap(~ scenario, ncol = 1) +            # One column: baseline, 2y, 4y, 6y, 8y
  theme_bw() +
  theme(
    panel.spacing = unit(0.25, "lines"),
    axis.text.x   = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "top"
  ) +
  labs(
    x = "Year",
    y = "Annual precipitation",
    fill = "Phase"
  )

print(p_bar)

p_bar +
  scale_x_continuous(breaks = seq(min(df_prec_long$year), max(df_prec_long$year), by = 4))


# -------------------------------
# 1) Prepare data for overlay plot
# -------------------------------

# Baseline series (reference)
df_base <- df_prec_long %>%
  filter(scenario == "baseline") %>%
  select(year, base_precip = precipitation_value)

# Non-baseline scenarios with baseline joined as a reference column
df_overlay <- df_prec_long %>%
  filter(scenario != "baseline") %>%
  left_join(df_base, by = "year") %>%
  mutate(
    year = as.numeric(year),
    scenario = factor(scenario, levels = c("2y", "4y", "6y", "8y"))
  )

# -------------------------------
# 2) Build shading rectangles for drought years (optional)
# -------------------------------

df_shade <- bind_rows(
  tibble(scenario = "2y", year = yr_2),
  tibble(scenario = "4y", year = yr_4),
  tibble(scenario = "6y", year = yr_6),
  tibble(scenario = "8y", year = yr_8)
) %>%
  mutate(
    scenario = factor(scenario, levels = levels(df_overlay$scenario)),
    xmin = year - 0.5,
    xmax = year + 0.5,
    ymin = -Inf,
    ymax =  Inf
  )

# -------------------------------
# 3) Plot: baseline bar behind + scenario bar on top (overlay)
# -------------------------------

p_overlay <- ggplot(df_overlay, aes(x = year)) +

  # Optional: background shading for drought years
  geom_rect(
    data = df_shade,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    alpha = 0.15
  ) +

  # Baseline bars (reference, behind)
  geom_col(
    aes(y = base_precip),
    width = 0.9,
    alpha = 0.35
  ) +

  # Scenario bars (on top)
  geom_col(
    aes(y = precipitation_value),
    width = 0.65,
    alpha = 0.85
  ) +

  facet_wrap(~ scenario, ncol = 1) +
  theme_bw() +
  theme(
    panel.spacing = unit(0.25, "lines"),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "none"
  ) +
  labs(
    x = "Year",
    y = "Annual precipitation"
  )

print(p_overlay)

# Optional: reduce x-axis tick density to avoid clutter
p_overlay + scale_x_continuous(breaks = seq(min(df_overlay$year), max(df_overlay$year), by = 4))

library(dplyr)
library(tidyr)
library(ggplot2)

# This assumes df_prec_annual has:
# year, precipitation, precipitation_2y, precipitation_4y, precipitation_6y, precipitation_8y

# Convert from wide to long (one row per year x scenario)
df_prec_long <- df_prec_annual %>%
  pivot_longer(
    cols = c(precipitation, precipitation_2y, precipitation_4y, precipitation_6y, precipitation_8y),
    names_to = "scenario_raw",
    values_to = "precipitation_value"
  ) %>%
  mutate(
    # Create readable scenario labels
    scenario = case_when(
      scenario_raw == "precipitation"    ~ "baseline",
      scenario_raw == "precipitation_2y" ~ "2y",
      scenario_raw == "precipitation_4y" ~ "4y",
      scenario_raw == "precipitation_6y" ~ "6y",
      scenario_raw == "precipitation_8y" ~ "8y",
      TRUE ~ NA_character_
    ),
    scenario = factor(scenario, levels = c("baseline", "2y", "4y", "6y", "8y")),
    year = as.numeric(year)
  ) %>%
  select(year, scenario, precipitation_value)

# Overlapped filled areas (NOT stacked)
p_area_identity <- ggplot(df_prec_long, aes(x = year, y = precipitation_value, fill = scenario)) +
  geom_area(position = "identity", alpha = 0.25) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "top"
  ) +
  labs(x = "Year", y = "Annual precipitation", fill = "Scenario") +
  scale_x_continuous(breaks = seq(min(df_prec_long$year), max(df_prec_long$year), by = 4))

print(p_area_identity)

library(dplyr)
library(tidyr)
library(ggplot2)

# Baseline as a reference series
df_base <- df_prec_long %>%
  filter(scenario == "baseline") %>%
  select(year, base_precip = precipitation_value)

# Join baseline to each scenario and compute ribbons
df_ribbon <- df_prec_long %>%
  filter(scenario != "baseline") %>%
  left_join(df_base, by = "year") %>%
  mutate(
    # ymin/ymax define the ribbon envelope
    ymin = pmin(base_precip, precipitation_value),
    ymax = pmax(base_precip, precipitation_value)
  )

p_ribbon <- ggplot() +
  # Ribbon shows the gap between baseline and scenario (the imposed change)
  geom_ribbon(
    data = df_ribbon,
    aes(x = year, ymin = ymin, ymax = ymax, fill = scenario),
    alpha = 0.25
  ) +
  # Plot baseline on top as a reference line
  geom_line(
    data = df_base,
    aes(x = year, y = base_precip),
    linewidth = 0.6
  ) +
  facet_wrap(~ scenario, ncol = 1) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "none"
  ) +
  labs(x = "Year", y = "Annual precipitation (baseline + deficit ribbon)")

print(p_ribbon)

library(dplyr)
library(tidyr)
library(ggplot2)

# ---------------------------------------------
# 1) Prepare long data and baseline reference
# ---------------------------------------------

# Convert wide -> long (if you do not already have df_prec_long)
df_prec_long <- df_prec_annual %>%
  pivot_longer(
    cols = c(precipitation, precipitation_2y, precipitation_4y, precipitation_6y, precipitation_8y),
    names_to = "scenario_raw",
    values_to = "precipitation_value"
  ) %>%
  mutate(
    scenario = case_when(
      scenario_raw == "precipitation"    ~ "baseline",
      scenario_raw == "precipitation_2y" ~ "2y",
      scenario_raw == "precipitation_4y" ~ "4y",
      scenario_raw == "precipitation_6y" ~ "6y",
      scenario_raw == "8y"               ~ "8y",  # not expected to happen
      scenario_raw == "precipitation_8y" ~ "8y",
      TRUE ~ NA_character_
    ),
    year = as.numeric(year),
    scenario = factor(scenario, levels = c("baseline", "2y", "4y", "6y", "8y"))
  ) %>%
  select(year, scenario, precipitation_value)

# Extract baseline series
df_base <- df_prec_long %>%
  filter(scenario == "baseline") %>%
  select(year, base_precip = precipitation_value)

# ---------------------------------------------
# 2) Build "deficit bars" dataset
# ---------------------------------------------
# deficit = baseline - scenario
# For your design, deficit is >0 only in drought years (where you multiplied by 0.7).
# We keep only positive deficits so we only draw the drought-year "gap".

df_deficit <- df_prec_long %>%
  filter(scenario != "baseline") %>%
  left_join(df_base, by = "year") %>%
  mutate(
    deficit = base_precip - precipitation_value
  ) %>%
  filter(deficit > 0) %>%  # only drought years
  mutate(
    scenario = factor(scenario, levels = c("2y", "4y", "6y", "8y"))
  )

# Keep a baseline dataset duplicated across scenarios to facet properly
df_base_faceted <- df_base %>%
  tidyr::crossing(scenario = factor(c("2y", "4y", "6y", "8y"), levels = c("2y", "4y", "6y", "8y")))

# ---------------------------------------------
# 3) Plot: baseline bars + deficit "overlay bars"
# ---------------------------------------------
# We draw:
# - baseline as the full bar height
# - deficit as a smaller bar starting at the scenario value (i.e., sitting on top of scenario)
#   so it visually shows the missing amount up to baseline.

p_def_hist <- ggplot() +
  # Baseline bars (reference)
  geom_col(
    data = df_base_faceted,
    aes(x = year, y = base_precip),
    width = 0.9,
    alpha = 0.35
  ) +
  # Deficit bars (the "gap" up to baseline) drawn on top of the scenario height
  geom_col(
    data = df_deficit,
    aes(x = year, y = deficit),
    width = 0.9,
    alpha = 0.6,
    position = position_stack()  # default, but we will use 'ymin/ymax' logic via geom_rect alternative below if needed
  ) +
  facet_wrap(~ scenario, ncol = 1) +
  theme_bw() +
  theme(
    panel.spacing = unit(0.25, "lines"),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "none"
  ) +
  labs(
    x = "Year",
    y = "Annual precipitation (baseline with deficit overlay)"
  ) +
  scale_x_continuous(breaks = seq(min(df_base$year), max(df_base$year), by = 4))

print(p_def_hist)

library(dplyr)
library(tidyr)
library(ggplot2)

# ---------------------------------------------
# 1) Build long table with scenario tags
# ---------------------------------------------
df_prec_long <- df_prec_annual %>%
  pivot_longer(
    cols = c(precipitation, precipitation_2y, precipitation_4y, precipitation_6y, precipitation_8y),
    names_to = "scenario_raw",
    values_to = "precip_scenario"
  ) %>%
  mutate(
    # Create readable scenario labels
    scenario = case_when(
      scenario_raw == "precipitation"    ~ "baseline",
      scenario_raw == "precipitation_2y" ~ "2y",
      scenario_raw == "precipitation_4y" ~ "4y",
      scenario_raw == "precipitation_6y" ~ "6y",
      scenario_raw == "precipitation_8y" ~ "8y",
      TRUE ~ NA_character_
    ),
    year = as.numeric(year),
    scenario = factor(scenario, levels = c("baseline", "2y", "4y", "6y", "8y"))
  ) %>%
  select(year, scenario, precip_scenario)

# ---------------------------------------------
# 2) Extract baseline as a reference series
# ---------------------------------------------
df_base <- df_prec_long %>%
  filter(scenario == "baseline") %>%
  transmute(year, base_precip = precip_scenario)

# ---------------------------------------------
# 3) Create deficit rectangles for ALL scenarios (overlaid)
#    Each rectangle spans one year (xmin/xmax) and the vertical gap
#    from scenario precipitation up to baseline precipitation.
# ---------------------------------------------
df_def_rect <- df_prec_long %>%
  filter(scenario != "baseline") %>%
  left_join(df_base, by = "year") %>%
  mutate(
    # Rectangle boundaries for one-year wide bands
    xmin = year - 0.5,
    xmax = year + 0.5,
    ymin = precip_scenario,
    ymax = base_precip
  ) %>%
  # Keep only years where baseline is higher than scenario (i.e., drought years in your design)
  filter(ymax > ymin)

# ---------------------------------------------
# 4) Plot: baseline bars + overlaid deficit bands
# ---------------------------------------------
p_overlaid <- ggplot() +
  # Baseline bars (reference, behind everything)
  geom_col(
    data = df_base,
    aes(x = year, y = base_precip),
    width = 0.9,
    alpha = 0.25
  ) +
  # Overlaid deficit bands (one layer per scenario, same panel)
  geom_rect(
    data = df_def_rect,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = scenario),
    alpha = 0.25,
    inherit.aes = FALSE
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "top"
  ) +
  labs(
    x = "Year",
    y = "Annual precipitation (baseline + deficit bands)",
    fill = "Scenario"
  ) +
  scale_x_continuous(breaks = seq(min(df_base$year), max(df_base$year), by = 4))

print(p_overlaid)


library(dplyr)
library(tidyr)
library(ggplot2)

# ---------------------------------------------
# 1) Convert from wide to long (scenario tagging)
# ---------------------------------------------
df_prec_long <- df_prec_annual %>%
  pivot_longer(
    cols = c(precipitation, precipitation_2y, precipitation_4y, precipitation_6y, precipitation_8y),
    names_to = "scenario_raw",
    values_to = "precip_scenario"
  ) %>%
  mutate(
    # Create readable scenario labels
    scenario = case_when(
      scenario_raw == "precipitation"    ~ "baseline",
      scenario_raw == "precipitation_2y" ~ "2y",
      scenario_raw == "precipitation_4y" ~ "4y",
      scenario_raw == "precipitation_6y" ~ "6y",
      scenario_raw == "precipitation_8y" ~ "8y",
      TRUE ~ NA_character_
    ),
    year = as.numeric(year),
    scenario = factor(scenario, levels = c("baseline", "2y", "4y", "6y", "8y"))
  ) %>%
  select(year, scenario, precip_scenario)

# ---------------------------------------------
# 2) Extract baseline as a reference series
# ---------------------------------------------
df_base <- df_prec_long %>%
  filter(scenario == "baseline") %>%
  transmute(year, base_precip = precip_scenario)

# ---------------------------------------------
# 3) Build rectangles to split the baseline bar
#    into (bottom = deficit) + (top = scenario)
# ---------------------------------------------
df_split <- df_prec_long %>%
  filter(scenario != "baseline") %>%
  left_join(df_base, by = "year") %>%
  mutate(
    # Deficit is the imposed reduction relative to baseline
    deficit = base_precip - precip_scenario,

    # Define year-wide rectangles
    xmin = year - 0.5,
    xmax = year + 0.5,

    # Bottom band: deficit (from 0 to deficit)
    ymin_def = 0,
    ymax_def = pmax(deficit, 0),

    # Top band: scenario precipitation (from deficit to baseline)
    ymin_scn = pmax(deficit, 0),
    ymax_scn = base_precip
  )

# Keep only years where there is an actual deficit (your drought years)
df_def_rect <- df_split %>%
  filter(deficit > 0) %>%
  transmute(year, scenario, xmin, xmax, ymin = ymin_def, ymax = ymax_def)

df_scn_rect <- df_split %>%
  transmute(year, scenario, xmin, xmax, ymin = ymin_scn, ymax = ymax_scn)

# ---------------------------------------------
# 4) Plot: bottom deficit band + top scenario band
# ---------------------------------------------
p_bottom_deficit <- ggplot() +
  # Draw the bottom deficit band (only drought years)
  geom_rect(
    data = df_def_rect,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = scenario),
    alpha = 0.35,
    inherit.aes = FALSE
  ) +
  # Draw the top portion (scenario precipitation) for all years
  geom_rect(
    data = df_scn_rect,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    alpha = 0.25,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ scenario, ncol = 1) +
  theme_bw() +
  theme(
    panel.spacing = unit(0.25, "lines"),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "top"
  ) +
  labs(
    x = "Year",
    y = "Annual precipitation (baseline height; bottom = deficit)",
    fill = "Scenario"
  ) +
  scale_x_continuous(breaks = seq(min(df_base$year), max(df_base$year), by = 4))

print(p_bottom_deficit)
