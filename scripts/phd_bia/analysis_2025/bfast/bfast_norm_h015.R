# ============================================================
# 1) LIBRARIES
# ============================================================
library(dplyr)
library(bfast)
library(zoo)       # as.yearmon(), rollmean()
library(tidyr)     # pivot_longer(), pivot_wider()
library(ggplot2)

# ============================================================
# 2) PATHS AND USER SETTINGS
# ============================================================
path_file <- "/Users/biancarius/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/monthly_mean_tables/"

files <- list(
  regclim = "MAN_regularclimate_monthly.csv",
  y7      = "MAN_30prec_7y_monthly.csv",
  y5      = "MAN_30prec_5y_monthly.csv",
  y3      = "MAN_30prec_3y_monthly.csv",
  y1      = "MAN_30prec_1y_monthly.csv"
)

# Variables to analyze
columns_to_process <- c("npp", "ctotal", "evapm")  # add "wue" if you want

# BFAST settings (keep fixed for fair comparisons)
bfast_h <- 0.15
bfast_max_iter <- 1

# Collapse definition for y1 (keep collapse month included)
collapse_ym <- "2006-12"

# Output paths
output_csv <- "~/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/analysis_2025/bfast/bfast_bp_slopes.csv"

# ============================================================
# 3) HELPER FUNCTIONS
# ============================================================

# ---- 3.1) Add simulation time (run_year starts at 1)
add_run_time <- function(df) {
  df %>%
    mutate(date = as.Date(zoo::as.yearmon(date))) %>%
    arrange(date) %>%
    mutate(
      month_index = row_number(),                  # 1, 2, 3, ...
      run_year    = floor((month_index - 1) / 12) + 1,  # 1, 2, 3, ...
      run_time    = (month_index - 1) / 12 + 1          # continuous time, starts at 1
    )
}

# ---- 3.2) Robust min-max normalization
normalize_minmax <- function(x, xmin, xmax) {
  x <- as.numeric(x)
  if (all(is.na(x))) return(x)
  if (is.na(xmin) || is.na(xmax)) return(rep(NA_real_, length(x)))
  if ((xmax - xmin) == 0) return(rep(0, length(x)))
  (x - xmin) / (xmax - xmin)
}

# ---- 3.3) Scenario-wise min/max table
compute_minmax_scenario <- function(df, vars) {
  vars_present <- intersect(vars, names(df))
  out <- lapply(vars_present, function(v) {
    x <- as.numeric(df[[v]])
    tibble(variable = v,
           min = min(x, na.rm = TRUE),
           max = max(x, na.rm = TRUE))
  }) %>% bind_rows()
  out
}

# ---- 3.4) Apply scenario-wise normalization
apply_minmax_table <- function(df, minmax_table) {
  df_out <- df
  for (i in seq_len(nrow(minmax_table))) {
    v <- minmax_table$variable[i]
    if (v %in% names(df_out)) {
      df_out[[v]] <- normalize_minmax(df_out[[v]], minmax_table$min[i], minmax_table$max[i])
    }
  }
  df_out
}

# ---- 3.5) Convert selected variables to a list of ts objects
to_ts_list_selected <- function(df, cols_ts, freq = 12) {
  cols_present <- intersect(cols_ts, names(df))
  ts_list <- lapply(cols_present, function(v) {
    ts(as.numeric(df[[v]]), start = c(1, 1), frequency = freq)  # start at simulation year 1
  })
  setNames(ts_list, cols_present)
}

# ---- 3.6) Run BFAST and extract breakpoints, slopes, magnitudes
run_bfast_summary <- function(ts_obj, h = 0.15, max.iter = 1) {
  
  res <- bfast(ts_obj, h = h, max.iter = max.iter)
  niter <- length(res$output)
  out <- res$output[[niter]]
  
  # Breakpoint indices (in the Vt piecewise regression)
  bp_idx <- breakpoints(out$bp.Vt)$breakpoints
  bp_idx <- bp_idx[!is.na(bp_idx)]
  
  # Segment slopes from the fitted piecewise linear model
  slopes <- coef(out$bp.Vt)[, 2]
  
  # Breakpoint magnitudes (may be NULL depending on configuration)
  mags <- res$Mags
  if (is.null(mags)) mags <- numeric(0)
  
  # Convert breakpoint indices into run_time and run_year (simulation-based)
  # Note: bp_idx refers to time index of the trend component (monthly)
  bp_run_time <- if (length(bp_idx) > 0) (bp_idx - 1) / 12 + 1 else numeric(0)
  bp_run_year <- if (length(bp_idx) > 0) floor((bp_idx - 1) / 12) + 1 else integer(0)
  
  list(
    bfast_object = res,
    bp_idx       = bp_idx,
    bp_run_time  = bp_run_time,
    bp_run_year  = bp_run_year,
    slopes       = as.numeric(slopes),
    mags         = as.numeric(mags)
  )
}

# ---- 3.7) Run BFAST for all scenarios x variables
run_bfast_all <- function(ts_lists, variables, h = 0.15, max.iter = 1) {
  results <- list()
  
  for (sc in names(ts_lists)) {
    for (v in variables) {
      if (!v %in% names(ts_lists[[sc]])) next
      key <- paste(sc, v, sep = "_")
      results[[key]] <- run_bfast_summary(ts_lists[[sc]][[v]], h = h, max.iter = max.iter)
    }
  }
  
  results
}

# ---- 3.8) Build wide table: bp1_date, bp1_run_year, mag1, slope1, ...
results_to_wide_df <- function(results_list, normalization_label = "scenario_wise") {
  
  keys <- names(results_list)
  
  combos <- do.call(rbind, lapply(keys, function(k) {
    parts <- strsplit(k, "_")[[1]]
    data.frame(
      normalization = normalization_label,
      scenario = parts[1],
      variable = parts[2],
      stringsAsFactors = FALSE
    )
  })) %>%
    as_tibble() %>%
    distinct()
  
  rows <- vector("list", nrow(combos))
  
  for (j in seq_len(nrow(combos))) {
    
    scenario <- combos$scenario[j]
    variable <- combos$variable[j]
    key <- paste(scenario, variable, sep = "_")
    
    r <- results_list[[key]]
    
    cols <- list(
      normalization = normalization_label,
      scenario = scenario,
      variable = variable
    )
    
    # Breakpoints: store BOTH run_time and run_year (and optionally a readable date label)
    if (!is.null(r) && length(r$bp_idx) > 0) {
      for (i in seq_along(r$bp_idx)) {
        cols[[paste0("bp", i, "_run_time")]] <- r$bp_run_time[i]
        cols[[paste0("bp", i, "_run_year")]] <- r$bp_run_year[i]
      }
    }
    
    # Magnitudes aligned with breakpoints
    if (!is.null(r) && length(r$mags) > 0) {
      for (i in seq_along(r$mags)) {
        cols[[paste0("mag", i)]] <- r$mags[i]
      }
    }
    
    # Slopes per segment
    if (!is.null(r) && length(r$slopes) > 0) {
      for (i in seq_along(r$slopes)) {
        cols[[paste0("slope", i)]] <- r$slopes[i]
      }
    }
    
    rows[[j]] <- as_tibble(cols)
  }
  
  out <- bind_rows(rows)
  
  # Order columns logically
  bp_rt_cols <- grep("^bp[0-9]+_run_time$", names(out), value = TRUE)
  bp_ry_cols <- grep("^bp[0-9]+_run_year$", names(out), value = TRUE)
  mag_cols   <- grep("^mag[0-9]+$", names(out), value = TRUE)
  slope_cols <- grep("^slope[0-9]+$", names(out), value = TRUE)
  
  out %>%
    select(normalization, scenario, variable,
           all_of(bp_rt_cols), all_of(bp_ry_cols),
           all_of(mag_cols), all_of(slope_cols))
}

# ============================================================
# 4) LOAD DATA AND PREPROCESS
# ============================================================

# Load and add simulation time
df_list <- lapply(files, function(fname) {
  read.csv(file.path(path_file, fname)) %>% add_run_time()
})

# Filter y1 before collapse (including collapse month)
collapse_date <- as.Date(zoo::as.yearmon(collapse_ym))
df_list$y1 <- df_list$y1 %>% filter(date <= collapse_date)

# ============================================================
# 5) SCENARIO-WISE NORMALIZATION
# ============================================================

minmax_by_scenario <- lapply(df_list, compute_minmax_scenario, vars = columns_to_process)

df_list_norm <- mapply(
  FUN = apply_minmax_table,
  df = df_list,
  minmax_table = minmax_by_scenario,
  SIMPLIFY = FALSE
)

# ============================================================
# 6) RUN BFAST
# ============================================================

ts_lists <- lapply(df_list_norm, to_ts_list_selected, cols_ts = columns_to_process, freq = 12)

bfast_results <- run_bfast_all(
  ts_lists,
  variables = columns_to_process,
  h = bfast_h,
  max.iter = bfast_max_iter
)

# Build wide results table and save CSV
df_bfast_wide <- results_to_wide_df(bfast_results, normalization_label = "scenario_wise")
print(df_bfast_wide)

write.csv(df_bfast_wide, output_csv, row.names = FALSE)

# ============================================================
# 7) FIGURE 1 — RAW + 12-MONTH ROLLING MEAN + BREAKPOINTS
# ============================================================

# ---- 7.1) Build long table for plotting (robust run_time recreation)
ts_long <- lapply(names(df_list_norm), function(sc) {
  
  df <- df_list_norm[[sc]]
  
  df %>%
    mutate(
      # Recreate run_time locally to avoid any upstream propagation issues
      run_time = (row_number() - 1) / 12 + 1
    ) %>%
    select(run_time, all_of(columns_to_process)) %>%
    pivot_longer(
      cols = all_of(columns_to_process),
      names_to = "variable",
      values_to = "value"
    ) %>%
    mutate(scenario = sc)
  
}) %>% bind_rows()

# ---- 7.2) Add 12-month centered rolling mean (smoothed line)
ts_long_smoothed <- ts_long %>%
  group_by(scenario, variable) %>%
  arrange(run_time) %>%
  mutate(
    value_roll12 = zoo::rollmean(value, k = 12, fill = NA, align = "center")
  ) %>%
  ungroup()

# ---- 7.3) Breakpoint table for vertical lines (run_time)
bp_plot <- lapply(names(bfast_results), function(k) {
  
  parts <- strsplit(k, "_")[[1]]
  scenario <- parts[1]
  variable <- parts[2]
  
  r <- bfast_results[[k]]
  if (is.null(r) || length(r$bp_run_time) == 0) return(NULL)
  
  tibble(
    scenario = scenario,
    variable = variable,
    run_time = r$bp_run_time
  )
  
}) %>% bind_rows()

# Desired scenario order in the plot
scenario_order <- c("regclim", "y7", "y5", "y3", "y1")

ts_long_smoothed <- ts_long_smoothed %>%
  mutate(
    scenario = factor(scenario, levels = scenario_order)
  )

bp_plot <- bp_plot %>%
  mutate(
    scenario = factor(scenario, levels = scenario_order)
  )

# ---- 7.4) Plot
fig1 <- ggplot(ts_long_smoothed, aes(x = run_time)) +
  
  # Raw series (background)
  geom_line(aes(y = value),
            color = "grey70", linewidth = 0.3, alpha = 0.6) +
  
  # Smoothed series (main signal)
  geom_line(aes(y = value_roll12),
            color = "black", linewidth = 0.6) +
  
  # Breakpoint lines
  geom_vline(
    data = bp_plot,
    aes(xintercept = run_time),
    linetype = "dashed",
    linewidth = 0.4,
    color = "grey30"
  ) +
  
  facet_grid(variable ~ scenario, scales = "free_y") +
  
  scale_x_continuous(
    breaks = seq(1, ceiling(max(ts_long_smoothed$run_time, na.rm = TRUE)), by = 5),
    labels = function(x) sprintf("%d", x)
  ) +
  
  labs(
    x = "Simulation year",
    y = "Normalized value",
    title = "Timing of structural changes (raw + 12-month rolling mean)"
  ) +
  
  theme_bw() +
  theme(
    strip.background = element_blank(),
    panel.grid = element_blank(),
    strip.text = element_text(size = 10),
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 10),
    plot.title = element_text(size = 11, face = "bold")
  )

print(fig1)
