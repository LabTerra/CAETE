  # ============================================================
  # 1) LIBRARIES
  # ============================================================
  library(dplyr)
  library(bfast)
  library(zoo)       # as.yearmon(), rollmean()
  library(tidyr)     # pivot_longer(), pivot_wider()
  library(ggplot2)
  library(showtext)
  library(sysfonts)
  
  # ============================================================
  # 2) PATHS AND USER SETTINGS
  # ============================================================
  path_file <- "/Users/biancarius/Desktop/CAETE-DVM-alloc-allom-including_alloc2_Cm2/scripts/monthly_mean_tables/"
  
  files <- list(
    regclim = "MAN_regularclimate_monthly.csv",
    y8      = "MAN_30prec_8y_monthly.csv",
    y6      = "MAN_30prec_6y_monthly.csv",
    y4      = "MAN_30prec_4y_monthly.csv",
    y2      = "MAN_30prec_2y_monthly.csv"
  )
  
  # Variables to analyze
  columns_to_process <- c("npp", "ctotal", "evapm")
  
  # BFAST settings (keep fixed for fair comparisons)
  bfast_h <- 0.2
  bfast_max_iter <- 1
  
  # ------------------------------------------------------------
  # Optional: truncate one scenario due to "collapse"
  # ------------------------------------------------------------
  collapse_scenario <- "y2"     
  collapse_ym <- "2006-12"      # keep collapse month included
  
  # ============================================================
  # 3) HELPER FUNCTIONS
  # ============================================================
  
  # ---- 3.1) Add simulation time (run_year starts at 0)
  add_run_time <- function(df) {
    df %>%
      mutate(date = as.Date(zoo::as.yearmon(date))) %>%
      arrange(date) %>%
      mutate(
        month_index = row_number(),                      # 1, 2, 3, ...
        run_year    = floor((month_index - 1) / 12),     # 0, 1, 2, ...
        run_time    = (month_index - 1) / 12             # 0.0, 0.083, 0.167, ...
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
  
  # ---- 3.3) GLOBAL min/max table
  compute_minmax_global <- function(df_list, vars) {
    # Junta todos os cenários num único dataframe para achar o Min e Max global
    df_all <- bind_rows(df_list)
    vars_present <- intersect(vars, names(df_all))
    
    out <- lapply(vars_present, function(v) {
      x <- as.numeric(df_all[[v]])
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
      ts(as.numeric(df[[v]]), start = c(0, 1), frequency = freq)  # start at simulation year 1
    })
    setNames(ts_list, cols_present)
  }
  
  # ---- 3.6) Run BFAST and extract breakpoints, slopes, magnitudes
  run_bfast_summary <- function(ts_obj, h = 0.15, max.iter = 1) {
    
    res <- bfast(ts_obj, h = h, max.iter = max.iter)
    niter <- length(res$output)
    out <- res$output[[niter]]
    
    bp_idx <- breakpoints(out$bp.Vt)$breakpoints
    bp_idx <- bp_idx[!is.na(bp_idx)]
    
    slopes <- coef(out$bp.Vt)[, 2]
    
    mags <- res$Mags
    if (is.null(mags)) mags <- numeric(0)
    
    bp_run_time <- if (length(bp_idx) > 0) (bp_idx - 1) / 12 else numeric(0)
    bp_run_year <- if (length(bp_idx) > 0) floor((bp_idx - 1) / 12) else integer(0)
    
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
  
  # ---- 3.8) Build wide table
  results_to_wide_df <- function(results_list, normalization_label = "global_wise") {
    
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
      
      if (!is.null(r) && length(r$bp_idx) > 0) {
        for (i in seq_along(r$bp_idx)) {
          cols[[paste0("bp", i, "_run_time")]] <- r$bp_run_time[i]
          cols[[paste0("bp", i, "_run_year")]] <- r$bp_run_year[i]
        }
      }
      
      if (!is.null(r) && length(r$mags) > 0) {
        for (i in seq_along(r$mags)) {
          cols[[paste0("mag", i)]] <- r$mags[i]
        }
      }
      
      if (!is.null(r) && length(r$slopes) > 0) {
        for (i in seq_along(r$slopes)) {
          cols[[paste0("slope", i)]] <- r$slopes[i]
        }
      }
      
      rows[[j]] <- as_tibble(cols)
    }
    
    out <- bind_rows(rows)
    
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
  
  df_list <- lapply(files, function(fname) {
    read.csv(file.path(path_file, fname)) %>% add_run_time()
  })
  
  if (!is.null(collapse_scenario) && collapse_scenario %in% names(df_list)) {
    collapse_date <- as.Date(zoo::as.yearmon(collapse_ym))
    df_list[[collapse_scenario]] <- df_list[[collapse_scenario]] %>%
      filter(date <= collapse_date)
  }
  
  # ============================================================
  # 5) GLOBAL NORMALIZATION
  # ============================================================
  
  # Calcula a tabela de min/max globais
  minmax_global <- compute_minmax_global(df_list, columns_to_process)
  
  # Aplica a mesma tabela global para todos os cenários
  df_list_norm <- lapply(df_list, function(df) {
    apply_minmax_table(df, minmax_global)
  })
  
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
  
  df_bfast_wide <- results_to_wide_df(bfast_results, normalization_label = "global_wise")
  #write.csv(df_bfast_wide, "bfast_bp_slopes_poster_japao.csv", row.names = FALSE)
  
  # ============================================================
  # Helper: build piecewise trend segments (slopes) for plotting
  # ============================================================
  build_slope_segments <- function(bfast_results, scenarios_keep, freq = 12) {
    
    segs <- lapply(names(bfast_results), function(k) {
      
      parts <- strsplit(k, "_")[[1]]
      sc <- parts[1]
      var <- parts[2]
      
      if (!sc %in% scenarios_keep) return(NULL)
      
      r <- bfast_results[[k]]
      if (is.null(r) || is.null(r$bfast_object)) return(NULL)
      
      res <- r$bfast_object
      out <- res$output[[length(res$output)]]
      
      yhat <- tryCatch(as.numeric(fitted(out$bp.Vt)), error = function(e) NA_real_)
      if (all(is.na(yhat))) return(NULL)
      
      n <- length(yhat)
      bp_idx <- r$bp_idx
      bp_idx <- bp_idx[!is.na(bp_idx)]
      
      start_idx <- c(1, bp_idx + 1)
      end_idx   <- c(bp_idx, n)
      
      t_start <- (start_idx - 1) / freq
      t_end   <- (end_idx - 1) / freq
      
      y_start <- yhat[start_idx]
      y_end   <- yhat[end_idx]
      
      dt <- (t_end - t_start)
      slope_per_year <- ifelse(dt > 0, (y_end - y_start) / dt, NA_real_)
      
      tibble(
        scenario = sc,
        variable = var,
        seg_id   = seq_along(start_idx),
        t_start  = t_start,
        t_end    = t_end,
        y_start  = y_start,
        y_end    = y_end,
        slope_per_year = slope_per_year,
        x_label = (t_start + t_end) / 2,
        y_label = (y_start + y_end) / 2
      )
    })
    
    bind_rows(segs)
  }
  
  
  # ============================================================
  # 7) FIGURE 1 — RAW + 12-MONTH ROLLING MEAN + BREAKPOINTS
  # ============================================================
  variable_labels <- c(
    npp    = "NPP",
    ctotal = "Total carbon",
    evapm  = "Evap."
  )
  
  scenario_order <- c("y8","y2")
  scenario_labels <- c("8 years", "2 years")
  
  # ---- 7.1) Build long table for plotting
  ts_long <- lapply(scenario_order, function(sc) {
    
    df <- df_list_norm[[sc]]
    
    df %>%
      mutate(run_time = (row_number() - 1) / 12) %>%
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
  
  # ---- 7.3) Breakpoint table
  bp_plot <- lapply(names(bfast_results), function(k) {
    parts <- strsplit(k, "_")[[1]]
    scenario <- parts[1]
    variable <- parts[2]
    
    if (!scenario %in% scenario_order) return(NULL)
    
    r <- bfast_results[[k]]
    if (is.null(r) || length(r$bp_run_time) == 0) return(NULL)
    
    tibble(scenario = scenario, variable = variable, run_time = r$bp_run_time)
  }) %>% bind_rows()
  
  # ---- 7.3b) Build slope segments
  slope_segments <- build_slope_segments(
    bfast_results = bfast_results,
    scenarios_keep = scenario_order,
    freq = 12
  )
  
  slope_segments <- slope_segments %>%
    mutate(
      scenario = factor(scenario, levels = scenario_order, labels = scenario_labels),
      variable = factor(variable, levels = names(variable_labels), labels = variable_labels)
    )
  
  # ---- 7.4) Apply factor ordering + labels
  ts_long_smoothed <- ts_long_smoothed %>%
    mutate(
      scenario = factor(scenario, levels = scenario_order, labels = scenario_labels),
      variable = factor(variable, levels = names(variable_labels), labels = variable_labels)
    )
  
  bp_plot <- bp_plot %>%
    mutate(
      scenario = factor(scenario, levels = scenario_order, labels = scenario_labels),
      variable = factor(variable, levels = names(variable_labels), labels = variable_labels)
    )
  
  # ============================================================
  # 7.5) Plot (OVERLAY frequencies) - LAYOUT CUSTOMIZADO (PATCHWORK)
  # ============================================================
  library(patchwork)
  
  freq_colors <- c(
    "8 years" = "#3B5BA9",  # blue
    "2 years" = "#D55E00"   # orange
  )
  
  # 1. Função auxiliar para desenhar um único painel (com controle de Eixo Y compartilhado)
  plot_single_var <- function(var_chosen, title_text, show_y = TRUE) {
    p <- ggplot() +
      # Linha da média móvel
      geom_line(
        data = ts_long_smoothed %>% filter(variable == var_chosen),
        aes(x = run_time, y = value_roll12, color = scenario, group = scenario, linetype = "Rolling mean"),
        linewidth = 1.3, alpha = 0.5
      ) +
      # Segmentos de tendência (slopes)
      geom_segment(
        data = slope_segments %>% filter(variable == var_chosen),
        aes(x = t_start, xend = t_end, y = y_start, yend = y_end,
            color = scenario, linetype = "Piecewise slope"),
        linewidth = 1.0, alpha = 1.0
      ) +
      # Breakpoints
      geom_vline(
        data = bp_plot %>% filter(variable == var_chosen),
        aes(xintercept = run_time, color = scenario, linetype = "Breakpoint"),
        linewidth = 0.9, alpha = 1
      ) +
      # Escalas
      scale_color_manual(name = "Drought frequency", values = freq_colors) +
      scale_linetype_manual(
        name = NULL,
        values = c("Rolling mean" = "solid", "Piecewise slope" = "solid", "Breakpoint" = "dotted")
      ) +
      # MUDANÇA: Limites fixos de 0 a 1 para compartilhar a mesma proporção em todos
      scale_y_continuous(breaks = scales::breaks_width(0.25), limits = c(0, 1), name = "Normalized value") +
      scale_x_continuous(
        breaks = seq(0, ceiling(max(ts_long_smoothed$run_time, na.rm = TRUE)), by = 5),
        labels = function(x) sprintf("%d", x)
      ) +
      labs(x = "Simulation year", y = "Normalized value", title = title_text) +
      # Tema limpo
      theme_bw(base_family = "Helvetica") +
      theme(
        text = element_text(family = "Helvetica"),
        panel.grid = element_blank(),
        axis.text.x = element_text(size = 13),
        axis.title.x = element_text(size = 14),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
      )
    
    # Se show_y for FALSE, removemos os números e o título do eixo Y (para a Evap)
    if (!show_y) {
      p <- p + theme(
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()
      )
    } else {
      p <- p + theme(
        axis.text.y = element_text(size = 13),
        axis.title.y = element_text(size = 14)
      )
    }
    
    return(p)
  }
  
  # 2. Gerar os 3 gráficos isolados
  p_npp    <- plot_single_var("NPP", "NPP", show_y = TRUE)
  p_evap   <- plot_single_var("Evap.", "Evapotranspiration", show_y = FALSE) # <--- Y desligado para "colar" no NPP
  p_carbon <- plot_single_var("Total carbon", "Total Carbon", show_y = TRUE)
  
  # 3. Desenhar o Layout do Patchwork
  # A = NPP, B = Evap, C = Carbono, # = Espaço vazio
  design <- "
    AABB
    #CC#
  "
  
  # 4. Juntar tudo em uma figura só
  fig1_overlay <- p_npp + p_evap + p_carbon + 
    plot_layout(design = design, guides = "collect") & 
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 15, face = "bold")
    )
  
print(fig1_overlay)