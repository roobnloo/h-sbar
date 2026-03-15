# plot/make-plots.R
# Generate ggplot boxplots comparing H-SBAR vs Chan SBAR across simulation settings.
#
# For each setting (scenario/sigma combination), produces one figure with
# multiple metric panels (facets). Each panel has two boxplots: H-SBAR and Chan.
# Only Stage 2 (post-BEA) results are shown.
#
# Output: plot/plots-<label>.pdf  (one file per setting)
#
# Usage:  Rscript plot/make-plots.R
#         (run from the project root)

library(ggplot2)
library(dplyr)
library(tidyr)

# ============================================================
# Locate result files
# ============================================================

rds_files <- list.files("sim",
  pattern = "^run-sim-.*-results\\.rds$",
  full.names = TRUE
)
if (length(rds_files) == 0L) stop("No results .rds files found in sim/")

# ============================================================
# Flatten one results list into a long data frame
# ============================================================

VARIANTS <- c("sbar_s2", "chan_s2_rss", "chan_s2_sigma", "chan_s2_prof")
METHOD_LABELS <- c(
  sbar_s2 = "H-SBAR",
  chan_s2_rss = "Chan RSS",
  chan_s2_sigma = "Chan sigma",
  chan_s2_prof = "Chan profile"
)

SCALAR_METRICS <- c("ncp", "hd", "mse", "beta_err", "sigma2_err")
METRIC_LABELS <- c(
  ncp        = "# changepoints",
  hd         = "log(1 + Hausdorff distance)",
  mse        = "Prediction MSE",
  beta_err   = "AR coef. error",
  sigma2_err = expression(sigma^2 ~ error)
)

flatten_results <- function(results, setting_label) {
  rows <- vector("list", length(results) * length(VARIANTS))
  idx <- 1L
  for (i in seq_along(results)) {
    rep <- results[[i]]
    for (v in VARIANTS) {
      entry <- rep[[v]]
      if (is.null(entry)) {
        row <- as.list(rep(NA_real_, length(SCALAR_METRICS)))
        names(row) <- SCALAR_METRICS
      } else {
        row <- lapply(SCALAR_METRICS, function(m) as.numeric(entry[[m]]))
        names(row) <- SCALAR_METRICS
      }
      row$rep <- i
      row$variant <- v
      row$setting <- setting_label
      rows[[idx]] <- row
      idx <- idx + 1L
    }
  }
  df <- bind_rows(rows)
  df$method <- METHOD_LABELS[df$variant]
  df
}

# ============================================================
# Parse setting label from filename
# ============================================================

parse_label <- function(path) {
  base <- sub("^run-sim-(.+)-results\\.rds$", "\\1", basename(path))
  # e.g. "scenario1-sigma0.5"    -> "Scenario 1  sigma=0.5"
  #      "scenario4-sigscale1.5" -> "Scenario 4  sigscale=1.5"
  #      "scenario4"             -> "Scenario 4"
  base <- sub("scenario(\\d+)-(sigma|sigscale)(.+)", "Scenario \\1  \\2=\\3", base, perl = TRUE)
  base <- sub("scenario(\\d+)$", "Scenario \\1", base)
  base
}

# ============================================================
# Build full data frame
# ============================================================

all_df <- bind_rows(lapply(rds_files, function(f) {
  label <- parse_label(f)
  results <- readRDS(f)
  flatten_results(results, label)
}))

# Parse scenario and sigma_label for scenario-level plots
all_df$scenario    <- sub("^(Scenario \\d+).*$", "\\1", all_df$setting)
all_df$sigma_label <- trimws(sub("^Scenario \\d+\\s*", "", all_df$setting))
all_df$sigma_label[all_df$sigma_label == ""] <- "(base)"

# ============================================================
# Plotting helpers
# ============================================================

METHOD_COLORS <- c(
  "H-SBAR"       = "#2166AC",
  "Chan RSS"     = "#D6604D",
  "Chan sigma"   = "#4DAC26",
  "Chan profile" = "#8E44AD"
)

all_df$method <- factor(all_df$method, levels = names(METHOD_COLORS))

make_boxplot <- function(df, metric, y_label, setting) {
  sub_df <- df[, c("method", metric)]
  colnames(sub_df)[2] <- "value"
  sub_df <- sub_df[!is.na(sub_df$value), ]
  if (metric == "hd") sub_df$value <- log1p(sub_df$value)

  ggplot(sub_df, aes(x = method, y = value, fill = method)) +
    geom_boxplot(outlier.size = 0.8, linewidth = 0.4, width = 0.55) +
    scale_fill_manual(values = METHOD_COLORS, guide = "none") +
    labs(
      title = setting,
      x     = NULL,
      y     = y_label
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(angle = 20, hjust = 1, size = 9),
      panel.grid.minor = element_blank()
    )
}

make_combined_plot <- function(df, setting) {
  plot_list <- lapply(seq_along(SCALAR_METRICS), function(i) {
    m <- SCALAR_METRICS[i]
    lbl <- if (is.character(METRIC_LABELS[[m]])) METRIC_LABELS[[m]] else m
    make_boxplot(df, m, lbl, if (i == 1L) setting else "")
  })

  # Use patchwork if available, otherwise cowplot, otherwise base layout
  if (requireNamespace("patchwork", quietly = TRUE)) {
    library(patchwork)
    wrap_plots(plot_list, nrow = 1L) +
      plot_annotation(
        title    = setting,
        subtitle = "Boxplots over 100 replications"
      )
  } else if (requireNamespace("gridExtra", quietly = TRUE)) {
    library(gridExtra)
    gridExtra::arrangeGrob(
      grobs = plot_list, nrow = 1L,
      top = grid::textGrob(setting, gp = grid::gpar(fontsize = 13, fontface = "bold"))
    )
  } else {
    stop("Install 'patchwork' or 'gridExtra' to combine panels.")
  }
}

# ============================================================
# Produce one PDF per setting
# ============================================================

settings <- unique(all_df$setting)
out_dir <- "plot"
if (!dir.exists(out_dir)) dir.create(out_dir)

for (s in settings) {
  sub <- all_df[all_df$setting == s, ]
  p <- make_combined_plot(sub, s)

  fname <- file.path(
    out_dir,
    paste0("plots-", gsub("[[:space:]]+", "_", gsub("=", "", s)), ".pdf")
  )
  ggsave(fname, p, width = 14, height = 4, device = "pdf")
  cat(sprintf("Saved %s\n", fname))
}

# ============================================================
# Also save a single combined PDF with all settings stacked
# ============================================================

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  all_plots <- lapply(settings, function(s) {
    make_combined_plot(all_df[all_df$setting == s, ], s)
  })
  combined <- Reduce(`/`, all_plots) # stack vertically with patchwork
  out_all <- file.path(out_dir, "plots-all.pdf")
  ggsave(out_all, combined,
    width = 14, height = 3 * length(settings), device = "pdf",
    limitsize = FALSE
  )
  cat(sprintf("Saved %s\n", out_all))
}

cat("\nDone.\n")

# ============================================================
# Scenario-level plots: log(1+hd) and AR coef error,
# grouped by sigma / sigscale on the x-axis
# ============================================================

SCENARIO_METRICS <- c("hd", "beta_err")
SCENARIO_METRIC_LABELS <- c(
  hd       = "log(1 + Hausdorff distance)",
  beta_err = "AR coef. error"
)

sigma_level_order <- function(labels) {
  u    <- unique(labels)
  nums <- suppressWarnings(as.numeric(gsub(".*=", "", u)))
  u[order(nums, na.last = TRUE)]
}

make_scenario_plot <- function(df_scen, scenario_name) {
  df_scen$sigma_label <- factor(
    df_scen$sigma_label,
    levels = sigma_level_order(unique(df_scen$sigma_label))
  )

  long <- tidyr::pivot_longer(df_scen, cols = all_of(SCENARIO_METRICS),
                              names_to = "metric", values_to = "value")
  long <- long[!is.na(long$value), ]
  long$value[long$metric == "hd"] <- log1p(long$value[long$metric == "hd"])
  long$metric <- factor(long$metric,
                        levels = SCENARIO_METRICS,
                        labels = unname(SCENARIO_METRIC_LABELS[SCENARIO_METRICS]))

  ggplot(long, aes(x = sigma_label, y = value, fill = method)) +
    geom_boxplot(outlier.size = 0.5, linewidth = 0.35, width = 0.7,
                 position = position_dodge(0.85)) +
    scale_fill_manual(values = METHOD_COLORS, name = NULL) +
    facet_wrap(~ metric, ncol = 1L, scales = "free_y") +
    labs(title = scenario_name, x = NULL, y = NULL) +
    theme_bw(base_size = 11) +
    theme(
      plot.title       = element_text(size = 10, face = "bold"),
      legend.position  = "right",
      axis.text.x      = element_text(angle = 20, hjust = 1, size = 8),
      panel.grid.minor = element_blank(),
      strip.text       = element_text(size = 9)
    )
}

scenarios <- unique(all_df$scenario)

for (sc in scenarios) {
  sub <- all_df[all_df$scenario == sc, ]
  p   <- make_scenario_plot(sub, sc)
  fname <- file.path(out_dir,
                     paste0("plots-scenario-", gsub("[[:space:]]+", "_", sc), ".pdf"))
  ggsave(fname, p, width = 10, height = 6, device = "pdf")
  cat(sprintf("Saved %s\n", fname))
}

if (requireNamespace("patchwork", quietly = TRUE)) {
  scen_plots   <- lapply(scenarios, function(sc) {
    make_scenario_plot(all_df[all_df$scenario == sc, ], sc)
  })
  combined_scen <- Reduce(`/`, scen_plots)
  out_scen_all  <- file.path(out_dir, "plots-scenarios-all.pdf")
  ggsave(out_scen_all, combined_scen,
    width = 10, height = 6 * length(scenarios), device = "pdf",
    limitsize = FALSE
  )
  cat(sprintf("Saved %s\n", out_scen_all))
}

cat("\nDone (scenario plots).\n")
