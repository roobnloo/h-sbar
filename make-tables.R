# make-tables.R
# Print mean (SE) summary tables comparing H-SBAR vs Chan across simulation settings.
#
# Usage:  Rscript make-tables.R <sim_dir>
#         (sim_dir defaults to "sim" if not provided)

library(dplyr)

# ============================================================
# Parse arguments
# ============================================================

args <- commandArgs(trailingOnly = TRUE)
sim_dir <- if (length(args) >= 1L) args[[1L]] else "sim"
if (!dir.exists(sim_dir)) stop(sprintf("Directory not found: %s", sim_dir))

# ============================================================
# Locate result files
# ============================================================

rds_files <- list.files(sim_dir,
  pattern = "^run-sim-.*-results\\.rds$",
  full.names = TRUE
)
if (length(rds_files) == 0L) stop(sprintf("No results .rds files found in %s", sim_dir))

# ============================================================
# Flatten RDS results into a long data frame
# ============================================================

variants <- c("sbar_s2", "chan_s2_rss", "chan_s2_sigma", "chan_s2_prof")
method_labels <- c(
  sbar_s2       = "H-SBAR",
  chan_s2_rss   = "Chan RSS",
  chan_s2_sigma = "Chan sigma",
  chan_s2_prof  = "Chan profile"
)
scalar_metrics <- c("correct_ncp", "hd", # "mse",
                    "beta_err", "sigma2_err")

parse_label <- function(path) {
  base <- sub("^run-sim-(.+)-results\\.rds$", "\\1", basename(path))
  # e.g. "scenario1-sigma0.5"    -> "Scenario 1  sigma=0.5"
  #      "scenario4-sigscale1.5" -> "Scenario 4  sigscale=1.5"
  #      "scenario4"             -> "Scenario 4"
  base <- sub("scenario(\\d+)-(sigma|sigscale)(.+)", "Scenario \\1  \\2=\\3", base, perl = TRUE)
  base <- sub("scenario(\\d+)$", "Scenario \\1", base)
  base
}

flatten_results <- function(results, setting_label) {
  rows <- vector("list", length(results) * length(variants))
  idx <- 1L
  for (i in seq_along(results)) {
    rep <- results[[i]]
    for (v in variants) {
      entry <- rep[[v]]
      if (is.null(entry)) {
        row <- as.list(rep(NA_real_, length(scalar_metrics)))
        names(row) <- scalar_metrics
      } else {
        row <- lapply(scalar_metrics, function(m) as.numeric(entry[[m]]))
        names(row) <- scalar_metrics
      }
      row$rep <- i
      row$variant <- v
      row$setting <- setting_label
      rows[[idx]] <- row
      idx <- idx + 1L
    }
  }
  df <- bind_rows(rows)
  df$method <- method_labels[df$variant]
  df
}

all_df <- bind_rows(lapply(rds_files, function(f) {
  flatten_results(readRDS(f), parse_label(f))
}))

# ============================================================
# Print summary tables: mean (SE) for each metric x method
# ============================================================

metric_print_labels <- c(
  correct_ncp = "Prop. correct m",
  hd          = "Hausdorff dist.",
  mse         = "Prediction MSE",
  beta_err    = "AR coef. error",
  sigma2_err  = "sigma2 error"
)

fmt_cell <- function(vals, is_prop = FALSE) {
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0L) {
    return("      -      ")
  }
  m <- mean(vals)
  se <- sd(vals) / sqrt(length(vals))
  if (is_prop) sprintf("%.3f (%.3f)", m, se) else sprintf("%.4f (%.4f)", m, se)
}

col_w <- 18L
method_names <- unname(method_labels)
settings <- unique(all_df$setting)

for (s in settings) {
  sub <- all_df[all_df$setting == s, ]
  cat(sprintf("\n%s\n", strrep("=", 60 + col_w * (length(method_names) - 1L))))
  cat(sprintf("Table: %s\n", s))
  cat(strrep("=", 60 + col_w * (length(method_names) - 1L)), "\n")

  hdr <- sprintf("%-18s", "Metric")
  for (mn in method_names) hdr <- paste0(hdr, sprintf(" %*s", col_w, mn))
  cat(hdr, "\n")
  cat(strrep("-", nchar(hdr)), "\n")

  for (m in scalar_metrics) {
    row <- sprintf("%-18s", metric_print_labels[m])
    for (mn in method_names) row <- paste0(row, sprintf(" %*s", col_w, fmt_cell(sub[[m]][sub$method == mn], is_prop = (m == "correct_ncp"))))
    cat(row, "\n")
  }
}

cat("\nDone.\n")
