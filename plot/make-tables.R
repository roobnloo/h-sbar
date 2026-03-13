# plot/make-tables.R
# Print mean (SE) summary tables comparing H-SBAR vs Chan across simulation settings.
#
# Usage:  Rscript plot/make-tables.R
#         (run from the project root)

library(dplyr)

# ============================================================
# Locate result files
# ============================================================

rds_files <- list.files("sim",
  pattern  = "^run-sim-.*-results\\.rds$",
  full.names = TRUE
)
if (length(rds_files) == 0L) stop("No results .rds files found in sim/")

# ============================================================
# Flatten RDS results into a long data frame
# ============================================================

variants <- c("sbar_s2", "chan_s2")
method_labels <- c(sbar_s2 = "H-SBAR", chan_s2 = "Chan")
scalar_metrics <- c("ncp", "hd", "mse", "beta_err", "sigma2_err")

parse_label <- function(path) {
  base <- sub("^run-sim-(.+)-results\\.rds$", "\\1", basename(path))
  base <- sub("scenario(\\d+)-sigma(.+)", "Scenario \\1  sigma=\\2", base)
  base <- sub("scenario(\\d+)$", "Scenario \\1", base)
  base
}

flatten_results <- function(results, setting_label) {
  rows <- vector("list", length(results) * length(variants))
  idx  <- 1L
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
      row$rep     <- i
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
  ncp        = "# changepoints",
  hd         = "Hausdorff dist.",
  mse        = "Prediction MSE",
  beta_err   = "AR coef. error",
  sigma2_err = "sigma2 error"
)

fmt_cell <- function(vals) {
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0L) return("      -      ")
  m  <- mean(vals)
  se <- sd(vals) / sqrt(length(vals))
  sprintf("%.4f (%.4f)", m, se)
}

col_w    <- 18L
settings <- unique(all_df$setting)

for (s in settings) {
  sub <- all_df[all_df$setting == s, ]
  cat(sprintf("\n%s\n", strrep("=", 60)))
  cat(sprintf("Table: %s\n", s))
  cat(strrep("=", 60), "\n")

  hdr <- sprintf("%-18s %*s %*s", "Metric", col_w, "H-SBAR", col_w, "Chan")
  cat(hdr, "\n")
  cat(strrep("-", nchar(hdr)), "\n")

  for (m in scalar_metrics) {
    cat(sprintf("%-18s %*s %*s\n",
      metric_print_labels[m],
      col_w, fmt_cell(sub[[m]][sub$method == "H-SBAR"]),
      col_w, fmt_cell(sub[[m]][sub$method == "Chan"])
    ))
  }
}

cat("\nDone.\n")
