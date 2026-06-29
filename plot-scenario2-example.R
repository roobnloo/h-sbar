# plot-scenario2-example.R
# Time plot of one scenario 2 rep with stage 1 (pre-BEA) and stage 2 (BEA)
# changepoint estimates overlaid as vertical lines.
#
# Usage:  Rscript plot-scenario2-example.R [seed] [outfile]
#   seed     RNG seed for the rep (default 42)
#   outfile  Output PDF path (default scenario2-example.pdf)

library(ggplot2)

source("generate-data.R")
source("hsbar.R")
source("cv-hsbar.R")
source("hsbar-bea.R")

args <- commandArgs(trailingOnly = TRUE)
SEED <- if (length(args) >= 1L) as.integer(args[[1L]]) else 42L
outfile <- if (length(args) >= 2L) args[[2L]] else "sim/scenario2-example.pdf"

P <- 2L
LAMBDA <- seq(0.01, 0.15, by = 0.01)
C_SCALE <- 1

# ---- Generate data ----------------------------------------------------------
dat <- generate_scenario2(seed = SEED)
y <- dat$Y
n <- dat$n

cat(sprintf("Scenario 2  n=%d  seed=%d\n", n, SEED))
cat(sprintf("True breaks: %s\n", paste(dat$break_points, collapse = ", ")))

# ---- Stage 1: CV + FISTA fit ------------------------------------------------
cat("Running CV-HSBAR...\n")
cv_fit <- cv_hsbar(y, p = P, lambda = LAMBDA, c_scale = C_SCALE, verbose = FALSE)
best_lambda <- cv_fit$best$lambda

cat(sprintf("Best lambda: %.4f\n", best_lambda))

sbar_fit <- hsbar(y, p = P, lambda = best_lambda, c_scale = C_SCALE)
cp_s1 <- sort(sbar_fit$cp)
cat(sprintf("Stage 1 CPs: %s\n", if (length(cp_s1) == 0L) "(none)" else paste(cp_s1, collapse = ", ")))

# ---- Stage 2: BEA pruning ---------------------------------------------------
bea_fit <- hsbar_bea(sbar_fit, y, p = P)
cp_s2 <- sort(bea_fit$cp)
cat(sprintf("Stage 2 CPs: %s\n", if (length(cp_s2) == 0L) "(none)" else paste(cp_s2, collapse = ", ")))

# ---- Plot -------------------------------------------------------------------
df <- data.frame(t = seq_len(n), y = y)

p_plot <- ggplot(df, aes(x = t, y = y)) +
  geom_line(linewidth = 0.3, color = "black") +
  labs(
    # title = "Scenario 2",
    x     = "Time",
    y     = expression(Y[t])
  ) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank())

p_plot <- p_plot +
  geom_vline(
    xintercept = dat$break_points,
    linetype   = "solid",
    color      = "black",
    linewidth  = 0.4
  )

if (length(cp_s1) > 0L) {
  p_plot <- p_plot +
    geom_vline(
      xintercept = cp_s1,
      linetype   = "dashed",
      color      = "#afafaf",
      linewidth  = 0.4
    )
}

if (length(cp_s2) > 0L) {
  p_plot <- p_plot +
    geom_vline(
      xintercept = cp_s2,
      linetype   = "dashed",
      color      = "red",
      linewidth  = 0.4
    )
}

# Legend annotation (manual)
x_max <- n
y_range <- range(y)
y_top <- y_range[2L]
ann_df <- data.frame(
  x = rep(0.02 * x_max, 3L),
  y = y_top - c(0, 0.07, 0.14) * diff(y_range)
)
p_plot <- p_plot +
  annotate("segment",
    x = ann_df$x[1L] - 0.01 * x_max, xend = ann_df$x[1L] + 0.01 * x_max,
    y = ann_df$y[1L], yend = ann_df$y[1L],
    linetype = "solid", color = "black", linewidth = 0.4
  ) +
  annotate("text",
    x = ann_df$x[1L] + 0.015 * x_max, y = ann_df$y[1L],
    label = "Truth", hjust = 0, size = 3, color = "black"
  ) +
  annotate("segment",
    x = ann_df$x[2L] - 0.01 * x_max, xend = ann_df$x[2L] + 0.01 * x_max,
    y = ann_df$y[2L], yend = ann_df$y[2L],
    linetype = "dashed", color = "#afafaf", linewidth = 0.4
  ) +
  annotate("text",
    x = ann_df$x[2L] + 0.015 * x_max, y = ann_df$y[2L],
    label = "Stage 1 (pre-BEA)", hjust = 0, size = 3, color = "gray40"
  ) +
  annotate("segment",
    x = ann_df$x[3L] - 0.01 * x_max, xend = ann_df$x[3L] + 0.01 * x_max,
    y = ann_df$y[3L], yend = ann_df$y[3L],
    linetype = "dashed", color = "red", linewidth = 0.4
  ) +
  annotate("text",
    x = ann_df$x[3L] + 0.015 * x_max, y = ann_df$y[3L],
    label = "Stage 2 (BEA)", hjust = 0, size = 3, color = "red"
  )

ggsave(outfile, p_plot, width = 10, height = 3.5, device = "pdf")
cat(sprintf("Saved %s\n", outfile))
