# test-cv-chan-sbar.R
# Cross-validate Chan et al. (2014) SBAR over a path of lambda,
# using Scenario 5 data: AR(1), two coefficient breaks, constant variance.
#
# Uses the interpolation-based CV method (tscv.md):
#   - equally spaced validation points (spacing > p)
#   - poisoned rows removed from the regression matrix
#   - MSPE as the selection criterion
#
# No c_scale parameter: Chan's model has AR coefficients only (no variance changes).

source("generate-data.R")
source("chan-sbar-bea.R")
source("cv-chan-sbar.R")

c_omega_n <- 1 # scale BEA penalty
ic_type <- "sigma_scaled"

# -----------------------------------------------------------------------
# 1. Generate data
# -----------------------------------------------------------------------
dat <- generate_scenario3(seed = 1)
cat(
  "True break points: t =", dat$break_points, "\n",
  " phi per regime  :", sapply(dat$phi_list, function(x) x[1L]), "\n",
  " sigma per regime:", dat$sigma_vec, "\n\n"
)

# -----------------------------------------------------------------------
# 2. Cross-validation
#    Vary lambda over a log grid.
#    val_spacing defaults to max(p+1, round(n/10)) = 30 with n=300, p=1,
#    giving ~9 equally spaced validation points.
# -----------------------------------------------------------------------
# lambda_path <- 10^seq(-3, -1, length.out = 100)
lambda_path <- seq(0.01, 0.1, by = 0.01)

cat("Running interpolation CV over lambda path ...\n")
cat(sprintf(
  "  %d lambda values on [%.4g, %.4g]\n\n",
  length(lambda_path),
  min(lambda_path),
  max(lambda_path)
))

cv <- cv_chan_sbar(
  y = dat$Y,
  p = dat$p,
  lambda = lambda_path,
  max_iter = 5000,
  val_spacing = 10,
  eps_tol = 1e-10,
  verbose = TRUE
)

# -----------------------------------------------------------------------
# 3. Report
# -----------------------------------------------------------------------
cat("\nValidation points:", cv$val_points, "\n")
cat(sprintf("Training rows: %d / %d\n\n", length(cv$keep_rows), dat$n))

cat("CV results:\n")
print(cv$cv_table, digits = 4, row.names = FALSE)

cat("\nOptimal lambda:\n")
print(cv$best, digits = 4, row.names = FALSE)

# -----------------------------------------------------------------------
# 4. Elbow plot
# -----------------------------------------------------------------------
plot_cv_chan_sbar(cv)

# -----------------------------------------------------------------------
# 5. Refit on full design matrix with CV-selected lambda
# -----------------------------------------------------------------------
best_ln <- cv$best$lambda

cat(sprintf(
  "\nRefitting on full data: lambda=%.4g ...\n",
  best_ln
))
fit <- chan_sbar_admm(
  y      = dat$Y,
  p      = dat$p,
  lambda = best_ln
)
cat("Converged:", fit$converged, "\n\n")

# -----------------------------------------------------------------------
# 6. Stage 1 changepoints
# -----------------------------------------------------------------------
cat("Stage 1 — changepoints (cp):", fit$cp, "\n")
cat("(True coefficient breaks at t =", dat$break_points, ")\n\n")

# -----------------------------------------------------------------------
# 7. Stage 2: BEA screening under all three IC criteria
# -----------------------------------------------------------------------
ic_types <- c("rss", "sigma_scaled", "profile_lik")
ic_labels <- c("RSS", "Sigma-scaled", "Profile likelihood")

cat("Running BEA screening (Chan 2014 Section 2.2) ...\n\n")
bea_list <- lapply(ic_types, function(ic) {
  chan_sbar_bea(fit, y = dat$Y, c_omega_n = c_omega_n, ic_type = ic)
})

for (i in seq_along(ic_types)) {
  b <- bea_list[[i]]
  cat(sprintf("IC: %s\n", ic_labels[i]))
  cat("  Stage 2 — refined changepoints:", if (length(b$cp) == 0L) "none" else b$cp, "\n")
  cat(sprintf("  omega_n = %.4g\n", b$omega_n))
  regimes <- sort(unique(c(1L, b$cp)))
  for (r in seq_along(regimes)) {
    t_start <- regimes[r]
    t_end <- if (r < length(regimes)) regimes[r + 1L] - 1L else dat$n
    cat(sprintf(
      "  t=%d..%d  AR1=%.3f  sigma2=%.3f\n",
      t_start, t_end, b$beta[t_start, 1L], b$sigma2[t_start]
    ))
  }
  cat("\n")
}

# -----------------------------------------------------------------------
# 8. Diagnostic plots: series with Stage 1 and Stage 2 break points
# -----------------------------------------------------------------------
op <- par(mfrow = c(3L, 1L), mar = c(3, 4, 2, 1))
for (i in seq_along(ic_types)) {
  b <- bea_list[[i]]
  plot(dat$Y,
    type = "l",
    main = sprintf(
      "BEA — %s  (cp: %s)",
      ic_labels[i],
      if (length(b$cp) == 0L) "none" else paste(b$cp, collapse = ", ")
    ),
    xlab = "t", ylab = "Y"
  )
  for (t in dat$break_points) abline(v = t, col = "red", lty = 2, lwd = 2)
  for (t in fit$cp) abline(v = t, col = "darkgray", lty = 3)
  for (t in b$cp) abline(v = t, col = "blue", lty = 2, lwd = 2)
  if (i == 1L) {
    legend("topleft",
      legend = c("True break", "Stage 1 (LASSO)", "Stage 2 (BEA)"),
      col = c("red", "darkgray", "blue"),
      lty = c(2, 3, 2), bty = "n", cex = 0.8
    )
  }
}
par(op)
