# test-cv-sbar-cov.R
# Cross-validate SBAR-COV over a path of lambda_n at fixed c_scale,
# using Scenario 1 data: AR(1), two coefficient breaks, constant variance.
#
# Uses the interpolation-based CV method (tscv.md):
#   - equally spaced validation points (spacing > p)
#   - poisoned rows removed from the regression matrix
#   - MSPE as the selection criterion
#
# c_scale is fixed (not cross-validated); only lambda_n is varied.

source("generate-data.R")
source("sbar-cov.R")
source("cv-sbar-cov.R")
source("sbar-cov-bea.R")

# -----------------------------------------------------------------------
# 1. Generate data (Scenario 1)
# -----------------------------------------------------------------------
dat <- generate_scenario1(seed = 42)
cat(
  "True break points: t =", dat$break_points, "\n",
  " phi per regime  :", sapply(dat$phi_list, function(x) x[1L]), "\n",
  " sigma per regime:", dat$sigma_vec, "\n\n"
)

# -----------------------------------------------------------------------
# 2. Cross-validation
#    Vary lambda_n over a log grid; c_scale fixed at 1.
#    val_spacing defaults to max(p+1, round(n/10)) = 30 with n=300, p=1,
#    giving ~9 equally spaced validation points.
# -----------------------------------------------------------------------
lambda_n_path <- 10^seq(-5, -1, length.out = 10)
c_scale_fixed <- 0.1

cat("Running interpolation CV over lambda_n path ...\n")
cat(sprintf(
  "  c_scale fixed at %.4g\n  %d lambda_n values on [%.4g, %.4g]\n\n",
  c_scale_fixed,
  length(lambda_n_path),
  min(lambda_n_path),
  max(lambda_n_path)
))

cv <- cv_sbar_cov(
  y         = dat$Y,
  p         = dat$p,
  lambda_n  = lambda_n_path,
  c_scale   = c_scale_fixed,
  verbose   = TRUE
)

# -----------------------------------------------------------------------
# 3. Report
# -----------------------------------------------------------------------
cat("\nValidation points:", cv$val_points, "\n")
cat(sprintf("Training rows: %d / %d\n\n", length(cv$keep_rows), dat$n))

cat("CV results:\n")
print(cv$cv_table, digits = 4, row.names = FALSE)

cat("\nOptimal lambda_n:\n")
print(cv$best, digits = 4, row.names = FALSE)

# -----------------------------------------------------------------------
# 4. Elbow plot
# -----------------------------------------------------------------------
plot_cv_sbar_cov(cv)

# -----------------------------------------------------------------------
# 5. Refit on full design matrix with CV-selected lambda_n
# -----------------------------------------------------------------------
best_ln <- cv$best$lambda_n

cat(sprintf(
  "\nRefitting on full data: lambda_n=%.4g, c_scale=%.4g ...\n",
  best_ln, c_scale_fixed
))
fit <- sbar_cov(
  y        = dat$Y,
  p        = dat$p,
  lambda_n = best_ln,
  c_scale  = c_scale_fixed
)
cat("Solver status:", fit$status, "\n\n")

# -----------------------------------------------------------------------
# 6. Stage 1 changepoints
# -----------------------------------------------------------------------
cat("Stage 1 — joint break points (cp):      ", fit$cp, "\n")
cat("Stage 1 — coeff break points (cp_theta):", fit$cp_theta, "\n")
cat("Stage 1 — var break points   (cp_psi):  ", fit$cp_psi, "\n")
cat("(True coefficient breaks at t =", dat$break_points, ")\n\n")

# -----------------------------------------------------------------------
# 7. Stage 2: BEA screening
# -----------------------------------------------------------------------
cat("Running BEA screening (Section 9) ...\n")
bea <- sbar_cov_bea(fit, y = dat$Y)
cat("Stage 2 — refined changepoints:", bea$cp, "\n")
cat(sprintf("  omega_n = %.4g\n\n", bea$omega_n))

# Regime summary from BEA
regimes <- sort(unique(c(1L, bea$cp)))
cat("Regime summary after BEA (start | AR1 | sigma2):\n")
for (r in seq_along(regimes)) {
  t_start <- regimes[r]
  t_end <- if (r < length(regimes)) regimes[r + 1L] - 1L else dat$n
  beta_r <- bea$beta[t_start, ]
  sig2_r <- bea$sigma2[t_start]
  cat(sprintf(
    "  t=%d..%d  AR1=%.3f  sigma2=%.3f\n",
    t_start, t_end, beta_r[1L], sig2_r
  ))
}

# -----------------------------------------------------------------------
# 8. Diagnostic plot: series with Stage 1 and Stage 2 break points
# -----------------------------------------------------------------------
plot(dat$Y,
  type = "l", main = "SBAR-COV fit (Scenario 1, CV-selected lambda_n)",
  xlab = "t", ylab = "Y"
)
for (t in dat$break_points) {
  abline(v = t, col = "red", lty = 2, lwd = 2)
}
for (t in fit$cp) {
  abline(v = t, col = "lightgray", lty = 3)
}
for (t in bea$cp) {
  abline(v = t, col = "blue", lty = 2, lwd = 2)
}
legend("topleft",
  legend = c("True break", "Stage 1 (LASSO)", "Stage 2 (BEA)"),
  col = c("red", "lightgray", "blue"),
  lty = c(2, 3, 2), bty = "n"
)
