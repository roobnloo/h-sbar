# test-hsbar.R
# Test H-SBAR on Scenario 1: baseline AR(1) with two coefficient breaks
# and constant variance (no variance change).
#
# True setup: T=300, p=1, breaks at t=100 and t=200.
# phi: -0.6 -> 0.75 -> -0.8; sigma constant at 0.1.

source("generate-data.R")
source("hsbar.R")
source("hsbar-bea.R")

# -----------------------------------------------------------------------
# 1. Generate data (Scenario 1)
# -----------------------------------------------------------------------
dat <- generate_scenario6(seed = 42)
cat(
  "True break points: t =", dat$break_points, "\n",
  " phi per regime  :", sapply(dat$phi_list, function(x) x[1L]), "\n",
  " sigma per regime:", dat$sigma_vec, "\n\n"
)

# -----------------------------------------------------------------------
# 2. Fit H-SBAR with joint co-location penalty (Section 8)
# -----------------------------------------------------------------------
cat("Fitting H-SBAR (joint penalty) ...\n")
fit <- hsbar(
  y = dat$Y,
  p = dat$p,
  lambda = 1e-3,
  c_scale = 1,
  alpha0 = 1,
  max_iter = 5000,
  # restart = FALSE,
  thr = 1e-5,
  verbose = TRUE
)
cat("Solver status:", fit$status, "\n\n")

# -----------------------------------------------------------------------
# 3. Stage 1 changepoints
# -----------------------------------------------------------------------
cat("Stage 1 — joint changepoints (cp):      ", fit$cp, "\n")
cat("Stage 1 — coeff changepoints (cp_theta):", fit$cp_theta, "\n")
cat("Stage 1 — var changepoints   (cp_psi):  ", fit$cp_psi, "\n")
cat("(True coefficient breaks at t =", dat$break_points, ")\n\n")

# -----------------------------------------------------------------------
# 4. Stage 2: BEA screening
# -----------------------------------------------------------------------
cat("Running BEA screening (Section 9) ...\n")
bea <- hsbar_bea(fit, y = dat$Y)
cat("Stage 2 — refined changepoints:", bea$cp, "\n")
cat(sprintf("  omega_n = %.4g\n\n", bea$omega_n))

# -----------------------------------------------------------------------
# 5. Diagnostic plot
# -----------------------------------------------------------------------
plot(dat$Y,
  type = "l", main = "H-SBAR fit (Scenario 1 — joint penalty)",
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
