# test-sbar-cov.R
# Test SBAR-COV on Scenario 1: baseline AR(1) with two coefficient breaks
# and constant variance (no variance change).
#
# True setup: T=300, p=1, breaks at t=100 and t=200.
# phi: -0.6 -> 0.75 -> -0.8; sigma constant at 0.1.

source("generate-data.R")
source("sbar-cov.R")

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
# 2. Fit SBAR-COV with joint co-location penalty (Section 8)
# -----------------------------------------------------------------------
cat("Fitting SBAR-COV (joint penalty) ...\n")
fit <- sbar_cov(
  y        = dat$Y,
  p        = dat$p,
  lambda_n = 0.01,
  c_scale  = 1,
  solver   = "CLARABEL",
  thr      = 1e-5
)
cat("Solver status:", fit$status, "\n\n")

# -----------------------------------------------------------------------
# 3. Report detected changepoints
# -----------------------------------------------------------------------
cat("Joint changepoints (cp):      ", fit$cp, "\n")
cat("Coeff changepoints (cp_theta):", fit$cp_theta, "\n")
cat("Var changepoints   (cp_psi):  ", fit$cp_psi, "\n")
cat("(True coefficient breaks at t =", dat$break_points, ")\n\n")

# -----------------------------------------------------------------------
# 4. Diagnostic plot
# -----------------------------------------------------------------------
plot(dat$Y,
  type = "l", main = "SBAR-COV fit (Scenario 1 — joint penalty)",
  xlab = "t", ylab = "Y"
)
for (t in dat$break_points) {
  abline(v = t, col = "red", lty = 2, lwd = 2)
}
for (t in fit$cp) {
  abline(v = t, col = "blue", lty = 3)
}
legend("topleft",
  legend = c("True break", "Est. joint break"),
  col = c("red", "blue"),
  lty = c(2, 3), bty = "n"
)
