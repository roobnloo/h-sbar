#' Estimate the scale parameter c for the H-SBAR penalty
#'
#' Fits a single homogeneous AR(p) model by OLS and returns
#' c = ||beta_hat||^2, which places theta and psi on the same
#' scale inside the group penalty.
#'
#' @param y  Numeric time series (length n)
#' @param p  AR order (no intercept)
#' @return   Scalar c > 0 for use as c_scale in hsbar()
fit_c <- function(y, p) {
  n <- length(y)
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }
  beta_hat <- qr.solve(y_lag, y)
  sum(beta_hat^2)
}
