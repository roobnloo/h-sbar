# hsbar-cvxr.R
# H-SBAR: Heterogeneous Structural Break AR with regime-specific variance.
#
# Joint co-location penalty (Section 8, eq. 8.1):
#
#   Q = (1/n) L_nat + lambda_n * sum_{i>=2} sqrt(||theta_i||_2^2 + c * psi_i^2)
#
# where L_nat = (1/2) sum_t [-log(phi_t) + phi_t*y_t^2
#                             - 2*y_t*g_t + g_t^2/phi_t]
#
# The joint norm ||[theta_i; sqrt(c)*psi_i]||_2 forces AR-coefficient breaks
# and precision breaks to be co-located (Section 8.4).
#
# Variables:
#   theta_i in R^p: i-th gamma-increment (precision-scaled AR vector)
#   psi_i   in R  : i-th precision increment
#
# Cumulative sums:
#   gamma_t = sum_{i<=t} theta_i = gamma_{j(t)}
#   phi_t   = sum_{i<=t} psi_i   = phi_{j(t)} = 1/sigma^2_{j(t)}
#
# Recovery: beta_{j(t)} = gamma_t / phi_t,  sigma^2_{j(t)} = 1/phi_t

library(CVXR)

#' Fit H-SBAR via the joint co-location penalty (Section 8, eq. 8.1)
#'
#' @param y         Numeric time series (length n)
#' @param p         AR order (no intercept)
#' @param lambda_n  Joint penalty strength (single tuning parameter)
#' @param c_scale   Fixed scale c > 0 (Section 8.3).  Sets the relative weight
#'                  of precision increments vs AR-coefficient increments in the
#'                  joint norm.  Default 1.
#' @param keep_rows Integer vector of row indices to include in the likelihood.
#'                  Defaults to all rows (\code{seq_len(n)}).
#' @param solver    CVXR solver (default "CLARABEL")
#' @param thr       Zero-threshold for changepoint detection
#' @param verbose   Print solver output?
#'
#' @return List: theta, psi, phi_vec, sigma2, beta,
#'               cp (joint changepoints), cp_theta, cp_psi, status, obj_val
hsbar_cvxr <- function(y,
                          p = 1,
                          lambda_n = 0.1,
                          c_scale = 1,
                          keep_rows = NULL,
                          solver = "CLARABEL",
                          thr = 1e-3,
                          verbose = FALSE) {
  n <- length(y)

  if (is.null(keep_rows)) keep_rows <- seq_len(n)
  n_tr <- length(keep_rows)

  # ------------------------------------------------------------------
  # 1. Lagged regressor matrix: y_lag[t, k] = y[t-k], pre-sample = 0
  # ------------------------------------------------------------------
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }

  y_lag_tr <- y_lag[keep_rows, , drop = FALSE] # n_tr x p
  y_col_tr <- matrix(y[keep_rows], n_tr, 1L) # n_tr x 1

  # ------------------------------------------------------------------
  # 2. Cumsum selector: l_sub[r, i] = 1 if i <= keep_rows[r]
  #    base::outer used explicitly: CVXR masks outer() from base.
  # ------------------------------------------------------------------
  l_sub <- base::outer(keep_rows, seq_len(n), ">=") + 0L # n_tr x n

  # ------------------------------------------------------------------
  # 3. CVXR decision variables
  # ------------------------------------------------------------------
  theta_mat <- Variable(c(n, p)) # row i = theta_i (gamma-increment)
  psi_vec <- Variable(c(n, 1L)) # precision increments

  # ------------------------------------------------------------------
  # 4. Affine expressions at training rows
  # ------------------------------------------------------------------
  gamma_cum_tr <- l_sub %*% theta_mat # n_tr x p
  g_vec_tr <- (y_lag_tr * gamma_cum_tr) %*% matrix(1, p, 1L) # n_tr x 1
  phi_tr <- l_sub %*% psi_vec # n_tr x 1

  # ------------------------------------------------------------------
  # 5. Negative log-likelihood (eq. 7.6)
  #    sum_t [-log(phi_t) + phi_t*y_t^2 - 2*y_t*g_t + g_t^2/phi_t]
  #    The quad-over-linear term g_t^2/phi_t is computed per-row since
  #    quad_over_lin(x, y) = x^T x / y (full inner product, not elementwise).
  # ------------------------------------------------------------------
  loss_log <- -sum_entries(log(phi_tr))
  loss_quad <- sum_entries(phi_tr * (y_col_tr^2))
  loss_cross <- -2 * sum_entries(g_vec_tr * y_col_tr)
  loss_qol <- Reduce("+", lapply(seq_len(n_tr), function(t) {
    quad_over_lin(g_vec_tr[t, 1L], phi_tr[t, 1L])
  }))

  total_loss <- (loss_log + loss_quad + loss_cross + loss_qol) / (2 * n_tr)

  # ------------------------------------------------------------------
  # 6. Joint co-location penalty (eq. 8.1):
  #    sum_{i>=2} ||[theta_i; sqrt(c) * psi_i]||_2
  #    hstack() is used (not cbind) to concatenate CVXR expressions.
  # ------------------------------------------------------------------
  sq_c <- sqrt(c_scale)
  pen_joint <- lambda_n * Reduce("+", lapply(seq(2L, n), function(i) {
    cvxr_norm(hstack(
      theta_mat[i, , drop = FALSE],
      sq_c * psi_vec[i, 1L, drop = FALSE]
    ), 2)
  }))

  # ------------------------------------------------------------------
  # 7. Solve
  # ------------------------------------------------------------------
  problem <- Problem(Minimize(total_loss + pen_joint))
  psolve(problem, solver = solver, verbose = verbose)

  sol_status <- status(problem)
  sol_obj <- value(problem)
  if (!sol_status %in% c("optimal", "optimal_inaccurate")) {
    warning("H-SBAR solver status: ", sol_status)
  }

  # ------------------------------------------------------------------
  # 8. Extract results and recover original parameters
  # ------------------------------------------------------------------
  theta_hat <- value(theta_mat)
  psi_hat <- as.vector(value(psi_vec))
  phi_hat <- cumsum(psi_hat)
  gamma_hat <- apply(theta_hat, 2, cumsum) # n x p
  sigma2_hat <- ifelse(phi_hat > 0, 1 / phi_hat, NA)
  beta_hat <- sweep(gamma_hat, 1, phi_hat, "/") # n x p

  # Changepoints via joint norm at positions i >= 2
  theta_tail <- theta_hat[-1L, , drop = FALSE] # (n-1) x p
  psi_tail <- psi_hat[-1L]
  joint_norms <- sqrt(rowSums(theta_tail^2) + c_scale * psi_tail^2)

  cp <- which(joint_norms > thr) + 1L
  cp_theta <- which(sqrt(rowSums(theta_tail^2)) > thr) + 1L
  cp_psi <- which(abs(psi_tail) > thr) + 1L

  # ------------------------------------------------------------------
  # 9. Post-processing: boundary trim and minimum-spacing filter
  #    - Drop candidates within p+3 of start or at the final index n
  #    - Of any two adjacent candidates with gap <= p+1, drop the earlier
  # ------------------------------------------------------------------
  filter_cp <- function(candidates) {
    x <- candidates[candidates > p + 3L & candidates < n]
    if (length(x) > 1L) {
      too_close <- which(diff(x) <= p + 1L)
      if (length(too_close) > 0L) x <- x[-too_close]
    }
    x
  }
  cp       <- filter_cp(cp)
  cp_theta <- filter_cp(cp_theta)
  cp_psi   <- filter_cp(cp_psi)

  list(
    theta    = theta_hat,
    psi      = psi_hat,
    phi_vec  = phi_hat,
    sigma2   = sigma2_hat,
    beta     = beta_hat,
    cp       = cp,
    cp_theta = cp_theta,
    cp_psi   = cp_psi,
    status   = sol_status,
    obj_val  = sol_obj
  )
}
