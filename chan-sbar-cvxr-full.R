source("generate-data.R")
source("chan-sbar-blockwise.R") # for chan_sbar_bcd_obj()
library(CVXR)

# ============================================================
# CVXR reference implementation of Chan et al. (2014) eq. 2.2
# using the full explicit design matrix (no cumsum trick).
#
# Minimises Q = (1 / (2 * n_tr)) * ||y_tr - x_full %*% theta_vec||^2
#             + lambda * sum_{i=2}^n ||theta_vec[((i-1)*p+1):(i*p)]||_2
#
# Design matrix x_full (n_tr x n*p):
#   x_full[r, (i-1)*p + k] = y_lag[keep_rows[r], k]  if keep_rows[r] >= i
#                           = 0                        otherwise
#
# theta_vec is a flat n*p vector; groups of p consecutive elements
# correspond to the AR increment at each time index.
# ============================================================
chan_sbar_cvxr_full <- function(y,
                                p = 1,
                                lambda = 0.1,
                                keep_rows = NULL,
                                solver = "CLARABEL",
                                thr = 1e-3,
                                verbose = FALSE) {
  n <- length(y)
  if (is.null(keep_rows)) keep_rows <- seq_len(n)
  n_tr <- length(keep_rows)

  # -----------------------------------------------------------------------
  # 1. Lagged regressor matrix: y_lag[t, k] = y[t-k], pre-sample = 0
  # -----------------------------------------------------------------------
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }
  y_lag_tr <- y_lag[keep_rows, , drop = FALSE] # n_tr x p

  # -----------------------------------------------------------------------
  # 2. Build explicit design matrix x_full (n_tr x n*p)
  #    Column ordering: columns (i-1)*p + 1 : i*p belong to group i.
  # -----------------------------------------------------------------------
  l_sub <- outer(keep_rows, seq_len(n), ">=") + 0L # n_tr x n
  x_full <- matrix(0, n_tr, n * p)
  for (k in seq_len(p)) {
    col_indices <- seq(k, n * p, by = p)
    x_full[, col_indices] <- sweep(l_sub, 1L, y_lag_tr[, k], "*")
  }

  # -----------------------------------------------------------------------
  # 3. CVXR problem: flat variable, single matrix product for the loss
  # -----------------------------------------------------------------------
  theta_vec <- Variable(n * p)

  fitted <- x_full %*% theta_vec # n_tr x 1
  # loss <- sum_entries(square(y[keep_rows] - fitted)) / (2 * n_tr)
  loss <- sum_squares(y[keep_rows] - fitted) / (2 * n_tr)

  # Group penalty on blocks i = 2..n (group 1 = baseline, unpenalised)
  # pen <- lambda * Reduce("+", lapply(seq(2L, n), function(i) {
  #   idx <- ((i - 1L) * p + 1L):(i * p)
  #   cvxr_norm(theta_vec[idx], 2)
  # }))
  pen <- lambda * CVXR::norm(theta_vec[-1], "1")

  problem <- Problem(Minimize(loss + pen))
  psolve(problem, solver = solver, verbose = verbose)

  sol_status <- status(problem)
  objval <- value(problem)
  if (!sol_status %in% c("optimal", "optimal_inaccurate")) {
    warning("chan_sbar_cvxr_full solver status: ", sol_status)
  }

  # -----------------------------------------------------------------------
  # 4. Extract theta: reshape flat vector -> n x p matrix
  # -----------------------------------------------------------------------
  theta_hat <- matrix(value(theta_vec), nrow = n, ncol = p, byrow = TRUE)
  beta_hat <- apply(theta_hat, 2, cumsum)

  # -----------------------------------------------------------------------
  # 5. Detect changepoints from theta norms (rows 2:n)
  # -----------------------------------------------------------------------
  theta_norms <- sqrt(rowSums(theta_hat[-1L, , drop = FALSE]^2))
  cp_raw <- which(theta_norms > thr) + 1L

  filter_cp <- function(cands) {
    x <- cands[cands > p + 3L & cands < n]
    if (length(x) > 1L) {
      too_close <- which(diff(x) <= p + 1L)
      if (length(too_close) > 0L) x <- x[-too_close]
    }
    x
  }

  list(
    theta   = theta_hat,
    beta    = beta_hat,
    cp      = filter_cp(cp_raw),
    status  = sol_status,
    obj_val = objval # chan_sbar_bcd_obj(theta_hat, y, p, lambda, keep_rows)
  )
}
