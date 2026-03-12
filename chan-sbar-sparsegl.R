library(sparsegl)

#' Group LASSO objective for Chan et al. (2014) SBAR
#'
#' Evaluates the penalised least-squares objective (eq. 2.2) at any theta.
#' Can be called independently of \code{chan_sbar_bcd()}.
#'
#' @param theta      n x p matrix of AR increments (row 1 = baseline, rows 2:n penalised).
#' @param y          Numeric time series (length n).
#' @param p          AR order.
#' @param lambda     Regularisation parameter.
#' @param keep_rows  Integer vector of row indices to include in the loss.
#'                   Defaults to all rows.
#'
#' @return Scalar objective value Q = f(theta) + g(theta).
chan_sbar_bcd_obj <- function(theta, y, p, lambda, keep_rows = NULL) {
  n <- nrow(theta)
  if (is.null(keep_rows)) keep_rows <- seq_len(n)
  n_tr <- length(keep_rows)

  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }
  y_lag_tr <- y_lag[keep_rows, , drop = FALSE]

  # Build full design matrix x_full (n_tr x n*p) then multiply by vec(theta)
  l_sub <- outer(keep_rows, seq_len(n), ">=") + 0L # n_tr x n
  x_full <- matrix(0, n_tr, n * p)
  for (k in seq_len(p)) {
    col_indices <- seq(k, n * p, by = p)
    x_full[, col_indices] <- sweep(l_sub, 1L, y_lag_tr[, k], "*")
  }
  theta_vec <- as.vector(t(theta)) # flatten row-major: group i -> cols (i-1)*p+1:i*p
  r <- y[keep_rows] - x_full %*% theta_vec
  f <- sum(r^2) / (2 * n_tr)
  g <- if (n == 1L) 0 else lambda * sum(sqrt(rowSums(theta[-1L, , drop = FALSE]^2)))

  f + g
}

#' Fit Chan et al. (2014) SBAR via group LASSO using sparsegl
#'
#' Solves the same group LASSO objective as \code{chan_sbar_bcd()} by
#' building the explicit cumsum design matrix and passing it to
#' \code{sparsegl::sparsegl()} with \code{asparse = 0} (pure group lasso).
#'
#' @param y          Numeric time series (length n).
#' @param p          AR order (no intercept).
#' @param lambda     Regularisation parameter.
#' @param keep_rows  Integer vector of row indices to include in the loss.
#'                   Defaults to all rows.
#' @param thr        Zero-threshold for changepoint detection.  Default 1e-3.
#'
#' @return List: theta (n x p), beta (n x p cumulative AR coefficients),
#'               cp (integer changepoint locations), obj_val, converged.
chan_sbar <- function(y,
                      p = 1,
                      lambda = 0.1,
                      keep_rows = NULL,
                      thr = 1e-3) {
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
  # 2. Build explicit design matrix X_full (n_tr x n*p)
  #    Column ordering: columns (i-1)*p + 1 : i*p belong to group i.
  #    X_full[r, (i-1)*p + k] = y_lag[keep_rows[r], k]  if keep_rows[r] >= i
  #                            = 0                        otherwise
  # -----------------------------------------------------------------------
  l_sub <- outer(keep_rows, seq_len(n), ">=") + 0L # n_tr x n
  x_full <- matrix(0, n_tr, n * p)
  for (k in seq_len(p)) {
    col_indices <- seq(k, n * p, by = p)
    x_full[, col_indices] <- sweep(l_sub, 1L, y_lag_tr[, k], "*")
  }

  # -----------------------------------------------------------------------
  # 3. Fit pure group lasso via sparsegl
  #    - group 1 (baseline regime) carries pf_group = 0 -> unpenalised
  #    - asparse = 0 -> no within-group L1 penalty
  #    - intercept = FALSE, standardize = FALSE: model is fully pre-encoded
  # -----------------------------------------------------------------------
  group <- rep(seq_len(n), each = p)
  pf_group <- c(0, rep(1, n - 1L))

  fit <- sparsegl(
    x           = x_full,
    y           = y[keep_rows],
    group       = group,
    lambda      = lambda,
    pf_group    = pf_group,
    asparse     = 0,
    intercept   = FALSE,
    standardize = FALSE
  )

  # -----------------------------------------------------------------------
  # 4. Extract theta: (n*p)-vector -> n x p matrix (byrow to match ordering)
  # -----------------------------------------------------------------------
  theta_vec <- as.numeric(fit$beta[, 1L])
  theta <- matrix(theta_vec, nrow = n, ncol = p, byrow = TRUE)

  # -----------------------------------------------------------------------
  # 5. Recover cumulative AR coefficients
  # -----------------------------------------------------------------------
  beta_hat <- apply(theta, 2, cumsum) # n x p

  # -----------------------------------------------------------------------
  # 6. Detect changepoints from theta norms (rows 2:n)
  # -----------------------------------------------------------------------
  theta_tail <- theta[-1L, , drop = FALSE]
  theta_norms <- sqrt(rowSums(theta_tail^2))
  cp_raw <- which(theta_norms > thr) + 1L

  filter_cp <- function(cands) {
    x <- cands[cands > p + 3L & cands < n]
    if (length(x) > 1L) {
      too_close <- which(diff(x) <= p + 1L)
      if (length(too_close) > 0L) x <- x[-too_close]
    }
    x
  }
  cp <- filter_cp(cp_raw)

  # -----------------------------------------------------------------------
  # 7. Objective value (via shared helper from chan-sbar-blockwise.R)
  # -----------------------------------------------------------------------
  obj_val <- chan_sbar_bcd_obj(theta, y, p, lambda, keep_rows)

  list(
    theta     = theta,
    beta      = beta_hat,
    cp        = cp,
    obj_val   = obj_val,
    converged = fit$jerr == 0L
  )
}
