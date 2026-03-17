# chan-sbar.R
# Chan, Yau & Zhang (2014) Group LASSO for Structural Break AR (SBAR).
#
# Stage 1: Minimises Q = f + g where:
#   f = (1 / (2 * n_tr)) * ||Y_tr - (X_n * theta)_tr||^2   squared OLS loss
#   g = lambda * sum_{i=2}^n ||theta_i||_2                  group LASSO
#
# theta[i, ] in R^p is the AR coefficient increment at time i.
# theta[1, ] = beta_1 (first-regime AR vector, not penalised).
# Changepoints are indices i >= 2 with theta[i, ] != 0.
#
# The design matrix X_n has (t, j)-block = Y_{t-1}^T if j <= t, else 0,
# so (X_n theta)[t] = Y_{t-1}^T cumsum(theta)[t, ].  This cumsum structure
# allows O(n) gradient evaluation via bwd_cumsum, mirroring hsbar.R.
#
# Algorithm: FISTA with backtracking line search and gradient-based restart
# (O'Donoghue & Candes).  Solves the same convex problem as the BCD and
# LARS algorithms described in the paper's supplementary material.
#
# Reference:
#   Chan, N. H., Yau, C. Y., & Zhang, R. M. (2014). Group LASSO for
#   structural break time series. JASA, 109(506), 590-599.

# ============================================================
# Stage 1: Group LASSO fit via FISTA
# ============================================================

#' Fit Chan et al. (2014) SBAR via group LASSO (FISTA)
#'
#' Estimates AR coefficient changepoints by minimising the group LASSO
#' objective (eq. 2.2) with a squared OLS loss.  Interface matches
#' \code{hsbar()} for direct comparison; variance-related outputs
#' (\code{psi}, \code{phi_vec}, \code{cp_psi}) are \code{NULL}.
#'
#' @param y        Numeric time series (length n).
#' @param p        AR order (no intercept).
#' @param lambda   Regularisation parameter.
#' @param keep_rows Integer vector of row indices to include in the loss.
#'                  Defaults to all rows.
#' @param alpha0   Initial step size for backtracking (default 1).
#' @param beta     Backtracking shrinkage factor in (0, 1) (default 0.5).
#' @param max_iter Maximum number of outer iterations (default 1000).
#' @param tol      Convergence tolerance on proximal gradient mapping
#'                 (default 1e-6).
#' @param thr      Zero-threshold for changepoint detection (default 1e-6).
#' @param restart  Apply gradient-based momentum restart? (default TRUE).
#' @param eps_tol  Stop if the relative objective change falls below this
#'                 between iterations: |dQ| < eps_tol*(1+|Q|) (default 1e-10).
#' @param verbose  Print iteration log? (default FALSE).
#' @param init_theta  Warm-start matrix for theta (n x p).
#'
#' @return List: theta, psi (NULL), phi_vec (NULL), sigma2, beta,
#'               cp, cp_theta, cp_psi (NULL), obj_val, n_iter, stop_crit.
#'   \code{stop_crit} is one of \code{"pg_norm"}, \code{"obj_change"},
#'   or \code{"max_iter"}.
chan_sbar_admm <- function(y,
                           p = 1,
                           lambda = 0.1,
                           keep_rows = NULL,
                           alpha0 = 1,
                           beta = 0.5,
                           max_iter = 1000,
                           tol = 1e-6,
                           thr = 1e-6,
                           restart = TRUE,
                           eps_tol = 1e-10,
                           verbose = FALSE,
                           init_theta = NULL) {
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
  y_tr <- y[keep_rows] # n_tr

  # -----------------------------------------------------------------------
  # 2. Cumulative-sum helpers
  #    fwd_cumsum: cumsum of length-n vector evaluated at training rows
  #    bwd_cumsum: scatter n_tr-vector to length n, then reverse cumsum
  #      t(X_n[keep_rows,]) %*% x_tr = bwd_cumsum(x_tr) applied per column
  # -----------------------------------------------------------------------
  fwd_cumsum <- function(x) cumsum(x)[keep_rows]

  bwd_cumsum <- function(x_tr) {
    z <- numeric(n)
    z[keep_rows] <- x_tr
    rev(cumsum(rev(z)))
  }

  # -----------------------------------------------------------------------
  # 3. Smooth loss  f = (1 / (2 * n_tr)) * ||y_tr - X_n(th)_tr||^2
  #    Fitted value at row t: y_lag[t,] %*% cumsum(theta)[t,]
  # -----------------------------------------------------------------------
  compute_f <- function(th) {
    gamma_tr <- apply(th, 2, fwd_cumsum) # n_tr x p
    r <- y_tr - rowSums(y_lag_tr * gamma_tr) # n_tr
    (1 / (2 * n_tr)) * sum(r^2)
  }

  # -----------------------------------------------------------------------
  # 4. Penalty  g = lambda * sum_{i=2}^n ||theta_i||_2
  # -----------------------------------------------------------------------
  compute_g_pen <- function(th) {
    if (n == 1L) {
      return(0)
    }
    lambda * sum(sqrt(rowSums(th[-1L, , drop = FALSE]^2)))
  }

  # -----------------------------------------------------------------------
  # 5. Gradient of f w.r.t. theta
  #    grad_f[i, ] = -(1/n_tr) * sum_{t >= i} r_t * y_lag[t, ]
  #                = -(1/n_tr) * bwd_cumsum applied to (r * y_lag_tr)
  # -----------------------------------------------------------------------
  compute_grad <- function(th) {
    gamma_tr <- apply(th, 2, fwd_cumsum)
    r <- y_tr - rowSums(y_lag_tr * gamma_tr) # observed - fitted
    -(1 / n_tr) * apply(r * y_lag_tr, 2, bwd_cumsum) # n x p
  }

  # -----------------------------------------------------------------------
  # 6. Proximal operator: group soft-threshold on rows 2:n
  # -----------------------------------------------------------------------
  prox_g <- function(a_th, alpha) {
    if (n == 1L) {
      return(a_th)
    }
    norms <- sqrt(rowSums(a_th[-1L, , drop = FALSE]^2))
    shrink <- ifelse(norms > 0, pmax(0, 1 - alpha * lambda / norms), 0)
    out_th <- a_th
    out_th[-1L, ] <- shrink * a_th[-1L, , drop = FALSE]
    out_th
  }

  # -----------------------------------------------------------------------
  # 7. Initialise
  #    w = true iterate; m = extrapolated point (start equal); s = momentum
  # -----------------------------------------------------------------------
  if (!is.null(init_theta)) {
    theta <- init_theta
  } else {
    theta <- matrix(0, n, p)
  }

  m_th <- theta
  s <- 1
  n_iter <- max_iter
  obj_prev <- Inf
  stop_crit <- "max_iter"

  # -----------------------------------------------------------------------
  # 8. FISTA main loop
  # -----------------------------------------------------------------------
  for (iter in seq_len(max_iter)) {
    # Gradient at the extrapolated point m
    f_m <- compute_f(m_th)
    grad_m <- compute_grad(m_th)

    # --- Backtracking from m ---
    alpha <- alpha0
    repeat {
      th_tent <- m_th - alpha * grad_m
      th_new <- prox_g(th_tent, alpha)

      f_new <- compute_f(th_new)
      d_th <- th_new - m_th
      inner <- sum(grad_m * d_th)
      sq_norm <- sum(d_th^2)
      if (f_new <= f_m + inner + sq_norm / (2 * alpha)) break

      alpha <- beta * alpha
      if (alpha < 1e-16) {
        warning("chan_sbar: backtracking step size collapsed at iteration ", iter)
        break
      }
    }

    # Proximal gradient mapping norm (stopping criterion)
    pg_norm <- sqrt(sum((th_new - theta)^2)) / alpha

    # --- Momentum restart (O'Donoghue-Candes) ---
    do_restart <- restart && (
      sum((th_new - theta) * (m_th - th_new)) > 0
    )

    # --- Nesterov momentum update ---
    s_new <- (1 + sqrt(1 + 4 * s^2)) / 2

    if (do_restart) {
      m_th <- th_new
      s_new <- 1
    } else {
      coef <- (s - 1) / s_new
      m_th <- th_new + coef * (th_new - theta)
    }

    theta <- th_new
    s <- s_new

    obj_cur <- f_new + compute_g_pen(theta)

    if (verbose) {
      g_pen <- compute_g_pen(theta)
      rst <- if (do_restart) " R" else "  "
      fmt <- "iter %4d%s| f=%.6f | g=%.6f | Q=%.6f | pg=%.2e | a=%.2e"
      message(sprintf(fmt, iter, rst, f_new, g_pen, obj_cur, pg_norm, alpha))
    }

    if (pg_norm < tol) {
      stop_crit <- "pg_norm"
      n_iter <- iter
      break
    }
    if (abs(obj_cur - obj_prev) < eps_tol * (1 + abs(obj_cur))) {
      stop_crit <- "obj_change"
      n_iter <- iter
      break
    }
    obj_prev <- obj_cur
  }

  # -----------------------------------------------------------------------
  # 9. Recover cumulative AR coefficients: beta[t, ] = cumsum(theta)[t, ]
  # -----------------------------------------------------------------------
  beta_hat <- apply(theta, 2, cumsum) # n x p

  # -----------------------------------------------------------------------
  # 10. Detect changepoints from theta norms (rows 2:n)
  # -----------------------------------------------------------------------
  theta_tail <- theta[-1L, , drop = FALSE]
  theta_norms <- sqrt(rowSums(theta_tail^2))
  cp_raw <- which(theta_norms > thr) + 1L

  # -----------------------------------------------------------------------
  # 11. Post-processing: boundary trim and minimum-spacing filter
  # -----------------------------------------------------------------------
  filter_cp <- function(candidates) {
    x <- candidates[candidates > p + 3L & candidates < n]
    if (length(x) > 1L) {
      too_close <- which(diff(x) <= p + 1L)
      if (length(too_close) > 0L) x <- x[-too_close]
    }
    x
  }
  cp <- filter_cp(cp_raw)
  cp_theta <- cp

  # -----------------------------------------------------------------------
  # 12. Sigma2 via segment-wise OLS refit on the full series
  # -----------------------------------------------------------------------
  breaks <- c(1L, cp, n + 1L)
  sigma2 <- numeric(n)
  for (b in seq_len(length(breaks) - 1L)) {
    rows <- breaks[b]:(breaks[b + 1L] - 1L)
    n_j <- length(rows)
    x_j <- y_lag[rows, , drop = FALSE]
    y_j <- y[rows]
    if (n_j > p) {
      qr_j <- qr(x_j)
      beta_j <- qr.coef(qr_j, y_j)
      if (anyNA(beta_j)) beta_j[is.na(beta_j)] <- 0
      rss_j <- sum((y_j - x_j %*% beta_j)^2)
      sigma2[rows] <- rss_j / n_j
    } else {
      sigma2[rows] <- NA_real_
    }
  }

  # -----------------------------------------------------------------------
  # 13. Final objective value
  # -----------------------------------------------------------------------
  obj_val <- compute_f(theta) + compute_g_pen(theta)

  list(
    theta = theta,
    psi = NULL,
    phi_vec = NULL,
    sigma2 = sigma2,
    beta = beta_hat,
    cp = cp,
    cp_theta = cp_theta,
    cp_psi = NULL,
    obj_val = obj_val,
    n_iter = n_iter,
    stop_crit = stop_crit
  )
}
