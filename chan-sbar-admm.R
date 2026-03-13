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
# Stage 2: chan_sbar_bea() prunes the over-selected Stage 1 candidates
# using the RSS-based information criterion of Chan et al. (2014) eq. 2.9:
#   IC(m, t) = Sn(t1,...,tm) + m * omega_n
# where Sn is the total OLS residual sum of squares (NOT the profiled NLL
# used in hsbar-bea.R).  Default omega_n = (p+1)*log(n) (MDL-like).
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
#' @param thr      Zero-threshold for changepoint detection (default 1e-3).
#' @param restart  Apply gradient-based momentum restart? (default TRUE).
#' @param verbose  Print iteration log? (default FALSE).
#' @param init_theta  Warm-start matrix for theta (n x p).
#'
#' @return List: theta, psi (NULL), phi_vec (NULL), sigma2, beta,
#'               cp, cp_theta, cp_psi (NULL), obj_val, n_iter.
chan_sbar_admm <- function(y,
                           p = 1,
                           lambda = 0.1,
                           keep_rows = NULL,
                           alpha0 = 1,
                           beta = 0.5,
                           max_iter = 1000,
                           tol = 1e-6,
                           thr = 1e-3,
                           restart = TRUE,
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

    if (verbose) {
      g_pen <- compute_g_pen(theta)
      f_cur <- compute_f(theta)
      rst <- if (do_restart) " R" else "  "
      fmt <- "iter %4d%s| f=%.6f | g=%.6f | Q=%.6f | pg=%.2e | a=%.2e"
      message(sprintf(fmt, iter, rst, f_cur, g_pen, f_cur + g_pen, pg_norm, alpha))
    }

    if (pg_norm < tol) {
      n_iter <- iter
      break
    }
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
    theta    = theta,
    psi      = NULL,
    phi_vec  = NULL,
    sigma2   = sigma2,
    beta     = beta_hat,
    cp       = cp,
    cp_theta = cp_theta,
    cp_psi   = NULL,
    obj_val  = obj_val,
    n_iter   = n_iter
  )
}

# ============================================================
# Stage 2: Backward Elimination Algorithm (Chan et al. 2014 Section 2.2)
# ============================================================

# ------------------------------------------------------------------
# Internal: OLS residual sum of squares for one segment
# ------------------------------------------------------------------
chan_seg_rss <- function(y_seg, x_seg) {
  n_j <- nrow(x_seg)
  p_seg <- ncol(x_seg)
  if (n_j <= p_seg) {
    return(Inf)
  }
  qr_j <- qr(x_seg)
  beta_j <- qr.coef(qr_j, y_seg)
  if (anyNA(beta_j)) beta_j[is.na(beta_j)] <- 0
  sum((y_seg - x_seg %*% beta_j)^2)
}

# ------------------------------------------------------------------
# Internal: IC = sum_j RSS_j + m * omega_n  (Chan 2014 eq. 2.9)
# ------------------------------------------------------------------
chan_compute_ic <- function(y, y_lag, cps, n, omega_n) {
  breaks <- c(1L, cps, n + 1L)
  total_rss <- 0
  for (b in seq_len(length(breaks) - 1L)) {
    rows <- breaks[b]:(breaks[b + 1L] - 1L)
    total_rss <- total_rss + chan_seg_rss(y[rows], y_lag[rows, , drop = FALSE])
  }
  total_rss + length(cps) * omega_n
}

# ------------------------------------------------------------------
# Internal: segment-wise OLS beta and sigma^2 after pruning
# ------------------------------------------------------------------
chan_refit_segments <- function(y, y_lag, cps, n, p) {
  breaks <- c(1L, cps, n + 1L)
  beta_out <- matrix(0, n, p)
  sig2_out <- numeric(n)
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
      beta_out[rows, ] <- matrix(beta_j, nrow = n_j, ncol = p, byrow = TRUE)
      sig2_out[rows] <- rss_j / n_j
    } else {
      sig2_out[rows] <- NA_real_
    }
  }
  list(beta = beta_out, sigma2 = sig2_out)
}

# ------------------------------------------------------------------
# Public: BEA pruning of Stage 1 changepoints (Chan 2014 Section 2.2)
# ------------------------------------------------------------------

#' Backward Elimination Algorithm for Chan et al. (2014) SBAR
#'
#' Prunes the over-selected Stage 1 candidates from \code{chan_sbar()}
#' using the RSS-based information criterion of Chan et al. (2014) eq. 2.9:
#' \deqn{IC(m, \mathbf{t}) = S_n(t_1,\ldots,t_m) + m \omega_n}
#' where \eqn{S_n} is the total OLS residual sum of squares.  This criterion
#' differs from \code{hsbar_bea()}, which uses a profiled log-likelihood.
#'
#' @param fit      Output of \code{chan_sbar()}.
#' @param y        Numeric time series (length n).
#' @param p        AR order.  Inferred from \code{fit$beta} if omitted.
#' @param omega_n  Penalty per break.  Default: \eqn{(p+1)\log(n)} (MDL-like).
#'
#' @return A list with:
#' \describe{
#'   \item{cp}{Integer vector of refined changepoint locations.}
#'   \item{beta}{n x p matrix: segment-wise OLS AR coefficients.}
#'   \item{sigma2}{Length-n vector of segment-wise MLE variances.}
#'   \item{ic}{IC value at the final changepoint set.}
#'   \item{omega_n}{Penalty per break used.}
#' }
chan_sbar_bea <- function(fit, y, p = NULL, omega_n = NULL) {
  n <- length(y)
  if (is.null(p)) p <- ncol(fit$beta)
  if (is.null(omega_n)) omega_n <- p * log(n)

  # Lagged regressor matrix (same convention as chan_sbar)
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }

  cps <- sort(fit$cp)

  if (length(cps) == 0L) {
    seg <- chan_refit_segments(y, y_lag, cps, n, p)
    return(list(
      cp      = integer(0L),
      beta    = seg$beta,
      sigma2  = seg$sigma2,
      ic      = chan_compute_ic(y, y_lag, cps, n, omega_n),
      omega_n = omega_n
    ))
  }

  current_ic <- chan_compute_ic(y, y_lag, cps, n, omega_n)

  # Iterative backward elimination
  repeat {
    m <- length(cps)
    if (m == 0L) break

    ic_without <- vapply(seq_len(m), function(i) {
      chan_compute_ic(y, y_lag, cps[-i], n, omega_n)
    }, numeric(1L))

    best_i <- which.min(ic_without)

    if (ic_without[best_i] <= current_ic) {
      cps <- cps[-best_i]
      current_ic <- ic_without[best_i]
    } else {
      break
    }
  }

  seg <- chan_refit_segments(y, y_lag, cps, n, p)

  list(
    cp      = cps,
    beta    = seg$beta,
    sigma2  = seg$sigma2,
    ic      = current_ic,
    omega_n = omega_n
  )
}
