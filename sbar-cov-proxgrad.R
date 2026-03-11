# sbar-cov-proxgrad.R
# SBAR-COV fitted by ISTA (proximal gradient descent) with backtracking.
#
# Minimises Q = f + g where:
#   f = (1/n_tr) * L_nat      smooth loss  (eq. 2 of prox_grad.md)
#   g = lambda * sum ||v_i||  group LASSO  (eq. 7.5 of sbar-cov.md)
#
# Each iteration:
#   1. Gradient step on f with backtracking line search
#   2. Joint group soft-threshold, closed-form prox of g
#      (eq. 7.7 of sbar-cov.md)
#
# See prox_grad.md for the full mathematical development.

#' Fit SBAR-COV via ISTA with backtracking line search
#'
#' @param y         Numeric time series (length n)
#' @param p         AR order (no intercept)
#' @param lambda_n  Joint penalty strength
#' @param c_scale   Scale parameter c > 0 (default 1)
#' @param keep_rows Integer vector of row indices to include in the likelihood.
#'                  Defaults to all rows.
#' @param alpha0    Initial step size for backtracking (default 1)
#' @param beta      Backtracking shrinkage factor in (0, 1) (default 0.5)
#' @param max_iter  Maximum number of outer iterations (default 1000)
#' @param tol       Convergence tolerance on proximal gradient mapping
#'                  (default 1e-6)
#' @param thr       Zero-threshold for changepoint detection (default 1e-3)
#' @param verbose   Print iteration log?
#'
#' @return List: theta, psi, phi_vec, sigma2, beta,
#'               cp, cp_theta, cp_psi, obj_val, n_iter
sbar_cov_ista <- function(y,
                          p = 1,
                          lambda_n = 0.1,
                          c_scale = 1,
                          keep_rows = NULL,
                          alpha0 = 1,
                          beta = 0.5,
                          max_iter = 1000,
                          tol = 1e-6,
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

  y_lag_tr <- y_lag[keep_rows, , drop = FALSE]  # n_tr x p
  y_tr <- y[keep_rows]                           # n_tr

  # -----------------------------------------------------------------------
  # 2. Cumulative-sum helpers (replace the O(n * n_tr) l_sub matrix)
  #    fwd_cumsum: cumsum of an n-vector evaluated at the training rows
  #    bwd_cumsum: scatter n_tr-vector to length n, then reverse cumsum
  #      t(l_sub) %*% x  =  rev(cumsum(rev(z)))  where z[keep_rows] = x
  # -----------------------------------------------------------------------
  fwd_cumsum <- function(x) cumsum(x)[keep_rows]

  bwd_cumsum <- function(x_tr) {
    z <- numeric(n)
    z[keep_rows] <- x_tr
    rev(cumsum(rev(z)))
  }

  # Helper: cumulative sums G_t and Phi_t at the training rows
  #    G_t   = y_lag[t,] . gamma_t  (inner product with precision-scaled AR)
  #    Phi_t = sum_{i<=t} psi_i     (precision at time t)
  compute_gp <- function(th, ps) {
    gamma_cum <- apply(th, 2, fwd_cumsum) # n_tr x p
    g <- rowSums(y_lag_tr * gamma_cum)    # n_tr
    phi <- fwd_cumsum(ps)                 # n_tr
    list(g = g, phi = phi)
  }

  # -----------------------------------------------------------------------
  # 3. Smooth loss  f = (1 / n_tr) * L_nat  (eq. 2 of prox_grad.md)
  # -----------------------------------------------------------------------
  compute_f <- function(g, phi) {
    if (any(phi <= 0)) return(Inf)
    (1 / (2 * n_tr)) * sum(-log(phi) + phi * y_tr^2 - 2 * y_tr * g + g^2 / phi)
  }

  # -----------------------------------------------------------------------
  # 4. Penalty  g = lambda * sum_{i>=2} ||[theta_i; sqrt(c)*psi_i]||_2
  # -----------------------------------------------------------------------
  compute_g_pen <- function(th, ps) {
    if (n == 1) return(0)
    norms <- sqrt(rowSums(th[-1L, , drop = FALSE]^2) + c_scale * ps[-1L]^2)
    lambda_n * sum(norms)
  }

  # -----------------------------------------------------------------------
  # 5. Gradients of f  (eq. 3-4 of prox_grad.md)
  # -----------------------------------------------------------------------
  compute_grad <- function(g, phi) {
    delta <- g / phi - y_tr                              # n_tr
    eta <- -1 / (2 * phi) + y_tr^2 / 2 - g^2 / (2 * phi^2)  # n_tr
    grad_th <- (1 / n_tr) * apply(delta * y_lag_tr, 2, bwd_cumsum) # n x p
    grad_ps <- (1 / n_tr) * bwd_cumsum(eta)                        # n
    list(grad_th = grad_th, grad_ps = grad_ps)
  }

  # -----------------------------------------------------------------------
  # 6. Proximal operator: joint group soft-threshold  (eq. 7 of prox_grad.md)
  #    For i >= 2: shrink (theta_i, psi_i) by (1 - alpha*lambda/||v_i||)+
  #    For i  = 1: no penalty, pass through unchanged.
  # -----------------------------------------------------------------------
  prox_g <- function(a_th, a_ps, alpha) {
    if (n == 1) return(list(th = a_th, ps = a_ps))
    norms <- sqrt(rowSums(a_th[-1L, , drop = FALSE]^2) + c_scale * a_ps[-1L]^2)
    shrink <- ifelse(norms > 0, pmax(0, 1 - alpha * lambda_n / norms), 0)
    out_th <- a_th
    out_ps <- a_ps
    out_th[-1L, ] <- shrink * a_th[-1L, , drop = FALSE]
    out_ps[-1L] <- shrink * a_ps[-1L]
    list(th = out_th, ps = out_ps)
  }

  # -----------------------------------------------------------------------
  # 7. Initialise: psi = (1, 0, ..., 0)  =>  phi_t = 1 for all t (feasible)
  #                theta = 0  =>  G_t = 0 for all t
  # -----------------------------------------------------------------------
  theta <- matrix(0, n, p)
  psi <- c(1, rep(0, n - 1))

  n_iter <- max_iter

  # -----------------------------------------------------------------------
  # 8. ISTA main loop
  # -----------------------------------------------------------------------
  for (iter in seq_len(max_iter)) {

    gp <- compute_gp(theta, psi)
    f_old <- compute_f(gp$g, gp$phi)
    grad <- compute_grad(gp$g, gp$phi)

    # --- Backtracking line search ---
    alpha <- alpha0
    repeat {
      # Gradient step
      th_tent <- theta - alpha * grad$grad_th
      ps_tent <- psi   - alpha * grad$grad_ps

      # Proximal step
      prx <- prox_g(th_tent, ps_tent, alpha)
      th_new <- prx$th
      ps_new <- prx$ps

      # Feasibility: all Phi_t > 0 required by the log barrier
      gp_new <- compute_gp(th_new, ps_new)
      if (all(gp_new$phi > 0)) {
        f_new <- compute_f(gp_new$g, gp_new$phi)
        # Sufficient decrease (Armijo, eq. 9 of prox_grad.md)
        dth <- th_new - theta
        dps <- ps_new - psi
        inner <- sum(grad$grad_th * dth) + sum(grad$grad_ps * dps)
        sq_norm <- sum(dth^2) + sum(dps^2)
        if (f_new <= f_old + inner + sq_norm / (2 * alpha)) break
      }

      alpha <- beta * alpha
      if (alpha < 1e-16) {
        warning("ISTA: backtracking step size collapsed at iteration ", iter)
        break
      }
    }

    # Proximal gradient mapping norm ||w_new - w|| / alpha  (stopping criterion)
    dth <- th_new - theta
    dps <- ps_new - psi
    pg_norm <- sqrt(sum(dth^2) + sum(dps^2)) / alpha

    theta <- th_new
    psi <- ps_new

    if (verbose) {
      g_pen <- compute_g_pen(theta, psi)
      fmt <- "iter %4d | f=%.6f | g=%.6f | Q=%.6f | pg=%.2e | a=%.2e"
      message(sprintf(fmt, iter, f_old, g_pen, f_old + g_pen, pg_norm, alpha))
    }

    if (pg_norm < tol) {
      n_iter <- iter
      break
    }
  }

  # -----------------------------------------------------------------------
  # 9. Recover original parameters from natural parametrisation
  # -----------------------------------------------------------------------
  phi_hat <- cumsum(psi)
  gamma_hat <- apply(theta, 2, cumsum)           # n x p
  sigma2_hat <- ifelse(phi_hat > 0, 1 / phi_hat, NA_real_)
  beta_hat <- sweep(gamma_hat, 1, phi_hat, "/")  # n x p

  # -----------------------------------------------------------------------
  # 10. Detect changepoints
  # -----------------------------------------------------------------------
  theta_tail <- theta[-1L, , drop = FALSE]
  psi_tail <- psi[-1L]
  joint_norms <- sqrt(rowSums(theta_tail^2) + c_scale * psi_tail^2)

  cp <- which(joint_norms > thr) + 1L
  cp_theta <- which(sqrt(rowSums(theta_tail^2)) > thr) + 1L
  cp_psi <- which(abs(psi_tail) > thr) + 1L

  gp_final <- compute_gp(theta, psi)
  obj_val <- compute_f(gp_final$g, gp_final$phi) + compute_g_pen(theta, psi)

  list(
    theta    = theta,
    psi      = psi,
    phi_vec  = phi_hat,
    sigma2   = sigma2_hat,
    beta     = beta_hat,
    cp       = cp,
    cp_theta = cp_theta,
    cp_psi   = cp_psi,
    obj_val  = obj_val,
    n_iter   = n_iter
  )
}
