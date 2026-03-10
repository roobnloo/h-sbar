# sbar-cov-admm.R
# ADMM solver for the SBAR-COV joint co-location penalty problem (optim.md).
#
# Minimises:
#   Q = (1/n) L_nat + lambda_n * sum_{i>=2} sqrt(||theta_i||^2 + c * psi_i^2)
#
# where L_nat = (1/2) sum_t [-log(Phi_t) + Phi_t*y_t^2
#                             - 2*y_t*G_t + G_t^2/Phi_t]
#
# ADMM split  theta = z_1, psi = z_2 decouples the smooth loss (Step 1,
# solved by L-BFGS via optim) from the non-smooth group penalty (Step 2,
# closed-form joint group soft-threshold).
#
# Return list has the same names as sbar_cov() plus ADMM diagnostics
# (iter, converged, obj_val) for direct comparison.

#' Fit SBAR-COV via ADMM
#'
#' @param y           Numeric time series (length n)
#' @param p           AR order (no intercept)
#' @param lambda_n    Joint penalty strength
#' @param c_scale     Fixed scale c > 0; balances precision vs AR increments
#' @param keep_rows   Row indices to include (defaults to all)
#' @param rho         ADMM augmented-Lagrangian penalty parameter
#' @param max_iter    Maximum outer ADMM iterations
#' @param tol_abs     Absolute convergence tolerance (Boyd et al. 2011 §3.3.1)
#' @param tol_rel     Relative convergence tolerance
#' @param inner_maxit Max L-BFGS iterations per smooth step
#' @param thr         Zero-threshold for changepoint detection
#' @param verbose     Print per-iteration diagnostics?
#'
#' @return List: theta, psi, phi_vec, sigma2, beta, cp, cp_theta, cp_psi,
#'               iter, converged, obj_val
sbar_cov_admm <- function(y,
                          p = 1,
                          lambda_n = 0.1,
                          c_scale = 1,
                          keep_rows = NULL,
                          rho = 1.0,
                          max_iter = 500,
                          tol_abs = 1e-4,
                          tol_rel = 1e-3,
                          inner_maxit = 20,
                          thr = 1e-3,
                          verbose = FALSE) {
  n <- length(y)

  if (is.null(keep_rows)) keep_rows <- seq_len(n)
  n_tr <- length(keep_rows)

  # ------------------------------------------------------------------
  # 1. Lagged regressor matrix: y_lag[t, k] = y[t-k], pre-sample = 0
  #    (identical to sbar-cov.R)
  # ------------------------------------------------------------------
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }
  y_lag_tr <- y_lag[keep_rows, , drop = FALSE] # n_tr x p
  y_col_tr <- y[keep_rows] # n_tr

  # ------------------------------------------------------------------
  # 2. Cumsum selector l_sub[r, i] = 1 if i <= keep_rows[r]
  #    l_sub_t = t(l_sub) used for reverse-cumsum gradient operations
  # ------------------------------------------------------------------
  l_sub <- base::outer(keep_rows, seq_len(n), ">=") + 0L # n_tr x n
  l_sub_t <- t(l_sub) # n x n_tr

  # ------------------------------------------------------------------
  # 3. Forward pass: compute G_t and Phi_t from (theta_mat, psi_vec).
  #    G_t is the dot product of the cumulative gamma with the lagged
  #    regressors; Phi_t is the cumulative precision. Both are n_tr vectors.
  # ------------------------------------------------------------------
  compute_g_phi <- function(theta_mat, psi_vec) {
    gamma_cum_tr <- l_sub %*% theta_mat # n_tr x p
    g_tr <- rowSums(gamma_cum_tr * y_lag_tr) # n_tr
    phi_tr <- as.vector(l_sub %*% psi_vec) # n_tr
    list(g = g_tr, phi = phi_tr)
  }

  # ------------------------------------------------------------------
  # 4. Negative log-likelihood L_nat (scalar, returns Inf if Phi_t <= 0)
  # ------------------------------------------------------------------
  l_nat <- function(g_tr, phi_tr) {
    if (any(phi_tr <= 0)) {
      return(Inf)
    }
    0.5 * sum(
      -log(phi_tr) + phi_tr * y_col_tr^2 - 2 * y_col_tr * g_tr +
        g_tr^2 / phi_tr
    )
  }

  # ------------------------------------------------------------------
  # 5. Full penalised objective Q_n^jnt (for reporting only)
  # ------------------------------------------------------------------
  objective <- function(theta_mat, psi_vec) {
    gp <- compute_g_phi(theta_mat, psi_vec)
    loss <- l_nat(gp$g, gp$phi) / n_tr
    th_t <- theta_mat[-1L, , drop = FALSE] # (n-1) x p
    ps_t <- psi_vec[-1L] # (n-1)
    pen <- lambda_n * sum(sqrt(rowSums(th_t^2) + c_scale * ps_t^2))
    loss + pen
  }

  # ------------------------------------------------------------------
  # 6. Smooth ADMM subproblem value + gradient (Step 1).
  #    Objective: (1/n_tr)*L_nat + (rho/2)||theta-c1||^2
  #                               + (rho/2)||psi-c2||^2
  #    grad_theta: reverse-cumsum of (delta * y_lag_tr), plus proximal term
  #    grad_psi:   reverse-cumsum of eta,                plus proximal term
  # ------------------------------------------------------------------
  smooth_fg <- function(theta_mat, psi_vec, c1_mat, c2_vec) {
    gp <- compute_g_phi(theta_mat, psi_vec)
    g_tr <- gp$g
    phi_tr <- gp$phi

    if (any(phi_tr <= 0)) {
      return(list(f = Inf, g = rep(NaN, n * p + n)))
    }

    d_theta <- theta_mat - c1_mat # n x p
    d_psi <- psi_vec - c2_vec # n

    f_loss <- l_nat(g_tr, phi_tr) / n_tr
    f_prox <- 0.5 * rho * (sum(d_theta^2) + sum(d_psi^2))

    # Gradient w.r.t. theta  (n x p)
    delta <- -y_col_tr + g_tr / phi_tr # n_tr
    grad_theta <- l_sub_t %*% (delta * y_lag_tr) / n_tr +
      rho * d_theta # n x p

    # Gradient w.r.t. psi  (n)
    eta <- -0.5 / phi_tr + 0.5 * y_col_tr^2 -
      0.5 * g_tr^2 / phi_tr^2 # n_tr
    grad_psi <- as.vector(l_sub_t %*% eta) / n_tr +
      rho * d_psi # n

    list(f = f_loss + f_prox, g = c(as.vector(grad_theta), grad_psi))
  }

  # ------------------------------------------------------------------
  # 7. Joint group soft-threshold — closed form (optim.md eq. 5).
  #    Applies ellipsoidal shrinkage to each (a_mat[i,], b_vec[i]) pair
  #    for i >= 2, zeroing the group when its weighted norm <= threshold.
  # ------------------------------------------------------------------
  joint_prox <- function(a_mat, b_vec, threshold) {
    th_a <- a_mat[-1L, , drop = FALSE] # (n-1) x p
    ps_b <- b_vec[-1L] # (n-1)
    denom <- sqrt(rowSums(th_a^2) + c_scale * ps_b^2) # (n-1)
    shrink <- pmax(0, 1 - threshold / denom) # (n-1), 0 if denom=0
    z1 <- a_mat
    z2 <- b_vec
    z1[-1L, ] <- shrink * th_a # broadcasts shrink over columns
    z2[-1L] <- shrink * ps_b
    list(z1 = z1, z2 = z2)
  }

  # ------------------------------------------------------------------
  # 8. Initialise: theta = 0, psi = (phi_init, 0, ..., 0) so Phi_t = phi_init
  # ------------------------------------------------------------------
  phi_init <- 1 / var(y)
  theta_mat <- matrix(0, n, p)
  psi_vec <- c(phi_init, rep(0, n - 1L)) # Phi_t = phi_init > 0 for all t

  z1 <- theta_mat
  z2 <- psi_vec
  u1 <- matrix(0, n, p)
  u2 <- rep(0, n)

  threshold <- lambda_n / rho
  dim_total <- sqrt(n * p + n)
  alpha     <- 1.5   # over-relaxation parameter (Boyd et al. 2011, §3.4.3)

  # ------------------------------------------------------------------
  # 9. ADMM outer loop
  # ------------------------------------------------------------------
  converged <- FALSE
  iter <- 0L

  for (iter in seq_len(max_iter)) {
    z1_old <- z1
    z2_old <- z2

    # Step 1 — smooth update via L-BFGS
    c1_mat <- z1 - u1
    c2_vec <- z2 - u2
    x0 <- c(as.vector(theta_mat), psi_vec)

    res <- optim(
      par = x0,
      fn = function(x) {
        th <- matrix(x[seq_len(n * p)], n, p)
        ps <- x[seq(n * p + 1L, n * p + n)]
        smooth_fg(th, ps, c1_mat, c2_vec)$f
      },
      gr = function(x) {
        th <- matrix(x[seq_len(n * p)], n, p)
        ps <- x[seq(n * p + 1L, n * p + n)]
        smooth_fg(th, ps, c1_mat, c2_vec)$g
      },
      method = "L-BFGS-B",
      control = list(maxit = inner_maxit, factr = 1e7)
    )

    theta_mat <- matrix(res$par[seq_len(n * p)], n, p)
    psi_vec <- res$par[seq(n * p + 1L, n * p + n)]

    # Over-relaxation: mix current theta with previous z (Boyd et al. §3.4.3)
    theta_relax <- alpha * theta_mat + (1 - alpha) * z1_old
    psi_relax   <- alpha * psi_vec   + (1 - alpha) * z2_old

    # Step 2 — joint group soft-threshold on relaxed variable
    prox_out <- joint_prox(theta_relax + u1, psi_relax + u2, threshold)
    z1 <- prox_out$z1
    z2 <- prox_out$z2

    # Step 3 — dual update uses relaxed variable; primal residual uses unrelaxed
    r1 <- theta_mat - z1
    r2 <- psi_vec   - z2
    u1 <- u1 + theta_relax - z1
    u2 <- u2 + psi_relax   - z2

    # Convergence check (Boyd et al. 2011, §3.3.1)
    r_pri <- sqrt(sum(r1^2) + sum(r2^2))
    s_dual <- rho * sqrt(sum((z1 - z1_old)^2) + sum((z2 - z2_old)^2))

    primal_scale <- max(
      sqrt(sum(theta_mat^2) + sum(psi_vec^2)),
      sqrt(sum(z1^2) + sum(z2^2))
    )
    dual_scale <- rho * sqrt(sum(u1^2) + sum(u2^2))

    eps_pri <- dim_total * tol_abs + tol_rel * primal_scale
    eps_dual <- dim_total * tol_abs + tol_rel * dual_scale

    if (verbose) {
      obj <- objective(z1, z2)
      cat(sprintf(
        "Iter %d: obj=%.6g  r_pri=%.2e (eps=%.2e)  s_dual=%.2e (eps=%.2e)\n",
        iter, obj, r_pri, eps_pri, s_dual, eps_dual
      ))
    }

    if (r_pri <= eps_pri && s_dual <= eps_dual) {
      converged <- TRUE
      break
    }
  }

  if (!converged) {
    warning("ADMM did not converge in ", max_iter, " iterations.")
  }

  # ------------------------------------------------------------------
  # 10. Recover original parameters from (z1, z2)
  # ------------------------------------------------------------------
  theta_hat <- z1
  psi_hat <- as.vector(z2)
  phi_hat <- cumsum(psi_hat)
  gamma_hat <- apply(theta_hat, 2, cumsum) # n x p
  sigma2_hat <- ifelse(phi_hat > 0, 1 / phi_hat, NA)
  beta_hat <- sweep(gamma_hat, 1, phi_hat, "/") # n x p

  # Changepoints at positions i >= 2 with nonzero joint norm
  theta_tail <- theta_hat[-1L, , drop = FALSE] # (n-1) x p
  psi_tail <- psi_hat[-1L]
  joint_norms <- sqrt(rowSums(theta_tail^2) + c_scale * psi_tail^2)

  cp <- which(joint_norms > thr) + 1L
  cp_theta <- which(sqrt(rowSums(theta_tail^2)) > thr) + 1L
  cp_psi <- which(abs(psi_tail) > thr) + 1L

  list(
    theta     = theta_hat,
    psi       = psi_hat,
    phi_vec   = phi_hat,
    sigma2    = sigma2_hat,
    beta      = beta_hat,
    cp        = cp,
    cp_theta  = cp_theta,
    cp_psi    = cp_psi,
    iter      = iter,
    converged = converged,
    obj_val   = objective(theta_hat, psi_hat)
  )
}
