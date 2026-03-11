# sbar-cov-fista.R
# SBAR-COV fitted by FISTA (accelerated proximal gradient) with backtracking.
#
# Minimises Q = f + g where:
#   f = (1/n_tr) * L_nat      smooth loss  (eq. 2 of prox_grad.md)
#   g = lambda * sum ||v_i||  group LASSO  (eq. 7.5 of sbar-cov.md)
#
# Each iteration evaluates the gradient/prox step at the extrapolated
# point m, then applies Nesterov momentum to the true iterates w.
# Gradient-based restart resets momentum when extrapolation overshoots.
#
# See prox_grad.md (FISTA section) for the full mathematical development.

#' Fit SBAR-COV via FISTA with backtracking line search
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
#' @param restart   Apply gradient-based momentum restart? (default TRUE)
#' @param verbose   Print iteration log?
#'
#' @return List: theta, psi, phi_vec, sigma2, beta,
#'               cp, cp_theta, cp_psi, obj_val, n_iter
sbar_cov <- function(y,
                     p = 1,
                     lambda_n = 0.1,
                     c_scale = 1,
                     keep_rows = NULL,
                     alpha0 = 1,
                     beta = 0.5,
                     max_iter = 1000,
                     tol = 1e-6,
                     thr = 1e-3,
                     restart = TRUE,
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
  y_tr <- y[keep_rows] # n_tr

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
  compute_gp <- function(th, ps) {
    gamma_cum <- apply(th, 2, fwd_cumsum) # n_tr x p
    g <- rowSums(y_lag_tr * gamma_cum) # n_tr
    phi <- fwd_cumsum(ps) # n_tr
    list(g = g, phi = phi)
  }

  # -----------------------------------------------------------------------
  # 3. Smooth loss  f = (1 / n_tr) * L_nat
  # -----------------------------------------------------------------------
  compute_f <- function(g, phi) {
    if (any(phi <= 0)) {
      return(Inf)
    }
    (1 / (2 * n_tr)) * sum(-log(phi) + phi * y_tr^2 - 2 * y_tr * g + g^2 / phi)
  }

  # -----------------------------------------------------------------------
  # 4. Penalty  g = lambda * sum_{i>=2} ||[theta_i; sqrt(c)*psi_i]||_2
  # -----------------------------------------------------------------------
  compute_g_pen <- function(th, ps) {
    if (n == 1) {
      return(0)
    }
    norms <- sqrt(rowSums(th[-1L, , drop = FALSE]^2) + c_scale * ps[-1L]^2)
    lambda_n * sum(norms)
  }

  # -----------------------------------------------------------------------
  # 5. Gradients of f
  # -----------------------------------------------------------------------
  compute_grad <- function(g, phi) {
    delta <- g / phi - y_tr
    eta <- -1 / (2 * phi) + y_tr^2 / 2 - g^2 / (2 * phi^2)
    grad_th <- (1 / n_tr) * apply(delta * y_lag_tr, 2, bwd_cumsum) # n x p
    grad_ps <- (1 / n_tr) * bwd_cumsum(eta) # n
    list(grad_th = grad_th, grad_ps = grad_ps)
  }

  # -----------------------------------------------------------------------
  # 6. Proximal operator: joint group soft-threshold
  # -----------------------------------------------------------------------
  prox_g <- function(a_th, a_ps, alpha) {
    if (n == 1) {
      return(list(th = a_th, ps = a_ps))
    }
    norms <- sqrt(rowSums(a_th[-1L, , drop = FALSE]^2) + c_scale * a_ps[-1L]^2)
    shrink <- ifelse(norms > 0, pmax(0, 1 - alpha * lambda_n / norms), 0)
    out_th <- a_th
    out_ps <- a_ps
    out_th[-1L, ] <- shrink * a_th[-1L, , drop = FALSE]
    out_ps[-1L] <- shrink * a_ps[-1L]
    list(th = out_th, ps = out_ps)
  }

  # -----------------------------------------------------------------------
  # 7. Initialise
  #    w = true iterate; m = extrapolated point (start equal); s = momentum
  # -----------------------------------------------------------------------
  theta <- matrix(0, n, p)
  psi <- c(1 / max(var(y_tr), 1e-6), rep(0, n - 1))

  m_th <- theta
  m_ps <- psi
  s <- 1

  n_iter <- max_iter

  # -----------------------------------------------------------------------
  # 8. FISTA main loop
  # -----------------------------------------------------------------------
  for (iter in seq_len(max_iter)) {
    # Gradient at the extrapolated point m
    gp_m <- compute_gp(m_th, m_ps)
    f_m <- compute_f(gp_m$g, gp_m$phi)
    grad_m <- compute_grad(gp_m$g, gp_m$phi)

    # --- Backtracking from m ---
    alpha <- alpha0
    repeat {
      th_tent <- m_th - alpha * grad_m$grad_th
      ps_tent <- m_ps - alpha * grad_m$grad_ps

      prx <- prox_g(th_tent, ps_tent, alpha)
      th_new <- prx$th
      ps_new <- prx$ps

      gp_new <- compute_gp(th_new, ps_new)
      if (all(gp_new$phi > 0)) {
        f_new <- compute_f(gp_new$g, gp_new$phi)
        d_th <- th_new - m_th
        d_ps <- ps_new - m_ps
        inner <- sum(grad_m$grad_th * d_th) + sum(grad_m$grad_ps * d_ps)
        sq_norm <- sum(d_th^2) + sum(d_ps^2)
        if (f_new <= f_m + inner + sq_norm / (2 * alpha)) break
      }

      alpha <- beta * alpha
      if (alpha < 1e-16) {
        warning("FISTA: backtracking step size collapsed at iteration ", iter)
        break
      }
    }

    # Proximal gradient mapping norm at the true iterate (stopping criterion)
    pg_norm <- sqrt(sum((th_new - theta)^2) + sum((ps_new - psi)^2)) / alpha

    # --- Momentum restart (O'Donoghue-Candes) ---
    # Reset when <w_new - w_old, m - w_new> > 0: the descent step overshoots
    # the momentum point, so momentum is fighting descent. Disable to run
    # plain FISTA.
    do_restart <- restart && (
      sum((th_new - theta) * (m_th - th_new)) +
        sum((ps_new - psi) * (m_ps - ps_new)) > 0
    )

    # --- Nesterov momentum update ---
    s_new <- (1 + sqrt(1 + 4 * s^2)) / 2

    if (do_restart) {
      m_th <- th_new
      m_ps <- ps_new
      s_new <- 1
    } else {
      coef <- (s - 1) / s_new
      m_th <- th_new + coef * (th_new - theta)
      m_ps <- ps_new + coef * (ps_new - psi)
    }

    theta <- th_new
    psi <- ps_new
    s <- s_new

    if (verbose) {
      g_pen <- compute_g_pen(theta, psi)
      f_cur <- compute_f(gp_new$g, gp_new$phi)
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
  # 9. Recover original parameters from natural parametrisation
  # -----------------------------------------------------------------------
  phi_hat <- cumsum(psi)
  gamma_hat <- apply(theta, 2, cumsum)
  sigma2_hat <- ifelse(phi_hat > 0, 1 / phi_hat, NA_real_)
  beta_hat <- sweep(gamma_hat, 1, phi_hat, "/")

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
