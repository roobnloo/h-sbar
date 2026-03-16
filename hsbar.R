# hsbar.R
# H-SBAR fitted by FISTA (accelerated proximal gradient) with backtracking.
#
# Minimises Q = f + g where:
#   f = (1/n_tr) * L_nat      smooth loss  (eq. 2 of prox_grad.md)
#   g = lambda * sum ||v_i||  group LASSO  (eq. 7.5 of hsbar.md)
#
# Each iteration evaluates the gradient/prox step at the extrapolated
# point m, then applies Nesterov momentum to the true iterates w.
# Gradient-based restart resets momentum when extrapolation overshoots.
#
# See prox_grad.md (FISTA section) for the full mathematical development.

#' Fit H-SBAR via FISTA with backtracking line search
#'
#' @param y         Numeric time series (length n)
#' @param p         AR order (no intercept)
#' @param lambda  Joint penalty strength
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
#' @param scale_y   Standardise y by sd(y[keep_rows]) before fitting to
#'                  improve conditioning when sigma is large? (default TRUE)
#' @param eps_tol   Stop if the relative objective change falls below this
#'                  between iterations: |dQ| < eps_tol*(1+|Q|) (default 1e-6).
#' @param verbose   Print iteration log?
#'
#' @return List: theta, psi, phi_vec, sigma2, beta,
#'               cp, cp_theta, cp_psi, obj_val, n_iter, stop_crit.
#'   \code{stop_crit} is one of \code{"pg_norm"}, \code{"obj_change"},
#'   or \code{"max_iter"}.
hsbar <- function(y,
                  p = 1,
                  lambda = 0.1,
                  c_scale = 1,
                  keep_rows = NULL,
                  alpha0 = 1,
                  beta = 0.5,
                  max_iter = 1000,
                  tol = 1e-6,
                  thr = 1e-3,
                  restart = TRUE,
                  scale_y = TRUE,
                  eps_tol = 1e-6,
                  verbose = FALSE,
                  init_theta = NULL,
                  init_psi = NULL) {
  n <- length(y)
  if (is.null(keep_rows)) keep_rows <- seq_len(n)
  n_tr <- length(keep_rows)

  # -----------------------------------------------------------------------
  # 1. Standardise y so the Lipschitz constant is O(1) regardless of sigma.
  #    Scaling y before the lag matrix is built automatically scales y_lag
  #    by the same factor, keeping the AR likelihood internally consistent.
  #    phi and sigma2 are restored to original units after the solver.
  # -----------------------------------------------------------------------
  y_scale <- if (scale_y) sd(y[keep_rows]) else 1.0
  if (y_scale < 1e-10) y_scale <- 1.0
  y <- y / y_scale

  # -----------------------------------------------------------------------
  # 2. Lagged regressor matrix: y_lag[t, k] = y[t-k], pre-sample = 0
  # -----------------------------------------------------------------------
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }

  y_lag_tr <- y_lag[keep_rows, , drop = FALSE] # n_tr x p
  y_tr <- y[keep_rows] # n_tr

  # -----------------------------------------------------------------------
  # 3. Cumulative-sum helpers (replace the O(n * n_tr) l_sub matrix)
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
  # 4. Smooth loss  f = (1 / n_tr) * L_nat
  # -----------------------------------------------------------------------
  compute_f <- function(g, phi) {
    if (any(phi <= 0)) {
      return(Inf)
    }
    (1 / (2 * n_tr)) * sum(-log(phi) + phi * y_tr^2 - 2 * y_tr * g + g^2 / phi)
  }

  # -----------------------------------------------------------------------
  # 5. Penalty  g = lambda * sum_{i>=2} ||[theta_i; sqrt(c)*psi_i]||_2
  # -----------------------------------------------------------------------
  compute_g_pen <- function(th, ps) {
    if (n == 1) {
      return(0)
    }
    norms <- sqrt(rowSums(th[-1L, , drop = FALSE]^2) + c_scale * ps[-1L]^2)
    lambda * sum(norms)
  }

  # -----------------------------------------------------------------------
  # 6. Gradients of f
  # -----------------------------------------------------------------------
  compute_grad <- function(g, phi) {
    delta <- g / phi - y_tr
    eta <- -1 / (2 * phi) + y_tr^2 / 2 - g^2 / (2 * phi^2)
    grad_th <- (1 / n_tr) * apply(delta * y_lag_tr, 2, bwd_cumsum) # n x p
    grad_ps <- (1 / n_tr) * bwd_cumsum(eta) # n
    list(grad_th = grad_th, grad_ps = grad_ps)
  }

  # -----------------------------------------------------------------------
  # 7. Proximal operator: joint group soft-threshold
  # -----------------------------------------------------------------------
  prox_g <- function(a_th, a_ps, alpha) {
    if (n == 1) {
      return(list(th = a_th, ps = a_ps))
    }
    norms <- sqrt(rowSums(a_th[-1L, , drop = FALSE]^2) + c_scale * a_ps[-1L]^2)
    shrink <- ifelse(norms > 0, pmax(0, 1 - alpha * lambda / norms), 0)
    out_th <- a_th
    out_ps <- a_ps
    out_th[-1L, ] <- shrink * a_th[-1L, , drop = FALSE]
    out_ps[-1L] <- shrink * a_ps[-1L]
    list(th = out_th, ps = out_ps)
  }

  # -----------------------------------------------------------------------
  # 8. Initialise
  #    w = true iterate; m = extrapolated point (start equal); s = momentum
  # -----------------------------------------------------------------------
  if (!is.null(init_theta) && !is.null(init_psi)) {
    theta <- init_theta
    psi <- init_psi
  } else {
    theta <- matrix(0, n, p)
    psi <- c(1 / max(var(y_tr), 1e-6), rep(0, n - 1))
  }

  m_th <- theta
  m_ps <- psi
  s <- 1

  n_iter <- max_iter
  obj_prev <- Inf
  stop_crit <- "max_iter"

  # -----------------------------------------------------------------------
  # 9. FISTA main loop
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

    # True proximal gradient norm at the updated iterate.
    # FISTA steps from extrapolated m, so ||th_new - theta_old|| / alpha mixes
    # momentum with the gradient; G_alpha(theta) is zero at the optimum and
    # converges monotonically rather than oscillating with the momentum.
    # gp_new already holds compute_gp(theta, psi) since theta == th_new.
    gp_t    <- gp_new
    grad_t  <- compute_grad(gp_t$g, gp_t$phi)
    prx_t   <- prox_g(theta - alpha * grad_t$grad_th,
                      psi   - alpha * grad_t$grad_ps, alpha)
    pg_norm <- sqrt(sum((prx_t$th - theta)^2) +
                      sum((prx_t$ps - psi)^2)) / alpha

    f_cur <- compute_f(gp_t$g, gp_t$phi)
    g_pen <- compute_g_pen(theta, psi)
    obj_cur <- f_cur + g_pen

    if (verbose) {
      rst <- if (do_restart) " R" else "  "
      fmt <- "iter %4d%s| f=%.6f | g=%.6f | Q=%.6f | pg=%.2e | a=%.2e"
      message(sprintf(fmt, iter, rst, f_cur, g_pen, obj_cur, pg_norm, alpha))
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
  # 10. Recover original parameters from natural parametrisation
  # -----------------------------------------------------------------------
  phi_hat <- cumsum(psi)
  gamma_hat <- apply(theta, 2, cumsum)
  sigma2_hat <- ifelse(phi_hat > 0, 1 / phi_hat, NA_real_)
  beta_hat <- sweep(gamma_hat, 1, phi_hat, "/")

  # Restore phi and sigma2 to original (unscaled) units.
  # The solver worked on y/y_scale, so phi_hat = y_scale^2 * phi_orig and
  # sigma2_hat = sigma2_orig / y_scale^2.  beta_hat is scale-invariant.
  phi_hat <- phi_hat / y_scale^2
  sigma2_hat <- sigma2_hat * y_scale^2

  # -----------------------------------------------------------------------
  # 11. Detect changepoints
  # -----------------------------------------------------------------------
  theta_tail <- theta[-1L, , drop = FALSE]
  psi_tail <- psi[-1L]
  joint_norms <- sqrt(rowSums(theta_tail^2) + c_scale * psi_tail^2)

  cp <- which(joint_norms > thr) + 1L
  cp_theta <- which(sqrt(rowSums(theta_tail^2)) > thr) + 1L
  cp_psi <- which(abs(psi_tail) > thr) + 1L

  # -----------------------------------------------------------------------
  # 12. Post-processing: boundary trim and minimum-spacing filter
  #     - Drop candidates within p+3 of start or at the final index n
  #     - Of any two adjacent candidates with gap <= p+1, drop the earlier
  # -----------------------------------------------------------------------
  filter_cp <- function(candidates) {
    x <- candidates[candidates > p + 3L & candidates < n]
    if (length(x) > 1L) {
      too_close <- which(diff(x) <= p + 1L)
      if (length(too_close) > 0L) x <- x[-too_close]
    }
    x
  }
  cp <- filter_cp(cp)
  cp_theta <- filter_cp(cp_theta)
  cp_psi <- filter_cp(cp_psi)

  # obj_val is evaluated in the scaled (y/y_scale) units used by the solver.
  # It differs from the original-unit likelihood by a constant log(y_scale)
  # per observation, which does not affect optimization or changepoint detection.
  gp_final <- compute_gp(theta, psi)
  obj_val <- compute_f(gp_final$g, gp_final$phi) + compute_g_pen(theta, psi)

  list(
    theta = theta,
    psi = psi,
    phi_vec = phi_hat,
    sigma2 = sigma2_hat,
    beta = beta_hat,
    cp = cp,
    cp_theta = cp_theta,
    cp_psi = cp_psi,
    obj_val = obj_val,
    n_iter = n_iter,
    stop_crit = stop_crit
  )
}
