# generate-data.R
# Simulation data generators for piecewise-stationary AR(p) processes.

#' Generate a piecewise-stationary AR(p) process (core generator)
#'
#' @param n           Total number of observations
#' @param break_points Integer vector of break-point indices (1-based).
#'   Each index is the FIRST observation of the new regime.
#'   Length m0 for m0 breaks (m0 + 1 regimes).
#' @param beta_list    List of AR coefficient vectors beta (no intercept), one per
#'   regime; each has length p.
#' @param sigma_vec   Numeric vector of innovation std devs, one per regime.
#' @param noise_ar    AR(1) coefficient for correlated noise (0 = i.i.d. Gaussian)
#' @param seed        RNG seed
#' @return List: Y, n, p, break_points, beta_list, sigma_vec, true_sigma2
generate_ar_piecewise <- function(n, break_points, beta_list, sigma_vec,
                                  noise_ar = 0, seed = 42, n_burnin = 200L) {
  n_regimes <- length(beta_list)
  stopifnot(length(sigma_vec) == n_regimes)
  stopifnot(length(break_points) == n_regimes - 1L)

  p <- length(beta_list[[1L]])
  n_total <- n + p + n_burnin

  set.seed(seed)

  # Regime assignment per observation (1-based)
  regime <- rep(1L, n)
  for (j in seq_along(break_points)) {
    regime[seq(break_points[j], n)] <- j + 1L
  }

  # Simulate (optionally correlated) noise over full burn-in + observation window
  raw_noise <- rnorm(n_total)
  if (noise_ar != 0) {
    eps <- numeric(n_total)
    eps[1L] <- raw_noise[1L]
    for (i in seq(2L, n_total)) {
      eps[i] <- noise_ar * eps[i - 1L] + raw_noise[i]
    }
  } else {
    eps <- raw_noise
  }

  # Simulate the piecewise AR process with proper burn-in
  # t_obs <= 0 during burn-in (regime 1); t_obs in 1..n during observations
  Y <- numeric(n_total)
  for (t in (p + 1L):n_total) {
    t_obs <- t - p - n_burnin
    r <- if (t_obs >= 1L) regime[t_obs] else 1L
    phi <- beta_list[[r]]
    sigma <- sigma_vec[r]
    lags <- Y[(t - 1L):(t - p)]
    Y[t] <- sum(phi * lags) + sigma * eps[t]
  }

  Y <- Y[(p + n_burnin + 1L):n_total]

  list(
    Y = Y,
    n = n,
    p = p,
    break_points = break_points,
    beta_list = beta_list,
    sigma_vec = sigma_vec,
    true_sigma2 = sigma_vec[regime]^2
  )
}


#' Scenario 1: Moderate AR(2), breaks in coefficients only.
#'
#' Regimes (all AR(2), all clearly stationary):
#'   1. beta = ( 0.5,  0.0) -- moderate positive lag-1, no lag-2
#'   2. beta = (-0.4,  0.3) -- sign flip on lag-1, mild lag-2
#'   3. beta = ( 0.6, -0.2) -- moderate positive lag-1, mild negative lag-2
#'
#' Coefficient jumps: |Delta beta1| in {0.9, 1.0}, |Delta beta2| in {0.3, 0.5}.
#'
#' @param seed  RNG seed
#' @param sigma Innovation std dev (constant across regimes, default 1).
#' @return List from generate_ar_piecewise
generate_scenario1 <- function(seed = 42, sigma = 1) {
  generate_ar_piecewise(
    n = 1002,
    break_points = c(334L, 668L),
    beta_list = list(c(0.5, 0.0), c(-0.4, 0.3), c(0.6, -0.2)),
    sigma_vec = c(sigma, sigma, sigma),
    seed = seed
  )
}


#' Scenario 2: Moderate AR(2), breaks in coefficients and variance.
#'
#' Mirrors scenario 1's break structure but adds variance shifts at a 2:1 ratio.
#'
#' Regimes (same AR(2) as scenario 1, sigma shifts by factor 2):
#'   1. beta = ( 0.5,  0.0),  sigma = 0.5 * sigma_scale
#'   2. beta = (-0.4,  0.3),  sigma = 1.0 * sigma_scale
#'   3. beta = ( 0.6, -0.2),  sigma = 0.5 * sigma_scale
#'
#' @param seed        RNG seed
#' @param sigma_scale Multiplier applied to all sigma values (default 1).
#'   Base sigma_vec is c(0.5, 1.0, 0.5).
#' @return List from generate_ar_piecewise
generate_scenario2 <- function(seed = 42, sigma_scale = 1) {
  generate_ar_piecewise(
    n = 1002L,
    break_points = c(334L, 668L),
    beta_list = list(c(0.5, 0.0), c(-0.4, 0.3), c(0.6, -0.2)),
    sigma_vec = c(0.5, 1.0, 0.5) * sigma_scale,
    seed = seed
  )
}


#' Scenario 3: Variance-dominated breaks, weak coefficient signal.
#'
#' Regimes (all AR(2), all clearly stationary):
#'   1. beta = ( 0.50,  0.10),  sigma = 0.2 * sigma_scale
#'   2. beta = ( 0.20,  0.30),  sigma = 1.0 * sigma_scale  -- variance spike
#'   3. beta = ( 0.50,  0.00),  sigma = 0.2 * sigma_scale
#'
#' Coefficient jumps: |Delta beta1| = 0.30 at both breaks;
#'                   |Delta beta2| in {0.20, 0.30}.
#' Variance ratio: 5:1 between regimes 2 and 1/3.
#'
#' @param seed        RNG seed
#' @param sigma_scale Multiplier applied to all sigma values (default 1).
#' @return List from generate_ar_piecewise
generate_scenario3 <- function(seed = 42, sigma_scale = 1) {
  generate_ar_piecewise(
    n = 1002L,
    break_points = c(334L, 668L),
    beta_list = list(c(0.50, 0.10), c(0.20, 0.30), c(0.50, 0.00)),
    sigma_vec = c(0.2, 1.0, 0.2) * sigma_scale,
    seed = seed
  )
}
