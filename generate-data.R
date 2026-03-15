# generate-data.R
# Simulation data generators for piecewise-stationary AR(p) processes.
# Scenario 1 adapted from Chan et al. (2014): CV-friendly dyadic AR(2), coefficient breaks only.
# Scenario 2: CV-friendly equal-thirds AR(2), coefficient and variance breaks.
# Scenario 3: Variance-dominated breaks, moderate coefficient signal.

#' Generate a piecewise-stationary AR(p) process (core generator)
#'
#' @param n           Total number of observations
#' @param break_points Integer vector of break-point indices (1-based).
#'   Each index is the FIRST observation of the new regime.
#'   Length m0 for m0 breaks (m0 + 1 regimes).
#' @param phi_list    List of AR coefficient vectors (no intercept), one per
#'   regime; each has length p.
#' @param sigma_vec   Numeric vector of innovation std devs, one per regime.
#' @param noise_ar    AR(1) coefficient for correlated noise (0 = i.i.d. Gaussian)
#' @param seed        RNG seed
#' @return List: Y, n, p, break_points, phi_list, sigma_vec, true_sigma2
generate_ar_piecewise <- function(n, break_points, phi_list, sigma_vec,
                                  noise_ar = 0, seed = 42, n_burnin = 200L) {
  n_regimes <- length(phi_list)
  stopifnot(length(sigma_vec) == n_regimes)
  stopifnot(length(break_points) == n_regimes - 1L)

  p <- length(phi_list[[1L]])
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
    phi <- phi_list[[r]]
    sigma <- sigma_vec[r]
    lags <- Y[(t - 1L):(t - p)]
    Y[t] <- sum(phi * lags) + sigma * eps[t]
  }

  Y <- Y[(p + n_burnin + 1L):n_total]

  list(
    Y            = Y,
    n            = n,
    p            = p,
    break_points = break_points,
    phi_list     = phi_list,
    sigma_vec    = sigma_vec,
    true_sigma2  = sigma_vec[regime]^2
  )
}


#' Scenario 1: CV-friendly dyadic AR(2), breaks in coefficients only.
#'
#' Mirrors the dyadic break structure from Chan et al. (2014) (n=1024, breaks
#' at t=513 and t=769) but uses moderate AR(2) coefficients well inside the
#' stationarity triangle and constant innovation variance.  The optimal lambda
#' falls in a predictable range and CV works well over a small linear grid
#' (e.g. seq(0.05, 0.35, 0.05)).
#'
#' Regimes (all AR(2), all clearly stationary):
#'   1. phi = ( 0.5,  0.0) -- moderate positive lag-1, no lag-2
#'   2. phi = (-0.4,  0.3) -- sign flip on lag-1, mild lag-2
#'   3. phi = ( 0.6, -0.2) -- moderate positive lag-1, mild negative lag-2
#'
#' Coefficient jumps: |Delta phi1| in {0.9, 1.0}, |Delta phi2| in {0.3, 0.5}.
#'
#' @param seed  RNG seed
#' @param sigma Innovation std dev (constant across regimes, default 1).
#' @return List from generate_ar_piecewise
generate_scenario1 <- function(seed = 42, sigma = 1) {
  generate_ar_piecewise(
    n            = 1024L,
    break_points = c(513L, 769L),
    phi_list     = list(c(0.5, 0.0), c(-0.4, 0.3), c(0.6, -0.2)),
    sigma_vec    = c(sigma, sigma, sigma),
    seed         = seed
  )
}


#' Scenario 2: CV-friendly equal-thirds AR(2), breaks in coefficients and variance.
#'
#' Mirrors scenario 7's break structure (n=1002, equal thirds at t=334 and
#' t=668) and uses the same moderate AR(2) coefficients as scenario 1, but
#' adds variance shifts at a 2:1 ratio.  The controlled heteroskedasticity
#' keeps per-segment CV error manageable, so a small linear lambda grid
#' (e.g. seq(0.05, 0.35, 0.05)) covers the optimum.
#'
#' Regimes (same AR(2) as scenario 1, sigma shifts by factor 2):
#'   1. phi = ( 0.5,  0.0),  sigma = 0.5 * sigma_scale
#'   2. phi = (-0.4,  0.3),  sigma = 1.0 * sigma_scale
#'   3. phi = ( 0.6, -0.2),  sigma = 0.5 * sigma_scale
#'
#' @param seed        RNG seed
#' @param sigma_scale Multiplier applied to all sigma values (default 1).
#'   Base sigma_vec is c(0.5, 1.0, 0.5).
#' @return List from generate_ar_piecewise
generate_scenario2 <- function(seed = 42, sigma_scale = 1) {
  generate_ar_piecewise(
    n            = 1002L,
    break_points = c(334L, 668L),
    phi_list     = list(c(0.5, 0.0), c(-0.4, 0.3), c(0.6, -0.2)),
    sigma_vec    = c(0.5, 1.0, 0.5) * sigma_scale,
    seed         = seed
  )
}


#' Scenario 3: Variance-dominated breaks, moderate coefficient signal.
#'
#' Same equal-thirds structure as scenario 2 (n=1002, breaks at t=334 and
#' t=668).  Coefficient jumps are moderate (~0.2-0.3 per component) — weaker
#' than scenario 2 (|Delta phi1| ~ 0.9-1.0).  The variance ratio is large
#' (5:1), so variance remains the dominant break signal, but the coefficient
#' signal is now large enough for H-SBAR to leverage both channels.  A method
#' insensitive to variance shifts will still struggle relative to H-SBAR.
#'
#' Regimes (all AR(2), all clearly stationary):
#'   1. phi = ( 0.50,  0.10),  sigma = 0.2 * sigma_scale
#'   2. phi = ( 0.20,  0.30),  sigma = 1.0 * sigma_scale  -- variance spike
#'   3. phi = ( 0.50,  0.00),  sigma = 0.2 * sigma_scale
#'
#' Coefficient jumps: |Delta phi1| = 0.30 at both breaks;
#'                   |Delta phi2| in {0.20, 0.30}.
#' Variance ratio: 5:1 between regimes 2 and 1/3.
#'
#' @param seed        RNG seed
#' @param sigma_scale Multiplier applied to all sigma values (default 1).
#'   Base sigma_vec is c(0.2, 1.0, 0.2).
#' @return List from generate_ar_piecewise
generate_scenario3 <- function(seed = 42, sigma_scale = 1) {
  generate_ar_piecewise(
    n            = 1002L,
    break_points = c(334L, 668L),
    phi_list     = list(c(0.50, 0.10), c(0.20, 0.30), c(0.50, 0.00)),
    sigma_vec    = c(0.2, 1.0, 0.2) * sigma_scale,
    seed         = seed
  )
}
