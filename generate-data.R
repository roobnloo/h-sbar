# generate-data.R
# Simulation data generators for piecewise-stationary AR(p) processes.
# Five scenarios adapted from Safikhani & Shojaie (2022) and Chan et al. (2014).

#' Generate a piecewise-stationary AR(p) process (core generator)
#'
#' @param n           Total number of observations
#' @param break_points Integer vector of break-point indices (1-based).
#'   Each index is the FIRST observation of the new regime.
#'   Length m0 for m0 breaks (m0 + 1 regimes).
#' @param phi_list    List of AR coefficient vectors (no intercept), one per
#'   regime; each has length p.
#' @param sigma_vec   Numeric vector of innovation std devs, one per regime.
#' @param noise_ar    AR(1) coefficient for correlated noise (0 = i.i.d. Gaussian).
#' @param seed        RNG seed
#' @return List: Y, n, p, break_points, phi_list, sigma_vec, true_sigma2
generate_ar_piecewise <- function(n, break_points, phi_list, sigma_vec,
                                  noise_ar = 0, seed = 42) {
  n_regimes <- length(phi_list)
  stopifnot(length(sigma_vec) == n_regimes)
  stopifnot(length(break_points) == n_regimes - 1L)

  p <- length(phi_list[[1L]])

  set.seed(seed)

  # Regime assignment per observation (1-based)
  regime <- rep(1L, n)
  for (j in seq_along(break_points)) {
    regime[seq(break_points[j], n)] <- j + 1L
  }

  # Simulate (optionally correlated) noise over full burn-in + observation window
  raw_noise <- rnorm(n + p)
  if (noise_ar != 0) {
    eps <- numeric(n + p)
    eps[1L] <- raw_noise[1L]
    for (i in seq(2L, n + p)) {
      eps[i] <- noise_ar * eps[i - 1L] + raw_noise[i]
    }
  } else {
    eps <- raw_noise
  }

  # Simulate the piecewise AR process (burn-in padding of length p)
  Y <- numeric(n + p)
  for (t in (p + 1L):(n + p)) {
    t_obs <- t - p
    r <- regime[t_obs]
    phi <- phi_list[[r]]
    sigma <- sigma_vec[r]
    lags <- Y[(t - 1L):(t - p)]
    Y[t] <- sum(phi * lags) + sigma * eps[t]
  }

  Y <- Y[(p + 1L):(n + p)]

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


#' Scenario 1: Baseline (centre breaks, constant variance)
#'
#' T=300, p=1, m0=2. Breaks at t=100 and t=200.
#' Large coefficient jumps; variance constant at sigma=0.1.
#' Reference: Safikhani & Shojaie (2022), low-noise benchmark.
#'
#' @param seed RNG seed
#' @return List from generate_ar_piecewise
generate_scenario1 <- function(seed = 42) {
  generate_ar_piecewise(
    n            = 300L,
    break_points = c(100L, 200L),
    phi_list     = list(c(-0.6), c(0.75), c(-0.8)),
    sigma_vec    = c(0.1, 0.1, 0.1),
    seed         = seed
  )
}


#' Scenario 2: Boundary sensitivity (breaks near endpoints)
#'
#' T=300, p=1, m0=2. Breaks at t=50 and t=250.
#' Tests detection when a regime occupies few observations.
#' Same coefficient jumps as Scenario 1; constant variance.
#'
#' @param seed RNG seed
#' @return List from generate_ar_piecewise
generate_scenario2 <- function(seed = 42) {
  generate_ar_piecewise(
    n            = 300L,
    break_points = c(50L, 250L),
    phi_list     = list(c(-0.6), c(0.75), c(-0.8)),
    sigma_vec    = c(0.1, 0.1, 0.1),
    seed         = seed
  )
}


#' Scenario 3: Small-jump detection (lower signal-to-noise)
#'
#' T=300, p=2, m0=2. Breaks at t=100 and t=200.
#' Small coefficient changes across regimes; constant variance.
#'
#' @param seed RNG seed
#' @return List from generate_ar_piecewise
generate_scenario3 <- function(seed = 42) {
  generate_ar_piecewise(
    n = 300L,
    break_points = c(100L, 200L),
    phi_list = list(
      c(0.4, 0.2),
      c(0.5, 0.1),
      c(0.3, 0.2)
    ),
    sigma_vec = c(0.1, 0.1, 0.1),
    seed = seed
  )
}


#' Scenario 4: Joint coefficient and variance shift (primary target case)
#'
#' T=500, p=1, m0=2. Breaks at t=150 and t=350.
#' Regime 2 ("crisis") has high persistence and high noise.
#' This is the main scenario for evaluating the SBAR-COV joint penalty.
#'
#' @param seed RNG seed
#' @return List from generate_ar_piecewise
generate_scenario4 <- function(seed = 42) {
  generate_ar_piecewise(
    n            = 500L,
    break_points = c(150L, 350L),
    phi_list     = list(c(0.5), c(0.9), c(0.2)),
    sigma_vec    = c(0.1, 0.4, 0.15),
    seed         = seed
  )
}


#' Scenario 5: Model misspecification (correlated AR(1) errors)
#'
#' T=300, p=1, m0=2. Break points at t=100 and t=200.
#' Uses Scenario 4 dynamics but innovations follow an AR(1) process:
#'   eps_t = 0.5 * eps_{t-1} + nu_t,  nu_t ~ N(0, 1).
#' Tests robustness when the i.i.d. noise assumption is violated.
#'
#' @param seed RNG seed
#' @return List from generate_ar_piecewise
generate_scenario5 <- function(seed = 42) {
  generate_ar_piecewise(
    n            = 300L,
    break_points = c(100L, 200L),
    phi_list     = list(c(0.5), c(0.9), c(0.2)),
    sigma_vec    = c(0.1, 0.4, 0.15),
    noise_ar     = 0.5,
    seed         = seed
  )
}
