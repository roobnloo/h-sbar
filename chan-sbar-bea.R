# chan-sbar-bea.R
# Stage 2 Backward Elimination Algorithm for Chan et al. (2014) SBAR.
#
# Prunes over-selected Stage 1 changepoints using an information criterion.
# Three IC variants are available (ic_type argument):
#
#   "rss"          IC = sum_j RSS_j + m * omega_n             (Chan 2014 eq. 2.9)
#   "sigma_scaled" IC = sum_j RSS_j + m * sigma2_hat * omega_n  (BIC-style)
#                  where sigma2_hat = RSS_global / (n - p)
#   "profile_lik"  IC = sum_j n_j * log(RSS_j / n_j) + m * omega_n  (scale-invariant)
#
# In all cases omega_n = c_omega_n * p * log(n)  (default c_omega_n = 1).
#
# Reference:
#   Chan, N. H., Yau, C. Y., & Zhang, R. M. (2014). Group LASSO for
#   structural break time series. JASA, 109(506), 590-599.

if (!exists("chan_sbar_admm", mode = "function")) source("chan-sbar-admm.R")

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
# Internal: IC = sum_j n_j * log(RSS_j / n_j) + m * omega_n
#           (profile likelihood; scale-invariant)
# ------------------------------------------------------------------
chan_compute_ic_profile <- function(y, y_lag, cps, n, omega_n) {
  breaks <- c(1L, cps, n + 1L)
  total <- 0
  for (b in seq_len(length(breaks) - 1L)) {
    rows <- breaks[b]:(breaks[b + 1L] - 1L)
    n_j <- length(rows)
    rss_j <- chan_seg_rss(y[rows], y_lag[rows, , drop = FALSE])
    total <- total + (n_j / 2) * log(rss_j / n_j)
  }
  total + length(cps) * omega_n
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
#' using an information criterion.  Three variants are available via
#' \code{ic_type}:
#' \describe{
#'   \item{"rss"}{Plain RSS: \eqn{IC = \sum_j RSS_j + m\,\omega_n}
#'     (Chan 2014 eq. 2.9).}
#'   \item{"sigma_scaled"}{BIC-style: \eqn{IC = \sum_j RSS_j + m\,\hat\sigma^2\,\omega_n}
#'     where \eqn{\hat\sigma^2 = RSS_{global}/(n-p)}.  Restores natural scaling
#'     when \eqn{\sigma \ne 1}.}
#'   \item{"profile_lik"}{Profile likelihood: \eqn{IC = \sum_j n_j\log(RSS_j/n_j) + m\,\omega_n}.
#'     Scale-invariant; equivalent to profiling out \eqn{\sigma^2} per segment.}
#' }
#'
#' @param fit       Output of \code{chan_sbar()}.
#' @param y         Numeric time series (length n).
#' @param p         AR order.  Inferred from \code{fit$beta} if omitted.
#' @param c_omega_n Scale factor: \eqn{\omega_n = c\_omega\_n \cdot p \cdot \log(n)}.
#'   Default 1.
#' @param ic_type   IC variant: \code{"rss"} (default), \code{"sigma_scaled"},
#'   or \code{"profile_lik"}.
#'
#' @return A list with:
#' \describe{
#'   \item{cp}{Integer vector of refined changepoint locations.}
#'   \item{beta}{n x p matrix: segment-wise OLS AR coefficients.}
#'   \item{sigma2}{Length-n vector of segment-wise MLE variances.}
#'   \item{ic}{IC value at the final changepoint set.}
#'   \item{omega_n}{Penalty per break used (after any sigma scaling).}
#'   \item{ic_type}{The IC variant used.}
#' }
chan_sbar_bea <- function(fit, y, p = NULL, c_omega_n = 1,
                          ic_type = c("rss", "sigma_scaled", "profile_lik")) {
  ic_type <- match.arg(ic_type)
  n <- length(y)
  if (is.null(p)) p <- ncol(fit$beta)
  omega_n <- p * log(n)

  # Lagged regressor matrix (same convention as chan_sbar)
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }

  # For sigma_scaled: estimate global sigma^2 from single-segment OLS
  if (ic_type == "sigma_scaled") {
    global_rss <- chan_seg_rss(y, y_lag)
    sigma2_hat <- global_rss / max(n - p, 1L)
    omega_n <- omega_n * sigma2_hat
  }
  omega_n <- c_omega_n * omega_n

  # Dispatch IC function once; BEA loop closes over it
  ic_fn <- switch(ic_type,
    rss          = function(cps) chan_compute_ic(y, y_lag, cps, n, omega_n),
    sigma_scaled = function(cps) chan_compute_ic(y, y_lag, cps, n, omega_n),
    profile_lik  = function(cps) chan_compute_ic_profile(y, y_lag, cps, n, omega_n)
  )

  cps <- sort(fit$cp)

  if (length(cps) == 0L) {
    seg <- chan_refit_segments(y, y_lag, cps, n, p)
    return(list(
      cp      = integer(0L),
      beta    = seg$beta,
      sigma2  = seg$sigma2,
      ic      = ic_fn(cps),
      omega_n = omega_n,
      ic_type = ic_type
    ))
  }

  current_ic <- ic_fn(cps)

  # Iterative backward elimination
  repeat {
    m <- length(cps)
    if (m == 0L) break

    ic_without <- vapply(seq_len(m), function(i) {
      ic_fn(cps[-i])
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
    omega_n = omega_n,
    ic_type = ic_type
  )
}
