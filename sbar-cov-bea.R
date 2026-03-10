# sbar-cov-bea.R
# Stage 2: Backward Elimination Algorithm for SBAR-COV (Section 9).
#
# The Stage 1 joint group LASSO (sbar_cov) over-selects candidate
# changepoints.  This module prunes them using the profile-likelihood
# information criterion from eq. 9.2: the total profiled NLL across
# segments (eq. 9.1) plus m times a per-break BIC penalty equal to
# (p + 1) * log(n) / 2.  Each joint break adds p + 1 parameters
# (p AR coefficients plus one variance), justifying the BIC count.
# The output reports refined changepoint locations together with
# segment-wise OLS estimates of beta_j and sigma_j^2.

if (!exists("sbar_cov", mode = "function")) source("sbar-cov.R")

# ------------------------------------------------------------------
# Internal: profiled NLL contribution for one segment (eq. 9.1 term)
# ------------------------------------------------------------------

seg_nll <- function(y_seg, x_seg) {
  n_j <- nrow(x_seg)
  p <- ncol(x_seg)
  if (n_j <= p) {
    return(Inf)
  } # underdetermined segment
  qr_j <- qr(x_seg)
  beta_j <- qr.coef(qr_j, y_seg)
  if (anyNA(beta_j)) beta_j[is.na(beta_j)] <- 0
  rss_j <- sum((y_seg - x_seg %*% beta_j)^2)
  rss_j <- max(rss_j, .Machine$double.eps)
  (n_j / 2) * log(rss_j / n_j)
}

# ------------------------------------------------------------------
# Internal: total G_n = sum_j NLL_j for a given changepoint vector
# ------------------------------------------------------------------

compute_g <- function(y, y_lag, cps, n) {
  breaks <- c(1L, cps, n + 1L)
  total <- 0
  for (b in seq_len(length(breaks) - 1L)) {
    rows <- breaks[b]:(breaks[b + 1L] - 1L)
    total <- total + seg_nll(y[rows], y_lag[rows, , drop = FALSE])
  }
  total
}

# ------------------------------------------------------------------
# Internal: segment-wise OLS beta and sigma^2 for the pruned model
# ------------------------------------------------------------------

refit_segments <- function(y, y_lag, cps, n, p) {
  breaks <- c(1L, cps, n + 1L)
  beta_out <- matrix(0, n, p)
  sig2_out <- numeric(n)
  for (b in seq_len(length(breaks) - 1L)) {
    rows <- breaks[b]:(breaks[b + 1L] - 1L)
    n_j <- length(rows)
    x_j <- y_lag[rows, , drop = FALSE]
    y_j <- y[rows]
    qr_j <- qr(x_j)
    beta_j <- qr.coef(qr_j, y_j)
    if (anyNA(beta_j)) beta_j[is.na(beta_j)] <- 0
    rss_j <- sum((y_j - x_j %*% beta_j)^2)
    beta_out[rows, ] <- matrix(beta_j, nrow = n_j, ncol = p, byrow = TRUE)
    sig2_out[rows] <- rss_j / n_j
  }
  list(beta = beta_out, sigma2 = sig2_out)
}

# ------------------------------------------------------------------
# Public: BEA pruning of Stage 1 changepoints
# ------------------------------------------------------------------

#' Backward Elimination Algorithm for SBAR-COV (Section 9)
#'
#' Prunes the over-selected candidate changepoints produced by Stage 1
#' (\code{sbar_cov}) using a BIC-based profile-likelihood IC.  Returns
#' the refined changepoint set together with segment-wise OLS estimates
#' of \eqn{\beta_j} and \eqn{\sigma_j^2} for the pruned model.
#'
#' @param fit     Output of \code{sbar_cov()} or \code{cv_sbar_cov()}.
#'                If from \code{cv_sbar_cov()}, \code{sbar_cov()} is
#'                re-run internally with the best \eqn{\lambda_n};
#'                \code{p} must be supplied explicitly in that case.
#' @param y       Numeric time series (length \eqn{n}).
#' @param p       AR order.  Inferred from \code{fit$beta} when
#'                \code{fit} is the output of \code{sbar_cov()}.
#' @param omega_n Penalty per break.  Default: \eqn{(p+1)\log(n)/2}
#'                (BIC, eq. 9.2).
#' @param ...     Additional arguments forwarded to \code{sbar_cov()}
#'                when \code{fit} is from \code{cv_sbar_cov()}.
#'
#' @return A list with:
#' \describe{
#'   \item{cp}{Integer vector of refined changepoint locations.}
#'   \item{beta}{n x p matrix: segment-wise OLS AR coefficients.
#'     Constant within each pruned segment; discarded breaks no longer
#'     produce distinct rows.}
#'   \item{sigma2}{Length-n vector of segment-wise MLE variances.}
#'   \item{ic}{IC value at the final changepoint set.}
#'   \item{omega_n}{Penalty per break used.}
#' }
sbar_cov_bea <- function(fit, y, p = NULL, omega_n = NULL, ...) {
  # ---- resolve input type --------------------------------------------------
  if (!is.null(fit$best)) {
    # fit is from cv_sbar_cov(): refit with the best lambda
    if (is.null(p)) {
      stop(
        "p must be provided when fit is the output of cv_sbar_cov()."
      )
    }
    fit <- sbar_cov(y, p = p, lambda_n = fit$best$lambda_n, ...)
  }

  n <- length(y)
  if (is.null(p)) p <- ncol(fit$beta)
  if (is.null(omega_n)) omega_n <- (p + 1) * log(n) / 2

  # ---- lagged regressor matrix (same convention as sbar-cov.R) -------------
  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }

  # ---- initialise ----------------------------------------------------------
  cps <- sort(fit$cp)

  if (length(cps) == 0L) {
    seg <- refit_segments(y, y_lag, cps, n, p)
    return(list(
      cp      = integer(0L),
      beta    = seg$beta,
      sigma2  = seg$sigma2,
      ic      = compute_g(y, y_lag, cps, n),
      omega_n = omega_n
    ))
  }

  current_ic <- compute_g(y, y_lag, cps, n) + length(cps) * omega_n

  # ---- iterative backward elimination --------------------------------------
  repeat {
    m <- length(cps)
    if (m == 0L) break

    ic_without <- vapply(seq_len(m), function(i) {
      compute_g(y, y_lag, cps[-i], n) + (m - 1L) * omega_n
    }, numeric(1L))

    best_i <- which.min(ic_without)

    if (ic_without[best_i] <= current_ic) {
      cps <- cps[-best_i]
      current_ic <- ic_without[best_i]
    } else {
      break
    }
  }

  # ---- refit pruned segments -----------------------------------------------
  seg <- refit_segments(y, y_lag, cps, n, p)

  list(
    cp      = cps,
    beta    = seg$beta,
    sigma2  = seg$sigma2,
    ic      = current_ic,
    omega_n = omega_n
  )
}
