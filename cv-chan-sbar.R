# cv-chan-sbar.R
# Interpolation-based cross-validation for Chan et al. (2014) SBAR.
#
# Mirrors cv-hsbar.R exactly, with two differences:
#   1. Calls chan_sbar() instead of hsbar().
#   2. Prediction at validation point v uses
#        y_hat_v = x_v %*% (l_val[j,] %*% theta)
#      i.e. the cumulative AR coefficient directly, with no phi division,
#      because cumsum(theta)[t,] = beta_t in Chan's model.
#
# See cv-hsbar.R for full methodology (Safikhani & Shojaie 2022).

if (!exists("chan_sbar_admm", mode = "function")) {
  source(file.path(dirname(sys.frame(1)$ofile), "chan-sbar-admm.R"))
}

# ============================================================
# Public: interpolation CV over a lambda path
# ============================================================

#' Interpolation cross-validation for Chan SBAR over a lambda path
#'
#' Holds out equally spaced points, fits on the thinned matrix via
#' \code{chan_sbar(keep_rows=...)}, predicts the held-out responses using
#' the cumulative AR coefficient, and reports MSPE per lambda value.
#'
#' @param y           Numeric time series (length n).
#' @param p           AR order (no intercept).
#' @param lambda      Numeric vector of lambda candidates.
#'                    Default: 10-point log grid on [1e-3, 1].
#' @param val_spacing Spacing s between validation points. Must be > p.
#'                    Default: \code{max(p + 1, round(n / 10))}.
#' @param verbose     Print per-lambda progress? Default TRUE.
#' @param ...         Additional arguments forwarded to \code{chan_sbar()}
#'                    (e.g. \code{thr}, \code{max_iter}).
#'
#' @return A list with:
#' \describe{
#'   \item{cv_table}{Data frame: lambda, mspe. One row per lambda value.}
#'   \item{best}{One-row data frame with the lambda minimising MSPE.}
#'   \item{val_points}{Integer vector of validation time indices.}
#'   \item{keep_rows}{Integer vector of training row indices.}
#' }
cv_chan_sbar <- function(y,
                         p = 1L,
                         lambda = NULL,
                         val_spacing = NULL,
                         verbose = TRUE,
                         ...) {
  n <- length(y)

  # ---- build lambda path (decreasing for warm restarts) ------------------
  lambda_vec <- if (is.null(lambda)) 10^seq(-3, 0, length.out = 10) else lambda
  lambda_vec <- sort(lambda_vec, decreasing = TRUE)
  n_lambda <- length(lambda_vec)

  # ---- validation set ----------------------------------------------------
  if (is.null(val_spacing)) {
    val_spacing <- max(p + 1L, round(n / 10))
  }
  if (val_spacing <= p) {
    stop(sprintf("val_spacing (%d) must be > p (%d).", val_spacing, p))
  }

  # Valid range: t >= p+1 (so all p lags exist) and t+p <= n
  first_valid <- p + 1L
  last_valid <- n - p
  val_points <- seq(first_valid, last_valid, by = val_spacing)
  k <- length(val_points)

  if (k < 2L) {
    stop("Fewer than 2 validation points. Reduce val_spacing or increase n.")
  }

  # ---- excluded rows for each validation point ---------------------------
  excl_rows <- sort(unique(as.integer(
    unlist(lapply(val_points, function(t) t:(t + p)))
  )))
  keep_rows <- setdiff(seq_len(n), excl_rows)

  if (length(keep_rows) < 2L * p) {
    stop("Too few training rows after exclusion. Reduce val_spacing or n.")
  }

  # ---- predictor matrix for validation points ----------------------------
  # x_val[j, ] = (y[t_j-1], ..., y[t_j-p])
  y_ext <- c(rep(0, p), y)
  x_val <- matrix(0, k, p)
  for (lag in seq_len(p)) {
    x_val[, lag] <- y_ext[val_points - lag + p]
  }
  y_val <- y[val_points]

  # ---- l_val: cumsum selector for validation rows ------------------------
  # l_val[j, t] = I(val_points[j] >= t), so l_val %*% theta = cumsum(theta)
  # evaluated at each validation time.
  l_val <- base::outer(val_points, seq_len(n), ">=") + 0L # k x n

  # ---- main CV loop ------------------------------------------------------
  mspe <- numeric(n_lambda)
  prev_theta <- NULL

  for (j in seq_len(n_lambda)) {
    ln <- lambda_vec[j]

    fit <- tryCatch(
      chan_sbar_admm( # nolint: object_usage_linter
        y, p,
        lambda = ln,
        keep_rows = keep_rows,
        init_theta = prev_theta,
        ...
      ),
      error = function(e) NULL
    )

    if (!is.null(fit)) prev_theta <- fit$theta

    if (is.null(fit)) {
      mspe[j] <- NA_real_
    } else {
      # Prediction: beta at validation time v = cumsum(theta)[v, ]
      #             = (l_val %*% theta)[j, ]
      gamma_val <- l_val %*% fit$theta # k x p
      y_hat <- rowSums(x_val * gamma_val) # k predictions
      mspe[j] <- mean((y_val - y_hat)^2)
    }

    if (verbose) {
      n_cp <- if (is.null(fit)) NA_integer_ else length(fit$cp)
      message(sprintf(
        "[%d/%d]  lambda=%.4g  MSPE=%.5g  ncp=%s",
        j, n_lambda, ln, mspe[j], n_cp
      ))
    }
  }

  # ---- assemble output ---------------------------------------------------
  cv_table <- data.frame(lambda = lambda_vec, mspe = mspe)
  best_idx <- which.min(mspe)

  list(
    cv_table   = cv_table,
    best       = cv_table[best_idx, , drop = FALSE],
    val_points = val_points,
    keep_rows  = keep_rows
  )
}

# ============================================================
# Plotting helper: elbow plot of MSPE along the lambda path
# ============================================================

#' Plot MSPE along the lambda path (elbow plot) for Chan SBAR CV
#'
#' @param cv_result  Output of \code{cv_chan_sbar()}.
#' @param log_x      Log-scale the lambda axis? Default TRUE.
#' @param ...        Extra arguments passed to \code{plot()}.
plot_cv_chan_sbar <- function(cv_result, log_x = TRUE, ...) {
  tbl <- cv_result$cv_table
  best <- cv_result$best

  x <- tbl$lambda
  bx <- best$lambda
  if (log_x) {
    x <- log10(x)
    bx <- log10(bx)
  }

  xlab <- if (log_x) expression(log[10](lambda[n])) else expression(lambda[n])

  plot(x, tbl$mspe,
    type = "b", pch = 19, col = "steelblue",
    xlab = xlab,
    ylab = "CV MSPE",
    main = "Chan SBAR cross-validation \u2013 MSPE",
    ...
  )
  abline(v = bx, lty = 2, col = "firebrick")
  legend("topright",
    legend = sprintf("Best = %.4g\nMSPE = %.5g", best$lambda, best$mspe),
    bty = "n", text.col = "firebrick"
  )

  invisible(cv_result)
}
