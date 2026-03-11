# cv-sbar-cov.R
# Interpolation-based cross-validation for SBAR-COV (Safikhani & Shojaie 2022).
#
# Methodology (see tscv.md):
#   1. Choose k equally spaced validation points T = {t_1,...,t_k} with
#      spacing s > p so that the AR predictors of each Y_t are never held out.
#   2. For each t in T remove from the regression matrix:
#        - the validation row  (row t  : Y_t is the response)
#        - the p poisoned rows (rows t+1..t+p : Y_t would appear as predictor)
#   3. Fit SBAR-COV on the thinned training matrix via sbar_cov(keep_rows=...).
#   4. Recover beta_hat[t,] at each validation time and predict
#        Y_hat_t = x_t %*% beta_hat[t,]   (x_t lags are in the training set)
#   5. MSPE(lambda) = mean((Y_t - Y_hat_t)^2) over t in T.
#
# Varies lambda over a log-linear path; c_scale is fixed (not cross-validated).

if (!exists("sbar_cov", mode = "function")) source("sbar-cov.R")

# ============================================================
# Public: interpolation CV over a lambda path
# ============================================================

#' Interpolation cross-validation for SBAR-COV over a lambda path
#'
#' Holds out equally spaced points, fits on the thinned matrix via
#' \code{sbar_cov(keep_rows=...)}, predicts the held-out responses,
#' and reports MSPE per lambda value.
#'
#' @param y           Numeric time series (length n).
#' @param p           AR order (no intercept).
#' @param lambda    Numeric vector of lambda candidates.
#'                    Default: 10-point log grid on [1e-3, 1].
#' @param c_scale     Fixed scale parameter c (Section 8.3).  Not cross-validated.
#'                    Default 1.
#' @param val_spacing Spacing s between validation points. Must be > p.
#'                    Default: \code{max(p + 1, round(n / 10))}.
#' @param verbose     Print per-lambda progress? Default TRUE.
#' @param ...         Additional arguments forwarded to \code{sbar_cov()}
#'                    (e.g. \code{solver}, \code{thr}).
#'
#' @return A list with:
#' \describe{
#'   \item{cv_table}{Data frame: lambda, mspe. One row per lambda value.}
#'   \item{best}{One-row data frame with the lambda minimising MSPE.}
#'   \item{val_points}{Integer vector of validation time indices.}
#'   \item{keep_rows}{Integer vector of training row indices.}
#' }
cv_sbar_cov <- function(y,
                        p = 1L,
                        lambda = NULL,
                        c_scale = 1,
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
  l_val <- base::outer(val_points, seq_len(n), ">=") + 0L # k x n

  # ---- main CV loop ------------------------------------------------------
  mspe <- numeric(n_lambda)
  warm_theta <- NULL
  warm_psi <- NULL

  for (j in seq_len(n_lambda)) {
    ln <- lambda_vec[j]

    fit <- tryCatch(
      sbar_cov( # nolint: object_usage_linter
        y, p,
        lambda = ln,
        c_scale = c_scale,
        keep_rows = keep_rows,
        init_theta = warm_theta,
        init_psi = warm_psi,
        ...
      ),
      error = function(e) NULL
    )

    if (is.null(fit)) {
      mspe[j] <- NA_real_
    } else {
      warm_theta <- fit$theta
      warm_psi <- fit$psi
      gamma_val <- l_val %*% fit$theta # k x q
      phi_val <- as.vector(l_val %*% fit$psi) # k
      beta_val <- sweep(gamma_val, 1, phi_val, "/") # k x q
      y_hat <- rowSums(x_val * beta_val) # k predictions
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

#' Plot MSPE along the lambda path (elbow plot)
#'
#' @param cv_result  Output of \code{cv_sbar_cov()}.
#' @param log_x      Log-scale the lambda axis? Default TRUE.
#' @param ...        Extra arguments passed to \code{plot()}.
plot_cv_sbar_cov <- function(cv_result, log_x = TRUE, ...) {
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
    main = "SBAR-COV cross-validation \u2013 MSPE",
    ...
  )
  abline(v = bx, lty = 2, col = "firebrick")
  legend("topright",
    legend = sprintf("Best = %.4g\nMSPE = %.5g", best$lambda, best$mspe),
    bty = "n", text.col = "firebrick"
  )

  invisible(cv_result)
}
