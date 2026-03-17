# cv-hsbar.R
# Interpolation-based cross-validation for H-SBAR (Safikhani & Shojaie 2022).
#
# Methodology (see tscv.md):
#   1. Choose k equally spaced validation points T = {t_1,...,t_k} with
#      spacing s > p so that the AR predictors of each Y_t are never held out.
#   2. For each t in T remove from the regression matrix:
#        - the validation row  (row t  : Y_t is the response)
#        - the p poisoned rows (rows t+1..t+p : Y_t would appear as predictor)
#   3. Fit H-SBAR on the thinned training matrix via hsbar(keep_rows=...).
#   4. Recover beta_hat[t,] at each validation time and predict
#        Y_hat_t = x_t %*% beta_hat[t,]   (x_t lags are in the training set)
#   5. MSPE(lambda) = mean((Y_t - Y_hat_t)^2) over t in T.
#
# Varies lambda over a log-linear path; c_scale is fixed (not cross-validated).

if (!exists("hsbar", mode = "function")) source("hsbar.R")

# ============================================================
# Public: interpolation CV over a lambda path
# ============================================================

#' Interpolation cross-validation for H-SBAR over a lambda path
#'
#' Holds out equally spaced points, fits on the thinned matrix via
#' \code{hsbar(keep_rows=...)}, predicts the held-out responses,
#' and reports MSPE per lambda value.
#'
#' @param y           Numeric time series (length n).
#' @param p           AR order (no intercept).
#' @param lambda    Numeric vector of lambda candidates.
#'                    Default: 10-point log grid on [1e-3, 1].
#' @param c_scale     Fixed scale parameter c (Section 8.3).  Not cross-validated.
#'                    Default 1.
#' @param val_spacing         Spacing s between validation points. Must be > p.
#'                            Default: 10.
#' @param drop_poisoned_rows  If \code{TRUE}, also exclude the p rows following
#'                            each validation point (rows \code{t+1..t+p}), which
#'                            contain \code{Y_t} as a lagged predictor and would
#'                            otherwise leak the held-out value into training.
#'                            If \code{FALSE} (default), only the validation row
#'                            itself is excluded, matching Abolfazl's implementation;
#'                            this admits a small downward bias in the CV error.
#' @param lambda_rule         One of \code{"min"} or \code{"1se"}.
#'                            \code{"min"} (default) returns the lambda minimising MSPE.
#'                            \code{"1se"} applies a 1-SE rule: among all
#'                            lambdas \emph{at least as large} as the minimiser,
#'                            selects the largest one whose MSPE is within one
#'                            standard error of the minimum — preferring a more
#'                            regularised model when the gain is not significant.
#'                            Matches the spirit of Abolfazl's \code{cv.var} rule.
#' @param verbose             Print per-lambda progress? Default TRUE.
#' @param ...         Additional arguments forwarded to \code{hsbar()}
#'                    (e.g. \code{solver}, \code{thr}).
#'
#' @return A list with:
#' \describe{
#'   \item{cv_table}{Data frame: lambda, mspe, mspe_se. One row per lambda value.}
#'   \item{best}{One-row data frame with the selected lambda (per \code{lambda_rule}).}
#'   \item{val_points}{Integer vector of validation time indices.}
#'   \item{keep_rows}{Integer vector of training row indices.}
#' }
cv_hsbar <- function(y,
                     p = 1L,
                     lambda = NULL,
                     c_scale = 1,
                     val_spacing = 10L,
                     drop_poisoned_rows = FALSE,
                     lambda_rule = c("min", "1se"),
                     verbose = TRUE,
                     ...) {
  n <- length(y)

  # ---- build lambda path (decreasing for warm restarts) ------------------
  lambda_vec <- if (is.null(lambda)) 10^seq(-3, 0, length.out = 10) else lambda
  lambda_vec <- sort(lambda_vec, decreasing = TRUE)
  n_lambda <- length(lambda_vec)

  # ---- validation set ----------------------------------------------------
  if (val_spacing <= p) {
    stop(sprintf("val_spacing (%d) must be > p (%d).", val_spacing, p))
  }

  # Valid range: t >= p+1 (so all p lags exist) and t+p <= n
  first_valid <- p + val_spacing
  last_valid <- n - p
  val_points <- seq(first_valid, last_valid, by = val_spacing)
  k <- length(val_points)

  if (k < 2L) {
    stop("Fewer than 2 validation points. Reduce val_spacing or increase n.")
  }

  # ---- excluded rows for each validation point ---------------------------
  # drop_poisoned_rows = TRUE : remove validation row + p downstream rows
  #   (rows t+1..t+p contain Y_t as a lag, leaking the held-out value into training)
  # drop_poisoned_rows = FALSE: remove only the validation row itself
  #   (matches Safikhani & Shojaie 2022 / Abolfazl's implementation; admits a small
  #    downward bias in CV error for lambda selection)
  excl_rows <- sort(unique(as.integer(
    unlist(lapply(val_points, function(t) {
      if (drop_poisoned_rows) t:(t + p) else t
    }))
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

  lambda_rule <- match.arg(lambda_rule, c("min", "1se"))

  # ---- main CV loop ------------------------------------------------------
  mspe    <- numeric(n_lambda)
  sq_err  <- matrix(NA_real_, n_lambda, k)  # per-point squared errors (for SE)
  cp_list <- vector("list", n_lambda)        # break points per lambda (for 1-SE)
  warm_theta <- NULL
  warm_psi <- NULL

  for (j in seq_len(n_lambda)) {
    ln <- lambda_vec[j]

    fit <- tryCatch(
      hsbar( # nolint: object_usage_linter
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
      y_hat       <- rowSums(x_val * beta_val) # k predictions
      sq_err[j, ] <- (y_val - y_hat)^2
      mspe[j]     <- mean(sq_err[j, ])
      cp_list[[j]] <- fit$cp
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
  mspe_se  <- apply(sq_err, 1, sd) / sqrt(k)
  cv_table <- data.frame(lambda = lambda_vec, mspe = mspe, mspe_se = mspe_se)

  min_idx  <- which.min(mspe)
  best_idx <- if (lambda_rule == "min") {
    min_idx
  } else {
    # 1-SE: segment-wise residual variance, following Abolfazl's cv.var rule.
    # For each validation point, find which regime it falls in (using the
    # break points from the min-MSPE fit), compute the variance of squared
    # errors within that regime, then aggregate:
    #   cv_var = (1 / k) * sqrt( sum_j var(sq_err in segment of val_point j) )
    # Note: Abolfazl uses all-time-point residuals for within-segment variance;
    # here we use only validation-point squared errors, which is noisier but
    # avoids a separate training-residual pass.
    cp_min    <- cp_list[[min_idx]]   # break points for the min-MSPE fit
    seg_id    <- vapply(val_points, function(t) sum(t > cp_min) + 1L, integer(1L))
    seg_vars  <- tapply(sq_err[min_idx, ], seg_id, stats::var)
    pt_var    <- as.numeric(seg_vars[as.character(seg_id)])
    # singleton segments yield NA var — fall back to global variance
    pt_var[is.na(pt_var)] <- stats::var(sq_err[min_idx, ], na.rm = TRUE)
    cv_var    <- (1 / k) * sqrt(sum(pt_var, na.rm = TRUE))
    threshold <- mspe[min_idx] + cv_var
    # among more-regularised lambdas (indices <= min_idx), pick the largest
    # lambda (smallest index) still within the threshold
    candidates <- which(!is.na(mspe) & seq_len(n_lambda) <= min_idx & mspe <= threshold)
    if (length(candidates) == 0L) min_idx else min(candidates)
  }

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
#' @param cv_result  Output of \code{cv_hsbar()}.
#' @param log_x      Log-scale the lambda axis? Default TRUE.
#' @param ...        Extra arguments passed to \code{plot()}.
plot_cv_hsbar <- function(cv_result, log_x = TRUE, ...) {
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
    main = "H-SBAR cross-validation \u2013 MSPE",
    ...
  )
  abline(v = bx, lty = 2, col = "firebrick")
  legend("topright",
    legend = sprintf("Best = %.4g\nMSPE = %.5g", best$lambda, best$mspe),
    bty = "n", text.col = "firebrick"
  )

  invisible(cv_result)
}
