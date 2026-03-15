source("generate-data.R")
source("chan-sbar-sparsegl.R")
source("chan-sbar-admm.R")

# Inline chan_sbar_cvxr_full() without the blockwise source dependency
library(CVXR)

chan_sbar_cvxr_full <- function(y,
                                p = 1,
                                lambda = 0.1,
                                keep_rows = NULL,
                                solver = "CLARABEL",
                                thr = 1e-3,
                                verbose = FALSE) {
  n <- length(y)
  if (is.null(keep_rows)) keep_rows <- seq_len(n)
  n_tr <- length(keep_rows)

  y_ext <- c(rep(0, p), y)
  y_lag <- matrix(0, n, p)
  for (k in seq_len(p)) {
    y_lag[, k] <- y_ext[(p - k + 1L):(n + p - k)]
  }
  y_lag_tr <- y_lag[keep_rows, , drop = FALSE]

  l_sub <- outer(keep_rows, seq_len(n), ">=") + 0L
  x_full <- matrix(0, n_tr, n * p)
  for (k in seq_len(p)) {
    col_indices <- seq(k, n * p, by = p)
    x_full[, col_indices] <- sweep(l_sub, 1L, y_lag_tr[, k], "*")
  }

  theta_vec_var <- Variable(n * p)
  fitted <- x_full %*% theta_vec_var
  loss <- sum_squares(y[keep_rows] - fitted) / (2 * n_tr)

  # Group L2 penalty on blocks i = 2..n (group 1 = baseline, unpenalised)
  pen <- lambda * Reduce("+", lapply(seq(2L, n), function(i) {
    idx <- ((i - 1L) * p + 1L):(i * p)
    cvxr_norm(theta_vec_var[idx], 2)
  }))

  problem <- Problem(Minimize(loss + pen))
  psolve(problem, solver = solver, verbose = verbose)

  sol_status <- status(problem)
  objval <- value(problem)
  if (!sol_status %in% c("optimal", "optimal_inaccurate")) {
    warning("chan_sbar_cvxr_full solver status: ", sol_status)
  }

  theta_hat <- matrix(value(theta_vec_var), nrow = n, ncol = p, byrow = TRUE)
  beta_hat <- apply(theta_hat, 2, cumsum)

  theta_norms <- sqrt(rowSums(theta_hat[-1L, , drop = FALSE]^2))
  cp_raw <- which(theta_norms > thr) + 1L

  filter_cp <- function(cands) {
    x <- cands[cands > p + 3L & cands < n]
    if (length(x) > 1L) {
      too_close <- which(diff(x) <= p + 1L)
      if (length(too_close) > 0L) x <- x[-too_close]
    }
    x
  }

  list(
    theta   = theta_hat,
    beta    = beta_hat,
    cp      = filter_cp(cp_raw),
    status  = sol_status,
    obj_val = chan_sbar_bcd_obj(theta_hat, y, p, lambda, keep_rows)
  )
}

set.seed(1)
lambda <- 0.005

for (scenario in 1:3) {
  cat("\n", rep("=", 60), "\n")
  cat("SCENARIO", scenario, "\n")
  cat(rep("=", 60), "\n")

  dat <- switch(scenario,
    "1" = generate_scenario1(),
    "2" = generate_scenario2(),
    "3" = generate_scenario3()
  )

  y <- dat$Y
  p <- dat$p

  cat("Data: n =", length(y), ", p =", p, "\n")

  # Run CVXR
  time_cvxr <- system.time(
    tryCatch(
      fit_cvxr <- chan_sbar_cvxr_full(y, p = p, lambda = lambda, verbose = FALSE),
      error = function(e) {
        cat("CVXR error:", e$message, "\n")
        fit_cvxr <<- NULL
      }
    )
  )

  # Run sparsegl
  time_sparsegl <- system.time(
    tryCatch(
      fit_sparsegl <- chan_sbar(y, p = p, lambda = lambda),
      error = function(e) {
        cat("sparsegl error:", e$message, "\n")
        fit_sparsegl <<- NULL
      }
    )
  )

  # Run BCD
  time_bcd <- system.time(
    tryCatch(
      fit_bcd <- chan_sbar_bcd(y, p = p, lambda = lambda),
      error = function(e) {
        cat("BCD error:", e$message, "\n")
        fit_bcd <<- NULL
      }
    )
  )

  # Run ADMM (FISTA)
  time_admm <- system.time(
    tryCatch(
      fit_admm <- chan_sbar_admm(y, p = p, lambda = lambda),
      error = function(e) {
        cat("ADMM error:", e$message, "\n")
        fit_admm <<- NULL
      }
    )
  )

  cat("\nResults:\n")
  if (!is.null(fit_cvxr)) {
    cat("CVXR      obj:", sprintf("%.6f", fit_cvxr$obj_val), "\n")
  } else {
    cat("CVXR      obj: FAILED\n")
  }
  if (!is.null(fit_sparsegl)) {
    cat("sparsegl  obj:", sprintf("%.6f", fit_sparsegl$obj_val), "\n")
  } else {
    cat("sparsegl  obj: FAILED\n")
  }
  if (!is.null(fit_admm)) {
    cat("ADMM      obj:", sprintf("%.6f", fit_admm$obj_val), "\n")
  } else {
    cat("ADMM      obj: FAILED\n")
  }

  if (!is.null(fit_cvxr) && !is.null(fit_sparsegl)) {
    cat("Diff CVXR-sparsegl:", sprintf("%.2e", fit_cvxr$obj_val - fit_sparsegl$obj_val), "\n")
  }
  if (!is.null(fit_cvxr) && !is.null(fit_admm)) {
    cat("Diff CVXR-ADMM:    ", sprintf("%.2e", fit_cvxr$obj_val - fit_admm$obj_val), "\n")
  }

  cat("\nTiming:\n")
  cat("CVXR     time(s):", sprintf("%.3f", time_cvxr["elapsed"]), "\n")
  cat("sparsegl time(s):", sprintf("%.3f", time_sparsegl["elapsed"]), "\n")
  cat("ADMM     time(s):", sprintf("%.3f", time_admm["elapsed"]), "\n")

  cat("\nChangepoints:\n")
  if (!is.null(fit_cvxr)) {
    cat("CVXR    :", paste(fit_cvxr$cp, collapse = ", "), "\n")
  } else {
    cat("CVXR    : FAILED\n")
  }
  if (!is.null(fit_sparsegl)) {
    cat("sparsegl:", paste(fit_sparsegl$cp, collapse = ", "), "\n")
  } else {
    cat("sparsegl: FAILED\n")
  }
  if (!is.null(fit_admm)) {
    cat("ADMM    :", paste(fit_admm$cp, collapse = ", "), "\n")
  } else {
    cat("ADMM    : FAILED\n")
  }
}
