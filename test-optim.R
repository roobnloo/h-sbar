source("generate-data.R")
source("hsbar-cvxr.R")
source("hsbar.R")

set.seed(1)
lambda <- 0.005

# Loop over scenarios 1 through 5
for (scenario in 1:5) {
  cat("\n", rep("=", 60), "\n")
  cat("SCENARIO", scenario, "\n")
  cat(rep("=", 60), "\n")

  # Generate data for current scenario
  dat <- switch(scenario,
    "1" = generate_scenario1(),
    "2" = generate_scenario2(),
    "3" = generate_scenario3(),
    "4" = generate_scenario4(),
    "5" = generate_scenario5()
  )

  y <- dat$Y
  p <- dat$p

  cat("Data: n =", length(y), ", p =", p, "\n")

  # Run CVXR
  time_cvxr <- system.time(
    tryCatch(
      fit_cvxr <- hsbar_cvxr(y, p = p, lambda = lambda, verbose = FALSE),
      error = function(e) {
        cat("CVXR error:", e$message, "\n")
        NULL
      }
    )
  )

  # Run FISTA
  time_fista <- system.time(
    fit_fista <- hsbar(y,
      p = p, lambda = lambda, alpha0 = 20,
      max_iter = 2000, tol = 1e-6, restart = TRUE, verbose = FALSE
    )
  )

  # Report results
  cat("\nResults:\n")
  if (!is.null(fit_cvxr)) {
    cat("CVXR   obj:", sprintf("%.6f", fit_cvxr$obj_val), "\n")
  } else {
    cat("CVXR   obj: FAILED\n")
  }
  cat("FISTA  obj:", sprintf("%.6f", fit_fista$obj_val), "\n")

  if (!is.null(fit_cvxr)) {
    cat("Difference:", sprintf("%.2e", fit_cvxr$obj_val - fit_fista$obj_val), "\n")
  }

  cat("\nTiming:\n")
  cat("CVXR  time(s):", sprintf("%.3f", time_cvxr["elapsed"]), "\n")
  cat("FISTA time(s):", sprintf("%.3f", time_fista["elapsed"]), "\n")
  cat("FISTA iters: ", fit_fista$n_iter, "\n")

  cat("\nChangepoints:\n")
  if (!is.null(fit_cvxr)) {
    cat("CVXR : ", paste(fit_cvxr$cp, collapse = ", "), "\n")
  } else {
    cat("CVXR : FAILED\n")
  }
  cat("FISTA: ", paste(fit_fista$cp, collapse = ", "), "\n")
}
