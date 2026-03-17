# run-sim.R
# Simulation study comparing H-SBAR vs Chan et al. (2014) SBAR.
# Each method: CV-selected lambda + BEA stage-2 pruning.
#
# Usage:
#   Rscript run-sim.R --scenario=N [--sigma=X] [--sigscale=X] [--nrep=N] [--outdir=PATH]
#
#   --scenario=N      Required (see generate-data.R).
#   --sigma=X         Innovation std dev (scenarios 1-3, default 1).
#   --sigscale=X      Multiplier for sigma_vec (scenarios 4-5, default 1).
#   --nrep=N          Number of replications (default 100).
#   --outdir=PATH     Directory for .rds output files (default ./sim, created if needed).
#
# Metrics:
#   Table 1: avg # breaks, % correct, mean/SE of relative break locations
#   Table 2: Hausdorff distance, prediction MSE, AR coef error, sigma2 error
#
# Results saved to run-sim-<label>-results.rds.
# To reproduce tables without re-running:
#   results <- readRDS("run-sim-<label>-results.rds")
# Then run from the "Summarisation helpers" section onward.

# ============================================================
# Command-line arguments
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, key, default = NULL) {
  pat <- paste0("^--", key, "=(.+)$")
  m <- regmatches(args, regexpr(pat, args, perl = TRUE))
  if (length(m) == 0L) default else sub(pat, "\\1", m)
}

scenario_str <- parse_arg(args, "scenario")
if (is.null(scenario_str)) stop("--scenario=N is required (1-10)")
scenario_arg <- as.integer(scenario_str)
if (!scenario_arg %in% c(1:10)) stop("--scenario must be 1-10")

sigma_arg <- as.numeric(parse_arg(args, "sigma", "1"))
sigma_scale_arg <- as.numeric(parse_arg(args, "sigscale", "1"))
nrep_arg <- as.integer(parse_arg(args, "nrep", "100"))
outdir_arg <- parse_arg(args, "outdir", "./sim")

# ============================================================
# Dependencies
# ============================================================

source("generate-data.R")
source("hsbar.R")
source("cv-hsbar.R")
source("hsbar-bea.R")
source("cv-chan-sbar.R")
source("chan-sbar-bea.R")

# ============================================================
# Scenario dispatch
# ============================================================

GENERATE_FN <- switch(as.character(scenario_arg),
  "1" = function(seed) generate_scenario1(seed = seed, sigma = sigma_arg),
  "2" = function(seed) generate_scenario2(seed = seed, sigma = sigma_arg),
  "3" = function(seed) generate_scenario3(seed = seed, sigma = sigma_arg),
  "4" = function(seed) generate_scenario4(seed = seed, sigma_scale = sigma_scale_arg),
  "5" = function(seed) generate_scenario5(seed = seed, sigma_scale = sigma_scale_arg),
  "6" = function(seed) generate_scenario6(seed = seed, sigma = sigma_arg),
  "7" = function(seed) generate_scenario7(seed = seed, sigma_scale = sigma_scale_arg),
  "8" = function(seed) generate_scenario8(seed = seed, sigma = sigma_arg),
  "9" = function(seed) generate_scenario9(seed = seed, sigma_scale = sigma_scale_arg),
  "10" = function(seed) generate_scenario10(seed = seed, sigma_scale = sigma_scale_arg)
)

meta <- GENERATE_FN(seed = 0L)
N <- meta$n
P <- meta$p
TRUE_BREAKS <- meta$break_points
M0 <- length(TRUE_BREAKS)
TRUE_PHI <- meta$phi_list
TRUE_SIGMA_VEC <- meta$sigma_vec

# ============================================================
# Tuning parameters (fixed across reps)
# ============================================================

N_REP <- nrep_arg
LAMBDA_SBAR <- seq(0.01, 0.15, by = 0.01)
LAMBDA_CHAN <- seq(0.01, 0.15, by = 0.01)
C_SCALE <- 1

# ============================================================
# Helpers
# ============================================================

#' Symmetric Hausdorff distance between two integer sets.
#' Returns n when one set is empty and the other is not.
hausdorff_dist <- function(est, tru, n) {
  if (length(est) == 0L && length(tru) == 0L) {
    return(0)
  }
  if (length(est) == 0L || length(tru) == 0L) {
    return(n)
  }
  fwd <- max(sapply(tru, function(b) min(abs(est - b)))) # coverage
  rev <- max(sapply(est, function(a) min(abs(tru - a)))) # precision
  max(fwd, rev)
}

#' Build an n x p matrix of true AR coefficients (regime assignment
#' matches generate_ar_piecewise: break at t means regime changes at t).
make_true_beta_mat <- function(n, breaks, phi_list) {
  p <- length(phi_list[[1L]])
  regime <- rep(1L, n)
  for (j in seq_along(breaks)) {
    regime[seq(breaks[j], n)] <- j + 1L
  }
  beta_mat <- matrix(0, n, p)
  for (t in seq_len(n)) {
    beta_mat[t, ] <- phi_list[[regime[t]]]
  }
  beta_mat
}

#' Predict y_hat[t] = x_t %*% beta[t,] for t = 1..n.
compute_yhat <- function(y, beta, p) {
  n <- length(y)
  y_ext <- c(rep(0, p), y)
  x_all <- matrix(0, n, p)
  for (lag in seq_len(p)) {
    x_all[, lag] <- y_ext[seq_len(n) + p - lag]
  }
  rowSums(x_all * beta)
}

#' Compute all metrics for one (stage, method) result.
compute_metrics <- function(cp_est, beta_est, sigma2_est,
                            y, true_beta, true_sigma2, n, p) {
  ncp <- length(cp_est)
  y_hat <- compute_yhat(y, beta_est, p)
  list(
    ncp         = ncp,
    correct_ncp = (ncp == M0),
    hd          = hausdorff_dist(cp_est, TRUE_BREAKS, n),
    mse         = mean((y - y_hat)^2),
    beta_err    = mean((beta_est - true_beta)^2),
    sigma2_err  = mean((sigma2_est - true_sigma2)^2),
    rel_cp      = if (ncp > 0L) sort(as.integer(cp_est)) / n else numeric(0),
    cp          = if (ncp > 0L) sort(as.integer(cp_est)) else integer(0)
  )
}

# ============================================================
# Single replication
# ============================================================

one_rep <- function(seed, lambda_sbar, lambda_chan, c_scale) {
  dat <- GENERATE_FN(seed)
  n <- dat$n
  p <- dat$p
  true_beta <- make_true_beta_mat(n, TRUE_BREAKS, dat$phi_list)
  true_sigma2 <- dat$true_sigma2

  # -- H-SBAR ------------------------------------------------------------
  sbar_s1 <- sbar_s2 <- NULL
  sbar_best_lambda <- NA_real_

  tryCatch(
    {
      cv_s <- cv_hsbar(dat$Y, p,
        lambda = lambda_sbar,
        c_scale = c_scale,
        max_iter = 5000
      )
      sbar_best_lambda <- cv_s$best$lambda
      fit_s <- hsbar(dat$Y, p,
        lambda = sbar_best_lambda,
        c_scale = c_scale,
        max_iter = 5000
      )
      sbar_s1 <- compute_metrics(
        fit_s$cp, fit_s$beta, fit_s$sigma2,
        dat$Y, true_beta, true_sigma2, n, p
      )
      bea_s <- hsbar_bea(fit_s, y = dat$Y)
      sbar_s2 <- compute_metrics(
        bea_s$cp, bea_s$beta, bea_s$sigma2,
        dat$Y, true_beta, true_sigma2, n, p
      )
    },
    error = function(e) {
      message(sprintf("  [seed %d] H-SBAR error: %s", seed, conditionMessage(e)))
    }
  )

  # -- Chan SBAR ---------------------------------------------------------
  chan_s1 <- chan_s2_rss <- chan_s2_sigma <- chan_s2_prof <- NULL
  chan_best_lambda <- NA_real_

  tryCatch(
    {
      cv_c <- cv_chan_sbar(dat$Y, p,
        lambda = lambda_chan,
        max_iter = 5000
      )
      chan_best_lambda <- cv_c$best$lambda
      fit_c <- chan_sbar_admm(dat$Y, p,
        lambda = chan_best_lambda,
        max_iter = 5000
      )
      chan_s1 <- compute_metrics(
        fit_c$cp, fit_c$beta, fit_c$sigma2,
        dat$Y, true_beta, true_sigma2, n, p
      )
      bea_rss <- chan_sbar_bea(fit_c, y = dat$Y, p = p, ic_type = "rss")
      bea_sigma <- chan_sbar_bea(fit_c, y = dat$Y, p = p, ic_type = "sigma_scaled")
      bea_prof <- chan_sbar_bea(fit_c, y = dat$Y, p = p, ic_type = "profile_lik")
      chan_s2_rss <- compute_metrics(
        bea_rss$cp, bea_rss$beta, bea_rss$sigma2,
        dat$Y, true_beta, true_sigma2, n, p
      )
      chan_s2_sigma <- compute_metrics(
        bea_sigma$cp, bea_sigma$beta, bea_sigma$sigma2,
        dat$Y, true_beta, true_sigma2, n, p
      )
      chan_s2_prof <- compute_metrics(
        bea_prof$cp, bea_prof$beta, bea_prof$sigma2,
        dat$Y, true_beta, true_sigma2, n, p
      )
    },
    error = function(e) {
      message(sprintf("  [seed %d] Chan error: %s", seed, conditionMessage(e)))
    }
  )

  list(
    sbar_s1 = sbar_s1,
    sbar_s2 = sbar_s2,
    chan_s1 = chan_s1,
    chan_s2_rss = chan_s2_rss,
    chan_s2_sigma = chan_s2_sigma,
    chan_s2_prof = chan_s2_prof,
    sbar_best_lambda = sbar_best_lambda,
    chan_best_lambda = chan_best_lambda
  )
}

# ============================================================
# Main simulation loop
# ============================================================

run_label <- if (scenario_arg %in% c(1:3, 6L, 8L)) {
  sprintf("Scenario %d  sigma=%.4g", scenario_arg, sigma_arg)
} else {
  sprintf("Scenario %d  sigma_scale=%.4g", scenario_arg, sigma_scale_arg)
}

cat(sprintf(
  "%s  n=%d  p=%d  m0=%d  true breaks: %s\n",
  run_label, N, P, M0, paste(TRUE_BREAKS, collapse = ", ")
))
cat(sprintf(
  "H-SBAR: %d lambda values on [%.2g, %.2g],  c_scale=%g\n",
  length(LAMBDA_SBAR), min(LAMBDA_SBAR), max(LAMBDA_SBAR), C_SCALE
))
cat(sprintf(
  "Chan SBAR: %d lambda values on [%.2g, %.2g]\n",
  length(LAMBDA_CHAN), min(LAMBDA_CHAN), max(LAMBDA_CHAN)
))
rds_label <- if (scenario_arg %in% c(1:3, 6L, 8L)) {
  sprintf("scenario%d-sigma%.4g", scenario_arg, sigma_arg)
} else {
  sprintf("scenario%d-sigscale%.4g", scenario_arg, sigma_scale_arg)
}
dir.create(outdir_arg, showWarnings = FALSE, recursive = TRUE)
rds_path <- file.path(outdir_arg, sprintf("run-sim-%s-results.rds", rds_label))

cat(sprintf("Running %d replications (seeds 1..%d) ...\n\n", N_REP, N_REP))
cat(sprintf("Results will be saved incrementally to %s\n\n", rds_path))

t_start <- proc.time()
results <- vector("list", N_REP)
for (i in seq_len(N_REP)) {
  cat(sprintf("[%3d/%d]", i, N_REP))
  results[[i]] <- one_rep(i, LAMBDA_SBAR, LAMBDA_CHAN, C_SCALE)
  local({
    r <- results[[i]]
    fmt_cp <- function(variant) {
      v <- r[[variant]]
      if (is.null(v) || length(v$cp) == 0L) "(none)" else paste(v$cp, collapse = ", ")
    }
    cat(sprintf(
      "  true=[%s]  hsbar=[%s]  chan_rss=[%s]  chan_sig=[%s]  chan_pl=[%s]",
      paste(TRUE_BREAKS, collapse = ", "),
      fmt_cp("sbar_s2"),
      fmt_cp("chan_s2_rss"),
      fmt_cp("chan_s2_sigma"),
      fmt_cp("chan_s2_prof")
    ))
  })
  cat("\n")
  saveRDS(results[seq_len(i)], rds_path)
}
elapsed <- (proc.time() - t_start)[["elapsed"]]
cat(sprintf("\nTotal elapsed: %.1f s  (%.1f s/rep)\n\n", elapsed, elapsed / N_REP))
cat(sprintf("Results saved to %s\n\n", rds_path))

# ============================================================
# Summarisation helpers
# ============================================================

#' Extract a scalar field from a list of rep results for one variant.
extract_scalar <- function(results, variant, field) {
  sapply(results, function(r) {
    v <- r[[variant]]
    if (is.null(v)) NA_real_ else as.numeric(v[[field]])
  })
}

#' Extract relative break locations; return matrix with ncol = m0 (NA if wrong #).
extract_rel_cp <- function(results, variant, m0) {
  t(sapply(results, function(r) {
    v <- r[[variant]]
    if (is.null(v) || length(v$rel_cp) != m0) rep(NA_real_, m0) else v$rel_cp
  }))
}

summarise_variant <- function(results, variant, m0) {
  ncp <- extract_scalar(results, variant, "ncp")
  correct <- extract_scalar(results, variant, "correct_ncp")
  hd <- extract_scalar(results, variant, "hd")
  mse <- extract_scalar(results, variant, "mse")
  beta_err <- extract_scalar(results, variant, "beta_err")
  sigma2_err <- extract_scalar(results, variant, "sigma2_err")
  rel_cp <- extract_rel_cp(results, variant, m0)

  n_ok <- sum(!is.na(ncp))
  n_correct <- sum(correct, na.rm = TRUE)

  # Conditional break-location stats (only reps with correct m)
  rel_ok <- rel_cp[!is.na(rel_cp[, 1L]), , drop = FALSE]
  n_rel <- nrow(rel_ok)
  mean_rel <- if (n_rel > 0L) colMeans(rel_ok) else rep(NA_real_, m0)
  se_rel <- if (n_rel > 1L) {
    apply(rel_ok, 2L, sd) / sqrt(n_rel)
  } else {
    rep(NA_real_, m0)
  }

  list(
    n_ok            = n_ok,
    avg_ncp         = mean(ncp, na.rm = TRUE),
    pct_correct     = 100 * n_correct / n_ok,
    mean_rel        = mean_rel,
    se_rel          = se_rel,
    mean_hd         = mean(hd, na.rm = TRUE),
    mean_mse        = mean(mse, na.rm = TRUE),
    mean_beta_err   = mean(beta_err, na.rm = TRUE),
    mean_sigma2_err = mean(sigma2_err, na.rm = TRUE)
  )
}

# ============================================================
# Print results
# ============================================================

VARIANTS <- c(
  "sbar_s1", "sbar_s2", "chan_s1",
  "chan_s2_rss", "chan_s2_sigma", "chan_s2_prof"
)
LABELS <- c(
  "H-SBAR S1", "H-SBAR S2", "Chan S1",
  "Chan S2 RSS", "Chan S2 sigma", "Chan S2 PL"
)

sums <- lapply(VARIANTS, function(v) summarise_variant(results, v, M0))
names(sums) <- VARIANTS

# --- Table 1: break-point estimation (Chan 2014 Table 1 style) ----------
cat("=================================================================\n")
cat(sprintf("Table 1: Break-point estimation  [%s]\n", run_label))
cat(sprintf(
  "         (True breaks: %s  |  True rel: %s)\n",
  paste(TRUE_BREAKS, collapse = ", "),
  paste(sprintf("%.4f", TRUE_BREAKS / N), collapse = ", ")
))
cat("=================================================================\n")

hdr <- sprintf(
  "%-14s %8s %8s %10s %8s %10s %8s",
  "Method", "Avg # 1st", "% m=2", "Mean b1", "SE b1",
  "Mean b2", "SE b2"
)
cat(hdr, "\n")
cat(strrep("-", nchar(hdr)), "\n")

for (k in seq_along(VARIANTS)) {
  s <- sums[[k]]
  cat(sprintf(
    "%-14s %8.2f %8.1f %10.4f %8.4f %10.4f %8.4f\n",
    LABELS[k],
    s$avg_ncp,
    s$pct_correct,
    if (M0 >= 1L) s$mean_rel[1L] else NA,
    if (M0 >= 1L) s$se_rel[1L] else NA,
    if (M0 >= 2L) s$mean_rel[2L] else NA,
    if (M0 >= 2L) s$se_rel[2L] else NA
  ))
}

# --- Table 2: estimation accuracy ---------------------------------------
cat("\n=================================================================\n")
cat(sprintf("Table 2: Estimation accuracy  [%s]\n", run_label))
cat("=================================================================\n")

hdr2 <- sprintf(
  "%-14s %10s %10s %10s %12s",
  "Method", "Mean HD", "Mean MSE", "Mean b-err", "Mean s2-err"
)
cat(hdr2, "\n")
cat(strrep("-", nchar(hdr2)), "\n")

for (k in seq_along(VARIANTS)) {
  s <- sums[[k]]
  cat(sprintf(
    "%-14s %10.2f %10.5f %10.5f %12.5f\n",
    LABELS[k],
    s$mean_hd,
    s$mean_mse,
    s$mean_beta_err,
    s$mean_sigma2_err
  ))
}

# --- Lambda selection summary ------------------------------------------
sbar_lambdas <- sapply(results, `[[`, "sbar_best_lambda")
chan_lambdas <- sapply(results, `[[`, "chan_best_lambda")

cat(sprintf(
  "\nH-SBAR best lambda: median=%.4g  range=[%.4g, %.4g]\n",
  median(sbar_lambdas, na.rm = TRUE),
  min(sbar_lambdas, na.rm = TRUE),
  max(sbar_lambdas, na.rm = TRUE)
))
cat(sprintf(
  "Chan best lambda:     median=%.4g  range=[%.4g, %.4g]\n",
  median(chan_lambdas, na.rm = TRUE),
  min(chan_lambdas, na.rm = TRUE),
  max(chan_lambdas, na.rm = TRUE)
))

cat("\nDone.\n")
