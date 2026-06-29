source("hsbar.R")
source("hsbar-bea.R")
source("fit-c.R")

FS       <- 360L
PLAG     <- 6
NON_BEAT <- c(
  "+", "~", "|", "[", "]", "!", "(", ")", "p",
  "t", "u", "`", "'", "^", "x", "s", "T", "*",
  "D", "=", "@", "Q", "?"
)

load_rr <- function(record) {
  path  <- sprintf("data-analysis/ecg/%sannotations.txt", record)
  lines <- readLines(path)
  lines <- lines[nzchar(trimws(lines))][-1]

  result <- lapply(lines, function(ln) {
    parts <- strsplit(trimws(ln), "\\s+")[[1]]
    if (length(parts) < 3) return(NULL)
    list(sample = as.integer(parts[2]), type = parts[3])
  })
  result <- Filter(Negate(is.null), result)

  df <- data.frame(
    sample = sapply(result, `[[`, "sample"),
    type   = sapply(result, `[[`, "type"),
    stringsAsFactors = FALSE
  )

  beats  <- df[!df$type %in% NON_BEAT, ]
  t_beat <- (beats$sample - 1L) / FS
  rr     <- diff(t_beat)

  list(
    rr     = rr,
    t_beat = t_beat[-length(t_beat)],
    t_max  = max(t_beat)
  )
}

parse_rhythm <- function(record) {
  path  <- sprintf("data-analysis/ecg/%sannotations.txt", record)
  lines <- readLines(path)
  lines <- lines[nzchar(trimws(lines))][-1]

  result <- lapply(lines, function(ln) {
    parts <- strsplit(trimws(ln), "\\s+")[[1]]
    if (length(parts) < 6) return(NULL)
    list(
      sample = as.integer(parts[2]),
      type   = parts[3],
      aux    = if (length(parts) >= 7) parts[7] else NA_character_
    )
  })
  result <- Filter(Negate(is.null), result)

  df <- data.frame(
    sample = sapply(result, `[[`, "sample"),
    type   = sapply(result, `[[`, "type"),
    aux    = sapply(result, `[[`, "aux"),
    stringsAsFactors = FALSE
  )

  mask <- df$type == "+" & !is.na(df$aux) & grepl("^\\(", df$aux)
  rhy  <- df[mask, ]
  rhy$rhythm <- sub("^\\(", "", rhy$aux)
  rhy$t_sec  <- (rhy$sample - 1L) / FS
  rhy
}

analyze_rr <- function(record, lambda, c_scale = NULL, warm_init = NULL) {
  cat(sprintf("\n=== Record %s (RR series) ===\n", record))

  ecg <- load_rr(record)
  y   <- ecg$rr
  n   <- length(y)

  cat(sprintf("Beats: %d  RR intervals: %d\n", n + 1L, n))

  rhy <- parse_rhythm(record)
  rhy$t_sec <- (rhy$sample - 1L) / FS
  cat(sprintf("Rhythm boundaries: %d\n", nrow(rhy)))
  print(rhy[, c("t_sec", "rhythm")])

  c_est <- if (is.null(c_scale)) fit_c(y, PLAG) else c_scale
  cat(sprintf("c_scale: %.6f (%s)\n", c_est,
              if (is.null(c_scale)) "estimated" else "provided"))

  cat("Fitting H-SBAR Stage 1...\n")
  fit <- hsbar(y, p = PLAG, lambda = lambda, c_scale = c_est,
               init_theta = warm_init$theta, init_psi = warm_init$psi,
               max_iter = 5000, verbose = FALSE)
  cat(sprintf("Stage 1 changepoints: %d\n", length(fit$cp)))
  cp1_sec <- if (length(fit$cp) > 0) ecg$t_beat[fit$cp] else numeric(0)

  cat("Running BEA pruning (Stage 2)...\n")
  bea     <- hsbar_bea(fit, y = y, omega = (PLAG + 1) * log(n) / 5)
  cp2_sec <- if (length(bea$cp) > 0) ecg$t_beat[bea$cp] else numeric(0)
  cat(sprintf("Stage 2 changepoints: %d\n", length(bea$cp)))

  if (length(cp2_sec) > 0 && nrow(rhy) > 0) {
    cat("Stage 2 breaks vs. nearest rhythm boundary:\n")
    nearest <- sapply(cp2_sec, function(t) {
      d <- abs(rhy$t_sec - t)
      i <- which.min(d)
      sprintf("%s (%.1fs away)", rhy$rhythm[i], d[i])
    })
    print(data.frame(break_sec = round(cp2_sec, 1), nearest_rhythm = nearest))
  }

  list(
    record  = record,
    lambda  = lambda,
    c_scale = c_est,
    rr      = y,
    t_beat  = ecg$t_beat,
    t_max   = ecg$t_max,
    rhy     = rhy,
    fit     = fit,
    bea     = bea,
    cp1_sec = cp1_sec,
    cp2_sec = cp2_sec
  )
}

args    <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("Usage: Rscript data-analysis/ecg_rr.R <record> <lambda> [c_scale] [--warm <rds>]")
record  <- args[[1L]]
lambda  <- as.numeric(args[[2L]])
c_scale <- if (length(args) >= 3L && !grepl("^--", args[[3L]])) as.numeric(args[[3L]]) else NULL

warm_idx  <- which(args == "--warm")
warm_init <- if (length(warm_idx) > 0L && length(args) >= warm_idx + 1L) {
  prev <- readRDS(args[[warm_idx + 1L]])
  cat(sprintf("Warm-starting from %s\n", args[[warm_idx + 1L]]))
  list(theta = prev$fit$theta, psi = prev$fit$psi)
} else NULL

result   <- analyze_rr(record, lambda, c_scale, warm_init)
out_dir  <- "data-analysis/ecg/results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, sprintf("rr_%s_lambda%s.rds", record, lambda))
saveRDS(result, out_path)
cat(sprintf("Results saved to %s\n", out_path))
