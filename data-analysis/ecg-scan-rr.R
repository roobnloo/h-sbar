source("hsbar.R")
source("hsbar-bea.R")
source("fit-c.R")

args   <- commandArgs(trailingOnly = TRUE)
RECORD <- if (length(args) > 0L) args[[1L]] else "231"
FS       <- 360L
PLAG     <- 6
TOL_SEC  <- 60

LAMBDAS  <- c(0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05)
C_SCALES <- c(0.05, 0.1, 0.2, 0.5, 1.0)

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

# ---- load data once ----

ecg    <- load_rr(RECORD)
y      <- ecg$rr
t_beat <- ecg$t_beat
n      <- length(y)

cat(sprintf("Record %s: %d RR intervals\n", RECORD, n))

rhy     <- parse_rhythm(RECORD)
ann_sec <- rhy$t_sec

cat(sprintf("Rhythm boundaries: %d\n", length(ann_sec)))
print(data.frame(t_min = round(ann_sec / 60, 2), rhythm = rhy$rhythm))
cat("\n")

# ---- match metric ----

match_stats <- function(cp_sec, ann_sec, tol = TOL_SEC) {
  if (length(cp_sec) == 0) {
    return(list(n_breaks = 0L, precision = NA_real_, recall = 0, f1 = 0))
  }
  hits_cp  <- sapply(cp_sec,  function(t) any(abs(ann_sec - t) <= tol))
  hits_ann <- sapply(ann_sec, function(t) any(abs(cp_sec  - t) <= tol))
  prec <- mean(hits_cp)
  rec  <- mean(hits_ann)
  f1   <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
  list(n_breaks = length(cp_sec), precision = prec, recall = rec, f1 = f1)
}

# ---- grid search ----

rows  <- list()
total <- length(LAMBDAS) * length(C_SCALES)
done  <- 0L

for (cs in C_SCALES) {
  warm_theta <- NULL
  warm_psi   <- NULL

  for (lam in sort(LAMBDAS, decreasing = TRUE)) {
    done <- done + 1L
    cat(sprintf("[%d/%d] lambda=%.3g  c_scale=%.3g  ...", done, total, lam, cs))

    fit <- tryCatch(
      hsbar(y, p = PLAG, lambda = lam, c_scale = cs,
            init_theta = warm_theta, init_psi = warm_psi,
            max_iter = 5000, verbose = FALSE),
      error = function(e) NULL
    )

    if (is.null(fit)) {
      cat("FAILED\n")
      next
    }

    warm_theta <- fit$theta
    warm_psi   <- fit$psi

    bea    <- hsbar_bea(fit, y = y, omega = (PLAG + 1) * log(n) / 5)
    cp_sec <- if (length(bea$cp) > 0) t_beat[bea$cp] else numeric(0)

    ms <- match_stats(cp_sec, ann_sec)
    cat(sprintf("S2_breaks=%d  prec=%.2f  rec=%.2f  f1=%.2f\n",
                ms$n_breaks,
                if (is.na(ms$precision)) NaN else ms$precision,
                ms$recall, ms$f1))

    rows[[length(rows) + 1L]] <- data.frame(
      lambda    = lam,
      c_scale   = cs,
      n_breaks  = ms$n_breaks,
      precision = round(ms$precision, 3),
      recall    = round(ms$recall, 3),
      f1        = round(ms$f1, 3),
      stringsAsFactors = FALSE
    )
  }
}

# ---- report ----

results <- do.call(rbind, rows)
results <- results[order(-results$f1, -results$recall), ]

cat("\n=== Results sorted by F1 (tolerance =", TOL_SEC, "s) ===\n")
print(results, row.names = FALSE)

best <- results[1L, ]
cat(sprintf(
  "\nBest: lambda=%.3g  c_scale=%.3g  ->  %d breaks, precision=%.2f, recall=%.2f, f1=%.2f\n",
  best$lambda, best$c_scale, best$n_breaks, best$precision, best$recall, best$f1
))

out_dir <- "data-analysis/ecg/results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_csv <- file.path(out_dir, sprintf("scan_rr_%s.csv", RECORD))
write.csv(results, out_csv, row.names = FALSE)
cat(sprintf("Full results saved to %s\n", out_csv))
