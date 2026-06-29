FS <- 360L
WIN_START <- 7L
WIN_END <- 13L

RHYTHM_COLORS <- c(
  "N"    = "#5aaa72",
  "B"    = "#e08a2e",
  "VT"   = "#c0392b",
  "AFIB" = "#7b5ea7",
  "AFL"  = "#2e86c1",
  "T"    = "#d4ac0d"
)

RHYTHM_LABELS <- c(
  "N"    = "Normal Sinus Rhythm",
  "B"    = "Ventricular Bigeminy",
  "VT"   = "Ventricular Tachycardia",
  "AFIB" = "Atrial Fibrillation",
  "AFL"  = "Atrial Flutter",
  "T"    = "Supraventricular Tachycardia"
)

parse_rhythm <- function(record) {
  path <- sprintf("data-analysis/ecg/%sannotations.txt", record)
  lines <- readLines(path)
  lines <- lines[nzchar(trimws(lines))][-1]

  result <- lapply(lines, function(ln) {
    parts <- strsplit(trimws(ln), "\\s+")[[1]]
    if (length(parts) < 6) {
      return(NULL)
    }
    list(
      sample = as.integer(parts[2]),
      type   = parts[3],
      aux    = if (length(parts) >= 7) parts[7] else NA_character_
    )
  })
  result <- Filter(Negate(is.null), result)

  df <- data.frame(
    sample = sapply(result, `[[`, "sample"),
    type = sapply(result, `[[`, "type"),
    aux = sapply(result, `[[`, "aux"),
    stringsAsFactors = FALSE
  )

  mask <- df$type == "+" & !is.na(df$aux) & grepl("^\\(", df$aux)
  rhy <- df[mask, ]
  rhy$rhythm <- sub("^\\(", "", rhy$aux)
  rhy$t_sec <- (rhy$sample - 1L) / FS
  rhy
}

load_ecg <- function(record) {
  path <- sprintf("data-analysis/ecg/%s.csv", record)
  df <- read.csv(path, check.names = FALSE)
  mlii_full <- df[["'MLII'"]]
  n_full <- length(mlii_full)
  idx <- seq(WIN_START * FS + 1L, WIN_END * FS)
  list(
    y     = mlii_full[idx],
    t_sec = seq(WIN_START, WIN_END - 1 / FS, by = 1 / FS),
    t_max = (n_full - 1L) / FS
  )
}

record <- "200"

pdf("data-analysis/ecg_groundtruth.pdf", width = 12, height = 3.5)
par(mar = c(4, 5, 2, 1))

ecg <- load_ecg(record)
rhy <- parse_rhythm(record)

intervals <- data.frame(
  t_start = rhy$t_sec,
  t_end = c(rhy$t_sec[-1], ecg$t_max),
  rhythm = rhy$rhythm,
  stringsAsFactors = FALSE
)
intervals$color <- RHYTHM_COLORS[intervals$rhythm]
intervals$color[is.na(intervals$color)] <- "#aaaaaa"

plot(ecg$t_sec, ecg$y,
  type = "n",
  xlim = c(WIN_START, WIN_END),
  xlab = "Time (s)", ylab = "Amplitude",
  main = "ECG amplitude",
  cex.main = 0.95, cex.lab = 0.85
)

usr <- par("usr")

for (i in seq_len(nrow(intervals))) {
  rect(intervals$t_start[i], usr[3],
    intervals$t_end[i], usr[4],
    col = adjustcolor(intervals$color[i], alpha.f = 0.22),
    border = NA
  )
}

lines(ecg$t_sec, ecg$y, col = "gray15", lwd = 0.5)
box()

in_window <- intervals$t_start < WIN_END & intervals$t_end > WIN_START
present <- unique(intervals$rhythm[in_window])
present <- present[present %in% names(RHYTHM_LABELS)]
legend(
  "bottomleft",
  legend  = sprintf("%-4s  %s", present, RHYTHM_LABELS[present]),
  fill    = RHYTHM_COLORS[present],
  border  = NA,
  cex     = 0.68,
  bg      = "white",
  box.lwd = 0.5
)

dev.off()
cat("Saved to data-analysis/ecg_groundtruth.pdf\n")
