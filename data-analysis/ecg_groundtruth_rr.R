FS <- 360L

RHYTHM_LABELS <- c(
  "N"    = "Normal Sinus Rhythm",
  "B"    = "Ventricular Bigeminy",
  "VT"   = "Ventricular Tachycardia",
  "AFIB" = "Atrial Fibrillation",
  "AFL"  = "Atrial Flutter",
  "T"    = "Supraventricular Tachycardia",
  "BII"  = "2nd-degree AV Block"
)

RHYTHM_COLORS <- c(
  "N"    = "#5aaa72",
  "B"    = "#e08a2e",
  "VT"   = "#c0392b",
  "AFIB" = "#7b5ea7",
  "AFL"  = "#2e86c1",
  "T"    = "#d4ac0d",
  "BII"  = "#1a7fa8"
)

NON_BEAT <- c(
  "+", "~", "|", "[", "]", "!", "(", ")", "p",
  "t", "u", "`", "'", "^", "x", "s", "T", "*",
  "D", "=", "@", "Q", "?"
)

load_rr <- function(record) {
  path <- sprintf("data-analysis/ecg/%sannotations.txt", record)
  lines <- readLines(path)
  lines <- lines[nzchar(trimws(lines))][-1]

  result <- lapply(lines, function(ln) {
    parts <- strsplit(trimws(ln), "\\s+")[[1]]
    if (length(parts) < 3) {
      return(NULL)
    }
    list(sample = as.integer(parts[2]), type = parts[3])
  })
  result <- Filter(Negate(is.null), result)

  df <- data.frame(
    sample = sapply(result, `[[`, "sample"),
    type = sapply(result, `[[`, "type"),
    stringsAsFactors = FALSE
  )

  beats <- df[!df$type %in% NON_BEAT, ]
  t_beat <- (beats$sample - 1L) / FS
  rr <- diff(t_beat)

  list(
    rr     = rr,
    t_beat = t_beat[-length(t_beat)],
    t_max  = max(t_beat)
  )
}

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

record <- "200"

pdf("data-analysis/ecg_groundtruth_rr.pdf", width = 12, height = 3.5)
par(mar = c(4, 5, 2, 1))

ecg <- load_rr(record)
rhy <- parse_rhythm(record)
rr_cap <- pmin(ecg$rr, 2.5)

intervals <- data.frame(
  t_start = rhy$t_sec,
  t_end = c(rhy$t_sec[-1], ecg$t_max),
  rhythm = rhy$rhythm,
  stringsAsFactors = FALSE
)
intervals$color <- RHYTHM_COLORS[intervals$rhythm]
intervals$color[is.na(intervals$color)] <- "#aaaaaa"

plot(ecg$t_beat / 60, rr_cap,
  type = "n",
  xlab = "Time (min)", ylab = "RR interval (s)",
  ylim = c(0, max(rr_cap) * 1.05),
  main = "RR interval",
  cex.main = 0.95, cex.lab = 0.85
)

usr <- par("usr")

for (i in seq_len(nrow(intervals))) {
  rect(intervals$t_start[i] / 60, usr[3],
    intervals$t_end[i] / 60, usr[4],
    col = adjustcolor(intervals$color[i], alpha.f = 0.22),
    border = NA
  )
}

lines(ecg$t_beat / 60, rr_cap, col = "gray15", lwd = 0.4)
box()

present <- unique(intervals$rhythm)
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
cat("Saved to data-analysis/ecg_groundtruth_rr.pdf\n")
