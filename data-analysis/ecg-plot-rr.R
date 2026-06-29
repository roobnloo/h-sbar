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

results_dir <- "data-analysis/ecg/results"
rds_files <- sort(list.files(results_dir, pattern = "^rr_.*\\.rds$", full.names = TRUE))
if (length(rds_files) == 0L) stop("No rr_*.rds results found in ", results_dir)

results <- lapply(rds_files, readRDS)

for (r in results) {
  rid <- r$record
  rhy <- r$rhy
  rr_cap <- pmin(r$rr, 2.5)
  t_min <- r$t_beat / 60

  pdf(sprintf("data-analysis/ecg_rr_hsbar_%s.pdf", rid), width = 11, height = 4)
  par(mar = c(4.5, 5, 2.5, 1))

  plot(t_min, rr_cap,
    type = "n",
    xlab = "Time (min)", ylab = "RR interval (s)",
    ylim = c(0, max(rr_cap) * 1.05),
    main = sprintf("MIT-BIH Record %s", rid)
  )

  usr <- par("usr")

  if (nrow(rhy) > 0) {
    intervals <- data.frame(
      t_start = rhy$t_sec,
      t_end = c(rhy$t_sec[-1], r$t_max),
      rhythm = rhy$rhythm,
      stringsAsFactors = FALSE
    )
    intervals$color <- RHYTHM_COLORS[intervals$rhythm]
    intervals$color[is.na(intervals$color)] <- "#aaaaaa"

    for (i in seq_len(nrow(intervals))) {
      rect(intervals$t_start[i] / 60, usr[3],
        intervals$t_end[i] / 60, usr[4],
        col = adjustcolor(intervals$color[i], alpha.f = 0.22), border = NA
      )
    }
  }

  lines(t_min, rr_cap, col = "gray15", lwd = 0.5)

  if (nrow(rhy) > 1) {
    tick_top <- usr[4]
    tick_bot <- usr[4] - (usr[4] - usr[3]) * 0.05
    segments(rhy$t_sec[-1] / 60, tick_top,
      rhy$t_sec[-1] / 60, tick_bot,
      col = "blue", lwd = 1.2, xpd = FALSE
    )
  }

  if (length(r$cp2_sec) > 0) {
    abline(v = r$cp2_sec / 60, lty = 2, col = "red", lwd = 1.4)
  }

  present <- unique(rhy$rhythm)
  present <- present[present %in% names(RHYTHM_LABELS)]

  legend("bottomleft",
    legend = c(
      sprintf("%-4s  %s", present, RHYTHM_LABELS[present]),
      "H-SBAR break (BEA)"
    ),
    fill = c(RHYTHM_COLORS[present], NA),
    border = c(rep(NA, length(present)), NA),
    lty = c(rep(NA, length(present)), 2),
    lwd = c(rep(NA, length(present)), 1.4),
    col = c(rep(NA, length(present)), "red"),
    cex = 0.7, bg = "white", box.lwd = 0.5
  )

  dev.off()
  cat(sprintf("Saved data-analysis/ecg_rr_hsbar_%s.pdf\n", rid))
}
