source("hsbar.R")
source("cv-hsbar.R")
source("hsbar-bea.R")
source("fit-c.R")

# Step 0: Load and validate
df <- read.csv("data-analysis/sp.csv")
df$Date <- as.Date(df$Date, "%m/%d/%Y")
df <- df[order(df$Date), ]
df <- df[!duplicated(df$Date), ]
stopifnot(all(df$Value > 0))
prices <- df$Value

# Step 1: Log returns
r <- diff(log(prices))
dates_r <- df$Date[-1]

# Step 2: Remove conditional mean; use AR(1) residuals if autocorrelation present
r_tilde <- r - mean(r)
lb <- Box.test(r_tilde, lag = 10, type = "Ljung-Box")
cat(sprintf("Ljung-Box test on demeaned returns: p = %.4f\n", lb$p.value))
if (lb$p.value < 0.05) {
  cat("Significant autocorrelation detected; fitting AR(1) and using residuals.\n")
  ar1 <- ar(r_tilde, order.max = 1, method = "ols")
  keep <- !is.na(ar1$resid)
  r_tilde <- ar1$resid[keep]
  dates_r <- dates_r[keep]
} else {
  cat("No significant autocorrelation; using demeaned returns.\n")
}

# Step 3: Audit outliers (flag only, do not remove)
mad_scale <- median(abs(r_tilde - median(r_tilde))) / 0.6745
outlier_idx <- which(abs(r_tilde) > 10 * mad_scale)
if (length(outlier_idx) > 0) {
  cat("Flagged outlier candidates (>10 robust SDs):\n")
  print(data.frame(date = dates_r[outlier_idx], r = r_tilde[outlier_idx]))
} else {
  cat("No outlier candidates flagged.\n")
}

# Step 4: Square to form AR object
x_sq <- r_tilde^2
plag <- 12

# # Step 5: CV to select lambda, then H-SBAR, then BEA pruning
# lambda_grid <- seq(0.01, 0.5, by = 0.01)

# cat("Running CV to select lambda...\n")
# cv_fit <- cv_hsbar(x_sq,
#   p = plag, lambda = lambda_grid, verbose = FALSE,
#   max_iter = 5000
# )
# best_lambda <- cv_fit$best$lambda
# cat(sprintf("CV selected lambda = %.4f\n", best_lambda))

c_est <- fit_c(x_sq, plag)
cat(sprintf("Estimated c_scale = %.6f\n", c_est))

cat("Fitting H-SBAR with best lambda...\n")
fit <- hsbar(x_sq, p = plag, lambda = 0.004, c_scale = 1, max_iter = 5000, verbose = FALSE)
cat(sprintf("Stage 1 changepoints: %d\n", length(fit$cp)))
cat("Stage 1 dates: ")
print(dates_r[fit$cp])

cp_dates_s1 <- dates_r[fit$cp]

cat("Running BEA pruning (stage 2)...\n")
bea_fit <- hsbar_bea(fit, y = x_sq, omega = (plag + 1) * log(length(x_sq)) / 5)
cp_dates <- dates_r[bea_fit$cp]
cat(sprintf("Stage 2 (BEA) changepoints: %d\n", length(bea_fit$cp)))
cat("Stage 2 dates: ")
print(cp_dates)

# Annual tick marks: Jan 1 of each year in the series
years <- seq(
  as.integer(format(min(dates_r), "%Y")),
  as.integer(format(max(dates_r), "%Y"))
)
year_ticks <- as.Date(paste0(years, "-01-01"))

add_year_axis <- function() {
  axis.Date(1, at = year_ticks, format = "%Y")
}

# Plots
pdf("data-analysis/sp_hsbar.pdf", width = 10, height = 13)
par(mfrow = c(3, 1), mar = c(3, 4, 2, 1))

plot_cv_hsbar(cv_fit)

plot(dates_r, r_tilde,
  type = "l", col = "steelblue", lwd = 0.6,
  xaxt = "n", xlab = "", ylab = "Log return", main = "S&P 500 daily log returns"
)
add_year_axis()
# abline(v = cp_dates_s1, lty = 2, col = "gray60")
abline(v = cp_dates, lty = 2, col = "red")

plot(dates_r, x_sq,
  type = "l", col = "steelblue", lwd = 0.6,
  xaxt = "n", xlab = "", ylab = expression(r^2),
  main = "Squared log returns (H-SBAR input)"
)
add_year_axis()
# abline(v = cp_dates_s1, lty = 2, col = "gray60")
abline(v = cp_dates, lty = 2, col = "red")

dev.off()
cat("Plots saved to data-analysis/sp_hsbar.pdf\n")
