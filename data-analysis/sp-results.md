# S&P 500 H-SBAR Results

**Input:** Daily log returns of S&P 500, squared and used as the AR(p) process input (p = 12).  
**Preprocessing:** Demeaned returns showed significant autocorrelation (Ljung-Box p ≈ 0); AR(1) residuals used instead.  
**Method:** H-SBAR with λ = 0.004, followed by BEA pruning (ω = (p+1) log(n) / 5).

Stage 1 produced 12 changepoints; BEA pruning reduced to **8 final changepoints**.

---

## Stage 2 (Post-BEA) Changepoints

### 1 — 2019-02-14

**Fed pivot / end of Q4 2018 bear market.**  
The S&P 500 fell ~20% in Q4 2018 amid fears of over-tightening by the Fed and slowing global growth. Markets bottomed on December 24, 2018. In January 2019, Fed Chair Powell signaled a pause in rate hikes; by mid-February, this pivot was well-established and the recovery rally was underway. This break marks the transition from the high-volatility late-2018 regime to the calmer 2019 bull market.

### 2 — 2020-04-21

**Post-COVID crash / acute volatility peak.**  
The COVID-19 crash ran from February 19 to March 23, 2020 (−34%). By April 21, the immediate acute phase had passed but markets remained highly uncertain. Notably, WTI crude oil futures went negative (−$37/barrel) on April 20, 2020. The CARES Act ($2.2T) had been signed March 27 and the Fed launched unlimited QE. This break likely marks the end of the extreme crash-volatility regime and the beginning of the uneven early-recovery phase.

### 3 — 2020-07-21

**Summer 2020 tech-led recovery regime.**  
By mid-July 2020, the S&P 500 had recovered nearly all COVID losses, driven by mega-cap tech. On July 21, the EU agreed on its €750B COVID recovery fund, a major positive signal for global markets. The US was experiencing a summer COVID wave but markets had decoupled from case counts. This break marks the start of a lower-volatility, tech-dominated recovery regime.

### 4 — 2020-11-16

**Vaccine announcement / Biden election — regime shift.**  
On November 7, 2020, Biden's presidential win was called; on November 9, Pfizer announced 90%+ vaccine efficacy. By November 16 both developments were fully absorbed, triggering a sharp rotation from growth/tech into cyclicals and value stocks. This break marks the transition to the vaccine-optimism regime and the beginning of the reopening trade.

### 5 — 2022-03-07

**Russia–Ukraine war / start of Fed rate hike cycle.**  
Russia invaded Ukraine on February 24, 2022, triggering a commodity price shock (WTI oil approached $130/barrel on March 8). The Fed began its most aggressive rate hike cycle since the 1980s, with the first 25bp hike on March 16. This break marks the onset of the 2022 stagflation/tightening bear market, a distinct high-volatility regime driven by geopolitical shock and monetary policy pivot.

### 6 — 2023-01-27

**Soft-landing optimism / end of 2022 bear market.**  
The S&P 500 gained ~6% in January 2023, one of its strongest January performances. On January 27, 2023, PCE inflation data came in softer than expected, reinforcing expectations that the Fed was nearing the end of its tightening cycle. This break marks the transition from the 2022 bear market regime to the 2023 soft-landing rally, with a meaningfully lower volatility regime.

### 7 — 2024-12-18

**Hawkish Fed cut — higher-for-longer reassertion.**  
The Fed cut rates by 25bp on December 18, 2024, but simultaneously reduced its 2025 rate-cut projections from four to two, signaling rates would stay elevated longer than expected. The S&P 500 fell ~3% on that day alone. This break marks the onset of a more cautious, higher-volatility regime as the market repriced the path of monetary easing.

### 8 — 2025-06-17

**Post-tariff shock stabilization.**  
Trump's "Liberation Day" tariffs (April 2, 2025) caused a ~15% peak-to-trough selloff, with extreme single-day swings (notably +8.8% on April 9 when a 90-day tariff pause was announced — flagged as an outlier by the robust MAD filter). Markets recovered through May–June 2025 as US–China trade tensions partially eased via a temporary agreement. This break likely marks the normalization of volatility after the tariff shock, as uncertainty about trade policy settled into a new baseline regime.

---

## Outliers Flagged (>10 robust SDs, not removed)

| Date | Log return |
|------|-----------|
| 2020-03-12 | −0.108 |
| 2020-03-16 | −0.116 |
| 2020-03-24 | +0.085 |
| 2025-04-09 | +0.088 |

The first three are COVID crash extremes; the last is the April 9, 2025 tariff-pause rally.
