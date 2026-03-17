Based on the simulation frameworks utilized by **Safikhani & Shojaie (2022)** and **Chan et al. (2014)**, here is a progressive set of experimental configurations for testing your structural break model.

These settings are adapted from their multivariate (VAR) scenarios into a univariate **Autoregressive (AR)** context, specifically tailored to evaluate your objective of capturing simultaneous shifts in coefficients ($\phi$) and innovation variance ($\sigma$).

# Simulation Settings for $AR(p)$ Joint Break Detection

### Configuration Overview

All scenarios assume a piecewise stationary process:


$$Y_t = \sum_{j=1}^{m_0+1} \left( \sum_{k=1}^p \phi_{j,k} Y_{t-k} + \sigma_j \epsilon_t \right) I(t_{j-1} \le t < t_j)$$


where $\epsilon_t \sim N(0, 1)$.

---

### Scenario 1: Baseline (Center Breaks, Constant Variance)

**Goal:** Establish a benchmark for coefficient detection where the signal is strong and located away from boundaries.

* **Parameters:** $T=300, p=1, m_0=2$.
* **Break Points:** $t_1=100, t_2=200$.
* **Coefficients ($\phi$):** $\phi_1 = -0.6, \phi_2 = 0.75, \phi_3 = -0.8$.
* **Variance ($\sigma$):** $\sigma_1 = \sigma_2 = \sigma_3 = 0.1$.

### Scenario 2: Boundary Sensitivity (Constant Variance)

**Goal:** Test the method's ability to locate breaks near the beginning or end of the series where information is asymmetric.

* **Parameters:** $T=300, p=1, m_0=2$.
* **Break Points:** $t_1=50, t_2=250$.
* **Coefficients ($\phi$):** $\phi_1 = -0.6, \phi_2 = 0.75, \phi_3 = -0.8$.
* **Variance ($\sigma$):** Constant across regimes; configurable via `--sigma` (default 0.1).

### Scenario 3: Small-Jump Detection (Lower Signal-to-Noise)

**Goal:** Evaluate localization accuracy when the "jump size" between regimes is small.

* **Parameters:** $T=300, p=2, m_0=2$.
* **Break Points:** $t_1=100, t_2=200$.
* **Coefficients ($\phi$):**
* Regime 1: $(\phi_{1,1}=0.4, \phi_{1,2}=0.2)$
* Regime 2: $(\phi_{2,1}=0.5, \phi_{2,2}=0.1)$
* Regime 3: $(\phi_{3,1}=0.3, \phi_{3,2}=0.2)$


* **Variance ($\sigma$):** Constant across regimes; configurable via `--sigma` (default 0.1).

### Scenario 4: Joint Coefficient & Variance Shift (Target Case)

**Goal:** Validate your primary extension—identifying structural breaks that occur simultaneously in system dynamics and noise.

* **Parameters:** $T=500, p=1, m_0=2$.
* **Break Points:** $t_1=150, t_2=350$.
* **Dynamics:**
* **Regime 1:** $\phi = 0.5, \sigma = 0.1$ (Low activity, low noise)
* **Regime 2:** $\phi = 0.9, \sigma = 0.4$ (High persistence, high noise - e.g., "Crisis" phase)
* **Regime 3:** $\phi = 0.2, \sigma = 0.15$ (Mean reverting, moderate noise)
* **Noise scaling:** All $\sigma$ values are multiplied by `sigma_scale` (configurable via `--sigma_scale`, default 1).

### Scenario 5: Model Misspecification (Correlated Error)

**Goal:** Test robustness when noise is not white noise. This adapts Safikhani & Shojaie’s dense covariance scenario to 1D.

* **Parameters:** $T=300, p=1, m_0=2$.
* **Noise Profile:** $\epsilon_t$ follows an $AR(1)$ process: $\epsilon_t = 0.5\epsilon_{t-1} + \nu_t$, where $\nu_t \sim N(0, 1)$.
* **Coefficients/Variance:** Use values from Scenario 4 to see if correlated noise masks the structural breaks.
* **Noise scaling:** All $\sigma$ values are multiplied by `sigma_scale` (configurable via `--sigma_scale`, default 1).

### Scenario 6: Dyadic Piecewise-Stationary AR(2) (Chan et al. 2014)

**Goal:** Replicate the dyadic simulation from Chan et al. (2014) used to benchmark the two-step SBAR procedure, with near-unit-root dynamics.

* **Parameters:** $T=1024, p=2, m_0=2$.
* **Break Points:** $t_1=513, t_2=769$ (dyadic).
* **Coefficients ($\phi$):**
  * Regime 1: $(\phi_1=0.9,\; \phi_2=0.0)$ — AR(1) embedded in AR(2)
  * Regime 2: $(\phi_1=1.69,\; \phi_2=-0.81)$ — near-unit-root AR(2)
  * Regime 3: $(\phi_1=1.32,\; \phi_2=-0.81)$ — near-unit-root AR(2)
* **Variance ($\sigma$):** Constant across regimes; configurable via `--sigma` (default 1).

### Scenario 7: Dyadic Piecewise-Stationary AR(2) with Variance Shifts

**Goal:** Extend Scenario 6 with heteroskedasticity to test joint break detection in coefficients and variance under near-unit-root dynamics.

* **Parameters:** $T=1002, p=2, m_0=2$.
* **Break Points:** $t_1=334, t_2=668$ (equal thirds).
* **Coefficients ($\phi$):** Same as Scenario 6 across all three regimes.
* **Variance ($\sigma$):**
  * Regime 1: $\sigma = 0.1 \times \texttt{sigma\_scale}$
  * Regime 2: $\sigma = 0.4 \times \texttt{sigma\_scale}$
  * Regime 3: $\sigma = 0.15 \times \texttt{sigma\_scale}$
* **Noise scaling:** All $\sigma$ values are multiplied by `sigma_scale` (configurable via `--sigscale`, default 1).

### Scenario 8: CV-Friendly Dyadic AR(2), Coefficient Breaks Only

**Goal:** A well-conditioned variant of Scenario 6 for cross-validation benchmarking, with moderate stationary coefficients that keep the optimal lambda in a predictable range.

* **Parameters:** $T=1024, p=2, m_0=2$.
* **Break Points:** $t_1=513, t_2=769$ (dyadic, as in Chan et al. 2014).
* **Coefficients ($\phi$):**
  * Regime 1: $(\phi_1=0.5,\; \phi_2=0.0)$ — moderate positive lag-1, no lag-2
  * Regime 2: $(\phi_1=-0.4,\; \phi_2=0.3)$ — sign flip on lag-1, mild lag-2
  * Regime 3: $(\phi_1=0.6,\; \phi_2=-0.2)$ — moderate positive lag-1, mild negative lag-2
* **Variance ($\sigma$):** Constant across regimes; configurable via `--sigma` (default 1).
* **Coefficient jumps:** $|\Delta\phi_1| \in \{0.9, 1.0\}$, $|\Delta\phi_2| \in \{0.3, 0.5\}$.

### Scenario 9: CV-Friendly Equal-Thirds AR(2), Coefficient and Variance Breaks

**Goal:** A well-conditioned variant of Scenario 7 combining the moderate AR(2) coefficients of Scenario 8 with controlled heteroskedasticity (2:1 variance ratio instead of 4:1).

* **Parameters:** $T=1002, p=2, m_0=2$.
* **Break Points:** $t_1=334, t_2=668$ (equal thirds).
* **Coefficients ($\phi$):** Same as Scenario 8 across all three regimes.
* **Variance ($\sigma$):**
  * Regime 1: $\sigma = 0.5 \times \texttt{sigma\_scale}$
  * Regime 2: $\sigma = 1.0 \times \texttt{sigma\_scale}$
  * Regime 3: $\sigma = 0.5 \times \texttt{sigma\_scale}$
* **Noise scaling:** All $\sigma$ values are multiplied by `sigma_scale` (configurable via `--sigscale`, default 1).

### Scenario 10: Variance-Dominated Breaks, Moderate Coefficient Signal

**Goal:** A difficulty level between Scenarios 9 and the original Scenario 10. Coefficient jumps are moderate (~0.2–0.3 per component) — weaker than Scenario 9 but strong enough for H-SBAR to leverage. The variance ratio remains large (5:1), so variance is still the dominant break signal; a method insensitive to variance shifts will still underperform relative to H-SBAR.

* **Parameters:** $T=1002, p=2, m_0=2$.
* **Break Points:** $t_1=334, t_2=668$ (equal thirds, same as Scenario 9).
* **Coefficients ($\phi$):**
  * Regime 1: $(\phi_1=0.50,\; \phi_2=0.10)$
  * Regime 2: $(\phi_1=0.20,\; \phi_2=0.30)$
  * Regime 3: $(\phi_1=0.50,\; \phi_2=0.00)$
* **Coefficient jumps:** $|\Delta\phi_1|=0.30$ at both breaks; $|\Delta\phi_2| \in \{0.20, 0.30\}$.
* **Variance ($\sigma$):**
  * Regime 1: $\sigma = 0.2 \times \texttt{sigma\_scale}$
  * Regime 2: $\sigma = 1.0 \times \texttt{sigma\_scale}$ — variance spike
  * Regime 3: $\sigma = 0.2 \times \texttt{sigma\_scale}$
* **Variance ratio:** 5:1 between the middle regime and the flanking regimes.
* **Noise scaling:** All $\sigma$ values are multiplied by `sigma_scale` (configurable via `--sigscale`, default 1).

---

### Implementation Summary for the Agent

| Feature | Simplest (Scenario 1) | Most Complex (Scenario 4/5) |
| --- | --- | --- |
| **Series Length ($T$)** | 300 | 500+ |
| **AR Order ($p$)** | 1 | 2 to 5 |
| **Mean Shifts ($\phi$)** | Discrete, large jumps | Small, clustered changes |
| **Variance Shifts ($\sigma$)** | None (Constant) | Regime-dependent $\sigma_j$ |
| **Noise Type** | i.i.d. Gaussian | Autocorrelated/Heteroscedastic |

**Recommendation:** Run 100 simulations per scenario and report the **Hausdorff distance** between the true and estimated break sets to measure localization accuracy.