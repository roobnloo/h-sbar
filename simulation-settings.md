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
* **Variance ($\sigma$):** Constant $\sigma = 0.1$.

### Scenario 3: Small-Jump Detection (Lower Signal-to-Noise)

**Goal:** Evaluate localization accuracy when the "jump size" between regimes is small.

* **Parameters:** $T=300, p=2, m_0=2$.
* **Break Points:** $t_1=100, t_2=200$.
* **Coefficients ($\phi$):**
* Regime 1: $(\phi_{1,1}=0.4, \phi_{1,2}=0.2)$
* Regime 2: $(\phi_{2,1}=0.5, \phi_{2,2}=0.1)$
* Regime 3: $(\phi_{3,1}=0.3, \phi_{3,2}=0.2)$


* **Variance ($\sigma$):** Constant $\sigma = 0.1$.

### Scenario 4: Joint Coefficient & Variance Shift (Target Case)

**Goal:** Validate your primary extension—identifying structural breaks that occur simultaneously in system dynamics and noise.

* **Parameters:** $T=500, p=1, m_0=2$.
* **Break Points:** $t_1=150, t_2=350$.
* **Dynamics:**
* **Regime 1:** $\phi = 0.5, \sigma = 0.1$ (Low activity, low noise)
* **Regime 2:** $\phi = 0.9, \sigma = 0.4$ (High persistence, high noise - e.g., "Crisis" phase)
* **Regime 3:** $\phi = 0.2, \sigma = 0.15$ (Mean reverting, moderate noise)



### Scenario 5: Model Misspecification (Correlated Error)

**Goal:** Test robustness when noise is not white noise. This adapts Safikhani & Shojaie’s dense covariance scenario to 1D.

* **Parameters:** $T=300, p=1, m_0=2$.
* **Noise Profile:** $\epsilon_t$ follows an $AR(1)$ process: $\epsilon_t = 0.5\epsilon_{t-1} + \nu_t$, where $\nu_t \sim N(0, 1)$.
* **Coefficients/Variance:** Use values from Scenario 4 to see if correlated noise masks the structural breaks.

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