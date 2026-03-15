Based on the simulation frameworks utilized by **Chan et al. (2014)**, here is the set of experimental configurations for testing your structural break model.

These settings are tailored to evaluate your objective of capturing simultaneous shifts in coefficients ($\phi$) and innovation variance ($\sigma$).

# Simulation Settings for $AR(p)$ Joint Break Detection

### Configuration Overview

All scenarios assume a piecewise stationary process:


$$Y_t = \sum_{j=1}^{m_0+1} \left( \sum_{k=1}^p \phi_{j,k} Y_{t-k} + \sigma_j \epsilon_t \right) I(t_{j-1} \le t < t_j)$$


where $\epsilon_t \sim N(0, 1)$.

---

### Scenario 1: CV-Friendly Dyadic AR(2), Coefficient Breaks Only

**Goal:** A well-conditioned variant of the Chan et al. (2014) dyadic scenario for cross-validation benchmarking, with moderate stationary coefficients that keep the optimal lambda in a predictable range.

* **Parameters:** $T=1024, p=2, m_0=2$.
* **Break Points:** $t_1=513, t_2=769$ (dyadic, as in Chan et al. 2014).
* **Coefficients ($\phi$):**
  * Regime 1: $(\phi_1=0.5,\; \phi_2=0.0)$ — moderate positive lag-1, no lag-2
  * Regime 2: $(\phi_1=-0.4,\; \phi_2=0.3)$ — sign flip on lag-1, mild lag-2
  * Regime 3: $(\phi_1=0.6,\; \phi_2=-0.2)$ — moderate positive lag-1, mild negative lag-2
* **Variance ($\sigma$):** Constant across regimes; configurable via `--sigma` (default 1).
* **Coefficient jumps:** $|\Delta\phi_1| \in \{0.9, 1.0\}$, $|\Delta\phi_2| \in \{0.3, 0.5\}$.

### Scenario 2: CV-Friendly Equal-Thirds AR(2), Coefficient and Variance Breaks

**Goal:** A well-conditioned variant combining the moderate AR(2) coefficients of Scenario 1 with controlled heteroskedasticity (2:1 variance ratio).

* **Parameters:** $T=1002, p=2, m_0=2$.
* **Break Points:** $t_1=334, t_2=668$ (equal thirds).
* **Coefficients ($\phi$):** Same as Scenario 1 across all three regimes.
* **Variance ($\sigma$):**
  * Regime 1: $\sigma = 0.5 \times \texttt{sigma\_scale}$
  * Regime 2: $\sigma = 1.0 \times \texttt{sigma\_scale}$
  * Regime 3: $\sigma = 0.5 \times \texttt{sigma\_scale}$
* **Noise scaling:** All $\sigma$ values are multiplied by `sigma_scale` (configurable via `--sigscale`, default 1).

### Scenario 3: Variance-Dominated Breaks, Moderate Coefficient Signal

**Goal:** A difficulty level above Scenario 2. Coefficient jumps are moderate (~0.2–0.3 per component) — weaker than Scenario 2 but strong enough for H-SBAR to leverage. The variance ratio remains large (5:1), so variance is still the dominant break signal; a method insensitive to variance shifts will still underperform relative to H-SBAR.

* **Parameters:** $T=1002, p=2, m_0=2$.
* **Break Points:** $t_1=334, t_2=668$ (equal thirds, same as Scenario 2).
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

### Implementation Summary

| Feature | Scenario 1 | Scenario 2 | Scenario 3 |
| --- | --- | --- | --- |
| **Series Length ($T$)** | 1024 | 1002 | 1002 |
| **AR Order ($p$)** | 2 | 2 | 2 |
| **Break Points** | Dyadic (513, 769) | Equal thirds (334, 668) | Equal thirds (334, 668) |
| **Coefficient Jumps** | Large (0.9–1.0) | Large (0.9–1.0) | Moderate (0.3) |
| **Variance Shifts ($\sigma$)** | None (constant) | 2:1 ratio | 5:1 ratio |

**Recommendation:** Run 100 simulations per scenario and report the **Hausdorff distance** between the true and estimated break sets to measure localization accuracy.
