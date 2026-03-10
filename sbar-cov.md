# Sketch of Chan, Yau & Zhang (2014): Group LASSO for Structural Break Time Series

## 1. The SBAR Model

Chan et al. consider an $(m+1)$-regime **structural break autoregressive (SBAR)** process:

$$Y_t = \sum_{j=1}^{m+1} \left[ \boldsymbol{\beta}_j^T \mathbf{Y}_{t-1} + \sigma(Y_{t-1}, \ldots, Y_{t-q})\,\varepsilon_t \right] \mathbf{1}(t_{j-1} \leq t < t_j), \tag{1.1}$$

where:

- $\mathbf{Y}_{t-1} = (Y_{t-1}, \ldots, Y_{t-p})^T \in \mathbb{R}^{p}$ is the lagged regressor vector,
- $\boldsymbol{\beta}_j = (\beta_{j1}, \ldots, \beta_{jp})^T \in \mathbb{R}^{p}$ are the AR coefficients in regime $j$,
- $1 = t_0 < t_1 < \cdots < t_m < t_{m+1} = n+1$ are the (unknown) change-point locations,
- $\sigma(\cdot)$ is a measurable function on $\mathbb{R}^q$ — **it does not carry a regime index**,
- $\{\varepsilon_t\}$ is white noise with mean zero and **unit variance**.

The number of change-points $m$ and the AR order $p$ are both positive integers. Crucially, **the noise scale $\sigma(\cdot)$ is assumed constant across regimes** — only the AR coefficient vector $\boldsymbol{\beta}_j$ changes at each break.

---

## 2. The Key Reparametrization

The central insight is to **recast break detection as a grouped variable selection problem** by encoding the AR coefficients as cumulative increments.

Define a new set of vectors $\{\boldsymbol{\theta}_i\}_{i=1}^n$ as follows:

$$\boldsymbol{\theta}_i = \begin{cases} \boldsymbol{\beta}_1 & i = 1, \\ \boldsymbol{\beta}_{j+1} - \boldsymbol{\beta}_j & \text{if } i = t_j \text{ (a true change-point)}, \\ \mathbf{0} & \text{otherwise.} \end{cases}$$

Under this reparametrization, the AR coefficients at any time $t$ in regime $j$ can be recovered as:

$$\boldsymbol{\beta}_j = \sum_{i=1}^{t_j} \boldsymbol{\theta}_i.$$

The collection $\boldsymbol{\theta}(n) = (\boldsymbol{\theta}_1, \boldsymbol{\theta}_2, \ldots, \boldsymbol{\theta}_n)^T$ is **sparse**: only $m+1$ of the $n$ vectors are nonzero — one for the initial coefficient vector and one at each true change-point. Detecting breaks is therefore equivalent to finding the nonzero $\boldsymbol{\theta}_i$'s.

---

## 3. The High-Dimensional Linear Regression Reformulation

Define the $n \times np$ design matrix $\mathbf{X}_n$ whose $(t, i)$-th block (of size $p$) equals $\mathbf{Y}_{t-1}^T$ for $i \leq t$ and $\mathbf{0}$ otherwise:

$$\mathbf{X}_n = \begin{pmatrix} \mathbf{Y}_0^T & \mathbf{0} & \cdots & \mathbf{0} \\ \mathbf{Y}_1^T & \mathbf{Y}_1^T & \cdots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{Y}_{n-1}^T & \mathbf{Y}_{n-1}^T & \cdots & \mathbf{Y}_{n-1}^T \end{pmatrix},$$

where $\mathbf{Y}_k^T = (Y_k, Y_{k-1}, \ldots, Y_{k-p+1})$.

Let $\mathbf{Y}_n^0 = (Y_1, \ldots, Y_n)^T$ and $\boldsymbol{\eta}(n) = (\sigma_1\varepsilon_1, \ldots, \sigma_n\varepsilon_n)^T$ with $\sigma_t = \sigma(Y_{t-1}, \ldots, Y_{t-q})$. Then model (1.1) becomes a **high-dimensional linear regression**:

$$\mathbf{Y}_n^0 = \mathbf{X}_n\,\boldsymbol{\theta}(n) + \boldsymbol{\eta}(n). \tag{2.1}$$

Since only $m+1$ of the $n$ coefficient groups are nonzero, one seeks a **group-sparse** solution to (2.1).

---

## 4. The Group LASSO Estimator (One-Step)

Chan et al. propose estimating $\boldsymbol{\theta}(n)$ by the **group LASSO**:

$$\hat{\boldsymbol{\theta}}(n) = \underset{\boldsymbol{\theta}(n)}{\arg\min}\; \frac{1}{n}\left\|\mathbf{Y}_n^0 - \mathbf{X}_n\boldsymbol{\theta}(n)\right\|^2 + \lambda_n \sum_{i=1}^{n} \|\boldsymbol{\theta}_i\|_2, \tag{2.2}$$

where $\lambda_n > 0$ is a regularization parameter and $\|\cdot\|_2$ is the Euclidean norm.

**Interpretation:**
- The first term is the **ordinary least squares** (sum of squared residuals) loss divided by $n$.
- The second term is the **group LASSO penalty**: the sum of $\ell_2$-norms of the coefficient groups $\boldsymbol{\theta}_i$. This encourages entire groups to be exactly zero, producing a sparse solution.
- When $\hat{\boldsymbol{\theta}}_i \neq \mathbf{0}$ for some $i \geq 2$, this signals a change in the AR parameter vector at time $i$.

The estimated change-point set is:

$$\hat{A}_n = \{i \geq 2 : \hat{\boldsymbol{\theta}}_i \neq \mathbf{0}\}. \tag{2.3}$$

The estimated AR coefficients in each regime are then recovered by cumulative summation:

$$\hat{\boldsymbol{\beta}}_1 = \hat{\boldsymbol{\theta}}_1, \qquad \hat{\boldsymbol{\beta}}_j = \sum_{i=1}^{\hat{t}_j} \hat{\boldsymbol{\theta}}_i, \quad j = 1, \ldots, \hat{m}. \tag{2.4}$$

Note: the criterion (2.2) is **not** a penalized likelihood — there is no explicit modelling of $\sigma$. The least squares loss treats $\sigma_t$ as a nuisance; because $\sigma$ does not vary by regime, no penalty on variance is needed.

---

## 5. Asymptotic Theory

The theoretical guarantees rest on three assumptions:

- **H1**: $\{\varepsilon_t\}$ is white noise with unit variance and $E|\varepsilon_1|^{4+\delta} < \infty$.
- **H2**: Within each regime the process is $\beta$-mixing stationary with $E|Y_t|^{4+\delta} < \infty$, and the minimum jump size $\min_i \|\boldsymbol{\beta}_i^0 - \boldsymbol{\beta}_{i-1}^0\| > \nu > 0$.
- **H3**: Minimum segment length grows at rate $n\gamma_n$ with $\gamma_n \to 0$, and the regularization sequence satisfies $\gamma_n / \lambda_n \to \infty$.

Key results:

- **Theorem 2.1** (Prediction consistency): The group LASSO estimator is prediction-consistent at rate $O(\sqrt{\log n / n})$.
- **Theorem 2.3** (Overestimation): The one-step group LASSO **overestimates** the number of change-points ($|\hat{A}_n| \geq m_0$ w.p.1), but each true change-point is detected within an $n\gamma_n$-neighborhood.

---

## 6. Two-Step Procedure (Model Selection)

Because the one-step estimator generally overestimates $m_0$, a **second step** selects the best subset of change-points from $\hat{A}_n$ using an information criterion (IC):

$$\text{IC}(m, \mathbf{t}) = S_n(t_1, \ldots, t_m) + m\,\omega_n, \tag{2.9}$$

where $S_n(t_1, \ldots, t_m) = \sum_{j=1}^{m+1} S_n(t_{j-1}, t_j)$ is the total **residual sum of squares** from segment-wise OLS fits, and $\omega_n$ is a penalty for model complexity (analogous to BIC or MDL).

The refined estimates are:

$$(\hat{\hat{m}}, \hat{\hat{\mathbf{t}}}) = \underset{m \in \{0,\ldots,|\hat{A}_n|\},\; \mathbf{t} \subseteq \hat{A}_n}{\arg\min}\; \text{IC}(m, \mathbf{t}). \tag{2.10}$$

For large $|\hat{A}_n|$, a **backward elimination algorithm (BEA)** is used to efficiently search over subsets. Under the conditions of Theorem 2.4, this two-step procedure yields consistent estimates of both $m_0$ and the change-point locations.

---

## 7. The Role of $\sigma$ and the Extension: SBAR-COV

A notable feature of the Chan et al. framework is that $\sigma(Y_{t-1}, \ldots, Y_{t-q})$ is **regime-invariant**: the conditional variance function does not change across breaks. This is a binding assumption — the method can only capture changes in the AR coefficients $\boldsymbol{\beta}_j$, not changes in volatility.

### 7.1 The SBAR-COV Model

We replace $\sigma$ with a regime-specific constant $\sigma_j > 0$, giving:

$$Y_t = \sum_{j=1}^{m+1}\left[\boldsymbol{\beta}_j^T \mathbf{Y}_{t-1} + \sigma_j\,\varepsilon_t\right]\mathbf{1}(t_{j-1} \leq t < t_j), \tag{7.1}$$

where $\{\varepsilon_t\}$ is white noise with mean zero and unit variance. Now a change-point at time $t_j$ may involve a jump in $\boldsymbol{\beta}$, a jump in $\sigma$, or both.

### 7.2 A Log-Scale Parametrization of $\sigma_j$

To mirror the $\boldsymbol{\theta}$-reparametrization of Section 2, a natural first attempt works on the log scale. Define $\varphi_j = \log \sigma_j^2$ (the log-variance in regime $j$) and introduce the scalar increment sequence $\boldsymbol{\xi}(n) = (\xi_1, \xi_2, \ldots, \xi_n)^T$ by:

$$\xi_i = \begin{cases} \varphi_1 = \log \sigma_1^2 & i = 1, \\ \varphi_{j+1} - \varphi_j = \log(\sigma_{j+1}^2 / \sigma_j^2) & \text{if } i = t_j \text{ (a change-point)}, \\ 0 & \text{otherwise.} \end{cases}$$

The log-variance at any time $t$ in regime $j$ is then recovered as a cumulative sum:

$$\varphi_{j(t)} = \log \sigma_{j(t)}^2 = \sum_{i=1}^{t} \xi_i =: s_t,$$

and hence $\sigma_{j(t)}^2 = e^{s_t}$. This guarantees $\sigma_{j(t)}^2 > 0$ for any $\xi_i$, and $\xi_i = 0$ means no change in variance at time $i$. Let $r_t = Y_t - \boldsymbol{\beta}_{j(t)}^T\mathbf{Y}_{t-1}$ denote the residual. Then:

$$r_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0,\, e^{s_t}).$$

### 7.3 The Gaussian Log-Likelihood (Log-Scale)

Under Gaussianity, dropping the constant $\frac{n}{2}\log(2\pi)$, the total negative log-likelihood is:

$$\mathcal{L}_n^{\log}\bigl(\boldsymbol{\xi}(n)\bigr) = \frac{1}{2}\sum_{t=1}^{n}\left[ s_t + r_t^2\, e^{-s_t} \right] = \frac{1}{2}\sum_{t=1}^{n}\left[\sum_{i=1}^{t}\xi_i + r_t^2\,\exp\!\left(-\sum_{i=1}^{t}\xi_i\right)\right], \tag{7.2}$$

where $r_t = Y_t - \boldsymbol{\beta}_{j(t)}^T\mathbf{Y}_{t-1}$. Two limiting cases make the structure transparent:

- **Constant variance** ($\xi_i = 0$ for all $i \geq 2$, so $s_t \equiv \varphi_1$): (7.2) reduces to a scaled OLS loss, recovering Chan et al.'s setting.
- **Known variance** (all $\xi_i$ known): (7.2) becomes a weighted least squares criterion with weights $e^{-s_t}$, a generalized-least-squares version of (2.2).

### 7.4 A Penalized Log-Scale Criterion (Non-Convex)

Penalizing the AR coefficient increments (from §2) and the log-variance increments $\boldsymbol{\xi}(n)$ jointly:

$$Q_n^{\log}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\xi}(n)\bigr) = \frac{1}{n}\mathcal{L}_n^{\log}\bigl(\boldsymbol{\xi}(n)\bigr) + \lambda_n^\beta \sum_{i=2}^{n}\|\boldsymbol{\theta}_i\|_2 + \lambda_n^\sigma \sum_{i=2}^{n}|\xi_i|, \tag{7.4}$$

where $\boldsymbol{\theta}(n)$ are Chan et al.'s AR coefficient increments from §2 and $\lambda_n^\beta, \lambda_n^\sigma > 0$.

**Remarks:**

1. *Separate tuning parameters.* Using distinct $\lambda_n^\beta$ and $\lambda_n^\sigma$ allows the method to detect breaks in the mean structure and the variance structure at different sensitivities and potentially different locations.

2. *Joint group penalty.* If breaks in $\boldsymbol{\beta}$ and $\sigma^2$ are assumed to co-occur, one can instead use a joint group penalty $\lambda_n \sum_{i=2}^n \|(\boldsymbol{\theta}_i^T, \xi_i)^T\|_2$, which forces both components to zero simultaneously.

3. *Non-convexity.* The loss $\mathcal{L}_n^{\log}$ in (7.2) is **not jointly convex** in $(\boldsymbol{\theta}(n), \boldsymbol{\xi}(n))$ because the term $r_t^2 e^{-s_t}$ couples the two parameter sequences nonlinearly. This complicates both computation and asymptotic theory.

4. *Profile approach.* For fixed $\boldsymbol{\xi}(n)$ (fixed segment variances), $Q_n^{\log}$ is convex in $\boldsymbol{\theta}(n)$ — it becomes a weighted group LASSO. For fixed $\boldsymbol{\theta}(n)$, the problem in $\boldsymbol{\xi}(n)$ reduces to a penalized log-likelihood for a change-in-variance model. This suggests an alternating/profile optimization strategy.

---

### 7.5 Natural Parametrization and Joint Convexity

The non-convexity of (7.2) motivates seeking a reparametrization that restores joint convexity, following Yu & Bien (2019), who observed that the Gaussian negative log-likelihood — non-convex in $(\boldsymbol{\beta}, \sigma^2)$ — becomes jointly convex under the **natural exponential family parameters** $\phi = 1/\sigma^2$ and $\boldsymbol{\gamma} = \phi\boldsymbol{\beta}$.

Following Yu & Bien, define per-regime **natural parameters**:

$$\phi_j = \frac{1}{\sigma_j^2} \quad \text{(precision)}, \qquad \boldsymbol{\gamma}_j = \phi_j\,\boldsymbol{\beta}_j \quad \text{(precision-scaled AR coefficients)}.$$

For observation $Y_t$ in regime $j(t)$, the negative log-likelihood contribution in terms of $(\phi_j, \boldsymbol{\gamma}_j)$ is

$$\ell_t = -\tfrac{1}{2}\log\phi_{j(t)} + \tfrac{1}{2}\phi_{j(t)}Y_t^2 - Y_t\,\boldsymbol{\gamma}_{j(t)}^T\mathbf{Y}_{t-1} + \frac{(\boldsymbol{\gamma}_{j(t)}^T\mathbf{Y}_{t-1})^2}{2\phi_{j(t)}}.\tag{7.5}$$

This is jointly convex in $(\boldsymbol{\gamma}_j, \phi_j)$ for $\phi_j > 0$: the terms $-\frac{1}{2}\log\phi_j$ and $\frac{\phi_j Y_t^2}{2}$ are convex in $\phi_j$; the cross term $-Y_t\boldsymbol{\gamma}_j^T\mathbf{Y}_{t-1}$ is linear; and $\frac{(\boldsymbol{\gamma}_j^T\mathbf{Y}_{t-1})^2}{2\phi_j}$ is the **quadratic-over-linear** function, jointly convex in $(\boldsymbol{\gamma}_j, \phi_j)$.

**Increment reparametrization.** We now introduce two sparse increment sequences, reusing the symbols $\boldsymbol{\theta}$ and $\boldsymbol{\psi}$ for the natural-parameter increments (note: $\boldsymbol{\theta}_i$ here denotes increments of $\boldsymbol{\gamma}$, not of $\boldsymbol{\beta}$ as in §2):

$$\boldsymbol{\theta}_i = \begin{cases} \boldsymbol{\gamma}_1 & i = 1, \\ \boldsymbol{\gamma}_{j+1} - \boldsymbol{\gamma}_j & \text{if } i = t_j, \\ \mathbf{0} & \text{otherwise,} \end{cases} \qquad \psi_i = \begin{cases} \phi_1 & i = 1, \\ \phi_{j+1} - \phi_j & \text{if } i = t_j, \\ 0 & \text{otherwise.} \end{cases}$$

Define the cumulative sums:

$$G_t := (\mathbf{X}_n\boldsymbol{\theta}(n))_t = \boldsymbol{\gamma}_{j(t)}^T\mathbf{Y}_{t-1}, \qquad \Phi_t := \sum_{i=1}^t \psi_i = \phi_{j(t)}.$$

The same design matrix $\mathbf{X}_n$ from (2.1) handles the $\boldsymbol{\theta}$ accumulation. In these coordinates, expanding $(\Phi_t Y_t - G_t)^2/\Phi_t = \Phi_t Y_t^2 - 2Y_t G_t + G_t^2/\Phi_t$, the total negative log-likelihood becomes:

$$\mathcal{L}_n^{\mathrm{nat}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) = \frac{1}{2}\sum_{t=1}^{n}\left[\underbrace{-\log \Phi_t}_{\text{convex in }\boldsymbol{\psi}} + \underbrace{\Phi_t\, Y_t^2}_{\text{linear in }\boldsymbol{\psi}} \underbrace{-\; 2Y_t\, G_t}_{\text{linear in }\boldsymbol{\theta}} + \underbrace{G_t^2 / \Phi_t}_{\text{quad-over-linear}}\right], \tag{7.6}$$

subject to $\Phi_t > 0$ for all $t$. Each of the four terms is convex in $(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n))$:

- $-\log \Phi_t$: $-\log$ of a positive linear function of $\boldsymbol{\psi}(n)$, convex;
- $\Phi_t Y_t^2$: linear in $\boldsymbol{\psi}(n)$ (since $Y_t^2$ is data), hence convex;
- $-2Y_t G_t$: linear in $\boldsymbol{\theta}(n)$, hence convex;
- $G_t^2/\Phi_t$: the **quadratic-over-linear** function $(G_t, \Phi_t)\mapsto G_t^2/\Phi_t$, jointly convex for $\Phi_t>0$, since $G_t$ is linear in $\boldsymbol{\theta}(n)$ and $\Phi_t$ is linear in $\boldsymbol{\psi}(n)$.

The structure of (7.6) is a direct analogue of the per-observation term (7.5), with per-regime constants $(\boldsymbol{\gamma}_j, \phi_j)$ replaced by their cumulative-sum representations $(G_t, \Phi_t)$.

**Penalized criterion.** Adding group-sparse penalties on the increments:

$$Q_n^{\mathrm{nat}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) = \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) + \lambda_n^\gamma \sum_{i=2}^n \|\boldsymbol{\theta}_i\|_2 + \lambda_n^\phi \sum_{i=2}^n |\psi_i|. \tag{7.7}$$

The entire objective is jointly convex in $(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n))$, making (7.7) amenable to standard convex optimization algorithms. Change-points are detected at positions $i \geq 2$ where $\hat{\boldsymbol{\theta}}_i \neq \mathbf{0}$ or $\hat{\psi}_i \neq 0$.

**Estimator.** The SBAR-COV natural-parametrization estimator is defined as the solution to the convex program:

$$\bigl(\hat{\boldsymbol{\theta}}(n),\, \hat{\boldsymbol{\psi}}(n)\bigr) = \operatorname*{argmin}_{\substack{\boldsymbol{\theta}(n)\,\in\,\mathbb{R}^{np} \\ \boldsymbol{\psi}(n)\,\in\,\mathbb{R}^n:\;\Phi_t > 0\;\forall\, t}} Q_n^{\mathrm{nat}}\bigl(\boldsymbol{\theta}(n),\, \boldsymbol{\psi}(n)\bigr). \tag{7.8}$$

The constraint set is $\boldsymbol{\theta}(n) \in \mathbb{R}^{np}$ (unconstrained) and $\boldsymbol{\psi}(n) \in \mathcal{C}$, where

$$\mathcal{C} = \Bigl\{\boldsymbol{\psi}(n)\in\mathbb{R}^n : \Phi_t := \textstyle\sum_{i=1}^t \psi_i > 0 \text{ for all } t = 1,\ldots,n\Bigr\}.$$

$\mathcal{C}$ is an open convex cone; it requires that the cumulative precision $\Phi_t$ remain strictly positive at every time point, not merely that each increment $\psi_i > 0$ (subsequent increments may be negative, representing a decrease in precision). Since the domain of $-\log\Phi_t$ is $\Phi_t>0$, any minimizer automatically satisfies $\hat{\Phi}_t > 0$.

**Recovering original parameters.** Given $(\hat{\boldsymbol{\theta}}(n), \hat{\boldsymbol{\psi}}(n))$:

$$\hat{\phi}_{j(t)} = \hat{\Phi}_t, \quad \hat{\sigma}_{j(t)}^2 = \hat{\Phi}_t^{-1}, \quad \hat{\boldsymbol{\gamma}}_{j(t)} = \sum_{i=1}^t \hat{\boldsymbol{\theta}}_i, \quad \hat{\boldsymbol{\beta}}_{j(t)} = \hat{\boldsymbol{\gamma}}_{j(t)} / \hat{\Phi}_t.$$

**Remarks.**

5. *Natural vs. log parametrization.* The convexifying reparametrization uses $\phi_j = 1/\sigma_j^2$ directly, not $\log(1/\sigma_j^2)$. The log scale (§7.2–§7.4) guarantees $\sigma^2 > 0$ automatically but sacrifices convexity; the natural scale preserves convexity but requires the explicit constraint $\Phi_t > 0$. In practice this can be enforced via a log-barrier or by initializing $\hat{\psi}_1 > 0$ and restricting subsequent increments.

6. *Interpretation of $\boldsymbol{\theta}$ increments.* Unlike Chan et al.'s $\boldsymbol{\theta}_i = \boldsymbol{\beta}_{j+1} - \boldsymbol{\beta}_j$ from §2, the increment $\boldsymbol{\theta}_i = \boldsymbol{\gamma}_{j+1} - \boldsymbol{\gamma}_j = \phi_{j+1}\boldsymbol{\beta}_{j+1} - \phi_j\boldsymbol{\beta}_j$ conflates changes in $\boldsymbol{\beta}$ and $\sigma^2$. A nonzero $\boldsymbol{\theta}_i$ signals a change in the natural parameter vector, which may originate from a shift in $\boldsymbol{\beta}$, a shift in $\sigma^2$, or both simultaneously.

---

### 7.6 Solving the Convex Program (7.8)

#### 7.6.1 Yu & Bien's Reduction Does Not Apply

Yu & Bien's key computational result (their Proposition 1) is that the natural lasso — a joint minimization over $(\boldsymbol{\theta}, \phi)$ in their notation — reduces, remarkably, to simply solving a **standard lasso** and reading off the optimal value as the variance estimate:

$$\hat{\sigma}^2_\lambda = \min_{\boldsymbol{\beta}\in\mathbb{R}^p} \left\{\frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + 2\lambda\|\boldsymbol{\beta}\|_1\right\}.$$

This reduction holds because in their setting the precision $\phi = 1/\sigma^2$ is a **single global scalar** shared by all observations. For fixed $\boldsymbol{\theta}$ (equivalently fixed $\boldsymbol{\beta}$), the profile problem in $\phi$ is one-dimensional and its first-order condition has a closed-form solution: $\hat{\phi} = n/\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$. Substituting this back yields the standard lasso objective as the profile likelihood in $\boldsymbol{\beta}$ alone.

This reduction **does not apply to SBAR-COV (7.8)** for a fundamental reason: the precision is **regime-specific**. Each observation $t$ is associated with a cumulative precision $\Phi_t = \sum_{i=1}^t \psi_i = \phi_{j(t)}$ that varies across time. The $\boldsymbol{\psi}(n)$ sequence has $n$ free parameters, and for fixed $\boldsymbol{\theta}(n)$, the profile problem in $\boldsymbol{\psi}(n)$ is:

$$\min_{\boldsymbol{\psi}(n)\in\mathcal{C}}\; \frac{1}{2n}\sum_{t=1}^n\left[-\log\Phi_t + \Phi_t Y_t^2 + \frac{G_t^2}{\Phi_t}\right] + \lambda_n^\phi \sum_{i=2}^n |\psi_i|,$$

where $\Phi_t = \sum_{i=1}^t\psi_i$ is a lower-triangular linear function of $\boldsymbol{\psi}(n)$. Because distinct $\Phi_t$'s share summands $\psi_i$ (each $\psi_i$ enters $\Phi_t$ for all $t \geq i$), the first-order conditions for $\psi_i$ couple observations across regimes and admit no closed-form solution. Consequently, optimizing (7.8) genuinely requires a custom iterative algorithm.

#### 7.6.2 Block Coordinate Descent

A natural approach exploits the **partial decoupling** of (7.8): for fixed $\boldsymbol{\psi}(n)$ the problem is convex in $\boldsymbol{\theta}(n)$ and vice versa. Alternating between the two blocks gives:

**$\boldsymbol{\theta}$-step** (fix $\boldsymbol{\psi}(n)$, hence fix all $\Phi_t > 0$). Using the expansion $(\Phi_t Y_t - G_t)^2/\Phi_t = \Phi_t(Y_t - G_t/\Phi_t)^2$, the objective in $\boldsymbol{\theta}(n)$ is:

$$\frac{1}{2n}\sum_{t=1}^n \Phi_t\!\left(Y_t - \frac{G_t}{\Phi_t}\right)^{\!2} + \lambda_n^\gamma\sum_{i=2}^n\|\boldsymbol{\theta}_i\|_2 = \frac{1}{2n}\bigl\|\mathbf{W}^{1/2}(\mathbf{y} - \mathbf{X}_n\boldsymbol{\theta}(n))\bigr\|_2^2 + \lambda_n^\gamma\sum_{i=2}^n\|\boldsymbol{\theta}_i\|_2,$$

where $\mathbf{W} = \mathrm{diag}(\Phi_1,\ldots,\Phi_n)$ and $\mathbf{y}_t^\star = \Phi_t Y_t$ is a precision-rescaled response. This is a **weighted group LASSO** — Chan et al.'s SBAR criterion generalized to GLS weights $\Phi_t$ — solvable by any standard group LASSO solver.

**$\boldsymbol{\psi}$-step** (fix $\boldsymbol{\theta}(n)$, hence fix all $G_t$). The objective in $\boldsymbol{\psi}(n)$ is:

$$\frac{1}{2n}\sum_{t=1}^n\left[-\log\Phi_t + \Phi_t Y_t^2 + \frac{G_t^2}{\Phi_t}\right] + \lambda_n^\phi\sum_{i=2}^n|\psi_i|, \qquad \Phi_t = \sum_{i=1}^t\psi_i,$$

a penalized change-in-precision problem. This is convex in $\boldsymbol{\psi}(n)$ (since each summand is convex in $\Phi_t$, which is linear in $\boldsymbol{\psi}$) and can be solved by proximal gradient descent applied to the cumulative-sum structure, or by passing the lower-triangular linear constraint $\Phi = L\boldsymbol{\psi}$ to an interior-point solver.

Convergence of block coordinate descent to the global minimum of (7.8) follows from the joint convexity of the objective and the fact that each subproblem has a unique solution (strict convexity in each block when the design and data are non-degenerate).

#### 7.6.3 General Convex Optimization

Alternatively, (7.8) can be solved in one pass by any general-purpose convex optimizer that recognizes the constituent atoms:

- $-\log\Phi_t$: log-concave barrier term, standard in DCP frameworks (e.g., CVXPY);
- $G_t^2/\Phi_t$: **quadratic-over-linear**, a built-in DCP atom, or equivalently a rotated second-order cone (RSOC) constraint $u_t\Phi_t \geq G_t^2$, $u_t \geq 0$;
- $\|\boldsymbol{\theta}_i\|_2$: group LASSO penalty, a DCP atom;
- $|\psi_i|$: LASSO penalty, a DCP atom.

The presence of the $-\log\Phi_t$ barrier makes the domain automatically open ($\Phi_t > 0$), so (7.8) can be passed directly to an interior-point solver without explicitly enforcing $\mathcal{C}$. The problem size is $O(n(p+1))$ variables, which is large for long time series; the block coordinate approach in §7.6.2 is likely more scalable in practice, since each sub-problem has an efficient solver.

#### 7.6.4 ADMM

A structured ADMM approach offers a middle ground: each iteration requires only gradient computations and closed-form proximal steps, avoiding both the large interior-point linear systems of §7.6.3 and the group LASSO sub-solves of §7.6.2.

**Variable splitting.** Let $\mathbf{L}\in\{0,1\}^{n\times n}$ denote the lower-triangular all-ones matrix, so that $\Phi_t = (\mathbf{L}\boldsymbol{\psi})_t = \sum_{i=1}^t\psi_i$. Introduce penalty-side copies $(\mathbf{z}_1, \mathbf{z}_2)$ of $(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n))$ to decouple the smooth loss from the non-smooth group penalties:

$$\min_{\boldsymbol{\theta},\,\boldsymbol{\psi},\,\mathbf{z}_1,\,\mathbf{z}_2}\; \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \lambda_n^\gamma\sum_{i=2}^n\|\mathbf{z}_{1,i}\|_2 + \lambda_n^\phi\sum_{i=2}^n|z_{2,i}| \quad\text{s.t.}\quad \boldsymbol{\theta} = \mathbf{z}_1,\;\boldsymbol{\psi} = \mathbf{z}_2. \tag{7.9}$$

The scaled augmented Lagrangian with dual variables $(\mathbf{u}_1, \mathbf{u}_2)$ and penalty $\rho > 0$ is:

$$\mathcal{A}_\rho = \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \lambda_n^\gamma\sum_{i\geq 2}\|\mathbf{z}_{1,i}\|_2 + \lambda_n^\phi\sum_{i\geq 2}|z_{2,i}| + \frac{\rho}{2}\|\boldsymbol{\theta} - \mathbf{z}_1 + \mathbf{u}_1\|^2 + \frac{\rho}{2}\|\boldsymbol{\psi} - \mathbf{z}_2 + \mathbf{u}_2\|^2.$$

**ADMM iterations.** Starting from any $(\mathbf{z}_1^0, \mathbf{z}_2^0, \mathbf{u}_1^0, \mathbf{u}_2^0)$ with $(\mathbf{L}\mathbf{z}_2^0)_t > 0$ for all $t$, iterate:

**Step 1 — smooth update** $(\boldsymbol{\theta}^{k+1}, \boldsymbol{\psi}^{k+1})$: solve

$$\min_{\boldsymbol{\theta},\,\boldsymbol{\psi}}\;\frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \frac{\rho}{2}\|\boldsymbol{\theta} - \mathbf{c}_1^k\|^2 + \frac{\rho}{2}\|\boldsymbol{\psi} - \mathbf{c}_2^k\|^2, \tag{$\star$}$$

where $\mathbf{c}_1^k = \mathbf{z}_1^k - \mathbf{u}_1^k$ and $\mathbf{c}_2^k = \mathbf{z}_2^k - \mathbf{u}_2^k$. This is a smooth, strictly convex problem (the $-\log\Phi_t$ barrier automatically enforces $\Phi_t > 0$) with closed-form gradients:

$$\nabla_{\boldsymbol{\theta}}\mathcal{A} = \frac{1}{n}\mathbf{X}_n^T\boldsymbol{\delta} + \rho(\boldsymbol{\theta} - \mathbf{c}_1^k), \qquad \delta_t = -Y_t + \frac{G_t}{\Phi_t},$$

$$\nabla_{\boldsymbol{\psi}}\mathcal{A} = \frac{1}{n}\mathbf{L}^T\boldsymbol{\eta} + \rho(\boldsymbol{\psi} - \mathbf{c}_2^k), \qquad \eta_t = -\frac{1}{2\Phi_t} + \frac{Y_t^2}{2} - \frac{G_t^2}{2\Phi_t^2},$$

where $\mathbf{L}^T\boldsymbol{\eta}$ is a reverse cumulative sum: $(\mathbf{L}^T\boldsymbol{\eta})_i = \sum_{t\geq i}\eta_t$. These expressions are $O(np)$ to compute, making gradient-based solvers (L-BFGS, Newton-CG) efficient.

**Step 2 — group soft-threshold** $\mathbf{z}_1^{k+1}$: for each $i \geq 2$,

$$\mathbf{z}_{1,i}^{k+1} = \mathcal{S}_{\lambda_n^\gamma/\rho}^{\mathrm{grp}}\!\bigl(\boldsymbol{\theta}_i^{k+1} + \mathbf{u}_{1,i}^k\bigr) := \left(1 - \frac{\lambda_n^\gamma/\rho}{\|\boldsymbol{\theta}_i^{k+1} + \mathbf{u}_{1,i}^k\|_2}\right)_{\!\!+}\!\!\bigl(\boldsymbol{\theta}_i^{k+1} + \mathbf{u}_{1,i}^k\bigr),$$

and $\mathbf{z}_{1,1}^{k+1} = \boldsymbol{\theta}_1^{k+1} + \mathbf{u}_{1,1}^k$ (no penalty on the first block). **Closed form.**

**Step 3 — scalar soft-threshold** $\mathbf{z}_2^{k+1}$: for each $i \geq 2$,

$$z_{2,i}^{k+1} = \mathcal{S}_{\lambda_n^\phi/\rho}\!\bigl(\psi_i^{k+1} + u_{2,i}^k\bigr) := \operatorname{sgn}(\psi_i^{k+1} + u_{2,i}^k)\left(|\psi_i^{k+1}+u_{2,i}^k| - \frac{\lambda_n^\phi}{\rho}\right)_{\!\!+},$$

and $z_{2,1}^{k+1} = \psi_1^{k+1} + u_{2,1}^k$. **Closed form.**

**Step 4 — dual update:**

$$\mathbf{u}_1^{k+1} = \mathbf{u}_1^k + \boldsymbol{\theta}^{k+1} - \mathbf{z}_1^{k+1}, \qquad \mathbf{u}_2^{k+1} = \mathbf{u}_2^k + \boldsymbol{\psi}^{k+1} - \mathbf{z}_2^{k+1}.$$

**Convergence.** Since $Q_n^{\mathrm{nat}}$ is jointly convex and closed, and the constraint $\boldsymbol{\theta}=\mathbf{z}_1$, $\boldsymbol{\psi}=\mathbf{z}_2$ is linear, the standard ADMM convergence theorem (Boyd et al., 2011, §3.2) guarantees that the primal residuals $\|\boldsymbol{\theta}^k - \mathbf{z}_1^k\|$, $\|\boldsymbol{\psi}^k - \mathbf{z}_2^k\|$ and the dual residuals $\|\mathbf{z}_1^k - \mathbf{z}_1^{k-1}\|$, $\|\mathbf{z}_2^k - \mathbf{z}_2^{k-1}\|$ converge to zero. The cost per outer iteration is dominated by solving $(\star)$ to moderate precision (e.g., 5–10 L-BFGS steps), which is $O(np)$ per gradient evaluation. Steps 2–4 are $O(np)$ in total. The overall per-iteration complexity is thus $O(np)$, comparable to a single weighted group LASSO solve.

---

## 8. Co-Located Breaks: A Joint Group Penalty

### 8.1 Motivation

The separated penalty in (7.7) treats the AR-coefficient increments $\boldsymbol{\theta}_i$ and precision increments $\psi_i$ as independent sparse signals, controlled by distinct regularization strengths $\lambda_n^\gamma$ and $\lambda_n^\phi$. Under this formulation a break detected in the AR structure need not coincide with a break in the precision, and vice versa. This is appropriate when the two types of structural change genuinely decouple — for instance, a purely volatility-driven shock that leaves the conditional mean dynamics unchanged.

In many economic and financial applications, however, regime changes tend to be simultaneous: a shift in the data-generating mechanism typically alters both the conditional mean and the conditional variance at the same time. Imposing this co-location constraint reduces the effective number of changepoint locations that must be estimated, borrows strength across the two components, and substantially simplifies tuning (one penalty level instead of two).

### 8.2 The Joint Criterion

Replace the separated penalty in (7.7) with the **joint group penalty**

$$Q_n^{\mathrm{jnt}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) = \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) + \lambda_n \sum_{i=2}^{n} \sqrt{\|\boldsymbol{\theta}_i\|_2^2 + c\,\psi_i^2}, \tag{8.1}$$

where $\lambda_n > 0$ is a single regularization parameter and $c > 0$ is a **fixed scale parameter** (not tuned by cross-validation). The loss $\mathcal{L}_n^{\mathrm{nat}}$ is the same jointly convex negative log-likelihood from (7.6). Since the joint penalty is a sum of Euclidean norms of linear functions of $(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n))$, and each norm is convex, $Q_n^{\mathrm{jnt}}$ inherits joint convexity. The estimator is:

$$\bigl(\hat{\boldsymbol{\theta}}(n),\, \hat{\boldsymbol{\psi}}(n)\bigr) = \operatorname*{argmin}_{\substack{\boldsymbol{\theta}(n)\,\in\,\mathbb{R}^{np} \\ \boldsymbol{\psi}(n)\,\in\,\mathbb{R}^n:\;\Phi_t > 0\;\forall\, t}} Q_n^{\mathrm{jnt}}\bigl(\boldsymbol{\theta}(n),\, \boldsymbol{\psi}(n)\bigr). \tag{8.2}$$

**Algebraic equivalence.** Define the scaled augmented vector at position $i$:

$$\mathbf{v}_i := \begin{pmatrix} \boldsymbol{\theta}_i \\ \sqrt{c}\,\psi_i \end{pmatrix} \in \mathbb{R}^{p+1}.$$

Then $\sqrt{\|\boldsymbol{\theta}_i\|_2^2 + c\,\psi_i^2} = \|\mathbf{v}_i\|_2$, so (8.1) is a **group LASSO on $\mathbf{v}_i$** — a standard $\ell_2$-norm penalty applied to the $(p+1)$-dimensional vector that concatenates the AR-increment block with the precision-increment scalar (after scaling by $\sqrt{c}$).

### 8.3 The Scale Parameter $c$

Unlike $\lambda_n$, the parameter $c$ is not a regularization strength — it does not determine the total amount of shrinkage. Rather, it sets the **relative scale** at which a precision increment $\psi_i$ is treated as equivalent to a unit displacement in the AR-increment space. Its role is analogous to a metric or weighting matrix inside the group norm.

**Dimensional analysis.** Under the natural parametrization, $\psi_i = \Delta\phi_j = \Delta(1/\sigma_j^2)$ has units of inverse variance, while $\|\boldsymbol{\theta}_i\|_2 = \|\Delta\boldsymbol{\gamma}_j\|_2 = \|\Delta(\phi_j\boldsymbol{\beta}_j)\|_2$ has units of inverse-variance times the AR coefficient. A natural anchor is:

$$c_0 = \frac{\|\boldsymbol{\gamma}_1\|_2^2}{\phi_1^2} = \|\boldsymbol{\beta}_1\|_2^2,$$

which makes $c_0\,\psi_i^2 = \|\boldsymbol{\beta}_1\|_2^2 (\Delta\phi)^2$ dimensionally comparable to $\|\Delta\boldsymbol{\gamma}\|_2^2 \approx \phi_1^2\|\Delta\boldsymbol{\beta}\|_2^2$. In practice, $c_0$ can be estimated from a preliminary unpenalized or lightly penalized fit by plugging in $\hat{\boldsymbol{\beta}}_1$ and $\hat{\phi}_1$. Alternatively, one may set $c = 1$ and rely on the fact that the penalty drives a single joint sparsity pattern regardless of the exact scale, at the cost of somewhat unequal shrinkage of the two components.

**Limiting cases.** The family (8.1) interpolates between two degenerate cases:

- $c \to 0$: the penalty reduces to $\lambda_n \sum_{i=2}^n \|\boldsymbol{\theta}_i\|_2$, penalizing only AR-coefficient increments. No regularization is placed on precision increments, and the method devolves to a generalized-least-squares version of Chan et al.'s SBAR.
- $c \to \infty$: the penalty is dominated by $\lambda_n\sqrt{c}\sum_{i=2}^n|\psi_i|$, penalizing only precision increments (an effective re-scaling of $\lambda_n$ by $\sqrt{c}$). No AR changepoints are sought.
- $0 < c < \infty$: the penalty jointly shrinks both components and enforces co-location of detected breaks.

### 8.4 Co-Location Property

The key structural feature of (8.1) is that the penalty term at position $i$ is zero if and only if **both** $\boldsymbol{\theta}_i = \mathbf{0}$ and $\psi_i = 0$:

$$\sqrt{\|\boldsymbol{\theta}_i\|_2^2 + c\,\psi_i^2} = 0 \iff \boldsymbol{\theta}_i = \mathbf{0} \text{ and } \psi_i = 0.$$

Consequently, the group soft-threshold operator (§8.5 below) zeros the entire pair $(\boldsymbol{\theta}_i, \psi_i)$ simultaneously or leaves both nonzero. This enforces a **shared sparsity pattern**: the detected changepoint set

$$\hat{A}_n^{\mathrm{jnt}} = \{i \geq 2 : \hat{\boldsymbol{\theta}}_i \neq \mathbf{0} \text{ or } \hat{\psi}_i \neq 0\}$$

coincides with the set where AR-structure breaks and precision breaks are jointly located. Under the separated penalty (7.7), a position could appear in the AR changepoint set but not the precision changepoint set, or vice versa; under (8.1) this is impossible.

**Contrast with separated penalty.** The separated penalty (7.7) requires two tuning parameters $(\lambda_n^\gamma, \lambda_n^\phi)$ and yields two potentially distinct changepoint sets. The joint penalty (8.1) requires one tuning parameter $\lambda_n$ and a single fixed scale $c$, and produces one unified changepoint set. This is a modeling restriction — it is correctly specified when breaks are truly co-located, and misspecified otherwise — but reduces variance in the estimated changepoint locations and simplifies model selection.

### 8.5 Proximal Operator and Closed-Form Thresholding

The proximal operator of $f_i(\boldsymbol{\theta}_i, \psi_i) = \lambda_n \sqrt{\|\boldsymbol{\theta}_i\|_2^2 + c\,\psi_i^2}$ at a point $(\mathbf{a}_i, b_i)$ is derived by making the substitution $\tilde{\mathbf{v}}_i = (\boldsymbol{\theta}_i^T, \sqrt{c}\,\psi_i)^T$ and $\tilde{\mathbf{a}}_i = (\mathbf{a}_i^T, \sqrt{c}\,b_i)^T$. In the transformed coordinates the penalty is the standard Euclidean norm $\lambda_n\|\tilde{\mathbf{v}}_i\|_2$, whose proximal operator is the vector soft-threshold. Inverting the substitution gives the **joint group soft-threshold**:

$$\mathcal{T}_{\lambda_n,c}(\mathbf{a}_i, b_i) := \left(1 - \frac{\lambda_n}{\sqrt{\|\mathbf{a}_i\|_2^2 + c\,b_i^2}}\right)_{\!\!+} (\mathbf{a}_i,\, b_i). \tag{8.3}$$

The scalar $\left(1 - \lambda_n/\|\tilde{\mathbf{a}}_i\|_2\right)_+$ is a common shrinkage factor applied **identically** to both $\mathbf{a}_i$ and $b_i$. This is the key property: both components are shrunk by the same multiplicative factor, so their ratio $b_i / \|\mathbf{a}_i\|_2$ is preserved whenever the group is not zeroed out entirely. In particular, a group is zeroed iff $\|\mathbf{a}_i\|_2^2 + c\,b_i^2 \leq \lambda_n^2$, which is an ellipsoidal dead-zone in the $(\mathbf{a}_i, b_i)$-space with semi-axes $\lambda_n$ along the $\boldsymbol{\theta}$ directions and $\lambda_n/\sqrt{c}$ along the $\psi$ direction.

### 8.6 Modified ADMM

The ADMM of §7.6.4 requires only a single modification: Steps 2 and 3 (the separate group and scalar soft-threshold steps) are **merged into one joint group soft-threshold step**.

The variable-splitting formulation (7.9) becomes

$$\min_{\boldsymbol{\theta},\,\boldsymbol{\psi},\,\mathbf{z}_1,\,\mathbf{z}_2}\; \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \lambda_n\sum_{i=2}^n\sqrt{\|\mathbf{z}_{1,i}\|_2^2 + c\,z_{2,i}^2} \quad\text{s.t.}\quad \boldsymbol{\theta} = \mathbf{z}_1,\;\boldsymbol{\psi} = \mathbf{z}_2. \tag{8.4}$$

**Step 1** (smooth update) is unchanged: solve $(\star)$ from §7.6.4 with the same gradients. **Steps 2 and 3** are replaced by the joint proximal step: for each $i \geq 2$,

$$\bigl(\mathbf{z}_{1,i}^{k+1},\, z_{2,i}^{k+1}\bigr) = \mathcal{T}_{\lambda_n/\rho,\,c}\!\bigl(\boldsymbol{\theta}_i^{k+1} + \mathbf{u}_{1,i}^k,\; \psi_i^{k+1} + u_{2,i}^k\bigr), \tag{8.5}$$

and $(\mathbf{z}_{1,1}^{k+1}, z_{2,1}^{k+1}) = (\boldsymbol{\theta}_1^{k+1} + \mathbf{u}_{1,1}^k,\; \psi_1^{k+1} + u_{2,1}^k)$ (no penalty on the initial block). **Step 4** (dual update) is unchanged. Each outer iteration remains $O(np)$, and the ADMM convergence guarantee from §7.6.4 carries over without modification since the joint penalty is still convex.

---

## 9. Two-Step Estimation: Backward Elimination for SBAR-COV

The joint group LASSO (8.2) is designed to over-select candidate changepoints: it returns a set $\hat{A}_n^{\mathrm{jnt}}$ that contains every true break with high probability, but also includes false positives. A second step is required to prune this candidate set to the true break locations.

### 9.1 Segment-wise Profile Likelihood

Given a partition of $\{1,\ldots,n\}$ into $m+1$ segments by break locations $\mathbf{t} = (t_1,\ldots,t_m)$ with $t_0 = p$ and $t_{m+1} = n$, define segment $j$ as observations $t_{j-1}+1,\ldots,t_j$ with $n_j = t_j - t_{j-1}$ observations. Let $\hat{\boldsymbol{\beta}}_j$ denote the OLS estimate within segment $j$ and define the segment residual sum of squares:

$$\mathrm{RSS}_j(\mathbf{t}) = \sum_{t=t_{j-1}+1}^{t_j} \bigl(Y_t - \hat{\boldsymbol{\beta}}_j^T \mathbf{Y}_{t-1}\bigr)^2.$$

Maximizing the Gaussian likelihood over $\sigma_j^2$ within each segment yields the profile MLE $\hat{\sigma}_j^2 = \mathrm{RSS}_j / n_j$. Substituting back and dropping constants, the segment-wise profiled negative log-likelihood is:

$$\mathrm{NLL}_j(\mathbf{t}) = \frac{n_j}{2}\log\hat{\sigma}_j^2 = \frac{n_j}{2}\log\frac{\mathrm{RSS}_j(\mathbf{t})}{n_j}.$$

The total goodness-of-fit for the $m$-break partition $\mathbf{t}$ is:

$$\mathcal{G}_n(\mathbf{t}) = \sum_{j=1}^{m+1} \mathrm{NLL}_j(\mathbf{t}) = \sum_{j=1}^{m+1} \frac{n_j}{2}\log\frac{\mathrm{RSS}_j(\mathbf{t})}{n_j}. \tag{9.1}$$

### 9.2 The Information Criterion

Because the joint penalty (8.1) enforces co-location, every changepoint in $\hat{A}_n^{\mathrm{jnt}}$ is a joint break involving both the AR coefficients and the precision. Each such break introduces $p+1$ new free parameters — $p$ for $\boldsymbol{\beta}_j$ and $1$ for $\sigma_j^2$. This uniform parameter count per break motivates the information criterion:

$$\mathrm{IC}(m, \mathbf{t}) = \mathcal{G}_n(\mathbf{t}) + m\,\omega_n, \qquad \omega_n = \frac{(p+1)\log n}{2}. \tag{9.2}$$

The choice $\omega_n = (p+1)\log(n)/2$ is the BIC penalty: it corresponds to $-2\log\hat{L} + k\log n$ with $k = p+1$ per break and $\mathcal{G}_n = -\log\hat{L}$ (up to constants). Under standard regularity conditions this choice achieves consistent model selection — $\omega_n \to \infty$ ensures false breaks are eventually rejected, and $\omega_n/n \to 0$ ensures true breaks are retained.

### 9.3 The Backward Elimination Algorithm

**Initialization.** Start with the full candidate set $\mathcal{A} = \hat{A}_n^{\mathrm{jnt}} = \{s_1 < \cdots < s_M\}$ from Stage 1, where $M = |\hat{A}_n^{\mathrm{jnt}}|$. Compute $W_M^* = \mathrm{IC}(M,\, \mathcal{A})$.

**Iterative pruning.** At each step with current candidate set $\mathcal{A}$ of size $m$:

1. For each $s_i \in \mathcal{A}$, compute the IC after removing $s_i$:
$$W_{m-1,i} = \mathrm{IC}\!\bigl(m-1,\, \mathcal{A} \setminus \{s_i\}\bigr).$$

2. Identify the most redundant point:
$$i^* = \operatorname*{argmin}_{i}\, W_{m-1,i}, \qquad W_{m-1}^* = W_{m-1,\,i^*}.$$

**Stopping criterion.** If $W_{m-1}^* \leq W_m^*$, remove $s_{i^*}$ from $\mathcal{A}$, set $m \leftarrow m-1$ and $W_m^* \leftarrow W_{m-1}^*$, then repeat. If $W_{m-1}^* > W_m^*$, stop and return $\mathcal{A}$ as the final estimated changepoint set $\hat{\hat{A}}_n$.

### 9.4 Complexity and Practical Considerations

**Computational cost.** Each pruning step evaluates $m$ candidate removals. For each removal, refitting is local: only the two segments adjacent to $s_i$ merge, so $\mathrm{RSS}_j$ for the merged segment must be recomputed. This can be done in $O(p^2 n_j)$ using updating formulas, giving a total cost of $O(M^2 p^2 n / M) = O(M p^2 n)$ over the entire BEA, or $O(p^2 n)$ per elimination step.

**Relationship to Stage 1.** The BEA does not re-estimate the natural parameters — it uses segment-wise OLS to evaluate the IC. This is consistent because the Stage 2 criterion (9.2) is the profiled likelihood, which marginalizes out the precision parameters in closed form. The output $\hat{\hat{A}}_n$ is a subset of $\hat{A}_n^{\mathrm{jnt}}$; the final parameter estimates $(\hat{\boldsymbol{\beta}}_j, \hat{\sigma}_j^2)$ are then obtained by refitting within each selected segment.

**Constraint from co-location.** Because (8.1) forces $(\boldsymbol{\theta}_i, \psi_i)$ to zero simultaneously, $\hat{A}_n^{\mathrm{jnt}}$ contains no coefficient-only or variance-only breaks. The BEA therefore always evaluates merged segments in which both AR structure and variance may change — it never faces the question of whether to retain a break that affects only one component.
