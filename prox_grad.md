# Proximal Gradient Descent for the Joint Group LASSO

## The Optimization Problem

We minimize a jointly convex objective in two vector variables $\boldsymbol{\theta}(n) \in \mathbb{R}^{np}$ and $\boldsymbol{\psi}(n) \in \mathbb{R}^n$:

$$Q_n^{\mathrm{jnt}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) = \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}\bigl(\boldsymbol{\theta}(n), \boldsymbol{\psi}(n)\bigr) + \lambda \sum_{i=2}^{n} \sqrt{\|\boldsymbol{\theta}_i\|_2^2 + c\,\psi_i^2}, \tag{1}$$

where $\lambda > 0$ and $c > 0$ are fixed scalars, and the smooth loss is:

$$\mathcal{L}_n^{\mathrm{nat}} = \frac{1}{2}\sum_{t=1}^{n}\left[-\log \Phi_t + \Phi_t Y_t^2 - 2Y_t G_t + \frac{G_t^2}{\Phi_t}\right]. \tag{2}$$

Here $G_t = (\mathbf{X}\boldsymbol{\theta})_t$ and $\Phi_t = \sum_{i=1}^t \psi_i$ are linear functions of the decision variables (via a fixed design matrix $\mathbf{X} \in \mathbb{R}^{n \times np}$ and the lower-triangular all-ones matrix $\mathbf{L} \in \{0,1\}^{n \times n}$ respectively). The domain requires $\Phi_t > 0$ for all $t$, which is an open convex cone enforced implicitly by the $-\log \Phi_t$ barrier.

The penalty term is a **group LASSO** on the $(p+1)$-dimensional vectors

$$\mathbf{v}_i = \begin{pmatrix}\boldsymbol{\theta}_i \\ \sqrt{c}\,\psi_i\end{pmatrix} \in \mathbb{R}^{p+1}, \qquad i = 2,\ldots,n.$$

So (1) can be written equivalently as $\frac{1}{n}\mathcal{L}_n^{\mathrm{nat}} + \lambda \sum_{i=2}^n \|\mathbf{v}_i\|_2$.

---

## Proximal Gradient Descent Formulation

Proximal gradient descent (PGD) exploits the **composite structure** of (1) by splitting the objective into a smooth part and a non-smooth part:

$$Q_n^{\mathrm{jnt}} = f(\boldsymbol{\theta}, \boldsymbol{\psi}) + g(\boldsymbol{\theta}, \boldsymbol{\psi}),$$

where

$$f(\boldsymbol{\theta}, \boldsymbol{\psi}) := \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta}, \boldsymbol{\psi}), \qquad g(\boldsymbol{\theta}, \boldsymbol{\psi}) := \lambda \sum_{i=2}^n \|\mathbf{v}_i\|_2.$$

The smooth part $f$ is continuously differentiable and convex on the domain $\mathcal{D} = \{\boldsymbol{\psi} : \Phi_t > 0\;\forall\,t\}$; the non-smooth part $g$ is convex and admits a closed-form proximal operator (§7.8 of hsbar.md). Each iteration takes a gradient step on $f$ followed by a proximal step for $g$.

---

## Gradients of the Smooth Part

Stacking the decision variables into $\mathbf{w} = (\boldsymbol{\theta}^T, \boldsymbol{\psi}^T)^T \in \mathbb{R}^{n(p+1)}$, the gradient of $f$ at a feasible point is:

$$\nabla_{\boldsymbol{\theta}} f = \frac{1}{n}\mathbf{X}_n^T \boldsymbol{\delta}, \qquad \nabla_{\boldsymbol{\psi}} f = \frac{1}{n}\mathbf{L}^T \boldsymbol{\eta}, \tag{3}$$

where the residual vectors are:

$$\delta_t = \frac{G_t}{\Phi_t} - Y_t, \qquad \eta_t = -\frac{1}{2\Phi_t} + \frac{Y_t^2}{2} - \frac{G_t^2}{2\Phi_t^2}. \tag{4}$$

Both expressions require only $O(np)$ arithmetic: $\boldsymbol{\delta}$ and $\boldsymbol{\eta}$ are $O(n)$ given $(G_t, \Phi_t)$, and the matrix–vector products $\mathbf{X}_n^T \boldsymbol{\delta}$ and $\mathbf{L}^T\boldsymbol{\eta}$ are each $O(np)$ owing to the lower-triangular structure. Specifically, $(\mathbf{L}^T\boldsymbol{\eta})_i = \sum_{t \geq i} \eta_t$ is a reverse cumulative sum, computable in $O(n)$.

---

## The Hessian and Lipschitz Constant

The Hessian of $f$ at $(\boldsymbol{\theta}, \boldsymbol{\psi})$ is the $n(p+1) \times n(p+1)$ block matrix:

$$H_f = \frac{1}{n}\begin{pmatrix} \mathbf{X}_n^T \mathbf{D}_{\Phi}^{-1} \mathbf{X}_n & -\mathbf{X}_n^T \mathbf{D}_{\Phi}^{-2} \mathbf{D}_G \mathbf{L} \\ -\mathbf{L}^T \mathbf{D}_G \mathbf{D}_{\Phi}^{-2} \mathbf{X}_n & \mathbf{L}^T \mathbf{D}_h \mathbf{L} \end{pmatrix}, \tag{5}$$

where $\mathbf{D}_{\Phi} = \mathrm{diag}(\Phi_1,\ldots,\Phi_n)$, $\mathbf{D}_G = \mathrm{diag}(G_1,\ldots,G_n)$, and

$$h_t = \frac{1}{2\Phi_t^2} + \frac{G_t^2}{\Phi_t^3} > 0.$$

The block-diagonal entries $\mathbf{X}_n^T\mathbf{D}_\Phi^{-1}\mathbf{X}_n \succeq 0$ and $\mathbf{L}^T\mathbf{D}_h\mathbf{L} \succ 0$ confirm $H_f \succeq 0$, consistent with joint convexity of $f$.

**Lipschitz continuity.** The gradient $\nabla f$ is not globally Lipschitz on $\mathcal{D}$: as $\Phi_t \to 0^+$, the diagonal entries of $\mathbf{D}_h$ diverge. However, on any sublevel set $\{Q_n^{\mathrm{jnt}} \leq q\}$ the precision vector is bounded away from zero (since $-\log\Phi_t \leq 2qn$ implies $\Phi_t \geq e^{-2qn}$), and the local Lipschitz constant satisfies:

$$L_f \leq \lambda_{\max}(H_f) \leq \frac{1}{n}\left\|\begin{pmatrix}\mathbf{X}_n \\ \mathbf{L}\end{pmatrix}\right\|_2^2 \cdot \max_t h_t. \tag{6}$$

Since $L_f$ is unknown and state-dependent, a **backtracking line search** selects a valid step size at each iteration.

---

## The Proximal Operator

The proximal operator of $g$ at a point $(\mathbf{a}, \mathbf{b}) \in \mathbb{R}^{np} \times \mathbb{R}^n$ decouples over positions $i$ (non-overlapping groups). For $i \geq 2$, the joint group soft-threshold (hsbar.md §7.8) is:

$$\mathrm{prox}_{\alpha g}(\mathbf{a}, \mathbf{b})_i = \mathcal{T}_{\alpha\lambda, c}(\mathbf{a}_i, b_i) := \left(1 - \frac{\alpha\lambda}{\sqrt{\|\mathbf{a}_i\|_2^2 + c\,b_i^2}}\right)_{\!\!+} (\mathbf{a}_i,\, b_i). \tag{7}$$

The initial block $i = 1$ is unpenalized: $\mathrm{prox}_{\alpha g}(\mathbf{a}, \mathbf{b})_1 = (\mathbf{a}_1, b_1)$. Total cost is $O(np)$.

**Feasibility after the proximal step.** The proximal step (7) shrinks each group toward zero but does not directly enforce $\Phi_t = \sum_{i=1}^t \psi_i > 0$. Feasibility is maintained by requiring the gradient-step iterate to land in $\mathcal{D}$ before applying the proximal operator — guaranteed by a sufficiently small step size enforced by backtracking.

---

## The ISTA Algorithm with Backtracking

Starting from a feasible point $(\boldsymbol{\theta}^0, \boldsymbol{\psi}^0)$ with $\Phi_t^0 > 0$ for all $t$, each iteration $k \geq 0$ proceeds as follows.

**Step 1 — Gradient step.** Compute the tentative gradient-step iterate:

$$\tilde{\boldsymbol{\theta}} = \boldsymbol{\theta}^k - \alpha \nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}^k, \boldsymbol{\psi}^k), \qquad \tilde{\boldsymbol{\psi}} = \boldsymbol{\psi}^k - \alpha \nabla_{\boldsymbol{\psi}} f(\boldsymbol{\theta}^k, \boldsymbol{\psi}^k),$$

using the gradients (3)–(4). The step size $\alpha > 0$ is chosen by backtracking (see below).

**Step 2 — Proximal step.** Apply the joint group soft-threshold:

$$\bigl(\boldsymbol{\theta}^{k+1}_i, \psi^{k+1}_i\bigr) = \begin{cases} \mathcal{T}_{\alpha\lambda,\,c}(\tilde{\boldsymbol{\theta}}_i, \tilde{\psi}_i) & i \geq 2, \\ (\tilde{\boldsymbol{\theta}}_1, \tilde{\psi}_1) & i = 1. \end{cases} \tag{8}$$

**Backtracking line search.** Fix $\beta \in (0,1)$ (e.g., $\beta = 0.5$) and an initial trial step $\alpha_0$. At each iteration, start with $\alpha \leftarrow \alpha_0$ and halve $\alpha \leftarrow \beta\alpha$ until both conditions hold:

1. **Feasibility:** $\tilde{\Phi}_t > 0$ and $\Phi_t^{k+1} > 0$ for all $t$.
2. **Sufficient decrease (Armijo on $f$):**
$$f(\boldsymbol{\theta}^{k+1}, \boldsymbol{\psi}^{k+1}) \leq f(\boldsymbol{\theta}^k, \boldsymbol{\psi}^k) + \langle \nabla f^k,\, \mathbf{w}^{k+1} - \mathbf{w}^k \rangle + \frac{1}{2\alpha}\|\mathbf{w}^{k+1} - \mathbf{w}^k\|_2^2. \tag{9}$$

Condition (9) is the standard proximal gradient sufficient decrease condition, which guarantees that each iteration decreases $Q_n^{\mathrm{jnt}}$ (Parikh & Boyd, 2014, §4.2).

---

## The FISTA Algorithm (Accelerated)

The accelerated variant (Beck & Teboulle, 2009) adds a momentum sequence that achieves the optimal $O(1/k^2)$ convergence rate for convex composites. Introduce a momentum variable $s^k$ and an extrapolated point $\mathbf{m}^k$. Initialize $\mathbf{w}^0 = (\boldsymbol{\theta}^0, \boldsymbol{\psi}^0)$, $\mathbf{m}^0 = \mathbf{w}^0$, $s^0 = 1$.

**FISTA iteration** $k \geq 0$:

1. **Proximal gradient step at $\mathbf{m}^k$:**
$$\mathbf{w}^{k+1} = \mathrm{prox}_{\alpha g}\!\bigl(\mathbf{m}^k - \alpha\,\nabla f(\mathbf{m}^k)\bigr), \tag{10}$$
   with backtracking on $\alpha$ as above (using $\mathbf{m}^k$ in place of $\mathbf{w}^k$).

2. **Momentum update:**
$$s^{k+1} = \frac{1 + \sqrt{1 + 4(s^k)^2}}{2}. \tag{11}$$

3. **Extrapolation:**
$$\mathbf{m}^{k+1} = \mathbf{w}^{k+1} + \frac{s^k - 1}{s^{k+1}}\bigl(\mathbf{w}^{k+1} - \mathbf{w}^k\bigr). \tag{12}$$

The extrapolation coefficient $(s^k-1)/s^{k+1}$ starts near $0$ and converges to $1$ as $k\to\infty$. Note that $\mathbf{m}^{k+1}$ may not be feasible even when $\mathbf{w}^{k+1}$ is, so the backtracking at step $k+1$ must re-establish feasibility from $\mathbf{m}^{k+1}$.

**Gradient-based restart.** When the iterates approach a sparse solution, FISTA's momentum can overshoot. A restart strategy (O'Donoghue & Candès, 2015) resets $s^k \leftarrow 1$ and $\mathbf{m}^k \leftarrow \mathbf{w}^k$ whenever $\langle \nabla f(\mathbf{m}^k),\, \mathbf{w}^{k+1} - \mathbf{w}^k\rangle > 0$, ensuring monotone decrease in the objective and stability near the solution.

---

## Convergence Analysis

Let $Q^* = \min Q_n^{\mathrm{jnt}}$ denote the global minimum, attained since the objective is coercive on $\mathcal{D}$.

**ISTA convergence.** If $\nabla f$ is $L_f$-Lipschitz on a sublevel set containing the trajectory, with fixed step $\alpha = 1/L_f$:

$$Q_n^{\mathrm{jnt}}(\mathbf{w}^k) - Q^* \leq \frac{L_f \|\mathbf{w}^0 - \mathbf{w}^*\|_2^2}{2k}. \tag{13}$$

With backtracking the same bound holds with $L_f$ replaced by the largest step-size-inverse encountered.

**FISTA convergence.** Under the same conditions:

$$Q_n^{\mathrm{jnt}}(\mathbf{w}^k) - Q^* \leq \frac{2L_f \|\mathbf{w}^0 - \mathbf{w}^*\|_2^2}{(k+1)^2}. \tag{14}$$

Both rates are optimal for first-order methods applied to the composite convex class. Since $Q_n^{\mathrm{jnt}}$ is strictly convex when the design matrix has full column rank in the relevant subspace, the minimizer $\mathbf{w}^*$ is unique and iterate convergence $\mathbf{w}^k \to \mathbf{w}^*$ holds.

**Exact sparsity in finite steps.** Unlike ADMM, each PGD iterate $(\boldsymbol{\theta}_i^k, \psi_i^k)$ is exactly zero whenever the group norm of the gradient-step point satisfies $\sqrt{\|\tilde{\boldsymbol{\theta}}_i\|_2^2 + c\,\tilde{\psi}_i^2} \leq \alpha\lambda$ — the ellipsoidal dead-zone of (7). The support of $\mathbf{w}^k$ can decrease monotonically during ISTA (though not necessarily during FISTA due to extrapolation).

---

## Comparison with ADMM

| Property | PGD / FISTA | ADMM |
|---|---|---|
| Per-iteration cost | $O(np)$ | $O(np)$ (with L-BFGS inner loop) |
| Convergence rate | $O(1/k)$ ISTA, $O(1/k^2)$ FISTA | $O(1/k)$ (primal/dual residuals) |
| Tuning parameters | Step size $\alpha$ (or backtracking) | Penalty $\rho$ |
| Exact sparsity per iterate | Yes (ISTA) | Only at convergence |
| Warm starting across $\lambda$ | Straightforward | Requires re-initializing dual variables |
| Implementation complexity | Lower | Higher (requires inner solver) |

FISTA is preferable when exact sparsity in iterates is valuable (e.g., active-set identification, early stopping) and when solving for a single $\lambda$. ADMM is preferable for a fine $\lambda$-path because dual variables warm-start effectively across $\lambda$ values, amortizing the per-solve cost.

---

## Stopping Criterion

A practical stopping rule uses the **proximal gradient mapping** at step size $\alpha$:

$$G_\alpha(\mathbf{w}) := \frac{1}{\alpha}\bigl(\mathbf{w} - \mathrm{prox}_{\alpha g}(\mathbf{w} - \alpha\,\nabla f(\mathbf{w}))\bigr).$$

At the optimum $\mathbf{w}^*$ one has $G_\alpha(\mathbf{w}^*) = \mathbf{0}$ (first-order optimality for composite problems). Stop when:

$$\|G_\alpha(\mathbf{w}^k)\|_2 \leq \epsilon_{\mathrm{abs}} + \epsilon_{\mathrm{rel}}\,\|\mathbf{w}^k\|_2,$$

with tolerances $\epsilon_{\mathrm{abs}} = 10^{-6}$ and $\epsilon_{\mathrm{rel}} = 10^{-4}$ recommended in practice.
