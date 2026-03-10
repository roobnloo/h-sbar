# ADMM for the Joint Group LASSO

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

## ADMM Formulation

Introduce penalty-side copies $\mathbf{z}_1 \in \mathbb{R}^{np}$ and $\mathbf{z}_2 \in \mathbb{R}^n$ to decouple the smooth loss from the non-smooth group penalty:

$$\min_{\boldsymbol{\theta},\,\boldsymbol{\psi},\,\mathbf{z}_1,\,\mathbf{z}_2}\;\frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \lambda\sum_{i=2}^n\|\tilde{\mathbf{z}}_i\|_2 \quad\text{s.t.}\quad \boldsymbol{\theta} = \mathbf{z}_1,\;\boldsymbol{\psi} = \mathbf{z}_2, \tag{3}$$

where $\tilde{\mathbf{z}}_i = (\mathbf{z}_{1,i}^T,\, \sqrt{c}\,z_{2,i})^T$. The scaled augmented Lagrangian with dual variables $\mathbf{u}_1 \in \mathbb{R}^{np}$, $\mathbf{u}_2 \in \mathbb{R}^n$ and penalty $\rho > 0$ is:

$$\mathcal{A}_\rho = \frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \lambda\sum_{i\geq 2}\|\tilde{\mathbf{z}}_i\|_2 + \frac{\rho}{2}\|\boldsymbol{\theta} - \mathbf{z}_1 + \mathbf{u}_1\|^2 + \frac{\rho}{2}\|\boldsymbol{\psi} - \mathbf{z}_2 + \mathbf{u}_2\|^2.$$

---

## ADMM Iterations

Initialize $(\mathbf{z}_1^0, \mathbf{z}_2^0, \mathbf{u}_1^0, \mathbf{u}_2^0)$ with $(\mathbf{L}\mathbf{z}_2^0)_t > 0$ for all $t$. At each iteration $k$:

### Step 1 — Smooth update $(\boldsymbol{\theta}^{k+1}, \boldsymbol{\psi}^{k+1})$

Solve the unconstrained smooth problem:

$$\min_{\boldsymbol{\theta},\,\boldsymbol{\psi}}\;\frac{1}{n}\mathcal{L}_n^{\mathrm{nat}}(\boldsymbol{\theta},\boldsymbol{\psi}) + \frac{\rho}{2}\|\boldsymbol{\theta} - \mathbf{c}_1^k\|^2 + \frac{\rho}{2}\|\boldsymbol{\psi} - \mathbf{c}_2^k\|^2, \tag{4}$$

where $\mathbf{c}_1^k = \mathbf{z}_1^k - \mathbf{u}_1^k$ and $\mathbf{c}_2^k = \mathbf{z}_2^k - \mathbf{u}_2^k$.

This is strictly convex and smooth (the $-\log\Phi_t$ barrier automatically enforces $\Phi_t > 0$). Gradients are:

$$\nabla_{\boldsymbol{\theta}}\mathcal{A} = \frac{1}{n}\mathbf{X}^T\boldsymbol{\delta} + \rho(\boldsymbol{\theta} - \mathbf{c}_1^k), \qquad \delta_t = -Y_t + \frac{G_t}{\Phi_t},$$

$$\nabla_{\boldsymbol{\psi}}\mathcal{A} = \frac{1}{n}\mathbf{L}^T\boldsymbol{\eta} + \rho(\boldsymbol{\psi} - \mathbf{c}_2^k), \qquad \eta_t = -\frac{1}{2\Phi_t} + \frac{Y_t^2}{2} - \frac{G_t^2}{2\Phi_t^2}.$$

Note that $\mathbf{L}^T\boldsymbol{\eta}$ is a reverse cumulative sum: $(\mathbf{L}^T\boldsymbol{\eta})_i = \sum_{t \geq i}\eta_t$. Solve (4) to moderate precision using L-BFGS or Newton-CG; each gradient evaluation costs $O(np)$.

### Step 2 — Joint group soft-threshold $(\mathbf{z}_1^{k+1}, \mathbf{z}_2^{k+1})$

For $i = 1$: no penalty on the initial block, so set

$$(\mathbf{z}_{1,1}^{k+1},\; z_{2,1}^{k+1}) = (\boldsymbol{\theta}_1^{k+1} + \mathbf{u}_{1,1}^k,\; \psi_1^{k+1} + u_{2,1}^k).$$

For $i = 2,\ldots,n$: let $\mathbf{a}_i = \boldsymbol{\theta}_i^{k+1} + \mathbf{u}_{1,i}^k$ and $b_i = \psi_i^{k+1} + u_{2,i}^k$. Apply the **joint group soft-threshold**:

$$(\mathbf{z}_{1,i}^{k+1},\; z_{2,i}^{k+1}) = \mathcal{T}_{\lambda/\rho,\,c}(\mathbf{a}_i, b_i) := \left(1 - \frac{\lambda/\rho}{\sqrt{\|\mathbf{a}_i\|_2^2 + c\,b_i^2}}\right)_{\!\!+}(\mathbf{a}_i,\; b_i). \tag{5}$$

The scalar shrinkage factor is applied identically to both components. A group is zeroed entirely when $\|\mathbf{a}_i\|_2^2 + c\,b_i^2 \leq (\lambda/\rho)^2$ — an ellipsoidal dead-zone with semi-axes $\lambda/\rho$ in the $\boldsymbol{\theta}$ directions and $\lambda/(\rho\sqrt{c})$ in the $\psi$ direction. This step is $O(np)$ in total and has a **closed form**.

### Step 3 — Dual update

$$\mathbf{u}_1^{k+1} = \mathbf{u}_1^k + \boldsymbol{\theta}^{k+1} - \mathbf{z}_1^{k+1}, \qquad \mathbf{u}_2^{k+1} = \mathbf{u}_2^k + \boldsymbol{\psi}^{k+1} - \mathbf{z}_2^{k+1}.$$

---

## Convergence

Since $Q_n^{\mathrm{jnt}}$ is jointly convex and closed and the equality constraints in (3) are linear, standard ADMM convergence theory (Boyd et al., 2011, §3.2) guarantees that the primal residuals $\|\boldsymbol{\theta}^k - \mathbf{z}_1^k\|$ and $\|\boldsymbol{\psi}^k - \mathbf{z}_2^k\|$ and the dual residuals $\|\mathbf{z}_1^k - \mathbf{z}_1^{k-1}\|$ and $\|\mathbf{z}_2^k - \mathbf{z}_2^{k-1}\|$ all converge to zero. The dominant cost per outer iteration is solving (4) to moderate precision (e.g., 5–10 L-BFGS steps), giving an overall per-iteration complexity of $O(np)$.

**Stopping criterion.** Terminate when both the primal and dual residuals fall below tolerances $\epsilon^{\mathrm{pri}}$ and $\epsilon^{\mathrm{dual}}$ (e.g., absolute $10^{-4}$, relative $10^{-3}$, as in Boyd et al., 2011, §3.3.1).
