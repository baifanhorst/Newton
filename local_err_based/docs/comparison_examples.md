# Superiority of Deuflhard's Local Newton Method: Redesigned Examples

This document presents two redesigned test systems to evaluate and compare the local Newton method implementations in `alg_CL.py` (which uses a standard step-norm check $\|\Delta x^k\| \le \text{XTOL}$) and the intended Deuflhard method (which uses the contractivity factor $\Theta_k = \frac{\|\Delta x^k\|}{\|\Delta x^{k-1}\|}$ to check $\frac{\|\Delta x^k\|}{1 - \Theta_k} \le \text{XTOL}$ and detect divergence).

---

## Example 1: Slow Convergence (Multiple Root / Near-Singular Jacobian)

When a system converges slowly (such as near a multiple root or where the Jacobian is ill-conditioned), the contraction factor $\Theta_k$ is close to $1$. In these cases, the actual error $\|x^k - x^*\|$ can be much larger than the step size $\|\Delta x^k\|$. 

### Mathematical Formulation
Consider the system:
$$
F(u, v) = \begin{pmatrix} u^{10} \\ v \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$
The exact root is $x^* = (0, 0)^T$. The Jacobian is:
$$
J(u, v) = \begin{pmatrix} 10 u^9 & 0 \\ 0 & 1 \end{pmatrix}
$$
The Newton step is:
$$
\Delta x^k = \begin{pmatrix} -u_k / 10 \\ -v_k \end{pmatrix}
$$
This gives $u_{k+1} = 0.9 u_k$ and $v_{k+1} = 0$. After the first step, the step norm is:
$$
\|\Delta x^k\| = 0.1 |u_k|
$$
Thus, the contraction rate is constant:
$$
\Theta_k = \frac{\|\Delta x^k\|}{\|\Delta x^{k-1}\|} = \frac{0.1 |u_k|}{0.1 |u_{k-1}|} = 0.9
$$

### Comparison of Termination Criteria
If we start at $x^0 = (1.0, 1.0)^T$ with a tolerance $\text{XTOL} = 10^{-4}$:

1.  **Classical Step-Norm Check (`alg_CL`):**
    Stops when:
    $$
    \|\Delta x^k\| = 0.1 |u_k| \le 10^{-4} \implies |u_k| \le 10^{-3}
    $$
    At termination, the actual error is $\|x^k - x^*\|_2 = |u_k| \approx 10^{-3}$.
    **The classical method terminates prematurely, leaving an error 10 times larger than the requested tolerance.**

2.  **Deuflhard's Local Error Check (`alg_DH`):**
    Estimates the true coordinate error using:
    $$
    \text{err}_{k+1} = \frac{\Theta_k}{1 - \Theta_k} \|\Delta x^k\| = \frac{0.9}{1 - 0.9} \|\Delta x^k\| = 9 \|\Delta x^k\|
    $$
    Stops when:
    $$
    9 \|\Delta x^k\| = 0.9 |u_k| \le 10^{-4} \implies |u_k| \le 1.11 \times 10^{-4}
    $$
    At termination, the actual error is $\|x^k - x^*\|_2 = |u_k| \approx 1.11 \times 10^{-4}$, which **satisfies the requested tolerance**.

---

## Example 2: Cycle / Divergence Detection ($\Theta \ge 1$)

If the initial guess is outside the local contraction region, the iteration can cycle or diverge. A classical local method will blindly run up to `MAX_ITER`. Deuflhard's method monitors $\Theta_k$ and aborts immediately if contractivity is lost.

### Mathematical Formulation
Consider the system:
$$
F(u, v) = \begin{pmatrix} u^3 - 2u + 2 \\ v \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$
The Jacobian is:
$$
J(u, v) = \begin{pmatrix} 3u^2 - 2 & 0 \\ 0 & 1 \end{pmatrix}
$$
If we start at $x^0 = (0, 0)^T$:
*   **$k=0$:** $F(x^0) = (2, 0)^T$, $J(x^0) = \begin{pmatrix} -2 & 0 \\ 0 & 1 \end{pmatrix}$
    $$
    \Delta x^0 = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \implies x^1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \|\Delta x^0\| = 1.0
    $$
*   **$k=1$:** $F(x^1) = (1, 0)^T$, $J(x^1) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$
    $$
    \Delta x^1 = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \implies x^2 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad \|\Delta x^1\| = 1.0
    $$
*   **Contraction factor at $k=1$:**
    $$
    \Theta_1 = \frac{\|\Delta x^1\|}{\|\Delta x^0\|} = \frac{1.0}{1.0} = 1.0
    $$

### Comparison of Behaviors
1.  **Classical Solver (`alg_CL.py`):**
    Will run for the full `MAX_ITER = 20` steps, cycling between $x = (0, 0)^T$ and $x = (1, 0)^T$ indefinitely without detecting that the iteration is failing to converge.
2.  **Deuflhard Solver (as in `zzz_not_used.txt`):**
    Detects $\Theta_1 = 1.0 \ge 1.0$ at iteration 1 and **immediately aborts** with:
    `Iteration 1: No convergence, Theta = 1.00`
    This saves 19 iterations of redundant computation and alerts the user immediately.

---

## 3. Python Verification Script

Below is a script that implements both test cases to compare the behaviors.

```python
import numpy as np

# --- Example 1: Slow Convergence System ---
def F_slow(x):
    return np.array([x[0]**10, x[1]])

def J_slow(x):
    return np.array([
        [10 * x[0]**9, 0.0],
        [0.0, 1.0]
    ])

# --- Example 2: Cycling System ---
def F_cycle(x):
    return np.array([x[0]**3 - 2*x[0] + 2, x[1]])

def J_cycle(x):
    return np.array([
        [3 * x[0]**2 - 2.0, 0.0],
        [0.0, 1.0]
    ])

# --- Algorithms ---
def run_classical_newton(func, jac, x0, xtol=1e-5, max_iter=20):
    x = np.array(x0, dtype=float)
    print("Running Classical Newton:")
    for k in range(max_iter):
        Fx = func(x)
        Jx = jac(x)
        dx = np.linalg.solve(Jx, -Fx)
        norm_dx = np.linalg.norm(dx)
        x = x + dx
        print(f"  Iter {k}: x = {x}, ||dx|| = {norm_dx:.2e}")
        if norm_dx <= xtol:
            print(f"  Converged at step {k}. Final solution: {x}")
            print(f"  True Error ||x - x*||: {np.linalg.norm(x):.2e}\n")
            return x
    print("  Did not converge.\n")
    return x

def run_deuflhard_newton(func, jac, x0, xtol=1e-5, max_iter=20):
    x = np.array(x0, dtype=float)
    dx_prev = None
    print("Running Deuflhard Newton:")
    for k in range(max_iter):
        Fx = func(x)
        Jx = jac(x)
        dx = np.linalg.solve(Jx, -Fx)
        norm_dx = np.linalg.norm(dx)
        
        theta = None
        est_error = None
        if k > 0:
            theta = norm_dx / norm_dx_prev
            if theta >= 1.0:
                print(f"  Iteration {k}: No convergence, Theta = {theta:.2f} >= 1.0")
                return x
            est_error = (theta / (1.0 - theta)) * norm_dx
            
        x = x + dx
        theta_str = f"{theta:.4f}" if theta is not None else "N/A"
        est_err_str = f"{est_error:.2e}" if est_error is not None else "N/A"
        print(f"  Iter {k}: x = {x}, ||dx|| = {norm_dx:.2e}, Theta = {theta_str}, Est.Error = {est_err_str}")
        
        if est_error is not None and est_error <= xtol:
            print(f"  Converged at step {k}. Final solution: {x}")
            print(f"  True Error ||x - x*||: {np.linalg.norm(x):.2e}\n")
            return x
        if k == 0 and norm_dx <= xtol:
            print(f"  Converged at step 0. Final solution: {x}\n")
            return x
            
        dx_prev = norm_dx
        
    print("  Did not converge.\n")
    return x

if __name__ == "__main__":
    # Test 1: Slow convergence
    print("=== TEST 1: SLOW CONVERGENCE ===")
    run_classical_newton(F_slow, J_slow, x0=[1.0, 1.0], xtol=1e-5)
    run_deuflhard_newton(F_slow, J_slow, x0=[1.0, 1.0], xtol=1e-5)
    
    # Test 2: Cycling / Divergence
    print("=== TEST 2: CYCLING / DIVERGENCE ===")
    run_classical_newton(F_cycle, J_cycle, x0=[0.0, 0.0], xtol=1e-5)
    run_deuflhard_newton(F_cycle, J_cycle, x0=[0.0, 0.0], xtol=1e-5)
