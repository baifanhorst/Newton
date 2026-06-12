# Design of a Test Example for Deuflhard's Local Newton Method

This document designs a system of two nonlinear equations to test a local Newton method implementation based on the affine-invariant, error-oriented convergence theory presented in Peter Deuflhard's textbook: *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*.

---

## 1. The Test System: Circle-Parabola Intersection

A simple, mathematically clean, and geometrically intuitive test system for a two-dimensional nonlinear solver is the intersection of a circle and a parabola.

### Mathematical Formulation
Let $x = (u, v)^T \in \mathbb{R}^2$. The system $F(x) = 0$ is defined as:
$$
F(u, v) = \begin{pmatrix} f_1(u, v) \\ f_2(u, v) \end{pmatrix} = \begin{pmatrix} u^2 + v^2 - 4 \\ v - u^2 + 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

### Exact Roots (Analytical Solution)
Substituting $u^2 = v + 1$ into $f_1(u, v) = 0$ yields:
$$
(v + 1) + v^2 - 4 = 0 \implies v^2 + v - 3 = 0
$$
Solving for $v$ (since we require $u^2 \ge 0$, we need $v \ge -1$):
$$
v^* = \frac{-1 + \sqrt{13}}{2} \approx 1.30277564
$$
Then, solving for $u$:
$$
u^* = \pm \sqrt{v^* + 1} = \pm \sqrt{\frac{1 + \sqrt{13}}{2}} \approx \pm 1.51754447
$$
Thus, there are exactly two real roots:
$$
x_+^* \approx \begin{pmatrix} 1.51754447 \\ 1.30277564 \end{pmatrix}, \quad x_-^* \approx \begin{pmatrix} -1.51754447 \\ 1.30277564 \end{pmatrix}
$$

### The Jacobian Matrix
The Jacobian $J(x) = F'(x)$ is given by:
$$
J(u, v) = \begin{pmatrix} \frac{\partial f_1}{\partial u} & \frac{\partial f_1}{\partial v} \\ \frac{\partial f_2}{\partial u} & \frac{\partial f_2}{\partial v} \end{pmatrix} = \begin{pmatrix} 2u & 2v \\ -2u & 1 \end{pmatrix}
$$
The determinant of the Jacobian is:
$$
\det(J(u, v)) = 2u(1 + 2v)
$$
The Jacobian is non-singular everywhere except on the lines $u = 0$ and $v = -1/2$. At both solutions $x_\pm^*$, the determinant is:
$$
\det(J(x_\pm^*)) = \pm 2(1.51754447)(1 + 2(1.30277564)) \approx \pm 10.94 \neq 0
$$
This guarantees that local Newton convergence will be quadratic and well-behaved when starting sufficiently close to either root.

---

## 2. Deuflhard's Local Error-Oriented Newton Method

In Deuflhard's framework, local error-oriented Newton methods focus on checking convergence in the **domain space** (the iterates) rather than the residual space, using affine-invariant metrics.

### Algorithmic Steps
Given an initial guess $x^0$, an error tolerance $\text{XTOL}$, and a maximum iteration count $k_{\max}$:

1. For $k = 0, 1, \dots, k_{\max}$:
   - Evaluate $F(x^k)$ and the Jacobian $J(x^k)$.
   - Solve the linear system for the Newton step $\Delta x^k$:
     $$J(x^k) \Delta x^k = -F(x^k)$$
   - Update the iterate:
     $$x^{k+1} = x^k + \Delta x^k$$
   - For $k \ge 1$, estimate the contractivity factor $\Theta_k$:
     $$
     \Theta_k = \frac{\|\Delta x^k\|}{\|\Delta x^{k-1}\|}
     $$
     *(where $\|\cdot\|$ is a chosen norm, such as the standard $\ell_2$-norm).*
   - For $k \ge 1$, perform the affine-invariant termination check:
     - If $\Theta_k < 1.0$:
       - Estimate the actual error of the updated iterate $x^{k+1}$ using:
         $$
         \text{err}_{k+1} = \frac{\Theta_k}{1 - \Theta_k} \|\Delta x^k\|
         $$
       - If $\text{err}_{k+1} \le \text{XTOL}$, terminate iteration and return $x^{k+1}$ as the solution.
     - If $\Theta_k \ge 1.0$, it indicates that the iteration is not yet in the local contraction region. Warn or handle non-convergence if this persists.
   - For $k = 0$, since $\text{err}_1$ cannot be estimated via $\text{Theta}_0$, we can either:
     - Directly accept $x^1$ if $\|\Delta x^0\| \le \text{XTOL}$ (extreme proximity).
     - Or simply proceed to $k=1$ to compute $\Theta_1$.

---

## 3. Python Implementation Example

Below is a Python implementation of the designed test system and Deuflhard's local Newton solver.

```python
import numpy as np

def F(x):
    """System of equations F(x) = 0."""
    u, v = x[0], x[1]
    return np.array([
        u**2 + v**2 - 4.0,
        v - u**2 + 1.0
    ], dtype=float)

def J(x):
    """Jacobian matrix J(x) = F'(x)."""
    u, v = x[0], x[1]
    return np.array([
        [2*u, 2*v],
        [-2*u, 1.0]
    ], dtype=float)

def deuflhard_local_newton(x0, xtol=1e-10, max_iter=20):
    """
    Solves F(x) = 0 using Deuflhard's local error-oriented Newton method.
    """
    x = np.array(x0, dtype=float)
    dx_prev = None
    
    print(f"Starting Deuflhard local Newton solver...")
    print(f"Initial guess x0 = {x}\n")
    print(f"{'k':<4} | {'x^k':<32} | {'||dx^k||':<12} | {'Theta_k':<10} | {'Estimated Error':<16} | {'||F(x^k)||':<12}")
    print("-" * 100)
    
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        norm_Fx = np.linalg.norm(Fx)
        
        # Solve for the Newton step
        try:
            dx = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            print(f"Error: Jacobian is singular at x = {x}")
            return None, False
            
        norm_dx = np.linalg.norm(dx)
        
        # Estimate contraction factor and check convergence
        theta = None
        est_error = None
        
        if k > 0:
            theta = norm_dx / norm_dx_prev
            if theta < 1.0:
                est_error = (theta / (1.0 - theta)) * norm_dx
            else:
                est_error = float('inf') # No contractivity
        
        # Format values for printing
        x_str = f"[{x[0]:.8f}, {x[1]:.8f}]"
        theta_str = f"{theta:.4f}" if theta is not None else "N/A"
        est_err_str = f"{est_error:.2e}" if est_error is not None else "N/A"
        
        print(f"{k:<4} | {x_str:<32} | {norm_dx:<12.2e} | {theta_str:<10} | {est_err_str:<16} | {norm_Fx:<12.2e}")
        
        # Perform the update
        x_next = x + dx
        
        # Convergence Check (Deuflhard's criterion)
        if est_error is not None and est_error <= xtol:
            print("-" * 100)
            print(f"Converged at step k={k} (x^{k+1} error estimate: {est_error:.2e} <= xtol)")
            return x_next, True
            
        # Fallback for extremely close starting points
        if k == 0 and norm_dx <= xtol:
            print("-" * 100)
            print(f"Converged at step k=0 (initial step norm: {norm_dx:.2e} <= xtol)")
            return x_next, True
            
        # Prepare for next iteration
        x = x_next
        dx_prev = dx
        norm_dx_prev = norm_dx
        
    print("-" * 100)
    print("Warning: Solver did not converge within the maximum number of iterations.")
    return x, False

# --- Run the Test Cases ---
if __name__ == "__main__":
    # Test 1: Converge to the positive root
    sol_pos, success_pos = deuflhard_local_newton(x0=[1.5, 1.0], xtol=1e-12)
    print(f"Solution x_pos*: {sol_pos}\n")
    
    # Test 2: Converge to the negative root
    sol_neg, success_neg = deuflhard_local_newton(x0=[-1.5, 1.0], xtol=1e-12)
    print(f"Solution x_neg*: {sol_neg}\n")
```
