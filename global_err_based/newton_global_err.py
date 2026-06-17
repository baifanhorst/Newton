import numpy as np

def solve(func, jac, x0, 
          XTOL, MAX_NR_ITER, 
          step_0 = 1.0, step_min = 1e-4, MAX_LS_ITER = 10):
    """Global error-based Newton's method with line search for solving nonlinear equations.
    Parameters:
    func : function
        The function representing the system of equations.
    jac : function
        The Jacobian of the function.
    x0 : array_like
        Initial guess for the solution.
    XTOL : float
        Tolerance for convergence based on the norm of the error.
    MAX_NR_ITER : int
        Maximum number of Newton-Raphson iterations to perform.
    step_0 : float, optional
        Initial step size for line search (default is 1.0).
    step_min : float, optional
        Minimum step size for line search (default is 1e-4).
    MAX_LS_ITER : int, optional
        Maximum number of line search iterations to perform (default is 10).
    Returns:
    success : bool
        True if the method converged, False otherwise.
    switch_to_local : bool
        True if the method should switch to a local method, False otherwise.
    x : ndarray
        Approximate solution after convergence or maximum iterations.
    """


    x = x0.copy()

    # Variables to store data to avoid repeated calculations
    # The math notation follows each variable, which is used in the latex notes.
    norm_dx = 0.0 # ||dx^k||
    norm_dx_prev = 0.0 # ||dx^{k-1}||
    norm_dx_ls = 0.0 # ||dx^{k,i}||
    dx_ls = np.zeros_like(x) # dx^{k,i}
    step = step_0 # \lambda_k^i


    # The main NR loop
    for k in range(MAX_NR_ITER):
        print("Iteration:", k)
        F = func(x)
        J = jac(x)
        dx = np.linalg.solve(J, -F)
        norm_dx = np.linalg.norm(dx)
        if norm_dx < XTOL:
            print(f"Converged in {k+1} iterations.")
            return True, False, x

        ###################################
        # Line search
        ###################################
        if k==0:
            step = step_0
        else:
            mu = norm_dx_prev * norm_dx_ls / np.linalg.norm(dx_ls - dx) / norm_dx * step
            step = np.min([1.0, mu])
            if step < step_min:
                print("Line search step too small. Stopping.")
                return False, False, x

        step_found = False
        for i in range(MAX_LS_ITER):
            print("Line search: ", i, "Step size:", step)
            x_ls = x + step * dx
            F = func(x_ls)
            dx_ls = np.linalg.solve(J, -F)
            norm_dx_ls = np.linalg.norm(dx_ls)
            Theta = norm_dx_ls / norm_dx
            mu = 0.5 * step * step * norm_dx / np.linalg.norm(dx_ls - (1 - step) * dx)

            if Theta > 1 - 0.5 * step:
                step = np.min([step * 0.5, mu])
                if step < step_min:
                    print("Line search step too small. Stopping.")
                    return False, False, x
                else:
                    continue
            else:
                step_temp = np.min([1.0, mu])
                if (step == 1.0) and (step_temp == 1.0):
                    if norm_dx_ls < XTOL:
                        x = x_ls.copy()
                        print(f"Converged in {k+1} iterations.")
                        return True, False, x
                    elif Theta < 0.5:
                        # Exit with the current x_ls, which will be the initial guess of a local method
                        print("Switching to local method.")
                        return False, True, x_ls
                
                if step_temp >= 4 * step:
                    step = step_temp
                    continue
                
                step_found = True
                x = x_ls.copy()
                break

        if not step_found:
            print(f"Line search failed after {MAX_LS_ITER} iterations. Stopping.")
            return False, False, x  
        
        ### End of Line Search ###

        # Save the norm of dx for the next NR iteration
        norm_dx_prev = norm_dx


    return False, False, x