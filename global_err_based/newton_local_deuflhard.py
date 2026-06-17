import numpy as np
def solve(func, jac, x0, XTOL, MAX_ITER):
    """
    Deuflhard's method for solving nonlinear equations.
    
    Parameters:
    func : function
        The function representing the system of equations.
    jac : function
        The Jacobian of the function.
    x0 : array_like
        Initial guess for the solution.
    XTOL : float
        Tolerance for convergence based on the norm of the error.
    MAX_ITER : int
        Maximum number of iterations to perform.
    
    Returns:
    success : bool
        True if the method converged, False otherwise.
    x : ndarray
        Approximate solution after convergence or maximum iterations.
    """
    
    
    x = x0.copy()
    J = jac(x)
    F = func(x)
    dx = np.linalg.solve(J, -F)
    norm_err_old = np.linalg.norm(dx)
    
    if norm_err_old <= XTOL:
        x += dx
        print("Converged at step 0")
        return True, x
    else:
        x += dx

        for i in range(1, MAX_ITER):
            J = jac(x)
            F = func(x)
            dx = np.linalg.solve(J, -F)
            norm_err = np.linalg.norm(dx)
            theta = norm_err / norm_err_old
            if theta >= 1.0:
                print("Convergence failure")
                return False, x
            if norm_err / (1 - theta) <= XTOL:
                print(f"Converged at step {i}")
                return True, x
            else:
                x += dx
                norm_err_old = norm_err 
    
    return False, x
