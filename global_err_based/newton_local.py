import numpy as np
def solve(func, jac, x0, RTOL, MAX_ITER):
    """
    Classical Newton's method for solving nonlinear equations.
    
    Parameters:
    func : function
        The function representing the system of equations.
    jac : function
        The Jacobian of the function.
    x0 : array_like
        Initial guess for the solution.
    RTOL : float
        Tolerance for convergence based on the norm of the residue.
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
    norm_res = np.linalg.norm(F)
    
    if norm_res <= RTOL:
        print("Converged at step 0")
        return True, x
    else:
        dx = np.linalg.solve(J, -F)
        x += dx

        for i in range(1, MAX_ITER):
            
            J = jac(x)
            F = func(x)
            norm_res = np.linalg.norm(F)
            if norm_res <= RTOL:
                print(f"Converged at step {i}")
                return True, x
            
            dx = np.linalg.solve(J, -F)
            x += dx
    
    return False, x
