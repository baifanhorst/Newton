import numpy as np

def Newton(func, jac, x0, XTOL, MAX_ITER):
    """
    Classical Newton's method for solving nonlinear equations.
    
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
    x : ndarray
        Approximate solution after convergence or maximum iterations.
    """
    
    # List of error norms
    norm_err_list = []
    # Iteration count
    iter_list = []
    
    # Initial guess
    x = x0.copy()
    
    # Initial iteration
    iter_list.append(0)
    
    J = jac(x)
    F = func(x)
    dx = np.linalg.solve(J, -F)
    norm_err_old = np.linalg.norm(dx)
    norm_err_list.append(norm_err_old)
    
    if norm_err_old <= XTOL:
        x += dx
        print("Converged at step 0")
        print("Solution:", x)
    else:
        x += dx

        for i in range(1, MAX_ITER):

            iter_list.append(i)
            
            J = jac(x)
            F = func(x)
            dx = np.linalg.solve(J, -F)
            norm_err = np.linalg.norm(dx)
            norm_err_list.append(norm_err)
            if norm_err <= XTOL:
                x += dx
                print(f"Converged at step {i}")
                print("Solution:", x)
                break
            else:
                x += dx
                norm_err_old = norm_err 
    
    return x, iter_list, norm_err_list
