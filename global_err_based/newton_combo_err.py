import numpy as np
import newton_global_err as ng
import newton_local_deuflhard as nl

def solve(func, jac, x0, 
          XTOL = 1e-6, MAX_NR_ITER_GLOBAL = 50, 
          step_0 = 1.0, step_min = 1e-4, MAX_LS_ITER = 10,
          MAX_NR_ITER_LOCAL = 10
    ):
    """
    Global error-based Newton's method for solving nonlinear equations.
    Parameters:
    func : function
        The function representing the system of equations.
    jac : function
        The Jacobian of the function.
    x0 : array_like
        Initial guess for the solution.
    XTOL : float, optional
        Tolerance for convergence based on the norm of the error (default is 1e-6).
    MAX_NR_ITER_GLOBAL : int, optional
        Maximum number of Newton-Raphson iterations to perform in the global phase (default is 50).
    step_0 : float, optional
        Initial step size for line search (default is 1.0). 
    step_min : float, optional
        Minimum step size for line search (default is 1e-4).
    MAX_LS_ITER : int, optional
        Maximum number of line search iterations to perform (default is 10).
    MAX_NR_ITER_LOCAL : int, optional
        Maximum number of Newton-Raphson iterations to perform in the local phase (default is 10).
    Returns:
    success : bool
        True if the method converged, False otherwise.
    x : ndarray
        Approximate solution after convergence or maximum iterations.
    """

    success, switch_to_local, x = ng.solve(
        func, jac, x0, XTOL, MAX_NR_ITER_GLOBAL, step_0, step_min, MAX_LS_ITER
        )
    
    if switch_to_local:
        success, x = nl.solve(func, jac, x, XTOL, MAX_NR_ITER_LOCAL)
        return success, x
    else:
        return success, x