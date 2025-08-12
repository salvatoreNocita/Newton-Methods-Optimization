import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from Modified_Newton_Method.SolverInstruments import Solvers

class CheckConditions(object):
    """ This class includes all conditions that must be checked for convergence and during algorithms """

    def __init__(self):
        self.solvers = Solvers()

    def StoppingCriterion_notmet(self,xk: np.array, gradf: np.array, tolgrad:float, k: int, k_max: int)-> bool:
        return k < k_max and np.linalg.norm(gradf)> tolgrad

    def H_is_positive_definite(self,hessf,k_max,corr_fact) -> np.array:
        try:
            import scipy.sparse as sp
            if sp.issparse(hessf):
                hessf = hessf.toarray()         #Convert in dense to apply cholesky (no sparse version available)
        except Exception:
            pass
        try:
            return np.linalg.cholesky(hessf), hessf
        except np.linalg.LinAlgError:
            L, bk = self.solvers.Build_bk(hessf,k_max,corr_fact)
            return L, bk

    def check_residuals_norm(self,xk:np.array,function:str):
        """ Method converges to solution only if residual = -gradient(xk) if goinf to zero (for each i-th component of x).
            Therefore this function checks if this condition holds.
            INPUT:
            - xk: np.array -> is the outcoming solution of the algorithm
            - function: str -> is the current function running
            OUTPUT:
            - Infinite norm of residuals
        """
        match function:
            case 'extended_rosenbrock':
                n = len(xk)
                r = np.zeros(n)
                for i in range(n):
                    if i % 2 == 0:  # k odd (1-based)
                        if i + 1 < n:
                            r[i] = 10.0 * (xk[i]**2 - xk[i+1])
                        else:
                            r[i] = 0.0  # safety for odd n
                    else:  # k even (1-based)
                        r[i] = xk[i-1] - 1.0
            
            case 'discrete_boundary_value_problem':
                n = len(xk)
                r = np.zeros(n)
                for i in range(n):
                    xi = xk[i]
                    xim1 = xk[i-1] if i > 0 else 0.0
                    xip1 = xk[i+1] if i < n-1 else 0.0
                    r[i] = (xi - xim1)*(xi - xip1) + xi + 1
            
            case 'broyden_tridiagonal_function':
                n = len(xk)
                r = np.zeros(n)
                for i in range(n):
                    xi = xk[i]
                    xim1 = xk[i-1] if i > 0 else 0.0
                    xip1 = xk[i+1] if i < n-1 else 0.0
                    r[i] = (3 - 2*xi)*xi - xim1 - xip1 + 1

        print()
        print("-"*50)
        inf_norm_res = np.linalg.norm(r, ord=np.inf)
        print(f"Infinity norm of residuals: {inf_norm_res:.6e}")
        if inf_norm_res < 1e-6:
            print("Method converged based on residual norm")
        else:
            print("Method did NOT converge based on residual norm")