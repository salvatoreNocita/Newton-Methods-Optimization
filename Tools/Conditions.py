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

    def _residual(self, xk: np.array, function: str) -> np.ndarray:
        match function:
            case 'extended_rosenbrock':
                n = len(xk)
                r = np.zeros(n)
                for i in range(n):
                    if i % 2 == 0:
                        if i + 1 < n:
                            r[i] = 10.0 * (xk[i]**2 - xk[i+1])
                        else:
                            r[i] = 0.0
                    else:
                        r[i] = xk[i-1] - 1.0
                return r
            case 'discrete_boundary_value_problem':
                n = len(xk)
                h = 1.0 / (n + 1)
                r = np.empty(n)
                for i in range(n):
                    xi = xk[i]
                    xim1 = xk[i-1] if i > 0 else 0.0
                    xip1 = xk[i+1] if i < n-1 else 0.0
                    v = xi + (i+1)*h + 1.0
                    r[i] = 2.0*xi - xim1 - xip1 + (h*h)*(v**1.5)
                return r
            case 'broyden_tridiagonal_function':
                n = len(xk)
                r = np.zeros(n)
                for i in range(n):
                    xi = xk[i]
                    xim1 = xk[i-1] if i > 0 else 0.0
                    xip1 = xk[i+1] if i < n-1 else 0.0
                    r[i] = (3 - 2*xi)*xi - xim1 - xip1 + 1
                return r
            case _:
                raise ValueError(f"Unknown function: {function}")

    def StoppingCriterion_notmet(self, xk: np.array, gradf: np.array, tolgrad: float, k: int, k_max: int, function: str = None, tolres: float = None, mode: str = 'grad') -> bool:
        """Return True if the stopping criterion is NOT met.
        mode: 'grad' uses ||gradf||; 'res' uses ||r(xk)||_inf; 'both' requires both below tol to stop.
        For 'res' and 'both', provide `function` and `tolres`.
        """
        if k >= k_max:
            return False
        if mode == 'grad':
            return np.linalg.norm(gradf) > tolgrad
        elif mode == 'res':
            assert function is not None and tolres is not None, "Provide function and tolres for residual-based stopping"
            r = self._residual(xk, function)
            return np.linalg.norm(r, ord=np.inf) > tolres
        elif mode == 'both':
            assert function is not None and tolres is not None, "Provide function and tolres for residual-based stopping"
            r = self._residual(xk, function)
            return (np.linalg.norm(gradf) > tolgrad) or (np.linalg.norm(r, ord=np.inf) > tolres)
        else:
            raise ValueError("mode must be 'grad', 'res', or 'both'")

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
        r = self._residual(xk, function)

        print()
        print("-"*50)
        inf_norm_res = np.linalg.norm(r, ord=np.inf)
        print(f"Infinity norm of residuals: {inf_norm_res:.6e}")
        if inf_norm_res < 1e-6:
            print("Method converged based on residual norm")
        else:
            print("Method did NOT converge based on residual norm")