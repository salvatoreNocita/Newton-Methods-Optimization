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

    def StoppingCriterion_notmet(self, xk: np.array, gradf: np.array, tolgrad: float, k: int, k_max: int) -> bool:
        return k < k_max and np.linalg.norm(gradf) > tolgrad

    def check_solution_error(self, xk: np.array, function: str, tol: float = 10e-6):
        n = len(xk)
        if function == 'extended_rosenbrock':
            x_opt = np.ones(n)
        elif function == 'discrete_boundary_value_problem':
            h = 1.0 / (n + 1)
            x_opt = np.array([(i+1)*h * (1 - (i+1)*h) for i in range(n)])
        elif function == 'broyden_tridiagonal_function':
            x_opt = np.ones(n)
        else:
            raise ValueError(f"Unknown function: {function}")

        err = np.linalg.norm(xk - x_opt, ord=np.inf)
        print()
        print("-"*50)
        print(f"Infinity norm error to theoretical solution: {err:.6e}")
        if err < tol:
            print("Solution matches theoretical optimum within tolerance")
            return True
        else:
            print("Solution differs from theoretical optimum")
            return False

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