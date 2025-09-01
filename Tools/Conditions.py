import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import scipy.sparse as ssp
from Modified_Newton_Method.SolverInstruments import Solvers

class CheckConditions(object):
    """ This class includes all conditions that must be checked for convergence and during algorithms """

    def __init__(self):
        self.solvers = Solvers()

    def StoppingCriterion_notmet(self, xk: np.array, gradf: np.array, tolgrad: float, k: int, k_max: int) -> bool:
        return k < k_max and np.linalg.norm(gradf) > tolgrad

    def H_is_positive_definite(self,hessf,k_max,corr_fact) -> np.array:
        iter = 0
        if ssp.issparse(hessf):
            hessf = hessf.toarray()         #Convert in dense to apply cholesky (no sparse version available)
        try:
            return ssp.csc_matrix(np.linalg.cholesky(hessf)), hessf, iter
        except np.linalg.LinAlgError:
            L, bk, iter = self.solvers.Build_bk(hessf,k_max,corr_fact)
            return L, bk, iter
    
    def build_perturbation_vector(self,xk:np.array,h):
        pv = np.empty(len(xk))
        for i,x_i in enumerate(xk):
            pv[i] = h* max(np.abs(x_i),1.0)
        
        return pv