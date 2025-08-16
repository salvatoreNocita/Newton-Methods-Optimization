import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import time

import numpy as np
import scipy as sci
from Tools.Derivatives import ApproximateDerivatives,ExactDerivatives,SparseApproximativeDerivatives
from Tools.Functions import FunctionDefinition
from Tools.Linesearch import LineSearch
from .SolverInstruments import Solvers
from Tools.Conditions import CheckConditions
from functools import partial
from colorama import Fore


class TruncatedNewtonMethod:
    def __init__(self, x0: np.array,function: str,alpha0: float, kmax: int, tolgrad: float, c1: float, 
                 eta : float, rho: float, btmax: int, rate_of_convergence: str,
                 derivatives: str, derivative_method: str, perturbation: float):
        
        self.function = function
        self.objective_function = FunctionDefinition().get_objective_function(function)
        self.x0= x0
        self.alpha0= alpha0
        self.kmax= kmax
        self.tolgrad= tolgrad
        self.c1= c1
        self.eta = eta
        self.rho= rho
        self.btmax= btmax
        self.rate_of_convergence = rate_of_convergence
        self.norm_grad_seq = []
        self.k= 0
        self.bt= 0
        self.x_seq= [x0]
        self.bt_seq= []
        self.derivatives = derivatives
        self.gradient = np.zeros(len(self.x0))

        self.conditions = CheckConditions()
        self.linesearch = LineSearch()
        self.exact_d = ExactDerivatives()
        self.finit_d = ApproximateDerivatives(self.objective_function,derivative_method, perturbation)
        self.sp_finit_d = SparseApproximativeDerivatives(self.objective_function, derivative_method,perturbation)
        self.compute_gradient, self.compute_hess_vec = self.compute_gradient_hessian(self.x0,self.gradient)
        self.solvers = Solvers(matvec = self.compute_hess_vec)

    def compute_gradient_hessian(self, xk: np.array,gradient:np.array):
        if self.derivatives == 'exact':

            if self.function == 'extended_rosenbrock':

                def grad_fn(x):
                    return self.exact_d.extended_rosenbrock(x, hessian=False)
                def hess_vec(x,v, grad):
                    return self.exact_d.extended_rosenbrock_hessian_vector_product(x, v, grad)
            
            elif self.function == 'discrete_boundary_value_problem':

                def grad_fn(x):
                    return self.exact_d.discrete_boundary_value_problem(x, hessian=False)
                def hess_vec(x,v, grad):
                    return self.exact_d.dbv_hessian_vector_product(x, v, grad)
            
            elif self.function == 'broyden_tridiagonal_function':

                def grad_fn(x):
                    return self.exact_d.Broyden_tridiagonal_function(x, hessian=False)
                def hess_vec(x, v, grad):
                    return self.exact_d.broyden_hessian_vector_product(x, v, grad)
            
            elif self.function == 'rosenbrock':

                def grad_fn(x):
                    return self.exact_d.exact_rosenbrock(x, hessian=False)
                def hess_vec(x,v, grad):
                    return self.exact_d.rosenbrock_hessian_vector_product(x, v, grad)
            
            else:
                raise ValueError(f"Unknown function '{self.function}' for exact derivatives")
            
        
        elif self.derivatives == 'finite_differences' or self.derivatives == 'adaptive_finite_differences':
            adaptive = True if self.derivatives == 'adaptive_finite_differences' else False
            grad_fn = partial(self.sp_finit_d.approximate_gradient_parallel, adaptive=adaptive)
            hess_vec = partial(self.sp_finit_d.hessian_vector_product, adaptive = adaptive) # TO IMPLEMENT!!! (ADAPTIVNESS)
        
        return grad_fn, hess_vec

    
    def Step(self,xk: np.array, alphak: float, pk: np.array) -> np.array:
        xk_1= xk+ alphak*pk
        self.x_seq.append(xk_1)
        self.bt_seq.append(alphak)
        return xk_1
    
    def Run(self, print_every = 50):

        xk = self.x0
        self.x_seq.append(xk)

        gradf = self.compute_gradient(xk)
        gradf_norm = np.linalg.norm(gradf)

        eta_k = self._set_eta_k()

        iter_cg = []

        start_time = time.perf_counter()
        max_time = float(np.clip( 20 * (len(xk) / 1e3)**0.6 , a_min=10.0, a_max=300.0))     #Heuristic
        success = True

        while self.conditions.StoppingCriterion_notmet(xk,gradf,self.tolgrad,self.k,self.kmax) and \
                (time.perf_counter() - start_time) < max_time:
            
            start = time.time()
            tol_cg = min(self.eta, eta_k(gradf_norm)) * gradf_norm
            
            pk, it_cg = self.solvers.CG_find_pk(gradf, tol_cg, xk, self.kmax)

            alphak = self.linesearch.Backtracking(xk, pk, gradf, self.alpha0, self.bt, self.btmax,
                                                  self.rho, self.c1, self.objective_function)
            
            iter_cg.append(it_cg)
            self.bt_seq.append(alphak)

            if alphak is None:
                print(f"Backtracking strategy failed with {self.btmax} iterations")
                print(f"Method doesn't converge")
                exit
            else:
                xk = self.Step(xk, alphak, pk)
                gradf = self.compute_gradient(xk)
                self.gradient = gradf

            gradf = self.compute_gradient(xk)
            gradf_norm = np.linalg.norm(gradf)
            self.norm_grad_seq.append(gradf_norm)
            self.k += 1

            if self.k % print_every == 0:
                print(Fore.LIGHTBLUE_EX + '-' * 50 + Fore.RESET)
                print(Fore.LIGHTBLUE_EX + f'CURRENT ITERATION : {self.k} ' + Fore.RESET)
                print(Fore.LIGHTBLUE_EX + '-' * 50 + Fore.RESET)
                print(f'Iterate: {xk} \n Grad Norm: {gradf_norm} \n Alpha : {alphak} \n CG Iter: {it_cg}')
            
            end = time.time()
            if self.k % print_every == 0:
                print(Fore.RED + f"Iteration {self.k} took {end - start:.4f} seconds" + Fore.RESET)

        if gradf_norm > self.tolgrad or (time.perf_counter() - start_time) >= max_time:
            success = False
        
        return xk, self.objective_function(xk), self.norm_grad_seq, self.k, self.x_seq, success
    

    def _set_eta_k(self):
        match self.rate_of_convergence:

            case "superlinear":
                return lambda gradf_norm : np.sqrt(gradf_norm)
            
            case "quadratic":
                return lambda gradf_norm : gradf_norm
            
            case _:
                raise ValueError(f"Invalid Rate of Convergence : {self.rate_of_convergence}")
            
