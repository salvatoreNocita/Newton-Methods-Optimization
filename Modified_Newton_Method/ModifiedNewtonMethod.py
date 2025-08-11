import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import scipy as sci
from Tools.Derivatives import ApproximateDerivatives,ExactDerivatives,SparseApproximativeDerivatives
from Tools.Functions import FunctionDefinition
from Tools.Linesearch import LineSearch
from SolverInstruments import Solvers

class ModifiedNewton(object):
    """
    Modified Newton's method for unconstrained optimization with robust Hessian handling and efficient linear algebra.

    This implementation minimizes a nonlinear objective function by iteratively updating the solution using Newton-type steps.
    The algorithm can compute gradient and Hessian information either exactly (if available) or approximately via finite differences,
    automatically choosing dense or sparse representations depending on the problem size.

    The descent direction is computed by solving the Newton system using either:
    - Conjugate Gradient (CG), optionally preconditioned with Incomplete Cholesky (for large-scale/sparse problems)
    - Direct Cholesky factorization (for small/moderate dense problems)

    If the Hessian is not positive definite, the algorithm applies an iterative diagonal correction (`Build_bk`) until positive definiteness is achieved.
    The method is capable of handling a variety of test problems, including:
    - Extended Rosenbrock
    - Classic Rosenbrock
    - Discrete boundary value problem
    - Broyden tridiagonal function

    Attributes:
    - x0: np.array -> Initial point (column vector) for the optimization.
    - alpha0: float -> Initial step size (typically 1 for Newton's method).
    - function: str -> Name of the objective function to minimize.
    - kmax: int -> Maximum number of iterations.
    - tolgrad: float -> Tolerance for the gradient norm (stopping criterion).
    - c1: float -> Armijo condition parameter for line search.
    - rho: float -> Backtracking parameter for line search.
    - btmax: int -> Maximum number of backtracking iterations.
    - solver_linear_system: str -> Method for solving the Newton system ('cg' or 'chol').
    - H_correction_factor: float -> Multiplicative factor for Hessian correction if not positive definite.
    - precond: str -> Whether to use preconditioning ('yes' or 'no') in CG.
    - derivatives: str -> Whether to use 'exact' or 'finite_differences' for gradient and Hessian.
    - derivative_method: str -> Method for finite differences ('forward', 'backward', 'central').
    - perturbation: float -> Step size for finite difference approximations.

    Variables:
    - k: int -> Iteration counter.
    - bt: int -> Counter for backtracking iterations.
    - x_seq: list of np.array -> Sequence of iterates produced by the method.
    - bt_seq: list of int -> Sequence of backtracking iterations at each step.

    Methods:
    - compute_gradient_hessian(xk) -> Selects and returns the appropriate gradient and Hessian computation functions (exact or finite difference, dense or sparse).
    - CG_Find_pk(xk, bk, gradf) -> Computes the Newton direction by solving the linear system with Conjugate Gradient, optionally with Incomplete Cholesky preconditioning.
    - StoppingCriterion_notmet(xk, gradf) -> Returns True if the stopping criterion is not met (i.e., k < kmax and ||gradf|| > tolgrad).
    - H_is_positive_definite(hessf) -> Checks if Hessian is positive definite via Cholesky factorization; if not, calls Build_bk to modify it.
    - Build_bk(hessf) -> Iteratively adds a diagonal correction to the Hessian until it becomes positive definite.
    - make_symmetric(Hessf, xk) -> Ensures the Hessian is symmetric (useful for finite difference approximations).
    - Step(xk, alphak, pk) -> Performs an iteration step: x_{k+1} = x_k + alphak * pk.
    - Run() -> Executes the full optimization procedure and returns the final solution, function value, gradient norm, total iterations, and iteration history.
    """
    def __init__(self, x0: np.array,alpha0: float, function: str, kmax: int, tolgrad: float, c1: float, 
                 rho: float, btmax: int, solver_linear_system: str,H_correction_factor,precond: str,
                 derivatives: str, derivative_method: str, perturbation: float):
        
        self.function = function
        functions = FunctionDefinition()
        match self.function:
            case 'extended_rosenbrock':
                self.objective_function = functions.extended_rosenbrock            
            case 'discrete_boundary_value_problem':
                self.objective_function = functions.dbv_function                      
            case 'broyden_tridiagonal_function':
                self.objective_function = functions.btf_function
            case 'rosenbrock':
                self.objective_function = functions.rosenbrock_function                      

        self.x0= x0
        self.alpha0= alpha0
        self.kmax= kmax
        self.tolgrad= tolgrad
        self.c1= c1
        self.rho= rho
        self.btmax= btmax
        self.solver_linear_system= solver_linear_system
        self.H_correction_factor= H_correction_factor
        self.precond = precond
        self.k= 0
        self.bt= 0
        self.x_seq= [x0]
        self.bt_seq= []
        self.derivatives = derivatives
        
        self.linesearch = LineSearch()
        self.solvers = Solvers()
        self.exact_d = ExactDerivatives()
        self.finit_d = ApproximateDerivatives(self.objective_function,derivative_method, perturbation)
        self.sp_finit_d = SparseApproximativeDerivatives(self.objective_function, derivative_method,perturbation)
        self.compute_gradient, self.compute_hessian = self.compute_gradient_hessian(self.x0,derivatives)
    
    def H_is_positive_definite(self,hessf) -> np.array:
        try:
            import scipy.sparse as sp
            if sp.issparse(hessf):
                hessf = hessf.toarray()         #Converte in dense to apply cholesky (no sparse version available)
        except Exception:
            pass
        try:
            return np.linalg.cholesky(hessf), hessf
        except np.linalg.LinAlgError:
            L, bk = self.Build_bk(hessf)
            return L, bk
                  
    def Build_bk(self,hessf: np.array) -> np.ndarray:
        beta = 1e-3
        try:
            import scipy.sparse as sp
            if sp.issparse(hessf):
                diag_elements = hessf.diagonal()
                H = hessf.toarray()
            else:
                diag_elements = np.diag(hessf)
                H = hessf
        except Exception:
            diag_elements = np.diag(hessf)
            H = hessf
        tau0 = 0 if np.min(diag_elements) > 0 else (-np.min(diag_elements) + beta)
        tauk = tau0
        bk = H + tauk * np.identity(H.shape[0])
        flag = False
        i = 0
        while not flag and i < self.kmax:
            tauk_1 = max(self.H_correction_factor * tauk, beta)
            bk = bk + tauk_1 * np.identity(bk.shape[0])
            tauk = tauk_1
            try:
                L = np.linalg.cholesky(bk)
                flag = True
                return L, bk
            except np.linalg.LinAlgError:
                flag = False
            i += 1
        raise np.linalg.LinAlgError(f"Hessian can't be modified with {self.kmax}")

    def compute_gradient_hessian(self, xk: np.array,derivatives: str):
        if derivatives == 'exact':
            if self.function == 'extended_rosenbrock':

                def grad_fn(x):
                    return self.exact_d.extended_rosenbrock(x, hessian=False)
                def hess_fn(x):
                    _, H = self.exact_d.extended_rosenbrock(x, hessian=True)
                    return H.toarray() if hasattr(H, 'toarray') else np.asarray(H)
                return grad_fn, hess_fn
            
            elif self.function == 'discrete_boundary_value_problem':
                def grad_fn(x):
                    return self.exact_d.discrete_boundary_value_problem(x, hessian=False)
                def hess_fn(x):
                    _, H = self.exact_d.discrete_boundary_value_problem(x, hessian=True)
                    return H.toarray() if hasattr(H, 'toarray') else np.asarray(H)
                return grad_fn, hess_fn
            
            elif self.function == 'broyden_tridiagonal_function':
                def grad_fn(x):
                    return self.exact_d.Broyden_tridiagonal_function(x, hessian=False)
                def hess_fn(x):
                    _, H = self.exact_d.Broyden_tridiagonal_function(x, hessian=True)
                    return H.toarray() if hasattr(H, 'toarray') else np.asarray(H)
                return grad_fn, hess_fn
            
            elif self.function == 'rosenbrock':
                def grad_fn(x):
                    return self.exact_d.exact_rosenbrock(x, hessian=False)
                def hess_fn(x):
                    _, H = self.exact_d.exact_rosenbrock(x, hessian=True)
                    return H.toarray() if hasattr(H, 'toarray') else np.asarray(H)
                return grad_fn, hess_fn
            
            else:
                raise ValueError(f"Unknown function '{self.function}' for exact derivatives")
            
        elif derivatives == 'finite_differences':
            grad = self.sp_finit_d.approximate_gradient_parallel
            if len(xk) < 10**3:
                hessian = self.finit_d.hessian
                return grad, hessian
            
            else:
                if self.function == 'extended_rosenbrock':
                    hessian = self.sp_finit_d.hessian_approx_extendedros
                else:
                    hessian = self.sp_finit_d.hessian_approx_tridiagonal
                return grad, hessian
            
        else:
            raise ValueError("'derivatives' must be either 'exact' or 'finite_differences'")
        
    def Step(self,xk: np.array, alphak: float, pk: np.array) -> np.array:
        xk_1= xk+ alphak*pk
        self.x_seq.append(xk_1)
        return xk_1
                
    def StoppingCriterion_notmet(self,xk: np.array, gradf)-> bool:
        return self.k<self.kmax and np.linalg.norm(gradf)> self.tolgrad

    def Run(self)-> tuple[np.array, float, float, int, list[np.array], list[float]]:
        xk = self.x0
        grad = self.compute_gradient(xk)
        
        while self.StoppingCriterion_notmet(xk, grad):
            hessf = self.compute_hessian(xk)
            if isinstance(hessf, tuple) and len(hessf) == 2:
                _, hessf = hessf
            if self.derivatives == 'finite_differences':
                hessf = self.solvers.make_symmetric(hessf)
            L, bk = self.H_is_positive_definite(hessf)
            if self.solver_linear_system == 'cg':
                pk = self.solvers.CG_Find_pk(bk, grad, self.precond)
            elif self.solver_linear_system == 'chol':
                if len(xk) < 10**3:
                    pk = self.solvers.chol_Find_Pk(L, grad)
                else:
                    print(f"Is not possible to find pk with cholesky with dimension {len(xk)}")
                    exit
            alphak = self.linesearch.Backtracking(xk, pk, grad, self.alpha0, self.bt, self.btmax,
                                                  self.rho, self.c1, self.objective_function)
            self.bt_seq.append(alphak)
            if alphak is None:
                print(f"Backtracking strategy failed with {self.btmax} iterations")
                print(f"Method doesn't converge")
                exit
            else:
                xk = self.Step(xk, alphak, pk)
                grad = self.compute_gradient(xk)
                self.k += 1

        norm_gradfxk = np.linalg.norm(grad)
        return xk, self.objective_function(xk), norm_gradfxk, self.k, self.x_seq, self.bt_seq