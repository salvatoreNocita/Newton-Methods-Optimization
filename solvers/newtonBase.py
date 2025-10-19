# newton_methods.py
from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from functools import partial
from typing import Callable, Optional, List, Any

import numpy as np

# --- External deps from your project ---
from tools.derivatives import approximativeDerivatives, exactDerivatives, sparseApproximativeDerivatives
from tools.functions import functionDefinition
from tools.linesearch import lineSearch
from tools.conditions import checkConditions

try:
    from truncatedNewton.solverInstruments import solvers as TSolvers
except Exception:
    TSolvers = None

try:
    from modifiedNewton.solverInstruments import solvers as MSolvers
except Exception:
    MSolvers = None


class newtonBase:
    """
    Base class holding:
    - problem definition, derivative operators (exact/FD/sparse-FD),
    - linesearch and conditions,
    - shared bookkeeping (sequences, timers),
    - utilities (step, eta_k policy, timing window).
    Subclasses only implement the algorithm-specific `run_once` loop.
    """

    def __init__(
        self,
        *,
        x0: np.ndarray,
        function: str,
        alpha0: float,
        kmax: int,
        tolgrad: float,
        c1: float,
        rho: float,
        btmax: int,
        # derivatives config
        derivatives: str,
        derivative_method: str,
        perturbation: float,
        # optional for TNM
        rate_of_convergence: Optional[str] = None,
    ) -> None:
        self.function = function
        self.objective_function = functionDefinition().get_objective_function(function)
        self.x0 = np.asarray(x0, dtype=float)
        self.alpha0 = float(alpha0)
        self.kmax = int(kmax)
        self.tolgrad = float(tolgrad)
        self.c1 = float(c1)
        self.rho = float(rho)
        self.btmax = int(btmax)

        # Common book-keeping
        self.k = 0
        self.bt = 0
        self.x_seq: List[np.ndarray] = [self.x0.copy()]
        self.bt_seq: List[float] = []
        self.norm_grad_seq: List[float] = []
        self.inner_iters: List[int] = []
        self.execution_times: List[float] = []
        self.tol_seq: List[float] = []  # used by TNM (tol_cg) and kept for compatibility

        # Derivatives setup
        self.derivatives = derivatives  # 'exact' | 'finite_differences' | 'adaptive_finite_differences'
        self.adaptive_h = derivatives == 'adaptive_finite_differences'
        self.gradient = np.zeros_like(self.x0)

        # Shared utilities
        self.conditions = checkConditions()
        self.linesearch = lineSearch()
        self.exact_d = exactDerivatives()
        self.finit_d = approximativeDerivatives(self.objective_function, derivative_method, perturbation)
        self.sp_finit_d = sparseApproximativeDerivatives(self.objective_function, derivative_method, perturbation)

        # To be filled by subclasses via _build_derivative_operators()
        self.compute_gradient: Callable[[np.ndarray], np.ndarray]
        # Optional:
        self.compute_hessian: Optional[Callable[[np.ndarray, np.ndarray, bool], Any]] = None
        self.compute_hess_vec: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None

        self._build_derivative_operators()  # sets compute_gradient and either compute_hessian or compute_hess_vec

    # ---------- Common utilities ----------

    def _get_solvers(self, matvec = None):
        """
        Helper that returns a Solvers instance from either TSolverInstruments or SolverInstruments,
        depending on what's available in your project
        """
        if matvec is None:
            return MSolvers()
        else:
            return TSolvers(matvec=matvec)

    def _timing_budget(self, n: int, timing: bool) -> float:
        """
        Returns a time budget in seconds, based on problem size n.
        If timing is False, returns np.inf (no time limit).
        """
        if timing:
            return float(np.clip(20 * (n / 1e3) ** 0.6, a_min=10.0, a_max=300.0))
        return np.inf

    def step(self, xk: np.ndarray, alphak: float, pk: np.ndarray) -> np.ndarray:
        xkp1 = xk + alphak * pk
        self.x_seq.append(xkp1)
        self.bt_seq.append(alphak)
        return xkp1

    # ---------- Derivative operators factory ----------

    def _build_derivative_operators(self) -> None:
        """
        Populate:
          - self.compute_gradient(x) -> grad
          - and either:
              self.compute_hess_vec(x, v, grad)   # for TNM
            or
              self.compute_hessian(x, grad, adaptive) # for Modified Newton
        """
        if self.derivatives == 'exact':
            if self.function == 'extended_rosenbrock':
                self.compute_gradient = lambda x: self.exact_d.extended_rosenbrock(x, hessian=False)
                self.compute_hess_vec = lambda x, v, g: self.exact_d.extended_rosenbrock_hessian_vector_product(x, v, g)
                self.compute_hessian = lambda x, g, a: self._as_array(self.exact_d.extended_rosenbrock(x, hessian=True)[1])

            elif self.function == 'extended_powell':
                self.compute_gradient = lambda x: self.exact_d.extended_powell(x, hessian=False)
                self.compute_hess_vec = lambda x, v, g: self.exact_d.extended_powell_hessian_vector_product(x, v, g)
                self.compute_hessian = lambda x, g, a: self._as_array(self.exact_d.extended_powell(x, hessian=True)[1])

            elif self.function == 'broyden_tridiagonal_function':
                self.compute_gradient = lambda x: self.exact_d.Broyden_tridiagonal_function(x, hessian=False)
                self.compute_hess_vec = lambda x, v, g: self.exact_d.broyden_hessian_vector_product(x, v, g)
                self.compute_hessian = lambda x, g, a: self._as_array(self.exact_d.Broyden_tridiagonal_function(x, hessian=True)[1])

            elif self.function == 'rosenbrock':
                self.compute_gradient = lambda x: self.exact_d.exact_rosenbrock(x, hessian=False)
                self.compute_hess_vec = lambda x, v, g: self.exact_d.rosenbrock_hessian_vector_product(x, v, g)
                self.compute_hessian = lambda x, g, a: self._as_array(self.exact_d.exact_rosenbrock(x, hessian=True)[1])

            else:
                raise ValueError(f"Unknown function '{self.function}' for exact derivatives")

        elif self.derivatives in ('finite_differences', 'adaptive_finite_differences'):
            # Sparse parallel gradient; adaptive controlled by self.adaptive_h at call site if needed
            self.compute_gradient = partial(self.sp_finit_d.approximate_gradient_parallel, adaptive=self.adaptive_h)
            # For TNM we need Hessian-vector product
            self.compute_hess_vec = partial(self.sp_finit_d.hessian_vector_product, adaptive=self.adaptive_h)
            # For Modified Newton we may need the full Hessian (dense for n<1e3, sparse-structured otherwise)
            if self.function == 'extended_rosenbrock':
                self.compute_hessian = lambda x, g, a : self.sp_finit_d.hessian_approx_extendedros(x, g)
            elif self.function == 'broyden_tridiagonal_function':
                self.compute_hessian = lambda x, g, a : self.sp_finit_d.hessian_approx_broyden_tridiagonal(x, g)
            elif self.function == 'extended_powell':
                self.compute_hessian = lambda x, g, a : self.sp_finit_d.hessian_approx_extended_powell(x, g)
            else:
                self.compute_hessian = lambda x, g, a : self.finit_d.hessian(x, g)

        else:
            raise ValueError("'derivatives' must be 'exact', 'finite_differences', or 'adaptive_finite_differences'")

    @staticmethod
    def _as_array(H):
        return H.toarray() if hasattr(H, 'toarray') else np.asarray(H)


    def Run(self, *args, **kwargs):
        """
        Subclasses implement. Kept for compatibility with your current main().
        """
        raise NotImplementedError