import numpy as np
from colorama import Fore
import time
from ..newtonBase import newtonBase

class modifiedNewton(newtonBase):
    def __init__(
        self,
        *,
        solver_linear_system: str,        # 'cg' | 'chol'
        H_correction_factor: float,
        precond: str,                      # 'yes' | 'no' (usato solo se solver_linear_system == 'cg')
        **base_kwargs,                     # tutti i parametri comuni della NewtonBase passano qui
    ) -> None:
        super().__init__(**base_kwargs)

        self.solver_linear_system = solver_linear_system
        self.H_correction_factor = float(H_correction_factor)
        self.precond = precond
        self.solvers = self._get_solvers()

        if self.compute_hessian is None:
            raise RuntimeError("ModifiedNewton richiede una Hessiana completa.")


    def Run(self, timing: bool = False, print_every: int = 50):
        xk = self.x0.copy()
        self.x_seq = [xk]

        grad = self.compute_gradient(xk)
        self.gradient = grad
        self.norm_grad_seq = [float(np.linalg.norm(grad))]

        start_time = time.perf_counter()
        max_time = self._timing_budget(len(xk), timing)
        success = True

        while self.conditions.StoppingCriterion_notmet(xk, grad, self.tolgrad, self.k, self.kmax) and \
                (time.perf_counter() - start_time) < max_time:

            iter_start = time.time()

            # Build Hessian (dense or sparse-structured)
            H = self.compute_hessian(xk, grad, self.adaptive_h)
            if isinstance(H, tuple) and len(H) == 2:
                _, H = H  # be robust to functions returning (f, H)

            # If finite differences, force symmetry
            if self.derivatives in ('finite_differences', 'adaptive_finite_differences'):
                if hasattr(self.solvers, "make_symmetric"):
                    H = self.solvers.make_symmetric(H)
                else:
                    # minimal symmetry enforcement
                    H = 0.5 * (H + H.T)

            # Ensure PD (returns Cholesky L of corrected H, the corrected matrix bk, and inner iterations)
            L, bk, inner_iter = self.conditions.H_is_positive_definite(H, self.kmax, self.H_correction_factor)

            # Solve for pk
            if self.solver_linear_system == 'cg':
                # Support both CG_Find_pk signatures with or without precondition
                if hasattr(self.solvers, "CG_Find_pk"):
                    try:
                        pk = self.solvers.CG_Find_pk(bk, grad, self.precond)
                    except TypeError:
                        pk = self.solvers.CG_Find_pk(bk, grad)
                elif hasattr(self.solvers, "CG_find_pk"):
                    # Some TNM solvers take (grad, tol, xk, ...) so not reusable here.
                    # Fall back to direct solve if no suitable CG solver is available.
                    try:
                        pk = self.solvers.CG_find_pk(bk, grad)  # may not exist
                    except Exception as e:
                        raise RuntimeError("CG solver for ModifiedNewton not found in your Solvers.") from e
                else:
                    raise RuntimeError("No CG solver found in Solvers for ModifiedNewton.")
            elif self.solver_linear_system == 'chol':
                if hasattr(self.solvers, "chol_Find_Pk"):
                    pk = self.solvers.chol_Find_Pk(L, grad)
                else:
                    # Minimal fallback using numpy.linalg for dense arrays
                    try:
                        y = np.linalg.solve(L, -grad)
                        pk = np.linalg.solve(L.T, y)
                    except Exception as e:
                        raise RuntimeError("Cholesky path unavailable and fallback failed.") from e
            else:
                raise ValueError("solver_linear_system must be 'cg' or 'chol'.")

            alphak = self.linesearch.Backtracking(
                xk, pk, grad, self.alpha0, self.bt, self.btmax, self.rho, self.c1, self.objective_function
            )

            self.bt_seq.append(alphak)
            self.inner_iters.append(int(inner_iter))

            if alphak is None:
                print(f"Backtracking failed with {self.btmax} iterations. Method doesn't converge.")
                success = False
                break

            xk = self.step(xk, alphak, pk)
            grad = self.compute_gradient(xk)
            self.gradient = grad
            self.norm_grad_seq.append(float(np.linalg.norm(grad)))
            self.k += 1

            iter_end = time.time()
            self.execution_times.append(iter_end - iter_start)

            if print_every > 0 and self.k % print_every == 0:
                print(Fore.LIGHTBLUE_EX + '-' * 50 + Fore.RESET)
                print(Fore.LIGHTBLUE_EX + f'CURRENT ITERATION : {self.k} ' + Fore.RESET)
                print(Fore.LIGHTBLUE_EX + '-' * 50 + Fore.RESET)
                print(f'Iterate: {xk} \n Grad Norm: {self.norm_grad_seq[-1]} \n Alpha : {alphak} \n Inner Iter: {inner_iter}')
                print(Fore.RED + f"Iteration {self.k} took {iter_end - iter_start:.4f} seconds" + Fore.RESET)

        if float(np.linalg.norm(grad)) > self.tolgrad or (time.perf_counter() - start_time) >= max_time:
            success = False

        return (
            self.execution_times,
            xk,
            self.objective_function(xk),
            self.norm_grad_seq,
            self.k,
            success,
            self.inner_iters,
            self.bt_seq,
            None,
        )