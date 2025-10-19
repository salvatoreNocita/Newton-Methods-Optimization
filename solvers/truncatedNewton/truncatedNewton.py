import numpy as np
from colorama import Fore
import time
from ..newtonBase import newtonBase

# versione minimale e pratica
class truncatedNewton(newtonBase):
    def __init__(
        self,
        *,
        eta: float,
        rate_of_convergence: str | None = None,
        **base_kwargs,               
    ) -> None:
        super().__init__(**base_kwargs)
    
        self.eta = float(eta)
        self.rate_of_convergence = rate_of_convergence

        # Solvers that need a matvec for H*v
        if self.compute_hess_vec is None:
            raise RuntimeError("TruncatedNewton requires Hessian-vector product operator.")
        
        self.solvers = self._get_solvers(matvec=self.compute_hess_vec)

    def _eta_policy(self):
        if self.rate_of_convergence == "quadratic":
            return lambda gnorm: gnorm
        # default superlinear
        return lambda gnorm: np.sqrt(gnorm)

    def Run(self, timing: bool = False, print_every: int = 50):
        xk = self.x0.copy()
        self.x_seq = [xk]

        grad = self.compute_gradient(xk)
        grad_norm = float(np.linalg.norm(grad))
        self.gradient = grad
        self.norm_grad_seq = [grad_norm]

        eta_k = self._eta_policy()
        start_time = time.perf_counter()
        max_time = self._timing_budget(len(xk), timing)
        success = True

        while self.conditions.StoppingCriterion_notmet(xk, grad, self.tolgrad, self.k, self.kmax) and \
                (time.perf_counter() - start_time) < max_time:

            iter_start = time.time()

            tol_cg = min(self.eta, eta_k(grad_norm)) * grad_norm
            self.tol_seq.append(tol_cg)

            remaining_time = max_time - (time.perf_counter() - start_time)

            pk, it_cg = None, 0
            pk, it_cg = self.solvers.CG(grad, tol_cg, xk, remaining_time=remaining_time)
            

            alphak = self.linesearch.Backtracking(
                xk, pk, grad, self.alpha0, self.bt, self.btmax,
                self.rho, self.c1, self.objective_function
            )

            self.inner_iters.append(int(it_cg))
            self.bt_seq.append(alphak)

            if alphak is None:
                print(f"Backtracking failed with {self.btmax} iterations. Method doesn't converge.")
                success = False
                break

            xk = self.step(xk, alphak, pk)
            grad = self.compute_gradient(xk)
            grad_norm = float(np.linalg.norm(grad))
            self.gradient = grad
            self.norm_grad_seq.append(grad_norm)
            self.k += 1

            iter_end = time.time()
            self.execution_times.append(iter_end - iter_start)

            if print_every > 0 and self.k % print_every == 0:
                print(Fore.LIGHTBLUE_EX + '-' * 50 + Fore.RESET)
                print(Fore.LIGHTBLUE_EX + f'CURRENT ITERATION : {self.k} ' + Fore.RESET)
                print(Fore.LIGHTBLUE_EX + '-' * 50 + Fore.RESET)
                print(f'Iterate: {xk} \n Grad Norm: {grad_norm} \n Alpha : {alphak} \n CG Iter: {it_cg}')
                print(Fore.RED + f"Iteration {self.k} took {iter_end - iter_start:.4f} seconds" + Fore.RESET)

        if grad_norm > self.tolgrad or (time.perf_counter() - start_time) >= max_time:
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
            self.tol_seq,
        )