import numpy as np
import time

class solvers:
    def __init__(self, matvec=None):
        self.matvec = matvec

    def CG(self, grad : np.array, tol : float, point, remaining_time:float, max_iterations = 100):
        n = len(grad)
        b = - grad
        x_k = np.zeros(n)

        # If there's no time left, return immediately with the zero step and 0 iterations
        if remaining_time is not None and remaining_time <= 0:
            return x_k, 0

        start = time.perf_counter()

        res_k = b - self.matvec(point, x_k, grad)
        resnorm_k = np.linalg.norm(res_k)
        conj_direction_k = res_k.copy()
        k = 1

        while k < max_iterations and resnorm_k > tol:
            # Time check at the start of each CG iteration
            if remaining_time is not None and (time.perf_counter() - start) >= remaining_time:
                # Time budget exceeded: return the best iterate so far
                return x_k, k - 1 if k > 1 else 0

            z_k = self.matvec(point, conj_direction_k, grad)
            curvature = conj_direction_k.T @ z_k
            if curvature <= 0:
                if k == 1:
                    return b, k
                else:
                    return x_k, k

            alpha_k = (res_k.T @ res_k) / (curvature)
            x_k = x_k + alpha_k * conj_direction_k
            res_new = res_k - alpha_k * z_k
            resnorm_k = np.linalg.norm(res_new)
            if resnorm_k < tol:
                break

            beta_k = (res_new.T @ res_new) / (res_k.T @ res_k)
            conj_direction_k = res_new + beta_k * conj_direction_k
            res_k = res_new
            k += 1

        return x_k, k