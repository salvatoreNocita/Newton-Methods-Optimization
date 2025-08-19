import numpy as np

class Solvers:
    def __init__(self, matvec=None):
        self.matvec = matvec

    def CG_find_pk(self, grad : np.array, tol : float, point, max_iterations = 100):
        n = len(grad)
        b = - grad
        x_k = np.zeros(n)
    
        res_k = b - self.matvec(point, x_k, grad)
        resnorm_k = np.linalg.norm(res_k)
        conj_direction_k = res_k.copy()
        k = 1
        
        while k < max_iterations and resnorm_k > tol: 
            z_k = self.matvec(point, conj_direction_k, grad)
            curvature = conj_direction_k.T @ z_k
            if  curvature <= 0: 
                if k == 1:
                    return b,k
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