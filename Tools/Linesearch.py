import numpy as np

class lineSearch(object):
    """ This class perform inexact line search within Armijo conditions + Backtracking to find the step size.
        Armijo conditions are used to guarantee a sufficient decrease on descent direction.
        Backtracking is used to control step size avoiding being too small.
    """
    
    def __init__(self):
        pass

    def ArmijoConditions_notmet(self,xk: np.array,alphak: float, pk: np.array, gradf: np.array,
                                c1: float, objective_function,)-> bool:
        xk_1= xk + alphak * pk

        return objective_function(xk_1) > objective_function(xk) + c1 * alphak * (gradf.T @ pk)
    
    def Backtracking(self, xk: np.array, pk: np.array,gradf: np.array, alpha0: float, bt: int,
                     btmax: int, rho: float,c1: float,objective_function)-> float:
        
        alphak= alpha0

        while self.ArmijoConditions_notmet(xk,alphak,pk,gradf,c1,objective_function) and bt<btmax:
            
            alphak= rho*alphak
            bt += 1

            if bt > btmax:
                return None

        return alphak