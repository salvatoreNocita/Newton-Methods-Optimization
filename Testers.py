import numpy as np
from itertools import product
import math

class Test_settings(object):
    """ This class is used to retrieve Tests setting. """

    def __init__(self):
        pass

    def get_modified_Newton_params(self):
        NewtonBackTracking_ARG_f= {'x0': ['fixed','sampled'],
                                    'alpha0': 1,
                                    'btmax': 50,
                                    'rho': 0.5,
                                    'c1': 1e-4,
                                    'tolgrad': 1e-5,
                                    'kmax': 1000,
                                    'solver_linear_system': 'cg',
                                    'H_correction_factor': 3,
                                    'precond': ['yes','no'],
                                    'derivatives': ['adaptive_finite_differences','exact','finite_differences'],
                                    'derivative_method': ['forward','forward','backward','central'],
                                    'perturbation': [1e-2,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12]
                                }
        
        return NewtonBackTracking_ARG_f

    def get_truncated_Newton_params(self):
        NewtonBackTracking_ARG_f= {'x0': ['fixed','sampled'],
                                        'alpha0': 1,
                                        'btmax': 50,
                                        'rho': 0.5,
                                        'c1': 1e-4,
                                        'tolgrad': 1e-5,
                                        'kmax': 1000,
                                        'eta': 0.5,
                                        'rate_of_convergence': ['superlinear','quadratic'],
                                        'derivatives': ['exact','finite_differences','adaptive_finite_differences'],
                                        'derivative_method': ['forward','backward','central'],
                                        'perturbation': [1e-2,1e-4,1e-6,1e-8,1e-10,1e-12]
                                    }
        return NewtonBackTracking_ARG_f

    def expand_param_grid(self, params):
        combos = []
        keys = list(params.keys())
        values_lists = [
            v if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for v in params.values()
        ]
        for combo in product(*values_lists):
            c = dict(zip(keys, combo))

            # Filtra combinazioni inutili:
            if c.get("derivatives") == "exact":
                c["derivative_method"] = np.nan
                c["perturbation"] = np.nan

            combos.append(c)

        # Rimuove duplicati mantenendo l'ordine
        seen = set()
        unique_combos = []
        for c in combos:
            t = tuple(sorted(c.items()))
            if t not in seen:
                seen.add(t)
                unique_combos.append(c)

        return unique_combos

    def initialize_x0(self, function: str, n: int):
        """ Return a standard(theorical) initial guess x0 for the chosen test function. 
            INPUT:
            - function: str -> actual function
            - n: int -> problem dimension
            """
        match function:
            case "extended_rosenbrock":
                x0 = np.empty(n, dtype=float)
                x0[0::2] = -1.2
                x0[1::2] = 1.0
                return x0
            case "extended_powell":
                x0 = np.empty(n, dtype=float)
                x0[0::4] = 3.0
                x0[1::4] = -1.0
                x0[2::4] = 0.0
                x0[3::4] = 1.0
                return x0
            case "broyden_tridiagonal_function":
                return -np.ones(n, dtype=float)
            case "rosenbrock":
                x0 = np.array([-1.2, 1.0], dtype=float)
                return x0
            case _:
                raise ValueError(f"Unknown function '{function}'.")
    
    def sample_x0(self,x_bar:np.array,m:int,seed:int):
        """ Samples m initial guess from hypercube defined with the mean of banchmark initial solution x0(bar)
            INPUT:
            - x_bar: np.array -> Initial theorical solution of the problem 
            - m: int -> number of point to be generated 
            - seed: int -> seed for reproducibility 
        """
        x0_list = []
        n = len(x_bar)
        generator = np.random.default_rng(seed)
        l = x_bar - 1.0
        u = x_bar + 1.0
        
        for i in range(m):
            x0 = generator.uniform(l,u,size=n)
            x0_list.append(x0)

        return x0_list
    
    def experimental_convergence_rate(self,grad_norms,tail=3,eps=1e-16):
        """ This function retrieve experimental rate of convergence without knowning optimal solution of the problem.
            Used formula: p_k = log(e_k+1/e_k)/log(e_k/e_k-1) where e_k = ||gradf(x_k)|| 
            INPUT:
            - norm_grad_seq = retrieved sequence of gradients norms of each iteration
            - tail = number of values of the sequence taken into account (last elements of sequence, more reilable)
        """
        g = [max(float(v), eps) for v in grad_norms if v is not None and np.isfinite(v)]
        if len(g) < 3:
            #We cannot retrieve and experimental rate
            return np.nan

        g = g[-(tail+2):]  # (tail+2) to compute p_k
        pk_vals = []
        for i in range(1, len(g) - 1):
            x_k_1, x_k, x_k__1 = g[i-1], g[i], g[i+1]       #x_k__1 is x_k+1)
            if x_k_1 > 0 and x_k > 0 and x_k__1 > 0 and x_k_1 != x_k:
                p = math.log(x_k__1/x_k) / math.log(x_k/x_k_1)
                if np.isfinite(p):
                    pk_vals.append(p)

        return float(np.median(pk_vals)) if pk_vals else np.nan