import numpy as np

class Test_settings(object):
    """ This class is used to retrieve Tests setting. """

    def __init__():
        pass

    def get_params():
        NewtonBackTracking_ARG_f= {'alpha0': 1,
                                    'btmax': 50,
                                    'rho': 0.5,
                                    'c1': 1e-4,
                                    'tolgrad': 1e-3,
                                    'kmax': 1000,
                                    'solver_linear_system': 'cg',
                                    'H_correction_factor': 3,
                                    'precond': ['yes','no'],
                                    'derivatives': ['exact','finite_differences'],
                                    'derivative_method': ['forward','backward','central'],
                                    'perturbation': [1e-2,1e-4,1e-6,1e-8,1e-10,1e-12]
                                }
        
        return NewtonBackTracking_ARG_f

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
            case "discrete_boundary_value_problem":
                h = 1.0 / (n + 1)
                i_array = np.arange(1, n+1)
                return i_array * h * (1 - i_array * h)
            case "broyden_tridiagonal_function":
                return -np.ones(n, dtype=float)
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