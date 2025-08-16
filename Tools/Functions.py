import numpy as np

class FunctionDefinition(object):
    """This class computes specified function in a given point"""
    def __init__(self):
        pass
        
    def get_objective_function(self, function_name: str):
        self.function_map = {
            'extended_rosenbrock': self.extended_rosenbrock,
            'discrete_boundary_value_problem': self.dbv_function,
            'broyden_tridiagonal_function': self.btf_function,
            'rosenbrock': self.rosenbrock_function
        }
        if function_name in self.function_map:
            return self.function_map[function_name]
        else:
            raise ValueError(f"Function '{function_name}' is not defined in FunctionDefinition class.")

    def extended_rosenbrock(self,x):
        """
        Compute the value of the extended Rosenbrock function at point x.
        Optimized for performance using vectorized operations.

        Input:
        x : Point to evaluate the function (numpy array of even length).

        Returns:
        F_x : Value of the extended Rosenbrock function at x.
        """
        n = len(x)
        assert n % 2 == 0, "Dimension must be even."

        # Reshape x into pairs (x1, x2), (x3, x4), ..., (x_{n-1}, x_n)
        x_pairs = x.reshape(-1, 2)  # Shape: (n/2, 2)

        # Compute the two terms of the Rosenbrock function
        term1 = 10 * (x_pairs[:, 0]**2 - x_pairs[:, 1])  # 10(x_{2k-1}^2 - x_{2k})
        term2 = x_pairs[:, 0] - 1  # (x_{2k-1} - 1)

        # Combine the terms and compute the function value
        F_x = 0.5 * np.sum(term1**2 + term2**2)

        return F_x
    
    def dbv_function(self, x):
        """
        Computes F(x) = sum_i( f_i(x)^2 ) for the discrete boundary value problem.

        f_i(x) = 2*x[i] - x[i-1] - x[i+1] + h^2 * (x[i] + (i+1)*h + 1)^3 / 2
        with boundary conditions x[0] = 0 (implicit) and x[n+1] = 0 (handled via padding).

        Parameters:
        x (numpy.ndarray): Array of shape (n,) containing x_1 to x_n.

        Returns:
        float: The value of F(x).
        """
        import numpy as np

        n = len(x)
        h = 1.0 / (n + 1)
        
        # Handle boundary conditions: x_0 = x_{n+1} = 0
        x_padded = np.concatenate([[0.0], x, [0.0]])  # [x_0, x_1, ..., x_n, x_{n+1}]
        
        # Compute terms using vectorized operations
        x_i = x_padded[1:-1]  # x_1 to x_n
        x_im1 = x_padded[:-2]  # x_0 to x_{n-1}
        x_ip1 = x_padded[2:]   # x_2 to x_{n+1}
        
        i_array = np.arange(1, n + 1)  # i from 1 to n
        V = x_i + i_array * h + 1.0    # x_i + i*h + 1
        
        # Compute (V)^3 / 2 (problem statement has ^3/2, not ^1.5)
        V_term = (V ** 3) / 2.0
        
        f_i = 2.0 * x_i - x_im1 - x_ip1 + (h ** 2) * V_term
        return np.sum(f_i ** 2)
    
    def btf_function(self, x):
        """
        Generalized Broyden tridiagonal (Problem 5):
        F(x) = sum_{i=1}^n | (3 - 2*x_i)*x_i + 1 - x_{i-1} - x_{i+1} |^p
        con p = 7/3 e x_0 = x_{n+1} = 0.
        """
        p = 7.0 / 3.0
        n = len(x)

        # vicini con condizioni al bordo nulle
        x_im1 = np.roll(x, 1)
        x_im1[0] = 0.0
        x_ip1 = np.roll(x, -1)
        x_ip1[-1] = 0.0

        f = (3.0 - 2.0*x) * x + 1.0 - x_im1 - x_ip1
        return np.sum(np.abs(f) ** p)
    
    def rosenbrock_function(self, x):
        """ Compute the Rosenbrock function value for a given vector x. """
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))