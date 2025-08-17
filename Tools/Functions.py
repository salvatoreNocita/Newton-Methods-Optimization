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
    
    def extended_powell(self, x):
        """
        Compute the block-structured function
            F(x) = (1/n) * \sum_{k=1}^{n} f_k(x)^2
        where n is a multiple of 4 and, for each block j = 1, ..., n/4
            f_{4j-3}(x) = x_{4j-3} + 10 x_{4j-2}
            f_{4j-2}(x) = sqrt(5) * (x_{4j-1} - x_{4j})
            f_{4j-1}(x) = (x_{4j-2} - 2 x_{4j-1})**2
            f_{4j}(x)   = sqrt(10) * (x_{4j-3} - x_{4j})**2
        The returned value is F(x).

        Parameters
        ----------
        x : np.ndarray
            1-D array of length n, with n % 4 == 0.

        Returns
        -------
        float
            Function value F(x).
        """
        x = np.asarray(x, dtype=float)
        n = x.size

        # Work blockwise: [a, b, c, d] = [x_{4j-3}, x_{4j-2}, x_{4j-1}, x_{4j}]
        blocks = x.reshape(-1, 4)
        a = blocks[:, 0]
        b = blocks[:, 1]
        c = blocks[:, 2]
        d = blocks[:, 3]

        f1 = a + 10.0 * b
        f2 = np.sqrt(5.0) * (c - d)
        f3 = (b - 2.0 * c) ** 2
        f4 = np.sqrt(10.0) * (a - d) ** 2

        f = np.concatenate([f1, f2, f3, f4])
        return float((1.0 / n) * np.sum(f ** 2))
    
    def rosenbrock_function(self, x):
        """ Compute the Rosenbrock function value for a given vector x. """
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))