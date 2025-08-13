import numpy as np

class FunctionDefinition(object):
    """This class computes specified function in a given point"""
    def __init__(self):
        pass

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
    
    def dbv_function(self,x):
        """
        Calcola F(x) = sum_i( f_i(x)^2 ) per un DBVP,
        in modo vettoriale (senza loop Python) per x shape (n,).

        f_i(x) = 2*x[i] - x[i-1] - x[i+1] + h^2 * ( x[i] + (i+1)*h + 1 )^(3/2)
        con x[-1] = 0 se i=0, x[n] = 0 se i=n-1.
        Nota: il problema non ha il fattore 0.5 davanti alla somma.
        """
        n = len(x)
        h = 1.0/(n+1)
        # shift
        x_im1 = np.roll(x, 1)
        x_im1[0] = 0.0                  # x[-1] = 0
        x_ip1 = np.roll(x, -1)
        x_ip1[-1] = 0.0                 # x[n] = 0

        i_array = np.arange(n)
        V = x + (i_array+1)*h + 1.0 

        # Potenza 3/2 robusta: evita NaN se V < 0 durante l'ottimizzazione se fisicamente V>=0
        V32 = np.clip(V, 0.0, None)**1.5  

        f = 2.0*x - x_im1 - x_ip1 + (h*h)*V32
        return np.sum(f*f)  # scalare
    
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